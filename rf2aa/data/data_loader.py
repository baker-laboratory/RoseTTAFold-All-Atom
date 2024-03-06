import torch
from dataclasses import dataclass, fields
from typing import Optional, List

from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.data.data_loader_utils import MSAFeaturize, get_bond_distances, generate_xyz_prev
from rf2aa.kinematics import xyz_to_t2d
from rf2aa.util import get_prot_sm_mask, xyz_t_to_frame_xyz, same_chain_from_bond_feats, \
        Ls_from_same_chain_2d, idx_from_Ls, is_atom



@dataclass
class RawInputData:
    msa: torch.Tensor
    ins: torch.Tensor
    bond_feats: torch.Tensor
    xyz_t: torch.Tensor
    mask_t: torch.Tensor
    t1d: torch.Tensor
    chirals: torch.Tensor
    atom_frames: torch.Tensor
    taxids: Optional[List[str]] = None
    term_info: Optional[torch.Tensor] = None
    chain_lengths: Optional[List] = None
    idx: Optional[List] = None

    def query_sequence(self):
        return self.msa[0]
    
    def sequence_string(self):
        three_letter_sequence  = [ChemData().num2aa[num] for num in self.query_sequence()]
        return "".join([ChemData().aa_321[three] for three in three_letter_sequence])
    
    def is_atom(self):
        return is_atom(self.query_sequence())

    def length(self):
        return self.msa.shape[1] 
    
    def get_chain_bins_from_chain_lengths(self):
        if self.chain_lengths is None:
            raise ValueError("Cannot call get_chain_bins_from_chain_lengths without \
                             setting chain_lengths. Chain_lengths is set in merge_inputs")
        chain_bins = {}
        running_length = 0
        for chain, length in self.chain_lengths:
            chain_bins[chain] = (running_length, running_length+length)
            running_length = running_length + length
        return chain_bins
    
    def update_protein_features_after_atomize(self, residues_to_atomize):
        if self.chain_lengths is None:
            raise("Cannot update protein features without chain_lengths. \
                  merge_inputs must be called before this function")
        chain_bins = self.get_chain_bins_from_chain_lengths()
        keep = torch.ones(self.length())
        prev_absolute_index = None
        prev_C = None
        #need to atomize residues from N term to Cterm to handle atomizing neighbors
        residues_to_atomize = sorted(residues_to_atomize, key= lambda x: x.original_chain +str(x.index_in_original_chain))
        for residue in residues_to_atomize:
            original_chain_start_index, original_chain_end_index = chain_bins[residue.original_chain]
            absolute_index_in_combined_input = original_chain_start_index + residue.index_in_original_chain

            atomized_chain_start_index, atomized_chain_end_index = chain_bins[residue.chain]
            N_index = atomized_chain_start_index + residue.absolute_N_index_in_chain
            C_index = atomized_chain_start_index + residue.absolute_C_index_in_chain
            # if residue is first in the chain, no extra bond feats to following residue
            if absolute_index_in_combined_input != original_chain_start_index:
                self.bond_feats[absolute_index_in_combined_input-1, N_index] = ChemData().RESIDUE_ATOM_BOND
                self.bond_feats[N_index, absolute_index_in_combined_input-1] = ChemData().RESIDUE_ATOM_BOND
            
            # if residue is last in chain, no extra bonds feats to following residue
            if absolute_index_in_combined_input != original_chain_end_index-1:
                self.bond_feats[absolute_index_in_combined_input+1, C_index] = ChemData().RESIDUE_ATOM_BOND
                self.bond_feats[C_index,absolute_index_in_combined_input+1] = ChemData().RESIDUE_ATOM_BOND
            keep[absolute_index_in_combined_input] = 0
            
            # find neighboring residues that were atomized
            if prev_absolute_index is not None:
                if prev_absolute_index + 1  == absolute_index_in_combined_input:
                    self.bond_feats[prev_C, N_index] = 1
                    self.bond_feats[N_index, prev_C] = 1

            prev_absolute_index = absolute_index_in_combined_input
            prev_C = C_index
        # remove protein features 
        self.keep_features(keep.bool())

    def keep_features(self, keep):
        if not torch.all(keep[self.is_atom()]):
            raise ValueError("cannot remove atoms")
        self.msa = self.msa[:,keep]
        self.ins = self.ins[:,keep]
        self.bond_feats = self.bond_feats[keep][:,keep]
        self.xyz_t = self.xyz_t[:,keep]
        self.t1d = self.t1d[:,keep]
        self.mask_t = self.mask_t[:,keep]
        if self.term_info is not None:
            self.term_info = self.term_info[keep]
        if self.idx is not None:
            self.idx = self.idx[keep]
        # assumes all chirals are after all protein residues
        self.chirals[...,:-1] = self.chirals[...,:-1] - torch.sum(~keep)

    def construct_features(self, model_runner):
        loader_params = model_runner.config.loader_params
        B, L = 1, self.length()
        seq, msa_clust, msa_seed, msa_extra, mask_pos = MSAFeaturize(
            self.msa.long(),
            self.ins.long(),
            loader_params,
            p_mask=loader_params.get("p_msa_mask", 0),
            term_info=self.term_info,
            deterministic=model_runner.deterministic,
        )
        dist_matrix = get_bond_distances(self.bond_feats)

        # xyz_prev, mask_prev = generate_xyz_prev(self.xyz_t, self.mask_t, loader_params)
        # xyz_prev = torch.nan_to_num(xyz_prev)

        # NOTE: The above is the way things "should" be done, this is for compatability with training.
        xyz_prev = ChemData().INIT_CRDS.reshape(1,ChemData().NTOTAL,3).repeat(L,1,1)
        
        self.xyz_t = torch.nan_to_num(self.xyz_t)

        mask_t_2d = get_prot_sm_mask(self.mask_t, seq[0]) 
        mask_t_2d = mask_t_2d[:,None]*mask_t_2d[:,:,None] # (B, T, L, L)

        xyz_t_frame = xyz_t_to_frame_xyz(self.xyz_t[None], self.msa[0], self.atom_frames)
        t2d = xyz_to_t2d(xyz_t_frame, mask_t_2d[None])
        t2d = t2d[0]
        # get torsion angles from templates
        seq_tmp = self.t1d[...,:-1].argmax(dim=-1)
        alpha, _, alpha_mask, _ = model_runner.xyz_converter.get_torsions(self.xyz_t.reshape(-1,L,ChemData().NTOTAL,3), 
                                seq_tmp, mask_in=self.mask_t.reshape(-1,L,ChemData().NTOTAL))
        alpha = alpha.reshape(B,-1,L,ChemData().NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(B,-1,L,ChemData().NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, L, 3*ChemData().NTOTALDOFS)
        alpha_t = alpha_t[0]
        alpha_prev = torch.zeros((L,ChemData().NTOTALDOFS,2))

        same_chain = same_chain_from_bond_feats(self.bond_feats)
        return RFInput(
            msa_latent=msa_seed,
            msa_full=msa_extra,
            seq=seq,
            seq_unmasked=self.query_sequence(),
            bond_feats=self.bond_feats,
            dist_matrix=dist_matrix,
            chirals=self.chirals,
            atom_frames=self.atom_frames.long(),
            xyz_prev=xyz_prev,
            alpha_prev=alpha_prev,
            t1d=self.t1d,
            t2d=t2d,
            xyz_t=self.xyz_t[..., 1, :],
            alpha_t=alpha_t.float(),
            mask_t=mask_t_2d.float(),
            same_chain=same_chain.long(),
            idx=self.idx
        )


@dataclass
class RFInput:
    msa_latent: torch.Tensor
    msa_full: torch.Tensor
    seq: torch.Tensor
    seq_unmasked: torch.Tensor
    idx: torch.Tensor
    bond_feats: torch.Tensor
    dist_matrix: torch.Tensor
    chirals: torch.Tensor
    atom_frames: torch.Tensor
    xyz_prev: torch.Tensor
    alpha_prev: torch.Tensor
    t1d: torch.Tensor
    t2d: torch.Tensor
    xyz_t: torch.Tensor
    alpha_t: torch.Tensor
    mask_t: torch.Tensor
    same_chain: torch.Tensor
    msa_prev: Optional[torch.Tensor] = None
    pair_prev: Optional[torch.Tensor] = None
    state_prev: Optional[torch.Tensor] = None
    mask_recycle: Optional[torch.Tensor] = None

    def to(self, gpu):
        for field in fields(self):
            field_value = getattr(self, field.name)
            if torch.is_tensor(field_value):
                setattr(self, field.name, field_value.to(gpu))
    
    def add_batch_dim(self):
        """ mimic pytorch dataloader at inference time"""
        for field in fields(self):
            field_value = getattr(self, field.name)
            if torch.is_tensor(field_value):
                setattr(self, field.name, field_value[None])

