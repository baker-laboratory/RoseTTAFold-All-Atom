import os
import hydra
import torch
import torch.nn as nn
from dataclasses import asdict

from rf2aa.data.merge_inputs import merge_all
from rf2aa.data.covale import load_covalent_molecules
from rf2aa.data.nucleic_acid import load_nucleic_acid
from rf2aa.data.protein import generate_msa_and_load_protein
from rf2aa.data.small_molecule import load_small_molecule
from rf2aa.ffindex import *
from rf2aa.chemical import initialize_chemdata, load_pdb_ideal_sdf_strings
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.model.RoseTTAFoldModel import RoseTTAFoldModule
from rf2aa.training.recycling import recycle_step_legacy
from rf2aa.util import writepdb, is_atom, Ls_from_same_chain_2d
from rf2aa.util_module import XYZConverter


class ModelRunner:

    def __init__(self, config) -> None:
        self.config = config
        initialize_chemdata(self.config.chem_params)
        FFindexDB = namedtuple("FFindexDB", "index, data")
        self.ffdb = FFindexDB(read_index(config.database_params.hhdb+'_pdb.ffindex'),
                              read_data(config.database_params.hhdb+'_pdb.ffdata'))
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.xyz_converter = XYZConverter()
        self.deterministic = config.get("deterministic", False)
        self.molecule_db = load_pdb_ideal_sdf_strings()

    def parse_inference_config(self):
        residues_to_atomize = [] # chain letter, residue number, residue name
        chains = []
        protein_inputs = {}
        if self.config.protein_inputs is not None:
            for chain in self.config.protein_inputs:
                if chain in chains:
                    raise ValueError(f"Duplicate chain found with name: {chain}. Please specify unique chain names")
                elif len(chain) > 1:
                    raise ValueError(f"Chain name must be a single character, found chain with name: {chain}")
                else:
                    chains.append(chain)
                protein_input = generate_msa_and_load_protein(
                    self.config.protein_inputs[chain]["fasta_file"],
                    chain,
                    self
                ) 
                protein_inputs[chain] = protein_input
        
        na_inputs = {}
        if self.config.na_inputs is not None:
            for chain in self.config.na_inputs:
                na_input = load_nucleic_acid(
                    self.config.na_inputs[chain]["fasta"],
                    self.config.na_inputs[chain]["input_type"],
                    self
                )
                na_inputs[chain] = na_input

        sm_inputs = {} 
        # first if any of the small molecules are covalently bonded to the protein
        # merge the small molecule with the residue and add it as a separate ligand
        # also add it to residues_to_atomize for bookkeeping later on
        # need to handle atomizing multiple consecutive residues here too
        if self.config.covale_inputs is not None:
            covalent_sm_inputs, residues_to_atomize_covale = load_covalent_molecules(protein_inputs, self.config, self)
            sm_inputs.update(covalent_sm_inputs)
            residues_to_atomize.extend(residues_to_atomize_covale)
            
        if self.config.sm_inputs is not None:
            for chain in self.config.sm_inputs:
                if self.config.sm_inputs[chain]["input_type"] not in ["smiles", "sdf"]:
                    raise ValueError("Small molecule input type must be smiles or sdf")
                if chain in sm_inputs: # chain already processed as covale
                    continue
                if "is_leaving" in self.config.sm_inputs[chain]:
                    raise ValueError("Leaving atoms are not supported for non-covalently bonded molecules")
                sm_input = load_small_molecule(
                   self.config.sm_inputs[chain]["input"],
                   self.config.sm_inputs[chain]["input_type"],
                   self
                )
                sm_inputs[chain] = sm_input

        if self.config.residue_replacement is not None:
            # add to the sm_inputs list
            # add to residues to atomize
            raise NotImplementedError("Modres inference is not implemented")
        
        raw_data = merge_all(protein_inputs, na_inputs, sm_inputs, residues_to_atomize, deterministic=self.deterministic)
        self.raw_data = raw_data

    def load_model(self):
        self.model = RoseTTAFoldModule(
            **self.config.legacy_model_param,
            aamask = ChemData().allatom_mask.to(self.device),
            atom_type_index = ChemData().atom_type_index.to(self.device),
            ljlk_parameters = ChemData().ljlk_parameters.to(self.device),
            lj_correction_parameters = ChemData().lj_correction_parameters.to(self.device),
            num_bonds = ChemData().num_bonds.to(self.device),
            cb_len = ChemData().cb_length_t.to(self.device),
            cb_ang = ChemData().cb_angle_t.to(self.device),
            cb_tor = ChemData().cb_torsion_t.to(self.device),

        ).to(self.device)
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def construct_features(self):
        return self.raw_data.construct_features(self)

    def run_model_forward(self, input_feats):
        input_feats.add_batch_dim()
        input_feats.to(self.device)
        input_dict = asdict(input_feats)
        input_dict["bond_feats"] = input_dict["bond_feats"].long()
        input_dict["seq_unmasked"] = input_dict["seq_unmasked"].long()
        outputs = recycle_step_legacy(self.model, 
                                     input_dict, 
                                     self.config.loader_params.MAXCYCLE, 
                                     use_amp=False,
                                     nograds=True,
                                     force_device=self.device)
        return outputs


    def write_outputs(self, input_feats, outputs):
        logits, logits_aa, logits_pae, logits_pde, p_bind, \
                xyz, alpha_s, xyz_allatom, lddt, _, _, _ \
            = outputs
        seq_unmasked = input_feats.seq_unmasked
        bond_feats = input_feats.bond_feats
        err_dict = self.calc_pred_err(lddt, logits_pae, logits_pde, seq_unmasked)
        err_dict["same_chain"] = input_feats.same_chain
        plddts = err_dict["plddts"]
        Ls = Ls_from_same_chain_2d(input_feats.same_chain)
        plddts = plddts[0]
        writepdb(os.path.join(f"{self.config.output_path}", f"{self.config.job_name}.pdb"), 
                 xyz_allatom, 
                 seq_unmasked, 
                 bond_feats=bond_feats,
                 bfacts=plddts,
                 chain_Ls=Ls
                 )
        torch.save(err_dict, os.path.join(f"{self.config.output_path}", 
                                          f"{self.config.job_name}_aux.pt"))

    def infer(self):
        self.load_model()
        self.parse_inference_config()
        input_feats = self.construct_features()
        outputs = self.run_model_forward(input_feats)
        self.write_outputs(input_feats, outputs)

    def lddt_unbin(self, pred_lddt):
        # calculate lddt prediction loss
        nbin = pred_lddt.shape[1]
        bin_step = 1.0 / nbin
        lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)

        pred_lddt = nn.Softmax(dim=1)(pred_lddt)
        return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

    def pae_unbin(self, logits_pae, bin_step=0.5):
        nbin = logits_pae.shape[1]
        bins = torch.linspace(bin_step*0.5, bin_step*nbin-bin_step*0.5, nbin,
                                dtype=logits_pae.dtype, device=logits_pae.device)
        logits_pae = torch.nn.Softmax(dim=1)(logits_pae)
        return torch.sum(bins[None,:,None,None]*logits_pae, dim=1)

    def pde_unbin(self, logits_pde, bin_step=0.3):
        nbin = logits_pde.shape[1]
        bins = torch.linspace(bin_step*0.5, bin_step*nbin-bin_step*0.5, nbin,
                                dtype=logits_pde.dtype, device=logits_pde.device)
        logits_pde = torch.nn.Softmax(dim=1)(logits_pde)
        return torch.sum(bins[None,:,None,None]*logits_pde, dim=1)

    def calc_pred_err(self, pred_lddts, logit_pae, logit_pde, seq):
        """Calculates summary metrics on predicted lDDT and distance errors"""
        plddts = self.lddt_unbin(pred_lddts)
        pae = self.pae_unbin(logit_pae) if logit_pae is not None else None
        pde = self.pde_unbin(logit_pde) if logit_pde is not None else None
        sm_mask = is_atom(seq)[0]
        sm_mask_2d = sm_mask[None,:]*sm_mask[:,None]
        prot_mask_2d = (~sm_mask[None,:])*(~sm_mask[:,None])
        inter_mask_2d = sm_mask[None,:]*(~sm_mask[:,None]) + (~sm_mask[None,:])*sm_mask[:,None]
        # assumes B=1
        err_dict = dict(
            plddts = plddts.cpu(),
            pae = pae.cpu(),
            pde = pde.cpu(),
            mean_plddt = float(plddts.mean()),
            mean_pae = float(pae.mean()) if pae is not None else None,
            pae_prot = float(pae[0,prot_mask_2d].mean()) if pae is not None else None,
            pae_inter = float(pae[0,inter_mask_2d].mean()) if pae is not None else None,
        )
        return err_dict


@hydra.main(version_base=None, config_path='config/inference')
def main(config):
    runner = ModelRunner(config)
    runner.infer()

if __name__ == "__main__":
    main()
