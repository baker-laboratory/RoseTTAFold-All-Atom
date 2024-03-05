import sys
import warnings
import assertpy

import numpy as np
import random
import torch
import warnings
from assertpy import assert_that

import networkx as nx
import itertools
from itertools import combinations
from collections import OrderedDict, Counter
from openbabel import openbabel
from scipy.spatial.transform import Rotation
from icecream import ic

from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.kinematics import get_atomize_protein_chirals, generate_Cbeta
from rf2aa.scoring import *


def random_rot_trans(xyz, random_noise=20.0, deterministic: bool = False):
    if deterministic:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

    # xyz: (N, L, 27, 3)
    N, L = xyz.shape[:2]

    # pick random rotation axis
    R_mat = torch.tensor(Rotation.random(N).as_matrix(), dtype=xyz.dtype).to(xyz.device)
    xyz = torch.einsum('nij,nlaj->nlai', R_mat, xyz) + torch.rand(N,1,1,3, device=xyz.device)*random_noise
    return xyz

def get_prot_sm_mask(atom_mask, seq):
    """
    Parameters
    ----------
    atom_mask : (..., L, Natoms) 
    seq : (L) 

    Returns
    -------
    mask : (..., L) 
    """
    sm_mask = is_atom(seq).to(atom_mask.device) # (L)
    # Asserting that atom_mask is full for masked regions of proteins [should be]
    has_backbone = atom_mask[...,:3].all(dim=-1)
    # has_backbone_prot = has_backbone[...,~sm_mask]
    # n_protein_with_backbone = has_backbone.sum()
    # n_protein = (~sm_mask).sum()
    #assert_that((n_protein/n_protein_with_backbone).item()).is_greater_than(0.8)
    mask_prot = has_backbone & ~sm_mask # valid protein/NA residues (L)
    mask_ca_sm = atom_mask[...,1] & sm_mask # valid sm mol positions (L)

    mask = mask_prot | mask_ca_sm # valid positions
    return mask

def center_and_realign_missing(xyz, mask_t, seq=None, same_chain=None, should_center: bool = True):
    """
    Moves center of mass of xyz to origin, then moves positions with missing
    coordinates to nearest existing residue on same chain.

    Parameters
    ----------
    seq : (L)
    xyz : (L, Natms, 3)
    mask_t : (L, Natms)
    same_chain : (L, L)

    Returns
    -------
    xyz : (L, Natms, 3)
    
    """
    L = xyz.shape[0]

    if same_chain is None:
        same_chain = torch.full((L,L), True)

    # valid protein/NA/small mol. positions
    if seq is None:
        mask = torch.full((L,), True)
    else:
        mask = get_prot_sm_mask(mask_t, seq)

    # center c.o.m of existing residues at the origin
    if should_center:
        center_CA = xyz[mask,1].mean(dim=0) # (3)
        xyz = torch.where(mask.view(L,1,1), xyz - center_CA.view(1, 1, 3), xyz)

    # move missing residues to the closest valid residues on same chain
    exist_in_xyz = torch.where(mask)[0] # (L_sub)
    same_chain_in_xyz = same_chain[:,mask].bool() # (L, L_sub)
    seqmap = (torch.arange(L, device=xyz.device)[:,None] - exist_in_xyz[None,:]).abs() # (L, L_sub)
    seqmap[~same_chain_in_xyz] += 99999
    seqmap = torch.argmin(seqmap, dim=-1) # (L)
    idx = torch.gather(exist_in_xyz, 0, seqmap) # (L)
    offset_CA = torch.gather(xyz[:,1], 0, idx.reshape(L,1).expand(-1,3))
    has_neighbor = same_chain_in_xyz.all(-1) 
    offset_CA[~has_neighbor] = 0 # stay at origin if nothing on same chain has coords
    xyz = torch.where(mask.view(L, 1, 1), xyz, xyz + offset_CA.reshape(L,1,3))

    return xyz


# note: needs consistency with chemical.py
def is_protein(seq):
    return seq < ChemData().NPROTAAS

def is_nucleic(seq):
    return (seq>=ChemData().NPROTAAS) * (seq <= ChemData().NNAPROTAAS)

# fd hacky
def is_DNA(seq):
    return (seq>=ChemData().NPROTAAS) * (seq < ChemData().NPROTAAS+5)

# fd hacky
def is_RNA(seq):
    return (seq>=ChemData().NPROTAAS+5) * (seq < ChemData().NNAPROTAAS)

def is_atom(seq):
    return seq > ChemData().NNAPROTAAS

# build a frame from 3 points
#fd  -  more complicated version splits angle deviations between CA-N and CA-C (giving more accurate CB position)
#fd  -  makes no assumptions about input dims (other than last 1 is xyz)
def rigid_from_3_points(N, Ca, C, is_na=None, eps=1e-4):
    dims = N.shape[:-1]

    v1 = C-Ca
    v2 = N-Ca
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)
    u2 = v2-(torch.einsum('...li, ...li -> ...l', e1, v2)[...,None]*e1)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1) #[B,L,3,3] - rotation matrix

    v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
    cosref = torch.sum(e1*v2, dim=-1)

    costgt = torch.full(dims, -0.3616, device=N.device)
    if is_na is not None:
       costgt[is_na] = ChemData().costgtNA

    cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )

    cosdel = torch.sqrt(0.5*(1+cos2del)+eps)

    sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)

    Rp = torch.eye(3, device=N.device).repeat(*dims,1,1)
    Rp[...,0,0] = cosdel
    Rp[...,0,1] = -sindel
    Rp[...,1,0] = sindel
    Rp[...,1,1] = cosdel
    R = torch.einsum('...ij,...jk->...ik', R,Rp)

    return R, Ca

def idealize_reference_frame(seq, xyz_in):
    xyz = xyz_in.clone()

    namask = is_nucleic(seq)
    Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], namask)

    protmask = ~namask

    pmask_bs,pmask_rs = protmask.nonzero(as_tuple=True)
    nmask_bs,nmask_rs = namask.nonzero(as_tuple=True)
    xyz[pmask_bs,pmask_rs,0,:] = torch.einsum('...ij,j->...i', Rs[pmask_bs,pmask_rs], ChemData().init_N.to(device=xyz_in.device) ) + Ts[pmask_bs,pmask_rs]
    xyz[pmask_bs,pmask_rs,2,:] = torch.einsum('...ij,j->...i', Rs[pmask_bs,pmask_rs], ChemData().init_C.to(device=xyz_in.device) ) + Ts[pmask_bs,pmask_rs]
    xyz[nmask_bs,nmask_rs,0,:] = torch.einsum('...ij,j->...i', Rs[nmask_bs,nmask_rs], ChemData().init_O1.to(device=xyz_in.device) ) + Ts[nmask_bs,nmask_rs]
    xyz[nmask_bs,nmask_rs,2,:] = torch.einsum('...ij,j->...i', Rs[nmask_bs,nmask_rs], ChemData().init_O2.to(device=xyz_in.device) ) + Ts[nmask_bs,nmask_rs]

    return xyz

def xyz_to_frame_xyz(xyz, seq_unmasked, atom_frames):
    """
    xyz (1, L, natoms, 3)
    seq_unmasked (1, L)
    atom_frames (1, L, 3, 2)
    """ 
    xyz_frame = xyz.clone()
    atoms = is_atom(seq_unmasked)
    if torch.all(~atoms):
        return xyz_frame

    atom_crds = xyz_frame[atoms]
    atom_L, natoms, _ = atom_crds.shape
    frames_reindex = torch.zeros(atom_frames.shape[:-1])
    
    for i in range(atom_L):
        frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0])*natoms + atom_frames[..., i, :, 1]
    frames_reindex = frames_reindex.long()

    xyz_frame[atoms, :, :3] = atom_crds.reshape(atom_L*natoms, 3)[frames_reindex]
    return xyz_frame

def xyz_frame_from_rotation_mask(xyz,rotation_mask, atom_frames):
    """
    function to get xyz_frame for l1 feature in Structure module
    xyz (1, L, natoms, 3)
    rotation_mask (1, L)
    atom_frames (1, L, 3, 2)
    """
    xyz_frame = xyz.clone()
    if torch.all(~rotation_mask):
        return xyz_frame

    atom_crds = xyz_frame[rotation_mask]
    atom_L, natoms, _ = atom_crds.shape
    frames_reindex = torch.zeros(atom_frames.shape[:-1])
    
    for i in range(atom_L):
        frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0])*natoms + atom_frames[..., i, :, 1]
    frames_reindex = frames_reindex.long()
    xyz_frame[rotation_mask, :, :3] = atom_crds.reshape(atom_L*natoms, 3)[frames_reindex]
    return xyz_frame

def xyz_t_to_frame_xyz(xyz_t, seq_unmasked, atom_frames):
    """
    Parameters:
        xyz_t (1, T, L, natoms, 3)
        seq_unmasked (B, L)
        atom_frames (1, A, 3, 2)
    Returns:
	    xyz_t_frame (B, T, L, natoms, 3)
    """
    is_sm = is_atom(seq_unmasked[0])
    return xyz_t_to_frame_xyz_sm_mask(xyz_t, is_sm, atom_frames)

def xyz_t_to_frame_xyz_sm_mask(xyz_t, is_sm, atom_frames):
    """
    Parameters:
        xyz_t (1, T, L, natoms, 3)
        is_sm (L)
        atom_frames (1, A, 3, 2)
    Returns:
	xyz_t_frame (B, T, L, natoms, 3)
    """
    # ic(xyz_t.shape, is_sm.shape, atom_frames.shape)
    # xyz_t.shape: torch.Size([1, 1, 194, 36, 3]) 
    # is_sm.shape: torch.Size([194])
    # atom_frames.shape: torch.Size([1, 29, 3, 2])
    xyz_t_frame = xyz_t.clone()
    atoms = is_sm
    if torch.all(~atoms):
        return xyz_t_frame
    atom_crds_t = xyz_t_frame[:, :, atoms]

    B, T, atom_L, natoms, _ = atom_crds_t.shape
    frames_reindex = torch.zeros(atom_frames.shape[:-1])
    for i in range(atom_L):
        frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0])*natoms + atom_frames[..., i, :, 1]
    frames_reindex = frames_reindex.long()
    xyz_t_frame[:, :, atoms, :3] = atom_crds_t.reshape(T, atom_L*natoms, 3)[:, frames_reindex.squeeze(0)]
    return xyz_t_frame

def get_frames(xyz_in, xyz_mask, seq, frame_indices, atom_frames=None):
    #B,L,natoms = xyz_in.shape[:3]
    frames = frame_indices[seq]
    atoms = is_atom(seq)
    if torch.any(atoms):
        frames[:,atoms[0].nonzero().flatten(), 0] = atom_frames

    frame_mask = ~torch.all(frames[...,0, :] == frames[...,1, :], axis=-1)

    # frame_mask *= torch.all(
    #     torch.gather(xyz_mask,2,frames.reshape(B,L,-1)).reshape(B,L,-1,3),
    #     axis=-1)

    return frames, frame_mask

def get_tips(xyz, seq):
    B,L = xyz.shape[:2]

    xyz_tips = torch.gather(xyz, 2, tip_indices.to(xyz.device)[seq][:,:,None,None].expand(-1,-1,-1,3)).reshape(B, L, 3)
    if torch.isnan(xyz_tips).any(): # replace NaN tip atom with virtual Cb atom
        # three anchor atoms
        N  = xyz[:,:,0]
        Ca = xyz[:,:,1]
        C  = xyz[:,:,2]

        # recreate Cb given N,Ca,C
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca    

        xyz_tips = torch.where(torch.isnan(xyz_tips), Cb, xyz_tips)
    return xyz_tips

def superimpose(pred, true, atom_mask):
    
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)
    
    B, L, natoms = pred.shape[:3]

    # center to centroid
    pred_allatom = pred[atom_mask][None]
    true_allatom = true[atom_mask][None]

    cp = centroid(pred_allatom)
    ct = centroid(true_allatom)
    
    pred_allatom_origin = pred_allatom - cp
    true_allatom_origin = true_allatom - ct

    # Computation of the covariance matrix
    C = torch.matmul(pred_allatom_origin.permute(0,2,1), true_allatom_origin)

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign to ensure right-handedness
    d = torch.ones([B,3,3], device=pred.device)
    d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)

    # Rotation matrix U
    U = torch.matmul(d*V, W.permute(0,2,1)) # (IB, 3, 3)
    pred_rms = pred - cp
    true_rms = true - ct
    
    # Rotate pred
    rP = torch.matmul(pred_rms, U) # (IB, L*3, 3)
    
    return rP+ct

def writepdb(filename, *args, file_mode='w', **kwargs, ):
    f = open(filename, file_mode)
    writepdb_file(f, *args, **kwargs)

def writepdb_file(f, atoms, seq, modelnum=None, chain="A", idx_pdb=None, bfacts=None, 
             bond_feats=None, file_mode="w",atom_mask=None, atom_idx_offset=0, chain_Ls=None,
             remap_atomtype=True, lig_name='LG1', atom_names=None):

    def _get_atom_type(atom_name):
        atype = ''
        if atom_name[0].isalpha():
            atype += atom_name[0]
        atype += atom_name[1]
        return atype

    # if needed, correct mistake in atomic number assignment in RF2-allatom (fold&dock 3 & earlier)
    atom_names_ = [
        "F",  "Cl", "Br", "I",  "O",  "S",  "Se", "Te", "N",  "P",  "As", "Sb",
        "C",  "Si", "Ge", "Sn", "Pb", "B",  "Al", "Zn", "Hg", "Cu", "Au", "Ni", 
        "Pd", "Pt", "Co", "Rh", "Ir", "Pr", "Fe", "Ru", "Os", "Mn", "Re", "Cr", 
        "Mo", "W",  "V",  "U",  "Tb", "Y",  "Be", "Mg", "Ca", "Li", "K",  "ATM"]
    atom_num = [
        9,    17,   35,   53,   8,    16,   34,   52,   7,    15,   33,   51,
        6,    14,   32,   50,   82,   5,    13,   30,   80,   29,   79,   28,
        46,   78,   27,   45,   77,   59,   26,   44,   76,   25,   75,   24,   
        42,   74,   23,   92,   65,   39,   4,    12,   20,   3,    19,   0] 
    atomnum2atomtype_ = dict(zip(atom_num,atom_names_))
    if remap_atomtype:
        atomtype_map = {v:atomnum2atomtype_[k] for k,v in ChemData().atomnum2atomtype.items()}
    else:
        atomtype_map = {v:v for k,v in ChemData().atomnum2atomtype.items()} # no change
        
    ctr = 1+atom_idx_offset
    scpu = seq.cpu().squeeze(0)
    atomscpu = atoms.cpu().squeeze(0)

    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
    if chain_Ls is not None:
        chain_letters = np.concatenate([np.full(L, alphabet[i]) for i,L in enumerate(chain_Ls)])
    else:
        chain_letters = [chain]*len(scpu)

    if modelnum is not None:
        f.write(f"MODEL        {modelnum}\n")

    Bfacts = torch.clamp( bfacts.cpu(), 0, 1)
    atom_idxs = {}
    i_res_lig = 0
    for i_res,s,ch in zip(range(len(scpu)), scpu, chain_letters):
        natoms = atomscpu.shape[-2]
        #if (natoms!=NHEAVY and natoms!=NTOTAL and natoms!=3):
        #    print ('bad size!', natoms, NHEAVY, NTOTAL, atoms.shape)
        #    assert(False)

        if s >= len(ChemData().aa2long):
            atom_idxs[i_res] = ctr

            # hack to make sure H's are output properly (they are not in RFAA alphabet)
            if atom_names is not None:
                atom_type = _get_atom_type(atom_names[i_res_lig])
                atom_name = atom_names[i_res_lig]
            else:
                atom_type = atomtype_map[ChemData().num2aa[s]]
                atom_name = atom_type

            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %+2s\n"%(
                    "HETATM", ctr, atom_name, lig_name,
                    ch, idx_pdb.max()+10, atomscpu[i_res,1,0], atomscpu[i_res,1,1], atomscpu[i_res,1,2],
                    1.0, Bfacts[i_res],  atom_type) )
            i_res_lig += 1
            ctr += 1
            continue

        atms = ChemData().aa2long[s]

        for i_atm,atm in enumerate(atms):
            if atom_mask is not None and not atom_mask[i_res,i_atm]: continue # skip missing atoms
            if (i_atm<natoms and atm is not None and not torch.isnan(atomscpu[i_res,i_atm,:]).any()):
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, atm, ChemData().num2aa[s],
                    ch, idx_pdb[i_res], atomscpu[i_res,i_atm,0], atomscpu[i_res,i_atm,1], atomscpu[i_res,i_atm,2],
                    1.0, Bfacts[i_res] ) )
                ctr += 1
    if bond_feats != None:
        atom_bonds = (bond_feats > 0) * (bond_feats <5)
        atom_bonds = atom_bonds.cpu()
        b, i, j = atom_bonds.nonzero(as_tuple=True)
        for start, end in zip(i,j):
            #print (start,end,bond_feats)
            f.write(f"CONECT{atom_idxs[int(start.cpu().numpy())]:5d}{atom_idxs[int(end.cpu().numpy())]:5d}\n")
    if modelnum is not None:
        f.write("ENDMDL\n")


### Create atom frames for FAPE loss calculation ###
def get_nxgraph(mol):
    '''build NetworkX graph from openbabel's OBMol'''

    N = mol.NumAtoms()

    # pairs of bonded atoms, openbabel indexes from 1 so readjust to indexing from 0
    bonds = [(bond.GetBeginAtomIdx()-1, bond.GetEndAtomIdx()-1) for bond in openbabel.OBMolBondIter(mol)]

    # connectivity graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(bonds)

    return G

def find_all_rigid_groups(bond_feats):
    """
    remove all single bonds from the graph and find connected components
    """
    rigid_atom_bonds = (bond_feats>1)*(bond_feats<5)
    rigid_atom_bonds_np = rigid_atom_bonds[0].cpu().numpy()
    G = nx.from_numpy_array(rigid_atom_bonds_np)
    connected_components = nx.connected_components(G)
    connected_components = [cc for cc in connected_components if len(cc)>2]
    connected_components = [torch.tensor(list(combinations(cc,2))) for cc in connected_components]
    if connected_components:
        connected_components = torch.cat(connected_components, dim=0)
    else:
        connected_components = None
    return connected_components

def find_all_paths_of_length_n(G : nx.Graph,
                               n : int,
                               **karg) -> torch.Tensor:
    '''find all paths of length N in a networkx graph
    https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph'''

    def findPaths(G,u,n):
        if n==0:
            return [[u]]
        paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
        return paths

    # all paths of length n
    allpaths = [tuple(p) if p[0]<p[-1] else tuple(reversed(p))
                for node in G for p in findPaths(G,node,n)]

    if 'omit_permutation' in karg.keys() and not karg['omit_permutation']:
        allpaths = [tuple(p) for node in G for p in findPaths(G,node,n)]

    # unique paths
    allpaths = list(set(allpaths))

    #return torch.tensor(allpaths)
    return allpaths

def get_atom_frames(msa, G, **karg):
    """choose a frame of 3 bonded atoms for each atom in the molecule, rule based system that chooses frame based on atom priorities"""
    query_seq = msa
    frames = find_all_paths_of_length_n(G, 2, **karg)
    selected_frames = []
    for n in range(msa.shape[0]):
        frames_with_n = [frame for frame in frames if n == frame[1]]

        # some chemical groups don't have two bonded heavy atoms; so choose a frame with an atom 2 bonds away
        if not frames_with_n:
            frames_with_n = [frame for frame in frames if n in frame]
        # if the atom isn't in a 3 atom frame, it should be ignored in loss calc, set all the atoms to n
        if not frames_with_n:
            selected_frames.append([(0,1),(0,1),(0, 1)])
            continue
        frame_priorities = []
        for frame in frames_with_n:
            # hacky but uses the "query_seq" to convert index of the atom into an "atom type" and converts that into a priority
            indices = [index for index in frame if index!=n]
            aas = [ChemData().num2aa[int(query_seq[index].numpy())] for index in indices]
            if 'omit_permutation' in karg.keys() and not karg['omit_permutation']:
                frame_priorities.append([ChemData().atom2frame_priority[aa] for aa in aas])
            else:
                frame_priorities.append(sorted([ChemData().atom2frame_priority[aa] for aa in aas]))


            
        # np.argsort doesn't sort tuples correctly so just sort a list of indices using a key
        sorted_indices = sorted(range(len(frame_priorities)), key=lambda i: frame_priorities[i])
        # calculate residue offset for frame
        frame = [(frame-n, 1) for frame in frames_with_n[sorted_indices[0]]]
        selected_frames.append(frame)
    assert msa.shape[0] == len(selected_frames)
    return torch.tensor(selected_frames).long()


### Generate bond features for small molecules ###
def get_bond_feats(mol):                                                                                 
    """creates 2d bond graph for small molecules"""
    N = mol.NumAtoms()
    bond_feats = torch.zeros((N, N)).long()

    for bond in openbabel.OBMolBondIter(mol):
        i,j = (bond.GetBeginAtomIdx()-1, bond.GetEndAtomIdx()-1)
        bond_feats[i,j] = bond.GetBondOrder() if not bond.IsAromatic() else 4
        bond_feats[j,i] = bond_feats[i,j]

    return bond_feats.long()

def get_protein_bond_feats(protein_L):
    """ creates protein residue connectivity graphs """
    bond_feats = torch.zeros((protein_L, protein_L))
    residues = torch.arange(protein_L-1)
    bond_feats[residues, residues+1] = 5
    bond_feats[residues+1, residues] = 5
    return bond_feats

def get_protein_bond_feats_from_idx(protein_L, idx_protein):
    """ creates protein residue connectivity graphs """
    bond_feats = torch.zeros((protein_L, protein_L))
    residues = torch.arange(protein_L-1)
    mask = idx_protein[:,None] == idx_protein[None,:]+1
    bond_feats[mask] = 5
    bond_feats[mask.T] = 5
    return bond_feats

def get_atomize_protein_bond_feats(i_start, msa, ra, n_res_atomize=5):
    """ 
    generate atom bond features for atomized residues 
    currently ignores long-range bonds like disulfides
    """
    ra2ind = {}
    for i, two_d in enumerate(ra):
        ra2ind[tuple(two_d.numpy())] = i
    N = len(ra2ind.keys())
    bond_feats = torch.zeros((N, N))
    for i, res in enumerate(msa[0, i_start:i_start+n_res_atomize]):
        for j, bond in enumerate(ChemData().aabonds[res]):
            start_idx = ChemData().aa2long[res].index(bond[0])
            end_idx = ChemData().aa2long[res].index(bond[1])
            if (i, start_idx) not in ra2ind or (i, end_idx) not in ra2ind:
                #skip bonds with atoms that aren't observed in the structure
                continue
            start_idx = ra2ind[(i, start_idx)]
            end_idx = ra2ind[(i, end_idx)]

            # maps the 2d index of the start and end indices to btype
            bond_feats[start_idx, end_idx] = ChemData().aabtypes[res][j]
            bond_feats[end_idx, start_idx] = ChemData().aabtypes[res][j]
        #accounting for peptide bonds
        if i > 0:
            if (i-1, 2) not in ra2ind or (i, 0) not in ra2ind:
                #skip bonds with atoms that aren't observed in the structure
                continue
            start_idx = ra2ind[(i-1, 2)]
            end_idx = ra2ind[(i, 0)]
            bond_feats[start_idx, end_idx] = ChemData().SINGLE_BOND
            bond_feats[end_idx, start_idx] = ChemData().SINGLE_BOND
    return bond_feats


### Generate atom features for proteins ###
def atomize_protein(i_start, msa, xyz, mask, n_res_atomize=5):
    """ given an index i_start, make the following flank residues into "atom" nodes """
    residues_atomize = msa[0, i_start:i_start+n_res_atomize]
    residues_atom_types = [ChemData().aa2elt[num][:14] for num in residues_atomize]
    residue_atomize_mask = mask[i_start:i_start+n_res_atomize].float() # mask of resolved atoms in the sidechain
    residue_atomize_allatom_mask = ChemData().allatom_mask[residues_atomize][:, :14] # the indices that have heavy atoms in that sidechain
    xyz_atomize = xyz[i_start:i_start+n_res_atomize]

    # handle symmetries
    xyz_alt = torch.zeros_like(xyz.unsqueeze(0))
    xyz_alt.scatter_(2, ChemData().long2alt[msa[0],:,None].repeat(1,1,1,3), xyz.unsqueeze(0))
    xyz_alt_atomize = xyz_alt[0, i_start:i_start+n_res_atomize]

    coords_stack = torch.stack((xyz_atomize, xyz_alt_atomize), dim=0)
    swaps = (coords_stack[0] == coords_stack[1]).all(dim=1).all(dim=1).squeeze() #checks whether theres a swap at each position
    swaps = torch.nonzero(~swaps).squeeze() # indices with a swap eg. [2,3]
    if swaps.numel() != 0:
        # if there are residues with alternate numbering scheme, create a stack of coordinate with each combo of swaps
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=UserWarning)
            combs = torch.combinations(torch.tensor([0,1]), r=swaps.numel(), with_replacement=True) #[[0,0], [0,1], [1,1]]
        stack = torch.stack((combs, swaps.repeat(swaps.numel()+1,1)), dim=-1).squeeze()
        coords_stack = coords_stack.repeat(swaps.numel()+1,1,1,1)
        nat_symm = coords_stack[0].repeat(swaps.numel()+1,1,1,1) # (N_symm, num_atomize_residues, natoms, 3)
        swapped_coords = coords_stack[stack[...,0], stack[...,1]].squeeze(1) #
        nat_symm[:,swaps] = swapped_coords
    else:
        nat_symm = xyz_atomize.unsqueeze(0)
    # every heavy atom that is in the sidechain is modelled but losses only applied to resolved atoms
    ra = residue_atomize_allatom_mask.nonzero()
    lig_seq = torch.tensor([ChemData().aa2num[residues_atom_types[r][a]] if residues_atom_types[r][a] in ChemData().aa2num else ChemData().aa2num["ATM"] for r,a in ra])
    ins = torch.zeros_like(lig_seq)

    r,a = ra.T
    lig_xyz = torch.zeros((len(ra), 3))
    lig_xyz = nat_symm[:, r, a]
    lig_mask = residue_atomize_mask[r, a].repeat(nat_symm.shape[0], 1)
    bond_feats = get_atomize_protein_bond_feats(i_start, msa, ra, n_res_atomize=n_res_atomize)
    #HACK: use networkx graph to make the atom frames, correct implementation will include frames with "residue atoms"
    G = nx.from_numpy_array(bond_feats.numpy())
        
    frames = get_atom_frames(lig_seq, G)
    chirals = get_atomize_protein_chirals(residues_atomize, lig_xyz[0], residue_atomize_allatom_mask, bond_feats)
    return lig_seq, ins, lig_xyz, lig_mask, frames, bond_feats, ra, chirals

def atomize_discontiguous_residues(idxs, msa, xyz, mask, bond_feats, same_chain, dslfs=None):
    """
    this atomizes multiple discontiguous residues at the same time, this is the default interface into atomizing residues 
    (using the non assembly dataset)
    """
    protein_L = msa.shape[1]
    seq_atomize_all = []
    ins_atomize_all = []
    xyz_atomize_all = []
    mask_atomize_all = []
    frames_atomize_all = []
    chirals_atomize_all = []
    prev_C_index = None
    total_num_atoms = 0
    sgs = {}
    for idx in idxs:
        seq_atomize, ins_atomize, xyz_atomize, mask_atomize, frames_atomize, bond_feats_atomize, resatom2idx, chirals_atomize = \
            atomize_protein(idx, msa, xyz, mask, n_res_atomize=1)
        r,_ = resatom2idx.T
        #print ('atomize_discontiguous_residues', idx, resatom2idx)
        last_C = torch.all(resatom2idx==torch.tensor([r[-1],2]),dim=1).nonzero()
        sgs[idx.item()] = torch.all(resatom2idx==torch.tensor([r[-1],5]),dim=1).nonzero()

        natoms = seq_atomize.shape[0]
        L = bond_feats.shape[0]

        sgs[idx.item()] = L+sgs[idx.item()]

        # update the chirals to be after all the other atoms (still need to update to put it behind all the proteins)
        chirals_atomize[:, :-1] += total_num_atoms

        seq_atomize_all.append(seq_atomize)
        ins_atomize_all.append(ins_atomize)
        xyz_atomize_all.append(xyz_atomize)
        mask_atomize_all.append(mask_atomize)
        frames_atomize_all.append(frames_atomize)
        chirals_atomize_all.append(chirals_atomize)

        N_term = idx ==  0
        C_term = idx == protein_L-1

        # update bond_feats every iteration, update all other features at the end 
        bond_feats_new = torch.zeros((L+natoms, L+natoms))
        bond_feats_new[:L, :L] = bond_feats
        bond_feats_new[L:, L:] = bond_feats_atomize
        # add bond between protein and atomized N
        if not N_term and idx-1 not in idxs:
            bond_feats_new[idx-1, L] = 6 # protein (backbone)-atom bond 
            bond_feats_new[L, idx-1] = 6 # protein (backbone)-atom bond 
        # add bond between protein and C, assumes every residue is being atomized one at a time (eg n_res_atomize=1)
        if not C_term and idx+1 not in idxs:
            bond_feats_new[idx+1, L+int(last_C.numpy())] = 6 # protein (backbone)-atom bond 
            bond_feats_new[L+int(last_C.numpy()), idx+1] = 6 # protein (backbone)-atom bond 
        # handle drawing peptide bond between contiguous atomized residues
        if idx-1 in idxs:
            if prev_C_index is None:
                raise ValueError("prev_C_index is None even though the previous residue has been atomized")
            bond_feats_new[prev_C_index, L] = 1 # single bond
            bond_feats_new[L, prev_C_index] = 1 # single bond

        prev_C_index =  L+int(last_C.numpy()) #update prev_C to draw bond to upcoming residue
        # update same_chain every iteration
        same_chain_new = torch.zeros((L+natoms, L+natoms))
        same_chain_new[:L, :L] = same_chain
        residues_in_prot_chain = same_chain[idx].squeeze().nonzero()

        same_chain_new[L:, residues_in_prot_chain] = 1
        same_chain_new[residues_in_prot_chain, L:] = 1
        same_chain_new[L:, L:] = 1

        bond_feats = bond_feats_new
        same_chain = same_chain_new
        total_num_atoms += natoms

    # disulfides
    if dslfs is not None:
        for i,j in dslfs:
            start_idx = sgs[i].item()
            end_idx = sgs[j].item()
            bond_feats[start_idx, end_idx] = 1
            bond_feats[end_idx, start_idx] = 1

    seq_atomize_all = torch.cat(seq_atomize_all)
    ins_atomize_all = torch.cat(ins_atomize_all)
    xyz_atomize_all = cartprodcat(xyz_atomize_all)
    mask_atomize_all = cartprodcat(mask_atomize_all)
    
    # frames were calculated per residue -- we want them over all residues in case there are contiguous residues
    bond_feats_sm = bond_feats[protein_L:][:, protein_L:]
    G = nx.from_numpy_array(bond_feats_sm.detach().cpu().numpy())
    frames_atomize_all = get_atom_frames(seq_atomize_all, G)
    
    # frames_atomize_all = torch.cat(frames_atomize_all)
    chirals_atomize_all = torch.cat(chirals_atomize_all)

    return seq_atomize_all, ins_atomize_all, xyz_atomize_all, mask_atomize_all, frames_atomize_all, chirals_atomize_all, \
        bond_feats, same_chain

def reindex_protein_feats_after_atomize(
    residues_to_atomize,
    prot_partners,
    msa, 
    ins,
    xyz,
    mask,
    bond_feats,
    idx,
    xyz_t,
    f1d_t,
    mask_t,
    same_chain,
    ch_label,
    Ls_prot,
    Ls_sm, 
    akeys_sm,
    remove_residue=True
):
    """
    Removes residues that have been atomized from protein features.
    """
    Ls = Ls_prot + Ls_sm
    chain_bins = [sum(Ls[:i]) for i in range(len(Ls)+1)]
    akeys_sm = list(itertools.chain.from_iterable(akeys_sm)) # list of list of tuples get flattened to a list of tuples

    # get tensor indices of atomized residues
    residue_chain_nums = []
    residue_indices = []
    for residue in residues_to_atomize:
        # residue object is a list of tuples:
        #   ((chain_letter, res_number, res_name), (chain_letter, xform_index))

        #### Need to identify what chain you're in to get correct res idx
        residue_chid_xf = residue[1]
        residue_chain_num = [p[:2] for p in prot_partners].index(residue_chid_xf)
        residue_index = (int(residue[0][1]) - 1) + sum(Ls_prot[:residue_chain_num])  # residues are 1 indexed in the cif files

        # skip residues with all backbone atoms masked
        if torch.sum(mask[0, residue_index, :3]) <3: continue

        residue_chain_nums.append(residue_chain_num)
        residue_indices.append(residue_index)
        atomize_N = residue[0] + ("N",)
        atomize_C = residue[0] + ("C",)

        N_index = akeys_sm.index(atomize_N) + sum(Ls_prot)
        C_index = akeys_sm.index(atomize_C) + sum(Ls_prot)

        # if first residue in chain, no extra bond feats to previous residue
        if residue_index != 0 and residue_index not in Ls_prot:
            bond_feats[residue_index-1, N_index] = 6
            bond_feats[N_index, residue_index-1] = 6

        # if residue is last in chain, no extra bonds feats to following residue
        if residue_index not in [L-1 for L in Ls_prot]:
            bond_feats[residue_index+1, C_index] = 6
            bond_feats[C_index,residue_index+1] = 6

        lig_chain_num = np.digitize([N_index], chain_bins)[0] -1 # np.digitize is 1 indexed
        same_chain[chain_bins[lig_chain_num]:chain_bins[lig_chain_num+1], \
                   chain_bins[residue_chain_num]: chain_bins[residue_chain_num+1]] = 1
        same_chain[chain_bins[residue_chain_num]: chain_bins[residue_chain_num+1], \
                   chain_bins[lig_chain_num]:chain_bins[lig_chain_num+1]] = 1

    if remove_residue:
        # remove atomized residues from feature tensors
        i_res = torch.tensor([i for i in range(sum(Ls)) if i not in residue_indices])
        msa = msa[:,i_res]
        ins = ins[:,i_res]
        xyz = xyz[:,i_res]
        mask = mask[:,i_res]
        bond_feats = bond_feats[i_res][:,i_res]
        idx = idx[i_res]
        xyz_t = xyz_t[:,i_res]
        f1d_t = f1d_t[:,i_res]
        mask_t = mask_t[:,i_res]
        same_chain = same_chain[i_res][:,i_res]
        ch_label = ch_label[i_res]

        for i_ch in residue_chain_nums:
            Ls_prot[i_ch] -= 1

    return msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label, Ls_prot, Ls_sm


def pop_protein_feats(residue_indices, msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label, Ls):
    """
    remove protein features for an arbitrary set of residue indices
    """
    pop = torch.ones((sum(Ls)))
    pop[residue_indices] = 0
    pop = pop.bool()

    msa = msa[:,pop]
    ins = ins[:,pop]
    xyz = xyz[:,pop]
    mask = mask[:,pop]
    bond_feats = bond_feats[pop][:,pop]
    idx = idx[pop]
    xyz_t = xyz_t[:,pop]
    f1d_t = f1d_t[:,pop]
    mask_t = mask_t[:,pop]
    same_chain = same_chain[pop][:,pop]
    ch_label = ch_label[pop]

    return msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label

def get_automorphs(mol, xyz_sm, mask_sm, max_symm=1000):
    """Enumerate atom symmetry permutations."""
    try:
        automorphs = openbabel.vvpairUIntUInt()
        openbabel.FindAutomorphisms(mol, automorphs)

        automorphs = torch.tensor(automorphs)
        n_symmetry = automorphs.shape[0]

        xyz_sm = xyz_sm[None].repeat(n_symmetry,1,1)
        mask_sm = mask_sm[None].repeat(n_symmetry,1)

        xyz_sm = torch.scatter(xyz_sm, 1, automorphs[:,:,0:1].repeat(1,1,3),
                                    torch.gather(xyz_sm,1,automorphs[:,:,1:2].repeat(1,1,3)))
        mask_sm = torch.scatter(mask_sm, 1, automorphs[:,:,0],
                            torch.gather(mask_sm, 1, automorphs[:,:,1]))
    except Exception as e:
        xyz_sm = xyz_sm[None]
        mask_sm = mask_sm[None]
    if xyz_sm.shape[0] > max_symm:
        xyz_sm = xyz_sm[:max_symm]
        mask_sm = mask_sm[:max_symm]
    return xyz_sm, mask_sm

def expand_xyz_sm_to_ntotal(xyz_sm, mask_sm, N_symmetry=None):
    """
    for small molecules, takes a 1d xyz tensor and converts to using N_total
    """
    N_symm_sm, L =  xyz_sm.shape[:2]
    if N_symmetry is None:
        N_symmetry = N_symm_sm
    xyz = torch.full((N_symmetry, L, ChemData().NTOTAL, 3), np.nan).float()
    xyz[:N_symm_sm, :, 1, :] = xyz_sm

    mask = torch.full((N_symmetry, L, ChemData().NTOTAL), False).bool()
    mask[:N_symm_sm, :, 1] = mask_sm
    return xyz, mask

def same_chain_2d_from_Ls(Ls):
    """Given list of chain lengths, returns binary matrix with 1 if two residues are on the same chain."""
    same_chain = torch.zeros((sum(Ls),sum(Ls))).long()
    i_curr = 0
    for L in Ls:
        same_chain[i_curr:i_curr+L, i_curr:i_curr+L] = 1
        i_curr += L
    return same_chain

def Ls_from_same_chain_2d(same_chain):
    """Given binary matrix indicating whether two residues are on same chain, returns list of chain lengths"""
    if len(same_chain.shape)==3: # remove batch dimension
        same_chain = same_chain.squeeze(0)
    Ls = []
    i_curr = 0
    while i_curr < len(same_chain):
        idx = torch.where(same_chain[i_curr])[0]
        Ls.append(int(idx[-1]-idx[0]+1))
        i_curr = idx[-1]+1
    return Ls

def get_prot_seqstring(ch, modres):
    """Return string representing amino acid sequence of a parsed CIF chain."""
    idx = [int(k[1]) for k in ch.atoms]
    i_min, i_max = np.min(idx), np.max(idx)
    L = i_max - i_min + 1
    seq = ["-"]*L

    for k,v in ch.atoms.items():
        i_res = int(k[1])-i_min
        if k[2] in ChemData().to1letter: # standard AA
            aa = ChemData().to1letter[k[2]]
        elif k[2] in modres and modres[k[2]] in ChemData().to1letter: # nonstandard AA, map to standard
            aa = ChemData().to1letter[modres[k[2]]]
        else: # unknown AA, still try to store BB atoms
            aa = 'X'
        seq[i_res] = aa
    return ''.join(seq)

def map_identical_prot_chains(partners, chains, modres):
    """Identifies which chain letters represent unique protein sequences,
    assigns a number to each unique sequence, and returns dicts mapping sequence
    numbers to chain letters and vice versa.
    
    Parameters
    ----------
    partners : list of tuples (partner, transform_index, num_contacts, partner_type)
        Information about neighboring chains to the query ligand in an
        assembly. This function will use the subset of these tuples that
        represent protein chains, where `partner_type = 'polypeptide(L)'`
        and `partner` contains the chain letter. `transform_index` is an
        integer index of the coordinate transform for each partner chain.
    chains : dict
        Dictionary mapping chain letters to cifutils.Chain objects representing
        the chains in a PDB entry.
    modres : dict
        Maps modified residue names to their canonical equivalents. Any
        modified residue will be converted to its standard equivalent and
        coordinates for atoms with matching names will be saved.

    Returns
    -------
    chnum2chlet : dict
        Dictionary mapping integers to lists of chain letters which represent
        identical chains
    """
    chlet2seq = OrderedDict()
    for p in partners:
        if p[-1] != 'polypeptide(L)': continue
        if p[0] not in chlet2seq:
            chlet2seq[p[0]] = get_prot_seqstring(chains[p[0]], modres)

    seq2chlet = OrderedDict()
    for chlet, seq in chlet2seq.items():
        if seq not in seq2chlet:
            seq2chlet[seq] = set()
        seq2chlet[seq].add(chlet)

    chnum2chlet = OrderedDict([(i,v) for i,(k,v) in enumerate(seq2chlet.items())])
    #chlet2chnum = OrderedDict([(chlet,chnum) for chnum,chlet_s in chnum2chlet.items() for chlet in chlet_s])

    return chnum2chlet 

def cartprodcat(X_s):
    """Concatenate list of tensors on dimension 1 while taking their cartesian product
    over dimension 0."""
    X = X_s[0]
    for X_ in X_s[1:]:
        N, L = X.shape[:2]
        N_, L_ = X_.shape[:2]
        X_out = torch.full((N, N_, L+L_,)+X.shape[2:], np.nan)
        for i in range(N):
            for j in range(N_):
                X_out[i,j] = torch.concat([X[i], X_[j]], dim=0)
        dims = (N*N_,L+L_,)+X.shape[2:]
        X = X_out.view(*dims)
    return X

def idx_from_Ls(Ls):
    """Generate residue indexes from a list of chain lengths, 
    with a chain gap offset between indexes for each chain."""
    idx = []
    offset = 0
    for L in Ls:
        idx.append(torch.arange(L)+offset)
        offset = offset+L+ChemData().CHAIN_GAP
    return torch.cat(idx, dim=0)


def bond_feats_from_Ls(Ls):
    """Generate protein (or DNA/RNA) bond features from a list of chain
    lengths"""
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    offset = 0
    for L_ in Ls:
        bond_feats[offset:offset+L_, offset:offset+L_] = get_protein_bond_feats(L_)
        offset += L_
    return bond_feats

def same_chain_from_bond_feats(bond_feats):
    """Return binary matrix indicating if pairs of residues are on same chain,
    given their bond features.
    """
    assert(len(bond_feats.shape)==2) # assume no batch dimension
    L = bond_feats.shape[0]
    same_chain = torch.zeros((L,L))
    G = nx.from_numpy_array(bond_feats.detach().cpu().numpy())
    for idx in nx.connected_components(G):
        idx = list(idx)
        for i in idx:
            same_chain[i,idx] = 1
    return same_chain


def kabsch(xyz1, xyz2, eps=1e-6):
    """Superimposes `xyz2` coordinates onto `xyz1`, returns RMSD and rotation matrix."""
    # center to CA centroid
    xyz1 = xyz1 - xyz1.mean(0)
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.linalg.svd(C)

    # get sign to ensure right-handedness
    d = torch.ones([3,3])
    d[:,-1] = torch.sign(torch.linalg.det(V)*torch.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U

    L = xyz2_.shape[0]

    rmsd = torch.sqrt(torch.sum((xyz2_-xyz1)*(xyz2_-xyz1), axis=(0,1)) / L + eps)

    return rmsd, U


