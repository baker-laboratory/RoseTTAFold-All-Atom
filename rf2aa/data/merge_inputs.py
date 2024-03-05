import torch

from rf2aa.data.data_loader_utils import merge_a3m_hetero, merge_a3m_homo, merge_hetero_templates, get_term_feats
from rf2aa.data.data_loader import RawInputData
from rf2aa.util import center_and_realign_missing, same_chain_from_bond_feats, random_rot_trans, idx_from_Ls


def merge_protein_inputs(protein_inputs, deterministic: bool = False):
    if len(protein_inputs) == 0:
        return None,[]
    elif len(protein_inputs) == 1:
        chain = list(protein_inputs.keys())[0]
        input = list(protein_inputs.values())[0]
        xyz_t = input.xyz_t
        xyz_t[0:1] = random_rot_trans(xyz_t[0:1], deterministic=deterministic)
        input.xyz_t = xyz_t
        return input, [(chain, input.length())]
    # handle merging MSAs and such
    # first determine which sequence are identical, then which one have mergeable MSAs
    # then cat the templates, other feats
    pass

def merge_na_inputs(na_inputs):
    # should just be trivially catting features
    running_inputs = None
    chain_lengths = []
    for chid, input in na_inputs.items():
        running_inputs = merge_two_inputs(running_inputs, input)
        chain_lengths.append((chid, input.length()))
    return running_inputs, chain_lengths

def merge_sm_inputs(sm_inputs):
    # should be trivially catting features
    running_inputs = None
    chain_lengths = []
    for chid, input in sm_inputs.items():
        running_inputs = merge_two_inputs(running_inputs, input)
        chain_lengths.append((chid, input.length()))
    return running_inputs, chain_lengths

def merge_two_inputs(first_input, second_input):
    # merges two arbitrary inputs of data types
    if first_input is None and second_input is None:
        return None
    elif first_input is None:
        return second_input
    elif second_input is None:
        return first_input

    Ls = [first_input.length(), second_input.length()]
    L_total = sum(Ls)
    # merge msas

    a3m_first = {
        "msa": first_input.msa,
        "ins": first_input.ins,
    }
    a3m_second = {
        "msa": second_input.msa,
        "ins": second_input.ins,
    }
    a3m = merge_a3m_hetero(a3m_first, a3m_second, Ls)
    # merge bond_feats
    bond_feats = torch.zeros((L_total, L_total)).long()
    offset = 0
    for bf in [first_input.bond_feats, second_input.bond_feats]:
        L = bf.shape[0]
        bond_feats[offset:offset+L, offset:offset+L] = bf
        offset += L

    # merge templates
    xyz_t = torch.cat([first_input.xyz_t, second_input.xyz_t],dim=1)
    t1d = torch.cat([first_input.t1d, second_input.t1d],dim=1)
    mask_t = torch.cat([first_input.mask_t, second_input.mask_t],dim=1)

    # handle chirals (need to residue offset)
    if second_input.chirals.shape[0] > 0 :
        second_input.chirals[:, :-1] = second_input.chirals[:, :-1] + first_input.length()
    chirals =  torch.cat([first_input.chirals, second_input.chirals])

    # cat atom frames
    atom_frames = torch.cat([first_input.atom_frames, second_input.atom_frames])
    # return new object
    return RawInputData(
        a3m["msa"],
        a3m["ins"],
        bond_feats,
        xyz_t,
        mask_t,
        t1d,
        chirals,
        atom_frames,
        taxids=None
    )

def merge_all(
    protein_inputs,
    na_inputs, 
    sm_inputs,
    residues_to_atomize,
    deterministic: bool = False,
):

    #protein_lengths = [protein_input.length() for protein_input in protein_inputs.values()]
    #na_lengths = [na_input.length() for na_input in na_inputs.values()]
    #sm_lengths = [sm_input.length() for sm_input in sm_inputs.values()]
    #all_lengths = protein_lengths + na_lengths + sm_lengths
    
    #term_info = get_term_feats(all_lengths)
    #term_info[sum(protein_lengths):, :] = 0

    protein_inputs, protein_chain_lengths = merge_protein_inputs(protein_inputs, deterministic=deterministic)
    
    na_inputs, na_chain_lengths = merge_na_inputs(na_inputs)
    sm_inputs, sm_chain_lengths = merge_sm_inputs(sm_inputs)
    if protein_inputs is None and na_inputs is None and sm_inputs is None:
        raise ValueError("No valid inputs were provided") 
    running_inputs = merge_two_inputs(protein_inputs, na_inputs) #could handle pairing protein/NA MSAs here
    running_inputs = merge_two_inputs(running_inputs, sm_inputs)

    all_chain_lengths = protein_chain_lengths + na_chain_lengths + sm_chain_lengths
    running_inputs.chain_lengths = all_chain_lengths

    all_lengths = get_Ls_from_chain_lengths(running_inputs.chain_lengths)
    protein_lengths = get_Ls_from_chain_lengths(protein_chain_lengths)
    term_info = get_term_feats(all_lengths)
    term_info[sum(protein_lengths):, :] = 0
    running_inputs.term_info = term_info

    xyz_t = running_inputs.xyz_t
    mask_t = running_inputs.mask_t
    
    same_chain = same_chain = same_chain_from_bond_feats(running_inputs.bond_feats)
    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [center_and_realign_missing(xyz_t[i], mask_t[i], same_chain=same_chain) for i in range(ntempl)]
    )
    xyz_t = torch.nan_to_num(xyz_t)
    running_inputs.xyz_t = xyz_t
    running_inputs.idx = idx_from_Ls(all_lengths)

    # after everything is merged need to add bond feats for covales
    # reindex protein feats function
    if residues_to_atomize:
        running_inputs.update_protein_features_after_atomize(residues_to_atomize)

    return running_inputs

def get_Ls_from_chain_lengths(chain_lengths):
    return [val[1] for val in chain_lengths]

