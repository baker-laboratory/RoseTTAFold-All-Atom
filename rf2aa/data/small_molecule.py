import torch

from rf2aa.data.data_loader import RawInputData
from rf2aa.data.data_loader_utils import blank_template
from rf2aa.data.parsers import parse_mol
from rf2aa.kinematics import get_chirals
from rf2aa.util import get_bond_feats, get_nxgraph, get_atom_frames


def load_small_molecule(input_file, input_type, model_runner):
    if input_type == "smiles":
        is_string = True
    else:
        is_string = False

    obmol, msa, ins, xyz, mask = parse_mol(
        input_file, filetype=input_type, string=is_string, generate_conformer=True
    )
    return compute_features_from_obmol(obmol, msa, xyz, model_runner) 

def compute_features_from_obmol(obmol, msa, xyz, model_runner):
    L = msa.shape[0]
    ins = torch.zeros_like(msa)
    bond_feats = get_bond_feats(obmol)

    xyz_t, t1d, mask_t, _ = blank_template(
        model_runner.config.loader_params.n_templ,
        L,
        deterministic=model_runner.deterministic,
    )
    chirals = get_chirals(obmol, xyz[0])
    G = get_nxgraph(obmol)
    atom_frames = get_atom_frames(msa, G)
    msa, ins = msa[None], ins[None]
    return RawInputData(
        msa, ins, bond_feats, xyz_t, mask_t, t1d, chirals, atom_frames, taxids=None
    )

def remove_leaving_atoms(input, is_leaving):
    keep = ~is_leaving
    return input.keep_features(keep)
