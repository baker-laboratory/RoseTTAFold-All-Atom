import numpy as np
import torch

from rf2aa.data.parsers import parse_mixed_fasta, parse_multichain_fasta
from rf2aa.data.data_loader_utils import merge_a3m_hetero, merge_a3m_homo, blank_template
from rf2aa.data.data_loader import RawInputData
from rf2aa.util import get_protein_bond_feats

def load_nucleic_acid(fasta_fn, input_type, model_runner):
    if input_type not in ["dna", "rna"]:
        raise ValueError("Only DNA and RNA inputs allowed for nucleic acids")
    if input_type == "dna":
        dna_alphabet = True
        rna_alphabet = False
    elif input_type == "rna":
        dna_alphabet = False
        rna_alphabet = True

    loader_params = model_runner.config.loader_params
    msa, ins, L = parse_multichain_fasta(fasta_fn, rna_alphabet=rna_alphabet, dna_alphabet=dna_alphabet)
    if (msa.shape[0] > loader_params["MAXSEQ"]):
        idxs_tokeep = np.random.permutation(msa.shape[0])[:loader_params["MAXSEQ"]]
        idxs_tokeep[0] = 0
        msa = msa[idxs_tokeep]
        ins = ins[idxs_tokeep]
    if len(L) > 1:
        raise ValueError("Please provide separate fasta files for each nucleic acid chain")
    L = L[0]
    xyz_t, t1d, mask_t, _ = blank_template(loader_params["n_templ"], L)


    bond_feats = get_protein_bond_feats(L)
    chirals = torch.zeros(0, 5)
    atom_frames = torch.zeros(0, 3, 2)
    
    return RawInputData(
        torch.from_numpy(msa),
        torch.from_numpy(ins),
        bond_feats,
        xyz_t,
        mask_t,
        t1d,
        chirals,
        atom_frames,
        taxids=None,
    )