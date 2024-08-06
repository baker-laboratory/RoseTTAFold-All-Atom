import torch

from rf2aa.data.data_loader import RawInputData
from rf2aa.data.data_loader_utils import blank_template, TemplFeaturize
from rf2aa.data.parsers import parse_a3m, parse_templates_raw
from rf2aa.data.preprocessing import make_msa
from rf2aa.util import get_protein_bond_feats


def get_templates(
    qlen,
    ffdb,
    hhr_fn,
    atab_fn,
    seqID_cut,
    n_templ,
    pick_top: bool = True,
    offset: int = 0,
    random_noise: float = 5.0,
    deterministic: bool = False,
):
    (
        xyz_parsed,
        mask_parsed,
        qmap_parsed,
        f0d_parsed,
        f1d_parsed,
        seq_parsed,
        ids_parsed,
    ) = parse_templates_raw(ffdb, hhr_fn=hhr_fn, atab_fn=atab_fn)
    tplt = {
        "xyz": xyz_parsed.unsqueeze(0),
        "mask": mask_parsed.unsqueeze(0),
        "qmap": qmap_parsed.unsqueeze(0),
        "f0d": f0d_parsed.unsqueeze(0),
        "f1d": f1d_parsed.unsqueeze(0),
        "seq": seq_parsed.unsqueeze(0),
        "ids": ids_parsed,
    }
    params = {
        "SEQID": seqID_cut,
    }
    return TemplFeaturize(
        tplt,
        qlen,
        params,
        offset=offset,
        npick=n_templ,
        pick_top=pick_top,
        random_noise=random_noise,
        deterministic=deterministic,
    )


def load_protein(msa_file, hhr_fn, atab_fn, model_runner):
    # print(model_runner.config)
    if model_runner.config.loader_params.MAXSEQ == 0:
        print('hacking single chain single sequence')
        print('model_runner.config:')
        print(model_runner.config)
        print('model_runner:')
        for key, val in model_runner.__dict__.items():
            print(key)
            if key == 'ffdb':
                print('\t', val)
        chid = list(model_runner.config.protein_inputs.keys())[0]
        path_ = model_runner.config.protein_inputs[chid]["fasta_file"]
        msa, ins, taxIDs = parse_a3m(path_)
    else:
        msa, ins, taxIDs = parse_a3m(msa_file)
    # NOTE: this next line is a bug, but is the way that
    # the code is written in the original implementation!
    ins[0] = msa[0]

    print('protein msa shape', msa.shape)

    L = msa.shape[1]
    if hhr_fn is None or atab_fn is None or model_runner.config.loader_params.n_templ == 0:
        print("No templates provided")
        xyz_t, t1d, mask_t, _ = blank_template(1, L)
    else:
        xyz_t, t1d, mask_t, _ = get_templates(
            L,
            model_runner.ffdb,
            hhr_fn,
            atab_fn,
            seqID_cut=model_runner.config.loader_params.seqid,
            n_templ=model_runner.config.loader_params.n_templ,
            deterministic=model_runner.deterministic,
        )

    bond_feats = get_protein_bond_feats(L)
    chirals = torch.zeros(0, 5)
    atom_frames = torch.zeros(0, 3, 2)

    print('protein xyz_t shape', xyz_t.shape)
    return RawInputData(
        torch.from_numpy(msa),
        torch.from_numpy(ins),
        bond_feats,
        xyz_t,
        mask_t,
        t1d,
        chirals,
        atom_frames,
        taxids=taxIDs,
    )

def generate_msa_and_load_protein(fasta_file, chain, model_runner):
    msa_file, hhr_file, atab_file = make_msa(fasta_file, chain, model_runner)
    return load_protein(str(msa_file), str(hhr_file), str(atab_file), model_runner)
