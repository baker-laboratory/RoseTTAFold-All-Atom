import os
import torch
import pytest
import warnings
warnings.filterwarnings("ignore")

from rf2aa.data.dataloader_adaptor import prepare_input
from rf2aa.training.recycling import run_model_forward_legacy
from rf2aa.tensor_util import assert_equal
from rf2aa.tests.test_conditions import setup_array,\
      make_deterministic, dataset_pickle_path, model_pickle_path
from rf2aa.util_module import XYZConverter
from rf2aa.chemical import ChemicalData as ChemData


# goal is to test all the configs on a broad set of datasets

gpu = "cuda:0" if torch.cuda.is_available() else "cpu"

legacy_test_conditions = setup_array(["na_compl", "rna", "sm_compl", "sm_compl_covale"], ["legacy_train"], device=gpu)

@pytest.mark.parametrize("example,model", legacy_test_conditions)
def test_regression_legacy(example, model):
    dataset_name, dataset_inputs, model_name, model = setup_test(example, model)
    make_deterministic()
    output_i = run_model_forward_legacy(model, dataset_inputs, gpu)
    model_pickle = model_pickle_path(dataset_name, model_name)
    output_names = ("logits_c6d", "logits_aa", "logits_pae", \
                        "logits_pde", "p_bind", "xyz", "alpha", "xyz_allatom", \
                        "lddt", "seq", "pair", "state")
    
    if not os.path.exists(model_pickle):
        torch.save(output_i, model_pickle)
    else:
        output_regression = torch.load(model_pickle, map_location=gpu)
        for idx, output in enumerate(output_i):
            got = output
            want = output_regression[idx]
            if output_names[idx] == "logits_c6d":
                for i in range(len(want)):
                    
                    got_i = got[i]
                    want_i = want[i]
                    try:
                        assert_equal(got_i, want_i)
                    except Exception as e:
                        raise ValueError(f"{output_names[idx]} not same for model: {model_name} on dataset: {dataset_name}") from e
            elif output_names[idx] in ["alpha", "xyz_allatom", "seq", "pair", "state"]:
                try:
                    assert torch.allclose(got, want, atol=1e-4)
                except Exception as e:
                    raise ValueError(f"{output_names[idx]} not same for model: {model_name} on dataset: {dataset_name}") from e
            else:
                try:
                    assert_equal(got, want)
                except Exception as e:
                    raise ValueError(f"{output_names[idx]} not same for model: {model_name} on dataset: {dataset_name}") from e

def setup_test(example, model):
    model_name, model, config = model

    # initialize chemical database
    ChemData.reset() # force reload chemical data
    ChemData(config)

    model = model.to(gpu)
    dataset_name = example[0]
    dataloader_inputs = torch.load(dataset_pickle_path(dataset_name), map_location=gpu)
    xyz_converter = XYZConverter().to(gpu)
    task, item, network_input, true_crds, mask_crds, msa, mask_msa, unclamp, \
        negative, symmRs, Lasu, ch_label = prepare_input(dataloader_inputs,xyz_converter, gpu)
    return dataset_name, network_input, model_name, model

