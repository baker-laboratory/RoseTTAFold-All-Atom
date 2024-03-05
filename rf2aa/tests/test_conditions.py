import torch
import pandas as pd
import numpy as np
import itertools
from collections import OrderedDict
from hydra import initialize, compose

from rf2aa.setup_model import trainer_factory, seed_all
from rf2aa.chemical import ChemicalData as ChemData

# configurations to test
configs = ["legacy_train"]
datasets = ["compl", "na_compl", "rna", "sm_compl", "sm_compl_covale", "sm_compl_asmb"]

cfg_overrides = [
    "loader_params.p_msa_mask=0.0", 
    "loader_params.crop=100000",
    "loader_params.mintplt=0",
    "loader_params.maxtplt=2"
]

def make_deterministic(seed=0):
    seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_dataset_names():
    data = {}
    for name in datasets:
        data[name] = [name]
    return data

# set up models for regression tests
def setup_models(device="cpu"):
    models, chem_cfgs = [], []
    for config in configs:
        with initialize(version_base=None, config_path="../config/train"):
            cfg = compose(config_name=config, overrides=cfg_overrides)

            # initializing the model needs the chemical DB initialized.  Force a reload
            ChemData.reset()
            ChemData(cfg.chem_params)

            trainer = trainer_factory[cfg.experiment.trainer](cfg)
            seed_all()
            trainer.construct_model(device=device)
            models.append(trainer.model)
            chem_cfgs.append(cfg.chem_params)
            trainer = None 

    return dict(zip(configs, (zip(configs, models, chem_cfgs))))

# set up job array for regression
def setup_array(datasets, models, device="cpu"):
    test_data = setup_dataset_names()
    test_models = setup_models(device=device)
    test_data = [test_data[dataset] for dataset in datasets]
    test_models = [test_models[model] for model in models]
    return (list(itertools.product(test_data, test_models)))

def random_param_init(model):
    seed_all()
    with torch.no_grad():
        fake_state_dict = OrderedDict()
        for name, param in model.model.named_parameters():
            fake_state_dict[name] = torch.randn_like(param)
        model.model.load_state_dict(fake_state_dict)
        model.shadow.load_state_dict(fake_state_dict)
    return model

def dataset_pickle_path(dataset_name):
    return f"test_pickles/data/{dataset_name}_regression.pt"

def model_pickle_path(dataset_name, model_name):
    return f"test_pickles/model/{model_name}_{dataset_name}_regression.pt"

def loss_pickle_path(dataset_name, model_name, loss_name):
    return f"test_pickles/loss/{loss_name}_{model_name}_{dataset_name}_regression.pt"