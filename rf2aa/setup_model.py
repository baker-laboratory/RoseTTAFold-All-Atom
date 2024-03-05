import torch
import numpy as np


import hydra
import os

from rf2aa.training.EMA import EMA
from rf2aa.model.RoseTTAFoldModel import RoseTTAFoldModule
from rf2aa.util_module import XYZConverter
from rf2aa.chemical import ChemicalData as ChemData

#TODO: control environment variables from config
# limit thread counts
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"

## To reproduce errors
import random

def seed_all(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

torch.set_num_threads(4)
#torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, config) -> None:
        self.config = config

        assert self.config.ddp_params.batch_size == 1, "batch size is assumed to be 1"
        if self.config.experiment.output_dir is not None:
            self.output_dir = self.config.experiment.output_dir 
        else:
            self.output_dir = "models/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def move_constants_to_device(self, gpu):
        self.fi_dev = ChemData().frame_indices.to(gpu)
        self.xyz_converter = XYZConverter().to(gpu)

        self.l2a = ChemData().long2alt.to(gpu)
        self.aamask = ChemData().allatom_mask.to(gpu)
        self.num_bonds = ChemData().num_bonds.to(gpu)
        self.atom_type_index = ChemData().atom_type_index.to(gpu)
        self.ljlk_parameters = ChemData().ljlk_parameters.to(gpu)
        self.lj_correction_parameters = ChemData().lj_correction_parameters.to(gpu)
        self.hbtypes = ChemData().hbtypes.to(gpu)
        self.hbbaseatoms = ChemData().hbbaseatoms.to(gpu)
        self.hbpolys = ChemData().hbpolys.to(gpu)
        self.cb_len = ChemData().cb_length_t.to(gpu)
        self.cb_ang = ChemData().cb_angle_t.to(gpu)
        self.cb_tor = ChemData().cb_torsion_t.to(gpu)

class LegacyTrainer(Trainer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def construct_model(self, device="cpu"):
        self.model = RoseTTAFoldModule(
            **self.config.legacy_model_param,
            aamask = ChemData().allatom_mask.to(device),
            atom_type_index = ChemData().atom_type_index.to(device),
            ljlk_parameters = ChemData().ljlk_parameters.to(device),
            lj_correction_parameters = ChemData().lj_correction_parameters.to(device),
            num_bonds = ChemData().num_bonds.to(device),
            cb_len = ChemData().cb_length_t.to(device),
            cb_ang = ChemData().cb_angle_t.to(device),
            cb_tor = ChemData().cb_torsion_t.to(device),

        ).to(device)
        if self.config.training_params.EMA is not None:
            self.model = EMA(self.model, self.config.training_params.EMA)

@hydra.main(version_base=None, config_path='config/train')
def main(config):
    seed_all()
    trainer = trainer_factory[config.experiment.trainer](config=config)

trainer_factory = {
    "legacy": LegacyTrainer,
}

if __name__ == "__main__":
    main()
