import os
import rf2aa as rf
import ipd
import torch
from rf2aa.tests import sym
from rf2aa.set_seed import seed_all
from rf2aa.tests.testutil import *

def make_deterministic(seed=0):
    seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def path(fname):
    return os.path.abspath(f'{rf.projdir}/data/tests/{fname}')

def load(fname):
    return ipd.dev.load(path(fname))

__all__ = ['sym', 'make_deterministic', 'path', 'load']
