import os
import rf2aa as rf
import ipd

def path(fname):
   return os.path.abspath(f'{rf.projdir}/tests/data/{fname}')

def load(fname):
   return ipd.dev.load(path(fname))
