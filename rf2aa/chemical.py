import sys, os, gzip, json
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import threading

import scipy.sparse
import scipy.sparse.csgraph

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'

from rf2aa.scoring import *

# process ideal frames
def make_frame(X, Y):
    Xn = X / torch.linalg.norm(X)
    Y = Y - torch.dot(Y, Xn) * Xn
    Yn = Y / torch.linalg.norm(Y)
    Z = torch.cross(Xn,Yn)
    Zn =  Z / torch.linalg.norm(Z)
    return torch.stack((Xn,Yn,Zn), dim=-1)

# ang between vectors
def th_ang_v(ab,bc,eps:float=1e-8):
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)
    ab, bc = th_N(ab),th_N(bc)
    cos_angle = torch.clamp( (ab*bc).sum(-1), -1, 1)
    sin_angle = torch.sqrt(1-cos_angle.square() + eps)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

# dihedral between vectors
def th_dih_v(ab,bc,cd):
    def th_cross(a,b):
        a,b = torch.broadcast_tensors(a,b)
        return torch.cross(a,b, dim=-1)
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)

    ab, bc, cd = th_N(ab),th_N(bc),th_N(cd)
    n1 = th_N( th_cross(ab,bc) )
    n2 = th_N( th_cross(bc,cd) )
    sin_angle = (th_cross(n1,bc)*n2).sum(-1)
    cos_angle = (n1*n2).sum(-1)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

# dihedral between points
def th_dih(a,b,c,d):
    return th_dih_v(a-b,b-c,c-d)

# helper function to load chemical database with specified config
def initialize_chemdata(config, worker_id=None):
    ChemicalData(config)

# A singleton class that stores chemical data
class ChemicalData:
    _ChemicalDataStore = None

    # initialize once
    def __new__(cls, *args, **kwds):
        #print ('__new__', cls._ChemicalDataStore)
        if not cls._ChemicalDataStore:
           cls._ChemicalDataStore = super().__new__(cls)
           cls._ChemicalDataStore.init(*args, **kwds)
        return cls._ChemicalDataStore

    @classmethod
    def reset(cls):
        cls._ChemicalDataStore = None

    def init(self, params):
        self.load_base_data(params)
        self.load_derived_data(params)
        self.params = params

    def load_base_data(self, params):
        self.NAATOKENS = 20+2+10+1+47 # 20 AAs, UNK, MASK, 8 NAs,HIS_D, 47 atoms
        self.UNKINDEX = 20  # residue unknown
        self.MASKINDEX = 21  # protein mask

        self.NHEAVYPROT = 14
        self.NHEAVY = 23
        self.NTOTAL = 36
        self.NNAPROTAAS = 32
        self.NPROTAAS = 22 # include UNK/MAS

        self.CHAIN_GAP = 200

        # internal coords
        self.NPROTTORS = 7
        self.NPROTANGS = 3
        self.NNATORS = 10
        self.NTOTALTORS = self.NPROTTORS+self.NNATORS
        self.NTOTALDOFS = self.NTOTALTORS+self.NPROTANGS

        #bond types
        self.num2btype = [0,1,2,3,4,5,6,7] # UNK, SINGLE, DOUBLE, TRIPLE, AROMATIC, 
                                           # PEPTIDE/NA BACKBONE, PROTEIN-LIGAND (PEPTIDE), OTHER
        self.NBTYPES = len(self.num2btype)

        self.num2aa=[
            'ALA','ARG','ASN','ASP','CYS',
            'GLN','GLU','GLY','HIS','ILE',
            'LEU','LYS','MET','PHE','PRO',
            'SER','THR','TRP','TYR','VAL',
            'UNK','MAS',
            ' DA',' DC',' DG',' DT', ' DX',
            ' RA',' RC',' RG',' RU', ' RX',
            'HIS_D', # only used for cart_bonded
            'Al', 'As', 'Au', 'B',
            'Be', 'Br', 'C', 'Ca', 'Cl',
            'Co', 'Cr', 'Cu', 'F', 'Fe',
            'Hg', 'I', 'Ir', 'K', 'Li', 'Mg',
            'Mn', 'Mo', 'N', 'Ni', 'O',
            'Os', 'P', 'Pb', 'Pd', 'Pr',
            'Pt', 'Re', 'Rh', 'Ru', 'S',
            'Sb', 'Se', 'Si', 'Sn', 'Tb',
            'Te', 'U', 'W', 'V', 'Y', 'Zn',
            'ATM'
        ]

        self.aa2num= {x:i for i,x in enumerate(self.num2aa)}
        self.aa2num['MEN'] = 20

        # Mapping 3 letter AA to 1 letter AA (e.g. ALA to A)
        self.one_letter = ["A", "R", "N", "D", "C", \
                           "Q", "E", "G", "H", "I", \
                           "L", "K", "M", "F", "P", \
                           "S", "T", "W", "Y", "V", "?", "-"]

        self.n_non_protein = len(self.num2aa) - len(self.one_letter)

        self.aa_321 = {a:b for a,b in zip(self.num2aa,self.one_letter+['a']*self.n_non_protein)}

        self.frame_priority2atom = [
            "F",  "Cl", "Br", "I",  "O",  "S",  "Se", "Te", "N",  "P",  "As", "Sb", 
            "C",  "Si", "Sn", "Pb", "B",  "Al", "Zn", "Hg", "Cu", "Au", "Ni", "Pd", 
            "Pt", "Co", "Rh", "Ir", "Pr", "Fe", "Ru", "Os", "Mn", "Re", "Cr", "Mo", 
            "W",  "V",  "U",  "Tb", "Y",  "Be", "Mg", "Ca", "Li", "K",  "ATM"]

        # these atomic numbers are incorrect, but keeping for fold&dock3 and correcting it 
        # in util.writepdb() during output.
        self.atom_num= [
            9,    17,   35,   53,   8,    16,   34,   52,   7,    15,   33,   51, 
            6,    14,   32,   50,   82,   5,    13,   30,   80,   29,   79,   28, 
            46,   78,   27,   45,   77,   26,   44,   76,   25,   75,   24,   42, 
            23,   74,   92,   65,   39,   4,    12,   20,   3,    19,   0] # in same order as frame priority

        self.atom2frame_priority = {x:i for i,x in enumerate(self.frame_priority2atom)}
        self.atomnum2atomtype = dict(zip(self.atom_num, self.frame_priority2atom))
        self.atomtype2atomnum = {v:k for k,v in self.atomnum2atomtype.items()}

        self.to1letter = {
            "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
            "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
            "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
            "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V',
            "DA":'a', "DC":'c', "DG":'g', "DT":'t',
            "A":'b', "C":'d', "G":'h', "U":'u',
        }

        # this is taken from a query string for a link named "metals in PDB" on BioLiP website 
        # hopefully they put a lot of thought into it
        self.METAL_RES_NAMES = ['LA','NI','3CO','K','CR','ZN','CD','PD','TB','YT3','OS','EU','NA','RB','W','YB','HO3',
                  'CE','MN','TL','LI','MN3','AU3','AU','EU3','AL','3NI','FE2','PT','FE','CA','AG','CU1',
                  'LU','HG','CO','SR','MG','PB','CS','GA','BA','SM','SB','CU','MO','CU2']

        # full sc atom representation
        if (not params.use_phospate_frames_for_NA):
            # USE RIBOSE FRAME
            self.aa2long=[
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #0  ala
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), #1  arg
                (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), #2  asn
                (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), #3  asp
                (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), #4  cys
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), #5  gln
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), #6  glu
                (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), #7  gly
                (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","2HE ",  None,  None,  None,  None,  None,  None), #8  his
                (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), #9  ile
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), #10 leu
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), #11 lys
                (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), #12 met
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","2HD ","1HE ","2HE "," HZ ",  None,  None,  None,  None), #13 phe
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), #14 pro
                (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), #15 ser
                (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), #16 thr
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE "," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), #17 trp
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE ","2HE ","2HD "," HH ",  None,  None,  None,  None), #18 tyr
                (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), #19 val
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #20 unk
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #21 mask

                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N6 ",  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #22  DA
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #23  DC
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N2 "," O6 ",  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #24  DG
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C7 "," C6 ",  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H3 "," H71"," H72"," H73"," H6 ",  None), #25  DT
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'",  None,  None,  None,  None,  None,  None), #26  DX (unk DNA)
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," N3 "," C4 "," C5 "," C6 "," N6 "," N7 "," C8 "," N9 ",  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #27   A
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #28   C
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," N2 "," N3 "," C4 "," C5 "," C6 "," O6 "," N7 "," C8 "," N9 "," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #29   G
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H3 "," H5 "," H6 ",  None,  None,  None), #30   U
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'",  None,  None,  None,  None,  None,  None), #31  RX (unk RNA)

                (" N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","1HD ",  None,  None,  None,  None,  None,  None), #-1 his_d
            ]
            

            # build the "alternate" sc mapping
            self.aa2longalt=[
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ala
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), # arg
                (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), # asn
                (" N  "," CA "," C  "," O  "," CB "," CG "," OD2"," OD1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
                (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), # gln
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE2"," OE1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), # glu
                (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
                (" N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","2HE ",  None,  None,  None,  None,  None,  None), # his
                (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), # ile
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), # leu
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), # lys
                (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), # met
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","2HD ","2HE "," HZ ","1HE ","1HD "," HA ","1HB ","2HB ",  None,  None,  None,  None), # phe
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), # pro
                (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
                (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), # thr
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE "," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), # trp
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ "," OH ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","2HE ","1HE ","1HD "," HH ",  None,  None,  None,  None), # tyr
                (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), # val
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # unk
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # mask
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N6 ",  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #22  DA
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #23  DC
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N2 "," O6 ",  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #24  DG
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C7 "," C6 ",  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H3 "," H71"," H72"," H73"," H6 ",  None), #25  DT
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'",  None,  None,  None,  None,  None,  None), #26  DX (unk DNA)
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," N3 "," C4 "," C5 "," C6 "," N6 "," N7 "," C8 "," N9 ",  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #27   A
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #28   C
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," N2 "," N3 "," C4 "," C5 "," C6 "," O6 "," N7 "," C8 "," N9 "," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #29   G
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H3 "," H5 "," H6 ",  None,  None,  None), #30   U
                (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'"," O2'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'",  None,  None,  None,  None,  None,  None), #31  RX (unk RNA)
            ]

            self.aa2type = [
                ("Nbb", "CAbb","CObb","OCbb","CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # ala
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "CH2", "NtrR","aroC","Narg","Narg",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol","Hpol","Hpol","Hpol"), # arg
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CNH2","ONH2","NH2O",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hpol","Hpol",  None,  None,  None,  None,  None,  None,  None), # asn
                ("Nbb", "CAbb","CObb","OCbb","CH2", "COO", "OOC", "OOC",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
                ("Nbb", "CAbb","CObb","OCbb","CH2", "SH1",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","HS",    None,  None,  None,  None,  None,  None,  None,  None), # cys
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "CNH2","ONH2","NH2O",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol",  None,  None,  None,  None,  None), # gln
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "COO", "OOC", "OOC",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None), # glu
                ("Nbb", "CAbb","CObb","OCbb",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "Nhis","aroC","aroC","Ntrp",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hpol","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # his
                ("Nbb", "CAbb","CObb","OCbb","CH1", "CH2", "CH3", "CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None), # ile
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH1", "CH3", "CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None), # leu
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "CH2", "CH2", "Nlys",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol","Hpol"), # lys
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "S",   "CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None), # met
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "aroC","aroC","aroC","aroC","aroC",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Haro","Haro","Haro","Haro","Haro",  None,  None,  None,  None), # phe
                ("Npro","CAbb","CObb","OCbb","CH2", "CH2", "CH2",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # pro
                ("Nbb", "CAbb","CObb","OCbb","CH2", "OH",    None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hpol","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # ser
                ("Nbb", "CAbb","CObb","OCbb","CH1", "OH",  "CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hpol","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # thr
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "aroC","CH0", "CH0", "aroC","Ntrp","aroC","aroC","aroC",  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Haro","Hapo","Hapo","Hapo","Hpol","Haro","Haro","Haro","Haro",  None,  None,  None), # trp
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "aroC","aroC","aroC","aroC","CH0", "OHY",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Haro","Haro","Haro","Haro","Hapo","Hapo","Hapo","Hpol",  None,  None,  None,  None), # tyr
                ("Nbb", "CAbb","CObb","OCbb","CH1", "CH3", "CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None), # val
                ("Nbb", "CAbb","CObb","OCbb","CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # unk
                ("Nbb", "CAbb","CObb","OCbb","CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # mask
                ("Oet3", "CH1","CH2","OOC","Phos", "OOC", "Oet2","CH2", "CH1", "CH1", "Oet2", "Npro","aroC","Nhis","aroC","Nhis","aroC","aroC","Nhis","aroC","NH2O",  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Haro","Hpol","Hpol","Haro",  None,  None), # DA
                ("Oet3", "CH1","CH2","OOC","Phos", "OOC", "Oet2","CH2", "CH1", "CH1", "Oet2", "Npro","CObb","OCbb","Nhis","aroC","NH2O","aroC","aroC",  None,  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol","Haro","Haro",  None,  None), # DC
                ("Oet3", "CH1","CH2","OOC","Phos", "OOC", "Oet2","CH2", "CH1", "CH1", "Oet2", "Npro","aroC","Nhis","aroC","Ntrp","CObb","aroC","Nhis","aroC","NH2O","OCbb",  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol","Hpol","Haro",  None,  None), # DG
                ("Oet3", "CH1","CH2","OOC","Phos", "OOC", "Oet2","CH2", "CH1", "CH1", "Oet2", "Npro","CObb","OCbb","Ntrp","CObb","OCbb","aroC","CH3", "aroC",  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo","Hapo","Hapo","Haro",  None), # DT
                ("Oet3", "CH1","CH2","OOC","Phos", "OOC", "Oet2","CH2", "CH1", "CH1", "Oet2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None), #  DX (unk DNA)
                ("Oet3", "CH1","CH2","OOC","Phos", "OOC", "Oet2","CH2", "CH1", "CH1", "Oet2", "OH",  "Nhis","aroC","Nhis","aroC","aroC","aroC","NH2O","Nhis","aroC","Npro",  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo","Haro","Hpol","Hpol","Haro",  None,  None), # A
                ("Oet3", "CH1","CH2","OOC","Phos", "OOC", "Oet2","CH2", "CH1", "CH1", "Oet2", "OH",  "Npro","CObb","OCbb","Nhis","aroC","NH2O","aroC","aroC",  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo","Hpol","Hpol","Haro","Haro",  None,  None), # C
                ("Oet3", "CH1","CH2","OOC","Phos", "OOC", "Oet2","CH2", "CH1", "CH1", "Oet2", "OH",  "Ntrp","aroC","NH2O","Nhis","aroC","aroC","CObb","OCbb","Nhis","aroC","Npro","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo","Hpol","Hpol","Hpol","Haro",  None,  None), # G
                ("Oet3", "CH1","CH2","OOC","Phos", "OOC", "Oet2","CH2", "CH1", "CH1", "Oet2", "OH",  "Npro","CObb","OCbb","Ntrp","CObb","OCbb","aroC","aroC",  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo","Hpol","Hapo","Haro",  None,  None,  None), # U
                ("Oet3", "CH1","CH2","OOC","Phos", "OOC", "Oet2","CH2", "CH1", "CH1", "Oet2", "OH",    None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo",  None,  None,  None,  None,  None,  None), # RX (unk RNA)
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "Nhis","aroC","aroC","Ntrp",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hpol","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # HIS-D NOT CORRECT!!!!!!!!!!
                (None, "genAl",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Al
                (None, "genAs",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # As
                (None, "genAu",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Au
                (None, "genB",   None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # B
                (None, "genBe",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Be
                (None, "genBr",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Br
                (None, "genC",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # C
                (None, "genCa",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ca
                (None, "genCl",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Cl
                (None, "genCo",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Co
                (None, "genCr",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Cr
                (None, "genCu",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Cu
                (None, "genF",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # F
                (None, "genFe",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Fe
                (None, "genHg",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Hg
                (None, "genI",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # I
                (None, "genIr",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ir
                (None, "genK",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # K
                (None, "genLi",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Li
                (None, "genMg",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Mg
                (None, "genMn",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Mn
                (None, "genMo",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Mo
                (None, "genN",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # N
                (None, "genNi",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ni
                (None, "genO",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # O
                (None, "genOs",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Os
                (None, "genP",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # P
                (None, "genPb",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pb
                (None, "genPd",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pd
                (None, "genPr",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pr
                (None, "genPt",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pt
                (None, "genRe",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Re
                (None, "genRh",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Rh
                (None, "genRu",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ru
                (None, "genS",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # S
                (None, "genSb",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Sb
                (None, "genSe",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Se
                (None, "genSi",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Si
                (None, "genSn",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Sn
                (None, "genTb",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Tb
                (None, "genTe",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Te
                (None, "genU",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # U
                (None, "genW",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # W   
                (None, "genV",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # V
                (None, "genY",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Y
                (None, "genZn",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Zn
                (None, "genATM",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # ATM

            ]

            self.aa2elt = [
                ("N","C","C","O","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H",None,None,None,None,None,None,None,None),#ala
                ("N","C","C","O","C","C","C","N","C","N","N",None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H","H","H","H","H"),#arg
                ("N","C","C","O","C","C","O","N",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H",None,None,None,None,None,None,None),#asn
                ("N","C","C","O","C","C","O","O",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H",None,None,None,None,None,None,None,None,None),#asp
                ("N","C","C","O","C","S",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H",None,None,None,None,None,None,None,None),#cys
                ("N","C","C","O","C","C","C","O","N",None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H",None,None,None,None,None),#gln
                ("N","C","C","O","C","C","C","O","O",None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H",None,None,None,None,None,None,None),#glu
                ("N","C","C","O",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H",None,None,None,None,None,None,None,None,None,None),#gly
                ("N","C","C","O","C","C","N","C","C","N",None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H",None,None,None,None,None,None),#his
                ("N","C","C","O","C","C","C","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H","H","H",None,None),#ile
                ("N","C","C","O","C","C","C","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H","H","H",None,None),#leu
                ("N","C","C","O","C","C","C","C","N",None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H","H","H","H","H"),#lys
                ("N","C","C","O","C","C","S","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H",None,None,None,None),#met
                ("N","C","C","O","C","C","C","C","C","C","C",None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H",None,None,None,None),#phe
                ("N","C","C","O","C","C","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H",None,None,None,None,None,None),#pro
                ("N","C","C","O","C","O",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H",None,None,None,None,None,None,None,None),#ser
                ("N","C","C","O","C","O","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H",None,None,None,None,None,None),#thr
                ("N","C","C","O","C","C","C","C","C","C","N","C","C","C",None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H","H",None,None,None),#trp
                ("N","C","C","O","C","C","C","C","C","C","C","O",None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H",None,None,None,None),#tyr
                ("N","C","C","O","C","C","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H",None,None,None,None),#val
                ("N","C","C","O","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H",None,None,None,None,None,None,None,None),#unk
                ("N","C","C","O","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H",None,None,None,None,None,None,None,None),#mask
                ("O" ,"C" ,"C" ,"O" ,"P" ,"O" ,"O" ,"C" ,"C" ,"C" ,"O" ,"N" ,"C" ,"N" ,"C" ,"N" ,"C" ,"C" ,"N" ,"C" ,"N" ,None,None,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,None,None),#DA
                ("O" ,"C" ,"C" ,"O" ,"P" ,"O" ,"O" ,"C" ,"C" ,"C" ,"O" ,"N" ,"C" ,"O" ,"N" ,"C" ,"N" ,"C" ,"C" ,None,None,None,None,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,None,None),#DC
                ("O" ,"C" ,"C" ,"O" ,"P" ,"O" ,"O" ,"C" ,"C" ,"C" ,"O" ,"N" ,"C" ,"N" ,"C" ,"N", "C" ,"C" ,"N" ,"C" ,"N" ,"O" ,None,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,None,None),#DG
                ("O" ,"C" ,"C" ,"O" ,"P" ,"O" ,"O" ,"C" ,"C" ,"C" ,"O" ,"N" ,"C" ,"O" ,"N" ,"C" ,"O" ,"C" ,"C" ,"C" ,None,None,None,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,None),#DT
                ("O" ,"C" ,"C" ,"O" ,"P" ,"O" ,"O" ,"C" ,"C" ,"C" ,"O" ,None,None,None,None,None,None,None,None,None,None,None,None,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,None,None,None,None,None,None),#DX
                ("O" ,"C" ,"C" ,"O" ,"P" ,"O" ,"O" ,"C" ,"C" ,"C" ,"O" ,"O" ,"N" ,"C" ,"N" ,"C" ,"C" ,"C" ,"N" ,"N" ,"C" ,"N" ,None,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,None,None),#A
                ("O" ,"C" ,"C" ,"O" ,"P" ,"O" ,"O" ,"C" ,"C" ,"C" ,"O" ,"O" ,"N" ,"C" ,"O" ,"N" ,"C" ,"N" ,"C" ,"C" ,None,None,None,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,None,None),#C
                ("O" ,"C" ,"C" ,"O" ,"P" ,"O" ,"O" ,"C" ,"C" ,"C" ,"O" ,"O" ,"N" ,"C" ,"N" ,"N" ,"C" ,"C" ,"C" ,"O" ,"N" ,"C" ,"N" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,None,None),#G
                ("O" ,"C" ,"C" ,"O" ,"P" ,"O" ,"O" ,"C" ,"C" ,"C" ,"O" ,"O" ,"N" ,"C" ,"O" ,"N" ,"C" ,"O" ,"C" ,"C" ,None,None,None,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,None,None,None),#U
                ("O" ,"C" ,"C" ,"O" ,"P" ,"O" ,"O" ,"C" ,"C" ,"C" ,"O" ,"O" ,None,None,None,None,None,None,None,None,None,None,None,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,"H" ,None,None,None,None,None),#RX
            ]

            # frames for generic FAPE
            ##NOTE: 1st entry must be "backbone frame"
            self.frames=[
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # ala
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "], [" CG "," CD "," NE "] ],  # arg
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # asn
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # asp
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "] ],  # cys
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "] ],  # gln
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "] ],  # glu
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # gly
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # his
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG1"] ],  # ile
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # leu
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "], [" CG "," CD "," CE "] ],  # lys
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," SD "] ],  # met
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # phe
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "]],  # pro
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," OG "] ],  # ser
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," OG1"] ],  # thr
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # trp
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # tyr
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "] ],  # val
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # unk
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # mask
                [ [" O4'"," C1'"," C2'"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N9 "], [" C4'"," C3'"," O3'"] ], #DA
                [ [" O4'"," C1'"," C2'"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N1 "], [" C4'"," C3'"," O3'"] ], #DC
                [ [" O4'"," C1'"," C2'"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N9 "], [" C4'"," C3'"," O3'"] ], #DG
                [ [" O4'"," C1'"," C2'"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N1 "], [" C4'"," C3'"," O3'"] ], #DT
                [ [" O4'"," C1'"," C2'"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C4'"," C3'"," O3'"] ], #DX
                [ [" O4'"," C1'"," C2'"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N9 "], [" C4'"," C3'"," O3'"] ], #A
                [ [" O4'"," C1'"," C2'"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N1 "], [" C4'"," C3'"," O3'"] ], #C
                [ [" O4'"," C1'"," C2'"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N9 "], [" C4'"," C3'"," O3'"] ], #G
                [ [" O4'"," C1'"," C2'"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N1 "], [" C4'"," C3'"," O3'"] ], #U
                [ [" O4'"," C1'"," C2'"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C4'"," C3'"," O3'"] ], #RX
            ]

            #fd Rosetta ideal coords
            #fd   - uses same "frame-building" as AF2
            # FRAMES:
            #   base = 0
            #   omega/phi/psi = 1-3 (omega unused)
            #   chi_1-4(prot) = 4-7
            #   CB_bend = 8
            #   NA alpha/beta/gamma/delta = 9-12  (NA epsilon/zeta no frame)
            #   NA nu2/nu1/nu0 = 13-15
            #   chi_1(NA) = 16
            self.ideal_coords = [
                [ # 0 ala
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3341, -0.4928,  0.9132)],
                    [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
                    ['1HB ', 8, (-0.1265, -1.7863, -1.1851)],
                    ['2HB ', 8, (-1.6173, -0.8147, -1.1541)],
                    ['3HB ', 8, (-0.2229, -0.2744, -2.1172)],
                ],
                [ # 1 arg
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3467, -0.5055,  0.9018)],
                    [' CB ', 8, (-0.5042,-0.7698,-1.2118)],
                    ['1HB ', 4, ( 0.3635, -0.5318,  0.8781)],
                    ['2HB ', 4, ( 0.3639, -0.5323, -0.8789)],
                    [' CG ', 4, (0.6396,1.3794, 0.000)],
                    ['1HG ', 5, (0.3639, -0.5139,  0.8900)],
                    ['2HG ', 5, (0.3641, -0.5140, -0.8903)],
                    [' CD ', 5, (0.5492,1.3801, 0.000)],
                    ['1HD ', 6, (0.3637, -0.5135,  0.8895)],
                    ['2HD ', 6, (0.3636, -0.5134, -0.8893)],
                    [' NE ', 6, (0.5423,1.3491, 0.000)],
                    [' NH1', 7, (0.2012,2.2965, 0.000)],
                    [' NH2', 7, (2.0824,1.0030, 0.000)],
                    [' CZ ', 7, (0.7650,1.1090, 0.000)],
                    [' HE ', 7, (0.4701,-0.8955, 0.000)],
                    ['1HH1', 7, (-0.8059,2.3776, 0.000)],
                    ['1HH2', 7, (2.5160,0.0898, 0.000)],
                    ['2HH1', 7, (0.7745,3.1277, 0.000)],
                    ['2HH2', 7, (2.6554,1.8336, 0.000)],
                ],
                [ # 2 asn
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3233, -0.4967,  0.9162)],
                    [' CB ', 8, (-0.5341,-0.7799,-1.1874)],
                    ['1HB ', 4, ( 0.3641, -0.5327,  0.8795)],
                    ['2HB ', 4, ( 0.3639, -0.5323, -0.8789)],
                    [' CG ', 4, (0.5778,1.3881, 0.000)],
                    [' ND2', 5, (0.5839,-1.1711, 0.000)],
                    [' OD1', 5, (0.6331,1.0620, 0.000)],
                    ['1HD2', 5, (1.5825, -1.2322, 0.000)],
                    ['2HD2', 5, (0.0323, -2.0046, 0.000)],
                ],
                [ # 3 asp
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3233, -0.4967,  0.9162)],
                    [' CB ', 8, (-0.5162,-0.7757,-1.2144)],
                    ['1HB ', 4, ( 0.3639, -0.5324,  0.8791)],
                    ['2HB ', 4, ( 0.3640, -0.5325, -0.8792)],
                    [' CG ', 4, (0.5926,1.4028, 0.000)],
                    [' OD1', 5, (0.5746,1.0629, 0.000)],
                    [' OD2', 5, (0.5738,-1.0627, 0.000)],
                ],
                [ # 4 cys
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3481, -0.5059,  0.9006)],
                    [' CB ', 8, (-0.5046,-0.7727,-1.2189)],
                    ['1HB ', 4, ( 0.3639, -0.5324,  0.8791)],
                    ['2HB ', 4, ( 0.3638, -0.5322, -0.8787)],
                    [' SG ', 4, (0.7386,1.6511, 0.000)],
                    [' HG ', 5, (0.1387,1.3221, 0.000)],
                ],
                [ # 5 gln
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3363, -0.5013,  0.9074)],
                    [' CB ', 8, (-0.5226,-0.7776,-1.2109)],
                    ['1HB ', 4, ( 0.3638, -0.5323,  0.8789)],
                    ['2HB ', 4, ( 0.3638, -0.5322, -0.8788)],
                    [' CG ', 4, (0.6225,1.3857, 0.000)],
                    ['1HG ', 5, ( 0.3531, -0.5156,  0.8931)],
                    ['2HG ', 5, ( 0.3531, -0.5156, -0.8931)],
                    [' CD ', 5, (0.5788,1.4021, 0.000)],
                    [' NE2', 6, (0.5908,-1.1895, 0.000)],
                    [' OE1', 6, (0.6347,1.0584, 0.000)],
                    ['1HE2', 6, (1.5825, -1.2525, 0.000)],
                    ['2HE2', 6, (0.0380, -2.0229, 0.000)],
                ],
                [ # 6 glu
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3363, -0.5013,  0.9074)],
                    [' CB ', 8, (-0.5197,-0.7737,-1.2137)],
                    ['1HB ', 4, ( 0.3638, -0.5323,  0.8789)],
                    ['2HB ', 4, ( 0.3638, -0.5322, -0.8788)],
                    [' CG ', 4, (0.6287,1.3862, 0.000)],
                    ['1HG ', 5, ( 0.3531, -0.5156,  0.8931)],
                    ['2HG ', 5, ( 0.3531, -0.5156, -0.8931)],
                    [' CD ', 5, (0.5850,1.3849, 0.000)],
                    [' OE1', 6, (0.5752,1.0618, 0.000)],
                    [' OE2', 6, (0.5741,-1.0635, 0.000)],
                ],
                [ # 7 gly
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    ['1HA ', 0, ( -0.3676, -0.5329,  0.8771)],
                    ['2HA ', 0, ( -0.3674, -0.5325, -0.8765)],
                ],
                [ # 8 his
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3299, -0.5180,  0.9001)],
                    [' CB ', 8, (-0.5163,-0.7809,-1.2129)],
                    ['1HB ', 4, ( 0.3640, -0.5325,  0.8793)],
                    ['2HB ', 4, ( 0.3637, -0.5321, -0.8786)],
                    [' CG ', 4, (0.6016,1.3710, 0.000)],
                    [' CD2', 5, (0.8918,-1.0184, 0.000)],
                    [' CE1', 5, (2.0299,0.8564, 0.000)],
                    ['1HE ', 5, (2.8542, 1.5693,  0.000)],
                    ['2HD ', 5, ( 0.6584, -2.0835, 0.000) ],
                    [' ND1', 6, (-1.8631, -1.0722,  0.000)],
                    [' NE2', 6, (-1.8625,  1.0707, 0.000)],
                    ['2HE ', 6, (-1.5439,  2.0292, 0.000)],
                ],
                [ # 9 ile
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3405, -0.5028,  0.9044)],
                    [' CB ', 8, (-0.5140,-0.7885,-1.2184)],
                    [' HB ', 4, (0.3637, -0.4714,  0.9125)],
                    [' CG1', 4, (0.5339,1.4348,0.000)],
                    [' CG2', 4, (0.5319,-0.7693,-1.1994)],
                    ['1HG2', 4, (1.6215, -0.7588, -1.1842)],
                    ['2HG2', 4, (0.1785, -1.7986, -1.1569)],
                    ['3HG2', 4, (0.1773, -0.3016, -2.1180)],
                    [' CD1', 5, (0.6106,1.3829, 0.000)],
                    ['1HG1', 5, (0.3637, -0.5338,  0.8774)],
                    ['2HG1', 5, (0.3640, -0.5322, -0.8793)],
                    ['1HD1', 5, (1.6978,  1.3006, 0.000)],
                    ['2HD1', 5, (0.2873,  1.9236, -0.8902)],
                    ['3HD1', 5, (0.2888, 1.9224, 0.8896)],
                ],
                [ # 10 leu
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.525, -0.000, -0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3435, -0.5040,  0.9027)],
                    [' CB ', 8, (-0.5175,-0.7692,-1.2220)],
                    ['1HB ', 4, ( 0.3473, -0.5346,  0.8827)],
                    ['2HB ', 4, ( 0.3476, -0.5351, -0.8836)],
                    [' CG ', 4, (0.6652,1.3823, 0.000)],
                    [' CD1', 5, (0.5083,1.4353, 0.000)],
                    [' CD2', 5, (0.5079,-0.7600,1.2163)],
                    [' HG ', 5, (0.3640, -0.4825, -0.9075)],
                    ['1HD1', 5, (1.5984,  1.4353, 0.000)],
                    ['2HD1', 5, (0.1462,  1.9496, -0.8903)],
                    ['3HD1', 5, (0.1459, 1.9494, 0.8895)],
                    ['1HD2', 5, (1.5983, -0.7606,  1.2158)],
                    ['2HD2', 5, (0.1456, -0.2774,  2.1243)],
                    ['3HD2', 5, (0.1444, -1.7871,  1.1815)],
                ],
                [ # 11 lys
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3335, -0.5005,  0.9097)],
                    ['1HB ', 4, ( 0.3640, -0.5324,  0.8791)],
                    ['2HB ', 4, ( 0.3639, -0.5324, -0.8790)],
                    [' CB ', 8, (-0.5259,-0.7785,-1.2069)],
                    ['1HG ', 5, (0.3641, -0.5229,  0.8852)],
                    ['2HG ', 5, (0.3637, -0.5227, -0.8841)],
                    [' CG ', 4, (0.6291,1.3869, 0.000)],
                    [' CD ', 5, (0.5526,1.4174, 0.000)],
                    ['1HD ', 6, (0.3641, -0.5239,  0.8848)],
                    ['2HD ', 6, (0.3638, -0.5219, -0.8850)],
                    [' CE ', 6, (0.5544,1.4170, 0.000)],
                    [' NZ ', 7, (0.5566,1.3801, 0.000)],
                    ['1HE ', 7, (0.4199, -0.4638,  0.9482)],
                    ['2HE ', 7, (0.4202, -0.4631, -0.8172)],
                    ['1HZ ', 7, (1.6223, 1.3980, 0.0658)],
                    ['2HZ ', 7, (0.2970,  1.9326, -0.7584)],
                    ['3HZ ', 7, (0.2981, 1.9319, 0.8909)],
                ],
                [ # 12 met
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3303, -0.4990,  0.9108)],
                    ['1HB ', 4, ( 0.3635, -0.5318,  0.8781)],
                    ['2HB ', 4, ( 0.3641, -0.5326, -0.8795)],
                    [' CB ', 8, (-0.5331,-0.7727,-1.2048)],
                    ['1HG ', 5, (0.3637, -0.5256,  0.8823)],
                    ['2HG ', 5, (0.3638, -0.5249, -0.8831)],
                    [' CG ', 4, (0.6298,1.3858,0.000)],
                    [' SD ', 5, (0.6953,1.6645,0.000)],
                    [' CE ', 6, (0.3383,1.7581,0.000)],
                    ['1HE ', 6, (1.7054,  2.0532, -0.0063)],
                    ['2HE ', 6, (0.1906,  2.3099, -0.9072)],
                    ['3HE ', 6, (0.1917, 2.3792, 0.8720)],
                ],
                [ # 13 phe
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3303, -0.4990,  0.9108)],
                    ['1HB ', 4, ( 0.3635, -0.5318,  0.8781)],
                    ['2HB ', 4, ( 0.3641, -0.5326, -0.8795)],
                    [' CB ', 8, (-0.5150,-0.7729,-1.2156)],
                    [' CG ', 4, (0.6060,1.3746, 0.000)],
                    [' CD1', 5, (0.7078,1.1928, 0.000)],
                    [' CD2', 5, (0.7084,-1.1920, 0.000)],
                    [' CE1', 5, (2.0900,1.1940, 0.000)],
                    [' CE2', 5, (2.0897,-1.1939, 0.000)],
                    [' CZ ', 5, (2.7809, 0.000, 0.000)],
                    ['1HD ', 5, (0.1613, 2.1362, 0.000)],
                    ['2HD ', 5, (0.1621, -2.1360, 0.000)],
                    ['1HE ', 5, (2.6335,  2.1384, 0.000)],
                    ['2HE ', 5, (2.6344, -2.1378, 0.000)],
                    [' HZ ', 5, (3.8700, 0.000, 0.000)],
                ],
                [ # 14 pro
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' HA ', 0, (-0.3868, -0.5380,  0.8781)],
                    ['1HB ', 4, ( 0.3762, -0.5355,  0.8842)],
                    ['2HB ', 4, ( 0.3762, -0.5355, -0.8842)],
                    [' CB ', 8, (-0.5649,-0.5888,-1.2966)],
                    [' CG ', 4, (0.3657,1.4451,0.0000)],
                    [' CD ', 5, (0.3744,1.4582, 0.0)],
                    ['1HG ', 5, (0.3798, -0.5348,  0.8830)],
                    ['2HG ', 5, (0.3798, -0.5348, -0.8830)],
                    ['1HD ', 6, (0.3798, -0.5348,  0.8830)],
                    ['2HD ', 6, (0.3798, -0.5348, -0.8830)],
                ],
                [ # 15 ser
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3425, -0.5041,  0.9048)],
                    ['1HB ', 4, ( 0.3637, -0.5321,  0.8786)],
                    ['2HB ', 4, ( 0.3636, -0.5319, -0.8782)],
                    [' CB ', 8, (-0.5146,-0.7595,-1.2073)],
                    [' OG ', 4, (0.5021,1.3081, 0.000)],
                    [' HG ', 5, (0.2647, 0.9230, 0.000)],
                ],
                [ # 16 thr
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3364, -0.5015,  0.9078)],
                    [' HB ', 4, ( 0.3638, -0.5006,  0.8971)],
                    ['1HG2', 4, ( 1.6231, -0.7142, -1.2097)],
                    ['2HG2', 4, ( 0.1792, -1.7546, -1.2237)],
                    ['3HG2', 4, ( 0.1808, -0.2222, -2.1269)],
                    [' CB ', 8, (-0.5172,-0.7952,-1.2130)],
                    [' CG2', 4, (0.5334,-0.7239,-1.2267)],
                    [' OG1', 4, (0.4804,1.3506,0.000)],
                    [' HG1', 5, (0.3194,  0.9056, 0.000)],
                ],
                [ # 17 trp
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3436, -0.5042,  0.9031)],
                    ['1HB ', 4, ( 0.3639, -0.5323,  0.8790)],
                    ['2HB ', 4, ( 0.3638, -0.5322, -0.8787)],
                    [' CB ', 8, (-0.5136,-0.7712,-1.2173)],
                    [' CG ', 4, (0.5984,1.3741, 0.000)],
                    [' CD1', 5, (0.8151,1.0921, 0.000)],
                    [' CD2', 5, (0.8753,-1.1538, 0.000)],
                    [' CE2', 5, (2.1865,-0.6707, 0.000)],
                    [' CE3', 5, (0.6541,-2.5366, 0.000)],
                    [' NE1', 5, (2.1309,0.7003, 0.000)],
                    [' CH2', 5, (3.0315,-2.8930, 0.000)],
                    [' CZ2', 5, (3.2813,-1.5205, 0.000)],
                    [' CZ3', 5, (1.7521,-3.3888, 0.000)],
                    ['1HD ', 5, (0.4722, 2.1252,  0.000)],
                    ['1HE ', 5, ( 2.9291,  1.3191,  0.000)],
                    [' HE3', 5, (-0.3597, -2.9356,  0.000)],
                    [' HZ2', 5, (4.3053, -1.1462,  0.000)],
                    [' HZ3', 5, ( 1.5712, -4.4640,  0.000)],
                    [' HH2', 5, ( 3.8700, -3.5898,  0.000)],
                ],
                [ # 18 tyr
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3305, -0.4992,  0.9112)],
                    ['1HB ', 4, ( 0.3642, -0.5327,  0.8797)],
                    ['2HB ', 4, ( 0.3637, -0.5321, -0.8785)],
                    [' CB ', 8, (-0.5305,-0.7799,-1.2051)],
                    [' CG ', 4, (0.6104,1.3840, 0.000)],
                    [' CD1', 5, (0.6936,1.2013, 0.000)],
                    [' CD2', 5, (0.6934,-1.2011, 0.000)],
                    [' CE1', 5, (2.0751,1.2013, 0.000)],
                    [' CE2', 5, (2.0748,-1.2011, 0.000)],
                    [' OH ', 5, (4.1408, 0.000, 0.000)],
                    [' CZ ', 5, (2.7648, 0.000, 0.000)],
                    ['1HD ', 5, (0.1485, 2.1455,  0.000)],
                    ['2HD ', 5, (0.1484, -2.1451,  0.000)],
                    ['1HE ', 5, (2.6200, 2.1450,  0.000)],
                    ['2HE ', 5, (2.6199, -2.1453,  0.000)],
                    [' HH ', 6, (0.3190, 0.9057,  0.000)],
                ],
                [ # 19 val
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3497, -0.5068,  0.9002)],
                    [' CB ', 8, (-0.5105,-0.7712,-1.2317)],
                    [' CG1', 4, (0.5326,1.4252, 0.000)],
                    [' CG2', 4, (0.5177,-0.7693,1.2057)],
                    [' HB ', 4, (0.3541, -0.4754, -0.9148)],
                    ['1HG1', 4, (1.6228,  1.4063,  0.000)],
                    ['2HG1', 4, (0.1790,  1.9457, -0.8898)],
                    ['3HG1', 4, (0.1798, 1.9453, 0.8903)],
                    ['1HG2', 4, (1.6073, -0.7659,  1.1989)],
                    ['2HG2', 4, (0.1586, -0.2971,  2.1203)],
                    ['3HG2', 4, (0.1582, -1.7976,  1.1631)],
                ],
                [ # 20 unk
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3341, -0.4928,  0.9132)],
                    [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
                    ['1HB ', 8, (-0.1265, -1.7863, -1.1851)],
                    ['2HB ', 8, (-1.6173, -0.8147, -1.1541)],
                    ['3HB ', 8, (-0.2229, -0.2744, -2.1172)],
                ],
                [ # 21 mask
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3341, -0.4928,  0.9132)],
                    [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
                    ['1HB ', 8, (-0.1265, -1.7863, -1.1851)],
                    ['2HB ', 8, (-1.6173, -0.8147, -1.1541)],
                    ['3HB ', 8, (-0.2229, -0.2744, -2.1172)],
                ],
                [ # 22 DA
                    [" O4'",0, (-0.3894, 1.3649, 0.0000)],
                    [" C1'",0, (0.0000, 0.0000, 0.0000)],
                    [" C2'",0, (1.5186, 0.0000, 0.0000)],
                    [" N9 ",0, (-0.5962, -0.6345, -1.1746)],
                    [" H1'",0, (-0.3458, -0.5730,  0.8726)],
                    [" C3'",14, (0.3241, 1.4829, 0.000)],
                    [" H2'",14, (0.4107, -0.5097, -0.8844)],
                    ["H2''",14, (0.4106, -0.5096,  0.8840)],
                    [" C4'",13, (0.3513, 1.4879, 0.000)],
                    [" O3'",13, (0.5035, -0.6292,  1.1839)],
                    [" H3'",13, (0.4310, -0.5496, -0.8502)],
                    [" C5'",12, (0.63111,  1.3722, 0.000)],
                    [" H4'",12, (0.3168, -0.5077, -0.7763)],
                    [" O5'",11, (0.4966, 1.3523, 0.000)],
                    [" H5'",11, (0.3740, -0.5366,  0.8839)],
                    ["H5''",11, (0.3748, -0.5365, -0.8842)],
                    [" P  ",10, (0.81940, 1.3660, 0.000)],
                    [" OP1", 9, (0.4669, 1.4096, 0.000)],
                    [" OP2", 9, (0.4615, -0.9241,  1.0676)],

                    [" C4 ",16, (0.8119, 1.1084, 0.0000)],
                    [" N3 ",16, (0.4328, 2.3976, 0.0000)],
                    [" C2 ",16, (1.4957, 3.1983, 0.0000)],
                    [" N1 ",16, (2.7960, 2.8816, 0.0000)],
                    [" C6 ",16, (3.1433, 1.5760, 0.0000)],
                    [" C5 ",16, (2.1084, 0.6255, 0.0000)],
                    [" N7 ",16, (2.1145, -0.7627, 0.0000)],
                    [" C8 ",16, (0.8438, -1.0825, 0.0000)],
                    [" N6 ",16, (4.4402, 1.2598, 0.0000)],
                    [" H2 ",16, (1.2740, 4.2755, 0.0000)],
                    [" H8 ",16, (0.4867, -2.1227, 0.0000)],
                    [" H61",16, (5.1313, 1.9828, 0.0000)],
                    [" H62",16, (4.7211, 0.3001, 0.0000)],
                ],
                [ # 23 DC
                    [" O4'",0, (-0.3894, 1.3649, 0.0000)],
                    [" C1'",0, (0.0000, 0.0000, 0.0000)],
                    [" C2'",0, (1.5186, 0.0000, 0.0000)],
                    [" N1 ",0, (-0.5962, -0.6345, -1.1746)],
                    [" H1'",0, (-0.3458, -0.5730,  0.8726)],
                    [" C3'",14, (0.3241, 1.4829, 0.000)],
                    [" H2'",14, (0.4107, -0.5097, -0.8844)],
                    ["H2''",14, (0.4106, -0.5096,  0.8840)],
                    [" C4'",13, (0.3513, 1.4879, 0.000)],
                    [" O3'",13, (0.5035, -0.6292,  1.1839)],
                    [" H3'",13, (0.4310, -0.5496, -0.8502)],
                    [" C5'",12, (0.63111,  1.3722, 0.000)],
                    [" H4'",12, (0.3168, -0.5077, -0.7763)],
                    [" O5'",11, (0.4966, 1.3523, 0.000)],
                    [" H5'",11, (0.3740, -0.5366,  0.8839)],
                    ["H5''",11, (0.3748, -0.5365, -0.8842)],
                    [" P  ",10, (0.81940, 1.3660, 0.000)],
                    [" OP1", 9, (0.4669, 1.4096, 0.000)],
                    [" OP2", 9, (0.4615, -0.9241,  1.0676)],
                    [" C2 ",16, (0.6758, 1.2249, 0.0000)],
                    [" O2 ",16, (0.0158, 2.2756, 0.0000)],
                    [" N3 ",16, (2.0283, 1.2334, 0.0000)],
                    [" C4 ",16, (2.7022, 0.0815, 0.0000)],
                    [" N4 ",16, (4.0356, 0.1372, 0.0000)],
                    [" C5 ",16, (2.0394, -1.1794, 0.0000)],
                    [" C6 ",16, (0.7007, -1.1745, 0.0000)],
                    [" H42",16, (4.5715, -0.7074, 0.0000)],
                    [" H41",16, (4.4992, 1.0229, 0.0000)],
                    [" H5 ",16, (2.6061, -2.1225, 0.0000)],
                    [" H6 ",16, (0.1563, -2.1302, 0.0000)],
                ],
                [ # 24 DG
                    [" O4'",0, (-0.3894, 1.3649, 0.0000)],
                    [" C1'",0, (0.0000, 0.0000, 0.0000)],
                    [" C2'",0, (1.5186, 0.0000, 0.0000)],
                    [" N9 ",0, (-0.5962, -0.6345, -1.1746)],
                    [" H1'",0, (-0.3458, -0.5730,  0.8726)],
                    [" C3'",14, (0.3241, 1.4829, 0.000)],
                    [" H2'",14, (0.4107, -0.5097, -0.8844)],
                    ["H2''",14, (0.4106, -0.5096,  0.8840)],
                    [" C4'",13, (0.3513, 1.4879, 0.000)],
                    [" O3'",13, (0.5035, -0.6292,  1.1839)],
                    [" H3'",13, (0.4310, -0.5496, -0.8502)],
                    [" C5'",12, (0.63111,  1.3722, 0.000)],
                    [" H4'",12, (0.3168, -0.5077, -0.7763)],
                    [" O5'",11, (0.4966, 1.3523, 0.000)],
                    [" H5'",11, (0.3740, -0.5366,  0.8839)],
                    ["H5''",11, (0.3748, -0.5365, -0.8842)],
                    [" P  ",10, (0.81940, 1.3660, 0.000)],
                    [" OP1", 9, (0.4669, 1.4096, 0.000)],
                    [" OP2", 9, (0.4615, -0.9241,  1.0676)],
                    [" C4 ",16, (0.8171, 1.1043, 0.0000)],
                    [" N3 ",16, (0.4110, 2.3918, 0.0000)],
                    [" C2 ",16, (1.4330, 3.2319, 0.0000)],
                    [" N1 ",16, (2.7493, 2.8397, 0.0000)],
                    [" C6 ",16, (3.1894, 1.5195, 0.0000)],
                    [" C5 ",16, (2.1029, 0.6070, 0.0000)],
                    [" N7 ",16, (2.0942, -0.7800, 0.0000)],
                    [" C8 ",16, (0.8285, -1.0956, 0.0000)],
                    [" N2 ",16, (1.2085, 4.5537, 0.0000)],
                    [" O6 ",16, (4.4017, 1.2743, 0.0000)],
                    [" H1 ",16, (3.4453, 3.5579, 0.0000)],
                    [" H8 ",16, (0.4623, -2.1330, 0.0000)],
                    [" H22",16, (0.2708, 4.9015, 0.0000)],
                    [" H21",16, (1.9785, 5.1920, 0.0000)],
                ],
                [ # 25 DT
                    [" O4'",0, (-0.3894, 1.3649, 0.0000)],
                    [" C1'",0, (0.0000, 0.0000, 0.0000)],
                    [" C2'",0, (1.5186, 0.0000, 0.0000)],
                    [" N1 ",0, (-0.5962, -0.6345, -1.1746)],
                    [" H1'",0, (-0.3458, -0.5730,  0.8726)],
                    [" C3'",14, (0.3241, 1.4829, 0.000)],
                    [" H2'",14, (0.4107, -0.5097, -0.8844)],
                    ["H2''",14, (0.4106, -0.5096,  0.8840)],
                    [" C4'",13, (0.3513, 1.4879, 0.000)],
                    [" O3'",13, (0.5035, -0.6292,  1.1839)],
                    [" H3'",13, (0.4310, -0.5496, -0.8502)],
                    [" C5'",12, (0.63111,  1.3722, 0.000)],
                    [" H4'",12, (0.3168, -0.5077, -0.7763)],
                    [" O5'",11, (0.4966, 1.3523, 0.000)],
                    [" H5'",11, (0.3740, -0.5366,  0.8839)],
                    ["H5''",11, (0.3748, -0.5365, -0.8842)],
                    [" P  ",10, (0.81940, 1.3660, 0.000)],
                    [" OP1", 9, (0.4669, 1.4096, 0.000)],
                    [" OP2", 9, (0.4615, -0.9241,  1.0676)],
                    [" C2 ",16, (0.6495, 1.2140, 0.0000)],
                    [" O2 ",16, (0.0636, 2.2854, 0.0000)],
                    [" N3 ",16, (2.0191, 1.1297, 0.0000)],
                    [" C4 ",16, (2.7859, -0.0198, 0.0000)],
                    [" O4 ",16, (4.0113, 0.0622, 0.0000)],
                    [" C5 ",16, (2.0397, -1.2580, 0.0000)],
                    [" C7 ",16, (2.7845, -2.5550, 0.0000)],
                    [" C6 ",16, (0.7021, -1.1863, 0.0000)],
                    [" H3 ",16, (2.5175, 1.9968, 0.0000)],
                    [" H71",16, (2.0680, -3.3898, 0.0000)],
                    [" H72",16, (3.4147, -2.6153, -0.9071)],
                    [" H73",16, (3.4193, -2.6153, 0.8885)],
                    [" H6 ",16, (0.1317, -2.1273, 0.0000)],
                ],
                [ # 26 DX
                    [" O4'",0, (-0.3894, 1.3649, 0.0000)],
                    [" C1'",0, (0.0000, 0.0000, 0.0000)],
                    [" C2'",0, (1.5186, 0.0000, 0.0000)],
                    [" H1'",0, (-0.3458, -0.5730,  0.8726)],
                    [" C3'",14, (0.3241, 1.4829, 0.000)],
                    [" H2'",14, (0.4107, -0.5097, -0.8844)],
                    ["H2''",14, (0.4106, -0.5096,  0.8840)],
                    [" C4'",13, (0.3513, 1.4879, 0.000)],
                    [" O3'",13, (0.5035, -0.6292,  1.1839)],
                    [" H3'",13, (0.4310, -0.5496, -0.8502)],
                    [" C5'",12, (0.63111,  1.3722, 0.000)],
                    [" H4'",12, (0.3168, -0.5077, -0.7763)],
                    [" O5'",11, (0.4966, 1.3523, 0.000)],
                    [" H5'",11, (0.3740, -0.5366,  0.8839)],
                    ["H5''",11, (0.3748, -0.5365, -0.8842)],
                    [" P  ",10, (0.81940, 1.3660, 0.000)],
                    [" OP1", 9, (0.4669, 1.4096, 0.000)],
                    [" OP2", 9, (0.4615, -0.9241,  1.0676)],
                ],
                [ # 27 A
                    [" O4'",0, (-0.4082, 1.3525, 0.0000)],
                    [" C1'",0, (0.0000, 0.0000, 0.0000)],
                    [" C2'",0, (1.5292, 0.0000, 0.0000)],
                    [" N9 ",0, (-0.5661, -0.6642, -1.1894)],
                    [" H1'",0, (-0.3326, -0.4415,  0.8101)],
                    [" C3'",14, (0.3028, 1.4927, 0.000)],
                    [" H2'",14, (0.3582, -0.4393, -0.7998)],
                    [" O2'",14, (0.4613, -0.6189,  1.1921)],
                    ["HO2'",14, (0.2499, -1.5749,  1.1568)],
                    [" C4'",13, (0.3316, 1.4845, 0.000)],
                    [" O3'",13, (0.5685, -0.6954,  1.0960)],
                    [" H3'",13, (0.3202, -0.4010, -0.8356)],
                    [" C5'",12, (0.6616, 1.3553, 0.000)],
                    [" H4'",12, (0.3168, -0.5077, -0.7763)],
                    [" O5'",11, (0.5217, 1.3206, 0.000)],
                    [" H5'",11, (0.3193, -0.4703,  0.7986)],
                    ["H5''",11, (0.3199, -0.4711, -0.7970)],
                    [" P  ",10, (0.8180, 1.3676, 0.000)],
                    [" OP1", 9, (0.4594, 1.4113, 0.000)],
                    [" OP2", 9, (0.4582, -0.9215,  1.0698)],
                    [" N1 ",16, (2.7963, 2.8824, 0.0000)],
                    [" C2 ",16, (1.4955, 3.2007, 0.0000)],
                    [" N3 ",16, (0.4333, 2.3980, 0.0000)],
                    [" C4 ",16, (0.8127, 1.1078, 0.0000)],
                    [" C5 ",16, (2.1082, 0.6254, 0.0000)],
                    [" C6 ",16, (3.1432, 1.5774, 0.0000)],
                    [" N6 ",16, (4.4400, 1.2609, 0.0000)],
                    [" N7 ",16, (2.1146, -0.7630, 0.0000)],
                    [" C8 ",16, (0.8442, -1.0830, 0.0000)],
                    [" H2 ",16, (1.2972, 4.1608, 0.0000)],
                    [" H61",16, (5.1172, 1.9697, 0.0000)],
                    [" H62",16, (4.7154, 0.3206, 0.0000)],
                    [" H8 ",16, (0.5258, -2.0104, 0.0000)],
                ],
                [ # 28 C
                    [" O4'",0, (-0.4082, 1.3525, 0.0000)],
                    [" C1'",0, (0.0000, 0.0000, 0.0000)],
                    [" C2'",0, (1.5292, 0.0000, 0.0000)],
                    [" N1 ",0, (-0.5661, -0.6642, -1.1894)],
                    [" H1'",0, (-0.3326, -0.4415,  0.8101)],
                    [" C3'",14, (0.3028, 1.4927, 0.000)],
                    [" H2'",14, (0.3582, -0.4393, -0.7998)],
                    [" O2'",14, (0.4613, -0.6189,  1.1921)],
                    ["HO2'",14, (0.2499, -1.5749,  1.1568)],
                    [" C4'",13, (0.3316, 1.4845, 0.000)],
                    [" O3'",13, (0.5685, -0.6954,  1.0960)],
                    [" H3'",13, (0.3202, -0.4010, -0.8356)],
                    [" C5'",12, (0.6616, 1.3553, 0.000)],
                    [" H4'",12, (0.3168, -0.5077, -0.7763)],
                    [" O5'",11, (0.5217, 1.3206, 0.000)],
                    [" H5'",11, (0.3193, -0.4703,  0.7986)],
                    ["H5''",11, (0.3199, -0.4711, -0.7970)],
                    [" P  ",10, (0.8180, 1.3676, 0.000)],
                    [" OP1", 9, (0.4594, 1.4113, 0.000)],
                    [" OP2", 9, (0.4582, -0.9215,  1.0698)],

                    [" C2 ",16, (0.6650, 1.2325, 0.0000)],
                    [" O2 ",16, (-0.0001, 2.2799, 0.0000)],
                    [" N3 ",16, (2.0175, 1.2603, 0.0000)],
                    [" C4 ",16, (2.7090, 0.1210, 0.0000)],
                    [" N4 ",16, (4.0423, 0.1969, 0.0000)],
                    [" C5 ",16, (2.0635, -1.1476, 0.0000)],
                    [" C6 ",16, (0.7250, -1.1627, 0.0000)],
                    [" H42",16, (4.5791, -0.6226, 0.0000)],
                    [" H41",16, (4.4833, 1.0723, 0.0000)],
                    [" H5 ",16, (2.5806, -1.9803, 0.0000)],
                    [" H6 ",16, (0.2622, -2.0258, 0.0000)],
                ],
                [ # 29 G
                    [" O4'",0, (-0.4082, 1.3525, 0.0000)],
                    [" C1'",0, (0.0000, 0.0000, 0.0000)],
                    [" C2'",0, (1.5292, 0.0000, 0.0000)],
                    [" N9 ",0, (-0.5661, -0.6642, -1.1894)],
                    [" H1'",0, (-0.3326, -0.4415,  0.8101)],
                    [" C3'",14, (0.3028, 1.4927, 0.000)],
                    [" H2'",14, (0.3582, -0.4393, -0.7998)],
                    [" O2'",14, (0.4613, -0.6189,  1.1921)],
                    ["HO2'",14, (0.2499, -1.5749,  1.1568)],
                    [" C4'",13, (0.3316, 1.4845, 0.000)],
                    [" O3'",13, (0.5685, -0.6954,  1.0960)],
                    [" H3'",13, (0.3202, -0.4010, -0.8356)],
                    [" C5'",12, (0.6616, 1.3553, 0.000)],
                    [" H4'",12, (0.3168, -0.5077, -0.7763)],
                    [" O5'",11, (0.5217, 1.3206, 0.000)],
                    [" H5'",11, (0.3193, -0.4703,  0.7986)],
                    ["H5''",11, (0.3199, -0.4711, -0.7970)],
                    [" P  ",10, (0.8180, 1.3676, 0.000)],
                    [" OP1", 9, (0.4594, 1.4113, 0.000)],
                    [" OP2", 9, (0.4582, -0.9215,  1.0698)],
                    [" N1 ",16, (2.7458, 2.8461, 0.0000)],
                    [" C2 ",16, (1.4286, 3.2360, 0.0000)],
                    [" N2 ",16, (1.1989, 4.5575, 0.0000)],
                    [" N3 ",16, (0.4087, 2.3932, 0.0000)],
                    [" C4 ",16, (0.8167, 1.1068, 0.0000)],
                    [" C5 ",16, (2.1036, 0.6115, 0.0000)],
                    [" C6 ",16, (3.1883, 1.5266, 0.0000)],
                    [" O6 ",16, (4.4006, 1.2842, 0.0000)],
                    [" N7 ",16, (2.0980, -0.7759, 0.0000)],
                    [" C8 ",16, (0.8317, -1.0936, 0.0000)],
                    [" H1 ",16, (3.4279, 3.5496, 0.0000)],
                    [" H22",16, (0.2781, 4.8947, 0.0000)],
                    [" H21",16, (1.9487, 5.1879, 0.0000)],
                    [" H8 ",16, (0.5085, -2.0185, 0.0000)],
                ],
                [ # 30 U
                    [" O4'",0, (-0.4082, 1.3525, 0.0000)],
                    [" C1'",0, (0.0000, 0.0000, 0.0000)],
                    [" C2'",0, (1.5292, 0.0000, 0.0000)],
                    [" N1 ",0, (-0.5661, -0.6642, -1.1894)],
                    [" H1'",0, (-0.3326, -0.4415,  0.8101)],
                    [" C3'",14, (0.3028, 1.4927, 0.000)],
                    [" H2'",14, (0.3582, -0.4393, -0.7998)],
                    [" O2'",14, (0.4613, -0.6189,  1.1921)],
                    ["HO2'",14, (0.2499, -1.5749,  1.1568)],
                    [" C4'",13, (0.3316, 1.4845, 0.000)],
                    [" O3'",13, (0.5685, -0.6954,  1.0960)],
                    [" H3'",13, (0.3202, -0.4010, -0.8356)],
                    [" C5'",12, (0.6616, 1.3553, 0.000)],
                    [" H4'",12, (0.3168, -0.5077, -0.7763)],
                    [" O5'",11, (0.5217, 1.3206, 0.000)],
                    [" H5'",11, (0.3193, -0.4703,  0.7986)],
                    ["H5''",11, (0.3199, -0.4711, -0.7970)],
                    [" P  ",10, (0.8180, 1.3676, 0.000)],
                    [" OP1", 9, (0.4594, 1.4113, 0.000)],
                    [" OP2", 9, (0.4582, -0.9215,  1.0698)],
                    [" C2 ",16, (0.6307, 1.2305, 0.0000)],
                    [" O2 ",16, (0.0260, 2.2886, 0.0000)],
                    [" N3 ",16, (2.0031, 1.1816, 0.0000)],
                    [" C4 ",16, (2.7953, 0.0532, 0.0000)],
                    [" O4 ",16, (4.0212, 0.1751, 0.0000)],
                    [" C5 ",16, (2.0746, -1.1833, 0.0000)],
                    [" C6 ",16, (0.7378, -1.1648, 0.0000)],
                    [" H3 ",16, (2.4701, 2.0428, 0.0000)],
                    [" H5 ",16, (2.5579, -2.0361, 0.0000)],
                    [" H6 ",16, (0.2681, -2.0239, 0.0000)],
                ],
                [ # 31 RX
                    [" O4'",0, (-0.4082, 1.3525, 0.0000)],
                    [" C1'",0, (0.0000, 0.0000, 0.0000)],
                    [" C2'",0, (1.5292, 0.0000, 0.0000)],
                    [" H1'",0, (-0.3326, -0.4415,  0.8101)],
                    [" C3'",14, (0.3028, 1.4927, 0.000)],
                    [" H2'",14, (0.3582, -0.4393, -0.7998)],
                    [" O2'",14, (0.4613, -0.6189,  1.1921)],
                    ["HO2'",14, (0.2499, -1.5749,  1.1568)],
                    [" C4'",13, (0.3316, 1.4845, 0.000)],
                    [" O3'",13, (0.5685, -0.6954,  1.0960)],
                    [" H3'",13, (0.3202, -0.4010, -0.8356)],
                    [" C5'",12, (0.6616, 1.3553, 0.000)],
                    [" H4'",12, (0.3168, -0.5077, -0.7763)],
                    [" O5'",11, (0.5217, 1.3206, 0.000)],
                    [" H5'",11, (0.3193, -0.4703,  0.7986)],
                    ["H5''",11, (0.3199, -0.4711, -0.7970)],
                    [" P  ",10, (0.8180, 1.3676, 0.000)],
                    [" OP1", 9, (0.4594, 1.4113, 0.000)],
                    [" OP2", 9, (0.4582, -0.9215,  1.0698)],
                ],
            ]

        else:
            # USE PHOSPHATE FRAME
            self.aa2long=[
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #0  ala
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), #1  arg
                (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), #2  asn
                (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), #3  asp
                (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), #4  cys
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), #5  gln
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), #6  glu
                (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), #7  gly
                (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","2HE ",  None,  None,  None,  None,  None,  None), #8  his
                (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), #9  ile
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), #10 leu
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), #11 lys
                (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), #12 met
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","2HD ","1HE ","2HE "," HZ ",  None,  None,  None,  None), #13 phe
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), #14 pro
                (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), #15 ser
                (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), #16 thr
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE "," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), #17 trp
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE ","2HE ","2HD "," HH ",  None,  None,  None,  None), #18 tyr
                (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), #19 val
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #20 unk
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #21 mask
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N6 ",  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #22  DA
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #23  DC
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N2 "," O6 ",  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #24  DG
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C7 "," C6 ",  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H3 "," H71"," H72"," H73"," H6 ",  None), #25  DT
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'",  None,  None,  None,  None,  None,  None), #26  DX (unk DNA)
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," N3 "," C4 "," C5 "," C6 "," N6 "," N7 "," C8 "," N9 ",  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #27   A
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #28   C
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," N2 "," N3 "," C4 "," C5 "," C6 "," O6 "," N7 "," C8 "," N9 "," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #29   G
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H3 "," H5 "," H6 ",  None,  None,  None), #30   U
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'",  None,  None,  None,  None,  None,  None), #31  RX (unk RNA)
                (" N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","1HD ",  None,  None,  None,  None,  None,  None), #-1 his_d
            ]

            # build the "alternate" sc mapping
            self.aa2longalt=[
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ala
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), # arg
                (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), # asn
                (" N  "," CA "," C  "," O  "," CB "," CG "," OD2"," OD1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
                (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), # gln
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE2"," OE1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), # glu
                (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
                (" N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","2HE ",  None,  None,  None,  None,  None,  None), # his
                (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), # ile
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), # leu
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), # lys
                (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), # met
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","2HD ","2HE "," HZ ","1HE ","1HD "," HA ","1HB ","2HB ",  None,  None,  None,  None), # phe
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), # pro
                (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
                (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), # thr
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE "," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), # trp
                (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ "," OH ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","2HE ","1HE ","1HD "," HH ",  None,  None,  None,  None), # tyr
                (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), # val
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # unk
                (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # mask
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N6 ",  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #22  DA
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #23  DC
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N9 "," C4 "," N3 "," C2 "," N1 "," C6 "," C5 "," N7 "," C8 "," N2 "," O6 ",  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #24  DG
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C7 "," C6 ",  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'"," H3 "," H71"," H72"," H73"," H6 ",  None), #25  DT
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C2'"," C1'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"H5''"," H5'"," H4'"," H3'","H2''"," H2'"," H1'",  None,  None,  None,  None,  None,  None), #26  DX (unk DNA)
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," N3 "," C4 "," C5 "," C6 "," N6 "," N7 "," C8 "," N9 ",  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H2 "," H61"," H62"," H8 ",  None,  None), #27   A
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," N4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H42"," H41"," H5 "," H6 ",  None,  None), #28   C
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," N2 "," N3 "," C4 "," C5 "," C6 "," O6 "," N7 "," C8 "," N9 "," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H1 "," H22"," H21"," H8 ",  None,  None), #29   G
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'"," N1 "," C2 "," O2 "," N3 "," C4 "," O4 "," C5 "," C6 ",  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'"," H3 "," H5 "," H6 ",  None,  None,  None), #30   U
                (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'"," O2'",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H5'","H5''"," H4'"," H3'"," H2'","HO2'"," H1'",  None,  None,  None,  None,  None,  None), #31  RX (unk RNA)
            ]

            self.aa2type = [
                ("Nbb", "CAbb","CObb","OCbb","CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # ala
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "CH2", "NtrR","aroC","Narg","Narg",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol","Hpol","Hpol","Hpol"), # arg
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CNH2","ONH2","NH2O",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hpol","Hpol",  None,  None,  None,  None,  None,  None,  None), # asn
                ("Nbb", "CAbb","CObb","OCbb","CH2", "COO", "OOC", "OOC",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
                ("Nbb", "CAbb","CObb","OCbb","CH2", "SH1",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","HS",    None,  None,  None,  None,  None,  None,  None,  None), # cys
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "CNH2","ONH2","NH2O",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol",  None,  None,  None,  None,  None), # gln
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "COO", "OOC", "OOC",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None), # glu
                ("Nbb", "CAbb","CObb","OCbb",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "Nhis","aroC","aroC","Ntrp",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hpol","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # his
                ("Nbb", "CAbb","CObb","OCbb","CH1", "CH2", "CH3", "CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None), # ile
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH1", "CH3", "CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None), # leu
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "CH2", "CH2", "Nlys",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol","Hpol"), # lys
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "S",   "CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None), # met
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "aroC","aroC","aroC","aroC","aroC",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Haro","Haro","Haro","Haro","Haro",  None,  None,  None,  None), # phe
                ("Npro","CAbb","CObb","OCbb","CH2", "CH2", "CH2",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # pro
                ("Nbb", "CAbb","CObb","OCbb","CH2", "OH",    None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hpol","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # ser
                ("Nbb", "CAbb","CObb","OCbb","CH1", "OH",  "CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hpol","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # thr
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "aroC","CH0", "CH0", "aroC","Ntrp","aroC","aroC","aroC",  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Haro","Hapo","Hapo","Hapo","Hpol","Haro","Haro","Haro","Haro",  None,  None,  None), # trp
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "aroC","aroC","aroC","aroC","CH0", "OHY",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Haro","Haro","Haro","Haro","Hapo","Hapo","Hapo","Hpol",  None,  None,  None,  None), # tyr
                ("Nbb", "CAbb","CObb","OCbb","CH1", "CH3", "CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None), # val
                ("Nbb", "CAbb","CObb","OCbb","CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # unk
                ("Nbb", "CAbb","CObb","OCbb","CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # mask
                ("OOC","Phos", "OOC", "Oet2","CH2", "CH1", "Oet3","CH1", "Oet2","CH2", "CH1", "Npro","aroC","Nhis","aroC","Nhis","aroC","aroC","Nhis","aroC","NH2O",  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Haro","Hpol","Hpol","Haro",  None,  None), # DA
                ("OOC","Phos", "OOC", "Oet2","CH2", "CH1", "Oet3","CH1", "Oet2","CH2", "CH1", "Npro","CObb","OCbb","Nhis","aroC","NH2O","aroC","aroC",  None,  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol","Haro","Haro",  None,  None), # DC
                ("OOC","Phos", "OOC", "Oet2","CH2", "CH1", "Oet3","CH1", "Oet2","CH2", "CH1", "Npro","aroC","Nhis","aroC","Ntrp","CObb","aroC","Nhis","aroC","NH2O","OCbb",  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol","Hpol","Haro",  None,  None), # DG
                ("OOC","Phos", "OOC", "Oet2","CH2", "CH1", "Oet3","CH1", "Oet2","CH2", "CH1", "Npro","CObb","OCbb","Ntrp","CObb","OCbb","aroC","CH3", "aroC",  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo","Hapo","Hapo","Haro",  None), # DT
                ("OOC","Phos", "OOC", "Oet2","CH2", "CH1", "Oet3","CH1", "Oet2","CH2", "CH1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None), #  DX (unk DNA)
                ("OOC","Phos", "OOC", "Oet2","CH2", "CH1", "Oet3","CH1", "Oet2","CH1", "CH2", "OH",  "Nhis","aroC","Nhis","aroC","aroC","aroC","NH2O","Nhis","aroC","Npro",  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo","Haro","Hpol","Hpol","Haro",  None,  None), # A
                ("OOC","Phos", "OOC", "Oet2","CH2", "CH1", "Oet3","CH1", "Oet2","CH1", "CH2", "OH",  "Npro","CObb","OCbb","Nhis","aroC","NH2O","aroC","aroC",  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo","Hpol","Hpol","Haro","Haro",  None,  None), # C
                ("OOC","Phos", "OOC", "Oet2","CH2", "CH1", "Oet3","CH1", "Oet2","CH1", "CH2", "OH",  "Ntrp","aroC","NH2O","Nhis","aroC","aroC","CObb","OCbb","Nhis","aroC","Npro","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo","Hpol","Hpol","Hpol","Haro",  None,  None), # G
                ("OOC","Phos", "OOC", "Oet2","CH2", "CH1", "Oet3","CH1", "Oet2","CH1", "CH2", "OH",  "Npro","CObb","OCbb","Ntrp","CObb","OCbb","aroC","aroC",  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo","Hpol","Hapo","Haro",  None,  None,  None), # U
                ("OOC","Phos", "OOC", "Oet2","CH2", "CH1", "Oet3","CH1", "Oet2","CH1", "CH2", "OH",    None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hapo",  None,  None,  None,  None,  None,  None), # RX (unk RNA)
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "Nhis","aroC","aroC","Ntrp",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hpol","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # HIS-D NOT CORRECT!!!!!!!!!!
                ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "Nhis","aroC","aroC","Ntrp",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,"HNbb","Hapo","Hapo","Hapo","Hpol","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # HIS-D NOT CORRECT!!!!!!!!!!
                (None, "genAl",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Al
                (None, "genAs",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # As
                (None, "genAu",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Au
                (None, "genB",   None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # B
                (None, "genBe",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Be
                (None, "genBr",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Br
                (None, "genC",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # C
                (None, "genCa",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ca
                (None, "genCl",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Cl
                (None, "genCo",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Co
                (None, "genCr",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Cr
                (None, "genCu",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Cu
                (None, "genF",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # F
                (None, "genFe",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Fe
                (None, "genHg",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Hg
                (None, "genI",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # I
                (None, "genIr",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ir
                (None, "genK",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # K
                (None, "genLi",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Li
                (None, "genMg",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Mg
                (None, "genMn",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Mn
                (None, "genMo",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Mo
                (None, "genN",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # N
                (None, "genNi",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ni
                (None, "genO",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # O
                (None, "genOs",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Os
                (None, "genP",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # P
                (None, "genPb",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pb
                (None, "genPd",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pd
                (None, "genPr",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pr
                (None, "genPt",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pt
                (None, "genRe",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Re
                (None, "genRh",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Rh
                (None, "genRu",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ru
                (None, "genS",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # S
                (None, "genSb",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Sb
                (None, "genSe",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Se
                (None, "genSi",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Si
                (None, "genSn",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Sn
                (None, "genTb",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Tb
                (None, "genTe",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Te
                (None, "genU",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # U
                (None, "genW",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # W   
                (None, "genV",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # V
                (None, "genY",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Y
                (None, "genZn",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Zn
                (None, "genATM",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # ATM

            ]

            self.aa2elt = [
                ("N","C","C","O","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H",None,None,None,None,None,None,None,None),#ala
                ("N","C","C","O","C","C","C","N","C","N","N",None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H","H","H","H","H"),#arg
                ("N","C","C","O","C","C","O","N",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H",None,None,None,None,None,None,None),#asn
                ("N","C","C","O","C","C","O","O",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H",None,None,None,None,None,None,None,None,None),#asp
                ("N","C","C","O","C","S",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H",None,None,None,None,None,None,None,None),#cys
                ("N","C","C","O","C","C","C","O","N",None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H",None,None,None,None,None),#gln
                ("N","C","C","O","C","C","C","O","O",None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H",None,None,None,None,None,None,None),#glu
                ("N","C","C","O",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H",None,None,None,None,None,None,None,None,None,None),#gly
                ("N","C","C","O","C","C","N","C","C","N",None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H",None,None,None,None,None,None),#his
                ("N","C","C","O","C","C","C","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H","H","H",None,None),#ile
                ("N","C","C","O","C","C","C","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H","H","H",None,None),#leu
                ("N","C","C","O","C","C","C","C","N",None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H","H","H","H","H"),#lys
                ("N","C","C","O","C","C","S","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H",None,None,None,None),#met
                ("N","C","C","O","C","C","C","C","C","C","C",None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H",None,None,None,None),#phe
                ("N","C","C","O","C","C","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H",None,None,None,None,None,None),#pro
                ("N","C","C","O","C","O",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H",None,None,None,None,None,None,None,None),#ser
                ("N","C","C","O","C","O","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H",None,None,None,None,None,None),#thr
                ("N","C","C","O","C","C","C","C","C","C","N","C","C","C",None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H","H",None,None,None),#trp
                ("N","C","C","O","C","C","C","C","C","C","C","O",None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H",None,None,None,None),#tyr
                ("N","C","C","O","C","C","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H","H",None,None,None,None),#val
                ("N","C","C","O","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H",None,None,None,None,None,None,None,None),#unk
                ("N","C","C","O","C",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H",None,None,None,None,None,None,None,None),#mask
                ("O","P","O","O","C","C","O","C","O","C","C","N","C","N","C","N","C","C","N","C","N",None,None,"H","H","H","H","H","H","H","H","H","H","H",None,None),#DA
                ("O","P","O","O","C","C","O","C","O","C","C","N","C","O","N","C","N","C","C",None,None,None,None,"H","H","H","H","H","H","H","H","H","H","H",None,None),#DC
                ("O","P","O","O","C","C","O","C","O","C","C","N","C","N","C","N","C","C","N","C","N","O",None,"H","H","H","H","H","H","H","H","H","H","H",None,None),#DG
                ("O","P","O","O","C","C","O","C","O","C","C","N","C","O","N","C","O","C","C","C",None,None,None,"H","H","H","H","H","H","H","H","H","H","H","H",None),#DT
                ("O","P","O","O","C","C","O","C","O","C","C",None,None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H",None,None,None,None,None,None),#DX
                ("O","P","O","O","C","C","O","C","O","C","C","O","N","C","N","C","C","C","N","N","C","N",None,"H","H","H","H","H","H","H","H","H","H","H",None,None),#A
                ("O","P","O","O","C","C","O","C","O","C","C","O","N","C","O","N","C","N","C","C",None,None,None,"H","H","H","H","H","H","H","H","H","H","H",None,None),#C
                ("O","P","O","O","C","C","O","C","O","C","C","O","N","C","N","N","C","C","C","O","N","C","N","H","H","H","H","H","H","H","H","H","H","H",None,None),#G
                ("O","P","O","O","C","C","O","C","O","C","C","O","N","C","O","N","C","O","C","C",None,None,None,"H","H","H","H","H","H","H","H","H","H",None,None,None),#U
                ("O","P","O","O","C","C","O","C","O","C","C","O",None,None,None,None,None,None,None,None,None,None,None,"H","H","H","H","H","H","H","H",None,None,None,None,None),#RX
            ]

            # frames for generic FAPE
            ##NOTE: 1st entry must be "backbone frame"
            self.frames=[
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # ala
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "], [" CG "," CD "," NE "] ],  # arg
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # asn
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # asp
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "] ],  # cys
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "] ],  # gln
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "] ],  # glu
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # gly
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # his
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG1"] ],  # ile
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # leu
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "], [" CG "," CD "," CE "] ],  # lys
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," SD "] ],  # met
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # phe
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "]],  # pro
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," OG "] ],  # ser
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," OG1"] ],  # thr
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # trp
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # tyr
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "] ],  # val
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # unk
                [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # mask
                [ [" OP1"," P  "," OP2"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N9 "], [" C4'"," C3'"," O3'"] ], #DA
                [ [" OP1"," P  "," OP2"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N1 "], [" C4'"," C3'"," O3'"] ], #DC
                [ [" OP1"," P  "," OP2"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N9 "], [" C4'"," C3'"," O3'"] ], #DG
                [ [" OP1"," P  "," OP2"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N1 "], [" C4'"," C3'"," O3'"] ], #DT
                [ [" OP1"," P  "," OP2"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C4'"," C3'"," O3'"] ], #DX
                [ [" OP1"," P  "," OP2"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N9 "], [" C4'"," C3'"," O3'"] ], #A
                [ [" OP1"," P  "," OP2"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N1 "], [" C4'"," C3'"," O3'"] ], #C
                [ [" OP1"," P  "," OP2"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N9 "], [" C4'"," C3'"," O3'"] ], #G
                [ [" OP1"," P  "," OP2"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C2'"," C1'"," N1 "], [" C4'"," C3'"," O3'"] ], #U
                [ [" OP1"," P  "," OP2"], [" OP1"," P  "," O5'"], [" P  "," O5'"," C5'"], [" O5'"," C5'"," C4'"], [" C5'"," C4'"," C3'"], [" C5'"," C4'"," O4'"], [" C4'"," O4'"," C1'"], [" C4'"," C3'"," O3'"] ], #RX
            ]

            #fd Rosetta ideal coords
            #fd   - uses same "frame-building" as AF2
            # FRAMES:
            #   base = 0
            #   omega/phi/psi = 1-3 (omega unused)
            #   chi_1-4(prot) = 4-7
            #   CB_bend = 8
            #   NA alpha/beta/gamma/delta = 9-12  (NA epsilon/zeta no frame)
            #   NA nu2/nu1/nu0 = 13-15
            #   chi_1(NA) = 16
            self.ideal_coords = [
                [ # 0 ala
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3341, -0.4928,  0.9132)],
                    [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
                    ['1HB ', 8, (-0.1265, -1.7863, -1.1851)],
                    ['2HB ', 8, (-1.6173, -0.8147, -1.1541)],
                    ['3HB ', 8, (-0.2229, -0.2744, -2.1172)],
                ],
                [ # 1 arg
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3467, -0.5055,  0.9018)],
                    [' CB ', 8, (-0.5042,-0.7698,-1.2118)],
                    ['1HB ', 4, ( 0.3635, -0.5318,  0.8781)],
                    ['2HB ', 4, ( 0.3639, -0.5323, -0.8789)],
                    [' CG ', 4, (0.6396,1.3794, 0.000)],
                    ['1HG ', 5, (0.3639, -0.5139,  0.8900)],
                    ['2HG ', 5, (0.3641, -0.5140, -0.8903)],
                    [' CD ', 5, (0.5492,1.3801, 0.000)],
                    ['1HD ', 6, (0.3637, -0.5135,  0.8895)],
                    ['2HD ', 6, (0.3636, -0.5134, -0.8893)],
                    [' NE ', 6, (0.5423,1.3491, 0.000)],
                    [' NH1', 7, (0.2012,2.2965, 0.000)],
                    [' NH2', 7, (2.0824,1.0030, 0.000)],
                    [' CZ ', 7, (0.7650,1.1090, 0.000)],
                    [' HE ', 7, (0.4701,-0.8955, 0.000)],
                    ['1HH1', 7, (-0.8059,2.3776, 0.000)],
                    ['1HH2', 7, (2.5160,0.0898, 0.000)],
                    ['2HH1', 7, (0.7745,3.1277, 0.000)],
                    ['2HH2', 7, (2.6554,1.8336, 0.000)],
                ],
                [ # 2 asn
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3233, -0.4967,  0.9162)],
                    [' CB ', 8, (-0.5341,-0.7799,-1.1874)],
                    ['1HB ', 4, ( 0.3641, -0.5327,  0.8795)],
                    ['2HB ', 4, ( 0.3639, -0.5323, -0.8789)],
                    [' CG ', 4, (0.5778,1.3881, 0.000)],
                    [' ND2', 5, (0.5839,-1.1711, 0.000)],
                    [' OD1', 5, (0.6331,1.0620, 0.000)],
                    ['1HD2', 5, (1.5825, -1.2322, 0.000)],
                    ['2HD2', 5, (0.0323, -2.0046, 0.000)],
                ],
                [ # 3 asp
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3233, -0.4967,  0.9162)],
                    [' CB ', 8, (-0.5162,-0.7757,-1.2144)],
                    ['1HB ', 4, ( 0.3639, -0.5324,  0.8791)],
                    ['2HB ', 4, ( 0.3640, -0.5325, -0.8792)],
                    [' CG ', 4, (0.5926,1.4028, 0.000)],
                    [' OD1', 5, (0.5746,1.0629, 0.000)],
                    [' OD2', 5, (0.5738,-1.0627, 0.000)],
                ],
                [ # 4 cys
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3481, -0.5059,  0.9006)],
                    [' CB ', 8, (-0.5046,-0.7727,-1.2189)],
                    ['1HB ', 4, ( 0.3639, -0.5324,  0.8791)],
                    ['2HB ', 4, ( 0.3638, -0.5322, -0.8787)],
                    [' SG ', 4, (0.7386,1.6511, 0.000)],
                    [' HG ', 5, (0.1387,1.3221, 0.000)],
                ],
                [ # 5 gln
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3363, -0.5013,  0.9074)],
                    [' CB ', 8, (-0.5226,-0.7776,-1.2109)],
                    ['1HB ', 4, ( 0.3638, -0.5323,  0.8789)],
                    ['2HB ', 4, ( 0.3638, -0.5322, -0.8788)],
                    [' CG ', 4, (0.6225,1.3857, 0.000)],
                    ['1HG ', 5, ( 0.3531, -0.5156,  0.8931)],
                    ['2HG ', 5, ( 0.3531, -0.5156, -0.8931)],
                    [' CD ', 5, (0.5788,1.4021, 0.000)],
                    [' NE2', 6, (0.5908,-1.1895, 0.000)],
                    [' OE1', 6, (0.6347,1.0584, 0.000)],
                    ['1HE2', 6, (1.5825, -1.2525, 0.000)],
                    ['2HE2', 6, (0.0380, -2.0229, 0.000)],
                ],
                [ # 6 glu
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3363, -0.5013,  0.9074)],
                    [' CB ', 8, (-0.5197,-0.7737,-1.2137)],
                    ['1HB ', 4, ( 0.3638, -0.5323,  0.8789)],
                    ['2HB ', 4, ( 0.3638, -0.5322, -0.8788)],
                    [' CG ', 4, (0.6287,1.3862, 0.000)],
                    ['1HG ', 5, ( 0.3531, -0.5156,  0.8931)],
                    ['2HG ', 5, ( 0.3531, -0.5156, -0.8931)],
                    [' CD ', 5, (0.5850,1.3849, 0.000)],
                    [' OE1', 6, (0.5752,1.0618, 0.000)],
                    [' OE2', 6, (0.5741,-1.0635, 0.000)],
                ],
                [ # 7 gly
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    ['1HA ', 0, ( -0.3676, -0.5329,  0.8771)],
                    ['2HA ', 0, ( -0.3674, -0.5325, -0.8765)],
                ],
                [ # 8 his
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3299, -0.5180,  0.9001)],
                    [' CB ', 8, (-0.5163,-0.7809,-1.2129)],
                    ['1HB ', 4, ( 0.3640, -0.5325,  0.8793)],
                    ['2HB ', 4, ( 0.3637, -0.5321, -0.8786)],
                    [' CG ', 4, (0.6016,1.3710, 0.000)],
                    [' CD2', 5, (0.8918,-1.0184, 0.000)],
                    [' CE1', 5, (2.0299,0.8564, 0.000)],
                    ['1HE ', 5, (2.8542, 1.5693,  0.000)],
                    ['2HD ', 5, ( 0.6584, -2.0835, 0.000) ],
                    [' ND1', 6, (-1.8631, -1.0722,  0.000)],
                    [' NE2', 6, (-1.8625,  1.0707, 0.000)],
                    ['2HE ', 6, (-1.5439,  2.0292, 0.000)],
                ],
                [ # 9 ile
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3405, -0.5028,  0.9044)],
                    [' CB ', 8, (-0.5140,-0.7885,-1.2184)],
                    [' HB ', 4, (0.3637, -0.4714,  0.9125)],
                    [' CG1', 4, (0.5339,1.4348,0.000)],
                    [' CG2', 4, (0.5319,-0.7693,-1.1994)],
                    ['1HG2', 4, (1.6215, -0.7588, -1.1842)],
                    ['2HG2', 4, (0.1785, -1.7986, -1.1569)],
                    ['3HG2', 4, (0.1773, -0.3016, -2.1180)],
                    [' CD1', 5, (0.6106,1.3829, 0.000)],
                    ['1HG1', 5, (0.3637, -0.5338,  0.8774)],
                    ['2HG1', 5, (0.3640, -0.5322, -0.8793)],
                    ['1HD1', 5, (1.6978,  1.3006, 0.000)],
                    ['2HD1', 5, (0.2873,  1.9236, -0.8902)],
                    ['3HD1', 5, (0.2888, 1.9224, 0.8896)],
                ],
                [ # 10 leu
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.525, -0.000, -0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3435, -0.5040,  0.9027)],
                    [' CB ', 8, (-0.5175,-0.7692,-1.2220)],
                    ['1HB ', 4, ( 0.3473, -0.5346,  0.8827)],
                    ['2HB ', 4, ( 0.3476, -0.5351, -0.8836)],
                    [' CG ', 4, (0.6652,1.3823, 0.000)],
                    [' CD1', 5, (0.5083,1.4353, 0.000)],
                    [' CD2', 5, (0.5079,-0.7600,1.2163)],
                    [' HG ', 5, (0.3640, -0.4825, -0.9075)],
                    ['1HD1', 5, (1.5984,  1.4353, 0.000)],
                    ['2HD1', 5, (0.1462,  1.9496, -0.8903)],
                    ['3HD1', 5, (0.1459, 1.9494, 0.8895)],
                    ['1HD2', 5, (1.5983, -0.7606,  1.2158)],
                    ['2HD2', 5, (0.1456, -0.2774,  2.1243)],
                    ['3HD2', 5, (0.1444, -1.7871,  1.1815)],
                ],
                [ # 11 lys
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3335, -0.5005,  0.9097)],
                    ['1HB ', 4, ( 0.3640, -0.5324,  0.8791)],
                    ['2HB ', 4, ( 0.3639, -0.5324, -0.8790)],
                    [' CB ', 8, (-0.5259,-0.7785,-1.2069)],
                    ['1HG ', 5, (0.3641, -0.5229,  0.8852)],
                    ['2HG ', 5, (0.3637, -0.5227, -0.8841)],
                    [' CG ', 4, (0.6291,1.3869, 0.000)],
                    [' CD ', 5, (0.5526,1.4174, 0.000)],
                    ['1HD ', 6, (0.3641, -0.5239,  0.8848)],
                    ['2HD ', 6, (0.3638, -0.5219, -0.8850)],
                    [' CE ', 6, (0.5544,1.4170, 0.000)],
                    [' NZ ', 7, (0.5566,1.3801, 0.000)],
                    ['1HE ', 7, (0.4199, -0.4638,  0.9482)],
                    ['2HE ', 7, (0.4202, -0.4631, -0.8172)],
                    ['1HZ ', 7, (1.6223, 1.3980, 0.0658)],
                    ['2HZ ', 7, (0.2970,  1.9326, -0.7584)],
                    ['3HZ ', 7, (0.2981, 1.9319, 0.8909)],
                ],
                [ # 12 met
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3303, -0.4990,  0.9108)],
                    ['1HB ', 4, ( 0.3635, -0.5318,  0.8781)],
                    ['2HB ', 4, ( 0.3641, -0.5326, -0.8795)],
                    [' CB ', 8, (-0.5331,-0.7727,-1.2048)],
                    ['1HG ', 5, (0.3637, -0.5256,  0.8823)],
                    ['2HG ', 5, (0.3638, -0.5249, -0.8831)],
                    [' CG ', 4, (0.6298,1.3858,0.000)],
                    [' SD ', 5, (0.6953,1.6645,0.000)],
                    [' CE ', 6, (0.3383,1.7581,0.000)],
                    ['1HE ', 6, (1.7054,  2.0532, -0.0063)],
                    ['2HE ', 6, (0.1906,  2.3099, -0.9072)],
                    ['3HE ', 6, (0.1917, 2.3792, 0.8720)],
                ],
                [ # 13 phe
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3303, -0.4990,  0.9108)],
                    ['1HB ', 4, ( 0.3635, -0.5318,  0.8781)],
                    ['2HB ', 4, ( 0.3641, -0.5326, -0.8795)],
                    [' CB ', 8, (-0.5150,-0.7729,-1.2156)],
                    [' CG ', 4, (0.6060,1.3746, 0.000)],
                    [' CD1', 5, (0.7078,1.1928, 0.000)],
                    [' CD2', 5, (0.7084,-1.1920, 0.000)],
                    [' CE1', 5, (2.0900,1.1940, 0.000)],
                    [' CE2', 5, (2.0897,-1.1939, 0.000)],
                    [' CZ ', 5, (2.7809, 0.000, 0.000)],
                    ['1HD ', 5, (0.1613, 2.1362, 0.000)],
                    ['2HD ', 5, (0.1621, -2.1360, 0.000)],
                    ['1HE ', 5, (2.6335,  2.1384, 0.000)],
                    ['2HE ', 5, (2.6344, -2.1378, 0.000)],
                    [' HZ ', 5, (3.8700, 0.000, 0.000)],
                ],
                [ # 14 pro
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' HA ', 0, (-0.3868, -0.5380,  0.8781)],
                    ['1HB ', 4, ( 0.3762, -0.5355,  0.8842)],
                    ['2HB ', 4, ( 0.3762, -0.5355, -0.8842)],
                    [' CB ', 8, (-0.5649,-0.5888,-1.2966)],
                    [' CG ', 4, (0.3657,1.4451,0.0000)],
                    [' CD ', 5, (0.3744,1.4582, 0.0)],
                    ['1HG ', 5, (0.3798, -0.5348,  0.8830)],
                    ['2HG ', 5, (0.3798, -0.5348, -0.8830)],
                    ['1HD ', 6, (0.3798, -0.5348,  0.8830)],
                    ['2HD ', 6, (0.3798, -0.5348, -0.8830)],
                ],
                [ # 15 ser
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3425, -0.5041,  0.9048)],
                    ['1HB ', 4, ( 0.3637, -0.5321,  0.8786)],
                    ['2HB ', 4, ( 0.3636, -0.5319, -0.8782)],
                    [' CB ', 8, (-0.5146,-0.7595,-1.2073)],
                    [' OG ', 4, (0.5021,1.3081, 0.000)],
                    [' HG ', 5, (0.2647, 0.9230, 0.000)],
                ],
                [ # 16 thr
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3364, -0.5015,  0.9078)],
                    [' HB ', 4, ( 0.3638, -0.5006,  0.8971)],
                    ['1HG2', 4, ( 1.6231, -0.7142, -1.2097)],
                    ['2HG2', 4, ( 0.1792, -1.7546, -1.2237)],
                    ['3HG2', 4, ( 0.1808, -0.2222, -2.1269)],
                    [' CB ', 8, (-0.5172,-0.7952,-1.2130)],
                    [' CG2', 4, (0.5334,-0.7239,-1.2267)],
                    [' OG1', 4, (0.4804,1.3506,0.000)],
                    [' HG1', 5, (0.3194,  0.9056, 0.000)],
                ],
                [ # 17 trp
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3436, -0.5042,  0.9031)],
                    ['1HB ', 4, ( 0.3639, -0.5323,  0.8790)],
                    ['2HB ', 4, ( 0.3638, -0.5322, -0.8787)],
                    [' CB ', 8, (-0.5136,-0.7712,-1.2173)],
                    [' CG ', 4, (0.5984,1.3741, 0.000)],
                    [' CD1', 5, (0.8151,1.0921, 0.000)],
                    [' CD2', 5, (0.8753,-1.1538, 0.000)],
                    [' CE2', 5, (2.1865,-0.6707, 0.000)],
                    [' CE3', 5, (0.6541,-2.5366, 0.000)],
                    [' NE1', 5, (2.1309,0.7003, 0.000)],
                    [' CH2', 5, (3.0315,-2.8930, 0.000)],
                    [' CZ2', 5, (3.2813,-1.5205, 0.000)],
                    [' CZ3', 5, (1.7521,-3.3888, 0.000)],
                    ['1HD ', 5, (0.4722, 2.1252,  0.000)],
                    ['1HE ', 5, ( 2.9291,  1.3191,  0.000)],
                    [' HE3', 5, (-0.3597, -2.9356,  0.000)],
                    [' HZ2', 5, (4.3053, -1.1462,  0.000)],
                    [' HZ3', 5, ( 1.5712, -4.4640,  0.000)],
                    [' HH2', 5, ( 3.8700, -3.5898,  0.000)],
                ],
                [ # 18 tyr
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3305, -0.4992,  0.9112)],
                    ['1HB ', 4, ( 0.3642, -0.5327,  0.8797)],
                    ['2HB ', 4, ( 0.3637, -0.5321, -0.8785)],
                    [' CB ', 8, (-0.5305,-0.7799,-1.2051)],
                    [' CG ', 4, (0.6104,1.3840, 0.000)],
                    [' CD1', 5, (0.6936,1.2013, 0.000)],
                    [' CD2', 5, (0.6934,-1.2011, 0.000)],
                    [' CE1', 5, (2.0751,1.2013, 0.000)],
                    [' CE2', 5, (2.0748,-1.2011, 0.000)],
                    [' OH ', 5, (4.1408, 0.000, 0.000)],
                    [' CZ ', 5, (2.7648, 0.000, 0.000)],
                    ['1HD ', 5, (0.1485, 2.1455,  0.000)],
                    ['2HD ', 5, (0.1484, -2.1451,  0.000)],
                    ['1HE ', 5, (2.6200, 2.1450,  0.000)],
                    ['2HE ', 5, (2.6199, -2.1453,  0.000)],
                    [' HH ', 6, (0.3190, 0.9057,  0.000)],
                ],
                [ # 19 val
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3497, -0.5068,  0.9002)],
                    [' CB ', 8, (-0.5105,-0.7712,-1.2317)],
                    [' CG1', 4, (0.5326,1.4252, 0.000)],
                    [' CG2', 4, (0.5177,-0.7693,1.2057)],
                    [' HB ', 4, (0.3541, -0.4754, -0.9148)],
                    ['1HG1', 4, (1.6228,  1.4063,  0.000)],
                    ['2HG1', 4, (0.1790,  1.9457, -0.8898)],
                    ['3HG1', 4, (0.1798, 1.9453, 0.8903)],
                    ['1HG2', 4, (1.6073, -0.7659,  1.1989)],
                    ['2HG2', 4, (0.1586, -0.2971,  2.1203)],
                    ['3HG2', 4, (0.1582, -1.7976,  1.1631)],
                ],
                [ # 20 unk
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3341, -0.4928,  0.9132)],
                    [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
                    ['1HB ', 8, (-0.1265, -1.7863, -1.1851)],
                    ['2HB ', 8, (-1.6173, -0.8147, -1.1541)],
                    ['3HB ', 8, (-0.2229, -0.2744, -2.1172)],
                ],
                [ # 21 mask
                    [' N  ', 0, (-0.5272, 1.3593, 0.000)],
                    [' CA ', 0, (0.000, 0.000, 0.000)],
                    [' C  ', 0, (1.5233, 0.000, 0.000)],
                    [' O  ', 3, (0.6303, 1.0574, 0.000)],
                    [' H  ', 2, (0.4920,-0.8821,  0.0000)],
                    [' HA ', 0, (-0.3341, -0.4928,  0.9132)],
                    [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
                    ['1HB ', 8, (-0.1265, -1.7863, -1.1851)],
                    ['2HB ', 8, (-1.6173, -0.8147, -1.1541)],
                    ['3HB ', 8, (-0.2229, -0.2744, -2.1172)],
                ],
                [ # 22 DA
                    [" OP1", 0, (-0.7319, 1.2920, 0.000)],
                    [" P  ", 0, (0.000, 0.000, 0.000)],
                    [" OP2", 0, (1.4855, 0.000, 0.000)],
                    [" O5'", 0, (-0.4948, -0.8559,  1.2489)],
                    [" C5'", 9, (0.7411, 1.2354, 0.000)],
                    [" C4'",10, (0.5207,  1.4178, 0.000)],
                    [" H5'",10, (0.3748, -0.5360, -0.8839)],
                    ["H5''",10, (0.3744, -0.5368,  0.8842)],
                    [" C3'",11, ( 0.6388,  1.3889, 0.000)],
                    [" H4'",11, ( 0.2823, -0.5105,  0.9326)],
                    [" O4'",11, (0.4804, -0.6610, -1.1947)],
                    [" C1'",13, (0.4913, 1.3316, 0.0000)],
                    [" H1'",14, (0.4561, -0.4898, 0.8726)],
                    [" N9 ",14, (0.4467, -0.7474, -1.1746)],
                    [" C2'",14, (0.4167, 1.4603, 0.0000)],
                    [" H2'",15, (0.4107, -0.5097, -0.8844)],
                    ["H2''",15, (0.4106, -0.5096, 0.8840)],
                    [" O3'",12, ( 0.4966,  1.3432, 0.000)],
                    [" H3'",12, (0.4359, -0.4915, -0.8827)],
                    [" C4 ",16, (0.8119, 1.1084, 0.0000)],
                    [" N3 ",16, (0.4328, 2.3976, 0.0000)],
                    [" C2 ",16, (1.4957, 3.1983, 0.0000)],
                    [" N1 ",16, (2.7960, 2.8816, 0.0000)],
                    [" C6 ",16, (3.1433, 1.5760, 0.0000)],
                    [" C5 ",16, (2.1084, 0.6255, 0.0000)],
                    [" N7 ",16, (2.1145, -0.7627, 0.0000)],
                    [" C8 ",16, (0.8438, -1.0825, 0.0000)],
                    [" N6 ",16, (4.4402, 1.2598, 0.0000)],
                    [" H2 ",16, (1.2740, 4.2755, 0.0000)],
                    [" H8 ",16, (0.4867, -2.1227, 0.0000)],
                    [" H61",16, (5.1313, 1.9828, 0.0000)],
                    [" H62",16, (4.7211, 0.3001, 0.0000)],
                ],
                [ # 23 DC
                    [" OP1", 0, (-0.7319, 1.2920, 0.000)],
                    [" P  ", 0, (0.000, 0.000, 0.000)],
                    [" OP2", 0, (1.4855, 0.000, 0.000)],
                    [" O5'", 0, (-0.4948, -0.8559,  1.2489)],
                    [" C5'", 9, (0.7411, 1.2354, 0.000)],
                    [" C4'",10, (0.5207,  1.4178, 0.000)],
                    [" H5'",10, (0.3748, -0.5360, -0.8839)],
                    ["H5''",10, (0.3744, -0.5368,  0.8842)],
                    [" C3'",11, ( 0.6388,  1.3889, 0.000)],
                    [" H4'",11, ( 0.2823, -0.5105,  0.9326)],
                    [" O4'",11, (0.4804, -0.6610, -1.1947)],
                    [" C1'",13, (0.4913, 1.3316, 0.0000)],
                    [" H1'",14, (0.4561, -0.4898, 0.8726)],
                    [" N1 ",14, (0.4467, -0.7474, -1.1746)],
                    [" C2'",14, (0.4167, 1.4603, 0.0000)],
                    [" H2'",15, (0.4107, -0.5097, -0.8844)],
                    ["H2''",15, (0.4106, -0.5096, 0.8840)],
                    [" O3'",12, ( 0.4966,  1.3432, 0.000)],
                    [" H3'",12, (0.4359, -0.4915, -0.8827)],
                [" C2 ",16, (0.6758, 1.2249, 0.0000)],
                [" O2 ",16, (0.0158, 2.2756, 0.0000)],
                [" N3 ",16, (2.0283, 1.2334, 0.0000)],
                [" C4 ",16, (2.7022, 0.0815, 0.0000)],
                [" N4 ",16, (4.0356, 0.1372, 0.0000)],
                [" C5 ",16, (2.0394, -1.1794, 0.0000)],
                [" C6 ",16, (0.7007, -1.1745, 0.0000)],
                [" H42",16, (4.5715, -0.7074, 0.0000)],
                [" H41",16, (4.4992, 1.0229, 0.0000)],
                [" H5 ",16, (2.6061, -2.1225, 0.0000)],
                [" H6 ",16, (0.1563, -2.1302, 0.0000)],
                ],
                [ # 24 DG
                    [" OP1", 0, (-0.7319, 1.2920, 0.000)],
                    [" P  ", 0, (0.000, 0.000, 0.000)],
                    [" OP2", 0, (1.4855, 0.000, 0.000)],
                    [" O5'", 0, (-0.4948, -0.8559,  1.2489)],
                    [" C5'", 9, (0.7411, 1.2354, 0.000)],
                    [" C4'",10, (0.5207,  1.4178, 0.000)],
                    [" H5'",10, (0.3748, -0.5360, -0.8839)],
                    ["H5''",10, (0.3744, -0.5368,  0.8842)],
                    [" C3'",11, ( 0.6388,  1.3889, 0.000)],
                    [" H4'",11, ( 0.2823, -0.5105,  0.9326)],
                    [" O4'",11, (0.4804, -0.6610, -1.1947)],
                    [" C1'",13, (0.4913, 1.3316, 0.0000)],
                    [" H1'",14, (0.4561, -0.4898, 0.8726)],
                    [" N9 ",14, (0.4467, -0.7474, -1.1746)],
                    [" C2'",14, (0.4167, 1.4603, 0.0000)],
                    [" H2'",15, (0.4107, -0.5097, -0.8844)],
                    ["H2''",15, (0.4106, -0.5096, 0.8840)],
                    [" O3'",12, ( 0.4966,  1.3432, 0.000)],
                    [" H3'",12, (0.4359, -0.4915, -0.8827)],
                [" C4 ",16, (0.8171, 1.1043, 0.0000)],
                [" N3 ",16, (0.4110, 2.3918, 0.0000)],
                [" C2 ",16, (1.4330, 3.2319, 0.0000)],
                [" N1 ",16, (2.7493, 2.8397, 0.0000)],
                [" C6 ",16, (3.1894, 1.5195, 0.0000)],
                [" C5 ",16, (2.1029, 0.6070, 0.0000)],
                [" N7 ",16, (2.0942, -0.7800, 0.0000)],
                [" C8 ",16, (0.8285, -1.0956, 0.0000)],
                [" N2 ",16, (1.2085, 4.5537, 0.0000)],
                [" O6 ",16, (4.4017, 1.2743, 0.0000)],
                [" H1 ",16, (3.4453, 3.5579, 0.0000)],
                [" H8 ",16, (0.4623, -2.1330, 0.0000)],
                [" H22",16, (0.2708, 4.9015, 0.0000)],
                [" H21",16, (1.9785, 5.1920, 0.0000)],
                ],
                [ # 25 DT
                    [" OP1", 0, (-0.7319, 1.2920, 0.000)],
                    [" P  ", 0, (0.000, 0.000, 0.000)],
                    [" OP2", 0, (1.4855, 0.000, 0.000)],
                    [" O5'", 0, (-0.4948, -0.8559,  1.2489)],
                    [" C5'", 9, (0.7411, 1.2354, 0.000)],
                    [" C4'",10, (0.5207,  1.4178, 0.000)],
                    [" H5'",10, (0.3748, -0.5360, -0.8839)],
                    ["H5''",10, (0.3744, -0.5368,  0.8842)],
                    [" C3'",11, ( 0.6388,  1.3889, 0.000)],
                    [" H4'",11, ( 0.2823, -0.5105,  0.9326)],
                    [" O4'",11, (0.4804, -0.6610, -1.1947)],
                    [" C1'",13, (0.4913, 1.3316, 0.0000)],
                    [" H1'",14, (0.4561, -0.4898, 0.8726)],
                    [" N1 ",14, (0.4467, -0.7474, -1.1746)],
                    [" C2'",14, (0.4167, 1.4603, 0.0000)],
                    [" H2'",15, (0.4107, -0.5097, -0.8844)],
                    ["H2''",15, (0.4106, -0.5096, 0.8840)],
                    [" O3'",12, ( 0.4966,  1.3432, 0.000)],
                    [" H3'",12, (0.4359, -0.4915, -0.8827)],
                    [" C2 ",16, (0.6495, 1.2140, 0.0000)],
                    [" O2 ",16, (0.0636, 2.2854, 0.0000)],
                    [" N3 ",16, (2.0191, 1.1297, 0.0000)],
                    [" C4 ",16, (2.7859, -0.0198, 0.0000)],
                    [" O4 ",16, (4.0113, 0.0622, 0.0000)],
                    [" C5 ",16, (2.0397, -1.2580, 0.0000)],
                    [" C7 ",16, (2.7845, -2.5550, 0.0000)],
                    [" C6 ",16, (0.7021, -1.1863, 0.0000)],
                    [" H3 ",16, (2.5175, 1.9968, 0.0000)],
                    [" H71",16, (2.0680, -3.3898, 0.0000)],
                    [" H72",16, (3.4147, -2.6153, -0.9071)],
                    [" H73",16, (3.4193, -2.6153, 0.8885)],
                    [" H6 ",16, (0.1317, -2.1273, 0.0000)],
                ],
                [ # 26 DX
                    [" OP1", 0, (-0.7319, 1.2920, 0.000)],
                    [" P  ", 0, (0.000, 0.000, 0.000)],
                    [" OP2", 0, (1.4855, 0.000, 0.000)],
                    [" O5'", 0, (-0.4948, -0.8559,  1.2489)],
                    [" C5'", 9, (0.7411, 1.2354, 0.000)],
                    [" C4'",10, (0.5207,  1.4178, 0.000)],
                    [" H5'",10, (0.3748, -0.5360, -0.8839)],
                    ["H5''",10, (0.3744, -0.5368,  0.8842)],
                    [" C3'",11, ( 0.6388,  1.3889, 0.000)],
                    [" H4'",11, ( 0.2823, -0.5105,  0.9326)],
                    [" O4'",11, (0.4804, -0.6610, -1.1947)],
                    [" C1'",13, (0.4913, 1.3316, 0.0000)],
                    [" H1'",14, (0.4561, -0.4898, 0.8726)],
                    [" C2'",14, (0.4167, 1.4603, 0.0000)],
                    [" H2'",15, (0.4107, -0.5097, -0.8844)],
                    ["H2''",15, (0.4106, -0.5096, 0.8840)],
                    [" O3'",12, ( 0.4966,  1.3432, 0.000)],
                    [" H3'",12, (0.4359, -0.4915, -0.8827)],
                ],
                [ # 27 A
                    [" OP1", 0, (-0.7319, 1.2920, 0.000)],
                    [" P  ", 0, (0.000, 0.000, 0.000)],
                    [" OP2", 0, (1.4855, 0.000, 0.000)],
                    [" O5'", 0, (-0.4948, -0.8559,  1.2489)],
                    [" C5'", 9, (0.7289, 1.2185, 0.000)],
                    [" C4'",10, (0.5541, 1.4027, 0.000)],
                    [" H5'",10, (0.3201, -0.4698, -0.7986)],
                    ["H5''",10, (0.3206, -0.4706,  0.7970)],
                    [" C3'",11, ( 0.6673, 1.3669, 0.000)],
                    [" H4'",11, ( 0.3173, -0.5074,  0.7763)],
                    [" O4'",11, ( 0.4914, -0.6338, -1.2098)],
                    [" C1'",13, (0.4828, 1.3277, -0.0000)],
                    [" H1'",14, (0.3265, -0.4460, 0.8101)],
                    [" N9 ",14, (0.4722, -0.7339, -1.1894)],
                    [" C2'",14, (0.4641, 1.4573, 0.0000)],
                    [" H2'",15, (0.3582, -0.4393, -0.7998)],
                    [" O2'",15, (0.4613, -0.6189, 1.1921)],
                    ["HO2'",15, (0.2499, -1.5749, 1.1568)],
                    [" O3'",12, ( 0.5548,  1.3039, 0.000)],
                    [" H3'",12, ( 0.3215, -0.4857, -0.7888)],
                    [" N1 ",16, (2.7963, 2.8824, 0.0000)],
                    [" C2 ",16, (1.4955, 3.2007, 0.0000)],
                    [" N3 ",16, (0.4333, 2.3980, 0.0000)],
                    [" C4 ",16, (0.8127, 1.1078, 0.0000)],
                    [" C5 ",16, (2.1082, 0.6254, 0.0000)],
                    [" C6 ",16, (3.1432, 1.5774, 0.0000)],
                    [" N6 ",16, (4.4400, 1.2609, 0.0000)],
                    [" N7 ",16, (2.1146, -0.7630, 0.0000)],
                    [" C8 ",16, (0.8442, -1.0830, 0.0000)],
                    [" H2 ",16, (1.2972, 4.1608, 0.0000)],
                    [" H61",16, (5.1172, 1.9697, 0.0000)],
                    [" H62",16, (4.7154, 0.3206, 0.0000)],
                    [" H8 ",16, (0.5258, -2.0104, 0.0000)],
                ],
                [ # 28 C
                    [" OP1", 0, (-0.7319, 1.2920, 0.000)],
                    [" P  ", 0, (0.000, 0.000, 0.000)],
                    [" OP2", 0, (1.4855, 0.000, 0.000)],
                    [" O5'", 0, (-0.4948, -0.8559,  1.2489)],
                    [" C5'", 9, (0.7289, 1.2185, 0.000)],
                    [" C4'",10, (0.5541, 1.4027, 0.000)],
                    [" H5'",10, (0.3201, -0.4698, -0.7986)],
                    ["H5''",10, (0.3206, -0.4706,  0.7970)],
                    [" C3'",11, ( 0.6673, 1.3669, 0.000)],
                    [" H4'",11, ( 0.3173, -0.5074,  0.7763)],
                    [" O4'",11, ( 0.4914, -0.6338, -1.2098)],
                    [" C1'",13, (0.4828, 1.3277, -0.0000)],
                    [" H1'",14, (0.3265, -0.4460, 0.8101)],
                    [" N1 ",14, (0.4722, -0.7339, -1.1894)],
                    [" C2'",14, (0.4641, 1.4573, 0.0000)],
                    [" H2'",15, (0.3582, -0.4393, -0.7998)],
                    [" O2'",15, (0.4613, -0.6189, 1.1921)],
                    ["HO2'",15, (0.2499, -1.5749, 1.1568)],
                    [" O3'",12, ( 0.5548,  1.3039, 0.000)],
                    [" H3'",12, ( 0.3215, -0.4857, -0.7888)],
                    [" C2 ",16, (0.6650, 1.2325, 0.0000)],
                    [" O2 ",16, (-0.0001, 2.2799, 0.0000)],
                    [" N3 ",16, (2.0175, 1.2603, 0.0000)],
                    [" C4 ",16, (2.7090, 0.1210, 0.0000)],
                    [" N4 ",16, (4.0423, 0.1969, 0.0000)],
                    [" C5 ",16, (2.0635, -1.1476, 0.0000)],
                    [" C6 ",16, (0.7250, -1.1627, 0.0000)],
                    [" H42",16, (4.5791, -0.6226, 0.0000)],
                    [" H41",16, (4.4833, 1.0723, 0.0000)],
                    [" H5 ",16, (2.5806, -1.9803, 0.0000)],
                    [" H6 ",16, (0.2622, -2.0258, 0.0000)],
                ],
                [ # 29 G
                    [" OP1", 0, (-0.7319, 1.2920, 0.000)],
                    [" P  ", 0, (0.000, 0.000, 0.000)],
                    [" OP2", 0, (1.4855, 0.000, 0.000)],
                    [" O5'", 0, (-0.4948, -0.8559,  1.2489)],
                    [" C5'", 9, (0.7289, 1.2185, 0.000)],
                    [" C4'",10, (0.5541, 1.4027, 0.000)],
                    [" H5'",10, (0.3201, -0.4698, -0.7986)],
                    ["H5''",10, (0.3206, -0.4706,  0.7970)],
                    [" C3'",11, ( 0.6673, 1.3669, 0.000)],
                    [" H4'",11, ( 0.3173, -0.5074,  0.7763)],
                    [" O4'",11, ( 0.4914, -0.6338, -1.2098)],
                    [" C1'",13, (0.4828, 1.3277, -0.0000)],
                    [" H1'",14, (0.3265, -0.4460, 0.8101)],
                    [" N9 ",14, (0.4722, -0.7339, -1.1894)],
                    [" C2'",14, (0.4641, 1.4573, 0.0000)],
                    [" H2'",15, (0.3582, -0.4393, -0.7998)],
                    [" O2'",15, (0.4613, -0.6189, 1.1921)],
                    ["HO2'",15, (0.2499, -1.5749, 1.1568)],
                    [" O3'",12, ( 0.5548,  1.3039, 0.000)],
                    [" H3'",12, ( 0.3215, -0.4857, -0.7888)],
                    [" N1 ",16, (2.7458, 2.8461, 0.0000)],
                    [" C2 ",16, (1.4286, 3.2360, 0.0000)],
                    [" N2 ",16, (1.1989, 4.5575, 0.0000)],
                    [" N3 ",16, (0.4087, 2.3932, 0.0000)],
                    [" C4 ",16, (0.8167, 1.1068, 0.0000)],
                    [" C5 ",16, (2.1036, 0.6115, 0.0000)],
                    [" C6 ",16, (3.1883, 1.5266, 0.0000)],
                    [" O6 ",16, (4.4006, 1.2842, 0.0000)],
                    [" N7 ",16, (2.0980, -0.7759, 0.0000)],
                    [" C8 ",16, (0.8317, -1.0936, 0.0000)],
                    [" H1 ",16, (3.4279, 3.5496, 0.0000)],
                    [" H22",16, (0.2781, 4.8947, 0.0000)],
                    [" H21",16, (1.9487, 5.1879, 0.0000)],
                    [" H8 ",16, (0.5085, -2.0185, 0.0000)],
                ],
                [ # 30 U
                    [" OP1", 0, (-0.7319, 1.2920, 0.000)],
                    [" P  ", 0, (0.000, 0.000, 0.000)],
                    [" OP2", 0, (1.4855, 0.000, 0.000)],
                    [" O5'", 0, (-0.4948, -0.8559,  1.2489)],
                    [" C5'", 9, (0.7289, 1.2185, 0.000)],
                    [" C4'",10, (0.5541, 1.4027, 0.000)],
                    [" H5'",10, (0.3201, -0.4698, -0.7986)],
                    ["H5''",10, (0.3206, -0.4706,  0.7970)],
                    [" C3'",11, ( 0.6673, 1.3669, 0.000)],
                    [" H4'",11, ( 0.3173, -0.5074,  0.7763)],
                    [" O4'",11, ( 0.4914, -0.6338, -1.2098)],
                    [" C1'",13, (0.4828, 1.3277, -0.0000)],
                    [" H1'",14, (0.3265, -0.4460, 0.8101)],
                    [" N1 ",14, (0.4722, -0.7339, -1.1894)],
                    [" C2'",14, (0.4641, 1.4573, 0.0000)],
                    [" H2'",15, (0.3582, -0.4393, -0.7998)],
                    [" O2'",15, (0.4613, -0.6189, 1.1921)],
                    ["HO2'",15, (0.2499, -1.5749, 1.1568)],
                    [" O3'",12, ( 0.5548,  1.3039, 0.000)],
                    [" H3'",12, ( 0.3215, -0.4857, -0.7888)],
                    [" C2 ",16, (0.6307, 1.2305, 0.0000)],
                    [" O2 ",16, (0.0260, 2.2886, 0.0000)],
                    [" N3 ",16, (2.0031, 1.1816, 0.0000)],
                    [" C4 ",16, (2.7953, 0.0532, 0.0000)],
                    [" O4 ",16, (4.0212, 0.1751, 0.0000)],
                    [" C5 ",16, (2.0746, -1.1833, 0.0000)],
                    [" C6 ",16, (0.7378, -1.1648, 0.0000)],
                    [" H3 ",16, (2.4701, 2.0428, 0.0000)],
                    [" H5 ",16, (2.5579, -2.0361, 0.0000)],
                    [" H6 ",16, (0.2681, -2.0239, 0.0000)],
                ],
                [ # 31 RX
                    [" OP1", 0, (-0.7319, 1.2920, 0.000)],
                    [" P  ", 0, (0.000, 0.000, 0.000)],
                    [" OP2", 0, (1.4855, 0.000, 0.000)],
                    [" O5'", 0, (-0.4948, -0.8559,  1.2489)],
                    [" C5'", 9, (0.7289, 1.2185, 0.000)],
                    [" C4'",10, (0.5541, 1.4027, 0.000)],
                    [" H5'",10, (0.3201, -0.4698, -0.7986)],
                    ["H5''",10, (0.3206, -0.4706,  0.7970)],
                    [" C3'",11, ( 0.6673, 1.3669, 0.000)],
                    [" H4'",11, ( 0.3173, -0.5074,  0.7763)],
                    [" O4'",11, ( 0.4914, -0.6338, -1.2098)],
                    [" C1'",13, (0.4828, 1.3277, -0.0000)],
                    [" H1'",14, (0.3265, -0.4460, 0.8101)],
                    [" C2'",14, (0.4641, 1.4573, 0.0000)],
                    [" H2'",15, (0.3582, -0.4393, -0.7998)],
                    [" O2'",15, (0.4613, -0.6189, 1.1921)],
                    ["HO2'",15, (0.2499, -1.5749, 1.1568)],
                    [" O3'",12, ( 0.5548,  1.3039, 0.000)],
                    [" H3'",12, ( 0.3215, -0.4857, -0.7888)],
                ],
            ]

        if params.get("use_cif_ordering_for_trp", False):
            self.aa2long[17] = (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE "," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None)
                
        self.aabonds=[
            #       0               1               2                3               4              5               6               7               8               9              10              11              12              13              14              15              16              17              18              19              20              21              22               23             24
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB ","1HB "),(" CB ","2HB "),(" CB ","3HB ")) , # ala
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD "),(" CG ","1HG "),(" CG ","2HG "),(" CD "," NE "),(" CD ","1HD "),(" CD ","2HD "),(" NE "," CZ "),(" NE "," HE "),(" CZ "," NH1"),(" CZ "," NH2"),(" NH1","1HH1"),(" NH1","2HH1"),(" NH2","1HH2"),(" NH2","2HH2")) , # arg
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," OD1"),(" CG "," ND2"),(" ND2","1HD2"),(" ND2","2HD2")) , # asn
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," OD1"),(" CG "," OD2")) , # asp
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," SG "),(" CB ","1HB "),(" CB ","2HB "),(" SG "," HG ")) , # cys
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD "),(" CG ","1HG "),(" CG ","2HG "),(" CD "," OE1"),(" CD "," NE2"),(" NE2","1HE2"),(" NE2","2HE2")) , # gln
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD "),(" CG ","1HG "),(" CG ","2HG "),(" CD "," OE1"),(" CD "," OE2")) , # glu
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA ","1HA "),(" CA ","2HA "),(" C  "," O  ")) , # gly
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," ND1"),(" CG "," CD2"),(" ND1"," CE1"),(" CD2"," NE2"),(" CD2","2HD "),(" CE1"," NE2"),(" CE1","1HE "),(" NE2","2HE ")) , # his
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG1"),(" CB "," CG2"),(" CB "," HB "),(" CG1"," CD1"),(" CG1","1HG1"),(" CG1","2HG1"),(" CG2","1HG2"),(" CG2","2HG2"),(" CG2","3HG2"),(" CD1","1HD1"),(" CD1","2HD1"),(" CD1","3HD1")) , # ile
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD1"),(" CG "," CD2"),(" CG "," HG "),(" CD1","1HD1"),(" CD1","2HD1"),(" CD1","3HD1"),(" CD2","1HD2"),(" CD2","2HD2"),(" CD2","3HD2")) , # leu
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD "),(" CG ","1HG "),(" CG ","2HG "),(" CD "," CE "),(" CD ","1HD "),(" CD ","2HD "),(" CE "," NZ "),(" CE ","1HE "),(" CE ","2HE "),(" NZ ","1HZ "),(" NZ ","2HZ "),(" NZ ","3HZ ")) , # lys
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," SD "),(" CG ","1HG "),(" CG ","2HG "),(" SD "," CE "),(" CE ","1HE "),(" CE ","2HE "),(" CE ","3HE ")) , # met
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD1"),(" CG "," CD2"),(" CD1"," CE1"),(" CD1","1HD "),(" CD2"," CE2"),(" CD2","2HD "),(" CE1"," CZ "),(" CE1","1HE "),(" CE2"," CZ "),(" CE2","2HE "),(" CZ "," HZ ")) , # phe
            ((" N  "," CA "),(" N  "," CD "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD "),(" CG ","1HG "),(" CG ","2HG "),(" CD ","1HD "),(" CD ","2HD ")) , # pro
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," OG "),(" CB ","1HB "),(" CB ","2HB "),(" OG "," HG ")) , # ser
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," OG1"),(" CB "," CG2"),(" CB "," HB "),(" OG1"," HG1"),(" CG2","1HG2"),(" CG2","2HG2"),(" CG2","3HG2")) , # thr
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD1"),(" CG "," CD2"),(" CD1"," NE1"),(" CD1","1HD "),(" CD2"," CE2"),(" CD2"," CE3"),(" NE1"," CE2"),(" NE1","1HE "),(" CE2"," CZ2"),(" CE3"," CZ3"),(" CE3"," HE3"),(" CZ2"," CH2"),(" CZ2"," HZ2"),(" CZ3"," CH2"),(" CZ3"," HZ3"),(" CH2"," HH2")) , # trp
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD1"),(" CG "," CD2"),(" CD1"," CE1"),(" CD1","1HD "),(" CD2"," CE2"),(" CD2","2HD "),(" CE1"," CZ "),(" CE1","1HE "),(" CE2"," CZ "),(" CE2","2HE "),(" CZ "," OH "),(" OH "," HH ")) , # tyr
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG1"),(" CB "," CG2"),(" CB "," HB "),(" CG1","1HG1"),(" CG1","2HG1"),(" CG1","3HG1"),(" CG2","1HG2"),(" CG2","2HG2"),(" CG2","3HG2")), # val
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB ","1HB "),(" CB ","2HB "),(" CB ","3HB ")) , # unk
            ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB ","1HB "),(" CB ","2HB "),(" CB ","3HB ")) , # mask
            ((" P  "," OP2"),(" P  "," OP1"),(" P  "," O5'"),(" O5'"," C5'"),(" C5'"," C4'"),(" C5'","H5''"),(" C5'"," H5'"),(" C4'"," O4'"),(" C4'"," C3'"),(" C4'"," H4'"),(" O4'"," C1'"),(" C3'"," O3'"),(" C3'"," C2'"),(" C3'"," H3'"),(" C2'"," C1'"),(" C2'","H2''"),(" C2'"," H2'"),(" C1'"," N9 "),(" C1'"," H1'"),(" N1 "," C2 "),(" N1 "," C6 "),(" C2 "," N3 "),(" C2 "," H2 "),(" N3 "," C4 "),(" C4 "," C5 "),(" C4 "," N9 "),(" C5 "," C6 "),(" C5 "," N7 "),(" C6 "," N6 "),(" N6 "," H61"),(" N6 "," H62"),(" N7 "," C8 "),(" C8 "," N9 "),(" C8 "," H8 ")) , # DA
            ((" P  "," OP2"),(" P  "," OP1"),(" P  "," O5'"),(" O5'"," C5'"),(" C5'"," C4'"),(" C5'","H5''"),(" C5'"," H5'"),(" C4'"," O4'"),(" C4'"," C3'"),(" C4'"," H4'"),(" O4'"," C1'"),(" C3'"," O3'"),(" C3'"," C2'"),(" C3'"," H3'"),(" C2'"," C1'"),(" C2'","H2''"),(" C2'"," H2'"),(" C1'"," N1 "),(" C1'"," H1'"),(" N1 "," C2 "),(" N1 "," C6 "),(" C2 "," O2 "),(" C2 "," N3 "),(" N3 "," C4 "),(" C4 "," N4 "),(" C4 "," C5 "),(" N4 "," H42"),(" N4 "," H41"),(" C5 "," C6 "),(" C5 "," H5 "),(" C6 "," H6 ")), # DC
            ((" P  "," OP2"),(" P  "," OP1"),(" P  "," O5'"),(" O5'"," C5'"),(" C5'"," C4'"),(" C5'","H5''"),(" C5'"," H5'"),(" C4'"," O4'"),(" C4'"," C3'"),(" C4'"," H4'"),(" O4'"," C1'"),(" C3'"," O3'"),(" C3'"," C2'"),(" C3'"," H3'"),(" C2'"," C1'"),(" C2'","H2''"),(" C2'"," H2'"),(" C1'"," N9 "),(" C1'"," H1'"),(" N1 "," C2 "),(" N1 "," C6 "),(" N1 "," H1 "),(" C2 "," N2 "),(" C2 "," N3 "),(" N2 "," H22"),(" N2 "," H21"),(" N3 "," C4 "),(" C4 "," C5 "),(" C4 "," N9 "),(" C5 "," C6 "),(" C5 "," N7 "),(" C6 "," O6 "),(" N7 "," C8 "),(" C8 "," N9 "),(" C8 "," H8 ")), # DG
            ((" P  "," OP2"),(" P  "," OP1"),(" P  "," O5'"),(" O5'"," C5'"),(" C5'"," C4'"),(" C5'","H5''"),(" C5'"," H5'"),(" C4'"," O4'"),(" C4'"," C3'"),(" C4'"," H4'"),(" O4'"," C1'"),(" C3'"," O3'"),(" C3'"," C2'"),(" C3'"," H3'"),(" C2'"," C1'"),(" C2'","H2''"),(" C2'"," H2'"),(" C1'"," N1 "),(" C1'"," H1'"),(" N1 "," C2 "),(" N1 "," C6 "),(" C2 "," O2 "),(" C2 "," N3 "),(" N3 "," C4 "),(" N3 "," H3 "),(" C4 "," O4 "),(" C4 "," C5 "),(" C5 "," C7 "),(" C5 "," C6 "),(" C7 "," H71"),(" C7 "," H72"),(" C7 "," H73"),(" C6 "," H6 ")), # DT
            ((" P  "," OP2"),(" P  "," OP1"),(" P  "," O5'"),(" O5'"," C5'"),(" C5'"," C4'"),(" C5'","H5''"),(" C5'"," H5'"),(" C4'"," O4'"),(" C4'"," C3'"),(" C4'"," H4'"),(" O4'"," C1'"),(" C3'"," O3'"),(" C3'"," C2'"),(" C3'"," H3'"),(" C2'"," C1'"),(" C2'","H2''"),(" C2'"," H2'"),(" C1'"," H1'")) , # DX
            ((" P  "," OP2"),(" P  "," OP1"),(" P  "," O5'"),(" O5'"," C5'"),(" C5'"," C4'"),(" C5'"," H5'"),(" C5'","H5''"),(" C4'"," O4'"),(" C4'"," C3'"),(" C4'"," H4'"),(" O4'"," C1'"),(" C3'"," O3'"),(" C3'"," C2'"),(" C3'"," H3'"),(" C2'"," C1'"),(" C2'"," O2'"),(" C2'"," H2'"),(" O2'","HO2'"),(" C1'"," N9 "),(" C1'"," H1'"),(" N1 "," C2 "),(" N1 "," C6 "),(" C2 "," N3 "),(" C2 "," H2 "),(" N3 "," C4 "),(" C4 "," C5 "),(" C4 "," N9 "),(" C5 "," C6 "),(" C5 "," N7 "),(" C6 "," N6 "),(" N6 "," H61"),(" N6 "," H62"),(" N7 "," C8 "),(" C8 "," N9 "),(" C8 "," H8 ")), # A
            ((" P  "," OP2"),(" P  "," OP1"),(" P  "," O5'"),(" O5'"," C5'"),(" C5'"," C4'"),(" C5'"," H5'"),(" C5'","H5''"),(" C4'"," O4'"),(" C4'"," C3'"),(" C4'"," H4'"),(" O4'"," C1'"),(" C3'"," O3'"),(" C3'"," C2'"),(" C3'"," H3'"),(" C2'"," C1'"),(" C2'"," O2'"),(" C2'"," H2'"),(" O2'","HO2'"),(" C1'"," N1 "),(" C1'"," H1'"),(" N1 "," C2 "),(" N1 "," C6 "),(" C2 "," O2 "),(" C2 "," N3 "),(" N3 "," C4 "),(" C4 "," N4 "),(" C4 "," C5 "),(" N4 "," H42"),(" N4 "," H41"),(" C5 "," C6 "),(" C5 "," H5 "),(" C6 "," H6 ")), # C
            ((" P  "," OP2"),(" P  "," OP1"),(" P  "," O5'"),(" O5'"," C5'"),(" C5'"," C4'"),(" C5'"," H5'"),(" C5'","H5''"),(" C4'"," O4'"),(" C4'"," C3'"),(" C4'"," H4'"),(" O4'"," C1'"),(" C3'"," O3'"),(" C3'"," C2'"),(" C3'"," H3'"),(" C2'"," C1'"),(" C2'"," O2'"),(" C2'"," H2'"),(" O2'","HO2'"),(" C1'"," N9 "),(" C1'"," H1'"),(" N1 "," C2 "),(" N1 "," C6 "),(" N1 "," H1 "),(" C2 "," N2 "),(" C2 "," N3 "),(" N2 "," H22"),(" N2 "," H21"),(" N3 "," C4 "),(" C4 "," C5 "),(" C4 "," N9 "),(" C5 "," C6 "),(" C5 "," N7 "),(" C6 "," O6 "),(" N7 "," C8 "),(" C8 "," N9 "),(" C8 "," H8 ")), # G
            ((" P  "," OP2"),(" P  "," OP1"),(" P  "," O5'"),(" O5'"," C5'"),(" C5'"," C4'"),(" C5'"," H5'"),(" C5'","H5''"),(" C4'"," O4'"),(" C4'"," C3'"),(" C4'"," H4'"),(" O4'"," C1'"),(" C3'"," O3'"),(" C3'"," C2'"),(" C3'"," H3'"),(" C2'"," C1'"),(" C2'"," O2'"),(" C2'"," H2'"),(" O2'","HO2'"),(" C1'"," N1 "),(" C1'"," H1'"),(" N1 "," C2 "),(" N1 "," C6 "),(" C2 "," O2 "),(" C2 "," N3 "),(" N3 "," C4 "),(" N3 "," H3 "),(" C4 "," O4 "),(" C4 "," C5 "),(" C5 "," C6 "),(" C5 "," H5 "),(" C6 "," H6 ")), # U
            ((" P  "," OP2"),(" P  "," OP1"),(" P  "," O5'"),(" O5'"," C5'"),(" C5'"," C4'"),(" C5'"," H5'"),(" C5'","H5''"),(" C4'"," O4'"),(" C4'"," C3'"),(" C4'"," H4'"),(" O4'"," C1'"),(" C3'"," O3'"),(" C3'"," C2'"),(" C3'"," H3'"),(" C2'"," C1'"),(" C2'"," O2'"),(" C2'"," H2'"),(" O2'","HO2'"),(" C1'"," H1'")), # RX
        ]

        self.NO_BOND = 0
        self.SINGLE_BOND = 1
        self.DOUBLE_BOND = 2
        self.TRIPLE_BOND = 3
        self.AROMATIC_BOND = 4
        self.RESIDUE_BB_BOND = 5
        self.RESIDUE_ATOM_BOND = 6

        # BTYPES ONLY DEFINED FOR CANONICAL 20 amino acids + UNK, MAS treated as glycines
        SINGLE_BOND = self.SINGLE_BOND
        DOUBLE_BOND = self.DOUBLE_BOND
        AROMATIC_BOND = self.AROMATIC_BOND
        self.aabtypes = [
            #    0              1           2             3            4           5             6            7            8           9             10                11             12           13               14            15              16           17              18              19           20             21           22            23            24      
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND), # ala
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND,   SINGLE_BOND, DOUBLE_BOND,     SINGLE_BOND,   SINGLE_BOND, SINGLE_BOND,   SINGLE_BOND, SINGLE_BOND), # arg
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND), # asn
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND,   SINGLE_BOND), # asp
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND), # cys
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   DOUBLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND), # gln 
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   DOUBLE_BOND,   SINGLE_BOND), # glu
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND), # gly
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, AROMATIC_BOND, AROMATIC_BOND,   AROMATIC_BOND, AROMATIC_BOND, SINGLE_BOND,     AROMATIC_BOND, SINGLE_BOND,   SINGLE_BOND), # his
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND,   SINGLE_BOND, SINGLE_BOND), # ile
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND,   SINGLE_BOND, SINGLE_BOND), # leu
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND,   SINGLE_BOND, SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND, SINGLE_BOND), # lys
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND), # met
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, AROMATIC_BOND, AROMATIC_BOND,   AROMATIC_BOND, SINGLE_BOND,   AROMATIC_BOND,   SINGLE_BOND,   AROMATIC_BOND, SINGLE_BOND, AROMATIC_BOND,   SINGLE_BOND,   SINGLE_BOND), # phe
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND,   SINGLE_BOND), # pro
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND), # ser
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND), # thr
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, AROMATIC_BOND, AROMATIC_BOND,   AROMATIC_BOND, SINGLE_BOND,   AROMATIC_BOND,   AROMATIC_BOND, AROMATIC_BOND, SINGLE_BOND, AROMATIC_BOND,   AROMATIC_BOND, SINGLE_BOND, AROMATIC_BOND, SINGLE_BOND, AROMATIC_BOND, SINGLE_BOND, SINGLE_BOND), # trp
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, AROMATIC_BOND, AROMATIC_BOND,   AROMATIC_BOND, SINGLE_BOND,   AROMATIC_BOND,   SINGLE_BOND,   AROMATIC_BOND, SINGLE_BOND, AROMATIC_BOND,   SINGLE_BOND,   SINGLE_BOND, SINGLE_BOND), # tyr
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND,   SINGLE_BOND,   SINGLE_BOND,     SINGLE_BOND), # val
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND), # unk (treated as gly)
            (SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, SINGLE_BOND, DOUBLE_BOND) # mask (treated as gly)
        ]
        self.aachirals = [
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #ala
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #arg
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #asn
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #asp
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #cys
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #gln
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #glu
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #gly
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #his
            (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #ileu
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #leu
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #lys
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #met
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #phe
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #pro
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #ser
            (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #thr
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #trp
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #tyr
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #val
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #unk
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #mas
        ]
        self.aachirals = torch.tensor(self.aachirals)

        # tip atom
        self.aa2tip = [
                " CB ", # ala
                " CZ ", # arg
                " ND2", # asn
                " CG ", # asp
                " SG ", # cys
                " NE2", # gln
                " CD ", # glu
                " CA ", # gly
                " NE2", # his
                " CD1", # ile
                " CG ", # leu
                " NZ ", # lys
                " SD ", # met
                " CZ ", # phe
                " CG ", # pro
                " OG ", # ser
                " OG1", # thr
                " CH2", # trp
                " OH ", # tyr
                " CB ", # val
                " CB ", # unknown (gap etc)
                " CB ", # masked
                " N1 ", # DA
                " N3 ", # DC
                " N1 ", # DG
                " N3 ", # DT
                " C1'", # DX
                " N1 ", # A
                " N3 ", # C
                " N1 ", # G
                " N3 ", # U
                " C1'", # RX
                " NE2", # HIS_D
                ]

        # ideal N, CA, C initial coordinates (protein)
        self.init_N = torch.tensor([-0.5272, 1.3593, 0.000]).float()
        self.init_CA = torch.zeros_like(self.init_N)
        self.init_C = torch.tensor([1.5233, 0.000, 0.000]).float()
        self.INIT_CRDS = torch.full((self.NTOTAL, 3), np.nan)
        self.INIT_CRDS[:3] = torch.stack((self.init_N, self.init_CA, self.init_C), dim=0) # (3,3)

        if (not params.use_phospate_frames_for_NA):
            # despite the name, this uses ideal O4p,C1p,C2p initial coordinates
            self.init_O1 = torch.tensor([-0.3894, 1.3649, 0.0000]).float()
            self.init_P = torch.zeros_like(self.init_O1)
            self.init_O2 = torch.tensor([1.5186, 0.000, 0.000]).float()
            self.costgtNA = -0.2744
        else:
            # this uses ideal P,OP1,OP2 initial coordinates
            self.init_O1 = torch.tensor([-0.7319, 1.2920, 0.000]).float()
            self.init_P = torch.zeros_like(self.init_O1)
            self.init_O2 = torch.tensor([1.4855, 0.000, 0.000]).float()
            self.costgtNA = -0.4929

        self.INIT_NA_CRDS = torch.full((self.NTOTAL, 3), np.nan)
        self.INIT_NA_CRDS[:3] = torch.stack((self.init_O1, self.init_P, self.init_O2), dim=0) # (3,3)

        # non-backbone torsions
        # (bb torsions are hard-coded)
        self.torsions=[
            [ None, None, None, None ],  # ala
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," NE "], [" CG "," CD "," NE "," CZ "] ],  # arg
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," OD1"], None, None ],  # asn
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," OD1"], None, None ],  # asp
            [ [" N  "," CA "," CB "," SG "], [" CA "," CB "," SG "," HG "], None, None ],  # cys
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," OE1"], None ],  # gln
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," OE1"], None ],  # glu
            [ None, None, None, None ],  # gly
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," ND1"], [" CD2"," CE1","1HE "," NE2"], None ],  # his (protonation handled as a pseudo-torsion)
            [ [" N  "," CA "," CB "," CG1"], [" CA "," CB "," CG1"," CD1"], None, None ],  # ile
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], None, None ],  # leu
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," CE "], [" CG "," CD "," CE "," NZ "] ],  # lys
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," SD "], [" CB "," CG "," SD "," CE "], None ],  # met
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], None, None ],  # phe
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD ","1HD "], None ],  # pro
            [ [" N  "," CA "," CB "," OG "], [" CA "," CB "," OG "," HG "], None, None ],  # ser
            [ [" N  "," CA "," CB "," OG1"], [" CA "," CB "," OG1"," HG1"], None, None ],  # thr
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], None, None ],  # trp
            [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], [" CE1"," CZ "," OH "," HH "], None ],  # tyr
            [ [" N  "," CA "," CB "," CG1"], None, None, None ],  # val
            [ None, None, None, None ],  # unk
            [ None, None, None, None ],  # mask
            [ [" O4'"," C1'"," N9 "," C4 "], None, None, None  ],#DA
            [ [" O4'"," C1'"," N1 "," C2 "], None, None, None  ],#DC
            [ [" O4'"," C1'"," N9 "," C4 "], None, None, None  ],#DG
            [ [" O4'"," C1'"," N1 "," C2 "], None, None, None  ],#DT
            [ None, None, None, None ],  # DX
            [ [" O4'"," C1'"," N9 "," C4 "], None, None, None  ],#A
            [ [" O4'"," C1'"," N1 "," C2 "], None, None, None  ],#C
            [ [" O4'"," C1'"," N9 "," C4 "], None, None, None  ],#G
            [ [" O4'"," C1'"," N1 "," C2 "], None, None, None  ],#U
            [ None, None, None, None ],  # RX
        ]

        self.NFRAMES = max([len(f) for f in self.frames])

        atomized_protein_frames = torch.load(script_dir+"atomized_protein_frames.pt")

    def load_derived_data(self, params):
        # resolve tip atom indices
        self.tip_indices = torch.full((self.NAATOKENS,), 0)
        for i in range(self.NAATOKENS):
            if i > self.NNAPROTAAS-1:
                # all atoms are at index 1 in the atom array 
                self.tip_indices[i] = 1
            else:
                tip_atm = self.aa2tip[i]
                atm_long = self.aa2long[i]
                self.tip_indices[i] = atm_long.index(tip_atm)

        # resolve torsion indices
        #  a negative index indicates the previous residue
        # order:
        #    omega/phi/psi: 0-2
        #    chi_1-4(prot): 3-6
        #    cb/cg bend: 7-9
        #    eps(p)/zeta(p): 10-11
        #    alpha/beta/gamma/delta: 12-15
        #    nu2/nu1/nu0: 16-18
        #    chi_1(na): 19
        self.torsion_indices = torch.full((self.NAATOKENS,self.NTOTALDOFS,4),0)
        self.torsion_can_flip = torch.full((self.NAATOKENS,self.NTOTALDOFS),False,dtype=torch.bool)
        for i in range(self.NPROTAAS):
            i_l, i_a = self.aa2long[i], self.aa2longalt[i]

            # protein omega/phi/psi
            self.torsion_indices[i,0,:] = torch.tensor([-1,-2,0,1]) # omega
            self.torsion_indices[i,1,:] = torch.tensor([-2,0,1,2]) # phi
            self.torsion_indices[i,2,:] = torch.tensor([0,1,2,3]) # psi (+pi)

            # protein chis
            for j in range(4):
                if self.torsions[i][j] is None:
                    continue
                for k in range(4):
                    a = self.torsions[i][j][k]
                    self.torsion_indices[i,3+j,k] = i_l.index(a)
                    if (i_l.index(a) != i_a.index(a)):
                        self.torsion_can_flip[i,3+j] = True ##bb tors never flip

            # CB/CG angles (only masking uses these indices)
            self.torsion_indices[i,7,:] = torch.tensor([0,2,1,4]) # CB ang1
            self.torsion_indices[i,8,:] = torch.tensor([0,2,1,4]) # CB ang2
            self.torsion_indices[i,9,:] = torch.tensor([0,2,4,5]) # CG ang (arg 1 ignored)

        # HIS is a special case for flip
        self.torsion_can_flip[8,4]=False

        # DNA/RNA
        if (not params.use_phospate_frames_for_NA):
            # ribose frame
            for i in range(self.NPROTAAS,self.NNAPROTAAS):
                self.torsion_indices[i,10,:] = torch.tensor([-2,-9,-10,4])  # epsilon_prev
                self.torsion_indices[i,11,:] = torch.tensor([-9,-10,4,6])   # zeta_prev
                self.torsion_indices[i,12,:] = torch.tensor([7,6,4,3])     # alpha c5'-o5'-p-op1
                self.torsion_indices[i,13,:] = torch.tensor([8,7,6,4])     # beta c4'-c5'-o5'-p
                self.torsion_indices[i,14,:] = torch.tensor([9,8,7,6])     # gamma c3'-c4'-c5'-o5'
                self.torsion_indices[i,15,:] = torch.tensor([2,9,8,7])     # delta c2'-c3'-c4'-c5'

                self.torsion_indices[i,16,:] = torch.tensor([1,2,9,8])     # nu2
                self.torsion_indices[i,17,:] = torch.tensor([0,1,2,9])     # nu1
                self.torsion_indices[i,18,:] = torch.tensor([2,1,0,8])     # nu0

                # NA chi
                if self.torsions[i][0] is not None:
                    i_l = self.aa2long[i]
                    for k in range(4):
                        a = self.torsions[i][0][k]
                        self.torsion_indices[i,19,k] = i_l.index(a) # chi
        else:
            # phosphate frame
            for i in range(self.NPROTAAS,self.NNAPROTAAS):
                # NA BB tors
                self.torsion_indices[i,10,:] = torch.tensor([-5,-7,-8,1])  # epsilon_prev
                self.torsion_indices[i,11,:] = torch.tensor([-7,-8,1,3])   # zeta_prev
                self.torsion_indices[i,12,:] = torch.tensor([0,1,3,4])     # alpha (+2pi/3)
                self.torsion_indices[i,13,:] = torch.tensor([1,3,4,5])     # beta
                self.torsion_indices[i,14,:] = torch.tensor([3,4,5,7])     # gamma
                self.torsion_indices[i,15,:] = torch.tensor([4,5,7,8])     # delta

                if (i<self.NPROTAAS+5):
                    # is DNA
                    self.torsion_indices[i,16,:] = torch.tensor([4,5,6,10])     # nu2
                    self.torsion_indices[i,17,:] = torch.tensor([5,6,10,9])     # nu1
                    self.torsion_indices[i,18,:] = torch.tensor([6,10,9,7])     # nu0
                else:   
                    # is RNA (fd: my fault since I flipped C1'/C2' order for DNA and RNA)
                    self.torsion_indices[i,16,:] = torch.tensor([4,5,6,9])     # nu2
                    self.torsion_indices[i,17,:] = torch.tensor([5,6,9,10])     # nu1
                    self.torsion_indices[i,18,:] = torch.tensor([6,9,10,7])     # nu0

                # NA chi
                if self.torsions[i][0] is not None:
                    i_l = self.aa2long[i]
                    for k in range(4):
                        a = self.torsions[i][0][k]
                        self.torsion_indices[i,19,k] = i_l.index(a) # chi

        # build the mapping from atoms in the full rep (Nx27) to the "alternate" rep
        self.allatom_mask = torch.zeros((self.NAATOKENS,self.NTOTAL), dtype=torch.bool)
        self.long2alt = torch.zeros((self.NAATOKENS,self.NTOTAL), dtype=torch.long)
        for i in range(self.NNAPROTAAS):
            i_l, i_lalt = self.aa2long[i],  self.aa2longalt[i]
            for j,a in enumerate(i_l):
                if (a is None):
                    self.long2alt[i,j] = j
                else:
                    self.long2alt[i,j] = i_lalt.index(a)
                    self.allatom_mask[i,j] = True
        for i in range(self.NNAPROTAAS, self.NAATOKENS):
            for j in range(self.NTOTAL):
                self.long2alt[i, j] = j
        self.allatom_mask[self.NNAPROTAAS:,1] = True

        # bond graph traversal
        self.num_bonds = torch.zeros((self.NAATOKENS,self.NTOTAL,self.NTOTAL), dtype=torch.long)
        for i in range(self.NNAPROTAAS):
            num_bonds_i = np.zeros((self.NTOTAL,self.NTOTAL))
            for (bnamei,bnamej) in self.aabonds[i]:
                bi,bj = self.aa2long[i].index(bnamei),self.aa2long[i].index(bnamej)
                num_bonds_i[bi,bj] = 1
            num_bonds_i = scipy.sparse.csgraph.shortest_path (num_bonds_i,directed=False)
            num_bonds_i[num_bonds_i>=4] = 4
            self.num_bonds[i,...] = torch.tensor(num_bonds_i)


        # atom type indices
        self.idx2aatype = []
        for x in self.aa2type:
            for y in x:
                if y and y not in self.idx2aatype:
                    self.idx2aatype.append(y)
        self.aatype2idx = {x:i for i,x in enumerate(self.idx2aatype)}

        # element indices
        self.idx2elt = []
        for x in self.aa2elt:
            for y in x:
                if y and y not in self.idx2elt:
                    self.idx2elt.append(y)
        self.elt2idx = {x:i for i,x in enumerate(self.idx2elt)}

        # LJ/LK scoring parameters
        self.atom_type_index = torch.zeros((self.NAATOKENS,self.NTOTAL), dtype=torch.long)
        self.element_index = torch.zeros((self.NAATOKENS,self.NTOTAL), dtype=torch.long)

        self.ljlk_parameters = torch.zeros((self.NAATOKENS,self.NTOTAL,5), dtype=torch.float)
        self.lj_correction_parameters = torch.zeros((self.NAATOKENS,self.NTOTAL,4), dtype=bool) # donor/acceptor/hpol/disulf
        for i in range(self.NNAPROTAAS):
            for j,a in enumerate(self.aa2type[i]):
                if (a is not None):
                    self.atom_type_index[i,j] = self.aatype2idx[a]
                    self.ljlk_parameters[i,j,:] = torch.tensor( type2ljlk[a] )
                    self.lj_correction_parameters[i,j,0] = (type2hb[a]==HbAtom.DO)+(type2hb[a]==HbAtom.DA)
                    self.lj_correction_parameters[i,j,1] = (type2hb[a]==HbAtom.AC)+(type2hb[a]==HbAtom.DA)
                    self.lj_correction_parameters[i,j,2] = (type2hb[a]==HbAtom.HP)
                    self.lj_correction_parameters[i,j,3] = (a=="SH1" or a=="HS")
            for j,a in enumerate(self.aa2elt[i]):
                if (a is not None):
                    self.element_index[i,j] = self.elt2idx[a]


        self.hbtypes = torch.full((self.NAATOKENS,self.NTOTAL,3),-1, dtype=torch.long) # (donortype, acceptortype, acchybtype)
        self.hbbaseatoms = torch.full((self.NAATOKENS,self.NTOTAL,2),-1, dtype=torch.long) # (B,B0) for acc; (D,-1) for don
        self.hbpolys = torch.zeros((HbDonType.NTYPES,HbAccType.NTYPES,3,15)) # weight,xmin,xmax,ymin,ymax,c9,...,c0

        for i in range(self.NNAPROTAAS):
            for j,a in enumerate(self.aa2type[i]):
                if (a in type2dontype):
                    j_hs = self.donorHs(self.aa2long[i][j],self.aabonds[i],self.aa2long[i])
                    for j_h in j_hs:
                        self.hbtypes[i,j_h,0] = type2dontype[a]
                        self.hbbaseatoms[i,j_h,0] = j
                if (a in type2acctype):
                    j_b, j_b0 = self.acceptorBB0(self.aa2long[i][j],type2hybtype[a],self.aabonds[i],self.aa2long[i])
                    self.hbtypes[i,j,1] = type2acctype[a]
                    self.hbtypes[i,j,2] = type2hybtype[a]
                    self.hbbaseatoms[i,j,0] = j_b
                    self.hbbaseatoms[i,j,1] = j_b0

        for i in range(HbDonType.NTYPES):
            for j in range(HbAccType.NTYPES):
                weight = dontype2wt[i]*acctype2wt[j]

                pdist,pbah,pahd = hbtypepair2poly[(i,j)]
                xrange,yrange,coeffs = hbpolytype2coeffs[pdist]
                self.hbpolys[i,j,0,0] = weight
                self.hbpolys[i,j,0,1:3] = torch.tensor(xrange)
                self.hbpolys[i,j,0,3:5] = torch.tensor(yrange)
                self.hbpolys[i,j,0,5:] = torch.tensor(coeffs)
                xrange,yrange,coeffs = hbpolytype2coeffs[pahd]
                self.hbpolys[i,j,1,0] = weight
                self.hbpolys[i,j,1,1:3] = torch.tensor(xrange)
                self.hbpolys[i,j,1,3:5] = torch.tensor(yrange)
                self.hbpolys[i,j,1,5:] = torch.tensor(coeffs)
                xrange,yrange,coeffs = hbpolytype2coeffs[pbah]
                self.hbpolys[i,j,2,0] = weight
                self.hbpolys[i,j,2,1:3] = torch.tensor(xrange)
                self.hbpolys[i,j,2,3:5] = torch.tensor(yrange)
                self.hbpolys[i,j,2,5:] = torch.tensor(coeffs)

        # cartbonded scoring parameters
        # (0) inter-res
        self.cb_lengths_CN = (1.32868, 369.445)
        self.cb_angles_CACN = (2.02807,160)
        self.cb_angles_CNCA = (2.12407,96.53)
        self.cb_torsions_CACNH = (0.0,41.830) # also used for proline CACNCD
        self.cb_torsions_CANCO = (0.0,38.668)

        # note for the below, the extra amino acid corrsponds to cb params for HIS_D
        # (1) intra-res lengths
        self.cb_lengths = [[] for i in range(self.NAATOKENS+1)]
        for cst in cartbonded_data_raw['lengths']:
            res_idx = self.aa2num[ cst['res'] ]
            self.cb_lengths[res_idx].append( (
                self.aa2long[res_idx].index(cst['atm1']),
                self.aa2long[res_idx].index(cst['atm2']),
                cst['x0'],cst['K']
            ) )
        ncst_per_res=max([len(i) for i in self.cb_lengths])
        self.cb_length_t = torch.zeros(self.NAATOKENS+1,ncst_per_res,4)
        for i in range(self.NNAPROTAAS+1):
            src = i
            if (self.num2aa[i]=='UNK' or self.num2aa[i]=='MAS'):
                src=self.aa2num['ALA']
            if (len(self.cb_lengths[src])>0):
                self.cb_length_t[i,:len(self.cb_lengths[src]),:] = torch.tensor(self.cb_lengths[src])

        # (2) intra-res angles
        self.cb_angles = [[] for i in range(self.NAATOKENS+1)]
        for cst in cartbonded_data_raw['angles']:
            res_idx = self.aa2num[ cst['res'] ]
            self.cb_angles[res_idx].append( (
                self.aa2long[res_idx].index(cst['atm1']),
                self.aa2long[res_idx].index(cst['atm2']),
                self.aa2long[res_idx].index(cst['atm3']),
                cst['x0'],cst['K']
            ) )
        ncst_per_res=max([len(i) for i in self.cb_angles])
        self.cb_angle_t = torch.zeros(self.NAATOKENS+1,ncst_per_res,5)
        for i in range(self.NNAPROTAAS+1):
            src = i
            if (self.num2aa[i]=='UNK' or self.num2aa[i]=='MAS'):
                src=self.aa2num['ALA']

            if (len(self.cb_angles[src])>0):
                self.cb_angle_t[i,:len(self.cb_angles[src]),:] = torch.tensor(self.cb_angles[src])

        # (3) intra-res torsions
        self.cb_torsions = [[] for i in range(self.NAATOKENS+1)]
        for cst in cartbonded_data_raw['torsions']:
            res_idx = self.aa2num[ cst['res'] ]
            self.cb_torsions[res_idx].append( (
                self.aa2long[res_idx].index(cst['atm1']),
                self.aa2long[res_idx].index(cst['atm2']),
                self.aa2long[res_idx].index(cst['atm3']),
                self.aa2long[res_idx].index(cst['atm4']),
                cst['x0'],cst['K'],cst['period']
            ) )
        ncst_per_res=max([len(i) for i in self.cb_torsions])
        self.cb_torsion_t = torch.zeros(self.NAATOKENS+1,ncst_per_res,7)
        self.cb_torsion_t[...,6]=1.0 # periodicity
        for i in range(self.NNAPROTAAS):
            src = i
            if (self.num2aa[i]=='UNK' or self.num2aa[i]=='MAS'):
                src=self.aa2num['ALA']

            if (len(self.cb_torsions[src])>0):
                self.cb_torsion_t[i,:len(self.cb_torsions[src]),:] = torch.tensor(self.cb_torsions[src])

        # kinematic parameters
        self.base_indices = torch.full((self.NAATOKENS,self.NTOTAL),0, dtype=torch.long) # base frame that builds each atom
        self.xyzs_in_base_frame = torch.ones((self.NAATOKENS,self.NTOTAL,4)) # coords of each atom in the base frame
        self.RTs_by_torsion = torch.eye(4).repeat(self.NAATOKENS,self.NTOTALTORS,1,1) # torsion frames
        self.reference_angles = torch.ones((self.NAATOKENS,self.NPROTANGS,2)) # reference values for bendable angles

        ## PROTEIN
        for i in range(self.NPROTAAS):
            i_l = self.aa2long[i]
            for name, base, coords in self.ideal_coords[i]:
                idx = i_l.index(name)
                self.base_indices[i,idx] = base
                self.xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

            # omega frame
            self.RTs_by_torsion[i,0,:3,:3] = torch.eye(3)
            self.RTs_by_torsion[i,0,:3,3] = torch.zeros(3)

            # phi frame
            self.RTs_by_torsion[i,1,:3,:3] = make_frame(
                self.xyzs_in_base_frame[i,0,:3] - self.xyzs_in_base_frame[i,1,:3],
                torch.tensor([1.,0.,0.])
            )
            self.RTs_by_torsion[i,1,:3,3] = self.xyzs_in_base_frame[i,0,:3]

            # psi frame
            self.RTs_by_torsion[i,2,:3,:3] = make_frame(
                self.xyzs_in_base_frame[i,2,:3] - self.xyzs_in_base_frame[i,1,:3],
                self.xyzs_in_base_frame[i,1,:3] - self.xyzs_in_base_frame[i,0,:3]
            )
            self.RTs_by_torsion[i,2,:3,3] = self.xyzs_in_base_frame[i,2,:3]

            # chi1 frame
            if self.torsions[i][0] is not None:
                a0,a1,a2 = self.torsion_indices[i,3,0:3]
                self.RTs_by_torsion[i,3,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,a2,:3]-self.xyzs_in_base_frame[i,a1,:3],
                    self.xyzs_in_base_frame[i,a0,:3]-self.xyzs_in_base_frame[i,a1,:3],
                )
                self.RTs_by_torsion[i,3,:3,3] = self.xyzs_in_base_frame[i,a2,:3]

            # chi2/3/4 frame
            for j in range(1,4):
                if self.torsions[i][j] is not None:
                    a2 = self.torsion_indices[i,3+j,2]
                    if ((i==18 and j==2) or (i==8 and j==2)):  # TYR CZ-OH & HIS CE1-HE1 a special case
                        a0,a1 = self.torsion_indices[i,3+j,0:2]
                        self.RTs_by_torsion[i,3+j,:3,:3] = make_frame(
                            self.xyzs_in_base_frame[i,a2,:3]-self.xyzs_in_base_frame[i,a1,:3],
                            self.xyzs_in_base_frame[i,a0,:3]-self.xyzs_in_base_frame[i,a1,:3] )
                    else:
                        self.RTs_by_torsion[i,3+j,:3,:3] = make_frame(
                            self.xyzs_in_base_frame[i,a2,:3],
                            torch.tensor([-1.,0.,0.]), )
                    self.RTs_by_torsion[i,3+j,:3,3] = self.xyzs_in_base_frame[i,a2,:3]

            # CB/CG angles
            NCr = 0.5*(self.xyzs_in_base_frame[i,0,:3]+self.xyzs_in_base_frame[i,2,:3])
            CAr = self.xyzs_in_base_frame[i,1,:3]
            CBr = self.xyzs_in_base_frame[i,4,:3]
            CGr = self.xyzs_in_base_frame[i,5,:3]
            self.reference_angles[i,0,:]=th_ang_v(CBr-CAr,NCr-CAr)
            NCp = self.xyzs_in_base_frame[i,2,:3]-self.xyzs_in_base_frame[i,0,:3]
            NCpp = NCp - torch.dot(NCp,NCr)/ torch.dot(NCr,NCr) * NCr
            self.reference_angles[i,1,:]=th_ang_v(CBr-CAr,NCpp)
            self.reference_angles[i,2,:]=th_ang_v(CGr,torch.tensor([-1.,0.,0.]))

        ## NUCLEIC ACIDS
        if (not params.use_phospate_frames_for_NA):
            # ribose frame
            for i in range(self.NPROTAAS, self.NNAPROTAAS):
                i_l = self.aa2long[i]

                for name, base, coords in self.ideal_coords[i]:
                    idx = i_l.index(name)
                    self.base_indices[i,idx] = base
                    self.xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

                # epsilon(p)/zeta(p) - like omega in protein, not used to build atoms
                #                    - keep as identity
                self.RTs_by_torsion[i,self.NPROTTORS+0,:3,:3] = torch.eye(3)
                self.RTs_by_torsion[i,self.NPROTTORS+0,:3,3] = torch.zeros(3)
                self.RTs_by_torsion[i,self.NPROTTORS+1,:3,:3] = torch.eye(3)
                self.RTs_by_torsion[i,self.NPROTTORS+1,:3,3] = torch.zeros(3)

                # nu1
                self.RTs_by_torsion[i,self.NPROTTORS+7,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,2,:3] , self.xyzs_in_base_frame[i,0,:3]
                )
                self.RTs_by_torsion[i,self.NPROTTORS+7,:3,3] = self.xyzs_in_base_frame[i,2,:3]
    
                # nu0 - currently not used for atom generation
                self.RTs_by_torsion[i,self.NPROTTORS+8,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,0,:3] , self.xyzs_in_base_frame[i,2,:3]
                )
                self.RTs_by_torsion[i,self.NPROTTORS+8,:3,3] = self.xyzs_in_base_frame[i,0,:3] # C2'
    
                # NA chi
                if self.torsions[i][0] is not None:
                    a0,a1,a2 = self.torsion_indices[i,19,0:3]
                    self.RTs_by_torsion[i,self.NPROTTORS+9,:3,:3] = make_frame(
                        self.xyzs_in_base_frame[i,a2,:3], self.xyzs_in_base_frame[i,a0,:3]
                    )
                    self.RTs_by_torsion[i,self.NPROTTORS+9,:3,3] = self.xyzs_in_base_frame[i,a2,:3]

                # nu2
                self.RTs_by_torsion[i,self.NPROTTORS+6,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,9,:3] , torch.tensor([-1.,0.,0.])
                )
                self.RTs_by_torsion[i,self.NPROTTORS+6,:3,3] = self.xyzs_in_base_frame[i,6,:3]
    

                # alpha
                self.RTs_by_torsion[i,self.NPROTTORS+2,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,4,:3], torch.tensor([-1.,0.,0.])
                )
                self.RTs_by_torsion[i,self.NPROTTORS+2,:3,3] = self.xyzs_in_base_frame[i,4,:3]
    
                # beta
                self.RTs_by_torsion[i,self.NPROTTORS+3,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,6,:3] , torch.tensor([-1.,0.,0.])
                )
                self.RTs_by_torsion[i,self.NPROTTORS+3,:3,3] = self.xyzs_in_base_frame[i,6,:3]
    
                # gamma
                self.RTs_by_torsion[i,self.NPROTTORS+4,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,7,:3] , torch.tensor([-1.,0.,0.])
                )
                self.RTs_by_torsion[i,self.NPROTTORS+4,:3,3] = self.xyzs_in_base_frame[i,7,:3]
    
                # delta
                self.RTs_by_torsion[i,self.NPROTTORS+5,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,8,:3] , torch.tensor([-1.,0.,0.])
                )
                self.RTs_by_torsion[i,self.NPROTTORS+5,:3,3] = self.xyzs_in_base_frame[i,8,:3]
        else:
            # phosphate frame
            for i in range(self.NPROTAAS, self.NNAPROTAAS):
                i_l = self.aa2long[i]

                for name, base, coords in self.ideal_coords[i]:
                    idx = i_l.index(name)
                    self.base_indices[i,idx] = base
                    self.xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

                # epsilon(p)/zeta(p) - like omega in protein, not used to build atoms
                #                    - keep as identity
                self.RTs_by_torsion[i,self.NPROTTORS+0,:3,:3] = torch.eye(3)
                self.RTs_by_torsion[i,self.NPROTTORS+0,:3,3] = torch.zeros(3)
                self.RTs_by_torsion[i,self.NPROTTORS+1,:3,:3] = torch.eye(3)
                self.RTs_by_torsion[i,self.NPROTTORS+1,:3,3] = torch.zeros(3)

                # alpha
                self.RTs_by_torsion[i,self.NPROTTORS+2,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,3,:3] - self.xyzs_in_base_frame[i,1,:3], # P->O5'
                    self.xyzs_in_base_frame[i,0,:3] - self.xyzs_in_base_frame[i,1,:3]  # P<-OP1
                )
                self.RTs_by_torsion[i,self.NPROTTORS+2,:3,3] = self.xyzs_in_base_frame[i,3,:3] # O5'

                # beta
                self.RTs_by_torsion[i,self.NPROTTORS+3,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,4,:3] , torch.tensor([-1.,0.,0.])
                )
                self.RTs_by_torsion[i,self.NPROTTORS+3,:3,3] = self.xyzs_in_base_frame[i,4,:3] # C5'

                # gamma
                self.RTs_by_torsion[i,self.NPROTTORS+4,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,5,:3] , torch.tensor([-1.,0.,0.])
                )
                self.RTs_by_torsion[i,self.NPROTTORS+4,:3,3] = self.xyzs_in_base_frame[i,5,:3] # C4'

                # delta
                self.RTs_by_torsion[i,self.NPROTTORS+5,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,7,:3] , torch.tensor([-1.,0.,0.])
                )
                self.RTs_by_torsion[i,self.NPROTTORS+5,:3,3] = self.xyzs_in_base_frame[i,7,:3] # C3'

                # nu2
                self.RTs_by_torsion[i,self.NPROTTORS+6,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,6,:3] , torch.tensor([-1.,0.,0.])
                )
                self.RTs_by_torsion[i,self.NPROTTORS+6,:3,3] = self.xyzs_in_base_frame[i,6,:3] # O4'

                # nu1
                if i<self.NPROTAAS+5:
                    # is DNA
                    C1idx,C2idx = 10,9
                else:
                    # is RNA
                    C1idx,C2idx = 9,10

                self.RTs_by_torsion[i,self.NPROTTORS+7,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,C1idx,:3] , torch.tensor([-1.,0.,0.])
                )
                self.RTs_by_torsion[i,self.NPROTTORS+7,:3,3] = self.xyzs_in_base_frame[i,C1idx,:3] # C1'

                # nu0
                self.RTs_by_torsion[i,self.NPROTTORS+8,:3,:3] = make_frame(
                    self.xyzs_in_base_frame[i,C2idx,:3] , torch.tensor([-1.,0.,0.])
                )
                self.RTs_by_torsion[i,self.NPROTTORS+8,:3,3] = self.xyzs_in_base_frame[i,C2idx,:3] # C2'

                # NA chi
                if self.torsions[i][0] is not None:
                    a2 = self.torsion_indices[i,19,2]
                    self.RTs_by_torsion[i,self.NPROTTORS+9,:3,:3] = make_frame(
                        self.xyzs_in_base_frame[i,a2,:3] , torch.tensor([-1.,0.,0.])
                    )
                    self.RTs_by_torsion[i,self.NPROTTORS+9,:3,3] = self.xyzs_in_base_frame[i,a2,:3]

        #Small molecules
        self.xyzs_in_base_frame[self.NNAPROTAAS:,1, :3] = 0
        # general FAPE parameters
        self.frame_indices = torch.full((self.NAATOKENS,self.NFRAMES,3,2),0, dtype=torch.long)
        for i in range(self.NNAPROTAAS):
            i_l = self.aa2long[i]
            for j,x in enumerate(self.frames[i]):
                if x is not None:
                    # frames are stored as (residue offset, atom position)
                    self.frame_indices[i,j,0] = torch.tensor((0, i_l.index(x[0])))
                    self.frame_indices[i,j,1] = torch.tensor((0, i_l.index(x[1])))
                    self.frame_indices[i,j,2] = torch.tensor((0, i_l.index(x[2])))

    def seq2chars(self, seq):
        out = ''.join([self.aa_321[self.num2aa[a]] for a in seq])
        return out

    # hbond scoring parameters
    def donorHs(self, D,bonds,atoms):
        dHs = []
        for (i,j) in bonds:
            if (i==D):
                idx_j = atoms.index(j)
                if (idx_j>=self.NHEAVY):  # if atom j is a hydrogen
                    dHs.append(idx_j)
            if (j==D):
                idx_i = atoms.index(i)
                if (idx_i>=self.NHEAVY):  # if atom j is a hydrogen
                    dHs.append(idx_i)
        assert (len(dHs)>0)
        return dHs

    def acceptorBB0(self, A,hyb,bonds,atoms):
        if (hyb == HbHybType.SP2):
            for (i,j) in bonds:
                if (i==A):
                    B = atoms.index(j)
                    if (B<self.NHEAVY):
                        break
                if (j==A):
                    B = atoms.index(i)
                    if (B<self.NHEAVY):
                        break
            for (i,j) in bonds:
                if (i==atoms[B]):
                    B0 = atoms.index(j)
                    if (B0<self.NHEAVY):
                        break
                if (j==atoms[B]):
                    B0 = atoms.index(i)
                    if (B0<self.NHEAVY):
                        break
        elif (hyb == HbHybType.SP3 or hyb == HbHybType.RING):
            for (i,j) in bonds:
                if (i==A):
                    B = atoms.index(j)
                    if (B<self.NHEAVY):
                        break
                if (j==A):
                    B = atoms.index(i)
                    if (B<self.NHEAVY):
                        break
            for (i,j) in bonds:
                if (i==A and j!=atoms[B]):
                    B0 = atoms.index(j)
                    break
                if (j==A and i!=atoms[B]):
                    B0 = atoms.index(i)
                    break

        return B,B0


def load_pdb_ideal_sdf_strings(base_path: Optional[str] = script_dir, return_only_sdf_strings: bool = False):
    """
    returns a dictionary of that maps all the 3letter ligand codes in the pdb to relevant information:
        string of the sdf file with idealized coordinates for that molecule
        atom names for the atoms in order
        leaving groups if there are covalent modifications to the ligand
        pdbx_align 
    """
    file_name = 'ligands.json.gz'
    if base_path is not None:
        file_name = Path(base_path) / file_name

    with gzip.open(file_name,'rt') as file:
        mols = json.load(file)

    if return_only_sdf_strings:
        mols_only_sdf = {key: value["sdf"] for key, value in mols.items()}
        return mols_only_sdf
    else:
        return mols

def load_tanimoto_sim_matrix(base_path=script_dir):
    """
    precomputed tanimoto similarities between ligands in the pdb until august 2021
    returns num_molecules X num_molecules matrix of tanimoto scores and
    list of names corresponding to indices in that matrix
    """
    file_name = 'tanimoto_ligands.npz'
    if base_path is not None:
        file_name = Path(base_path) / file_name

    tanimoto_data = np.load(file_name, allow_pickle=True)
    
    sim = tanimoto_data['sim']
    names = tanimoto_data['names']
    return sim, names
