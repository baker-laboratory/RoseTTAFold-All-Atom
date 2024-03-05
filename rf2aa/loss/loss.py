import torch
from icecream import ic
import numpy as np
import scipy
import pandas as pd
import networkx as nx

from rf2aa.util import (
    is_protein,
    rigid_from_3_points,
    is_nucleic,
    find_all_paths_of_length_n,
    find_all_rigid_groups
)
from rf2aa.chemical import ChemicalData as ChemData

from rf2aa.kinematics import get_dih, get_ang
from rf2aa.scoring import HbHybType

from typing import List, Dict, Optional


#fd more efficient LJ loss
class LJLoss(torch.autograd.Function):
    @staticmethod
    def ljVdV(deltas, sigma, epsilon, lj_lin, eps):
        # deltas - (N,natompair,3)
        N = deltas.shape[0]

        dist = torch.sqrt( torch.sum ( torch.square( deltas ), dim=-1 ) + eps )
        linpart = dist<lj_lin*sigma[None]
        deff = dist.clone()
        deff[linpart] = lj_lin*sigma.repeat(N,1)[linpart]
        sd = sigma / deff
        sd2 = sd*sd
        sd6 = sd2 * sd2 * sd2
        sd12 = sd6 * sd6
        ljE = epsilon * (sd12 - 2 * sd6)
        ljE[linpart] += epsilon.repeat(N,1)[linpart] * (
            -12 * sd12[linpart]/deff[linpart] + 12 * sd6[linpart]/deff[linpart]
        ) * (dist[linpart]-deff[linpart])

        # works for linpart too
        dljEdd_over_r = epsilon * (-12 * sd12/deff + 12 * sd6/deff) / (dist)

        return ljE.sum(dim=-1), dljEdd_over_r

    @staticmethod
    def forward(
        ctx, xs, seq, aamask, bond_feats, dist_matrix, ljparams, ljcorr, num_bonds, 
        lj_lin=0.75, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
        eps=1e-8, training=True
    ):
        N, L, A = xs.shape[:3]
        assert (N==1) # see comment below

        ds_res = torch.sqrt( torch.sum ( torch.square( 
            xs.detach()[:,:,None,1,:]-xs.detach()[:,None,:,1,:]), dim=-1 ))
        rs = torch.triu_indices(L,L,0, device=xs.device)
        ri,rj = rs[0],rs[1]

        # batch during inference for huge systems
        BATCHSIZE = 65536//N

        ljval = 0
        dljEdx = torch.zeros_like(xs, dtype=torch.float)

        for i_batch in range((len(ri)-1)//BATCHSIZE + 1):
            idx = torch.arange(
                i_batch*BATCHSIZE, 
                min( (i_batch+1)*BATCHSIZE, len(ri)),
                device=xs.device
            )
            rii,rjj = ri[idx],rj[idx] # residue pairs we consider

            ridx,ai,aj = (
                aamask[seq[rii]][:,:,None]*aamask[seq[rjj]][:,None,:]
            ).nonzero(as_tuple=True)

            deltas = xs[:,rii,:,None,:]-xs[:,rjj,None,:,:] # N,BATCHSIZE,Natm,Natm,3
            seqi,seqj = seq[rii[ridx]], seq[rjj[ridx]]

            mask = torch.ones_like(ridx, dtype=torch.bool) # are atoms defined?

            # mask out atom pairs from too-distant residues (C-alpha dist > 24A)
            ca_dist = torch.linalg.norm(deltas[:,:,1,1],dim=-1)
            mask *= (ca_dist[:,ridx]<24).any(dim=0)  # will work for batch>1 but very inefficient

            intrares = (rii[ridx]==rjj[ridx])
            mask[intrares*(ai<aj)] = False  # upper tri (atoms)

            ## count-pair
            # a) intra-protein
            mask[intrares] *= num_bonds[seqi[intrares],ai[intrares],aj[intrares]]>=4
            pepbondres = ri[ridx]+1==rj[ridx]
            mask[pepbondres] *= (
                num_bonds[seqi[pepbondres],ai[pepbondres],2]
                + num_bonds[seqj[pepbondres],0,aj[pepbondres]]
                + 1) >=4

            # b) intra-ligand
            atommask = (ai==1)*(aj==1)
            dist_matrix = torch.nan_to_num(dist_matrix, posinf=4.0) #NOTE: need to run nan_to_num to remove infinities
            resmask = (dist_matrix[0,rii,rjj] >= 4) # * will only work for batch=1
            mask[atommask] *= resmask[ ridx[atommask] ]

            # c) protein/ligand
            ##fd NOTE1: changed 6->5 in masking (atom 5 is CG which should always be 4+ bonds away from connected atom)
            ##fd NOTE2: this does NOT work correctly for nucleic acids
            ##fd     for NAs atoms 0-4 are masked, but also 5,7,8 and 9 should be masked!
            bbatommask = (ai<5)*(aj<5)
            resmask = (bond_feats[0,rii,rjj] != 6) # * will only work for batch=1
            mask[bbatommask] *= resmask[ ridx[bbatommask] ]

            # apply mask.  only interactions to be scored remain
            ai,aj,seqi,seqj,ridx = ai[mask],aj[mask],seqi[mask],seqj[mask],ridx[mask]
            deltas = deltas[:,ridx,ai,aj]

            # hbond correction
            use_hb_dis = (
                ljcorr[seqi,ai,0]*ljcorr[seqj,aj,1] 
                + ljcorr[seqi,ai,1]*ljcorr[seqj,aj,0] ).nonzero()
            use_ohdon_dis = ( # OH are both donors & acceptors
                ljcorr[seqi,ai,0]*ljcorr[seqi,ai,1]*ljcorr[seqj,aj,0] 
                +ljcorr[seqi,ai,0]*ljcorr[seqj,aj,0]*ljcorr[seqj,aj,1] 
            ).nonzero()
            use_hb_hdis = (
                ljcorr[seqi,ai,2]*ljcorr[seqj,aj,1] 
                +ljcorr[seqi,ai,1]*ljcorr[seqj,aj,2] 
            ).nonzero()

            # disulfide correction
            potential_disulf = (ljcorr[seqi,ai,3]*ljcorr[seqj,aj,3] ).nonzero()

            ljrs = ljparams[seqi,ai,0] + ljparams[seqj,aj,0]
            ljrs[use_hb_dis] = lj_hb_dis
            ljrs[use_ohdon_dis] = lj_OHdon_dis
            ljrs[use_hb_hdis] = lj_hbond_hdis

            ljss = torch.sqrt( ljparams[seqi,ai,1] * ljparams[seqj,aj,1] + eps )
            ljss [potential_disulf] = 0.0

            natoms = torch.sum(aamask[seq])
            ljval_i,dljEdd_i = LJLoss.ljVdV(deltas,ljrs,ljss,lj_lin,eps)

            ljval += ljval_i / natoms

            # sum per-atom-pair grads into per-atom grads
            # note this is stochastic op on GPU
            idxI,idxJ = rii[ridx]*A + ai, rjj[ridx]*A + aj

            dljEdx.view(N,-1,3).index_add_(1, idxI, dljEdd_i[...,None]*deltas, alpha=1.0/natoms)
            dljEdx.view(N,-1,3).index_add_(1, idxJ, dljEdd_i[...,None]*deltas, alpha=-1.0/natoms)

        ctx.save_for_backward(dljEdx)

        return ljval



    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        dljEdx, = ctx.saved_tensors
        return (
            grad_output * dljEdx, 
            None, None, None, None, None, None, None, None, None, None, None, None, None
        )

# Rosetta-like version of LJ (fa_atr+fa_rep)
#   lj_lin is switch from linear to 12-6.  Smaller values more sharply penalize clashes
def calc_lj(
    seq, xs, aamask, bond_feats, dist_matrix, ljparams, ljcorr, num_bonds,  
    lj_lin=0.75, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8,
    training=True
):
    lj = LJLoss.apply
    ljval = lj(
        xs, seq, aamask, bond_feats, dist_matrix, ljparams, ljcorr, num_bonds, 
        lj_lin, lj_hb_dis, lj_OHdon_dis, lj_hbond_hdis, eps, training)

    return ljval


def calc_chiral_loss(pred, chirals):
    """
    calculate error in dihedral angles for chiral atoms
    Input:
     - pred: predicted coords (B, L, :, 3)
     - chirals: True coords (B, nchiral, 5), skip if 0 chiral sites, 5 dimension are indices for 4 atoms that make dihedral and the ideal angle they should form
    Output:
     - mean squared error of chiral angles
    """
    if chirals.shape[1] == 0:
        return torch.tensor(0.0, device=pred.device)
    chiral_dih = pred[:, chirals[..., :-1].long(), 1]
    pred_dih = get_dih(chiral_dih[...,0, :], chiral_dih[...,1, :], chiral_dih[...,2, :], chiral_dih[...,3, :]) # n_symm, b, n, 36, 3
    l = torch.square(pred_dih-chirals[...,-1]).mean()
    return l

@torch.enable_grad()
def calc_lj_grads(
    seq, xyz, alpha, toaa, bond_feats, dist_matrix, 
    aamask, ljparams, ljcorr, num_bonds, 
    lj_lin=0.85, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8
):
    xyz.requires_grad_(True)
    alpha.requires_grad_(True)
    _, xyzaa = toaa(seq, xyz, alpha)
    Elj = calc_lj(
        seq[0], 
        xyzaa[...,:3], 
        aamask,
        bond_feats,
        dist_matrix,
        ljparams, 
        ljcorr, 
        num_bonds,
        lj_lin, 
        lj_hb_dis, 
        lj_OHdon_dis, 
        lj_hbond_hdis, 
        lj_maxrad,
        eps
    )
    return torch.autograd.grad(Elj, (xyz,alpha))

@torch.enable_grad()
def calc_chiral_grads(xyz, chirals):
    xyz.requires_grad_(True)
    l = calc_chiral_loss(xyz, chirals)
    if l.item() == 0.0:
        return (torch.zeros(xyz.shape, device=xyz.device),) # autograd returns a tuple..
    return torch.autograd.grad(l, xyz)
