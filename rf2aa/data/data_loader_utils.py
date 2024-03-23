import torch
import warnings
import time
from icecream import ic
from torch.utils import data
import os, csv, random, pickle, gzip, itertools, time, ast, copy, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(script_dir+'/../')

import numpy as np
import scipy
import networkx as nx

from rf2aa.data.parsers import parse_a3m, parse_pdb
from rf2aa.chemical import ChemicalData as ChemData


from rf2aa.util import random_rot_trans, \
    is_atom, is_protein, is_nucleic, is_atom


def MSABlockDeletion(msa, ins, nb=5):
    '''
    Input: MSA having shape (N, L)
    output: new MSA with block deletion
    '''
    N, L = msa.shape
    block_size = max(int(N*0.3), 1)
    block_start = np.random.randint(low=1, high=N, size=nb) # (nb)
    to_delete = block_start[:,None] + np.arange(block_size)[None,:]
    to_delete = np.unique(np.clip(to_delete, 1, N-1))
    #
    mask = np.ones(N, bool)
    mask[to_delete] = 0

    return msa[mask], ins[mask]

def cluster_sum(data, assignment, N_seq, N_res):
    csum = torch.zeros(N_seq, N_res, data.shape[-1], device=data.device).scatter_add(0, assignment.view(-1,1,1).expand(-1,N_res,data.shape[-1]), data.float())
    return csum

def get_term_feats(Ls):
    """Creates N/C-terminus binary features"""
    term_info = torch.zeros((sum(Ls),2)).float()
    start = 0
    for L_chain in Ls:
        term_info[start, 0] = 1.0 # flag for N-term
        term_info[start+L_chain-1,1] = 1.0 # flag for C-term
        start += L_chain
    return term_info


def MSAFeaturize(msa, ins, params, p_mask=0.15, eps=1e-6, nmer=1, L_s=[], 
    term_info=None, tocpu=False, fixbb=False, seed_msa_clus=None, deterministic=False):
    '''
    Input: full MSA information (after Block deletion if necessary) & full insertion information
    Output: seed MSA features & extra sequences
    
    Seed MSA features:
        - aatype of seed sequence (20 regular aa + 1 gap/unknown + 1 mask)
        - profile of clustered sequences (22)
        - insertion statistics (2)
        - N-term or C-term? (2)
    extra sequence features:
        - aatype of extra sequence (22)
        - insertion info (1)
        - N-term or C-term? (2)
    '''
    if deterministic:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        # TODO: delete me, just for testing purposes
        msa = msa[:2]

    if fixbb:
        p_mask = 0
        msa = msa[:1]
        ins = ins[:1]
    N, L = msa.shape
    
    if term_info is None:
        if len(L_s)==0:
            L_s = [L]
        term_info = get_term_feats(L_s)
    term_info = term_info.to(msa.device)

    #binding_site = torch.zeros((L,1), device=msa.device).float()
    binding_site = torch.zeros((L,0), device=msa.device).float() # keeping this off for now (Jue 12/19)
        
    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=ChemData().NAATOKENS) # N x L x NAATOKENS
    raw_profile = raw_profile.float().mean(dim=0) # L x NAATOKENS

    # Select Nclust sequence randomly (seed MSA or latent MSA)
    Nclust = (min(N, params['MAXLAT'])-1) // nmer 
    Nclust = Nclust*nmer + 1
    
    if N > Nclust*2:
        Nextra = N - Nclust
    else:
        Nextra = N
    Nextra = min(Nextra, params['MAXSEQ']) // nmer
    Nextra = max(1, Nextra * nmer)
    #
    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()
    for i_cycle in range(params['MAXCYCLE']):
        sample_mono = torch.randperm((N-1)//nmer, device=msa.device)
        sample = [sample_mono + imer*((N-1)//nmer) for imer in range(nmer)]
        sample = torch.stack(sample, dim=-1)
        sample = sample.reshape(-1)

        # add MSA clusters pre-chosen before calling this function
        if seed_msa_clus is not None:
            sample_orig_shape = sample.shape
            sample_seed = seed_msa_clus[i_cycle]
            sample_more = torch.tensor([i for i in sample if i not in sample_seed])
            N_sample_more = len(sample) - len(sample_seed)
            if N_sample_more > 0:
                sample_more = sample_more[torch.randperm(len(sample_more))[:N_sample_more]]
                sample = torch.cat([sample_seed, sample_more])
            else:
                sample = sample_seed[:len(sample)] # take all clusters from pre-chosen ones

        msa_clust = torch.cat((msa[:1,:], msa[1:,:][sample[:Nclust-1]]), dim=0)
        ins_clust = torch.cat((ins[:1,:], ins[1:,:][sample[:Nclust-1]]), dim=0)

        # 15% random masking 
        # - 10%: aa replaced with a uniformly sampled random amino acid
        # - 10%: aa replaced with an amino acid sampled from the MSA profile
        # - 10%: not replaced
        # - 70%: replaced with a special token ("mask")
        random_aa = torch.tensor([[0.05]*20 + [0.0]*(ChemData().NAATOKENS-20)], device=msa.device)
        same_aa = torch.nn.functional.one_hot(msa_clust, num_classes=ChemData().NAATOKENS)
        # explicitly remove probabilities from nucleic acids and atoms
        #same_aa[..., ChemData().NPROTAAS:] = 0
        #raw_profile[...,ChemData().NPROTAAS:] = 0
        probs = 0.1*random_aa + 0.1*raw_profile + 0.1*same_aa
        #probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)
        
        # explicitly set the probability of masking for nucleic acids and atoms
        #probs[...,is_protein(seq),ChemData().MASKINDEX]=0.7
        #probs[...,~is_protein(seq), :] = 0 # probably overkill but set all none protein elements to 0
        #probs[1:, ~is_protein(seq),20] = 1.0 # want to leave the gaps as gaps
        #probs[0,is_nucleic(seq), ChemData().MASKINDEX] = 1.0
        #probs[0,is_atom(seq), ChemData().aa2num["ATM"]] = 1.0
        
        sampler = torch.distributions.categorical.Categorical(probs=probs)
        mask_sample = sampler.sample()

        mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < p_mask
        mask_pos[msa_clust>ChemData().MASKINDEX]=False # no masking on NAs
        use_seq = msa_clust
        msa_masked = torch.where(mask_pos, mask_sample, use_seq)
        b_seq.append(msa_masked[0].clone())

        ## get extra sequenes
        if N > Nclust*2:  # there are enough extra sequences
            msa_extra = msa[1:,:][sample[Nclust-1:]]
            ins_extra = ins[1:,:][sample[Nclust-1:]]
            extra_mask = torch.full(msa_extra.shape, False, device=msa_extra.device)
        elif N - Nclust < 1:
            msa_extra = msa_masked.clone()
            ins_extra = ins_clust.clone()
            extra_mask = mask_pos.clone()
        else:
            msa_add = msa[1:,:][sample[Nclust-1:]]
            ins_add = ins[1:,:][sample[Nclust-1:]]
            mask_add = torch.full(msa_add.shape, False, device=msa_add.device)
            msa_extra = torch.cat((msa_masked, msa_add), dim=0)
            ins_extra = torch.cat((ins_clust, ins_add), dim=0)
            extra_mask = torch.cat((mask_pos, mask_add), dim=0)
        N_extra = msa_extra.shape[0]
        
        # clustering (assign remaining sequences to their closest cluster by Hamming distance
        msa_clust_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=ChemData().NAATOKENS)
        msa_extra_onehot = torch.nn.functional.one_hot(msa_extra, num_classes=ChemData().NAATOKENS)
        count_clust = torch.logical_and(~mask_pos, msa_clust != 20).float() # 20: index for gap, ignore both masked & gaps
        count_extra = torch.logical_and(~extra_mask, msa_extra != 20).float()
        agreement = torch.matmul((count_extra[:,:,None]*msa_extra_onehot).view(N_extra, -1), (count_clust[:,:,None]*msa_clust_onehot).view(Nclust, -1).T)
        assignment = torch.argmax(agreement, dim=-1)

        # seed MSA features
        # 1. one_hot encoded aatype: msa_clust_onehot
        # 2. cluster profile
        count_extra = ~extra_mask
        count_clust = ~mask_pos
        msa_clust_profile = cluster_sum(count_extra[:,:,None]*msa_extra_onehot, assignment, Nclust, L)
        msa_clust_profile += count_clust[:,:,None]*msa_clust_profile
        count_profile = cluster_sum(count_extra[:,:,None], assignment, Nclust, L).view(Nclust, L)
        count_profile += count_clust
        count_profile += eps
        msa_clust_profile /= count_profile[:,:,None]
        # 3. insertion statistics
        msa_clust_del = cluster_sum((count_extra*ins_extra)[:,:,None], assignment, Nclust, L).view(Nclust, L)
        msa_clust_del += count_clust*ins_clust
        msa_clust_del /= count_profile
        ins_clust = (2.0/np.pi)*torch.arctan(ins_clust.float()/3.0) # (from 0 to 1)
        msa_clust_del = (2.0/np.pi)*torch.arctan(msa_clust_del.float()/3.0) # (from 0 to 1)
        ins_clust = torch.stack((ins_clust, msa_clust_del), dim=-1)
        #
        if fixbb:
            assert params['MAXCYCLE'] == 1
            msa_clust_profile = msa_clust_onehot
            msa_extra_onehot = msa_clust_onehot
            ins_clust[:] = 0
            ins_extra[:] = 0
            # This is how it is done in rfdiff, but really it seems like it should be all 0.
            # Keeping as-is for now for consistency, as it may be used in downstream masking done
            # by apply_masks.
            mask_pos = torch.full_like(msa_clust, 1).bool()
        msa_seed = torch.cat((msa_clust_onehot, msa_clust_profile, ins_clust, term_info[None].expand(Nclust,-1,-1)), dim=-1)

        # extra MSA features
        ins_extra = (2.0/np.pi)*torch.arctan(ins_extra[:Nextra].float()/3.0) # (from 0 to 1)
        try:
            msa_extra = torch.cat((msa_extra_onehot[:Nextra], ins_extra[:,:,None], term_info[None].expand(Nextra,-1,-1)), dim=-1)
        except Exception as e:
            print('msa_extra.shape',msa_extra.shape)
            print('ins_extra.shape',ins_extra.shape)

        if (tocpu):
            b_msa_clust.append(msa_clust.cpu())
            b_msa_seed.append(msa_seed.cpu())
            b_msa_extra.append(msa_extra.cpu())
            b_mask_pos.append(mask_pos.cpu())
        else:
            b_msa_clust.append(msa_clust)
            b_msa_seed.append(msa_seed)
            b_msa_extra.append(msa_extra)
            b_mask_pos.append(mask_pos)
    
    b_seq = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos = torch.stack(b_mask_pos)

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

def blank_template(n_tmpl, L, random_noise=5.0, deterministic: bool = False):
    if deterministic:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

    xyz = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(n_tmpl,L,1,1) \
        + torch.rand(n_tmpl,L,1,3)*random_noise - random_noise/2
    t1d = torch.nn.functional.one_hot(torch.full((n_tmpl, L), 20).long(), num_classes=ChemData().NAATOKENS-1).float() # all gaps
    conf = torch.zeros((n_tmpl, L, 1)).float()
    t1d = torch.cat((t1d, conf), -1)
    mask_t = torch.full((n_tmpl,L,ChemData().NTOTAL), False)
    return xyz, t1d, mask_t, np.full((n_tmpl), "")


def TemplFeaturize(tplt, qlen, params, offset=0, npick=1, npick_global=None, pick_top=True, same_chain=None, random_noise=5, deterministic: bool = False):
    if deterministic:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
    
    seqID_cut = params['SEQID']

    if npick_global == None:
        npick_global=max(npick, 1)

    ntplt = len(tplt['ids'])
    if (ntplt < 1) or (npick < 1): #no templates in hhsearch file or not want to use templ
        return blank_template(npick_global, qlen, random_noise)
    
    # ignore templates having too high seqID
    if seqID_cut <= 100.0:
        tplt_valid_idx = torch.where(tplt['f0d'][0,:,4] < seqID_cut)[0]
        tplt['ids'] = np.array(tplt['ids'])[tplt_valid_idx]
    else:
        tplt_valid_idx = torch.arange(len(tplt['ids']))
    
    # check again if there are templates having seqID < cutoff
    ntplt = len(tplt['ids'])
    npick = min(npick, ntplt)
    if npick<1: # no templates
        return blank_template(npick_global, qlen, random_noise)

    if not pick_top: # select randomly among all possible templates
        sample = torch.randperm(ntplt)[:npick]
    else: # only consider top 50 templates
        sample = torch.randperm(min(50,ntplt))[:npick]

    xyz = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(npick_global,qlen,1,1) + torch.rand(1,qlen,1,3)*random_noise
    mask_t = torch.full((npick_global,qlen,ChemData().NTOTAL),False) # True for valid atom, False for missing atom
    t1d = torch.full((npick_global, qlen), 20).long()
    t1d_val = torch.zeros((npick_global, qlen)).float()
    for i,nt in enumerate(sample):
        tplt_idx = tplt_valid_idx[nt]
        sel = torch.where(tplt['qmap'][0,:,1]==tplt_idx)[0]
        pos = tplt['qmap'][0,sel,0] + offset

        ntmplatoms = tplt['xyz'].shape[2] # will be bigger for NA templates
        xyz[i,pos,:ntmplatoms] = tplt['xyz'][0,sel]
        mask_t[i,pos,:ntmplatoms] = tplt['mask'][0,sel].bool()

        # 1-D features: alignment confidence 
        t1d[i,pos] = tplt['seq'][0,sel]
        t1d_val[i,pos] = tplt['f1d'][0,sel,2] # alignment confidence
        # xyz[i] = center_and_realign_missing(xyz[i], mask_t[i], same_chain=same_chain)

    t1d = torch.nn.functional.one_hot(t1d, num_classes=ChemData().NAATOKENS-1).float() # (no mask token)
    t1d = torch.cat((t1d, t1d_val[...,None]), dim=-1)

    tplt_ids = np.array(tplt["ids"])[sample].flatten() # np.array of chain ids (ordered)
    return xyz, t1d, mask_t, tplt_ids

def merge_hetero_templates(xyz_t_prot, f1d_t_prot, mask_t_prot, tplt_ids, Ls_prot, deterministic: bool = False):
    """Diagonally tiles template coordinates, 1d input features, and masks across
    template and residue dimensions. 1st template is concatenated directly on residue
    dimension after a random rotation & translation.
    """
    N_tmpl_tot = sum([x.shape[0] for x in xyz_t_prot])

    xyz_t_out, f1d_t_out, mask_t_out, _ = blank_template(N_tmpl_tot, sum(Ls_prot))
    tplt_ids_out = np.full((N_tmpl_tot),"", dtype=object) # rk bad practice.. should fix
    i_tmpl = 0
    i_res = 0
    for xyz_, f1d_, mask_, ids in zip(xyz_t_prot, f1d_t_prot, mask_t_prot, tplt_ids):
        N_tmpl, L_tmpl = xyz_.shape[:2]
        if i_tmpl == 0:
            i1, i2 = 1, N_tmpl
        else:
            i1, i2 = i_tmpl, i_tmpl+N_tmpl - 1
 
        # 1st template is concatenated directly, so that all atoms are set in xyz_prev
        xyz_t_out[0, i_res:i_res+L_tmpl] = random_rot_trans(xyz_[0:1], deterministic=deterministic)
        f1d_t_out[0, i_res:i_res+L_tmpl] = f1d_[0]
        mask_t_out[0, i_res:i_res+L_tmpl] = mask_[0]

        if not tplt_ids_out[0]: # only add first template
            tplt_ids_out[0] = ids[0]
        # remaining templates are diagonally tiled
        xyz_t_out[i1:i2, i_res:i_res+L_tmpl] = xyz_[1:]
        f1d_t_out[i1:i2, i_res:i_res+L_tmpl] = f1d_[1:]
        mask_t_out[i1:i2, i_res:i_res+L_tmpl] = mask_[1:]
        tplt_ids_out[i1:i2] = ids[1:] 
        if i_tmpl == 0:
            i_tmpl += N_tmpl
        else:
            i_tmpl += N_tmpl-1
        i_res += L_tmpl

    return xyz_t_out, f1d_t_out, mask_t_out, tplt_ids_out

def generate_xyz_prev(xyz_t, mask_t, params):
    """
    allows you to use different initializations for the coordinate track specified in params
    """
    L = xyz_t.shape[1]
    if params["BLACK_HOLE_INIT"]:
        xyz_t, _, mask_t = blank_template(1, L)
    return xyz_t[0].clone(), mask_t[0].clone()

### merge msa & insertion statistics of two proteins having different taxID
def merge_a3m_hetero(a3mA, a3mB, L_s):
    # merge msa
    query = torch.cat([a3mA['msa'][0], a3mB['msa'][0]]).unsqueeze(0) # (1, L)

    msa = [query]
    if a3mA['msa'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['msa'][1:], (0,sum(L_s[1:])), "constant", 20) # pad gaps
        msa.append(extra_A)
    if a3mB['msa'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['msa'][1:], (L_s[0],0), "constant", 20)
        msa.append(extra_B)
    msa = torch.cat(msa, dim=0)

    # merge ins
    query = torch.cat([a3mA['ins'][0], a3mB['ins'][0]]).unsqueeze(0) # (1, L)
    ins = [query]
    if a3mA['ins'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['ins'][1:], (0,sum(L_s[1:])), "constant", 0) # pad gaps
        ins.append(extra_A)
    if a3mB['ins'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['ins'][1:], (L_s[0],0), "constant", 0)
        ins.append(extra_B)
    ins = torch.cat(ins, dim=0)

    a3m = {'msa': msa, 'ins': ins}

    # merge taxids
    if 'taxid' in a3mA and 'taxid' in a3mB:
        a3m['taxid'] = np.concatenate([np.array(a3mA['taxid']), np.array(a3mB['taxid'])[1:]])

    return a3m

# merge msa & insertion statistics of units in homo-oligomers
def merge_a3m_homo(msa_orig, ins_orig, nmer, mode="default"):
     N, L = msa_orig.shape[:2]
     if mode == "repeat":

         # AAAAAA
         # AAAAAA

         msa = torch.tile(msa_orig,(1,nmer))
         ins = torch.tile(ins_orig,(1,nmer))

     elif mode == "diag":

         # AAAAAA
         # A-----
         # -A----
         # --A---
         # ---A--
         # ----A-
         # -----A

         N = N - 1
         new_N = 1 + N * nmer
         new_L = L * nmer
         msa = torch.full((new_N, new_L), 20, dtype=msa_orig.dtype, device=msa_orig.device)
         ins = torch.full((new_N, new_L), 0, dtype=ins_orig.dtype, device=msa_orig.device)

         start_L = 0
         start_N = 1
         for i_c in range(nmer):
             msa[0, start_L:start_L+L] = msa_orig[0] 
             msa[start_N:start_N+N, start_L:start_L+L] = msa_orig[1:]
             ins[0, start_L:start_L+L] = ins_orig[0]
             ins[start_N:start_N+N, start_L:start_L+L] = ins_orig[1:]
             start_L += L
             start_N += N
     else:

         # AAAAAA
         # A-----
         # -AAAAA

         msa = torch.full((2*N-1, L*nmer), 20, dtype=msa_orig.dtype, device=msa_orig.device)
         ins = torch.full((2*N-1, L*nmer), 0, dtype=ins_orig.dtype, device=msa_orig.device)

         msa[:N, :L] = msa_orig
         ins[:N, :L] = ins_orig
         start = L

         for i_c in range(1,nmer):
             msa[0, start:start+L] = msa_orig[0] 
             msa[N:, start:start+L] = msa_orig[1:]
             ins[0, start:start+L] = ins_orig[0]
             ins[N:, start:start+L] = ins_orig[1:]
             start += L        

     return {"msa": msa, "ins": ins}

def merge_msas(a3m_list, L_s):
    """
    takes a list of a3m dictionaries with keys msa, ins and a list of protein lengths and creates a
    combined MSA 
    """
    seen = set()
    taxIDs = []
    a3mA = a3m_list[0]
    taxIDs.extend(a3mA["taxID"])
    seen.update(a3mA["hash"])
    msaA, insA = a3mA["msa"], a3mA["ins"]
    for i in range(1, len(a3m_list)):
        a3mB = a3m_list[i]
        pair_taxIDs = set(taxIDs).intersection(set(a3mB["taxID"]))
        if a3mB["hash"] in seen or len(pair_taxIDs) < 5: #homomer/not enough pairs 
            a3mA = {"msa": msaA, "ins": insA}
            L_s_to_merge = [sum(L_s[:i]), L_s[i]]
            a3mA = merge_a3m_hetero(a3mA, a3mB, L_s_to_merge)
            msaA, insA = a3mA["msa"], a3mA["ins"]
            taxIDs.extend(a3mB["taxID"])
        else:
            final_pairsA = []
            final_pairsB = []
            msaB, insB = a3mB["msa"], a3mB["ins"]
            for pair in pair_taxIDs:
                pair_a3mA = np.where(np.array(taxIDs)==pair)[0]
                pair_a3mB = np.where(a3mB["taxID"]==pair)[0]
                msaApair = torch.argmin(torch.sum(msaA[pair_a3mA, :] == msaA[0, :],axis=-1))
                msaBpair = torch.argmin(torch.sum(msaB[pair_a3mB, :] == msaB[0, :],axis=-1))
                final_pairsA.append(pair_a3mA[msaApair])
                final_pairsB.append(pair_a3mB[msaBpair])
            paired_msaB = torch.full((msaA.shape[0], L_s[i]), 20).long() # (N_seq_A, L_B)
            paired_msaB[final_pairsA] = msaB[final_pairsB]
            msaA = torch.cat([msaA, paired_msaB], dim=1)
            insA = torch.zeros_like(msaA) # paired MSAs in our dataset dont have insertions 
        seen.update(a3mB["hash"])
        
    return msaA, insA

def remove_all_gap_seqs(a3m):
    """Removes sequences that are all gaps from an MSA represented as `a3m` dictionary"""
    idx_seq_keep = ~(a3m['msa']==ChemData().UNKINDEX).all(dim=1)
    a3m['msa'] = a3m['msa'][idx_seq_keep]
    a3m['ins'] = a3m['ins'][idx_seq_keep]
    return a3m

def join_msas_by_taxid(a3mA, a3mB, idx_overlap=None):
    """Joins (or "pairs") 2 MSAs by matching sequences with the same
    taxonomic ID. If more than 1 sequence exists in both MSAs with the same tax
    ID, only the sequence with the highest sequence identity to the query (1st
    sequence in MSA) will be paired.
    
    Sequences that aren't paired will be padded and added to the bottom of the
    joined MSA.  If a subregion of the input MSAs overlap (represent the same
    chain), the subregion residue indices can be given as `idx_overlap`, and
    the overlap region of the unpaired sequences will be included in the joined
    MSA.
    
    Parameters
    ----------
    a3mA : dict
        First MSA to be joined, with keys `msa` (N_seq, L_seq), `ins` (N_seq,
        L_seq), `taxid` (N_seq,), and optionally `is_paired` (N_seq,), a
        boolean tensor indicating whether each sequence is fully paired. Can be
        a multi-MSA (contain >2 sub-MSAs).
    a3mB : dict
        2nd MSA to be joined, with keys `msa`, `ins`, `taxid`, and optionally
        `is_paired`. Can be a multi-MSA ONLY if not overlapping with 1st MSA.
    idx_overlap : tuple or list (optional)
        Start and end indices of overlap region in 1st MSA, followed by the
        same in 2nd MSA.

    Returns
    -------
    a3m : dict
        Paired MSA, with keys `msa`, `ins`, `taxid` and `is_paired`.
    """
    # preprocess overlap region
    L_A, L_B = a3mA['msa'].shape[1], a3mB['msa'].shape[1]
    if idx_overlap is not None:
        i1A, i2A, i1B, i2B = idx_overlap
        i1B_new, i2B_new = (0, i1B) if i2B==L_B else (i2B, L_B) # MSA B residues that don't overlap MSA A
        assert((i1B==0) or (i2B==a3mB['msa'].shape[1])), \
            "When overlapping with 1st MSA, 2nd MSA must comprise at most 2 sub-MSAs "\
            "(i.e. residue range should include 0 or a3mB['msa'].shape[1])"
    else:
        i1B_new, i2B_new = (0, L_B)
        
    # pair sequences
    taxids_shared = a3mA['taxid'][np.isin(a3mA['taxid'],a3mB['taxid'])]
    i_pairedA, i_pairedB = [], []

    for taxid in taxids_shared:
        i_match = np.where(a3mA['taxid']==taxid)[0]
        i_match_best = torch.argmin(torch.sum(a3mA['msa'][i_match]==a3mA['msa'][0], axis=1))
        i_pairedA.append(i_match[i_match_best])

        i_match = np.where(a3mB['taxid']==taxid)[0]
        i_match_best = torch.argmin(torch.sum(a3mB['msa'][i_match]==a3mB['msa'][0], axis=1))
        i_pairedB.append(i_match[i_match_best])

    # unpaired sequences
    i_unpairedA = np.setdiff1d(np.arange(a3mA['msa'].shape[0]), i_pairedA)
    i_unpairedB = np.setdiff1d(np.arange(a3mB['msa'].shape[0]), i_pairedB)
    N_paired, N_unpairedA, N_unpairedB = len(i_pairedA), len(i_unpairedA), len(i_unpairedB)

    # handle overlap region
    # if msa A consists of sub-MSAs 1,2,3 and msa B of 2,4 (i.e overlap region is 2),
    # this diagram shows how the variables below make up the final multi-MSA
    # (* denotes nongaps, - denotes gaps)
    #  1 2 3 4
    # |*|*|*|*|   msa_paired
    # |*|*|*|-|   msaA_unpaired
    # |-|*|-|*|   msaB_unpaired
    if idx_overlap is not None:
        assert((a3mA['msa'][i_pairedA, i1A:i2A]==a3mB['msa'][i_pairedB, i1B:i2B]) |
               (a3mA['msa'][i_pairedA, i1A:i2A]==ChemData().UNKINDEX)).all(),\
            'Paired MSAs should be identical (or 1st MSA should be all gaps) in overlap region'

        # overlap region gets sequences from 2nd MSA bc sometimes 1st MSA will be all gaps here
        msa_paired = torch.cat([a3mA['msa'][i_pairedA, :i1A],
                                a3mB['msa'][i_pairedB, i1B:i2B],
                                a3mA['msa'][i_pairedA, i2A:],
                                a3mB['msa'][i_pairedB, i1B_new:i2B_new] ], dim=1)
        msaA_unpaired = torch.cat([a3mA['msa'][i_unpairedA],
                                 torch.full((N_unpairedA, i2B_new-i1B_new), ChemData().UNKINDEX) ], dim=1)
        msaB_unpaired = torch.cat([torch.full((N_unpairedB, i1A), ChemData().UNKINDEX),
                                 a3mB['msa'][i_unpairedB, i1B:i2B],
                                 torch.full((N_unpairedB, L_A-i2A), ChemData().UNKINDEX),
                                 a3mB['msa'][i_unpairedB, i1B_new:i2B_new] ], dim=1)
    else:
        # no overlap region, simple offset pad & stack
        # this code is actually a special case of "if" block above, but writing
        # this out explicitly here to make the logic more clear
        msa_paired = torch.cat([a3mA['msa'][i_pairedA], a3mB['msa'][i_pairedB, i1B_new:i2B_new]], dim=1)
        msaA_unpaired = torch.cat([a3mA['msa'][i_unpairedA],
                                 torch.full((N_unpairedA, L_B), ChemData().UNKINDEX)], dim=1) # pad with gaps
        msaB_unpaired = torch.cat([torch.full((N_unpairedB, L_A), ChemData().UNKINDEX),
                                 a3mB['msa'][i_unpairedB]], dim=1) # pad with gaps

    # stack paired & unpaired
    msa = torch.cat([msa_paired, msaA_unpaired, msaB_unpaired], dim=0)
    taxids = np.concatenate([a3mA['taxid'][i_pairedA], a3mA['taxid'][i_unpairedA], a3mB['taxid'][i_unpairedB]])

    # label "fully paired" sequences (a row of MSA that was never padded with gaps)
    # output seq is fully paired if seqs A & B both started out as paired and were paired to
    # each other on tax ID. 
    # NOTE: there is a rare edge case that is ignored here for simplicity: if
    # pMSA 0+1 and 1+2 are joined and then joined to 2+3, a seq that exists in
    # 0+1 and 2+3 but NOT 1+2 will become fully paired on the last join but
    # will not be labeled as such here
    is_pairedA = a3mA['is_paired'] if 'is_paired' in a3mA else torch.ones((a3mA['msa'].shape[0],)).bool()
    is_pairedB = a3mB['is_paired'] if 'is_paired' in a3mB else torch.ones((a3mB['msa'].shape[0],)).bool()
    is_paired = torch.cat([is_pairedA[i_pairedA] & is_pairedB[i_pairedB],
                           torch.zeros((N_unpairedA + N_unpairedB,)).bool()])

    # insertion features in paired MSAs are assumed to be zero
    a3m = dict(msa=msa, ins=torch.zeros_like(msa), taxid=taxids, is_paired=is_paired)
    return a3m


def load_minimal_multi_msa(hash_list, taxid_list, Ls, params):
    """Load a multi-MSA, which is a MSA that is paired across more than 2
    chains. This loads the MSA for unique chains. Use 'expand_multi_msa` to
    duplicate portions of the MSA for homo-oligomer repeated chains.

    Given a list of unique MSA hashes, loads all MSAs (using paired MSAs where
    it can) and pairs sequences across as many sub-MSAs as possible by matching
    taxonomic ID. For details on how pairing is done, see
    `join_msas_by_taxid()`

    Parameters
    ----------
    hash_list : list of str 
        Hashes of MSAs to load and join. Must not contain duplicates.
    taxid_list : list of str
        Taxonomic IDs of query sequences of each input MSA.
    Ls : list of int
        Lengths of the chains corresponding to the hashes.

    Returns
    -------
    a3m_out : dict
        Multi-MSA with all input MSAs. Keys: `msa`,`ins` [torch.Tensor (N_seq, L)], 
        `taxid` [np.array (Nseq,)], `is_paired` [torch.Tensor (N_seq,)]
    hashes_out : list of str
        Hashes of MSAs in the order that they are joined in `a3m_out`.
        Contains the same elements as the input `hash_list` but may be in a
        different order.
    Ls_out : list of int
        Lengths of each chain in `a3m_out`
    """
    assert(len(hash_list)==len(set(hash_list))), 'Input MSA hashes must be unique'

    # the lists below are constructed such that `a3m_list[i_a3m]` is a multi-MSA
    # comprising sub-MSAs whose indices in the input lists are 
    # `i_in = idx_list_groups[i_a3m][i_submsa]`, i.e. the sub-MSA hashes are
    # `hash_list[i_in]` and lengths are `Ls[i_in]`.
    # Each sub-MSA spans a region of its multi-MSA `a3m_list[i_a3m][:,i_start:i_end]`, 
    # where `(i_start,i_end) = res_range_groups[i_a3m][i_submsa]`
    a3m_list = []         # list of multi-MSAs
    idx_list_groups = []  # list of lists of indices of input chains making up each multi-MSA
    res_range_groups = [] # list of lists of start and end residues of each sub-MSA in multi-MSA

    # iterate through all pairs of hashes and look for paired MSAs (pMSAs)
    # NOTE: in the below, if pMSAs are loaded for hashes 0+1 and then 2+3, and
    # later a pMSA is found for 0+2, the last MSA will not be loaded. The 0+1
    # and 2+3 pMSAs will still be joined on taxID at the end, but sequences
    # only present in the 0+2 pMSA pMSAs will be missed. this is probably very
    # rare and so is ignored here for simplicity.
    N = len(hash_list)
    for i1, i2 in itertools.permutations(range(N),2):

        idx_list = [x for group in idx_list_groups for x in group] # flattened list of loaded hashes
        if i1 in idx_list and i2 in idx_list: continue # already loaded
        if i1 == '' or i2 == '': continue # no taxID means no pMSA

        # a paired MSA exists
        if taxid_list[i1]==taxid_list[i2]:
            
            h1, h2 = hash_list[i1], hash_list[i2]
            fn = params['COMPL_DIR']+'/pMSA/'+h1[:3]+'/'+h2[:3]+'/'+h1+'_'+h2+'.a3m.gz'

            if os.path.exists(fn):
                msa, ins, taxid = parse_a3m(fn, paired=True)
                a3m_new = dict(msa=torch.tensor(msa), ins=torch.tensor(ins), taxid=taxid,
                               is_paired=torch.ones(msa.shape[0]).bool())
                res_range1 = (0,Ls[i1])
                res_range2 = (Ls[i1],msa.shape[1])

                # both hashes are new, add paired MSA to list
                if i1 not in idx_list and i2 not in idx_list:
                    a3m_list.append(a3m_new)
                    idx_list_groups.append([i1,i2])
                    res_range_groups.append([res_range1, res_range2])

                # one of the hashes is already in a multi-MSA
                # find that multi-MSA and join the new pMSA to it
                elif i1 in idx_list:
                    # which multi-MSA & sub-MSA has the hash with index `i1`?
                    i_a3m = np.where([i1 in group for group in idx_list_groups])[0][0]
                    i_submsa = np.where(np.array(idx_list_groups[i_a3m])==i1)[0][0]
                    
                    idx_overlap = res_range_groups[i_a3m][i_submsa] + res_range1
                    a3m_list[i_a3m] = join_msas_by_taxid(a3m_list[i_a3m], a3m_new, idx_overlap)
                    
                    idx_list_groups[i_a3m].append(i2)
                    L = res_range_groups[i_a3m][-1][1] # length of current multi-MSA
                    L_new = res_range2[1] - res_range2[0]
                    res_range_groups[i_a3m].append((L, L+L_new))

                elif i2 in idx_list:
                    # which multi-MSA & sub-MSA has the hash with index `i2`?
                    i_a3m = np.where([i2 in group for group in idx_list_groups])[0][0]
                    i_submsa = np.where(np.array(idx_list_groups[i_a3m])==i2)[0][0]
                    
                    idx_overlap = res_range_groups[i_a3m][i_submsa] + res_range2
                    a3m_list[i_a3m] = join_msas_by_taxid(a3m_list[i_a3m], a3m_new, idx_overlap)
                    
                    idx_list_groups[i_a3m].append(i1)
                    L = res_range_groups[i_a3m][-1][1] # length of current multi-MSA
                    L_new = res_range1[1] - res_range1[0]
                    res_range_groups[i_a3m].append((L, L+L_new))
                    
    # add unpaired MSAs
    # ungroup hash indices now, since we're done making multi-MSAs
    idx_list = [x for group in idx_list_groups for x in group]
    for i in range(N):
        if i not in idx_list:
            fn = params['PDB_DIR'] + '/a3m/' + hash_list[i][:3] + '/' + hash_list[i] + '.a3m.gz'
            msa, ins, taxid = parse_a3m(fn)
            a3m_new = dict(msa=torch.tensor(msa), ins=torch.tensor(ins), 
                           taxid=taxid, is_paired=torch.ones(msa.shape[0]).bool())
            a3m_list.append(a3m_new)
            idx_list.append(i)
            
    Ls_out = [Ls[i] for i in idx_list]
    hashes_out = [hash_list[i] for i in idx_list]
            
    # join multi-MSAs & unpaired MSAs
    a3m_out = a3m_list[0]
    for i in range(1, len(a3m_list)):
        a3m_out = join_msas_by_taxid(a3m_out, a3m_list[i])

    return a3m_out, hashes_out, Ls_out    


def expand_multi_msa(a3m, hashes_in, hashes_out, Ls_in, Ls_out):
    """Expands a multi-MSA of unique chains into an MSA of a
    hetero-homo-oligomer in which some chains appear more than once. The query
    sequences (1st sequence of MSA) are concatenated directly along the
    residue dimention. The remaining sequences are offset-tiled (i.e. "padded &
    stacked") so that exact repeat sequences aren't paired.

    For example, if the original multi-MSA contains unique chains 1,2,3 but
    the final chain order is 1,2,1,3,3,1, this function will output an MSA like
    (where - denotes a block of gap characters):

        1 2 - 3 - -
        - - 1 - 3 -
        - - - - - 1

    Parameters
    ----------
    a3m : dict
        Contains torch.Tensors `msa` and `ins` (N_seq, L) and np.array `taxid` (Nseq,),
        representing the multi-MSA of unique chains.
    hashes_in : list of str
        Unique MSA hashes used in `a3m`.
    hashes_out : list of str
        Non-unique MSA hashes desired in expanded MSA.
    Ls_in : list of int
        Lengths of each chain in `a3m`
    Ls_out : list of int
        Lengths of each chain desired in expanded MSA.
    params : dict
        Data loading parameters

    Returns
    -------
    a3m : dict
        Contains torch.Tensors `msa` and `ins` of expanded MSA. No
        taxids because no further joining needs to be done.
    """
    assert(len(hashes_out)==len(Ls_out))
    assert(set(hashes_in)==set(hashes_out))
    assert(a3m['msa'].shape[1]==sum(Ls_in))

    # figure out which oligomeric repeat is represented by each hash in `hashes_out`
    # each new repeat will be offset in sequence dimension of final MSA
    counts = dict()
    n_copy = [] # n-th copy of this hash in `hashes`
    for h in hashes_out:
        if h in counts:
            counts[h] += 1
        else:
            counts[h] = 1
        n_copy.append(counts[h])

    # num sequences in source & destination MSAs
    N_in = a3m['msa'].shape[0]
    N_out = (N_in-1)*max(n_copy)+1 # concatenate query seqs, pad&stack the rest

    # source MSA
    msa_in, ins_in = a3m['msa'], a3m['ins']

    # initialize destination MSA to gap characters
    msa_out = torch.full((N_out, sum(Ls_out)), ChemData().UNKINDEX)
    ins_out = torch.full((N_out, sum(Ls_out)), 0)

    # for each destination chain
    for i_out, h_out in enumerate(hashes_out):
        # identify index of source chain
        i_in = np.where(np.array(hashes_in)==h_out)[0][0]

        # residue indexes
        i1_res_in = sum(Ls_in[:i_in])
        i2_res_in = sum(Ls_in[:i_in+1])
        i1_res_out = sum(Ls_out[:i_out])
        i2_res_out = sum(Ls_out[:i_out+1])

        # copy over query sequence
        # NOTE: There is a bug in these next two lines!
        # The second line should be ins_out[0, i1_res_out:i2_res_out] = ins_in[0, i1_res_in:i2_res_in]
        msa_out[0, i1_res_out:i2_res_out] = msa_in[0, i1_res_in:i2_res_in]
        ins_out[0, i1_res_out:i2_res_out] = msa_in[0, i1_res_in:i2_res_in]

        # offset non-query sequences along sequence dimension based on repeat number of a given hash
        i1_seq_out = 1+(n_copy[i_out]-1)*(N_in-1)
        i2_seq_out = 1+n_copy[i_out]*(N_in-1)
        # copy over non-query sequences
        msa_out[i1_seq_out:i2_seq_out, i1_res_out:i2_res_out] = msa_in[1:, i1_res_in:i2_res_in]
        ins_out[i1_seq_out:i2_seq_out, i1_res_out:i2_res_out] = ins_in[1:, i1_res_in:i2_res_in]

    # only 1st oligomeric repeat can be fully paired
    is_paired_out = torch.cat([a3m['is_paired'], torch.zeros((N_out-N_in,)).bool()]) 

    a3m_out = dict(msa=msa_out, ins=ins_out, is_paired=is_paired_out)
    a3m_out = remove_all_gap_seqs(a3m_out)

    return a3m_out

def load_multi_msa(chain_ids, Ls, chid2hash, chid2taxid, params):
    """Loads multi-MSA for an arbitrary number of protein chains. Tries to
    locate paired MSAs and pair sequences across all chains by taxonomic ID.
    Unpaired sequences are padded and stacked on the bottom.
    """
    # get MSA hashes (used to locate a3m files) and taxonomic IDs (used to determine pairing)
    hashes = []
    hashes_unique = []
    taxids_unique = []
    Ls_unique = []
    for chid,L_ in zip(chain_ids, Ls):
        hashes.append(chid2hash[chid])
        if chid2hash[chid] not in hashes_unique:
            hashes_unique.append(chid2hash[chid])
            taxids_unique.append(chid2taxid.get(chid))
            Ls_unique.append(L_)

    # loads multi-MSA for unique chains
    a3m_prot, hashes_unique, Ls_unique = \
        load_minimal_multi_msa(hashes_unique, taxids_unique, Ls_unique, params)

    # expands multi-MSA to repeat chains of homo-oligomers
    a3m_prot = expand_multi_msa(a3m_prot, hashes_unique, hashes, Ls_unique, Ls, params)

    return a3m_prot

def choose_multimsa_clusters(msa_seq_is_paired, params):
    """Returns indices of fully-paired sequences in a multi-MSA to use as seed
    clusters during MSA featurization.
    """
    frac_paired = msa_seq_is_paired.float().mean()
    if frac_paired > 0.25: # enough fully paired sequences, just let MSAFeaturize choose randomly
        return None
    else:
        # ensure that half of the clusters are fully-paired sequences,
        # and let the rest be chosen randomly
        N_seed = params['MAXLAT']//2
        msa_seed_clus = []
        for i_cycle in range(params['MAXCYCLE']):
            idx_paired = torch.where(msa_seq_is_paired)[0]
            msa_seed_clus.append(idx_paired[torch.randperm(len(idx_paired))][:N_seed])
        return msa_seed_clus


#fd 
def get_bond_distances(bond_feats):
    atom_bonds = (bond_feats > 0)*(bond_feats<5)
    dist_matrix = scipy.sparse.csgraph.shortest_path(atom_bonds.long().numpy(), directed=False)
    # dist_matrix = torch.tensor(np.nan_to_num(dist_matrix, posinf=4.0)) # protein portion is inf and you don't want to mask it out
    return torch.from_numpy(dist_matrix).float()


def get_pdb(pdbfilename, plddtfilename, item, lddtcut, sccut):
    xyz, mask, res_idx = parse_pdb(pdbfilename)
    plddt = np.load(plddtfilename)
    
    # update mask info with plddt (ignore sidechains if plddt < 90.0)
    mask_lddt = np.full_like(mask, False)
    mask_lddt[plddt > sccut] = True
    mask_lddt[:,:5] = True
    mask = np.logical_and(mask, mask_lddt)
    mask = np.logical_and(mask, (plddt > lddtcut)[:,None])
    
    return {'xyz':torch.tensor(xyz), 'mask':torch.tensor(mask), 'idx': torch.tensor(res_idx), 'label':item}

def get_msa(a3mfilename, item, maxseq=5000):
    msa,ins, taxIDs = parse_a3m(a3mfilename, maxseq=5000)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins), 'taxIDs':taxIDs, 'label':item}
