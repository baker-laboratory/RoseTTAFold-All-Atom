import torch
import torch.nn as nn

import numpy as np

from contextlib import ExitStack

from rf2aa.chemical import ChemicalData as ChemData

def recycle_step_legacy(ddp_model, input, n_cycle, use_amp, nograds=False, force_device=None):
    if force_device is not None:
        gpu = force_device
    else:
        gpu = ddp_model.device

    xyz_prev, alpha_prev, mask_recycle = \
        input["xyz_prev"], input["alpha_prev"], input["mask_recycle"]
    output_i = (None, None, xyz_prev, alpha_prev, mask_recycle)
    for i_cycle in range(n_cycle):
        with ExitStack() as stack:
            stack.enter_context(torch.cuda.amp.autocast(enabled=use_amp))
            if i_cycle < n_cycle -1 or nograds is True:
                stack.enter_context(torch.no_grad())
                if force_device is None:
                    stack.enter_context(ddp_model.no_sync())
            return_raw = (i_cycle < n_cycle -1)
            use_checkpoint = not nograds and (i_cycle == n_cycle -1)

            input_i = add_recycle_inputs(input, output_i, i_cycle, gpu, return_raw=return_raw, use_checkpoint=use_checkpoint)
            output_i = ddp_model(**input_i)
    return output_i


def run_model_forward_legacy(model, network_input, device="cpu"):
    """ run model forward pass, no recycling or ddp with legacy model (for tests)"""
    gpu = device
    xyz_prev, alpha_prev, mask_recycle = \
        network_input["xyz_prev"], network_input["alpha_prev"], network_input["mask_recycle"]
    output_i = (None, None, xyz_prev, alpha_prev, mask_recycle)
    input_i = add_recycle_inputs(network_input, output_i, 0, gpu, return_raw=False, use_checkpoint=False)
    input_i["seq_unmasked"] = input_i["seq_unmasked"].to(gpu)
    input_i["sctors"] = input_i["sctors"].to(gpu)
    model.eval()
    with torch.no_grad():
        output_i = model(**input_i)

    return output_i

def add_recycle_inputs(network_input, output_i, i_cycle, gpu, return_raw=False, use_checkpoint=False):
    input_i = {}
    for key in network_input:
        if key in ['msa_latent', 'msa_full', 'seq']:
            input_i[key] = network_input[key][:,i_cycle].to(gpu, non_blocking=True)
        else:
            input_i[key] = network_input[key]

    L = input_i["msa_latent"].shape[2]
    msa_prev, pair_prev, _, alpha, mask_recycle = output_i
    xyz_prev = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,L,1,1).to(gpu, non_blocking=True)

    input_i['msa_prev'] = msa_prev
    input_i['pair_prev'] = pair_prev
    input_i['xyz'] = xyz_prev
    input_i['mask_recycle'] = mask_recycle
    input_i['sctors'] = alpha
    input_i['return_raw'] = return_raw
    input_i['use_checkpoint'] = use_checkpoint

    input_i.pop('xyz_prev')
    input_i.pop('alpha_prev')
    return input_i
