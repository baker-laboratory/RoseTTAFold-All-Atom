import torch


def debug_nans(latent_feats):
    for k, v in latent_feats.items():
        if torch.is_tensor(v):
            print('k', k)
            print('sum nans', torch.sum(v.isnan()))

def debug_unused_params(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print('name', name)

def debug_used_params(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print('name', name)

def debug_device(rf_inputs):
    for name, tensor in rf_inputs.items():
        if torch.is_tensor(tensor):
            if not tensor.is_cuda():
                print('name', name)
                print('dev', tensor.device)

def debug_grads(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.norm().item()}")
