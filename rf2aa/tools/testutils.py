import functools
import hydra
from hydra import compose, initialize, core

def hydra_sandbox(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        hydra_instance = None
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra_instance = hydra.core.global_hydra.GlobalHydra().instance()
        hydra.core.global_hydra.GlobalHydra().clear()
        result = func(*args, **kwargs)
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        if hydra_instance:
            hydra.core.global_hydra.GlobalHydra.set_instance(hydra_instance)
        return result

    return wrapper

@hydra_sandbox
def construct_conf_symtest(overrides=[]):
    core.global_hydra.GlobalHydra().clear()
    initialize(version_base=None, config_path='config/inference', job_name='test_app')
    conf = compose(config_name='sym_test.yaml', overrides=overrides, return_hydra_config=True)
    return conf
