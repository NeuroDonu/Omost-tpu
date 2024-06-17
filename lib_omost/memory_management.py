import torch
from contextlib import contextmanager
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

high_vram = False
#gpu = torch.device('cuda')
cpu = torch.device('cpu')
xla = torch.device('xla')
#torch.zeros((1, 1)).to(gpu, torch.float32)
#torch.cuda.empty_cache()

torch.zeros((1, 1)).to(xla.device(), torch.float32)
models_in_xla = []


@contextmanager
def movable_bnb_model(m):
    if hasattr(m, 'quantization_method'):
        m.quantization_method_backup = m.quantization_method
        del m.quantization_method
    try:
        yield None
    finally:
        if hasattr(m, 'quantization_method_backup'):
            m.quantization_method = m.quantization_method_backup
            del m.quantization_method_backup
    return


def load_models_to_xla(models):
    global models_in_xla

    if not isinstance(models, (tuple, list)):
        models = [models]

    models_to_remain = [m for m in set(models) if m in models_in_xla]
    models_to_load = [m for m in set(models) if m not in models_in_xla]
    models_to_unload = [m for m in set(models_in_xla) if m not in models_to_remain]

    if not high_vram:
        for m in models_to_unload:
            with movable_bnb_model(m):
                m.to(cpu)
            print('Unload to CPU:', m.__class__.__name__)
        models_in_xla = models_to_remain

    for m in models_to_load:
        with movable_bnb_model(m):
            m.to(xla)
        print('Load to TPU:', m.__class__.__name__)

    models_in_xla = list(set(models_in_xla + models))
    torch.cuda.empty_cache()
    return


def unload_all_models(extra_models=None):
    global models_in_xla

    if extra_models is None:
        extra_models = []

    if not isinstance(extra_models, (tuple, list)):
        extra_models = [extra_models]

    models_in_xla = list(set(models_in_xla + extra_models))

    return load_models_to_xla([])
