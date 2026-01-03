import inspect
import os

import torch
from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    orig_torch_load = torch.load
    use_weights_only = "weights_only" in inspect.signature(torch.load).parameters
    if use_weights_only:
        # Trusted local model file; override torch.load default for fairseq.
        def torch_load_no_weights_only(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_torch_load(*args, **kwargs)

        torch.load = torch_load_no_weights_only
    try:
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["assets/hubert/hubert_base.pt"],
            suffix="",
        )
    finally:
        if use_weights_only:
            torch.load = orig_torch_load
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
