"""Train TCS model"""

import os
import json
import torch
import numpy as np

from tcs_transformer.utils.parse_config import ConfigParser
from tcs_transformer.core.trainer import Trainer
from tcs_transformer.models import tcs_model as module_arch
from tcs_transformer.utils.common import prepare_device, get_transformed_info


def get_trainer_args(config, train=True):
    """Prepare trainer arguments from configuration"""

    logger = config.get_logger("train")

    # Get configuration parameters
    MIN_PRED_FUNC_FREQ = config["data_loader"]["args"]["min_pred_func_freq"]
    MIN_CONTENT_PRED_FREQ = config["data_loader"]["args"]["min_content_pred_freq"]

    transformed_dir = config["data_loader"]["args"]["transformed_dir"]
    transformed_info_dir = os.path.join(transformed_dir, "info")

    # Load transformed vocabulary info
    pred_func2cnt, content_pred2cnt, pred2ix, content_predarg2ix, pred_func2ix = (
        get_transformed_info(transformed_info_dir)
    )

    sorted_pred_func2ix = sorted(pred_func2ix.items(), key=lambda x: x[1])
    pred_funcs = [pred_func for pred_func, ix in sorted_pred_func2ix]

    print("Initializing encoder ...")
    # Determine embedding count based on encoder type
    if config["encoder_arch"]["type"] == "PASEncoder":
        num_embs = len(content_predarg2ix)
    else:
        num_embs = len(pred2ix)

    encoder = config.init_obj("encoder_arch", module_arch, num_embs=num_embs)
    for p in encoder.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    print("Initializing decoder ...")
    decoder = config.init_obj(
        "decoder_arch", module_arch, num_sem_funcs=len(pred_funcs), train=train
    )

    if not config["decoder_arch"]["args"]["sparse_sem_funcs"]:
        for p in decoder.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Number of encoder parameters: {num_params}")
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Number of decoder parameters: {num_params}")

    # Metrics (empty list - no metrics currently implemented)
    criterion = None
    metric_ftns = []

    trainer_args = {
        "encoder": encoder,
        "pred2ix": pred2ix,
        "pred_func2ix": pred_func2ix,
        "decoder": decoder,
        "criterion": criterion,
        "metric_ftns": metric_ftns,
        "config": config,
    }

    return trainer_args


def train_model(config_path="configs/config.json", seed=42, resume=None):
    """Main training function"""

    # Load configuration
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Update seed if provided
    config_dict["seed"] = seed

    config = ConfigParser(config_dict, resume=resume)

    print("=" * 80)
    print("TCS Model Training")
    print("=" * 80)
    print(f"Configuration: {config_path}")
    print(f"Random seed: {seed}")
    print("=" * 80)

    # Set deterministic behavior
    torch.use_deterministic_algorithms(mode=True)
    torch.autograd.set_detect_anomaly(False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float64)

    # Fix random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare for GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    ddp = config["ddp"]
    world_size = None

    if device_ids == []:
        device = "cpu"
        num_thread = torch.get_num_threads()
        print(f"Using {device} with {num_thread} threads")
        world_size = min(int(num_thread), 2)
    else:
        device = f"cuda:{device_ids[0]}"
        print(f"Using {device} with device_ids: {device_ids}")
        if len(device_ids) > 1:
            ddp = True
            world_size = min(len(device_ids), 12)
        else:
            ddp = False

    # Get trainer args and start training
    trainer_args = get_trainer_args(config)
    trainer = Trainer(
        world_size=world_size, device=device, rank=-1, ddp=ddp, **trainer_args
    )
    trainer.train()

    print("\nTraining completed!")
