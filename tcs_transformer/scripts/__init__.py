"""Training Scripts for TCS-Transformer Models"""

# Import from subdirectories
from . import tcs_scripts
from . import transformer_scripts

# Import hybrid training script (top-level in scripts)
from .train_hybrid import (
    train_hybrid_model,
    load_hybrid_checkpoint,
    generate_text_hybrid,
    load_tcs_vae,
    setup_hybrid_model,
    setup_optimizer,
    train_step,
    evaluate_step,
    format_log_message,
    save_checkpoint,
    run_training_loop,
)

__all__ = [
    "tcs_scripts",
    "transformer_scripts",
    "train_hybrid_model",
    "load_hybrid_checkpoint",
    "generate_text_hybrid",
    "load_tcs_vae",
    "setup_hybrid_model",
    "setup_optimizer",
    "train_step",
    "evaluate_step",
    "format_log_message",
    "save_checkpoint",
    "run_training_loop",
]
