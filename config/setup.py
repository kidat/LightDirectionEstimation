import torch
from adet.config import get_cfg
from pathlib import Path

def setup_cfg(args):
    """Configure Detectron2 settings"""
    cfg = get_cfg()
    cfg.VERSION = 2
    
    if not Path(args.config_file).exists():
        raise FileNotFoundError(f"Config file {args.config_file} not found!")
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.clone()