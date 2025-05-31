import sys
import os
import argparse
import glob
import multiprocessing as mp
from pathlib import Path
from processing.image_processor import LightEstimator
from utils.logging_utils import setup_logging

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 + MoGe: Object-Shadow 3D Estimation")
    parser.add_argument(
        "--config-file", 
        default="../SSIS/configs/SSIS/MS_R_101_BiFPN_SSISv2_demo.yaml", 
        metavar="FILE",
        help="Path to config file"
    )
    parser.add_argument(
        "--input", 
        default="./", 
        help="Path to input image or directory"
    )
    parser.add_argument(
        "--output", 
        default="./result/", 
        help="Output directory"
    )
    parser.add_argument(
        "--confidence-threshold", 
        type=float, 
        default=0.1,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--opts", 
        default=[], 
        nargs=argparse.REMAINDER,
        help="Modify config options using command-line"
    )
    return parser

if __name__ == "__main__":
    # Configure environment paths
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../SSIS")))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../MoGe")))
    
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logging()
    
    # Validate paths
    if not Path(args.config_file).exists():
        raise FileNotFoundError(f"Config file {args.config_file} not found!")
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input path {args.input} not found!")
    
    # Initialize and process
    estimator = LightEstimator(args)
    input_paths = []
    
    if os.path.isdir(args.input):
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            input_paths.extend(glob.glob(os.path.join(args.input, ext)))
    else:
        input_paths = [args.input]
    
    for path in tqdm.tqdm(input_paths, desc="Processing Images"):
        estimator.process_image(path)