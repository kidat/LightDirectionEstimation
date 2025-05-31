import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'SSIS')))

from demo.predictor import VisualizationDemo
from moge.model.v1 import MoGeModel
import torch

def load_ssis(cfg):
    return VisualizationDemo(cfg)

def load_moge(device):
    return MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()