import os
import sys
import torch
from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
from transformers import pipeline
from PIL import Image

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# ZoeD_N
conf = get_config("zoedepth", "eval", 'nyu')

conf.pretrained_resource = "local::./checkpoints/depth_anything_metric_depth_indoor.pt"
model_zoe_n = build_model(conf)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


yolo_segmentation = YOLO('yolov8x-seg.pt')

depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

def depth_estimation(img):
    depth = zoe.infer_pil(Image.fromarray(img))