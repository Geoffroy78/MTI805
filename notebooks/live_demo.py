import torch
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import numpy as np
from transformers import pipeline
from PIL import Image
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import datetime

# ZoeD_N
conf = get_config("zoedepth", "eval", 'nyu')

conf.pretrained_resource = "local::./checkpoints/depth_anything_metric_depth_indoor.pt"
model_zoe_n = build_model(conf)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


yolo_segmentation = YOLO('yolov8x-seg.pt')


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not ret:
        break    
    depth = zoe.infer_pil(Image.fromarray(img))  
    results = yolo_segmentation(source=img.copy())
    for result in results:
        masks = result.masks
        classes = [result.names[id] for id in result.boxes.cls.tolist()]
        if masks is None:
            continue
        for c, mask_data in zip(classes, masks):
            mask = np.zeros(img.shape, np.uint8)
            contour = mask_data.xy.pop().astype(np.int32)
            contour = contour.reshape(-1, 1, 2)

            _ = cv2.drawContours(mask, [contour], -1, (255,255,255), cv2.FILLED)
            mean_depth = np.median(depth[mask[:,:,0] > 0])
            img = cv2.addWeighted(img, 1, mask, 0.5, 0)
            text = f'Depth: {mean_depth:.2f}'
            cv2.putText(img, text, (contour[0][0][0], contour[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)
    plt.clf()  
    plt.imshow(img)
    plt.title(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    plt.pause(0.0000000001)
    plt.draw()
   