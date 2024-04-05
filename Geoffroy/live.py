import cv2
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from imutils.video import FPS
import datetime
fps = FPS().start()

def image_to_depth_image(raw_image):
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
    h, w = image.shape[:2]
    
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        depth = depth_anything(image)
    
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    
    depth = depth.cpu().numpy().astype(np.uint8)
    
    #depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    return depth

# Fonction pour afficher une image en 3D
def afficher_image_3d(image):
    # Normaliser les valeurs de couleur dans la plage 0-1
    image_normalized = image.astype(float) / 255

    # Réduire la taille de l'image
    reduced_image = image_normalized[::4, ::4]  # Réduction de moitié de la résolution en largeur et en hauteur

    # Créer une grille 3D basée sur les dimensions de l'image
    x, y = np.meshgrid(np.arange(reduced_image.shape[1]), np.arange(reduced_image.shape[0]))
    z = np.zeros_like(x)

    # Afficher l'image
    ax.plot_surface(x, y, z, facecolors=reduced_image, rstride=1, cstride=1)

    # Configurer les limites des axes
    ax.set_xlim(0, reduced_image.shape[1])
    ax.set_ylim(0, reduced_image.shape[0])
    ax.set_zlim(0, 1)  # Plage arbitraire pour z

    # Afficher le résultat
    plt.show()

video_capture = cv2.VideoCapture(0)

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vits')).to(DEVICE).eval()

# Créer la figure 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
result, video_frame = video_capture.read()  # Lis les images de la caméra
depth_image = image_to_depth_image(video_frame)
cv2.imshow(
        "My Face Detection Project", depth_image
)
while cv2.getWindowProperty('My Face Detection Project', cv2.WND_PROP_VISIBLE) >= 1:
    result, video_frame = video_capture.read()  # Lis les images de la caméra
    depth_image = image_to_depth_image(video_frame)
    
    cv2.imshow(
        "My Face Detection Project", depth_image
    )
    fps.update()
    #afficher_image_3d(video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
fps.stop()

print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
video_capture.release()
cv2.destroyAllWindows()
