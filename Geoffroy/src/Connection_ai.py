import numpy as np
from transformers import pipeline
from ultralytics import YOLO
import time
from PIL import Image
import cv2

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
# depth_anything_vitl14, depth-anything-large-hf
depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf")
# model_depth = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")

def image(path):
    print(path)
    try:
        img = cv2.imread(path)
    except Exception as err:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img

def save_image(image, save_path):
    cv2.imwrite(save_path, image)

def analyze_image(frame):
    start = time.perf_counter()
    results = model(frame)
    end = time.perf_counter()
    total_time =end - start
    fps = 1/total_time
    #print(type(results[0].boxes), results[0].boxes)
    # Extraction des résultats
    annotated_frame = frame
    dictionnaire = results[0].__dict__
    boxes = results[0].boxes.cpu().numpy()
    masks = results[0].masks
    classes = [results[0].names[id] for id in results[0].boxes.cls.tolist()]

    # Traitement des masques
    if masks:
        resulting_masks = []
        for c, mask_data in zip(classes, masks):
            if mask_data is not None:
                contour = mask_data.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                mask = cv2.drawContours(np.zeros(annotated_frame.shape[:2], np.uint8), [contour], -1, (255,255,255), cv2.FILLED)
                resulting_masks.append((c, mask))
        
        depth = np.array(depth_estimator(Image.fromarray(frame))["depth"])
        list_mean = []

        for i, (c, mask) in enumerate(resulting_masks):
            mean_depth = depth[mask > 0].mean()
            list_mean.append(mean_depth)
        
        # Extraction des coordonnées des boîtes englobantes
        xyxys = boxes.xyxy
        
        # Dessin des boîtes englobantes et des étiquettes
        for xyxy, c, mean_depth in zip(xyxys, classes, list_mean):
            cv2.rectangle(annotated_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'{c}, depth: {mean_depth:.2f}', (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated_frame

def analyze_video(path, save_path):
    cap = cv2.VideoCapture(path)

    # Définir les propriétés de la vidéo en fonction du premier frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Définir le codec et le format pour l'enregistrement
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    # Traitez chaque frame de la vidéo
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Traitez le frame (par exemple, effectuez une analyse)
        result_frame = analyze_image(frame)
        out.write(result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     # Libérez les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
