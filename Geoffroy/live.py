import cv2
from src.Connection_ai import *
video_capture = cv2.VideoCapture(0)
result, video_frame = video_capture.read()  # Lis les images de la caméra
depth_image = analyze_image(video_frame)
cv2.imshow(
        "My Face Detection Project", depth_image
)
while cv2.getWindowProperty('My Face Detection Project', cv2.WND_PROP_VISIBLE) >= 1:
    result, video_frame = video_capture.read()  # Lis les images de la caméra
    depth_image = analyze_image(video_frame)
    
    cv2.imshow(
        "My Face Detection Project", depth_image
    )
    #fps.update()
    #afficher_image_3d(video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
#fps.stop()

#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
video_capture.release()
cv2.destroyAllWindows()
