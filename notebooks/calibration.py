import numpy as np
import cv2 as cv

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
pattern_size = (9, 6)
amount_of_images = 50

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

calibration_pattern_points = []
image_projections_calibration_pattern_points = [] 

camera = cv.VideoCapture('/dev/video4')
camera.set(cv.CAP_PROP_BUFFERSIZE, 1)
ret, frame = camera.read()
i = 1
while amount_of_images > 0:
    ret, frame = camera.read()
    if not ret:
        print("Camera frame not read")
        continue
    img = frame
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

    if ret:
        calibration_pattern_points.append(objp)

        corners_refined = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        image_projections_calibration_pattern_points.append(corners_refined)

        cv.drawChessboardCorners(img, pattern_size, corners_refined, ret)
        cv.destroyAllWindows()
        cv.imshow(f'img {i}', img)
        cv.waitKey(500)
        amount_of_images -= 1
        i += 1
    else:
        print('Chessboard not found')


ret, camera_matrix, dist_coeff, rotation_vectors, translation_vectors = cv.calibrateCamera(calibration_pattern_points, image_projections_calibration_pattern_points, gray.shape[::-1], None, None)

if ret:
    print('Camera calibrated')
    # Save the camera calibration result
    np.savez('./assets/camera_calibration.npz', camera_matrix=camera_matrix, dist_coeff=dist_coeff, rotation_vectors=rotation_vectors, translation_vectors=translation_vectors)