import cv2
import numpy as np
import glob
from pathlib import Path
import matplotlib.pyplot as plt

def calibrate_camera(image_dir, rows, cols, square_edge_length, plot_corners=True):
    """
    Calibrate a camera using chessboard images.

    Parameters:
    - image_dir: Path to the directory containing chessboard images.
    - rows, cols: Number of corners along the chessboard's rows and columns.
    - square_edge_length: Length of the edge of a chessboard square in millimeters.
    - plot_corners: If True, plot images with detected corners.

    Returns:
    - focal_length: The focal length of the camera.
    - principal_point: The principal point of the camera.
    - distortion_coefficients: Distortion coefficients of the camera.
    - image_size: The size of the calibration images (width, height).
    """

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * square_edge_length

    object_points = []
    image_points = []

    # Create directory for images with detected corners
    corners_dir = Path(image_dir.parent) / (f"{image_dir.name}_corners")
    corners_dir.mkdir(exist_ok=True)

    images = glob.glob(str(Path(image_dir) / '*.jpg'))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        width, height = gray.shape

        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        if ret:
            object_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners2)

            if plot_corners:
                cv2.drawChessboardCorners(img, (rows, cols), corners2, ret)
                 # Generate path for the new image with corners
                corner_img_path = corners_dir / Path(fname).name
                cv2.imwrite(str(corner_img_path), img)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    print("Reprojection error: \n", ret) 
    print("\nCamera matrix: \n", mtx) 
    print("\nDistortion coefficients: \n", dist) 

    focal_length = (mtx[0, 0], mtx[1, 1])  # fx, fy
    principal_point = (mtx[0, 2], mtx[1, 2])  # cx, cy

    return focal_length, principal_point, dist, (width, height)

# # Example usage
# image_directory = '/data/chessboard/images'
# rows = 9  # Number of inner corners per row
# cols = 6  # Number of inner corners per column
# square_edge_length = 23.5  # in millimeters

# focal_length, principal_point, radial_distortion = calibrate_camera(image_directory, rows, cols, square_edge_length)
# print(f"Focal Length: {focal_length}")
# print(f"Principal Point: {principal_point}")
# print(f"Radial Distortion: {radial_distortion}")
