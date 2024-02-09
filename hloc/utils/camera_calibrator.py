import cv2
import numpy as np
import glob
from pathlib import Path

def calibrate_camera(image_dir, rows, cols, square_edge_length):
    """
    Calibrate a camera using chessboard images.

    Parameters:
    - image_dir: Path to the directory containing chessboard images.
    - rows, cols: Number of corners along the chessboard's rows and columns.
    - square_edge_length: Length of the edge of a chessboard square in millimeters.

    Returns:
    - focal_length: The focal length of the camera.
    - principal_point: The principal point of the camera.
    - radial_distortion: One coefficient for the polynomial radial distortion model.
    """

    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * square_edge_length

    object_points = []  # 3D points in real world space
    image_points = []  # 2D points in image plane

    images = glob.glob(str(Path(image_dir) / '*.jpg'))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        # If found, add object points, image points
        if ret:
            object_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners2)

    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    print("Error in projection : \n", ret) 
    print("\nCamera matrix : \n", mtx) 
    print("\nDistortion coefficients : \n", dist) 

    focal_length = (mtx[0, 0], mtx[1, 1])  # fx, fy
    principal_point = (mtx[0, 2], mtx[1, 2])  # cx, cy
    #radial_distortion = dist[0][0]   Assuming one coefficient for radial distortion

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
