import sys, os; sys.path.insert(0, os.getcwd())
import cv2, numpy as np, logging

from camera.astra_camera import create_charuco_board, detect_charuco
board, aruco_dict = create_charuco_board()
print("1. astra_camera.create_charuco_board: OK")

img = np.ones((480, 640, 3), dtype=np.uint8) * 200
ok, cc, ci = detect_charuco(img, board, aruco_dict)
print("2. detect_charuco(blank): " + str(ok))

from camera.stereo_calibrate_live import setup_charuco, detect_board
board2, detector = setup_charuco()
cc, ci, mc, mi = detect_board(detector, img)
print("3. stereo_calibrate_live.detect_board: cc=" + str(cc is not None) + " ci=" + str(ci is not None))

from camera.test_depth import test_charuco_detection
logger = logging.getLogger("test")
ret = test_charuco_detection(logger, img)
print("4. test_depth.test_charuco_detection: " + ret["status"])
print("\nAll files updated successfully for OpenCV 4.13!")
