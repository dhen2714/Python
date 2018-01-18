"""
Functions for processing images taken by Leopard camera.
"""
import numpy as np
import cv2
import glob
import os
import re
from checkerboard_motion import pixel_reprojections

def split_images(path):
    """
    1) Get images in path.
    2) Open, split.
    3) Save split images in new folder, with names left01, right01, etc.
    """
    searchName = os.path.join(path, "*.pgm")
    fileNames = glob.glob(searchName)
    n = len(fileNames)

    for (img, i) in zip(fileNames, range(n)):
        img = cv2.imread(img, 0)
        left = img[:,:640]
        right = img[:,640:]
        cv2.imwrite("left{}.pgm".format(i), left)
        cv2.imwrite("right{}.pgm".format(i), right)
    return

def raw_greyscale_array(fileName, rows, cols, bit_depth = np.uint8):
    """
    Reads .raw greyscale image and outputs image as numpy array.
    """
    with open(fileName, 'rb') as f:
        img = np.fromfile(f, dtype = bit_depth, count = rows*cols)
    return img.reshape((rows, cols))

def raw_2_pgm(directory, rows, cols, bit_depth = np.uint8, stereo = False):
    """
    Find all .raw images in given directory, and saves them as .pgm files in a
    new directory "PGM".
    """
    raw_imgs = glob.glob(os.path.join(directory, "*.raw"))

    for raw_img in raw_imgs:
        print("Processing: ", raw_img)

        path, fileName = os.path.split(raw_img)
        newPath = os.path.join(path, "PGM")

        if not os.path.isdir(newPath):
            os.mkdir(newPath)
            print("New directory created at: ", newPath)

        items = re.match("([a-z\-]+)([0-9]+).([a-z]+)", fileName).groups()
        img = raw_greyscale_array(raw_img, rows, cols, bit_depth)

        if stereo:
            fName1 = os.path.join(newPath,"left{}.pgm".format(items[1]))
            fName2 = os.path.join(newPath,"right{}.pgm".format(items[1]))
            cv2.imwrite(fName1, img[:,int(cols/2):])
            cv2.imwrite(fName2, img[:,:int(cols/2)])
        else:
            fName = os.path.join(newPath,"{0}{1}.pgm".format(items[0],items[1]))
            cv2.imwrite(fName, img)
    return

def calibrate_leopard_stereo(directory, check_rows, check_cols, length,
    pixel_format = "pgm", showCorners = True):
    """
    Calibrates leoprad stereo cameras.
    Outputs results into .npz files.
    """
    # Termination criteria for getting sub-pixel corner positions.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare corner points, e.g. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((check_rows*check_cols,3), np.float32)
    objp[:,:2] = length*np.mgrid[0:check_rows,0:check_cols].T.reshape(-1,2)

    # Arrays to store object points and image points from all images.
    objpoints = [] # 3d points in real world space.
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    images = glob.glob(os.path.join(directory, "*.{}".format(pixel_format)))

    if not images:
        print("No images found in {}".format(directory))
        return

    for image in images:
        print("Processing ... ", image)
        img = cv2.imread(image)
        img_left = img[:,640:]
        img_right = img[:,:640]
        grey_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        grey_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Find corners.
        # cv2.findChessboardCorners(img, patternsize(rows, columns))
        ret1, corners1 = cv2.findChessboardCorners(grey_left,
                                                   (check_rows, check_cols),
                                                   None)
        ret2, corners2 = cv2.findChessboardCorners(grey_right,
                                                   (check_rows, check_cols),
                                                   None)

        if ret1 and ret2:
            corners1 = cv2.cornerSubPix(grey_left, corners1, (5,5), (-1,-1),
                                        criteria)
            corners2 = cv2.cornerSubPix(grey_right, corners2, (5,5), (-1,-1),
                                        criteria)
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

            if showCorners:
                cv2.drawChessboardCorners(img_left,
                                          (check_rows, check_cols),
                                          corners1,
                                          ret1)
                cv2.drawChessboardCorners(img_right,
                                          (check_rows, check_cols),
                                          corners2,
                                          ret2)
                cv2.imshow("{}".format(image),
                    np.hstack((img_left, img_right)))
                cv2.waitKey(0)
        else:
            print("Could not find corners in both views of".format(image))


    input("All images processed. Press ENTER to continue.")
    cv2.destroyAllWindows()
    print("Calibrating...")

    ret1, K1, dc1, rv1, tv1 = cv2.calibrateCamera(objpoints, imgpoints_left,
                                                 img_left.shape[:2],None,None)

    ret2, K2, dc2, rv2, tv2 = cv2.calibrateCamera(objpoints, imgpoints_right,
                                                 img_left.shape[:2],None,None)


    print("Mean reprojection error for left camera: ", ret1)
    print("Mean reprojection error for right camera: ", ret2)

    left_reproj, left_pix = pixel_reprojections(check_rows, check_cols,
                objpoints, imgpoints_left, K1, rv1, tv1, dc1)
    right_reproj, right_pix = pixel_reprojections(check_rows, check_cols,
                objpoints, imgpoints_right, K2, rv2, tv2, dc2)

    retval, K1, dc1, K2, dc2, R, T, E, F = cv2.stereoCalibrate(
                                        objpoints,
                                        imgpoints_left,
                                        imgpoints_right,
                                        K1, dc1,
                                        K2, dc2,
                                        img_left.shape[:2],
                                        cv2.CALIB_FIX_INTRINSIC)

    print("Final mean reprojection error: ", retval)
    np.savez("Stereo_calibration",
             retval = retval,
             K1 = K1, dc1 = dc1, K2 = K2, dc2 = dc2, R = R, T = T,
             Ematrix = E, Fmatrix = F,
             Left_pix_coords = left_pix, Right_pix_coords = right_pix,
             Left_pix_reproj = left_reproj, Right_pix_reproj = right_reproj)

    np.savez("Left_calibration", ret = ret1, K = K1, dc = dc1,
             rvecs = rv1, tvecs = tv1,
             Checkerboard_coords = left_pix,
             Checkerboard_reprojected = left_reproj)

    np.savez("Right_calibration", ret = ret2, K = K2, dc = dc2,
             rvecs = rv2, tvecs = tv2,
             Checkerboard_coords = right_pix,
             Checkerboard_reprojected = right_reproj)

    print("Calibration finished, results saved.")
    return

if __name__ == "__main__":
    directory = r"C:/Users/dhen2714/OneDrive - The University of Sydney (Students)/Phd/Experiments/VolunteerForeheadTracking/Tests/PreSetup/20171220_Calibration_intrinsic/Images"

    raw_2_pgm(directory, 480, 1280, stereo = True)
