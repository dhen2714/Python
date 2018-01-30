"""
Functions for processing images taken by Leopard camera.
"""
import numpy as np
import cv2
import glob
import os
import re
from checkerboard_motion import pixel_reprojections
import pandas as pd
import matplotlib.pyplot as plt

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
    imageNames = []

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

        if (ret1 and ret2): # Process only if corners found for both views
            corners1 = cv2.cornerSubPix(grey_left, corners1, (5,5), (-1,-1),
                                        criteria)
            corners2 = cv2.cornerSubPix(grey_right, corners2, (5,5), (-1,-1),
                                        criteria)
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            imageNames.append(os.path.splitext(os.path.split(image)[1])[0])

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


    print("Mean reprojection error for left camera (no stereo cal): ", ret1)
    print("Mean reprojection error for right camera (no stero cal): ", ret2)

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

    # Show reprojection errors for each image.
    rms_left = np.zeros(len(imageNames))
    rms_right = np.zeros(len(imageNames))
    for image, i in zip(imageNames, range(len(imageNames))):
        rM1, _ = cv2.Rodrigues(rv1[i]) # Convert rot vec to matrix
        rv2_new = np.dot(R, rM1)
        rv2_new, _ = cv2.Rodrigues(rv2_new)
        tv2_new = T + np.dot(R, tv1[i])
        left_reproj, _ = cv2.projectPoints(
            objpoints[i], rv1[i], tv1[i], K1, dc1)
        right_reproj, _ = cv2.projectPoints(
            objpoints[i], rv2_new, tv2_new, K2, dc2)
        errs_left = np.squeeze(imgpoints_left[i] - left_reproj)
        errs_right = np.squeeze(imgpoints_right[i] - right_reproj)
        rms_left[i] = np.mean(np.sqrt((
            errs_left[:,0]**2 + errs_left[:,1]**2)))
        rms_right[i] = np.mean(np.sqrt((
            errs_right[:,0]**2 + errs_right[:,1]**2)))

    df = pd.DataFrame(np.array([rms_left, rms_right]).T,
                      columns = ['Left', 'Right'],
                      index = imageNames)
    df.plot.bar()
    plt.ylabel('Mean reprojection error (pixels)')
    plt.title('Stereo calibration mean reprojection error')
    plt.show()

    np.savez('Objpoints.npz', objpoints = objpoints,
        left = imgpoints_left, right = imgpoints_right)
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

def correct_dist2(vec, fc, c ,k, p):
    """
    Corrects distortion using Brown's (or Brown-Conrady) distortion model.
    This is the same method used by OpenCV.
    """
    # May be 2 or 3 radial coefficients depending on calibration method
    if len(k) == 3:
        k3 = k[2]
    else:
        k3 = 0

    ud = vec[:,0]
    vd = vec[:,1]
    xn = (ud - c[0])/fc[0] # Normalise points
    yn = (vd - c[1])/fc[1]
    r2 = xn*xn + yn*yn
    r4 = r2*r2
    r6 = r4*r2

    k_radial = 1 + k[0]*r2 + k[1]*r4 + k3*r6
    x = xn*k_radial + 2*p[0]*xn*yn + p[1]*(r2 + 2*xn*xn)
    y = yn*k_radial + p[0]*(r2 + 2*yn*yn) + 2*p[1]*xn*yn

    x = fc[0]*x + c[0] # Convert back to pix coords
    y = fc[1]*y + c[1]
    return np.array([x,y]).T

def correct_dist(vec,fc,c,k,p):
    """
    From Andre's IDL code.
    Inputs:
        vec - Nx2 array of distorted pixel coordinates.
        fc  - 2 element focal length.
        c   - principal point.
        k   - 2 (or 3) radial distortion coefficients.
        p   - 2 tangential distortion coefficients.
    Outputs:
        cvec - Nx2 array of distortion-corrected pixel coordinates.
    """
    # May be 2 or 3 radial coefficients depending on calibration method
    if len(k) == 3:
        k3 = k[2]
    else:
        k3 = 0

    ud = vec[:,0]
    vd = vec[:,1]
    xn = (ud - c[0])/fc[0] # Normalise points
    yn = (vd - c[1])/fc[1]
    x = xn
    y = yn

    for i in range(20):
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        k_radial = 1 + k[0]*r2 + k[1]*r4 + k3*r6
        delta_x = 2*p[0]*x*y + p[1]*(r2 + 2*x*x)
        delta_y = 2*p[1]*x*y + p[0]*(r2 + 2*y*y)
        x = (xn - delta_x)/k_radial
        y = (yn - delta_y)/k_radial

    x = fc[0]*x + c[0] # Undo normalisation
    y = fc[1]*y + c[1]
    cvec = np.array([x,y]).T
    return cvec

def get_checkerboard_points3d(directory, check_rows, check_cols,
    P1, P2, fc1, fc2, dc1, dc2, pp1, pp2, ext = '.pgm'):
    """
    Processes images in directory, returning the average 3d position of
    checkerboard points
    """
    # Termination criteria for getting sub-pixel corner positions.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Prepare corner points, e.g. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((check_rows*check_cols,3), np.float32)
    # Variable 'length' is size of checkerboard square.
    objp[:,:2] = np.mgrid[0:check_rows,0:check_cols].T.reshape(-1,2)
    objpoints = []
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    for f in os.scandir(directory):
        if os.path.splitext(f.name)[1] == ext:
            img = cv2.imread(os.path.join(directory, f.name))
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

            if (ret1 and ret2):
                corners1 = cv2.cornerSubPix(grey_left, corners1, (5,5),
                                            (-1,-1),
                                            criteria)
                corners2 = cv2.cornerSubPix(grey_right, corners2, (5,5),
                                            (-1,-1),
                                            criteria)
                objpoints.append(objp)
                imgpoints_left.append(corners1)
                imgpoints_right.append(corners2)

    n_images = len(objpoints)
    n_points = check_rows*check_cols
    X = np.ones((n_points, 4, n_images))
    for i in range(n_images):
        left = correct_dist(np.squeeze(imgpoints_left[i]), fc1, pp1,
            dc1[[0,1,4]], dc1[[2,3]])
        right = correct_dist(np.squeeze(imgpoints_right[i]), fc2, pp2,
            dc2[[0,1,4]], dc2[[2,3]])
        # left = np.squeeze(imgpoints_left[i])
        # right = np.squeeze(imgpoints_right[i])
        points3d = cv2.triangulatePoints(P1, P2, left.T, right.T)
        X[:,:,i] = np.apply_along_axis(lambda v: v/v[-1],0, points3d).T

    cv2.drawChessboardCorners(img_left,
                              (check_rows, check_cols),
                              corners1,
                              ret1)
    cv2.drawChessboardCorners(img_right,
                              (check_rows, check_cols),
                              corners2,
                              ret2)
    cv2.imwrite("{}.jpg".format(directory),
        np.hstack((img_left, img_right)))

    X = np.mean(X, axis = 2)
    left_reproj = np.dot(P1, X.T)
    right_reproj = np.dot(P2, X.T)
    lr = np.apply_along_axis(lambda v: v/v[-1], 0, left_reproj).T
    rr = np.apply_along_axis(lambda v: v/v[-1], 0, right_reproj).T
    errs_left = np.abs(np.squeeze(imgpoints_left[i]) - lr[:,:2])
    errs_right = np.abs(np.squeeze(imgpoints_right[i]) - rr[:,:2])
    print("MEAN ERRS L: ", np.mean(errs_left, axis = 0))
    print("MEAN ERRS R: ", np.mean(errs_right, axis = 0))
    return X

if __name__ == "__main__":
    directory = r"C:/Users/dhen2714/OneDrive - The University of Sydney (Students)/Phd/Experiments/VolunteerForeheadTracking/Tests/PreSetup/20171220_Calibration_intrinsic/Images"

    raw_2_pgm(directory, 480, 1280, stereo = True)
