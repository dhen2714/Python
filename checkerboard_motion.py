"""
Workflow for getting motion of checkerboard:
    1) Have all images in a directory, with names left01, left02, ...,
    right01, right02, ... etc.
    2) Use get_corners(directory_name,left/right,n_rows,n_cols,length)
    for both left and right camera images.
    3) Input two dictionaries from previous step into triangulate_corners()
    along with camera projection matrices. Returns dictionary with keys : vals
        view : 3D array of points
    4) Input dictionary from previous step into get_motions().
"""

import numpy as np
import cv2
import glob
import motion3d as m3d
import os

def get_corners(directory,prefix,check_rows,check_cols,length,
                showResults=True):
    """

    """
    # Termination criteria for getting sub-pixel corner positions.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare corner points, e.g. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((check_rows*check_cols,3), np.float32)
    # Variable 'length' is size of checkerboard square.
    objp[:,:2] = length*np.mgrid[0:check_rows,0:check_cols].T.reshape(-1,2)

    # Dictionary with key corresponding to view number, value array of coords.
    d = dict()

    images = glob.glob("{}*.jpg".format(directory + "/" + prefix))

    for fname in images:
        print("Processing ... ",fname)
        view = fname.split("\\")[1].split(".")[0]

        img = cv2.imread(fname)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find corners.
        # cv2.findChessboardCorners(img, patternsize(rows, columns))
        ret, corners = cv2.findChessboardCorners(grey,
                                                 (check_rows, check_cols),
                                                 None)

        # If found, store pixel coordinates of corners after refinement.
        if ret:
            cv2.cornerSubPix(grey, corners, (11,11), (-1,-1), criteria)
            d.update({view : np.squeeze(corners)})

            # Draw and display the corners
            if showResults:
                cv2.drawChessboardCorners(img, (check_rows, check_cols), corners,
                                          ret)
                cv2.imshow('{}'.format(view), img)
                cv2.waitKey(1)

    if showResults:
        input("Press ENTER to continue.")
        cv2.destroyAllWindows()
    return d

def triangulate_corners(d1,d2,P1,P2):
    """
    Find 3D positions of checkerboard corners.
    Returns dictionary, with keys corresponding to view number and values being
    Nx3 arrays of 3D checkerboard positions.
    """
    # Assumes filenames are 'left01', 'right01' ...
    nums2 = [i.split('t')[1] for i in d2.keys()]
    dict3D = dict()

    for key, vals in d1.items():
        num = key.split('t')[1]
        if num in nums2:
            x1 = d1[key]
            x2 = d2['right' + num]
            X = cv2.triangulatePoints(P1,P2,x1.T,x2.T)
            X = np.apply_along_axis(lambda v: v/v[-1],0,X)
            # New dictionary keys are 'view01', 'view02', ...
            newKey = "view" + num
            dict3D.update({newKey : X})
    return dict3D

def get_motions(dict3D,reference="view01"):
    """
    Use Horn's method to find the motions for each frame.
    """
    reference_view = dict3D[reference]
    dict_motions = dict()

    for key, vals in dict3D.items():
        X1 = vals
        H = m3d.hornmm(X1,reference_view)
        dict_motions.update({key : H})
    return dict_motions

def calibrate_stereo(directory,check_rows,check_cols,length,
                     left_prefix = "left",
                     right_prefix = "right",
                     showResults=True):
    """
    Calibrate stereo camera setup.
    Outputs results into a .npz file.
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
    os.path.join(directory, "{}*.pgm")
    left_search = os.path.join(directory, "{}*.pgm".format(left_prefix))
    right_search = os.path.join(directory, "{}*.pgm".format(right_prefix))
    left_images = glob.glob(left_search)
    right_images = glob.glob(right_search)
    if not (left_images and right_images):
        print("No images found in", directory)
        quit()

    set_number = 1

    for images in [left_images, right_images]:
        for fname in images:
            print("Processing ... ",fname)
            view = fname.split("\\")[1].split(".")[0]

            img = cv2.imread(fname)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find corners.
            # cv2.findChessboardCorners(img, patternsize(rows, columns))
            ret, corners = cv2.findChessboardCorners(grey,
                                                     (check_rows, check_cols),
                                                     None)

            # If found, store pixel coordinates of corners after refinement.
            if ret:
                cv2.cornerSubPix(grey, corners, (5,5), (-1,-1), criteria)
                if set_number == 1:
                    objpoints.append(objp)
                    imgpoints_left.append(corners)
                elif set_number == 2:
                    imgpoints_right.append(corners)


                # Draw and display the corners
                if showResults:
                    cv2.drawChessboardCorners(img, (check_rows, check_cols),
                                              corners, ret)
                    cv2.imshow('{}'.format(view), img)
                    cv2.waitKey(1)
        set_number += 1

    if showResults:
        input("All images processed. Press ENTER to continue.")
        cv2.destroyAllWindows()

    print("Calibrating...")
    ret1, K1, dc1, rv1, tv1 = cv2.calibrateCamera(objpoints, imgpoints_left,
                                                 img.shape[:2],None,None)
    #ret1, K1, dc1, rv1, tv1 = cv2.calibrateCamera(objpoints, imgpoints_left,
    #                                             img.shape[:2],K1,dc1)
    ret2, K2, dc2, rv2, tv2 = cv2.calibrateCamera(objpoints, imgpoints_right,
                                                 img.shape[:2],None,None)
    #ret2, K2, dc2, rv2, tv2 = cv2.calibrateCamera(objpoints, imgpoints_right,
    #                                             img.shape[:2],K2,dc2)
    print("Reprojection error for left camera: ", ret1)
    print("Reprojection error for right camera: ", ret2)
    left_reproj, left_pix = pixel_reprojections(check_rows, check_cols,
                objpoints, imgpoints_left, K1, rv1, tv1, dc1)
    right_reproj, right_pix = pixel_reprojections(check_rows, check_cols,
                objpoints, imgpoints_right, K2, rv2, tv2, dc2)

    retval, K1, dc1, K2, dc2, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                               imgpoints_left,
                                                               imgpoints_right,
                                                               K1, dc1,
                                                               K2, dc2,
                                                               img.shape[:2],
                                                               cv2.CALIB_FIX_INTRINSIC)
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

    print("Calibration finished!")
    return

def reprojection_error(points3d, points2d, K, rvecs, tvecs, distCoeffs):
    n = len(points3d)
    errs = np.zeros((n,2))
    for i in range(n):
        project, _ = cv2.projectPoints(points3d[i], rvecs[i], tvecs[i],
                                       K, distCoeffs)
        errs[i,:] = np.mean(np.abs((project - points2d[i])), axis = 0)
    return np.mean(errs, axis = 0)

def pixel_reprojections(check_rows, check_cols, points3d, points2d,
                              K, rvecs, tvecs, distCoeffs):
    """
    Reprojects 3D checkerboard positions to pixel coordinates for given camera
    matrix, extrinsic parameters and distortion coefficients.

    Returns two Ix2xJ arrays, where I is the number of checkerboard corners,
    and J is the number of image pairs.
        - pix_reprojected is the array of reprojected pixel coordinates.
        - pix_measured is the array of observed pixel coordinates.
    """
    n = len(points3d)
    pix_measured = np.zeros((check_rows*check_cols,2,n))
    pix_reprojected = np.zeros((check_rows*check_cols,2,n))
    for i in range(n):
        project, _ = cv2.projectPoints(points3d[i], rvecs[i], tvecs[i],
                                       K, distCoeffs)
        pix_reprojected[:,:,i] = np.squeeze(project)
        pix_measured[:,:,i] = np.squeeze(points2d[i])
    return pix_reprojected, pix_measured

def print_stereo_calibration_results(retval,P1,dc1,P2,dc2,R,T,E,F):
    print("Camera matrices: \n",P1,"\n",P2)
    print("Distortion coefficients: \n",dc1,"\n",dc2)
    print("Rotation matrix: \n",R)
    print("Translation vector: \n",T)
    return

if __name__ == "__main__":
    images = "images/"
    #d = get_corners('images','left',6,9,True)
    #print(d['left02'])
    ret, P1, dc1, P2, dc2, R, T, E, F = calibrate_stereo(images,6,9,30,False)
    print_stereo_calibration_results(ret, P1, dc1, P2, dc2, R, T, E, F)
    print(ret)
