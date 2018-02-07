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
import matplotlib.pyplot as plt
import pandas as pd

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

    images = glob.glob('{}*.jpg'.format(directory + '/' + prefix))

    for fname in images:
        print('Processing ... ',fname)
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
                cv2.drawChessboardCorners(img,
                                          (check_rows, check_cols),
                                          corners,
                                          ret)
                cv2.imshow('{}'.format(view), img)
                cv2.waitKey(1)

    if showResults:
        input('Press ENTER to continue.')
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
            newKey = 'view' + num
            dict3D.update({newKey : X})
    return dict3D

def get_motions(dict3D,reference = 'view01'):
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

def sort_image_filenames(list1, list2, prefix1, prefix2):
    """
    Given two numpy arrays of calibration image filenames, sort them by number.
    """
    names1 = [os.path.split(_)[1] for _ in list1]
    names2 = [os.path.split(_)[1] for _ in list2]
    nums1 = [int(os.path.splitext(_)[0].replace(prefix1, '')) for _ in names1]
    nums2 = [int(os.path.splitext(_)[0].replace(prefix2, '')) for _ in names2]
    return list1[np.argsort(nums1)], list2[np.argsort(nums2)]

def calibrate_stereo(directory, check_rows, check_cols, length,
                     left_prefix = 'left',
                     right_prefix = 'right',
                     pixel_format = 'pgm',
                     showResults = True):
    """
    Calibrate stereo camera setup.
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
    left_search = os.path.join(directory, '{}*.{}'.format(left_prefix,
                                                          pixel_format))
    right_search = os.path.join(directory, '{}*.{}'.format(right_prefix,
                                                           pixel_format))
    left_images = np.array(glob.glob(left_search))
    right_images = np.array(glob.glob(right_search))

    # Sort filenames by their number.
    left_images, right_images = sort_image_filenames(left_images,
                                                     right_images,
                                                     left_prefix,
                                                     right_prefix)

    ret_left = [] # Images for which checkerboard points found
    ret_right = []

    if not (len(left_images) and len(right_images)):
        print('No images found matching {}, {}'.format(left_search,
                                                       right_search))
        return

    set_number = 1
    unsuccessful_views = set()

    for image_left, image_right in zip(left_images, right_images):
        view_left = os.path.split(image_left)[1].split('.')[0]
        view_right = os.path.split(image_right)[1].split('.')[0]
        print('Processing ... ', view_left, view_right)

        img_left = cv2.imread(image_left)
        img_right = cv2.imread(image_right)
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
            ret_left.append(image_left)
            ret_right.append(image_right)

            if showResults:
                cv2.drawChessboardCorners(img_left,
                                          (check_rows, check_cols),
                                          corners1,
                                          ret1)
                cv2.drawChessboardCorners(img_right,
                                          (check_rows, check_cols),
                                          corners2,
                                          ret2)
                cv2.imshow('{}, {}'.format(view_left, view_right),
                           np.hstack((img_left, img_right)))
                cv2.waitKey(0)
        else:
            print('Could not find corners in both {} and {}'.format(
                    view_left, view_right))


    input('All images processed. Press ENTER to continue.')
    cv2.destroyAllWindows()

    ret1, K1, dc1, rv1, tv1 = cv2.calibrateCamera(objpoints, imgpoints_left,
                                                 img_left.shape[:2],None,None)

    ret2, K2, dc2, rv2, tv2 = cv2.calibrateCamera(objpoints, imgpoints_right,
                                                 img_left.shape[:2],None,None)


    print('Reprojection error for left camera: ', ret1)
    print('Reprojection error for right camera: ', ret2)

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
    rms_left = np.zeros(len(ret_left))
    rms_right = np.zeros(len(ret_right))
    for image_left, image_right, i in zip(ret_left, ret_right,
                                          range(len(ret_left))):
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
                      columns = [left_prefix, right_prefix])
                      #index = imageNames)
    df.plot.bar()
    plt.ylabel('Mean reprojection error (pixels)')
    plt.title('Stereo calibration mean reprojection error')
    plt.show()

    print('Final reprojection error: ', retval)
    np.savez('Stereo_calibration',
             retval = retval,
             K1 = K1, dc1 = dc1, K2 = K2, dc2 = dc2, R = R, T = T,
             Ematrix = E, Fmatrix = F,
             Left_pix_coords = left_pix, Right_pix_coords = right_pix,
             Left_pix_reproj = left_reproj, Right_pix_reproj = right_reproj)

    np.savez('Left_calibration', ret = ret1, K = K1, dc = dc1,
             rvecs = rv1, tvecs = tv1,
             Checkerboard_coords = left_pix,
             Checkerboard_reprojected = left_reproj)

    np.savez('Right_calibration', ret = ret2, K = K2, dc = dc2,
             rvecs = rv2, tvecs = tv2,
             Checkerboard_coords = right_pix,
             Checkerboard_reprojected = right_reproj)

    print('Calibration finished, results saved.')
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

def print_stereo_cal_npz(npz_file = 'Stereo_calibration.npz'):
    """
    Displays results of stereo calibration for a given npz stereo cal file.
    """
    data = np.load(npz_file)
    K1 = data['K1']; K2 = data['K2']
    dc1 = data['dc1']; dc2 = data['dc2']
    R1 = np.eye(3); R2 = data['R']; t = data['T']
    reprojection_error = data['retval']

    P1 = np.dot(K1, np.hstack((R1, np.zeros((3,1)))))
    P2 = np.dot(K2, np.hstack((R2, t.reshape(3,1))))

    print('P1 : \n {} \n'.format(repr(P1)) +
          'P2 : \n {} \n'.format(repr(P2)) +
          'K1 : \n {} \n'.format(repr(K1)) +
          'K2 : \n {} \n'.format(repr(K2)) +
          'R : \n {} \n'.format(repr(R2))  +
          't : \n {} \n'.format(repr(t))   +
          'Distortion (k1, k2, p1, p2, k3): \n' +
          'Camera 1: {} \n'.format(repr(dc1)) +
          'Camera 2: {} \n'.format(repr(dc2)) +
          'Reprojection error: {} \n'.format(reprojection_error))
    return

def load_stereo_calibration_results(npzFile = 'Stereo_calibration.npz'):
    f = np.load(npzFile)
    K1 = f['K1']; K2 = f['K2']; dc1 = f['dc1']; dc2 = f['dc2']
    R = f['R']; t = f['T']
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3,1)))))
    P2 = np.dot(K2, np.hstack((R, t.reshape(3,1))))
    return P1, P2, K1, K2, np.squeeze(dc1), np.squeeze(dc2), R, t

def load_cam_calibration_results(npzFile):
    f = np.load(npzFile)
    K = f['K']; dc = f['dc']; rv = f['rvecs']; tv = f['tvecs']
    reprojection_error = f['ret']
    return K, np.squeeze(dc), rv, tv, reprojection_error

def print_stereo_calibration_results(retval,P1,dc1,P2,dc2,R,T,E,F):
    print('Camera matrices: \n',P1,'\n',P2)
    print('Distortion coefficients: \n',dc1,'\n',dc2)
    print('Rotation matrix: \n',R)
    print('Translation vector: \n',T)
    return

if __name__ == '__main__':
    images = 'images/'
    #d = get_corners('images','left',6,9,True)
    #print(d['left02'])
    ret, P1, dc1, P2, dc2, R, T, E, F = calibrate_stereo(images,6,9,30,False)
    print_stereo_calibration_results(ret, P1, dc1, P2, dc2, R, T, E, F)
    print(ret)
