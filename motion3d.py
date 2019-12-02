import numpy as np


def mdot(*args):
    """
    Matrix multiplication for more than 2 arrays.
    """
    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret, a)
    return ret


def vec2mat(*args):
    """
    Converts a six vector represenation of motion to a 4x4 matrix.
    Assumes yaw, pitch, roll are in degrees.
    Inputs:
        *args - either 6 numbers (yaw,pitch,roll,x,y,z) or an array with
             6 elements.
     Outputs:
        t     - 4x4 matrix representation of six vector.
    """
    if len(args) == 6:
        yaw = args[0]
        pitch = args[1]
        roll = args[2]
        x = args[3]
        y = args[4]
        z = args[5]
    elif len(args) == 1:
        yaw = args[0][0]
        pitch = args[0][1]
        roll = args[0][2]
        x = args[0][3]
        y = args[0][4]
        z = args[0][5]

    ax = np.radians(yaw)
    ay = np.radians(pitch)
    az = np.radians(roll)
    t1 = np.array([[1, 0, 0, 0],
                   [0, np.cos(ax), -np.sin(ax), 0],
                   [0, np.sin(ax), np.cos(ax), 0],
                   [0, 0, 0, 1]])
    t2 = np.array([[np.cos(ay), 0, np.sin(ay), 0],
                   [0, 1, 0, 0],
                   [-np.sin(ay), 0, np.cos(ay), 0],
                   [0, 0, 0, 1]])
    t3 = np.array([[np.cos(az), -np.sin(az), 0, 0],
                   [np.sin(az), np.cos(az), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    tr = np.array([[1, 0, 0, x],
                   [0, 1, 0, y],
                   [0, 0, 1, z],
                   [0, 0, 0, 1]])
    return mdot(tr, t3, t2, t1)


def mat2vec(H):
    """
    Converts a 4x4 representation of pose to a 6 vector.
    Inputs:
        H - 4x4 matrix.
    Outputs:
        v - [yaw,pitch,roll,x,y,z] (yaw,pitch,roll are in degrees)
    """
    sy = -H[2, 0]
    cy = 1-(sy*sy)

    if cy > 0.00001:
        cy = np.sqrt(cy)
        cx = H[2, 2]/cy
        sx = H[2, 1]/cy
        cz = H[0, 0]/cy
        sz = H[1, 0]/cy
    else:
        cy = 0.0
        cx = H[1, 1]
        sx = -H[1, 2]
        cz = 1.0
        sz = 0.0

    r2deg = (180/np.pi)
    return np.array([np.arctan2(sx, cx)*r2deg, np.arctan2(sy, cy)*r2deg,
                     np.arctan2(sz, cz)*r2deg,
                     H[0, 3], H[1, 3], H[2, 3]])


def stavdahl_average(*arrays):
    """
    Compute average transformation using Oyvind Stavdahl's method.
    """
    num_arr = len(arrays)
    sum_arr = sum(arrays)

    # Get average rotation
    U, S, Vt = np.linalg.svd(sum_arr[:3, :3]) # Vt, transpose of V is returned
    if np.linalg.det(sum_arr[:3, :3]) < 0:
        J = np.diag([1, 1, -1])
    else:
        J = np.diag([1, 1, 1])
    Rav = np.vstack((mdot(Vt.T, J, U.T).T, np.zeros((1, 3))))

    # Get average translation
    tav = sum_arr[:, 3]/num_arr
    return np.hstack((Rav, tav.reshape(4, 1)))


def random_motion(rot=5, tr=10):
    """
    Create random 4x4 homogeneous matrix representing a random rigid body
    motion. Max range of rotations and translations specified by rot and tr.
    """
    yaw = rot*2*(np.random.rand() - 0.5)
    pitch = rot*2*(np.random.rand() - 0.5)
    roll = rot*2*(np.random.rand() - 0.5)
    x = tr*2*(np.random.rand() - 0.5)
    y = tr*2*(np.random.rand() - 0.5)
    z = tr*2*(np.random.rand() - 0.5)
    return vec2mat(yaw, pitch, roll, x, y, z)


def transform_pose_array(poses, reference_pose=None):
    """
    Transforms poses to motions.
    Inputs:
        poses - Nx6 array of poses in [Rx Ry Rz x y z] format
        reference_pose - reference pose, defaults to first pose
    Outputs:
        motions - Nx6 array of motions in [Rx Ry Rz x y z format]
    """
    num_poses = len(poses)
    motions = np.zeros((num_poses, 6))

    if reference_pose is None:
        reference_pose = 0
    
    ref_mat = vec2mat(poses[reference_pose])

    for i in range(num_poses):
        pose_mat = vec2mat(poses[i])
        motion = np.dot(pose_mat, np.linalg.inv(ref_mat))
        motions[i, :] = mat2vec(motion)

    return motions


def calibrate_motion_array(motions, calibration):
    """
    Calibrates a motion array B to coordinate frame A by applying A = XBX^-1
    Inputs:
        motions - Nx6 array of motions in [Rx Ry Rz x y z] format in coordinate
                  frame B
        calibration - the calibration X converting measurements in B to A
    Outputs:
        cal_motions - calibrated motions in coordinate frame A
    """
    num_motions = len(motions)
    cal_motions = np.zeros((num_motions, 6))

    cal_inv = np.linalg.inv(calibration)

    for i in range(num_motions):
        motion = vec2mat(motions[i, :])
        cal_motion = mdot(calibration, motion, cal_inv)
        cal_motions[i, :] = mat2vec(cal_motion)

    return cal_motions


def hornmm(X1, X2):
    """
    Translated from hornmm.pro

    Least squares solution to X1 = H*X2.
    Outputs H, the transformation from X2 to X1.
    Inputs X1 and X2 are Nx3 (or Nx4 homogeneous) arrays of 3D points.

    Implements method in "Closed-form solution of absolute orientation using
    unit quaternions",
    Horn B.K.P, J Opt Soc Am A 4(4):629-642, April 1987.
    """
    N = X2.shape[0]

    xc = np.sum(X2[:, 0])/N
    yc = np.sum(X2[:, 1])/N
    zc = np.sum(X2[:, 2])/N
    xfc = np.sum(X1[:, 0])/N
    yfc = np.sum(X1[:, 1])/N
    zfc = np.sum(X1[:, 2])/N

    xn = X2[:, 0] - xc
    yn = X2[:, 1] - yc
    zn = X2[:, 2] - zc
    xfn = X1[:, 0] - xfc
    yfn = X1[:, 1] - yfc
    zfn = X1[:, 2] - zfc

    sxx = np.dot(xn, xfn)
    sxy = np.dot(xn, yfn)
    sxz = np.dot(xn, zfn)
    syx = np.dot(yn, xfn)
    syy = np.dot(yn, yfn)
    syz = np.dot(yn, zfn)
    szx = np.dot(zn, xfn)
    szy = np.dot(zn, yfn)
    szz = np.dot(zn, zfn)

    M = np.array([[sxx, syy, sxz],
                  [syx, syy, syz],
                  [szx, szy, szz]])

    N = np.array([[(sxx+syy+szz), (syz-szy), (szx-sxz), (sxy-syx)],
                  [(syz-szy), (sxx-syy-szz), (sxy+syx), (szx+sxz)],
                  [(szx-sxz), (sxy+syx), (-sxx+syy-szz), (syz+szy)],
                  [(sxy-syx), (szx+sxz), (syz+szy), (-sxx-syy+szz)]])

    eVal, eVec = np.linalg.eig(N)
    index = np.argmax(eVal)
    vec = eVec[:, index]
    q0 = vec[0]
    qx = vec[1]
    qy = vec[2]
    qz = vec[3]

    X = np.array([[(q0*q0+qx*qx-qy*qy-qz*qz), 2*(qx*qy-q0*qz), 2*(qx*qz+q0*qy), 0],
                  [2*(qy*qx+q0*qz), (q0*q0-qx*qx+qy*qy-qz*qz), 2*(qy*qz-q0*qx), 0],
                  [2*(qz*qx-q0*qy), 2*(qz*qy+q0*qx), (q0*q0-qx*qx-qy*qy+qz*qz), 0],
                  [0, 0, 0, 1]])

    Xpos = np.array([xc, yc, zc, 1])
    Xfpos = np.array([xfc, yfc, zfc, 1])
    d = Xpos - np.dot(np.linalg.inv(X), Xfpos)

    Tr = np.array([[1, 0, 0, -d[0]],
                   [0, 1, 0, -d[1]],
                   [0, 0, 1, -d[2]],
                   [0, 0, 0, 1]])
    return np.dot(X, Tr)


if __name__ == '__main__':
    arr1 = np.eye(4)
    arr2 = np.eye(4)
    arr3 = stavdahl_average(arr1, arr2)
    # print(__name__)
