import numpy as np


class Quaternion:
    """
    Quaternion
        q = w + x*i + y*j + z*k
    Can be initialised with:
        - 4 arguments w, x, y, z
        - list or numpy array with 4 elements w, x, y, z
        - or Quaternion(angle = a, axis = b), a is angle in degrees and b is
        3 element list or numpy array. Axis is normalised, and a unit
        quaternion is initialised
        - default initialised as 1*w + 0*i + 0*j + 0*k
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 4:
            self.w = args[0]
            self.x = args[1]
            self.y = args[2]
            self.z = args[3]
        elif len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
            self.w = args[0][0]
            self.x = args[0][1]
            self.y = args[0][2]
            self.z = args[0][3]
        elif "angle" in kwargs.keys() and "axis" in kwargs.keys():
            n = kwargs["axis"] / np.linalg.norm(kwargs["axis"])
            self.w = np.cos(np.radians(kwargs["angle"]) / 2)
            self.x = n[0] * np.sin(np.radians(kwargs["angle"]) / 2)
            self.y = n[1] * np.sin(np.radians(kwargs["angle"]) / 2)
            self.z = n[2] * np.sin(np.radians(kwargs["angle"]) / 2)
        else:
            self.w = 1
            self.x = 0
            self.y = 0
            self.z = 0

    def __str__(self):
        return "{} + {}i + {}j + {}k".format(self.w, self.x, self.y, self.z)

    def __repr__(self):
        return "Quaternion({}, {}, {}, {})".format(self.w, self.x, self.y, self.z)

    def __add__(self, other):
        if type(self) != type(other):
            raise TypeError("Must add two quaternions.")
        return Quaternion(
            self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z
        )

    def __mul__(self, other):
        """self*x, where x is int, float or quaternion."""
        if isinstance(other, (int, float)):
            return Quaternion(
                other * self.w, other * self.x, other * self.y, other * self.z
            )
        elif type(self) != type(other):
            raise TypeError("Must multiply quaternion with real number or quaternion.")

        w = self.scalar * other.scalar - np.dot(self.vector, other.vector)
        xyz = (
            self.scalar * other.vector
            + other.scalar * self.vector
            + np.cross(self.vector, other.vector)
        )
        return Quaternion(w, xyz[0], xyz[1], xyz[2])

    def __rmul__(self, other):
        """x*Quaternion, where x is int or float."""
        if isinstance(other, (int, float)):
            return Quaternion(
                other * self.w, other * self.x, other * self.y, other * self.z
            )
        else:
            raise TypeError("Must multiply quaternion with real number or quaternion.")

    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def normalize(self):
        return self.unit

    @property
    def unit(self):
        return self * (1 / self.norm)

    @property
    def axis(self):
        return self.unit.vector / (np.sin(np.radians(self.angle) / 2))

    @property
    def angle(self):
        return np.degrees(2 * np.arccos(self.unit.w))

    @property
    def scalar(self):
        return self.w

    @property
    def vector(self):
        return np.array([self.x, self.y, self.z])

    @property
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    @property
    def inverse(self):
        return self.conjugate * (1 / self.norm**2)

    @property
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    @property
    def mat(self):
        """Returns 3x3 rotation matrix. Normalizes quaternion first."""
        q0 = self.unit.w
        qx = self.unit.x
        qy = self.unit.y
        qz = self.unit.z
        return np.array(
            [
                [
                    q0**2 + qx**2 - qy**2 - qz**2,
                    2 * (qx * qy - q0 * qz),
                    2 * (qx * qz + q0 * qy),
                ],
                [
                    2 * (qx * qy + q0 * qz),
                    (q0**2 - qx**2 + qy**2 - qz**2),
                    2 * (qy * qz - q0 * qx),
                ],
                [
                    2 * (qz * qx - q0 * qy),
                    2 * (qz * qy + q0 * qx),
                    (q0**2 - qx**2 - qy**2 + qz**2),
                ],
            ]
        )

    @property
    def hmat(self):
        """Returns 4x4 homogeneous rotation matrix."""
        H = np.eye(4)
        H[:3, :3] = self.mat
        return H

    @property
    def euler(self):
        """
        Returns Euler angle representation, assumes ZYX ordering, returns
        array in format [rX, rY, rZ] in degrees.
        """
        R = self.mat
        sy = -R[2, 0]
        cy = 1 - (sy * sy)

        if cy > 0.00001:
            cy = np.sqrt(cy)
            cx = R[2, 2] / cy
            sx = R[2, 1] / cy
            cz = R[0, 0] / cy
            sz = R[1, 0] / cy
        else:
            cy = 0.0
            cx = R[1, 1]
            sx = -R[1, 2]
            cz = 1.0
            sz = 0.0
        return np.array(
            [
                np.degrees(np.arctan2(sx, cx)),
                np.degrees(np.arctan2(sy, cy)),
                np.degrees(np.arctan2(sz, cz)),
            ]
        )


class DualNumber:
    """
    Dual number is denoted r + eps*d, where eps^2 is 0.
    For a dual number q, r is Real(q), d is Dual(q).
    """

    def __init__(self, real=0, dual=0):
        self.real = real
        self.dual = dual

    def __repr__(self):
        return "{}({}, {}).".format(
            self.__class__.__name__, repr(self.real), repr(self.dual)
        )

    def __str__(self):
        return "({}) + eps*({})".format(self.real, self.dual)

    def __add__(self, other):
        if type(self) != type(other):
            raise TypeError("Must add two dual numbers of same type.")
        return self.__class__((self.real + other.real), (self.dual + other.dual))

    def __sub__(self, other):
        return self.__class__(self.real + (-other.real), self.dual + (-other.dual))

    def __pos__(self):
        return self.__class__(self.real, self.dual)

    def __neg__(self):
        return self.__class__(-self.real, -self.dual)

    def __mul__(self, other):
        """self*x, where x is int, float or dual number"""
        if isinstance(other, (int, float)):
            return self.__class__(self.real * other, self.dual * other)
        elif type(self) != type(other):
            raise TypeError(
                "Must multiply dual number with real number of dual number"
                + "of same type."
            )
        return self.__class__(
            self.real * other.real, (self.real * other.dual + self.dual * other.real)
        )

    def __rmul__(self, other):
        """x*self, where x is int or float."""
        if isinstance(other, (int, float)):
            return self.__class__(self.real * other, self.dual * other)
        else:
            raise TypeError(
                "Must multiply dual number with real number of dual number"
                + "of same type."
            )


class DualQuaternion(DualNumber):
    """
    Dual quaternions are dual numbers q + eps*q', where q and q' are quaternions.
    """

    def __init__(self, *args):
        if len(args) == 8:
            super().__init__(
                Quaternion([args[0], args[1], args[2], args[3]]),
                Quaternion([args[4], args[5], args[6], args[7]]),
            )
        elif (len(args) == 1) and isinstance(args[0], (list, np.ndarray)):
            super().__init__(
                Quaternion([args[0][0], args[0][1], args[0][2], args[0][3]]),
                Quaternion([args[0][4], args[0][5], args[0][6], args[0][7]]),
            )
        elif len(args) == 2 and all(isinstance(i, (list, np.ndarray)) for i in args):
            super().__init__(Quaternion(args[0]), Quaternion(args[1]))
        elif len(args) == 2 and all(isinstance(i, (Quaternion)) for i in args):
            super().__init__(args[0], args[1])
        else:
            super().__init__(Quaternion(), Quaternion())

    @property
    def norm(self):
        return DualNumber(
            self.real * self.real.conjugate,
            self.real * self.dual.conjugate + self.dual * self.real.conjugate,
        )

    @property
    def mat(self):
        """
        Returns 4x4 homogenous matrix representation of self.
        """
        H = np.eye(4)
        H = self.real.hmat
        H[:3, 3] = 2 * (self.dual * self.real.conjugate).vector
        return H

    @property
    def conjugate(self):
        return DualQuaternion(self.real.conjugate, self.dual.conjugate)

    @property
    def scalar(self):
        """Returns scalar (quaternion) dual number."""
        return DualNumber(self.real.scalar, self.dual.scalar)

    @property
    def vector(self):
        """Returns vector (quaternion) dual number."""
        return DualNumber(self.real.vector, self.dual.vector)

    @property
    def screw_angle(self):
        """Returns screw angle."""
        return 2 * np.arccos(self.scalar.real)

    @property
    def screw_distance(self):
        """Returns screw distance."""
        theta = self.screw_angle
        if theta == 0:
            retval = np.nan
        else:
            retval = (-2 * self.scalar.dual) / np.sin(theta / 2)
        return retval


def rot2q(R: np.ndarray) -> Quaternion:
    """Converts 3x3 rotation matrix to quaternion"""
    q0 = 1 + R[0, 0] + R[1, 1] + R[2, 2]
    qx = 1 + R[0, 0] - R[1, 1] - R[2, 2]
    qy = 1 - R[0, 0] + R[1, 1] - R[2, 2]
    qz = 1 - R[0, 0] - R[1, 1] + R[2, 2]
    qCheck = [q0, qx, qy, qz]

    if np.max(qCheck) == qx:
        qx = np.sqrt(qx / 4)
        qy = (R[1, 0] + R[0, 1]) / (4 * qx)
        qz = (R[0, 2] + R[2, 0]) / (4 * qx)
        q0 = (R[2, 1] - R[1, 2]) / (4 * qx)
    elif np.max(qCheck) == qy:
        qy = np.sqrt(qy / 4)
        qx = (R[1, 0] + R[0, 1]) / (4 * qy)
        qz = (R[2, 1] + R[1, 2]) / (4 * qy)
        q0 = (R[0, 2] - R[2, 0]) / (4 * qy)
    elif np.max(qCheck) == qz:
        qz = np.sqrt(qz / 4)
        qx = (R[0, 2] + R[2, 0]) / (4 * qz)
        qy = (R[2, 1] + R[1, 2]) / (4 * qz)
        q0 = (R[1, 0] - R[0, 1]) / (4 * qz)
    else:
        q0 = np.sqrt(q0 / 4)
        qx = (R[2, 1] - R[1, 2]) / (4 * q0)
        qy = (R[0, 2] - R[2, 0]) / (4 * q0)
        qz = (R[1, 0] - R[0, 1]) / (4 * q0)
    return Quaternion(q0, qx, qy, qz)


def hom2dq(H: np.ndarray) -> DualQuaternion:
    """Converts a 4x4 homogeneous, rigid body transformation matrix H into a
    dual quaternion."""
    R = H[:3, :3]
    t = H[:3, 3]
    q1 = rot2q(R)
    q2 = 0.5 * Quaternion(0, t[0], t[1], t[2]) * q1
    return DualQuaternion(q1, q2)


def qmult(q1: Quaternion, q2: Quaternion) -> np.array:
    """Quaternion multiplication."""
    q1vec = q1[1:]
    q2vec = q2[1:]
    q1scalar = q1[0]
    q2scalar = q2[0]

    qscalar = q1scalar * q2scalar - np.dot(q1vec, q2vec)
    qvec = q1scalar * q2vec + q2scalar * q1vec + np.cross(q1vec, q2vec)
    return np.array([qscalar, qvec[0], qvec[1], qvec[2]])


def skew(w):
    """Skew symmetric matrix from 3-vector."""
    R = np.zeros((3, 3))
    R[0, 1] = -w[2]
    R[0, 2] = w[1]
    R[1, 2] = -w[0]
    R[1, 0] = w[2]
    R[2, 0] = -w[1]
    R[2, 1] = w[0]
    return R


def dqcrosscalib(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Hand-eye calibration AX = XB.

    Inputs:
        A, B : Nx4x4 arrays representing N motions in two coordinate frames.
    Outputs:
        X    : 4x4 array relating the two coordinate systems.

    Method by K Daniilidis, "Hand-Eye Calibration Using Dual Quaternions",
    1999.
    """
    n = len(A)
    T = np.zeros((8, 6 * n))
    S = np.zeros((6, 8))

    for i in range(n):
        a = hom2dq(A[i, :, :])
        b = hom2dq(B[i, :, :])
        a1 = a.real.vector
        b1 = b.real.vector
        a2 = a.dual.vector
        b2 = b.dual.vector
        S[:3, 0] = a1 - b1
        S[:3, 1:4] = skew(a1 + b1)
        S[:3, 4:] = np.zeros((3, 4))
        S[3:, 0] = a2 - b2
        S[3:, 1:4] = skew(a2 + b2)
        S[3:, 4] = a1 - b1
        S[3:, 5:] = skew(a1 + b1)
        T[:, 6 * i : 6 * (i + 1)] = S.T

    # np.linalg.svd() returns the transpose of V.
    U, S, Vt = np.linalg.svd(T.T)
    V = Vt.T
    u1 = V[:4, 6]
    v1 = V[4:8, 6]
    u2 = V[:4, 7]
    v2 = V[4:8, 7]

    alpha = np.dot(u1, v1)
    beta = np.dot(u1, v2) + np.dot(u2, v1)
    gamma = np.dot(u2, v2)
    s = np.roots([alpha, beta, gamma])

    if s[0] == 0:
        val = np.dot(u2 * u2)
    else:
        val1 = s[0] ** 2 * np.dot(u1, u1) + 2 * s[0] * np.dot(u1, u2) + np.dot(u2, u2)
        val2 = s[1] ** 2 * np.dot(u1, u1) + 2 * s[1] * np.dot(u1, u2) + np.dot(u2, u2)

        if val1 > val2:
            s = s[0]
            val = val1
        else:
            s = s[1]
            val = val2

    # This will be the case when inputs are pure rotations.
    if val == 0:
        q = V[:, 6] + V[:, 7]
        Xdq = DualQuaternion(q)
    else:
        l2 = np.sqrt(1 / val)
        l1 = s * l2
        q = l1 * V[:, 6] + l2 * V[:, 7]
        Xdq = DualQuaternion(q)
        if Xdq.real.scalar < 0:
            Xdq.real = -Xdq.real
            Xdq.dual = -Xdq.dual
    return Xdq.mat


"""Testing"""
if __name__ == "__main__":
    import motion3d

    n_motions = 3
    A = np.zeros((n_motions, 4, 4))
    B = np.zeros((n_motions, 4, 4))
    X_true = motion3d.random_motion(5, 10)
    # X_true = motion3d.vec2mat([4,12,15,4,3,9])
    A[0, :, :] = motion3d.vec2mat([4, 34, 5, 3, 4, 12])
    A[1, :, :] = motion3d.vec2mat([3, 4, 23, 7, 91, 2])
    A[2, :, :] = motion3d.vec2mat([17, 13, 4, 36, 18, 32])

    for i in range(n_motions):
        Ai = motion3d.random_motion(5, 10)
        # Ai = A[:,:,i]
        Bi = motion3d.mdot(np.linalg.inv(X_true), Ai, X_true)
        A[i, :, :] = Ai
        B[i, :, :] = Bi

    print(X_true)
    print(dqcrosscalib(A, B))
