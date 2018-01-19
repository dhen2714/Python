import numpy as np

class Quaternion:
    """
    Quaternion:
        q = w+ x*i + y*j + z*k
    """
    def __init__(self, *args):
        if len(args) == 4:
            self.w = args[0]
            self.x = args[1]
            self.y = args[2]
            self.z = args[3]
        elif len(args) == 1 and len(args[0]) == 4:
            self.w = args[0][0]
            self.x = args[0][1]
            self.y = args[0][2]
            self.z = args[0][3]
        else:
            self.w = 1
            self.x = 0
            self.y = 0
            self.z = 0

    def __str__(self):
        return "{} + {}*i + {}*j + {}*k".format(self.w, self.x, self.y, self.z)

    def __repr__(self):
        return "Quaternion({}, {}, {}, {})".format(
            self.w, self.x, self.y, self.z)

    def __add__(self, other):
        if type(self) != type(other):
            raise TypeError("Must add two quaternions.")
        return Quaternion(self.w + other.w, self.x + other.x,
            self.y + other.y, self.z + other.z)

    def __mul__(self, other):
        """self*x, where x is int, float or quaternion."""
        if isinstance(other, (int, float)):
            return Quaternion(other*self.w, other*self.x, other*self.y,
                other*self.z)
        elif type(self) != type(other):
            raise TypeError(
                "Must multiply quaternion with real number or quaternion.")

        w = self.scalar*other.scalar - np.dot(self.vector, other.vector)
        xyz = (self.scalar*other.vector + other.scalar*self.vector +
            np.cross(self.vector, other.vector))
        return Quaternion(w, xyz[0], xyz[1], xyz[2])

    def __rmul__(self, other):
        """x*Quaternion, where x is int or float."""
        if isinstance(other, (int, float)):
            return Quaternion(other*self.w, other*self.x, other*self.y,
                other*self.z)
        else:
            raise TypeError(
                "Must multiply quaternion with real number or quaternion.")

    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

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
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

class DualNumber:
    """
    Dual number is denoted r + eps*d, where eps^2 is 0.
    For a dual number q, r is Real(q), d is Dual(q).
    """
    def __init__(self, real = 0, dual = 0):
        self.real = real
        self.dual = dual

    def __repr__(self):
        return "dual_number({}, {}).".format(self.real, self.dual)

    def __str__(self):
        return "({}) + eps*({})".format(self.real, self.dual)

class DualQuaternion(DualNumber):
    """
    Dual quaternions are dual numbers where both real and dual parts are
    quaternions.
    """
    def __init__(self, *args):
        if len(args) == 8:
            super().__init__(Quaternion([args[0], args[1], args[2], args[3]]),
                             Quaternion([args[4], args[5], args[6], args[7]]))
        elif (len(args) == 1) and (len(args[0]) == 8):
            super().__init__(Quaternion([args[0][0], args[0][1], args[0][2],
                                       args[0][3]]),
                             Quaternion([args[0][4], args[0][5], args[0][6],
                                       args[0][7]]))
        elif len(args) == 2:
            super().__init__(args[0], args[1])
        else:
            super().__init__()

    def __repr__(self):
        return "DualQuaternion({}, {})".format(repr(self.real),
            repr(self.dual))

    def mat(self):
        """
        Returns 4x4 homogenous matrix representation of self.
        """
        H = np.eye(4)

        # q0 = self.real[0]
        # qx = self.real[1]
        # qy = self.real[2]
        # qz = self.real[3]
        q0 = self.real.w
        qx = self.real.x
        qy = self.real.y
        qz = self.real.z

        R = np.array([[q0**2+qx**2-qy**2-qz**2,2*(qx*qy-q0*qz),
                       2*(qx*qz+q0*qy)],
                      [2*(qx*qy*q0*qz),(q0**2-qx**2+qy**2-qz**2),
                       2*(qy*qz-q0*qx)],
                      [2*(qz*qx-q0*qy),2*(qz*qy+q0*qx),
                       (q0**2-qx**2-qy**2+qz**2)]])

        # q = self.real; q[1:] = -self.real[1:]
        q = self.real.conjugate
        # t = 2*qmult(self.dual,q)
        t = 2*(self.dual*q)
        H[:3,:3] = R
        #H[:3,3] = t[1:]
        H[:3,3] = t.vector
        return H

    @property
    def conjugate(self):
        return DualQuaternion(self.real.conjugate, self.dual.conjugate)

    @property
    def scalar(self):
        """Returns scalar part."""
        sc_real = 0.5*(self.real + self.conj.real)
        sc_dual = 0.5*(self.dual + self.conj.dual)
        return dual_number(sc_real.scalar, sc_dual.scalar)

    @property
    def screw_angle(self):
        """Returns screw angle."""
        return 2*np.arccos(self.scalar.real)

    @property
    def screw_distance(self):
        """Returns screw distance."""
        theta = self.screw_angle
        return (-2*self.scalar.dual)/np.sin(theta/2)

def hom2dq(H):
    """Converts a 4x4 homogeneous, rigid body transformation matrix H into a
    dual quaternion."""

    R = H[:3,:3]
    t = H[:3,3]

    q0 = 1 + R[0,0] + R[1,1] + R[2,2]
    qx = 1 + R[0,0] - R[1,1] - R[2,2]
    qy = 1 - R[0,0] + R[1,1] - R[2,2]
    qz = 1 - R[0,0] - R[1,1] + R[2,2]
    qCheck = [q0,qx,qy,qz]

    if np.max(qCheck) == qx:
        qx = -np.sqrt(qx/4)
        qy = (R[1,0]+R[0,1])/(4*qx)
        qz = (R[0,2]+R[2,0])/(4*qx)
        q0 = (R[2,1]-R[1,2])/(4*qx)
    elif np.max(qCheck) == qy:
        qy = np.sqrt(qy/4)
        qx = (R[1,0]+R[0,1])/(4*qy)
        qz = (R[2,1]+R[1,2])/(4*qy)
        q0 = (R[0,2]-R[2,0])/(4*qy)
    elif np.max(qCheck) == qz:
        qz = np.sqrt(qz/4)
        qx = (R[0,2]+R[2,0])/(4*qz)
        qy = (R[2,1]+R[1,2])/(4*qz)
        q0 = (R[1,0]-R[0,1])/(4*qz)
    else:
        q0 = np.sqrt(q0/4)
        qx = (R[2,1]-R[1,2])/(4*q0)
        qy = (R[0,2]-R[2,0])/(4*q0)
        qz = (R[1,0]-R[0,1])/(4*q0)

    dq_real = np.array([q0,qx,qy,qz])
    t_q = np.array([0,t[0],t[1],t[2]])
    dq_dual = 0.5*qmult(t_q,dq_real)

    return DualQuaternion(Quaternion(dq_real), Quaternion(dq_dual))

def qmult(q1,q2):
    """Quaternion multiplication."""
    q1vec = q1[1:]; q2vec = q2[1:]
    q1scalar = q1[0]; q2scalar = q2[0]

    qscalar = q1scalar*q2scalar - np.dot(q1vec,q2vec)
    qvec = q1scalar*q2vec + q2scalar*q1vec + np.cross(q1vec,q2vec)

    q = np.array([qscalar,qvec[0],qvec[1],qvec[2]])

    return q

def skew(w):
    """Skew symmetric matrix from 3-vector."""
    R = np.zeros((3,3))
    R[0,1] = -w[2]; R[0,2] = w[1]; R[1,2] = -w[0]
    R[1,0] = w[2]; R[2,0] = -w[1]; R[2,1] = w[0]
    return R

def dqcrosscalib(A,B):
    """
    Hand-eye calibration AX = XB.

    Inputs:
        A, B : 4x4xN arrays representing N motions in two coordinate frames.
    Outputs:
        X    : 4x4 array relating the two coordinate systems.

    Method by K Daniilidis, "Hand-Eye Calibration Using Dual Quaternions",
    1999.
    """
    n = A.shape[2]
    T = np.zeros((8,6*n))
    S = np.zeros((6,8))

    for i in range(n):
        a = hom2dq(A[:,:,i]); b = hom2dq(B[:,:,i])
        a1 = a.real.vector; b1 = b.real.vector
        a2 = a.dual.vector; b2 = b.dual.vector
        S[:3,0] = a1 - b1
        S[:3,1:4] = skew(a1 + b1)
        S[:3,4:] = np.zeros((3,4))
        S[3:,0] = a2 - b2
        S[3:,1:4] = skew(a2+b2)
        S[3:,4] = a1 - b1
        S[3:,5:] = skew(a1 + b1)
        T[:,6*i:6*(i+1)] = S.T

    # np.linalg.svd() returns the transpose of V.
    U, S, Vt = np.linalg.svd(T.T); V = Vt.T
    u1 = V[:4,6]; v1 = V[4:8,6]
    u2 = V[:4,7]; v2 = V[4:8,7]

    alpha = np.dot(u1,v1)
    beta = np.dot(u1,v2) + np.dot(u2,v1)
    gamma = np.dot(u2,v2)
    s = np.roots([alpha, beta, gamma])

    if s[0] == 0:
        val = np.dot(u2*u2)
    else:
        val1 = s[0]**2*np.dot(u1,u1) + 2*s[0]*np.dot(u1,u2) + np.dot(u2,u2)
        val2 = s[1]**2*np.dot(u1,u1) + 2*s[1]*np.dot(u1,u2) + np.dot(u2,u2)

        if val1 > val2:
            s = s[0]
            val = val1
        else:
            s = s[1]
            val = val2

    # This will be the case when inputs are pure rotations.
    if val == 0:
        q = V[:,6] + V[:,7]
        Xdq = DualQuaternion(q)
    else:
        l2 = np.sqrt(1/val)
        l1 = s*l2
        q = l1*V[:,6] + l2*V[:,7]
        Xdq = DualQuaternion(q)
        if Xdq.real.scalar < 0:
            Xdq.real = -Xdq.real
            Xdq.dual = -Xdq.dual
    return Xdq.mat()

"""Testing"""
if __name__ == "__main__":
    import motion3d
    n_motions = 3
    A = np.zeros((4,4,n_motions))
    B = np.zeros((4,4,n_motions))
    X_true = motion3d.random_motion(5,10)
    #X_true = motion3d.vec2mat([4,12,15,4,3,9])
    A[:,:,0] = motion3d.vec2mat([4,34,5,3,4,12])
    A[:,:,1] = motion3d.vec2mat([3,4,23,7,91,2])
    A[:,:,2] = motion3d.vec2mat([17,13,4,36,18,32])

    for i in range(n_motions):
        Ai = motion3d.random_motion(5,10)
        #Ai = A[:,:,i]
        Bi = motion3d.mdot(np.linalg.inv(X_true),Ai,X_true)
        A[:,:,i] = Ai; B[:,:,i] = Bi

    print(X_true)
    print(dqcrosscalib(A,B))
