import numpy as np
import scipy.spatial.transform as st

def tr(R):
    return (R[0][0] + R[1][1] + R[2][2])

def rot_dist(R1,R2):
    #print(R1)
    #print(R2)
    R=R1@(R2.T)
    temp = np.clip((tr(R)-1)/2, -1,1)
    return np.arccos(temp)

def normalize(vect):
    if not np.linalg.norm(vect)==0:
        return vect/np.linalg.norm(vect)
    else:
        return np.zeros(3)

def v1v2_to_angleaxis(v1,v2):
    if np.all((v1 - v2)<10e-8):
        return 0, np.array([0,0,0])
    alpha = np.arccos(np.dot(v1, v2))  
    #adjust rotation to [0,2pi]
    if alpha > 2*np.pi: 
        alpha = alpha % (2*np.pi) 

    if alpha < 0:
        n = -alpha // (2*np.pi)
        alpha += (n+1)*2*np.pi

    alpha_rot_axis = np.cross(v1, v2)
    alpha_rot_axis /= np.linalg.norm(alpha_rot_axis)
    assert np.isclose(np.linalg.norm(alpha_rot_axis),1)
    assert not np.isnan(alpha)
    assert not np.any(np.isnan(alpha_rot_axis))
    return alpha, alpha_rot_axis

def decompose_rotvec(target_rotvec, R_bd): #prerot = R_bd
    #decompose target_rotvec into prerot@xy_rotvec@Rot_z
    R_bt = st.Rotation.from_rotvec(target_rotvec).as_matrix() #rotation matrix from base to target

    pd_d = np.array([0,0,1]) #not needed, can be put directly in equation really
    R_dt = R_bd.T@R_bt
    pt_d = R_dt@np.array([0,0,1]) #target frame z vector expressed in initial ee offset frame
    alpha, alpha_rot_axis = v1v2_to_angleaxis(pd_d, pt_d) #dotprod,crossprod to find (positive) alpha and corresponding axis

    assert alpha_rot_axis[2] <0.01
    alpha_rot_axis[2] = 0 
    alpha_rot_axis = normalize(alpha_rot_axis)
    R_d1 = st.Rotation.from_rotvec(alpha*alpha_rot_axis).as_matrix()

    R_1b = (R_bd@R_d1).T # R_1b = R_b1.T is true for SO(3) matrices (rotation matrices -- special orthogonal group 3)
    R_12 = R_1b@R_bt #2 is same as target prior to joint limiting. 

    rotvec_12 = st.Rotation.from_matrix(R_12).as_rotvec()
    beta = np.linalg.norm(rotvec_12)*np.sign(rotvec_12[2])
    assert rotvec_12[0] < 0.01
    assert rotvec_12[1] < 0.01
    return alpha_rot_axis, alpha, beta #beta rot axis = [0 0 1] by the definition of this decomposition

def pose_is_within_limits(target_pose, tcp_limits, R_bd):

    axis, a, b = decompose_rotvec(target_pose[3:6], R_bd)

    #Making sure the initial pose is within tcp limits. This makes it safe fpr the user to modify tcp limits
    x,y,z = target_pose[:3]
    return ((tcp_limits.alpha[0] <= a <= tcp_limits.alpha[1]) and \
            (tcp_limits.beta[0] <= b <= tcp_limits.beta[1]) and \
            (tcp_limits.x[0]-0.001 <= x <= tcp_limits.x[1]-0.001) and \
            (tcp_limits.y[0]-0.001 <= y <= tcp_limits.y[1]-0.001) and \
            (tcp_limits.z[0]-0.001 <= z <= tcp_limits.z[1]+0.001))

class TCPLimiter():
    def __init__(self, tcp_limits, R_bd):
        self.R_bd = R_bd
        self.tcp_limits = tcp_limits
        self.n_whole_rot_beta = 0
        self.old_local_beta = None
        self.old_beta = None
    
    def get_limited_pose(self, pose):
        # The target rotation can be decomposed into three rotations. This decomposition is chosen for convenience when it comes to enforcing joint limits.
            # Firstly, rotate from base system to the initial end-effector rotation.
            # Then, we perform an axis-angle rotation where the z-component is 0. With this, we can effectively "point" the z-axis in any direction. This rotation has the angle "alpha".
            # Lastly, we rotate around the new z-axis by the angle "beta".
        target_pose=pose.copy()
        alpha_rot_axis, alpha, local_beta = decompose_rotvec(target_pose[3:6], self.R_bd)

        if self.old_local_beta is not None:
            if local_beta-self.old_local_beta <-3*np.pi/2:
                self.n_whole_rot_beta += 1
            if local_beta-self.old_local_beta >3*np.pi/2:
                self.n_whole_rot_beta -= 1

        self.old_local_beta = local_beta
        # add whole rot if 
        # add full loops to beta
        beta = local_beta + self.n_whole_rot_beta*2*np.pi 

        # enforce limits
        alpha_lim = self.tcp_limits.alpha #[0, 0.2] #alpha is actually strictly nonnegative
        beta_lim = self.tcp_limits.beta #[-2*np.pi-0.1, 0.1]

        beta = np.clip(beta, beta_lim[0], beta_lim[1])
        alpha = np.clip(alpha, alpha_lim[0], alpha_lim[1]) 
        #record previous beta for keeping track of full rotations
        if self.old_beta is not None:
            pass
            #assert np.abs(beta-self.old_beta) < 0.5 #safety check
        self.old_beta = beta


        R_d1_limited = st.Rotation.from_rotvec(alpha*alpha_rot_axis).as_matrix()
        R_12_limited = st.Rotation.from_rotvec(np.array([0,0,beta])).as_matrix()
        new_target_rot = st.Rotation.from_matrix(self.R_bd@R_d1_limited@R_12_limited).as_rotvec()

        target_pose[3:6] = new_target_rot

        x_lim = self.tcp_limits.x #[0.55,0.9]#[0.5,0.9]
        y_lim = self.tcp_limits.y #[-0.4,0.2]#[-0.4, 0.4]
        z_lim = self.tcp_limits.z #[0.03,0.26]#[0.217, 0.4]

        target_pose[:3] = np.clip(target_pose[:3], [x_lim[0], y_lim[0], z_lim[0]], [x_lim[1], y_lim[1], z_lim[1]])
        assert len(target_pose) == len(pose)
        return target_pose