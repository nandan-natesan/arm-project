"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm

# Helper function
import numpy as np

def matrix_exp_6(se3_mat):
    """
    Computes the Matrix Exponential of an se(3) matrix using Rodrigues' formula.
    Replace scipy.linalg.expm with this.
    """
    # Extract the skew-symmetric angular velocity matrix
    omg_mat = se3_mat[0:3, 0:3]
    
    # Extract the linear velocity vector
    v = se3_mat[0:3, 3]
    
    # Check if the joint is effectively prismatic (no rotation)
    # We check if the angular velocity is close to zero
    if np.isclose(np.linalg.norm(omg_mat), 0):
        # For prismatic joints: T = [I, v*theta]
        return np.eye(4) + se3_mat
    else:
        # Extract rotation magnitude (theta) from the skew matrix
        # omega_mat = [[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]]
        # The norm of the vector w is roughly the magnitude of elements in omg_mat
        # A robust way to get theta from omg_mat * theta:
        theta = np.sqrt(0.5 * np.trace(omg_mat @ omg_mat.T))
        
        # Normalize the skew matrix
        omg_mat_norm = omg_mat / theta
        
        # Rodrigues' formula for Rotation (SO3)
        # R = I + sin(theta)[w] + (1-cos(theta))[w]^2
        R = (np.eye(3) + 
             np.sin(theta) * omg_mat_norm + 
             (1 - np.cos(theta)) * (omg_mat_norm @ omg_mat_norm))
        
        # Formula for Translation
        # p = (I*theta + (1-cos(theta))[w] + (theta-sin(theta))[w]^2) * v / theta
        G = (np.eye(3) * theta + 
             (1 - np.cos(theta)) * omg_mat_norm + 
             (theta - np.sin(theta)) * (omg_mat_norm @ omg_mat_norm))
        p = np.dot(G, v) / theta
        
        # Construct final 4x4 matrix
        T_res = np.eye(4)
        T_res[0:3, 0:3] = R
        T_res[0:3, 3] = p
        return T_res
def vec_to_se3(S):
    """
    Helper: Converts a 6-element screw vector to a 4x4 se(3) matrix.
    S = [w_x, w_y, w_z, v_x, v_y, v_z]
    """
    # Extract angular (w) and linear (v) components
    w = S[0:3]

    v = S[3:6]
    
    # Form the skew-symmetric matrix for angular velocity (w_hat)
    w_hat = np.array([
        [0,     -w[2],  w[1]],
        [w[2],   0,    -w[0]],
        [-w[1],  w[0],  0   ]
    ])
    
    # Construct the 4x4 se(3) matrix
    se3_mat = np.zeros((4, 4))
    se3_mat[0:3, 0:3] = w_hat
    se3_mat[0:3, 3] = v

    return se3_mat


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle

def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    @param      a      (float) link length (meters)
    @param      alpha  (float) link twist (radians)
    @param      d      (float) link offset (meters)
    @param      theta  (float) joint angle (radians)

    @return     (4x4 np.array) The transformation matrix.
    """
    
    # Pre-compute sin/cos for efficiency and readability
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # Construct the standard DH Transformation Matrix
    T = np.array([
        [ct,    -st * ca,   st * sa,    a * ct],
        [st,     ct * ca,  -ct * sa,    a * st],
        [0,      sa,        ca,         d     ],
        [0,      0,         0,          1     ]
    ])

    return T


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

    @param      dh_params     The dh parameters as a 2D list/array [a, alpha, d, theta_offset]
    @param      joint_angles  The joint angles of the links [q1, q2, ...]
    @param      link          (int) The index of the target link (1 to N). 
                              If link=N, returns End Effector pose relative to World.

    @return     (4x4 np.array) Transformation matrix representing the pose of the desired link
    """
    
    # 1. Prepare the parameters
    # Use copy=True to avoid modifying the original dh_params list passed by the user
    # params = np.array(dh_params, copy=True)
    params = np.array([[0,1.570796327,103.91,0],
                       [205.73,0,0,1.3342],
                       [200,0,0,-1.3342],
                       [0,1.570796327,0,1.570796327],
                       [0,0,174.15,0]])
    
    # Add current joint angles to the theta column (index 3)
    # This combines the static offset (from DH table) with the dynamic angle (from motors)
    # params[i, 3] = theta_offset + joint_angle
    num_joints = len(joint_angles)

    params[:3, 3] -= joint_angles[:3]
    params[3, 3] -= joint_angles[3]
    params[4, 3] -= joint_angles[4]

    # params[:num_joints, 3] += joint_angles
    
    # 2. Initialize the global transformation as Identity
    T_base = np.array([
    [ 0,  1,  0,  0],
    [1,  0,  0,  0],
    [ 0,  0,  1,  0],
    [ 0,  0,  0,  1]
])

    T_global = T_base 
    for i in range(link):
        row = params[i]
        
        # Extract DH parameters for this row
        a     = row[0]
        alpha = row[1]
        d     = row[2]
        theta = row[3]
        
        # Calculate transform for this specific link (T_{i-1, i})
        T_link = get_transform_from_dh(a, alpha, d, theta)
        
        # Multiply to the global transform (order matters: T_current * T_new)
        T_global = np.dot(T_global, T_link)
        
    return T_global

        


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

    @param      T     (4x4) transformation matrix
    @param      convention  (str) 'zyx' (default, RPY) or 'zyz'

    @return     (3,) numpy array of euler angles [angle_1, angle_2, angle_3]
                For 'zyx', returns [yaw, pitch, roll] (alpha, beta, gamma)
    """
    
    # Extract the 3x3 Rotation Matrix portion from T
    R = T[0:3, 0:3]
    
    # Initialize angles
    alpha, beta, gamma = 0.0, 0.0, 0.0
    
    # Tolerance for gimbal lock detection
    tol = 1e-6

    # Z-Y-Z Classic Euler angles
    # Corresponds to R = Rz(alpha) * Ry(beta) * Rz(gamma)
        
    if R[2, 2] > 1.0 - tol:
        # beta = 0
        beta = 0
        alpha = 0
        gamma = np.arctan2(R[1, 0], R[0, 0])
    elif R[2, 2] < -1.0 + tol:
        # beta = 180
        beta = np.pi
        alpha = 0
        gamma = np.arctan2(R[1, 0], R[0, 0])
    else:
        beta = np.arctan2(np.sqrt(R[2, 0]**2 + R[2, 1]**2), R[2, 2])
        alpha = np.arctan2(R[1, 2], R[0, 2])
        gamma = np.arctan2(R[2, 1], -R[2, 0])
        
    return np.array([alpha, beta, gamma])



def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

    @param      T     (4x4) transformation matrix

    @return     (list) [x, y, z, roll, pitch, yaw]
    """
    
    # 1. Extract the Position (Translation)
    # The last column, first 3 rows contains the position vector p
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    
    # 2. Extract the Euler Angles (Orientation)
    # Uses the helper function from the previous step (defaulting to 'zyx' / RPY)
    # Note: verify if your helper returns [yaw, pitch, roll] or [roll, pitch, yaw].
    # The function I provided previously returned [yaw, pitch, roll].
    yaw, pitch, roll = get_euler_angles_from_T(T)
    
    return [x, y, z, roll, pitch, yaw]


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a representing the pose of the desired link

    @param      joint_angles  (list or np.array) The joint angles [theta1, theta2, ...]
    @param      m_mat         (4x4 np.array) The M matrix (Home configuration)
    @param      s_lst         (6xN np.array) List of screw vectors (columns)

    @return     (4x4 np.array) Homogeneous matrix representing the pose of the desired link
    """
    
    # 1. Initialize the transformation matrix as Identity
    # This will accumulate the joint transformations: T = e^(S1*t1) * e^(S2*t2) ...
    T = np.eye(4)
#     m_mat = np.array([[1.0,0.0,0.0,408.575]
# [0.0,1.0,0.0,0.0],
#     [0.0,0.0,1.0,304.57],
# [0.0,0.0,0.0,1.0]
# ])    
#     s_lst = np.array([[0.0,0.0,0.0,0.0,1.0],
#     [0.0,1.0,1.0,1.0,0.0],
#     [0.0,-104.57,-304.57,-304.57,0.0],
#     [0.0,0.0,0.0,0.0,304.57],
#     [0.0,0.0,50,250,0.0],
#     ])
# screw vectors
    m_mat = np.array(m_mat)
    s_lst = np.array(s_lst)
    # 2. Iterate through each joint
    # We assume s_lst has shape (6, N) where N is number of joints
    num_joints = len(joint_angles)
    
    for i in range(num_joints):
        theta = joint_angles[i]

        # Get the screw vector for the current joint (i-th column)
        S = s_lst[:, i]

        # Convert screw vector to 4x4 se(3) matrix representation [S]
        S_se3 = vec_to_se3(S)

        # Calculate matrix exponential: e^([S] * theta)
        # This creates the transformation matrix for this specific joint's motion
        # joint_trans = scipy.linalg.expm(S_se3 * theta)
        joint_trans = matrix_exp_6(S_se3 * theta)

        # Accumulate the result (Order matters: Base -> Tip)
        T = T @ joint_trans
        

    # 3. Multiply by the Home Configuration (M) at the very end
    # Formula: T_final = (Product of Exponentials) * M
    T_robot = np.dot(T, m_mat)
    T_base = np.array([
            [ 0,  -1,  0,  0],
            [1,  0,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]
        ])
        
   

    # Pre-multiply by the base frame transformation
    T_final = np.dot(T_base, T_robot)
    # T_final = T @ m_mat
    
    return T_final


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dh_params, pose):

    tol = 1e-6

    #---------
    # parse data from inputs
    #---------
    # assign geometry from dh params
    dh_params = np.asarray(dh_params, dtype=float)
    l1 = dh_params[1, 0] #a2
    l2 = dh_params[2, 0] #a3
    d1 = dh_params[0, 2] #base height
    Lt = dh_params[4, 2] #tool offset (d5)

    pose_world = np.asarray(pose, dtype=float)

    # transform input pose from world to robot frame
    T_base = np.array([
    [ 0,  1,  0,  0],
    [1,  0,  0,  0],
    [ 0,  0,  1,  0],
    [ 0,  0,  0,  1]
    ])
    pose_robot = np.dot(T_base, pose_world[:3])
    
    # extract x, y, z, psi from pose_world
    x_p, y_p, z_p = pose_robot[0], pose_robot[1], pose_robot[2]
    psi = pose[4]
    # psi = wrist pitch about x/y horizontal plane
    # ensure angles are between pi and -pi
    psi = clamp(psi)
    
    #---------
    # determine theta1 for joint 1 (base yaw)
    #---------
    # project pose x, y onto horizontal (x,y) plane
    r_xy = np.hypot(x_p, y_p)
    # check if degenerate case
    if r_xy < tol:
        # end located on z axis - arm is in singular configuration
        # infinite solutions therefore return two representative yaw angles
        th1_list = [0.0, np.pi]
    else:
        th1_list = [np.arctan2(y_p, x_p), np.arctan2(y_p, x_p)+np.pi]
    
    # reduce problem to 2D RR elbow starting at joint 2
    z_0 = z_p - d1
    
    sols = []
    # -------
    # determine theta3 for joint 3
    #---------
    for th1 in th1_list:
        th1 = clamp(th1)
    #determine r using th1
        r = np.cos(th1) * x_p + np.sin(th1) * y_p
        # determine wrist center using 2D RR elbow
        r_wc = r - Lt * np.cos(psi)
        z_wc = z_0 - Lt * np.sin(psi)

        # assign distance from shoulder to wrist (l3) for clarity
        l3 = np.hypot(r_wc, z_wc)
        # calculate cos(theta_3)
        
        c3 = (l3**2 - l1**2 - l2**2) / (2 * l1 * l2)
        #check if degenerate condition
        if c3 > 1.0 + tol or c3 < -1.0 - tol:
            # cannot reach position with this branch
            continue
        else:
            c3 = np.clip(c3, -1.0, 1.0)

        # calc theta for joint 3    
        # two outputs depending on sign
        th3_list = [np.arccos(c3), -np.arccos(c3)]

        # -------
        # determine theta2 for joint 2
        #---------
        for th3 in th3_list:
            # check singularity for shoulder/wrist
            if l3 < tol:
                th2 = 0.0
            # calc theta2 for joint 2
            else:
                th2 = np.arctan2(z_wc, r_wc) - np.arctan2(l2*np.sin(th3), l1 + l2*np.cos(th3))

            # calc theta for joint 4 given theta2 and theta3
            # assuming single orientation angle defined as the angle from the x-y base plane
            th4 = psi - (th2 + th3)
            # -------
            # build 4x4 solution output
            #---------
            # th1, th2, th3, th4
            # base, shoulder, elbow, wrist pitch
            # 2 x base orientations (left/ right)
            # 2 x elbow orientations (up/ down)
            sols.append([th1, th2, th3, th4])

    if len(sols) == 0:
        raise ValueError("Pose is not in workspace - cannot reach position")
    else:
        return np.array(sols, dtype=float)

def IK_solutionfilter(q_IKsol, q_curr, wrist_phi):
    """!
    @jbrief Filters possible joint configs that produce goal pose to select best candidate for motion planning

    @param q_IKsol Solutions provided by IK_geometric, shape (M,4)
    @param q_curr Current joint positions from RXARM, shape (5,)
    @param wrist_phi Desired wrist roll angle (th5)

    @return Best joint configuration array, shape (5,1) [th1, th2, th3, th4, th5]
    """

    def wrap_to_pi(a):
    #Map angle(s) to (-pi, pi]. Works on scalars or arrays.
        return (a + np.pi) % (2 * np.pi) - np.pi

        joint_limits = [
            (-np.pi, np.pi), # th1, waist
            (-108*np.pi/180, 113*np.pi/180), # th2, shoulder
            (-108*np.pi/180, 93*np.pi/180), # th3, elbow
            (-100*np.pi/180, 123*np.pi/180), # th4, wrist pitch
            (-np.pi, np.pi) # th5, wrist roll
            ]

        # penalize movement of particular joints by assigning weights
        w = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

        q_IKsol = np.asarray(q_IKsol, dtype=float)
        q_curr = np.asarray(q_curr, dtype=float).reshape(-1)

        if q_IKsol.ndim != 2 or q_IKsol.shape[1] != 4:
            raise ValueError(f"q_IKsol must be shape (M,4); got {q_IKsol.shape}")
        if q_curr.size < 4:
            raise ValueError(f"q_curr must have at least 4 elements; got {q_curr.size}")

        # Wrap current joints to (-pi, pi]
        q_curr4 = wrap_to_pi(q_curr[:4])

        # Wrap desired wrist roll
        th5 = wrap_to_pi(float(wrist_phi))

        # assign default variables for comparison
        q_best = None
        best_cost = np.inf

        for q_4 in q_IKsol:
            # wrap to [-pi, pi]
            q_4 = wrap_to_pi(np.asarray(q_4, dtype=float))

            # build 5-joint candidate using wrapped th5
            q_test = np.array([q_4[0], q_4[1], q_4[2], q_4[3], th5], dtype=float)

            # check potential solution against joint limits
            feasible = True
            for j in range(5):
                mn, mx = joint_limits[j]
                if not (mn <= q_test[j] <= mx):
                    feasible = False
                    break
            if not feasible:
                continue

            # assign cost to current potential solution
            dq = wrap_to_pi(q_4 - q_curr4)
            cost = float(np.sum(w * dq * dq))

            # optional singularity avoidance
            cost += 0.05 / (abs(np.sin(q_4[2])) + 1e-3)

            # compare cost of potential solution to current best
            if cost < best_cost:
                best_cost = cost
                q_best = q_test
        # catch for no potential solutions
        if q_best is None:
            raise ValueError("No feasible IK candidate within joint limits.")

        # return the best configuration
        return q_best