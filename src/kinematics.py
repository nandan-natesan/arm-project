"""!
Implements Forward and Inverse kinematics for the RX200 5-DOF arm.

Includes:
    - DH-parameter based Forward Kinematics  (FK_dh)   — currently active
    - Product of Exponentials FK             (FK_pox)  — alternative, see docstring
    - Geometric IK with tool-down / tool-flat auto-selection (IK_geometric)
    - Stacking IK with explicit psi control  (IK_geometric_stack)
    - Joint limit checking and feasibility search for stacking

Switching FK method:
    Both FK_dh and FK_pox compute the same end-effector pose. To switch:
    1. In rxarm.py  →  RXArm.get_ee_pose(), change FK_dh(...) to FK_pox(...)
    2. FK_pox reads M_matrix and S_list from config/rx200_pox.csv (loaded in RXArm.__init__)
    3. FK_dh uses hardcoded DH params with joint-angle offsets

Wrist angle pipeline (how the gripper aligns to a block):
    1. camera.py segment_blocks_watershed() calls cv2.minAreaRect() on each block
       contour, which returns an orientation angle in degrees.
    2. That angle_deg is passed through IK as block_angle_deg.
    3. In IK_geometric_stack (tool-down mode, psi ≈ -π/2):
         - angle is quantized to nearest 45° and converted to radians (ba)
         - theta5 (wrist roll) = ba + theta1  (base angle), then clamped to (-π, π]
       This rotates the gripper so its fingers align with the block's long axis.
    4. In IK_geometric (simpler version):
         - tool-down:  theta5 = theta1  (gripper aligned to world X)
         - tool-flat:  theta5 = 0
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Helpers: matrix exponential, screw vectors, angle clamping
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
#  Forward Kinematics — DH Parameters
# ═══════════════════════════════════════════════════════════════════════

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
    
    # RX200 DH table: [a (mm), alpha (rad), d (mm), theta_offset (rad)]
    # Rows: base→shoulder, shoulder→elbow, elbow→wrist_angle, wrist_angle→wrist_rotate, wrist→EE
    params = np.array([[0,       1.570796327, 103.91, 0         ],
                       [205.73,  0,           0,      1.3342    ],
                       [200,     0,           0,     -1.3342    ],
                       [0,       1.570796327, 0,      1.570796327],
                       [0,       0,           174.15, 0         ]])

    # Subtract motor angles from theta offsets (sign convention for this arm)
    params[:3, 3] -= joint_angles[:3]
    params[3, 3] -= joint_angles[3]
    params[4, 3] -= joint_angles[4]

    # Base frame rotation: aligns DH frame 0 with our world frame convention
    T_base = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
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


# ═══════════════════════════════════════════════════════════════════════
#  Forward Kinematics — Product of Exponentials (PoX)
#
#  Alternative to FK_dh. Uses screw axes at home configuration instead
#  of DH link parameters. To switch:
#    In rxarm.py RXArm.get_ee_pose(), replace:
#       ee_T = FK_dh(None, self.get_positions(), self.num_joints)
#    with:
#       ee_T = FK_pox(self.get_positions(), self.M_matrix, self.S_list)
#    M_matrix and S_list are loaded from config/rx200_pox.csv at init.
# ═══════════════════════════════════════════════════════════════════════

def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Product of Exponentials FK: T = e^([S1]*q1) * ... * e^([Sn]*qn) * M

    @param      joint_angles  (list or np.array) The joint angles [theta1, theta2, ...]
    @param      m_mat         (4x4 np.array) Home configuration matrix M
    @param      s_lst         (6xN np.array) Screw axes at home config (columns)

    @return     (4x4 np.array) Homogeneous transform for the end-effector
    """
    T = np.eye(4)
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

    T_final = np.dot(T_base, T_robot)
    
    return T_final



# ═══════════════════════════════════════════════════════════════════════
#  Inverse Kinematics — Geometric approach for 5-DOF RX200
# ═══════════════════════════════════════════════════════════════════════

def IK_geometric(pose, block_angle_deg):
    """!
    @brief      Produce joint configurations for provided pose

                Converts a desired end-effector pose vector as np.array to joint angles. Determines if tool down or flat is required based on reachability arc

    @param      pose       The desired pose vector as np.array [x, y, z]

    @return     Joint configuration in a numpy array 1x4 where each col is a joint angle
    """

    # geometry
    l1_nom = 103.91
    l1_bias = 1.5
    l1 = l1_nom + l1_bias
    l2 = 205.73
    l3 = 200.0
    l4 = 174.15

    # psi candidates for tool down or tool flat
    psi_down = -np.pi/2
    psi_flat = 0.0
    psi = psi_down  # set default

    # block angle
    block_angle_rad = np.deg2rad(block_angle_deg)

    #extract x, y, z from pose
    x_p, y_p, z_p = float(pose[0]), float(pose[1]), float(pose[2])

    # base yaw/ theta1
    theta1 = np.arctan2(-x_p, y_p)

    # 2D RR planar coords at shoulder
    x0 = np.hypot(x_p, y_p)
    y0 = z_p - l1

    # ----------------------------
    # Decide tool down or flat (psi) using reach of arm from base to wrist
    # ----------------------------
    # assign tolerance to rmax so boundary is slightly less than numerical boundary
    # prevents unwanted behaviour at numerical boundary
    rmax_nom = (l2 + l3) + 100
    rmax_tol = 0
    rmax = rmax_nom - rmax_tol

    # Tool-down wrist center + outer reach
    x_wc_down = x0 - l4 * np.cos(psi_down)
    y_wc_down = y0 - l4 * np.sin(psi_down)
    l_wc_down = np.hypot(x_wc_down, y_wc_down)

    # check if reach is within the tool-down arc and pose height to set psi
    # z pose limit will need tuning, or add further checks to segment out psi accordingly
    if l_wc_down >= rmax + 1e-6 or z_p >= 200 + 1e-6:
        psi = psi_flat
    else:
        psi = psi_down

    # ----------------------------
    # IK solve for chosen psi
    # ----------------------------

    # wrist center
    x_wc = x0 - l4 * np.cos(psi)
    y_wc = y0 - l4 * np.sin(psi)

    # reach
    l_wc2 = x_wc**2 + y_wc**2
    l_wc = np.sqrt(l_wc2)

    # reachability check
    if l_wc > (l2 + l3) + 1e-6 or l_wc < abs(l2 - l3) - 1e-6:
        return None

    # shoulder geometry alpha/ gamma1 / gamma2
    c = (l_wc2 - l2**2 - l3**2) / (2.0*l2*l3)
    c = np.clip(c, -1.0, 1.0)

    # elbow-up branch
    s = np.sqrt(max(0.0, 1.0 - c*c))
    alpha = np.arctan2(s, c)

    gamma1 = np.arctan2(y_wc, x_wc)
    gamma2 = np.arctan2(l3*np.sin(alpha), l2 + l3*np.cos(alpha))

    # shoulder and elbow / theta2, theta3
    theta2 = np.pi/2 - gamma1 - gamma2
    theta3 = alpha

    # wrist pitch / theta4
    theta4 = np.pi/2 - psi - theta2 - theta3

    # Wrist roll (theta5) — controls gripper orientation about the approach axis
    # Tool-flat: no roll needed (gripper horizontal)
    # Tool-down: set theta5 = theta1 to keep gripper aligned with world X axis
    #   To instead align with a detected block angle, use: theta5 = theta1 + block_angle_rad
    if psi == psi_flat:
        theta5 = 0.0
    else:
        theta5 = theta1
        theta5 = clamp(theta5)

    # Convert geometric to motor angles
    q1 = theta1
    q2 = theta2 - 0.245
    q3 = theta3 - 1.3258
    q4 = theta4
    q5 = theta5

    # return joint positions
    return np.array([q1, q2, q3, q4, q5], dtype=float)


# ═══════════════════════════════════════════════════════════════════════
#  Joint limits and feasibility search (used for stacking challenges)
# ═══════════════════════════════════════════════════════════════════════

def check_joint_limits(q):
    """Returns True if all motor angles are within RX200 hardware limits."""
    limits = [
        (-3.14,  3.14),
        (-1.885, 1.972),
        (-1.885, 1.623),
        (-1.745, 2.147),
        (-3.14,  3.14),
    ]
    for i, (lo, hi) in enumerate(limits):
        if q[i] < lo - 1e-4 or q[i] > hi + 1e-4:
            return False
    return True

def IK_geometric_stack(pose, psi, block_angle_deg=0.0, elbow_up=True):
    """
    IK with explicit approach angle psi and elbow configuration.
    psi = -pi/2 -> tool points straight down
    psi = 0     -> tool points horizontally forward
    """
    l1 = 103.91 + 1.5
    l2 = 205.73
    l3 = 200.0
    l4 = 174.15

    x_p, y_p, z_p = float(pose[0]), float(pose[1]), float(pose[2])
    theta1 = np.arctan2(-x_p, y_p)
    x0 = np.hypot(x_p, y_p)
    y0 = z_p - l1

    x_wc = x0 - l4 * np.cos(psi)
    y_wc = y0 - l4 * np.sin(psi)
    l_wc2 = x_wc**2 + y_wc**2
    l_wc = np.sqrt(l_wc2)

    if l_wc > (l2 + l3) + 1e-6 or l_wc < abs(l2 - l3) - 1e-6:
        return None

    c = (l_wc2 - l2**2 - l3**2) / (2.0 * l2 * l3)
    c = np.clip(c, -1.0, 1.0)
    s_val = np.sqrt(max(0.0, 1.0 - c * c))
    if not elbow_up:
        s_val = -s_val

    alpha = np.arctan2(s_val, c)
    gamma1 = np.arctan2(y_wc, x_wc)
    gamma2 = np.arctan2(l3 * np.sin(alpha), l2 + l3 * np.cos(alpha))

    theta2 = np.pi / 2 - gamma1 - gamma2
    theta3 = alpha
    theta4 = np.pi / 2 - psi - theta2 - theta3

    # Wrist roll: when tool is pointing down (psi ≈ -π/2) and a block angle is given,
    # align the gripper to the block's detected orientation.
    # The angle from cv2.minAreaRect is quantized to 45° steps for robustness,
    # then combined with base rotation (theta1) to produce a world-frame gripper angle.
    theta5 = 0.0
    if abs(psi - (-np.pi / 2)) < 0.1 and abs(block_angle_deg) > 1e-3:
        ba = np.deg2rad(round(block_angle_deg / 45.0) * 45.0 % 180.0)
        theta5 = ba + theta1
        theta5 = clamp(theta5)

    q = np.array([
        theta1,
        theta2 - 0.245,
        theta3 - 1.3258,
        theta4,
        theta5
    ], dtype=float)
    return q

def find_feasible_ik(pose, block_angle_deg=0.0, preferred_psi=-np.pi/2):
    """
    Tries psi values from preferred_psi toward 0, with both elbow configs.
    Returns (q, psi) for the most vertical feasible solution, or (None, None).
    """
    psi_candidates = np.linspace(preferred_psi, 0.0, 13)
    for psi in psi_candidates:
        for elbow_up in [True, False]:
            q = IK_geometric_stack(pose, psi, block_angle_deg, elbow_up)
            if q is not None and check_joint_limits(q):
                return q, psi
    return None, None


def compute_best_psi(xyz):
    """
    Find the psi closest to -pi/2 (straight down) that produces a fully
    valid IK solution (reachable AND within joint limits).
    Returns psi in [-pi/2, 0], or None if nothing works.
    """
    xyz = np.asarray(xyz, dtype=float)
    psi_candidates = np.linspace(-np.pi / 2, 0.0, 30)
    for psi in psi_candidates:
        for elbow_up in [True, False]:
            q = IK_geometric_stack(xyz, psi, 0.0, elbow_up)
            if q is not None and check_joint_limits(q):
                return psi
    return None


def compute_paired_psi(xyz_high, xyz_low):
    """
    Find the most vertical psi that gives a valid IK for BOTH points.
    Ensures approach and place use the same arm configuration so the
    descent is smooth with no wrist pitch change.
    """
    xyz_high = np.asarray(xyz_high, dtype=float)
    xyz_low = np.asarray(xyz_low, dtype=float)
    psi_candidates = np.linspace(-np.pi / 2, 0.0, 30)
    for psi in psi_candidates:
        ok_h = False
        ok_l = False
        for elbow_up in [True, False]:
            q = IK_geometric_stack(xyz_high, psi, 0.0, elbow_up)
            if q is not None and check_joint_limits(q):
                ok_h = True
                break
        for elbow_up in [True, False]:
            q = IK_geometric_stack(xyz_low, psi, 0.0, elbow_up)
            if q is not None and check_joint_limits(q):
                ok_l = True
                break
        if ok_h and ok_l:
            return psi
    return None

