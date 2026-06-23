"""四元数与旋转矩阵工具函数"""

import numpy as np


def rotation_matrix_from_vectors(forward, up):
    """从前进方向和上方向构建旋转矩阵 (3x3)
    旋转矩阵将 body 坐标系 (X=forward, Y=left, Z=up) 映射到世界坐标系
    """
    f = forward / (np.linalg.norm(forward) + 1e-10)
    u = up / (np.linalg.norm(up) + 1e-10)
    if abs(np.dot(f, u)) > 0.99:
        u = np.array([0.0, 1.0, 0.0])
    l = np.cross(u, f)  # left = up × forward
    l /= (np.linalg.norm(l) + 1e-10)
    u = np.cross(f, l)  # 重新计算确保正交
    return np.column_stack([f, l, u])  # 列: [forward, left, up]


def rotation_matrix_to_quaternion(m):
    """旋转矩阵 -> 四元数 [w, x, y, z]"""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return np.array([
            s / 4.0,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s
        ])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return np.array([
            (m[2, 1] - m[1, 2]) / s,
            s / 4.0,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s
        ])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return np.array([
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            s / 4.0,
            (m[1, 2] + m[2, 1]) / s
        ])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        return np.array([
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            s / 4.0
        ])


def look_at_quaternion(target, position):
    """计算从 position 指向 target 的四元数，保持水平（只绕 Z 轴旋转）"""
    forward = target - position
    if np.linalg.norm(forward) < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    forward = forward.copy()
    forward[2] = 0  # 保持水平，只绕 Z 轴旋转
    if np.linalg.norm(forward) < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    mat = rotation_matrix_from_vectors(forward, up)
    return rotation_matrix_to_quaternion(mat)


def quaternion_forward(quat):
    """从四元数 [w, x, y, z] 计算前方向量（X 轴正方向旋转后的结果）"""
    w, x, y, z = quat
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    fz = 2.0 * (x * z - w * y)
    v = np.array([fx, fy, fz])
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else np.array([1.0, 0.0, 0.0])


# 雷达 IMU 到 DJI IMU 的旋转矩阵
LIDAR_TO_DJI_R = np.array([
    [-0.839384, 0.0720621, 0.538741],
    [0.0487258, 0.997158, -0.0574631],
    [-0.541351, -0.021983, -0.840509]
])


def quat_to_matrix(q):
    """四元数 [w, x, y, z] -> 旋转矩阵 3x3"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def quat_map_to_lidar(quat_map):
    """将四元数从地图坐标系转换到雷达 IMU 坐标系
    R_lidar = R_map @ LIDAR_TO_DJI_R^T
    参数: quat_map - [w, x, y, z] 地图坐标系四元数
    返回: [w, x, y, z] 雷达 IMU 坐标系四元数
    """
    R_map = quat_to_matrix(quat_map)
    R_lidar = R_map @ LIDAR_TO_DJI_R.T
    return rotation_matrix_to_quaternion(R_lidar)
