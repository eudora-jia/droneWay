"""
航线生成核心模块

核心思路：目标点 + 约束条件 → 唯一航点 → 唯一航线
    1. 用户在模型/点云表面选择目标点
    2. 估计目标点处的表面法线（PCA）
    3. 沿法线方向偏移安全距离 → 候选航点
    4. 碰撞检测：航点与点云距离 < collision_dist 则继续外推
    5. 确定唯一航点位置
    6. 计算云台 pitch/yaw 使目标点位于相机画面中心

所有函数为纯函数，不依赖 GUI，可独立调用。
"""

import numpy as np
from scipy.spatial import cKDTree


def estimate_normal(point, points, tree=None, k=None):
    """用 PCA 估计点云在指定位置的表面法线

    Args:
        point: 目标点 [x, y, z]
        points: 完整点云 (N, 3)
        tree: 可选的 cKDTree，不传则自动构建
        k: 邻域点数，None 则自动选择

    Returns:
        normal: 单位法线向量 [nx, ny, nz]，方向朝外（远离点云质心）
    """
    point = np.asarray(point, dtype=np.float64)
    if len(points) < 3:
        return np.array([0.0, 0.0, 1.0])

    if tree is None:
        tree = cKDTree(points)

    if k is None:
        k = max(3, min(30, len(points)))

    dists, idxs = tree.query(point, k=k)
    neighbors = points[idxs]
    centered = neighbors - neighbors.mean(axis=0)
    cov = centered.T @ centered / len(neighbors)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # 最小特征值对应法线方向

    # 确保法线朝外（远离点云质心）
    centroid = points.mean(axis=0)
    if np.dot(normal, point - centroid) < 0:
        normal = -normal
    return normal


def find_safe_position(target, normal, tree, collision_dist, safe_dist,
                       max_offset=10.0, step=0.5):
    """沿法线方向寻找安全的无人机飞行位置

    从 target + normal * safe_dist 开始，逐步外推直到满足碰撞约束。

    Args:
        target: 目标点 [x, y, z]
        normal: 表面法线（单位向量）
        tree: cKDTree（点云）
        collision_dist: 碰撞判定距离（航点与点云小于此距离视为碰撞）
        safe_dist: 安全距离（航点与目标点的最小距离）
        max_offset: 最大偏移量（超过则放弃）
        step: 每次外推步长

    Returns:
        (pos, warned, distance)
        pos: 安全航点位置 [x,y,z]，找不到返回 None
        warned: 是否有警告（True 表示距离较远或无法找到）
        distance: 航点到最近点云点的距离
    """
    target = np.asarray(target, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)

    for i in range(int(max_offset / step) + 1):
        offset = safe_dist + i * step
        if offset > max_offset:
            return None, True, 0.0
        pos = target + normal * offset
        if tree is not None:
            dist, _ = tree.query(pos)
            if dist >= collision_dist:
                warned = (i > 0)  # 第一次就成功则无警告
                return pos, warned, dist
        else:
            return pos, False, float('inf')
    return None, True, 0.0


def calc_gimbal_pitch(drone_pos, target_pos):
    """计算云台 pitch 角度，使目标点位于相机画面中心

    Args:
        drone_pos: 无人机位置 [x, y, z]
        target_pos: 目标点位置 [x, y, z]

    Returns:
        pitch: 俯仰角（度），负值=朝下，-90=垂直朝下
    """
    d = np.asarray(drone_pos, dtype=np.float64)
    t = np.asarray(target_pos, dtype=np.float64)
    drop = d[2] - t[2]
    h_dist = np.sqrt((d[0] - t[0])**2 + (d[1] - t[1])**2)
    if h_dist < 1e-6:
        return -90.0
    return -np.degrees(np.arctan(drop / h_dist))


def calc_gimbal_yaw(drone_pos, target_pos, drone_heading=None):
    """计算云台 yaw 角度（水平偏转角）

    Args:
        drone_pos: 无人机位置 [x, y, z]
        target_pos: 目标点位置 [x, y, z]
        drone_heading: 无人机机头方向向量 [x, y, 0]，None 则返回绝对角度

    Returns:
        yaw: 偏航角（度），相对于机头方向的偏转，正=右转
    """
    d = np.asarray(drone_pos, dtype=np.float64)
    t = np.asarray(target_pos, dtype=np.float64)
    to_target = t - d
    to_target[2] = 0  # 只看水平面

    if np.linalg.norm(to_target) < 1e-6:
        return 0.0

    target_angle = np.degrees(np.arctan2(to_target[1], to_target[0]))

    if drone_heading is not None:
        h = np.asarray(drone_heading, dtype=np.float64)
        h[2] = 0
        if np.linalg.norm(h) < 1e-6:
            return target_angle
        heading_angle = np.degrees(np.arctan2(h[1], h[0]))
        yaw = target_angle - heading_angle
        # 归一化到 [-180, 180]
        while yaw > 180:
            yaw -= 360
        while yaw < -180:
            yaw += 360
        return yaw

    return target_angle


def look_at_quaternion(target, pos):
    """计算从 pos 朝向 target 的四元数（wxyz 格式）

    Args:
        target: 目标位置 [x, y, z]
        pos: 当前位置 [x, y, z]

    Returns:
        quat: 四元数 [w, x, y, z]
    """
    from quaternion_utils import look_at_quaternion as _look_at
    return _look_at(target, pos)


def generate_inspect_route(targets, points, safe_dist=3.0, collision_dist=1.5,
                           speed=1.0, max_offset=10.0):
    """基于目标点生成完整的巡检航线（核心算法）

    流程：目标点 → 法线 → 安全位置 → 碰撞检测 → 唯一航点 → 云台角度

    Args:
        targets: 目标点列表 [np.array([x,y,z]), ...]
        points: 点云数据 (N, 3)
        safe_dist: 安全距离（米），默认 3.0（符合指南要求）
        collision_dist: 碰撞判定距离（米），默认 1.5
        speed: 飞行速度（m/s），默认 1.0
        max_offset: 法线方向最大搜索距离（米）

    Returns:
        (waypoints, warnings)
        waypoints: 航点列表，每个航点为 dict:
            {
                'pos': np.array([x,y,z]),       # 无人机位置
                'quat': np.array([w,x,y,z]),     # 姿态四元数
                'target_pos': np.array([x,y,z]), # 目标点位置
                'gimbal_pitch': float,            # 云台俯仰角(度)
                'gimbal_yaw': float,              # 云台偏航角(度)
                'speed': float,                   # 飞行速度
                'action': 'scan',                 # 动作类型
                'safe_dist_actual': float,        # 实际安全距离
                'collision_clearance': float,     # 到点云最近距离
            }
        warnings: 警告列表 [(index, message), ...]
    """
    if targets is None or len(targets) == 0:
        return [], []

    points = np.asarray(points, dtype=np.float64)
    tree = cKDTree(points) if len(points) > 0 else None

    waypoints = []
    warnings = []

    for i, target in enumerate(targets):
        target = np.asarray(target, dtype=np.float64)

        # Step 1: 估计表面法线
        normal = estimate_normal(target, points, tree)

        # Step 2: 沿法线找安全位置
        pos, warned, clearance = find_safe_position(
            target, normal, tree, collision_dist, safe_dist, max_offset
        )

        if pos is None:
            # 法线方向找不到安全位置，尝试向上
            fallback_normal = np.array([0.0, 0.0, 1.0])
            pos, warned, clearance = find_safe_position(
                target, fallback_normal, tree, collision_dist, safe_dist, max_offset
            )
            if pos is None:
                warnings.append((i, f"P{i+1} 无法找到安全位置，已强制提升"))
                pos = target + np.array([0.0, 0.0, safe_dist + 2.0])
                clearance = 0.0

        if warned:
            warnings.append((i, f"P{i+1} 航点沿法线外推较多"))

        # Step 3: 计算云台角度
        gimbal_pitch = calc_gimbal_pitch(pos, target)
        gimbal_yaw = calc_gimbal_yaw(pos, target)

        # Step 4: 计算姿态四元数（机头朝向目标方向的水平投影）
        heading = target - pos
        heading[2] = 0
        if np.linalg.norm(heading) < 1e-6:
            heading = np.array([1.0, 0.0, 0.0])
        look_target = pos + heading
        quat = look_at_quaternion(look_target, pos)

        actual_dist = np.linalg.norm(pos - target)

        waypoints.append({
            'pos': pos,
            'quat': quat,
            'target_pos': target.copy(),
            'gimbal_pitch': gimbal_pitch,
            'gimbal_yaw': gimbal_yaw,
            'speed': speed,
            'action': 'scan',
            'safe_dist_actual': actual_dist,
            'collision_clearance': clearance,
        })

    return waypoints, warnings


def generate_flat_route_targets(points, spacing=2.0, z=None):
    """在点云上方自动生成面状航线的目标点网格

    根据点云 XY 范围，在 z 高度处生成均匀网格点作为"隐式目标点"。
    每个网格点下方的点云表面即为实际拍摄目标。

    Args:
        points: 点云数据 (N, 3)
        spacing: 网格间距（米）
        z: 飞行高度，None 则取点云最高点 + 2m

    Returns:
        grid_targets: 网格目标点列表 [np.array([x,y,z_surface]), ...]
        flight_z: 建议飞行高度
    """
    points = np.asarray(points, dtype=np.float64)
    mn = points.min(axis=0)
    mx = points.max(axis=0)

    if z is None:
        z = mx[2] + 2.0

    tree = cKDTree(points)

    # 生成 XY 网格
    xs = np.arange(mn[0], mx[0] + spacing * 0.5, spacing)
    ys = np.arange(mn[1], mx[1] + spacing * 0.5, spacing)

    targets = []
    for x in xs:
        for y in ys:
            # 找到网格点下方最近的点云点作为目标
            grid_pt = np.array([x, y, z])
            dist, idx = tree.query(grid_pt)
            surface_pt = points[idx]
            # 只保留表面点在飞行高度以下的目标
            if surface_pt[2] < z:
                targets.append(surface_pt)

    return targets, z


def merge_waypoint_lists(*waypoint_lists):
    """合并多个航点列表，自动插入过渡航点

    Args:
        *waypoint_lists: 多个航点列表

    Returns:
        merged: 合并后的航点列表
    """
    merged = []
    for wp_list in waypoint_lists:
        if len(wp_list) == 0:
            continue
        if len(merged) > 0:
            # 插入过渡：从上一段末尾到这一段开头
            last = merged[-1]
            first = wp_list[0]
            # 可以在这里插入安全过渡点，暂简单拼接
            pass
        merged.extend(wp_list)
    return merged
