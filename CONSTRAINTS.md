# 航线生成约束条件

## 无人机坐标系（前左上 Forward-Left-Up）
- **Forward**: 机头方向（由螃蟹飞决定：⊥ 航线方向）
- **Left**: 机头左侧（= Forward × Up）
- **Up**: 垂直向上 [0, 0, 1]

## 云台坐标系
- 与无人机坐标系相同（yaw=0, roll=0）
- 云台只能绕 Left 轴旋转（pitch）
- pitch > 0: 抬头（看向 Up 方向）
- pitch < 0: 低头（看向 -Up 方向）
- pitch = -90°: 垂直朝下（桥顶面典型值）

## 约束条件

### C1: 碰撞检测（安全距离）
- 无人机位置与点云/STL 的最近距离 ≥ `collision_dist`（安全距离 × 1.5）
- 包含拐角检测：90° 边缘处检查相邻两个面的距离

### C2: 云台 pitch 角度范围
- `-90° ≤ pitch ≤ +55°`
- 计算方法：
  ```
  los = target - drone_pos          # 无人机→目标视线
  fwd = heading（机头方向）           # Forward 轴
  los_fwd = dot(los, fwd)           # 视线在 Forward 轴的分量
  los_up  = dot(los, [0,0,1])       # 视线在 Up 轴的分量
  pitch = atan2(los_up, los_fwd)    # 正值=抬头，负值=低头
  ```
- **特殊情况**：桥顶面时 pitch = -90°（垂直朝下），机头方向 ⊥ 云台方向

### C3: 视线不穿模（LOS 无遮挡）
- 无人机→目标点连线中段（20%~70%处）不穿透模型表面
- 采样检测：沿视线等距采样，每点与点云最近距离 ≥ 0.3m

### C4: 螃蟹飞（机头方向约束）
- 机头方向 ⊥ 航线方向（在水平面上）
- `heading = cross(航线方向水平分量, [0,0,1])`
- 确保机头朝向目标一侧（dot(heading, 目标-无人机) > 0）

### C5: 云台 yaw=0, roll=0 约束
- 云台视线必须在"Forward + Up"构成的平面内
- 即：视线在 Left 轴的分量为 0
- `dot(los, left) = 0`（允许小误差 < 0.1）
- 违反时说明无人机位置需要沿航线方向调整

### C6: 巡检距离约束（优化目标）
- **硬约束**：`distance(drone, target) ≤ 设置的巡检距离`
- **优化目标**：在满足 C1~C5 的前提下，实际巡检距离尽量接近设置值
  - 巡检距离越小 → 覆盖范围越小 → 重叠率越高
  - 巡检距离越大 → 覆盖范围越大 → 重叠率越低
  - 实际巡检距离影响覆盖宽度：`cover = 2 × 实际距离 × tan(FOV/2)`
- 搜索策略：从 `inspect_dist` 开始搜索，找到满足 C1~C5 的最近位置
  - 优先尝试 `inspect_dist`（最近）
  - 不满足时逐步增大距离，直到找到可行解
  - 避免不必要的远距离（降低重叠率）

### C7: 重叠率约束
- 航点间距由覆盖宽度和旁向重叠率决定：
  ```
  cover = 2 × 实际巡检距离 × tan(有效FOV / 2)
  wp_spacing = cover × (1 - 旁向重叠率)
  ```
- 最小拍摄间隔限制实际间距：
  ```
  min_spacing = 飞行速度 × 最小拍摄间隔（M4T=0.7s）
  actual_spacing = max(wp_spacing, min_spacing)
  actual_overlap = 1 - actual_spacing / cover
  ```
- 若 `actual_overlap < 目标重叠率`，发出警告
