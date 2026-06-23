# 桥梁巡检无人机航线规划工具

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖包：
- `PyQt5 >= 5.15` — GUI 框架
- `numpy >= 1.20` — 数值计算
- `vtk >= 9.0` — 3D 可视化

## 运行

```bash
python main.py
```

## 打包为 Windows 可执行文件

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "BridgeRoutePlanner" main.py
```

生成的 exe 文件在 `dist/BridgeRoutePlanner.exe`

## 功能说明

### 1. 加载点云
- 点击 **"打开 PCD 文件"** 加载 `.pcd` 格式点云
- 支持 ASCII 和 Binary 格式的 PCD 文件
- 点云加载后自动根据范围填充默认航线参数

### 2. 平面航线（桥底面弓字形扫描）
- 设置扫描区域 X/Y 范围、飞行高度、行间距、速度
- 生成弓字形（蛇形）航线，适用于桥底面巡检

### 3. 立方体航线（桥柱环绕扫描）
- 设置柱体中心坐标、尺寸、离柱面距离、步长
- **螺旋线**：绕柱体螺旋上升扫描
- **Z字形**：逐层上下扫描

### 4. 航线管理
- **保存航线**：导出为 JSON 格式，包含位置、四元数、速度
- **加载航线**：从 JSON 文件恢复航线
- **清除航线**：清除当前航线（保留点云）

## JSON 航线格式

```json
{
  "version": "1.0",
  "description": "桥梁巡检航线",
  "waypoint_count": 100,
  "waypoints": [
    {
      "position": { "x": 1.234, "y": 5.678, "z": 10.0 },
      "quaternion": { "w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0 },
      "speed": 3.0,
      "action": "fly"
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `position` | object | 航点三维坐标 (米) |
| `quaternion` | object | 姿态四元数 (w, x, y, z)，描述无人机朝向 |
| `speed` | float | 飞行速度 (m/s) |
| `action` | string | 动作类型: `fly`(飞行) / `scan`(扫描) |

## 配置文件 config.json

外参标定参数存放在 `config.json`，用于保存航线时将四元数从地图坐标系转换到雷达 IMU 坐标系。

```json
{
  "lidar_to_dji_imu_rotation": [
    [-0.839384, 0.0720621, 0.538741],
    [0.0487258, 0.997158, -0.0574631],
    [-0.541351, -0.021983, -0.840509]
  ]
}
```

| 参数 | 说明 |
|------|------|
| `lidar_to_dji_imu_rotation` | 3x3 旋转矩阵，雷达 IMU 到 DJI IMU 的坐标变换 |

### 坐标系说明

- 界面展示使用**地图坐标系**（方便预览）
- 导出 JSON 中的四元数使用**雷达 IMU 坐标系**（供飞控使用）
- 转换公式：`R_lidar = R_map @ lidar_to_dji_imu_R.T`
