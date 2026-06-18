"""
桥梁巡检无人机航线规划工具
Bridge Inspection Drone Waypoint Planner

功能：
- 加载 PCD 点云文件并 3D 可视化
- 设计平面航线（桥底面弓字形扫描）
- 设计立方体航线（桥柱螺旋线/Z字形扫描）
- 航点含位置(xyz) + 四元数(wxyz) + 速度
- 导出/导入 JSON 航线文件
- 交互式画框选择扫描区域
"""

import sys
import json
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QSplitter, QSizePolicy, QMenu,
    QProgressBar, QCheckBox, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

# ─── VTK 导入（兼容多个版本）────────────────────────────────
try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkRenderingCore import (
        vtkActor, vtkPolyDataMapper, vtkRenderer,
        vtkPoints, vtkPolyData, vtkVertexGlyphFilter,
        vtkFollower, vtkVectorText
    )
    from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkLineSource
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyLine
    from vtkmodules.vtkCommonCore import vtkPoints as vtkPointsBase
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
    VTK_AVAILABLE = True
except ImportError:
    try:
        import vtk
        from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        from vtk.util.numpy_support import numpy_to_vtk
        VTK_AVAILABLE = True
        # 旧版 VTK 用 vtk.vtkXxx 风格
        from vtk import (
            vtkActor, vtkPolyDataMapper, vtkRenderer,
            vtkPoints, vtkPolyData, vtkVertexGlyphFilter,
            vtkSphereSource, vtkLineSource, vtkCellArray,
            vtkPolyLine, vtkFollower, vtkVectorText,
            vtkTransform, vtkTransformPolyDataFilter
        )
    except ImportError:
        VTK_AVAILABLE = False
        print("[WARNING] VTK not installed. Run: pip install vtk")


# ─── Foxglove 风格交互：左键平移，右键旋转 ───────────────────
if VTK_AVAILABLE:
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

    class FoxgloveInteractorStyle(vtkInteractorStyleTrackballCamera):
        """仿 Foxglove 交互风格：左键=平移，右键=旋转，滚轮=缩放"""

        def __init__(self):
            super().__init__()
            self._vtk_viewer = None
            self._rotating = False
            self._last_x = 0
            self._last_y = 0

        def set_viewer(self, viewer):
            self._vtk_viewer = viewer

        # ── 左键：平移（Pan）──
        def OnLeftButtonDown(self):
            if self._vtk_viewer and getattr(self._vtk_viewer, '_pick_drag_active', False):
                return
            self.StartPan()

        def OnLeftButtonUp(self):
            self.EndPan()

        # ── 右键：旋转（手动实现，避免 VTK 事件冲突）──
        def OnRightButtonDown(self):
            rwi = self.GetInteractor()
            self._last_x, self._last_y = rwi.GetEventPosition()
            self._rotating = True

        def OnRightButtonUp(self):
            self._rotating = False

        def OnMouseMove(self):
            if self._rotating:
                rwi = self.GetInteractor()
                renderer = self.GetCurrentRenderer()
                if renderer is None:
                    self.FindPokedRenderer(rwi.GetEventPosition()[0], rwi.GetEventPosition()[1])
                    renderer = self.GetCurrentRenderer()
                if renderer is None:
                    return

                cur_x, cur_y = rwi.GetEventPosition()
                dx = cur_x - self._last_x
                dy = cur_y - self._last_y
                self._last_x, self._last_y = cur_x, cur_y

                camera = renderer.GetActiveCamera()
                camera.Azimuth(-dx * 0.4)
                camera.Elevation(dy * 0.4)
                camera.OrthogonalizeViewUp()
                renderer.ResetCameraClippingRange()
                rwi.Render()
                return

            super().OnMouseMove()


# ─── PCD 文件解析 ────────────────────────────────────────────
def parse_pcd(filepath):
    """解析 PCD 文件，返回 numpy 数组 (N, 3) float64"""
    header = {}
    data_start = 0

    with open(filepath, 'rb') as f:
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('DATA'):
                header['DATA'] = line.split()[1]
                data_start = f.tell()
                break
            if line:
                parts = line.split()
                header[parts[0]] = parts[1:]

    fields = header.get('FIELDS', ['x', 'y', 'z'])
    sizes = header.get('SIZE', ['4'] * len(fields))
    types = header.get('TYPE', ['F'] * len(fields))
    width = int(header.get('WIDTH', ['0'])[0])
    height = int(header.get('HEIGHT', ['1'])[0])
    num_points = int(header.get('POINTS', [str(width * height)])[0])

    if num_points == 0:
        return np.empty((0, 3), dtype=np.float64)

    xyz_indices = []
    for i, f in enumerate(fields):
        if f.lower() in ('x', 'y', 'z'):
            xyz_indices.append(i)

    if len(xyz_indices) != 3:
        xyz_indices = [0, 1, 2]

    if header['DATA'] == 'ascii':
        with open(filepath, 'rb') as f:
            f.seek(data_start)
            raw = f.read().decode('ascii', errors='ignore')
        points = []
        for line in raw.strip().split('\n'):
            vals = line.strip().split()
            if len(vals) >= 3:
                points.append([float(vals[i]) for i in xyz_indices])
        return np.array(points, dtype=np.float64) if points else np.empty((0, 3))

    elif header['DATA'] == 'binary':
        with open(filepath, 'rb') as f:
            f.seek(data_start)
            raw = f.read()

        dtypes = []
        for s, t in zip(sizes, types):
            s = int(s)
            if t.upper() == 'F':
                dtypes.append(('f{}'.format(len(dtypes)), np.float32 if s == 4 else np.float64))
            elif t.upper() == 'U':
                dtypes.append(('f{}'.format(len(dtypes)), {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[s]))
            else:
                dtypes.append(('f{}'.format(len(dtypes)), {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[s]))

        structured = np.frombuffer(raw, dtype=np.dtype(dtypes), count=num_points)
        xyz = np.zeros((num_points, 3), dtype=np.float64)
        for j, idx in enumerate(xyz_indices):
            xyz[:, j] = structured['f{}'.format(idx)].astype(np.float64)
        return xyz

    return np.empty((0, 3), dtype=np.float64)


# ─── 法向量估算（PCA）──────────────────────────────────────
def estimate_normals(points, k=30):
    """
    用 PCA 估算点云法向量
    points: (N, 3) 数组
    k: 每个点取 k 个最近邻做法向量估算
    返回: normals (N, 3) 单位法向量
    """
    from scipy.spatial import cKDTree
    n = len(points)
    normals = np.zeros((n, 3))

    tree = cKDTree(points)
    dists, indices = tree.query(points, k=k)

    for i in range(n):
        neighbors = points[indices[i]]
        centroid = neighbors.mean(axis=0)
        cov = np.cov((neighbors - centroid).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 最小特征值对应的特征向量就是法向量
        normals[i] = eigenvectors[:, 0]

    # 统一法向量朝向（确保都朝同一侧，这里强制朝下，因为桥底面法向量朝下）
    # 通过检查法向量 Z 分量来统一
    for i in range(n):
        if normals[i, 2] > 0:
            normals[i] = -normals[i]  # 翻转朝下

    return normals


def nearest_neighbor_path(points):
    """
    贪心最近邻排序：将散乱点排成一条连续路径
    points: (N, 3) 数组
    返回: 排序后的索引数组
    """
    from scipy.spatial import cKDTree
    n = len(points)
    if n <= 1:
        return np.arange(n)

    tree = cKDTree(points)
    visited = np.zeros(n, dtype=bool)
    order = []

    # 从左下角点开始
    start = np.argmin(points[:, 0] + points[:, 1])
    order.append(start)
    visited[start] = True

    for _ in range(n - 1):
        current = order[-1]
        # 找最近的未访问点
        dists, indices = tree.query(points[current], k=n)
        for idx in indices:
            if not visited[idx]:
                order.append(idx)
                visited[idx] = True
                break

    return np.array(order)


# ─── 四元数工具 ──────────────────────────────────────────────
def rotation_matrix_from_vectors(forward, up):
    """从前进方向和上方向构建旋转矩阵 (3x3)，右手坐标系"""
    f = forward / (np.linalg.norm(forward) + 1e-10)
    u = up / (np.linalg.norm(up) + 1e-10)
    r = np.cross(u, f)  # right = up x forward（右手系）
    r /= (np.linalg.norm(r) + 1e-10)
    u = np.cross(f, r)  # 重新计算确保正交
    return np.column_stack([r, u, f])


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


# ─── 3D 可视化组件 ───────────────────────────────────────────
class VTKViewer(QWidget):
    """嵌入 PyQt5 的 VTK 3D 点云/航线可视化组件，支持交互式画框选点"""

    # 选框完成信号：[[min_x,min_y,min_z], [max_x,max_y,max_z]]
    box_selected = pyqtSignal(list)
    # 航点编辑完成信号：(航点索引, 新位置 ndarray, 新朝向 ndarray or None)
    waypoint_edited = pyqtSignal(int, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        if not VTK_AVAILABLE:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("VTK not installed.\nRun: pip install vtk"))
            return

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.renderer.SetBackground(0.15, 0.15, 0.18)

        self._actors = []
        self.points_data = None
        self._cloud_actor = None  # 点云 actor 引用

        # ─── 选框模式状态 ───
        self.pick_mode = False      # 是否处于画框模式
        self._dragging = False      # 是否正在拖拽
        self._drag_start = None     # 拖拽起点 (3D)
        self._pick_drag_active = False  # 画框拖拽已激活（阻止 interactor style 平移）
        self.box_actors = []        # 选框线框 actor
        self._observers = []        # 事件观察者 ID

        # ─── 航点编辑状态 ───
        self._wp_editing = False        # 是否正在编辑航点
        self._wp_edit_idx = -1          # 编辑中的航点索引
        self._wp_edit_z = 0.0           # 拖拽平面 Z 高度
        self._wp_edit_offset = None     # 拖拽偏移量
        self._wp_edit_actor = None      # 被编辑航点的 actor（高亮用）
        self._waypoint_actors = []      # 航点 actor 列表（用于拾取）
        self._waypoints_ref = None      # 航点数据引用
        self._safe_distance = 2.0       # 安全距离阈值（米）
        self.show_heading = True        # 是否显示机头方向箭头

        # ─── 点拾取器 ───
        self.picker = vtk.vtkPointPicker() if VTK_AVAILABLE else None

        self._timer_id = self.startTimer(30)

    def timerEvent(self, event):
        if VTK_AVAILABLE and self.interactor:
            self.interactor.ProcessEvents()

    def clear_actors(self):
        for actor in self._actors:
            self.renderer.RemoveActor(actor)
        self._actors.clear()
        self._cloud_actor = None

    def add_point_cloud(self, points):
        """显示点云（按高度着色，大点云自动降采样渲染）"""
        if not VTK_AVAILABLE or len(points) == 0:
            return
        self.clear_actors()
        self.points_data = points  # 保留完整数据用于拾取和分析

        # ─── 大点云降采样渲染（超过 500 万点时）───
        MAX_RENDER_POINTS = 5_000_000
        if len(points) > MAX_RENDER_POINTS:
            # 体素降采样：按网格取重心
            voxel_size = self._estimate_voxel_size(points, MAX_RENDER_POINTS)
            render_points = self._voxel_downsample(points, voxel_size)
            print(f"[VTK] Downsampled {len(points)} -> {len(render_points)} points (voxel={voxel_size:.2f})")
        else:
            render_points = points

        vtk_points = vtkPoints()
        vtk_array = numpy_to_vtk(render_points.astype(np.float64), deep=True)
        vtk_array.SetName('Points')
        vtk_points.SetData(vtk_array)

        polydata = vtkPolyData()
        polydata.SetPoints(vtk_points)

        # ─── 高度着色（向量化计算，替代逐点循环）───
        z_vals = render_points[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        z_range = z_max - z_min if z_max > z_min else 1.0

        t = (z_vals - z_min) / z_range  # 归一化到 [0, 1]

        # 分段计算 RGB（纯 numpy 向量化）
        r = np.zeros(len(t), dtype=np.uint8)
        g = np.zeros(len(t), dtype=np.uint8)
        b = np.zeros(len(t), dtype=np.uint8)

        mask = t < 0.25
        g[mask] = (t[mask] * 4 * 255).astype(np.uint8)
        b[mask] = 255

        mask = (t >= 0.25) & (t < 0.5)
        g[mask] = 255
        b[mask] = ((0.5 - t[mask]) * 4 * 255).astype(np.uint8)

        mask = (t >= 0.5) & (t < 0.75)
        r[mask] = ((t[mask] - 0.5) * 4 * 255).astype(np.uint8)
        g[mask] = 255

        mask = t >= 0.75
        r[mask] = 255
        g[mask] = ((1.0 - t[mask]) * 4 * 255).astype(np.uint8)

        # 合并为 (N, 3) 数组，直接设置到 VTK
        rgb = np.column_stack([r, g, b]).astype(np.uint8)
        vtk_colors = numpy_to_vtk(rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_colors.SetName('Colors')
        polydata.GetPointData().SetScalars(vtk_colors)

        glyph = vtkVertexGlyphFilter()
        glyph.SetInputData(polydata)
        glyph.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToDefault()

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(2)

        self.renderer.AddActor(actor)
        self._actors.append(actor)
        self._cloud_actor = actor

        self.renderer.ResetCamera()
        self._update_view()
        print(f"[VTK] Point cloud loaded: {len(points)} points (rendered: {len(render_points)})")

    @staticmethod
    def _estimate_voxel_size(points, target_count):
        """估算体素大小，使降采样后点数接近 target_count"""
        # 用包围盒体积估算
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        volume = np.prod(mx - mn + 1e-10)
        # 体素体积 = 总体积 / 目标点数
        voxel_vol = volume / target_count
        voxel_size = voxel_vol ** (1.0 / 3.0)
        return max(voxel_size, 0.01)

    @staticmethod
    def _voxel_downsample(points, voxel_size):
        """体素网格降采样：每个体素取重心代表点（纯 numpy，无 Python 循环）"""
        mn = points.min(axis=0)
        # 量化到体素网格坐标
        voxel_idx = ((points - mn) / voxel_size).astype(np.int64)

        # 编码为唯一整数键
        max_idx = voxel_idx.max(axis=0) + 1
        keys = voxel_idx[:, 0] + voxel_idx[:, 1] * max_idx[0] + voxel_idx[:, 2] * max_idx[0] * max_idx[1]

        # 按键排序
        sort_order = np.argsort(keys)
        sorted_keys = keys[sort_order]
        sorted_points = points[sort_order]

        # 找到每个体素的起始位置
        boundaries = np.concatenate([[0], np.where(np.diff(sorted_keys))[0] + 1, [len(sorted_keys)]])

        # 用 reduceat 计算每个体素的点数和坐标总和
        counts = np.diff(boundaries).astype(np.float64)
        sums_x = np.add.reduceat(sorted_points[:, 0], boundaries[:-1])
        sums_y = np.add.reduceat(sorted_points[:, 1], boundaries[:-1])
        sums_z = np.add.reduceat(sorted_points[:, 2], boundaries[:-1])

        result = np.column_stack([sums_x, sums_y, sums_z]) / counts[:, None]
        return result

    @staticmethod
    def _compute_perpendicular_headings(waypoints):
        """计算每个航点垂直于航线的朝向（朝向桥面）。

        直行段：飞行方向旋转90°得到垂直方向
        拐角处：取前后两段垂直方向的平均值，朝向Z形内侧
        """
        n = len(waypoints)
        positions = [wp['pos'] for wp in waypoints]
        headings = []

        # 先计算每段的飞行方向和垂直方向
        seg_perps = []
        for i in range(n - 1):
            d = positions[i + 1] - positions[i]
            norm = np.linalg.norm(d)
            if norm < 1e-10:
                seg_perps.append(np.array([0.0, 1.0, 0.0]))
                continue
            d = d / norm
            # 垂直方向：飞行方向在 XY 平面旋转 90°
            # 同时考虑 Z 分量的影响
            perp = np.array([-d[1], d[0], 0.0])
            pnorm = np.linalg.norm(perp)
            if pnorm < 1e-10:
                # 飞行方向是纯 Z 方向，用 X 轴作为垂直
                perp = np.array([1.0, 0.0, 0.0])
            else:
                perp = perp / pnorm
            seg_perps.append(perp)

        for i in range(n):
            if n == 1:
                headings.append(np.array([0.0, 1.0, 0.0]))
            elif i == 0:
                # 第一个点：用第一段的垂直方向
                headings.append(seg_perps[0])
            elif i == n - 1:
                # 最后一个点：用最后一段的垂直方向
                headings.append(seg_perps[-1])
            else:
                # 中间点：取前后两段垂直方向的平均（处理拐角）
                avg = seg_perps[i - 1] + seg_perps[i]
                norm = np.linalg.norm(avg)
                if norm < 1e-10:
                    # 前后垂直方向相反（U 型转弯），用第一段
                    headings.append(seg_perps[i])
                else:
                    headings.append(avg / norm)

        return headings

    @staticmethod
    def _is_corner(waypoints, idx):
        """判断航点是否在拐角处（飞行方向发生显著变化）"""
        n = len(waypoints)
        if idx == 0 or idx == n - 1:
            return False
        d1 = waypoints[idx]['pos'] - waypoints[idx - 1]['pos']
        d2 = waypoints[idx + 1]['pos'] - waypoints[idx]['pos']
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        if n1 < 1e-10 or n2 < 1e-10:
            return False
        cos_angle = np.dot(d1, d2) / (n1 * n2)
        # 夹角大于 30° 视为拐角
        return cos_angle < np.cos(np.radians(30))

    def add_route(self, waypoints):
        """显示航线和航点"""
        if not VTK_AVAILABLE or len(waypoints) == 0:
            return

        # 清除旧的航线 actor（保留点云和选框）
        to_remove = []
        for i, actor in enumerate(self._actors):
            if actor != self._cloud_actor and actor not in self.box_actors:
                to_remove.append(i)
        for i in reversed(to_remove):
            self.renderer.RemoveActor(self._actors[i])
            del self._actors[i]

        n = len(waypoints)

        # ─── 航线路径（绿色线）───
        vtk_pts = vtkPoints()
        for wp in waypoints:
            vtk_pts.InsertNextPoint(wp['pos'].tolist())

        polyline = vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            polyline.GetPointIds().SetId(i, i)

        cells = vtkCellArray()
        cells.InsertNextCell(polyline)

        polydata = vtkPolyData()
        polydata.SetPoints(vtk_pts)
        polydata.SetLines(cells)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 1.0, 0.3)
        actor.GetProperty().SetLineWidth(2.5)
        self.renderer.AddActor(actor)
        self._actors.append(actor)

        # ─── 航点标记（红色球 + 编号标签）───
        self._waypoint_actors = []
        self._waypoints_ref = waypoints
        # 计算标签偏移量（基于航点间距）
        if n >= 2:
            avg_d = np.mean([np.linalg.norm(waypoints[i+1]['pos'] - waypoints[i]['pos']) for i in range(n-1)])
            label_offset = avg_d * 0.15
        else:
            label_offset = 0.3

        for i, wp in enumerate(waypoints):
            sphere = vtkSphereSource()
            sphere.SetCenter(wp['pos'].tolist())
            sphere.SetRadius(0.15)
            sphere.Update()

            m = vtkPolyDataMapper()
            m.SetInputConnection(sphere.GetOutputPort())
            a = vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetColor(1.0, 0.2, 0.2)
            self.renderer.AddActor(a)
            self._actors.append(a)
            self._waypoint_actors.append(a)

            # 航点编号标签
            txt_src = vtkVectorText()
            txt_src.SetText(str(i + 1))
            txt_m = vtkPolyDataMapper()
            txt_m.SetInputConnection(txt_src.GetOutputPort())
            follower = vtkFollower()
            follower.SetMapper(txt_m)
            follower.SetScale(label_offset, label_offset, label_offset)
            follower.SetPosition(wp['pos'][0], wp['pos'][1], wp['pos'][2] + label_offset * 1.5)
            follower.GetProperty().SetColor(1.0, 1.0, 0.2)  # 黄色编号
            follower.SetCamera(self.renderer.GetActiveCamera())
            self.renderer.AddActor(follower)
            self._actors.append(follower)

        # ─── 机头方向箭头（可选显示）───
        if getattr(self, 'show_heading', True) and n >= 2:
            # 计算箭头长度
            dists = [np.linalg.norm(waypoints[i+1]['pos'] - waypoints[i]['pos']) for i in range(n-1)]
            avg_dist = np.mean(dists)
            arrow_len = np.clip(avg_dist * 0.25, 0.3, 2.0)

            # 计算每个航点的垂直于航线方向（朝向桥面）
            headings = self._compute_perpendicular_headings(waypoints)

            arrow_src = vtk.vtkArrowSource()
            arrow_src.SetTipResolution(12)
            arrow_src.SetShaftResolution(8)
            arrow_src.SetShaftRadius(0.02)
            arrow_src.SetTipLength(0.35)
            arrow_src.SetTipRadius(0.06)
            arrow_src.Update()

            for i, wp in enumerate(waypoints):
                pos = wp['pos']
                fwd = headings[i]

                # 构造旋转矩阵：箭头默认沿 X 轴，旋转到 fwd 方向
                up_ref = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(fwd, up_ref)) > 0.99:
                    up_ref = np.array([0.0, 1.0, 0.0])
                right = np.cross(fwd, up_ref)
                right /= np.linalg.norm(right)
                actual_up = np.cross(right, fwd)
                actual_up /= np.linalg.norm(actual_up)

                mat = vtk.vtkMatrix4x4()
                mat.Identity()
                for c in range(3):
                    mat.SetElement(0, c, [fwd, right, actual_up][c][0])
                    mat.SetElement(1, c, [fwd, right, actual_up][c][1])
                    mat.SetElement(2, c, [fwd, right, actual_up][c][2])
                mat.SetElement(0, 3, pos[0])
                mat.SetElement(1, 3, pos[1])
                mat.SetElement(2, 3, pos[2])

                transform = vtkTransform()
                transform.PostMultiply()
                transform.Concatenate(mat)
                transform.Scale(arrow_len, arrow_len, arrow_len)

                tfilter = vtkTransformPolyDataFilter()
                tfilter.SetInputConnection(arrow_src.GetOutputPort())
                tfilter.SetTransform(transform)
                tfilter.Update()

                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(tfilter.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                # 拐角处用黄色，直行段用青色
                is_corner = self._is_corner(waypoints, i)
                if is_corner:
                    actor.GetProperty().SetColor(1.0, 0.9, 0.0)  # 黄色
                else:
                    actor.GetProperty().SetColor(0.0, 0.9, 1.0)  # 青色
                actor.GetProperty().SetOpacity(0.9)
                self.renderer.AddActor(actor)
                self._actors.append(actor)

        self._update_view()
        print(f"[VTK] Route displayed: {n} waypoints")

    # ─── 拖拽画框模式 ─────────────────────────────────────────
    def enter_pick_mode(self):
        """进入拖拽画框模式，自动切到俯视视角"""
        if self.points_data is None or len(self.points_data) == 0:
            print("[Pick] No point cloud loaded")
            return

        self.pick_mode = True
        self._dragging = False
        self._drag_start = None
        self._clear_box()

        # 自动切到俯视图
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0, 0, 100)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 1, 0)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

        print("[Pick] Ctrl + Left Drag to draw box. Esc to cancel.")

    def exit_pick_mode(self):
        """退出画框模式"""
        self.pick_mode = False
        self._dragging = False
        self._drag_start = None
        self._pick_drag_active = False
        print("[Pick] Pick mode exited.")

    def _pick_3d(self, screen_x, screen_y):
        """从屏幕坐标拾取 3D 点"""
        if not self.picker:
            return None
        self.picker.Pick(screen_x, screen_y, 0, self.renderer)
        pos = self.picker.GetPickPosition()
        if pos == (0.0, 0.0, 0.0):
            return None
        return np.array(pos)

    def _on_mouse_down(self, obj, event):
        """鼠标按下：Ctrl+左键 → 航点编辑 或 画框选区"""
        self._pick_drag_active = False
        if not self.interactor.GetControlKey():
            return  # 没按 Ctrl，交给 FoxgloveInteractorStyle 处理（左键平移）

        pos = self.interactor.GetEventPosition()

        # 优先检查：Ctrl+左键点击附近有航点 → 进入航点编辑
        wp_idx = self._find_nearest_waypoint(pos[0], pos[1])
        if wp_idx >= 0:
            self._start_wp_edit(wp_idx, pos[0], pos[1])
            self._pick_drag_active = True
            return

        # 次优先：pick_mode 下画框
        if not self.pick_mode:
            return
        p = self._pick_3d(pos[0], pos[1])
        if p is not None:
            self._dragging = True
            self._drag_start = p
            self._pick_drag_active = True
            self._clear_box()
            print(f"[Pick] Start: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    def _on_mouse_move(self, obj, event):
        """鼠标拖动：航点编辑 或 画框预览"""
        if self._wp_editing:
            pos = self.interactor.GetEventPosition()
            self._update_wp_edit(pos[0], pos[1])
            return
        if not self._dragging or self._drag_start is None:
            return
        pos = self.interactor.GetEventPosition()
        p = self._pick_3d(pos[0], pos[1])
        if p is not None:
            self._draw_preview_box(self._drag_start, p)

    def _on_mouse_up(self, obj, event):
        """鼠标松开：完成航点编辑 或 画框"""
        if self._wp_editing:
            self._end_wp_edit()
            self._pick_drag_active = False
            return
        if not self._dragging or self._drag_start is None:
            self._pick_drag_active = False
            return
        pos = self.interactor.GetEventPosition()
        p = self._pick_3d(pos[0], pos[1])
        if p is None:
            p = self._drag_start

        self._dragging = False
        self._pick_drag_active = False
        p1, p2 = self._drag_start, p

        # 确保框有最小尺寸
        if np.linalg.norm(p2 - p1) < 0.1:
            print("[Pick] Box too small, ignored")
            self._clear_box()
            return

        # 画最终框
        self._draw_preview_box(p1, p2)

        # 计算 Z 范围（用框内点云的实际 Z 范围）
        mn = np.minimum(p1, p2)
        mx = np.maximum(p1, p2)
        if self.points_data is not None:
            mask = np.all((self.points_data >= mn - 0.5) & (self.points_data <= mx + 0.5), axis=1)
            pts_in_box = self.points_data[mask]
            if len(pts_in_box) > 0:
                mn[2] = pts_in_box[:, 2].min()
                mx[2] = pts_in_box[:, 2].max()

        print(f"[Pick] Box: ({mn[0]:.1f},{mn[1]:.1f},{mn[2]:.1f}) -> ({mx[0]:.1f},{mx[1]:.1f},{mx[2]:.1f})")

        # 退出选框模式
        self.exit_pick_mode()

        # 发出信号，由主窗口弹出菜单让用户选航线类型
        self.box_selected.emit([mn.tolist(), mx.tolist()])

    # ─── 航点编辑 ──────────────────────────────────────────────
    def _find_nearest_waypoint(self, screen_x, screen_y):
        """找到屏幕坐标最近的航点，返回索引（-1 表示未找到）"""
        if not self._waypoint_actors or not self._waypoints_ref:
            return -1

        # 将屏幕坐标转为世界坐标射线
        ren = self.renderer
        ren.SetDisplayPoint(screen_x, screen_y, 0)
        ren.DisplayToWorld()
        near = np.array(ren.GetWorldPoint()[:3])
        ren.SetDisplayPoint(screen_x, screen_y, 1)
        ren.DisplayToWorld()
        far = np.array(ren.GetWorldPoint()[:3])
        ray_dir = far - near
        ray_dir /= np.linalg.norm(ray_dir)

        # 找最近的航点（射线到点的距离）
        best_idx = -1
        best_dist = float('inf')
        PICK_THRESHOLD = 15.0  # 像素阈值

        cam = ren.GetActiveCamera()
        vp = ren.GetViewport()
        win_size = ren.GetSize()

        for i, wp in enumerate(self._waypoints_ref):
            pos = wp['pos']
            # 世界坐标 → 屏幕坐标
            ren.SetWorldPoint(pos[0], pos[1], pos[2], 1)
            ren.WorldToDisplay()
            dp = ren.GetDisplayPoint()
            dx = dp[0] - screen_x
            dy = dp[1] - screen_y
            dist_px = (dx**2 + dy**2) ** 0.5
            if dist_px < best_dist:
                best_dist = dist_px
                best_idx = i

        if best_dist > PICK_THRESHOLD:
            return -1
        return best_idx

    def _screen_to_xy_plane(self, screen_x, screen_y, z_plane):
        """屏幕坐标射线与 Z=z_plane 平面的交点"""
        ren = self.renderer
        ren.SetDisplayPoint(screen_x, screen_y, 0)
        ren.DisplayToWorld()
        near = np.array(ren.GetWorldPoint()[:3])
        ren.SetDisplayPoint(screen_x, screen_y, 1)
        ren.DisplayToWorld()
        far = np.array(ren.GetWorldPoint()[:3])
        ray_dir = far - near

        # 射线: P = near + t * ray_dir
        # 平面: P.z = z_plane
        if abs(ray_dir[2]) < 1e-10:
            return None
        t = (z_plane - near[2]) / ray_dir[2]
        if t < 0:
            return None
        return near + t * ray_dir

    def _start_wp_edit(self, idx, screen_x, screen_y):
        """开始编辑航点"""
        self._wp_editing = True
        self._wp_edit_idx = idx
        wp = self._waypoints_ref[idx]
        self._wp_edit_z = wp['pos'][2]

        # 计算拖拽偏移（鼠标点击位置与航点位置的 XY 偏差）
        world_p = self._screen_to_xy_plane(screen_x, screen_y, self._wp_edit_z)
        if world_p is not None:
            self._wp_edit_offset = wp['pos'][:2] - world_p[:2]
        else:
            self._wp_edit_offset = np.zeros(2)

        # 高亮选中的航点
        if idx < len(self._waypoint_actors):
            self._waypoint_actors[idx].GetProperty().SetColor(1.0, 1.0, 0.0)  # 黄色高亮

        print(f"[WP Edit] Selected waypoint #{idx} at ({wp['pos'][0]:.2f}, {wp['pos'][1]:.2f}, {wp['pos'][2]:.2f})")

    def _update_wp_edit(self, screen_x, screen_y):
        """拖动更新航点位置"""
        if not self._wp_editing:
            return
        world_p = self._screen_to_xy_plane(screen_x, screen_y, self._wp_edit_z)
        if world_p is None:
            return

        new_pos = world_p[:2] + self._wp_edit_offset
        wp = self._waypoints_ref[self._wp_edit_idx]
        wp['pos'][0] = new_pos[0]
        wp['pos'][1] = new_pos[1]

        # 更新球体位置
        if self._wp_edit_idx < len(self._waypoint_actors):
            actor = self._waypoint_actors[self._wp_edit_idx]
            actor.SetPosition(new_pos[0] - actor.GetCenter()[0],
                              new_pos[1] - actor.GetCenter()[1], 0)

        # 更新箭头位置
        self.vtk_widget.GetRenderWindow().Render()

    def _end_wp_edit(self):
        """完成航点编辑，通知主窗口"""
        if not self._wp_editing:
            return
        idx = self._wp_edit_idx
        wp = self._waypoints_ref[idx]
        new_pos = wp['pos'].copy()

        # 恢复航点颜色（安全距离检测会重新标色）
        if idx < len(self._waypoint_actors):
            self._waypoint_actors[idx].GetProperty().SetColor(1.0, 0.2, 0.2)

        self._wp_editing = False
        self._wp_edit_idx = -1

        # 发出信号通知主窗口
        self.waypoint_edited.emit(idx, new_pos, None)
        print(f"[WP Edit] Waypoint #{idx} moved to ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})")

    def _draw_preview_box(self, p1, p2):
        """实时绘制预览框（黄色半透明线框）"""
        self._clear_box()

        mn = np.minimum(p1, p2)
        mx = np.maximum(p1, p2)

        # 8 个顶点
        corners = [
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
        ]

        # 12 条边
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]

        for i1, i2 in edges:
            line = vtkLineSource()
            line.SetPoint1(corners[i1])
            line.SetPoint2(corners[i2])
            m = vtkPolyDataMapper()
            m.SetInputConnection(line.GetOutputPort())
            a = vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetColor(1.0, 1.0, 0.0)
            a.GetProperty().SetLineWidth(2)
            a.GetProperty().SetOpacity(0.8)
            self.renderer.AddActor(a)
            self.box_actors.append(a)

        self.vtk_widget.GetRenderWindow().Render()

    def _clear_box(self):
        """清除选框线框"""
        for a in self.box_actors:
            self.renderer.RemoveActor(a)
        self.box_actors.clear()

    def _update_view(self):
        """更新相机和视图"""
        if not VTK_AVAILABLE:
            return
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(50, -80, 60)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        cam.Elevation(30)
        cam.Azimuth(-45)
        self.vtk_widget.GetRenderWindow().Render()

    def _on_global_key_press(self, obj, event):
        """全局快捷键：视角切换"""
        key = self.interactor.GetKeyCode()
        cam = self.renderer.GetActiveCamera()

        if key == '1':
            # 俯视图 (Top View)
            cam.SetPosition(0, 0, 100)
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(0, 1, 0)
            self.renderer.ResetCamera()
            print("[View] Top View")
        elif key == '2':
            # 正视图 (Front View) - 从 Y 负方向看
            cam.SetPosition(0, -100, 0)
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(0, 0, 1)
            self.renderer.ResetCamera()
            print("[View] Front View")
        elif key == '3':
            # 侧视图 (Side View) - 从 X 正方向看
            cam.SetPosition(100, 0, 0)
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(0, 0, 1)
            self.renderer.ResetCamera()
            print("[View] Side View")
        elif key == '4':
            # 透视图 (Perspective) - 默认
            self.renderer.ResetCamera()
            cam.Elevation(30)
            cam.Azimuth(-45)
            print("[View] Perspective")
        elif key == '5':
            # 仰视图 (Bottom View) - 从下往上看桥底面
            cam.SetPosition(0, 0, -100)
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(0, 1, 0)
            self.renderer.ResetCamera()
            print("[View] Bottom View (looking up)")
        elif key == '\x1b':
            # Esc 取消画框模式
            if self.pick_mode:
                self._clear_box()
                self.exit_pick_mode()

        self.vtk_widget.GetRenderWindow().Render()

    def setup_scene(self):
        """初始化场景（坐标轴 + 网格）"""
        if not VTK_AVAILABLE:
            return

        # 坐标轴
        axes = [
            ((0, 0, 0), (8, 0, 0), (1, 0, 0)),  # X 红
            ((0, 0, 0), (0, 8, 0), (0, 1, 0)),  # Y 绿
            ((0, 0, 0), (0, 0, 8), (0, 0, 1)),  # Z 蓝
        ]
        for start, end, color in axes:
            line = vtkLineSource()
            line.SetPoint1(start)
            line.SetPoint2(end)
            m = vtkPolyDataMapper()
            m.SetInputConnection(line.GetOutputPort())
            a = vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetColor(color)
            a.GetProperty().SetLineWidth(3)
            self.renderer.AddActor(a)
            self._actors.append(a)

        # 地面网格
        grid_pts = vtkPoints()
        grid_cells = vtkCellArray()
        idx = 0
        for i in range(-20, 21, 2):
            # X 方向线
            grid_pts.InsertNextPoint(i, -20, 0)
            grid_pts.InsertNextPoint(i, 20, 0)
            line = vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)
            line.GetPointIds().SetId(0, idx)
            line.GetPointIds().SetId(1, idx + 1)
            grid_cells.InsertNextCell(line)
            idx += 2
            # Y 方向线
            grid_pts.InsertNextPoint(-20, i, 0)
            grid_pts.InsertNextPoint(20, i, 0)
            line = vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)
            line.GetPointIds().SetId(0, idx)
            line.GetPointIds().SetId(1, idx + 1)
            grid_cells.InsertNextCell(line)
            idx += 2

        grid_poly = vtkPolyData()
        grid_poly.SetPoints(grid_pts)
        grid_poly.SetLines(grid_cells)
        m = vtkPolyDataMapper()
        m.SetInputData(grid_poly)
        a = vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(0.3, 0.3, 0.3)
        a.GetProperty().SetOpacity(0.4)
        self.renderer.AddActor(a)
        self._actors.append(a)

        self._update_view()
        self.interactor.Initialize()

        # 设置 Foxglove 风格交互（左键平移，右键旋转）
        style = FoxgloveInteractorStyle()
        style.set_viewer(self)
        self.interactor.SetInteractorStyle(style)

        # 绑定全局快捷键（视角切换 + Esc 取消画框）
        self.interactor.AddObserver("KeyPressEvent", self._on_global_key_press)

        # 一次性绑定鼠标事件（Ctrl+左键画框，由 pick_mode 标志位控制）
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_mouse_down)
        self.interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self._on_mouse_up)


# ─── 主窗口 ──────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """桥梁巡检航线规划工具 - 主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("桥梁巡检无人机航线规划工具")
        self.resize(1400, 900)

        self.points = None
        self.waypoints = []
        self._surface_bounds = None

        self._init_ui()
        self._apply_style()
        self.viewer.setup_scene()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        # ─── 左侧 3D 视图 ───
        self.viewer = VTKViewer()
        splitter.addWidget(self.viewer)

        # 连接选框完成信号
        self.viewer.box_selected.connect(self._on_box_selected)
        self.viewer.waypoint_edited.connect(self._on_waypoint_edited)

        # ─── 右侧控制面板 ───
        ctrl = QWidget()
        ctrl.setMinimumWidth(320)
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setSpacing(6)

        # -- 加载点云 --
        grp_load = QGroupBox("加载点云")
        gl = QVBoxLayout(grp_load)
        self.btn_load = QPushButton("打开 PCD 文件")
        gl.addWidget(self.btn_load)
        self.lbl_pc_info = QLabel("未加载点云")
        gl.addWidget(self.lbl_pc_info)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        gl.addWidget(self.progress_bar)
        ctrl_layout.addWidget(grp_load)

        # -- 桥梁参数 --
        grp_bridge = QGroupBox("桥梁参数")
        bp = QGridLayout(grp_bridge)
        bp.setSpacing(4)

        bp.addWidget(QLabel("桥型:"), 0, 0)
        self.cmb_bridge_type = QComboBox()
        self.cmb_bridge_type.addItems(["跨河桥", "跨线桥", "高架桥"])
        bp.addWidget(self.cmb_bridge_type, 0, 1, 1, 3)

        bp.addWidget(QLabel("桥长(m):"), 1, 0)
        self.edt_bridge_len = QLineEdit("100"); bp.addWidget(self.edt_bridge_len, 1, 1)
        bp.addWidget(QLabel("桥宽(m):"), 1, 2)
        self.edt_bridge_wid = QLineEdit("15"); bp.addWidget(self.edt_bridge_wid, 1, 3)

        bp.addWidget(QLabel("净空(m):"), 2, 0)
        self.edt_bridge_clr = QLineEdit("8"); bp.addWidget(self.edt_bridge_clr, 2, 1)
        bp.addWidget(QLabel("跨距(m):"), 2, 2)
        self.edt_bridge_span = QLineEdit("30"); bp.addWidget(self.edt_bridge_span, 2, 3)

        btn_apply_bridge = QPushButton("应用到航线默认值")
        btn_apply_bridge.setStyleSheet("QPushButton { background: #2a3a5a; padding: 6px; } QPushButton:hover { background: #3a4a6a; }")
        btn_apply_bridge.clicked.connect(self._apply_bridge_params)
        bp.addWidget(btn_apply_bridge, 3, 0, 1, 4)

        ctrl_layout.addWidget(grp_bridge)

        # -- 框选区域 --
        grp_pick = QGroupBox("区域选择")
        pk = QVBoxLayout(grp_pick)
        self.btn_pick_region = QPushButton("框选区域 (Ctrl+拖拽)")
        self.btn_pick_region.setStyleSheet("QPushButton { background: #2a4a2a; font-weight: bold; padding: 8px; } QPushButton:hover { background: #3a5a3a; }")
        pk.addWidget(self.btn_pick_region)

        sd_row = QHBoxLayout()
        sd_row.addWidget(QLabel("安全距离(m):"))
        self.edt_safe_dist = QLineEdit("2.0")
        self.edt_safe_dist.setMaximumWidth(60)
        self.edt_safe_dist.textChanged.connect(self._on_safe_dist_changed)
        sd_row.addWidget(self.edt_safe_dist)
        sd_row.addStretch()
        pk.addLayout(sd_row)

        lbl_wp_hint = QLabel("Ctrl+左键点击航点可拖动编辑位置")
        lbl_wp_hint.setStyleSheet("color: #888; font-size: 10px;")
        lbl_wp_hint.setWordWrap(True)
        pk.addWidget(lbl_wp_hint)

        ctrl_layout.addWidget(grp_pick)

        # -- 面状航线（桥底面弓字形扫描）--
        grp_flat = QGroupBox("面状航线（弓字形扫描）")
        fl = QGridLayout(grp_flat)
        fl.setSpacing(4)

        fl.addWidget(QLabel("X 范围:"), 0, 0)
        self.edt_xmin = QLineEdit("-10"); fl.addWidget(self.edt_xmin, 0, 1)
        fl.addWidget(QLabel("~"), 0, 2)
        self.edt_xmax = QLineEdit("10"); fl.addWidget(self.edt_xmax, 0, 3)

        fl.addWidget(QLabel("Y 范围:"), 1, 0)
        self.edt_ymin = QLineEdit("-10"); fl.addWidget(self.edt_ymin, 1, 1)
        fl.addWidget(QLabel("~"), 1, 2)
        self.edt_ymax = QLineEdit("10"); fl.addWidget(self.edt_ymax, 1, 3)

        fl.addWidget(QLabel("高度Z:"), 2, 0)
        self.edt_z = QLineEdit("5"); fl.addWidget(self.edt_z, 2, 1)
        fl.addWidget(QLabel("间距:"), 2, 2)
        self.edt_spacing = QLineEdit("2"); fl.addWidget(self.edt_spacing, 2, 3)

        fl.addWidget(QLabel("速度(m/s):"), 3, 0)
        self.edt_flat_speed = QLineEdit("3"); fl.addWidget(self.edt_flat_speed, 3, 1)

        self.btn_flat = QPushButton("生成面状航线")
        fl.addWidget(self.btn_flat, 4, 0, 1, 4)
        ctrl_layout.addWidget(grp_flat)

        # -- 等距面扫描 --
        grp_surface = QGroupBox("等距面扫描（沿表面法向）")
        sl = QGridLayout(grp_surface)
        sl.setSpacing(4)

        sl.addWidget(QLabel("离面距离:"), 0, 0)
        self.edt_standoff = QLineEdit("2.0"); sl.addWidget(self.edt_standoff, 0, 1)
        sl.addWidget(QLabel("速度:"), 0, 2)
        self.edt_surface_speed = QLineEdit("2.0"); sl.addWidget(self.edt_surface_speed, 0, 3)

        sl.addWidget(QLabel("K邻域:"), 1, 0)
        self.edt_k_neighbors = QLineEdit("30"); sl.addWidget(self.edt_k_neighbors, 1, 1)
        sl.addWidget(QLabel("法向:"), 1, 2)
        self.cbo_normal_dir = QComboBox()
        self.cbo_normal_dir.addItems(["朝下", "朝上", "自动"])
        sl.addWidget(self.cbo_normal_dir, 1, 3)

        self.btn_surface = QPushButton("生成等距面航线")
        sl.addWidget(self.btn_surface, 2, 0, 1, 4)
        ctrl_layout.addWidget(grp_surface)

        # -- 立方体/圆柱体航线（桥柱环绕扫描）--
        grp_cube = QGroupBox("柱体航线（桥柱环绕扫描）")
        cl = QGridLayout(grp_cube)
        cl.setSpacing(4)

        cl.addWidget(QLabel("类型:"), 0, 0)
        self.cbo_pillar_type = QComboBox()
        self.cbo_pillar_type.addItems(["立方体", "圆柱体"])
        self.cbo_pillar_type.currentIndexChanged.connect(self._on_pillar_type_changed)
        cl.addWidget(self.cbo_pillar_type, 0, 1, 1, 3)

        cl.addWidget(QLabel("中心(x,y,z):"), 1, 0)
        self.edt_cx = QLineEdit("0"); cl.addWidget(self.edt_cx, 1, 1)
        self.edt_cy = QLineEdit("0"); cl.addWidget(self.edt_cy, 1, 2)
        self.edt_cz = QLineEdit("0"); cl.addWidget(self.edt_cz, 1, 3)

        cl.addWidget(QLabel("尺寸(Wx,Wy,H):"), 2, 0)
        self.edt_dx = QLineEdit("4"); cl.addWidget(self.edt_dx, 2, 1)
        self.edt_dy = QLineEdit("4"); cl.addWidget(self.edt_dy, 2, 2)
        self.edt_dz = QLineEdit("8"); cl.addWidget(self.edt_dz, 2, 3)

        cl.addWidget(QLabel("水平步距:"), 3, 0)
        self.edt_cstep = QLineEdit("2"); cl.addWidget(self.edt_cstep, 3, 1)
        cl.addWidget(QLabel("垂直步距:"), 3, 2)
        self.edt_vstep = QLineEdit("2"); cl.addWidget(self.edt_vstep, 3, 3)

        cl.addWidget(QLabel("离柱距离:"), 4, 0)
        self.edt_dist = QLineEdit("3"); cl.addWidget(self.edt_dist, 4, 1)
        cl.addWidget(QLabel("速度:"), 4, 2)
        self.edt_cspeed = QLineEdit("2"); cl.addWidget(self.edt_cspeed, 4, 3)

        cl.addWidget(QLabel("路径类型:"), 5, 0)
        self.cbo_cube_type = QComboBox()
        self.cbo_cube_type.addItems(["螺旋线", "Z字形"])
        cl.addWidget(self.cbo_cube_type, 5, 1, 1, 3)

        self.btn_cube = QPushButton("生成柱体航线")
        cl.addWidget(self.btn_cube, 6, 0, 1, 4)
        ctrl_layout.addWidget(grp_cube)

        # -- 航线管理 --
        grp_route = QGroupBox("航线管理")
        rl = QVBoxLayout(grp_route)
        self.lbl_info = QLabel("航点: 0")
        rl.addWidget(self.lbl_info)
        self.btn_clear = QPushButton("清除航线")
        rl.addWidget(self.btn_clear)
        self.btn_save = QPushButton("保存航线 (JSON)")
        rl.addWidget(self.btn_save)
        self.btn_load_route = QPushButton("加载航线 (JSON)")
        rl.addWidget(self.btn_load_route)

        # 均匀分布按钮
        self.btn_resample = QPushButton("航点均匀分布")
        self.btn_resample.setStyleSheet("QPushButton { background: #3a3a2a; padding: 6px; } QPushButton:hover { background: #4a4a3a; }")
        self.btn_resample.clicked.connect(self._resample_waypoints)
        rl.addWidget(self.btn_resample)

        # 机头方向显示开关
        self.chk_show_heading = QCheckBox("显示机头方向箭头")
        self.chk_show_heading.setChecked(True)
        self.chk_show_heading.stateChanged.connect(self._toggle_heading)
        rl.addWidget(self.chk_show_heading)

        ctrl_layout.addWidget(grp_route)

        # -- 快捷键提示 --
        lbl_help = QLabel("快捷键: 1=俯视 2=正视 3=侧视 4=透视 5=仰视  Esc=取消框选")
        lbl_help.setStyleSheet("color: #666; font-size: 10px; padding: 4px;")
        lbl_help.setWordWrap(True)
        ctrl_layout.addWidget(lbl_help)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter)

        # -- 信号连接 --
        self.btn_load.clicked.connect(self.load_point_cloud)
        self.btn_pick_region.clicked.connect(lambda: self.viewer.enter_pick_mode())
        self.btn_flat.clicked.connect(self.generate_flat_route)
        self.btn_surface.clicked.connect(self.generate_surface_route)
        self.btn_cube.clicked.connect(self.generate_cube_route)
        self.btn_clear.clicked.connect(self.clear_route)
        self.btn_save.clicked.connect(self.save_route)
        self.btn_load_route.clicked.connect(self.load_route)

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        return lbl

    def _on_pillar_type_changed(self, idx):
        """切换柱体类型时更新尺寸标签"""
        if idx == 1:  # 圆柱体：Wx=Wy=直径
            self.edt_dy.setText(self.edt_dx.text())

    def _resample_waypoints(self):
        """将现有航点均匀重新分布"""
        if len(self.waypoints) < 2:
            QMessageBox.information(self, "提示", "航点不足，无法均匀分布")
            return

        # 计算原始路径总长度
        positions = [wp['pos'] for wp in self.waypoints]
        total_len = sum(np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions)-1))

        # 按原始间距的平均值作为目标间距
        avg_spacing = total_len / (len(self.waypoints) - 1)

        # 沿路径均匀插值
        new_waypoints = self._interpolate_waypoints(self.waypoints, avg_spacing)
        self.waypoints = new_waypoints
        self._display_route()
        print(f"[Resample] {len(self.waypoints)} waypoints, spacing={avg_spacing:.2f}m")

    @staticmethod
    def _interpolate_waypoints(waypoints, spacing):
        """沿路径均匀插值航点"""
        if len(waypoints) < 2:
            return waypoints

        positions = np.array([wp['pos'] for wp in waypoints])
        # 累积弧长
        seg_lens = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cum_len[-1]

        if total_len < 1e-10:
            return waypoints

        # 目标弧长位置
        n_new = max(2, int(total_len / spacing) + 1)
        target_lens = np.linspace(0, total_len, n_new)

        new_wps = []
        seg_idx = 0
        for t_len in target_lens:
            # 找到所在段
            while seg_idx < len(cum_len) - 2 and cum_len[seg_idx + 1] < t_len:
                seg_idx += 1
            # 线性插值
            seg_start = cum_len[seg_idx]
            seg_end = cum_len[seg_idx + 1]
            seg_len = seg_end - seg_start
            if seg_len < 1e-10:
                alpha = 0.0
            else:
                alpha = (t_len - seg_start) / seg_len
            alpha = np.clip(alpha, 0.0, 1.0)

            pos = waypoints[seg_idx]['pos'] * (1 - alpha) + waypoints[seg_idx + 1]['pos'] * alpha
            quat = waypoints[seg_idx]['quat'] * (1 - alpha) + waypoints[seg_idx + 1]['quat'] * alpha
            # 四元数归一化
            quat = quat / np.linalg.norm(quat)

            new_wps.append({
                'pos': pos,
                'quat': quat,
                'speed': waypoints[seg_idx]['speed'],
                'action': waypoints[seg_idx]['action']
            })

        return new_wps

    def _apply_style(self):
        # 检测中文字体可用性
        from PyQt5.QtGui import QFontDatabase
        available = QFontDatabase().families()
        cn_font = "Microsoft YaHei"
        for candidate in ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "PingFang SC"]:
            if candidate in available:
                cn_font = candidate
                break

        self.setStyleSheet(f"""
            QMainWindow {{ background: #1e1e22; }}
            QWidget {{ color: #ddd; font-family: "{cn_font}", "Segoe UI", Arial; font-size: 12px; }}
            QGroupBox {
                border: 1px solid #444; border-radius: 6px;
                margin-top: 8px; padding: 10px 8px; font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
            QLineEdit {
                background: #2b2b30; border: 1px solid #555; border-radius: 3px;
                padding: 3px 6px; color: #eee;
            }
            QLineEdit:focus { border-color: #4a9eff; }
            QPushButton {
                background: #3a3a42; border: 1px solid #555; border-radius: 4px;
                padding: 6px 14px; color: #eee; min-height: 24px;
            }
            QPushButton:hover { background: #4a4a55; }
            QPushButton:pressed { background: #555566; }
            QComboBox {
                background: #2b2b30; border: 1px solid #555; border-radius: 3px;
                padding: 3px 6px; color: #eee;
            }
            QComboBox QAbstractItemView { background: #2b2b30; color: #eee; selection-background-color: #4a9eff; }
        """)

    # ─── 选框完成回调 ────────────────────────────────────────
    def _on_box_selected(self, box):
        """收到选框完成信号，弹出菜单让用户选航线类型"""
        mn = np.array(box[0])
        mx = np.array(box[1])
        self._last_box = (mn, mx)

        # 弹出菜单
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #2b2b30; color: #eee; border: 1px solid #555; padding: 4px; }
            QMenu::item { padding: 6px 20px; }
            QMenu::item:selected { background: #4a9eff; }
        """)
        menu.addAction("Flat Route (Zigzag)", lambda: self._apply_box("flat", mn, mx))
        menu.addAction("Cube Route (Pillar)", lambda: self._apply_box("cube", mn, mx))
        menu.addAction("Surface Scan (Equidistant)", lambda: self._apply_box("surface", mn, mx))

        # 在鼠标位置弹出
        cursor_pos = self.cursor().pos()
        menu.exec_(cursor_pos)

    def _apply_box(self, mode, mn, mx):
        """根据用户选择的航线类型，填充参数并生成航线"""
        if mode == "flat":
            self.edt_xmin.setText(f"{mn[0]:.1f}")
            self.edt_xmax.setText(f"{mx[0]:.1f}")
            self.edt_ymin.setText(f"{mn[1]:.1f}")
            self.edt_ymax.setText(f"{mx[1]:.1f}")
            self.edt_z.setText(f"{mx[2] + 3:.1f}")
            print(f"[Pick] Flat: X[{mn[0]:.1f}, {mx[0]:.1f}] Y[{mn[1]:.1f}, {mx[1]:.1f}] Z={mx[2]+3:.1f}")
            self.generate_flat_route()

        elif mode == "cube":
            center = (mn + mx) / 2
            size = mx - mn
            self.edt_cx.setText(f"{center[0]:.1f}")
            self.edt_cy.setText(f"{center[1]:.1f}")
            self.edt_cz.setText(f"{mn[2]:.1f}")
            self.edt_dx.setText(f"{max(size[0], 0.5):.1f}")
            self.edt_dy.setText(f"{max(size[1], 0.5):.1f}")
            self.edt_dz.setText(f"{max(size[2], 0.5):.1f}")
            print(f"[Pick] Cube: center=({center[0]:.1f},{center[1]:.1f},{mn[2]:.1f}) size=({size[0]:.1f},{size[1]:.1f},{size[2]:.1f})")
            self.generate_cube_route()

        elif mode == "surface":
            self._surface_bounds = (mn, mx)
            print(f"[Pick] Surface: X[{mn[0]:.1f}, {mx[0]:.1f}] Y[{mn[1]:.1f}, {mx[1]:.1f}] Z[{mn[2]:.1f}, {mx[2]:.1f}]")
            self.generate_surface_route()

    # ─── 加载点云 ───
    def load_point_cloud(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 PCD 点云文件", "", "PCD 文件 (*.pcd);;所有文件 (*)"
        )
        if not path:
            return

        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度模式
        self.lbl_pc_info.setText(f"正在加载 {os.path.basename(path)}...")
        QApplication.processEvents()

        try:
            self.points = parse_pcd(path)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(30)
            QApplication.processEvents()

            self.viewer.add_point_cloud(self.points)
            self.progress_bar.setValue(80)
            QApplication.processEvents()

            n = len(self.points)
            self.lbl_pc_info.setText(f"已加载: {os.path.basename(path)} ({n:,} 点)")

            # 根据点云范围自动填充平面航线默认值
            if n > 0:
                mn = self.points.min(axis=0)
                mx = self.points.max(axis=0)
                pad = max((mx - mn).max() * 0.05, 1.0)
                self.edt_xmin.setText(f"{mn[0] - pad:.1f}")
                self.edt_xmax.setText(f"{mx[0] + pad:.1f}")
                self.edt_ymin.setText(f"{mn[1] - pad:.1f}")
                self.edt_ymax.setText(f"{mx[1] + pad:.1f}")
                self.edt_z.setText(f"{mx[2] + 3:.1f}")

                # 柱体中心默认为点云中心
                center = (mn + mx) / 2
                self.edt_cx.setText(f"{center[0]:.1f}")
                self.edt_cy.setText(f"{center[1]:.1f}")
                self.edt_cz.setText(f"{mn[2]:.1f}")
                self.edt_dz.setText(f"{mx[2] - mn[2]:.1f}")

            self.progress_bar.setValue(100)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载点云失败:\n{str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    # ─── 生成平面航线 ───
    def generate_flat_route(self):
        try:
            xmin = float(self.edt_xmin.text())
            xmax = float(self.edt_xmax.text())
            ymin = float(self.edt_ymin.text())
            ymax = float(self.edt_ymax.text())
            z = float(self.edt_z.text())
            spacing = float(self.edt_spacing.text())
            speed = float(self.edt_flat_speed.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效数字")
            return

        if spacing <= 0 or xmin >= xmax or ymin >= ymax:
            QMessageBox.warning(self, "输入错误", "请检查参数范围")
            return

        self.waypoints = []
        y = ymin
        direction = 1  # 1: X 正向, -1: X 反向

        while y <= ymax:
            if direction == 1:
                x_start, x_end = xmin, xmax
            else:
                x_start, x_end = xmax, xmin

            # 起点
            self.waypoints.append({
                'pos': np.array([x_start, y, z]),
                'quat': np.array([1.0, 0.0, 0.0, 0.0]),
                'speed': speed,
                'action': 'fly'
            })

            # 终点
            self.waypoints.append({
                'pos': np.array([x_end, y, z]),
                'quat': look_at_quaternion(np.array([x_end + direction, y, z]),
                                           np.array([x_end, y, z])),
                'speed': speed,
                'action': 'fly'
            })

            # 移到下一行
            y += spacing
            if y <= ymax:
                self.waypoints.append({
                    'pos': np.array([x_end, y, z]),
                    'quat': np.array([1.0, 0.0, 0.0, 0.0]),
                    'speed': speed,
                    'action': 'fly'
                })
            direction *= -1

        # 更新航点朝向
        for i in range(len(self.waypoints) - 1):
            self.waypoints[i]['quat'] = look_at_quaternion(
                self.waypoints[i + 1]['pos'], self.waypoints[i]['pos']
            )

        self._display_route()

    # ─── 生成等距面扫描航线 ───
    def generate_surface_route(self):
        """沿点云表面等距飞行的航线生成"""
        if self.points is None or len(self.points) == 0:
            QMessageBox.warning(self, "警告", "未加载点云")
            return

        # 获取选区范围（如果没有画框，用整个点云）
        if hasattr(self, '_surface_bounds') and self._surface_bounds:
            mn, mx = self._surface_bounds
        else:
            mn = self.points.min(axis=0)
            mx = self.points.max(axis=0)

        try:
            standoff = float(self.edt_standoff.text())
            speed = float(self.edt_surface_speed.text())
            k = int(self.edt_k_neighbors.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效数字")
            return

        normal_dir = self.cbo_normal_dir.currentText()  # "朝下" / "朝上" / "自动"

        # 提取选区内的点云
        mask = np.all((self.points >= mn) & (self.points <= mx), axis=1)
        surface_pts = self.points[mask]

        if len(surface_pts) < 10:
            QMessageBox.warning(self, "警告", "所选区域内点太少")
            return

        print(f"[Surface] Processing {len(surface_pts)} points, k={k}, standoff={standoff}m")

        try:
            # 估算表面法向量
            normals = estimate_normals(surface_pts, k=k)

            # 根据用户选择统一法向量方向
            if normal_dir == "朝下":
                # 强制朝下（Z 负方向）
                for i in range(len(normals)):
                    if normals[i, 2] > 0:
                        normals[i] = -normals[i]
            elif normal_dir == "朝上":
                # 强制朝上（Z 正方向）
                for i in range(len(normals)):
                    if normals[i, 2] < 0:
                        normals[i] = -normals[i]
            # "Auto" 保持 estimate_normals 的默认朝下

            # 沿法向量偏移生成航点
            offset_pts = surface_pts + normals * standoff

            # 用最近邻排序形成连续路径
            order = nearest_neighbor_path(offset_pts)
            ordered_pts = offset_pts[order]
            ordered_normals = normals[order]

            # 生成航点（朝向表面）
            self.waypoints = []
            for i in range(len(ordered_pts)):
                pos = ordered_pts[i]
                # 朝向表面（法向量方向 = 指向表面）
                surface_point = pos - ordered_normals[i] * standoff
                quat = look_at_quaternion(surface_point, pos)

                self.waypoints.append({
                    'pos': pos,
                    'quat': quat,
                    'speed': speed,
                    'action': 'scan'
                })

            self._display_route()
            print(f"[Surface] Generated {len(self.waypoints)} waypoints")

        except ImportError:
            QMessageBox.critical(self, "错误",
                                 "等距面扫描需要 scipy 库。\n请运行: pip install scipy")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"等距面扫描失败:\n{str(e)}")

    # ─── 生成立方体航线 ───
    def generate_cube_route(self):
        try:
            cx = float(self.edt_cx.text())
            cy = float(self.edt_cy.text())
            cz = float(self.edt_cz.text())
            dx = float(self.edt_dx.text())
            dy = float(self.edt_dy.text())
            dz = float(self.edt_dz.text())
            cstep = float(self.edt_cstep.text())
            vstep = float(self.edt_vstep.text())
            dist = float(self.edt_dist.text())
            speed = float(self.edt_cspeed.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效数字")
            return

        if cstep <= 0 or vstep <= 0 or dz <= 0:
            QMessageBox.warning(self, "输入错误", "步距和高度必须为正数")
            return

        # 定义 4 个柱面（矩形柱体的四条边）
        half_x = dx / 2
        half_y = dy / 2
        pillars = [
            {'cx': cx - half_x, 'cy': cy, 'face': '-x'},
            {'cx': cx + half_x, 'cy': cy, 'face': '+x'},
            {'cx': cx, 'cy': cy - half_y, 'face': '-y'},
            {'cx': cx, 'cy': cy + half_y, 'face': '+y'},
        ]

        self.waypoints = []
        route_type = self.cbo_cube_type.currentText()

        for pillar in pillars:
            px, py = pillar['cx'], pillar['cy']

            if route_type == "Spiral":
                # 螺旋线：绕柱面螺旋上升
                num_turns = max(1, int(dz / vstep))
                num_pts_per_turn = max(8, int(360 / max(1, cstep)))
                total_pts = num_turns * num_pts_per_turn

                for i in range(total_pts + 1):
                    t = i / total_pts
                    angle = t * num_turns * 2 * np.pi
                    z = cz + t * dz

                    rx = px + dist * np.cos(angle)
                    ry = py + dist * np.sin(angle)
                    pos = np.array([rx, ry, z])

                    # 朝向柱面中心
                    inward = np.array([px, py, z]) - pos
                    if np.linalg.norm(inward) > 1e-6:
                        quat = look_at_quaternion(np.array([px, py, z]), pos)
                    else:
                        quat = np.array([1.0, 0.0, 0.0, 0.0])

                    self.waypoints.append({
                        'pos': pos,
                        'quat': quat,
                        'speed': speed,
                        'action': 'scan'
                    })

            elif route_type == "Zigzag":
                # Z字形：上下扫描，每层旋转一定角度
                num_layers = max(1, int(dz / vstep))
                num_cols = max(1, int(360 / max(1, cstep)))

                for layer in range(num_layers + 1):
                    z = cz + layer * vstep
                    for col in range(num_cols + 1):
                        if layer % 2 == 0:
                            angle = (col / num_cols) * 2 * np.pi
                        else:
                            angle = (1 - col / num_cols) * 2 * np.pi

                        rx = px + dist * np.cos(angle)
                        ry = py + dist * np.sin(angle)
                        pos = np.array([rx, ry, z])

                        quat = look_at_quaternion(np.array([px, py, z]), pos)

                        self.waypoints.append({
                            'pos': pos,
                            'quat': quat,
                            'speed': speed,
                            'action': 'scan'
                        })

        self._display_route()

    @staticmethod
    def _make_labeled_input(parent_layout, label, default=""):
        """创建一行 Label + QLineEdit，返回 QLineEdit"""
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        edt = QLineEdit(default)
        edt.setMaximumWidth(80)
        row.addWidget(edt)
        row.addStretch()
        parent_layout.addLayout(row)
        return edt

    def _apply_bridge_params(self):
        """根据桥参数自动填充航线默认值"""
        try:
            bridge_len = float(self.edt_bridge_len.text())
            bridge_wid = float(self.edt_bridge_wid.text())
            clearance = float(self.edt_bridge_clr.text())
            span = float(self.edt_bridge_span.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的桥梁参数")
            return

        bridge_type = self.cmb_bridge_type.currentIndex()

        # 不同桥型的默认飞行高度偏移
        if bridge_type == 0:    # 跨河：桥底面往下偏移
            z_offset = 3.0
        elif bridge_type == 1:  # 跨线：桥底面往下偏移更多（避开线路）
            z_offset = 5.0
        else:                   # 高架：桥底面偏移适中
            z_offset = 4.0

        # 用点云中心作为参考原点（如果有加载点云）
        if self.points is not None and len(self.points) > 0:
            center = (self.points.min(axis=0) + self.points.max(axis=0)) / 2
            cx, cy = center[0], center[1]
        else:
            cx, cy = 0.0, 0.0

        half_len = bridge_len / 2
        half_wid = bridge_wid / 2

        # 平面航线：覆盖桥底面区域
        self.edt_xmin.setText(f"{cx - half_len:.1f}")
        self.edt_xmax.setText(f"{cx + half_len:.1f}")
        self.edt_ymin.setText(f"{cy - half_wid:.1f}")
        self.edt_ymax.setText(f"{cy + half_wid:.1f}")

        # Z = 桥底面高度（用点云最高点估算） - 偏移
        if self.points is not None and len(self.points) > 0:
            z_bottom = self.points[:, 2].max() - z_offset
        else:
            z_bottom = clearance
        self.edt_z.setText(f"{z_bottom:.1f}")

        # 立方体航线：桥墩扫描
        self.edt_cx.setText(f"{cx:.1f}")
        self.edt_cy.setText(f"{cy:.1f}")
        self.edt_cz.setText(f"{clearance:.1f}")
        self.edt_dx.setText(f"{bridge_wid * 0.3:.1f}")
        self.edt_dy.setText(f"{bridge_wid * 0.3:.1f}")
        self.edt_dz.setText(f"{clearance:.1f}")

        # 扫描间距根据跨距调整
        spacing = max(span / 5, 2.0)
        self.edt_spacing.setText(f"{spacing:.1f}")

        bridge_name = self.cmb_bridge_type.currentText()
        print(f"[Bridge] Applied: {bridge_name}, L={bridge_len}m, W={bridge_wid}m, Clearance={clearance}m, Span={span}m")
        self.lbl_info.setText(f"桥梁: {bridge_name}, {bridge_len}m x {bridge_wid}m")

    def _toggle_heading(self, state):
        """切换机头方向箭头显示"""
        self.viewer.show_heading = (state == Qt.Checked)
        if self.waypoints:
            self.viewer.add_route(self.waypoints)
            self._check_safety_distance()

    def _on_safe_dist_changed(self, text):
        """安全距离输入变更"""
        try:
            val = float(text)
            if val > 0:
                self.viewer._safe_distance = val
                # 重新检查安全距离
                if self.waypoints:
                    self._check_safety_distance()
        except ValueError:
            pass

    def _display_route(self):
        self.viewer.add_route(self.waypoints)
        self.lbl_info.setText(f"航点: {len(self.waypoints)}")
        self._check_safety_distance()

    def _on_waypoint_edited(self, idx, new_pos, new_quat):
        """航点被拖动编辑后的回调"""
        if idx < len(self.waypoints):
            self.waypoints[idx]['pos'] = new_pos
            if new_quat is not None:
                self.waypoints[idx]['quat'] = new_quat
            # 重新渲染航线（更新路径线和箭头）
            self.viewer.add_route(self.waypoints)
            self._check_safety_distance()

    def _check_safety_distance(self):
        """检测航点间安全距离 + 碰撞检测"""
        if len(self.waypoints) < 2:
            return
        safe_dist = self.viewer._safe_distance
        violations = []
        for i in range(len(self.waypoints) - 1):
            d = np.linalg.norm(self.waypoints[i+1]['pos'] - self.waypoints[i]['pos'])
            if d < safe_dist:
                violations.append((i, i+1, d))

        # 标红过近的航点
        for i, j, d in violations:
            if i < len(self.viewer._waypoint_actors):
                self.viewer._waypoint_actors[i].GetProperty().SetColor(1.0, 0.0, 0.0)
            if j < len(self.viewer._waypoint_actors):
                self.viewer._waypoint_actors[j].GetProperty().SetColor(1.0, 0.0, 0.0)

        # 碰撞检测：航点与点云过近
        collision_count = 0
        collision_dist = safe_dist * 0.5  # 碰撞距离 = 安全距离的一半
        if self.points is not None and len(self.points) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(self.points)
            for i, wp in enumerate(self.waypoints):
                dist, _ = tree.query(wp['pos'])
                if dist < collision_dist:
                    collision_count += 1
                    if i < len(self.viewer._waypoint_actors):
                        # 品红色 = 碰撞
                        self.viewer._waypoint_actors[i].GetProperty().SetColor(1.0, 0.0, 1.0)

        # 状态栏信息
        msgs = [f"航点: {len(self.waypoints)}"]
        if violations:
            msgs.append(f"{len(violations)} 对过近 (<{safe_dist}m)")
        if collision_count:
            msgs.append(f"{collision_count} 个碰撞 (<{collision_dist:.1f}m)")
        self.lbl_info.setText(" | ".join(msgs))
        self.viewer.vtk_widget.GetRenderWindow().Render()

    # ─── 清除航线 ───
    def clear_route(self):
        self.waypoints = []
        self.viewer.clear_actors()
        if self.points is not None:
            self.viewer.add_point_cloud(self.points)
        self.lbl_info.setText("航点: 0")

    # ─── 保存航线 ───
    def save_route(self):
        if not self.waypoints:
            QMessageBox.information(self, "提示", "没有航线可保存")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存航线", "", "JSON 文件 (*.json)"
        )
        if not path:
            return

        # 桥梁参数
        bridge_params = {
            "type": self.cmb_bridge_type.currentText(),
            "type_index": self.cmb_bridge_type.currentIndex(),
            "length_m": self.edt_bridge_len.text(),
            "width_m": self.edt_bridge_wid.text(),
            "clearance_m": self.edt_bridge_clr.text(),
            "span_m": self.edt_bridge_span.text(),
        }

        data = {
            "version": "2.0",
            "description": "Bridge inspection route",
            "bridge": bridge_params,
            "waypoint_count": len(self.waypoints),
            "waypoints": []
        }

        for wp in self.waypoints:
            data["waypoints"].append({
                "position": {
                    "x": round(float(wp['pos'][0]), 4),
                    "y": round(float(wp['pos'][1]), 4),
                    "z": round(float(wp['pos'][2]), 4)
                },
                "quaternion": {
                    "w": round(float(wp['quat'][0]), 6),
                    "x": round(float(wp['quat'][1]), 6),
                    "y": round(float(wp['quat'][2]), 6),
                    "z": round(float(wp['quat'][3]), 6)
                },
                "speed": round(float(wp['speed']), 2),
                "action": wp['action']
            })

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "已保存", f"航线已保存到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")

    # ─── 加载航线 ───
    def load_route(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载航线", "", "JSON 文件 (*.json)"
        )
        if not path:
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.waypoints = []
            for wp in data.get('waypoints', []):
                pos = wp['position']
                quat = wp['quaternion']
                self.waypoints.append({
                    'pos': np.array([pos['x'], pos['y'], pos['z']], dtype=np.float64),
                    'quat': np.array([quat['w'], quat['x'], quat['y'], quat['z']], dtype=np.float64),
                    'speed': wp.get('speed', 2.0),
                    'action': wp.get('action', 'fly')
                })

            # 恢复桥梁参数
            bridge = data.get('bridge', {})
            if bridge:
                idx = bridge.get('type_index', 0)
                if 0 <= idx < self.cmb_bridge_type.count():
                    self.cmb_bridge_type.setCurrentIndex(idx)
                self.edt_bridge_len.setText(str(bridge.get('length_m', '100')))
                self.edt_bridge_wid.setText(str(bridge.get('width_m', '15')))
                self.edt_bridge_clr.setText(str(bridge.get('clearance_m', '8')))
                self.edt_bridge_span.setText(str(bridge.get('span_m', '30')))

            self._display_route()
            QMessageBox.information(self, "已加载", f"已加载 {len(self.waypoints)} 个航点")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败:\n{str(e)}")


# ─── 入口 ────────────────────────────────────────────────────
def main():
    if not VTK_AVAILABLE:
        print("Error: VTK is required")
        print("Run: pip install vtk")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
