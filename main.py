"""
桥梁巡检无人机航线规划工具
Bridge Inspection Drone Waypoint Planner

功能：
- 加载 PCD 点云文件并 3D 可视化
- 设计平面航线（桥底面弓字形扫描）
- 设计立方体航线（桥柱螺旋线/Z字形扫描）
- 航点含位置(xyz) + 四元数(wxyz) + 速度
- 导出/导入 JSON 航线文件
"""

import sys
import json
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt
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
            vtkPolyLine, vtkFollower, vtkVectorText
        )
    except ImportError:
        VTK_AVAILABLE = False
        print("[WARNING] VTK not installed. Run: pip install vtk")


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


# ─── 3D 可视化组件 ───────────────────────────────────────────
class VTKViewer(QWidget):
    """嵌入 PyQt5 的 VTK 3D 点云/航线可视化组件"""

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

        self._timer_id = self.startTimer(30)

    def timerEvent(self, event):
        if VTK_AVAILABLE and self.interactor:
            self.interactor.ProcessEvents()

    def clear_actors(self):
        for actor in self._actors:
            self.renderer.RemoveActor(actor)
        self._actors.clear()

    def add_point_cloud(self, points):
        """显示点云"""
        if not VTK_AVAILABLE or len(points) == 0:
            return
        self.clear_actors()
        self.points_data = points

        vtk_points = vtkPoints()
        vtk_array = numpy_to_vtk(points.astype(np.float64), deep=True)
        vtk_array.SetName('Points')
        vtk_points.SetData(vtk_array)

        polydata = vtkPolyData()
        polydata.SetPoints(vtk_points)

        glyph = vtkVertexGlyphFilter()
        glyph.SetInputData(polydata)
        glyph.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.7, 0.85, 1.0)
        actor.GetProperty().SetPointSize(2)

        self.renderer.AddActor(actor)
        self._actors.append(actor)

        self.renderer.ResetCamera()
        self._update_view()
        print(f"[VTK] Point cloud loaded: {len(points)} points")

    def add_route(self, waypoints):
        """显示航线和航点"""
        if not VTK_AVAILABLE or len(waypoints) == 0:
            return

        # 清除旧的航线 actor（保留点云）
        to_remove = []
        for i, actor in enumerate(self._actors):
            if actor.GetProperty().GetPointSize() != 2:
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

        # ─── 航点标记（红色球）───
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

        self._update_view()
        print(f"[VTK] Route displayed: {n} waypoints")

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


# ─── 主窗口 ──────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """桥梁巡检航线规划工具 - 主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bridge Inspection Drone Waypoint Planner")
        self.resize(1400, 900)

        self.points = None
        self.waypoints = []

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

        # ─── 右侧控制面板 ───
        ctrl = QWidget()
        ctrl.setFixedWidth(340)
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setSpacing(6)

        # -- 加载点云 --
        grp_load = QGroupBox("Load Point Cloud")
        gl = QVBoxLayout(grp_load)
        self.btn_load = QPushButton("Open PCD File")
        gl.addWidget(self.btn_load)
        self.lbl_pc_info = QLabel("No point cloud loaded")
        gl.addWidget(self.lbl_pc_info)
        ctrl_layout.addWidget(grp_load)

        # -- 平面航线（桥底面弓字形扫描）--
        grp_flat = QGroupBox("Flat Surface Route (Zigzag Scan)")
        fl = QVBoxLayout(grp_flat)
        fl.addWidget(self._label("Scan Area X Range:"))
        h = QHBoxLayout()
        self.edt_xmin = QLineEdit("-10"); self.edt_xmax = QLineEdit("10")
        h.addWidget(QLabel("Min X")); h.addWidget(self.edt_xmin)
        h.addWidget(QLabel("Max X")); h.addWidget(self.edt_xmax)
        fl.addLayout(h)

        fl.addWidget(self._label("Scan Area Y Range:"))
        h = QHBoxLayout()
        self.edt_ymin = QLineEdit("-10"); self.edt_ymax = QLineEdit("10")
        h.addWidget(QLabel("Min Y")); h.addWidget(self.edt_ymin)
        h.addWidget(QLabel("Max Y")); h.addWidget(self.edt_ymax)
        fl.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Altitude Z:"))
        self.edt_z = QLineEdit("5"); h.addWidget(self.edt_z)
        h.addWidget(QLabel("Line Spacing:"))
        self.edt_spacing = QLineEdit("2"); h.addWidget(self.edt_spacing)
        fl.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Speed (m/s):"))
        self.edt_flat_speed = QLineEdit("3"); h.addWidget(self.edt_flat_speed)
        fl.addLayout(h)

        self.btn_flat = QPushButton("Generate Flat Route")
        fl.addWidget(self.btn_flat)
        ctrl_layout.addWidget(grp_flat)

        # -- 立方体航线（桥柱环绕扫描）--
        grp_cube = QGroupBox("Cube Route (Pillar Surround Scan)")
        cl = QVBoxLayout(grp_cube)

        cl.addWidget(self._label("Pillar Center (x,y,z):"))
        h = QHBoxLayout()
        self.edt_cx = QLineEdit("0"); self.edt_cy = QLineEdit("0"); self.edt_cz = QLineEdit("0")
        h.addWidget(self.edt_cx); h.addWidget(self.edt_cy); h.addWidget(self.edt_cz)
        cl.addLayout(h)

        cl.addWidget(self._label("Pillar Size (Wx, Wy, H):"))
        h = QHBoxLayout()
        self.edt_dx = QLineEdit("4"); self.edt_dy = QLineEdit("4"); self.edt_dz = QLineEdit("8")
        h.addWidget(self.edt_dx); h.addWidget(self.edt_dy); h.addWidget(self.edt_dz)
        cl.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("H Step:"))
        self.edt_cstep = QLineEdit("2"); h.addWidget(self.edt_cstep)
        h.addWidget(QLabel("V Step:"))
        self.edt_vstep = QLineEdit("2"); h.addWidget(self.edt_vstep)
        cl.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Standoff Dist:"))
        self.edt_dist = QLineEdit("3"); h.addWidget(self.edt_dist)
        h.addWidget(QLabel("Speed:"))
        self.edt_cspeed = QLineEdit("2"); h.addWidget(self.edt_cspeed)
        cl.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Route Type:"))
        self.cbo_cube_type = QComboBox()
        self.cbo_cube_type.addItems(["Spiral", "Zigzag"])
        h.addWidget(self.cbo_cube_type)
        cl.addLayout(h)

        self.btn_cube = QPushButton("Generate Cube Route")
        cl.addWidget(self.btn_cube)
        ctrl_layout.addWidget(grp_cube)

        # -- 航线管理 --
        grp_route = QGroupBox("Route Management")
        rl = QVBoxLayout(grp_route)
        self.lbl_info = QLabel("Waypoints: 0")
        rl.addWidget(self.lbl_info)
        self.btn_clear = QPushButton("Clear Route")
        rl.addWidget(self.btn_clear)
        self.btn_save = QPushButton("Save Route (JSON)")
        rl.addWidget(self.btn_save)
        self.btn_load_route = QPushButton("Load Route (JSON)")
        rl.addWidget(self.btn_load_route)
        ctrl_layout.addWidget(grp_route)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter)

        # -- 信号连接 --
        self.btn_load.clicked.connect(self.load_point_cloud)
        self.btn_flat.clicked.connect(self.generate_flat_route)
        self.btn_cube.clicked.connect(self.generate_cube_route)
        self.btn_clear.clicked.connect(self.clear_route)
        self.btn_save.clicked.connect(self.save_route)
        self.btn_load_route.clicked.connect(self.load_route)

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        return lbl

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #1e1e22; }
            QWidget { color: #ddd; font-family: "Segoe UI", Arial; font-size: 12px; }
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
                padding: 6px 14px; color: #eee;
            }
            QPushButton:hover { background: #4a4a55; }
            QPushButton:pressed { background: #555566; }
            QComboBox {
                background: #2b2b30; border: 1px solid #555; border-radius: 3px;
                padding: 3px 6px; color: #eee;
            }
            QComboBox QAbstractItemView { background: #2b2b30; color: #eee; selection-background-color: #4a9eff; }
        """)

    # ─── 加载点云 ───
    def load_point_cloud(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open PCD Point Cloud", "", "PCD Files (*.pcd);;All Files (*)"
        )
        if not path:
            return
        try:
            self.points = parse_pcd(path)
            self.viewer.add_point_cloud(self.points)
            self.lbl_pc_info.setText(f"Loaded: {os.path.basename(path)} ({len(self.points)} pts)")

            # 根据点云范围自动填充平面航线默认值
            if len(self.points) > 0:
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

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load point cloud:\n{str(e)}")

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
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers")
            return

        if spacing <= 0 or xmin >= xmax or ymin >= ymax:
            QMessageBox.warning(self, "Invalid Input", "Please check parameter ranges")
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
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers")
            return

        if cstep <= 0 or vstep <= 0 or dz <= 0:
            QMessageBox.warning(self, "Invalid Input", "Step size and height must be positive")
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

    # ─── 显示航线 ───
    def _display_route(self):
        self.viewer.add_route(self.waypoints)
        self.lbl_info.setText(f"Waypoints: {len(self.waypoints)}")

    # ─── 清除航线 ───
    def clear_route(self):
        self.waypoints = []
        self.viewer.clear_actors()
        if self.points is not None:
            self.viewer.add_point_cloud(self.points)
        self.lbl_info.setText("Waypoints: 0")

    # ─── 保存航线 ───
    def save_route(self):
        if not self.waypoints:
            QMessageBox.information(self, "Info", "No route to save")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Route", "", "JSON Files (*.json)"
        )
        if not path:
            return

        data = {
            "version": "1.0",
            "description": "Bridge inspection route",
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
            QMessageBox.information(self, "Saved", f"Route saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed:\n{str(e)}")

    # ─── 加载航线 ───
    def load_route(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Route", "", "JSON Files (*.json)"
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

            self._display_route()
            QMessageBox.information(self, "Loaded", f"Loaded {len(self.waypoints)} waypoints")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed:\n{str(e)}")


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
