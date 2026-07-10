"""VTK 3D 点云/航线可视化组件"""

import numpy as np
from quaternion_utils import quaternion_forward
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

# VTK 延迟到 __init__ 中导入，加快模块加载速度


# ─── Foxglove 风格交互：左键平移，右键旋转 ───────────────────
# 不继承任何 vtkInteractorStyle 子类，直接用 vtkObject 的 observer 机制绑定事件
class FoxgloveInteractorStyle(object):
    """通过 VTK observer 机制实现：左键=平移，右键=旋转，滚轮=缩放"""

    def __init__(self):
        self._vtk_viewer = None
        self._rwi = None
        self._mode = None
        self._prev_pos = None
        self._renderer = None

    def set_viewer(self, viewer):
        self._vtk_viewer = viewer
        self._renderer = viewer.renderer

    def set_interactor(self, rwi):
        self._rwi = rwi
        rwi.AddObserver("LeftButtonPressEvent", self._on_left_down)
        rwi.AddObserver("LeftButtonReleaseEvent", self._on_left_up)
        rwi.AddObserver("RightButtonPressEvent", self._on_right_down)
        rwi.AddObserver("RightButtonReleaseEvent", self._on_right_up)
        rwi.AddObserver("MiddleButtonPressEvent", self._on_mid_down)
        rwi.AddObserver("MiddleButtonReleaseEvent", self._on_mid_up)
        rwi.AddObserver("MouseMoveEvent", self._on_move)
        rwi.AddObserver("MouseWheelForwardEvent", self._on_wheel_fwd)
        rwi.AddObserver("MouseWheelBackwardEvent", self._on_wheel_bwd)

    def _on_left_down(self, obj, event):
        v = self._vtk_viewer
        rwi = self._rwi
        if v is None:
            self._mode = 'pan'
            self._prev_pos = rwi.GetEventPosition()
            return

        # FPV模式下左键不做普通操作
        if v.fpv_mode:
            return

        ctrl = rwi.GetControlKey()
        pos = rwi.GetEventPosition()

        if v.polygon_mode:
            v._poly_click_start = pos
            return

        if v.place_mode:
            v._poly_click_start = pos
            return

        if v.inspect_mode:
            v._poly_click_start = pos
            return

        if v.line_mode:
            v._poly_click_start = pos
            return

        if ctrl:
            wp_idx = v._find_nearest_waypoint(pos[0], pos[1])
            if wp_idx >= 0:
                v._start_wp_edit(wp_idx, pos[0], pos[1])
                return

        self._mode = 'pan'
        self._prev_pos = pos

    def _on_left_up(self, obj, event):
        v = self._vtk_viewer
        rwi = self._rwi

        if v and v.polygon_mode and v._poly_click_start is not None:
            pos = rwi.GetEventPosition()
            start = v._poly_click_start
            v._poly_click_start = None
            if abs(pos[0] - start[0]) < 5 and abs(pos[1] - start[1]) < 5:
                p = v._pick_3d(pos[0], pos[1])
                if p is not None:
                    v._add_polygon_point(p)
            self._mode = None
            self._prev_pos = None
            return

        if v and v.place_mode and v._poly_click_start is not None:
            pos = rwi.GetEventPosition()
            start = v._poly_click_start
            v._poly_click_start = None
            if abs(pos[0] - start[0]) < 5 and abs(pos[1] - start[1]) < 5:
                p = v._pick_3d(pos[0], pos[1])
                if p is not None:
                    v._update_place_preview(p)
            self._mode = None
            self._prev_pos = None
            return

        if v and v.inspect_mode and v._poly_click_start is not None:
            pos = rwi.GetEventPosition()
            start = v._poly_click_start
            v._poly_click_start = None
            if abs(pos[0] - start[0]) < 5 and abs(pos[1] - start[1]) < 5:
                p = v._pick_3d(pos[0], pos[1])
                if p is not None:
                    v._add_inspect_point(p)
            self._mode = None
            self._prev_pos = None
            return

        if v and v.line_mode and v._poly_click_start is not None:
            pos = rwi.GetEventPosition()
            start = v._poly_click_start
            v._poly_click_start = None
            if abs(pos[0] - start[0]) < 5 and abs(pos[1] - start[1]) < 5:
                p = v._pick_3d(pos[0], pos[1])
                if p is not None:
                    v._add_line_point(p)
            self._mode = None
            self._prev_pos = None
            return

        if v and v._wp_editing:
            v._end_wp_edit()
            self._mode = None
            self._prev_pos = None
            return

        self._mode = None
        self._prev_pos = None

    def _on_right_down(self, obj, event):
        v = self._vtk_viewer
        rwi = self._rwi
        # FPV模式下右键由FPV系统处理
        if v and v.fpv_mode:
            return
        if v and v.polygon_mode:
            if len(v._poly_points) >= 3:
                pts = [p.tolist() for p in v._poly_points]
                v.polygon_finished.emit(pts)
                v.exit_polygon_mode(clear_markers=False)
            else:
                v.exit_polygon_mode()
            return
        if v and v.place_mode:
            if v._place_preview_pos is not None:
                v.place_picked.emit(v._place_preview_pos)
                v.exit_place_mode(clear_marker=False)
            else:
                v.exit_place_mode()
            return
        if v and v.inspect_mode:
            if len(v._inspect_points) > 0:
                pts = [p.tolist() for p in v._inspect_points]
                v.inspect_points_confirmed.emit(pts)
                v.exit_inspect_mode(clear_markers=False)
            else:
                v.exit_inspect_mode()
            return
        if v and v.line_mode:
            if len(v._line_points) == 2:
                pts = [p.tolist() for p in v._line_points]
                v.line_points_confirmed.emit(pts)
                v.exit_line_mode(clear_markers=False)
            else:
                v.exit_line_mode()
            return
        self._mode = 'rotate'
        self._prev_pos = rwi.GetEventPosition()

    def _on_right_up(self, obj, event):
        self._mode = None
        self._prev_pos = None

    def _on_mid_down(self, obj, event):
        self._mode = 'dolly'
        self._prev_pos = self._rwi.GetEventPosition()

    def _on_mid_up(self, obj, event):
        self._mode = None
        self._prev_pos = None

    def _on_wheel_fwd(self, obj, event):
        if self._vtk_viewer and self._vtk_viewer.fpv_mode:
            return
        self._dolly(1.1)
        self._rwi.Render()

    def _on_wheel_bwd(self, obj, event):
        if self._vtk_viewer and self._vtk_viewer.fpv_mode:
            return
        self._dolly(0.9)
        self._rwi.Render()

    def _dolly(self, factor):
        ren = self._renderer
        if ren is None:
            return
        cam = ren.GetActiveCamera()
        if cam.GetParallelProjection():
            cam.SetParallelScale(cam.GetParallelScale() / factor)
        else:
            cam.Dolly(factor)
        ren.ResetCameraClippingRange()

    def _on_move(self, obj, event):
        v = self._vtk_viewer
        rwi = self._rwi

        # FPV模式下由FPV系统处理鼠标，跳过普通相机操作
        if v and v.fpv_mode:
            return

        if v and v._wp_editing:
            pos = rwi.GetEventPosition()
            v._update_wp_edit(pos[0], pos[1])
            return

        if self._mode is None or self._prev_pos is None:
            return

        pos = rwi.GetEventPosition()
        dx = pos[0] - self._prev_pos[0]
        dy = pos[1] - self._prev_pos[1]
        self._prev_pos = pos

        ren = self._renderer
        if ren is None:
            return
        cam = ren.GetActiveCamera()

        if self._mode == 'pan':
            view_focus = cam.GetFocalPoint()
            ren.SetWorldPoint(view_focus[0], view_focus[1], view_focus[2], 1.0)
            ren.WorldToDisplay()
            fd = ren.GetDisplayPoint()
            ren.SetDisplayPoint(fd[0] + dx, fd[1] + dy, fd[2])
            ren.DisplayToWorld()
            nf = ren.GetWorldPoint()
            w = nf[3]
            if abs(w) > 1e-10:
                nf = [nf[0]/w, nf[1]/w, nf[2]/w]
            else:
                nf = list(view_focus)
            cam.SetFocalPoint(nf[0], nf[1], nf[2])
            cp = cam.GetPosition()
            cam.SetPosition(
                cp[0] + nf[0] - view_focus[0],
                cp[1] + nf[1] - view_focus[1],
                cp[2] + nf[2] - view_focus[2])

        elif self._mode == 'rotate':
            cam.Azimuth(-dx * 0.5)
            cam.Elevation(-dy * 0.5)
            cam.OrthogonalizeViewUp()

        elif self._mode == 'dolly':
            factor = 1.0 + dy * 0.01
            if factor > 0.01:
                self._dolly(factor)

        ren.ResetCameraClippingRange()
        rwi.Render()


# ─── 3D 可视化组件 ───────────────────────────────────────────
class VTKViewer(QWidget):
    """嵌入 PyQt5 的 VTK 3D 点云/航线可视化组件，支持交互式画框选点"""

    waypoint_edited = pyqtSignal(int, object, object)
    polygon_finished = pyqtSignal(list)
    place_picked = pyqtSignal(object)  # 点击放置模式
    inspect_points_confirmed = pyqtSignal(list)  # 巡检点确认
    line_points_confirmed = pyqtSignal(list)  # 直线起终点确认 [start, end]

    def __init__(self, parent=None):
        super().__init__(parent)
        # ─── VTK 延迟导入（仅在实例化时加载，加快模块导入速度）───
        try:
            import vtk
            from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            from vtkmodules.util.numpy_support import numpy_to_vtk
            from vtkmodules.vtkRenderingCore import (
                vtkActor, vtkPolyDataMapper, vtkRenderer,
                vtkPoints, vtkPolyData, vtkVertexGlyphFilter,
                vtkFollower, vtkVectorText, vtkBillboardTextActor3D
            )
            from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkCubeSource, vtkLineSource, vtkArrowSource
            from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
            from vtkmodules.vtkCommonTransforms import vtkTransform
            from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyLine
            from vtkmodules.vtkFiltersCore import vtkGlyph3D
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
            vtk.vtkOutputWindow.SetGlobalWarningDisplay(0)
            self._vtk_available = True
        except ImportError:
            try:
                import vtk
                from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
                from vtk.util.numpy_support import numpy_to_vtk
                from vtk import (
                    vtkActor, vtkPolyDataMapper, vtkRenderer,
                    vtkPoints, vtkPolyData, vtkVertexGlyphFilter,
                    vtkSphereSource, vtkCubeSource, vtkLineSource, vtkArrowSource, vtkCellArray,
                    vtkPolyLine, vtkFollower, vtkVectorText,
                    vtkBillboardTextActor3D,
                    vtkTransformPolyDataFilter, vtkTransform,
                    vtkGlyph3D, vtkInteractorStyleUser,
                )
                vtk.vtkOutputWindow.SetGlobalWarningDisplay(0)
                self._vtk_available = True
            except ImportError:
                self._vtk_available = False
                layout = QVBoxLayout(self)
                layout.addWidget(QLabel("VTK not installed.\nRun: pip install vtk"))
                return

        # 保存 VTK 类为实例属性，供其他方法使用
        self._vtk = vtk
        self._QVTKRenderWindowInteractor = QVTKRenderWindowInteractor
        self._numpy_to_vtk = numpy_to_vtk
        self._vtkActor = vtkActor
        self._vtkPolyDataMapper = vtkPolyDataMapper
        self._vtkRenderer = vtkRenderer
        self._vtkPoints = vtkPoints
        self._vtkPolyData = vtkPolyData
        self._vtkVertexGlyphFilter = vtkVertexGlyphFilter
        self._vtkSphereSource = vtkSphereSource
        self._vtkCubeSource = vtkCubeSource
        self._vtkLineSource = vtkLineSource
        self._vtkArrowSource = vtkArrowSource
        self._vtkCellArray = vtkCellArray
        self._vtkPolyLine = vtkPolyLine
        self._vtkBillboardTextActor3D = vtkBillboardTextActor3D
        self._vtkTransformPolyDataFilter = vtkTransformPolyDataFilter
        self._vtkTransform = vtkTransform
        self._vtkGlyph3D = vtkGlyph3D
        self._vtkInteractorStyleUser = vtkInteractorStyleUser

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        # ─── 视角切换按钮（右上角覆盖层）───
        from PyQt5.QtWidgets import QFrame, QButtonGroup
        view_frame = QFrame(self.vtk_widget)
        view_frame.setStyleSheet("QFrame { background: rgba(240,240,238,200); border: 1px solid #ccc; border-radius: 6px; }")
        view_frame.setFixedSize(260, 44)
        view_layout = QHBoxLayout(view_frame)
        view_layout.setContentsMargins(4, 2, 4, 2)
        view_layout.setSpacing(4)

        self._view_btns = QButtonGroup(self)
        # (图标, 标签, 名称, 颜色)
        views = [
            ("⬇", "俯", "top", "#5b9bd5"),
            ("⬆", "仰", "bottom", "#70ad47"),
            ("⬛", "前", "front", "#ed7d31"),
            ("▐", "侧", "side", "#9b59b6"),
            ("◆", "透", "persp", "#607d8b"),
        ]
        tooltips = ["从上往下看 (Top)", "从下往上看 (Bottom)", "从正面看 (Front)", "从侧面看 (Side)", "自由透视 (Perspective)"]
        for i, (icon, label, name, color) in enumerate(views):
            btn = QPushButton(f"{icon}\n{label}")
            btn.setFixedSize(42, 32)
            btn.setToolTip(tooltips[i])
            btn.setStyleSheet(f"QPushButton {{ background: #e8e8e6; border: 1px solid #bbb; border-radius: 4px; color: #333; font-size: 11px; padding: 1px; }} QPushButton:checked {{ background: {color}; color: #fff; border-color: {color}; }}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, n=name: self._set_view(n))
            view_layout.addWidget(btn)
            self._view_btns.addButton(btn, i)

        self._view_btns.button(0).setChecked(True)
        QTimer.singleShot(100, lambda: view_frame.move(self.vtk_widget.width() - 270, 8))

        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.renderer.SetBackground(0.95, 0.95, 0.93)  # 奶白色背景

        self._actors = []
        self.points_data = None
        self._cloud_tree = None
        self._cloud_actor = None

        # ─── 点击放置模式 ───
        self.place_mode = False
        self._place_preview_pos = None
        self._place_preview_actor = None

        # ─── 航点编辑状态 ───
        self._wp_editing = False
        self._wp_edit_idx = -1
        self._wp_edit_z = 0.0
        self._wp_edit_offset = None
        self._wp_edit_actor = None
        self._waypoint_actors = []
        self._waypoints_ref = None
        self._safe_distance = 2.0
        self._takeoff_z = 1.0
        self._takeoff_yaw = 0.0
        self._safe_point = (0.0, 0.0, 5.0)
        self.show_heading = True

        # ─── 多边形选择模式 ───
        self.polygon_mode = False
        self._poly_points = []
        self._poly_markers = []
        self._poly_line_actor = None
        self._poly_click_start = None

        # ─── 巡检选点模式 ───
        self.inspect_mode = False
        self._inspect_points = []
        self._inspect_markers = []

        # ─── 直线起终点选点模式 ───
        self.line_mode = False
        self._line_points = []
        self._line_markers = []

        # ─── 裁剪平面 ───
        self._clip_plane_actors = {}  # {'x': actor, 'y': actor, 'z': actor}
        self._clip_plane_positions = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self._clip_plane_visible = {'x': False, 'y': False, 'z': False}
        self._clip_bounds = None  # 点云包围盒，用于确定平面大小

        # ─── FPV 无人机视角模式 ───
        self.fpv_mode = False
        self._fpv_first_person = True  # True=第一人称, False=第三人称
        self._fpv_pos = np.array([0.0, 0.0, 5.0])  # 无人机位置
        self._fpv_yaw = 0.0    # 偏航角（度），0=朝+X方向
        self._fpv_pitch = 0.0  # 俯仰角（度），0=水平，正=向下看
        self._fpv_speed = 0.5  # 飞行速度（米/按键）
        self._fpv_look_speed = 0.3  # 鼠标灵敏度
        self._fpv_keys = set()  # 当前按下的键
        self._fpv_mouse_active = False  # 右键按下时鼠标控制视角
        self._fpv_prev_mouse = None
        self._fpv_drone_actor = None  # 无人机模型actor
        self._fpv_on_mark = None  # 打点回调函数
        self._fpv_fov = 80.0  # FPV相机视场角

        self._timer_id = self.startTimer(16)  # ~60fps

    def timerEvent(self, event):
        if getattr(self, '_vtk_available', False) and getattr(self, 'interactor', None):
            self.interactor.ProcessEvents()
            # FPV模式下持续处理按键移动
            if getattr(self, 'fpv_mode', False):
                self.fpv_tick()

    def _create_clip_plane_actor(self, axis, color):
        """创建裁剪平面actor（半透明矩形）"""
        vtk = self._vtk
        plane_source = vtk.vtkPlaneSource()
        mapper = self._vtkPolyDataMapper()
        mapper.SetInputConnection(plane_source.GetOutputPort())
        actor = self._vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(0.3)
        actor.GetProperty().SetLighting(False)
        actor.SetVisibility(False)
        # 保存plane_source引用以便后续更新
        actor._plane_source = plane_source
        actor._axis = axis
        self.renderer.AddActor(actor)
        return actor

    def _update_clip_plane_geometry(self, axis, position):
        """更新裁剪平面的几何形状（位置和大小）"""
        if axis not in self._clip_plane_actors:
            return
        actor = self._clip_plane_actors[axis]
        plane_source = actor._plane_source

        # 获取点云包围盒
        bounds = self._clip_bounds
        if bounds is None:
            return

        # 计算平面大小（取包围盒在另外两个轴上的范围）
        if axis == 'x':
            y_range = bounds['y_max'] - bounds['y_min']
            z_range = bounds['z_max'] - bounds['z_min']
            size = max(y_range, z_range) * 1.2
            center = [position, (bounds['y_min'] + bounds['y_max']) / 2, (bounds['z_min'] + bounds['z_max']) / 2]
            normal = [1, 0, 0]
            # 设置平面的两个角点
            p1 = [position, bounds['y_min'] - size * 0.1, bounds['z_min'] - size * 0.1]
            p2 = [position, bounds['y_max'] + size * 0.1, bounds['z_min'] - size * 0.1]
            p3 = [position, bounds['y_min'] - size * 0.1, bounds['z_max'] + size * 0.1]
        elif axis == 'y':
            x_range = bounds['x_max'] - bounds['x_min']
            z_range = bounds['z_max'] - bounds['z_min']
            size = max(x_range, z_range) * 1.2
            center = [(bounds['x_min'] + bounds['x_max']) / 2, position, (bounds['z_min'] + bounds['z_max']) / 2]
            normal = [0, 1, 0]
            p1 = [bounds['x_min'] - size * 0.1, position, bounds['z_min'] - size * 0.1]
            p2 = [bounds['x_max'] + size * 0.1, position, bounds['z_min'] - size * 0.1]
            p3 = [bounds['x_min'] - size * 0.1, position, bounds['z_max'] + size * 0.1]
        else:  # z
            x_range = bounds['x_max'] - bounds['x_min']
            y_range = bounds['y_max'] - bounds['y_min']
            size = max(x_range, y_range) * 1.2
            center = [(bounds['x_min'] + bounds['x_max']) / 2, (bounds['y_min'] + bounds['y_max']) / 2, position]
            normal = [0, 0, 1]
            p1 = [bounds['x_min'] - size * 0.1, bounds['y_min'] - size * 0.1, position]
            p2 = [bounds['x_max'] + size * 0.1, bounds['y_min'] - size * 0.1, position]
            p3 = [bounds['x_min'] - size * 0.1, bounds['y_max'] + size * 0.1, position]

        plane_source.SetOrigin(p1)
        plane_source.SetPoint1(p2)
        plane_source.SetPoint2(p3)
        plane_source.SetNormal(normal)
        plane_source.Update()

    def set_clip_plane(self, axis, position):
        """设置裁剪平面位置"""
        if not self._vtk_available:
            return
        self._clip_plane_positions[axis] = position
        if axis in self._clip_plane_actors:
            self._update_clip_plane_geometry(axis, position)
            self.vtk_widget.GetRenderWindow().Render()

    def show_clip_plane(self, axis, visible):
        """显示/隐藏裁剪平面"""
        if not self._vtk_available:
            return
        self._clip_plane_visible[axis] = visible
        if visible and axis not in self._clip_plane_actors:
            # 创建平面actor
            colors = {'x': (1, 0.3, 0.3), 'y': (0.3, 1, 0.3), 'z': (0.3, 0.3, 1)}
            self._clip_plane_actors[axis] = self._create_clip_plane_actor(axis, colors[axis])
            # 设置初始位置
            if self._clip_bounds is not None:
                if axis == 'x':
                    pos = self._clip_bounds['x_max']
                elif axis == 'y':
                    pos = self._clip_bounds['y_max']
                else:
                    pos = self._clip_bounds['z_max']
                self._clip_plane_positions[axis] = pos
            self._update_clip_plane_geometry(axis, self._clip_plane_positions[axis])
        if axis in self._clip_plane_actors:
            self._clip_plane_actors[axis].SetVisibility(visible)
            self.vtk_widget.GetRenderWindow().Render()

    def _update_clip_bounds(self, points):
        """更新点云包围盒，用于裁剪平面大小"""
        if points is None or len(points) == 0:
            self._clip_bounds = None
            return
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        self._clip_bounds = {
            'x_min': mn[0], 'x_max': mx[0],
            'y_min': mn[1], 'y_max': mx[1],
            'z_min': mn[2], 'z_max': mx[2],
        }

    def clear_clip_planes(self):
        """清除所有裁剪平面"""
        for axis, actor in self._clip_plane_actors.items():
            self.renderer.RemoveActor(actor)
        self._clip_plane_actors.clear()
        self._clip_plane_visible = {'x': False, 'y': False, 'z': False}
        if self.vtk_widget.GetRenderWindow():
            self.vtk_widget.GetRenderWindow().Render()

    # ─── FPV 无人机视角模式 ─────────────────────────────────────

    def toggle_fpv(self, enable=None):
        """切换FPV模式"""
        if enable is None:
            enable = not self.fpv_mode
        self.fpv_mode = enable

        if enable:
            self._create_drone_model()
            self._setup_fpv_keyboard()
            self._enter_fpv_camera()
        else:
            self._remove_drone_model()
            self._exit_fpv_camera()

        if self.vtk_widget.GetRenderWindow():
            self.vtk_widget.GetRenderWindow().Render()

    def _create_drone_model(self):
        """加载STL无人机模型"""
        if self._fpv_drone_actor is not None:
            return
        vtk = self._vtk
        import os

        # 查找STL文件（优先M4T_v2_simple.stl）
        stl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'M4T_v2_simple.stl')
        if not os.path.exists(stl_path):
            # 尝试当前工作目录
            stl_path = os.path.join(os.getcwd(), 'M4T_v2_simple.stl')
        if not os.path.exists(stl_path):
            # 回退到简单线框
            self._create_drone_model_fallback()
            return

        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()

        stl_polydata = reader.GetOutput()
        if stl_polydata.GetNumberOfPoints() == 0:
            self._create_drone_model_fallback()
            return

        # 根据场景大小缩放无人机模型
        bounds = stl_polydata.GetBounds()  # (xmin,xmax,ymin,ymax,zmin,zmax)
        stl_size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])

        scene_size = 5.0
        if self.points_data is not None and len(self.points_data) > 0:
            mn = self.points_data.min(axis=0)
            mx = self.points_data.max(axis=0)
            scene_size = np.linalg.norm(mx - mn) * 0.015  # 场景对角线的1.5%

        scale = max(0.5, scene_size) / stl_size if stl_size > 0 else 1.0

        transform = vtk.vtkTransform()
        transform.Scale(scale, scale, scale)
        # 修正模型姿态：机身放平 + 机头朝前
        transform.RotateX(90)
        transform.RotateY(90)
        # 居中：把模型中心移到原点
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2
        transform.Translate(-cx, -cy, -cz)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(stl_polydata)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        mapper = self._vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())

        actor = self._vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.3, 0.3, 0.3)  # 灰色机身
        actor.GetProperty().SetLighting(True)

        self.renderer.AddActor(actor)
        self._actors.append(actor)
        self._fpv_drone_actor = actor

    def _create_drone_model_fallback(self):
        """STL加载失败时的简单线框模型"""
        vtk = self._vtk
        arm_len = 1.0
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(-arm_len, 0, 0)
        pts.InsertNextPoint(arm_len, 0, 0)
        pts.InsertNextPoint(0, -arm_len, 0)
        pts.InsertNextPoint(0, arm_len, 0)
        pts.InsertNextPoint(0, 0, 0)
        pts.InsertNextPoint(arm_len * 2, 0, 0)
        pts.InsertNextPoint(arm_len * 2, 0, 0)
        pts.InsertNextPoint(arm_len * 1.5, arm_len * 0.4, 0)
        pts.InsertNextPoint(arm_len * 2, 0, 0)
        pts.InsertNextPoint(arm_len * 1.5, -arm_len * 0.4, 0)

        lines = vtk.vtkCellArray()
        for pair in [(0,1),(2,3),(4,5),(6,7),(8,9)]:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)
            line.GetPointIds().SetId(0, pair[0])
            line.GetPointIds().SetId(1, pair[1])
            lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(pts)
        polydata.SetLines(lines)
        mapper = self._vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = self._vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 1, 0)
        actor.GetProperty().SetLineWidth(3)
        actor.GetProperty().SetLighting(False)
        self.renderer.AddActor(actor)
        self._actors.append(actor)
        self._fpv_drone_actor = actor

    def _remove_drone_model(self):
        """移除无人机模型"""
        if self._fpv_drone_actor is not None:
            self.renderer.RemoveActor(self._fpv_drone_actor)
            if self._fpv_drone_actor in self._actors:
                self._actors.remove(self._fpv_drone_actor)
            self._fpv_drone_actor = None

    def _update_drone_model(self):
        """更新无人机模型位置和朝向"""
        if self._fpv_drone_actor is None:
            return
        vtk = self._vtk
        t = vtk.vtkTransform()
        t.Translate(self._fpv_pos.tolist())
        t.RotateZ(self._fpv_yaw)
        t.RotateX(-self._fpv_pitch)  # VTK的旋转方向
        self._fpv_drone_actor.SetUserTransform(t)

    def _setup_fpv_keyboard(self):
        """设置FPV键盘事件"""
        rwi = self.interactor
        if rwi is None:
            return
        # 移除旧的键盘观察者（如果有）
        if hasattr(self, '_fpv_key_press_obs'):
            rwi.RemoveObserver(self._fpv_key_press_obs)
            rwi.RemoveObserver(self._fpv_key_release_obs)
        self._fpv_key_press_obs = rwi.AddObserver("KeyPressEvent", self._fpv_on_key_press)
        self._fpv_key_release_obs = rwi.AddObserver("KeyReleaseEvent", self._fpv_on_key_release)
        # 鼠标事件
        if hasattr(self, '_fpv_right_press_obs'):
            rwi.RemoveObserver(self._fpv_right_press_obs)
            rwi.RemoveObserver(self._fpv_right_release_obs)
            rwi.RemoveObserver(self._fpv_move_obs)
            rwi.RemoveObserver(self._fpv_left_press_obs)
        self._fpv_left_press_obs = rwi.AddObserver("LeftButtonPressEvent", self._fpv_on_left_down)
        self._fpv_right_press_obs = rwi.AddObserver("RightButtonPressEvent", self._fpv_on_right_down)
        self._fpv_right_release_obs = rwi.AddObserver("RightButtonReleaseEvent", self._fpv_on_right_up)
        self._fpv_move_obs = rwi.AddObserver("MouseMoveEvent", self._fpv_on_mouse_move)

    def _fpv_on_key_press(self, obj, event):
        """FPV键盘按下"""
        if not self.fpv_mode:
            return
        key = obj.GetKeySym().lower()
        self._fpv_keys.add(key)

        # 空格键 = 打点
        if key == 'space':
            self._fpv_mark_waypoint()
            return

        # V键 = 退出FPV
        if key == 'v':
            self.toggle_fpv(False)
            return

        # C键 = 切换第一人称/第三人称
        if key == 'c':
            self._fpv_first_person = not self._fpv_first_person
            if self._fpv_drone_actor is not None:
                self._fpv_drone_actor.SetVisibility(not self._fpv_first_person)
            if not self._fpv_first_person:
                # 第三人称：相机在无人机后方看过去
                self._update_third_person_camera()
            else:
                # 第一人称：相机在无人机位置
                self._update_fpv_camera()
            return

    def _fpv_on_key_release(self, obj, event):
        """FPV键盘释放"""
        key = obj.GetKeySym().lower()
        self._fpv_keys.discard(key)

    def _fpv_on_left_down(self, obj, event):
        """FPV左键按下 - 拾取点云点并记录航点"""
        if not self.fpv_mode:
            return
        pos = obj.GetEventPosition()
        picked = self._pick_3d(pos[0], pos[1])
        if picked is not None and self._fpv_on_mark is not None:
            self._fpv_on_mark(picked, self._fpv_pos.copy(), self._fpv_yaw, self._fpv_pitch)

    def _fpv_on_right_down(self, obj, event):
        """FPV右键按下 - 开始鼠标控制视角"""
        if not self.fpv_mode:
            return
        self._fpv_mouse_active = True
        self._fpv_prev_mouse = obj.GetEventPosition()

    def _fpv_on_right_up(self, obj, event):
        """FPV右键释放"""
        self._fpv_mouse_active = False
        self._fpv_prev_mouse = None

    def _fpv_on_mouse_move(self, obj, event):
        """FPV鼠标移动 - 控制视角"""
        if not self.fpv_mode or not self._fpv_mouse_active:
            return
        pos = obj.GetEventPosition()
        if self._fpv_prev_mouse is not None:
            dx = pos[0] - self._fpv_prev_mouse[0]
            dy = pos[1] - self._fpv_prev_mouse[1]
            self._fpv_yaw -= dx * self._fpv_look_speed
            self._fpv_pitch += dy * self._fpv_look_speed
            self._fpv_pitch = max(-89, min(89, self._fpv_pitch))
        self._fpv_prev_mouse = pos
        self._update_fpv_camera()

    def _fpv_mark_waypoint(self):
        """在当前位置记录航点"""
        if self._fpv_on_mark is not None:
            # 计算相机看向的方向，找到与点云的交点
            direction = self._fpv_get_look_direction()
            target = self._fpv_find_intersection(direction)
            if target is not None:
                self._fpv_on_mark(target, self._fpv_pos.copy(), self._fpv_yaw, self._fpv_pitch)

    def _fpv_get_look_direction(self):
        """获取FPV相机朝向的单位向量"""
        yaw_rad = np.radians(self._fpv_yaw)
        pitch_rad = np.radians(self._fpv_pitch)
        dx = np.cos(pitch_rad) * np.cos(yaw_rad)
        dy = np.cos(pitch_rad) * np.sin(yaw_rad)
        dz = -np.sin(pitch_rad)
        return np.array([dx, dy, dz])

    def _fpv_find_intersection(self, direction, max_dist=100.0):
        """从无人机位置沿方向射线，找到与点云的最近交点"""
        if self.points_data is None or len(self.points_data) == 0:
            return None
        # 用KDTree快速查找
        if self._cloud_tree is None:
            from scipy.spatial import cKDTree
            self._cloud_tree = cKDTree(self.points_data)

        # 沿射线采样若干点
        best_point = None
        best_dist = max_dist
        for t in np.arange(0.5, max_dist, 0.3):
            sample = self._fpv_pos + direction * t
            dist, idx = self._cloud_tree.query(sample)
            if dist < 0.5 and t < best_dist:
                best_dist = t
                best_point = self.points_data[idx]

        return best_point

    def _enter_fpv_camera(self):
        """进入FPV相机模式"""
        cam = self.renderer.GetActiveCamera()
        if self.points_data is not None and len(self.points_data) > 0:
            mn = self.points_data.min(axis=0)
            mx = self.points_data.max(axis=0)
            center = (mn + mx) / 2
            # 放在点云X最小方向前方
            self._fpv_pos = np.array([mn[0] - 5, center[1], center[2]])
            # 计算朝向点云中心的偏航角
            dx = center[0] - self._fpv_pos[0]
            dy = center[1] - self._fpv_pos[1]
            self._fpv_yaw = np.degrees(np.arctan2(dy, dx))
            self._fpv_pitch = 0.0

        # 设置裁剪范围（近平面0.1，远平面1000）
        cam.SetClippingRange(0.1, 1000.0)

        # FPV视角下隐藏无人机模型
        if self._fpv_drone_actor is not None:
            self._fpv_drone_actor.SetVisibility(False)

        self._update_fpv_camera()

    def _exit_fpv_camera(self):
        """退出FPV相机模式，恢复默认视角"""
        self.renderer.ResetCamera()
        self._fpv_keys.clear()
        self._fpv_mouse_active = False

    def _update_fpv_camera(self, render=True):
        """更新FPV相机到无人机位置和朝向"""
        cam = self.renderer.GetActiveCamera()
        pos = self._fpv_pos
        direction = self._fpv_get_look_direction()
        focal = pos + direction * 10  # 看向的方向

        cam.SetPosition(pos.tolist())
        cam.SetFocalPoint(focal.tolist())
        cam.SetViewUp(0, 0, 1)
        cam.SetViewAngle(self._fpv_fov)

        self._update_drone_model()
        if render and self.vtk_widget.GetRenderWindow():
            self.vtk_widget.GetRenderWindow().Render()

    def _update_third_person_camera(self, render=True):
        """第三人称视角：相机在无人机后上方，跟随无人机"""
        cam = self.renderer.GetActiveCamera()
        pos = self._fpv_pos
        yaw_rad = np.radians(self._fpv_yaw)

        # 相机在无人机后方10米、上方5米
        behind_dist = 10.0
        up_dist = 5.0
        cam_pos = np.array([
            pos[0] - behind_dist * np.cos(yaw_rad),
            pos[1] - behind_dist * np.sin(yaw_rad),
            pos[2] + up_dist
        ])

        cam.SetPosition(cam_pos.tolist())
        cam.SetFocalPoint(pos.tolist())
        cam.SetViewUp(0, 0, 1)
        cam.SetViewAngle(self._fpv_fov)

        self._update_drone_model()
        if render and self.vtk_widget.GetRenderWindow():
            self.vtk_widget.GetRenderWindow().Render()

    def fpv_tick(self):
        """FPV模式下每帧更新（处理持续按键移动）"""
        if not self.fpv_mode:
            return False

        moved = False
        yaw_rad = np.radians(self._fpv_yaw)
        # 前进方向（水平面）
        forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0])
        right = np.array([-np.sin(yaw_rad), np.cos(yaw_rad), 0])

        for key in self._fpv_keys:
            if key == 'w':
                self._fpv_pos += forward * self._fpv_speed
                moved = True
            elif key == 's':
                self._fpv_pos -= forward * self._fpv_speed
                moved = True
            elif key == 'a':
                self._fpv_pos -= right * self._fpv_speed
                moved = True
            elif key == 'd':
                self._fpv_pos += right * self._fpv_speed
                moved = True
            elif key == 'q':
                self._fpv_pos[2] += self._fpv_speed
                moved = True
            elif key == 'e':
                self._fpv_pos[2] -= self._fpv_speed
                moved = True

        if moved:
            # 只更新相机参数，最后统一渲染一次
            if self._fpv_first_person:
                self._update_fpv_camera(render=False)
            else:
                self._update_third_person_camera(render=False)
            if self.vtk_widget.GetRenderWindow():
                self.vtk_widget.GetRenderWindow().Render()
        return moved

    def clear_actors(self):
        for actor in self._actors:
            self.renderer.RemoveActor(actor)
        self._actors.clear()
        self._cloud_actor = None
        # FPV模式下保留无人机模型
        if self._fpv_drone_actor is not None:
            self.renderer.AddActor(self._fpv_drone_actor)
            self._actors.append(self._fpv_drone_actor)

    def add_point_cloud(self, points, render_mode='auto', point_size=0.05, colors=None, normals=None):
        """显示点云（支持球体/立方体/像素/圆片渲染模式）
        colors: (N, 3) uint8 外部传入的RGB颜色，为None时按高度着色
        normals: (N, 3) 法线数组，render_mode='splat'时用于圆片朝向
        """
        if not self._vtk_available or len(points) == 0:
            return
        self.clear_actors()
        self._add_scene_axes()
        self.points_data = points
        self._cloud_tree = None  # 重建KDTree缓存
        self._update_clip_bounds(points)  # 更新裁剪平面包围盒

        MAX_RENDER_POINTS = 5_000_000
        if len(points) > MAX_RENDER_POINTS:
            voxel_size = self._estimate_voxel_size(points, MAX_RENDER_POINTS)
            render_points = self._voxel_downsample(points, voxel_size)
            # 降采样颜色：取体素内第一个点的颜色
            if colors is not None:
                mn = points.min(axis=0)
                voxel_idx_all = ((points - mn) / voxel_size).astype(np.int64)
                max_idx = voxel_idx_all.max(axis=0) + 1
                keys_all = voxel_idx_all[:, 0] + voxel_idx_all[:, 1] * max_idx[0] + voxel_idx_all[:, 2] * max_idx[0] * max_idx[1]
                # 对render_points中的每个点，找到对应的颜色
                render_voxel_idx = ((render_points - mn) / voxel_size).astype(np.int64)
                render_keys = render_voxel_idx[:, 0] + render_voxel_idx[:, 1] * max_idx[0] + render_voxel_idx[:, 2] * max_idx[0] * max_idx[1]
                # 用searchsorted匹配
                sort_order = np.argsort(keys_all)
                sorted_keys = keys_all[sort_order]
                sorted_colors = colors[sort_order]
                idx = np.searchsorted(sorted_keys, render_keys)
                idx = np.clip(idx, 0, len(sorted_keys) - 1)
                render_colors = sorted_colors[idx]
            else:
                render_colors = None
        else:
            render_points = points
            render_colors = colors

        vtk_points = self._vtkPoints()
        vtk_array = self._numpy_to_vtk(render_points.astype(np.float64), deep=True)
        vtk_array.SetName('Points')
        vtk_points.SetData(vtk_array)

        polydata = self._vtkPolyData()
        polydata.SetPoints(vtk_points)

        if render_colors is not None:
            # 使用外部传入的颜色
            rgb = render_colors.astype(np.uint8)
        else:
            # 按高度着色（蓝→青→绿→黄→红）
            z_vals = render_points[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            z_range = z_max - z_min if z_max > z_min else 1.0

            t = (z_vals - z_min) / z_range

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

            rgb = np.column_stack([r, g, b]).astype(np.uint8)

        vtk_colors = self._numpy_to_vtk(rgb, deep=True, array_type=self._vtk.VTK_UNSIGNED_CHAR)
        vtk_colors.SetName('Colors')
        polydata.GetPointData().SetScalars(vtk_colors)

        SPHERE_THRESHOLD = 200_000
        use_glyph = (render_mode in ('sphere', 'cube')) or (render_mode == 'auto' and len(render_points) <= SPHERE_THRESHOLD)
        use_splat = (render_mode == 'splat')

        if use_splat and normals is not None:
            # 圆片渲染模式：每个点显示为朝法线方向的圆片
            vtk_normals = self._numpy_to_vtk(normals.astype(np.float32), deep=True, array_type=self._vtk.VTK_FLOAT)
            vtk_normals.SetName('Normals')
            polydata.GetPointData().SetNormals(vtk_normals)

            src = self._vtk.vtkRegularPolygonSource()
            src.SetRadius(point_size)
            src.SetNumberOfSides(12)
            src.SetNormal(0, 0, 1)  # 默认法线，会被glyph旋转
            src.Update()

            glyph = self._vtkGlyph3D()
            glyph.SetInputData(polydata)
            glyph.SetSourceConnection(src.GetOutputPort())
            glyph.SetVectorModeToUseNormal()
            glyph.OrientOn()
            glyph.SetScaleModeToDataScalingOff()
            glyph.Update()
            mapper = self._vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            mapper.ScalarVisibilityOn()
        elif use_glyph:
            if render_mode == 'cube':
                src = self._vtkCubeSource()
                src.SetXLength(point_size * 2)
                src.SetYLength(point_size * 2)
                src.SetZLength(point_size * 2)
            else:
                src = self._vtkSphereSource()
                src.SetRadius(point_size)
                src.SetThetaResolution(6)
                src.SetPhiResolution(6)
            glyph = self._vtkGlyph3D()
            glyph.SetInputData(polydata)
            glyph.SetSourceConnection(src.GetOutputPort())
            glyph.ScalingOff()
            glyph.Update()
            mapper = self._vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            mapper.ScalarVisibilityOn()
        else:
            glyph = self._vtkVertexGlyphFilter()
            glyph.SetInputData(polydata)
            glyph.Update()
            mapper = self._vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            mapper.ScalarVisibilityOn()
            mapper.SetScalarModeToDefault()

        actor = self._vtkActor()
        actor.SetMapper(mapper)
        if not use_glyph:
            actor.GetProperty().SetPointSize(max(1, int(point_size * 40)))

        self.renderer.AddActor(actor)
        self._actors.append(actor)
        self._cloud_actor = actor

        self.renderer.ResetCamera()
        self._update_view()

    @staticmethod
    def _estimate_voxel_size(points, target_count):
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        volume = np.prod(mx - mn + 1e-10)
        voxel_vol = volume / target_count
        voxel_size = voxel_vol ** (1.0 / 3.0)
        return max(voxel_size, 0.01)

    @staticmethod
    def _voxel_downsample(points, voxel_size):
        mn = points.min(axis=0)
        voxel_idx = ((points - mn) / voxel_size).astype(np.int64)
        max_idx = voxel_idx.max(axis=0) + 1
        keys = voxel_idx[:, 0] + voxel_idx[:, 1] * max_idx[0] + voxel_idx[:, 2] * max_idx[0] * max_idx[1]

        sort_order = np.argsort(keys)
        sorted_keys = keys[sort_order]
        sorted_points = points[sort_order]

        boundaries = np.concatenate([[0], np.where(np.diff(sorted_keys))[0] + 1, [len(sorted_keys)]])

        counts = np.diff(boundaries).astype(np.float64)
        sums_x = np.add.reduceat(sorted_points[:, 0], boundaries[:-1])
        sums_y = np.add.reduceat(sorted_points[:, 1], boundaries[:-1])
        sums_z = np.add.reduceat(sorted_points[:, 2], boundaries[:-1])

        result = np.column_stack([sums_x, sums_y, sums_z]) / counts[:, None]
        return result

    @staticmethod
    def estimate_normals(points, k=20):
        """用PCA方法估算点云法线，返回 (N,3) 法线数组"""
        from scipy.spatial import cKDTree
        n = len(points)
        normals = np.zeros((n, 3), dtype=np.float64)
        tree = cKDTree(points)
        dists, indices = tree.query(points, k=min(k, n))

        for i in range(n):
            neighbors = points[indices[i]]
            center = neighbors.mean(axis=0)
            cov = np.cov((neighbors - center).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # 最小特征值对应的特征向量就是法线
            normals[i] = eigvecs[:, 0]

        # 统一法线朝向（都朝上，即Z分量为正）
        neg_mask = normals[:, 2] < 0
        normals[neg_mask] *= -1
        # Z分量接近0的，用Y方向统一
        zero_z = np.abs(normals[:, 2]) < 0.1
        if zero_z.any():
            neg_y = normals[zero_z, 1] < 0
            normals[zero_z][neg_y] *= -1

        return normals

    @staticmethod
    def upsample_for_display(points, normals, colors=None, factor=3, radius=None):
        """渲染时增密：在每个点的切平面内插值生成新点
        factor: 每个点生成的新点数（不含原始点）
        radius: 插值半径，None时自动计算
        返回: (new_points, new_colors)
        """
        n = len(points)
        if n == 0:
            return points, colors

        if radius is None:
            # 自动计算：基于平均最近邻距离
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            dists, _ = tree.query(points, k=2)
            radius = np.median(dists[:, 1]) * 0.5

        # 构建切平面坐标系
        # 选一个不平行于法线的参考向量
        abs_n = np.abs(normals)
        ref = np.zeros_like(normals)
        # 对每行选Z最小的轴作为参考
        min_axis = abs_n.argmin(axis=1)
        ref[np.arange(n), min_axis] = 1.0

        tangent1 = np.cross(normals, ref)
        tangent1_norm = np.linalg.norm(tangent1, axis=1, keepdims=True)
        tangent1_norm[tangent1_norm < 1e-10] = 1.0
        tangent1 = tangent1 / tangent1_norm

        tangent2 = np.cross(normals, tangent1)
        tangent2_norm = np.linalg.norm(tangent2, axis=1, keepdims=True)
        tangent2_norm[tangent2_norm < 1e-10] = 1.0
        tangent2 = tangent2 / tangent2_norm

        # 在切平面内生成随机偏移
        np.random.seed(42)  # 固定种子保证一致性
        angles = np.random.uniform(0, 2 * np.pi, (n, factor))
        dists = np.random.uniform(0, radius, (n, factor))

        offsets_x = dists * np.cos(angles)  # (n, factor)
        offsets_y = dists * np.sin(angles)  # (n, factor)

        # 生成新点：原始点 + 切平面偏移
        new_points_list = [points]
        for i in range(factor):
            offset = (offsets_x[:, i:i+1] * tangent1 +
                      offsets_y[:, i:i+1] * tangent2)
            new_points_list.append(points + offset)

        all_points = np.vstack(new_points_list)

        # 处理颜色
        all_colors = None
        if colors is not None:
            color_list = [colors] + [colors] * factor
            all_colors = np.vstack(color_list)

        return all_points, all_colors

    @staticmethod
    def _compute_forward_headings(waypoints):
        n = len(waypoints)
        if n < 2:
            return [np.array([1.0, 0.0, 0.0])]
        headings = []
        for i in range(n):
            # 优先使用航点自带的四元数计算航向
            quat = waypoints[i].get('quat')
            if quat is not None:
                h = quaternion_forward(quat)
                headings.append(h)
                continue
            if i < n - 1:
                d = waypoints[i + 1]['pos'] - waypoints[i]['pos']
            else:
                d = waypoints[i]['pos'] - waypoints[i - 1]['pos']
            norm = np.linalg.norm(d)
            if norm < 1e-10:
                headings.append(np.array([1.0, 0.0, 0.0]))
            else:
                headings.append(d / norm)
        return headings

    @staticmethod
    def _is_corner(waypoints, idx):
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
        return cos_angle < np.cos(np.radians(30))

    def add_route(self, waypoints):
        """显示航线和航点"""
        if not self._vtk_available or len(waypoints) == 0:
            return

        to_remove = []
        for i, actor in enumerate(self._actors):
            if actor != self._cloud_actor:
                to_remove.append(i)
        for i in reversed(to_remove):
            self.renderer.RemoveActor(self._actors[i])
            del self._actors[i]

        self._add_scene_axes()

        n = len(waypoints)
        BATCH_THRESHOLD = 50

        # ── 航线路径 ──
        vtk_pts = self._vtkPoints()
        for wp in waypoints:
            vtk_pts.InsertNextPoint(wp['pos'].tolist())

        polyline = self._vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            polyline.GetPointIds().SetId(i, i)

        cells = self._vtkCellArray()
        cells.InsertNextCell(polyline)

        polydata = self._vtkPolyData()
        polydata.SetPoints(vtk_pts)
        polydata.SetLines(cells)

        mapper = self._vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = self._vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 1.0, 0.3)
        actor.GetProperty().SetLineWidth(3.5)
        self.renderer.AddActor(actor)
        self._actors.append(actor)

        # ── 起飞线: origin → takeoff_z → 安全点 → 第一个航点 ──
        origin = [0.0, 0.0, 0.0]
        takeoff_pt = [0.0, 0.0, self._takeoff_z]
        safe_pt = list(self._safe_point)
        first_wp = waypoints[0]['pos'].tolist()

        path_points = [origin, takeoff_pt, safe_pt, first_wp]
        for k in range(len(path_points) - 1):
            line = self._vtkLineSource()
            line.SetPoint1(path_points[k])
            line.SetPoint2(path_points[k + 1])
            mapper = self._vtkPolyDataMapper()
            mapper.SetInputConnection(line.GetOutputPort())
            actor = self._vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.2, 0.5, 1.0)
            actor.GetProperty().SetLineWidth(3)
            actor.GetProperty().SetOpacity(0.6)
            self.renderer.AddActor(actor)
            self._actors.append(actor)

        # 起飞点球体
        takeoff_sphere = self._vtkSphereSource()
        takeoff_sphere.SetCenter(takeoff_pt)
        takeoff_sphere.SetRadius(0.2)
        takeoff_sphere.Update()
        tm = self._vtkPolyDataMapper()
        tm.SetInputConnection(takeoff_sphere.GetOutputPort())
        ta = self._vtkActor()
        ta.SetMapper(tm)
        ta.GetProperty().SetColor(0.2, 0.5, 1.0)
        self.renderer.AddActor(ta)
        self._actors.append(ta)

        # 安全点球体（绿色）
        safe_sphere = self._vtkSphereSource()
        safe_sphere.SetCenter(safe_pt)
        safe_sphere.SetRadius(0.25)
        safe_sphere.Update()
        sm = self._vtkPolyDataMapper()
        sm.SetInputConnection(safe_sphere.GetOutputPort())
        sa = self._vtkActor()
        sa.SetMapper(sm)
        sa.GetProperty().SetColor(0.2, 0.9, 0.3)
        self.renderer.AddActor(sa)
        self._actors.append(sa)

        # 起飞方向箭头（原点处，按偏航角）
        yaw_rad = np.radians(self._takeoff_yaw)
        arrow_dir = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0])
        arrow_len = 2.0
        arrow_end = np.array(takeoff_pt) + arrow_dir * arrow_len
        arrow_line = self._vtkLineSource()
        arrow_line.SetPoint1(takeoff_pt)
        arrow_line.SetPoint2(arrow_end.tolist())
        arrow_mapper = self._vtkPolyDataMapper()
        arrow_mapper.SetInputConnection(arrow_line.GetOutputPort())
        arrow_actor = self._vtkActor()
        arrow_actor.SetMapper(arrow_mapper)
        arrow_actor.GetProperty().SetColor(1.0, 0.5, 0.0)
        arrow_actor.GetProperty().SetLineWidth(3)
        self.renderer.AddActor(arrow_actor)
        self._actors.append(arrow_actor)

        self._waypoint_actors = []
        self._waypoints_ref = waypoints
        if n >= 2:
            avg_d = np.mean([np.linalg.norm(waypoints[i+1]['pos'] - waypoints[i]['pos']) for i in range(n-1)])
            label_offset = avg_d * 0.08
        else:
            label_offset = 0.2

        if n > BATCH_THRESHOLD:
            # ── 批量模式：用 vtkGlyph3D 一次性渲染所有球体 ──
            glyph_pts = self._vtkPoints()
            for wp in waypoints:
                glyph_pts.InsertNextPoint(wp['pos'].tolist())
            glyph_poly = self._vtkPolyData()
            glyph_poly.SetPoints(glyph_pts)

            glyph_src = self._vtkSphereSource()
            glyph_src.SetRadius(0.15)
            glyph_src.SetThetaResolution(8)
            glyph_src.SetPhiResolution(8)

            glyph = self._vtkGlyph3D()
            glyph.SetInputData(glyph_poly)
            glyph.SetSourceConnection(glyph_src.GetOutputPort())
            glyph.ScalingOff()
            glyph.Update()

            sm = self._vtkPolyDataMapper()
            sm.SetInputConnection(glyph.GetOutputPort())
            sa = self._vtkActor()
            sa.SetMapper(sm)
            sa.GetProperty().SetColor(1.0, 0.2, 0.2)
            self.renderer.AddActor(sa)
            self._actors.append(sa)
            self._waypoint_actors = [sa] * n
        else:
            # ── 少量航点：逐个创建（支持单独拖拽编辑）──
            for wp in waypoints:
                sphere = self._vtkSphereSource()
                sphere.SetCenter(wp['pos'].tolist())
                sphere.SetRadius(0.15)
                sphere.Update()
                m = self._vtkPolyDataMapper()
                m.SetInputConnection(sphere.GetOutputPort())
                a = self._vtkActor()
                a.SetMapper(m)
                a.GetProperty().SetColor(1.0, 0.2, 0.2)
                self.renderer.AddActor(a)
                self._actors.append(a)
                self._waypoint_actors.append(a)

            # 标签只在少量航点时显示
            for i, wp in enumerate(waypoints):
                label = self._vtkBillboardTextActor3D()
                label.SetInput(str(i + 1))
                label.SetPosition(wp['pos'][0], wp['pos'][1], wp['pos'][2] + label_offset * 2)
                label.SetScale(label_offset * 0.6, label_offset * 0.6, label_offset * 0.6)
                label.GetTextProperty().SetColor(0.0, 0.2, 0.8)
                label.GetTextProperty().SetFontSize(24)
                label.GetTextProperty().SetBackgroundColor(1.0, 1.0, 1.0)
                label.GetTextProperty().SetBackgroundOpacity(0.8)
                label.GetTextProperty().SetFrame(True)
                label.GetTextProperty().SetFrameColor(0.8, 0.8, 0.8)
                self.renderer.AddActor(label)
                self._actors.append(label)

        # ── 投影线：航点 → 目标方向投射到点云上的点 ──
        has_target = any('target_pos' in wp for wp in waypoints)
        if has_target:
            for i, wp in enumerate(waypoints):
                if 'target_pos' not in wp:
                    continue
                pos = wp['pos']
                tgt = wp['target_pos']
                proj_pt = self._project_to_cloud(pos, tgt)
                if proj_pt is not None:
                    # 连线：航点 → 投影点
                    line = self._vtkLineSource()
                    line.SetPoint1(pos.tolist())
                    line.SetPoint2(proj_pt.tolist())
                    mapper = self._vtkPolyDataMapper()
                    mapper.SetInputConnection(line.GetOutputPort())
                    actor = self._vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(1.0, 0.8, 0.0)
                    actor.GetProperty().SetLineWidth(2)
                    self.renderer.AddActor(actor)
                    self._actors.append(actor)
                    # 投影点标记（小黄色球体）
                    sphere = self._vtkSphereSource()
                    sphere.SetCenter(proj_pt.tolist())
                    sphere.SetRadius(0.12)
                    sphere.Update()
                    sm = self._vtkPolyDataMapper()
                    sm.SetInputConnection(sphere.GetOutputPort())
                    sa = self._vtkActor()
                    sa.SetMapper(sm)
                    sa.GetProperty().SetColor(1.0, 0.9, 0.2)
                    self.renderer.AddActor(sa)
                    self._actors.append(sa)

        self._update_view()

    def _project_to_cloud(self, pos, target_pos, max_dist=50.0, steps=200):
        """从航点沿目标方向投射射线，找到与点云的最近交点
        pos: 航点位置 [x,y,z]
        target_pos: 目标点位置 [x,y,z]
        返回: 投影点 np.array 或 None
        """
        if self.points_data is None or len(self.points_data) == 0:
            return None
        origin = np.array(pos)
        direction = np.array(target_pos) - origin
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return None
        direction = direction / dist
        # 沿射线采样
        t_vals = np.linspace(0.1, min(max_dist, dist * 1.5), steps)
        sample_pts = origin + t_vals[:, np.newaxis] * direction[np.newaxis, :]
        # KDTree 查询每个采样点到最近点云点的距离
        from scipy.spatial import cKDTree
        if not hasattr(self, '_cloud_tree') or self._cloud_tree is None:
            self._cloud_tree = cKDTree(self.points_data)
        dists, idxs = self._cloud_tree.query(sample_pts)
        # 找到距离最小的采样点（射线与点云相交处）
        min_idx = np.argmin(dists)
        if dists[min_idx] < 1.5:  # 射线经过点云附近
            return self.points_data[idxs[min_idx]]
        return None

    def _pick_3d(self, screen_x, screen_y, z_plane=None):
        """屏幕坐标拾取3D点
        z_plane=数值: 射线与指定Z平面求交
        z_plane=None: 拾取点云表面（PointPicker → KDTree射线 → Z=0回退）
        """
        if z_plane is not None:
            return self._ray_z_plane(screen_x, screen_y, z_plane)

        # 1) VTK PointPicker 拾取渲染表面最近点
        picker = self._vtk.vtkPointPicker()
        picker.PickFromListOn()
        if self._cloud_actor:
            picker.AddPickList(self._cloud_actor)
        picker.SetTolerance(0.025)
        picked = picker.Pick(screen_x, screen_y, 0, self.renderer)
        if picked:
            pos = picker.GetPickPosition()
            return np.array(pos)

        # 2) KDTree射线查找：沿射线采样，找最近的点云点
        if self.points_data is not None and len(self.points_data) > 0:
            ren = self.renderer
            ren.SetDisplayPoint(screen_x, screen_y, 0)
            ren.DisplayToWorld()
            near = np.array(ren.GetWorldPoint()[:3])
            w = ren.GetWorldPoint()[3]
            if abs(w) > 1e-10:
                near /= w
            ren.SetDisplayPoint(screen_x, screen_y, 1)
            ren.DisplayToWorld()
            far = np.array(ren.GetWorldPoint()[:3])
            w = ren.GetWorldPoint()[3]
            if abs(w) > 1e-10:
                far /= w
            ray_dir = far - near
            ray_len = np.linalg.norm(ray_dir)
            if ray_len > 1e-10:
                ray_dir /= ray_len
                if self._cloud_tree is None:
                    from scipy.spatial import cKDTree
                    self._cloud_tree = cKDTree(self.points_data)
                best_point = None
                best_dist = float('inf')
                for t in np.arange(0.5, ray_len, 0.3):
                    sample = near + ray_dir * t
                    dist, idx = self._cloud_tree.query(sample)
                    if dist < 1.0 and dist < best_dist:
                        best_dist = dist
                        best_point = self.points_data[idx]
                if best_point is not None:
                    return best_point

        # 3) 回退：Z=0 平面
        return self._ray_z_plane(screen_x, screen_y, 0.0)

    def _ray_z_plane(self, screen_x, screen_y, z_plane):
        """射线与Z平面求交（用 DisplayToWorld，直接用屏幕像素坐标）"""
        ren = self.renderer

        ren.SetDisplayPoint(screen_x, screen_y, 0)
        ren.DisplayToWorld()
        near = np.array(ren.GetWorldPoint()[:3])
        w = ren.GetWorldPoint()[3]
        if abs(w) > 1e-10:
            near /= w

        ren.SetDisplayPoint(screen_x, screen_y, 1)
        ren.DisplayToWorld()
        far = np.array(ren.GetWorldPoint()[:3])
        w = ren.GetWorldPoint()[3]
        if abs(w) > 1e-10:
            far /= w

        ray_dir = far - near
        if abs(ray_dir[2]) < 1e-10:
            return None
        t = (z_plane - near[2]) / ray_dir[2]
        return near + t * ray_dir

    # ─── 航点编辑 ──────────────────────────────────────────────
    def _find_nearest_waypoint(self, screen_x, screen_y):
        if not self._waypoint_actors or not self._waypoints_ref:
            return -1

        ren = self.renderer
        ren.SetDisplayPoint(screen_x, screen_y, 0)
        ren.DisplayToWorld()
        near = np.array(ren.GetWorldPoint()[:3])
        ren.SetDisplayPoint(screen_x, screen_y, 1)
        ren.DisplayToWorld()
        far = np.array(ren.GetWorldPoint()[:3])
        ray_dir = far - near
        ray_dir /= np.linalg.norm(ray_dir)

        best_idx = -1
        best_dist = float('inf')
        PICK_THRESHOLD = 15.0

        for i, wp in enumerate(self._waypoints_ref):
            pos = wp['pos']
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
        ren = self.renderer
        vp = ren.GetViewport()
        win_size = ren.GetRenderWindow().GetSize()
        if win_size[0] == 0 or win_size[1] == 0:
            return None
        vx = (screen_x / win_size[0] - vp[0]) / (vp[2] - vp[0])
        vy = (screen_y / win_size[1] - vp[1]) / (vp[3] - vp[1])

        ren.SetViewPoint(vx, vy, 0)
        ren.ViewToWorld()
        near = np.array(ren.GetWorldPoint()[:3])
        w = ren.GetWorldPoint()[3]
        if abs(w) > 1e-10:
            near /= w

        ren.SetViewPoint(vx, vy, 1)
        ren.ViewToWorld()
        far = np.array(ren.GetWorldPoint()[:3])
        w = ren.GetWorldPoint()[3]
        if abs(w) > 1e-10:
            far /= w

        ray_dir = far - near

        if abs(ray_dir[2]) < 1e-10:
            return None
        t = (z_plane - near[2]) / ray_dir[2]
        if t < 0:
            return None
        return near + t * ray_dir

    def _start_wp_edit(self, idx, screen_x, screen_y):
        self._wp_editing = True
        self._wp_edit_idx = idx
        wp = self._waypoints_ref[idx]
        self._wp_edit_z = wp['pos'][2]

        world_p = self._screen_to_xy_plane(screen_x, screen_y, self._wp_edit_z)
        if world_p is not None:
            self._wp_edit_offset = wp['pos'][:2] - world_p[:2]
        else:
            self._wp_edit_offset = np.zeros(2)

        if idx < len(self._waypoint_actors):
            self._waypoint_actors[idx].GetProperty().SetColor(1.0, 1.0, 0.0)

    def _update_wp_edit(self, screen_x, screen_y):
        if not self._wp_editing:
            return
        world_p = self._screen_to_xy_plane(screen_x, screen_y, self._wp_edit_z)
        if world_p is None:
            return

        new_pos = world_p[:2] + self._wp_edit_offset
        wp = self._waypoints_ref[self._wp_edit_idx]
        wp['pos'][0] = new_pos[0]
        wp['pos'][1] = new_pos[1]

        if self._wp_edit_idx < len(self._waypoint_actors):
            actor = self._waypoint_actors[self._wp_edit_idx]
            actor.SetPosition(new_pos[0] - actor.GetCenter()[0],
                              new_pos[1] - actor.GetCenter()[1], 0)

        self.vtk_widget.GetRenderWindow().Render()

    def _end_wp_edit(self):
        if not self._wp_editing:
            return
        idx = self._wp_edit_idx
        wp = self._waypoints_ref[idx]
        new_pos = wp['pos'].copy()

        if idx < len(self._waypoint_actors):
            self._waypoint_actors[idx].GetProperty().SetColor(1.0, 0.2, 0.2)

        self._wp_editing = False
        self._wp_edit_idx = -1

        self.waypoint_edited.emit(idx, new_pos, None)

    # ─── 多边形选择模式 ─────────────────────────────────────
    def enter_polygon_mode(self):
        if self.points_data is None or len(self.points_data) == 0:
            print("[Polygon] No point cloud loaded")
            return
        self.polygon_mode = True
        self._poly_points = []
        self._clear_polygon()
        self._set_view("top")
        print("[Polygon] 左键点击添加顶点，右键结束绘制，Esc取消")

    def exit_polygon_mode(self, clear_markers=True):
        self.polygon_mode = False
        self._poly_points = []
        self._poly_click_start = None
        if clear_markers:
            self._clear_polygon()
        print("[Polygon] Exited polygon mode.")

    # ─── 点击放置模式 ─────────────────────────────────────
    def enter_place_mode(self):
        if self.points_data is None or len(self.points_data) == 0:
            print("[Place] No point cloud loaded - 请先加载点云")
            return
        self._clear_place_preview()
        self._clear_polygon()
        self.place_mode = True
        self._place_preview_pos = None
        self._place_preview_actor = None
        self._set_view("top")

    def exit_place_mode(self, clear_marker=True):
        self.place_mode = False
        self._place_preview_pos = None
        if clear_marker:
            self._clear_place_preview()
        print("[Place] Exited place mode.")

    def _update_place_preview(self, pos):
        """左键点击时更新预览位置"""
        self._clear_place_preview()
        self._place_preview_pos = pos

        sphere = self._vtkSphereSource()
        sphere.SetCenter(pos.tolist())
        sphere.SetRadius(0.3)
        sphere.Update()
        m = self._vtkPolyDataMapper()
        m.SetInputConnection(sphere.GetOutputPort())
        a = self._vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(0.2, 0.8, 0.2)
        a.GetProperty().SetOpacity(0.7)
        self.renderer.AddActor(a)
        self._place_preview_actor = a
        self.vtk_widget.GetRenderWindow().Render()

    def _clear_place_preview(self):
        if self._place_preview_actor:
            self.renderer.RemoveActor(self._place_preview_actor)
            self._place_preview_actor = None

    def _add_polygon_point(self, pos):
        self._poly_points.append(pos)

        sphere = self._vtkSphereSource()
        sphere.SetCenter(pos.tolist())
        sphere.SetRadius(0.3)
        sphere.Update()
        m = self._vtkPolyDataMapper()
        m.SetInputConnection(sphere.GetOutputPort())
        a = self._vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(1.0, 1.0, 0.0)
        self.renderer.AddActor(a)
        self._poly_markers.append(a)

        self._update_polygon_line()
        self.vtk_widget.GetRenderWindow().Render()

    def _update_polygon_line(self):
        if self._poly_line_actor:
            self.renderer.RemoveActor(self._poly_line_actor)
            self._poly_line_actor = None

        n = len(self._poly_points)
        if n < 2:
            return

        vtk_pts = self._vtkPoints()
        for p in self._poly_points:
            vtk_pts.InsertNextPoint(p.tolist())

        polyline = self._vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            polyline.GetPointIds().SetId(i, i)

        cells = self._vtkCellArray()
        cells.InsertNextCell(polyline)

        polydata = self._vtkPolyData()
        polydata.SetPoints(vtk_pts)
        polydata.SetLines(cells)

        mapper = self._vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = self._vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        actor.GetProperty().SetLineWidth(2)
        actor.GetProperty().SetOpacity(0.8)
        self.renderer.AddActor(actor)
        self._poly_line_actor = actor

    def _clear_polygon(self):
        for a in self._poly_markers:
            self.renderer.RemoveActor(a)
        self._poly_markers.clear()
        if self._poly_line_actor:
            self.renderer.RemoveActor(self._poly_line_actor)
            self._poly_line_actor = None

    # ─── 巡检选点模式 ─────────────────────────────────────────
    def enter_inspect_mode(self):
        if self.points_data is None or len(self.points_data) == 0:
            print("[Inspect] No point cloud loaded")
            return
        self.inspect_mode = True
        self._poly_click_start = None
        self._set_view("top")
        print("[Inspect] 左键点击添加巡检点，右键确认，Esc取消")

    def exit_inspect_mode(self, clear_markers=True):
        self.inspect_mode = False
        self._poly_click_start = None
        if clear_markers:
            self._clear_inspect_points()
        print("[Inspect] Exited inspect mode.")

    def _add_inspect_point(self, pos):
        self._inspect_points.append(pos)
        sphere = self._vtkSphereSource()
        sphere.SetCenter(pos.tolist())
        sphere.SetRadius(0.2)
        sphere.Update()
        m = self._vtkPolyDataMapper()
        m.SetInputConnection(sphere.GetOutputPort())
        a = self._vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(1.0, 0.8, 0.0)  # 黄色
        self.renderer.AddActor(a)
        self._inspect_markers.append(a)
        self.vtk_widget.GetRenderWindow().Render()

    def _clear_inspect_points(self):
        for a in self._inspect_markers:
            self.renderer.RemoveActor(a)
        self._inspect_markers.clear()
        self._inspect_points.clear()

    # ─── 直线起终点选点 ───
    def enter_line_mode(self):
        if self.points_data is None or len(self.points_data) == 0:
            print("[Line] No point cloud loaded")
            return
        self.line_mode = True
        self._poly_click_start = None
        self._set_view("top")
        print("[Line] 左键点击选起点，再点选终点，右键确认，Esc取消")

    def exit_line_mode(self, clear_markers=True):
        self.line_mode = False
        self._poly_click_start = None
        if clear_markers:
            self._clear_line_points()
        print("[Line] Exited line mode.")

    def _add_line_point(self, pos):
        if len(self._line_points) >= 2:
            return
        self._line_points.append(pos)
        sphere = self._vtkSphereSource()
        sphere.SetCenter(pos.tolist())
        sphere.SetRadius(0.3)
        sphere.Update()
        m = self._vtkPolyDataMapper()
        m.SetInputConnection(sphere.GetOutputPort())
        a = self._vtkActor()
        a.SetMapper(m)
        if len(self._line_points) == 1:
            a.GetProperty().SetColor(0.2, 0.8, 0.2)  # 绿色=起点
        else:
            a.GetProperty().SetColor(0.9, 0.2, 0.2)  # 红色=终点
        self.renderer.AddActor(a)
        self._line_markers.append(a)
        self.vtk_widget.GetRenderWindow().Render()
        if len(self._line_points) == 2:
            print("[Line] 已选起点和终点，右键确认生成航线")

    def _clear_line_points(self):
        for a in self._line_markers:
            self.renderer.RemoveActor(a)
        self._line_markers.clear()
        self._line_points.clear()

    def _update_view(self):
        if not self._vtk_available:
            return
        # FPV模式下由FPV系统控制相机，跳过默认视角设置
        if getattr(self, 'fpv_mode', False):
            self.vtk_widget.GetRenderWindow().Render()
            return
        cam = self.renderer.GetActiveCamera()
        # 用场景中心而非原点
        bounds = self.renderer.ComputeVisiblePropBounds()
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        cam.Elevation(30)
        cam.Azimuth(-45)
        self.vtk_widget.GetRenderWindow().Render()

    def _set_view(self, name):
        if not self._vtk_available:
            return
        cam = self.renderer.GetActiveCamera()
        # 获取场景中心
        bounds = self.renderer.ComputeVisiblePropBounds()
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2
        # 计算场景范围
        dx = max(bounds[1] - bounds[0], 1)
        dy = max(bounds[3] - bounds[2], 1)
        dz = max(bounds[5] - bounds[4], 1)
        dist = max(dx, dy, dz) * 1.5

        cam.SetFocalPoint(cx, cy, cz)

        if name == "top":
            cam.SetPosition(cx, cy, cz + dist)
            cam.SetViewUp(0, 1, 0)
        elif name == "bottom":
            cam.SetPosition(cx, cy, cz - dist)
            cam.SetViewUp(0, -1, 0)
        elif name == "front":
            cam.SetPosition(cx, cy - dist, cz)
            cam.SetViewUp(0, 0, 1)
        elif name == "side":
            cam.SetPosition(cx + dist, cy, cz)
            cam.SetViewUp(0, 0, 1)
        elif name == "persp":
            cam.SetPosition(cx + dist * 0.6, cy - dist * 0.8, cz + dist * 0.6)
            cam.SetViewUp(0, 0, 1)

        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _on_global_key_press(self, obj, event):
        key = self.interactor.GetKeyCode()
        key_view_map = {
            '1': 'top', '2': 'front', '3': 'side', '4': 'persp', '5': 'bottom',
        }
        if key in key_view_map:
            self._set_view(key_view_map[key])
        elif key == '\x1b':
            if self.polygon_mode:
                self.exit_polygon_mode()
            elif self.place_mode:
                self.exit_place_mode()
            elif self.inspect_mode:
                self.exit_inspect_mode()
            elif self.line_mode:
                self.exit_line_mode()
            self.vtk_widget.GetRenderWindow().Render()

    def _add_scene_axes(self):
        """添加坐标轴（带箭头）和网格到场景"""
        arrow_len = 4.0
        axes_config = [
            # (color, rotate_func) - 箭头默认沿X轴
            ((1, 0, 0), None),                    # X轴：无需旋转
            ((0, 1, 0), (90, 0, 0, 1)),            # Y轴：绕Z旋转90°
            ((0, 0, 1), (-90, 0, 1, 0)),           # Z轴：绕Y旋转-90°
        ]
        for color, rot in axes_config:
            arrow = self._vtkArrowSource()
            arrow.SetShaftRadius(0.03)
            arrow.SetTipRadius(0.08)
            arrow.SetTipLength(0.25)
            arrow.Update()

            transform = self._vtkTransform()
            if rot:
                angle, ax_x, ax_y, ax_z = rot
                transform.RotateWXYZ(angle, ax_x, ax_y, ax_z)
            transform.Scale(arrow_len, arrow_len, arrow_len)
            filt = self._vtkTransformPolyDataFilter()
            filt.SetInputConnection(arrow.GetOutputPort())
            filt.SetTransform(transform)
            filt.Update()
            m = self._vtkPolyDataMapper()
            m.SetInputConnection(filt.GetOutputPort())

            a = self._vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetColor(color)
            self.renderer.AddActor(a)
            self._actors.append(a)

        grid_pts = self._vtkPoints()
        grid_cells = self._vtkCellArray()
        idx = 0
        for i in range(-20, 21, 2):
            grid_pts.InsertNextPoint(i, -20, 0)
            grid_pts.InsertNextPoint(i, 20, 0)
            line = self._vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)
            line.GetPointIds().SetId(0, idx)
            line.GetPointIds().SetId(1, idx + 1)
            grid_cells.InsertNextCell(line)
            idx += 2
            grid_pts.InsertNextPoint(-20, i, 0)
            grid_pts.InsertNextPoint(20, i, 0)
            line = self._vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)
            line.GetPointIds().SetId(0, idx)
            line.GetPointIds().SetId(1, idx + 1)
            grid_cells.InsertNextCell(line)
            idx += 2

        grid_poly = self._vtkPolyData()
        grid_poly.SetPoints(grid_pts)
        grid_poly.SetLines(grid_cells)
        m = self._vtkPolyDataMapper()
        m.SetInputData(grid_poly)
        a = self._vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(0.5, 0.5, 0.5)
        a.GetProperty().SetOpacity(0.6)
        self.renderer.AddActor(a)
        self._actors.append(a)

    def setup_scene(self):
        """初始化场景（坐标轴 + 网格）"""
        if not self._vtk_available:
            return

        self._add_scene_axes()

        self._update_view()
        self.interactor.Initialize()

        # 用空 style 替换默认的 TrackballCamera，避免它抢先处理鼠标事件
        self.interactor.SetInteractorStyle(self._vtkInteractorStyleUser())

        style = FoxgloveInteractorStyle()
        style.set_viewer(self)
        style.set_interactor(self.interactor)

        self.interactor.AddObserver("KeyPressEvent", self._on_global_key_press)
