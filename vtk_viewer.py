"""VTK 3D 点云/航线可视化组件"""

import math
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

        pos = rwi.GetEventPosition()

        # 选点模式下（含FPV），仍需设置_click_start
        if v.polygon_mode or v.place_mode or v.inspect_mode or v.line_mode:
            v._poly_click_start = pos
            return

        # FPV模式下左键不做普通操作
        if v.fpv_mode:
            return

        ctrl = rwi.GetControlKey()
        if ctrl:
            wp_idx = v._find_nearest_waypoint(pos[0], pos[1])
            if wp_idx >= 0:
                v._start_wp_edit(wp_idx, pos[0], pos[1])

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
                v._picked_normal = None
                p = v._pick_3d(pos[0], pos[1])
                if p is not None:
                    normal = v._picked_normal.copy() if v._picked_normal is not None else np.array([0.0, 0.0, 1.0])
                    v._picked_normal = None
                    v._add_polygon_point(p, normal)
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
                    # 吸附到最近点云点（仅点云模式，STL/OBJ射线已精确）
                    if not v._get_all_mesh_actors() and v.points_data is not None and len(v.points_data) > 0:
                        if v._cloud_tree is None:
                            from scipy.spatial import cKDTree
                            v._cloud_tree = cKDTree(v.points_data)
                        _, idx = v._cloud_tree.query(p)
                        p = v.points_data[idx]
                    v._add_inspect_point(p)
            self._mode = None
            self._prev_pos = None
            return

        if v and v.line_mode and v._poly_click_start is not None:
            pos = rwi.GetEventPosition()
            start = v._poly_click_start
            v._poly_click_start = None
            if abs(pos[0] - start[0]) < 5 and abs(pos[1] - start[1]) < 5:
                # FPV模式下先更新相机参数并渲染，确保DisplayToWorld计算正确
                if v.fpv_mode:
                    if getattr(v, '_fpv_first_person', True):
                        v._update_fpv_camera(render=True)
                    else:
                        v._update_third_person_camera(render=True)
                p = v._pick_3d(pos[0], pos[1])
                if p is not None:
                    # 吸附到最近点云点（仅点云模式，STL/OBJ射线已精确）
                    if not v._get_all_mesh_actors() and v.points_data is not None and len(v.points_data) > 0:
                        if v._cloud_tree is None:
                            from scipy.spatial import cKDTree
                            v._cloud_tree = cKDTree(v.points_data)
                        _, idx = v._cloud_tree.query(p)
                        p = v.points_data[idx]
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
        # 选点模式下（含FPV），右键结束绘制
        if v and v.polygon_mode:
            if len(v._poly_points) >= 3:
                pts = [(p.tolist(), n.tolist()) for p, n in zip(v._poly_points, v._poly_normals)]
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
                pts = [(p.tolist(), n.tolist()) for p, n in v._inspect_points]
                v.inspect_points_confirmed.emit(pts)
                v.exit_inspect_mode(clear_markers=False)
            else:
                v.exit_inspect_mode()
            return
        if v and v.line_mode:
            if len(v._line_points) == 2:
                pts = [(p.tolist(), n.tolist()) for p, n in v._line_points]
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
            ren.SetDisplayPoint(fd[0] - dx, fd[1] - dy, fd[2])
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
    line_point_picked = pyqtSignal(int, object)  # 直线选点实时通知 (index, point)
    anim_finished = pyqtSignal()  # 航线动画播放结束

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
                vtkFollower, vtkVectorText, vtkBillboardTextActor3D,
                vtkTextActor
            )
            from vtkmodules.vtkFiltersSources import vtkSphereSource, vtkCubeSource, vtkLineSource, vtkArrowSource
            from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
            from vtkmodules.vtkCommonTransforms import vtkTransform
            from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyLine, vtkPolygon
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
                    vtkPolyLine, vtkPolygon, vtkFollower, vtkVectorText,
                    vtkBillboardTextActor3D, vtkTextActor,
                    vtkTransformPolyDataFilter, vtkTransform,
                    vtkGlyph3D, vtkInteractorStyleUser,
                    vtkFloatArray, vtkTexture, vtkPNGReader, vtkOBJReader, vtkOBJImporter,
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
        self._vtkPolygon = vtkPolygon
        self._vtkFloatArray = vtkFloatArray
        self._vtkTexture = vtkTexture
        self._vtkPNGReader = vtkPNGReader
        self._vtkBillboardTextActor3D = vtkBillboardTextActor3D
        self._vtkTransformPolyDataFilter = vtkTransformPolyDataFilter
        self._vtkTransform = vtkTransform
        self._vtkGlyph3D = vtkGlyph3D
        self._vtkInteractorStyleUser = vtkInteractorStyleUser
        self._vtkTextActor = vtkTextActor

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # ─── 视角切换按钮（浮动在 VTK 上方）───
        from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QButtonGroup
        from PyQt5.QtCore import Qt as QtCore
        self._view_frame = QFrame(self)
        self._view_frame.setStyleSheet("QFrame { background: transparent; border: none; }")
        self._view_frame.setFixedSize(250, 28)
        view_layout = QHBoxLayout(self._view_frame)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(1)

        views = [
            ("TOP", "top"),
            ("BOT", "bottom"),
            ("FRONT", "front"),
            ("SIDE", "side"),
            ("3D", "persp"),
        ]
        self._view_btns_list = []
        for i, (label, name) in enumerate(views):
            btn = QLabel(label)
            btn.setAlignment(QtCore.AlignCenter)
            btn.setFixedSize(48, 26)
            btn.setStyleSheet(
                "QLabel { color: #555; background: #2a2a2a; border: 1px solid #3a3a3a; "
                "font-size: 10px; font-family: Consolas, monospace; }"
            )
            btn.mousePressEvent = lambda e, n=name, idx=i: self._on_view_label_click(n, idx)
            view_layout.addWidget(btn)
            self._view_btns_list.append((btn, name))
        # 默认高亮3D
        self._view_btn_active = 4
        self._view_btns_list[4][0].setStyleSheet(
            "QLabel { color: #ffa500; background: #333; border: 1px solid #ffa500; "
            "font-size: 10px; font-family: Consolas, monospace; font-weight: bold; }"
        )
        self._view_frame.raise_()
        QTimer.singleShot(100, lambda: self._view_frame.move(self.width() - 260, 8))

        self.renderer.SetBackground(243/255, 243/255, 244/255)  # #F3F3F4
        self.renderer.TwoSidedLightingOn()  # 双面光照，防止背面全黑

        # 添加侧面辅助光，增强棱角立体感
        vtk = self._vtk
        light = vtk.vtkLight()
        light.SetPosition(1, -1, 1)
        light.SetFocalPoint(0, 0, 0)
        light.SetIntensity(0.4)
        light.SetColor(1.0, 1.0, 0.95)
        self.renderer.AddLight(light)

        self._actors = []
        self.points_data = None
        self._cloud_tree = None
        self._cloud_actor = None

        # ─── STL 网格模型 ───
        self._stl_polydata = None
        self._stl_actor = None
        self._obj_actors = []  # OBJ导入器创建的多个actor
        self._stl_locator = None  # vtkCellLocator 用于法线查询
        self._stl_normals = None  # vtkPolyData 法线数据
        self._stl_points_np = None
        self._stl_normals_np = None
        self._stl_tree = None
        self._stl_cell_locator = None
        self._picked_normal = None  # 最近一次拾取点的法线

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
        self._axes_actors = []  # 坐标轴和网格actor
        self._waypoints_ref = None
        self._safe_distance = 2.0
        self._takeoff_z = 1.0
        self._takeoff_yaw = 0.0
        self._safe_point = (0.0, 0.0, 5.0)
        self.show_heading = True
        self.show_gimbal_dir = False

        # ─── 多边形选择模式 ───
        self.polygon_mode = False
        self._poly_points = []
        self._poly_normals = []
        self._poly_markers = []
        self._poly_ref_pos = None
        self._poly_ref_normal = None
        self._poly_line_actor = None
        self._poly_surface_actor = None
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
        self._fpv_speed = 0.5  # 默认飞行速度（米/帧，60fps时约30m/s）
        self._fpv_boost_speed = 2.0  # Shift加速时的速度
        self._fpv_look_speed = 0.3  # 鼠标灵敏度
        self._fpv_keys = set()  # 当前按下的键
        self._fpv_mouse_active = False  # 右键按下时鼠标控制视角
        self._fpv_prev_mouse = None
        self._fpv_drone_actor = None  # 无人机模型actor
        self._fpv_on_mark = None  # 打点回调函数
        self._fpv_fov = 80.0  # FPV相机视场角

        # ─── 航线动画播放 ───
        self._anim_playing = False
        self._anim_wp_idx = 0
        self._anim_waypoints = []
        self._anim_drone_actor = None
        self._anim_coverage_actor = None
        self._anim_cone_actor = None
        self._anim_timer = None
        self._anim_speed = 1.0
        self._camera_hfov = 80.0  # 水平FOV（度），由外部设置
        self._camera_vfov = 60.0  # 垂直FOV（度），由外部设置

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
            # 先清空按键状态，防止残留
            self._fpv_keys.clear()
            self._fpv_mouse_active = False
            # 移除FPV键盘和鼠标observer
            rwi = self.vtk_widget.GetRenderWindow().GetInteractor()
            if rwi is not None:
                for attr in ['_fpv_key_press_obs', '_fpv_key_release_obs',
                             '_fpv_right_press_obs', '_fpv_right_release_obs',
                             '_fpv_move_obs', '_fpv_left_press_obs']:
                    obs_id = getattr(self, attr, None)
                    if obs_id is not None:
                        try:
                            rwi.RemoveObserver(obs_id)
                        except Exception:
                            pass
                        setattr(self, attr, None)
            self._remove_drone_model()
            self._exit_fpv_camera()

        if self.vtk_widget.GetRenderWindow():
            self.vtk_widget.GetRenderWindow().Render()

    def _create_drone_model(self):
        """加载STL无人机模型"""
        if self._fpv_drone_actor is not None:
            return
        vtk = self._vtk
        import os, sys

        # 多路径查找STL文件
        search_dirs = [
            os.path.dirname(os.path.abspath(__file__)),  # vtk_viewer.py同级
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # 上级目录
            os.getcwd(),  # 当前工作目录
            os.path.dirname(os.path.abspath(sys.argv[0])),  # 可执行文件目录
        ]
        # PyInstaller打包后的临时目录
        if hasattr(sys, '_MEIPASS'):
            search_dirs.insert(0, sys._MEIPASS)

        stl_path = None
        for d in search_dirs:
            candidate = os.path.join(d, 'M4T_v2_simple.stl')
            if os.path.exists(candidate):
                stl_path = candidate
                break

        if stl_path is None:
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

        # 缩放无人机模型到真实尺寸（M4T对角轴距约0.35m，整体约0.5m）
        bounds = stl_polydata.GetBounds()  # (xmin,xmax,ymin,ymax,zmin,zmax)
        stl_size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])

        real_size = 0.5  # 无人机真实尺寸（米）
        scale = real_size / stl_size if stl_size > 0 else 1.0

        transform = vtk.vtkTransform()
        transform.Scale(scale, scale, scale)
        # 修正模型姿态：机身放平 + 机头朝前
        transform.RotateX(90)
        transform.RotateY(90)
        transform.RotateZ(180)
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

    # ─── 航线动画播放 ─────────────────────────────────────────
    def start_route_animation(self, waypoints, speed=1.0, camera_fov=None):
        """开始航线动画播放（第三人称跟随视角）"""
        if not waypoints:
            return
        self.stop_route_animation()
        self._anim_waypoints = waypoints
        self._anim_speed = speed
        self._anim_camera_fov = camera_fov if camera_fov else self._fpv_fov
        self._anim_wp_idx = 0
        self._anim_playing = True
        # 创建动画用无人机模型
        self._create_anim_drone()
        # 初始化相机到第一个航点
        wp0 = waypoints[0]
        pos = np.array(wp0['pos'])
        self._update_anim_camera(pos, wp0['quat'])
        # 启动动画定时器
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_tick)
        interval = max(20, int(100 / speed))
        self._anim_timer.start(interval)

    def stop_route_animation(self):
        """停止动画"""
        self._anim_playing = False
        if self._anim_timer:
            self._anim_timer.stop()
            self._anim_timer = None
        self._cleanup_anim()
        # 恢复默认视角
        self.renderer.ResetCamera()
        if self.vtk_widget.GetRenderWindow():
            self.vtk_widget.GetRenderWindow().Render()
        self.anim_finished.emit()

    def _create_anim_drone(self):
        """创建动画用无人机模型（加载STL无人机模型）"""
        if self._anim_drone_actor is not None:
            return
        vtk = self._vtk
        import os, sys

        # 多路径查找STL文件
        search_dirs = [
            os.path.dirname(os.path.abspath(__file__)),
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            os.getcwd(),
            os.path.dirname(os.path.abspath(sys.argv[0])),
        ]
        if hasattr(sys, '_MEIPASS'):
            search_dirs.insert(0, sys._MEIPASS)

        stl_path = None
        for d in search_dirs:
            candidate = os.path.join(d, 'M4T_v2_simple.stl')
            if os.path.exists(candidate):
                stl_path = candidate
                break

        if stl_path is None:
            # 回退到简单线框
            self._create_anim_drone_fallback()
            return

        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()
        stl_polydata = reader.GetOutput()
        if stl_polydata.GetNumberOfPoints() == 0:
            self._create_anim_drone_fallback()
            return

        # 缩放到真实尺寸
        bounds = stl_polydata.GetBounds()
        stl_size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        real_size = 0.6
        scale = real_size / stl_size if stl_size > 0 else 1.0

        transform = vtk.vtkTransform()
        transform.Scale(scale, scale, scale)
        transform.RotateX(90)
        transform.RotateY(90)
        transform.RotateZ(180)
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
        actor.GetProperty().SetColor(0.3, 0.3, 0.3)
        actor.GetProperty().SetLighting(True)
        # 无人机始终渲染在最前面（不被桥梁遮挡）
        actor.GetProperty().SetRenderLinesAsTubes(True)
        self.renderer.AddActor(actor)
        self._anim_drone_actor = actor

    def _create_anim_drone_fallback(self):
        """STL加载失败时的简单线框模型"""
        vtk = self._vtk
        arm_len = 1.0
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(-arm_len, 0, 0)
        pts.InsertNextPoint(arm_len, 0, 0)
        pts.InsertNextPoint(0, -arm_len, 0)
        pts.InsertNextPoint(0, arm_len, 0)
        pts.InsertNextPoint(0, 0, 0)
        pts.InsertNextPoint(0, 0, arm_len * 0.5)
        lines = vtk.vtkCellArray()
        for pair in [(0,1), (2,3), (4,5)]:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pair[0])
            line.GetPointIds().SetId(1, pair[1])
            lines.InsertNextCell(line)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.5, 0.0)
        actor.GetProperty().SetLineWidth(5)
        self.renderer.AddActor(actor)
        self._anim_drone_actor = actor

    def _update_anim_drone(self, pos, quat):
        """更新动画无人机位置和朝向"""
        if self._anim_drone_actor is None:
            return
        from quaternion_utils import quaternion_forward
        forward = quaternion_forward(quat)
        yaw = np.degrees(np.arctan2(forward[1], forward[0]))
        vtk = self._vtk
        t = vtk.vtkTransform()
        t.Translate(pos.tolist())
        t.RotateZ(yaw)
        t.RotateX(180)  # 修正机身朝向：底朝天→正常
        self._anim_drone_actor.SetUserTransform(t)

    def _update_anim_camera(self, pos, quat):
        """第三人称相机跟随无人机"""
        from quaternion_utils import quaternion_forward
        cam = self.renderer.GetActiveCamera()
        forward = quaternion_forward(quat)
        back = -forward
        back_len = np.linalg.norm(back)
        if back_len < 1e-10:
            back = np.array([-1.0, 0.0, 0.0])
        else:
            back = back / back_len
        cam_pos = pos + back * 15 + np.array([0, 0, 8])
        cam.SetPosition(cam_pos.tolist())
        cam.SetFocalPoint(pos.tolist())
        cam.SetViewUp(0, 0, 1)
        cam.SetViewAngle(getattr(self, '_anim_camera_fov', self._fpv_fov))
        cam.SetClippingRange(0.1, 5000.0)

    def _update_gimbal_coverage(self, wp):
        """计算云台FOV覆盖的STL三角面，创建高亮actor"""
        # 清除旧高亮
        if self._anim_coverage_actor is not None:
            self.renderer.RemoveActor(self._anim_coverage_actor)
            self._anim_coverage_actor = None
        if self._anim_cone_actor is not None:
            self.renderer.RemoveActor(self._anim_cone_actor)
            self._anim_cone_actor = None

        if self._stl_polydata is None:
            return
        pos = np.array(wp['pos'])
        target = np.array(wp.get('target_pos', pos))
        # 使用航线动画FOV（实际相机FOV），而非FPV视角FOV
        fov = getattr(self, '_anim_camera_fov', self._fpv_fov)
        half_fov_rad = np.radians(fov / 2.0)

        # 云台方向
        gimbal_dir = target - pos
        gim_len = np.linalg.norm(gimbal_dir)
        if gim_len < 1e-10:
            return
        gimbal_dir = gimbal_dir / gim_len

        # 遍历STL所有cell（三角面），筛选在FOV锥体内的面
        polydata = self._stl_polydata
        n_cells = polydata.GetNumberOfCells()
        visible_ids = []

        for cid in range(n_cells):
            cell = polydata.GetCell(cid)
            # 计算三角面中心
            pts = cell.GetPoints()
            center = np.array([0.0, 0.0, 0.0])
            for j in range(3):
                p = pts.GetPoint(j)
                center += np.array(p)
            center /= 3.0

            # 从无人机到面中心的方向
            to_center = center - pos
            dist = np.linalg.norm(to_center)
            if dist < 1e-10:
                continue
            to_center = to_center / dist

            # 检查是否在FOV锥体内
            cos_angle = np.dot(to_center, gimbal_dir)
            if cos_angle <= 0:
                continue  # 在身后
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            if angle <= half_fov_rad:
                visible_ids.append(cid)

        if not visible_ids:
            return

        # 创建高亮polydata（只包含可见面）
        vtk = self._vtk
        new_pts = vtk.vtkPoints()
        new_cells = vtk.vtkCellArray()
        pt_idx = 0
        for cid in visible_ids:
            cell = polydata.GetCell(cid)
            cell_pts = cell.GetPoints()
            ids = []
            for j in range(3):
                p = cell_pts.GetPoint(j)
                new_pts.InsertNextPoint(p)
                ids.append(pt_idx)
                pt_idx += 1
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, ids[0])
            tri.GetPointIds().SetId(1, ids[1])
            tri.GetPointIds().SetId(2, ids[2])
            new_cells.InsertNextCell(tri)

        highlight_poly = vtk.vtkPolyData()
        highlight_poly.SetPoints(new_pts)
        highlight_poly.SetPolys(new_cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(highlight_poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.2, 0.0)  # 亮橙红色
        actor.GetProperty().SetOpacity(0.7)
        actor.GetProperty().SetAmbient(0.4)
        actor.GetProperty().SetDiffuse(0.6)
        actor.GetProperty().SetLighting(True)
        actor.GetProperty().BackfaceCullingOff()
        self.renderer.AddActor(actor)
        self._anim_coverage_actor = actor

        # 添加FOV锥体线框，增强视觉效果
        self._add_fov_cone(pos, gimbal_dir, fov, gim_len)

    def _add_fov_cone(self, pos, direction, fov_deg, max_dist):
        """在场景中绘制FOV锥体线框"""
        vtk = self._vtk
        half_fov = np.radians(fov_deg / 2.0)
        cone_len = min(max_dist * 0.8, 50.0)  # 锥体长度，限制最大50米
        cone_radius = cone_len * np.tan(half_fov)

        # 构建锥体侧面的线框（8条母线 + 底面圆）
        lines = vtk.vtkCellArray()
        points = vtk.vtkPoints()
        pid = 0

        # 锥体顶点（无人机位置）
        points.InsertNextPoint(pos.tolist())
        pid += 1

        # 构建坐标系：direction为Z轴，找两个正交向量
        z_axis = direction / np.linalg.norm(direction)
        # 找一个不平行的向量
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z_axis, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, up)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # 底面圆心
        circle_center = pos + direction * cone_len

        # 底面圆上的点（16个）
        n_circle = 16
        for i in range(n_circle):
            angle = 2.0 * np.pi * i / n_circle
            pt = circle_center + (x_axis * np.cos(angle) + y_axis * np.sin(angle)) * cone_radius
            points.InsertNextPoint(pt.tolist())
            pid += 1

        # 8条母线（从顶点到底面圆）
        for i in range(8):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 0)  # 顶点
            line.GetPointIds().SetId(1, 1 + i * 2)  # 底面圆上的点
            lines.InsertNextCell(line)

        # 底面圆连线
        for i in range(n_circle):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 1 + i)
            line.GetPointIds().SetId(1, 1 + (i + 1) % n_circle)
            lines.InsertNextCell(line)

        cone_poly = vtk.vtkPolyData()
        cone_poly.SetPoints(points)
        cone_poly.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(cone_poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.8, 0.0)  # 金黄色线框
        actor.GetProperty().SetOpacity(0.6)
        actor.GetProperty().SetLineWidth(1.5)
        self.renderer.AddActor(actor)
        self._anim_cone_actor = actor

    def _anim_tick(self):
        """动画每帧更新"""
        if not self._anim_playing or self._anim_wp_idx >= len(self._anim_waypoints):
            self.stop_route_animation()
            return
        if self._anim_wp_idx == 0:
            print(f"[Anim] Drone actor: {self._anim_drone_actor is not None}")
            print(f"[Anim] Bridge visible: stl={self._stl_actor.GetVisibility() if self._stl_actor else 'N/A'}")
            wp0 = self._anim_waypoints[0]
            pos0 = np.array(wp0['pos'])
            cam = self.renderer.GetActiveCamera()
            cp = cam.GetPosition()
            print(f"[Anim] wp0 pos={pos0}, cam_pos=({cp[0]:.1f},{cp[1]:.1f},{cp[2]:.1f})")
            print(f"[Anim] drone bounds: {self._anim_drone_actor.GetBounds()}")
        wp = self._anim_waypoints[self._anim_wp_idx]
        pos = np.array(wp['pos'])
        quat = wp['quat']
        # 移动动画无人机
        self._update_anim_drone(pos, quat)
        # 更新跟随相机
        self._update_anim_camera(pos, quat)
        # 高亮云台覆盖
        self._update_gimbal_coverage(wp)
        # 投影相机画面到表面
        self._update_anim_projection(wp)
        if self.vtk_widget.GetRenderWindow():
            self.vtk_widget.GetRenderWindow().Render()
        self._anim_wp_idx += 1

    def _update_anim_projection(self, wp):
        """在当前航点位置投影相机画面到STL/点云表面"""
        # 清除旧投影
        if hasattr(self, '_anim_proj_actors'):
            for a in self._anim_proj_actors:
                self.renderer.RemoveActor(a)
        self._anim_proj_actors = []
        # 检查投影开关
        if not getattr(self, '_anim_proj_enabled', True):
            return
        if 'target_pos' not in wp:
            return
        pos = np.array(wp['pos'], dtype=float)
        tgt = np.array(wp['target_pos'], dtype=float)
        los = tgt - pos
        dist = np.linalg.norm(los)
        if dist < 0.5:
            return
        hfov_rad = math.radians(self._camera_hfov)
        vfov_rad = math.radians(self._camera_vfov)
        half_w = dist * math.tan(hfov_rad / 2.0)
        half_h = dist * math.tan(vfov_rad / 2.0)
        # 航线方向
        quat = wp.get('quat')
        if quat is not None:
            from quaternion_utils import quaternion_forward
            heading = quaternion_forward(quat)
        else:
            heading = np.array([1.0, 0.0, 0.0])
        fwd_h = np.array([heading[0], heading[1], 0.0])
        fwd_len = np.linalg.norm(fwd_h)
        if fwd_len < 1e-6:
            fwd_h = np.array([1.0, 0.0, 0.0])
        else:
            fwd_h = fwd_h / fwd_len
        right_h = np.cross(np.array([0.0, 0.0, 1.0]), fwd_h)
        rn = np.linalg.norm(right_h)
        if rn < 1e-6:
            right_h = np.array([0.0, 1.0, 0.0])
        else:
            right_h = right_h / rn
        # 4个角点
        corners_3d = [
            tgt + fwd_h * half_h + right_h * half_w,
            tgt + fwd_h * half_h - right_h * half_w,
            tgt - fwd_h * half_h - right_h * half_w,
            tgt - fwd_h * half_h + right_h * half_w,
        ]
        # 投影每个角到表面
        proj_corners = []
        for corner in corners_3d:
            pt = self._project_to_stl(pos, corner)
            if pt is None:
                pt = self._project_to_cloud(pos, corner)
            if pt is None:
                pt = corner
            proj_corners.append(pt)
        # 纹理贴图多边形
        vtk = self._vtk
        vtk_pts = self._vtkPoints()
        tc = self._vtkFloatArray()
        tc.SetNumberOfComponents(2)
        tc.SetName("TextureCoordinates")
        tex_coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        for k, c in enumerate(proj_corners):
            vtk_pts.InsertNextPoint(c.tolist())
            tc.InsertNextTuple2(tex_coords[k][0], tex_coords[k][1])
        polygon = self._vtkPolygon()
        pid = polygon.GetPointIds()
        pid.SetNumberOfIds(4)
        for k in range(4):
            pid.SetId(k, k)
        cells = self._vtkCellArray()
        cells.InsertNextCell(polygon)
        polydata = self._vtkPolyData()
        polydata.SetPoints(vtk_pts)
        polydata.SetPolys(cells)
        polydata.GetPointData().SetTCoords(tc)
        tex = self._get_coverage_texture()
        mapper = self._vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = self._vtkActor()
        actor.SetMapper(mapper)
        actor.SetTexture(tex)
        actor.GetProperty().SetOpacity(0.6)
        actor.GetProperty().SetLighting(False)
        actor.SetPosition(0, 0, 0.01)
        self.renderer.AddActor(actor)
        self._anim_proj_actors.append(actor)
        # 边框
        edge_pts = self._vtkPoints()
        for c in proj_corners:
            edge_pts.InsertNextPoint(c.tolist())
        edge_line = self._vtkPolyLine()
        edge_line.GetPointIds().SetNumberOfIds(5)
        for k in range(4):
            edge_line.GetPointIds().SetId(k, k)
        edge_line.GetPointIds().SetId(4, 0)
        edge_cells = self._vtkCellArray()
        edge_cells.InsertNextCell(edge_line)
        edge_poly = self._vtkPolyData()
        edge_poly.SetPoints(edge_pts)
        edge_poly.SetLines(edge_cells)
        edge_mapper = self._vtkPolyDataMapper()
        edge_mapper.SetInputData(edge_poly)
        edge_actor = self._vtkActor()
        edge_actor.SetMapper(edge_mapper)
        edge_actor.GetProperty().SetColor(0.0, 0.8, 1.0)
        edge_actor.GetProperty().SetLineWidth(2)
        edge_actor.GetProperty().SetLighting(False)
        edge_actor.SetPosition(0, 0, 0.01)
        self.renderer.AddActor(edge_actor)
        self._anim_proj_actors.append(edge_actor)

    def _cleanup_anim(self):
        """清理动画资源"""
        if self._anim_drone_actor is not None:
            self.renderer.RemoveActor(self._anim_drone_actor)
            self._anim_drone_actor = None
        if self._anim_coverage_actor is not None:
            self.renderer.RemoveActor(self._anim_coverage_actor)
            self._anim_coverage_actor = None
        if self._anim_cone_actor is not None:
            self.renderer.RemoveActor(self._anim_cone_actor)
            self._anim_cone_actor = None
        if hasattr(self, '_anim_proj_actors'):
            for a in self._anim_proj_actors:
                self.renderer.RemoveActor(a)
            self._anim_proj_actors = []
        if self.vtk_widget.GetRenderWindow():
            self.vtk_widget.GetRenderWindow().Render()

    # ─── STL 网格模型 ─────────────────────────────────────────
    def add_stl_mesh(self, path, color=(0.7, 0.7, 0.7), opacity=1.0):
        """加载并渲染STL网格模型（带光照）
        path: STL文件路径
        color: (R,G,B) 0~1
        opacity: 透明度 0~1
        """
        if not self._vtk_available:
            return False
        vtk = self._vtk
        import os
        if not os.path.exists(path):
            print(f"[STL] File not found: {path}")
            return False

        # 清除旧STL
        self.clear_stl_mesh()

        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()

        polydata = reader.GetOutput()
        if polydata.GetNumberOfPoints() == 0:
            print(f"[STL] Empty mesh: {path}")
            return False

        # 先清除STL文件中存储的旧法线，避免干扰
        polydata.GetPointData().SetNormals(None)
        polydata.GetCellData().SetNormals(None)

        # 计算法线，统一朝外
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(polydata)
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOff()
        normals_filter.ConsistencyOn()
        normals_filter.SplittingOff()
        normals_filter.AutoOrientNormalsOn()
        normals_filter.Update()
        polydata_with_normals = normals_filter.GetOutput()

        # 调试：检查法线是否有效
        pn = polydata_with_normals.GetPointData().GetNormals()
        if pn:
            n_tuples = pn.GetNumberOfTuples()
            sample = [pn.GetTuple(i) for i in range(min(3, n_tuples))]
            print(f"[STL] Point normals: {n_tuples} tuples, samples={sample}")
        else:
            print("[STL] WARNING: No point normals computed!")
            polydata_with_normals = polydata

        # 存储数据
        self._stl_polydata = polydata_with_normals
        point_normals = polydata_with_normals.GetPointData().GetNormals()

        # 构建 KDTree 用于快速最近点查询
        stl_points = polydata_with_normals.GetPoints()
        n_pts = stl_points.GetNumberOfPoints()
        stl_np = np.array([stl_points.GetPoint(i) for i in range(n_pts)])
        self._stl_points_np = stl_np

        if point_normals is not None:
            normals_np = np.array([point_normals.GetTuple(i) for i in range(n_pts)])
            # 归一化
            norms = np.linalg.norm(normals_np, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            self._stl_normals_np = normals_np / norms
        else:
            self._stl_normals_np = None

        from scipy.spatial import cKDTree
        self._stl_tree = cKDTree(stl_np)

        # 创建CellLocator用于表面碰撞检测
        cell_loc = vtk.vtkCellLocator()
        cell_loc.SetDataSet(polydata_with_normals)
        cell_loc.BuildLocator()
        self._stl_cell_locator = cell_loc

        # 尝试从同名PLY文件加载部件颜色
        ply_path = os.path.splitext(path)[0] + '.ply'
        has_ply_color = False
        if os.path.exists(ply_path):
            try:
                from pcd_parser import parse_ply
                ply_points, ply_colors = parse_ply(ply_path)
                if ply_colors is not None and len(ply_points) > 0:
                    from scipy.spatial import cKDTree as _CKDTree
                    ply_tree = _CKDTree(ply_points)
                    _, indices = ply_tree.query(stl_np)
                    stl_colors = ply_colors[indices]  # (N, 3) uint8
                    # 设置顶点颜色
                    vtk_colors = vtk.vtkUnsignedCharArray()
                    vtk_colors.SetNumberOfComponents(3)
                    vtk_colors.SetName("Colors")
                    for c in stl_colors:
                        vtk_colors.InsertNextTuple3(int(c[0]), int(c[1]), int(c[2]))
                    polydata_with_normals.GetPointData().SetScalars(vtk_colors)
                    has_ply_color = True
                    print(f"[STL] Loaded colors from PLY: {ply_path}")
            except Exception as e:
                print(f"[STL] Failed to load PLY colors: {e}")

        # 渲染
        mapper = self._vtkPolyDataMapper()
        mapper.SetInputData(polydata_with_normals)
        self._stl_colors = has_ply_color  # 保存PLY颜色状态
        if has_ply_color:
            mapper.ScalarVisibilityOn()
            mapper.SetColorModeToDirectScalars()
        else:
            mapper.ScalarVisibilityOff()
        actor = self._vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(color)
        prop.SetOpacity(opacity)
        prop.SetAmbient(0.4)
        prop.SetDiffuse(0.8)
        prop.SetSpecular(0.2)
        prop.SetSpecularPower(20)
        prop.SetInterpolationToPhong()
        prop.SetLighting(True)
        prop.BackfaceCullingOff()  # 不剔除背面，底面也能正常显示
        print(f"[STL] Color={color}, Opacity={opacity}")

        self.renderer.AddActor(actor)
        self._actors.append(actor)
        self._stl_actor = actor

        # 显示点云边框（可选）
        actor.GetProperty().SetEdgeVisibility(False)

        n_triangles = polydata_with_normals.GetNumberOfCells()
        print(f"[STL] Loaded: {path} ({n_triangles} triangles)")
        return True

    def add_obj_mesh(self, path, color=(0.7, 0.7, 0.7), opacity=1.0):
        """加载并渲染OBJ网格模型（支持MTL材质+纹理贴图）
        快速显示 → 延迟计算法线/KDTree/CellLocator/纹理
        """
        if not self._vtk_available:
            return False
        vtk = self._vtk
        import os
        if not os.path.exists(path):
            print(f"[OBJ] File not found: {path}")
            return False

        self.clear_stl_mesh()

        obj_dir = os.path.dirname(os.path.abspath(path))

        # ── 解析MTL材质（纯文本解析，极快） ──
        mtl_kd = None
        mtl_map_kd = None
        mtl_file = None
        try:
            with open(path, 'r', errors='ignore') as f:
                for line in f:
                    if line.strip().startswith('mtllib '):
                        mtl_file = line.strip().split(None, 1)[1].strip()
                        break
        except Exception:
            pass

        if mtl_file:
            mtl_path = os.path.join(obj_dir, mtl_file)
            if os.path.exists(mtl_path):
                print(f"[OBJ] MTL: {mtl_path}")
                try:
                    with open(mtl_path, 'r', errors='ignore') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 4 and parts[0] == 'Kd':
                                mtl_kd = (float(parts[1]), float(parts[2]), float(parts[3]))
                            elif len(parts) >= 2 and parts[0] == 'map_Kd':
                                mtl_map_kd = parts[1]
                except Exception as e:
                    print(f"[OBJ] MTL parse error: {e}")

        # ── 加载几何体（vtkOBJReader，较快） ──
        reader = vtk.vtkOBJReader()
        reader.SetFileName(path)
        reader.Update()

        polydata = reader.GetOutput()
        n_pts = polydata.GetNumberOfPoints()
        if n_pts == 0:
            print(f"[OBJ] Empty mesh: {path}")
            return False

        n_tri = polydata.GetNumberOfCells()
        print(f"[OBJ] Geometry: {n_pts} vertices, {n_tri} triangles")

        # ── 立即创建Actor显示（用材质颜色，不等纹理/法线） ──
        mapper = self._vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        actor = self._vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        if mtl_kd is not None:
            prop.SetColor(mtl_kd)
            self._stl_colors = True
            print(f"[OBJ] Material Kd: ({mtl_kd[0]:.2f}, {mtl_kd[1]:.2f}, {mtl_kd[2]:.2f})")
        else:
            prop.SetColor(color)
            self._stl_colors = False
        prop.SetOpacity(opacity)
        prop.SetAmbient(0.4)
        prop.SetDiffuse(0.8)
        prop.SetSpecular(0.2)
        prop.SetSpecularPower(20)
        prop.SetInterpolationToPhong()
        prop.SetLighting(True)
        prop.BackfaceCullingOff()  # 不剔除背面，内部墙也显示颜色

        self.renderer.AddActor(actor)
        self._actors.append(actor)
        self._stl_actor = actor
        self._obj_actors = [actor]

        # 先存raw polydata，后续延迟处理会替换
        self._stl_polydata = polydata

        # 刷新UI让模型先显示出来
        if hasattr(self, 'vtk_widget') and self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()

        # ── 延迟执行重操作（法线、KDTree、CellLocator、纹理） ──
        self._obj_deferred = {
            'path': path,
            'polydata': polydata,
            'mtl_kd': mtl_kd,
            'mtl_map_kd': mtl_map_kd,
            'obj_dir': obj_dir,
            'actor': actor,
            'opacity': opacity,
        }
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self._deferred_obj_finish)

        return True

    def _deferred_obj_finish(self):
        """OBJ延迟加载：法线 + KDTree + CellLocator + 纹理"""
        info = getattr(self, '_obj_deferred', None)
        if info is None:
            return
        self._obj_deferred = None

        vtk = self._vtk
        import os
        polydata = info['polydata']
        actor = info['actor']

        # 检查actor是否还在（可能已被clear_stl_mesh清除）
        if actor not in self._actors:
            return

        print("[OBJ] Deferred: computing normals...")

        # 法线计算
        polydata.GetPointData().SetNormals(None)
        polydata.GetCellData().SetNormals(None)
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(polydata)
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOff()
        normals_filter.ConsistencyOn()
        normals_filter.SplittingOff()
        normals_filter.AutoOrientNormalsOn()
        normals_filter.Update()
        polydata_with_normals = normals_filter.GetOutput()

        pn = polydata_with_normals.GetPointData().GetNormals()
        if pn:
            print(f"[OBJ] Normals: {pn.GetNumberOfTuples()} tuples")
        else:
            polydata_with_normals = polydata

        self._stl_polydata = polydata_with_normals

        # 更新mapper使用带法线的polydata
        mapper = actor.GetMapper()
        mapper.SetInputData(polydata_with_normals)

        # KDTree + CellLocator
        stl_points = polydata_with_normals.GetPoints()
        n_pts = stl_points.GetNumberOfPoints()
        stl_np = np.array([stl_points.GetPoint(i) for i in range(n_pts)])
        self._stl_points_np = stl_np

        if pn:
            normals_np = np.array([pn.GetTuple(i) for i in range(n_pts)])
            norms = np.linalg.norm(normals_np, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            self._stl_normals_np = normals_np / norms
        else:
            self._stl_normals_np = None

        print("[OBJ] Deferred: building KDTree...")
        from scipy.spatial import cKDTree
        self._stl_tree = cKDTree(stl_np)

        print("[OBJ] Deferred: building CellLocator...")
        cell_loc = vtk.vtkCellLocator()
        cell_loc.SetDataSet(polydata_with_normals)
        cell_loc.BuildLocator()
        self._stl_cell_locator = cell_loc

        # 纹理（最后加载，最慢）
        mtl_map_kd = info.get('mtl_map_kd')
        if mtl_map_kd:
            tex_path = os.path.join(info['obj_dir'], mtl_map_kd)
            if os.path.exists(tex_path):
                fsize = os.path.getsize(tex_path)
                print(f"[OBJ] Deferred: loading texture ({fsize / 1024 / 1024:.1f} MB)...")
                try:
                    png_reader = vtk.vtkPNGReader()
                    png_reader.SetFileName(tex_path)
                    png_reader.Update()
                    img = png_reader.GetOutput()
                    if img and img.GetNumberOfPoints() > 0:
                        tcoords = polydata_with_normals.GetPointData().GetTCoords()
                        if tcoords and tcoords.GetNumberOfTuples() > 0:
                            texture = vtk.vtkTexture()
                            texture.SetInputData(img)
                            texture.InterpolateOn()
                            texture.RepeatOff()
                            actor.SetTexture(texture)
                            self._stl_colors = True
                            print(f"[OBJ] Texture applied ({tcoords.GetNumberOfTuples()} UV coords)")
                        else:
                            print("[OBJ] No UV coords, texture skipped")
                    else:
                        print("[OBJ] Texture read failed")
                except Exception as e:
                    print(f"[OBJ] Texture error: {e}")

        if hasattr(self, 'vtk_widget') and self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()

        print("[OBJ] Deferred loading complete")

    def get_stl_normal(self, point):
        """获取STL表面法线：找最近三角面，返回朝向观察者的几何法线"""
        if self._stl_polydata is None or self._stl_tree is None:
            return None
        # 找最近顶点
        _, idx = self._stl_tree.query(point)
        # 收集所有包含该顶点的三角面法线
        normals = []
        n_cells = self._stl_polydata.GetNumberOfCells()
        for ci in range(n_cells):
            cell = self._stl_polydata.GetCell(ci)
            if cell is None:
                continue
            n_pts = cell.GetNumberOfPoints()
            found = False
            for j in range(n_pts):
                if cell.GetPointId(j) == idx:
                    found = True
                    break
            if not found:
                continue
            v0 = np.array(self._stl_polydata.GetPoint(cell.GetPointId(0)))
            v1 = np.array(self._stl_polydata.GetPoint(cell.GetPointId(1)))
            v2 = np.array(self._stl_polydata.GetPoint(cell.GetPointId(2)))
            n = np.cross(v1 - v0, v2 - v0)
            nrm = np.linalg.norm(n)
            if nrm < 1e-10:
                continue
            normals.append(n / nrm)
        if not normals:
            return None
        # 获取观察者位置，选朝向观察者的法线
        if self.fpv_mode:
            viewer_pos = np.array(self._fpv_pos, dtype=float)
        else:
            cam = self.renderer.GetActiveCamera()
            viewer_pos = np.array(cam.GetPosition(), dtype=float)
        to_viewer = viewer_pos - point
        best_normal = None
        best_dot = -float('inf')
        for n in normals:
            d = np.dot(n, to_viewer)
            if d > best_dot:
                best_dot = d
                best_normal = n
        return best_normal

    def get_stl_geometric_normal(self, point):
        """获取STL表面真实几何法线（不做viewer方向翻转）
        用于航线生成：需要区分顶面/底面
        """
        if self._stl_polydata is None or self._stl_tree is None:
            return None
        _, idx = self._stl_tree.query(point)
        normals = []
        n_cells = self._stl_polydata.GetNumberOfCells()
        for ci in range(n_cells):
            cell = self._stl_polydata.GetCell(ci)
            if cell is None:
                continue
            n_pts = cell.GetNumberOfPoints()
            found = False
            for j in range(n_pts):
                if cell.GetPointId(j) == idx:
                    found = True
                    break
            if not found:
                continue
            v0 = np.array(self._stl_polydata.GetPoint(cell.GetPointId(0)))
            v1 = np.array(self._stl_polydata.GetPoint(cell.GetPointId(1)))
            v2 = np.array(self._stl_polydata.GetPoint(cell.GetPointId(2)))
            n = np.cross(v1 - v0, v2 - v0)
            nrm = np.linalg.norm(n)
            if nrm < 1e-10:
                continue
            normals.append(n / nrm)
        if not normals:
            return None
        # 返回平均法线（不翻转）
        avg = np.mean(normals, axis=0)
        nrm = np.linalg.norm(avg)
        if nrm < 1e-10:
            return normals[0]
        return avg / nrm

    def _pick_stl_top_surface(self, screen_x, screen_y):
        """沿相机射线与STL求交，返回最近的交点（用户看到的第一个面）"""
        if self._stl_polydata is None:
            return None, None
        # 计算射线
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
        if ray_len < 1e-10:
            return None, None
        ray_dir /= ray_len

        # Möller–Trumbore：找最近的交点
        polydata = self._stl_polydata
        n_cells = polydata.GetNumberOfCells()
        best_t = float('inf')
        best_pos = None
        best_normal = None
        for ci in range(n_cells):
            cell = polydata.GetCell(ci)
            if cell is None or cell.GetNumberOfPoints() < 3:
                continue
            ids = [cell.GetPointId(j) for j in range(3)]
            v0 = np.array(polydata.GetPoint(ids[0]))
            v1 = np.array(polydata.GetPoint(ids[1]))
            v2 = np.array(polydata.GetPoint(ids[2]))
            e1 = v1 - v0
            e2 = v2 - v0
            h = np.cross(ray_dir, e2)
            a = np.dot(e1, h)
            if abs(a) < 1e-10:
                continue
            f = 1.0 / a
            s = near - v0
            u = f * np.dot(s, h)
            if u < 0.0 or u > 1.0:
                continue
            q = np.cross(s, e1)
            v_tri = f * np.dot(ray_dir, q)
            if v_tri < 0.0 or u + v_tri > 1.0:
                continue
            t = f * np.dot(e2, q)
            if 1e-6 < t < best_t:
                best_t = t
                best_pos = near + t * ray_dir
                tri_normal = np.cross(e1, e2)
                nrm = np.linalg.norm(tri_normal)
                if nrm > 1e-10:
                    tri_normal = tri_normal / nrm
                    # 确保法线朝向相机（与射线方向相反）
                    if np.dot(tri_normal, ray_dir) > 0:
                        tri_normal = -tri_normal
                    best_normal = tri_normal
        return best_pos, best_normal

    def _get_all_mesh_actors(self):
        """获取所有网格模型actor（STL + OBJ）"""
        actors = []
        if self._stl_actor is not None:
            actors.append(self._stl_actor)
        actors.extend(self._obj_actors)
        return actors

    def set_stl_opacity(self, opacity):
        """设置STL/OBJ模型透明度"""
        for a in self._get_all_mesh_actors():
            a.GetProperty().SetOpacity(opacity)
        if self._get_all_mesh_actors() and hasattr(self, 'vtk_widget') and self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()

    def clear_stl_mesh(self):
        """清除STL/OBJ网格模型"""
        if self._stl_actor is not None:
            self.renderer.RemoveActor(self._stl_actor)
            if self._stl_actor in self._actors:
                self._actors.remove(self._stl_actor)
            self._stl_actor = None
        # 清除OBJ导入器创建的多个actor
        for a in self._obj_actors:
            self.renderer.RemoveActor(a)
            if a in self._actors:
                self._actors.remove(a)
        self._obj_actors = []
        self._stl_polydata = None
        self._stl_locator = None
        self._stl_normals = None
        self._stl_colors = False
        self._stl_points_np = None
        self._stl_normals_np = None
        self._stl_tree = None
        self._stl_cell_locator = None

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
        if not self.fpv_mode:
            return
        key = obj.GetKeySym().lower()
        self._fpv_keys.discard(key)

    def _fpv_on_left_down(self, obj, event):
        """FPV左键按下 - 拾取点云点并记录航点"""
        if not self.fpv_mode:
            return
        # 选点模式下不触发FPV打点（由_on_left_up处理）
        if self.polygon_mode or self.place_mode or self.inspect_mode or self.line_mode:
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
        """FPV鼠标移动 - 只控制yaw（机头左右转向）"""
        if not self.fpv_mode or not self._fpv_mouse_active:
            return
        pos = obj.GetEventPosition()
        if self._fpv_prev_mouse is not None:
            dx = pos[0] - self._fpv_prev_mouse[0]
            self._fpv_yaw -= dx * self._fpv_look_speed
        self._fpv_prev_mouse = pos
        # 根据当前视角模式调用不同的更新方法
        if getattr(self, '_fpv_first_person', True):
            self._update_fpv_camera()
        else:
            self._update_third_person_camera()

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
        """从无人机位置沿方向射线，找到与STL或点云的最近交点"""
        origin = self._fpv_pos.copy()
        ray_dir = direction.copy()
        ray_len = np.linalg.norm(ray_dir)
        if ray_len < 1e-10:
            return None
        ray_dir = ray_dir / ray_len

        # 优先：STL 射线求交
        if self._stl_polydata is not None:
            polydata = self._stl_polydata
            n_cells = polydata.GetNumberOfCells()
            best_t = float('inf')
            best_pos = None
            for ci in range(n_cells):
                cell = polydata.GetCell(ci)
                if cell is None or cell.GetNumberOfPoints() < 3:
                    continue
                ids = [cell.GetPointId(j) for j in range(3)]
                v0 = np.array(polydata.GetPoint(ids[0]))
                v1 = np.array(polydata.GetPoint(ids[1]))
                v2 = np.array(polydata.GetPoint(ids[2]))
                e1 = v1 - v0
                e2 = v2 - v0
                h = np.cross(ray_dir, e2)
                a = np.dot(e1, h)
                if abs(a) < 1e-10:
                    continue
                f = 1.0 / a
                s = origin - v0
                u = f * np.dot(s, h)
                if u < 0.0 or u > 1.0:
                    continue
                q = np.cross(s, e1)
                v_tri = f * np.dot(ray_dir, q)
                if v_tri < 0.0 or u + v_tri > 1.0:
                    continue
                t = f * np.dot(e2, q)
                if 1e-6 < t < best_t and t < max_dist:
                    best_t = t
                    best_pos = origin + t * ray_dir
            if best_pos is not None:
                return best_pos

        # 回退：点云
        if self.points_data is None or len(self.points_data) == 0:
            return None
        if self._cloud_tree is None:
            from scipy.spatial import cKDTree
            self._cloud_tree = cKDTree(self.points_data)

        ts = np.arange(0.5, max_dist, 0.3)
        sample_pts = origin + ray_dir * ts[:, np.newaxis]
        dists, idxs = self._cloud_tree.query(sample_pts, workers=-1)
        valid = (dists < 0.5)
        if valid.any():
            t_valid = ts[valid]
            best = np.argmin(t_valid)
            return self.points_data[idxs[valid][best]]
        return None

    def _enter_fpv_camera(self):
        """进入FPV相机模式"""
        cam = self.renderer.GetActiveCamera()
        # 获取数据源边界（点云或STL）
        data_min = data_max = None
        if self.points_data is not None and len(self.points_data) > 0:
            data_min = self.points_data.min(axis=0)
            data_max = self.points_data.max(axis=0)
        elif self._stl_polydata is not None:
            bounds = self._stl_polydata.GetBounds()  # (xmin,xmax,ymin,ymax,zmin,zmax)
            data_min = np.array([bounds[0], bounds[2], bounds[4]])
            data_max = np.array([bounds[1], bounds[3], bounds[5]])

        # 使用用户设置的起始位置，或根据数据边界自动计算
        if not hasattr(self, '_fpv_start_pos') or self._fpv_start_pos is None:
            if data_min is not None:
                center = (data_min + data_max) / 2
                size = np.linalg.norm(data_max - data_min)
                # 从侧前方稍高处看向中心
                self._fpv_pos = center + np.array([-size * 0.4, -size * 0.4, size * 0.3])
            else:
                self._fpv_pos = np.array([0.0, 0.0, 0.0])
        else:
            self._fpv_pos = np.array(self._fpv_start_pos, dtype=float)

        # 计算朝向数据几何中心的偏航角
        if data_min is not None:
            center = (data_min + data_max) / 2
            dx = center[0] - self._fpv_pos[0]
            dy = center[1] - self._fpv_pos[1]
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                self._fpv_yaw = np.degrees(np.arctan2(dy, dx))
            else:
                self._fpv_yaw = 0.0
        else:
            self._fpv_yaw = 0.0
        self._fpv_pitch = 0.0

        cam.SetClippingRange(0.1, 5000.0)

        # 禁用VTK自动裁剪范围计算（关键！）
        self._orig_reset_clip = self.renderer.ResetCameraClippingRange
        self.renderer.ResetCameraClippingRange = lambda: None

        # FPV视角下隐藏无人机模型
        if self._fpv_drone_actor is not None:
            self._fpv_drone_actor.SetVisibility(False)

        # FPV模式：关闭背面剔除（进入模型内部也能看到表面）
        for a in self._get_all_mesh_actors():
            a.GetProperty().BackfaceCullingOff()

        self._update_fpv_camera()

    def _exit_fpv_camera(self):
        """退出FPV相机模式，恢复默认视角"""
        self._fpv_keys.clear()
        self._fpv_mouse_active = False
        # 恢复ResetCameraClippingRange
        if hasattr(self, '_orig_reset_clip') and self._orig_reset_clip is not None:
            self.renderer.ResetCameraClippingRange = self._orig_reset_clip
            self._orig_reset_clip = None
        # 不恢复背面剔除——模型加载时已设定，OBJ=Off，STL=On
        self.renderer.ResetCamera()
        if self.vtk_widget.GetRenderWindow():
            self.vtk_widget.GetRenderWindow().Render()

    def _update_fpv_camera(self, render=True):
        """更新FPV相机到无人机位置和朝向"""
        cam = self.renderer.GetActiveCamera()
        pos = self._fpv_pos
        direction = self._fpv_get_look_direction()
        focal = pos + direction * 10

        cam.SetPosition(pos.tolist())
        cam.SetFocalPoint(focal.tolist())
        cam.SetViewUp(0, 0, 1)
        cam.SetViewAngle(self._fpv_fov)
        cam.SetClippingRange(0.1, 5000.0)

        # 确保STL/OBJ保持表面渲染模式
        for a in self._get_all_mesh_actors():
            a.GetProperty().SetRepresentationToSurface()
            a.GetProperty().EdgeVisibilityOff()

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
        right = np.array([np.sin(yaw_rad), -np.cos(yaw_rad), 0])

        min_dist = 5.0  # 最小碰撞距离（防止靠近STL时深度缓冲精度问题）
        # Shift加速
        boost = 'shift_l' in self._fpv_keys or 'shift_r' in self._fpv_keys
        speed = self._fpv_boost_speed if boost else self._fpv_speed
        for key in self._fpv_keys:
            if key in ('shift_l', 'shift_r'):
                continue
            old_pos = self._fpv_pos.copy()
            if key == 'w':
                self._fpv_pos += forward * speed
            elif key == 's':
                self._fpv_pos -= forward * speed
            elif key == 'a':
                self._fpv_pos -= right * speed
            elif key == 'd':
                self._fpv_pos += right * speed
            elif key == 'q':
                self._fpv_pos[2] += speed
            elif key == 'e':
                self._fpv_pos[2] -= speed
            else:
                continue
            # 碰撞检测：离STL表面太近则回退
            if self._stl_tree is not None:
                dist, _ = self._stl_tree.query(self._fpv_pos)
                if dist < min_dist:
                    self._fpv_pos = old_pos
                else:
                    moved = True
            else:
                moved = True

        if moved:
            # 只更新相机参数，最后统一渲染一次
            if self._fpv_first_person:
                self._update_fpv_camera(render=False)
            else:
                self._update_third_person_camera(render=False)
            cam = self.renderer.GetActiveCamera()
            cam.SetClippingRange(0.1, 5000.0)
            if self.vtk_widget.GetRenderWindow():
                self.vtk_widget.GetRenderWindow().Render()
                # 渲染后再次强制裁剪范围，防止VTK内部重置
                cam.SetClippingRange(0.1, 5000.0)
        return moved

    def clear_actors(self):
        self._remove_legend()
        for actor in self._actors:
            self.renderer.RemoveActor(actor)
        self._actors.clear()
        self._cloud_actor = None
        # FPV模式下保留无人机模型
        if self._fpv_drone_actor is not None:
            self.renderer.AddActor(self._fpv_drone_actor)
            self._actors.append(self._fpv_drone_actor)
        # 保留STL/OBJ模型
        for a in self._get_all_mesh_actors():
            self.renderer.AddActor(a)
            self._actors.append(a)

    def add_point_cloud(self, points, render_mode='auto', point_size=0.05, colors=None, normals=None, reset_camera=True, use_lighting=True):
        """显示点云（支持球体/立方体/像素/圆片渲染模式）
        colors: (N, 3) uint8 外部传入的RGB颜色，为None时按高度着色
        normals: (N, 3) 法线数组，render_mode='splat'时用于圆片朝向
        reset_camera: 是否重置相机视角
        use_lighting: 是否启用光照（高度着色等方案应关闭，避免不同视角颜色不一致）
        """
        if not self._vtk_available or len(points) == 0:
            return
        # 过滤NaN/Inf点和极端哨兵值
        valid = np.isfinite(points).all(axis=1) & (np.abs(points) < 1e10).all(axis=1)
        if not valid.all():
            points = points[valid]
            if colors is not None:
                colors = colors[valid]
            if normals is not None:
                normals = normals[valid]
        if len(points) == 0:
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
        polydata.GetPointData().SetActiveScalars('Colors')

        SPHERE_THRESHOLD = 200_000
        # 高程着色时不用 glyph（避免球面插值冲淡颜色），用 GL_POINTS 区分大小
        use_glyph = use_lighting and ((render_mode in ('sphere', 'cube')) or (render_mode == 'auto' and len(render_points) <= SPHERE_THRESHOLD))
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
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToDirectScalars()
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
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToDirectScalars()
        else:
            glyph = self._vtkVertexGlyphFilter()
            glyph.SetInputData(polydata)
            glyph.Update()
            mapper = self._vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            mapper.ScalarVisibilityOn()
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToDirectScalars()

        actor = self._vtkActor()
        actor.SetMapper(mapper)
        if not use_glyph:
            # 按渲染模式区分点大小：球体>立方体>自动>像素
            size_multiplier = {'sphere': 80, 'cube': 60, 'auto': 40, 'pixel': 40, 'splat': 40}.get(render_mode, 40)
            actor.GetProperty().SetPointSize(max(1, int(point_size * size_multiplier)))

        # 增强光照：提高棱角辨识度
        prop = actor.GetProperty()
        if use_lighting:
            prop.SetAmbient(0.3)
            prop.SetDiffuse(0.6)
            prop.SetSpecular(0.2)
            prop.SetSpecularPower(20)
        else:
            prop.LightingOff()
            # 圆片模式保留少量环境光，让法线朝向差异可见
            if use_splat:
                prop.LightingOn()
                prop.SetAmbient(0.8)
                prop.SetDiffuse(0.2)
                prop.SetSpecular(0.0)

        self.renderer.AddActor(actor)
        self._actors.append(actor)
        self._cloud_actor = actor

        # FPV模式下不重置相机，或者明确指定不重置
        if reset_camera and not getattr(self, 'fpv_mode', False):
            self.renderer.ResetCamera()
        self._update_view(reset_camera=reset_camera)

    def _remove_legend(self):
        """移除已有的图例"""
        if hasattr(self, '_legend_actors'):
            for a in self._legend_actors:
                self.renderer.RemoveActor(a)
            self._legend_actors.clear()

    def show_height_legend(self, z_min, z_max):
        """显示高程颜色图例（彩色标注 + 最小/最大高度）"""
        if not self._vtk_available:
            return
        self._remove_legend()
        self._legend_actors = []

        font_size = 13
        # 颜色条标注：高→低，红→黄→绿→青→蓝
        # 对应 height 着色方案的 5 个区间
        color_labels = [
            ("%.1fm" % z_max, (1.0, 0.0, 0.0), 0.92),   # 红 (最高)
            ("", (1.0, 1.0, 0.0), 0.82),                  # 黄
            ("", (0.0, 1.0, 0.0), 0.72),                  # 绿
            ("", (0.0, 1.0, 1.0), 0.62),                  # 青
            ("%.1fm" % z_min, (0.0, 0.0, 1.0), 0.52),    # 蓝 (最低)
        ]

        # 标题
        title = self._vtkTextActor()
        title.SetInput("高程")
        tp = title.GetTextProperty()
        tp.SetFontSize(font_size + 2)
        tp.SetColor(1.0, 1.0, 1.0)
        tp.SetBold(True)
        tp.SetShadow(True)
        coord = title.GetPositionCoordinate()
        coord.SetCoordinateSystemToNormalizedViewport()
        coord.SetValue(0.88, 0.95)
        self.renderer.AddActor(title)
        self._legend_actors.append(title)

        # 颜色块 + 标注
        for label, color, y_pos in color_labels:
            # 彩色方块用 ■ 字符表示
            block = self._vtkTextActor()
            block.SetInput("■")
            tp = block.GetTextProperty()
            tp.SetFontSize(font_size + 4)
            tp.SetColor(*color)
            tp.SetBold(True)
            tp.SetShadow(False)
            coord = block.GetPositionCoordinate()
            coord.SetCoordinateSystemToNormalizedViewport()
            coord.SetValue(0.88, y_pos)
            self.renderer.AddActor(block)
            self._legend_actors.append(block)

            # 高度标签
            if label:
                ta = self._vtkTextActor()
                ta.SetInput(label)
                tp = ta.GetTextProperty()
                tp.SetFontSize(font_size)
                tp.SetColor(1.0, 1.0, 1.0)
                tp.SetBold(True)
                tp.SetShadow(True)
                coord = ta.GetPositionCoordinate()
                coord.SetCoordinateSystemToNormalizedViewport()
                coord.SetValue(0.92, y_pos)
                self.renderer.AddActor(ta)
                self._legend_actors.append(ta)
        self._legend_actors.append(title)

        self._update_view(reset_camera=False)

    @staticmethod
    def _estimate_voxel_size(points, target_count):
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        volume = np.prod(mx - mn + 1e-10)
        if not np.isfinite(volume) or volume <= 0:
            return 0.01
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
        """用PCA方法估算点云法线（批量向量化），返回 (N,3) 法线数组"""
        from scipy.spatial import cKDTree
        n = len(points)
        if n < 3:
            return np.zeros((n, 3), dtype=np.float64)

        k = min(k, n)
        tree = cKDTree(points)
        # 多线程批量查询
        _, indices = tree.query(points, k=k, workers=-1)

        # 批量提取邻域: (N, k, 3)
        neighbors = points[indices]
        # 批量计算均值: (N, 1, 3)
        centers = neighbors.mean(axis=1, keepdims=True)
        # 批量去均值: (N, k, 3)
        centered = neighbors - centers
        # 批量协方差矩阵: (N, 3, 3) = (N, 3, k) @ (N, k, 3)
        cov = np.einsum('nik,njk->nij', centered, centered) / (k - 1)
        # 批量特征值分解
        eigvals, eigvecs = np.linalg.eigh(cov)
        # 最小特征值对应的特征向量 = 法线 (第一列)
        normals = eigvecs[:, :, 0].copy()

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

        if normals is None or len(normals) != n:
            # 法线无效时直接返回原始点
            return points, colors

        normals = np.asarray(normals, dtype=np.float64)
        if normals.ndim != 2 or normals.shape[1] != 3:
            # 法线 shape 异常，丢弃
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

        # 在切平面内生成随机偏移（向量化）
        np.random.seed(42)  # 固定种子保证一致性
        angles = np.random.uniform(0, 2 * np.pi, (n, factor))
        dists = np.random.uniform(0, radius, (n, factor))

        offsets_x = dists * np.cos(angles)  # (n, factor)
        offsets_y = dists * np.sin(angles)  # (n, factor)

        # 向量化生成所有新点: (n, factor, 3)
        # tangent1, tangent2: (n, 3) → (n, 1, 3)
        t1 = tangent1[:, np.newaxis, :]  # (n, 1, 3)
        t2 = tangent2[:, np.newaxis, :]  # (n, 1, 3)
        # offsets: (n, factor) → (n, factor, 1)
        ox = offsets_x[:, :, np.newaxis]  # (n, factor, 1)
        oy = offsets_y[:, :, np.newaxis]  # (n, factor, 1)
        # 新点 = 原始点 + 偏移
        base = points[:, np.newaxis, :]  # (n, 1, 3)
        new_pts = base + ox * t1 + oy * t2  # (n, factor, 3)

        # 合并: 原始点 + 所有新点
        all_points = np.vstack([points, new_pts.reshape(-1, 3)])

        # 处理颜色
        all_colors = None
        if colors is not None:
            all_colors = np.vstack([colors] + [colors] * factor)

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

    def add_route(self, waypoints, reset_camera=True):
        """显示航线和航点"""
        if not self._vtk_available or len(waypoints) == 0:
            return

        # 清除多边形平面残留
        if self._poly_surface_actor:
            self.renderer.RemoveActor(self._poly_surface_actor)
            self._poly_surface_actor = None

        mesh_actors = set(id(a) for a in self._get_all_mesh_actors())
        to_remove = []
        for i, actor in enumerate(self._actors):
            if actor != self._cloud_actor and actor != self._fpv_drone_actor and id(actor) not in mesh_actors:
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
        actor.GetProperty().SetLineWidth(2)
        actor.GetProperty().SetOpacity(0.9)
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
            glyph_src.SetRadius(0.08)
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
            sa.GetProperty().SetOpacity(0.9)
            self.renderer.AddActor(sa)
            self._actors.append(sa)
            self._waypoint_actors = [sa] * n
        else:
            # ── 少量航点：逐个创建（支持单独拖拽编辑）──
            for wp in waypoints:
                sphere = self._vtkSphereSource()
                sphere.SetCenter(wp['pos'].tolist())
                sphere.SetRadius(0.08)
                sphere.Update()
                m = self._vtkPolyDataMapper()
                m.SetInputConnection(sphere.GetOutputPort())
                a = self._vtkActor()
                a.SetMapper(m)
                a.GetProperty().SetColor(1.0, 0.2, 0.2)
                a.GetProperty().SetOpacity(0.9)
                self.renderer.AddActor(a)
                self._actors.append(a)
                self._waypoint_actors.append(a)

            # 标签只在少量航点时显示
            for i, wp in enumerate(waypoints):
                lbl_pos = [wp['pos'][0], wp['pos'][1], wp['pos'][2] + label_offset * 2]
                lbl_scale = label_offset * 0.5

                # 背景标签（黑色，稍偏移形成阴影）
                bg = self._vtkBillboardTextActor3D()
                bg.SetInput(str(i + 1))
                bg.SetPosition(lbl_pos[0] + lbl_scale * 0.05, lbl_pos[1] + lbl_scale * 0.05, lbl_pos[2] - lbl_scale * 0.05)
                bg.SetScale(lbl_scale, lbl_scale, lbl_scale)
                bg.GetTextProperty().SetColor(0.0, 0.0, 0.0)
                bg.GetTextProperty().SetFontSize(18)
                bg.GetTextProperty().SetBold(True)
                self.renderer.AddActor(bg)
                self._actors.append(bg)

                # 前景标签（亮黄色）
                fg = self._vtkBillboardTextActor3D()
                fg.SetInput(str(i + 1))
                fg.SetPosition(lbl_pos[0], lbl_pos[1], lbl_pos[2])
                fg.SetScale(lbl_scale, lbl_scale, lbl_scale)
                fg.GetTextProperty().SetColor(1.0, 1.0, 0.0)
                fg.GetTextProperty().SetFontSize(18)
                fg.GetTextProperty().SetBold(True)
                self.renderer.AddActor(fg)
                self._actors.append(fg)

        # ── 首尾航点坐标标签 ──
        coord_scale = label_offset * 0.6
        for idx, tag in [(0, "首"), (n - 1, "末")] if n >= 2 else [(0, "首")]:
            wp = waypoints[idx]
            p = wp['pos']
            coord_text = f"{tag}点({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})"
            # 背景（黑色阴影）
            bg = self._vtkBillboardTextActor3D()
            bg.SetInput(coord_text)
            bg.SetPosition(p[0] + coord_scale * 0.04, p[1] + coord_scale * 0.04, p[2] + label_offset * 3 + coord_scale * 0.04)
            bg.SetScale(coord_scale, coord_scale, coord_scale)
            bg.GetTextProperty().SetColor(0.0, 0.0, 0.0)
            bg.GetTextProperty().SetFontSize(16)
            bg.GetTextProperty().SetBold(True)
            self.renderer.AddActor(bg)
            self._actors.append(bg)
            # 前景（青色）
            fg = self._vtkBillboardTextActor3D()
            fg.SetInput(coord_text)
            fg.SetPosition(p[0], p[1], p[2] + label_offset * 3)
            fg.SetScale(coord_scale, coord_scale, coord_scale)
            fg.GetTextProperty().SetColor(0.0, 1.0, 1.0)
            fg.GetTextProperty().SetFontSize(16)
            fg.GetTextProperty().SetBold(True)
            self.renderer.AddActor(fg)
            self._actors.append(fg)

        # ── 相机覆盖区域投影（显示重叠率）──
        has_target = any('target_pos' in wp for wp in waypoints)
        if has_target and getattr(self, '_coverage_enabled', True):
            hfov_rad = math.radians(self._camera_hfov)
            vfov_rad = math.radians(self._camera_vfov)
            headings = self._compute_forward_headings(waypoints)
            for i, wp in enumerate(waypoints):
                if 'target_pos' not in wp:
                    continue
                pos = np.array(wp['pos'], dtype=float)
                tgt = np.array(wp['target_pos'], dtype=float)
                los = tgt - pos
                dist = np.linalg.norm(los)
                if dist < 0.5:
                    continue
                half_w = dist * math.tan(hfov_rad / 2.0)
                half_h = dist * math.tan(vfov_rad / 2.0)
                heading = headings[i] if i < len(headings) else np.array([1.0, 0.0, 0.0])
                fwd_h = np.array([heading[0], heading[1], 0.0])
                fwd_len = np.linalg.norm(fwd_h)
                if fwd_len < 1e-6:
                    fwd_h = np.array([1.0, 0.0, 0.0])
                else:
                    fwd_h = fwd_h / fwd_len
                right_h = np.cross(np.array([0.0, 0.0, 1.0]), fwd_h)
                rn = np.linalg.norm(right_h)
                if rn < 1e-6:
                    right_h = np.array([0.0, 1.0, 0.0])
                else:
                    right_h = right_h / rn
                corners_3d = [
                    tgt + fwd_h * half_h + right_h * half_w,
                    tgt + fwd_h * half_h - right_h * half_w,
                    tgt - fwd_h * half_h - right_h * half_w,
                    tgt - fwd_h * half_h + right_h * half_w,
                ]
                proj_corners = []
                for corner in corners_3d:
                    pt = self._project_to_stl(pos, corner)
                    if pt is None:
                        pt = self._project_to_cloud(pos, corner)
                    if pt is None:
                        pt = corner
                    proj_corners.append(pt)
                # 纹理贴图（网格图案，用于显示重叠率）
                vtk_pts = self._vtkPoints()
                tc = self._vtkFloatArray()
                tc.SetNumberOfComponents(2)
                tc.SetName("TextureCoordinates")
                tex_coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                for k, c in enumerate(proj_corners):
                    vtk_pts.InsertNextPoint(c.tolist())
                    tc.InsertNextTuple2(tex_coords[k][0], tex_coords[k][1])
                polygon = self._vtkPolygon()
                pid = polygon.GetPointIds()
                pid.SetNumberOfIds(4)
                for k in range(4):
                    pid.SetId(k, k)
                cells = self._vtkCellArray()
                cells.InsertNextCell(polygon)
                polydata = self._vtkPolyData()
                polydata.SetPoints(vtk_pts)
                polydata.SetPolys(cells)
                polydata.GetPointData().SetTCoords(tc)
                tex = self._get_grid_texture()
                mapper = self._vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                actor = self._vtkActor()
                actor.SetMapper(mapper)
                actor.SetTexture(tex)
                actor.GetProperty().SetOpacity(0.4)
                actor.GetProperty().SetLighting(False)
                actor.SetPosition(0, 0, 0.01)
                self.renderer.AddActor(actor)
                self._actors.append(actor)
                # 边框
                edge_pts = self._vtkPoints()
                for c in proj_corners:
                    edge_pts.InsertNextPoint(c.tolist())
                edge_line = self._vtkPolyLine()
                edge_line.GetPointIds().SetNumberOfIds(5)
                for k in range(4):
                    edge_line.GetPointIds().SetId(k, k)
                edge_line.GetPointIds().SetId(4, 0)
                edge_cells = self._vtkCellArray()
                edge_cells.InsertNextCell(edge_line)
                edge_poly = self._vtkPolyData()
                edge_poly.SetPoints(edge_pts)
                edge_poly.SetLines(edge_cells)
                edge_mapper = self._vtkPolyDataMapper()
                edge_mapper.SetInputData(edge_poly)
                edge_actor = self._vtkActor()
                edge_actor.SetMapper(edge_mapper)
                edge_actor.GetProperty().SetColor(0.0, 0.8, 1.0)
                edge_actor.GetProperty().SetLineWidth(2)
                edge_actor.GetProperty().SetLighting(False)
                edge_actor.SetPosition(0, 0, 0.01)
                self.renderer.AddActor(edge_actor)
                self._actors.append(edge_actor)

        # ── 机头方向箭头 ──
        if self.show_heading:
            headings = self._compute_forward_headings(waypoints)
            for i, wp in enumerate(waypoints):
                pos = wp['pos']
                heading = headings[i] if i < len(headings) else np.array([1.0, 0.0, 0.0])
                # 水平方向（去掉Z分量）
                h_horiz = np.array([heading[0], heading[1], 0.0])
                h_len = np.linalg.norm(h_horiz)
                if h_len < 1e-10:
                    continue
                h_horiz /= h_len
                arrow_len = 1.0
                arrow_end = pos + h_horiz * arrow_len
                line = self._vtkLineSource()
                line.SetPoint1(pos.tolist())
                line.SetPoint2(arrow_end.tolist())
                mapper = self._vtkPolyDataMapper()
                mapper.SetInputConnection(line.GetOutputPort())
                actor = self._vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0.0, 0.8, 1.0)
                actor.GetProperty().SetLineWidth(3)
                self.renderer.AddActor(actor)
                self._actors.append(actor)

        # ── 云台方向线（航点→目标点）──
        if getattr(self, 'show_gimbal_dir', False):
            for i, wp in enumerate(waypoints):
                if 'target_pos' not in wp:
                    continue
                pos = wp['pos']
                tgt = wp['target_pos']
                line = self._vtkLineSource()
                line.SetPoint1(pos.tolist())
                line.SetPoint2(tgt.tolist())
                mapper = self._vtkPolyDataMapper()
                mapper.SetInputConnection(line.GetOutputPort())
                actor = self._vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.4, 0.0)
                actor.GetProperty().SetLineWidth(2)
                actor.GetProperty().SetOpacity(0.7)
                self.renderer.AddActor(actor)
                self._actors.append(actor)

        if reset_camera:
            self._update_view()
        else:
            self.vtk_widget.GetRenderWindow().Render()

    def _get_grid_texture(self):
        """获取网格纹理（带缓存），用于生成航线时显示覆盖重叠率"""
        if hasattr(self, '_grid_texture_cached') and self._grid_texture_cached is not None:
            return self._grid_texture_cached
        from PIL import Image, ImageDraw
        import os, tempfile
        w, h = 512, 512
        img = Image.new('RGBA', (w, h), (30, 30, 50, 200))
        draw = ImageDraw.Draw(img)
        for y in range(h):
            r = int(30 + 80 * y / h)
            g = int(50 + 60 * y / h)
            b = int(100 + 80 * y / h)
            draw.line([(0, y), (w, y)], fill=(r, g, b, 180))
        grid = 32
        for x in range(0, w, grid):
            draw.line([(x, 0), (x, h)], fill=(0, 200, 255, 100), width=1)
        for y in range(0, h, grid):
            draw.line([(0, y), (w, y)], fill=(0, 200, 255, 100), width=1)
        draw.line([(w//2, 0), (w//2, h)], fill=(0, 255, 100, 200), width=2)
        draw.line([(0, h//2), (w, h//2)], fill=(0, 255, 100, 200), width=2)
        cx, cy, r = w//2, h//2, min(w, h)//4
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0, 255, 100, 200), width=2)
        tmp_path = os.path.join(tempfile.gettempdir(), '_grid_tex.png')
        img.save(tmp_path)
        reader = self._vtkPNGReader()
        reader.SetFileName(tmp_path)
        reader.Update()
        tex = self._vtkTexture()
        tex.SetInputConnection(reader.GetOutputPort())
        tex.InterpolateOn()
        self._grid_texture_cached = tex
        return tex

    def _get_coverage_texture(self):
        """获取相机覆盖区域纹理（带缓存），优先加载 lena.png"""
        if hasattr(self, '_coverage_texture_cached') and self._coverage_texture_cached is not None:
            return self._coverage_texture_cached
        import os, tempfile
        vtk = self._vtk
        # 查找 lena.png（支持 PyInstaller 打包路径）
        import sys
        search_dirs = []
        if getattr(sys, 'frozen', False):
            search_dirs.append(sys._MEIPASS)
        search_dirs += [
            os.path.dirname(os.path.abspath(__file__)),
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            os.getcwd(),
            tempfile.gettempdir(),
        ]
        lena_path = None
        for d in search_dirs:
            candidate = os.path.join(d, 'lena.png')
            if os.path.exists(candidate):
                lena_path = candidate
                break
        if lena_path:
            reader = self._vtkPNGReader()
            reader.SetFileName(lena_path)
            reader.Update()
            tex = self._vtkTexture()
            tex.SetInputConnection(reader.GetOutputPort())
            tex.InterpolateOn()
            self._coverage_texture_cached = tex
            print(f"[Coverage] Loaded lena.png from {lena_path}")
            return tex
        # 回退：生成网格测试图
        from PIL import Image, ImageDraw
        # 生成测试图像（带网格 + 渐变）
        w, h = 512, 512
        img = Image.new('RGBA', (w, h), (30, 30, 50, 200))
        draw = ImageDraw.Draw(img)
        # 渐变背景
        for y in range(h):
            r = int(30 + 80 * y / h)
            g = int(50 + 60 * y / h)
            b = int(100 + 80 * y / h)
            draw.line([(0, y), (w, y)], fill=(r, g, b, 180))
        # 网格
        grid = 32
        for x in range(0, w, grid):
            draw.line([(x, 0), (x, h)], fill=(0, 200, 255, 100), width=1)
        for y in range(0, h, grid):
            draw.line([(0, y), (w, y)], fill=(0, 200, 255, 100), width=1)
        # 中心十字
        draw.line([(w//2, 0), (w//2, h)], fill=(0, 255, 100, 200), width=2)
        draw.line([(0, h//2), (w, h//2)], fill=(0, 255, 100, 200), width=2)
        # 中心圆
        cx, cy, r = w//2, h//2, min(w, h)//4
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0, 255, 100, 200), width=2)
        # 角落标记
        mark = 40
        for (sx, sy) in [(0, 0), (w-mark, 0), (0, h-mark), (w-mark, h-mark)]:
            draw.rectangle([sx, sy, sx+mark, sy+mark], outline=(255, 255, 0, 200), width=2)
        # 保存到临时文件
        tmp_path = os.path.join(tempfile.gettempdir(), '_coverage_tex.png')
        img.save(tmp_path)
        # 加载为VTK纹理
        reader = self._vtkPNGReader()
        reader.SetFileName(tmp_path)
        reader.Update()
        tex = self._vtkTexture()
        tex.SetInputConnection(reader.GetOutputPort())
        tex.InterpolateOn()
        self._coverage_texture_cached = tex
        return tex

    def _project_to_stl(self, pos, target_pos, max_dist=50.0):
        """从航点沿目标方向投射射线，找到与STL表面的交点
        pos: 航点位置 [x,y,z]
        target_pos: 目标点位置 [x,y,z]
        返回: 投影点 np.array 或 None
        """
        locator = getattr(self, '_stl_cell_locator', None)
        if locator is None or self._stl_polydata is None:
            return None
        vtk = self._vtk
        origin = np.array(pos, dtype=float)
        target = np.array(target_pos, dtype=float)
        direction = target - origin
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return None
        direction = direction / dist
        p1 = (origin - direction * 0.5).tolist()
        p2 = (origin + direction * min(max_dist, dist * 2.0)).tolist()
        t = vtk.reference(0.0)
        x = [0.0, 0.0, 0.0]
        pcoords = [0.0, 0.0, 0.0]
        subId = vtk.reference(0)
        cellId = vtk.reference(0)
        cell = vtk.vtkGenericCell()
        hit = locator.IntersectWithLine(p1, p2, 1e-6, t, x, pcoords, subId, cellId, cell)
        if hit:
            return np.array(x)
        return None

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

        # FPV模式下使用相机参数直接计算射线
        if self.fpv_mode:
            return self._fpv_pick(screen_x, screen_y)

        # 0) 优先：STL/OBJ 表面拾取
        mesh_actors = self._get_all_mesh_actors()
        if mesh_actors:
            # 多边形/巡检/直线模式：用射线求交，只取正面
            if self.polygon_mode or self.inspect_mode or self.line_mode:
                pos, normal = self._pick_stl_top_surface(screen_x, screen_y)
                if pos is not None:
                    if normal is not None:
                        self._picked_normal = normal
                    return pos
            # 其他模式：CellPicker
            cell_picker = self._vtk.vtkCellPicker()
            cell_picker.PickFromListOn()
            for a in mesh_actors:
                cell_picker.AddPickList(a)
            cell_picker.SetTolerance(0.01)
            if cell_picker.Pick(screen_x, screen_y, 0, self.renderer):
                pos = np.array(cell_picker.GetPickPosition())
                cell_id = cell_picker.GetCellId()
                if cell_id >= 0 and self._stl_normals_np is not None:
                    polydata = self._stl_polydata
                    cell = polydata.GetCell(cell_id)
                    if cell and cell.GetNumberOfPoints() >= 3:
                        ids = [cell.GetPointId(j) for j in range(3)]
                        n0 = self._stl_normals_np[ids[0]]
                        n1 = self._stl_normals_np[ids[1]]
                        n2 = self._stl_normals_np[ids[2]]
                        normal = (n0 + n1 + n2) / 3.0
                        norm = np.linalg.norm(normal)
                        if norm > 1e-10:
                            normal /= norm
                        cam = self.renderer.GetActiveCamera()
                        viewer_pos = np.array(cam.GetPosition(), dtype=float)
                        if np.dot(normal, viewer_pos - pos) < 0:
                            normal = -normal
                        self._picked_normal = normal
                return pos

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
                # 批量生成射线采样点并一次查询
                ts = np.arange(0.5, ray_len, 0.3)
                sample_pts = near + ray_dir * ts[:, np.newaxis]  # (M, 3)
                dists, idxs = self._cloud_tree.query(sample_pts, workers=-1)
                valid = dists < 1.0
                if valid.any():
                    best = np.argmin(np.where(valid, dists, np.inf))
                    return self.points_data[idxs[best]]

        # 3) 回退：Z=0 平面
        return self._ray_z_plane(screen_x, screen_y, 0.0)

    def _fpv_pick(self, screen_x, screen_y):
        """FPV模式下拾取点云点 - 直接用DisplayToWorld计算射线"""
        # 优先：STL/OBJ 表面射线求交（精确法线）
        if self._get_all_mesh_actors():
            pos, normal = self._pick_stl_top_surface(screen_x, screen_y)
            if pos is not None:
                if normal is not None:
                    self._picked_normal = normal
                return pos

        if self.points_data is None or len(self.points_data) == 0:
            return None

        # 用DisplayToWorld计算射线
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
        if ray_len < 1e-10:
            return None

        ray_dir /= ray_len

        if self._cloud_tree is None:
            from scipy.spatial import cKDTree
            self._cloud_tree = cKDTree(self.points_data)

        ts = np.arange(0.5, ray_len, 0.3)
        sample_pts = near + ray_dir * ts[:, np.newaxis]
        dists, idxs = self._cloud_tree.query(sample_pts, workers=-1)
        valid = dists < 1.0
        if valid.any():
            best = np.argmin(np.where(valid, dists, np.inf))
            result = self.points_data[idxs[best]]
            return result

        return None

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
        has_stl = bool(self._get_all_mesh_actors())
        has_cloud = self.points_data is not None and len(self.points_data) > 0
        if not has_stl and not has_cloud:
            print("[Polygon] No point cloud or STL loaded")
            return
        self.polygon_mode = True
        self._poly_points = []
        self._poly_normals = []
        self._clear_polygon()
        print("[Polygon] 左键点击添加顶点，右键结束绘制，Esc取消")

    def exit_polygon_mode(self, clear_markers=True):
        self.polygon_mode = False
        self._poly_points = []
        self._poly_normals = []
        self._poly_click_start = None
        if clear_markers:
            self._clear_polygon()
        print("[Polygon] Exited polygon mode.")

    # ─── 点击放置模式 ─────────────────────────────────────
    def enter_place_mode(self):
        has_stl = bool(self._get_all_mesh_actors())
        has_cloud = self.points_data is not None and len(self.points_data) > 0
        if not has_stl and not has_cloud:
            print("[Place] No point cloud or STL loaded")
            return
        self._clear_place_preview()
        self._clear_polygon()
        self.place_mode = True
        self._place_preview_pos = None
        self._place_preview_actor = None

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
        sphere.SetRadius(0.2)
        sphere.Update()
        m = self._vtkPolyDataMapper()
        m.SetInputConnection(sphere.GetOutputPort())
        a = self._vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(1.0, 0.8, 0.0)
        a.GetProperty().SetOpacity(0.7)
        self.renderer.AddActor(a)
        self._place_preview_actor = a
        self.vtk_widget.GetRenderWindow().Render()

    def _clear_place_preview(self):
        if self._place_preview_actor:
            self.renderer.RemoveActor(self._place_preview_actor)
            self._place_preview_actor = None

    def _add_polygon_point(self, pos, normal=None):
        # 存储位置和表面法线
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0])
        pos = np.array(pos, dtype=float)
        normal = np.array(normal, dtype=float)
        self._poly_points.append(pos)
        self._poly_normals.append(normal)

        sphere = self._vtkSphereSource()
        sphere.SetCenter(pos.tolist())
        sphere.SetRadius(0.2)
        sphere.Update()
        m = self._vtkPolyDataMapper()
        m.SetInputConnection(sphere.GetOutputPort())
        a = self._vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(1.0, 1.0, 0.0)
        a.GetProperty().SetLighting(False)
        self.renderer.AddActor(a)
        self._poly_markers.append(a)

        self._update_polygon_line()
        self.vtk_widget.GetRenderWindow().Render()

    def _update_polygon_line(self):
        if self._poly_line_actor:
            self.renderer.RemoveActor(self._poly_line_actor)
            self._poly_line_actor = None
        if self._poly_surface_actor:
            self.renderer.RemoveActor(self._poly_surface_actor)
            self._poly_surface_actor = None

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
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        actor = self._vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        actor.GetProperty().SetLineWidth(3)
        actor.GetProperty().SetOpacity(0.9)
        actor.GetProperty().SetLighting(False)
        self.renderer.AddActor(actor)
        self._poly_line_actor = actor

        # 3个点以上时绘制透明填充平面
        if n >= 3:
            polygon = self._vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(n)
            for i in range(n):
                polygon.GetPointIds().SetId(i, i)

            poly_cells = self._vtkCellArray()
            poly_cells.InsertNextCell(polygon)

            surf_data = self._vtkPolyData()
            surf_data.SetPoints(vtk_pts)
            surf_data.SetPolys(poly_cells)

            surf_mapper = self._vtkPolyDataMapper()
            surf_mapper.SetInputData(surf_data)
            surf_mapper.SetResolveCoincidentTopologyToPolygonOffset()
            surf_actor = self._vtkActor()
            surf_actor.SetMapper(surf_mapper)
            surf_actor.GetProperty().SetColor(0.2, 0.6, 1.0)
            surf_actor.GetProperty().SetOpacity(0.25)
            surf_actor.GetProperty().SetEdgeVisibility(True)
            surf_actor.GetProperty().SetEdgeColor(0.3, 0.7, 1.0)
            surf_actor.GetProperty().SetLineWidth(2)
            surf_actor.GetProperty().SetLighting(False)
            self.renderer.AddActor(surf_actor)
            self._poly_surface_actor = surf_actor

    def _clear_polygon(self):
        for a in self._poly_markers:
            self.renderer.RemoveActor(a)
        self._poly_markers.clear()
        if self._poly_line_actor:
            self.renderer.RemoveActor(self._poly_line_actor)
            self._poly_line_actor = None
        if self._poly_surface_actor:
            self.renderer.RemoveActor(self._poly_surface_actor)
            self._poly_surface_actor = None

    # ─── 巡检选点模式 ─────────────────────────────────────────
    def enter_inspect_mode(self):
        has_stl = bool(self._get_all_mesh_actors())
        has_cloud = self.points_data is not None and len(self.points_data) > 0
        if not has_stl and not has_cloud:
            print("[Inspect] No point cloud or STL loaded")
            return
        self.inspect_mode = True
        self._poly_click_start = None
        print("[Inspect] 左键点击添加巡检点，右键确认，Esc取消")

    def exit_inspect_mode(self, clear_markers=True):
        self.inspect_mode = False
        self._poly_click_start = None
        if clear_markers:
            self._clear_inspect_points()
        print("[Inspect] Exited inspect mode.")

    def _add_inspect_point(self, pos):
        # 存储 (位置, 法线) 对，法线来自拾取时的精确三角面法线
        normal = self._picked_normal if self._picked_normal is not None else np.array([0.0, 0.0, 1.0])
        self._inspect_points.append((pos.copy(), normal.copy()))
        self._picked_normal = None  # 用完清掉
        sphere = self._vtkSphereSource()
        sphere.SetCenter(pos.tolist())
        sphere.SetRadius(0.2)
        sphere.Update()
        m = self._vtkPolyDataMapper()
        m.SetInputConnection(sphere.GetOutputPort())
        a = self._vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(1.0, 0.8, 0.0)  # 黄色
        a.GetProperty().SetLighting(False)
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
        has_stl = bool(self._get_all_mesh_actors())
        has_cloud = self.points_data is not None and len(self.points_data) > 0
        if not has_stl and not has_cloud:
            print("[Line] No point cloud or STL loaded")
            return
        self.line_mode = True
        self._poly_click_start = None
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
        # 存储 (位置, 法线) 对
        normal = self._picked_normal if self._picked_normal is not None else np.array([0.0, 0.0, 1.0])
        self._line_points.append((pos.copy(), normal.copy()))
        self._picked_normal = None
        idx = len(self._line_points) - 1  # 0=起点, 1=终点
        sphere = self._vtkSphereSource()
        sphere.SetCenter(pos.tolist())
        sphere.SetRadius(0.2)
        sphere.Update()
        m = self._vtkPolyDataMapper()
        m.SetInputConnection(sphere.GetOutputPort())
        a = self._vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(1.0, 0.8, 0.0)  # 与点状航线一致
        a.GetProperty().SetLighting(False)
        self.renderer.AddActor(a)
        self._line_markers.append(a)
        self.vtk_widget.GetRenderWindow().Render()
        # 实时通知主窗口更新坐标
        self.line_point_picked.emit(idx, pos.tolist())
        if len(self._line_points) == 2:
            print("[Line] 已选起点和终点，右键确认生成航线")

    def _clear_line_points(self):
        for a in self._line_markers:
            self.renderer.RemoveActor(a)
        self._line_markers.clear()
        self._line_points.clear()

    def _update_view(self, reset_camera=True):
        if not self._vtk_available:
            return
        # FPV模式下由FPV系统控制相机，跳过默认视角设置
        if getattr(self, 'fpv_mode', False):
            self.vtk_widget.GetRenderWindow().Render()
            return
        if reset_camera:
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

    def _on_view_label_click(self, name, idx):
        """视角标签点击"""
        _active_style = (
            "QLabel { color: #ffa500; background: #333; border: 1px solid #ffa500; "
            "font-size: 10px; font-family: Consolas, monospace; font-weight: bold; }"
        )
        _inactive_style = (
            "QLabel { color: #555; background: #2a2a2a; border: 1px solid #3a3a3a; "
            "font-size: 10px; font-family: Consolas, monospace; }"
        )
        for j, (lbl, n) in enumerate(self._view_btns_list):
            lbl.setStyleSheet(_active_style if j == idx else _inactive_style)
        self._view_btn_active = idx
        self._set_view(name)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_view_frame'):
            self._view_frame.move(self.width() - 250, 8)

    def _set_view(self, name):
        if not self._vtk_available or getattr(self, 'fpv_mode', False):
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
        # FPV模式下禁用视角切换快捷键
        if not getattr(self, 'fpv_mode', False):
            key_view_map = {
                '1': 'top', '2': 'front', '3': 'side', '4': 'persp', '5': 'bottom',
            }
            if key in key_view_map:
                self._set_view(key_view_map[key])
        if key == '\x1b':
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
        # 清除旧的坐标轴actor
        for a in self._axes_actors:
            if a in self._actors:
                self._actors.remove(a)
            self.renderer.RemoveActor(a)
        self._axes_actors = []

        # 根据数据范围自动缩放坐标轴大小
        data_size = 10.0
        if self.points_data is not None and len(self.points_data) > 0:
            data_size = np.linalg.norm(self.points_data.max(axis=0) - self.points_data.min(axis=0))
        elif self._stl_polydata is not None:
            b = self._stl_polydata.GetBounds()
            data_size = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
        arrow_len = max(1.0, data_size * 0.05)
        axes_config = [
            # (color, rotate_func) - 箭头默认沿X轴
            ((1, 0, 0), None),                    # X轴：无需旋转
            ((0, 1, 0), (90, 0, 0, 1)),            # Y轴：绕Z旋转90°
            ((0, 0, 1), (-90, 0, 1, 0)),           # Z轴：绕Y旋转-90°
        ]
        for color, rot in axes_config:
            arrow = self._vtkArrowSource()
            arrow.SetShaftRadius(0.015)
            arrow.SetTipRadius(0.04)
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
            a.GetProperty().SetLighting(False)
            self.renderer.AddActor(a)
            self._actors.append(a)
            self._axes_actors.append(a)

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
        a.GetProperty().SetColor(86/255, 165/255, 251/255)  # #56A5FB
        a.GetProperty().SetOpacity(0.6)
        a.GetProperty().SetLighting(False)
        self.renderer.AddActor(a)
        self._actors.append(a)
        self._axes_actors.append(a)

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
