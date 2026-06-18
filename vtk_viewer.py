"""VTK 3D 点云/航线可视化组件"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

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
        from vtk import (
            vtkActor, vtkPolyDataMapper, vtkRenderer,
            vtkPoints, vtkPolyData, vtkVertexGlyphFilter,
            vtkSphereSource, vtkLineSource, vtkCellArray,
            vtkPolyLine, vtkFollower, vtkVectorText,
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

        def set_viewer(self, viewer):
            self._vtk_viewer = viewer

        # ── 左键 ──
        def OnLeftButtonDown(self):
            v = self._vtk_viewer
            if v is None:
                self.StartPan()
                return

            rwi = self.GetInteractor()
            ctrl = rwi.GetControlKey()
            pos = rwi.GetEventPosition()

            if v.polygon_mode:
                v._poly_click_start = pos
                return

            if v.place_mode:
                v._poly_click_start = pos
                return

            if ctrl:
                wp_idx = v._find_nearest_waypoint(pos[0], pos[1])
                if wp_idx >= 0:
                    v._start_wp_edit(wp_idx, pos[0], pos[1])
                    return

            self.StartPan()

        def OnLeftButtonUp(self):
            v = self._vtk_viewer

            if v and v.polygon_mode and v._poly_click_start is not None:
                rwi = self.GetInteractor()
                pos = rwi.GetEventPosition()
                start = v._poly_click_start
                v._poly_click_start = None
                if abs(pos[0] - start[0]) < 5 and abs(pos[1] - start[1]) < 5:
                    p = v._pick_3d(pos[0], pos[1])
                    if p is not None:
                        v._add_polygon_point(p)
                return

            if v and v.place_mode and v._poly_click_start is not None:
                rwi = self.GetInteractor()
                pos = rwi.GetEventPosition()
                start = v._poly_click_start
                v._poly_click_start = None
                if abs(pos[0] - start[0]) < 5 and abs(pos[1] - start[1]) < 5:
                    p = v._pick_3d(pos[0], pos[1])
                    if p is not None:
                        v._update_place_preview(p)
                return

            if v and v._wp_editing:
                v._end_wp_edit()
                return
            self.EndPan()

        # ── 右键：翻滚 / 多边形模式下结束绘制 ──
        def OnRightButtonDown(self):
            v = self._vtk_viewer
            if v and v.polygon_mode:
                if len(v._poly_points) >= 3:
                    pts = [p.tolist() for p in v._poly_points]
                    v.polygon_finished.emit(pts)
                v.exit_polygon_mode()
                return
            if v and v.place_mode:
                if v._place_preview_pos is not None:
                    v.place_picked.emit(v._place_preview_pos)
                v.exit_place_mode()
                return
            self.StartRotate()

        def OnRightButtonUp(self):
            self.EndRotate()

        # ── 移动 ──
        def OnMouseMove(self):
            v = self._vtk_viewer
            rwi = self.GetInteractor()

            if v and v._wp_editing:
                pos = rwi.GetEventPosition()
                v._update_wp_edit(pos[0], pos[1])
                return

            super().OnMouseMove()


# ─── 3D 可视化组件 ───────────────────────────────────────────
class VTKViewer(QWidget):
    """嵌入 PyQt5 的 VTK 3D 点云/航线可视化组件，支持交互式画框选点"""

    waypoint_edited = pyqtSignal(int, object, object)
    polygon_finished = pyqtSignal(list)
    place_picked = pyqtSignal(object)  # 点击放置模式

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

        # ─── 视角切换按钮（右上角覆盖层）───
        from PyQt5.QtWidgets import QFrame, QButtonGroup
        view_frame = QFrame(self.vtk_widget)
        view_frame.setStyleSheet("QFrame { background: rgba(240,240,238,200); border: 1px solid #ccc; border-radius: 6px; }")
        view_frame.setFixedSize(180, 32)
        view_layout = QHBoxLayout(view_frame)
        view_layout.setContentsMargins(4, 2, 4, 2)
        view_layout.setSpacing(2)

        self._view_btns = QButtonGroup(self)
        views = [("俯", "top"), ("仰", "bottom"), ("正", "front"), ("侧", "side"), ("透", "persp")]
        for i, (label, name) in enumerate(views):
            btn = QPushButton(label)
            btn.setFixedSize(28, 24)
            btn.setStyleSheet("QPushButton { background: #e0e0de; border: 1px solid #bbb; border-radius: 3px; color: #000; font-size: 11px; } QPushButton:checked { background: #4a9eff; color: #fff; }")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, n=name: self._set_view(n))
            view_layout.addWidget(btn)
            self._view_btns.addButton(btn, i)

        self._view_btns.button(0).setChecked(True)
        QTimer.singleShot(100, lambda: view_frame.move(self.vtk_widget.width() - 190, 8))

        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.renderer.SetBackground(0.95, 0.95, 0.93)  # 奶白色背景

        self._actors = []
        self.points_data = None
        self._cloud_actor = None

        # ─── 点击放置模式 ───
        self.place_mode = False

        # ─── 航点编辑状态 ───
        self._wp_editing = False
        self._wp_edit_idx = -1
        self._wp_edit_z = 0.0
        self._wp_edit_offset = None
        self._wp_edit_actor = None
        self._waypoint_actors = []
        self._waypoints_ref = None
        self._safe_distance = 2.0
        self.show_heading = True

        # ─── 多边形选择模式 ───
        self.polygon_mode = False
        self._poly_points = []
        self._poly_markers = []
        self._poly_line_actor = None
        self._poly_click_start = None

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
        self.points_data = points

        MAX_RENDER_POINTS = 5_000_000
        if len(points) > MAX_RENDER_POINTS:
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
    def _compute_forward_headings(waypoints):
        n = len(waypoints)
        if n < 2:
            return [np.array([1.0, 0.0, 0.0])]
        headings = []
        for i in range(n):
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
        if not VTK_AVAILABLE or len(waypoints) == 0:
            return

        to_remove = []
        for i, actor in enumerate(self._actors):
            if actor != self._cloud_actor:
                to_remove.append(i)
        for i in reversed(to_remove):
            self.renderer.RemoveActor(self._actors[i])
            del self._actors[i]

        n = len(waypoints)

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

        # 起飞线：从地面 (0,0,1) 到航线起点
        takeoff_start = [0.0, 0.0, 1.0]
        takeoff_end = waypoints[0]['pos'].tolist()
        takeoff_line = vtkLineSource()
        takeoff_line.SetPoint1(takeoff_start)
        takeoff_line.SetPoint2(takeoff_end)
        takeoff_mapper = vtkPolyDataMapper()
        takeoff_mapper.SetInputConnection(takeoff_line.GetOutputPort())
        takeoff_actor = vtkActor()
        takeoff_actor.SetMapper(takeoff_mapper)
        takeoff_actor.GetProperty().SetColor(0.2, 0.5, 1.0)
        takeoff_actor.GetProperty().SetLineWidth(3)
        takeoff_actor.GetProperty().SetLineStipplePattern(0xF0F0)
        takeoff_actor.GetProperty().SetLineStippleRepeatFactor(1)
        self.renderer.AddActor(takeoff_actor)
        self._actors.append(takeoff_actor)

        # 起飞点标记
        takeoff_sphere = vtkSphereSource()
        takeoff_sphere.SetCenter(takeoff_start)
        takeoff_sphere.SetRadius(0.4)
        takeoff_sphere.Update()
        tm = vtkPolyDataMapper()
        tm.SetInputConnection(takeoff_sphere.GetOutputPort())
        ta = vtkActor()
        ta.SetMapper(tm)
        ta.GetProperty().SetColor(0.2, 0.5, 1.0)
        self.renderer.AddActor(ta)
        self._actors.append(ta)

        self._waypoint_actors = []
        self._waypoints_ref = waypoints
        if n >= 2:
            avg_d = np.mean([np.linalg.norm(waypoints[i+1]['pos'] - waypoints[i]['pos']) for i in range(n-1)])
            label_offset = avg_d * 0.08
        else:
            label_offset = 0.2

        for i, wp in enumerate(waypoints):
            sphere = vtkSphereSource()
            sphere.SetCenter(wp['pos'].tolist())
            sphere.SetRadius(0.3)
            sphere.Update()

            m = vtkPolyDataMapper()
            m.SetInputConnection(sphere.GetOutputPort())
            a = vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetColor(1.0, 0.2, 0.2)
            self.renderer.AddActor(a)
            self._actors.append(a)
            self._waypoint_actors.append(a)

            txt_src = vtkVectorText()
            txt_src.SetText(str(i + 1))
            txt_m = vtkPolyDataMapper()
            txt_m.SetInputConnection(txt_src.GetOutputPort())
            follower = vtkFollower()
            follower.SetMapper(txt_m)
            follower.SetScale(label_offset * 0.6, label_offset * 0.6, label_offset * 0.6)
            follower.SetPosition(wp['pos'][0], wp['pos'][1], wp['pos'][2] + label_offset * 2)
            follower.GetProperty().SetColor(1.0, 1.0, 0.2)
            follower.SetCamera(self.renderer.GetActiveCamera())
            self.renderer.AddActor(follower)
            self._actors.append(follower)

        if getattr(self, 'show_heading', True) and n >= 2:
            dists = [np.linalg.norm(waypoints[i+1]['pos'] - waypoints[i]['pos']) for i in range(n-1)]
            avg_dist = np.mean(dists)
            line_len = np.clip(avg_dist * 0.3, 0.3, 2.0)

            headings = self._compute_forward_headings(waypoints)

            for i, wp in enumerate(waypoints):
                pos = wp['pos']
                fwd = headings[i]
                end = pos + fwd * line_len

                line = vtkLineSource()
                line.SetPoint1(pos.tolist())
                line.SetPoint2(end.tolist())
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(line.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetLineWidth(3)
                is_corner = self._is_corner(waypoints, i)
                if is_corner:
                    actor.GetProperty().SetColor(1.0, 0.9, 0.0)
                else:
                    actor.GetProperty().SetColor(0.0, 0.9, 1.0)
                self.renderer.AddActor(actor)
                self._actors.append(actor)

        self._update_view()
        print(f"[VTK] Route displayed: {n} waypoints")

    def _pick_3d(self, screen_x, screen_y, z_plane=None):
        """屏幕坐标拾取3D点：射线与XY平面求交"""
        ren = self.renderer
        ren.SetDisplayPoint(screen_x, screen_y, 0)
        ren.DisplayToWorld()
        near = np.array(ren.GetWorldPoint()[:3])
        ren.SetDisplayPoint(screen_x, screen_y, 1)
        ren.DisplayToWorld()
        far = np.array(ren.GetWorldPoint()[:3])
        ray_dir = far - near

        if z_plane is None:
            if self.points_data is not None and len(self.points_data) > 0:
                z_plane = self.points_data[:, 2].max()
            else:
                z_plane = 0.0

        if abs(ray_dir[2]) < 1e-10:
            return None
        t = (z_plane - near[2]) / ray_dir[2]
        if t < 0:
            return None
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
        ren.SetDisplayPoint(screen_x, screen_y, 0)
        ren.DisplayToWorld()
        near = np.array(ren.GetWorldPoint()[:3])
        ren.SetDisplayPoint(screen_x, screen_y, 1)
        ren.DisplayToWorld()
        far = np.array(ren.GetWorldPoint()[:3])
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

        print(f"[WP Edit] Selected waypoint #{idx} at ({wp['pos'][0]:.2f}, {wp['pos'][1]:.2f}, {wp['pos'][2]:.2f})")

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
        print(f"[WP Edit] Waypoint #{idx} moved to ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})")

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

    def exit_polygon_mode(self):
        self.polygon_mode = False
        self._poly_points = []
        self._poly_click_start = None
        self._clear_polygon()
        print("[Polygon] Exited polygon mode.")

    # ─── 点击放置模式 ─────────────────────────────────────
    def enter_place_mode(self):
        if self.points_data is None or len(self.points_data) == 0:
            print("[Place] No point cloud loaded")
            return
        self.place_mode = True
        self._place_preview_pos = None
        self._place_preview_actor = None
        print("[Place] 左键选择位置，右键确认，Esc取消")

    def exit_place_mode(self):
        self.place_mode = False
        self._place_preview_pos = None
        self._clear_place_preview()
        print("[Place] Exited place mode.")

    def _update_place_preview(self, pos):
        """左键点击时更新预览位置"""
        self._clear_place_preview()
        self._place_preview_pos = pos

        sphere = vtkSphereSource()
        sphere.SetCenter(pos.tolist())
        sphere.SetRadius(0.5)
        sphere.Update()
        m = vtkPolyDataMapper()
        m.SetInputConnection(sphere.GetOutputPort())
        a = vtkActor()
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

        sphere = vtkSphereSource()
        sphere.SetCenter(pos.tolist())
        sphere.SetRadius(0.3)
        sphere.Update()
        m = vtkPolyDataMapper()
        m.SetInputConnection(sphere.GetOutputPort())
        a = vtkActor()
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

        vtk_pts = vtkPoints()
        for p in self._poly_points:
            vtk_pts.InsertNextPoint(p.tolist())

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

    def _update_view(self):
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

    def _set_view(self, name):
        if not VTK_AVAILABLE:
            return
        cam = self.renderer.GetActiveCamera()
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 1, 0)

        if name == "top":
            cam.SetPosition(0, 0, 100)
        elif name == "bottom":
            cam.SetPosition(0, 0, -100)
            cam.SetViewUp(0, -1, 0)
        elif name == "front":
            cam.SetPosition(0, -100, 0)
        elif name == "side":
            cam.SetPosition(100, 0, 0)
        elif name == "persp":
            cam.SetPosition(50, -80, 60)
            cam.SetViewUp(0, 0, 1)
            self.renderer.ResetCamera()
            cam.Elevation(30)
            cam.Azimuth(-45)
            self.vtk_widget.GetRenderWindow().Render()
            return

        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def _on_global_key_press(self, obj, event):
        key = self.interactor.GetKeyCode()
        cam = self.renderer.GetActiveCamera()

        if key == '1':
            cam.SetPosition(0, 0, 100)
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(0, 1, 0)
            self.renderer.ResetCamera()
            print("[View] Top View")
        elif key == '2':
            cam.SetPosition(0, -100, 0)
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(0, 0, 1)
            self.renderer.ResetCamera()
            print("[View] Front View")
        elif key == '3':
            cam.SetPosition(100, 0, 0)
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(0, 0, 1)
            self.renderer.ResetCamera()
            print("[View] Side View")
        elif key == '4':
            self.renderer.ResetCamera()
            cam.Elevation(30)
            cam.Azimuth(-45)
            print("[View] Perspective")
        elif key == '5':
            cam.SetPosition(0, 0, -100)
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(0, 1, 0)
            self.renderer.ResetCamera()
            print("[View] Bottom View (looking up)")
        elif key == '\x1b':
            if self.polygon_mode:
                self.exit_polygon_mode()
            elif self.place_mode:
                self.exit_place_mode()

        self.vtk_widget.GetRenderWindow().Render()

    def setup_scene(self):
        """初始化场景（坐标轴 + 网格）"""
        if not VTK_AVAILABLE:
            return

        axes = [
            ((0, 0, 0), (8, 0, 0), (1, 0, 0)),
            ((0, 0, 0), (0, 8, 0), (0, 1, 0)),
            ((0, 0, 0), (0, 0, 8), (0, 0, 1)),
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

        grid_pts = vtkPoints()
        grid_cells = vtkCellArray()
        idx = 0
        for i in range(-20, 21, 2):
            grid_pts.InsertNextPoint(i, -20, 0)
            grid_pts.InsertNextPoint(i, 20, 0)
            line = vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)
            line.GetPointIds().SetId(0, idx)
            line.GetPointIds().SetId(1, idx + 1)
            grid_cells.InsertNextCell(line)
            idx += 2
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

        style = FoxgloveInteractorStyle()
        style.set_viewer(self)
        self.interactor.SetInteractorStyle(style)

        self.interactor.AddObserver("KeyPressEvent", self._on_global_key_press)
