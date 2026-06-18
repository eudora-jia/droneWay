"""桥梁巡检航线规划工具 - 主窗口"""

import json
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QSplitter, QButtonGroup,
    QProgressBar, QCheckBox, QGridLayout, QScrollArea, QTabWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase

from pcd_parser import parse_pcd
from quaternion_utils import look_at_quaternion
from vtk_viewer import VTKViewer


class MainWindow(QMainWindow):
    """桥梁巡检航线规划工具 - 主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("桥梁巡检无人机航线规划工具")
        self.resize(1400, 900)

        self.points = None
        self.waypoints = []
        self._kdtree = None
        self._kdtree_points_id = None

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

        self.viewer.waypoint_edited.connect(self._on_waypoint_edited)
        self.viewer.polygon_finished.connect(self._on_polygon_finished)
        self.viewer.place_picked.connect(self._on_place_picked)
        self._place_target = None  # "cube" or "cylinder"

        # ─── 右侧控制面板 ───
        ctrl = QWidget()
        ctrl.setMinimumWidth(320)
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setSpacing(6)

        # -- 模式切换 --
        grp_mode = QGroupBox("工作模式")
        mode_layout = QHBoxLayout(grp_mode)
        self._mode_group = QButtonGroup(self)
        self.btn_mode_preview = QPushButton("预览模式")
        self.btn_mode_preview.setCheckable(True)
        self.btn_mode_preview.setChecked(True)
        self.btn_mode_route = QPushButton("航线模式")
        self.btn_mode_route.setCheckable(True)
        self._mode_group.addButton(self.btn_mode_preview)
        self._mode_group.addButton(self.btn_mode_route)
        mode_layout.addWidget(self.btn_mode_preview)
        mode_layout.addWidget(self.btn_mode_route)
        ctrl_layout.addWidget(grp_mode)

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

        self._route_widgets = []

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
        btn_apply_bridge.setStyleSheet("QPushButton { background: #d0d8e8; padding: 6px; } QPushButton:hover { background: #c0c8d8; }")
        btn_apply_bridge.clicked.connect(self._apply_bridge_params)
        bp.addWidget(btn_apply_bridge, 3, 0, 1, 4)

        ctrl_layout.addWidget(grp_bridge)
        self._route_widgets.append(grp_bridge)

        # -- 安全距离 --
        grp_pick = QGroupBox("安全设置")
        pk = QVBoxLayout(grp_pick)
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
        self._route_widgets.append(grp_pick)

        # ─── 航线类型 Tab ───
        route_tabs = QTabWidget()
        route_tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #ccc; } QTabBar::tab { background: #e0e0de; padding: 6px 12px; color: #000; } QTabBar::tab:selected { background: #fff; }")

        # -- Tab 1: 面状航线 --
        tab_flat = QWidget()
        fl = QGridLayout(tab_flat)
        fl.setSpacing(4)

        self.btn_poly_select = QPushButton("多边形选择区域")
        self.btn_poly_select.setStyleSheet("QPushButton { background: #d8e8d8; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c8d8c8; }")
        self.btn_poly_select.clicked.connect(self._start_polygon_select)
        fl.addWidget(self.btn_poly_select, 0, 0, 1, 4)

        fl.addWidget(QLabel("高度Z:"), 0, 0)
        self.edt_z = QLineEdit("5"); fl.addWidget(self.edt_z, 0, 1)
        fl.addWidget(QLabel("线间距:"), 0, 2)
        self.edt_spacing = QLineEdit("2"); fl.addWidget(self.edt_spacing, 0, 3)

        fl.addWidget(QLabel("航点距离:"), 1, 0)
        self.edt_wp_spacing = QLineEdit("2"); fl.addWidget(self.edt_wp_spacing, 1, 1)
        fl.addWidget(QLabel("速度(m/s):"), 1, 2)
        self.edt_flat_speed = QLineEdit("3"); fl.addWidget(self.edt_flat_speed, 1, 3)

        fl.addWidget(QLabel("曲度:"), 2, 0)
        self.edt_curvature = QLineEdit("0"); fl.addWidget(self.edt_curvature, 2, 1)

        self.btn_flat = QPushButton("生成面状航线")
        fl.addWidget(self.btn_flat, 3, 0, 1, 4)
        route_tabs.addTab(tab_flat, "面状航线")

        # -- Tab 2: 立方体航线 --
        tab_cube = QWidget()
        cl = QGridLayout(tab_cube)
        cl.setSpacing(4)

        cl.addWidget(QLabel("底面中心(x,y,z):"), 0, 0)
        self.edt_cx = QLineEdit("0"); cl.addWidget(self.edt_cx, 0, 1)
        self.edt_cy = QLineEdit("0"); cl.addWidget(self.edt_cy, 0, 2)
        self.edt_cz = QLineEdit("0"); cl.addWidget(self.edt_cz, 0, 3)

        cl.addWidget(QLabel("长(X):"), 1, 0)
        self.edt_dx = QLineEdit("4"); cl.addWidget(self.edt_dx, 1, 1)
        cl.addWidget(QLabel("宽(Y):"), 1, 2)
        self.edt_dy = QLineEdit("4"); cl.addWidget(self.edt_dy, 1, 3)

        cl.addWidget(QLabel("高(Z):"), 2, 0)
        self.edt_dz = QLineEdit("8"); cl.addWidget(self.edt_dz, 2, 1)
        cl.addWidget(QLabel("离柱距离:"), 2, 2)
        self.edt_dist = QLineEdit("3"); cl.addWidget(self.edt_dist, 2, 3)

        cl.addWidget(QLabel("水平步距:"), 3, 0)
        self.edt_cstep = QLineEdit("2"); cl.addWidget(self.edt_cstep, 3, 1)
        cl.addWidget(QLabel("垂直步距:"), 3, 2)
        self.edt_vstep = QLineEdit("2"); cl.addWidget(self.edt_vstep, 3, 3)

        cl.addWidget(QLabel("速度:"), 4, 0)
        self.edt_cspeed = QLineEdit("2"); cl.addWidget(self.edt_cspeed, 4, 1)
        cl.addWidget(QLabel("起始角度(°):"), 4, 2)
        self.edt_cube_start_angle = QLineEdit("0"); cl.addWidget(self.edt_cube_start_angle, 4, 3)

        cube_btn_row = QHBoxLayout()
        self.btn_cube_place = QPushButton("点击放置")
        self.btn_cube_place.setStyleSheet("QPushButton { background: #d8e8d8; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c8d8c8; }")
        self.btn_cube_place.clicked.connect(lambda: self._start_place_mode("cube"))
        cube_btn_row.addWidget(self.btn_cube_place)
        self.btn_cube = QPushButton("生成立方体航线")
        cube_btn_row.addWidget(self.btn_cube)
        cl.addLayout(cube_btn_row, 5, 0, 1, 4)
        route_tabs.addTab(tab_cube, "立方体航线")

        # -- Tab 3: 圆柱体航线 --
        tab_cyl = QWidget()
        cyl = QGridLayout(tab_cyl)
        cyl.setSpacing(4)

        cyl.addWidget(QLabel("底面中心(x,y,z):"), 0, 0)
        self.edt_cyl_cx = QLineEdit("0"); cyl.addWidget(self.edt_cyl_cx, 0, 1)
        self.edt_cyl_cy = QLineEdit("0"); cyl.addWidget(self.edt_cyl_cy, 0, 2)
        self.edt_cyl_cz = QLineEdit("0"); cyl.addWidget(self.edt_cyl_cz, 0, 3)

        cyl.addWidget(QLabel("直径:"), 1, 0)
        self.edt_cyl_diam = QLineEdit("4"); cyl.addWidget(self.edt_cyl_diam, 1, 1)
        cyl.addWidget(QLabel("高(Z):"), 1, 2)
        self.edt_cyl_h = QLineEdit("8"); cyl.addWidget(self.edt_cyl_h, 1, 3)

        cyl.addWidget(QLabel("离柱距离:"), 2, 0)
        self.edt_cyl_dist = QLineEdit("3"); cyl.addWidget(self.edt_cyl_dist, 2, 1)
        cyl.addWidget(QLabel("水平步距(°):"), 2, 2)
        self.edt_cyl_astep = QLineEdit("15"); cyl.addWidget(self.edt_cyl_astep, 2, 3)

        cyl.addWidget(QLabel("垂直步距:"), 3, 0)
        self.edt_cyl_vstep = QLineEdit("2"); cyl.addWidget(self.edt_cyl_vstep, 3, 1)
        cyl.addWidget(QLabel("速度:"), 3, 2)
        self.edt_cyl_speed = QLineEdit("2"); cyl.addWidget(self.edt_cyl_speed, 3, 3)

        cyl.addWidget(QLabel("路径:"), 4, 0)
        self.cbo_cyl_type = QComboBox()
        self.cbo_cyl_type.addItems(["螺旋线", "Z字形"])
        cyl.addWidget(self.cbo_cyl_type, 4, 1)
        cyl.addWidget(QLabel("起始角度(°):"), 4, 2)
        self.edt_cyl_start_angle = QLineEdit("0"); cyl.addWidget(self.edt_cyl_start_angle, 4, 3)

        cyl_btn_row = QHBoxLayout()
        self.btn_cyl_place = QPushButton("点击放置")
        self.btn_cyl_place.setStyleSheet("QPushButton { background: #d8e8d8; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c8d8c8; }")
        self.btn_cyl_place.clicked.connect(lambda: self._start_place_mode("cylinder"))
        cyl_btn_row.addWidget(self.btn_cyl_place)
        self.btn_cylinder = QPushButton("生成圆柱体航线")
        cyl_btn_row.addWidget(self.btn_cylinder)
        cyl.addLayout(cyl_btn_row, 5, 0, 1, 4)
        route_tabs.addTab(tab_cyl, "圆柱体航线")

        ctrl_layout.addWidget(route_tabs)
        self._route_widgets.append(route_tabs)

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

        self.chk_show_heading = QCheckBox("显示机头方向")
        self.chk_show_heading.setChecked(True)
        self.chk_show_heading.stateChanged.connect(self._toggle_heading)
        rl.addWidget(self.chk_show_heading)

        ctrl_layout.addWidget(grp_route)
        self._route_widgets.append(grp_route)

        # -- 快捷键提示 --
        lbl_help = QLabel("快捷键: 1=俯视 2=正视 3=侧视 4=透视 5=仰视  Esc=取消多边形")
        lbl_help.setStyleSheet("color: #666; font-size: 10px; padding: 4px;")
        lbl_help.setWordWrap(True)
        ctrl_layout.addWidget(lbl_help)

        ctrl_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(ctrl)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(400)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter)

        # -- 信号连接 --
        self.btn_load.clicked.connect(self.load_point_cloud)
        self.btn_flat.clicked.connect(self.generate_flat_route)
        self.btn_cube.clicked.connect(self.generate_cube_route)
        self.btn_cylinder.clicked.connect(self.generate_cylinder_route)
        self.btn_clear.clicked.connect(self.clear_route)
        self.btn_save.clicked.connect(self.save_route)
        self.btn_load_route.clicked.connect(self.load_route)

        self.btn_mode_preview.clicked.connect(lambda: self._switch_mode("preview"))
        self.btn_mode_route.clicked.connect(lambda: self._switch_mode("route"))

        self._switch_mode("preview")

    def _switch_mode(self, mode):
        is_route = (mode == "route")
        for w in self._route_widgets:
            w.setVisible(is_route)
        self.btn_mode_preview.setChecked(not is_route)
        self.btn_mode_route.setChecked(is_route)
        if is_route:
            self.btn_mode_preview.setStyleSheet("")
            self.btn_mode_route.setStyleSheet("QPushButton { background: #4a9eff; font-weight: bold; color: #fff; }")
        else:
            self.btn_mode_preview.setStyleSheet("QPushButton { background: #4a9eff; font-weight: bold; color: #fff; }")
            self.btn_mode_route.setStyleSheet("")

    def _on_pillar_type_changed(self, idx):
        if idx == 1:
            self.edt_dy.setText(self.edt_dx.text())

    def _apply_style(self):
        available = QFontDatabase().families()
        cn_font = "Microsoft YaHei"
        for candidate in ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "PingFang SC"]:
            if candidate in available:
                cn_font = candidate
                break

        self.setStyleSheet(f"""
            QMainWindow {{ background: #f0f0ee; }}
            QWidget {{ color: #000; font-family: "{cn_font}", "Segoe UI", Arial; font-size: 12px; }}
            QGroupBox {{
                border: 1px solid #ccc; border-radius: 6px;
                margin-top: 8px; padding: 10px 8px; font-weight: bold; color: #000;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #000; }}
            QLineEdit {{
                background: #fff; border: 1px solid #bbb; border-radius: 3px;
                padding: 3px 6px; color: #000;
            }}
            QLineEdit:focus {{ border-color: #4a9eff; }}
            QPushButton {{
                background: #e0e0de; border: 1px solid #bbb; border-radius: 4px;
                padding: 6px 14px; color: #000; min-height: 24px;
            }}
            QPushButton:hover {{ background: #d0d0ce; }}
            QPushButton:pressed {{ background: #c0c0be; }}
            QComboBox {{
                background: #fff; border: 1px solid #bbb; border-radius: 3px;
                padding: 3px 6px; color: #000;
            }}
            QComboBox QAbstractItemView {{ background: #fff; color: #000; selection-background-color: #4a9eff; }}
        """)

    # ─── 多边形选择 ──────────────────────────────────────────
    def _start_polygon_select(self):
        self.viewer.enter_polygon_mode()

    def _on_polygon_finished(self, pts):
        poly = np.array(pts)
        mn = poly.min(axis=0)
        mx = poly.max(axis=0)
        if self.points is not None and len(self.points) > 0:
            z_val = self.points[:, 2].max() + 3
        else:
            z_val = mx[2] + 3
        self.edt_z.setText(f"{z_val:.1f}")
        self._polygon_vertices = pts
        self.generate_flat_route()
        print(f"[Polygon] {len(pts)} vertices, bbox=[{mn[0]:.1f},{mx[0]:.1f}]x[{mn[1]:.1f},{mx[1]:.1f}]")

    # ─── 点击放置模式 ──────────────────────────────────────
    def _start_place_mode(self, target):
        self._place_target = target
        self.viewer.enter_place_mode()

    def _on_place_picked(self, pos):
        if self._place_target == "cube":
            self.edt_cx.setText(f"{pos[0]:.1f}")
            self.edt_cy.setText(f"{pos[1]:.1f}")
            self.edt_cz.setText(f"{pos[2]:.1f}")
            self.generate_cube_route()
            self.viewer._set_view("persp")
        elif self._place_target == "cylinder":
            self.edt_cyl_cx.setText(f"{pos[0]:.1f}")
            self.edt_cyl_cy.setText(f"{pos[1]:.1f}")
            self.edt_cyl_cz.setText(f"{pos[2]:.1f}")
            self.generate_cylinder_route()
            self.viewer._set_view("persp")
        self._place_target = None

    # ─── 加载点云 ───
    def load_point_cloud(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 PCD 点云文件", "", "PCD 文件 (*.pcd);;所有文件 (*)"
        )
        if not path:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
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

            if n > 0:
                mn = self.points.min(axis=0)
                mx = self.points.max(axis=0)

                self.edt_z.setText(f"{mx[2] + 3:.1f}")

                center = (mn + mx) / 2
                self.edt_cx.setText(f"{center[0]:.1f}")
                self.edt_cy.setText(f"{center[1]:.1f}")
                self.edt_cz.setText(f"{mn[2]:.1f}")
                self.edt_dz.setText(f"{mx[2] - mn[2]:.1f}")

                self.edt_cyl_cx.setText(f"{center[0]:.1f}")
                self.edt_cyl_cy.setText(f"{center[1]:.1f}")
                self.edt_cyl_cz.setText(f"{mn[2]:.1f}")
                self.edt_cyl_h.setText(f"{mx[2] - mn[2]:.1f}")

            self.progress_bar.setValue(100)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载点云失败:\n{str(e)}")
        finally:
            self.progress_bar.setVisible(False)

    # ─── 生成平面航线 ───
    def generate_flat_route(self):
        try:
            z = float(self.edt_z.text())
            spacing = float(self.edt_spacing.text())
            wp_spacing = float(self.edt_wp_spacing.text())
            speed = float(self.edt_flat_speed.text())
            curvature = float(self.edt_curvature.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效数字")
            return

        if spacing <= 0 or wp_spacing <= 0:
            QMessageBox.warning(self, "输入错误", "间距必须为正数")
            return

        if hasattr(self, '_polygon_vertices') and self._polygon_vertices:
            poly = np.array(self._polygon_vertices)
            xmin, ymin = poly[:, 0].min(), poly[:, 1].min()
            xmax, ymax = poly[:, 0].max(), poly[:, 1].max()
            use_polygon = True
        elif self.points is not None and len(self.points) > 0:
            mn = self.points.min(axis=0)
            mx = self.points.max(axis=0)
            xmin, ymin = mn[0], mn[1]
            xmax, ymax = mx[0], mx[1]
            use_polygon = False
        else:
            QMessageBox.warning(self, "提示", "请先加载点云或多边形选择区域")
            return

        if xmin >= xmax or ymin >= ymax:
            QMessageBox.warning(self, "输入错误", "区域范围无效")
            return

        y_center = (ymin + ymax) / 2
        y_half = (ymax - ymin) / 2 if ymax != ymin else 1.0

        def curved_z(y):
            t = (y - y_center) / y_half
            return z + curvature * t * t

        def point_in_polygon(x, y, polygon_xy):
            n = len(polygon_xy)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = polygon_xy[i]
                xj, yj = polygon_xy[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside

        poly_xy = None
        if use_polygon:
            poly_xy = [(p[0], p[1]) for p in self._polygon_vertices]

        self.waypoints = []
        # 起始点选择离原点最近的角
        corners = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
        best = min(corners, key=lambda c: c[0]**2 + c[1]**2)
        start_y = best[1]
        start_x = best[0]
        # 从起始角开始：Y方向朝另一端扫描，X方向朝另一端扫描
        direction = 1 if start_x == xmin else -1
        y = start_y
        y_step = spacing if start_y == ymin else -spacing

        while (y_step > 0 and y <= ymax) or (y_step < 0 and y >= ymin):
            if direction == 1:
                x_start, x_end = xmin, xmax
            else:
                x_start, x_end = xmax, xmin

            if use_polygon and poly_xy:
                xs = np.linspace(xmin, xmax, max(100, int((xmax - xmin) / 0.5)))
                inside_xs = [x for x in xs if point_in_polygon(x, y, poly_xy)]
                if not inside_xs:
                    y += y_step
                    continue
                if direction == 1:
                    x_start, x_end = inside_xs[0], inside_xs[-1]
                else:
                    x_start, x_end = inside_xs[-1], inside_xs[0]

            # 沿扫描线均匀分布多个航点
            line_len = abs(x_end - x_start)
            n_pts = max(2, int(line_len / wp_spacing) + 1)
            z_line = curved_z(y)

            for j in range(n_pts):
                x = x_start + (x_end - x_start) * j / (n_pts - 1)
                pos = np.array([x, y, z_line])
                # 机头朝向飞行方向（沿扫描线X方向）
                if j < n_pts - 1:
                    x_next = x_start + (x_end - x_start) * (j + 1) / (n_pts - 1)
                    target = np.array([x_next, y, z_line])
                else:
                    target = np.array([x_end, y, z_line])
                quat = look_at_quaternion(target, pos)
                self.waypoints.append({
                    'pos': pos,
                    'quat': quat,
                    'speed': speed,
                    'action': 'fly'
                })

            # 转折到下一条扫描线
            y += y_step
            has_next = (y_step > 0 and y <= ymax) or (y_step < 0 and y >= ymin)
            if has_next:
                z_next = curved_z(y)
                next_x = xmin if direction == 1 else xmax
                quat = look_at_quaternion(
                    np.array([next_x, y, z_next]),
                    np.array([x_end, y, z_next])
                )
                self.waypoints.append({
                    'pos': np.array([x_end, y, z_next]),
                    'quat': quat,
                    'speed': speed,
                    'action': 'fly'
                })
            direction *= -1

        if not self.waypoints:
            QMessageBox.warning(self, "提示", "多边形区域内无有效航点")
            return

        self._display_route()

    # ─── 生成圆柱体航线 ───
    def generate_cylinder_route(self):
        try:
            cx = float(self.edt_cyl_cx.text())
            cy = float(self.edt_cyl_cy.text())
            cz = float(self.edt_cyl_cz.text())
            diam = float(self.edt_cyl_diam.text())
            h = float(self.edt_cyl_h.text())
            dist = float(self.edt_cyl_dist.text())
            astep = float(self.edt_cyl_astep.text())
            vstep = float(self.edt_cyl_vstep.text())
            speed = float(self.edt_cyl_speed.text())
            start_angle = np.radians(float(self.edt_cyl_start_angle.text()))
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效数字")
            return

        if diam <= 0 or h <= 0 or vstep <= 0:
            QMessageBox.warning(self, "输入错误", "直径、高度和步距必须为正数")
            return

        radius = diam / 2 + dist
        route_type = self.cbo_cyl_type.currentText()
        self.waypoints = []

        if route_type == "螺旋线":
            num_turns = max(1, int(h / vstep))
            num_pts_per_turn = max(8, int(360 / max(1, astep)))
            total_pts = num_turns * num_pts_per_turn

            for i in range(total_pts + 1):
                t = i / total_pts
                angle = start_angle + t * num_turns * 2 * np.pi
                z = cz + t * h

                rx = cx + radius * np.cos(angle)
                ry = cy + radius * np.sin(angle)
                pos = np.array([rx, ry, z])

                tangent = np.array([-np.sin(angle), np.cos(angle), 0])
                target = pos + tangent
                quat = look_at_quaternion(target, pos)

                self.waypoints.append({
                    'pos': pos,
                    'quat': quat,
                    'speed': speed,
                    'action': 'scan'
                })

        elif route_type == "Z字形":
            num_layers = max(1, int(h / vstep))
            num_cols = max(1, int(360 / max(1, astep)))

            for layer in range(num_layers + 1):
                z = cz + layer * vstep
                end_col = num_cols + 1 if layer == num_layers else num_cols
                for col in range(end_col):
                    if layer % 2 == 0:
                        angle = start_angle + (col / num_cols) * 2 * np.pi
                    else:
                        angle = start_angle + (1 - col / num_cols) * 2 * np.pi

                    rx = cx + radius * np.cos(angle)
                    ry = cy + radius * np.sin(angle)
                    pos = np.array([rx, ry, z])

                    # 机头沿圆弧切线方向
                    if layer % 2 == 0:
                        tangent = np.array([-np.sin(angle), np.cos(angle), 0])
                    else:
                        tangent = np.array([np.sin(angle), -np.cos(angle), 0])
                    target = pos + tangent
                    quat = look_at_quaternion(target, pos)

                    self.waypoints.append({
                        'pos': pos,
                        'quat': quat,
                        'speed': speed,
                        'action': 'scan'
                    })

        self._display_route()
        print(f"[Cylinder] Generated {len(self.waypoints)} waypoints ({route_type})")

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
            start_angle = np.radians(float(self.edt_cube_start_angle.text()))
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效数字")
            return

        if cstep <= 0 or vstep <= 0 or dz <= 0:
            QMessageBox.warning(self, "输入错误", "步距和高度必须为正数")
            return

        half_x = dx / 2
        half_y = dy / 2
        pillars = [
            {'cx': cx - half_x, 'cy': cy, 'face': '-x'},
            {'cx': cx + half_x, 'cy': cy, 'face': '+x'},
            {'cx': cx, 'cy': cy - half_y, 'face': '-y'},
            {'cx': cx, 'cy': cy + half_y, 'face': '+y'},
        ]

        self.waypoints = []

        for pillar in pillars:
            px, py = pillar['cx'], pillar['cy']

            num_layers = max(1, int(dz / vstep))
            num_cols = max(1, int(360 / max(1, cstep)))

            for layer in range(num_layers + 1):
                z = cz + layer * vstep
                end_col = num_cols + 1 if layer == num_layers else num_cols
                for col in range(end_col):
                    if layer % 2 == 0:
                        angle = start_angle + (col / num_cols) * 2 * np.pi
                    else:
                        angle = start_angle + (1 - col / num_cols) * 2 * np.pi

                    rx = px + dist * np.cos(angle)
                    ry = py + dist * np.sin(angle)
                    pos = np.array([rx, ry, z])

                    # 机头沿圆弧切线方向
                    if layer % 2 == 0:
                        tangent = np.array([-np.sin(angle), np.cos(angle), 0])
                    else:
                        tangent = np.array([np.sin(angle), -np.cos(angle), 0])
                    target = pos + tangent
                    quat = look_at_quaternion(target, pos)

                    self.waypoints.append({
                        'pos': pos,
                        'quat': quat,
                        'speed': speed,
                        'action': 'scan'
                    })

        self._display_route()

    def _apply_bridge_params(self):
        try:
            bridge_len = float(self.edt_bridge_len.text())
            bridge_wid = float(self.edt_bridge_wid.text())
            clearance = float(self.edt_bridge_clr.text())
            span = float(self.edt_bridge_span.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的桥梁参数")
            return

        bridge_type = self.cmb_bridge_type.currentIndex()

        if bridge_type == 0:
            z_offset = 3.0
        elif bridge_type == 1:
            z_offset = 5.0
        else:
            z_offset = 4.0

        if self.points is not None and len(self.points) > 0:
            center = (self.points.min(axis=0) + self.points.max(axis=0)) / 2
            cx, cy = center[0], center[1]
        else:
            cx, cy = 0.0, 0.0

        if self.points is not None and len(self.points) > 0:
            z_bottom = self.points[:, 2].max() - z_offset
        else:
            z_bottom = clearance
        self.edt_z.setText(f"{z_bottom:.1f}")

        self.edt_cx.setText(f"{cx:.1f}")
        self.edt_cy.setText(f"{cy:.1f}")
        self.edt_cz.setText(f"{clearance:.1f}")
        self.edt_dx.setText(f"{bridge_wid * 0.3:.1f}")
        self.edt_dy.setText(f"{bridge_wid * 0.3:.1f}")
        self.edt_dz.setText(f"{clearance:.1f}")

        self.edt_cyl_cx.setText(f"{cx:.1f}")
        self.edt_cyl_cy.setText(f"{cy:.1f}")
        self.edt_cyl_cz.setText(f"{clearance:.1f}")
        self.edt_cyl_diam.setText(f"{bridge_wid * 0.3:.1f}")
        self.edt_cyl_h.setText(f"{clearance:.1f}")

        spacing = max(span / 5, 2.0)
        self.edt_spacing.setText(f"{spacing:.1f}")

        bridge_name = self.cmb_bridge_type.currentText()
        print(f"[Bridge] Applied: {bridge_name}, L={bridge_len}m, W={bridge_wid}m, Clearance={clearance}m, Span={span}m")
        self.lbl_info.setText(f"桥梁: {bridge_name}, {bridge_len}m x {bridge_wid}m")

    def _toggle_heading(self, state):
        self.viewer.show_heading = (state == Qt.Checked)
        if self.waypoints:
            self.viewer.add_route(self.waypoints)
            self._check_safety_distance()

    def _on_safe_dist_changed(self, text):
        try:
            val = float(text)
            if val > 0:
                self.viewer._safe_distance = val
                if self.waypoints:
                    self._check_safety_distance()
        except ValueError:
            pass

    def _display_route(self):
        self.viewer.add_route(self.waypoints)
        self.lbl_info.setText(f"航点: {len(self.waypoints)}")
        self._check_safety_distance()

    def _on_waypoint_edited(self, idx, new_pos, new_quat):
        if idx < len(self.waypoints):
            self.waypoints[idx]['pos'] = new_pos
            if new_quat is not None:
                self.waypoints[idx]['quat'] = new_quat
            self.viewer.add_route(self.waypoints)
            self._check_safety_distance()

    def _get_kdtree(self):
        if self.points is None or len(self.points) == 0:
            return None
        pts_id = id(self.points)
        if self._kdtree is not None and self._kdtree_points_id == pts_id:
            return self._kdtree
        from scipy.spatial import cKDTree
        self._kdtree = cKDTree(self.points)
        self._kdtree_points_id = pts_id
        return self._kdtree

    def _check_safety_distance(self):
        if len(self.waypoints) < 2:
            return
        safe_dist = self.viewer._safe_distance
        violations = []
        for i in range(len(self.waypoints) - 1):
            d = np.linalg.norm(self.waypoints[i+1]['pos'] - self.waypoints[i]['pos'])
            if d < safe_dist:
                violations.append((i, i+1, d))

        for i, j, d in violations:
            if i < len(self.viewer._waypoint_actors):
                self.viewer._waypoint_actors[i].GetProperty().SetColor(1.0, 0.0, 0.0)
            if j < len(self.viewer._waypoint_actors):
                self.viewer._waypoint_actors[j].GetProperty().SetColor(1.0, 0.0, 0.0)

        collision_count = 0
        collision_dist = safe_dist * 0.5
        tree = self._get_kdtree()
        if tree is not None:
            wp_positions = np.array([wp['pos'] for wp in self.waypoints])
            dists, _ = tree.query(wp_positions)
            for i, dist in enumerate(dists):
                if dist < collision_dist:
                    collision_count += 1
                    if i < len(self.viewer._waypoint_actors):
                        self.viewer._waypoint_actors[i].GetProperty().SetColor(1.0, 0.0, 1.0)

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
        self.viewer._clear_polygon()
        self.viewer._clear_place_preview()
        # 只清除航线 actor，保留点云
        to_remove = []
        for i, actor in enumerate(self.viewer._actors):
            if actor != self.viewer._cloud_actor:
                to_remove.append(i)
        for i in reversed(to_remove):
            self.viewer.renderer.RemoveActor(self.viewer._actors[i])
            del self.viewer._actors[i]
        self.viewer._waypoint_actors = []
        self.viewer._waypoints_ref = None
        self.viewer.vtk_widget.GetRenderWindow().Render()
        self.lbl_info.setText("航点: 0")

    # ─── 保存航线（nav_msgs/Path 格式）───
    def save_route(self):
        if not self.waypoints:
            QMessageBox.information(self, "提示", "没有航线可保存")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存航线", "", "JSON 文件 (*.json)"
        )
        if not path:
            return

        poses = []
        for wp in self.waypoints:
            q = wp['quat']  # 内部存储: (w, x, y, z)
            poses.append({
                "header": {
                    "stamp": {"sec": 0, "nsec": 0},
                    "frame_id": "map"
                },
                "pose": {
                    "position": {
                        "x": round(float(wp['pos'][0]), 4),
                        "y": round(float(wp['pos'][1]), 4),
                        "z": round(float(wp['pos'][2]), 4)
                    },
                    "orientation": {
                        "x": round(float(q[1]), 6),
                        "y": round(float(q[2]), 6),
                        "z": round(float(q[3]), 6),
                        "w": round(float(q[0]), 6)
                    }
                }
            })

        data = {
            "header": {
                "stamp": {"sec": 0, "nsec": 0},
                "frame_id": "map"
            },
            "poses": poses,
            "bridge": {
                "type": self.cmb_bridge_type.currentText(),
                "type_index": self.cmb_bridge_type.currentIndex(),
                "length_m": self.edt_bridge_len.text(),
                "width_m": self.edt_bridge_wid.text(),
                "clearance_m": self.edt_bridge_clr.text(),
                "span_m": self.edt_bridge_span.text()
            }
        }

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "已保存", f"航线已保存到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")

    # ─── 加载航线（兼容 nav_msgs/Path 和旧格式）───
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

            # nav_msgs/Path 格式
            if 'poses' in data:
                for ps in data['poses']:
                    pose = ps['pose']
                    pos = pose['position']
                    ori = pose['orientation']  # ROS: (x, y, z, w)
                    self.waypoints.append({
                        'pos': np.array([pos['x'], pos['y'], pos['z']], dtype=np.float64),
                        'quat': np.array([ori['w'], ori['x'], ori['y'], ori['z']], dtype=np.float64),
                        'speed': 2.0,
                        'action': 'fly'
                    })
            # 旧格式
            elif 'waypoints' in data:
                for wp in data['waypoints']:
                    pos = wp['position']
                    quat = wp['quaternion']  # 旧格式: (w, x, y, z)
                    self.waypoints.append({
                        'pos': np.array([pos['x'], pos['y'], pos['z']], dtype=np.float64),
                        'quat': np.array([quat['w'], quat['x'], quat['y'], quat['z']], dtype=np.float64),
                        'speed': wp.get('speed', 2.0),
                        'action': wp.get('action', 'fly')
                    })

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
