"""桥梁巡检航线规划工具 - 主窗口"""

import json
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QSplitter, QButtonGroup, QSlider,
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
        self._polygon_vertices = None

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

        bp.addWidget(QLabel("桥梁名称:"), 0, 0)
        self.edt_bridge_name = QLineEdit("我的桥梁")
        bp.addWidget(self.edt_bridge_name, 0, 1, 1, 3)

        bp.addWidget(QLabel("桥型:"), 1, 0)
        self.cmb_bridge_type = QComboBox()
        self.cmb_bridge_type.addItems(["跨河桥", "跨线桥", "高架桥"])
        bp.addWidget(self.cmb_bridge_type, 1, 1, 1, 3)

        bp.addWidget(QLabel("桥长(m):"), 2, 0)
        self.edt_bridge_len = QLineEdit("100"); bp.addWidget(self.edt_bridge_len, 2, 1)
        bp.addWidget(QLabel("桥宽(m):"), 2, 2)
        self.edt_bridge_wid = QLineEdit("15"); bp.addWidget(self.edt_bridge_wid, 2, 3)

        bp.addWidget(QLabel("净空(m):"), 3, 0)
        self.edt_bridge_clr = QLineEdit("8"); bp.addWidget(self.edt_bridge_clr, 3, 1)
        bp.addWidget(QLabel("跨距(m):"), 3, 2)
        self.edt_bridge_span = QLineEdit("30"); bp.addWidget(self.edt_bridge_span, 3, 3)

        btn_apply_bridge = QPushButton("应用")
        btn_apply_bridge.setStyleSheet("QPushButton { background: #d0d8e8; padding: 6px; } QPushButton:hover { background: #c0c8d8; }")
        btn_apply_bridge.clicked.connect(self._apply_bridge_params)
        bp.addWidget(btn_apply_bridge, 4, 0, 1, 4)

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

        tk_row = QHBoxLayout()
        tk_row.addWidget(QLabel("起飞高度(m):"))
        self.edt_takeoff_z = QLineEdit("1.0")
        self.edt_takeoff_z.setMaximumWidth(60)
        tk_row.addWidget(self.edt_takeoff_z)
        tk_row.addWidget(QLabel("初始偏航角(°):"))
        self.edt_takeoff_yaw = QLineEdit("0")
        self.edt_takeoff_yaw.setMaximumWidth(60)
        tk_row.addWidget(self.edt_takeoff_yaw)
        tk_row.addStretch()
        pk.addLayout(tk_row)

        minz_row = QHBoxLayout()
        minz_row.addWidget(QLabel("最低飞行Z值(m):"))
        self.edt_min_z = QLineEdit("-999")
        self.edt_min_z.setMaximumWidth(60)
        minz_row.addWidget(self.edt_min_z)
        minz_row.addWidget(QLabel("低于此值视为碰撞"))
        minz_row.addStretch()
        pk.addLayout(minz_row)

        btn_apply_safety = QPushButton("应用")
        btn_apply_safety.setStyleSheet("QPushButton { background: #d0d8e8; padding: 4px; } QPushButton:hover { background: #c0c8d8; }")
        btn_apply_safety.clicked.connect(self._apply_safety_settings)
        pk.addWidget(btn_apply_safety)

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

        fl.addWidget(QLabel("高度Z:"), 0, 0)
        self.edt_z = QLineEdit("5"); fl.addWidget(self.edt_z, 0, 1)
        fl.addWidget(QLabel("线间距:"), 0, 2)
        self.edt_spacing = QLineEdit("2"); fl.addWidget(self.edt_spacing, 0, 3)

        fl.addWidget(QLabel("航点距离:"), 1, 0)
        self.edt_wp_spacing = QLineEdit("2"); fl.addWidget(self.edt_wp_spacing, 1, 1)
        fl.addWidget(QLabel("速度(m/s):"), 1, 2)
        self.edt_flat_speed = QLineEdit("3"); fl.addWidget(self.edt_flat_speed, 1, 3)

        fl.addWidget(QLabel("曲度:"), 2, 0)
        self.sld_curvature = QSlider(Qt.Horizontal)
        self.sld_curvature.setRange(0, 100)
        self.sld_curvature.setValue(0)
        fl.addWidget(self.sld_curvature, 2, 1, 1, 2)
        self.lbl_curvature_val = QLabel("0.00")
        self.lbl_curvature_val.setMinimumWidth(30)
        fl.addWidget(self.lbl_curvature_val, 2, 3)
        self.sld_curvature.valueChanged.connect(lambda v: self.lbl_curvature_val.setText(f"{v/100:.2f}"))

        btn_apply_flat = QPushButton("应用")
        btn_apply_flat.setStyleSheet("QPushButton { background: #d0d8e8; padding: 4px; } QPushButton:hover { background: #c0c8d8; }")
        btn_apply_flat.clicked.connect(self._apply_flat_params)
        fl.addWidget(btn_apply_flat, 3, 0, 1, 2)

        self.btn_poly_select = QPushButton("点击放置（右键确认生成）")
        self.btn_poly_select.setStyleSheet("QPushButton { background: #d8e8d8; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c8d8c8; }")
        self.btn_poly_select.clicked.connect(self._start_polygon_select)
        fl.addWidget(self.btn_poly_select, 3, 2, 1, 2)

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

        cl.addWidget(QLabel("起始角度(°):"), 5, 0)
        self.sld_cube_start_angle = QSlider(Qt.Horizontal)
        self.sld_cube_start_angle.setRange(0, 360)
        self.sld_cube_start_angle.setValue(0)
        cl.addWidget(self.sld_cube_start_angle, 5, 1, 1, 2)
        self.lbl_cube_angle_val = QLabel("0°")
        self.lbl_cube_angle_val.setMinimumWidth(30)
        cl.addWidget(self.lbl_cube_angle_val, 5, 3)
        self.sld_cube_start_angle.valueChanged.connect(lambda v: self.lbl_cube_angle_val.setText(f"{v}°"))

        btn_apply_cube = QPushButton("应用")
        btn_apply_cube.setStyleSheet("QPushButton { background: #d0d8e8; padding: 4px; } QPushButton:hover { background: #c0c8d8; }")
        btn_apply_cube.clicked.connect(self._apply_cube_params)
        cl.addWidget(btn_apply_cube, 6, 0, 1, 2)

        self.btn_cube_place = QPushButton("点击放置（右键确认生成）")
        self.btn_cube_place.setStyleSheet("QPushButton { background: #d8e8d8; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c8d8c8; }")
        self.btn_cube_place.clicked.connect(lambda: self._start_place_mode("cube"))
        cl.addWidget(self.btn_cube_place, 6, 2, 1, 2)
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

        cyl.addWidget(QLabel("起始角度(°):"), 5, 0)
        self.sld_cyl_start_angle = QSlider(Qt.Horizontal)
        self.sld_cyl_start_angle.setRange(0, 360)
        self.sld_cyl_start_angle.setValue(0)
        cyl.addWidget(self.sld_cyl_start_angle, 5, 1, 1, 2)
        self.lbl_cyl_angle_val = QLabel("0°")
        self.lbl_cyl_angle_val.setMinimumWidth(30)
        cyl.addWidget(self.lbl_cyl_angle_val, 5, 3)
        self.sld_cyl_start_angle.valueChanged.connect(lambda v: self.lbl_cyl_angle_val.setText(f"{v}°"))

        btn_apply_cyl = QPushButton("应用")
        btn_apply_cyl.setStyleSheet("QPushButton { background: #d0d8e8; padding: 4px; } QPushButton:hover { background: #c0c8d8; }")
        btn_apply_cyl.clicked.connect(self._apply_cyl_params)
        cyl.addWidget(btn_apply_cyl, 6, 0, 1, 2)

        self.btn_cyl_place = QPushButton("点击放置（右键确认生成）")
        self.btn_cyl_place.setStyleSheet("QPushButton { background: #d8e8d8; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c8d8c8; }")
        self.btn_cyl_place.clicked.connect(lambda: self._start_place_mode("cylinder"))
        cyl.addWidget(self.btn_cyl_place, 6, 2, 1, 2)
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
            curvature = self.sld_curvature.value() / 100.0
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
        span = y_half * 2

        def curved_z(y):
            t = (y - y_center) / y_half  # -1 到 1
            return z + curvature * span * (1 - t * t) / 4

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
                # 机头垂直于航线方向，朝扫描区域中心
                if y >= y_center:
                    heading = np.array([0.0, -1.0, 0.0])
                else:
                    heading = np.array([0.0, 1.0, 0.0])
                target = pos + heading
                quat = look_at_quaternion(target, pos)
                self.waypoints.append({
                    'pos': pos,
                    'quat': quat,
                    'speed': speed,
                    'action': 'fly'
                })

            # 下一条扫描线
            y += y_step
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
            start_angle = np.radians(self.sld_cyl_start_angle.value())
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

                # 机头朝向圆柱中心（径向内法线）
                inward = np.array([cx - rx, cy - ry, 0.0])
                inward_norm = np.linalg.norm(inward)
                if inward_norm > 1e-10:
                    heading = inward / inward_norm
                else:
                    heading = np.array([1.0, 0.0, 0.0])
                target = pos + heading
                quat = look_at_quaternion(target, pos)

                self.waypoints.append({
                    'pos': pos,
                    'quat': quat,
                    'speed': speed,
                    'action': 'scan'
                })

        elif route_type == "Z字形":
            num_cols = max(1, int(360 / max(1, astep)))
            num_layers = max(1, int(h / vstep))

            for col in range(num_cols + 1):
                angle = start_angle + (col / num_cols) * 2 * np.pi
                rx = cx + radius * np.cos(angle)
                ry = cy + radius * np.sin(angle)

                # 机头朝向圆柱中心
                inward = np.array([cx - rx, cy - ry, 0.0])
                inward_norm = np.linalg.norm(inward)
                if inward_norm > 1e-10:
                    heading = inward / inward_norm
                else:
                    heading = np.array([1.0, 0.0, 0.0])

                # 偶数列：从下往上；奇数列：从上往下
                if col % 2 == 0:
                    layers_range = range(num_layers + 1)
                else:
                    layers_range = range(num_layers, -1, -1)

                for layer in layers_range:
                    z = cz + layer * vstep
                    pos = np.array([rx, ry, z])
                    target = pos + heading
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
            start_angle = np.radians(self.sld_cube_start_angle.value())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效数字")
            return

        if cstep <= 0 or vstep <= 0 or dz <= 0:
            QMessageBox.warning(self, "输入错误", "步距和高度必须为正数")
            return

        half_x = dx / 2
        half_y = dy / 2

        # 矩形四个角（逆时针），起始角决定从哪个角开始
        cos_a = np.cos(start_angle)
        sin_a = np.sin(start_angle)
        corners_raw = [
            (-half_x, -half_y),
            ( half_x, -half_y),
            ( half_x,  half_y),
            (-half_x,  half_y),
        ]
        corners = []
        for rx, ry in corners_raw:
            lx = rx * cos_a - ry * sin_a
            ly = rx * sin_a + ry * cos_a
            corners.append((cx + lx, cy + ly))

        num_layers = max(1, int(dz / vstep))

        # 每条边独立按步距分布点
        edge_points = []  # [边0的点列表, 边1的点列表, ...]
        for i in range(4):
            c0 = corners[i]
            c1 = corners[(i + 1) % 4]
            ex = c1[0] - c0[0]
            ey = c1[1] - c0[1]
            edge_len = np.sqrt(ex * ex + ey * ey)
            n_pts = max(2, int(edge_len / cstep) + 1)
            pts = []
            for j in range(n_pts):
                ratio = j / (n_pts - 1)
                px = c0[0] + ratio * ex
                py = c0[1] + ratio * ey
                # 机头垂直于边方向，朝矩形内侧
                d_len = np.sqrt(ex * ex + ey * ey)
                if d_len > 1e-10:
                    heading = np.array([-ey / d_len, ex / d_len, 0.0])
                else:
                    heading = np.array([1.0, 0.0, 0.0])
                pts.append((np.array([px, py]), heading))
            edge_points.append(pts)

        self.waypoints = []

        for layer in range(num_layers + 1):
            z = cz + layer * vstep
            reverse = (layer % 2 == 1)

            # 按顺序遍历4条边
            edges_order = range(4) if not reverse else range(3, -1, -1)
            for ei in edges_order:
                pts = edge_points[ei]
                pt_order = range(len(pts)) if not reverse else range(len(pts) - 1, -1, -1)
                for pi in pt_order:
                    # 跳过非最后一层的每条边最后一个点（避免与下一条边起点重复）
                    if pi == (len(pts) - 1 if not reverse else 0) and ei != (3 if not reverse else 0) and layer < num_layers:
                        continue
                    pos_2d, heading = pts[pi]
                    pos = np.array([pos_2d[0], pos_2d[1], z])
                    target = pos + heading
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

        bridge_name = self.edt_bridge_name.text().strip()
        if not bridge_name:
            bridge_name = self.cmb_bridge_type.currentText()
        self._bridge_name = bridge_name
        print(f"[Bridge] Applied: {bridge_name}, L={bridge_len}m, W={bridge_wid}m, Clearance={clearance}m, Span={span}m")
        self.lbl_info.setText(f"桥梁: {bridge_name}, {bridge_len}m x {bridge_wid}m")

    def _apply_safety_settings(self):
        """应用安全设置（起飞高度、初始偏航角）并刷新航线显示"""
        if self.waypoints:
            try:
                self.viewer._takeoff_z = float(self.edt_takeoff_z.text())
            except ValueError:
                self.viewer._takeoff_z = 1.0
            try:
                self.viewer._takeoff_yaw = float(self.edt_takeoff_yaw.text())
            except ValueError:
                self.viewer._takeoff_yaw = 0.0
            self.viewer.add_route(self.waypoints)
            self._check_safety_distance()
        print(f"[Safety] 起飞高度={self.edt_takeoff_z.text()}m, 初始偏航角={self.edt_takeoff_yaw.text()}°")

    def _apply_flat_params(self):
        """应用面状航线参数并重新生成航线"""
        if hasattr(self, '_polygon_vertices') and self._polygon_vertices:
            self.generate_flat_route()
        elif self.points is not None and len(self.points) > 0:
            self.generate_flat_route()
        else:
            QMessageBox.warning(self, "提示", "请先加载点云或多边形选择区域")

    def _apply_cube_params(self):
        """应用立方体航线参数并重新生成航线"""
        if self.waypoints:
            self.generate_cube_route()

    def _apply_cyl_params(self):
        """应用圆柱体航线参数并重新生成航线"""
        if self.waypoints:
            self.generate_cylinder_route()

    def _toggle_heading(self, state):
        self.viewer.show_heading = (state == Qt.Checked)
        if self.waypoints:
            try:
                self.viewer._takeoff_z = float(self.edt_takeoff_z.text())
            except ValueError:
                self.viewer._takeoff_z = 1.0
            try:
                self.viewer._takeoff_yaw = float(self.edt_takeoff_yaw.text())
            except ValueError:
                self.viewer._takeoff_yaw = 0.0
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
        try:
            self.viewer._takeoff_z = float(self.edt_takeoff_z.text())
        except ValueError:
            self.viewer._takeoff_z = 1.0
        try:
            self.viewer._takeoff_yaw = float(self.edt_takeoff_yaw.text())
        except ValueError:
            self.viewer._takeoff_yaw = 0.0
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

        # 最低Z值检查
        low_z_count = 0
        try:
            min_z = float(self.edt_min_z.text())
        except ValueError:
            min_z = -999
        if min_z > -900:
            for i, wp in enumerate(self.waypoints):
                if wp['pos'][2] < min_z:
                    low_z_count += 1
                    if i < len(self.viewer._waypoint_actors):
                        self.viewer._waypoint_actors[i].GetProperty().SetColor(1.0, 0.5, 0.0)

        msgs = [f"航点: {len(self.waypoints)}"]
        if violations:
            msgs.append(f"{len(violations)} 对过近 (<{safe_dist}m)")
        if collision_count:
            msgs.append(f"{collision_count} 个碰撞 (<{collision_dist:.1f}m)")
        if low_z_count:
            msgs.append(f"{low_z_count} 个低于Z={min_z}m")
        self.lbl_info.setText(" | ".join(msgs))
        self.viewer.vtk_widget.GetRenderWindow().Render()

    # ─── 清除航线 ───
    def clear_route(self):
        self.waypoints = []
        self.viewer._clear_polygon()
        self.viewer._clear_place_preview()

        ren = self.viewer.renderer
        cloud = self.viewer._cloud_actor

        # 批量收集非点云 actor，一次性移除
        to_remove = [a for a in self.viewer._actors if a != cloud]
        for a in to_remove:
            ren.RemoveActor(a)
        self.viewer._actors = [cloud] if cloud else []
        self.viewer._waypoint_actors = []
        self.viewer._waypoints_ref = None

        # 恢复坐标轴和网格
        self.viewer._add_scene_axes()
        self.viewer.vtk_widget.GetRenderWindow().Render()
        self.lbl_info.setText("航点: 0")

    # ─── 保存航线（nav_msgs/Path 格式）───
    def save_route(self):
        if not self.waypoints:
            QMessageBox.information(self, "提示", "没有航线可保存")
            return

        from datetime import datetime
        ts = datetime.now().strftime("%y%m%d%H%M")
        default_name = f"{getattr(self, '_bridge_name', '航线')}_{ts}.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存航线", default_name, "JSON 文件 (*.json)"
        )
        if not path:
            return

        try:
            takeoff_z = float(self.edt_takeoff_z.text())
        except ValueError:
            takeoff_z = 1.0
        try:
            takeoff_yaw = float(self.edt_takeoff_yaw.text())
        except ValueError:
            takeoff_yaw = 0.0

        # 起飞偏航角转四元数
        yaw_rad = np.radians(takeoff_yaw)
        takeoff_quat = np.array([np.cos(yaw_rad / 2), 0, 0, np.sin(yaw_rad / 2)])

        def _wp_to_pose(wp):
            q = wp['quat']
            return {
                "header": {"stamp": {"sec": 0, "nsec": 0}, "frame_id": "map"},
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
            }

        poses = []
        # 第1个点：原点 (0,0,0)，方向为起飞偏航角
        poses.append({
            "header": {"stamp": {"sec": 0, "nsec": 0}, "frame_id": "map"},
            "pose": {
                "position": {"x": 0, "y": 0, "z": 0},
                "orientation": {
                    "x": round(float(takeoff_quat[1]), 6),
                    "y": round(float(takeoff_quat[2]), 6),
                    "z": round(float(takeoff_quat[3]), 6),
                    "w": round(float(takeoff_quat[0]), 6)
                }
            }
        })
        for wp in self.waypoints:
            poses.append(_wp_to_pose(wp))

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
                all_poses = data['poses']
                start_idx = 0

                # 检测格式：第1个点是(0,0,0)原点
                if len(all_poses) >= 2:
                    p0 = all_poses[0]['pose']['position']
                    if abs(p0['x']) < 1e-6 and abs(p0['y']) < 1e-6 and abs(p0['z']) < 1e-6:
                        o0 = all_poses[0]['pose']['orientation']
                        qw, qx, qy = o0['w'], o0['x'], o0['y']
                        yaw_rad = 2 * np.arctan2(o0['z'], qw)
                        self.edt_takeoff_yaw.setText(f"{np.degrees(yaw_rad):.1f}")
                        # 旧格式：第2个点也是(0,0,z)起飞点
                        p1 = all_poses[1]['pose']['position']
                        if abs(p1['x']) < 1e-6 and abs(p1['y']) < 1e-6:
                            self.edt_takeoff_z.setText(f"{p1['z']:.1f}")
                            start_idx = 2
                        else:
                            start_idx = 1

                for ps in all_poses[start_idx:]:
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
