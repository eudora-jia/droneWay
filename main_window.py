"""桥梁巡检航线规划工具 - 主窗口"""

import json
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QButtonGroup, QSlider,
    QProgressBar, QCheckBox, QGridLayout, QScrollArea, QTabWidget,
    QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from pcd_parser import parse_pcd
from quaternion_utils import look_at_quaternion, quat_map_to_odom
from vtk_viewer import VTKViewer


class NoWheelSlider(QSlider):
    """禁用滚轮事件的 QSlider，避免滚动鼠标时意外改变值"""
    def wheelEvent(self, event):
        event.accept()  # 接受但不处理，阻止事件继续传播


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

        # ─── 左侧 3D 视图 ───
        self.viewer = VTKViewer()

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

        # 裁剪框（XYZ过滤）
        clip_row0 = QHBoxLayout()
        self.chk_clip = QCheckBox("裁剪框:")
        self.chk_clip.setChecked(False)
        clip_row0.addWidget(self.chk_clip)
        self.btn_clip_apply = QPushButton("应用")
        self.btn_clip_apply.setMaximumWidth(50)
        self.btn_clip_apply.setEnabled(False)
        clip_row0.addWidget(self.btn_clip_apply)
        gl.addLayout(clip_row0)

        clip_row_x = QHBoxLayout()
        clip_row_x.addWidget(QLabel("X:"))
        self.edt_clip_xmin = QLineEdit("-999"); self.edt_clip_xmin.setMaximumWidth(60); self.edt_clip_xmin.setEnabled(False)
        self.edt_clip_xmax = QLineEdit("999"); self.edt_clip_xmax.setMaximumWidth(60); self.edt_clip_xmax.setEnabled(False)
        clip_row_x.addWidget(self.edt_clip_xmin)
        clip_row_x.addWidget(QLabel("~"))
        clip_row_x.addWidget(self.edt_clip_xmax)
        gl.addLayout(clip_row_x)

        clip_row_y = QHBoxLayout()
        clip_row_y.addWidget(QLabel("Y:"))
        self.edt_clip_ymin = QLineEdit("-999"); self.edt_clip_ymin.setMaximumWidth(60); self.edt_clip_ymin.setEnabled(False)
        self.edt_clip_ymax = QLineEdit("999"); self.edt_clip_ymax.setMaximumWidth(60); self.edt_clip_ymax.setEnabled(False)
        clip_row_y.addWidget(self.edt_clip_ymin)
        clip_row_y.addWidget(QLabel("~"))
        clip_row_y.addWidget(self.edt_clip_ymax)
        gl.addLayout(clip_row_y)

        clip_row_z = QHBoxLayout()
        clip_row_z.addWidget(QLabel("Z:"))
        self.edt_clip_zmin = QLineEdit("-999"); self.edt_clip_zmin.setMaximumWidth(60); self.edt_clip_zmin.setEnabled(False)
        self.edt_clip_zmax = QLineEdit("999"); self.edt_clip_zmax.setMaximumWidth(60); self.edt_clip_zmax.setEnabled(False)
        clip_row_z.addWidget(self.edt_clip_zmin)
        clip_row_z.addWidget(QLabel("~"))
        clip_row_z.addWidget(self.edt_clip_zmax)
        gl.addLayout(clip_row_z)

        # 点云渲染模式 + 大小
        render_row = QHBoxLayout()
        render_row.addWidget(QLabel("渲染:"))
        self.cmb_render_mode = QComboBox()
        self.cmb_render_mode.addItems(["自动", "球体", "立方体", "像素"])
        self.cmb_render_mode.setMaximumWidth(80)
        render_row.addWidget(self.cmb_render_mode)
        render_row.addWidget(QLabel("大小:"))
        self.sld_point_size = NoWheelSlider(Qt.Horizontal)
        self.sld_point_size.setRange(1, 20)
        self.sld_point_size.setValue(5)
        self.sld_point_size.setMaximumWidth(80)
        render_row.addWidget(self.sld_point_size)
        self.lbl_point_size = QLabel("0.05")
        self.lbl_point_size.setMaximumWidth(35)
        render_row.addWidget(self.lbl_point_size)
        gl.addLayout(render_row)
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
        self.edt_safe_dist = QLineEdit("1.0")
        self.edt_safe_dist.setMaximumWidth(60)
        self.edt_safe_dist.textChanged.connect(self._on_safe_dist_changed)
        sd_row.addWidget(self.edt_safe_dist)
        sd_row.addStretch()
        pk.addLayout(sd_row)

        tk_row = QHBoxLayout()
        tk_row.addWidget(QLabel("起飞高度(m):"))
        self.edt_takeoff_z = QLineEdit("1.2")
        self.edt_takeoff_z.setMaximumWidth(60)
        self.edt_takeoff_z.setReadOnly(True)
        self.edt_takeoff_z.setStyleSheet("QLineEdit { background: #eee; color: #666; }")
        tk_row.addWidget(self.edt_takeoff_z)
        tk_row.addWidget(QLabel("初始偏航角(°):"))
        self.edt_takeoff_yaw = QLineEdit("-90")
        self.edt_takeoff_yaw.setMaximumWidth(60)
        tk_row.addWidget(self.edt_takeoff_yaw)
        tk_row.addStretch()
        pk.addLayout(tk_row)

        safe_row = QHBoxLayout()
        safe_row.addWidget(QLabel("安全点(x,y,z):"))
        self.edt_safe_x = QLineEdit("0"); self.edt_safe_x.setMaximumWidth(50)
        self.edt_safe_y = QLineEdit("0"); self.edt_safe_y.setMaximumWidth(50)
        self.edt_safe_z = QLineEdit("5.0"); self.edt_safe_z.setMaximumWidth(50)
        safe_row.addWidget(self.edt_safe_x)
        safe_row.addWidget(self.edt_safe_y)
        safe_row.addWidget(self.edt_safe_z)
        safe_row.addStretch()
        pk.addLayout(safe_row)

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
        self.edt_z = QLineEdit("2"); fl.addWidget(self.edt_z, 0, 1)
        fl.addWidget(QLabel("线间距:"), 0, 2)
        self.edt_spacing = QLineEdit("2"); fl.addWidget(self.edt_spacing, 0, 3)

        fl.addWidget(QLabel("航点距离:"), 1, 0)
        self.edt_wp_spacing = QLineEdit("2"); fl.addWidget(self.edt_wp_spacing, 1, 1)
        fl.addWidget(QLabel("速度(m/s):"), 1, 2)
        self.edt_flat_speed = QLineEdit("1"); fl.addWidget(self.edt_flat_speed, 1, 3)

        # 相机型号 → FOV 映射
        self._camera_fov_map = {
            "DJI Mavic 3E": 84,
            "DJI Mavic 3T": 82,
            "DJI Matrice 4T": 82,
            "DJI M350+P1(24mm)": 84,
            "DJI M350+P1(35mm)": 54,
            "DJI M350+P1(50mm)": 40,
            "DJI M350+L2(雷达)": 70,
            "自定义": 80,
        }

        fl.addWidget(QLabel("相机型号:"), 2, 0)
        self.cmb_camera = QComboBox()
        self.cmb_camera.addItems(self._camera_fov_map.keys())
        fl.addWidget(self.cmb_camera, 2, 1)
        self.cmb_camera.currentTextChanged.connect(self._on_camera_changed)

        fl.addWidget(QLabel("FOV(°):"), 2, 2)
        self.edt_camera_fov = QLineEdit("84"); fl.addWidget(self.edt_camera_fov, 2, 3)

        fl.addWidget(QLabel("航向重叠(%):"), 3, 0)
        self.edt_forward_overlap = QLineEdit("60"); fl.addWidget(self.edt_forward_overlap, 3, 1)
        fl.addWidget(QLabel("旁向重叠(%):"), 3, 2)
        self.edt_side_overlap = QLineEdit("30"); fl.addWidget(self.edt_side_overlap, 3, 3)

        btn_calc_overlap = QPushButton("自动算间距")
        btn_calc_overlap.setStyleSheet("QPushButton { background: #e8e0d0; padding: 4px; } QPushButton:hover { background: #d8d0c0; }")
        btn_calc_overlap.clicked.connect(self._calc_overlap_spacing)
        fl.addWidget(btn_calc_overlap, 4, 0, 1, 4)

        fl.addWidget(QLabel("曲度:"), 5, 0)
        self.sld_curvature = NoWheelSlider(Qt.Horizontal)
        self.sld_curvature.setRange(0, 100)
        self.sld_curvature.setValue(0)
        fl.addWidget(self.sld_curvature, 5, 1, 1, 2)
        self.lbl_curvature_val = QLabel("0.00")
        self.lbl_curvature_val.setMinimumWidth(30)
        fl.addWidget(self.lbl_curvature_val, 5, 3)
        self.sld_curvature.valueChanged.connect(lambda v: self.lbl_curvature_val.setText(f"{v/100:.2f}"))

        btn_apply_flat = QPushButton("应用")
        btn_apply_flat.setStyleSheet("QPushButton { background: #d0d8e8; padding: 4px; } QPushButton:hover { background: #c0c8d8; }")
        btn_apply_flat.clicked.connect(self._apply_flat_params)
        fl.addWidget(btn_apply_flat, 6, 0, 1, 2)

        self.btn_poly_select = QPushButton("点击放置（右键确认生成）")
        self.btn_poly_select.setStyleSheet("QPushButton { background: #d8e8d8; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c8d8c8; }")
        self.btn_poly_select.clicked.connect(self._start_polygon_select)
        fl.addWidget(self.btn_poly_select, 6, 2, 1, 2)

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
        self.edt_dz = QLineEdit("2"); cl.addWidget(self.edt_dz, 2, 1)
        cl.addWidget(QLabel("离柱距离:"), 2, 2)
        self.edt_dist = QLineEdit("1"); cl.addWidget(self.edt_dist, 2, 3)

        cl.addWidget(QLabel("水平步距:"), 3, 0)
        self.edt_cstep = QLineEdit("2"); cl.addWidget(self.edt_cstep, 3, 1)
        cl.addWidget(QLabel("垂直步距:"), 3, 2)
        self.edt_vstep = QLineEdit("2"); cl.addWidget(self.edt_vstep, 3, 3)

        cl.addWidget(QLabel("速度:"), 4, 0)
        self.edt_cspeed = QLineEdit("1"); cl.addWidget(self.edt_cspeed, 4, 1)

        cl.addWidget(QLabel("起始角度(°):"), 5, 0)
        self.sld_cube_start_angle = NoWheelSlider(Qt.Horizontal)
        self.sld_cube_start_angle.setRange(0, 360)
        self.sld_cube_start_angle.setValue(0)
        cl.addWidget(self.sld_cube_start_angle, 5, 1)
        self.lbl_cube_angle_val = QLabel("0°")
        self.lbl_cube_angle_val.setMinimumWidth(30)
        cl.addWidget(self.lbl_cube_angle_val, 5, 2)
        btn_cube_auto_angle = QPushButton("自动")
        btn_cube_auto_angle.setMaximumWidth(50)
        btn_cube_auto_angle.clicked.connect(self._auto_cube_angle)
        cl.addWidget(btn_cube_auto_angle, 5, 3)
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
        self.edt_cyl_h = QLineEdit("2"); cyl.addWidget(self.edt_cyl_h, 1, 3)

        cyl.addWidget(QLabel("离柱距离:"), 2, 0)
        self.edt_cyl_dist = QLineEdit("1"); cyl.addWidget(self.edt_cyl_dist, 2, 1)
        cyl.addWidget(QLabel("水平步距(°):"), 2, 2)
        self.edt_cyl_astep = QLineEdit("15"); cyl.addWidget(self.edt_cyl_astep, 2, 3)

        cyl.addWidget(QLabel("垂直步距:"), 3, 0)
        self.edt_cyl_vstep = QLineEdit("2"); cyl.addWidget(self.edt_cyl_vstep, 3, 1)
        cyl.addWidget(QLabel("速度:"), 3, 2)
        self.edt_cyl_speed = QLineEdit("1"); cyl.addWidget(self.edt_cyl_speed, 3, 3)

        cyl.addWidget(QLabel("路径:"), 4, 0)
        self.cbo_cyl_type = QComboBox()
        self.cbo_cyl_type.addItems(["螺旋线", "Z字形"])
        cyl.addWidget(self.cbo_cyl_type, 4, 1)

        cyl.addWidget(QLabel("起始角度(°):"), 5, 0)
        self.sld_cyl_start_angle = NoWheelSlider(Qt.Horizontal)
        self.sld_cyl_start_angle.setRange(0, 360)
        self.sld_cyl_start_angle.setValue(0)
        cyl.addWidget(self.sld_cyl_start_angle, 5, 1)
        self.lbl_cyl_angle_val = QLabel("0°")
        self.lbl_cyl_angle_val.setMinimumWidth(30)
        cyl.addWidget(self.lbl_cyl_angle_val, 5, 2)
        btn_cyl_auto_angle = QPushButton("自动")
        btn_cyl_auto_angle.setMaximumWidth(50)
        btn_cyl_auto_angle.clicked.connect(self._auto_cyl_angle)
        cyl.addWidget(btn_cyl_auto_angle, 5, 3)
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

        # -- Tab 4: 直线航线 --
        tab_line = QWidget()
        ll = QGridLayout(tab_line)
        ll.setSpacing(4)

        ll.addWidget(QLabel("起点(x,y,z):"), 0, 0)
        self.edt_line_x1 = QLineEdit("0"); ll.addWidget(self.edt_line_x1, 0, 1)
        self.edt_line_y1 = QLineEdit("0"); ll.addWidget(self.edt_line_y1, 0, 2)
        self.edt_line_z1 = QLineEdit("5"); ll.addWidget(self.edt_line_z1, 0, 3)

        ll.addWidget(QLabel("终点(x,y,z):"), 1, 0)
        self.edt_line_x2 = QLineEdit("10"); ll.addWidget(self.edt_line_x2, 1, 1)
        self.edt_line_y2 = QLineEdit("0"); ll.addWidget(self.edt_line_y2, 1, 2)
        self.edt_line_z2 = QLineEdit("5"); ll.addWidget(self.edt_line_z2, 1, 3)

        btn_pick_line = QPushButton("选择起终点")
        btn_pick_line.setStyleSheet("QPushButton { background: #e8e0d0; padding: 4px; } QPushButton:hover { background: #d8d0c0; }")
        btn_pick_line.clicked.connect(self._start_line_mode)
        ll.addWidget(btn_pick_line, 0, 4, 2, 1)

        ll.addWidget(QLabel("航点距离:"), 2, 0)
        self.edt_line_spacing = QLineEdit("2"); ll.addWidget(self.edt_line_spacing, 2, 1)
        ll.addWidget(QLabel("速度(m/s):"), 2, 2)
        self.edt_line_speed = QLineEdit("1"); ll.addWidget(self.edt_line_speed, 2, 3)

        btn_apply_line = QPushButton("生成直线航线")
        btn_apply_line.setStyleSheet("QPushButton { background: #d0d8e8; padding: 6px; font-weight: bold; } QPushButton:hover { background: #c0c8d8; }")
        btn_apply_line.clicked.connect(self.generate_line_route)
        ll.addWidget(btn_apply_line, 3, 0, 1, 4)

        route_tabs.addTab(tab_line, "直线航线")

        # -- Tab 5: 点状航线 --
        tab_inspect = QWidget()
        il = QGridLayout(tab_inspect)
        il.setSpacing(4)

        il.addWidget(QLabel("巡检点列表:"), 0, 0)
        self.btn_inspect = QPushButton("选择巡检点")
        self.btn_inspect.setStyleSheet("QPushButton { background: #e8e0d0; font-weight: bold; padding: 6px; } QPushButton:hover { background: #d8d0c0; }")
        self.btn_inspect.clicked.connect(self._start_inspect_mode)
        il.addWidget(self.btn_inspect, 0, 1)
        self.btn_clear_inspect = QPushButton("清除")
        self.btn_clear_inspect.setMaximumWidth(60)
        self.btn_clear_inspect.clicked.connect(self._clear_inspect_points)
        il.addWidget(self.btn_clear_inspect, 0, 2)

        self.lst_inspect = QListWidget()
        self.lst_inspect.setMaximumHeight(100)
        il.addWidget(self.lst_inspect, 1, 0, 1, 3)

        il.addWidget(QLabel("巡检距离:"), 2, 0)
        self.edt_inspect_dist = QLineEdit("3.0")
        self.edt_inspect_dist.setMaximumWidth(50)
        il.addWidget(self.edt_inspect_dist, 2, 1)
        il.addWidget(QLabel("m"), 2, 2)

        self.btn_gen_inspect = QPushButton("生成点状航线")
        self.btn_gen_inspect.setStyleSheet("QPushButton { background: #d0e8d0; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c0d8c0; }")
        self.btn_gen_inspect.clicked.connect(self.generate_inspect_route)
        il.addWidget(self.btn_gen_inspect, 3, 0, 1, 3)

        self._inspect_target_points = []  # 巡检目标点列表
        self.viewer.inspect_points_confirmed.connect(self._on_inspect_confirmed)
        self.viewer.line_points_confirmed.connect(self._on_line_confirmed)

        route_tabs.addTab(tab_inspect, "点状航线")

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
        self.btn_copy = QPushButton("复制航线到剪贴板")
        self.btn_copy.clicked.connect(self.copy_route_to_clipboard)
        rl.addWidget(self.btn_copy)
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
        scroll.setFixedWidth(340)
        main_layout.addWidget(self.viewer)
        main_layout.addWidget(scroll)

        # -- 信号连接 --
        self.btn_load.clicked.connect(self.load_point_cloud)
        self.chk_clip.toggled.connect(self._on_clip_toggled)
        self.btn_clip_apply.clicked.connect(self._apply_clip)
        self.cmb_render_mode.currentTextChanged.connect(self._on_render_mode_changed)
        self.sld_point_size.valueChanged.connect(self._on_point_size_changed)
        self.btn_clear.clicked.connect(self.clear_route)
        self.btn_save.clicked.connect(self.save_route)
        self.btn_load_route.clicked.connect(self.load_route)

        # 立方体区域参数变化时重新计算
        self.edt_cx.textChanged.connect(lambda: self._on_cube_area_changed())
        self.edt_cy.textChanged.connect(lambda: self._on_cube_area_changed())
        self.edt_dx.textChanged.connect(lambda: self._on_cube_area_changed())
        self.edt_dy.textChanged.connect(lambda: self._on_cube_area_changed())

        # 圆柱体区域参数变化时重新计算
        self.edt_cyl_cx.textChanged.connect(lambda: self._on_cyl_area_changed())
        self.edt_cyl_cy.textChanged.connect(lambda: self._on_cyl_area_changed())
        self.edt_cyl_diam.textChanged.connect(lambda: self._on_cyl_area_changed())
        self.edt_cyl_dist.textChanged.connect(lambda: self._on_cyl_area_changed())

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
        # 跳过 QFontDatabase().families() 扫描（Windows 下很慢），直接用 CSS 回退列表
        self.setStyleSheet("""
            QMainWindow { background: #f0f0ee; }
            QWidget { color: #000; font-family: "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "PingFang SC", "Segoe UI", Arial; font-size: 12px; }
            QGroupBox {
                border: 1px solid #ccc; border-radius: 6px;
                margin-top: 8px; padding: 10px 8px; font-weight: bold; color: #000;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #000; }
            QLineEdit {
                background: #fff; border: 1px solid #bbb; border-radius: 3px;
                padding: 3px 6px; color: #000;
            }
            QLineEdit:focus { border-color: #4a9eff; }
            QPushButton {
                background: #e0e0de; border: 1px solid #bbb; border-radius: 4px;
                padding: 6px 14px; color: #000; min-height: 24px;
            }
            QPushButton:hover { background: #d0d0ce; }
            QPushButton:pressed { background: #c0c0be; }
            QComboBox {
                background: #fff; border: 1px solid #bbb; border-radius: 3px;
                padding: 3px 6px; color: #000;
            }
            QComboBox QAbstractItemView { background: #fff; color: #000; selection-background-color: #4a9eff; }
        """)

    # ─── 多边形选择 ──────────────────────────────────────────
    def _start_polygon_select(self):
        self.viewer.enter_polygon_mode()

    def _on_polygon_finished(self, pts):
        poly = np.array(pts)
        mn = poly.min(axis=0)
        mx = poly.max(axis=0)
        if self.points is not None and len(self.points) > 0:
            z_val = max(self.points[:, 2].max() + 3, 2.0)
        else:
            z_val = 2.0
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

            self.viewer.add_point_cloud(self.points, self._get_render_mode(), self._get_point_size())
            self.progress_bar.setValue(80)
            QApplication.processEvents()

            n = len(self.points)
            self.lbl_pc_info.setText(f"已加载: {os.path.basename(path)} ({n:,} 点)")

            if n > 0:
                mn = self.points.min(axis=0)
                mx = self.points.max(axis=0)

                # 起飞Z 固定 1.2m
                self.edt_takeoff_z.setText("1.2")

                # 安全点 Z = 点云最高点 + 2m
                safe_z = float(mx[2]) + 2.0
                self.edt_safe_z.setText(f"{safe_z:.1f}")

                # 最低飞行Z值 = 点云最低点
                self.edt_min_z.setText(f"{float(mn[2]):.1f}")

                self.edt_z.setText(f"{mx[2] + 3:.1f}")

                center = (mn + mx) / 2
                self.edt_cx.setText(f"{center[0]:.1f}")
                self.edt_cy.setText(f"{center[1]:.1f}")

                # 立方体四角
                dx_default = 4.0
                dy_default = 4.0
                try:
                    dx_default = float(self.edt_dx.text())
                    dy_default = float(self.edt_dy.text())
                except ValueError:
                    pass
                half_x, half_y = dx_default / 2, dy_default / 2
                cube_corners = [
                    (center[0] - half_x, center[1] - half_y),
                    (center[0] + half_x, center[1] - half_y),
                    (center[0] + half_x, center[1] + half_y),
                    (center[0] - half_x, center[1] + half_y),
                ]

                # 立方体底面Z
                cube_cz = self._compute_default_cz(cube_corners, 1.2)
                self.edt_cz.setText(f"{cube_cz:.1f}")

                # 圆柱体圆周四点
                self.edt_cyl_cx.setText(f"{center[0]:.1f}")
                self.edt_cyl_cy.setText(f"{center[1]:.1f}")
                cyl_radius = 2.0
                cyl_dist = 3.0
                try:
                    cyl_radius = float(self.edt_cyl_diam.text()) / 2
                    cyl_dist = float(self.edt_cyl_dist.text())
                except ValueError:
                    pass
                R = cyl_radius + cyl_dist
                cyl_corners = [
                    (center[0] + R, center[1]),
                    (center[0], center[1] + R),
                    (center[0] - R, center[1]),
                    (center[0], center[1] - R),
                ]
                cyl_cz = self._compute_default_cz(cyl_corners, 1.2)
                self.edt_cyl_cz.setText(f"{cyl_cz:.1f}")
                max_z_cube = self._compute_max_z_for_area(cube_corners)
                if max_z_cube is not None:
                    max_dz = max(1.0, max_z_cube - cube_cz)
                    self.edt_dz.setText(f"{min(mx[2] - mn[2], max_dz):.1f}")
                else:
                    self.edt_dz.setText(f"{mx[2] - mn[2]:.1f}")

                # 圆柱体高度受限于圆周上四点最近点Z - 0.5
                max_z_cyl = self._compute_max_z_for_area(cyl_corners)
                if max_z_cyl is not None:
                    max_h = max(1.0, max_z_cyl - cyl_cz)
                    self.edt_cyl_h.setText(f"{min(mx[2] - mn[2], max_h):.1f}")
                else:
                    self.edt_cyl_h.setText(f"{mx[2] - mn[2]:.1f}")

            self.progress_bar.setValue(100)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载点云失败:\n{str(e)}")
        finally:
            self.progress_bar.setVisible(False)

        # 设置XYZ裁剪框默认范围
        if self.points is not None and len(self.points) > 0:
            mn = self.points.min(axis=0)
            mx = self.points.max(axis=0)
            self.edt_clip_xmin.setText(f"{mn[0]:.1f}")
            self.edt_clip_xmax.setText(f"{mx[0]:.1f}")
            self.edt_clip_ymin.setText(f"{mn[1]:.1f}")
            self.edt_clip_ymax.setText(f"{mx[1]:.1f}")
            self.edt_clip_zmin.setText(f"{mn[2]:.1f}")
            self.edt_clip_zmax.setText(f"{mx[2]:.1f}")

    # ─── Z值过滤 ───
    def _get_render_mode(self):
        """获取当前渲染模式: 'auto'/'sphere'/'cube'/'pixel'"""
        text = self.cmb_render_mode.currentText()
        return {"自动": "auto", "球体": "sphere", "立方体": "cube", "像素": "pixel"}.get(text, "auto")

    def _get_point_size(self):
        return self.sld_point_size.value() * 0.01

    def _on_point_size_changed(self, val):
        self.lbl_point_size.setText(f"{val * 0.01:.2f}")
        self._refresh_point_cloud()

    def _on_render_mode_changed(self, text):
        self._refresh_point_cloud()

    def _refresh_point_cloud(self):
        if self.chk_clip.isChecked():
            self._apply_clip()
        elif self.points is not None:
            self.viewer.add_point_cloud(self.points, self._get_render_mode(), self._get_point_size())

    def _on_clip_toggled(self, checked):
        for w in [self.edt_clip_xmin, self.edt_clip_xmax,
                  self.edt_clip_ymin, self.edt_clip_ymax,
                  self.edt_clip_zmin, self.edt_clip_zmax, self.btn_clip_apply]:
            w.setEnabled(checked)
        self._refresh_point_cloud()

    def _apply_clip(self):
        if self.points is None or len(self.points) == 0:
            return
        try:
            xmin = float(self.edt_clip_xmin.text())
            xmax = float(self.edt_clip_xmax.text())
            ymin = float(self.edt_clip_ymin.text())
            ymax = float(self.edt_clip_ymax.text())
            zmin = float(self.edt_clip_zmin.text())
            zmax = float(self.edt_clip_zmax.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的裁剪范围")
            return
        p = self.points
        mask = (p[:,0]>=xmin)&(p[:,0]<=xmax)&(p[:,1]>=ymin)&(p[:,1]<=ymax)&(p[:,2]>=zmin)&(p[:,2]<=zmax)
        filtered = p[mask]
        if len(filtered) == 0:
            QMessageBox.information(self, "提示", "裁剪后无点云数据")
            return
        self.viewer.add_point_cloud(filtered, self._get_render_mode(), self._get_point_size())
        n_total = len(self.points)
        n_show = len(filtered)
        self.lbl_pc_info.setText(f"已加载: {n_total:,} 点 (显示 {n_show:,})")

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
            use_polygon = True
        elif self.points is not None and len(self.points) > 0:
            mn = self.points.min(axis=0)
            mx = self.points.max(axis=0)
            poly = np.array([
                [mn[0], mn[1]], [mx[0], mn[1]],
                [mx[0], mx[1]], [mn[0], mx[1]]
            ])
            use_polygon = False
        else:
            QMessageBox.warning(self, "提示", "请先加载点云或多边形选择区域")
            return

        # 计算多边形主方向（用最长边的方向作为扫描方向）
        n_poly = len(poly)
        max_len = 0
        main_dir = np.array([1.0, 0.0])
        for i in range(n_poly):
            edge = poly[(i + 1) % n_poly][:2] - poly[i][:2]
            length = np.linalg.norm(edge)
            if length > max_len:
                max_len = length
                main_dir = edge / length

        # 构建旋转矩阵：将多边形旋转到轴对齐
        # 扫描方向沿X轴，垂直方向沿Y轴
        cos_a = main_dir[0]
        sin_a = main_dir[1]
        R_to_axis = np.array([[cos_a, sin_a], [-sin_a, cos_a]])  # 旋转到轴对齐
        R_from_axis = np.array([[cos_a, -sin_a], [sin_a, cos_a]])  # 旋转回来

        # 旋转多边形到轴对齐坐标系
        poly_rot = np.dot(poly[:, :2], R_to_axis.T)
        xmin, ymin = poly_rot[:, 0].min(), poly_rot[:, 1].min()
        xmax, ymax = poly_rot[:, 0].max(), poly_rot[:, 1].max()

        if xmin >= xmax or ymin >= ymax:
            QMessageBox.warning(self, "输入错误", "区域范围无效")
            return

        y_center = (ymin + ymax) / 2
        y_half = (ymax - ymin) / 2 if ymax != ymin else 1.0
        span = y_half * 2

        def curved_z(y_local):
            t = (y_local - y_center) / y_half
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

        self.waypoints = []
        direction = 1
        y = ymin
        y_step = spacing

        while y <= ymax + y_step * 0.5:
            if direction == 1:
                x_start, x_end = xmin, xmax
            else:
                x_start, x_end = xmax, xmin

            # 在旋转坐标系中裁剪到多边形内
            xs = np.linspace(xmin, xmax, max(100, int((xmax - xmin) / 0.5)))
            inside_xs = [x for x in xs if point_in_polygon(x, y, poly_rot)]
            if not inside_xs:
                y += spacing
                continue
            if direction == 1:
                x_start, x_end = inside_xs[0], inside_xs[-1]
            else:
                x_start, x_end = inside_xs[-1], inside_xs[0]

            line_len = abs(x_end - x_start)
            n_pts = max(2, int(line_len / wp_spacing) + 1)
            z_line = curved_z(y)

            for j in range(n_pts):
                x_local = x_start + (x_end - x_start) * j / (n_pts - 1)
                # 从轴对齐坐标系旋转回原始坐标系
                xy_orig = np.dot([x_local, y], R_from_axis.T)
                pos = np.array([xy_orig[0], xy_orig[1], z_line])
                # 机头方向：垂直于扫描线，朝区域中心
                # normal 是垂直于扫描方向的单位向量（在原始坐标系中）
                normal = np.array([-sin_a, cos_a])
                if y >= y_center:
                    heading = np.array([-normal[0], -normal[1], 0.0])
                else:
                    heading = np.array([normal[0], normal[1], 0.0])
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

        # 高度受限于圆周四点最近点Z - 0.5
        radius = diam / 2 + dist
        cyl_circle_pts = [
            (cx + radius, cy),
            (cx, cy + radius),
            (cx - radius, cy),
            (cx, cy - radius),
        ]
        max_z = self._compute_max_z_for_area(cyl_circle_pts)
        if max_z is not None and (cz + h) > max_z:
            h = max(1.0, max_z - cz)
            self.edt_cyl_h.setText(f"{h:.1f}")
            QMessageBox.information(self, "高度调整", f"圆柱体高度已调整为 {h:.1f}m（受限于区域上方点云）")

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

            for col in range(num_cols):  # 不含num_cols，因为2π=0与col=0重复
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

    # ─── 生成直线航线 ───
    def generate_line_route(self):
        try:
            x1 = float(self.edt_line_x1.text())
            y1 = float(self.edt_line_y1.text())
            z1 = float(self.edt_line_z1.text())
            x2 = float(self.edt_line_x2.text())
            y2 = float(self.edt_line_y2.text())
            z2 = float(self.edt_line_z2.text())
            spacing = float(self.edt_line_spacing.text())
            speed = float(self.edt_line_speed.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效数字")
            return

        if spacing <= 0:
            QMessageBox.warning(self, "输入错误", "航点距离必须为正数")
            return

        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])
        length = np.linalg.norm(p2 - p1)
        if length < 1e-10:
            QMessageBox.warning(self, "输入错误", "起点和终点不能重合")
            return

        n_pts = max(2, int(length / spacing) + 1)
        direction = (p2 - p1) / length

        self.waypoints = []
        for i in range(n_pts):
            t = i / (n_pts - 1)
            pos = p1 + t * (p2 - p1)
            target = pos + direction
            quat = look_at_quaternion(target, pos)
            self.waypoints.append({
                'pos': pos,
                'quat': quat,
                'speed': speed,
                'action': 'fly'
            })

        self._display_route()
        print(f"[Line] Generated {len(self.waypoints)} waypoints")

    # ─── 巡检点功能 ─────────────────────────────────────────
    def _start_inspect_mode(self):
        self.viewer.enter_inspect_mode()

    def _on_inspect_confirmed(self, pts):
        """巡检点选点确认回调"""
        self._inspect_target_points = [np.array(p) for p in pts]
        self.lst_inspect.clear()
        for i, p in enumerate(self._inspect_target_points):
            self.lst_inspect.addItem(f"P{i+1}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
        print(f"[Inspect] {len(self._inspect_target_points)} inspection points confirmed")

    def _clear_inspect_points(self):
        self._inspect_target_points.clear()
        self.lst_inspect.clear()
        self.viewer._clear_inspect_points()
        self.viewer.vtk_widget.GetRenderWindow().Render()

    def _start_line_mode(self):
        self.viewer.enter_line_mode()

    def _on_line_confirmed(self, pts):
        """直线起终点选点确认回调"""
        if len(pts) == 2:
            s, e = pts[0], pts[1]
            self.edt_line_x1.setText(f"{s[0]:.1f}")
            self.edt_line_y1.setText(f"{s[1]:.1f}")
            self.edt_line_z1.setText(f"{s[2]:.1f}")
            self.edt_line_x2.setText(f"{e[0]:.1f}")
            self.edt_line_y2.setText(f"{e[1]:.1f}")
            self.edt_line_z2.setText(f"{e[2]:.1f}")
            print(f"[Line] 起点({s[0]:.1f},{s[1]:.1f},{s[2]:.1f}) 终点({e[0]:.1f},{e[1]:.1f},{e[2]:.1f})")

    def _estimate_normal(self, point):
        """用 PCA 估计点云在该位置的法线方向"""
        tree = self._get_kdtree()
        if tree is None or len(self.points) < 3:
            return np.array([0.0, 0.0, 1.0])
        k = max(3, min(30, len(self.points)))
        dists, idxs = tree.query(point, k=k)
        neighbors = self.points[idxs]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / len(neighbors)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # 最小特征值 = 法线方向
        # 确保法线朝外（远离点云质心）
        centroid = self.points.mean(axis=0)
        if np.dot(normal, point - centroid) < 0:
            normal = -normal
        return normal

    def _find_safe_position(self, target, normal, tree, collision_dist, safe_dist, max_offset=10.0):
        """沿法线方向找安全飞行位置，返回 (pos, warned)"""
        for i in range(20):
            offset = safe_dist + i * 0.5
            if offset > max_offset:
                return None, True
            pos = target + normal * offset
            if tree is not None:
                dist, _ = tree.query(pos)
                if dist >= collision_dist:
                    return pos, False
            else:
                return pos, False
        return None, True

    def generate_inspect_route(self):
        """从巡检目标点自动生成无人机航线（带碰撞检测）"""
        if not self._inspect_target_points:
            QMessageBox.warning(self, "提示", "请先选择巡检点位")
            return

        try:
            inspect_dist = float(self.edt_inspect_dist.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的巡检距离")
            return

        if inspect_dist <= 0:
            QMessageBox.warning(self, "输入错误", "巡检距离必须为正数")
            return

        safe_dist = self.viewer._safe_distance
        collision_dist = safe_dist * 0.5
        tree = self._get_kdtree()

        self.waypoints = []
        warnings = []

        for i, target in enumerate(self._inspect_target_points):
            normal = self._estimate_normal(target)
            pos, warned = self._find_safe_position(
                target, normal, tree, collision_dist, inspect_dist
            )
            if pos is None:
                # 法线方向找不到安全位置，尝试向上
                pos = target + np.array([0, 0, inspect_dist])
                if tree is not None:
                    dist, _ = tree.query(pos)
                    if dist < collision_dist:
                        warnings.append(f"P{i+1} 无法找到安全位置")
                        pos = target + np.array([0, 0, inspect_dist + 2.0])
                warned = True

            if warned:
                warnings.append(f"P{i+1} 航点可能过近")

            # 朝向巡检目标点
            quat = look_at_quaternion(target, pos)
            self.waypoints.append({
                'pos': pos,
                'quat': quat,
                'speed': 1.0,
                'action': 'scan'
            })

        self._display_route()

        if warnings:
            QMessageBox.warning(self, "碰撞警告",
                f"生成 {len(self.waypoints)} 个航点\n\n" + "\n".join(warnings))
        print(f"[Inspect] Generated {len(self.waypoints)} waypoints from {len(self._inspect_target_points)} targets")

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

        # 高度受限于区域四角最近点Z - 0.5
        half_x_check = dx / 2
        half_y_check = dy / 2
        cos_a_check = np.cos(start_angle)
        sin_a_check = np.sin(start_angle)
        cube_corners = [
            (cx + (-half_x_check) * cos_a_check - (-half_y_check) * sin_a_check,
             cy + (-half_x_check) * sin_a_check + (-half_y_check) * cos_a_check),
            (cx + (half_x_check) * cos_a_check - (-half_y_check) * sin_a_check,
             cy + (half_x_check) * sin_a_check + (-half_y_check) * cos_a_check),
            (cx + (half_x_check) * cos_a_check - (half_y_check) * sin_a_check,
             cy + (half_x_check) * sin_a_check + (half_y_check) * cos_a_check),
            (cx + (-half_x_check) * cos_a_check - (half_y_check) * sin_a_check,
             cy + (-half_x_check) * sin_a_check + (half_y_check) * cos_a_check),
        ]
        max_z = self._compute_max_z_for_area(cube_corners)
        if max_z is not None and (cz + dz) > max_z:
            dz = max(1.0, max_z - cz)
            self.edt_dz.setText(f"{dz:.1f}")
            QMessageBox.information(self, "高度调整", f"立方体高度已调整为 {dz:.1f}m（受限于区域上方点云）")

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
                # 每个点独立计算指向立方体中心的方向
                to_cx = cx - px
                to_cy = cy - py
                to_c_len = np.sqrt(to_cx * to_cx + to_cy * to_cy)
                if to_c_len > 1e-10:
                    heading = np.array([to_cx / to_c_len, to_cy / to_c_len, 0.0])
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
                    # 跳过每条边最后一个点（避免与下一条边起点重复）
                    if pi == (len(pts) - 1 if not reverse else 0) and ei != (3 if not reverse else 0):
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
            takeoff_z, takeoff_yaw = self._get_takeoff_params()
            self.viewer._takeoff_z = takeoff_z
            self.viewer._takeoff_yaw = takeoff_yaw
            self.viewer.add_route(self.waypoints)
            self._check_safety_distance()
        print(f"[Safety] 起飞高度={self.edt_takeoff_z.text()}m, 初始偏航角={self.edt_takeoff_yaw.text()}°")

    def _on_camera_changed(self, name):
        """相机型号切换时自动填入FOV"""
        fov = self._camera_fov_map.get(name, 80)
        self.edt_camera_fov.setText(str(fov))

    def _calc_overlap_spacing(self):
        """根据相机FOV和重叠率自动计算航点距离和线间距"""
        import math
        try:
            h = float(self.edt_z.text())
            fov = float(self.edt_camera_fov.text())
            fwd_overlap = float(self.edt_forward_overlap.text()) / 100.0
            side_overlap = float(self.edt_side_overlap.text()) / 100.0
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效数值")
            return
        if h <= 0 or fov <= 0 or fov >= 180:
            QMessageBox.warning(self, "提示", "高度和FOV需为正数，FOV<180°")
            return
        if not (0 <= fwd_overlap < 1) or not (0 <= side_overlap < 1):
            QMessageBox.warning(self, "提示", "重叠率需在0~99%之间")
            return
        # 地面覆盖宽度 = 2 * H * tan(FOV/2)
        cover = 2.0 * h * math.tan(math.radians(fov / 2.0))
        wp_spacing = round(cover * (1.0 - fwd_overlap), 2)
        line_spacing = round(cover * (1.0 - side_overlap), 2)
        self.edt_wp_spacing.setText(str(max(0.1, wp_spacing)))
        self.edt_spacing.setText(str(max(0.1, line_spacing)))
        print(f"[Overlap] H={h}m FOV={fov}° 覆盖={cover:.1f}m → 航点间距={wp_spacing}m 线间距={line_spacing}m")

    def _apply_flat_params(self):
        """应用面状航线参数并重新生成航线"""
        if hasattr(self, '_polygon_vertices') and self._polygon_vertices:
            self.generate_flat_route()
        else:
            QMessageBox.warning(self, "提示", "请先点击选择区域按钮并绘制多边形")

    def _auto_cube_angle(self):
        """自动计算立方体起始角度：使第一个航点离起飞点最近"""
        if self.points is None or len(self.points) == 0:
            QMessageBox.warning(self, "提示", "请先加载点云")
            return
        try:
            cx = float(self.edt_cx.text())
            cy = float(self.edt_cy.text())
            dx = float(self.edt_dx.text())
            dy = float(self.edt_dy.text())
        except ValueError:
            return
        half_x, half_y = dx / 2, dy / 2
        best_angle, best_dist = 0, float('inf')
        for deg in range(361):
            a = np.radians(deg)
            cos_a, sin_a = np.cos(a), np.sin(a)
            # corners[0] = (-halfX, -halfY) 旋转后
            px = cx + (-half_x * cos_a - (-half_y) * sin_a)
            py = cy + (-half_x * sin_a + (-half_y) * cos_a)
            d = px * px + py * py
            if d < best_dist:
                best_dist = d
                best_angle = deg
        self.sld_cube_start_angle.setValue(best_angle)

    def _auto_cyl_angle(self):
        """自动计算圆柱体起始角度：使第一个航点离起飞点最近"""
        if self.points is None or len(self.points) == 0:
            QMessageBox.warning(self, "提示", "请先加载点云")
            return
        try:
            cx = float(self.edt_cyl_cx.text())
            cy = float(self.edt_cyl_cy.text())
        except ValueError:
            return
        # 最优角度：从圆心指向起飞点(0,0)的方向
        angle_rad = np.arctan2(-cy, -cx)
        angle_deg = int(np.degrees(angle_rad)) % 360
        self.sld_cyl_start_angle.setValue(angle_deg)

    def _apply_cube_params(self):
        """应用立方体航线参数并重新生成航线"""
        if self.waypoints:
            self.generate_cube_route()
        else:
            QMessageBox.warning(self, "提示", "请先点击放置并右键确认生成航线")

    def _apply_cyl_params(self):
        """应用圆柱体航线参数并重新生成航线"""
        if self.waypoints:
            self.generate_cylinder_route()
        else:
            QMessageBox.warning(self, "提示", "请先点击放置并右键确认生成航线")

    def _toggle_heading(self, state):
        self.viewer.show_heading = (state == Qt.Checked)
        if self.waypoints:
            takeoff_z, takeoff_yaw = self._get_takeoff_params()
            self.viewer._takeoff_z = takeoff_z
            self.viewer._takeoff_yaw = takeoff_yaw
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

    def _on_cube_area_changed(self):
        """立方体区域参数变化时重新计算cz和dz"""
        if self.points is None or len(self.points) == 0:
            return
        try:
            cx = float(self.edt_cx.text())
            cy = float(self.edt_cy.text())
            takeoff_z = float(self.edt_takeoff_z.text())
            dx = float(self.edt_dx.text())
            dy = float(self.edt_dy.text())
        except ValueError:
            return
        half_x, half_y = dx / 2, dy / 2
        corners = [
            (cx - half_x, cy - half_y), (cx + half_x, cy - half_y),
            (cx + half_x, cy + half_y), (cx - half_x, cy + half_y),
        ]
        cz = self._compute_default_cz(corners, takeoff_z)
        self.edt_cz.setText(f"{cz:.1f}")
        max_z = self._compute_max_z_for_area(corners)
        if max_z is not None:
            max_dz = max(1.0, max_z - cz)
            try:
                cur_dz = float(self.edt_dz.text())
                if cur_dz > max_dz:
                    self.edt_dz.setText(f"{max_dz:.1f}")
            except ValueError:
                self.edt_dz.setText(f"{max_dz:.1f}")

    def _on_cyl_area_changed(self):
        """圆柱体区域参数变化时重新计算cz和h"""
        if self.points is None or len(self.points) == 0:
            return
        try:
            cx = float(self.edt_cyl_cx.text())
            cy = float(self.edt_cyl_cy.text())
            takeoff_z = float(self.edt_takeoff_z.text())
            diam = float(self.edt_cyl_diam.text())
            dist = float(self.edt_cyl_dist.text())
        except ValueError:
            return
        R = diam / 2 + dist
        corners = [
            (cx + R, cy), (cx, cy + R),
            (cx - R, cy), (cx, cy - R),
        ]
        cz = self._compute_default_cz(corners, takeoff_z)
        self.edt_cyl_cz.setText(f"{cz:.1f}")
        max_z = self._compute_max_z_for_area(corners)
        if max_z is not None:
            max_h = max(1.0, max_z - cz)
            try:
                cur_h = float(self.edt_cyl_h.text())
                if cur_h > max_h:
                    self.edt_cyl_h.setText(f"{max_h:.1f}")
            except ValueError:
                self.edt_cyl_h.setText(f"{max_h:.1f}")

    def _compute_default_cz(self, corners, takeoff_z):
        """计算底面中心默认Z值：>= takeoff_z+0.5 且 >= 四个角附近点云最低Z+0.5
        使用 XY 平面距离（2D），不依赖 KDTree（因为 KDTree 是 3D 的）"""
        cz_min = takeoff_z + 0.5
        if self.points is not None and len(self.points) > 0:
            xy = self.points[:, :2]
            for cx, cy in corners:
                diff = xy - [cx, cy]
                dist_sq = diff[:, 0] ** 2 + diff[:, 1] ** 2
                mask = dist_sq < 4.0  # 半径2m内的点
                if np.any(mask):
                    nearby_z = np.min(self.points[mask, 2])
                    cz_min = max(cz_min, nearby_z + 0.5)
        return cz_min

    def _compute_max_z_for_area(self, corners, ref_z=None):
        """计算区域最大允许飞行Z值：水平2m内、在ref_z以上的最高点Z - 0.5
        ref_z: 参考高度，只考虑此高度以上的点。默认使用多边形区域的Z均值。
        """
        if self.points is None or len(self.points) == 0:
            return None
        if ref_z is None:
            ref_z = min(c[2] if len(c) > 2 else 0 for c in corners) if corners else 0
        xy = self.points[:, :2]
        max_z = -float('inf')
        for cx, cy in corners:
            diff = xy - [cx, cy]
            dist_sq = diff[:, 0] ** 2 + diff[:, 1] ** 2
            mask = dist_sq < 4.0  # 水平2m内
            if ref_z is not None:
                mask = mask & (self.points[:, 2] > ref_z)
            if np.any(mask):
                z = np.max(self.points[mask, 2])
                if z > max_z:
                    max_z = z
        return max_z - 0.5 if max_z > -float('inf') else None

    def _on_takeoff_z_changed(self, text):
        try:
            float(text)
        except ValueError:
            return
        self._update_cube_cz_and_dz()
        self._update_cyl_cz_and_h()

    def _update_cube_cz_and_dz(self):
        """重新计算立方体的 cz 和 dz 上限"""
        if self.points is None or len(self.points) == 0:
            return
        try:
            cx = float(self.edt_cx.text())
            cy = float(self.edt_cy.text())
            takeoff_z = float(self.edt_takeoff_z.text())
            dx = float(self.edt_dx.text())
            dy = float(self.edt_dy.text())
        except ValueError:
            return
        half_x, half_y = dx / 2, dy / 2
        corners = [
            (cx - half_x, cy - half_y), (cx + half_x, cy - half_y),
            (cx + half_x, cy + half_y), (cx - half_x, cy + half_y),
        ]
        cz = self._compute_default_cz(corners, takeoff_z)
        self.edt_cz.setText(f"{cz:.1f}")
        max_z = self._compute_max_z_for_area(corners)
        if max_z is not None:
            max_dz = max(1.0, max_z - cz)
            try:
                cur_dz = float(self.edt_dz.text())
                if cur_dz > max_dz:
                    self.edt_dz.setText(f"{max_dz:.1f}")
            except ValueError:
                self.edt_dz.setText(f"{max_dz:.1f}")

    def _update_cyl_cz_and_h(self):
        """重新计算圆柱体的 cz 和 h 上限"""
        if self.points is None or len(self.points) == 0:
            return
        try:
            cx = float(self.edt_cyl_cx.text())
            cy = float(self.edt_cyl_cy.text())
            takeoff_z = float(self.edt_takeoff_z.text())
            diam = float(self.edt_cyl_diam.text())
            dist = float(self.edt_cyl_dist.text())
        except ValueError:
            return
        R = diam / 2 + dist
        corners = [
            (cx + R, cy), (cx, cy + R),
            (cx - R, cy), (cx, cy - R),
        ]
        cz = self._compute_default_cz(corners, takeoff_z)
        self.edt_cyl_cz.setText(f"{cz:.1f}")
        max_z = self._compute_max_z_for_area(corners)
        if max_z is not None:
            max_h = max(1.0, max_z - cz)
            try:
                cur_h = float(self.edt_cyl_h.text())
                if cur_h > max_h:
                    self.edt_cyl_h.setText(f"{max_h:.1f}")
            except ValueError:
                self.edt_cyl_h.setText(f"{max_h:.1f}")

    def _on_cube_area_changed(self):
        """立方体区域参数变化时重新计算cz和dz"""
        self._update_cube_cz_and_dz()

    def _on_cyl_area_changed(self):
        """圆柱体区域参数变化时重新计算cz和h"""
        self._update_cyl_cz_and_h()

    def _display_route(self):
        takeoff_z, takeoff_yaw = self._get_takeoff_params()
        self.viewer._takeoff_z = takeoff_z
        self.viewer._takeoff_yaw = takeoff_yaw
        try:
            self.viewer._safe_point = (
                float(self.edt_safe_x.text()),
                float(self.edt_safe_y.text()),
                float(self.edt_safe_z.text()),
            )
        except ValueError:
            self.viewer._safe_point = (0.0, 0.0, 5.0)
        self.viewer.add_route(self.waypoints)
        self.lbl_info.setText(f"航点: {len(self.waypoints)}")
        self._check_safety_distance()

    def _get_takeoff_params(self):
        """解析起飞高度和初始偏航角，返回 (takeoff_z, takeoff_yaw)"""
        try:
            takeoff_z = float(self.edt_takeoff_z.text())
        except ValueError:
            takeoff_z = 1.0
        try:
            takeoff_yaw = float(self.edt_takeoff_yaw.text())
        except ValueError:
            takeoff_yaw = 0.0
        return takeoff_z, takeoff_yaw

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
        violations, collisions, low_z = self._collect_collision_warnings()

        for i, j, d in violations:
            if i < len(self.viewer._waypoint_actors):
                self.viewer._waypoint_actors[i].GetProperty().SetColor(1.0, 0.0, 0.0)
            if j < len(self.viewer._waypoint_actors):
                self.viewer._waypoint_actors[j].GetProperty().SetColor(1.0, 0.0, 0.0)

        for idx, dist in collisions:
            if idx >= 0 and idx < len(self.viewer._waypoint_actors):
                self.viewer._waypoint_actors[idx].GetProperty().SetColor(1.0, 0.0, 1.0)

        try:
            min_z = float(self.edt_min_z.text())
        except ValueError:
            min_z = -999
        for idx, z_val in low_z:
            if idx < len(self.viewer._waypoint_actors):
                self.viewer._waypoint_actors[idx].GetProperty().SetColor(1.0, 0.5, 0.0)

        collision_dist = safe_dist * 0.5
        seg_collisions = set(idx for idx, _ in collisions if idx >= 0)
        safe_collision = any(idx == -1 for idx, _ in collisions)
        msgs = [f"航点: {len(self.waypoints)}"]
        if violations:
            msgs.append(f"{len(violations)} 对过近 (<{safe_dist}m)")
        if seg_collisions:
            msgs.append(f"{len(seg_collisions)} 航点碰撞 (<{collision_dist:.1f}m)")
        if safe_collision:
            msgs.append("安全点路径碰撞")
        if low_z:
            msgs.append(f"{len(low_z)} 个低于Z={min_z}m")
        self.lbl_info.setText(" | ".join(msgs))
        self.viewer.vtk_widget.GetRenderWindow().Render()

    def _collect_collision_warnings(self):
        """收集碰撞检测数据，返回 (violations, collisions, low_z)
        violations: [(i, j, dist), ...] 航点间距过近
        collisions: [(idx, dist), ...] 航点距点云过近（含线段采样）
        low_z: [(idx, z_val), ...] 航点低于最低Z值
        """
        safe_dist = self.viewer._safe_distance
        collision_dist = safe_dist * 0.5
        sample_step = safe_dist * 0.5

        # ── 航点间距检测（向量化）──
        violations = []
        if len(self.waypoints) >= 2:
            positions = np.array([wp['pos'] for wp in self.waypoints])
            diffs = np.diff(positions, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            too_close = np.where(dists < safe_dist)[0]
            violations = [(int(i), int(i + 1), float(dists[i])) for i in too_close]

        # 构建完整路径点列表：安全点 + 所有航点
        try:
            sx = float(self.edt_safe_x.text())
            sy = float(self.edt_safe_y.text())
            sz = float(self.edt_safe_z.text())
        except ValueError:
            sx, sy, sz = 0.0, 0.0, 5.0
        safe_pos = np.array([sx, sy, sz])
        all_positions = [safe_pos] + [wp['pos'] for wp in self.waypoints]

        # ── 线段采样碰撞检测（批量查询）──
        collisions = []
        tree = self._get_kdtree()
        if tree is not None and len(all_positions) >= 2:
            all_pts = []
            seg_labels = []
            for seg_i in range(len(all_positions) - 1):
                p1 = all_positions[seg_i]
                p2 = all_positions[seg_i + 1]
                d = np.linalg.norm(p2 - p1)
                if d < 1e-10:
                    continue
                n_samples = max(2, int(d / sample_step) + 1)
                ts = np.linspace(0, 1, n_samples + 1)
                pts = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
                all_pts.append(pts)
                label = -1 if seg_i == 0 else seg_i
                seg_labels.extend([label] * len(pts))

            if all_pts:
                all_pts_arr = np.vstack(all_pts)
                dists, _ = tree.query(all_pts_arr)
                hit_mask = dists < collision_dist
                collisions = [(seg_labels[i], float(dists[i])) for i in np.where(hit_mask)[0]]

        # ── 最低Z值检测 ──
        low_z = []
        try:
            min_z = float(self.edt_min_z.text())
        except ValueError:
            min_z = -999
        if min_z > -900:
            for i, wp in enumerate(self.waypoints):
                if wp['pos'][2] < min_z:
                    low_z.append((i, wp['pos'][2]))

        return violations, collisions, low_z

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

        # 碰撞检测警告（不影响保存）
        safe_dist = self.viewer._safe_distance
        violations, collisions, low_z = self._collect_collision_warnings()
        warnings = []
        for i, j, d in violations:
            warnings.append(f"{i+1}-{j+1} 号航点间距 {d:.2f}m < {safe_dist}m")
        for idx, dist in collisions:
            if idx == -1:
                warnings.append(f"安全点路径碰撞 (距点云 {dist:.2f}m)")
            else:
                warnings.append(f"{idx+1} 号航点碰撞 (距点云 {dist:.2f}m)")
        try:
            min_z = float(self.edt_min_z.text())
        except ValueError:
            min_z = -999
        for idx, z_val in low_z:
            warnings.append(f"{idx+1} 号航点Z={z_val:.1f}m < {min_z}m")
        if warnings:
            msg = f"检测到 {len(warnings)} 个问题:\n\n" + "\n".join(warnings[:10])
            if len(warnings) > 10:
                msg += f"\n...等共 {len(warnings)} 个"
            msg += "\n\n是否仍要保存？"
            reply = QMessageBox.warning(self, "碰撞警告", msg,
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        data = self._build_route_json()
        if data is None:
            return

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "已保存", f"航线已保存到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")

    def _build_route_json(self):
        """构建航线JSON数据（供保存和复制共用）"""
        if not self.waypoints:
            return None

        _, takeoff_yaw = self._get_takeoff_params()

        yaw_rad = np.radians(takeoff_yaw)
        takeoff_quat = quat_map_to_odom(np.array([np.cos(yaw_rad / 2), 0, 0, np.sin(yaw_rad / 2)]))

        def _wp_to_pose(wp):
            q = quat_map_to_odom(wp['quat'])
            return {
                "header": {"stamp": {"sec": 0, "nsec": 0}, "frame_id": "camera_init"},
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
        # Pose 0: origin (0,0,0)
        poses.append({
            "header": {"stamp": {"sec": 0, "nsec": 0}, "frame_id": "camera_init"},
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

        # Pose 1: safety point
        try:
            sx = float(self.edt_safe_x.text())
            sy = float(self.edt_safe_y.text())
            sz = float(self.edt_safe_z.text())
        except ValueError:
            sx, sy, sz = 0.0, 0.0, 5.0
        safe_pos = np.array([sx, sy, sz])
        # 安全点朝向第一个航点
        first_wp_pos = self.waypoints[0]['pos']
        safe_heading = first_wp_pos - safe_pos
        safe_heading[2] = 0
        if np.linalg.norm(safe_heading) > 1e-10:
            safe_target = safe_pos + safe_heading
            safe_quat = look_at_quaternion(safe_target, safe_pos)
        else:
            safe_quat = takeoff_quat
        safe_quat_odom = quat_map_to_odom(safe_quat)
        poses.append({
            "header": {"stamp": {"sec": 0, "nsec": 0}, "frame_id": "camera_init"},
            "pose": {
                "position": {
                    "x": round(sx, 4), "y": round(sy, 4), "z": round(sz, 4)
                },
                "orientation": {
                    "x": round(float(safe_quat_odom[1]), 6),
                    "y": round(float(safe_quat_odom[2]), 6),
                    "z": round(float(safe_quat_odom[3]), 6),
                    "w": round(float(safe_quat_odom[0]), 6)
                }
            }
        })

        for wp in self.waypoints:
            poses.append(_wp_to_pose(wp))

        return {
            "header": {"stamp": {"sec": 0, "nsec": 0}, "frame_id": "camera_init"},
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

    def copy_route_to_clipboard(self):
        """复制航线JSON到剪贴板"""
        data = self._build_route_json()
        if data is None:
            QMessageBox.information(self, "提示", "没有航线可复制")
            return
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        QApplication.clipboard().setText(json_str)
        QMessageBox.information(self, "已复制", f"航线已复制到剪贴板（{len(data['poses'])} 个航点）")

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
