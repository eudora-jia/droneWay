"""桥梁巡检航线规划工具 - 主窗口"""

import json
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QButtonGroup, QSlider,
    QProgressBar, QCheckBox, QGridLayout, QScrollArea, QTabWidget,
    QListWidget, QListWidgetItem, QAction, QActionGroup, QStackedWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from pcd_parser import parse_pcd, parse_ply
from quaternion_utils import look_at_quaternion, quat_map_to_odom
from vtk_viewer import VTKViewer


class NoWheelSlider(QSlider):
    """禁用滚轮事件的 QSlider，避免滚动鼠标时意外改变值"""
    def wheelEvent(self, event):
        event.accept()  # 接受但不处理，阻止事件继续传播


class MainWindow(QMainWindow):
    """桥梁巡检航线规划工具 - 主窗口"""

    # ─── 中英文翻译字典 ───
    _T = {
        "zh": {
            "win_title": "桥梁巡检无人机航线规划工具",
            "menu_file": "文件", "menu_view": "展示",
            "act_load_pc": "加载点云", "act_save_ros": "保存ROS航线",
            "act_load_route": "加载ROS航线", "act_copy_route": "复制ROS航线到剪贴板",
            "act_export_maicro": "导出maicro航线文件",
            "act_copy_maicro": "复制maicro航线到剪贴板",
            "act_clip": "裁剪框", "menu_render": "点云渲染模式", "menu_size": "点云大小",
            "menu_upsample": "点云渲染增密",
            "menu_color": "点云颜色方案",
            "color_original": "原色", "color_height": "高度着色",
            "color_thermal": "热力图", "color_grayscale": "灰度",
            "color_red": "纯红", "color_green": "纯绿", "color_blue": "纯蓝",
            "act_fpv": "FPV无人机视角", "act_fpv_pos": "无人机设置",
            "menu_lang": "语言", "lang_zh": "中文", "lang_en": "English",
            "menu_settings": "设置", "act_bridge_params": "桥梁参数",
            "act_camera_params": "无人机参数",
            "act_range_calc": "拍摄范围计算器",
            "grp_mode": "工作模式", "btn_route": "航线模式",
            "lbl_pc_info": "未加载点云",
            "grp_bridge": "桥梁参数", "lbl_bridge_name": "桥梁名称:",
            "lbl_bridge_type": "桥型:", "lbl_bridge_len": "桥长(m):",
            "lbl_bridge_wid": "桥宽(m):", "lbl_clearance": "净空(m):",
            "lbl_span": "跨距(m):", "btn_apply": "应用",
            "grp_safe": "安全设置", "lbl_safe_dist": "安全距离(m):",
            "lbl_takeoff_z": "起飞高度(m):", "lbl_takeoff_yaw": "起飞偏航角(°):",
            "lbl_safe_point": "安全点(x,y,z):", "lbl_min_z": "最低飞行Z值(m):",
            "lbl_min_z_hint": "低于此值视为碰撞",
            "lbl_wp_hint": "Ctrl+左键点击航点可拖动编辑位置",
            "lbl_route_type": "航线类型:",
            "route_flat": "面状航线", "route_cube": "立方体航线",
            "route_cyl": "圆柱体航线", "route_line": "直线航线",
            "route_inspect": "点状航线",
            "lbl_height_z": "高度Z:", "lbl_line_spacing": "线间距:",
            "lbl_wp_spacing": "航点距离:", "lbl_speed": "速度(m/s):",
            "lbl_camera": "相机型号:", "lbl_fov": "水平FOV(°):",
            "lbl_fwd_overlap": "航向重叠(%):", "lbl_side_overlap": "旁向重叠(%):",
            "btn_calc_overlap": "自动算间距", "lbl_curvature": "曲度:",
            "btn_pick_area": "选择顶点（右键确认）",
            "lbl_center": "底面中心(x,y,z):", "lbl_len_x": "长(X):",
            "lbl_wid_y": "宽(Y):", "lbl_height_z2": "高(Z):",
            "lbl_dist": "巡检距离:", "lbl_h_step": "水平步距:",
            "lbl_v_step": "垂直步距:", "lbl_speed2": "速度:",
            "lbl_start_angle": "起始角度(°):", "btn_auto": "自动",
            "lbl_diam": "直径:", "lbl_h_step_angle": "水平步距(°):",
            "lbl_path": "路径:", "lbl_start_pt": "起点(x,y,z):",
            "lbl_end_pt": "终点(x,y,z):", "btn_pick_endpoints": "选择起终点",
            "btn_gen_line": "生成直线航线",
            "lbl_inspect_list": "巡检点列表:", "btn_select_inspect": "选择巡检点",
            "btn_clear": "清除", "lbl_inspect_dist": "巡检距离:",
            "btn_gen_inspect": "生成点状航线",
            "grp_route_mgmt": "航线管理", "lbl_wp_count": "航点: 0",
            "btn_clear_route": "清除", "btn_save_ros2": "保存ROS航线",
            "btn_copy": "复制ROS航线到剪贴板", "btn_load_route2": "加载航线 (JSON)",
            "chk_heading": "显示机头方向",
            "lbl_shortcuts": "快捷键: 1=俯视 2=正视 3=侧视 4=透视 5=仰视  Esc=取消",
            "clip_enable": "启用", "btn_clip_apply": "应用",
            "lbl_clip_x": "X:", "lbl_clip_y": "Y:", "lbl_clip_z": "Z:",
            "lbl_render": "渲染:", "lbl_size": "大小:",
            "btn_gen_cube": "应用", "btn_gen_cube2": "点击放置（右键确认生成）",
            "btn_gen_cyl": "应用", "btn_gen_cyl2": "点击放置（右键确认生成）",
        },
        "en": {
            "win_title": "Bridge Inspection Drone Route Planner",
            "menu_file": "File", "menu_view": "View",
            "act_load_pc": "Load Point Cloud", "act_save_ros": "Save ROS Route",
            "act_load_route": "Load ROS Route", "act_copy_route": "Copy ROS Route to Clipboard",
            "act_export_maicro": "Export Maicro Route",
            "act_copy_maicro": "Copy Maicro Route to Clipboard",
            "act_clip": "Clip Box", "menu_render": "Render Mode", "menu_size": "Point Size",
            "menu_upsample": "Display Upsample",
            "menu_color": "Color Scheme",
            "color_original": "Original", "color_height": "Height",
            "color_thermal": "Thermal", "color_grayscale": "Grayscale",
            "color_red": "Red", "color_green": "Green", "color_blue": "Blue",
            "act_fpv": "FPV Drone View", "act_fpv_pos": "Drone Settings",
            "menu_lang": "Language", "lang_zh": "中文", "lang_en": "English",
            "menu_settings": "Settings", "act_bridge_params": "Bridge Params",
            "act_camera_params": "Drone Params",
            "act_range_calc": "Coverage Calculator",
            "grp_mode": "Mode", "btn_route": "Route",
            "lbl_pc_info": "No point cloud loaded",
            "grp_bridge": "Bridge Params", "lbl_bridge_name": "Bridge Name:",
            "lbl_bridge_type": "Type:", "lbl_bridge_len": "Length(m):",
            "lbl_bridge_wid": "Width(m):", "lbl_clearance": "Clearance(m):",
            "lbl_span": "Span(m):", "btn_apply": "Apply",
            "grp_safe": "Safety", "lbl_safe_dist": "Safe Dist(m):",
            "lbl_takeoff_z": "Takeoff Z(m):", "lbl_takeoff_yaw": "Takeoff Yaw(°):",
            "lbl_safe_point": "Safe Point(x,y,z):", "lbl_min_z": "Min Z(m):",
            "lbl_min_z_hint": "Below this = collision",
            "lbl_wp_hint": "Ctrl+Click waypoint to drag",
            "lbl_route_type": "Route Type:",
            "route_flat": "Flat", "route_cube": "Cube",
            "route_cyl": "Cylinder", "route_line": "Line",
            "route_inspect": "Inspect",
            "lbl_height_z": "Height Z:", "lbl_line_spacing": "Line Spacing:",
            "lbl_wp_spacing": "WP Spacing:", "lbl_speed": "Speed(m/s):",
            "lbl_camera": "Camera:", "lbl_fov": "H-FOV(°):",
            "lbl_fwd_overlap": "Fwd Overlap(%):", "lbl_side_overlap": "Side Overlap(%):",
            "btn_calc_overlap": "Auto Calc", "lbl_curvature": "Curvature:",
            "btn_pick_area": "Place (Right-click to confirm)",
            "lbl_center": "Center(x,y,z):", "lbl_len_x": "Len(X):",
            "lbl_wid_y": "Wid(Y):", "lbl_height_z2": "H(Z):",
            "lbl_dist": "Inspect Dist(m):", "lbl_h_step": "H Step:",
            "lbl_v_step": "V Step:", "lbl_speed2": "Speed:",
            "lbl_start_angle": "Start Angle(°):", "btn_auto": "Auto",
            "lbl_diam": "Diameter:", "lbl_h_step_angle": "H Step(°):",
            "lbl_path": "Path:", "lbl_start_pt": "Start(x,y,z):",
            "lbl_end_pt": "End(x,y,z):", "btn_pick_endpoints": "Pick Points",
            "btn_gen_line": "Generate Line Route",
            "lbl_inspect_list": "Inspection Points:", "btn_select_inspect": "Select Points",
            "btn_clear": "Clear", "lbl_inspect_dist": "Inspect Dist(m):",
            "btn_gen_inspect": "Generate Inspect Route",
            "grp_route_mgmt": "Route Mgmt", "lbl_wp_count": "Waypoints: 0",
            "btn_clear_route": "Clear", "btn_save_ros2": "Save ROS Route",
            "btn_copy": "Copy ROS Route to Clipboard", "btn_load_route2": "Load Route (JSON)",
            "chk_heading": "Show Heading",
            "lbl_shortcuts": "Keys: 1=Top 2=Front 3=Side 4=Persp 5=Bottom  Esc=Cancel",
            "clip_enable": "Enable", "btn_clip_apply": "Apply",
            "lbl_clip_x": "X:", "lbl_clip_y": "Y:", "lbl_clip_z": "Z:",
            "lbl_render": "Render:", "lbl_size": "Size:",
            "btn_gen_cube": "Apply", "btn_gen_cube2": "Place (Right-click to confirm)",
            "btn_gen_cyl": "Apply", "btn_gen_cyl2": "Place (Right-click to confirm)",
        },
    }

    def __init__(self):
        super().__init__()
        self._lang = 'zh'
        self.setWindowTitle(self._T['zh']['win_title'])
        self.resize(1400, 900)

        # 相机型号 → FOV 映射（所有航线共用）
        # FOV均为水平FOV（实测校准）
        # M4T中长焦：标称对角线35°，4:3画幅实测水平覆盖1.1m@3m → 水平FOV≈20.8°
        self._camera_fov_map = {
            "DJI Mavic 3E": 84,
            "DJI Mavic 3T": 82,
            "M4T 广角": 82,
            "M4T 中长焦": 20.8,
            "M4T 长焦": 15,
            "DJI M350+P1(24mm)": 84,
            "DJI M350+P1(35mm)": 54,
            "DJI M350+P1(50mm)": 40,
            "DJI M350+L2(雷达)": 70,
            "自定义": 80,
        }
        self._camera_name = "M4T 中长焦"
        self._camera_fov = 20.8  # 水平FOV
        self._camera_aspect = 4 / 3  # 画幅比例 宽:高
        self._camera_zoom = 3.0  # 默认3倍变焦
        self._camera_zoom_map = {
            "M4T 广角": 1.0,
            "M4T 中长焦": 3.0,
            "M4T 长焦": 7.0,
        }
        self._camera_min_interval = 0.7  # 最小拍摄间隔（秒），M4T为0.7s
        self._forward_overlap = 30
        self._side_overlap = 30

        # 云台参数
        self._gimbal_yaw = 0.0       # 偏航角（度），0=与机头同向
        self._gimbal_stabilized = True  # 是否稳定
        self._gimbal_pitch_min = -90.0  # 云台俯仰最小值（M4T广角）
        self._gimbal_pitch_max = 55.0   # 云台俯仰最大值（M4T广角）

        # ─── 菜单栏 ───
        self._init_menu_bar()

        self.points = None
        self._point_colors = None
        self._point_normals = None
        self._fpv_start_pos = [0.0, 0.0, 0.0]  # 无人机起始位置
        self.waypoints = []
        self._kdtree = None
        self._kdtree_points_id = None

        self._init_ui()
        self._apply_style()
        self.viewer.setup_scene()

    def _init_menu_bar(self):
        """初始化菜单栏"""
        menubar = self.menuBar()
        menubar.setStyleSheet("QMenuBar { font-size: 13px; } QMenuBar::item { padding: 4px 10px; }")
        self._menubar = menubar

        # ─── 文件 ───
        self._file_menu = menubar.addMenu("文件")

        self._act_load_pc = QAction("加载点云", self)
        self._act_load_pc.triggered.connect(self.load_point_cloud)
        self._file_menu.addAction(self._act_load_pc)

        self._act_load_stl = QAction("加载模型(STL/OBJ)", self)
        self._act_load_stl.triggered.connect(self.load_stl_mesh)
        self._file_menu.addAction(self._act_load_stl)

        self._file_menu.addSeparator()

        self._act_save_ros = QAction("保存ROS航线", self)
        self._act_save_ros.triggered.connect(self.save_route)
        self._file_menu.addAction(self._act_save_ros)

        self._act_load_route = QAction("加载航线", self)
        self._act_load_route.triggered.connect(self.load_route)
        self._file_menu.addAction(self._act_load_route)

        self._act_copy_route = QAction("复制ROS航线到剪贴板", self)
        self._act_copy_route.triggered.connect(self.copy_route_to_clipboard)
        self._file_menu.addAction(self._act_copy_route)

        self._file_menu.addSeparator()

        self._act_export_maicro = QAction("导出maicro航线文件", self)
        self._act_export_maicro.triggered.connect(self.export_maicro_route)
        self._file_menu.addAction(self._act_export_maicro)

        self._act_copy_maicro = QAction("复制maicro航线到剪贴板", self)
        self._act_copy_maicro.triggered.connect(self.copy_maicro_route_to_clipboard)
        self._file_menu.addAction(self._act_copy_maicro)

        # ─── 展示 ───
        self._view_menu = menubar.addMenu("展示")

        self._act_clip_toggle = QAction("裁剪框", self)
        self._act_clip_toggle.triggered.connect(self._show_clip_dialog)
        self._view_menu.addAction(self._act_clip_toggle)

        self._view_menu.addSeparator()

        self._render_menu = self._view_menu.addMenu("点云渲染模式")
        self._render_mode_group = QActionGroup(self)
        self._render_mode_acts = {}
        for name in ["自动", "球体", "立方体", "像素", "圆片"]:
            act = QAction(name, self)
            act.setCheckable(True)
            act.setActionGroup(self._render_mode_group)
            if name == "自动":
                act.setChecked(True)
            act.triggered.connect(lambda checked, n=name: self._on_menu_render_mode(n))
            self._render_menu.addAction(act)
            self._render_mode_acts[name] = act

        self._view_menu.addSeparator()
        self._size_menu = self._view_menu.addMenu("点云大小")
        self._size_group = QActionGroup(self)
        self._size_acts = {}
        for val in [1, 3, 5, 8, 10, 15, 20]:
            label = f"{val * 0.01:.2f}"
            act = QAction(label, self)
            act.setCheckable(True)
            act.setActionGroup(self._size_group)
            if val == 5:
                act.setChecked(True)
            act.triggered.connect(lambda checked, v=val: self._on_menu_point_size(v))
            self._size_menu.addAction(act)
            self._size_acts[val] = act

        self._view_menu.addSeparator()
        self._upsample_menu = self._view_menu.addMenu("点云渲染增密")
        self._upsample_group = QActionGroup(self)
        self._upsample_acts = {}
        self._upsample_factor = 0  # 0=关闭, 1=2倍, 2=5倍, 3=10倍
        for name, factor in [("关闭", 0), ("2倍", 1), ("5倍", 2), ("10倍", 3)]:
            act = QAction(name, self)
            act.setCheckable(True)
            act.setActionGroup(self._upsample_group)
            if factor == 0:
                act.setChecked(True)
            act.triggered.connect(lambda checked, f=factor: self._on_upsample_changed(f))
            self._upsample_menu.addAction(act)
            self._upsample_acts[factor] = act

        # 颜色方案
        self._color_menu = self._view_menu.addMenu("点云颜色方案")
        self._color_group = QActionGroup(self)
        self._color_acts = {}
        self._color_scheme = "height"  # 默认高度着色
        for name, scheme in [("高度着色", "height"), ("原色", "original"), ("热力图", "thermal"),
                             ("灰度", "grayscale"), ("纯红", "red"), ("纯绿", "green"), ("纯蓝", "blue")]:
            act = QAction(name, self)
            act.setCheckable(True)
            act.setActionGroup(self._color_group)
            if scheme == "height":
                act.setChecked(True)
            act.triggered.connect(lambda checked, s=scheme: self._on_color_scheme_changed(s))
            self._color_menu.addAction(act)
            self._color_acts[scheme] = act

        self._view_menu.addSeparator()

        # STL透明度
        self._stl_opacity_menu = self._view_menu.addMenu("STL透明度")
        self._stl_opacity_group = QActionGroup(self)
        for name, val in [("不透明", 1.0), ("75%", 0.75), ("50%", 0.5), ("25%", 0.25), ("隐藏", 0.0)]:
            act = QAction(name, self)
            act.setCheckable(True)
            act.setActionGroup(self._stl_opacity_group)
            if val == 1.0:
                act.setChecked(True)
            act.triggered.connect(lambda checked, v=val: self.viewer.set_stl_opacity(v))
            self._stl_opacity_menu.addAction(act)

        self._act_fpv = QAction("FPV无人机视角 (V)", self)
        self._act_fpv.setCheckable(True)
        self._act_fpv.triggered.connect(self._toggle_fpv_mode)
        self._view_menu.addAction(self._act_fpv)

        self._act_fpv_pos = QAction("无人机设置", self)
        self._act_fpv_pos.triggered.connect(self._set_fpv_start_pos)
        self._view_menu.addAction(self._act_fpv_pos)

        # ─── 航线管理 ───
        self._route_menu = menubar.addMenu("航线管理")

        self._act_show_heading = QAction("显示机头方向", self)
        self._act_show_heading.setCheckable(True)
        self._act_show_heading.setChecked(True)
        self._act_show_heading.triggered.connect(self._toggle_heading)
        self._route_menu.addAction(self._act_show_heading)

        self._act_show_gimbal = QAction("显示云台方向", self)
        self._act_show_gimbal.setCheckable(True)
        self._act_show_gimbal.setChecked(False)
        self._act_show_gimbal.triggered.connect(self._toggle_gimbal_dir)
        self._route_menu.addAction(self._act_show_gimbal)

        self._route_menu.addSeparator()

        self._act_anim_play = QAction("航线动画播放", self)
        self._act_anim_play.triggered.connect(self._toggle_route_animation)
        self._route_menu.addAction(self._act_anim_play)

        # ─── 设置 ───
        self._settings_menu = menubar.addMenu("设置")
        self._act_bridge_params = QAction("桥梁参数", self)
        self._act_bridge_params.triggered.connect(self._show_bridge_dialog)
        self._settings_menu.addAction(self._act_bridge_params)

        self._act_camera_params = QAction("无人机参数", self)
        self._act_camera_params.triggered.connect(self._show_camera_dialog)
        self._settings_menu.addAction(self._act_camera_params)

        self._act_range_calc = QAction("拍摄范围计算器", self)
        self._act_range_calc.triggered.connect(self._show_range_calculator_dialog)
        self._settings_menu.addAction(self._act_range_calc)

        self._settings_menu.addSeparator()
        self._lang_menu = self._settings_menu.addMenu("语言")
        self._lang_group = QActionGroup(self)
        self._act_lang_zh = QAction("中文", self)
        self._act_lang_zh.setCheckable(True)
        self._act_lang_zh.setChecked(True)
        self._act_lang_zh.setActionGroup(self._lang_group)
        self._act_lang_zh.triggered.connect(lambda: self._switch_language('zh'))
        self._lang_menu.addAction(self._act_lang_zh)
        self._act_lang_en = QAction("English", self)
        self._act_lang_en.setCheckable(True)
        self._act_lang_en.setActionGroup(self._lang_group)
        self._act_lang_en.triggered.connect(lambda: self._switch_language('en'))
        self._lang_menu.addAction(self._act_lang_en)

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
        self.viewer.anim_finished.connect(self._on_anim_stopped)
        self._place_target = None  # "cube" or "cylinder"
        self._polygon_vertices = None

        # ─── 右侧控制面板 ───
        ctrl = QWidget()
        ctrl.setMinimumWidth(320)
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setSpacing(6)

        # -- 进度条 --
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        ctrl_layout.addWidget(self.progress_bar)

        # 状态栏
        self.statusBar().showMessage("未加载点云")

        # 裁剪状态（通过弹窗控制，展示菜单触发）
        self._clip_enabled = {'x': False, 'y': False, 'z': False}
        self._clip_positions = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # 点云渲染模式 + 大小（隐藏，通过展示菜单控制）
        self.cmb_render_mode = QComboBox()
        self.cmb_render_mode.addItems(["自动", "球体", "立方体", "像素"])
        self.cmb_render_mode.setVisible(False)
        self.sld_point_size = NoWheelSlider(Qt.Horizontal)
        self.sld_point_size.setRange(1, 20)
        self.sld_point_size.setValue(5)
        self.sld_point_size.setVisible(False)
        self.lbl_point_size = QLabel("0.05")
        self.lbl_point_size.setVisible(False)

        self._route_widgets = []

        # -- 桥梁参数（默认值，通过设置菜单弹窗编辑）--
        self._bridge_name_val = "我的桥梁"
        self._bridge_type_val = 0
        self._bridge_len_val = "100"
        self._bridge_wid_val = "15"
        self._bridge_clr_val = "8"
        self._bridge_span_val = "30"
        self._bridge_render = False
        self._bridge_info_label = QLabel(self.viewer)
        self._bridge_info_label.setStyleSheet(
            "QLabel { background: rgba(0,0,0,160); color: #fff; padding: 6px 10px; "
            "border-radius: 4px; font-size: 12px; }"
        )
        self._bridge_info_label.setVisible(False)
        self._bridge_info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # -- 安全距离 --
        grp_pick = QGroupBox("安全设置")
        pk = QVBoxLayout(grp_pick)
        sd_row = QHBoxLayout()
        sd_row.addWidget(QLabel("安全距离(m):"))
        self.edt_safe_dist = QLineEdit("1.0")
        self.edt_safe_dist.setMaximumWidth(60)
        self.edt_safe_dist.textChanged.connect(self._on_safe_dist_changed)
        self.viewer._safe_distance = 1.0  # 确保同步
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

        # ─── 航线类型选择 ───
        route_type_row = QHBoxLayout()
        route_type_row.addWidget(QLabel("航线类型:"))
        self.cmb_route_type = QComboBox()
        self.cmb_route_type.addItems(["面状航线", "立方体航线", "圆柱体航线", "直线航线", "点状航线"])
        route_type_row.addWidget(self.cmb_route_type)
        ctrl_layout.addLayout(route_type_row)

        self._route_stack = QStackedWidget()
        self.cmb_route_type.currentIndexChanged.connect(self._on_route_type_changed)

        # -- Tab 1: 面状航线 --
        tab_flat = QWidget()
        fl = QGridLayout(tab_flat)
        fl.setSpacing(4)

        fl.addWidget(QLabel("巡检距离(m):"), 0, 0)
        self.edt_flat_inspect_dist = QLineEdit("3.0"); fl.addWidget(self.edt_flat_inspect_dist, 0, 1)
        fl.addWidget(QLabel("速度(m/s):"), 0, 2)
        self.edt_flat_speed = QLineEdit("1"); fl.addWidget(self.edt_flat_speed, 0, 3)

        fl.addWidget(QLabel("航点间距:"), 1, 0)
        self.edt_wp_spacing = QLineEdit("自动")
        self.edt_wp_spacing.setReadOnly(True)
        self.edt_wp_spacing.setStyleSheet("QLineEdit { background: #eee; color: #666; }")
        fl.addWidget(self.edt_wp_spacing, 1, 1)
        fl.addWidget(QLabel("线间距:"), 1, 2)
        self.edt_spacing = QLineEdit("自动")
        self.edt_spacing.setReadOnly(True)
        self.edt_spacing.setStyleSheet("QLineEdit { background: #eee; color: #666; }")
        fl.addWidget(self.edt_spacing, 1, 3)

        btn_calc_overlap = QPushButton("自动算间距")
        btn_calc_overlap.setStyleSheet("QPushButton { background: #e8e0d0; padding: 4px; } QPushButton:hover { background: #d8d0c0; }")
        btn_calc_overlap.clicked.connect(self._calc_overlap_spacing)
        fl.addWidget(btn_calc_overlap, 2, 0, 1, 4)

        btn_apply_flat = QPushButton("应用")
        btn_apply_flat.setStyleSheet("QPushButton { background: #d0d8e8; padding: 4px; } QPushButton:hover { background: #c0c8d8; }")
        btn_apply_flat.clicked.connect(self._apply_flat_params)
        fl.addWidget(btn_apply_flat, 3, 0, 1, 4)

        self.btn_poly_select = QPushButton("选择顶点（右键确认）")
        self.btn_poly_select.setStyleSheet("QPushButton { background: #d8e8d8; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c8d8c8; }")
        self.btn_poly_select.clicked.connect(self._start_polygon_select)
        fl.addWidget(self.btn_poly_select, 4, 0, 1, 4)

        self._route_stack.addWidget(tab_flat)

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
        cl.addWidget(QLabel("巡检距离(m):"), 2, 2)
        self.edt_dist = QLineEdit("1"); cl.addWidget(self.edt_dist, 2, 3)

        cl.addWidget(QLabel("水平步距:"), 3, 0)
        self.edt_cstep = QLineEdit("自动")
        self.edt_cstep.setReadOnly(True)
        self.edt_cstep.setStyleSheet("QLineEdit { background: #eee; color: #666; }")
        cl.addWidget(self.edt_cstep, 3, 1)
        cl.addWidget(QLabel("垂直步距:"), 3, 2)
        self.edt_vstep = QLineEdit("自动")
        self.edt_vstep.setReadOnly(True)
        self.edt_vstep.setStyleSheet("QLineEdit { background: #eee; color: #666; }")
        cl.addWidget(self.edt_vstep, 3, 3)

        cl.addWidget(QLabel("速度:"), 4, 0)
        self.edt_cspeed = QLineEdit("1"); cl.addWidget(self.edt_cspeed, 4, 1)

        btn_cube_calc = QPushButton("自动算间距")
        btn_cube_calc.setStyleSheet("QPushButton { background: #e8e0d0; padding: 4px; } QPushButton:hover { background: #d8d0c0; }")
        btn_cube_calc.clicked.connect(self._calc_cube_spacing)
        cl.addWidget(btn_cube_calc, 4, 2, 1, 2)

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
        cl.addWidget(btn_apply_cube, 6, 0, 1, 4)

        self.btn_cube_place = QPushButton("点击放置（右键确认生成）")
        self.btn_cube_place.setStyleSheet("QPushButton { background: #d8e8d8; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c8d8c8; }")
        self.btn_cube_place.clicked.connect(lambda: self._start_place_mode("cube"))
        cl.addWidget(self.btn_cube_place, 7, 0, 1, 4)
        self._route_stack.addWidget(tab_cube)

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

        cyl.addWidget(QLabel("巡检距离(m):"), 2, 0)
        self.edt_cyl_dist = QLineEdit("1"); cyl.addWidget(self.edt_cyl_dist, 2, 1)
        cyl.addWidget(QLabel("水平步距(°):"), 2, 2)
        self.edt_cyl_astep = QLineEdit("自动")
        self.edt_cyl_astep.setReadOnly(True)
        self.edt_cyl_astep.setStyleSheet("QLineEdit { background: #eee; color: #666; }")
        cyl.addWidget(self.edt_cyl_astep, 2, 3)

        cyl.addWidget(QLabel("垂直步距:"), 3, 0)
        self.edt_cyl_vstep = QLineEdit("自动")
        self.edt_cyl_vstep.setReadOnly(True)
        self.edt_cyl_vstep.setStyleSheet("QLineEdit { background: #eee; color: #666; }")
        cyl.addWidget(self.edt_cyl_vstep, 3, 1)
        cyl.addWidget(QLabel("速度:"), 3, 2)
        self.edt_cyl_speed = QLineEdit("1"); cyl.addWidget(self.edt_cyl_speed, 3, 3)

        btn_cyl_calc = QPushButton("自动算间距")
        btn_cyl_calc.setStyleSheet("QPushButton { background: #e8e0d0; padding: 4px; } QPushButton:hover { background: #d8d0c0; }")
        btn_cyl_calc.clicked.connect(self._calc_cyl_spacing)
        cyl.addWidget(btn_cyl_calc, 4, 0, 1, 2)

        cyl.addWidget(QLabel("路径:"), 4, 2)
        self.cbo_cyl_type = QComboBox()
        self.cbo_cyl_type.addItems(["螺旋线", "Z字形"])
        cyl.addWidget(self.cbo_cyl_type, 4, 3)

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
        cyl.addWidget(btn_apply_cyl, 6, 0, 1, 4)

        self.btn_cyl_place = QPushButton("点击放置（右键确认生成）")
        self.btn_cyl_place.setStyleSheet("QPushButton { background: #d8e8d8; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c8d8c8; }")
        self.btn_cyl_place.clicked.connect(lambda: self._start_place_mode("cylinder"))
        cyl.addWidget(self.btn_cyl_place, 7, 0, 1, 4)
        self._route_stack.addWidget(tab_cyl)

        # -- Tab 4: 直线航线 --
        tab_line = QWidget()
        ll = QGridLayout(tab_line)
        ll.setSpacing(4)

        btn_pick_line = QPushButton("选择起终点")
        btn_pick_line.setStyleSheet("QPushButton { background: #e8e0d0; padding: 4px; font-weight: bold; } QPushButton:hover { background: #d8d0c0; }")
        btn_pick_line.clicked.connect(self._start_line_mode)
        ll.addWidget(btn_pick_line, 0, 0, 1, 4)

        ll.addWidget(QLabel("起点(x,y,z):"), 1, 0)
        self.edt_line_x1 = QLineEdit("0"); ll.addWidget(self.edt_line_x1, 1, 1)
        self.edt_line_y1 = QLineEdit("0"); ll.addWidget(self.edt_line_y1, 1, 2)
        self.edt_line_z1 = QLineEdit("5"); ll.addWidget(self.edt_line_z1, 1, 3)

        ll.addWidget(QLabel("终点(x,y,z):"), 2, 0)
        self.edt_line_x2 = QLineEdit("10"); ll.addWidget(self.edt_line_x2, 2, 1)
        self.edt_line_y2 = QLineEdit("0"); ll.addWidget(self.edt_line_y2, 2, 2)
        self.edt_line_z2 = QLineEdit("5"); ll.addWidget(self.edt_line_z2, 2, 3)

        ll.addWidget(QLabel("巡检距离(m):"), 3, 0)
        self.edt_line_inspect_dist = QLineEdit("3.0"); ll.addWidget(self.edt_line_inspect_dist, 3, 1)
        ll.addWidget(QLabel("速度(m/s):"), 3, 2)
        self.edt_line_speed = QLineEdit("1"); ll.addWidget(self.edt_line_speed, 3, 3)

        ll.addWidget(QLabel("航点间距:"), 4, 0)
        self.edt_line_spacing = QLineEdit("自动")
        self.edt_line_spacing.setReadOnly(True)
        self.edt_line_spacing.setStyleSheet("QLineEdit { background: #eee; color: #666; }")
        ll.addWidget(self.edt_line_spacing, 4, 1)

        btn_apply_line = QPushButton("应用")
        btn_apply_line.setStyleSheet("QPushButton { background: #d0d8e8; padding: 6px; font-weight: bold; } QPushButton:hover { background: #c0c8d8; }")
        btn_apply_line.clicked.connect(self.generate_line_route)
        ll.addWidget(btn_apply_line, 5, 0, 1, 4)

        self._route_stack.addWidget(tab_line)

        # -- Tab 5: 点状航线 --
        tab_inspect = QWidget()
        il = QGridLayout(tab_inspect)
        il.setSpacing(4)
        for c in range(4):
            il.setColumnStretch(c, 1)

        self.btn_inspect = QPushButton("选择点")
        self.btn_inspect.setStyleSheet("QPushButton { background: #e8e0d0; font-weight: bold; padding: 6px; } QPushButton:hover { background: #d8d0c0; }")
        self.btn_inspect.clicked.connect(self._start_inspect_mode)
        il.addWidget(self.btn_inspect, 0, 0, 1, 4)

        il.addWidget(QLabel("巡检点列表:"), 1, 0)
        self.lst_inspect = QListWidget()
        self.lst_inspect.setMaximumHeight(100)
        il.addWidget(self.lst_inspect, 2, 0, 1, 4)

        il.addWidget(QLabel("巡检距离(m):"), 3, 0)
        self.edt_inspect_dist = QLineEdit("3.0")
        self.edt_inspect_dist.setMaximumWidth(50)
        il.addWidget(self.edt_inspect_dist, 3, 1)
        il.addWidget(QLabel("速度(m/s):"), 3, 2)
        self.edt_inspect_speed = QLineEdit("1.0")
        self.edt_inspect_speed.setMaximumWidth(50)
        il.addWidget(self.edt_inspect_speed, 3, 3)

        self.btn_gen_inspect = QPushButton("生成点状航线")
        self.btn_gen_inspect.setStyleSheet("QPushButton { background: #d0e8d0; font-weight: bold; padding: 6px; } QPushButton:hover { background: #c0d8c0; }")
        self.btn_gen_inspect.clicked.connect(self.generate_inspect_route)
        il.addWidget(self.btn_gen_inspect, 4, 0, 1, 4)

        self._inspect_target_points = []  # 巡检目标点列表
        self._inspect_target_normals = []  # 每个目标点对应的表面法线
        self._line_start_normal = None     # 直线起点法线
        self._line_end_normal = None       # 直线终点法线
        self._polygon_normals = []         # 多边形顶点表面法线
        self._stl_triangles = None         # STL 三角面顶点数组，用于射线相交检测
        self._stl_triangles_np = None      # STL 三角面 numpy 数组，用于表面距离计算
        self._stl_distance_tree = None     # STL KDTree，用于碰撞检测
        self.viewer.inspect_points_confirmed.connect(self._on_inspect_confirmed)
        self.viewer.line_points_confirmed.connect(self._on_line_confirmed)
        self.viewer.line_point_picked.connect(self._on_line_point_picked)

        self._route_stack.addWidget(tab_inspect)

        self.btn_clear = QPushButton("清除")
        self.btn_clear.setStyleSheet("QPushButton { background: #e8d0d0; padding: 6px; } QPushButton:hover { background: #d8c0c0; }")
        ctrl_layout.addWidget(self.btn_clear)

        ctrl_layout.addWidget(self._route_stack)
        self._route_widgets.append(self._route_stack)

        # -- 航线管理 --
        grp_route = QGroupBox("航线管理")
        rl = QVBoxLayout(grp_route)
        self.lbl_info = QLabel("航点: 0")
        rl.addWidget(self.lbl_info)
        self.lbl_route_time = QLabel("")
        self.lbl_route_time.setStyleSheet("color: #555; font-size: 11px;")
        rl.addWidget(self.lbl_route_time)

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
        self.cmb_render_mode.currentTextChanged.connect(self._on_render_mode_changed)
        self.sld_point_size.valueChanged.connect(self._on_point_size_changed)
        self.btn_clear.clicked.connect(self.clear_route)

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

    def _on_route_type_changed(self, index):
        """航线类型切换时清理上一种类型的选择状态"""
        self._route_stack.setCurrentIndex(index)
        # 清理多边形选择状态
        if hasattr(self, '_polygon_vertices') and self._polygon_vertices:
            self._polygon_vertices = None
            self._polygon_normals = []
        self.viewer.exit_polygon_mode()
        # 清理直线选择状态
        self.viewer.exit_line_mode()
        # 清理巡检点选择状态
        self.viewer.exit_inspect_mode()

    # ─── 多边形选择 ──────────────────────────────────────────
    def _start_polygon_select(self):
        self.viewer.enter_polygon_mode()

    def _on_polygon_finished(self, pts):
        # pts 为 [(pos, normal), ...] 对
        positions = [p for p, n in pts]
        normals = [n for p, n in pts]
        poly = np.array(positions)
        mn = poly.min(axis=0)
        mx = poly.max(axis=0)
        self._polygon_vertices = positions
        # 存储多边形顶点的表面法线（拾取时的精确三角面法线）
        self._polygon_normals = [np.array(n) for n in normals]
        self.viewer._clear_polygon()
        self._calc_overlap_spacing()
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
        elif self._place_target == "cylinder":
            self.edt_cyl_cx.setText(f"{pos[0]:.1f}")
            self.edt_cyl_cy.setText(f"{pos[1]:.1f}")
            self.edt_cyl_cz.setText(f"{pos[2]:.1f}")
            self.generate_cylinder_route()
        self._place_target = None

    # ─── 加载点云 ───
    def load_point_cloud(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开点云文件", "", "点云文件 (*.pcd *.ply);;PCD 文件 (*.pcd);;PLY 文件 (*.ply);;所有文件 (*)"
        )
        if not path:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.statusBar().showMessage(f"正在加载 {os.path.basename(path)}...")
        QApplication.processEvents()

        try:
            self._point_colors = None
            self._point_normals = None
            ext = os.path.splitext(path)[1].lower()
            if ext == '.ply':
                self.points, self._point_colors = parse_ply(path)
            else:
                self.points = parse_pcd(path)

            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(60)
            QApplication.processEvents()

            # 过滤NaN/Inf点和极端哨兵值（如±float32_max）
            valid = np.isfinite(self.points).all(axis=1) & (np.abs(self.points) < 1e10).all(axis=1)
            if not valid.all():
                self.points = self.points[valid]
                if self._point_colors is not None:
                    self._point_colors = self._point_colors[valid]
                if self._point_normals is not None:
                    self._point_normals = self._point_normals[valid]

            n = len(self.points)
            colors = self._apply_color_scheme(self.points, self._point_colors)
            self.viewer.add_point_cloud(self.points, self._get_render_mode(), self._get_point_size(), colors=colors, use_lighting=(self._color_scheme == "original"))
            self._update_height_legend()
            self.progress_bar.setValue(80)
            QApplication.processEvents()

            self.statusBar().showMessage(f"已加载: {os.path.basename(path)} ({n:,} 点)")

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

    def load_stl_mesh(self):
        """加载STL/OBJ网格模型"""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开模型文件", "", "模型文件 (*.stl *.obj);;STL 文件 (*.stl);;OBJ 文件 (*.obj);;所有文件 (*)"
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext == '.obj':
            ok = self.viewer.add_obj_mesh(path)
        else:
            ok = self.viewer.add_stl_mesh(path)
        if ok:
            self.statusBar().showMessage(f"已加载模型: {os.path.basename(path)}")
            # 预计算三角面顶点数组（用于射线穿模检测）
            self._precompute_stl_triangles()
            # STL 不需要点云相关菜单
            self._render_menu.setEnabled(False)
            self._upsample_menu.setEnabled(False)
            self._color_menu.setEnabled(False)
        else:
            QMessageBox.warning(self, "错误", "加载模型失败")

    def _ensure_normals(self):
        """确保法线已估算（延迟执行，只在需要时调用）"""
        if self._point_normals is not None or self.points is None or len(self.points) == 0:
            return
        n = len(self.points)
        self.statusBar().showMessage(f"正在估算法线 ({n:,} 点)...")
        QApplication.processEvents()
        k = min(20, n - 1)
        self._point_normals = self.viewer.estimate_normals(self.points, k=k)
        self.statusBar().showMessage(f"法线估算完成")

    # ─── Z值过滤 ───
    def _get_render_mode(self):
        """获取当前渲染模式: 'auto'/'sphere'/'cube'/'pixel'/'splat'"""
        text = self.cmb_render_mode.currentText()
        return {"自动": "auto", "球体": "sphere", "立方体": "cube", "像素": "pixel", "圆片": "splat"}.get(text, "auto")

    def _get_point_size(self):
        return self.sld_point_size.value() * 0.01

    def _on_point_size_changed(self, val):
        self.lbl_point_size.setText(f"{val * 0.01:.2f}")
        # 同步菜单选中状态
        act = self._size_acts.get(val)
        if act:
            act.setChecked(True)
        else:
            # 滑块值不在菜单预设中时，取消所有菜单选中
            for a in self._size_acts.values():
                a.setChecked(False)
        self._refresh_point_cloud()

    def _update_height_legend(self):
        """如果当前是高度着色模式，显示高程图例"""
        if self._color_scheme == "height" and self.points is not None and len(self.points) > 0:
            z_vals = self.points[:, 2]
            self.viewer.show_height_legend(float(z_vals.min()), float(z_vals.max()))
        else:
            self.viewer._remove_legend()

    def _on_color_scheme_changed(self, scheme):
        self._color_scheme = scheme
        self._refresh_point_cloud()

    def _apply_color_scheme(self, points, colors):
        """根据颜色方案处理颜色，返回 (N,3) uint8 颜色数组"""
        n = len(points)
        scheme = self._color_scheme

        if scheme == "original":
            if colors is not None:
                return colors
            scheme = "height"  # 无原色时回退到高度着色

        if scheme == "height":
            z_vals = points[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            z_range = z_max - z_min if z_max > z_min else 1.0
            t = (z_vals - z_min) / z_range
            r = np.zeros(n, dtype=np.uint8)
            g = np.zeros(n, dtype=np.uint8)
            b = np.zeros(n, dtype=np.uint8)
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
            return np.column_stack([r, g, b])

        if scheme == "thermal":
            z_vals = points[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            z_range = z_max - z_min if z_max > z_min else 1.0
            t = (z_vals - z_min) / z_range
            r = np.clip(t * 2 * 255, 0, 255).astype(np.uint8)
            g = np.clip((t - 0.5) * 2 * 255, 0, 255).astype(np.uint8)
            b = np.clip((1.0 - t) * 255, 0, 255).astype(np.uint8)
            return np.column_stack([r, g, b])

        if scheme == "grayscale":
            if colors is not None:
                gray = (0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]).astype(np.uint8)
            else:
                z_vals = points[:, 2]
                z_min, z_max = z_vals.min(), z_vals.max()
                z_range = z_max - z_min if z_max > z_min else 1.0
                gray = ((z_vals - z_min) / z_range * 255).astype(np.uint8)
            return np.column_stack([gray, gray, gray])

        color_map = {"red": (220, 50, 50), "green": (50, 180, 50), "blue": (50, 100, 220)}
        if scheme in color_map:
            c = color_map[scheme]
            return np.tile(np.array(c, dtype=np.uint8), (n, 1))

        return colors if colors is not None else np.tile(np.array([180, 180, 180], dtype=np.uint8), (n, 1))

    def _on_render_mode_changed(self, text):
        self._refresh_point_cloud()

    def _on_upsample_changed(self, factor):
        self._upsample_factor = factor
        self._refresh_point_cloud_with_upsample()

    def _refresh_point_cloud_with_upsample(self):
        """带增密的刷新"""
        if self.points is None or len(self.points) == 0:
            return

        display_points = self.points
        display_colors = self._apply_color_scheme(self.points, self._point_colors)
        display_normals = self._point_normals

        # 增密需要法线
        if self._upsample_factor > 0:
            self._ensure_normals()
            display_normals = self._point_normals

        # 如果需要增密，且法线已估算
        if self._upsample_factor > 0 and self._point_normals is not None:
            factors = {1: 2, 2: 5, 3: 10}
            n_new = factors.get(self._upsample_factor, 0)
            if n_new > 0:
                display_points, display_colors = self.viewer.upsample_for_display(
                    self.points, self._point_normals, self._point_colors, factor=n_new
                )
                # 增密后的法线（重复原始法线）
                if self._point_normals is not None:
                    display_normals = np.vstack([self._point_normals] * (n_new + 1))

        if any(self._clip_enabled.values()):
            # 裁剪模式下也需要处理增密
            p = display_points
            mask = np.ones(len(p), dtype=bool)
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            for axis in ['x', 'y', 'z']:
                if self._clip_enabled[axis]:
                    idx = axis_map[axis]
                    pos = self._clip_positions[axis]
                    mask &= (p[:, idx] <= pos)
            filtered = p[mask]
            filtered_colors = display_colors[mask] if display_colors is not None else None
            filtered_normals = display_normals[mask] if display_normals is not None else None
            if len(filtered) > 0:
                self.viewer.add_point_cloud(filtered, self._get_render_mode(), self._get_point_size(), colors=filtered_colors, normals=filtered_normals, reset_camera=False, use_lighting=(self._color_scheme == "original"))
                self._update_height_legend()
        else:
            self.viewer.add_point_cloud(display_points, self._get_render_mode(), self._get_point_size(), colors=display_colors, normals=display_normals, reset_camera=False, use_lighting=(self._color_scheme == "original"))
            self._update_height_legend()

    def _refresh_point_cloud(self):
        if any(self._clip_enabled.values()):
            self._apply_clip()
        elif self.points is not None:
            # 圆片模式需要法线
            if self._get_render_mode() == 'splat':
                self._ensure_normals()
            colors = self._apply_color_scheme(self.points, self._point_colors)
            self.viewer.add_point_cloud(self.points, self._get_render_mode(), self._get_point_size(), colors=colors, normals=self._point_normals, reset_camera=False, use_lighting=(self._color_scheme == "original"))
            self._update_height_legend()

    def _on_menu_render_mode(self, name):
        """菜单栏渲染模式切换"""
        self.cmb_render_mode.setCurrentText(name)

    def _on_menu_point_size(self, val):
        """菜单栏点云大小切换"""
        self.sld_point_size.setValue(val)

    def _toggle_fpv_mode(self):
        """切换FPV无人机视角模式"""
        has_cloud = self.points is not None and len(self.points) > 0
        has_stl = self.viewer._stl_actor is not None
        if not has_cloud and not has_stl:
            QMessageBox.information(self, "提示", "请先加载点云或模型")
            self._act_fpv.setChecked(False)
            return

        enable = self._act_fpv.isChecked()
        if enable:
            # 设置起始位置和打点回调
            self.viewer._fpv_start_pos = self._fpv_start_pos
            self.viewer._fpv_on_mark = self._fpv_mark_waypoint
            self.viewer.toggle_fpv(True)
            self.statusBar().showMessage("FPV模式：WASD移动，QE升降，右键控制视角，左键选点，空格打点，V退出")
        else:
            self.viewer.toggle_fpv(False)
            self.statusBar().showMessage("已退出FPV模式")

    def _fpv_mark_waypoint(self, target_point, drone_pos, yaw, pitch):
        """FPV模式下打点记录航点"""
        # target_point是相机中心与点云的交点
        # drone_pos是无人机位置
        # 这里可以将target_point作为航点添加到航线中
        x, y, z = target_point
        self.statusBar().showMessage(
            f"FPV打点: 目标({x:.1f}, {y:.1f}, {z:.1f}) "
            f"无人机({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f}) "
            f"偏航{yaw:.0f}° 俯仰{pitch:.0f}°"
        )
        # TODO: 将航点添加到航线系统中

    def _set_fpv_start_pos(self):
        """无人机设置：起始位置和飞行速度"""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QGridLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("无人机设置")
        dlg.setMinimumWidth(280)
        layout = QGridLayout(dlg)

        layout.addWidget(QLabel("── 起始位置 ──"), 0, 0, 1, 2)

        layout.addWidget(QLabel("X(m):"), 1, 0)
        edt_x = QLineEdit(f"{self._fpv_start_pos[0]:.1f}")
        layout.addWidget(edt_x, 1, 1)

        layout.addWidget(QLabel("Y(m):"), 2, 0)
        edt_y = QLineEdit(f"{self._fpv_start_pos[1]:.1f}")
        layout.addWidget(edt_y, 2, 1)

        layout.addWidget(QLabel("Z(m):"), 3, 0)
        edt_z = QLineEdit(f"{self._fpv_start_pos[2]:.1f}")
        layout.addWidget(edt_z, 3, 1)

        layout.addWidget(QLabel("── 飞行速度 ──"), 4, 0, 1, 2)

        layout.addWidget(QLabel("速度(m/帧):"), 5, 0)
        edt_speed = QLineEdit(f"{self.viewer._fpv_speed:.1f}")
        layout.addWidget(edt_speed, 5, 1)
        lbl_hint = QLabel("60fps时实际速度=此值×60")
        lbl_hint.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(lbl_hint, 6, 0, 1, 2)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns, 7, 0, 1, 2)

        if dlg.exec_() == QDialog.Accepted:
            try:
                self._fpv_start_pos = [float(edt_x.text()), float(edt_y.text()), float(edt_z.text())]
                self.viewer._fpv_start_pos = self._fpv_start_pos
                self.viewer._fpv_speed = float(edt_speed.text())
                self.statusBar().showMessage(
                    f"无人机位置: ({self._fpv_start_pos[0]:.1f}, {self._fpv_start_pos[1]:.1f}, {self._fpv_start_pos[2]:.1f}) "
                    f"速度: {self.viewer._fpv_speed:.1f}m/帧"
                )
            except ValueError:
                pass

    def _show_bridge_dialog(self):
        """弹出桥梁参数对话框"""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("桥梁参数")
        dlg.setMinimumWidth(300)
        layout = QGridLayout(dlg)
        layout.setSpacing(6)

        layout.addWidget(QLabel("桥梁名称:"), 0, 0)
        edt_name = QLineEdit(self._bridge_name_val)
        layout.addWidget(edt_name, 0, 1, 1, 3)

        layout.addWidget(QLabel("桥型:"), 1, 0)
        cmb_type = QComboBox()
        cmb_type.addItems(["跨河桥", "跨线桥", "高架桥"])
        cmb_type.setCurrentIndex(self._bridge_type_val)
        layout.addWidget(cmb_type, 1, 1, 1, 3)

        layout.addWidget(QLabel("桥长(m):"), 2, 0)
        edt_len = QLineEdit(self._bridge_len_val)
        layout.addWidget(edt_len, 2, 1)
        layout.addWidget(QLabel("桥宽(m):"), 2, 2)
        edt_wid = QLineEdit(self._bridge_wid_val)
        layout.addWidget(edt_wid, 2, 3)

        layout.addWidget(QLabel("净空(m):"), 3, 0)
        edt_clr = QLineEdit(self._bridge_clr_val)
        layout.addWidget(edt_clr, 3, 1)
        layout.addWidget(QLabel("跨距(m):"), 3, 2)
        edt_span = QLineEdit(self._bridge_span_val)
        layout.addWidget(edt_span, 3, 3)

        chk_render = QCheckBox("渲染")
        chk_render.setChecked(self._bridge_render)
        layout.addWidget(chk_render, 4, 0, 1, 2)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns, 5, 0, 1, 4)

        if dlg.exec_() == QDialog.Accepted:
            self._bridge_name_val = edt_name.text()
            self._bridge_type_val = cmb_type.currentIndex()
            self._bridge_len_val = edt_len.text()
            self._bridge_wid_val = edt_wid.text()
            self._bridge_clr_val = edt_clr.text()
            self._bridge_span_val = edt_span.text()
            self._bridge_render = chk_render.isChecked()
            self._apply_bridge_params()
            self._update_bridge_info_label()

    def _update_bridge_info_label(self):
        """更新渲染区右上角的桥梁参数信息标签"""
        if self._bridge_render:
            type_names = ["跨河桥", "跨线桥", "高架桥"]
            type_name = type_names[self._bridge_type_val]
            text = (f"桥梁: {self._bridge_name_val}\n"
                    f"类型: {type_name}\n"
                    f"长: {self._bridge_len_val}m  宽: {self._bridge_wid_val}m\n"
                    f"净空: {self._bridge_clr_val}m  跨距: {self._bridge_span_val}m")
            self._bridge_info_label.setText(text)
            self._bridge_info_label.adjustSize()
            self._bridge_info_label.setVisible(True)
            self._position_bridge_info_label()
        else:
            self._bridge_info_label.setVisible(False)

    def _position_bridge_info_label(self):
        """定位桥梁信息标签到渲染区右上角"""
        vr = self.viewer.size()
        lb = self._bridge_info_label.size()
        self._bridge_info_label.move(vr.width() - lb.width() - 10, 10)

    def resizeEvent(self, event):
        """窗口大小变化时重新定位信息标签"""
        super().resizeEvent(event)
        if self._bridge_info_label.isVisible():
            self._position_bridge_info_label()

    def _show_camera_dialog(self):
        """弹出无人机参数对话框（相机+云台）"""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("无人机参数")
        dlg.setMinimumWidth(380)
        layout = QGridLayout(dlg)
        layout.setSpacing(6)

        # ── 相机参数 ──
        layout.addWidget(QLabel("── 相机 ──"), 0, 0, 1, 4)

        layout.addWidget(QLabel("相机型号:"), 1, 0)
        cmb_camera = QComboBox()
        cmb_camera.addItems(self._camera_fov_map.keys())
        cmb_camera.setCurrentText(self._camera_name)
        layout.addWidget(cmb_camera, 1, 1, 1, 3)

        layout.addWidget(QLabel("水平FOV(°):"), 2, 0)
        edt_fov = QLineEdit(str(self._camera_fov))
        edt_fov.setReadOnly(True)
        edt_fov.setStyleSheet("QLineEdit { background: #eee; color: #666; }")
        layout.addWidget(edt_fov, 2, 1)

        layout.addWidget(QLabel("变焦(倍):"), 2, 2)
        edt_zoom = QLineEdit(f"{self._camera_zoom:.1f}")
        layout.addWidget(edt_zoom, 2, 3)

        def _calc_effective_fov(default_fov, default_zoom, new_zoom):
            """default_fov: 默认变焦下的FOV, default_zoom: 默认变焦倍数, new_zoom: 目标变焦"""
            import math
            if new_zoom <= 0:
                new_zoom = 1.0
            # 先算出1倍时的基础FOV: base = 2*atan(tan(default/2) * default_zoom)
            base_half = math.atan(math.tan(math.radians(default_fov / 2)) * default_zoom)
            # 再算目标变焦下的FOV
            return math.degrees(2 * math.atan(math.tan(base_half) / new_zoom))

        def on_zoom_changed(text):
            try:
                zoom = float(text)
                if zoom <= 0:
                    return
                name = cmb_camera.currentText()
                default_fov = self._camera_fov_map.get(name, 80)
                default_zoom = self._camera_zoom_map.get(name, 1.0)
                eff_fov = _calc_effective_fov(default_fov, default_zoom, zoom)
                edt_fov.setText(f"{eff_fov:.1f}")
            except ValueError:
                pass
        edt_zoom.textChanged.connect(on_zoom_changed)

        # 各相机对应的云台限位 [min, max]
        gimbal_limits_map = {
            "M4T 广角": [-120.0, 30.0],
            "M4T 中长焦": [-90.0, 55.0],
            "M4T 长焦": [-90.0, 70.0],
        }

        def on_camera_changed(name):
            default_fov = self._camera_fov_map.get(name, 80)
            default_zoom = self._camera_zoom_map.get(name, 1.0)
            edt_zoom.setText(f"{default_zoom:.1f}")
            edt_fov.setText(f"{default_fov:.1f}")
            limits = gimbal_limits_map.get(name)
            if limits:
                edt_pitch_min.setText(f"{limits[0]:.1f}")
                edt_pitch_max.setText(f"{limits[1]:.1f}")
        cmb_camera.currentTextChanged.connect(on_camera_changed)

        layout.addWidget(QLabel("航向重叠(%):"), 3, 0)
        edt_fwd = QLineEdit(str(self._forward_overlap))
        layout.addWidget(edt_fwd, 3, 1)
        layout.addWidget(QLabel("旁向重叠(%):"), 3, 2)
        edt_side = QLineEdit(str(self._side_overlap))
        layout.addWidget(edt_side, 3, 3)

        layout.addWidget(QLabel("最小拍摄间隔(s):"), 4, 0)
        edt_min_interval = QLineEdit(f"{self._camera_min_interval:.1f}")
        layout.addWidget(edt_min_interval, 4, 1)
        lbl_interval_hint = QLabel("(M4T=0.7s)")
        lbl_interval_hint.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(lbl_interval_hint, 4, 2, 1, 2)

        # ── 云台参数 ──
        layout.addWidget(QLabel("── 云台 ──"), 5, 0, 1, 4)

        layout.addWidget(QLabel("偏航角(°):"), 6, 0)
        edt_yaw = QLineEdit(f"{self._gimbal_yaw:.1f}")
        layout.addWidget(edt_yaw, 6, 1)
        lbl_yaw_hint = QLabel("(0=与机头同向)")
        lbl_yaw_hint.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(lbl_yaw_hint, 6, 2, 1, 2)

        layout.addWidget(QLabel("俯仰限位(°):"), 7, 0)
        edt_pitch_min = QLineEdit(f"{self._gimbal_pitch_min:.1f}")
        layout.addWidget(edt_pitch_min, 7, 1)
        layout.addWidget(QLabel("~"), 7, 2)
        edt_pitch_max = QLineEdit(f"{self._gimbal_pitch_max:.1f}")
        layout.addWidget(edt_pitch_max, 7, 3)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns, 8, 0, 1, 4)

        if dlg.exec_() == QDialog.Accepted:
            self._camera_name = cmb_camera.currentText()
            self._camera_fov = float(edt_fov.text())
            self._camera_zoom = float(edt_zoom.text())
            self._camera_min_interval = float(edt_min_interval.text())
            self._forward_overlap = int(edt_fwd.text())
            self._side_overlap = int(edt_side.text())
            self._gimbal_yaw = float(edt_yaw.text())
            self._gimbal_pitch_min = float(edt_pitch_min.text())
            self._gimbal_pitch_max = float(edt_pitch_max.text())

    def _show_range_calculator_dialog(self):
        """弹出拍摄范围计算器对话框"""
        import math
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QSlider

        dlg = QDialog(self)
        dlg.setWindowTitle("拍摄范围计算器")
        dlg.setMinimumWidth(400)
        layout = QGridLayout(dlg)
        layout.setSpacing(8)

        # 相机型号选择
        layout.addWidget(QLabel("相机型号:"), 0, 0)
        cmb_camera = QComboBox()
        cmb_camera.addItems(self._camera_fov_map.keys())
        layout.addWidget(cmb_camera, 0, 1, 1, 3)

        # 广角端焦距（根据相机型号自动填充）
        layout.addWidget(QLabel("广角端焦距:"), 1, 0)
        lbl_wide_focal = QLabel("24 mm")
        layout.addWidget(lbl_wide_focal, 1, 1)

        # 传感器尺寸（对角线，根据FOV和焦距反推）
        layout.addWidget(QLabel("传感器尺寸:"), 1, 2)
        lbl_sensor = QLabel("0 mm")
        layout.addWidget(lbl_sensor, 1, 3)

        # 光学变焦倍数
        layout.addWidget(QLabel("光学变焦:"), 2, 0)
        sld_optical = QSlider(Qt.Horizontal)
        sld_optical.setRange(10, 1000)  # 1.0x ~ 100.0x
        sld_optical.setValue(10)
        layout.addWidget(sld_optical, 2, 1, 1, 2)
        lbl_optical = QLabel("1.0 x")
        layout.addWidget(lbl_optical, 2, 3)

        # 数码变焦倍数
        layout.addWidget(QLabel("数码变焦:"), 3, 0)
        edt_digital = QLineEdit("1.0")
        edt_digital.setMaximumWidth(80)
        layout.addWidget(edt_digital, 3, 1)
        layout.addWidget(QLabel("x"), 3, 2)

        # 目标距离
        layout.addWidget(QLabel("目标距离:"), 4, 0)
        edt_distance = QLineEdit("10")
        edt_distance.setMaximumWidth(80)
        layout.addWidget(edt_distance, 4, 1)
        layout.addWidget(QLabel("m"), 4, 2)

        # 分隔线
        from PyQt5.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line, 5, 0, 1, 4)

        # 计算结果
        layout.addWidget(QLabel("当前焦距:"), 6, 0)
        lbl_current_focal = QLabel("- mm")
        lbl_current_focal.setStyleSheet("QLabel { font-weight: bold; }")
        layout.addWidget(lbl_current_focal, 6, 1)

        layout.addWidget(QLabel("水平FOV:"), 6, 2)
        lbl_fov_result = QLabel("- °")
        lbl_fov_result.setStyleSheet("QLabel { font-weight: bold; }")
        layout.addWidget(lbl_fov_result, 6, 3)

        layout.addWidget(QLabel("拍摄范围:"), 7, 0)
        lbl_range = QLabel("- m × - m")
        lbl_range.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; }")
        layout.addWidget(lbl_range, 7, 1, 1, 3)

        # 相机焦距范围映射（广角端焦距, 最大光学变焦倍数）
        camera_focal_range = {
            "DJI Mavic 3E": (24, 1),
            "DJI Mavic 3T": (24, 1),
            "M4T 广角": (24, 1),
            "M4T 中长焦": (72, 1),
            "M4T 长焦": (24, 6.75),
            "DJI M350+P1(24mm)": (24, 1),
            "DJI M350+P1(35mm)": (35, 1),
            "DJI M350+P1(50mm)": (50, 1),
            "DJI M350+L2(雷达)": (24, 1),
            "自定义": (24, 10),
        }

        def calculate():
            """执行计算"""
            try:
                cam_name = cmb_camera.currentText()
                wide_focal, max_optical = camera_focal_range.get(cam_name, (24, 1))

                # 更新广角端焦距显示
                lbl_wide_focal.setText(f"{wide_focal} mm")

                # 计算传感器尺寸（用FOV和焦距反推）
                fov = self._camera_fov_map.get(cam_name, 80)
                fov_rad = math.radians(fov)
                # 传感器对角线 = 2 × 焦距 × tan(FOV/2)
                sensor_size = 2 * wide_focal * math.tan(fov_rad / 2)
                lbl_sensor.setText(f"{sensor_size:.1f} mm")

                # 更新滑块范围
                sld_optical.setRange(10, int(max_optical * 10))

                # 获取变焦倍数
                optical_zoom = sld_optical.value() / 10.0
                digital_zoom = float(edt_digital.text())
                total_zoom = optical_zoom * digital_zoom

                # 更新光学变焦显示
                lbl_optical.setText(f"{optical_zoom:.1f} x")

                # 计算当前焦距
                current_focal = wide_focal * total_zoom
                lbl_current_focal.setText(f"{current_focal:.1f} mm")

                # 计算FOV
                if current_focal > 0:
                    current_fov = 2 * math.degrees(math.atan(sensor_size / (2 * current_focal)))
                    lbl_fov_result.setText(f"{current_fov:.1f}°")
                else:
                    current_fov = 180
                    lbl_fov_result.setText("180°")

                # 计算拍摄范围
                distance = float(edt_distance.text())
                if current_fov < 180:
                    fov_rad = math.radians(current_fov)
                    range_size = 2 * distance * math.tan(fov_rad / 2)
                    lbl_range.setText(f"{range_size:.1f} m × {range_size:.1f} m")
                else:
                    lbl_range.setText("∞")

            except (ValueError, ZeroDivisionError):
                pass

        # 连接信号
        cmb_camera.currentTextChanged.connect(lambda: calculate())
        sld_optical.valueChanged.connect(lambda: calculate())
        edt_digital.textChanged.connect(lambda: calculate())
        edt_distance.textChanged.connect(lambda: calculate())

        # 快捷按钮
        from PyQt5.QtWidgets import QPushButton
        btn_row = QHBoxLayout()
        btn_wide = QPushButton("广角端")
        btn_wide.clicked.connect(lambda: sld_optical.setValue(10))
        btn_row.addWidget(btn_wide)
        btn_tele = QPushButton("长焦端")
        btn_tele.clicked.connect(lambda: sld_optical.setValue(sld_optical.maximum()))
        btn_row.addWidget(btn_tele)
        btn_current = QPushButton("当前相机")
        def use_current_camera():
            idx = cmb_camera.findText(self._camera_name)
            if idx >= 0:
                cmb_camera.setCurrentIndex(idx)
        btn_current.clicked.connect(use_current_camera)
        btn_row.addWidget(btn_current)
        layout.addLayout(btn_row, 8, 0, 1, 4)

        # 关闭按钮
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns, 9, 0, 1, 4)

        # 初始计算
        calculate()

        dlg.exec_()

    def _switch_language(self, lang):
        """切换界面语言"""
        if lang == self._lang:
            return
        self._lang = lang
        self._apply_language()

    # 内联 QLabel 的中文→英文映射（用于递归翻译）
    _INLINE_LABELS = {
        "工作模式": "Mode",
        "加载点云": "Point Cloud",
        "桥梁参数": "Bridge Params",
        "安全设置": "Safety",
        "航线管理": "Route Mgmt",
        "裁剪框": "Clip Box",
        "桥梁名称:": "Bridge Name:",
        "桥型:": "Type:",
        "跨河桥": "River Bridge", "跨线桥": "Overpass", "高架桥": "Viaduct",
        "桥长(m):": "Length(m):",
        "桥宽(m):": "Width(m):",
        "净空(m):": "Clearance(m):",
        "跨距(m):": "Span(m):",
        "安全距离(m):": "Safe Dist(m):",
        "起飞高度(m):": "Takeoff Z(m):",
        "起飞偏航角(°):": "Takeoff Yaw(°):",
        "安全点(x,y,z):": "Safe Pt(x,y,z):",
        "最低飞行Z值(m):": "Min Z(m):",
        "低于此值视为碰撞": "Below this = collision",
        "航线类型:": "Route Type:",
        "高度Z:": "Height Z:",
        "线间距:": "Line Spacing:",
        "航点距离:": "WP Spacing:",
        "速度(m/s):": "Speed(m/s):",
        "相机型号:": "Camera:",
        "水平FOV(°):": "H-FOV(°):",
        "航向重叠(%):": "Fwd Overlap(%):",
        "旁向重叠(%):": "Side Overlap(%):",
        "曲度:": "Curvature:",
        "底面中心(x,y,z):": "Center(x,y,z):",
        "长(X):": "Len(X):",
        "宽(Y):": "Wid(Y):",
        "高(Z):": "Ht(Z):",
        "巡检距离(m):": "Inspect Dist(m):",
        "水平步距:": "H Step:",
        "垂直步距:": "V Step:",
        "速度:": "Speed:",
        "起始角度(°):": "Start Angle(°):",
        "直径:": "Diameter:",
        "水平步距(°):": "H Step(°):",
        "路径:": "Path:",
        "螺旋线": "Spiral", "Z字形": "Zigzag",
        "起点(x,y,z):": "Start(x,y,z):",
        "终点(x,y,z):": "End(x,y,z):",
        "巡检点列表:": "Inspect Points:",
        "m": "m",
        "自动": "Auto", "球体": "Sphere", "立方体": "Cube", "像素": "Pixel",
        "启用": "Enable",
        "X:": "X:", "Y:": "Y:", "Z:": "Z:",
        "渲染:": "Render:",
        "大小:": "Size:",
        "快捷键: 1=俯视 2=正视 3=侧视 4=透视 5=仰视  Esc=取消":
            "Keys: 1=Top 2=Front 3=Side 4=Persp 5=Bottom  Esc=Cancel",
    }

    def _apply_language(self):
        """应用当前语言到所有UI文本"""
        t = self._T[self._lang]

        # ─── 菜单栏 ───
        self._file_menu.setTitle(t["menu_file"])
        self._act_load_pc.setText(t["act_load_pc"])
        self._act_save_ros.setText(t["act_save_ros"])
        self._act_load_route.setText(t["act_load_route"])
        self._act_copy_route.setText(t["act_copy_route"])
        self._act_export_maicro.setText(t["act_export_maicro"])
        self._act_copy_maicro.setText(t["act_copy_maicro"])
        self._view_menu.setTitle(t["menu_view"])
        self._act_clip_toggle.setText(t["act_clip"])
        self._render_menu.setTitle(t["menu_render"])
        # 更新渲染模式子菜单文本
        render_map = {"自动": "Auto", "球体": "Sphere", "立方体": "Cube", "像素": "Pixel", "圆片": "Splat"}
        if self._lang == "zh":
            render_map = {v: k for k, v in render_map.items()}
        for old_name, act in self._render_mode_acts.items():
            new_name = render_map.get(old_name, old_name)
            act.setText(new_name)
        self._size_menu.setTitle(t["menu_size"])
        # 更新增密子菜单文本
        upsample_map = {"关闭": "Off", "2倍": "2x", "5倍": "5x", "10倍": "10x"}
        if self._lang == "zh":
            upsample_map = {v: k for k, v in upsample_map.items()}
        for factor, act in self._upsample_acts.items():
            old_text = act.text()
            new_text = upsample_map.get(old_text, old_text)
            act.setText(new_text)
        self._upsample_menu.setTitle(t["menu_upsample"])

        color_map = {"原色": "Original", "高度着色": "Height", "热力图": "Thermal",
                     "灰度": "Grayscale", "纯红": "Red", "纯绿": "Green", "纯蓝": "Blue"}
        if self._lang == "zh":
            color_map = {v: k for k, v in color_map.items()}
        for scheme, act in self._color_acts.items():
            old_text = act.text()
            new_text = color_map.get(old_text, old_text)
            act.setText(new_text)
        self._color_menu.setTitle(t["menu_color"])
        self._lang_menu.setTitle(t["menu_lang"])
        self._settings_menu.setTitle(t["menu_settings"])
        self._act_bridge_params.setText(t["act_bridge_params"])
        self._act_camera_params.setText(t["act_camera_params"])
        self._act_range_calc.setText(t["act_range_calc"])
        self._act_fpv.setText(t["act_fpv"])
        self._act_fpv_pos.setText(t["act_fpv_pos"])

        self.setWindowTitle(t["win_title"])

        # 状态栏（仅在未加载点云时切换文本）
        status = self.statusBar().currentMessage()
        if "未加载" in status or "No point cloud" in status:
            self.statusBar().showMessage(t["lbl_pc_info"])

        # ─── 构建翻译映射 ───
        if self._lang == 'zh':
            # 英→中：用 _INLINE_LABELS 的反向映射
            text_map = {v: k for k, v in self._INLINE_LABELS.items()}
            # 已知控件的翻译
            known = {
                "Apply": "应用",
                "Place (Right-click to confirm)": "选择顶点（右键确认）",
                "Pick Points": "选择起终点",
                "Generate Line Route": "生成直线航线",
                "Select Points": "选择巡检点",
                "Clear": "清除",
                "Generate Inspect Route": "生成点状航线",
                "Clear Route": "清除",
                "Auto Calc": "自动算间距",
                "Show Heading": "显示机头方向",
                "Waypoints: 0": "航点: 0",
                "No point cloud loaded": "未加载点云",
            }
            text_map.update(known)
        else:
            # 中→英
            text_map = dict(self._INLINE_LABELS)
            known = {
                "应用": "Apply",
                "选择顶点（右键确认）": "Select Vertices (Right-click to confirm)",
                "选择起终点": "Pick Points",
                "生成直线航线": "Generate Line Route",
                "选择巡检点": "Select Points",
                "清除": "Clear",
                "生成点状航线": "Generate Inspect Route",
                "清除": "Clear",
                "自动算间距": "Auto Calc",
                "显示机头方向": "Show Heading",
                "显示云台方向": "Show Gimbal",
                "航线动画播放": "Route Animation",
                "暂停动画": "Pause Animation",
                "停止动画": "Stop Animation",
                "航点: 0": "Waypoints: 0",
                "未加载点云": "No point cloud loaded",
            }
            text_map.update(known)

        # ─── 递归遍历所有控件，批量替换文本 ───
        def _translate_widgets(widget):
            from PyQt5.QtWidgets import QGroupBox, QPushButton, QCheckBox, QLabel, QComboBox
            for child in widget.findChildren(QWidget):
                if isinstance(child, QGroupBox):
                    old = child.title()
                    if old in text_map:
                        child.setTitle(text_map[old])
                elif isinstance(child, (QPushButton, QCheckBox)):
                    old = child.text()
                    if old in text_map:
                        child.setText(text_map[old])
                elif isinstance(child, QLabel):
                    old = child.text()
                    if old in text_map:
                        child.setText(text_map[old])

        _translate_widgets(self.centralWidget())
        _translate_widgets(self.menuBar())

        # ComboBox 航线类型（需要特殊处理，因为 items 是列表）
        route_names = [t["route_flat"], t["route_cube"], t["route_cyl"],
                       t["route_line"], t["route_inspect"]]
        idx = self.cmb_route_type.currentIndex()
        self.cmb_route_type.clear()
        self.cmb_route_type.addItems(route_names)
        self.cmb_route_type.setCurrentIndex(idx)

        print(f"[Lang] Switched to {self._lang}")

    def _apply_clip(self):
        """应用裁剪：根据启用的轴和位置过滤点云"""
        if self.points is None or len(self.points) == 0:
            return
        p = self.points
        mask = np.ones(len(p), dtype=bool)
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        for axis in ['x', 'y', 'z']:
            if self._clip_enabled[axis]:
                idx = axis_map[axis]
                pos = self._clip_positions[axis]
                mask &= (p[:, idx] <= pos)
        filtered = p[mask]
        filtered_raw_colors = self._point_colors[mask] if self._point_colors is not None else None
        filtered_colors = self._apply_color_scheme(filtered, filtered_raw_colors)
        filtered_normals = self._point_normals[mask] if self._point_normals is not None else None
        if len(filtered) == 0:
            QMessageBox.information(self, "提示", "裁剪后无点云数据")
            return
        self.viewer.add_point_cloud(filtered, self._get_render_mode(), self._get_point_size(), colors=filtered_colors, normals=filtered_normals, reset_camera=False, use_lighting=(self._color_scheme == "original"))
        self._update_height_legend()
        n_total = len(self.points)
        n_show = len(filtered)
        self.statusBar().showMessage(f"已加载: {n_total:,} 点 (显示 {n_show:,})")

    def _show_clip_dialog(self):
        """弹出裁剪对话框"""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QSlider
        if self.points is None or len(self.points) == 0:
            QMessageBox.information(self, "提示", "请先加载点云")
            return

        mn = self.points.min(axis=0)
        mx = self.points.max(axis=0)

        dlg = QDialog(self)
        dlg.setWindowTitle("裁剪")
        dlg.setMinimumWidth(350)
        layout = QGridLayout(dlg)
        layout.setSpacing(8)

        # X轴裁剪
        chk_x = QCheckBox("X轴裁剪")
        chk_x.setChecked(self._clip_enabled['x'])
        layout.addWidget(chk_x, 0, 0)
        sld_x = QSlider(Qt.Horizontal)
        sld_x.setRange(0, 1000)
        x_val = self._clip_positions.get('x', mx[0])
        sld_x.setValue(int((x_val - mn[0]) / (mx[0] - mn[0]) * 1000) if mx[0] > mn[0] else 1000)
        sld_x.setEnabled(self._clip_enabled['x'])
        layout.addWidget(sld_x, 0, 1)
        lbl_x = QLabel(f"位置: {x_val:.1f}")
        layout.addWidget(lbl_x, 0, 2)

        # Y轴裁剪
        chk_y = QCheckBox("Y轴裁剪")
        chk_y.setChecked(self._clip_enabled['y'])
        layout.addWidget(chk_y, 1, 0)
        sld_y = QSlider(Qt.Horizontal)
        sld_y.setRange(0, 1000)
        y_val = self._clip_positions.get('y', mx[1])
        sld_y.setValue(int((y_val - mn[1]) / (mx[1] - mn[1]) * 1000) if mx[1] > mn[1] else 1000)
        sld_y.setEnabled(self._clip_enabled['y'])
        layout.addWidget(sld_y, 1, 1)
        lbl_y = QLabel(f"位置: {y_val:.1f}")
        layout.addWidget(lbl_y, 1, 2)

        # Z轴裁剪
        chk_z = QCheckBox("Z轴裁剪")
        chk_z.setChecked(self._clip_enabled['z'])
        layout.addWidget(chk_z, 2, 0)
        sld_z = QSlider(Qt.Horizontal)
        sld_z.setRange(0, 1000)
        z_val = self._clip_positions.get('z', mx[2])
        sld_z.setValue(int((z_val - mn[2]) / (mx[2] - mn[2]) * 1000) if mx[2] > mn[2] else 1000)
        sld_z.setEnabled(self._clip_enabled['z'])
        layout.addWidget(sld_z, 2, 1)
        lbl_z = QLabel(f"位置: {z_val:.1f}")
        layout.addWidget(lbl_z, 2, 2)

        # 滑块值变化时更新标签和平面
        def on_x_changed(val):
            pos = mn[0] + (mx[0] - mn[0]) * val / 1000
            lbl_x.setText(f"位置: {pos:.1f}")
            if chk_x.isChecked():
                self.viewer.set_clip_plane('x', pos)
        def on_y_changed(val):
            pos = mn[1] + (mx[1] - mn[1]) * val / 1000
            lbl_y.setText(f"位置: {pos:.1f}")
            if chk_y.isChecked():
                self.viewer.set_clip_plane('y', pos)
        def on_z_changed(val):
            pos = mn[2] + (mx[2] - mn[2]) * val / 1000
            lbl_z.setText(f"位置: {pos:.1f}")
            if chk_z.isChecked():
                self.viewer.set_clip_plane('z', pos)
        sld_x.valueChanged.connect(on_x_changed)
        sld_y.valueChanged.connect(on_y_changed)
        sld_z.valueChanged.connect(on_z_changed)

        # 勾选时显示/隐藏平面
        def on_x_toggled(checked):
            sld_x.setEnabled(checked)
            self.viewer.show_clip_plane('x', checked)
            if checked:
                pos = mn[0] + (mx[0] - mn[0]) * sld_x.value() / 1000
                self.viewer.set_clip_plane('x', pos)
        def on_y_toggled(checked):
            sld_y.setEnabled(checked)
            self.viewer.show_clip_plane('y', checked)
            if checked:
                pos = mn[1] + (mx[1] - mn[1]) * sld_y.value() / 1000
                self.viewer.set_clip_plane('y', pos)
        def on_z_toggled(checked):
            sld_z.setEnabled(checked)
            self.viewer.show_clip_plane('z', checked)
            if checked:
                pos = mn[2] + (mx[2] - mn[2]) * sld_z.value() / 1000
                self.viewer.set_clip_plane('z', pos)
        chk_x.toggled.connect(on_x_toggled)
        chk_y.toggled.connect(on_y_toggled)
        chk_z.toggled.connect(on_z_toggled)

        # 确定/取消按钮
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(btns, 3, 0, 1, 3)

        def on_accept():
            self._clip_enabled['x'] = chk_x.isChecked()
            self._clip_enabled['y'] = chk_y.isChecked()
            self._clip_enabled['z'] = chk_z.isChecked()
            if chk_x.isChecked():
                self._clip_positions['x'] = mn[0] + (mx[0] - mn[0]) * sld_x.value() / 1000
            if chk_y.isChecked():
                self._clip_positions['y'] = mn[1] + (mx[1] - mn[1]) * sld_y.value() / 1000
            if chk_z.isChecked():
                self._clip_positions['z'] = mn[2] + (mx[2] - mn[2]) * sld_z.value() / 1000
            self._apply_clip()
            dlg.accept()

        def on_reject():
            # 恢复原始状态
            for axis in ['x', 'y', 'z']:
                self.viewer.show_clip_plane(axis, self._clip_enabled[axis])
                if self._clip_enabled[axis]:
                    self.viewer.set_clip_plane(axis, self._clip_positions[axis])
            dlg.reject()

        btns.accepted.connect(on_accept)
        btns.rejected.connect(on_reject)

        dlg.exec_()

    # ─── 生成平面航线（目标点驱动） ───
    def generate_flat_route(self):
        try:
            inspect_dist = float(self.edt_flat_inspect_dist.text())
            speed = float(self.edt_flat_speed.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效数字")
            return

        if inspect_dist <= 0:
            QMessageBox.warning(self, "输入错误", "巡检距离必须为正数")
            return

        safe_dist = self.viewer._safe_distance
        collision_dist = safe_dist * 1.5

        # 自动计算间距（如果还是默认的"自动"）
        if self.edt_spacing.text() == "自动" or self.edt_wp_spacing.text() == "自动":
            self._calc_overlap_spacing()
        try:
            spacing = float(self.edt_spacing.text())
            wp_spacing = float(self.edt_wp_spacing.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "间距计算失败，请手动设置")
            return

        if spacing == 0 or wp_spacing == 0:
            QMessageBox.warning(self, "提示", "间距不能为0")
            return

        if not hasattr(self, '_polygon_vertices') or not self._polygon_vertices:
            QMessageBox.warning(self, "提示", "请先选择多边形区域")
            return

        poly = np.array(self._polygon_vertices)

        # 多边形内判断（2D XY）
        def point_in_polygon_2d(points_2d, polygon):
            n = len(polygon)
            inside = np.zeros(len(points_2d), dtype=bool)
            j = n - 1
            for i in range(n):
                xi, yi = polygon[i]
                xj, yj = polygon[j]
                dy = yj - yi
                if abs(dy) > 1e-12:
                    cond = ((yi > points_2d[:, 1]) != (yj > points_2d[:, 1])) & \
                           (points_2d[:, 0] < (xj - xi) * (points_2d[:, 1] - yi) / dy + xi)
                    inside[cond] = ~inside[cond]
                j = i
            return inside

        # 计算多边形平面法线（叉积，朝向相机）
        if len(poly) >= 3:
            e1 = poly[1] - poly[0]
            e2 = poly[2] - poly[0]
            poly_normal = np.cross(e1, e2)
            nrm = np.linalg.norm(poly_normal)
            if nrm > 1e-10:
                poly_normal /= nrm
            else:
                poly_normal = np.array([0, 0, 1.0])
        else:
            poly_normal = np.array([0, 0, 1.0])
        cam = self.viewer.renderer.GetActiveCamera()
        cam_pos = np.array(cam.GetPosition(), dtype=float)
        poly_center = poly.mean(axis=0)
        if np.dot(poly_normal, cam_pos - poly_center) < 0:
            poly_normal = -poly_normal
        print(f"[FlatRoute] poly_normal=({poly_normal[0]:.3f},{poly_normal[1]:.3f},{poly_normal[2]:.3f})")

        # 确定扫描主方向（多边形最长边方向投影到多边形平面）
        n_poly = len(poly)
        max_len = 0
        raw_dir = np.array([1.0, 0.0, 0.0])
        for i in range(n_poly):
            edge = poly[(i + 1) % n_poly] - poly[i]
            length = np.linalg.norm(edge)
            if length > max_len:
                max_len = length
                raw_dir = edge / length
        # 移除法线分量，得到平面内的方向
        main_dir_3d = raw_dir - np.dot(raw_dir, poly_normal) * poly_normal
        nrm = np.linalg.norm(main_dir_3d)
        if nrm > 1e-10:
            main_dir_3d /= nrm
        else:
            main_dir_3d = np.array([1.0, 0.0, 0.0])
        # 扫描副方向 = 法线 × 主方向
        cross_dir_3d = np.cross(poly_normal, main_dir_3d)
        nrm = np.linalg.norm(cross_dir_3d)
        if nrm > 1e-10:
            cross_dir_3d /= nrm

        # 在多边形平面内生成均匀网格
        # 投影多边形顶点到 (main_dir, cross_dir) 2D 坐标系
        poly_2d = np.column_stack([
            np.dot(poly - poly_center, main_dir_3d),
            np.dot(poly - poly_center, cross_dir_3d)
        ])
        u_min, u_max = poly_2d[:, 0].min(), poly_2d[:, 0].max()
        v_min, v_max = poly_2d[:, 1].min(), poly_2d[:, 1].max()

        # 生成均匀网格点，过滤在多边形内的
        grid_targets = []
        grid_scan_dirs = []  # 每个目标点对应的扫描方向
        v = v_min + spacing * 0.5
        row_idx = 0
        while v <= v_max:
            row_u = np.arange(u_min + wp_spacing * 0.5, u_max, wp_spacing)
            if row_idx % 2 == 1:
                row_u = row_u[::-1]  # 蛇形扫描
                scan_dir = -main_dir_3d
            else:
                scan_dir = main_dir_3d
            for u in row_u:
                pt_2d = np.array([u, v])
                # 判断是否在多边形内
                inside = False
                j = len(poly_2d) - 1
                for i in range(len(poly_2d)):
                    yi, xi = poly_2d[i][1], poly_2d[i][0]
                    yj, xj = poly_2d[j][1], poly_2d[j][0]
                    if ((yi > pt_2d[1]) != (yj > pt_2d[1])) and \
                       (pt_2d[0] < (xj - xi) * (pt_2d[1] - yi) / (yj - yi) + xi):
                        inside = not inside
                    j = i
                if inside:
                    # 2D → 3D：在多边形平面上
                    target_3d = poly_center + u * main_dir_3d + v * cross_dir_3d
                    grid_targets.append(target_3d)
                    grid_scan_dirs.append(scan_dir)
            v += spacing
            row_idx += 1

        print(f"[FlatRoute] grid: {len(grid_targets)} targets, spacing={spacing:.2f}, wp_spacing={wp_spacing:.2f}")
        region_points = np.array(grid_targets) if grid_targets else np.empty((0, 3))

        if len(region_points) == 0:
            QMessageBox.warning(self, "提示", "多边形区域内无数据点")
            return

        # 获取KDTree用于碰撞检测
        tree = self._get_kdtree()

        self.waypoints = []
        warnings = []
        _up = np.array([0.0, 0.0, 1.0])
        self._check_speed_overlap(speed, wp_spacing, main_dir_3d, warnings)

        # 判断表面类型
        normal_up_dot = np.dot(poly_normal, _up)
        is_top_surface = normal_up_dot > 0.7
        is_bottom_surface = normal_up_dot < -0.7

        def _check_flat_candidate(pos_c, _target, _heading, _left):
            """检查候选位置（STL版/点云版）"""
            # C1: 碰撞检测
            if self._stl_triangles_np is not None:
                stl_dist = self._stl_surface_distance(pos_c)
                if stl_dist < collision_dist:
                    return False, f'collision_stl({stl_dist:.2f})'
            elif tree is not None:
                dist, _ = tree.query(pos_c)
                if dist < collision_dist:
                    return False, f'collision({dist:.2f})'
            # C2: 云台 pitch
            pitch = self._calc_gimbal_pitch(pos_c, _target, _heading)
            if not (self._gimbal_pitch_min <= pitch <= self._gimbal_pitch_max):
                return False, 'gimbal'
            # C5: LOS 在 Fwd-Up 平面内
            los = _target - pos_c
            if abs(np.dot(los, _left)) > 0.1:
                return False, 'gimbal_yaw'
            # C3: LOS 不穿模
            los_len = np.linalg.norm(los)
            if los_len > 0.5:
                if self.viewer._stl_polydata is not None and self._stl_triangles is not None:
                    if self._ray_stl_intersect(pos_c, _target):
                        return False, 'los_stl'
                elif tree is not None:
                    for tt in np.linspace(0.05, 0.95, max(5, int(los_len / 0.5))):
                        sample = pos_c + los * tt
                        d, _ = tree.query(sample)
                        if d < 0.3:
                            return False, 'los'
            return True, 'ok'

        for i, target in enumerate(region_points):
            normal = poly_normal.copy()
            scan_dir = grid_scan_dirs[i] if i < len(grid_scan_dirs) else main_dir_3d

            # 检查法线方向是否正确（仅点云模式需要，STL几何法线已正确）
            if tree is not None and self.viewer._stl_polydata is None:
                candidate_pos = target + normal * inspect_dist
                dist_to_surface, _ = tree.query(candidate_pos)
                if dist_to_surface < collision_dist:
                    normal = -normal

            # 螃蟹飞：heading = cross(航线方向水平分量, [0,0,1])
            route_dir_h = np.array([scan_dir[0], scan_dir[1], 0.0])
            rdh_len = np.linalg.norm(route_dir_h)
            if rdh_len > 1e-6:
                route_dir_h = route_dir_h / rdh_len
            else:
                route_dir_h = np.array([1.0, 0.0, 0.0])
            heading_h = np.cross(route_dir_h, _up)
            hn = np.linalg.norm(heading_h)
            if hn > 1e-6:
                heading_h = heading_h / hn
            else:
                heading_h = np.array([1.0, 0.0, 0.0])

            # Left 向量用于 C5 检查
            left = np.cross(heading_h, _up)
            left_len = np.linalg.norm(left)
            if left_len > 1e-6:
                left = left / left_len

            # 构建搜索方向（根据表面类型）
            if is_bottom_surface:
                pitch_max_rad = np.radians(self._gimbal_pitch_max)
                base_dirs = []
                for pitch_deg in [45, 35, 25, 50, 15, 10, 55]:
                    pitch_rad = np.radians(pitch_deg)
                    h_factor = np.cos(pitch_rad)
                    drop_factor = -np.sin(pitch_rad)
                    base_dirs.append(np.array([heading_h[0]*h_factor, heading_h[1]*h_factor, drop_factor]))
                    base_dirs.append(np.array([-heading_h[0]*h_factor, -heading_h[1]*h_factor, drop_factor]))
                base_dirs.extend([-_up, _up, heading_h, -heading_h])
                search_directions = [d / np.linalg.norm(d) for d in base_dirs if np.linalg.norm(d) > 1e-6]
            elif is_top_surface:
                base_dirs = [_up]
                for frac in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                    base_dirs.append(_up + heading_h * frac)
                    base_dirs.append(_up - heading_h * frac)
                base_dirs.extend([heading_h, -heading_h])
                search_directions = [d / np.linalg.norm(d) for d in base_dirs if np.linalg.norm(d) > 1e-6]
            else:
                normal_in_plane = np.dot(normal, heading_h) * heading_h + np.dot(normal, _up) * _up
                if np.linalg.norm(normal_in_plane) > 1e-6:
                    normal_in_plane = normal_in_plane / np.linalg.norm(normal_in_plane)
                else:
                    normal_in_plane = heading_h.copy()
                base_dirs = [normal_in_plane]
                for frac in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                    base_dirs.append(normal_in_plane + _up * frac)
                    base_dirs.append(normal_in_plane - _up * frac)
                base_dirs.extend([heading_h, -heading_h, _up, -_up])
                search_directions = [d / np.linalg.norm(d) for d in base_dirs if np.linalg.norm(d) > 1e-6]

            # 搜索满足约束的位置
            drone_pos = None
            for offset in np.arange(inspect_dist, inspect_dist + 20.0, 0.5):
                for move_dir in search_directions:
                    pos_c = target + move_dir * offset
                    ok, reason = _check_flat_candidate(pos_c, target, heading_h, left)
                    if ok:
                        drone_pos = pos_c
                        break
                if drone_pos is not None:
                    break

            warned = False
            if drone_pos is None:
                drone_pos = target + normal * inspect_dist
                warned = True
                warnings.append(f"目标{i+1} 无法满足所有约束")

            # 确保 heading 朝向目标（否则云台需要转180°）
            los_to_target = target - drone_pos
            final_heading = heading_h.copy()
            if np.dot(final_heading, los_to_target) < 0:
                final_heading = -final_heading

            # 云台俯仰角
            gimbal_pitch = self._calc_gimbal_pitch(drone_pos, target, final_heading)
            gimbal_pitch = np.clip(gimbal_pitch, self._gimbal_pitch_min, self._gimbal_pitch_max)

            # 机头方向
            quat = look_at_quaternion(drone_pos + final_heading, drone_pos)
            self.waypoints.append({
                'pos': drone_pos,
                'quat': quat,
                'speed': speed,
                'action': 'fly',
                'gimbal_pitch': gimbal_pitch,
                'target_pos': target.copy()
            })

        if not self.waypoints:
            QMessageBox.warning(self, "提示", "多边形区域内无有效航点")
            return
        print(f"[FlatRoute] main_dir=({main_dir_3d[0]:.3f},{main_dir_3d[1]:.3f},{main_dir_3d[2]:.3f}), spacing={spacing:.2f}, wp_spacing={wp_spacing:.2f}")
        print(f"[FlatRoute] generated {len(self.waypoints)} waypoints")
        for i, wp in enumerate(self.waypoints[:5]):
            t = wp.get('target_pos', wp['pos'])
            print(f"  wp{i}: pos=({wp['pos'][0]:.1f},{wp['pos'][1]:.1f},{wp['pos'][2]:.1f}) target=({t[0]:.1f},{t[1]:.1f},{t[2]:.1f})")

        self._display_route()

        if warnings:
            QMessageBox.warning(self, "约束警告",
                f"生成 {len(self.waypoints)} 个航点\n\n" + "\n".join(warnings))

    # ─── 生成圆柱体航线 ───
    def generate_cylinder_route(self):
        import math
        # 自动计算步距（如果还是默认的"自动"）
        if self.edt_cyl_astep.text() == "自动" or self.edt_cyl_vstep.text() == "自动":
            self._calc_cyl_spacing()
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
        tree = self._get_kdtree()
        safe_dist = self.viewer._safe_distance
        collision_dist = safe_dist * 1.5

        warnings = []
        up = np.array([0.0, 0.0, 1.0])

        def _check_cyl_candidate(pos_c, target_c):
            """检查候选位置是否满足所有约束（STL版/点云版）"""
            # C1: 碰撞检测
            if self._stl_triangles_np is not None:
                stl_dist = self._stl_surface_distance(pos_c)
                if stl_dist < collision_dist:
                    return False, f'collision_stl({stl_dist:.2f})'
            elif tree is not None:
                d, _ = tree.query(pos_c)
                if d < collision_dist:
                    return False, f'collision({d:.2f})'
            # C2: 云台 pitch
            pitch = self._calc_gimbal_pitch(pos_c, target_c)
            if not (self._gimbal_pitch_min <= pitch <= self._gimbal_pitch_max):
                return False, 'gimbal'
            # C3: LOS 不穿模
            los = target_c - pos_c
            los_len = np.linalg.norm(los)
            if los_len > 0.5:
                if self.viewer._stl_polydata is not None and self._stl_triangles is not None:
                    if self._ray_stl_intersect(pos_c, target_c):
                        return False, 'los_stl'
                elif tree is not None:
                    for tt in np.linspace(0.05, 0.95, max(5, int(los_len / 0.5))):
                        sample = pos_c + los * tt
                        d, _ = tree.query(sample)
                        if d < 0.3:
                            return False, 'los'
            return True, 'ok'

        # 检查速度vs拍摄间隔：水平方向（弧长步距）
        hfov = self._camera_fov
        h_cover = 2.0 * dist * math.tan(math.radians(hfov / 2.0))
        arc_step = h_cover * (1.0 - self._side_overlap / 100.0)
        self._check_speed_overlap(speed, arc_step, np.array([1.0, 0.0, 0.0]), warnings)

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
                cyl_center = np.array([cx, cy, z])
                outward = np.array([rx - cx, ry - cy, 0.0])
                out_norm = np.linalg.norm(outward)
                if out_norm > 1e-10:
                    outward = outward / out_norm
                else:
                    outward = np.array([1.0, 0.0, 0.0])

                # 综合约束检查（STL版/点云版）
                ok, reason = _check_cyl_candidate(pos, cyl_center)
                if not ok:
                    # 搜索安全位置：沿 outward 方向偏移
                    found = False
                    for offset in np.arange(0.5, 10.0, 0.5):
                        for direction in [outward, up, -up, outward + up*0.5, outward - up*0.5]:
                            d_norm = np.linalg.norm(direction)
                            if d_norm < 1e-6:
                                continue
                            cand = pos + direction / d_norm * offset
                            ok2, _ = _check_cyl_candidate(cand, cyl_center)
                            if ok2:
                                pos = cand
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        warnings.append(f"螺旋点{i+1} 无法满足所有约束")

                # 机头朝向圆柱中心（径向内法线）
                inward = np.array([cx - pos[0], cy - pos[1], 0.0])
                inward_norm = np.linalg.norm(inward)
                if inward_norm > 1e-10:
                    heading = inward / inward_norm
                else:
                    heading = np.array([1.0, 0.0, 0.0])
                target = pos + heading
                quat = look_at_quaternion(target, pos)
                gimbal_pitch = self._calc_gimbal_pitch(pos, cyl_center)
                gimbal_pitch = np.clip(gimbal_pitch, self._gimbal_pitch_min, self._gimbal_pitch_max)
                self.waypoints.append({
                    'pos': pos,
                    'quat': quat,
                    'speed': speed,
                    'action': 'scan',
                    'gimbal_pitch': gimbal_pitch,
                    'target_pos': cyl_center.copy()
                })

        elif route_type == "Z字形":
            num_cols = max(1, int(360 / max(1, astep)))
            num_layers = max(1, int(h / vstep))

            for col in range(num_cols):
                angle = start_angle + (col / num_cols) * 2 * np.pi
                rx = cx + radius * np.cos(angle)
                ry = cy + radius * np.sin(angle)

                outward = np.array([rx - cx, ry - cy, 0.0])
                out_norm = np.linalg.norm(outward)
                if out_norm > 1e-10:
                    outward = outward / out_norm
                else:
                    outward = np.array([1.0, 0.0, 0.0])

                # 机头朝向圆柱中心
                inward = -outward
                heading = inward.copy()

                # 偶数列：从下往上；奇数列：从上往下
                if col % 2 == 0:
                    layers_range = range(num_layers + 1)
                else:
                    layers_range = range(num_layers, -1, -1)

                for layer in layers_range:
                    z = cz + layer * vstep
                    pos = np.array([rx, ry, z])
                    cyl_center = np.array([cx, cy, z])

                    # 综合约束检查（STL版/点云版）
                    ok, reason = _check_cyl_candidate(pos, cyl_center)
                    if not ok:
                        found = False
                        for offset in np.arange(0.5, 10.0, 0.5):
                            for direction in [outward, up, -up, outward + up*0.5, outward - up*0.5]:
                                d_norm = np.linalg.norm(direction)
                                if d_norm < 1e-6:
                                    continue
                                cand = pos + direction / d_norm * offset
                                ok2, _ = _check_cyl_candidate(cand, cyl_center)
                                if ok2:
                                    pos = cand
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            warnings.append(f"Z字形点(col{col+1},layer{layer}) 无法满足所有约束")

                    target = pos + heading
                    quat = look_at_quaternion(target, pos)
                    gimbal_pitch = self._calc_gimbal_pitch(pos, cyl_center)
                    gimbal_pitch = np.clip(gimbal_pitch, self._gimbal_pitch_min, self._gimbal_pitch_max)
                    self.waypoints.append({
                        'pos': pos,
                        'quat': quat,
                        'speed': speed,
                        'action': 'scan',
                        'gimbal_pitch': gimbal_pitch,
                        'target_pos': cyl_center.copy()
                    })

        self._display_route()
        print(f"[Cylinder] Generated {len(self.waypoints)} waypoints ({route_type})")

        if warnings:
            QMessageBox.warning(self, "约束警告",
                f"生成 {len(self.waypoints)} 个航点\n\n" + "\n".join(warnings[:10]))

    # ─── 生成直线航线 ───
    def generate_line_route(self):
        import math
        try:
            x1 = float(self.edt_line_x1.text())
            y1 = float(self.edt_line_y1.text())
            z1 = float(self.edt_line_z1.text())
            x2 = float(self.edt_line_x2.text())
            y2 = float(self.edt_line_y2.text())
            z2 = float(self.edt_line_z2.text())
            inspect_dist = float(self.edt_line_inspect_dist.text())
            speed = float(self.edt_line_speed.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效数字")
            return

        if inspect_dist <= 0:
            QMessageBox.warning(self, "输入错误", "巡检距离必须为正数")
            return

        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])
        length = np.linalg.norm(p2 - p1)
        if length < 1e-10:
            QMessageBox.warning(self, "输入错误", "起点和终点不能重合")
            return

        # 航点间距 = 覆盖宽度 × (1 - 旁向重叠)
        route_dir = p2 - p1
        eff_fov = self._fov_for_direction(route_dir)
        cover = 2.0 * inspect_dist * math.tan(math.radians(eff_fov / 2.0))
        spacing = cover * (1.0 - self._side_overlap / 100.0)
        spacing = max(0.1, spacing)
        self.edt_line_spacing.setText(f"{spacing:.2f}")

        n_pts = max(2, int(length / spacing) + 1)
        tree = self._get_kdtree()
        safe_dist = self.viewer._safe_distance
        collision_dist = safe_dist * 1.5

        # 检查速度vs拍摄间隔
        warnings_line = []
        self._check_speed_overlap(speed, spacing, route_dir, warnings_line)
        if warnings_line:
            QMessageBox.warning(self, "拍摄间隔警告", "\n".join(warnings_line))

        # === 确定偏移方向：优先用拾取时存储的法线 ===
        mid_point = (p1 + p2) / 2.0
        if (hasattr(self, '_line_start_normal') and self._line_start_normal is not None
                and hasattr(self, '_line_end_normal') and self._line_end_normal is not None):
            # 插值法线：起点法线和终点法线之间插值
            normal = (self._line_start_normal + self._line_end_normal)
            nrm = np.linalg.norm(normal)
            if nrm > 1e-10:
                normal = normal / nrm
            else:
                normal = self._estimate_normal(mid_point)
        else:
            normal = self._estimate_normal(mid_point)

        # 检查法线方向是否正确（仅点云模式需要，STL几何法线已正确）
        if tree is not None and self.viewer._stl_polydata is None:
            candidate_pos = mid_point + normal * inspect_dist
            dist_to_surface, _ = tree.query(candidate_pos)
            if dist_to_surface < collision_dist:
                normal = -normal

        # 螃蟹飞：机头垂直于航线方向
        route_h = np.array([route_dir[0], route_dir[1], 0.0])
        rhn = np.linalg.norm(route_h)
        if rhn > 1e-6:
            route_h = route_h / rhn
        else:
            route_h = np.array([1.0, 0.0, 0.0])
        crab_heading = np.cross(route_h, np.array([0.0, 0.0, 1.0]))
        chn = np.linalg.norm(crab_heading)
        if chn > 1e-6:
            crab_heading = crab_heading / chn
        else:
            crab_heading = np.array([1.0, 0.0, 0.0])

        self.waypoints = []
        warnings = []
        up = np.array([0.0, 0.0, 1.0])

        # Left 向量用于 C5 检查
        left = np.cross(crab_heading, up)
        left_len = np.linalg.norm(left)
        if left_len > 1e-6:
            left = left / left_len

        def _check_candidate(pos_c):
            """检查候选位置是否满足所有约束"""
            # C1: 碰撞检测（STL 版 / 点云版）
            if self._stl_triangles_np is not None:
                stl_dist = self._stl_surface_distance(pos_c)
                if stl_dist < collision_dist:
                    return False, f'collision_stl({stl_dist:.2f})'
            elif tree is not None:
                dist, _ = tree.query(pos_c)
                if dist < collision_dist:
                    return False, f'collision({dist:.2f})'
            # C2: 云台 pitch
            pitch = self._calc_gimbal_pitch(pos_c, target, crab_heading)
            if not (self._gimbal_pitch_min <= pitch <= self._gimbal_pitch_max):
                return False, 'gimbal'
            # C5: LOS 在 Fwd-Up 平面内
            los = target - pos_c
            if abs(np.dot(los, left)) > 0.1:
                return False, 'gimbal_yaw'
            # C3: LOS 不穿模
            los_len = np.linalg.norm(los)
            if los_len > 0.5:
                if self.viewer._stl_polydata is not None and self._stl_triangles is not None:
                    if self._ray_stl_intersect(pos_c, target):
                        return False, 'los_stl'
                elif tree is not None:
                    for tt in np.linspace(0.05, 0.95, max(5, int(los_len / 0.5))):
                        sample = pos_c + los * tt
                        d, _ = tree.query(sample)
                        if d < 0.3:
                            return False, 'los'
            return True, 'ok'

        # 判断表面类型
        normal_up_dot = np.dot(normal, up)
        is_top_surface = normal_up_dot > 0.7
        is_bottom_surface = normal_up_dot < -0.7

        for i in range(n_pts):
            t = i / (n_pts - 1)
            target = p1 + t * (p2 - p1)

            # 构建搜索方向（根据表面类型）
            if is_bottom_surface:
                # 底面：沿 crab_heading 方向 + 向下偏移
                pitch_max_rad = np.radians(self._gimbal_pitch_max)
                base_dirs = []
                for pitch_deg in [45, 35, 25, 50, 15, 10, 55]:
                    pitch_rad = np.radians(pitch_deg)
                    h_dist_factor = np.cos(pitch_rad)
                    drop_factor = -np.sin(pitch_rad)
                    base_dirs.append(np.array([crab_heading[0]*h_dist_factor, crab_heading[1]*h_dist_factor, drop_factor]))
                    base_dirs.append(np.array([-crab_heading[0]*h_dist_factor, -crab_heading[1]*h_dist_factor, drop_factor]))
                base_dirs.extend([-up, up, crab_heading, -crab_heading])
                search_directions = [d / np.linalg.norm(d) for d in base_dirs if np.linalg.norm(d) > 1e-6]
            elif is_top_surface:
                # 顶面：上方 + 对角线
                base_dirs = [up]
                for frac in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                    base_dirs.append(up + crab_heading * frac)
                    base_dirs.append(up - crab_heading * frac)
                base_dirs.extend([crab_heading, -crab_heading])
                search_directions = [d / np.linalg.norm(d) for d in base_dirs if np.linalg.norm(d) > 1e-6]
            else:
                # 垂直面：沿法线 + 对角线
                normal_in_plane = np.dot(normal, crab_heading) * crab_heading + np.dot(normal, up) * up
                if np.linalg.norm(normal_in_plane) > 1e-6:
                    normal_in_plane = normal_in_plane / np.linalg.norm(normal_in_plane)
                else:
                    normal_in_plane = crab_heading.copy()
                base_dirs = [normal_in_plane]
                for frac in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                    base_dirs.append(normal_in_plane + up * frac)
                    base_dirs.append(normal_in_plane - up * frac)
                base_dirs.extend([crab_heading, -crab_heading, up, -up])
                search_directions = [d / np.linalg.norm(d) for d in base_dirs if np.linalg.norm(d) > 1e-6]

            # 搜索满足约束的位置
            drone_pos = None
            for offset in np.arange(inspect_dist, inspect_dist + 20.0, 0.5):
                for move_dir in search_directions:
                    pos_c = target + move_dir * offset
                    ok, reason = _check_candidate(pos_c)
                    if ok:
                        drone_pos = pos_c
                        break
                if drone_pos is not None:
                    break

            warned = False
            if drone_pos is None:
                drone_pos = target + normal * inspect_dist
                warned = True
                warnings.append(f"航点{i+1} 无法满足所有约束")

            # 确保 heading 朝向目标（否则云台需要转180°）
            los_to_target = target - drone_pos
            heading_used = crab_heading.copy()
            if np.dot(heading_used, los_to_target) < 0:
                heading_used = -heading_used

            # 云台俯仰角
            gimbal_pitch = self._calc_gimbal_pitch(drone_pos, target, heading_used)
            gimbal_pitch = np.clip(gimbal_pitch, self._gimbal_pitch_min, self._gimbal_pitch_max)

            # 机头方向：螃蟹飞（垂直于航线）
            quat = look_at_quaternion(drone_pos + heading_used, drone_pos)
            self.waypoints.append({
                'pos': drone_pos,
                'quat': quat,
                'speed': speed,
                'action': 'fly',
                'gimbal_pitch': gimbal_pitch,
                'target_pos': target.copy()
            })

        self._display_route()
        print(f"[Line] Generated {len(self.waypoints)} waypoints, spacing={spacing:.2f}m, inspect_dist={inspect_dist}m")
        if warnings:
            QMessageBox.warning(self, "约束警告",
                f"生成 {len(self.waypoints)} 个航点\n\n" + "\n".join(warnings[:10]))

    # ─── 巡检点功能 ─────────────────────────────────────────
    def _start_inspect_mode(self):
        self.viewer.enter_inspect_mode()

    def _on_inspect_confirmed(self, pts):
        """巡检点选点确认回调，pts 为 [(pos, normal), ...] 对"""
        self._inspect_target_points = [np.array(p) for p, n in pts]
        # 存储每个目标点对应的表面法线（拾取时精确计算的三角面法线）
        self._inspect_target_normals = [np.array(n) for p, n in pts]
        self.lst_inspect.clear()
        for i, p in enumerate(self._inspect_target_points):
            self.lst_inspect.addItem(f"P{i+1}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
        print(f"[Inspect] {len(self._inspect_target_points)} inspection points confirmed")

    def _clear_inspect_points(self):
        self._inspect_target_points.clear()
        self._inspect_target_normals.clear()
        self.lst_inspect.clear()
        self.viewer._clear_inspect_points()
        self.viewer.vtk_widget.GetRenderWindow().Render()

    def _start_line_mode(self):
        self.viewer.enter_line_mode()

    def _on_line_point_picked(self, idx, pos):
        """直线选点实时更新坐标字段"""
        if idx == 0:
            self.edt_line_x1.setText(f"{pos[0]:.1f}")
            self.edt_line_y1.setText(f"{pos[1]:.1f}")
            self.edt_line_z1.setText(f"{pos[2]:.1f}")
        elif idx == 1:
            self.edt_line_x2.setText(f"{pos[0]:.1f}")
            self.edt_line_y2.setText(f"{pos[1]:.1f}")
            self.edt_line_z2.setText(f"{pos[2]:.1f}")
            # 两点都选好后自动计算航点间距
            try:
                import math
                inspect_dist = float(self.edt_line_inspect_dist.text())
                p1 = np.array([float(self.edt_line_x1.text()), float(self.edt_line_y1.text()), float(self.edt_line_z1.text())])
                p2 = np.array([float(self.edt_line_x2.text()), float(self.edt_line_y2.text()), float(self.edt_line_z2.text())])
                eff_fov = self._fov_for_direction(p2 - p1)
                cover = 2.0 * inspect_dist * math.tan(math.radians(eff_fov / 2.0))
                spacing = cover * (1.0 - self._side_overlap / 100.0)
                self.edt_line_spacing.setText(f"{spacing:.2f}")
            except ValueError:
                pass

    def _on_line_confirmed(self, pts):
        """直线起终点选点确认回调，pts 为 [(pos, normal), ...]"""
        if len(pts) == 2:
            s, s_normal = np.array(pts[0][0]), np.array(pts[0][1])
            e, e_normal = np.array(pts[1][0]), np.array(pts[1][1])
            # 存储法线供 generate_line_route 使用
            self._line_start_normal = s_normal
            self._line_end_normal = e_normal
            self.edt_line_x1.setText(f"{s[0]:.1f}")
            self.edt_line_y1.setText(f"{s[1]:.1f}")
            self.edt_line_z1.setText(f"{s[2]:.1f}")
            self.edt_line_x2.setText(f"{e[0]:.1f}")
            self.edt_line_y2.setText(f"{e[1]:.1f}")
            self.edt_line_z2.setText(f"{e[2]:.1f}")
            # 自动计算航点间距
            try:
                import math
                inspect_dist = float(self.edt_line_inspect_dist.text())
                eff_fov = self._fov_for_direction(e - s)
                cover = 2.0 * inspect_dist * math.tan(math.radians(eff_fov / 2.0))
                spacing = cover * (1.0 - self._side_overlap / 100.0)
                self.edt_line_spacing.setText(f"{spacing:.2f}")
            except ValueError:
                pass
            print(f"[Line] 起点({s[0]:.1f},{s[1]:.1f},{s[2]:.1f}) 终点({e[0]:.1f},{e[1]:.1f},{e[2]:.1f})")
            # 选完两个点后自动生成航线
            self.generate_line_route()

    def _vertical_fov(self):
        """根据水平FOV和画幅比例计算垂直FOV"""
        import math
        return math.degrees(2 * math.atan(math.tan(math.radians(self._camera_fov / 2.0)) / self._camera_aspect))

    def _fov_for_direction(self, route_dir):
        """根据航线方向选择合适的FOV：平行Z轴用垂直FOV，垂直Z轴用水平FOV"""
        import math
        d = np.array(route_dir, dtype=float)
        nrm = np.linalg.norm(d)
        if nrm < 1e-10:
            return self._camera_fov
        d = d / nrm
        # Z方向分量占比
        z_ratio = abs(d[2])
        hfov = self._camera_fov
        vfov = self._vertical_fov()
        # z_ratio=1 纯垂直 → 用vfov；z_ratio=0 纯水平 → 用hfov
        return hfov * (1.0 - z_ratio) + vfov * z_ratio

    def _check_speed_overlap(self, speed, wp_spacing, route_dir, warnings):
        """检查飞行速度是否满足最小拍摄间隔下的重叠率要求
        speed: 飞行速度 (m/s)
        wp_spacing: 航点间距 (m)
        route_dir: 航线方向向量（用于选择FOV）
        warnings: 警告列表（直接追加）
        返回: 实际可达重叠率 (%)
        """
        import math
        interval = self._camera_min_interval
        min_spacing = speed * interval  # 最小拍摄间距
        eff_fov = self._fov_for_direction(route_dir)
        inspect_dist = 3.0  # 参考距离，用于计算覆盖宽度
        cover = 2.0 * inspect_dist * math.tan(math.radians(eff_fov / 2.0))
        if cover < 0.01:
            return self._side_overlap
        # 实际重叠率 = 1 - 实际间距/覆盖宽度
        actual_spacing = max(min_spacing, wp_spacing)
        actual_overlap = max(0.0, (1.0 - actual_spacing / cover)) * 100.0
        if min_spacing > wp_spacing:
            warnings.append(
                f"速度{speed:.1f}m/s + 最小拍摄间隔{interval}s → "
                f"最小间距{min_spacing:.2f}m > 设计间距{wp_spacing:.2f}m，"
                f"实际旁向重叠率≈{actual_overlap:.0f}%（目标{self._side_overlap}%）"
            )
        return actual_overlap

    def _precompute_stl_triangles(self):
        """从 STL polydata 提取三角面顶点数组 + 距离计算数据"""
        poly = self.viewer._stl_polydata
        if poly is None:
            self._stl_triangles = None
            self._stl_distance_tree = None
            self._stl_triangles_np = None
            return
        pts = poly.GetPoints()
        n_cells = poly.GetNumberOfCells()
        verts = []
        for ci in range(n_cells):
            cell = poly.GetCell(ci)
            if cell.GetNumberOfPoints() != 3:
                continue
            p0 = pts.GetPoint(cell.GetPointId(0))
            p1 = pts.GetPoint(cell.GetPointId(1))
            p2 = pts.GetPoint(cell.GetPointId(2))
            verts.append([p0, p1, p2])
        if verts:
            self._stl_triangles = np.array(verts, dtype=np.float64)  # (M, 3, 3)
            print(f"[C3] 预计算 {len(verts)} 个三角面用于穿模检测")
        else:
            self._stl_triangles = None
        # 预计算 STL 顶点 KDTree（用于快速筛选候选三角面）
        stl_pts = getattr(self.viewer, '_stl_points_np', None)
        if stl_pts is not None and len(stl_pts) > 0:
            from scipy.spatial import cKDTree
            self._stl_distance_tree = cKDTree(stl_pts)
            # 构建 顶点→三角面 索引（用于精确距离计算）
            self._stl_triangles_np = np.array(verts, dtype=np.float64) if verts else None
            print(f"[C1] STL KDTree + 三角面索引已预计算 ({len(stl_pts)} 顶点, {len(verts)} 三角面)")
        else:
            self._stl_distance_tree = None
            self._stl_triangles_np = None

    def _point_to_triangle_dist(self, p, v0, v1, v2):
        """计算点 p 到三角面 (v0,v1,v2) 的最短距离（精确）"""
        # 三角面的两条边
        e0 = v1 - v0
        e1 = v2 - v0
        # p 相对于 v0 的向量
        d = v0 - p
        a = np.dot(e0, e0)
        b = np.dot(e0, e1)
        c = np.dot(e1, e1)
        dd = np.dot(d, e0)
        ee = np.dot(d, e1)
        det = a * c - b * b
        s = b * ee - c * dd
        t = b * dd - a * ee
        if s + t <= det:
            if s < 0:
                if t < 0:
                    # 区域4
                    if dd < 0:
                        t = 0
                        s = min(max(-dd / a, 0), 1)
                    else:
                        s = 0
                        t = min(max(-ee / c, 0), 1)
                else:
                    # 区域3
                    s = 0
                    t = min(max(-ee / c, 0), 1)
            elif t < 0:
                # 区域5
                t = 0
                s = min(max(-dd / a, 0), 1)
            else:
                # 区域0
                inv_det = 1.0 / det
                s *= inv_det
                t *= inv_det
        else:
            if s < 0:
                # 区域2
                tmp0 = b + dd
                tmp1 = c + ee
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2 * b + c
                    s = min(max(numer / denom, 0), 1)
                    t = 1 - s
                else:
                    s = 0
                    t = min(max(-ee / c, 0), 1)
            elif t < 0:
                # 区域6
                tmp0 = b + ee
                tmp1 = a + dd
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2 * b + c
                    t = min(max(numer / denom, 0), 1)
                    s = 1 - t
                else:
                    t = 0
                    s = min(max(-dd / a, 0), 1)
            else:
                # 区域1
                numer = c + ee - b - dd
                denom = a - 2 * b + c
                s = min(max(numer / denom, 0), 1)
                t = 1 - s
        closest = v0 + s * e0 + t * e1
        return np.linalg.norm(p - closest)

    def _stl_surface_distance(self, pos):
        """计算点 pos 到 STL 表面的精确最短距离
        遍历所有三角面，用包围盒快速过滤，精确计算点到三角面距离
        """
        if self._stl_triangles_np is None:
            return float('inf')
        pos = np.asarray(pos, dtype=np.float64)
        tri_verts = self._stl_triangles_np  # (M, 3, 3)
        min_dist = float('inf')
        for ti in range(len(tri_verts)):
            tri = tri_verts[ti]
            tri_min = tri.min(axis=0)
            tri_max = tri.max(axis=0)
            # 包围盒快速过滤：计算点到包围盒的距离
            bbox_dist_sq = 0.0
            for d in range(3):
                if pos[d] < tri_min[d]:
                    diff = tri_min[d] - pos[d]
                    bbox_dist_sq += diff * diff
                elif pos[d] > tri_max[d]:
                    diff = pos[d] - tri_max[d]
                    bbox_dist_sq += diff * diff
            # 如果包围盒距离已经 >= 当前最小距离，跳过
            if bbox_dist_sq >= min_dist * min_dist:
                continue
            # 精确计算点到三角面距离
            dist = self._point_to_triangle_dist(pos, tri[0], tri[1], tri[2])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _ray_stl_intersect(self, ray_origin, ray_target):
        """检测射线 origin→target 是否在到达目标前穿过 STL 模型
        返回 True = 穿模（LOS 被遮挡）
        使用 Möller–Trumbore 算法，向量化批量检测所有三角面
        """
        if self._stl_triangles is None:
            return False
        origin = np.asarray(ray_origin, dtype=np.float64)
        target = np.asarray(ray_target, dtype=np.float64)
        direction = target - origin
        ray_len = np.linalg.norm(direction)
        if ray_len < 1e-10:
            return False
        direction = direction / ray_len

        # 三角面顶点: (M, 3, 3)
        v0 = self._stl_triangles[:, 0, :]
        v1 = self._stl_triangles[:, 1, :]
        v2 = self._stl_triangles[:, 2, :]

        # Möller–Trumbore（向量化）
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(direction, edge2)
        a = np.einsum('ij,ij->i', edge1, h)

        valid = np.abs(a) > 1e-10
        if not valid.any():
            return False

        f = 1.0 / a[valid]
        s = origin - v0[valid]
        u = f * np.einsum('ij,ij->i', s, h[valid])

        # 逐元素检查重心坐标
        in_tri = (u >= 0.0) & (u <= 1.0)
        q = np.cross(s, edge1[valid])
        v = f * np.einsum('j,ij->i', direction, q)
        in_tri = in_tri & (v >= 0.0) & (u + v <= 1.0)

        if not in_tri.any():
            return False

        t = f[in_tri] * np.einsum('ij,ij->i', edge2[valid][in_tri], q[in_tri])
        t_ratio = t / ray_len
        hits_before_target = (t > 1e-6) & (t_ratio < 0.95)

        return hits_before_target.any()

    def _estimate_normal(self, point):
        """获取表面法线：优先STL几何法线，回退PCA估算"""
        # 优先：STL 网格真实几何法线（不翻转，用于区分顶面/底面）
        if self.viewer._stl_polydata is not None:
            normal = self.viewer.get_stl_geometric_normal(point)
            if normal is not None:
                return normal

        # 获取观察者位置（仅PCA回退时用于朝向判断）
        if self.viewer.fpv_mode:
            viewer_pos = np.array(self.viewer._fpv_pos, dtype=float)
        else:
            cam = self.viewer.renderer.GetActiveCamera()
            viewer_pos = np.array(cam.GetPosition(), dtype=float)

        # 回退：PCA 估算（优先用STL顶点，回退点云）
        stl_pts = getattr(self.viewer, '_stl_points_np', None)
        if stl_pts is not None and len(stl_pts) >= 3:
            pca_points = stl_pts
        elif self.points is not None and len(self.points) >= 3:
            pca_points = self.points
        else:
            return np.array([0.0, 0.0, 1.0])
        from scipy.spatial import cKDTree
        tree = cKDTree(pca_points)
        k = max(3, min(30, len(pca_points)))
        dists, idxs = tree.query(point, k=k)
        neighbors = pca_points[idxs]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / len(neighbors)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # 最小特征值 = 法线方向
        if np.dot(normal, viewer_pos - point) < 0:
            normal = -normal
        return normal

    def _find_safe_position(self, target, normal, tree, collision_dist, safe_dist, max_offset=10.0, heading=None):
        """根据约束条件直接计算航点位置。
        约束：巡检距离、安全距离、云台角度范围、视线无遮挡、yaw=0/roll=0。
        heading: 可选，传入时检查 C5（视线在 Forward-Up 平面内）
        """
        pitch_min = self._gimbal_pitch_min  # -90
        pitch_max = self._gimbal_pitch_max  # 55
        inspect_dist = safe_dist  # 巡检距离

        # 计算 Left 向量用于 C5 检查
        left_vec = None
        if heading is not None:
            left_vec = np.cross(heading, np.array([0.0, 0.0, 1.0]))
            left_len = np.linalg.norm(left_vec)
            if left_len > 1e-6:
                left_vec = left_vec / left_len

        def _check(pos):
            """综合检查：碰撞+云台+视线+C5，返回 (通过, 原因)"""
            # C2: 云台角度
            pitch = self._calc_gimbal_pitch(pos, target, heading)
            if not (pitch_min <= pitch <= pitch_max):
                return False, 'gimbal'
            # C1: 碰撞
            if tree is not None:
                dist, _ = tree.query(pos)
                if dist < collision_dist:
                    return False, 'collision'
            # C5: 视线必须在 Forward-Up 平面内（yaw=0, roll=0）
            if left_vec is not None:
                los = target - pos
                if abs(np.dot(los, left_vec)) > 0.1:
                    return False, 'gimbal_yaw'
            # C3: 视线不穿模（只检查中间段，阈值0.3m）
            if tree is not None:
                seg = target - pos
                seg_len = np.linalg.norm(seg)
                if seg_len > 1.0:
                    n_samples = max(5, int(seg_len / 0.5))
                    for t in np.linspace(0.05, 0.95, n_samples):
                        sample = pos + seg * t
                        d, _ = tree.query(sample)
                        if d < 0.3:
                            return False, 'los'
            return True, 'ok'

        # 构建候选方向：先法线偏转（斜向），最后才是纯法线
        directions = []
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(normal, up)) > 0.9:
            up = np.array([1.0, 0.0, 0.0])
        perp1 = np.cross(normal, up)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(normal, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        for angle_deg in [30, 45, 60, 15, 75]:
            angle_rad = np.radians(angle_deg)
            for sign in [1, -1]:
                d1 = normal * np.cos(angle_rad) + perp1 * np.sin(angle_rad) * sign
                directions.append(d1 / np.linalg.norm(d1))
                d2 = normal * np.cos(angle_rad) + perp2 * np.sin(angle_rad) * sign
                directions.append(d2 / np.linalg.norm(d2))
        directions.append(normal)

        # 在每个方向上，用巡检距离直接尝试
        for direction in directions:
            pos = target + direction * inspect_dist
            ok, reason = _check(pos)
            if ok:
                pitch = self._calc_gimbal_pitch(pos, target, heading)
                print(f"[FindSafe] target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f}) "
                      f"dir=({direction[0]:.2f},{direction[1]:.2f},{direction[2]:.2f}) "
                      f"dist={inspect_dist:.1f} pitch={pitch:.1f}° → OK")
                return pos, False

        # 巡检距离不行，沿法线逐步加大距离
        for offset in np.arange(inspect_dist + 1.0, max_offset + 1.0, 1.0):
            pos = target + normal * offset
            ok, reason = _check(pos)
            if ok:
                pitch = self._calc_gimbal_pitch(pos, target, heading)
                print(f"[FindSafe] target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f}) "
                      f"offset={offset:.1f} pitch={pitch:.1f}° → OK (extended)")
                return pos, False

        print(f"[FindSafe] target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f}) → FAILED")
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
        collision_dist = safe_dist * 1.5
        tree = self._get_kdtree()

        self.waypoints = []
        warnings = []

        # 检查速度vs拍摄间隔
        if len(self._inspect_target_points) >= 2:
            min_seg = min(np.linalg.norm(self._inspect_target_points[i+1] - self._inspect_target_points[i])
                         for i in range(len(self._inspect_target_points) - 1))
            route_dir_inspect = self._inspect_target_points[-1] - self._inspect_target_points[0]
            self._check_speed_overlap(1.0, min_seg, route_dir_inspect, warnings)

        up = np.array([0.0, 0.0, 1.0])

        for i, target in enumerate(self._inspect_target_points):
            target = np.asarray(target, dtype=np.float64)

            # ── 航线方向：用前后相邻点计算（非首尾连线）──
            if len(self._inspect_target_points) >= 2:
                if i == 0:
                    rd = self._inspect_target_points[1] - target
                elif i == len(self._inspect_target_points) - 1:
                    rd = target - self._inspect_target_points[-2]
                else:
                    rd = self._inspect_target_points[i+1] - self._inspect_target_points[i-1]
                rd_h = np.array([rd[0], rd[1], 0.0])
                rd_h_len = np.linalg.norm(rd_h)
                if rd_h_len > 1e-6:
                    route_dir_h = rd_h / rd_h_len
                else:
                    route_dir_h = np.array([1.0, 0.0, 0.0])
            else:
                route_dir_h = np.array([1.0, 0.0, 0.0])

            # ── 螃蟹飞 heading：⊥ 航线方向（水平面）──
            heading = np.cross(route_dir_h, up)
            hn = np.linalg.norm(heading)
            if hn > 1e-6:
                heading = heading / hn
            else:
                heading = np.array([1.0, 0.0, 0.0])
            # 确保 heading 朝向目标一侧
            to_target = target - np.zeros(3)  # 参考点暂用原点
            if np.dot(heading, np.array([to_target[0], to_target[1], 0.0])) < 0:
                heading = -heading

            # ── 法线 ──
            if i < len(self._inspect_target_normals):
                normal = self._inspect_target_normals[i]
            else:
                normal = self._estimate_normal(target)
            # 确保法线朝外（仅点云模式需要，STL几何法线已正确）
            if tree is not None and self.viewer._stl_polydata is None:
                candidate_pos = target + normal * inspect_dist
                dist_to_surface, _ = tree.query(candidate_pos)
                if dist_to_surface < collision_dist:
                    normal = -normal

            print(f"[Inspect] P{i+1} target=({target[0]:.2f},{target[1]:.2f},{target[2]:.2f}) "
                  f"normal=({normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}) "
                  f"heading=({heading[0]:.2f},{heading[1]:.2f},{heading[2]:.2f})")

            # ── 云台坐标系 ──
            left = np.cross(heading, up)
            left_len = np.linalg.norm(left)
            if left_len > 1e-6:
                left = left / left_len
            else:
                left = np.array([0.0, 1.0, 0.0])

            # ── 判断表面类型，选择搜索方向（C5: 必须在 Forward-Up 平面内）──
            normal_up_dot = np.dot(normal, up)
            is_top_surface = normal_up_dot > 0.7      # 法线朝上 → 顶面
            is_bottom_surface = normal_up_dot < -0.7   # 法线朝下 → 底面
            is_horizontal = is_top_surface or is_bottom_surface

            if is_top_surface:
                # 顶面：无人机在上方，LOS 朝下（在 Fwd-Up 平面内）
                # 纯 up 可能被立方体侧面挡住（STL距离太近）
                # 需要对角线方向：向上+水平远离边缘
                primary_dir = up.copy()
                # 对角线：多个角度的 向上+水平（避开立方体侧面）
                fallback_dirs = [
                    up + heading * 0.3, up - heading * 0.3,       # 浅角度
                    up + route_dir_h * 0.3, up - route_dir_h * 0.3,
                    up + heading * 0.7, up - heading * 0.7,       # 中角度
                    up + route_dir_h * 0.7, up - route_dir_h * 0.7,
                    up + heading * 1.5, up - heading * 1.5,       # 陡角度
                    up + route_dir_h * 1.5, up - route_dir_h * 1.5,
                    heading, -heading, route_dir_h, -route_dir_h,  # 纯水平
                ]
            elif is_bottom_surface:
                # 底面：无人机在下方，LOS 朝上
                # C5 要求 LOS 在 Fwd-Up 平面内（Left 分量=0）
                # → 无人机只能沿 heading 方向偏移（不能沿 route_dir 方向）
                # 用 -heading：无人机在目标前方下方，LOS=(heading方向, 向上)
                # pitch = atan2(drop, offset)，drop=offset*sin(pitch_max) 时 pitch 最大
                primary_dir = -heading.copy()  # 前方下方
                fallback_dirs = [heading, -up, up]
            else:
                # 垂直面：沿法线方向搜索（远离表面）
                # heading 可能平行于表面，不能用于远离表面
                # 将法线投影到 Fwd-Up 平面内（去掉 Left 分量）以满足 C5
                normal_along_heading = np.dot(normal, heading) * heading
                normal_along_up = np.dot(normal, up) * up
                normal_in_plane = normal_along_heading + normal_along_up
                if np.linalg.norm(normal_in_plane) > 1e-6:
                    primary_dir = normal_in_plane / np.linalg.norm(normal_in_plane)
                else:
                    # 法线完全沿 Left 方向，用 heading 作为回退
                    primary_dir = heading.copy()
                # 对角线方向：法线+上/下（绕过桥体结构）
                fallback_dirs = [
                    normal_in_plane + up * 0.5, normal_in_plane - up * 0.5,  # 浅对角
                    normal_in_plane + up * 1.0, normal_in_plane - up * 1.0,  # 中对角
                    normal_in_plane + up * 2.0, normal_in_plane - up * 2.0,  # 陡对角
                    heading, -heading, up, -up,  # 基础方向
                ]

            def _check_candidate(pos_c):
                """检查候选位置是否满足所有约束"""
                # C1: 碰撞检测（STL 版 / 点云版）
                if self._stl_triangles_np is not None:
                    # STL 版：精确计算到三角面的最短距离
                    stl_dist = self._stl_surface_distance(pos_c)
                    if stl_dist < collision_dist:
                        return False, f'collision_stl({stl_dist:.2f})'
                elif tree is not None:
                    # 点云版：用 KDTree 查询最近点距离
                    dist, _ = tree.query(pos_c)
                    if dist < collision_dist:
                        return False, f'collision({dist:.2f})'
                # C2: 云台 pitch 范围（严格拒绝，不 clip）
                pitch = self._calc_gimbal_pitch(pos_c, target, heading)
                if not (self._gimbal_pitch_min <= pitch <= self._gimbal_pitch_max):
                    return False, 'gimbal'
                # C5: 视线在 Forward-Up 平面内
                los = target - pos_c
                los_left = np.dot(los, left)
                if abs(los_left) > 0.1:
                    return False, 'gimbal_yaw'
                # C3: 视线不穿模
                los_len = np.linalg.norm(los)
                if los_len > 0.5:
                    if self.viewer._stl_polydata is not None and self._stl_triangles is not None:
                        # ── STL 射线相交检测（Möller–Trumbore 算法）──
                        # 检测 LOS 是否在到达目标前穿过 STL 三角面
                        if self._ray_stl_intersect(pos_c, target):
                            return False, 'los_stl'
                    elif tree is not None:
                        # ── 点云距离检测（fallback）──
                        n_samples = max(5, int(los_len / 0.5))
                        for t in np.linspace(0.05, 0.95, n_samples):
                            sample = pos_c + los * t
                            d, _ = tree.query(sample)
                            if d < 0.3:
                                return False, 'los'
                return True, 'ok'

            # ── C6 搜索：从 inspect_dist 开始，优先最近 ──
            best_pos = None

            if is_bottom_surface:
                # ── 底面专用搜索 ──
                # C5 要求 LOS 在 Fwd-Up 平面内 → 只能沿 heading 方向偏移
                # 无人机在目标下方+前方（-heading 方向），总距离 = offset
                # pitch = atan2(drop, h_dist)，其中 drop² + h_dist² = offset²
                # 所以 drop = offset * sin(pitch)，h_dist = offset * cos(pitch)
                pitch_max_rad = np.radians(self._gimbal_pitch_max)  # 55°
                search_debug = []
                for offset in np.arange(inspect_dist, inspect_dist + 10.0, 0.5):
                    # 尝试不同 pitch 角度（从 45° 开始，优先中等角度）
                    for pitch_deg in [45, 35, 25, 50, 15, 10, 55]:
                        pitch_rad = np.radians(pitch_deg)
                        h_dist = offset * np.cos(pitch_rad)  # 水平距离（沿 heading）
                        drop = -offset * np.sin(pitch_rad)   # 垂直距离（向下！无人机在目标下方）
                        # 无人机 = target + 前方偏移(-heading) + 向下偏移(-up)
                        # 例：offset=3, pitch=35° → h_dist=2.46, drop=-1.72
                        # pos = target + (-heading)*2.46 + up*(-1.72) → 在目标前下方
                        pos_c = target + (-heading) * h_dist + up * drop
                        ok, reason = _check_candidate(pos_c)
                        if not ok and len(search_debug) < 6:
                            search_debug.append(f"  offset={offset:.1f} pitch={pitch_deg}° → {reason}")
                        if ok:
                            best_pos = pos_c
                            print(f"[Inspect] P{i+1} 底面搜索成功: offset={offset:.1f} pitch={pitch_deg}° "
                                  f"pos=({pos_c[0]:.2f},{pos_c[1]:.2f},{pos_c[2]:.2f})")
                            break
                    if best_pos is not None:
                        break
                # 回退：沿 +heading 方向（无人机在目标后方下方）
                if best_pos is None:
                    for offset in np.arange(inspect_dist, inspect_dist + 10.0, 0.5):
                        for pitch_deg in [45, 35, 25, 50, 15, 10]:
                            pitch_rad = np.radians(pitch_deg)
                            h_dist = offset * np.cos(pitch_rad)
                            drop = -offset * np.sin(pitch_rad)
                            pos_c = target + heading * h_dist + up * drop
                            ok, reason = _check_candidate(pos_c)
                            if not ok and len(search_debug) < 10:
                                search_debug.append(f"  +heading offset={offset:.1f} pitch={pitch_deg}° → {reason}")
                            if ok:
                                best_pos = pos_c
                                print(f"[Inspect] P{i+1} 底面(+heading)搜索成功: offset={offset:.1f} pitch={pitch_deg}°")
                                break
                        if best_pos is not None:
                            break
                # 最终回退：直接向下（-up）
                if best_pos is None:
                    for offset in np.arange(inspect_dist, inspect_dist + 5.0, 0.5):
                        pos_c = target - up * offset
                        ok, reason = _check_candidate(pos_c)
                        if not ok and len(search_debug) < 14:
                            search_debug.append(f"  -up offset={offset:.1f} → {reason}")
                        if ok:
                            best_pos = pos_c
                            print(f"[Inspect] P{i+1} 底面(-up)搜索成功: offset={offset:.1f}")
                            break
                if best_pos is None:
                    print(f"[Inspect] P{i+1} 底面搜索全部失败:")
                    for line in search_debug:
                        print(line)
            else:
                # ── 通用搜索（顶面/垂直面）──
                all_dirs = [primary_dir] + fallback_dirs
                if not any(np.allclose(normal, d, atol=0.1) for d in all_dirs):
                    all_dirs.append(normal)

                search_attempts = 0
                for offset in np.arange(inspect_dist, inspect_dist + 20.0, 0.5):
                    for move_dir in all_dirs:
                        d_norm = np.linalg.norm(move_dir)
                        if d_norm < 1e-6:
                            continue
                        pos_c = target + move_dir / d_norm * offset
                        ok, reason = _check_candidate(pos_c)
                        if not ok and search_attempts < 5:
                            print(f"  [Search] offset={offset:.1f} dir=({move_dir[0]:.2f},{move_dir[1]:.2f},{move_dir[2]:.2f}) → {reason}")
                            search_attempts += 1
                        if ok:
                            best_pos = pos_c
                            print(f"[Inspect] P{i+1} 搜索成功: offset={offset:.1f} dir=({move_dir[0]:.2f},{move_dir[1]:.2f},{move_dir[2]:.2f})")
                            break
                    if best_pos is not None:
                        break

            warned = False
            if best_pos is None:
                pos = target + normal * inspect_dist
                warned = True
                warnings.append(f"P{i+1} 无法满足所有约束")
            else:
                pos = best_pos

            # 打印距离信息（调试）
            if self._stl_triangles_np is not None:
                stl_d = self._stl_surface_distance(pos)
                print(f"[Inspect] P{i+1} → pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) STL距离={stl_d:.2f}m 碰撞阈值={collision_dist:.1f}m")
            elif tree is not None:
                pc_d, _ = tree.query(pos)
                print(f"[Inspect] P{i+1} → pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) 点云距离={pc_d:.2f}m 碰撞阈值={collision_dist:.1f}m")
            else:
                print(f"[Inspect] P{i+1} → pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")

            # ── 机头方向：螃蟹飞（⊥ 航线）──
            # 确保 heading 朝向目标（否则云台需要转180°）
            los_to_target = target - pos
            if np.dot(heading, los_to_target) < 0:
                heading = -heading
            quat = look_at_quaternion(pos + heading, pos)
            gimbal_pitch = self._calc_gimbal_pitch(pos, target, heading)
            # pitch 超范围时记录警告（不 clip）
            if not (self._gimbal_pitch_min <= gimbal_pitch <= self._gimbal_pitch_max):
                warnings.append(f"P{i+1} pitch={gimbal_pitch:.1f}°超范围，已强制使用最近可行位置")
                gimbal_pitch = np.clip(gimbal_pitch, self._gimbal_pitch_min, self._gimbal_pitch_max)
            print(f"[Inspect] P{i+1} gimbal_pitch={gimbal_pitch:.1f}° heading=({heading[0]:.2f},{heading[1]:.2f},{heading[2]:.2f})")
            self.waypoints.append({
                'pos': pos,
                'quat': quat,
                'speed': 1.0,
                'action': 'scan',
                'gimbal_pitch': gimbal_pitch,
                'target_pos': target.copy()
            })

        self._display_route()

        if warnings:
            QMessageBox.warning(self, "碰撞警告",
                f"生成 {len(self.waypoints)} 个航点\n\n" + "\n".join(warnings))
        print(f"[Inspect] Generated {len(self.waypoints)} waypoints from {len(self._inspect_target_points)} targets")

    # ─── 生成立方体航线 ───
    def generate_cube_route(self):
        # 自动计算步距（如果还是默认的"自动"）
        if self.edt_cstep.text() == "自动" or self.edt_vstep.text() == "自动":
            self._calc_cube_spacing()
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
        tree = self._get_kdtree()
        safe_dist = self.viewer._safe_distance
        collision_dist = safe_dist * 1.5
        warnings = []

        # 检查速度vs拍摄间隔（水平方向沿边移动）
        self._check_speed_overlap(speed, cstep, np.array([1.0, 0.0, 0.0]), warnings)

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
                    cube_center = np.array([cx, cy, z])

                    outward = np.array([pos_2d[0] - cx, pos_2d[1] - cy, 0.0])
                    out_norm = np.linalg.norm(outward)
                    if out_norm > 1e-10:
                        outward = outward / out_norm
                    else:
                        outward = np.array([1.0, 0.0, 0.0])

                    # 综合约束检查：碰撞 + 云台角度 + 视线不穿模
                    need_search = False
                    if tree is not None:
                        dist, _ = tree.query(pos)
                        if dist < collision_dist:
                            need_search = True
                    pitch = self._calc_gimbal_pitch(pos, cube_center)
                    if not (self._gimbal_pitch_min <= pitch <= self._gimbal_pitch_max):
                        need_search = True
                    if not need_search and tree is not None:
                        seg = cube_center - pos
                        seg_len = np.linalg.norm(seg)
                        if seg_len > 1.0:
                            for tt in np.linspace(0.05, 0.95, max(5, int(seg_len / 0.5))):
                                sample = pos + seg * tt
                                d, _ = tree.query(sample)
                                if d < 0.3:
                                    need_search = True
                                    break

                    if need_search:
                        # 拐角检测：检查附近点是否分布在多个方向
                        max_off = 10.0
                        if tree is not None:
                            dist_q, _ = tree.query(pos)
                            if dist_q < collision_dist:
                                search_r = collision_dist * 2.0
                                nearby_idxs = tree.query_ball_point(pos, search_r)
                                if len(nearby_idxs) > 0:
                                    nearby_pts = tree.data[nearby_idxs]
                                    to_pts = nearby_pts - pos
                                    horiz = to_pts[:, :2]
                                    horiz_norms = np.linalg.norm(horiz, axis=1, keepdims=True)
                                    horiz_norms[horiz_norms < 1e-10] = 1.0
                                    horiz_dirs = horiz / horiz_norms
                                    outward_2d = outward[:2]
                                    dot_products = horiz_dirs @ outward_2d
                                    opposite_face = np.sum(dot_products < -0.5)
                                    side_face = np.sum(np.abs(dot_products) <= 0.5)
                                    if opposite_face > 0 or side_face > 0:
                                        max_off = 15.0

                        safe_pos, warned = self._find_safe_position(
                            cube_center, outward, tree, collision_dist, 0.0, max_offset=max_off)
                        if safe_pos is not None:
                            pos = safe_pos
                        if warned:
                            warnings.append(f"边{ei+1}层{layer} 无法满足所有约束")

                    # 重新计算heading（pos可能已调整）
                    to_cx = cx - pos[0]
                    to_cy = cy - pos[1]
                    to_c_len = np.sqrt(to_cx * to_cx + to_cy * to_cy)
                    if to_c_len > 1e-10:
                        heading = np.array([to_cx / to_c_len, to_cy / to_c_len, 0.0])

                    target = pos + heading
                    quat = look_at_quaternion(target, pos)
                    gimbal_pitch = self._calc_gimbal_pitch(pos, cube_center)
                    gimbal_pitch = np.clip(gimbal_pitch, self._gimbal_pitch_min, self._gimbal_pitch_max)
                    self.waypoints.append({
                        'pos': pos,
                        'quat': quat,
                        'speed': speed,
                        'action': 'scan',
                        'gimbal_pitch': gimbal_pitch,
                        'target_pos': cube_center.copy()
                    })

        self._display_route()

        if warnings:
            QMessageBox.warning(self, "约束警告",
                f"生成 {len(self.waypoints)} 个航点\n\n" + "\n".join(warnings[:10]))

    def _apply_bridge_params(self):
        try:
            bridge_len = float(self._bridge_len_val)
            bridge_wid = float(self._bridge_wid_val)
            clearance = float(self._bridge_clr_val)
            span = float(self._bridge_span_val)
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的桥梁参数")
            return

        bridge_type = self._bridge_type_val

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

        bridge_name = self._bridge_name_val.strip()
        if not bridge_name:
            bridge_name = ["跨河桥", "跨线桥", "高架桥"][self._bridge_type_val]
        self._bridge_name = bridge_name
        self.lbl_info.setText(f"桥梁: {bridge_name}, {bridge_len}m x {bridge_wid}m")

    def _apply_safety_settings(self):
        """应用安全设置（起飞高度、初始偏航角）并刷新航线显示"""
        if self.waypoints:
            takeoff_z, takeoff_yaw = self._get_takeoff_params()
            self.viewer._takeoff_z = takeoff_z
            self.viewer._takeoff_yaw = takeoff_yaw
            self.viewer.add_route(self.waypoints, reset_camera=False)
            self._check_safety_distance()
        print(f"[Safety] 起飞高度={self.edt_takeoff_z.text()}m, 初始偏航角={self.edt_takeoff_yaw.text()}°")

    def _on_camera_changed(self, name):
        """相机型号切换时自动填入FOV"""
        fov = self._camera_fov_map.get(name, 80)
        self.edt_camera_fov.setText(str(fov))

    def _calc_overlap_spacing(self):
        """根据相机FOV、巡检距离和重叠率自动计算航点距离和线间距"""
        import math
        try:
            # 使用巡检距离（目标距离）而非飞行高度
            inspect_dist = float(self.edt_flat_inspect_dist.text())
            fov = self._camera_fov
            fwd_overlap = self._forward_overlap / 100.0
            side_overlap = self._side_overlap / 100.0
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效数值")
            return
        if inspect_dist <= 0 or fov <= 0 or fov >= 180:
            QMessageBox.warning(self, "提示", "巡检距离和FOV需为正数，FOV<180°")
            return
        if not (0 <= fwd_overlap < 1) or not (0 <= side_overlap < 1):
            QMessageBox.warning(self, "提示", "重叠率需在0~99%之间")
            return
        # 拍摄范围 = 2 × 巡检距离 × tan(FOV/2)
        cover = 2.0 * inspect_dist * math.tan(math.radians(fov / 2.0))
        # 航点距离由旁向重叠率计算，线间距由航向重叠率计算
        wp_spacing = round(cover * (1.0 - side_overlap), 2)
        line_spacing = round(cover * (1.0 - fwd_overlap), 2)
        self.edt_wp_spacing.setText(str(max(0.1, wp_spacing)))
        self.edt_spacing.setText(str(max(0.1, line_spacing)))
        max_speed = wp_spacing / self._camera_min_interval if self._camera_min_interval > 0 else 999
        print(f"[Overlap] 巡检距离={inspect_dist}m FOV={fov}° 覆盖={cover:.1f}m → 航点间距={wp_spacing}m 线间距={line_spacing}m 最大速度={max_speed:.1f}m/s")

    def _calc_cube_spacing(self):
        """根据相机FOV、巡检距离和重叠率自动计算立方体航线的水平步距和垂直步距"""
        import math
        try:
            dist = float(self.edt_dist.text())
            fwd_overlap = self._forward_overlap / 100.0
            side_overlap = self._side_overlap / 100.0
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效数值")
            return
        if dist <= 0 or self._camera_fov <= 0 or self._camera_fov >= 180:
            QMessageBox.warning(self, "提示", "巡检距离和FOV需为正数，FOV<180°")
            return
        hfov = self._camera_fov
        vfov = self._vertical_fov()
        # 水平覆盖用水平FOV，垂直覆盖用垂直FOV
        h_cover = 2.0 * dist * math.tan(math.radians(hfov / 2.0))
        v_cover = 2.0 * dist * math.tan(math.radians(vfov / 2.0))
        h_step = round(h_cover * (1.0 - side_overlap), 2)
        v_step = round(v_cover * (1.0 - fwd_overlap), 2)
        self.edt_cstep.setText(str(max(0.1, h_step)))
        self.edt_vstep.setText(str(max(0.1, v_step)))
        max_speed = h_step / self._camera_min_interval if self._camera_min_interval > 0 else 999
        print(f"[Cube Overlap] 巡检距离={dist}m 水平FOV={hfov}° 垂直FOV={vfov:.1f}° → 水平步距={h_step}m 垂直步距={v_step}m 最大速度={max_speed:.1f}m/s")

    def _calc_cyl_spacing(self):
        """根据相机FOV、巡检距离和重叠率自动计算圆柱体航线的水平步距和垂直步距"""
        import math
        try:
            dist = float(self.edt_cyl_dist.text())
            diam = float(self.edt_cyl_diam.text())
            fwd_overlap = self._forward_overlap / 100.0
            side_overlap = self._side_overlap / 100.0
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效数值")
            return
        if dist <= 0 or self._camera_fov <= 0 or self._camera_fov >= 180:
            QMessageBox.warning(self, "提示", "巡检距离和FOV需为正数，FOV<180°")
            return
        hfov = self._camera_fov
        vfov = self._vertical_fov()
        # 垂直覆盖用垂直FOV
        v_cover = 2.0 * dist * math.tan(math.radians(vfov / 2.0))
        v_step = round(v_cover * (1.0 - fwd_overlap), 2)
        self.edt_cyl_vstep.setText(str(max(0.1, v_step)))
        # 水平步距（角度）：水平覆盖用水平FOV
        h_cover = 2.0 * dist * math.tan(math.radians(hfov / 2.0))
        radius = diam / 2 + dist
        arc_step = h_cover * (1.0 - side_overlap)
        angle_step = round(math.degrees(arc_step / radius), 1)
        self.edt_cyl_astep.setText(str(max(1.0, angle_step)))
        max_speed = arc_step / self._camera_min_interval if self._camera_min_interval > 0 else 999
        print(f"[Cyl Overlap] 巡检距离={dist}m 水平FOV={hfov}° 垂直FOV={vfov:.1f}° → 弧长步距={arc_step:.2f}m 角度步距={angle_step}° 垂直步距={v_step}m 最大速度={max_speed:.1f}m/s")

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

    def _toggle_heading(self, checked):
        self.viewer.show_heading = checked
        self._refresh_route_display()

    def _toggle_gimbal_dir(self, checked):
        self.viewer.show_gimbal_dir = checked
        self._refresh_route_display()

    def _toggle_route_animation(self):
        """开始/停止航线动画播放"""
        if self.viewer._anim_playing:
            self.viewer.stop_route_animation()
            return
        if not self.waypoints:
            QMessageBox.warning(self, "提示", "请先生成航线")
            return
        self._on_anim_started()
        self.viewer.start_route_animation(self.waypoints, speed=1.0, camera_fov=self._camera_fov)

    def _on_anim_started(self):
        """动画开始：禁用其他操作"""
        self._act_anim_play.setText("停止动画")
        self._file_menu.setEnabled(False)
        self._view_menu.setEnabled(False)
        self._settings_menu.setEnabled(False)
        # 禁用侧边栏
        for w in self.findChildren(QPushButton):
            if w.text() not in ("停止动画",):
                w.setEnabled(False)
        for w in self.findChildren(QComboBox):
            w.setEnabled(False)

    def _on_anim_stopped(self):
        """动画结束：恢复操作"""
        self._act_anim_play.setText("航线动画播放")
        self._file_menu.setEnabled(True)
        self._view_menu.setEnabled(True)
        self._settings_menu.setEnabled(True)
        for w in self.findChildren(QPushButton):
            w.setEnabled(True)
        for w in self.findChildren(QComboBox):
            w.setEnabled(True)

    def _refresh_route_display(self):
        if self.waypoints:
            takeoff_z, takeoff_yaw = self._get_takeoff_params()
            self.viewer._takeoff_z = takeoff_z
            self.viewer._takeoff_yaw = takeoff_yaw
            self.viewer.add_route(self.waypoints, reset_camera=False)
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
        self.viewer._camera_hfov = self._camera_fov
        self.viewer._camera_vfov = self._vertical_fov()
        try:
            self.viewer._safe_point = (
                float(self.edt_safe_x.text()),
                float(self.edt_safe_y.text()),
                float(self.edt_safe_z.text()),
            )
        except ValueError:
            self.viewer._safe_point = (0.0, 0.0, 5.0)
        self.viewer.add_route(self.waypoints, reset_camera=False)
        n = len(self.waypoints)
        info = f"航点: {n}"
        if n >= 1:
            p0 = self.waypoints[0]['pos']
            info += f"  |  首点: ({p0[0]:.1f}, {p0[1]:.1f}, {p0[2]:.1f})"
        if n >= 2:
            pn = self.waypoints[-1]['pos']
            info += f"  |  末点: ({pn[0]:.1f}, {pn[1]:.1f}, {pn[2]:.1f})"
        self.lbl_info.setText(info)
        self._update_route_time_label()
        self._check_safety_distance()

    @staticmethod
    def _calc_gimbal_pitch(drone_pos, target_pos, heading=None):
        """计算云台pitch角度（度），使目标点位于相机画面中心
        drone_pos: 无人机位置 [x,y,z]
        target_pos: 目标点位置 [x,y,z]
        heading: 机头方向（前左上坐标系的 Forward 轴），可选
                传入时用 atan2(los_up, los_fwd) 计算（螃蟹飞正确）
                不传时用 atan(drop, h_dist) 计算（兼容旧行为）
        返回: pitch角度（负值=朝下，正值=朝上，-90=垂直朝下，+90=垂直朝上）
        """
        d = np.array(drone_pos, dtype=np.float64)
        t = np.array(target_pos, dtype=np.float64)
        los = t - d  # 无人机→目标视线

        if heading is not None:
            fwd = np.array(heading, dtype=np.float64)
            fwd_len = np.linalg.norm(fwd)
            if fwd_len > 1e-9:
                fwd = fwd / fwd_len
                los_fwd = np.dot(los, fwd)
                los_up = los[2]  # dot(los, [0,0,1])
                if los_fwd > 1e-9:
                    # 目标在机头前方：用 heading 公式（螃蟹飞正确）
                    return np.degrees(np.arctan2(los_up, los_fwd))
                # 目标在侧面或后方：降级到水平距离公式

        # 降级：无 heading 或目标不在机头前方时用水平距离
        drop = d[2] - t[2]  # 正=无人机在上，负=无人机在下
        h_dist = np.sqrt((d[0] - t[0])**2 + (d[1] - t[1])**2)
        if h_dist < 1e-6:
            return -90.0 if drop >= 0 else 90.0
        return -np.degrees(np.arctan(drop / h_dist))

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
            self.viewer.add_route(self.waypoints, reset_camera=False)
            self._check_safety_distance()

    def _get_kdtree(self):
        # 优先使用STL网格的KDTree
        if self.viewer._stl_tree is not None:
            return self.viewer._stl_tree
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

        collision_dist = safe_dist * 1.5
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
        self._update_route_time_label()
        self.viewer.vtk_widget.GetRenderWindow().Render()

    def _update_route_time_label(self):
        """计算并显示航线总飞行时间"""
        if not self.waypoints:
            self.lbl_route_time.setText("")
            return
        total_time = 0.0
        for i in range(1, len(self.waypoints)):
            p0 = np.array(self.waypoints[i - 1]['pos'])
            p1 = np.array(self.waypoints[i]['pos'])
            dist = np.linalg.norm(p1 - p0)
            speed = self.waypoints[i].get('speed', 1.0)
            if speed < 0.1:
                speed = 0.1
            total_time += dist / speed
        # 加上起飞和降落时间估算
        total_time += 5.0  # 起飞悬停
        total_time += 5.0  # 降落悬停
        if total_time < 60:
            self.lbl_route_time.setText(f"预计飞行时间: {total_time:.0f} 秒")
        else:
            m = int(total_time // 60)
            s = int(total_time % 60)
            self.lbl_route_time.setText(f"预计飞行时间: {m}分{s}秒")

    def _collect_collision_warnings(self):
        """收集碰撞检测数据，返回 (violations, collisions, low_z)
        violations: [(i, j, dist), ...] 航点间距过近
        collisions: [(idx, dist), ...] 航点距点云过近（含线段采样）
        low_z: [(idx, z_val), ...] 航点低于最低Z值
        """
        safe_dist = self.viewer._safe_distance
        collision_dist = safe_dist * 1.5
        sample_step = safe_dist * 0.5

        # ── 航点间距检测（只检查非相邻航点，相邻航点按重叠率设计本就近）──
        violations = []
        if len(self.waypoints) >= 3:
            from scipy.spatial import cKDTree
            positions = np.array([wp['pos'] for wp in self.waypoints])
            tree_wp = cKDTree(positions)
            pairs = tree_wp.query_pairs(safe_dist)
            for i, j in pairs:
                if abs(i - j) > 1:  # 跳过相邻航点
                    d = np.linalg.norm(positions[i] - positions[j])
                    violations.append((int(min(i, j)), int(max(i, j)), float(d)))

        # 构建完整路径点列表：安全点 + 所有航点
        try:
            sx = float(self.edt_safe_x.text())
            sy = float(self.edt_safe_y.text())
            sz = float(self.edt_safe_z.text())
        except ValueError:
            sx, sy, sz = 0.0, 0.0, 5.0
        safe_pos = np.array([sx, sy, sz])
        all_positions = [safe_pos] + [wp['pos'] for wp in self.waypoints]

        # ── 线段采样碰撞检测（STL 版 / 点云版）──
        collisions = []
        if self._stl_triangles_np is not None and len(all_positions) >= 2:
            # STL 版：精确计算到三角面的最短距离
            for seg_i in range(len(all_positions) - 1):
                p1 = all_positions[seg_i]
                p2 = all_positions[seg_i + 1]
                d = np.linalg.norm(p2 - p1)
                if d < 1e-10:
                    continue
                n_samples = max(2, int(d / sample_step) + 1)
                ts = np.linspace(0, 1, n_samples + 1)
                pts = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
                for pt in pts:
                    stl_dist = self._stl_surface_distance(pt)
                    if stl_dist < collision_dist:
                        label = -1 if seg_i == 0 else seg_i
                        collisions.append((label, float(stl_dist)))
        else:
            # 点云版：用 KDTree
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
        self.viewer._clear_line_points()
        self.viewer._clear_inspect_points()
        self._inspect_target_points.clear()
        self._inspect_target_normals.clear()
        self._line_start_normal = None
        self._line_end_normal = None
        self._polygon_normals.clear()

        # 清除直线航线起终点坐标
        for edt in [self.edt_line_x1, self.edt_line_y1, self.edt_line_z1,
                     self.edt_line_x2, self.edt_line_y2, self.edt_line_z2]:
            edt.setText("0")
        self.edt_line_spacing.setText("自动")

        ren = self.viewer.renderer
        cloud = self.viewer._cloud_actor
        drone = self.viewer._fpv_drone_actor
        stl = self.viewer._stl_actor

        # 批量收集非点云、非无人机、非STL actor，一次性移除
        to_remove = [a for a in self.viewer._actors if a != cloud and a != drone and a != stl]
        for a in to_remove:
            ren.RemoveActor(a)
        self.viewer._actors = [a for a in self.viewer._actors if a == cloud or a == drone or a == stl]
        self.viewer._waypoint_actors = []
        self.viewer._waypoints_ref = None

        # 恢复坐标轴和网格
        self.viewer._add_scene_axes()
        self.viewer.vtk_widget.GetRenderWindow().Render()
        self.lbl_info.setText("航点: 0")
        self._update_route_time_label()

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
                "type": ["跨河桥", "跨线桥", "高架桥"][self._bridge_type_val],
                "type_index": self._bridge_type_val,
                "length_m": self._bridge_len_val,
                "width_m": self._bridge_wid_val,
                "clearance_m": self._bridge_clr_val,
                "span_m": self._bridge_span_val
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

    def copy_maicro_route_to_clipboard(self):
        """复制maicro格式航线到剪贴板"""
        if not self.waypoints:
            QMessageBox.information(self, "提示", "没有航线可复制")
            return

        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # 获取相机参数
        cam_name = self._camera_name
        if "长焦" in cam_name:
            camera_source = "ZOOM_CAMERA"
        else:
            camera_source = "WIDE_CAMERA"
        focal_length_map = {
            "DJI Mavic 3E": 24, "DJI Mavic 3T": 24,
            "M4T 广角": 24, "M4T 中长焦": 72, "M4T 长焦": 162,
            "DJI M350+P1(24mm)": 24, "DJI M350+P1(35mm)": 35,
            "DJI M350+P1(50mm)": 50, "DJI M350+L2(雷达)": 24,
            "自定义": 24,
        }
        focal_length = focal_length_map.get(cam_name, 24)
        bridge_name = getattr(self, '_bridge_name', '桥梁')

        way_point_list = []
        for i, wp in enumerate(self.waypoints):
            pos = wp['pos']
            gimbal_pitch = wp.get('gimbal_pitch', -90.0)
            speed = wp.get('speed', 1.0)
            action = wp.get('action', 'fly')
            shoot = (action == 'scan')

            q = wp['quat']
            yaw = np.degrees(np.arctan2(
                2.0 * (q[0] * q[3] + q[1] * q[2]),
                1.0 - 2.0 * (q[2]**2 + q[3]**2)
            ))

            wp_data = {
                "index": i,
                "lat": 0.0, "lon": 0.0, "alt": float(pos[2]),
                "x": round(float(pos[0]), 6),
                "y": round(float(pos[1]), 6),
                "z": round(float(pos[2]), 6),
                "devicePartName": f"航点{i+1}",
                "cameraSource": camera_source,
                "focalLength": focal_length,
                "gimbalPitch": round(float(gimbal_pitch), 2),
                "speed": round(float(speed), 1),
                "head": round(float(yaw), 2),
                "yaw": round(float(yaw), 2),
                "shoot": shoot,
                "thermal": False,
                "headingMode": "fixed"
            }

            if 'target_pos' in wp:
                tgt = wp['target_pos']
                wp_data["aimTarget"] = {
                    "focalRing": 0,
                    "distance": round(float(np.linalg.norm(np.array(pos) - np.array(tgt))), 2),
                    "alt": round(float(tgt[2]), 6),
                    "lon": 0.0, "lat": 0.0
                }

            way_point_list.append(wp_data)

        maicro_data = {
            "aircraftModel": "DJI_MATRICE_4_SERIES",
            "createdTime": ts,
            "bridgeName": bridge_name,
            "partName": "自定义航线",
            "name": f"{bridge_name}_自定义航线",
            "photoCount": len([w for w in way_point_list if w.get("shoot")]),
            "partType": 1,
            "exposure": {"shutter": 500, "ev": 0, "iso": 400},
            "photoMode": 1,
            "initialSpeed": 2,
            "wayPointList": way_point_list
        }

        json_str = json.dumps(maicro_data, indent=2, ensure_ascii=False)
        QApplication.clipboard().setText(json_str)
        QMessageBox.information(self, "已复制",
            f"maicro航线已复制到剪贴板（{len(way_point_list)} 个航点）")

    # ─── 导出 maicro 航线文件 ───
    def export_maicro_route(self):
        """导出maicro格式航线文件"""
        if not self.waypoints:
            QMessageBox.information(self, "提示", "没有航线可导出")
            return

        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        default_name = f"{getattr(self, '_bridge_name', '航线')}_{ts}.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出maicro航线文件", default_name, "JSON 文件 (*.json)"
        )
        if not path:
            return

        # 获取相机参数
        cam_name = self._camera_name
        cam_fov = self._camera_fov
        # maicro 相机源映射
        if "长焦" in cam_name:
            camera_source = "ZOOM_CAMERA"
        else:
            camera_source = "WIDE_CAMERA"
        focal_length_map = {
            "DJI Mavic 3E": 24, "DJI Mavic 3T": 24,
            "M4T 广角": 24, "M4T 中长焦": 72, "M4T 长焦": 162,
            "DJI M350+P1(24mm)": 24, "DJI M350+P1(35mm)": 35,
            "DJI M350+P1(50mm)": 50, "DJI M350+L2(雷达)": 24,
            "自定义": 24,
        }
        focal_length = focal_length_map.get(cam_name, 24)

        bridge_name = getattr(self, '_bridge_name', '桥梁')

        # 构建航点列表
        way_point_list = []
        for i, wp in enumerate(self.waypoints):
            pos = wp['pos']
            gimbal_pitch = wp.get('gimbal_pitch', -90.0)
            speed = wp.get('speed', 1.0)
            action = wp.get('action', 'fly')
            shoot = (action == 'scan')

            # 从 quat 计算 yaw 角度
            q = wp['quat']
            yaw = np.degrees(np.arctan2(
                2.0 * (q[0] * q[3] + q[1] * q[2]),
                1.0 - 2.0 * (q[2]**2 + q[3]**2)
            ))

            wp_data = {
                "index": i,
                "lat": 0.0,
                "lon": 0.0,
                "alt": float(pos[2]),
                "x": round(float(pos[0]), 6),
                "y": round(float(pos[1]), 6),
                "z": round(float(pos[2]), 6),
                "devicePartName": f"航点{i+1}",
                "cameraSource": camera_source,
                "focalLength": focal_length,
                "gimbalPitch": round(float(gimbal_pitch), 2),
                "speed": round(float(speed), 1),
                "head": round(float(yaw), 2),
                "yaw": round(float(yaw), 2),
                "shoot": shoot,
                "thermal": False,
                "headingMode": "fixed"
            }

            # 如果有投影目标点，添加 aimTarget
            if 'target_pos' in wp:
                tgt = wp['target_pos']
                wp_data["aimTarget"] = {
                    "focalRing": 0,
                    "distance": round(float(np.linalg.norm(np.array(pos) - np.array(tgt))), 2),
                    "alt": round(float(tgt[2]), 6),
                    "lon": 0.0,
                    "lat": 0.0
                }

            way_point_list.append(wp_data)

        maicro_data = {
            "aircraftModel": "DJI_MATRICE_4_SERIES",
            "createdTime": ts,
            "bridgeName": bridge_name,
            "partName": "自定义航线",
            "name": f"{bridge_name}_自定义航线",
            "photoCount": len([w for w in way_point_list if w.get("shoot")]),
            "partType": 1,
            "exposure": {
                "shutter": 500,
                "ev": 0,
                "iso": 400
            },
            "photoMode": 1,
            "initialSpeed": 2,
            "wayPointList": way_point_list
        }

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(maicro_data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "已导出",
                f"maicro航线已导出到:\n{path}\n\n航点数: {len(way_point_list)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")

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
                if 0 <= idx < 3:
                    self._bridge_type_val = idx
                self._bridge_len_val = str(bridge.get('length_m', '100'))
                self._bridge_wid_val = str(bridge.get('width_m', '15'))
                self._bridge_clr_val = str(bridge.get('clearance_m', '8'))
                self._bridge_span_val = str(bridge.get('span_m', '30'))

            self._display_route()
            QMessageBox.information(self, "已加载", f"已加载 {len(self.waypoints)} 个航点")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败:\n{str(e)}")
