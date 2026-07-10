"""生成斜拉桥3D模型：STL网格 + 带标签的PLY点云"""
import numpy as np
import struct

# 部件标签定义
LABEL_DECK = 0       # 桥面（含纵梁、横梁、加劲肋）
LABEL_TOWER = 1      # 桥塔
LABEL_CABLE = 2      # 斜拉索
LABEL_PIER = 3       # 桥墩（含盖梁、承台、桩基）
LABEL_ABUTMENT = 4   # 桥台
LABEL_RAILING = 5    # 护栏
LABEL_AUX_PIER = 6   # 辅助墩
LABEL_SIDEWALK = 7   # 人行道

LABEL_NAMES = {
    0: "桥面", 1: "桥塔", 2: "斜拉索", 3: "桥墩",
    4: "桥台", 5: "护栏", 6: "辅助墩", 7: "人行道"
}

# 每个标签对应的颜色 (R, G, B)
LABEL_COLORS = {
    0: (180, 180, 180),  # 桥面：浅灰
    1: (100, 140, 200),  # 桥塔：蓝灰
    2: (220, 200, 80),   # 拉索：金黄
    3: (140, 140, 140),  # 桥墩：中灰
    4: (160, 120, 90),   # 桥台：棕灰
    5: (200, 200, 200),  # 护栏：亮灰
    6: (130, 150, 130),  # 辅墩：灰绿
    7: (190, 190, 170),  # 人行道：米灰
}


def create_cable_stayed_bridge():
    """创建斜拉桥模型，返回 (triangles, labels, component_centers)
    triangles: [(v1,v2,v3), ...]
    labels: [int, ...] 每个三角形的部件标签
    component_centers: {label: (cx,cy,cz)} 每个部件的几何中心
    """
    triangles = []
    labels = []
    # 记录每个标签下所有三角形顶点，用于计算几何中心
    label_vertices = {}

    def _add_tri(v1, v2, v3, label):
        triangles.append([v1, v2, v3])
        labels.append(label)
        if label not in label_vertices:
            label_vertices[label] = []
        label_vertices[label].extend([v1, v2, v3])

    def add_box(cx, cy, cz, wx, wy, wz, label):
        """添加一个立方体"""
        v = [
            [cx - wx/2, cy - wy/2, cz - wz/2],
            [cx + wx/2, cy - wy/2, cz - wz/2],
            [cx + wx/2, cy + wy/2, cz - wz/2],
            [cx - wx/2, cy + wy/2, cz - wz/2],
            [cx - wx/2, cy - wy/2, cz + wz/2],
            [cx + wx/2, cy - wy/2, cz + wz/2],
            [cx + wx/2, cy + wy/2, cz + wz/2],
            [cx - wx/2, cy + wy/2, cz + wz/2],
        ]
        faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (1, 5, 6), (1, 6, 2),
            (2, 6, 7), (2, 7, 3),
            (3, 7, 4), (3, 4, 0),
        ]
        for i1, i2, i3 in faces:
            _add_tri(v[i1], v[i2], v[i3], label)

    def add_cylinder_approx(cx, cy, cz, radius, height, segments, label):
        """添加圆柱体（棱柱近似）"""
        bottom_circle = []
        top_circle = []
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            bottom_circle.append([x, y, cz])
            top_circle.append([x, y, cz + height])

        for i in range(segments):
            i1 = i
            i2 = (i + 1) % segments
            _add_tri(bottom_circle[i1], bottom_circle[i2], top_circle[i2], label)
            _add_tri(bottom_circle[i1], top_circle[i2], top_circle[i1], label)

        for i in range(1, segments - 1):
            _add_tri(bottom_circle[0], bottom_circle[i], bottom_circle[i + 1], label)

        for i in range(1, segments - 1):
            _add_tri(top_circle[0], top_circle[i + 1], top_circle[i], label)

    def add_cable(x1, y1, z1, x2, y2, z2, radius=0.15):
        """添加缆索"""
        segments = 6
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        if length < 0.001:
            return

        direction = np.array([dx, dy, dz]) / length
        if abs(direction[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)
        forward = np.cross(right, direction)
        forward = forward / np.linalg.norm(forward)

        bottom_circle = []
        top_circle = []
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            offset = right * radius * np.cos(angle) + forward * radius * np.sin(angle)
            bottom_circle.append([x1 + offset[0], y1 + offset[1], z1 + offset[2]])
            top_circle.append([x2 + offset[0], y2 + offset[1], z2 + offset[2]])

        for i in range(segments):
            i1 = i
            i2 = (i + 1) % segments
            _add_tri(bottom_circle[i1], bottom_circle[i2], top_circle[i2], LABEL_CABLE)
            _add_tri(bottom_circle[i1], top_circle[i2], top_circle[i1], LABEL_CABLE)

    # ══════════════════════════════════════════════
    # 基本参数
    # ══════════════════════════════════════════════
    bridge_length = 100.0
    bridge_width = 15.0
    bridge_thickness = 1.0
    tower_positions = [-bridge_length/4, bridge_length/4]
    tower_height = 35.0
    tower_width = 4.0
    tower_depth = 4.0
    pier_height = 12.0
    pier_width = 5.0
    pier_depth = 6.0

    # ─── 桥面（顶板） ───
    add_box(0, 0, bridge_thickness/2, bridge_length, bridge_width, bridge_thickness, LABEL_DECK)

    # ─── 纵梁（4根） ───
    girder_h = 1.2
    girder_w = 0.6
    girder_spacing = bridge_width / 5
    for i in range(-2, 3):
        if i == 0:
            continue
        add_box(0, i * girder_spacing, -girder_h/2,
                bridge_length - 1, girder_w, girder_h, LABEL_DECK)

    # ─── 横梁（每隔5m） ───
    cross_beam_h = 1.0
    cross_beam_w = 0.5
    for cx in np.arange(-bridge_length/2 + 2.5, bridge_length/2, 5.0):
        add_box(cx, 0, -cross_beam_h/2,
                cross_beam_w, bridge_width - 1, cross_beam_h, LABEL_DECK)

    # ─── 加劲肋 ───
    rib_h = 0.4
    rib_w = 0.2
    for ry in np.arange(-bridge_width/2 + 1, bridge_width/2, 2.5):
        add_box(0, ry, -rib_h/2, bridge_length - 2, rib_w, rib_h, LABEL_DECK)

    # ─── 桥塔 ───
    for tx in tower_positions:
        add_box(tx, -tower_depth/2, tower_height/2, tower_width, tower_depth/2, tower_height, LABEL_TOWER)
        add_box(tx, tower_depth/2, tower_height/2, tower_width, tower_depth/2, tower_height, LABEL_TOWER)
        add_box(tx, 0, tower_height - 1, tower_width + 2, tower_depth + 2, 2, LABEL_TOWER)
        add_box(tx, 0, tower_height * 0.6, tower_width + 1, tower_depth + 1, 1.5, LABEL_TOWER)
        add_box(tx, -tower_depth/2, 2.5, tower_width + 1.5, tower_depth/2 + 0.8, 5, LABEL_TOWER)
        add_box(tx, tower_depth/2, 2.5, tower_width + 1.5, tower_depth/2 + 0.8, 5, LABEL_TOWER)

    # ─── 斜拉索 ───
    for tx in tower_positions:
        for i in range(10):
            deck_x = tx - (i + 1) * (bridge_length/4) / 10
            if deck_x < -bridge_length/2:
                continue
            top_z = tower_height - 2
            bottom_z = bridge_thickness
            add_cable(tx, -tower_depth/2, top_z, deck_x, -bridge_width/2 + 1, bottom_z, 0.12)
            add_cable(tx, tower_depth/2, top_z, deck_x, bridge_width/2 - 1, bottom_z, 0.12)
        for i in range(10):
            deck_x = tx + (i + 1) * (bridge_length/4) / 10
            if deck_x > bridge_length/2:
                continue
            top_z = tower_height - 2
            bottom_z = bridge_thickness
            add_cable(tx, -tower_depth/2, top_z, deck_x, -bridge_width/2 + 1, bottom_z, 0.12)
            add_cable(tx, tower_depth/2, top_z, deck_x, bridge_width/2 - 1, bottom_z, 0.12)

    # ─── 桥墩（盖梁 + 墩柱 + 承台 + 桩基 + 斜撑） ───
    cap_beam_height = 1.5
    cap_beam_extend = 1.5
    footing_height = 2.0
    footing_extend = 2.0

    for px in tower_positions:
        add_box(px, 0, pier_height - cap_beam_height/2,
                pier_width + cap_beam_extend*2, pier_depth + cap_beam_extend*2, cap_beam_height, LABEL_PIER)
        add_box(px, 0, pier_height/2, pier_width, pier_depth, pier_height, LABEL_PIER)
        add_box(px, 0, -footing_height/2,
                pier_width + footing_extend*2, pier_depth + footing_extend*2, footing_height, LABEL_PIER)
        pile_radius = 0.8
        pile_height = 8.0
        for ox, oy in [(pier_width/2-1, pier_depth/2-1), (-pier_width/2+1, pier_depth/2-1),
                        (pier_width/2-1, -pier_depth/2+1), (-pier_width/2+1, -pier_depth/2+1)]:
            add_cylinder_approx(px + ox, oy, -footing_height - pile_height/2,
                                pile_radius, pile_height, 8, LABEL_PIER)
        # 斜撑
        brace_top_z = pier_height * 0.8
        brace_bot_z = pier_height * 0.2
        add_cable(px, -pier_depth/4, brace_bot_z, px, pier_depth/4, brace_top_z, 0.3)
        add_cable(px, -pier_depth/4, brace_top_z, px, pier_depth/4, brace_bot_z, 0.3)

    # ─── 辅助墩 ───
    aux_pier_height = 10.0
    aux_pier_width = 3.0
    aux_pier_depth = 4.0
    for px in [-bridge_length/2 + 12, tower_positions[0] - 12,
               tower_positions[1] + 12, bridge_length/2 - 12]:
        add_box(px, 0, aux_pier_height - 1,
                aux_pier_width + 2, aux_pier_depth + 2, 1.5, LABEL_AUX_PIER)
        add_box(px, -aux_pier_depth/3, aux_pier_height/2,
                aux_pier_width, aux_pier_depth/3, aux_pier_height, LABEL_AUX_PIER)
        add_box(px, aux_pier_depth/3, aux_pier_height/2,
                aux_pier_width, aux_pier_depth/3, aux_pier_height, LABEL_AUX_PIER)
        add_box(px, 0, -1, aux_pier_width + 3, aux_pier_depth + 3, 2, LABEL_AUX_PIER)
        for oy in [-aux_pier_depth/3, aux_pier_depth/3]:
            add_cylinder_approx(px, oy, -1 - 4, 0.6, 6, 8, LABEL_AUX_PIER)
        add_box(px, 0, aux_pier_height * 0.5,
                aux_pier_width * 0.6, aux_pier_depth - 0.5, 0.6, LABEL_AUX_PIER)

    # ─── 桥台 ───
    abutment_length = 8.0
    abutment_width = bridge_width + 4.0
    abutment_height = 8.0
    for side in [-1, 1]:
        ax = side * (bridge_length/2 + abutment_length/2)
        add_box(ax, 0, -abutment_height/2, abutment_length, abutment_width, abutment_height, LABEL_ABUTMENT)
        add_box(ax, 0, 0.5, abutment_length + 1, abutment_width + 1, 1.5, LABEL_ABUTMENT)
        add_box(ax, 0, -abutment_height - 1, abutment_length + 3, abutment_width + 3, 2, LABEL_ABUTMENT)
        add_box(ax, -abutment_width/2 - 0.5, -abutment_height/4,
                abutment_length - 2, 1.0, abutment_height/2, LABEL_ABUTMENT)
        add_box(ax, abutment_width/2 + 0.5, -abutment_height/4,
                abutment_length - 2, 1.0, abutment_height/2, LABEL_ABUTMENT)
        add_box(ax + side * 2, 0, -abutment_height * 0.7,
                3, abutment_width + 2, abutment_height * 0.4, LABEL_ABUTMENT)

    # ─── 护栏 ───
    rail_post_size = 0.25
    rail_post_height = 1.2
    for rx in np.arange(-bridge_length/2 + 1, bridge_length/2, 2.0):
        for ry in [-bridge_width/2 + 0.3, bridge_width/2 - 0.3]:
            add_box(rx, ry, bridge_thickness + rail_post_height/2,
                    rail_post_size, rail_post_size, rail_post_height, LABEL_RAILING)
    add_box(0, -bridge_width/2 + 0.3, bridge_thickness + rail_post_height,
            bridge_length - 1, 0.3, 0.2, LABEL_RAILING)
    add_box(0, bridge_width/2 - 0.3, bridge_thickness + rail_post_height,
            bridge_length - 1, 0.3, 0.2, LABEL_RAILING)
    add_box(0, -bridge_width/2 + 0.3, bridge_thickness + rail_post_height * 0.5,
            bridge_length - 1, 0.24, 0.15, LABEL_RAILING)
    add_box(0, bridge_width/2 - 0.3, bridge_thickness + rail_post_height * 0.5,
            bridge_length - 1, 0.24, 0.15, LABEL_RAILING)

    # ─── 人行道 ───
    sidewalk_width = 1.5
    sidewalk_height = 0.15
    add_box(0, -bridge_width/2 + sidewalk_width/2, bridge_thickness + sidewalk_height/2,
            bridge_length, sidewalk_width, sidewalk_height, LABEL_SIDEWALK)
    add_box(0, bridge_width/2 - sidewalk_width/2, bridge_thickness + sidewalk_height/2,
            bridge_length, sidewalk_width, sidewalk_height, LABEL_SIDEWALK)

    # 计算每个部件的几何中心
    component_centers = {}
    for label, verts in label_vertices.items():
        arr = np.array(verts)
        component_centers[label] = arr.mean(axis=0)

    return triangles, labels, component_centers


def sample_point_cloud(triangles, labels, component_centers, density=50):
    """从网格表面采样点云，修正法线方向朝外
    density: 每平方米采样点数
    返回: (points, normals, point_labels)
    """
    all_points = []
    all_normals = []
    all_labels = []

    for idx, (v1, v2, v3) in enumerate(triangles):
        v1 = np.array(v1)
        v2 = np.array(v2)
        v3 = np.array(v3)
        label = labels[idx]

        edge1 = v2 - v1
        edge2 = v3 - v1
        cross = np.cross(edge1, edge2)
        area = np.linalg.norm(cross) / 2.0
        if area < 1e-10:
            continue

        normal = cross / np.linalg.norm(cross)

        # 修正法线方向：确保朝向远离部件几何中心
        center = component_centers[label]
        tri_center = (v1 + v2 + v3) / 3.0
        outward = tri_center - center
        if np.dot(normal, outward) < 0:
            normal = -normal

        n_points = max(1, int(area * density))

        # 重心坐标采样
        r1 = np.sqrt(np.random.random(n_points))
        r2 = np.random.random(n_points)
        u = 1 - r1
        v = r1 * (1 - r2)
        w = r1 * r2

        points = u[:, None] * v1 + v[:, None] * v2 + w[:, None] * v3

        all_points.append(points)
        all_normals.append(np.tile(normal, (n_points, 1)))
        all_labels.extend([label] * n_points)

    return np.vstack(all_points), np.vstack(all_normals), np.array(all_labels)


def save_ply_with_labels(filename, points, normals, labels):
    """保存为PLY格式（xyz + 法向量 + 部件标签 + 部件颜色）"""
    n = len(points)
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int label\n")
        f.write("end_header\n")
        for i in range(n):
            px, py, pz = points[i]
            nx, ny, nz = normals[i]
            lbl = int(labels[i])
            r, g, b = LABEL_COLORS.get(lbl, (180, 180, 180))
            f.write(f"{px:.3f} {py:.3f} {pz:.3f} {nx:.4f} {ny:.4f} {nz:.4f} {r} {g} {b} {lbl}\n")


def save_stl_binary(filename, triangles):
    """保存为二进制STL"""
    with open(filename, 'wb') as f:
        header = b'Cable Stayed Bridge Model' + b'\0' * (80 - 25)
        f.write(header)
        f.write(struct.pack('<I', len(triangles)))
        for v1, v2, v3 in triangles:
            edge1 = np.array(v2) - np.array(v1)
            edge2 = np.array(v3) - np.array(v1)
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            f.write(struct.pack('<fff', *normal))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<fff', *v3))
            f.write(struct.pack('<H', 0))


if __name__ == "__main__":
    import os

    print("正在生成斜拉桥模型...")
    triangles, labels, comp_centers = create_cable_stayed_bridge()
    print(f"  网格: {len(triangles)} 个三角形")

    # 统计各部件
    from collections import Counter
    label_counts = Counter(labels)
    for lbl, cnt in sorted(label_counts.items()):
        print(f"    {LABEL_NAMES[lbl]}: {cnt} 个三角形")

    # 保存STL
    stl_file = "cable_stayed_bridge.stl"
    save_stl_binary(stl_file, triangles)
    print(f"  STL: {stl_file} ({os.path.getsize(stl_file)/1024:.1f} KB)")

    # 采样点云
    print("正在采样点云（法线已修正为朝外）...")
    points, normals, point_labels = sample_point_cloud(
        triangles, labels, comp_centers, density=30
    )
    print(f"  点云: {len(points)} 个点")

    # 统计各部件点数
    plabel_counts = Counter(point_labels)
    for lbl, cnt in sorted(plabel_counts.items()):
        print(f"    {LABEL_NAMES[lbl]}: {cnt} 点")

    # 保存PLY
    ply_file = "cable_stayed_bridge.ply"
    save_ply_with_labels(ply_file, points, normals, point_labels)
    print(f"  PLY: {ply_file} ({os.path.getsize(ply_file)/1024:.1f} KB)")

    # 尺寸
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)
    print(f"\n  模型尺寸: {xmax-xmin:.1f}m × {ymax-ymin:.1f}m × {zmax-zmin:.1f}m")
    print(f"  X: {xmin:.1f} ~ {xmax:.1f}")
    print(f"  Y: {ymin:.1f} ~ {ymax:.1f}")
    print(f"  Z: {zmin:.1f} ~ {zmax:.1f}")
