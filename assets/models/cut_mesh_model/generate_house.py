"""生成一个简单的房子STL模型（内部中空，可用于室内航线测试）"""
import numpy as np

def cross(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])

def normal_of(v0, v1, v2):
    n = cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(n)
    return n / length if length > 1e-10 else np.array([0, 0, 1.0])

triangles = []

def tri(p0, p1, p2):
    triangles.append((np.array(p0, dtype=float), np.array(p1, dtype=float), np.array(p2, dtype=float)))

def quad(p0, p1, p2, p3):
    tri(p0, p1, p2)
    tri(p0, p2, p3)

# 房子尺寸
W, D, H = 10.0, 8.0, 4.0   # 宽、深、墙高
roof_h = 3.0
t = 0.15  # 墙壁厚度

hw, hd = W/2, D/2

# === 外墙（4面，每面内外两层） ===
# 前墙 (y=0)
quad([-hw, 0, 0], [hw, 0, 0], [hw, 0, H], [-hw, 0, H])
quad([-hw+t, t, t], [hw-t, t, t], [hw-t, t, H-t], [-hw+t, t, H-t])
# 后墙 (y=D)
quad([hw, D, 0], [-hw, D, 0], [-hw, D, H], [hw, D, H])
quad([hw-t, D-t, t], [-hw+t, D-t, t], [-hw+t, D-t, H-t], [hw-t, D-t, H-t])
# 左墙 (x=-W/2)
quad([-hw, D, 0], [-hw, 0, 0], [-hw, 0, H], [-hw, D, H])
quad([-hw+t, D-t, t], [-hw+t, t, t], [-hw+t, t, H-t], [-hw+t, D-t, H-t])
# 右墙 (x=W/2)
quad([hw, 0, 0], [hw, D, 0], [hw, D, H], [hw, 0, H])
quad([hw-t, t, t], [hw-t, D-t, t], [hw-t, D-t, H-t], [hw-t, t, H-t])

# === 墙顶封边（外墙顶面到内墙顶面的窄条） ===
quad([-hw, 0, H], [hw, 0, H], [hw-t, t, H], [-hw+t, t, H])       # 前
quad([hw, D, H], [-hw, D, H], [-hw+t, D-t, H], [hw-t, D-t, H])   # 后
quad([-hw, D, H], [-hw, 0, H], [-hw+t, t, H], [-hw+t, D-t, H])   # 左
quad([hw, 0, H], [hw, D, H], [hw-t, D-t, H], [hw-t, t, H])       # 右

# === 地板 ===
quad([-hw, 0, 0], [-hw, D, 0], [hw, D, 0], [hw, 0, 0])

# === 屋顶（三角形棱柱，内外两层） ===
# 外表面
tri([-hw, 0, H], [hw, 0, H], [0, D/2, H+roof_h])      # 前坡
tri([-hw, 0, H], [0, D/2, H+roof_h], [-hw, D, H])      # 左坡
tri([hw, 0, H], [hw, D, H], [0, D/2, H+roof_h])         # 右坡
tri([hw, D, H], [-hw, D, H], [0, D/2, H+roof_h])        # 后坡

# 内表面
tri([hw-t, t, H-t], [-hw+t, t, H-t], [0, D/2, H+roof_h-t])
tri([-hw+t, t, H-t], [-hw+t, D-t, H-t], [0, D/2, H+roof_h-t])
tri([hw-t, D-t, H-t], [hw-t, t, H-t], [0, D/2, H+roof_h-t])
tri([-hw+t, D-t, H-t], [hw-t, D-t, H-t], [0, D/2, H+roof_h-t])

# === 门洞（前墙开口） ===
door_w, door_h = 2.0, 3.0
dx0, dx1 = -door_w/2, door_w/2
# 门洞内侧面
quad([dx0, 0, 0], [dx0, t, 0], [dx0, t, door_h], [dx0, 0, door_h])   # 左
quad([dx1, t, 0], [dx1, 0, 0], [dx1, 0, door_h], [dx1, t, door_h])   # 右
quad([dx0, t, 0], [dx1, t, 0], [dx1, t, door_h], [dx0, t, door_h])   # 顶

# === 窗户（左右墙各一个） ===
win_y0, win_y1 = 2.0, 5.0
win_z0, win_z1 = 1.0, 3.0
# 左窗
quad([-hw, win_y0, win_z0], [-hw, win_y1, win_z0], [-hw, win_y1, win_z1], [-hw, win_y0, win_z1])
quad([-hw+t, win_y1, win_z0], [-hw+t, win_y0, win_z0], [-hw+t, win_y0, win_z1], [-hw+t, win_y1, win_z1])
quad([-hw, win_y0, win_z0], [-hw+t, win_y0, win_z0], [-hw+t, win_y0, win_z1], [-hw, win_y0, win_z1])
quad([-hw+t, win_y1, win_z0], [-hw, win_y1, win_z0], [-hw, win_y1, win_z1], [-hw+t, win_y1, win_z1])
quad([-hw, win_y0, win_z1], [-hw, win_y1, win_z1], [-hw+t, win_y1, win_z1], [-hw+t, win_y0, win_z1])
quad([-hw+t, win_y0, win_z0], [-hw+t, win_y1, win_z0], [-hw, win_y1, win_z0], [-hw, win_y0, win_z0])
# 右窗
quad([hw, win_y1, win_z0], [hw, win_y0, win_z0], [hw, win_y0, win_z1], [hw, win_y1, win_z1])
quad([hw-t, win_y0, win_z0], [hw-t, win_y1, win_z0], [hw-t, win_y1, win_z1], [hw-t, win_y0, win_z1])
quad([hw-t, win_y0, win_z0], [hw, win_y0, win_z0], [hw, win_y0, win_z1], [hw-t, win_y0, win_z1])
quad([hw, win_y1, win_z0], [hw-t, win_y1, win_z0], [hw-t, win_y1, win_z1], [hw, win_y1, win_z1])
quad([hw, win_y0, win_z1], [hw, win_y1, win_z1], [hw-t, win_y1, win_z1], [hw-t, win_y0, win_z1])
quad([hw-t, win_y1, win_z0], [hw, win_y1, win_z0], [hw, win_y0, win_z0], [hw-t, win_y0, win_z0])


def write_ascii_stl(path, tris):
    with open(path, 'w') as f:
        f.write("solid house\n")
        for v0, v1, v2 in tris:
            n = normal_of(v0, v1, v2)
            f.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            f.write("    outer loop\n")
            for v in [v0, v1, v2]:
                f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid house\n")


if __name__ == '__main__':
    out = 'house.stl'
    write_ascii_stl(out, triangles)
    all_pts = np.array([v for tri in triangles for v in tri])
    print(f"Saved: {out} ({len(triangles)} triangles)")
    print(f"Bounds: x=[{all_pts[:,0].min():.1f}, {all_pts[:,0].max():.1f}] "
          f"y=[{all_pts[:,1].min():.1f}, {all_pts[:,1].max():.1f}] "
          f"z=[{all_pts[:,2].min():.1f}, {all_pts[:,2].max():.1f}]")
