import open3d as o3d
# 1. 加载 PCD
pcd = o3d.io.read_point_cloud("cable_stayed_bridge.ply")
# 2. 转换为体素栅格 (Occupancy Map 的基础)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.2)
# 3. 可视化
o3d.visualization.draw_geometries([voxel_grid])
