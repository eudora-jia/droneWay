"""点云文件解析模块（PCD + PLY）"""

import os
import struct

import numpy as np


def parse_pcd(filepath):
    """解析 ASCII 或 Binary PCD，返回 (N, 3) 浮点坐标数组。"""
    header = {}

    with open(filepath, 'rb') as f:
        while True:
            raw_line = f.readline()
            if not raw_line:
                raise ValueError("Invalid PCD file: missing DATA header")
            line = raw_line.decode('ascii', errors='ignore').strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            key = parts[0].upper()
            if key == 'DATA':
                if len(parts) != 2:
                    raise ValueError("Invalid PCD DATA header")
                data_format = parts[1].lower()
                data_start = f.tell()
                break
            header[key] = parts[1:]

    fields = header.get('FIELDS', ['x', 'y', 'z'])
    sizes = header.get('SIZE', ['4'] * len(fields))
    types = header.get('TYPE', ['F'] * len(fields))
    counts = header.get('COUNT', ['1'] * len(fields))
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("Invalid PCD header: FIELDS/SIZE/TYPE/COUNT length mismatch")
    counts = [int(count) for count in counts]
    if any(count < 1 for count in counts):
        raise ValueError("Invalid PCD header: COUNT values must be positive")
    width = int(header.get('WIDTH', ['0'])[0])
    height = int(header.get('HEIGHT', ['1'])[0])
    num_points = int(header.get('POINTS', [str(width * height)])[0])
    if num_points == 0:
        return np.empty((0, 3), dtype=np.float64)

    xyz_field_indices = {
        field.lower(): index
        for index, field in enumerate(fields)
        if field.lower() in ('x', 'y', 'z')
    }
    if set(xyz_field_indices) != {'x', 'y', 'z'}:
        raise ValueError("PCD file must contain x, y, and z fields")
    if any(counts[xyz_field_indices[axis]] != 1 for axis in ('x', 'y', 'z')):
        raise ValueError("PCD x, y, and z fields must have COUNT 1")

    field_offsets = np.cumsum([0] + counts[:-1])
    xyz_value_indices = [
        field_offsets[xyz_field_indices[axis]] for axis in ('x', 'y', 'z')
    ]

    if data_format == 'ascii':
        points = []
        max_index = max(xyz_value_indices)
        with open(filepath, 'rb') as f:
            f.seek(data_start)
            for raw_line in f:
                values = raw_line.decode('ascii', errors='ignore').split()
                if len(values) > max_index:
                    points.append([float(values[index]) for index in xyz_value_indices])
        return np.asarray(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)

    if data_format != 'binary':
        raise ValueError(f"Unsupported PCD DATA mode: {data_format}")

    dtype_fields = []
    for index, (size, value_type, count) in enumerate(zip(sizes, types, counts)):
        size = int(size)
        if value_type.upper() == 'F':
            dtype = {4: np.float32, 8: np.float64}.get(size)
        elif value_type.upper() == 'U':
            dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(size)
        else:
            dtype = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}.get(size)
        if dtype is None:
            raise ValueError(f"Unsupported PCD field size: {size}")
        if count == 1:
            dtype_fields.append((f'f{index}', dtype))
        else:
            dtype_fields.append((f'f{index}', dtype, (count,)))

    point_dtype = np.dtype(dtype_fields)
    expected_size = data_start + point_dtype.itemsize * num_points
    if os.path.getsize(filepath) < expected_size:
        raise ValueError("Truncated binary PCD data")
    structured = np.memmap(
        filepath, dtype=point_dtype, mode="r", offset=data_start, shape=(num_points,)
    )
    coordinate_dtype = np.result_type(
        *[
            point_dtype.fields[f"f{xyz_field_indices[axis]}"][0]
            for axis in ('x', 'y', 'z')
        ]
    )
    points = np.empty((num_points, 3), dtype=coordinate_dtype)
    for output_index, axis in enumerate(('x', 'y', 'z')):
        field_index = xyz_field_indices[axis]
        points[:, output_index] = structured[f"f{field_index}"]
    return points


def parse_ply(filepath):
    """解析 PLY 文件，返回 (points, colors)
    points: (N, 3) float64
    colors: (N, 3) uint8，无颜色属性时返回 None
    """
    with open(filepath, 'rb') as f:
        # 读取头部
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
            if not line:
                return np.empty((0, 3), dtype=np.float64), None

        # 解析头部信息
        format_type = 'ascii'
        num_vertices = 0
        properties = []
        in_vertex = False

        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'format':
                format_type = parts[1]
            elif parts[0] == 'element' and parts[1] == 'vertex':
                num_vertices = int(parts[2])
                in_vertex = True
            elif parts[0] == 'element':
                in_vertex = False
            elif parts[0] == 'property' and in_vertex:
                properties.append((parts[1], parts[2] if len(parts) > 2 else ''))

        if num_vertices == 0:
            return np.empty((0, 3), dtype=np.float64), None

        # 找出 xyz 和 rgb 的列索引
        xyz_indices = []
        rgb_indices = []
        for i, (ptype, pname) in enumerate(properties):
            if pname in ('x', 'y', 'z'):
                xyz_indices.append(i)
            if pname in ('red', 'green', 'blue'):
                rgb_indices.append(i)

        has_color = len(rgb_indices) == 3

        if format_type == 'ascii':
            return _parse_ply_ascii(f, num_vertices, xyz_indices, rgb_indices, has_color)
        elif format_type == 'binary_little_endian':
            return _parse_ply_binary_le(f, num_vertices, properties, xyz_indices, rgb_indices, has_color)
        else:
            return np.empty((0, 3), dtype=np.float64), None


def _parse_ply_ascii(f, num_vertices, xyz_indices, rgb_indices, has_color):
    """解析 ASCII PLY"""
    points = np.zeros((num_vertices, 3), dtype=np.float64)
    colors = np.zeros((num_vertices, 3), dtype=np.uint8) if has_color else None

    for i in range(num_vertices):
        line = f.readline().decode('ascii', errors='ignore').strip()
        if not line:
            break
        vals = line.split()
        for j, idx in enumerate(xyz_indices):
            points[i, j] = float(vals[idx])
        if has_color:
            for j, idx in enumerate(rgb_indices):
                colors[i, j] = int(vals[idx])

    return points, colors


def _parse_ply_binary_le(f, num_vertices, properties, xyz_indices, rgb_indices, has_color):
    """解析 binary_little_endian PLY"""
    type_map = {
        'float': ('f4', np.float32),
        'float32': ('f4', np.float32),
        'double': ('f8', np.float64),
        'float64': ('f8', np.float64),
        'uchar': ('u1', np.uint8),
        'uint8': ('u1', np.uint8),
        'int': ('i4', np.int32),
        'int32': ('i4', np.int32),
        'uint': ('u4', np.uint32),
        'uint32': ('u4', np.uint32),
        'short': ('i2', np.int16),
        'int16': ('i2', np.int16),
        'ushort': ('u2', np.uint16),
        'uint16': ('u2', np.uint16),
    }

    dtypes = []
    for ptype, pname in properties:
        np_type = type_map.get(ptype)
        if np_type is None:
            np_type = ('f4', np.float32)
        dtypes.append((pname, np.dtype(np_type[1])))

    row_size = sum(dt.itemsize for _, dt in dtypes)
    raw = f.read(row_size * num_vertices)

    if len(raw) < row_size * num_vertices:
        actual_count = len(raw) // row_size
        num_vertices = actual_count

    structured = np.frombuffer(raw[:row_size * num_vertices], dtype=np.dtype(dtypes), count=num_vertices)

    points = np.zeros((num_vertices, 3), dtype=np.float64)
    for j, name in enumerate(['x', 'y', 'z']):
        if name in structured.dtype.names:
            points[:, j] = structured[name].astype(np.float64)

    colors = None
    if has_color:
        colors = np.zeros((num_vertices, 3), dtype=np.uint8)
        for j, name in enumerate(['red', 'green', 'blue']):
            if name in structured.dtype.names:
                colors[:, j] = structured[name].astype(np.uint8)

    return points, colors
