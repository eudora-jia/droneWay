"""PCD 点云文件解析模块"""

import numpy as np


def parse_pcd(filepath):
    """解析 PCD 文件，返回 numpy 数组 (N, 3) float64"""
    header = {}
    data_start = 0

    with open(filepath, 'rb') as f:
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('DATA'):
                header['DATA'] = line.split()[1]
                data_start = f.tell()
                break
            if line:
                parts = line.split()
                header[parts[0]] = parts[1:]

    fields = header.get('FIELDS', ['x', 'y', 'z'])
    sizes = header.get('SIZE', ['4'] * len(fields))
    types = header.get('TYPE', ['F'] * len(fields))
    width = int(header.get('WIDTH', ['0'])[0])
    height = int(header.get('HEIGHT', ['1'])[0])
    num_points = int(header.get('POINTS', [str(width * height)])[0])

    if num_points == 0:
        return np.empty((0, 3), dtype=np.float64)

    xyz_indices = []
    for i, f in enumerate(fields):
        if f.lower() in ('x', 'y', 'z'):
            xyz_indices.append(i)

    if len(xyz_indices) != 3:
        xyz_indices = [0, 1, 2]

    if header['DATA'] == 'ascii':
        with open(filepath, 'rb') as f:
            f.seek(data_start)
            raw = f.read().decode('ascii', errors='ignore')
        points = []
        for line in raw.strip().split('\n'):
            vals = line.strip().split()
            if len(vals) >= 3:
                points.append([float(vals[i]) for i in xyz_indices])
        return np.array(points, dtype=np.float64) if points else np.empty((0, 3))

    elif header['DATA'] == 'binary':
        with open(filepath, 'rb') as f:
            f.seek(data_start)
            raw = f.read()

        dtypes = []
        for s, t in zip(sizes, types):
            s = int(s)
            if t.upper() == 'F':
                dtypes.append(('f{}'.format(len(dtypes)), np.float32 if s == 4 else np.float64))
            elif t.upper() == 'U':
                dtypes.append(('f{}'.format(len(dtypes)), {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[s]))
            else:
                dtypes.append(('f{}'.format(len(dtypes)), {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[s]))

        structured = np.frombuffer(raw, dtype=np.dtype(dtypes), count=num_points)
        xyz = np.zeros((num_points, 3), dtype=np.float64)
        for j, idx in enumerate(xyz_indices):
            xyz[:, j] = structured['f{}'.format(idx)].astype(np.float64)
        return xyz

    return np.empty((0, 3), dtype=np.float64)
