/**
 * 航线生成算法 - 从 main_window.py 移植
 */

// 雷达 IMU 到 DJI IMU 的旋转矩阵（从配置文件读取）
const LIDAR_TO_DJI_R = [
  [-0.839384, 0.0720621, 0.538741],
  [0.0487258, 0.997158, -0.0574631],
  [-0.541351, -0.021983, -0.840509]
]

// 四元数工具
function quatFromAxisAngle(axis, angle) {
  const half = angle / 2
  const s = Math.sin(half)
  return [Math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s]
}

// 四元数 [w, x, y, z] -> 旋转矩阵 3x3
function quatToMatrix(q) {
  const [w, x, y, z] = q
  return [
    [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
    [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
  ]
}

// 3x3 矩阵乘法
function matMul(a, b) {
  const r = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        r[i][j] += a[i][k] * b[k][j]
  return r
}

// 3x3 矩阵转置
function matTranspose(m) {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]]
  ]
}

// 旋转矩阵 -> 四元数 [w, x, y, z]
function matrixToQuat(m) {
  const trace = m[0][0] + m[1][1] + m[2][2]
  let w, x, y, z
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1)
    w = 0.25 / s
    x = (m[2][1] - m[1][2]) * s
    y = (m[0][2] - m[2][0]) * s
    z = (m[1][0] - m[0][1]) * s
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const s = 2 * Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2])
    w = (m[2][1] - m[1][2]) / s
    x = 0.25 * s
    y = (m[0][1] + m[1][0]) / s
    z = (m[0][2] + m[2][0]) / s
  } else if (m[1][1] > m[2][2]) {
    const s = 2 * Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2])
    w = (m[0][2] - m[2][0]) / s
    x = (m[0][1] + m[1][0]) / s
    y = 0.25 * s
    z = (m[1][2] + m[2][1]) / s
  } else {
    const s = 2 * Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1])
    w = (m[1][0] - m[0][1]) / s
    x = (m[0][2] + m[2][0]) / s
    y = (m[1][2] + m[2][1]) / s
    z = 0.25 * s
  }
  return [w, x, y, z]
}

/**
 * 将四元数从地图坐标系转换到雷达 IMU 坐标系
 * R_lidar = R_map @ lidar_to_dji_R^T
 * @param {number[]} quatMap - [w, x, y, z] 地图坐标系四元数
 * @returns {number[]} [w, x, y, z] 雷达 IMU 坐标系四元数
 */
export function quatMapToLidar(quatMap) {
  const R_map = quatToMatrix(quatMap)
  const R_lidar = matMul(R_map, matTranspose(LIDAR_TO_DJI_R))
  return matrixToQuat(R_lidar)
}

function lookAtQuaternion(target, pos) {
  const dx = target[0] - pos[0]
  const dy = target[1] - pos[1]
  const dz = target[2] - pos[2]
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz)
  if (len < 1e-10) return [1, 0, 0, 0]
  const dir = [dx / len, dy / len, dz / len]
  const forward = [1, 0, 0]
  const dot = forward[0] * dir[0] + forward[1] * dir[1] + forward[2] * dir[2]
  if (dot > 0.9999) return [1, 0, 0, 0]
  if (dot < -0.9999) return [0, 0, 1, 0]
  const axis = [
    forward[1] * dir[2] - forward[2] * dir[1],
    forward[2] * dir[0] - forward[0] * dir[2],
    forward[0] * dir[1] - forward[1] * dir[0],
  ]
  const axisLen = Math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
  if (axisLen < 1e-10) return [1, 0, 0, 0]
  return quatFromAxisAngle([axis[0] / axisLen, axis[1] / axisLen, axis[2] / axisLen], Math.acos(dot))
}

/**
 * 生成平面航线
 */
export function generateFlatRoute(params) {
  const { z, spacing, wpSpacing, speed, curvature, xmin, ymin, xmax, ymax } = params
  const yCenter = (ymin + ymax) / 2
  const yHalf = ymax !== ymin ? (ymax - ymin) / 2 : 1.0
  const span = yHalf * 2

  function curvedZ(y) {
    const t = (y - yCenter) / yHalf
    return z + curvature * span * (1 - t * t) / 4
  }

  const waypoints = []
  let direction = 1
  let y = ymin
  const yStep = spacing

  while (y <= ymax + yStep * 0.5) {
    const xStart = direction === 1 ? xmin : xmax
    const xEnd = direction === 1 ? xmax : xmin
    const lineLen = Math.abs(xEnd - xStart)
    const nPts = Math.max(2, Math.floor(lineLen / wpSpacing) + 1)
    const zLine = curvedZ(y)

    for (let j = 0; j < nPts; j++) {
      const x = xStart + (xEnd - xStart) * j / (nPts - 1)
      const pos = [x, y, zLine]
      const heading = y >= yCenter ? [0, -1, 0] : [0, 1, 0]
      const target = [pos[0] + heading[0], pos[1] + heading[1], pos[2] + heading[2]]
      const quat = lookAtQuaternion(target, pos)
      waypoints.push({ pos, quat, speed, action: 'fly' })
    }

    y += yStep
    direction *= -1
  }

  return waypoints
}

/**
 * 生成立方体航线
 */
export function generateCubeRoute(params) {
  const { cx, cy, cz, dx, dy, dz, cstep, vstep, speed, startAngle } = params
  const halfX = dx / 2
  const halfY = dy / 2
  const cosA = Math.cos(startAngle)
  const sinA = Math.sin(startAngle)

  const cornersRaw = [
    [-halfX, -halfY],
    [halfX, -halfY],
    [halfX, halfY],
    [-halfX, halfY],
  ]
  const corners = cornersRaw.map(([rx, ry]) => [
    cx + rx * cosA - ry * sinA,
    cy + rx * sinA + ry * cosA,
  ])

  const numLayers = Math.max(1, Math.floor(dz / vstep))

  // 每条边独立按步距分布点
  const edgePoints = []
  for (let i = 0; i < 4; i++) {
    const c0 = corners[i]
    const c1 = corners[(i + 1) % 4]
    const ex = c1[0] - c0[0]
    const ey = c1[1] - c0[1]
    const edgeLen = Math.sqrt(ex * ex + ey * ey)
    const nPts = Math.max(2, Math.floor(edgeLen / cstep) + 1)
    const pts = []
    for (let j = 0; j < nPts; j++) {
      const ratio = j / (nPts - 1)
      const px = c0[0] + ratio * ex
      const py = c0[1] + ratio * ey
      const dLen = Math.sqrt(ex * ex + ey * ey)
      const heading = dLen > 1e-10 ? [-ey / dLen, ex / dLen, 0] : [1, 0, 0]
      pts.push({ pos2d: [px, py], heading })
    }
    edgePoints.push(pts)
  }

  const waypoints = []
  for (let layer = 0; layer <= numLayers; layer++) {
    const z = cz + layer * vstep
    const reverse = layer % 2 === 1
    const edgesOrder = reverse ? [3, 2, 1, 0] : [0, 1, 2, 3]

    for (const ei of edgesOrder) {
      const pts = edgePoints[ei]
      const ptOrder = reverse
        ? Array.from({ length: pts.length }, (_, i) => pts.length - 1 - i)
        : Array.from({ length: pts.length }, (_, i) => i)

      for (const pi of ptOrder) {
        // 跳过非最后一层的每条边最后一个点
        const lastIdx = reverse ? 0 : pts.length - 1
        const lastEdge = reverse ? 0 : 3
        if (pi === lastIdx && ei !== lastEdge && layer < numLayers) continue

        const { pos2d, heading } = pts[pi]
        const pos = [pos2d[0], pos2d[1], z]
        const target = [pos[0] + heading[0], pos[1] + heading[1], pos[2] + heading[2]]
        const quat = lookAtQuaternion(target, pos)
        waypoints.push({ pos, quat, speed, action: 'scan' })
      }
    }
  }

  return waypoints
}

/**
 * 生成圆柱体航线（螺旋线）
 */
export function generateCylinderSpiral(params) {
  const { cx, cy, cz, diam, dist, h, astep, vstep, speed, startAngle } = params
  const radius = diam / 2 + dist
  const numTurns = Math.max(1, Math.floor(h / vstep))
  const numPtsPerTurn = Math.max(8, Math.floor(360 / Math.max(1, astep)))
  const totalPts = numTurns * numPtsPerTurn
  const waypoints = []

  for (let i = 0; i <= totalPts; i++) {
    const t = i / totalPts
    const angle = startAngle + t * numTurns * 2 * Math.PI
    const z = cz + t * h
    const rx = cx + radius * Math.cos(angle)
    const ry = cy + radius * Math.sin(angle)
    const pos = [rx, ry, z]
    const inward = [cx - rx, cy - ry, 0]
    const inwardNorm = Math.sqrt(inward[0] ** 2 + inward[1] ** 2)
    const heading = inwardNorm > 1e-10 ? inward.map(v => v / inwardNorm) : [1, 0, 0]
    const target = [pos[0] + heading[0], pos[1] + heading[1], pos[2] + heading[2]]
    const quat = lookAtQuaternion(target, pos)
    waypoints.push({ pos, quat, speed, action: 'scan' })
  }

  return waypoints
}

/**
 * 生成圆柱体航线（Z字形 - 垂直往复）
 */
export function generateCylinderZigzag(params) {
  const { cx, cy, cz, diam, dist, h, astep, vstep, speed, startAngle } = params
  const radius = diam / 2 + dist
  const numCols = Math.max(1, Math.floor(360 / Math.max(1, astep)))
  const numLayers = Math.max(1, Math.floor(h / vstep))
  const waypoints = []

  for (let col = 0; col <= numCols; col++) {
    const angle = startAngle + (col / numCols) * 2 * Math.PI
    const rx = cx + radius * Math.cos(angle)
    const ry = cy + radius * Math.sin(angle)
    const inward = [cx - rx, cy - ry, 0]
    const inwardNorm = Math.sqrt(inward[0] ** 2 + inward[1] ** 2)
    const heading = inwardNorm > 1e-10 ? inward.map(v => v / inwardNorm) : [1, 0, 0]

    const layersRange = col % 2 === 0
      ? Array.from({ length: numLayers + 1 }, (_, i) => i)
      : Array.from({ length: numLayers + 1 }, (_, i) => numLayers - i)

    for (const layer of layersRange) {
      const z = cz + layer * vstep
      const pos = [rx, ry, z]
      const target = [pos[0] + heading[0], pos[1] + heading[1], pos[2] + heading[2]]
      const quat = lookAtQuaternion(target, pos)
      waypoints.push({ pos, quat, speed, action: 'scan' })
    }
  }

  return waypoints
}
