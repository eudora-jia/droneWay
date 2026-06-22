import React, { useCallback } from 'react'

export default function SidePanel({ pointCloud, waypoints, onLoadPCD, onLoadRoute, onExportRoute }) {

  // 解析 PCD 文件（ASCII 格式简单解析）
  const handlePCDFile = useCallback((e) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = ev.target.result
      const points = parsePCD(text)
      onLoadPCD(points)
    }
    reader.readAsText(file)
  }, [onLoadPCD])

  // 加载航线 JSON
  const handleRouteFile = useCallback((e) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result)
        data.fileName = file.name
        onLoadRoute(data)
      } catch (err) {
        alert('JSON 解析失败: ' + err.message)
      }
    }
    reader.readAsText(file)
  }, [onLoadRoute])

  const panelStyle = {
    width: 280,
    height: '100%',
    background: '#f8f8f6',
    borderRight: '1px solid #ddd',
    overflowY: 'auto',
    padding: 12,
    fontFamily: 'system-ui, -apple-system, sans-serif',
    fontSize: 13
  }

  const btnStyle = {
    width: '100%',
    padding: '8px 12px',
    marginBottom: 8,
    border: '1px solid #ccc',
    borderRadius: 4,
    background: '#e8e8e6',
    cursor: 'pointer',
    fontSize: 13
  }

  const sectionStyle = {
    marginBottom: 16,
    padding: 10,
    background: '#fff',
    borderRadius: 6,
    border: '1px solid #e0e0e0'
  }

  return (
    <div style={panelStyle}>
      <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#333' }}>桥梁航线规划</h3>

      <div style={sectionStyle}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>文件</div>
        <label style={{ ...btnStyle, display: 'block', textAlign: 'center' }}>
          加载点云 (PCD)
          <input type="file" accept=".pcd" onChange={handlePCDFile} style={{ display: 'none' }} />
        </label>
        <label style={{ ...btnStyle, display: 'block', textAlign: 'center' }}>
          加载航线 (JSON)
          <input type="file" accept=".json" onChange={handleRouteFile} style={{ display: 'none' }} />
        </label>
        <button style={btnStyle} onClick={onExportRoute} disabled={!waypoints.length}>
          导出航线
        </button>
      </div>

      <div style={sectionStyle}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>状态</div>
        <div style={{ color: '#666' }}>
          点云: {pointCloud ? `${pointCloud.length} 点` : '未加载'}
        </div>
        <div style={{ color: '#666' }}>
          航点: {waypoints.length} 个
        </div>
      </div>

      <div style={sectionStyle}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>操作说明</div>
        <ul style={{ margin: 0, paddingLeft: 16, color: '#666', lineHeight: 1.8 }}>
          <li>左键拖动: 平移</li>
          <li>右键拖动: 旋转</li>
          <li>滚轮: 缩放</li>
        </ul>
      </div>
    </div>
  )
}

// 简单 PCD ASCII 解析
function parsePCD(text) {
  const lines = text.split('\n')
  const points = []
  let dataStart = false

  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed.startsWith('DATA')) {
      dataStart = true
      continue
    }
    if (!dataStart || !trimmed) continue

    const parts = trimmed.split(/\s+/).map(Number)
    if (parts.length >= 3 && !isNaN(parts[0])) {
      points.push([parts[0], parts[1], parts[2]])
    }
  }

  return points
}
