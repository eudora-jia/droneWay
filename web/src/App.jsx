import React, { useState, useCallback } from 'react'
import Viewer3D from './components/Viewer3D'
import SidePanel from './components/SidePanel'

export default function App() {
  const [pointCloud, setPointCloud] = useState(null)
  const [waypoints, setWaypoints] = useState([])
  const [routeFile, setRouteFile] = useState(null)

  const handleLoadPCD = useCallback((data) => {
    setPointCloud(data)
  }, [])

  const handleLoadRoute = useCallback((data) => {
    setWaypoints(data.waypoints || [])
    setRouteFile(data)
  }, [])

  const handleExportRoute = useCallback(() => {
    if (!routeFile) return
    const json = JSON.stringify(routeFile, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = routeFile.fileName || 'route.json'
    a.click()
    URL.revokeObjectURL(url)
  }, [routeFile])

  return (
    <div style={{ display: 'flex', width: '100%', height: '100%' }}>
      <SidePanel
        pointCloud={pointCloud}
        waypoints={waypoints}
        onLoadPCD={handleLoadPCD}
        onLoadRoute={handleLoadRoute}
        onExportRoute={handleExportRoute}
      />
      <Viewer3D
        pointCloud={pointCloud}
        waypoints={waypoints}
      />
    </div>
  )
}
