import React, { useRef, useEffect, useCallback } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

export default function Viewer3D({ pointCloud, waypoints }) {
  const containerRef = useRef(null)
  const sceneRef = useRef(null)
  const rendererRef = useRef(null)
  const cameraRef = useRef(null)
  const controlsRef = useRef(null)
  const cloudGroupRef = useRef(null)
  const routeGroupRef = useRef(null)

  // 初始化 Three.js 场景
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // 场景
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0xf5f5ef) // 奶白色
    sceneRef.current = scene

    // 相机
    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 10000)
    camera.position.set(50, -80, 60)
    camera.up.set(0, 0, 1)
    cameraRef.current = camera

    // 渲染器
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.setPixelRatio(window.devicePixelRatio)
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // 控制器：左键平移，右键旋转，滚轮缩放
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.PAN,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.ROTATE
    }
    controls.target.set(0, 0, 0)
    controls.update()
    controlsRef.current = controls

    // 坐标轴
    const axes = new THREE.AxesHelper(10)
    scene.add(axes)

    // 网格
    const grid = new THREE.GridHelper(100, 20, 0xcccccc, 0xe0e0e0)
    grid.rotation.x = Math.PI / 2 // XY 平面
    scene.add(grid)

    // 点云组
    const cloudGroup = new THREE.Group()
    scene.add(cloudGroup)
    cloudGroupRef.current = cloudGroup

    // 航线组
    const routeGroup = new THREE.Group()
    scene.add(routeGroup)
    routeGroupRef.current = routeGroup

    // 动画循环
    function animate() {
      requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    // 窗口大小变化
    const onResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(container.clientWidth, container.clientHeight)
    }
    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)
      container.removeChild(renderer.domElement)
      renderer.dispose()
    }
  }, [])

  // 更新点云
  useEffect(() => {
    const group = cloudGroupRef.current
    if (!group) return

    // 清除旧点云
    while (group.children.length) {
      const child = group.children[0]
      group.remove(child)
      child.geometry?.dispose()
      child.material?.dispose()
    }

    if (!pointCloud || pointCloud.length === 0) return

    const geometry = new THREE.BufferGeometry()
    const positions = new Float32Array(pointCloud.length * 3)
    const colors = new Float32Array(pointCloud.length * 3)

    for (let i = 0; i < pointCloud.length; i++) {
      const p = pointCloud[i]
      positions[i * 3] = p[0]
      positions[i * 3 + 1] = p[1]
      positions[i * 3 + 2] = p[2]
      // 根据 Z 值着色（蓝→红）
      const z = p[2]
      const t = Math.min(1, Math.max(0, z / 20))
      colors[i * 3] = t
      colors[i * 3 + 1] = 0.3
      colors[i * 3 + 2] = 1 - t
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))

    const material = new THREE.PointsMaterial({ size: 0.15, vertexColors: true })
    const points = new THREE.Points(geometry, material)
    group.add(points)

    // 自动调整相机
    const box = new THREE.Box3().setFromBufferAttribute(geometry.getAttribute('position'))
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())
    const maxDim = Math.max(size.x, size.y, size.z)
    const cam = cameraRef.current
    if (cam) {
      cam.position.set(center.x + maxDim, center.y - maxDim * 1.5, center.z + maxDim)
      cam.lookAt(center)
      controlsRef.current?.target.copy(center)
    }
  }, [pointCloud])

  // 更新航线
  useEffect(() => {
    const group = routeGroupRef.current
    if (!group) return

    while (group.children.length) {
      const child = group.children[0]
      group.remove(child)
      child.geometry?.dispose()
      child.material?.dispose()
    }

    if (!waypoints || waypoints.length === 0) return

    // 航点球体
    const sphereGeo = new THREE.SphereGeometry(0.3, 16, 16)
    waypoints.forEach((wp, i) => {
      const mat = new THREE.MeshBasicMaterial({ color: i === 0 ? 0x00ff00 : 0xff3333 })
      const sphere = new THREE.Mesh(sphereGeo, mat)
      sphere.position.set(wp.pos[0], wp.pos[1], wp.pos[2])
      group.add(sphere)
    })

    // 航线连接线
    if (waypoints.length >= 2) {
      const lineGeo = new THREE.BufferGeometry()
      const linePositions = new Float32Array(waypoints.length * 3)
      waypoints.forEach((wp, i) => {
        linePositions[i * 3] = wp.pos[0]
        linePositions[i * 3 + 1] = wp.pos[1]
        linePositions[i * 3 + 2] = wp.pos[2]
      })
      lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3))
      const lineMat = new THREE.LineBasicMaterial({ color: 0x00ccff, linewidth: 2 })
      const line = new THREE.Line(lineGeo, lineMat)
      group.add(line)
    }

    // 航向线段
    const headingLen = 1.5
    const headingGeo = new THREE.BufferGeometry()
    const headingPositions = []
    waypoints.forEach((wp) => {
      const pos = new THREE.Vector3(wp.pos[0], wp.pos[1], wp.pos[2])
      // 从四元数计算方向
      const quat = new THREE.Quaternion(wp.quat[1], wp.quat[2], wp.quat[3], wp.quat[0])
      const dir = new THREE.Vector3(1, 0, 0).applyQuaternion(quat)
      const end = pos.clone().add(dir.multiplyScalar(headingLen))
      headingPositions.push(pos.x, pos.y, pos.z, end.x, end.y, end.z)
    })
    headingGeo.setAttribute('position', new THREE.Float32BufferAttribute(headingPositions, 3))
    const headingMat = new THREE.LineBasicMaterial({ color: 0x00eaff })
    const headingLines = new THREE.LineSegments(headingGeo, headingMat)
    group.add(headingLines)
  }, [waypoints])

  return (
    <div ref={containerRef} style={{ flex: 1, height: '100%' }} />
  )
}
