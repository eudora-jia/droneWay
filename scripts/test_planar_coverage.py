"""Headless functional test for planar coverage overlay + FPV corner axes gizmo.

add_point_cloud crashes under offscreen Qt in this environment (pre-existing,
unrelated to these changes), so the test drives add_route / FPV directly.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt5.QtWidgets import QApplication

app = QApplication([])
from vtk_viewer import VTKViewer

v = VTKViewer()
v.resize(800, 600)
# offscreen QVTKOpenGLWidget rendering crashes in this environment (pre-existing);
# the logic under test does not need actual GL output.
v.vtk_widget.GetRenderWindow().Render = lambda *a, **k: None
print('viewer ok', flush=True)

wps = []
for y, z in [(-3, 2), (-1, 2), (1, 2), (3, 2), (3, 4), (1, 4), (-1, 4), (-3, 4)]:
    wps.append({
        'pos': np.array([-3.0, y, z]),
        'target_pos': np.array([0.0, y, z]),
        'speed': 2.0,
        'action': 'fly',
    })

plane = {'origin': np.array([0.0, 0.0, 4.0]), 'normal': np.array([-1.0, 0.0, 0.0])}

# ── planar coverage path ──
v._coverage_plane = plane
before = len(v._actors)
v.add_route(wps, reset_camera=False)
with_plane = len(v._actors) - before
print('actors added WITH plane:', with_plane, flush=True)

cov_actor = None
for a in reversed(v._actors):
    mapper = a.GetMapper()
    pd = mapper.GetInputDataObject(0, 0) if mapper else None
    if pd is not None and pd.GetNumberOfPoints() == 32 and pd.GetNumberOfPolys() == 8:
        cov_actor = a
        break
if cov_actor is None:
    print('PLANAR QUAD ACTOR: not found')
else:
    pd = cov_actor.GetMapper().GetInputDataObject(0, 0)
    pts = np.array([pd.GetPoint(i) for i in range(pd.GetNumberOfPoints())])
    print(f'planar quad x-residual max = {np.abs(pts[:, 0]).max():.4f} (plane lift offset)')
    print('quad z range:', round(pts[:, 2].min(), 2), round(pts[:, 2].max(), 2),
          'y range:', round(pts[:, 1].min(), 2), round(pts[:, 1].max(), 2))

# ── legacy surface projection path (no plane) ──
v._coverage_plane = None
before = len(v._actors)
v.add_route(wps, reset_camera=False)
print('actors added WITHOUT plane:', len(v._actors) - before, flush=True)

# ── FPV corner gizmo ──
v.toggle_fpv(True)
corner = v._fpv_orientation_marker
vp0 = tuple(round(x, 3) for x in corner.GetViewport())
cam0 = tuple(round(x, 3) for x in corner.GetActiveCamera().GetPosition())
print('corner draw:', corner.GetDraw(), 'viewport:', vp0, 'cam:', cam0, flush=True)
v._fpv_yaw = 0.0
v._update_fpv_camera()
cam1 = tuple(round(x, 3) for x in corner.GetActiveCamera().GetPosition())
vp1 = tuple(round(x, 3) for x in corner.GetViewport())
print('after yaw change cam:', cam1, 'viewport:', vp1,
      'cam_follows:', cam0 != cam1,
      'viewport_fixed:', vp0 == vp1 == (0.0, 0.0, 0.16, 0.16), flush=True)
v.toggle_fpv(False)
print('corner draw after exit:', corner.GetDraw(), flush=True)
print('TEST DONE')
