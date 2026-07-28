# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

vtk_datas, vtk_binaries, vtk_hiddenimports = collect_all('vtkmodules')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=vtk_binaries,
    datas=[('M4T_v2_simple.stl', '.'), ('lena.png', '.'), ('icon.png', '.'), ('icon.ico', '.')] + vtk_datas,
    hiddenimports=vtk_hiddenimports + ['PIL.Image', 'trimesh'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BridgeRoutePlanner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BridgeRoutePlanner',
)
