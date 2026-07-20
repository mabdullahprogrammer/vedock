# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(SPECPATH).resolve().parent
installer_source = project_root / 'installer' / 'vedock_installer.py'
asset_root = project_root / 'vedock_cli' / 'assets'


a = Analysis(
    [str(installer_source)],
    pathex=[],
    binaries=[],
    datas=[(str(asset_root / 'logo.png'), 'assets'), (str(asset_root / 'logo.ico'), 'assets')],
    hiddenimports=['webview.platforms.edgechromium'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['win32com', 'pythoncom', 'pywintypes', 'qtpy', 'gi'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VedockInstaller',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(asset_root / 'logo.ico')],
)
