# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

block_cipher = None

# Collect all necessary files and hidden imports for streamlit and other libraries
datas = []
binaries = []
hiddenimports = []

# Collect streamlit datas and hidden imports
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Collect altair datas and hidden imports
tmp_ret = collect_all('altair')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Collect pandas datas and hidden imports
tmp_ret = collect_all('pandas')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Collect pyarrow datas and hidden imports
tmp_ret = collect_all('pyarrow')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]
datas += copy_metadata('pyarrow')

# Collect sklearn datas and hidden imports
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Add conda DLLs (crucial for pyarrow and others on Windows)
import glob
conda_bin_dir = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.exists(conda_bin_dir):
    print(f"Collecting DLLs from {conda_bin_dir}")
    dlls = glob.glob(os.path.join(conda_bin_dir, '*.dll'))
    for dll in dlls:
        binaries.append((dll, '.'))

# Add other hidden imports explicitly if needed
hiddenimports += [
    'streamlit.web.cli',
    'numpy',
    'PIL',
    'rich',
    'click',
    'toml',
    'watchdog',
    'tornado',
    'streamlit_desktop_app',
    'webview',
    'sklearn.utils._typedefs',
    'sklearn.neighbors._partition_nodes',
]

# Add application specific data
datas += [
    ('app', 'app'),
    ('app/.streamlit', '.streamlit'),
]

a = Analysis(
    ['run_app.py'],
    pathex=['D:\\anaconda\\envs\\st\\Library\\bin'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Sth-in-EB',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Sth-in-EB',
)
