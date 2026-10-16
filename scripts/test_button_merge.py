"""Headless smoke test for merged buttons (自动算间距→应用, 规划安全过渡+应用)."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QPushButton

app = QApplication([])
import vtk_viewer
# 离屏 GL 在 setup_scene 的 Render 处会崩溃（环境已知问题），按钮清单测试不需要场景。
vtk_viewer.VTKViewer.setup_scene = lambda self: None
from main_window import MainWindow

w = MainWindow()
texts = [b.text() for b in w.findChildren(QPushButton)]
print('自动算间距 buttons:', texts.count('自动算间距'))
print('应用 buttons:', texts.count('应用'), '(expect 4: flat/cube/cyl/line tabs)')
print('plan transition button:', w.btn_plan_transition.text())

# 规划安全过渡 = 应用安全设置 + 规划过渡；无点云时应弹出提示并安全返回
shown = []


def close_modal():
    m = QApplication.activeModalWidget()
    if m is not None:
        shown.append(m.windowTitle())
        m.close()


QTimer.singleShot(300, close_modal)
w.btn_plan_transition.click()
app.processEvents()
print('modal after plan click:', shown)

# 语言切换仍应工作（翻译表删过条目）
w._switch_language('en')
w._switch_language('zh')
en_texts = None
w._switch_language('en')
en_texts = [b.text() for b in w.findChildren(QPushButton)]
print('en Auto Calc buttons:', en_texts.count('Auto Calc'))
w._switch_language('zh')
print('SMOKE DONE')
