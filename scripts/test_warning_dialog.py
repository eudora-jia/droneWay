"""Headless test for the bounded scrollable warning dialog (_show_warning_details)."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QApplication, QWidget, QDialogButtonBox, QPlainTextEdit)

app = QApplication([])
from main_window import MainWindow


class Dummy(QWidget):
    _show_warning_details = MainWindow._show_warning_details
    _WARNING_DETAIL_LIMIT = MainWindow._WARNING_DETAIL_LIMIT


d = Dummy()


def _active_dialog():
    return QApplication.activeModalWidget()


def close_it():
    w = _active_dialog()
    if w is not None:
        w.close()


def click_yes():
    w = _active_dialog()
    if w is None:
        return
    box = w.findChild(QDialogButtonBox)
    box.button(QDialogButtonBox.Yes).click()


def detail_lines():
    w = _active_dialog()
    if w is None:
        return -1
    return len(w.findChild(QPlainTextEdit).toPlainText().splitlines())


# 1) long list is truncated and the dialog stays bounded
QTimer.singleShot(200, lambda: (print('lines in box:', detail_lines()), close_it()))
res = d._show_warning_details("约束警告", "生成 400 个航点，500 条约束未完全满足",
                              [f"航点{i+1} 约束未满足: 距离过近" for i in range(500)])
print('closed-without-confirm returns:', res)

# 2) confirm mode: Yes -> True
QTimer.singleShot(200, click_yes)
res_yes = d._show_warning_details("碰撞警告", "检测到 3 个问题，是否仍要保存？",
                                  ["1 号航点碰撞", "2 号航点碰撞", "3 号航点碰撞"],
                                  confirm=True)
print('confirm Yes returns:', res_yes)

# 3) confirm mode: No -> False
QTimer.singleShot(200, close_it)
res_no = d._show_warning_details("碰撞警告", "检测到 3 个问题，是否仍要保存？",
                                 ["1 号航点碰撞"], confirm=True)
print('confirm No/Escape returns:', res_no)
print('DIALOG TEST DONE')
