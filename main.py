"""
桥梁巡检无人机航线规划工具
Bridge Inspection Drone Waypoint Planner
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QIcon
from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    # 设置应用图标（支持 PyInstaller 打包路径）
    icon_search = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    if getattr(sys, 'frozen', False):
        icon_search.insert(0, sys._MEIPASS)
    for d in icon_search:
        icon_path = os.path.join(d, 'icon.png')
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
            break

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
