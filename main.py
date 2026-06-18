"""
桥梁巡检无人机航线规划工具
Bridge Inspection Drone Waypoint Planner
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
