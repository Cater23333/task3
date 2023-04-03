from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from utils import MyApp


if __name__ == "__main__":  #主程序入口
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec())






