def findweb(name):  #this function is to find the url of the author
    import requests
    from bs4 import BeautifulSoup

    name = name.replace(' ', '+')
    url="https://dblp.org/search?q="+name
    bs = BeautifulSoup(requests.get(url).content, "html.parser")
    for node in bs.find_all('ul', class_='result-list'):
         linklist = node.find_all('a')
         print(linklist[0]['href'])
         return linklist[0]['href']  #return str



def getart_name(url):  #this function is to get the title of article
    import requests
    from bs4 import BeautifulSoup

    bs = BeautifulSoup(requests.get(url).content, "html.parser")
    art_name=[]
    for node in bs.find_all('span', class_='title'):
        art_name.append(node.string)
    return art_name  #return list



from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(683, 415)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 30, 311, 51))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 90, 141, 51))
        self.pushButton.setMaximumSize(QtCore.QSize(151, 51))
        self.pushButton.setObjectName("pushButton")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(370, 30, 231, 51))
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(70, 240, 111, 51))
        self.label_2.setObjectName("label_2")
        self.plainTextEdit_2 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_2.setGeometry(QtCore.QRect(300, 140, 271, 181))
        self.plainTextEdit_2.setObjectName("plainTextEdit_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 683, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt;\">请输入计算机科学家姓名（英语）</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "开始查询"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:18pt;\">结果：</span></p><p><br/></p></body></html>"))


class MyApp(QtWidgets.QMainWindow): #define a window class for main function to call
    def __init__(self):
        super().__init__()
        self.__ui = Ui_MainWindow()  # import object
        self.__ui.setupUi(self)
        self.__ui.pushButton.clicked.connect(self.find_name)  #connect function with button

    def find_name(self):  #the usage of button
            name = str(self.__ui.plainTextEdit.toPlainText()) #get value from plainTextEdit
            url=findweb(name)
            art_name = getart_name(url)
            for i,j in enumerate(art_name):
            #self.__ui.plainTextEdit_2.setPlainText((art_name[0]))  # put list into textEdit
                self.__ui.plainTextEdit_2.insertPlainText(str(i+1)+','+j+'\n'+'\n')  #put list into textEdit


