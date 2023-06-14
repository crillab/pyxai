from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QAbstractItemView, QHeaderView, QDesktopWidget, QBoxLayout, QFrame, QMessageBox, QFileDialog, QLabel, QSizePolicy, QScrollArea,  QStyleFactory, QMainWindow, QTableWidgetItem, QHBoxLayout, QMenu, QAction, QGroupBox, QListWidget, QWidget, QVBoxLayout, QGridLayout, QTableWidget

from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QColor
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

import sys
import webbrowser
import numpy
import colorsys
import math
from pyxai.sources.core.tools.ImageViewSync import QImageViewSync
from pyxai.sources.core.tools.vizualisation import PyPlotImageGenerator, PyPlotDiagramGenerator


class GraphicalInterface(QMainWindow):
    """Main Window."""
    def __init__(self, explainer, image_size=None):
        """Initializer."""
        app = QApplication(sys.argv)
        
        self.explainer = explainer
        self.image_size = image_size
        self.feature_names = explainer.get_feature_names()
        self.feature_values = dict()
        if self.image_size is not None:
            if not isinstance(self.image_size, tuple) or len(self.image_size) != 2:
                raise ValueError("The 'image' parameter must be a tuple of size 2 representing the number of pixels (x_axis, y_axis).") 
            self.pyplot_image_generator = PyPlotImageGenerator(image_size, 256)

        self.pyplot_diagram_generator = PyPlotDiagramGenerator()
        super().__init__(None)
        self.originalPalette = QApplication.palette()
        main_layout = QGridLayout()

        self.setWindowTitle("PyXAI")
        self.resize(400, 200)
        
        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.create_menu_bar()
        instance_layout = self.create_instance_group()
        main_layout.addLayout(instance_layout, 0, 0)

        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.VLine)
        self.separator.setFrameShadow(QFrame.Raised)
        layout_separator = QHBoxLayout()
        layout_separator.addWidget(self.separator)
       

        main_layout.addLayout(layout_separator, 0, 1)

        explanation_layout = self.create_explanation_group()
        main_layout.addLayout(explanation_layout, 0, 2)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        
        self.show()
        sys.exit(app.exec_())

    def create_instance_group(self):
        self.instance_group = QGroupBox("Instances")
        
        self.instance_group.repaint()
        self.list_instance = QListWidget()
        instances = tuple("Instance "+str(i) for i in range(1, len(self.explainer._history.keys())+1))
        
        #List of instances
        self.list_instance.addItems(instances)
        self.list_instance.clicked.connect(self.clicked_instance)
        self.list_instance.setMaximumWidth(100)
        self.list_instance.setMinimumWidth(100)

        #Table of the selected instance
        self.table_instance = QTableWidget(len(self.feature_names), 2)
        self.table_instance.verticalHeader().setVisible(False)
        self.table_instance.setHorizontalHeaderItem(0, QTableWidgetItem("Name"))
        self.table_instance.setHorizontalHeaderItem(1, QTableWidgetItem("Value"))
        self.table_instance.setMaximumWidth(220)
        self.table_instance.setMinimumWidth(220)
        for i, name in enumerate(self.feature_names):
            self.table_instance.setItem(i, 0, QTableWidgetItem(str(name)))
        
        if self.image_size is not None:
            #Image of the selected instance
            self.imageLabelLeft = QLabel()
            self.imageLabelLeft.setBackgroundRole(QPalette.Base)
            self.imageLabelLeft.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            #self.imageLabelLeft.setScaledContents(True)

            self.imageLabelLeft.setMinimumWidth(300)
            self.imageLabelLeft.setMinimumHeight(300)
            
            self.scrollAreaLeft = QScrollArea()
            self.scrollAreaLeft.setBackgroundRole(QPalette.Dark)
            self.scrollAreaLeft.setWidget(self.imageLabelLeft)
            self.scrollAreaLeft.setVisible(True)

            self.scrollAreaLeft.setMinimumWidth(302)
            self.scrollAreaLeft.setMinimumHeight(302)

            self.scrollAreaLeft.mouseMoveEvent = self.mouseMoveEventLeft
            self.scrollAreaLeft.mousePressEvent = self.mousePressEventLeft
            self.scrollAreaLeft.mouseReleaseEvent = self.mouseReleaseEventLeft

            self.imageLabelLeft.setCursor(Qt.OpenHandCursor)
            
        layout = QGridLayout()
        layout.addWidget(self.list_instance, 0, 0)
        layout.addWidget(self.table_instance, 0, 1)
        if self.image_size is not None:
            layout.addWidget(self.scrollAreaLeft, 0, 2)
        
        return layout
    
    def to_rgb(self, value):
        return QColor().fromRgb(int(value)) 
    

    def display_left(self, instance):
        image = self.pyplot_image_generator.generate_instance(instance)
        self.scaleFactor = 1.0
        size = self.imageLabelLeft.size()
        self.imageLabelLeft.setPixmap(QPixmap.fromImage(image).scaled(size))
        
    def display_right(self, instance, reason=None):
        if self.image_size is not None:
            image = self.pyplot_image_generator.generate_explanation(instance, reason)
            size = self.imageLabelRight.size()
            self.imageLabelRight.setPixmap(QPixmap.fromImage(image).scaled(size))
        else:
            image = self.pyplot_diagram_generator.generate_explanation(self.feature_values, instance, reason)
            qpixmap = QPixmap.fromImage(image)
            self.scrollAreaRight.setMinimumWidth(qpixmap.width()+16)
            #self.imageLabelRight.setMinimumWidth(qpixmap.width())
            #self.imageLabelRight.setMinimumHeight(qpixmap.width())
            self.imageLabelRight.setPixmap(qpixmap)
            self.imageLabelRight.adjustSize()
            
            
        

    def mousePressEventLeft(self, event):
        self.pressed = True
        self.imageLabelLeft.setCursor(Qt.ClosedHandCursor)
        self.initialPosX = self.scrollAreaLeft.horizontalScrollBar().value() + event.pos().x()
        self.initialPosY = self.scrollAreaLeft.verticalScrollBar().value() + event.pos().y()

    def mouseReleaseEventLeft(self, event):
        self.pressed = False
        self.imageLabelLeft.setCursor(Qt.OpenHandCursor)
        self.initialPosX = self.scrollAreaLeft.horizontalScrollBar().value()
        self.initialPosY = self.scrollAreaLeft.verticalScrollBar().value()

    def mouseMoveEventLeft(self, event):
        if self.pressed:
            self.scrollAreaLeft.horizontalScrollBar().setValue(self.initialPosX - event.pos().x())
            self.scrollAreaLeft.verticalScrollBar().setValue(self.initialPosY - event.pos().y())

    def create_explanation_group(self):
        self.instance_group = QGroupBox()
        
        self.table_explanation = QTableWidget(0, 4)
        self.table_explanation.verticalHeader().setVisible(False)
        self.table_explanation.setHorizontalHeaderItem(0, QTableWidgetItem("Id Method"))
        self.table_explanation.setHorizontalHeaderItem(1, QTableWidgetItem("Name"))
        self.table_explanation.setHorizontalHeaderItem(2, QTableWidgetItem("Id Reason"))
        self.table_explanation.setHorizontalHeaderItem(3, QTableWidgetItem("Lenght"))
        self.table_explanation.clicked.connect(self.clicked_explanation)

        header = self.table_explanation.horizontalHeader()       
        self.table_explanation.setColumnWidth(0, 80)
        self.table_explanation.setColumnWidth(1, 200)
        self.table_explanation.setColumnWidth(2, 80)
        self.table_explanation.setColumnWidth(3, 80)
        self.table_explanation.setMinimumWidth(80+200+80+80+2)
        self.table_explanation.setMaximumWidth(80+200+80+80+2)
        self.table_explanation.setSelectionBehavior(QAbstractItemView.SelectRows);
        #Image of the explanation
        self.imageLabelRight = QLabel()
        self.imageLabelRight.setBackgroundRole(QPalette.Base)
        self.imageLabelRight.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        #self.imageLabelRight.setScaledContents(True)

        self.imageLabelRight.setMinimumWidth(300)
        self.imageLabelRight.setMinimumHeight(300)
        
        self.scrollAreaRight = QScrollArea()
        self.scrollAreaRight.setBackgroundRole(QPalette.Dark)
        self.scrollAreaRight.setWidget(self.imageLabelRight)
        self.scrollAreaRight.setVisible(True)

        self.scrollAreaRight.setMinimumWidth(302)
        self.scrollAreaRight.setMinimumHeight(302)
        
        self.scrollAreaRight.mouseMoveEvent = self.mouseMoveEventLeft
        self.scrollAreaRight.mousePressEvent = self.mousePressEventLeft
        self.scrollAreaRight.mouseReleaseEvent = self.mouseReleaseEventLeft

        self.imageLabelRight.setCursor(Qt.OpenHandCursor)

        layout = QGridLayout()
        layout.addWidget(self.table_explanation, 0, 0)
        layout.addWidget(self.scrollAreaRight, 0, 1)
        
        
        return layout
    
    def clicked_instance(self, qmodelindex):
        index = self.list_instance.currentIndex().row()
        self.current_instance = tuple(self.explainer._history.keys())[index]
        self.feature_values.clear()
        for i, value in enumerate(self.current_instance):
            self.table_instance.setItem(i, 1, QTableWidgetItem(str(value)))
            self.feature_values[self.feature_names[i]] = value

        n_to_delete = self.table_explanation.rowCount()
        for _ in range(n_to_delete):
            self.table_explanation.removeRow(0)

        for id_method, (_, method, reasons) in enumerate(self.explainer._history[self.current_instance]):
            for id_reason, reason in enumerate(reasons):
                method = method.replace("reason", "").replace("reasons", "").replace("_", " ").capitalize()
                method = " ".join(word.capitalize() for word in method.split(" "))

                numrows = self.table_explanation.rowCount() 
                self.table_explanation.insertRow(numrows)
                self.table_explanation.setItem(numrows,0, QTableWidgetItem(str(id_method)))
                self.table_explanation.setItem(numrows,1, QTableWidgetItem(str(method)))
                self.table_explanation.setItem(numrows,2, QTableWidgetItem(str(id_reason)))
                self.table_explanation.setItem(numrows,3, QTableWidgetItem(str(len(reason))))
        
        if self.image_size is not None:
            self.display_left(self.current_instance)

    
    def clicked_explanation(self, qmodelindex):
        self.index = self.table_explanation.currentIndex().row()
        self.id_method = int(self.table_explanation.item(self.index, 0).text())
        self.id_reason = int(self.table_explanation.item(self.index, 2).text())
        
        reasons = self.explainer._history[self.current_instance][self.id_method][2]
        reason = reasons[self.id_reason]
        #print("reason:", reason)
        self.display_right(self.current_instance, reason)

    def create_menu_bar(self):
        self.save_action = QAction("&Save", self)
        self.exit_action = QAction("&Exit", self)
        self.documentation_action = QAction("&Documentation", self)
        
        self.save_action.triggered.connect(self.save)
        self.exit_action.triggered.connect(self.close)
        self.documentation_action.triggered.connect(self.documentation)

        menu_bar = self.menuBar()
        file_menu = QMenu("&File", self)
        help_menu = menu_bar.addMenu("&Help")   
        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(help_menu)

        file_menu.addAction(self.save_action)
        file_menu.addAction(self.exit_action)    
        help_menu.addAction(self.documentation_action)
    
    def save(self):
        print("Save")

    def documentation(self):
        webbrowser.open_new("http://www.cril.univ-artois.fr/pyxai/documentation/")