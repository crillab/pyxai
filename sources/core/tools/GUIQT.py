from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSplitter, QApplication, QAbstractItemView, QHeaderView, QBoxLayout, QFrame, QMessageBox, QFileDialog, QLabel, QSizePolicy, QScrollArea,  QStyleFactory, QMainWindow, QTableWidgetItem, QHBoxLayout, QMenu, QGroupBox, QListWidget, QWidget, QVBoxLayout, QGridLayout, QTableWidget

from PyQt6.QtGui import QImage, QAction, QPixmap, QPalette, QPainter, QColor
from PyQt6.QtPrintSupport import QPrintDialog, QPrinter

import sys
import webbrowser
import numpy
import dill
from pyxai.sources.core.tools.vizualisation import PyPlotImageGenerator, PyPlotDiagramGenerator

class EmptyExplainer():pass

class GraphicalInterface(QMainWindow):
    """Main Window."""
    def __init__(self, explainer, image=None, feature_names=None, time_series=None):
        """Initializer."""
        app = QApplication(sys.argv)
        app.setPalette(app.style().standardPalette())
        if explainer is not None:
            pass
        self.explainer = explainer
        self.image = image
        self.time_series = time_series
        if feature_names is not None:
            self.feature_names = feature_names
        elif explainer is not None:
            self.feature_names = explainer.get_feature_names()
        else:
            self.feature_names = []

        self.feature_values = dict()
        if self.image is not None:
            self.pyplot_image_generator = PyPlotImageGenerator(image)

        self.pyplot_diagram_generator = PyPlotDiagramGenerator(time_series)
        super().__init__(None)
        #main_layout = QGridLayout()
        self.imageLabelLeft = None
        self.setWindowTitle("PyXAI")
        self.resize(400, 200)
        
        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.create_menu_bar()
        splitter = QSplitter()
        
        instance_layout = self.create_instance_group()
        instance_widget = QWidget()
        instance_widget.setLayout(instance_layout)
        splitter.addWidget(instance_widget)
        
        #main_layout.addLayout(instance_layout, 0, 0)

        #self.separator = QFrame()
        #self.separator.setFrameShape(QFrame.VLine)
        #self.separator.setFrameShadow(QFrame.Raised)
        #layout_separator = QHBoxLayout()
        #layout_separator.addWidget(self.separator)
        

        #main_layout.addLayout(layout_separator, 0, 1)

        self.explanation_layout = self.create_explanation_group()
        self.explanation_widget = QWidget()
        self.explanation_widget.setLayout(self.explanation_layout)
        splitter.addWidget(self.explanation_widget)

        #main_layout.addLayout(explanation_layout, 0, 2)

        #widget = QWidget()
        #widget.setLayout(main_layout)
        self.setCentralWidget(splitter)
        
        self.show()
        sys.exit(app.exec())

    def create_instance_group(self):

        self.instance_group = QGroupBox("Instances")
        self.instance_group.repaint()
        self.list_instance = QListWidget()
        if self.explainer is not None:
            instances = tuple("Instance "+str(i) for i in range(1, len(self.explainer._history.keys())+1))
        else:
            instances = tuple()
        #List of instances
        self.list_instance.addItems(instances)
        self.list_instance.clicked.connect(self.clicked_instance)
        self.list_instance.setMaximumWidth(150)
        self.list_instance.setMinimumWidth(150)
        
        #Table of the selected instance

        self.table_instance = QTableWidget(len(self.feature_names)-1 if len(self.feature_names) != 0 else 0, 2)
        
        self.table_instance.verticalHeader().setVisible(False)
        self.table_instance.setHorizontalHeaderItem(0, QTableWidgetItem("Name"))
        self.table_instance.setHorizontalHeaderItem(1, QTableWidgetItem("Value"))
        self.table_instance.setMinimumWidth(220)
        
        for i, name in enumerate(self.feature_names[:-1]):
            self.table_instance.setItem(i, 0, QTableWidgetItem(str(name)))
        self.table_prediction = QTableWidget(1, 2)
        self.table_prediction.verticalHeader().setVisible(False)
        self.table_prediction.setHorizontalHeaderItem(0, QTableWidgetItem("Name"))
        self.table_prediction.setHorizontalHeaderItem(1, QTableWidgetItem("Value"))
        self.table_prediction.setMinimumWidth(220)
        self.table_prediction.setMaximumHeight(50)
        self.table_prediction.setMinimumHeight(50)
        if len(self.feature_names) != 0:
            self.table_prediction.setItem(0, 0, QTableWidgetItem(str(self.feature_names[-1])))

        if self.image is not None:
            #Image of the selected instance
            self.imageLabelLeft = QLabel()
            #self.imageLabelLeft.setBackgroundRole(QPalette.ColorRole.Base)
            self.imageLabelLeft.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            #self.imageLabelLeft.setScaledContents(True)

            self.imageLabelLeft.setMinimumWidth(300)
            self.imageLabelLeft.setMinimumHeight(300)
            
            self.scrollAreaLeft = QScrollArea()
            #self.scrollAreaLeft.setBackgroundRole(QPalette.ColorRole.Dark)
            self.scrollAreaLeft.setWidget(self.imageLabelLeft)
            self.scrollAreaLeft.setVisible(True)

            self.scrollAreaLeft.setMinimumWidth(302)
            self.scrollAreaLeft.setMinimumHeight(302)

            self.scrollAreaLeft.mouseMoveEvent = self.mouseMoveEventLeft
            self.scrollAreaLeft.mousePressEvent = self.mousePressEventLeft
            self.scrollAreaLeft.mouseReleaseEvent = self.mouseReleaseEventLeft

            self.imageLabelLeft.setCursor(Qt.CursorShape.OpenHandCursor)
            
        layout = QGridLayout()
        layout2 = QVBoxLayout()
        
        layout.addWidget(self.list_instance, 0, 0)
        layout.addLayout(layout2, 0, 1)
        label_prediction = QLabel(self)
        #label_prediction.setFrameStyle(QFrame.Panel | QFrame.Raised)
        label_prediction.setText("Prediction:")
        label_prediction.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        label_instance = QLabel(self)
        label_instance.setText("Instance:")
        label_instance.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        
        layout2.addWidget(label_prediction)
        layout2.addWidget(self.table_prediction)
        layout2.addWidget(label_instance)
        layout2.addWidget(self.table_instance)
        
        if self.image is not None:
            layout.addWidget(self.scrollAreaLeft, 0, 2)
        self.left_layout = layout
        return layout
    
    def to_rgb(self, value):
        return QColor().fromRgb(int(value)) 
    

    def display_left(self, instance):
        image = self.pyplot_image_generator.generate_instance(instance)
        self.scaleFactor = 1.0
        size = self.imageLabelLeft.size()
        self.imageLabelLeft.setPixmap(QPixmap.fromImage(image).scaled(size))
        
    def display_right(self, instance, reason=None):
        if self.image is not None:
            image = self.pyplot_image_generator.generate_explanation(instance, reason)
            size = self.imageLabelRight.size()
            self.imageLabelRight.setPixmap(QPixmap.fromImage(image).scaled(size))
            self.imageLabelRight.adjustSize()
        else:
            image = self.pyplot_diagram_generator.generate_explanation(self.feature_values, instance, reason)
            qpixmap = QPixmap.fromImage(image)
            self.scrollAreaRight.setMinimumWidth(qpixmap.width()+16)
            #self.imageLabelRight.setMinimumWidth(qpixmap.width())
            #self.imageLabelRight.setMinimumHeight(qpixmap.width())
            self.imageLabelRight.setPixmap(qpixmap)
            self.imageLabelRight.adjustSize()
            
            
        

    def mousePressEventLeft(self, event):
        if self.imageLabelLeft is not None:
            self.pressed = True
            self.imageLabelLeft.setCursor(Qt.ClosedHandCursor)
            self.initialPosX = self.scrollAreaLeft.horizontalScrollBar().value() + event.pos().x()
            self.initialPosY = self.scrollAreaLeft.verticalScrollBar().value() + event.pos().y()

    def mouseReleaseEventLeft(self, event):
        if self.imageLabelLeft is not None:
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
        
        self.table_explanation = QTableWidget(0, 5)
        self.table_explanation.verticalHeader().setVisible(False)
        self.table_explanation.setHorizontalHeaderItem(0, QTableWidgetItem("Id Method"))
        self.table_explanation.setHorizontalHeaderItem(1, QTableWidgetItem("Name"))
        self.table_explanation.setHorizontalHeaderItem(2, QTableWidgetItem("Id Reason"))
        self.table_explanation.setHorizontalHeaderItem(3, QTableWidgetItem("#Binaries"))
        self.table_explanation.setHorizontalHeaderItem(4, QTableWidgetItem("#Features"))
        self.table_explanation.clicked.connect(self.clicked_explanation)

        header = self.table_explanation.horizontalHeader()       
        self.table_explanation.setColumnWidth(0, 80)
        self.table_explanation.setColumnWidth(1, 200)
        self.table_explanation.setColumnWidth(2, 80)
        self.table_explanation.setColumnWidth(3, 80)
        self.table_explanation.setColumnWidth(4, 80)
        self.table_explanation.setMinimumWidth(80+200+80+80+80+2)
        self.table_explanation.setMaximumWidth(80+200+80+80+80+2)
        self.table_explanation.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        #Image of the explanation
        self.imageLabelRight = QLabel()
        #self.imageLabelRight.setBackgroundRole(QPalette.ColorRole.Base)
        self.imageLabelRight.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        #self.imageLabelRight.setScaledContents(True)

        self.imageLabelRight.setMinimumWidth(300)
        self.imageLabelRight.setMinimumHeight(300)
        
        self.scrollAreaRight = QScrollArea()
        #self.scrollAreaRight.setBackgroundRole(QPalette.ColorRole.Dark)
        self.scrollAreaRight.setWidget(self.imageLabelRight)
        self.scrollAreaRight.setVisible(True)

        self.scrollAreaRight.setMinimumWidth(302)
        self.scrollAreaRight.setMinimumHeight(302)
        
        self.scrollAreaRight.mouseMoveEvent = self.mouseMoveEventLeft
        self.scrollAreaRight.mousePressEvent = self.mousePressEventLeft
        self.scrollAreaRight.mouseReleaseEvent = self.mouseReleaseEventLeft

        self.imageLabelRight.setCursor(Qt.CursorShape.OpenHandCursor)

        layout = QGridLayout()
        layout.addWidget(self.table_explanation, 0, 0)
        layout.addWidget(self.scrollAreaRight, 0, 1)
        
        
        return layout
    
    def clicked_instance(self, qmodelindex):
        index = self.list_instance.currentIndex().row()
        self.current_instance = tuple(self.explainer._history.keys())[index]
        self.feature_values.clear()
        for i, value in enumerate(self.current_instance[0]):
            self.table_instance.setItem(i, 1, QTableWidgetItem(str(value)))
            self.feature_values[self.feature_names[i]] = value

        self.table_prediction.setItem(0, 1, QTableWidgetItem(str(self.current_instance[1])))

        n_to_delete = self.table_explanation.rowCount()
        for _ in range(n_to_delete):
            self.table_explanation.removeRow(0)
        self.imageLabelRight.clear()

        for id_method, (_, method, reasons) in enumerate(self.explainer._history[self.current_instance]):
            for id_reason, reason in enumerate(reasons):
                method = method.replace("reason", "").replace("reasons", "").replace("_", " ").capitalize()
                method = " ".join(word.capitalize() for word in method.split(" "))
                n_binaries = sum(len(reason[key]) for key in reason.keys())
                numrows = self.table_explanation.rowCount() 
                self.table_explanation.insertRow(numrows)
                self.table_explanation.setItem(numrows,0, QTableWidgetItem(str(id_method)))
                self.table_explanation.setItem(numrows,1, QTableWidgetItem(str(method)))
                self.table_explanation.setItem(numrows,2, QTableWidgetItem(str(id_reason)))
                self.table_explanation.setItem(numrows,3, QTableWidgetItem(str(n_binaries)))
                self.table_explanation.setItem(numrows,4, QTableWidgetItem(str(len(reason))))
        if self.image is not None:
            self.display_left(self.current_instance[0])
        self.adjustSize()
        self.update()
        self.show()
        
    
    def clicked_explanation(self, qmodelindex):
        self.index = self.table_explanation.currentIndex().row()
        self.id_method = int(self.table_explanation.item(self.index, 0).text())
        self.id_reason = int(self.table_explanation.item(self.index, 2).text())
        
        reasons = self.explainer._history[self.current_instance][self.id_method][2]
        reason = reasons[self.id_reason]
        self.display_right(self.current_instance[0], reason)
        
    def create_menu_bar(self):
        self.save_action = QAction("Save Explainer", self)
        self.load_action = QAction("Load Explainer", self)
        self.save_image_instance_action = QAction("Save Image Instance ", self)
        self.save_image_explanation_action = QAction("Save Image Explanation ", self)
        
        self.exit_action = QAction("&Exit", self)
        self.documentation_action = QAction("&Documentation", self)
        
        self.save_action.triggered.connect(self.save)
        self.load_action.triggered.connect(self.load)
        self.save_image_instance_action.triggered.connect(self.save_image_instance)
        self.save_image_explanation_action.triggered.connect(self.save_image_explanation)

        self.exit_action.triggered.connect(self.close)
        self.documentation_action.triggered.connect(self.documentation)

        menu_bar = self.menuBar()
        file_menu = QMenu("&File", self)
        help_menu = menu_bar.addMenu("&Help")   
        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(help_menu)

        file_menu.addAction(self.load_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_image_instance_action)
        file_menu.addAction(self.save_image_explanation_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)    
        help_menu.addAction(self.documentation_action)
    
    def save_image_instance(self):
        if self.imageLabelLeft is None or self.imageLabelLeft.pixmap() is None:
            msgBox = QMessageBox()
            msgBox.setText("No image displayed in the application at the moment for an instance.")
            msgBox.exec()
        else:
            pixmap = self.imageLabelLeft.pixmap()
            fileDialog = QFileDialog()
            fileDialog.setDefaultSuffix("png")
            name, _ = fileDialog.getSaveFileName(None, 'Save Image Instance', filter="Portable Network Graphics (*.png)")
            if not name.endswith(".png"): name = name + ".png"
            pixmap.save(name)

    def save_image_explanation(self):
        if self.imageLabelRight is None or self.imageLabelRight.pixmap() is None:
            msgBox = QMessageBox()
            msgBox.setText("No image displayed in the application at the moment for an explanation.")
            msgBox.exec()
        else:
            pixmap = self.imageLabelRight.pixmap()
            fileDialog = QFileDialog()
            fileDialog.setDefaultSuffix("png")
            name, _ = fileDialog.getSaveFileName(None, 'Save Image Instance', filter="Portable Network Graphics (*.png)")
            if not name.endswith(".png"): name = name + ".png"
            pixmap.save(name)

    def save(self):
        fileDialog = QFileDialog()
        fileDialog.setDefaultSuffix("explainer")
        name, _ = fileDialog.getSaveFileName(None, 'Save File', filter="Dill Explainer Object (*.explainer)")
        
        if not name.endswith(".explainer"): name = name + ".explainer"

        with open(name,'wb') as io:
            dill.dump([self.image, self.feature_names, self.explainer._history, self.time_series],io)


    def load(self):
        fileDialog = QFileDialog()
        fileDialog.setDefaultSuffix("explainer")
        name, _ = fileDialog.getOpenFileName(None, 'Load File', filter="Dill Explainer Object (*.explainer)")
        if name == "":
            return
        with open(name,'rb') as io:
            data=dill.load(io)
        
        self.image = data[0]
        self.feature_names = data[1]
        self.explainer = EmptyExplainer()
        self.explainer._history = data[2]
        self.time_series = data[3]
        
        self.list_instance.clear()    
        instances = tuple("Instance "+str(i) for i in range(1, len(self.explainer._history.keys())+1))
        self.list_instance.addItems(instances)
        self.table_instance.setRowCount(len(self.feature_names)-1)

        for i, name in enumerate(self.feature_names[:-1]):
            self.table_instance.setItem(i, 0, QTableWidgetItem(str(name)))
            self.table_instance.setItem(i, 1, QTableWidgetItem(str("")))
            
        self.table_prediction.setItem(0, 0, QTableWidgetItem(str(self.feature_names[-1])))
        self.table_prediction.setItem(0, 1, QTableWidgetItem(str("")))
        
        self.pyplot_diagram_generator = PyPlotDiagramGenerator(self.time_series)
        if self.image is not None:
            self.pyplot_image_generator = PyPlotImageGenerator(self.image)

            #Image of the selected instance
            if self.imageLabelLeft is None:
                self.imageLabelLeft = QLabel()
                #self.imageLabelLeft.setBackgroundRole(QPalette.ColorRole.Base)
                self.imageLabelLeft.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
                #self.imageLabelLeft.setScaledContents(True)

                self.imageLabelLeft.setMinimumWidth(300)
                self.imageLabelLeft.setMinimumHeight(300)
                
                self.scrollAreaLeft = QScrollArea()
                #self.scrollAreaLeft.setBackgroundRole(QPalette.ColorRole.Dark)
                self.scrollAreaLeft.setWidget(self.imageLabelLeft)
                self.scrollAreaLeft.setVisible(True)

                self.scrollAreaLeft.setMinimumWidth(302)
                self.scrollAreaLeft.setMinimumHeight(302)

                self.scrollAreaLeft.mouseMoveEvent = self.mouseMoveEventLeft
                self.scrollAreaLeft.mousePressEvent = self.mousePressEventLeft
                self.scrollAreaLeft.mouseReleaseEvent = self.mouseReleaseEventLeft

                self.imageLabelLeft.setCursor(Qt.CursorShape.OpenHandCursor)
                self.left_layout.addWidget(self.scrollAreaLeft, 0, 2) 
        else:
            if self.imageLabelLeft is not None:
                self.left_layout.removeWidget(self.scrollAreaLeft)
                self.scrollAreaLeft.setVisible(False)
                self.scrollAreaLeft.close()
                self.imageLabelLeft.clear()
                self.imageLabelLeft.close()
                self.imageLabelLeft = None

                
        n_to_delete = self.table_explanation.rowCount()
        for _ in range(n_to_delete):
            self.table_explanation.removeRow(0)
        self.imageLabelRight.clear()
        self.imageLabelRight.setMinimumWidth(300)
        self.imageLabelRight.setMinimumHeight(300)
        self.imageLabelRight.resize(300, 300)
        self.scrollAreaRight.setMinimumWidth(302)
        self.scrollAreaRight.setMinimumHeight(302)

    def documentation(self):
        webbrowser.open_new("http://www.cril.univ-artois.fr/pyxai/documentation/")