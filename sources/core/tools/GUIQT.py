from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTableWidgetItem, QHBoxLayout, QMenu, QAction, QGroupBox, QListWidget, QWidget, QVBoxLayout, QGridLayout, QTableWidget

import sys
import webbrowser

class GraphicalInterface(QMainWindow):
    """Main Window."""
    def __init__(self, explainer, image=None):
        """Initializer."""
        app = QApplication(sys.argv)
        
        self.explainer = explainer
        self.image = image
        self.feature_names = explainer.get_feature_names()
        if self.image is not None:
            if not isinstance(self.image, tuple) or len(self.image) != 2:
                raise ValueError("The 'image' parameter must be a tuple of size 2 representing the number of pixels (x_axis, y_axis).") 

        super().__init__(None)
        print("here1")
        main_layout = QGridLayout()

        self.setWindowTitle("Python Menus & Toolbars")
        self.resize(400, 200)
        
        self.create_menu_bar()
        instance_layout = self.create_instance_group()
        main_layout.addLayout(instance_layout, 0, 0)
        
        
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        self.show()
        sys.exit(app.exec_())

    def create_instance_group(self):
        self.instance_group = QGroupBox()
        self.list_instance = QListWidget()
        instances = tuple("Instance "+str(i) for i in range(1, len(self.explainer._history.keys())+1))
        
        self.list_instance.addItems(instances)
        self.list_instance.clicked.connect(self.clicked_instance)
        self.list_instance.setMaximumWidth(100)

        self.table_instance = QTableWidget(len(self.feature_names), 2)
        self.table_instance.setHorizontalHeaderItem(0, QTableWidgetItem("Name"))
        self.table_instance.setHorizontalHeaderItem(1, QTableWidgetItem("Value"))
        for i, name in enumerate(self.feature_names):
            self.table_instance.setItem(i, 0, QTableWidgetItem(str(name)))
        
        layout = QHBoxLayout()
        layout.addWidget(self.list_instance)
        layout.addWidget(self.table_instance)
        
        layout.addStretch(1)
        return layout
    
    def clicked_instance(self, qmodelindex):
        index = self.list_instance.currentIndex().row()
        print(index)
        instance = tuple(self.explainer._history.keys())[index]
        for i, value in enumerate(instance):
            self.table_instance.setItem(i, 1, QTableWidgetItem(str(value)))
        

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