#!/usr/bin/python
# -*- coding: UTF-8 -*-

from PyQt5.QtWidgets import *

app = QApplication([])
app.setStyle('Fusion')
window = QWidget()
layout = QVBoxLayout()

button = QPushButton('track_object')
def on_button_clicked():
    alert = QMessageBox()
    alert.setText('You clicked the button!')
    alert.exec_()

button.clicked.connect(on_button_clicked)
button.show()



app.exec_()