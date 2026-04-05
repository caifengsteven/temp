from PyQt5.QtWidgets import *
from PyQt5 import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import random


class CanvasWidget(QWidget):
    def __init__(self, parent=None):
        super(CanvasWidget, self).__init__(parent)
        self.setMinimumSize(800, 600)
        self.patternSize = 10
        self.pattern = [random.random() for i in range(self.patternSize)]
        self.selectPatterNodeId = -1
        self.tmpValue = -1

    def newPattern(self):
        self.pattern = [random.random() for i in range(self.patternSize)]
        self.selectPatterNodeId = -1
        self.tmpValue = -1
        self.update()

    def getPattern(self):
        return self.pattern

    def resizeEvent(self, QResizeEvent):
        self.padding = min(self.width(), self.height()) * 0.1
        self.canvasHeight = self.height() - 2 * self.padding
        self.canvasWidth = self.width() - 2 * self.padding
        self.step = self.canvasWidth / self.patternSize

    def mouseMoveEvent(self, QMouseEvent):
        if self.selectPatterNodeId == -1:
            return
        mouseY = QMouseEvent.y()
        self.tmpValue = (self.padding + self.canvasHeight - mouseY) / self.canvasHeight
        self.tmpValue = min(self.tmpValue, 1)
        self.tmpValue = max(self.tmpValue, 0)
        self.pattern[self.selectPatterNodeId] = self.tmpValue
        self.update()

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() != Qt.LeftButton:
            return
        mouseX = QMouseEvent.x()
        mouseY = QMouseEvent.y()
        for id, value in enumerate(self.pattern):
            x, y =self.__pattern_node_pos(id, value)
            if self.__distance2(mouseX, mouseY, x, y) < 250:
                self.selectPatterNodeId = id
                break
        self.update()

    def mouseReleaseEvent(self, QMouseEvent):
        self.selectPatterNodeId = -1
        self.tmpValue = -1
        self.update()

    def paintEvent(self, QPaintEvent):
        painter = QPainter(self)
        self.drawCoord(painter)
        self.drawPattern(painter)

    def drawPattern(self, painter):
        painter.setBrush(QBrush(QColor(190, 75, 72)))
        painter.setPen(QPen(QColor(190, 75, 72)))
        points = []
        for id, value in enumerate(self.pattern):
            x, y = self.__pattern_node_pos(id, value)
            pointRadius = 5
            if self.selectPatterNodeId == id:
                pointRadius *= 1.5
            painter.drawEllipse(x - pointRadius, y - pointRadius, 2 * pointRadius, 2 * pointRadius)
            points.append(QPointF(x, y))
        pen = QPen(QColor(190, 75, 72))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setFont(QFont("Helvetica", 30, QFont.Bold))
        painter.drawPolyline(QPolygonF(points))
        if self.tmpValue >= 0:
            painter.drawText(self.padding, self.padding / 2, "Id {}, value {:.2f}".format(self.selectPatterNodeId + 1, self.tmpValue))


    def drawCoord(self, painter):
        painter.setPen(QPen(QColor(50, 50, 50)))
        painter.drawLine(self.padding, self.padding, self.padding + self.canvasWidth, self.padding)
        painter.drawLine(self.padding, self.padding + self.canvasHeight,
                         self.padding + self.canvasWidth, self.padding + self.canvasHeight)
        painter.drawLine(self.padding, self.padding, self.padding, self.padding + self.canvasHeight)
        painter.drawLine(self.padding, self.padding + self.canvasHeight / 2,
                         self.padding + self.canvasWidth, self.padding + self.canvasHeight / 2)

        painter.drawText(self.padding - 20, self.padding + 5, "1.0")
        painter.drawText(self.padding - 20, self.padding + self.canvasHeight / 2 + 5, '0.5')
        painter.drawText(self.padding - 20, self.padding + self.canvasHeight + 5, '0.0')

        markHeight = 10
        for i in range(self.patternSize):
            painter.drawLine(self.padding + i * self.step, self.padding + self.canvasHeight,
                             self.padding + i * self.step, self.padding + self.canvasHeight + markHeight)
            painter.drawText(self.padding + i * self.step + self.step / 2, self.padding + self.canvasHeight + 2 * markHeight, str(i + 1))

    def __pattern_node_pos(self, id, value):
        x = self.padding + id * self.step + self.step / 2
        y = self.padding + self.canvasHeight - value * self.canvasHeight
        return x, y

    def __distance2(self, x0, y0, x1, y1):
        return (x0 - x1)**2 + (y0 - y1)**2


class Mainwindow(QWidget):
    def __init__(self, parent=None):
        super(Mainwindow, self).__init__(parent)
        layout_control = QHBoxLayout()
        self.btnNewPattern = QPushButton("New Pattern", self)
        self.btnSavePattern = QPushButton("Save Pattern", self)
        self.btnNewPattern.setStyleSheet("QPushButton{ font-family:'Microsoft YaHei';font-size:20px;}")
        self.btnSavePattern.setStyleSheet("QPushButton{ font-family:'Microsoft YaHei';font-size:20px;}")
        layout_control.addWidget(self.btnNewPattern)
        layout_control.addWidget(self.btnSavePattern)

        layout_whole = QVBoxLayout()
        self.canvas = CanvasWidget(self)
        layout_whole.addLayout(layout_control)
        layout_whole.addWidget(self.canvas)
        self.setLayout(layout_whole)

        self.btnNewPattern.clicked.connect(self.onNewPatternClicked)
        self.btnSavePattern.clicked.connect(self.onSavePatternClicked)

    def onNewPatternClicked(self):
        self.canvas.newPattern()

    def onSavePatternClicked(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Pattern", "train_data/Pattern",
                                                "Text files (*.txt);;")
        if len(file_path) == 0:
            return
        with open(file_path, 'w') as f:
            pattern = self.canvas.getPattern()
            for value in pattern:
                f.write('{:.4f}\n'.format(value))



def main():
    app = QApplication(sys.argv)
    mainWindow = Mainwindow()
    mainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()



