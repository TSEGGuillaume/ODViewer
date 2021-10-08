from re import S
from PyQt5.QtCore import QDir, QObject, QPoint, QLine, QStandardPaths, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QImage, QImageReader, QPen, QPixmap, QPalette, QPainter
from PyQt5.QtWidgets import QDialog, QFileSystemModel, QHBoxLayout, QLabel, QListView, QPushButton, QSizePolicy, QRadioButton, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, QVBoxLayout, QWidget, qApp, QFileDialog

import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

import os
import time
import numpy as np

class CPredictionsPainter(QPainter):
    def __init__(self, device):
        if type(self) == CPredictionsPainter:
            raise Exception("CPredictionsPainter must be subclassed.")

        super().__init__(device)

        self.initPainterStyle()

    def initPainterStyle(self):
        self.color = QColor(0, 255, 0, 255)
        self.pen = QPen(self.color)
        self.pen.setWidth(3)
        self.setPen(self.pen)

    def drawInferences(self, inference):
        pass

class CCirclePredictionsPainter(CPredictionsPainter):
    def __init__(self, device):
        super().__init__(device)

        self.initPainterStyle()

    def initPainterStyle(self):
        self.color = QColor(0, 255, 0, 255)
        self.pen = QPen(self.color)
        self.pen.setWidth(3)
        self.setPen(self.pen)

    def drawInferences(self, inference):
        boxes   = inference._boxes
        scores  = inference._scores
        labels  = inference._labels

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # Scores are sorted so we can break
            if score < 0.5: # TODO : récupérer la valeur d'un curseur [0, 1]
                break

            b = box.astype(int)
            x = b[0]
            y = b[1]
            w = b[2] - x
            h = b[3] - y 

            self.drawEllipse(x, y, w, h)
            print("draw (x, y, rx, ry) object: ({}, {}, {}, {}))".format(x + int(w/2), y + int(h/2), int(w/2), int(h/2)))

        self.end()  

class CBboxPredictionsPainter(CPredictionsPainter):
    def __init__(self, device):
        super().__init__(device)

        self.initPainterStyle()

    def initPainterStyle(self):
        self.color = QColor(0, 255, 0, 255)
        self.pen = QPen(self.color)
        self.pen.setWidth(3)
        self.setPen(self.pen)

    def drawInferences(self, inference):
        super().drawInferences(inference)

        boxes   = inference._boxes
        scores  = inference._scores
        labels  = inference._labels

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # Scores are sorted so we can break
            if score < 0.5: # TODO : récupérer la valeur d'un curseur [0, 1]
                break

            b = box.astype(int)
            x = b[0]
            y = b[1]
            w = b[2] - x
            h = b[3] - y 

            self.drawRect(x, y, w, h)
            print("draw (x, y, w, h) object : ({}, {}, {}, {}))".format(x, y, w, h))

        self.end()

class CCrossPredictionsPainter(CPredictionsPainter):
    def __init__(self, device):
        super().__init__(device)

        self.initPainterStyle()

    def initPainterStyle(self):
        self.color = QColor(0, 255, 0, 255)
        self.pen = QPen(self.color)
        self.pen.setWidth(3)
        self.setPen(self.pen)

    def drawInferences(self, inference):
        super().drawInferences(inference)

        boxes   = inference._boxes
        scores  = inference._scores
        labels  = inference._labels

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # Scores are sorted so we can break
            if score < 0.5: # TODO : récupérer la valeur d'un curseur [0, 1]
                break
            b = box.astype(int)
            x = b[0]
            y = b[1]
            w = b[2] - x
            h = b[3] - y 
            lenLines = 5
 
            hLine = QLine(x + w/2 - lenLines, y + h/2, x + w/2 + lenLines, y + h/2)
            vLine = QLine(x + w/2, y + h/2 - lenLines, x + w/2, y + (h/2) + lenLines)

            self.drawLines(hLine, vLine)
            print("draw (x, y) object : ({}, {})".format(x + int(w/2), y + int(h/2)))

        self.end()

class CInference(QObject):
    finished = pyqtSignal()

    def __init__(self, pImgPath, pModelPath, pBackbone=None, pThreshold=None, pSavePath=None):
        super().__init__()

        self.imgPath = pImgPath
        self.modelPath = pModelPath

        ### Gestion des paramètres
        if pBackbone != None:
            assert pBackbone == "resnet50" or pBackbone == "resnet101"
            self.backbone = pBackbone
        else:
            self.backbone = 'resnet50'

        if pThreshold != None:
            assert pThreshold >=0 and pThreshold <= 1
            self.scoreThreshold = pThreshold
        else:
            self.scoreThreshold = 0.5

        if pSavePath != None:
            self.inferenceSavePath = pSavePath
        else:
            self.inferenceSavePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.jpg")
    
    @pyqtSlot()
    def inferImage(self):
        self.performTFObjectDetection()
        print("End of inference")
        self.finished.emit()

    def performTFObjectDetection(self):
        """
        @Signature   : simpleInference
        @Brief       : Effectue une inférence sur une image selon les paramètres indiqués.

        @Paramètres :
            - imgPath(str)      : chemin absolu vers le fichier image à inférer (ex : D:\<path>\<filename>.jpg).
            - modelPath(str)    : chemin absolu vers le modèle d'inférence à utiliser (ex : D:\<path>\<filename>.h5).
            - pBackbone(str)    : indique le réseau backbone à utiliser pour effectuer l'inférence. Les backbones disponibles sont [resnet50, resnet101].
            - pThreshold(float) : indique le seuil minimal de probabilité pour prendre en compte la prédiction. Ce seuil doit ∈ [0.0, 1.0].
            - pSavePath(str)    : chemin absolu de sauvegarde pour l'image résultante de l'inférence  (ex : D:\<path>\<filename>.jpg).

        @Retour :
            - objects       : liste des objets détectés par le modèle
            - scoreF1       : le score F1 de l'inférence. 
            - enlapsedTime  : le temps écoulé pour réaliser l'inférence
        """
        print("Inference begins")

        ### Gestion du matériel
        gpu = "0" # ID du GPU pour l'inférence
        setup_gpu(gpu) # set the modified tf session as backend in keras

        ### Inférence
        image = read_image_bgr(self.imgPath) # Chargement de l'image
        model = models.load_model(self.modelPath, backbone_name=self.backbone) # Chargement du model Retinanet

        imgHeight, imgWidth, nbChannels = image.shape

        resizeFactor    = 1     # Facteur d'échelle 
        imageMinSide    = min(imgHeight, imgWidth) / resizeFactor
        imageMaxSide    = max(imgHeight, imgWidth) / resizeFactor

        # Prétraitement de l'image source pour le réseau
        image = preprocess_image(image)
        image, scale = resize_image(image, imageMinSide, imageMaxSide)
        print("Facteur de redimensionnement : ", scale)

        # Traitement de l'image par le réseau
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        enlapsedTime = time.time() - start
        print("Processing time : ", enlapsedTime)

        boxes /= scale # Réduction de la taille des bbox pour correspondre au redimensionnement de l'image

        self._boxes = boxes
        self._scores = scores
        self._labels = labels

class QImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Mecanism variables
        self.workingRoot = os.path.join("D:\\", "Retinanet")
        self.scaleFactor = 1.0

        self.computingThread = QThread()

        self.setupUI()

    def setupUI(self):
        # The main widget
        self.mainWidget = QWidget()

        # Abstract
        # Utilisation de deux models QFileSystemModel différents car les filtres sur fichier ne sont pas les mêmes
        self.fileSysImages = QFileSystemModel()
        self.fileSysImages.setReadOnly(True)
        self.fileSysImages.setFilter(QDir.Filter.NoDotAndDotDot | QDir.Filter.Files)
        self.fileSysImages.setNameFilters(self.defineImageFilter())
        self.fileSysImages.setNameFilterDisables(False)

        self.fileSysModels = QFileSystemModel()
        self.fileSysModels.setReadOnly(True)
        self.fileSysModels.setFilter(QDir.Filter.NoDotAndDotDot | QDir.Filter.Files)
        self.fileSysModels.setNameFilters(["*.h5"])
        self.fileSysModels.setNameFilterDisables(False)

        self.hlayMain = QHBoxLayout(self.mainWidget)
    
        self.vlayParam = QVBoxLayout()

        self.modelsListLbl = QLabel("Models list :")

        self.modelsListView = QListView()
        self.initializeInferenceParamListView(self.modelsListView, self.fileSysModels)

        self.modelFileBrowserBtn = QPushButton("Browse...")
        self.modelFileBrowserBtn.clicked.connect(self.openModelsFiles)

        self.imagesListLbl = QLabel("Images list :")

        self.imagesListView = QListView() 
        self.initializeInferenceParamListView(self.imagesListView, self.fileSysImages)
        self.imagesListView.selectionModel().currentChanged.connect(self.onImageClicked)
        self.imagesListView.doubleClicked.connect(self.onImageDoubleClicked) #TODO infer image

        self.imageFileBrowserBtn = QPushButton("Browse...")
        self.imageFileBrowserBtn.clicked.connect(self.openImagesFiles)

        self.vlayParam.addWidget(self.modelsListLbl)
        self.vlayParam.addWidget(self.modelsListView)
        self.vlayParam.addWidget(self.modelFileBrowserBtn)
        self.vlayParam.addWidget(self.imagesListLbl)
        self.vlayParam.addWidget(self.imagesListView)
        self.vlayParam.addWidget(self.imageFileBrowserBtn)

        # Panel Image
        self.vlayImage = QVBoxLayout()

        self.hlayParamVisualisation = QHBoxLayout()

        self.radioVisuBbox = QRadioButton("Bouding Box")
        self.radioVisuCircle = QRadioButton("Circle")
        self.radioVisuCross = QRadioButton("Cross")
        self.radioVisuNone = QRadioButton("None")

        self.radioVisuBbox.clicked.connect(self.onVisualisationStyleChanged)
        self.radioVisuCircle.clicked.connect(self.onVisualisationStyleChanged)
        self.radioVisuCross.clicked.connect(self.onVisualisationStyleChanged)
        self.radioVisuNone.clicked.connect(self.onVisualisationStyleChanged)

        # The cross visualisation is default
        self.radioVisuBbox.setChecked(True)

        self.radioVisuBbox.setEnabled(False)
        self.radioVisuCircle.setEnabled(False)
        self.radioVisuCross.setEnabled(False)
        self.radioVisuNone.setEnabled(False)

        self.hlayParamVisualisation.addWidget(self.radioVisuBbox)
        self.hlayParamVisualisation.addWidget(self.radioVisuCircle)
        self.hlayParamVisualisation.addWidget(self.radioVisuCross)
        self.hlayParamVisualisation.addWidget(self.radioVisuNone)

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setUpdatesEnabled(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Base)
        self.scrollArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(True)

        self.vlayImage.addLayout(self.hlayParamVisualisation)
        self.vlayImage.addWidget(self.scrollArea)

        self.hlayMain.addLayout(self.vlayParam)
        self.hlayMain.addLayout(self.vlayImage)

        self.setCentralWidget(self.mainWidget)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Inference Viewer")
        self.resize(1064, 536)

    def defineImageFilter(self):
        imageSupportedMime = ["*.jpg"] # Pas très propre mais solution pour le moment. La fonction QImageReader.supportedMimeTypes() retourne "jpeg" mais pas "jpg".
        for elem in QImageReader.supportedMimeTypes():
            imageSupportedMime.append("*." + elem.data().decode("utf-8").split("/")[-1])

        return imageSupportedMime

    def initializeImageFileDialog(self, acceptMode):
        dialog = QFileDialog()

        firstDialog = True

        if firstDialog == True:
            firstDialog = False
            picturesLocations = self.workingRoot #QStandardPaths.standardLocations(QStandardPaths.StandardLocation.PicturesLocation)
            dialog.setDirectory(picturesLocations if picturesLocations else QDir.currentPath)

        dialog.setAcceptMode(acceptMode)

        return dialog

    def initializeInferenceParamListView(self, pListView, pFileSystem):
        pListView.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Expanding)
        pListView.setMaximumWidth(300)

        pListView.setDisabled(True)
        pListView.setSelectionMode(QListView.SelectionMode.SingleSelection)

        pListView.setModel(pFileSystem)

    def openModelsFiles(self):
        dlg = self.initializeImageFileDialog(QFileDialog.AcceptMode.AcceptOpen)

        dirName = dlg.getExistingDirectory(self, "Select models directory")

        if dirName:
            self.modelsListView.setDisabled(False)

            self.fileSysModels.setRootPath(dirName)
            self.modelsListView.setRootIndex(self.fileSysModels.index(dirName))

    def openImagesFiles(self):
        dlg = self.initializeImageFileDialog(QFileDialog.AcceptMode.AcceptOpen)

        dirName = dlg.getExistingDirectory(self, "Select images directory")

        if dirName:
            self.imagesListView.setDisabled(False)

            self.fileSysImages.setRootPath(dirName)
            self.imagesListView.setRootIndex(self.fileSysImages.index(dirName))

    def openImagesDirectory(self):
        dlg = self.initializeImageFileDialog(QFileDialog.AcceptMode.AcceptOpen)

        dirName = dlg.getExistingDirectory(self, "Select images directory")

        self.imagesFiles = []
        if dirName:
            imageSupportedMime = []


            for imageFile in [f for f in os.listdir(dirName) if f.endswith(tuple(imageSupportedMime))]:
                self.imagesFiles.append(os.path.join(dirName, imageFile))


        self.showImage()

    def showImage(self, imagePath):
        image = QImage(imagePath)
        if image.isNull():
            QMessageBox.information(self, "Image Viewer", "Cannot load %s." % imagePath)
            return

        self.originalImage = QPixmap.fromImage(image)
        self.imageLabel.setPixmap(self.originalImage)

        previousFactor =  self.scaleFactor
        self.scaleFactor = 1.0

        self.scrollArea.setVisible(True)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()

        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()
            self.scaleImage(previousFactor) # Keep the scaling after loading image

    def onImageClicked(self):
        selectedElemIndex = self.imagesListView.currentIndex()
        imageFilePath = QFileSystemModel.filePath(self.fileSysImages, selectedElemIndex)

        self.showImage(imageFilePath)

    def onImageDoubleClicked(self):
        selectedElemIndex = self.modelsListView.currentIndex()
        modelFilePath = QFileSystemModel.filePath(self.fileSysModels, selectedElemIndex)

        selectedElemIndex = self.imagesListView.currentIndex()
        imageFilePath = QFileSystemModel.filePath(self.fileSysImages, selectedElemIndex)

        if modelFilePath != '':
            self.imagesListView.setDisabled(True)

            self.openedImagesInferences = CInference(imageFilePath, modelFilePath)

            self.openedImagesInferences.moveToThread(self.computingThread)
            self.computingThread.started.connect(self.openedImagesInferences.inferImage)
            self.openedImagesInferences.finished.connect(self.onInferenceFinished)

            self.computingThread.start()
        else:
            alertDlg = QMessageBox(self)

            alertDlg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            alertDlg.setIcon(QMessageBox.Icon.Information)
            alertDlg.setWindowTitle("Processing error")
            alertDlg.setText("No model loaded")
            alertDlg.setInformativeText("Please, select a model before infer image.")
            alertDlg.open()

    def onInferenceFinished(self):
        self.computingThread.quit()

        self.radioVisuBbox.setEnabled(True)
        self.radioVisuCircle.setEnabled(True)
        self.radioVisuCross.setEnabled(True)
        self.radioVisuNone.setEnabled(True)

        self.initInferencesPainter()
        self.drawInferences()

        self.imagesListView.setDisabled(False)

    def drawInferences(self):
        self.myPainter.drawInferences(self.openedImagesInferences)

        self.imageLabel.setPixmap(self.imageLabel.pixmap().copy())
        self.imageLabel.repaint()

    def initInferencesPainter(self):
        self.imageLabel.setPixmap(self.originalImage)

        if self.radioVisuBbox.isChecked():
            self.myPainter = CBboxPredictionsPainter(self.imageLabel.pixmap())
        elif self.radioVisuCircle.isChecked():
            self.myPainter = CCirclePredictionsPainter(self.imageLabel.pixmap())
        elif self.radioVisuCross.isChecked():
            self.myPainter = CCrossPredictionsPainter(self.imageLabel.pixmap())
        elif self.radioVisuNone.isChecked():
            pass

    def onVisualisationStyleChanged(self):
        self.initInferencesPainter()
        
        self.drawInferences()

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.scaleFactor = 1.0
        self.updateActions()
        self.imageLabel.adjustSize()

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "About Image Viewer",
                          "<p>The <b>Image Viewer</b> example shows how to combine "
                          "QLabel and QScrollArea to display an image. QLabel is "
                          "typically used for displaying text, but it can also display "
                          "an image. QScrollArea provides a scrolling view around "
                          "another widget. If the child widget exceeds the size of the "
                          "frame, QScrollArea automatically provides scroll bars.</p>"
                          "<p>The example demonstrates how QLabel's ability to scale "
                          "its contents (QLabel.scaledContents), and QScrollArea's "
                          "ability to automatically resize its contents "
                          "(QScrollArea.widgetResizable), can be used to implement "
                          "zooming and scaling features.</p>"
                          "<p>In addition the example shows how to use QPainter to "
                          "print an image.</p>")

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.openImagesDirectory)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)
        self.aboutAct = QAction("&About", self, triggered=self.about)
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)

        # TODO : 
        self.openAct.setDisabled(True)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    imageViewer = QImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
    # TODO QScrollArea support mouse
    # base on https://github.com/baoboa/pyqt5/blob/master/examples/widgets/imageviewer.py
    #
    # if you need Two Image Synchronous Scrolling in the window by PyQt5 and Python 3
    # please visit https://gist.github.com/acbetter/e7d0c600fdc0865f4b0ee05a17b858f2
