from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import sys
import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
from utils import SimpleFPS, draw_fps
import argparse
import time

class VideoThreadPiCam(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.grab_frame = True

    def run(self):
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}, raw={"size": (640, 480)})
        picam2.configure(camera_config)
        picam2.start()

        while True:
            if self.grab_frame:
                frame = picam2.capture_array()
                self.change_pixmap_signal.emit(frame)
                self.grab_frame = False
            else:
                time.sleep(0.0001)

class App(QWidget):
    def __init__(self, camera_test_only, model_path):
        super().__init__()
        self.camera_test_only = camera_test_only
        self.setWindowTitle("Qt UI")
        self.disply_width = 640
        self.display_height = 480
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)
        self.interpreter = self.load_model(model_path)
        self.thread = VideoThreadPiCam()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def load_model(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter

    def predict(self, frame):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
        frame = np.expand_dims(frame, axis=0).astype(np.uint8)
        frame = ((frame - 127.5) / 127.5).astype(np.uint8)  # Correction de normalisation
        self.interpreter.set_tensor(input_details[0]['index'], frame)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])[0]
        class_id = np.argmax(output_data)
        confidence = np.max(output_data)
        return class_id, confidence  # Renvoie aussi la confiance

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        class_id, confidence = self.predict(cv_img)
        label = f"{class_id} ({confidence}%)"  # Affichage confidence (A CORRIGER)
        display_img = cv2.putText(cv_img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        display_img = cv2.rectangle(display_img, (50, 60), (250, 200), (255, 0, 0), 2)  # Dessiner un rectangle exemple
        qt_img = self.convert_cv_qt(display_img)
        self.image_label.setPixmap(qt_img)
        self.thread.grab_frame = True

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../models/orchidees/model.tflite")
    parser.add_argument('--camera_test', action=argparse.BooleanOptionalAction)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.debug or args.camera_test:
        app = QApplication(sys.argv)
        a = App(camera_test_only=args.camera_test, model_path=args.model)
        a.show()
        sys.exit(app.exec_())
    else:
        print("No UI mode not supported with Teachable Machine model.")
