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
        print(f"DEBUG: input_shape -> {input_shape}")

        # Assurez-vous que input_shape est correct
        if len(input_shape) != 4 or input_shape[0] != 1:
            raise ValueError(f"Unexpected input_shape: {input_shape}")

        # Redimensionner l'image
        frame = cv2.resize(frame, (input_shape[2], input_shape[1])).astype(np.uint8)
        frame = np.expand_dims(frame, axis=0)
        self.interpreter.set_tensor(input_details[0]['index'], frame)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])[0]
        probabilities = tf.nn.softmax(output_data.astype(np.float32)).numpy()
        class_id = np.argmax(probabilities)
        confidence = np.max(probabilities)
        print("DEBUG : output_data ->", output_data)
        return class_id, confidence

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        class_id, confidence = self.predict(cv_img)
        label = f"{class_id} ({confidence * 100:.2f}%)"
        
        # Encadrement
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(cv_img, contours, -1, (0, 255, 0), 3)
        if contours:
            # Trouver le contour avec la plus grande aire
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(cv_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(cv_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        qt_img = self.convert_cv_qt(cv_img)
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
