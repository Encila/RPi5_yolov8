import torch
import tensorflow as tf
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ops  # for postprocess
from pathlib import Path
import cv2
import numpy as np

# .pt files contains names in there but exported onnx/tflite don't have them.
yolo_default_label_names = {0: 'orchidée pyramidale', 1: 'pervenche', 2: 'vesce', 3: 'autre orchidée'}

class YoloDetector:
    def __init__(self, model_path, task='detect'):
        self.model = YOLO(model_path, task=task)

        self.imgsz = 640  # assume 640 at the moment since it is the default one
        if model_path.suffix == '.onnx':
            # once exported to onnx, auto resizing doesn't seem to work as expected
            # probably there is a better way but I'll just read it from onnx file
            # and set the dimension when predict
            # note, square images only atm
            import onnx
            dummy_model = onnx.load(str(model_path))
            self.imgsz = dummy_model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
            del dummy_model
        if model_path.suffix == '.tflite':
            labels = open("../models/orchidees/labels.txt").read().splitlines()
            global yolo_default_label_names
            yolo_default_label_names = {i: label for i, label in enumerate(labels)}

    def predict(self, frame, conf):
        return self.model.predict(source=frame, save=False, conf=conf, save_txt=False, show=False, verbose=False,
                                  imgsz=self.imgsz)

    def get_label_names(self):
        if self.model.names is None or len(self.model.names) == 0:
            return yolo_default_label_names
        return self.model.names


class YoloDetectorTFLite:
    def __init__(self, model_path, use_coral_tpu=False):
        self.name = model_path.name
        
        self.use_coral_tpu = use_coral_tpu
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

    def predict(self, frame, conf):
        orig_imgs = [frame]

        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']

        # TODO check shape of input_shape and frame.shape
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        _, w, h, _ = input_shape

        # check width and height
        if frame.shape[0] != h or frame.shape[1] != w:
            input_img = cv2.resize(frame, (w, h))
        else:
            input_img = frame
        input_img = input_img[np.newaxis, ...]  # add batch dim
        
        input_img = input_img.astype(np.float32) / 255.  # change to float img
        self.interpreter.set_tensor(input_details[0]['index'], input_img)

        self.interpreter.invoke()

        preds = self.interpreter.get_tensor(output_details[0]['index'])
          
        if preds.size == 0:
            return []    
        
        ######################################################################
        # borrowed from ultralytics\models\yolo\detect\predict.py #postprocess

        # convert to torch to use ops.non_max_suppression
        # ultralytics is working on none-deeplearning based non_max_suppression
        # https://github.com/ultralytics/ultralytics/issues/1777
        # maybe someday, but for now, just workaround
        preds = torch.from_numpy(preds)
        preds = ops.non_max_suppression(preds,
                                        conf,
                                        0.7,  # todo, make into arg
                                        agnostic=False,
                                        max_det=300,
                                        classes=None)  # hack. just copied values from execution of yolov8n.pt

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]

            # tflite result are in [0, 1]
            # scale them by width (w == h)
            pred[:, :4] *= w

            pred[:, :4] = ops.scale_boxes(input_img.shape[1:], pred[:, :4], orig_img.shape)
            img_path = ""
            results.append(Results(orig_img, path=img_path, names=yolo_default_label_names, boxes=pred))

        return results

    def get_label_names(self):
        return yolo_default_label_names


class YoloDetectorWrapper:
    def __init__(self, model_path, use_coral_tpu=False):
        model_path = Path(model_path)

        if use_coral_tpu or model_path.suffix == '.tflite':
            self.detector = YoloDetectorTFLite(model_path, use_coral_tpu)
        else:
            self.detector = YoloDetector(model_path)

    def predict(self, frame, conf=0.5):
        return self.detector.predict(frame, conf=conf)

    def get_label_names(self):
        return self.detector.get_label_names()
