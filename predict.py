import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 

import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from PIL import Image
import cv2
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def predict(images,interpreter):
    """predict gambar putik dari gambar input lalu di crop"""
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(True,"yolov4") # mengatur architecture model menjadi yolov4 tiny
    input_size = 416 # size input gambar

    original_image = cv2.imread(images)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) #mengubah gambar dari BGR ke RGB (dari PIL)
    
    image_name = images.split('/')[-1]
    image_name = images.split('.')[0] # mengambil nama gambar

    image_data = cv2.resize(original_image, (input_size, input_size)) # ukuran gambar input di-resize menjadi ukuran 416x416
    image_data = image_data / 255. # normalisasi matrix gambar
    
    images_data = []  
    
    for i in range(1):
        images_data.append(image_data)
    
    images_data = np.asarray(images_data).astype(np.float32)
    
    # Predict Image menggunakan yolov4 tiny tflite
    
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size])) # return boxes location and pred_conf

    #Non Max Supression untuk menghasilkan prediksi yang valid
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.45
            )
    
    original_h, original_w, _ = original_image.shape #mengambil size dari gambar sebelum di resize
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
            
    # hold data detection dalam satu variabel
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
    
    # membaca nama kelas dari config file
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    allowed_classes = list(class_names.values())

    #proses crop gambar hasil prediksi
    img_name = image_name
    crop_path = os.path.join(os.getcwd(), 'detections', 'crop')
    
    try:
        cropped_img = crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes, img_name)
        image = Image.fromarray(cropped_img.astype(np.uint8)) #return gambar dari array hasil dari return value function crop_objects
        return image
    except:
        pass
