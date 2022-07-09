from predict import predict
import os
import cv2
import tensorflow as tf

def crop_images_for_dir(dir,interpreter):
    """membuat dataset dari folder yang berisi gambar bunga"""
    list_images = os.listdir(dir)
    for images in list_images:
        images = dir + "/" + images
        predict(images,interpreter)

def main():
    dir = "cc"
    interpreter = tf.lite.Interpreter(model_path="./checkpoints/putik_new.tflite")
    crop_images_for_dir(dir,interpreter)

if __name__ == "__main__":
    main()
