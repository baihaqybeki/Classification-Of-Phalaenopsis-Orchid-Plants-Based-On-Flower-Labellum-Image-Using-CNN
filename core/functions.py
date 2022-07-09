import os
import cv2
import numpy as np
from core.utils import read_class_names
from core.config import cfg


# berfungsi untuk crop setiap hasil deteksi dan menyimpan hasil crop sebagai gambar baru
def crop_objects(img, data, path, allowed_classes,img_name):
    boxes, scores, classes, num_objects = data
    index_max=np.argmax(scores)
    class_names = read_class_names(cfg.YOLO.CLASSES)
    
    #membuat dictionary untuk hold semua hasil deteksi
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            if i == index_max:
                counts[class_name] = counts.get(class_name, 0) + 1
                # mendapatkan koordinat bounding box
                xmin, ymin, xmax, ymax = boxes[i]
                
                # crop hasil deteksi dari gambar (ambil 5 pixels tambahan di semua sisi)
                cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
                
                # menyimpan gambar hasil crop
                img_name = "{}-{}.png".format(img_name,i)
                img_path = os.path.join(path, img_name)
                cv2.imwrite(img_path, cropped_img)
        else:
            continue
    return cropped_img