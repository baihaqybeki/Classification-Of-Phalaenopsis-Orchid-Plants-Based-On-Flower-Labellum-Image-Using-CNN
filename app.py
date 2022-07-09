import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import os
import numpy as np
from predict import predict

def import_and_predict(image_data, model):
    """function untuk prediksi kelas gambar"""
    size = (75,75) # ukuran gambar untuk prediksi untuk x dan y   
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS) 
    image = image.convert('RGB') #convert gambar ke rgb, in case gambar dalam bentuk bgr
    image = np.asarray(image) # mengubah gambar menjadi numpy array
    image = (image.astype(np.float32) / 255.0) #normalisasi matrix gambar
    img_reshape = image[np.newaxis,...] 

    prediction = model.predict(img_reshape) #predict gambar menggunakan kelas model.predict
    return prediction

model = tf.keras.models.load_model('checkpoint/model_putik.hdf5') # load model InveptionV3 untuk prediksi kelas

interpreter = tf.lite.Interpreter(model_path="./checkpoint/putik_new.tflite") # load model yolov4 tiny untuk proses crop putik dari gambar

st.write("""
         # Phalaenopsis Lamelligera and Cornu-Cervi Classification
         """
         )

st.write("This is an image classification web app to predict Phalaenopsis Lamelligera and Cornu-Cervi")

file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    filename = "uploaded.jpg"
    
    image.save(filename)
    gambar = predict(filename,interpreter) # crop putik dari input gambar
    
    if gambar is not None:
        prediction = import_and_predict(gambar, model) #predict kelas anggrek dari cropped images
    else:
        prediction = None
        st.write("please input the right image")
    
    os.remove("uploaded.jpg")
    st.image(image, use_column_width=True)
    
    if np.argmax(prediction) == 0:
        st.write("Cornucervi")
    elif np.argmax(prediction) == 1:
        st.write("Lamelligera")
    else:
        st.write("Input the correct image")
    
    st.text("Probability (0: Cornucervi, 1: Lamelligera)")
    st.write(prediction)