import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import smart_resize
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from resolve_symbols import resolve_symbols_on_img
from render_equations import render_equation
from make_predictions import make_prediction

st.set_page_config(page_title="Try it yourself", page_icon="📈")


#------
st.subheader("Upload an image of a handwritten equation below:")

#sidebar
#st.sidebar.markdown("## Controls")
#st.sidebar.markdown("You can **change** the values to change the *chart*.")

#user input of an image
input_img = st.file_uploader(label='bla',type=['png', 'jpg', 'jpeg'], label_visibility='hidden')

#class labels
with open('../CNN_model/class_names.txt', 'r') as f:
    lines = f.readlines()
class_labels = [label.split(' ')[-1][:-1] for label in lines]

#load model
efficientnet_model = tf.keras.models.load_model("../CNN_model/efficientnet_model_lw.h5")

#call preprocessing
if input_img is not None:

    image = Image.open(input_img)
    image.save("img.png")
    symbs, levels, stack, script_levels, extend_list, fig, ax = resolve_symbols_on_img("img.png",  plot=True)
    ax.set_frame_on(False)
    ax.tick_params(axis='both',which='both',bottom=False, left=False, labelbottom=False, labelleft=False) 
        
    pred_symbol_list = make_prediction(symbs, extend_list, efficientnet_model, class_labels)
    eqstr = render_equation(pred_symbol_list, levels, stack, script_levels, extend_list)

    st.subheader("Predicted symbols, order, and position:")

    st.pyplot(fig)
    
    
    st.markdown('<p class="big-font"> Predicted equation: </p>', unsafe_allow_html=True)
    
    st.text("Raw string: " +  eqstr.replace(" ",""))

    st.write("Rendered in LaTeX:  " r'' + eqstr)
