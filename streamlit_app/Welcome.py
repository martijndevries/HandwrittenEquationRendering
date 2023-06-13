import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import smart_resize
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from resolve_symbols import resolve_symbols_on_img
from render_equations import render_equation
from make_predictions import make_prediction


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

#------
st.title("Handwritten Equation Recognition")

#st.sidebar.success("Upload your own equation, or see an example")

st.markdown("Welcome! This tool will render a digital equation from handwriting. Simply upload an image with an equation or mathematical expression, and see the result for yourself!")


st.markdown("Try the example below:")

if not st.button('Render the equation'):
    st.image("math_martijn3.jpeg") 
else:
    ##class labels
    with open('../CNN_model/class_names.txt', 'r') as f:
        lines = f.readlines()
    class_labels = [label.split(' ')[-1][:-1] for label in lines]   

    #load model
    efficientnet_model = tf.keras.models.load_model("../CNN_model/efficientnet_model_lw.h5")

    symbs, levels, stack, script_levels, extend_list, fig, ax = resolve_symbols_on_img("math_martijn3.jpeg",  plot=True)
    ax.set_frame_on(False)
    ax.tick_params(axis='both',which='both',bottom=False, left=False, labelbottom=False, labelleft=False) 
        
    pred_symbol_list = make_prediction(symbs, extend_list, efficientnet_model, class_labels)
    eqstr = render_equation(pred_symbol_list, levels, stack, script_levels, extend_list)

    st.subheader("Predicted symbols, order, and position:")

    st.pyplot(fig)
    
    
    st.markdown('<p class="big-font"> Predicted equation: </p>', unsafe_allow_html=True)
    
    st.text("Raw string: " +  eqstr.replace(" ",""))

    st.write("Rendered in LaTeX:  " r'' + eqstr)
