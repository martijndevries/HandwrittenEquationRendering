import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import smart_resize
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from resolve_symbols import resolve_symbols_on_img
from render_equations import render_equation

st.set_page_config(layout="centered")


#Model Prediction step

def make_predictions(symbol_list, extend_list):

    pred_list = []
    pred_symbol_list, pred_idx_list = [], []
    for i, symbol in enumerate(symbol_list):

        rgb_im = np.zeros((symbol.shape[0], symbol.shape[1], 3))
        for j in range(3):
            rgb_im[:,:,j] = symbol.astype('uint8')
        resize_img = smart_resize(rgb_im, (100,100))
        resize_img = np.expand_dims(resize_img, axis=0)
        prediction = efficientnet_model.predict(resize_img, verbose=0)
    
        pred_dic = {k[6:]:v for v,k in sorted(zip(prediction[0], class_labels))[-4:]}
        y_classes = prediction.argmax(axis=-1)
        pred_idx_list.append(y_classes)
        label = class_labels[y_classes[0]][6:]
        
        #check if a symbol extends over multiple adjacent symbols
        #if it's not a square root, check if the square root is predicted at a lower probability
        #if so, just use that
        if extend_list[i] >  1 and label != '\\sqrt':
            if '\\sqrt' in pred_dic.keys(): 
                label = '\\sqrt'
        
            
        pred_symbol_list.append(label)
        pred_list.append(pred_dic)
    return pred_symbol_list


st.markdown("""
<style>
.big-font {
    font-size:50x !important;
}
</style>
""", unsafe_allow_html=True)

#------
st.title("Handwritten Equation Recognition")

st.subheader("Upload an image of a handwritten equation below:")
#user input of an image
input_img = st.file_uploader(label='bla',type=['png', 'jpg'], label_visibility='hidden')

#class labels
with open('../class_names.txt', 'r') as f:
    lines = f.readlines()
class_labels = [label.split(' ')[-1][:-1] for label in lines]

#load model
efficientnet_model = tf.keras.models.load_model("../CNN_model/efficientnet_model_lw2.h5")

#call preprocessing
if input_img is not None:

    image = Image.open(input_img)
    image.save("img.png")
    symbs, levels, stack, script_levels, extend_list, fig, ax = resolve_symbols_on_img("img.png",  plot=True)
    ax.set_frame_on(False)
    ax.tick_params(axis='both',which='both',bottom=False, left=False, labelbottom=False, labelleft=False) 
        
    pred_symbol_list = make_predictions(symbs, extend_list)
    eqstr = render_equation(pred_symbol_list, levels, stack, script_levels, extend_list)

    st.subheader("Predicted symbols, order, and position:")

    st.pyplot(fig)
    
    
    st.markdown('<p class="big-font"> Predicted equation: </p>', unsafe_allow_html=True)
    st.write("Rendered in LaTeX:  " r'' + eqstr)
    st.text("Raw string: " +  eqstr)
