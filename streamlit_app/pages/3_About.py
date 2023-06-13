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
    page_title="About this tool",
    page_icon="👋",
)


st.write("# About this tool")

st.markdown("""
    This app was created by Martijn de Vries. For more information, see the [github page](https://github.com/martijndevries/HandwrittenEquations)
    
    ## Some Explanation and Limitations
    - This tool supports most common letters and mathematical operators, including functions like log, lim, or trigonometric functions like sin or cos
    - To increase the chance of the equation being rendered correctly, make sure individual symbols are connected, and that symbols do not touch each other
    - Structures like fractions are recognized and rendered correctly, but fractions within fractions (or eg. fractions in superscripts) will not be properly recognized
    
    ## Contact:
    - Email: martijndevries91@gmail.com 
    - LinkedIn: [linkedin.com/in/mn-devries](https://www.linkedin.com/in/mn-devries)
    - Portfolio: [martijndv.com](https://www.martijndv.com)
    - Github: [github.com/martijndevries/](https://github.com/martijndevries/)
    """)
