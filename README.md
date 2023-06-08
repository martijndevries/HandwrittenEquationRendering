# Capstone Project - Rendering Handwritten Equations

By Martijn de Vries <br>
martijndevries91@gmail.com

## Problem Statement

As a data scientist at a small tech startup, I have been tasked with developing a new tool that can render equations into digital format. Using the 2011-2013 CROHME datasets of handwritten equations and mathematical symboos, I will build a tool that accepts an image of an handwritten equation, predicts which symbols are on the image, and then renders the equation and  LaTeX. Both the image of the equation and the latex representation should be returned. For the symbol recognition, I will use a pre-trained EfficientNetB0 Convolutional Network. In order to evaluate the success of the model, we will look both at how accurately the model can predict individual symbols. I will also calculate the 'Demerau-Levenshtein' distance between the predicted vs ground truth equation strings, in order to gauge how succesfull the full pipeline is in resolving the equation.


## Repository Overview
    
This repository consists of the following:

<ul>
   <li> The directory <code>./code</code> contains 4 notebooks and several .py files that go through each of the steps in the analysis
   
   <ol>
    <li> In <b>data_processing.ipynb</b> I inspect the inkML data and process them into image files to be used later in the analysis. Two types of images are generated: individual symbol images (to train the model on), and full equation images (to evaluate the performance of the full pipeline at the end </li>
   </ol>
    <li> The directory <code>./CNN_model/</code> contains the trained efficientNetB0 model, used to make predictions on images of individual symbols
   <li> The directory <code>./figures</code> contains all the figures that are saved during the analysis in the notebooks, in .png formats </li>
    <li> The slides for the project presentation are in the file <code>equation_rendering_slides.pdf</code> </li>
</ul>


## Data Overview


## Pipeline Summary

The full pipeline to turn a handwritten equation into a digital one consists of 3 major steps:

<ol>
    <li> <b>Resolving symbols</b> In this pre-processing step, individual symbols are detected on the image as well as some additional information essential to rendering the equation </li>
    <li> <b>Model prediction:</b> An EfficientNetB0 Convolutional Neural Network was trained to recognize individual symbols. The images from the pre-processing step are fed to the model, and a prediction is made for each symbol </li>
    <li> <b>Equation rendering:</b> In this post-processing step, the predicted labels for each symbol are stitched together into an equation, using the predictions and the information from the pre-processing step</li>
</ol>

A high-level overview of each step is given below. More detail on the exact steps taken can be found in the notebooks.

### 1) Resolving symbols

As a first step, when the image is uploaded it is thresholded towards black and white values. For the inkML equation files, this is trivial as they were created digitally. However, the tool should also be able to handle real-world pictures with imperfections and shadows. For those images, a three-step process is applied: firstly, the images is tresholded with adaptive gaussian tresholding. This process often will still result in small-scale features on the image that we would like to remove. To remove these features, a Gaussian blurring is applied, and then a second, binary tresholding is applied to remove these small-scale features (more information on tresholding in openCV can be found <a href=https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html>here</a>).

Next, the symbols are detected with openCVs <a href=https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a>findContours()</a> function. In order to further remove,


## Overall Conclusions