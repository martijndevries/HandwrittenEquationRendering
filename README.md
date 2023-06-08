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

As a first step, when the image is uploaded it is thresholded towards black and white values. For the inkML equation files, this is trivial as they were created digitally. However, the tool should also be able to handle real-world pictures with imperfections and shadows. For those images, a three-step process is applied: firstly, the images is tresholded with adaptive gaussian tresholding. This process often will still result in small-scale features on the image that we would like to remove. To remove these features, a Gaussian blurring, and then a second, binary tresholding is applied to remove these small-scale features (more information on tresholding in openCV can be found <a href=https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html>here</a>).

Next, the symbols are detected with openCVs <a href=https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a>findContours()</a> function. In order to further ignore small imperfections on the image, only contours with bounding boxes above a certain size (relative to the total image size) are considered. An example of a thresholded image (an equation handwritten by me), with the bounding boxes from findContours() overlayed, is shown below.

 <img src="./figures/contours_example.png" height="180px"/>

There are a few crucial steps that the pre-processing pipeline needs to take before individual symbols can be fed to the model for prediction: some boxes bound inner contours,  which should not be included. Additionally, some boxes are actually part of a single symbol, like in the 'equals' sign. Additionally, we need to know the order of the symbols, as well as additional information how they relate to other symbols in the equation: are they in a fraction, are they subscripts or superscript? Are they below a limit sign or above a summation sign? The pre-processing pipeline tries to take care of each of these things, primarily by making informed guessess about symbols using the relative locations of the boxes on the image. An example of the same image after pre-processing is shown below:

 <img src="./figures/preprocessed_example.png" height="180px"/>
 
 The pre-processing pipeline has correctly figured out: 1) which boxes are inner contours, and removed them, 2) that the 'equals' sign and the factorial should be considered single symbols, 3) the order the symbols should be read in, 4) the fact that the symbols in the 'limit' sign and within the fraction are related to eachother, in what I call a 'stack', and 5) which symbols are superscripts. 
 
We can also see that the pipeline is not foolproof. In the example above, the infinity symbol is incorrectly guessed not to be a part of the stack, because it extends a little too far out from under the limit sign.

The output of the preprocessing pipeline is as follows
<ol>
 <li>A list of 2D image arrays, each with a single symbol (ordered by appearnce in the equation)</li>
 <li>A list of the 'level' of each symbol, where a 'level' is a set of adjacent symbols that can be read left-to-right </li>
 <li> The 'stack value' of each symbol, where a value of 0 means the symbol is unstacked (there are no symbols above or below it that it has any relation to), and 1,2 and 3 indicate the top, middle (if it exists), and bottom levels of a stack respectively</li>
 <li> The 'script level' of each symbol where a level of 0 means the symbol is at base level, a level of -1 means the symbol is a subscript, a level of 1 means the symbol is a superscript, etc </li>
 <li> The 'extend list' of each symbol. This list for now only exists so that the equation rendering function knows when to close out a root sign. </li>
 <li> If plot=True, an image with the boundign boxes for each symbol (color coded on whether they are base level symbols, in a stack, or super/subscripts) is also shown, and a pyplot ax object is also returned
 </ol>

### 2) Model Prediction

This is the most 



## Overall Conclusions