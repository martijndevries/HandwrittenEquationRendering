# Handwritten Equation Renderer

## Problem Statement

As a data scientist at a small tech startup, I have been tasked with developing a new tool that can render equations into digital format. Using the 2011-2013 CROHME datasets of handwritten equations and mathematical symboos, I will build a tool that accepts an image of an handwritten equation, predicts which symbols are on the image, and then renders the equation using LaTex. Both the image of the equation and the latex representation should be returned. For the symbol recognition, I will use a pre-trained EfficientNetB0 Convolutional Network. In order to evaluate the success of the model, we will look both at how accurately the model can predict individual symbols. I will also calculate the 'Demerau-Levenshtein' distance between the actual equation and predicted equation

