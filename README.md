# An Image Denoising Network with Interpretation
**Author**: Sam Arrington, Edwin S. Yan, Daniel da Silva
 
_EN.605.647 - Neural Networks, Johns Hopkins University_


## Executive Summary
Image denoising is a task that comes up in countless applications and fields. It refers to the task of taking an image with a type of noise and producing a clean image with reduced or eliminated noise. Noise can be defined as an unwanted signal or disturbance interfering with the desired signal (Merriam Webster, 2012).  Common sources of noise include noise originating from an imperfect detector or lossy compression.

In this project, we added three types of noise to a stock image dataset and trained a neural network to remove the noise. The network was able to deliver significant improvements to image quality in each of the three noise types, though imperfections remain. The three types of noise studied were Gaussian noise, Salt/Pepper noise, and Poisson noise. The details of how we produced each of these noise types are reported in the Description of Problem section.

To solve this problem we utilized a single-layer feed-forward neural network with the noisy image as input and the clean image as output. Our initial neural network was trained with images originating from the MNIST dataset of Handwritten Digits (Li et al, 2012), and the best performing version utilized a hidden layer size of 256 nodes, rectified linear unit activations, the Adam optimizer (Kingma et al, 2014), and glorot_uniform
 weight initialization.  The network was trained with the mean-squared error loss function.
 
To achieve the best performance, we repeated the training with different hidden layer sizes and weight initialization methods. The results of these studies are outlined and plotted in the Description of Computational Performance section.

Based on this class's encouragement to look deeper into neural network behavior, we produced an interpretation of the optimization solution based on analogies to physics, linear algebra, and functional analysis. This interpretation is outlined in the Analysis of Performance section.

Overall, a network like this could be utilized in practical applications to improve the quality of an image before it is presented to a user or analyzed by downstream processing. This project was implemented using Python 3.8, Tensorflow, Keras, NumPy, Matplotlib, and Jupyter Notebooks.


## Environment
There are two ways to reproduce our environment in Linux (Tested on Ubuntu and Red Hat Enterprise):
1. **conda_env.yml** - Conda Environment Export
2. **requirements.txt** - Pip Packages Export


## Folder Structure
1. **noisy.py** - Helper Function that Generate Different Noises
2. **utility.py** - Helper Function that pre-process data and plot images
3. **SA_experiments.ipynb** & **SA_experiments_Extended_Range.ipynb** - MSE vs Hidden Nodes; AACS vs Hidden Nodes
4. **experiments.ipynb** - MSE vs Weight Initializers
5. **images_used_in_paper.ipynb** - Plots used in the presentation and final paper







