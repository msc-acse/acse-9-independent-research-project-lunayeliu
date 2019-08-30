# acse-9-independent-research-project-lunayeliu
Applying machine learning to the optimisation of numericalintegration in finite element method

## Introduction
The software developed is a novel smart FEM platform which combined machine learning techniques with the traditional deterministic method. To specific mechanics problem, this software is able to provide solutions more efficiently with the same accuracy than the traditional method.

As shown in the figure below, the input of the neural network here is the coordinates value of the element.  Theoutput of the network would be sent to a softmax function( a cross-entropy function).  Then a probabilitydistribution of the possible number of integration points needed is calculated.  The number with the highestpossibility would be chosen as the input number of the tensor quad.
The neural network in the machine learning module can select the optimal number of integration points for numerical integration according to elements shapes. The classification accuracy of the network on unseen data can reach $94.3\%$.

<p align="center">
  <img src="https://user-images.githubusercontent.com/43916396/63956539-86563f00-ca7e-11e9-867b-c98984a7f9e7.png" width="520" height="200"><br>
</p>

## Documentation

The documentation is generated automatically by Pydoc. Part of them can be found in the 'docs' repo in the format of html. Open with browser to check.

## Repo Structure
* __data__		- contains the data of 8-node hexahedral and 10-node tatrehedron element prepared for the training model in the 'trained model' repo
 
* __data_generation__ - holds the code for generating the data in the 'data' repo

* __elements__ - three kinds of FEM element: 8-node hexahedral, 4-node tetrahedron, 10-node tetrahedron 
 
* __FEM_modules__ - sepsific FEM code for linear elasticity problem
  
* __HPC_scripts__ - the scripts of generating dataset using HPC
  
* __integration_module__ - contains the smart integration method

* __trained_model__  - holds all the machine learning models

* __utils__ - some basic tools 

## Dependencies
To be able to run this software, the following packages and versions are required:

 - Python Version 3.7
 - NumPy Version 1.16.3
 - SciPy Version 1.3.0
 - PyTorch Version 1.1
 - Sikit-learn Version 0.21
 - matplotlib Version 3.1.0

## Test and example


## Author and Course Information

Author： Ye Liu

Github: @lunayeliu

Email: ye.liu18@imperial.ac.uk

CID: 01626306

Course: ACSE-9: Applied Computational Science Project

## License

Licensed under the MIT [license](https://github.com/msc-acse/acse-9-independent-research-project-lunayeliu/blob/master/LICENSE)
