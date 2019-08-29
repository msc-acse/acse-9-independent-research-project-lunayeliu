# acse-9-independent-research-project-lunayeliu
acse-9-independent-research-project-lunayeliu created by GitHub Classroom

## Introduction
The software developed is a novel smart FEM platform which combined machine learning techniques with the traditional deterministic method. To specific mechanics problem, this software is able to provide solutions more efficiently with the same accuracy than the traditional method.

As shown in the figure below, the input of the neural network here is the coordinates value of the element.  Theoutput of the network would be sent to a softmax function( a cross-entropy function).  Then a probabilitydistribution of the possible number of integration points needed is calculated.  The number with the highestpossibility would be chosen as the input number of the tensor quad.


<img src="https://user-images.githubusercontent.com/43916396/63956539-86563f00-ca7e-11e9-867b-c98984a7f9e7.png" width="520" height="200" div align=center />


<p align="center"> 
<img src="https://user-images.githubusercontent.com/43916396/63956539-86563f00-ca7e-11e9-867b-c98984a7f9e7.png">
</p>



## Installation instructions


## Documentation

## Repo Structure

## Dependencies
To be able to run this software, the following packages and versions are required:

 - NumPy Version 1.16.3
 - SciPy Version 1.3.0
 - PyTorch Version 1.1


## Author and Course Information
Author: Ye Liu
Course: ACSE-9: Applied Computational Science Project

## License
This project is licensed under the MIT [license](https://github.com/msc-acse/acse-9-independent-research-project-lunayeliu/blob/master/LICENSE)
