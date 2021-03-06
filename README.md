# edgeAI
This is an open-source python tool to generate tailored c-code for embedded devices.
The code can be used to deploy deep neural networks, qp-solvers and more
for several applications like control, image recognition, etc..
It can also be used to run simulations of an MPC setup and similar usages.

## Getting Started

### Installing
Just download or clone the repository and make sure all the following packages are installed.

### Packages
* NumPy
* h5py
* PICOS
* CVXOPT
* SciPy

### First examples
Three examples are provided in the folder examples.
Just change directory to examples, open a terminal and run
```
python ~.py
```
or
```
run ~.py
```
from an ipython console.

## Content
This section explains how **edgeAI** is organised:

### examples
In examples are a three scripts that can be run directly to generate code.
These examples show a few possibilities how the **edgeAI** toolbox can be used.

### external_data
**edgeAI** can use data saved in a *~.csv* to simulate disturbances or other external influences like the weather.

### systems
This folder contains the systems as a function returning a description for the simulation (system matrices) and control of a system (constraints, etc.).

### lib
This is the heart of **edgeAI**. It includes all the code for the code-generation and all the classes used to handle projects.

### neural_networks
edgeAI can read neural networks trained with keras (*~.h5* and *~.json*) and use them to generate code.
Additionally, scaling data for the input and the output of the network can be provided as *~.csv*.
How the files and the headers should be named can be seen in the examples.

## Authors
* **Benjamin Karg** - *Initial work* - [Deep-learning based embedded mixed-integer model predictive control](https://github.com/benjaminkarg/edgeAI)
  * B. Karg and S. Lucia. "Deep-learning based embedded mixed-integer model predictive control". In Proc. of the European Control Conference (in Press), 2018.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* Thank you very much to all the contributors to the libraries named in *Packages*
* Thanks to Prof. Sergio Lucia ;)
