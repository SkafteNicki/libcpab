# libcpab
CPAB Transformations [1]: finite-dimensional spaces of simple, fast, and 
highly-expressive diffeomorphisms derived from parametric, 
continuously-defined, velocity fields in **Tensorflow**.

The main idea behind this library is to offer a simple way to use and
incorporate diffiomorphic transformations. The diffiomorphic transformations
are based on the work of [Freifeld et al.](https://www.cs.bgu.ac.il/~orenfr/papers/freifeld_etal_PAMI_2017).
The library supports diffiomorphic transformations in 1D, 2D and 3D. 

This code is based on the original implementation CPAB transformations by
Oren Freifeld (Github repo: [cpabDiffeo](https://github.com/freifeld/cpabDiffeo)).

## Author of this software

Nicki Skafte Detlefsen (email: nsde@dtu.dk)

Thanks to Tobias Slott Jensen and Asger Ougaard for suppling the base code 
for the 1D and 3D cases.

This software is released under the MIT License (included with the software). 
Note, however, that using this code (and/or the results of running it) to support
any form of publication (e.g., a book, a journal paper, a conference papar, a patent
application ect.) we request you to cite [1].

## Requirements

* Generic python packages: numpy, scipy, matplotlib
* Tensorflow
* To use the GPU implementation, you need a nvidia GPU and CUDA + cuDNN installed. 
  See [Tensorflows GPU installation instructions](https://www.tensorflow.org/install/) 
  for more details

## Installation

Clone this reposatory to a directory of your choice
```
git clone https://github.com/SkafteNicki/ddtn
```
Add this directory to your PYTHONPATH
```
export PYTHONPATH=$PYTHONPATH:$YOUR_FOLDER_PATH/ddtn
```

## How to use
The interface is simple to use and only have 7 different methods that should
get you started with diffiomorphic transformations. 
```
    # Import library
    from libcpab import cpab
 
    # Define a 2x2 transformation class
    T = cpab(tess_size=[2,2])
    
    # Important methods
    g = T.uniform_meshgrid(...) # sample uniform grid of points in transformer domain
    theta = T.sample_transformation(...) # sample random normal transformation vectors
    dim = T.get_theta_size() # get dimensionality of transformation parametrization
    params = T.get_params() # get different transformer parameters
    g_t = T.transform_grid(g, theta) # transform a grid of points using theta
    data_t1 = T.interpolate(data, g_t) # interpolate some data using the transformed grid
    data_t2 = T.transform_data(data, theta) # combination of the two last methods 
```
All these methods expects numpy arrays as input and returns numpy arrays. 
If you want the method to output tf tensors instead, just set the `return_tf_tensor`
argument in the `cpab` class to `True``

Additionally, we supply two case scrips:
* case1.py: simple use of the library to transform data
* case2.py: image registration by incorporating the library in a tensorflow optimization rutine
For a specific use of the transformations in a greater context, 
see this [paper](http://www2.compute.dtu.dk/~sohau/papers/cvpr2018/detlefsen_cvpr_2018.pdf)  
and this [github repo](https://github.com/SkafteNicki/ddtn).

## References
```
[1] @article{freifeld2017transformations,
  title={Transformations Based on Continuous Piecewise-Affine Velocity Fields},
  author={Freifeld, Oren and Hauberg, Soren and Batmanghelich, Kayhan and Fisher, John W},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2017},
  publisher={IEEE}
}

[2] @article{detlefsen2018transformations,
  title = {Deep Diffeomorphic Transformer Networks},
  author = {Nicki Skafte Detlefsen and Oren Freifeld and S{\o}ren Hauberg},
  journal = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018},
  publisher={IEEE}
}

```