# Interface specific notes

## Tensorflow interface
* For this version I have written both a fast c++/cuda version of the transformation
  and a version running pure tensorflow. As default it will try to use the fast
  version, however on windows and mac it defaults to the slow pure version. The 
  library will let you know which version it uses.
  
* I have precompiled the c++/cuda versions into dynamic libraries. If they do not
  work you may have to recompile them for your system. Please inspect the different 
  makefiles.

* As default will all the methods expect numpy arrays as input and output numpy.
  However, if you want to incorporate the transformations in a larger tensorflow
  graph (or just like to work with tensorflow tensors) just initialize the 
  transformer class with the optional argument `return_tf_tensors=True`

* This backend has a additional method called `T.fix_data_size()`. If this is used
  then the library can pre-compile a computational graph that makes computations
  faster. This only works when `return_tf_tensors=False`.

* All methods expects batches of data as input and outputs also data as batches.
  Batch format is:
  * 1D data: [batch size, number of features]
  * 2D data: [batch size, height of images, width of images, number of channels]
  * 3D data: [batch size, height of volume, width of volume, depth of volume]
  
## Pytorch interface
* All methods expect torch tensors as input and output as torch tensors. Use
  tensor.numpy() to get the corresponding numpy array.

* As default will the computations run on the cpu. If you want to run on the
  gpu, initialize the transformer class with the optional argument
  `device='gpu'`.
  
* I have only implemented this version in pure pytorch. I am looking at how to
  implement a faster c++/cuda version, but this probably requires more experience
  with the framework.
  
* All methods expects batches of data as input and outputs also data as batches.
  Batch format is:
  * 1D data: [batch size, number of features]
  * 2D data: [batch size, number of channels, height of images, width of images]
  * 3D data: [batch size, number of channels, depth of volume, height of volume, width of volume]

