#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include "cpabcpu.cc"

/*****************************************************************************
 * Array access macros.                                                      *
 *****************************************************************************/
#define IDX1(x, i)          (*((npy_float64*)PyArray_GETPTR1(x, i)))
//#define IDX2(x, i, j)       (*((npy_float64*)PyArray_GETPTR2(x, i, j)))
//#define IDX3(x, i, j, k)    (*((npy_float64*)PyArray_GETPTR3(x, i, j, k)))
//#define IDX4(x, i, j, k, l) (*((npy_float64*)PyArray_GETPTR4(x, i, j, k, l)))


/*****************************************************************************
 * Forward integration wrapper function                                      *
 *****************************************************************************/
static PyObject *npcpab_forward(PyObject *self, PyObject *args) {
  // Declare variables. 
  npy_int64 nStepSolver;
  PyArrayObject *points, *Trels, *new_points, *nc;

  // Grap input
  if (!PyArg_ParseTuple(args, "O!O!O!O!l",
                        &PyArray_Type, &points,
                        &PyArray_Type, &Trels,
                        &PyArray_Type, &new_points,
                        &PyArray_Type, &nc,
                        &nStepSolver)) {
    return NULL;
  }

  // Problem size
  const int ndim = PyArray_DIMS(points)[0];
  const int nP = PyArray_DIMS(points)[1];
  const int batch_size = PyArray_DIMS(Trels)[0];
  
  // Get data pointers
  npy_float64 *raw_points    = (npy_float64*)PyArray_DATA(points);
  npy_float64 *raw_trels     = (npy_float64*)PyArray_DATA(Trels);
  npy_float64 *raw_newpoints = (npy_float64*)PyArray_DATA(new_points);
  int nc_int[ndim];
  for (int k = 0; k < ndim; k++)
    nc_int[k] = IDX1(nc, k); // XXX: THIS ASSUME THAT nc IS A FLOAT64 ARRAY -- WHAT SHOULD IT BE?  
  
  // Call the work-horse
  cpab_forward_cpu(raw_points, raw_trels, ndim, nP, batch_size, nStepSolver, nc_int, raw_newpoints);

  // Get back to python
  Py_RETURN_NONE;
}

/*****************************************************************************
 * Backward integration wrapper function                                     *
 *****************************************************************************/
static PyObject *npcpab_backward(PyObject *self, PyObject *args) {
  // Declare variables. 
  npy_int64 nStepSolver;
  PyArrayObject *points, *As, *Bs, *grad, *nc;

  // Grap input
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!l",
                        &PyArray_Type, &points,
                        &PyArray_Type, &As,
                        &PyArray_Type, &Bs,
                        &PyArray_Type, &grad,
                        &PyArray_Type, &nc,
                        &nStepSolver)) {
    return NULL;
  }

  // Problem size
  const int ndim = PyArray_DIMS(points)[0];
  const int nP = PyArray_DIMS(points)[1];
  const int batch_size = PyArray_DIMS(As)[0];
  const int d = PyArray_DIMS(Bs)[0];
  const int nC = PyArray_DIMS(Bs)[1];
  
  // Get data pointers
  npy_float64 *raw_points = (npy_float64*)PyArray_DATA(points);
  npy_float64 *raw_As     = (npy_float64*)PyArray_DATA(As);
  npy_float64 *raw_Bs     = (npy_float64*)PyArray_DATA(Bs);
  npy_float64 *raw_grad   = (npy_float64*)PyArray_DATA(grad);
  int nc_int[ndim];
  for (int k = 0; k < ndim; k++)
    nc_int[k] = IDX1(nc, k); // XXX: THIS ASSUME THAT nc IS A FLOAT64 ARRAY -- WHAT SHOULD IT BE?  
  
  // Call the work-horse
  cpab_backward_cpu(raw_points, raw_As, raw_Bs, ndim, nP, batch_size, nStepSolver, d, nC, nc_int, raw_grad);

  // Get back to python
  Py_RETURN_NONE;
}


/*
void cpab_backward(const FLOAT *points, // [ndim, nP]
                   const FLOAT *As, // [batch_size, nC, ndim, ndim+1]
                   const FLOAT *Bs, // [d, nC, ndim, ndim+1]
                   int ndim, int nP, int batch_size, int nstepsolver, int d, int nC, // scalar
                   const int *nc, // [ndim]
                   FLOAT *grad){ // [d, batch_size, ndim, nP]
*/


// Method definition object for this extension, these argumens mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//          accepting arguments, accepting keyword arguments, being a
//          class method, or being a static method of a class.
// ml_doc:  Contents of this method's docstring
static PyMethodDef npcpab_methods[] = { 
    {   
        "npcpab_forward", npcpab_forward, METH_VARARGS,
        "CPAB forward integration; this function should not be called directly."
    },  
    {   
        "npcpab_backward", npcpab_backward, METH_VARARGS,
        "CPAB backward integration; this function should not be called directly."
    },  
    {NULL, NULL, 0, NULL}
};

// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for it's method definitions
static struct PyModuleDef nplibcpab_definition = { 
    PyModuleDef_HEAD_INIT,
    "nplibcpab",
    "A Python module that prints 'hello world' from C code.",
    -1, 
    npcpab_methods
};

// Module initialization
// Python calls this function when importing your extension. It is important
// that this function is named PyInit_[[your_module_name]] exactly, and matches
// the name keyword argument in setup.py's setup() call.
PyMODINIT_FUNC PyInit_nplibcpab(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&nplibcpab_definition);
}

