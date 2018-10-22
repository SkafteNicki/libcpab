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
 * main entry point                                                          *
 *****************************************************************************/
static PyObject *npcpab1(PyObject *self, PyObject *args) {
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



// Method definition object for this extension, these argumens mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//          accepting arguments, accepting keyword arguments, being a
//          class method, or being a static method of a class.
// ml_doc:  Contents of this method's docstring
static PyMethodDef npcpab1_methods[] = { 
    {   
        "npcpab1", npcpab1, METH_VARARGS,
        "Evaluate cpab for 1D signals; this function should not be called directly."
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
    npcpab1_methods
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

