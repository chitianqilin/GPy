#include "complex"
#include "Python.h"
#include "npy_common.h"

static PyObject *computeUpsilonMatrix(PyObject *self, PyObject *args);

void C_computeUpsilonMatrix(
    std::complex<double> gamma,
    double sigma2,
    double *t1,
    double *t2,
    long rows,
    long cols,
    npy_cdouble *P_UpsilonMatrix_npy_data);
