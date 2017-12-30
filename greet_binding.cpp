#include <iostream>
#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <map>
//#include <list>
#include <set>

#include <exception>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
//#include <ndarray.hpp>
//#include <boost/python/numpy.hpp>

#include <Eigen/Dense>

using namespace std;
//namespace python = boost::python;
using namespace boost::python;
//namespace numpy = boost::python::numpy;

using namespace Eigen;


void say_greeting(const char* name) {
  cout << "Hello, " << name << "!\n";
}


struct WrongSizeError : public std::exception {
  const char* what() const throw() { return "Unsupported array size."; }
};

struct WrongTypeError : public std::exception {
  const char* what() const throw() { return "Unsupported array type."; }
};

// Boost::Python needs the translators
void translate_sz(const WrongSizeError& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}

void translate_ty(const WrongTypeError& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}

// multiply matrix of double (m) by f
object multiply(numeric::array m, double f)
{
  PyObject* m_obj = PyArray_FROM_OTF(m.ptr(), NPY_DOUBLE, NPY_ARRAY_CARRAY);
  if (!m_obj)
    throw WrongTypeError();

  // to avoid memory leaks, let a Boost::Python object manage the array
  object temp(handle<>(m_obj));

  // check that m is a matrix of doubles
  int k = PyArray_NDIM((PyArrayObject*)m_obj);
  if (k != 2)
    throw WrongSizeError();

  // get direct access to the array data
  const double* data = static_cast<const double*>(PyArray_DATA((PyArrayObject*)m_obj));

  // make the output array, and get access to its data
  PyObject* res = PyArray_SimpleNew(2, PyArray_DIMS((PyArrayObject*)m_obj), NPY_DOUBLE);
  double* res_data = static_cast<double*>(PyArray_DATA((PyArrayObject*)res));

  const unsigned size = PyArray_SIZE((PyArrayObject*)m_obj); // number of elements in array
  for (unsigned i = 0; i < size; ++i)
    res_data[i] = f*data[i];

  return object(handle<>(res)); // go back to using Boost::Python constructs
}

void process(numeric::array ps) {
  PyArrayObject* pairs = (PyArrayObject*)PyArray_FROM_OTF(ps.ptr(), NPY_DOUBLE, NPY_ARRAY_CARRAY);
  if (! pairs) {
    throw WrongTypeError();
  }

  int dims = PyArray_NDIM(pairs);
  if (dims != 2) {
    throw WrongSizeError();
  }

  // to avoid memory leaks, let a Boost::Python object manage the array
  object temp(handle<>(pairs));

  int stride1 = PyArray_STRIDE(pairs, 0) / sizeof(double);
  int stride2 = PyArray_STRIDE(pairs, 1) / sizeof(double);

  //cout << "Stride 1: " << stride1 << ", 2: " << stride2 << endl;

  int dim1 = PyArray_DIM(pairs, 0);
  int dim2 = PyArray_DIM(pairs, 1);

  //cout << "Dim 1: " << dim1 << ", 2: " << dim2 << endl;

  if (dim2 != 2) {
    throw WrongSizeError();
  }

  double* data = static_cast<double*>(PyArray_DATA(pairs));

  map<double, set<double>> my_map;

  for (int idx = 0; idx < dim1; idx++) {
    double u = data[ idx * stride1 + 0 * stride2 ];
    double v = data[ idx * stride1 + 1 * stride2 ];
    //cout << "[" << idx << "]: U = " << u << ", V = " << v << endl;
    my_map[v].insert(u);
  }

  /*for (auto const & entry : my_map) {
    auto const & key = entry.first;
    auto const & values = entry.second;
    for (auto const & value : values) {
      //cout << "Key: " << key << " => " << value << endl;
    }
  }*/
  cout << "Size: " << my_map.size() << endl;

}

object twice(numeric::array xs) {
  PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(xs.ptr(), NPY_DOUBLE, NPY_ARRAY_CARRAY);
  if (! array) {
    throw WrongTypeError();
  }

  int dims = PyArray_NDIM(array);
  if (dims != 2) {
    throw WrongSizeError();
  }

  // to avoid memory leaks, let a Boost::Python object manage the array
  object temp(handle<>(array));

  int dim1 = PyArray_DIM(array, 0);
  int dim2 = PyArray_DIM(array, 1);

  double* data = static_cast<double*>(PyArray_DATA(array));

  typedef Matrix<double,Dynamic,Dynamic,RowMajor> MyMatrix;
  MyMatrix matrix = Map<MyMatrix>(data, dim1, dim2);

  MyMatrix m1 = matrix.block(1, 0, (dim1-1), dim2);
  MyMatrix m2 = matrix.block(0, 0, (dim1-1), dim2);

  //cout << "M1: " << m1 << endl;
  //cout << "M2: " << m2 << endl;

  MyMatrix m3 = m2 - m1;
  VectorXd norms = m3.rowwise().norm();
  cout << "Norms: " << norms << endl;

  double sum = 0.0;
  for (int idx = 0; idx < (dim1-1); idx++) {
    double norm = norms(idx);
    norms(idx) += sum;
    sum += norm;
  }

  cout << "Cumsum: " << norms << ", sum: " << sum << endl;

  norms /= sum;

  cout << "Knots: " << norms << endl;

  double* result_data = norms.data();
  cout << "First: " << result_data[0] << endl;

  long int result_dims[1];
  result_dims[0] = dim1-1;

  // make the output array, and get access to its data
  PyObject* result = PyArray_SimpleNewFromData(1, result_dims, NPY_DOUBLE, result_data);

  handle<> handle(result);
  numeric::array result_array(handle);
  return result_array.copy();
}

void determinant(numeric::array xs) {
  PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(xs.ptr(), NPY_DOUBLE, NPY_ARRAY_CARRAY);
  if (! array) {
    throw WrongTypeError();
  }

  int dims = PyArray_NDIM(array);
  if (dims != 2) {
    throw WrongSizeError();
  }

  // to avoid memory leaks, let a Boost::Python object manage the array
  object temp(handle<>(array));

  int dim1 = PyArray_DIM(array, 0);
  int dim2 = PyArray_DIM(array, 1);
  if (dim1 != dim2) {
    throw WrongSizeError();
  }

  double* data = static_cast<double*>(PyArray_DATA(array));

  typedef Matrix<double,Dynamic,Dynamic,RowMajor> MyMatrix;
  MyMatrix matrix = Map<MyMatrix>(data, dim1, dim2);
  cout << matrix.determinant() << endl;
}
// numpy::ndarray multiply(numpy::ndarray m, double f) {
//   return nullptr;
// }

void* init_numpy() {
  import_array();
}

BOOST_PYTHON_MODULE(greet) {
  init_numpy();
  numeric::array::set_module_and_type("numpy", "ndarray");

  def("say_greeting", say_greeting);
  def("multiply", multiply);
  def("process", process);
  def("twice", twice);
  def("determinant", determinant);
}
