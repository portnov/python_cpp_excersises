#include <iostream>
#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/iterator_range.hpp>
#include <map>
#include <list>
#include <vector>
#include <set>

#include <exception>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
//#include <ndarray.hpp>
//#include <boost/python/numpy.hpp>

#include <Eigen/Dense>

using namespace std;
namespace python = boost::python;
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

object create_knots_euclidean(numeric::array xs) {
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
  //cout << "Norms: " << norms << endl;

  double sum = 0.0;
  for (int idx = 0; idx < (dim1-1); idx++) {
    double norm = norms(idx);
    norms(idx) += sum;
    sum += norm;
  }

  //cout << "Cumsum: " << norms << ", sum: " << sum << endl;

  norms /= sum;

  //cout << "Knots: " << norms << endl;

  double* result_data = norms.data();
  //cout << "First: " << result_data[0] << endl;

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

typedef int vertex_index;

python::list topo_sort(int verts_number, python::list edges) {

  boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph(verts_number);

  for (int i = 0; i < len(edges); i++) {
    int vertex1 = python::extract<int>(edges[i][0]);
    int vertex2 = python::extract<int>(edges[i][1]);
    boost::add_edge(vertex1, vertex2, graph);
  }

  std::vector<vertex_index> result_vector;
  boost::topological_sort(graph, std::back_inserter(result_vector));

  python::list result;
  for (auto const & item : result_vector) {
    result.append(item);
  }

  return result;
}

// Create a typedef for the Graph type
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::property<boost::vertex_index_t, int>, boost::property<boost::edge_index_t, int>> Graph;

// typedef subgraph < Graph > SubGraph;
typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef typename boost::graph_traits<Graph>::edge_descriptor Edge;
typedef boost::graph_traits<Graph> GraphTraits;

// Iterators
typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
typedef boost::graph_traits<Graph>::edge_iterator edge_iter;

typedef boost::shared_ptr<std::vector<unsigned long>> vertex_component_map;

struct EdgeInComponent { 
    vertex_component_map mapping_;
    unsigned long which_;
    Graph const & master_;

    EdgeInComponent(vertex_component_map m, unsigned long which, Graph const & master) 
        : mapping_(m), which_(which), master_(master) {}

    template <typename Edge> bool operator()(Edge const&e) const {
        return mapping_->at(source(e,master_))==which_
            || mapping_->at(target(e,master_))==which_;
    } 
};

struct VertexInComponent { 
    vertex_component_map mapping_;
    unsigned long which_;

    VertexInComponent(vertex_component_map m, unsigned long which)
        : mapping_(m), which_(which) {}

    template <typename Vertex> bool operator()(Vertex const & v) const {
        return mapping_->at(v)==which_;
    } 
};

typedef boost::filtered_graph<Graph, EdgeInComponent, VertexInComponent> ComponentGraph;
typedef boost::property_map<ComponentGraph, boost::vertex_index_t>::type VertexIndexMap;
typedef boost::property_map<ComponentGraph, boost::edge_index_t>::type EdgeIndexMap;

python::list get_components(int verts_number, python::list edges) {

  Graph graph(verts_number);
  for (int i = 0; i < len(edges); i++) {
    int vertex1 = python::extract<int>(edges[i][0]);
    int vertex2 = python::extract<int>(edges[i][1]);
    boost::add_edge(vertex1, vertex2, graph);
  }

  vertex_component_map mapping = boost::make_shared<std::vector<unsigned long>>(verts_number);
  int num_components = boost::connected_components(graph, mapping->data());

  python::list result;

  for (int i = 0; i < num_components; i++) {
    ComponentGraph component = ComponentGraph(graph, EdgeInComponent(mapping, i, graph), VertexInComponent(mapping, i));
    std::map<Vertex,int> new_index_map;
    int new_index = 0;
    python::list vertices;
    python::list edges;

    for (auto const & vertex : boost::make_iterator_range(boost::vertices(component))) {
      vertices.append(vertex);
      new_index_map[vertex] = new_index;
      new_index ++;
    }
    cout << "component[" << i << "]: ";
    for (auto edge : boost::make_iterator_range(boost::edges(component))) {
      Vertex v1 = boost::source(edge, component);
      Vertex v2 = boost::target(edge, component);

      auto i1 = new_index_map[v1];
      auto i2 = new_index_map[v2];
      cout << i1 << " (old " << v1 << ") --> " << i2 << " (old " << v2 << "); ";
      edges.append(python::make_tuple(i1, i2));
    }
    cout << endl;
    result.append(python::make_tuple(vertices, edges));
  }

  return result;
}

void* init_numpy() {
  import_array();
}

BOOST_PYTHON_MODULE(greet) {
  init_numpy();
  numeric::array::set_module_and_type("numpy", "ndarray");

  def("say_greeting", say_greeting);
  def("multiply", multiply);
  def("process", process);
  def("create_knots_euclidean", create_knots_euclidean);
  def("determinant", determinant);
  def("topo_sort", topo_sort, return_value_policy<return_by_value>());
  def("get_components", get_components, return_value_policy<return_by_value>());
}

