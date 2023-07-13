#include "mfem.hpp"

using namespace std;
using namespace mfem;

void PrintVertex(Mesh * mesh, int vertex);
void PrintElementVertices(Mesh * mesh, int elem);
void PrintFaceVertices(Mesh * mesh, int face);
template <class T>
void PrintArray(const Array<T> & a, const char *aname);
void PrintSet(const std::set<int> & a, const char *aname);
void PrintVector(const Vector & a, const char *aname);

// for parallel 
void PrintVertex(Mesh * mesh, int vertex,  int printid);
void PrintElementVertices(Mesh * mesh, int elem,  int printid);
void PrintFaceVertices(Mesh * mesh, int face,  int printid);
template <class T>
void PrintArray(const Array<T> & a, const char *aname,  int printid);
void PrintSet(const std::set<int> & a, const char *aname,  int printid);
void PrintVector(const Vector & a, const char *aname,  int printid);
void PrintSparseMatrix(const SparseMatrix & a, const char *aname,  int printid);
