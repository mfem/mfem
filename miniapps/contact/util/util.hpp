#include "mfem.hpp"

using namespace std;
using namespace mfem;

void PrintVertex(Mesh * mesh, int vertex);
void PrintElementVertices(Mesh * mesh, int elem);
void PrintFaceVertices(Mesh * mesh, int face);
template <class T>
void PrintArray(const Array<T> & a, const char *aname)
{
   int sz = a.Size();
   mfem::out << aname << " = " ;
   for (int i = 0; i<sz; i++)
   {
      mfem::out << a[i] << "  ";
   }
   mfem::out << endl;
}
void PrintSet(const std::set<int> & a, const char *aname);
void PrintVector(const Vector & a, const char *aname);

// for parallel 
void PrintVertex(Mesh * mesh, int vertex,  int printid);
void PrintElementVertices(Mesh * mesh, int elem,  int printid);
void PrintFaceVertices(Mesh * mesh, int face,  int printid);
template <class T>
void PrintArray(const Array<T> & a, const char *aname,  int printid)
{
   int myid = Mpi::WorldRank();
   if (myid == printid)
   {
      int sz = a.Size();
      mfem::out << "myid = " << myid <<":   " << aname << " = " ;
      for (int i = 0; i<sz; i++)
      {
         mfem::out << a[i] << "  ";
      }
      mfem::out << endl;
   }
}
void PrintSet(const std::set<int> & a, const char *aname,  int printid);
void PrintVector(const Vector & a, const char *aname,  int printid);
void PrintVector(const std::vector<int> & a, const char *aname,  int printid);
void PrintVector(const std::vector<unsigned int> & a, const char *aname,  int printid);
void PrintSparseMatrix(const SparseMatrix & a, const char *aname,  int printid);
