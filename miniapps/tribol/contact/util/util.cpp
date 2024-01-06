#include "util.hpp"


void PrintVertex(Mesh * mesh, int vertex)
{
   Array<int> vertices;
   mfem::out << "vertex: " << vertex << ":  ";
   double *coords = mesh->GetVertex(vertex);
   mfem::out << "(" << coords[0] << ", " << coords[1] << ", " << coords[2] << ") \n";
}

void PrintElementVertices(Mesh * mesh, int elem)
{
   Array<int> vertices;
   mfem::out <<  "elem: " << elem << ". Vertices = \n" ;
   mesh->GetElementVertices(elem,vertices);
   for (int i = 0; i<vertices.Size(); i++)
   {
      PrintVertex(mesh,vertices[i]);
   }
   mfem::out << endl;
}

void PrintFaceVertices(Mesh * mesh, int face)
{
   Array<int> vertices;
   mfem::out << "face: " << face << ". Vertices = \n" ;
   mesh->GetFaceVertices(face,vertices);
   for (int i = 0; i<vertices.Size(); i++)
   {
      PrintVertex(mesh,vertices[i]);
   }
   mfem::out << endl;
}

void PrintSet(const std::set<int> & a, const char *aname)
{
   mfem::out << aname << " = " ;
   for (std::set<int>::iterator it = a.begin(); it!= a.end(); it++)
   {
      mfem::out << *it << "  ";
   }
   mfem::out << endl;
}

void PrintVector(const Vector & a, const char *aname)
{
   int sz = a.Size();
   mfem::out << aname << " = " ;
   for (int i = 0; i<sz; i++)
   {
      mfem::out << a[i] << "  ";
   }
   mfem::out << endl;
}

void PrintVertex(Mesh * mesh, int vertex,  int printid)
{
   int myid = Mpi::WorldRank();
   if (myid == printid)
   {
      mfem::out << "myid = " << myid <<":   " <<  "vertex: " << vertex << ":  ";
      double *coords = mesh->GetVertex(vertex);
      mfem::out << "(" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")\n";
   }
}

void PrintElementVertices(Mesh * mesh, int elem,  int printid)
{
   int myid = Mpi::WorldRank();
   Array<int> vertices;
   if (myid == printid)
   {
      mfem::out << "myid = " << myid <<":   " <<  "elem: " << elem <<
                ". Vertices = \n" ;
      mesh->GetElementVertices(elem,vertices);
      for (int i = 0; i<vertices.Size(); i++)
      {
         PrintVertex(mesh,vertices[i],printid);
      }
      mfem::out << endl;
   }
}

void PrintFaceVertices(Mesh * mesh, int face,  int printid)
{
   int myid = Mpi::WorldRank();
   Array<int> vertices;
   if (myid == printid)
   {
      mfem::out << "myid = " << myid <<":   " <<  "face: " << face <<
                ". Vertices = \n" ;
      mesh->GetFaceVertices(face,vertices);
      for (int i = 0; i<vertices.Size(); i++)
      {
         PrintVertex(mesh,vertices[i],printid);
      }
      mfem::out << endl;
   }
}


void PrintSet(const std::set<int> & a, const char *aname,  int printid)
{
   int myid = Mpi::WorldRank();
   if (myid == printid)
   {
      mfem::out << "myid = " << myid <<":   " << aname << " = " ;
      for (std::set<int>::iterator it = a.begin(); it!= a.end(); it++)
      {
         mfem::out << *it << "  ";
      }
      mfem::out << endl;
   }
}

void PrintVector(const Vector & a, const char *aname,  int printid)
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

void PrintVector(const std::vector<int> & a, const char *aname,  int printid)
{
   int myid = Mpi::WorldRank();
   if (myid == printid)
   {
      int sz = a.size();
      mfem::out << "myid = " << myid <<":   " << aname << " = " ;
      for (int i = 0; i<sz; i++)
      {
         mfem::out << a[i] << "  ";
      }
      mfem::out << endl;
   }
}

void PrintVector(const std::vector<unsigned int> & a, const char *aname,  int printid)
{
   int myid = Mpi::WorldRank();
   if (myid == printid)
   {
      int sz = a.size();
      mfem::out << "myid = " << myid <<":   " << aname << " = " ;
      for (int i = 0; i<sz; i++)
      {
         mfem::out << a[i] << "  ";
      }
      mfem::out << endl;
   }
}

void PrintSparseMatrix(const SparseMatrix & a, const char *aname,  int printid)
{
   int myid = Mpi::WorldRank();
   if (myid == printid)
   {
      mfem::out << "myid = " << myid <<":   " << aname << " = " ;
      a.PrintMatlab(mfem::out);
   }
   mfem::out << endl;
}