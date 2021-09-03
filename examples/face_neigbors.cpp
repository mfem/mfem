
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


void GetFaceElements(Mesh & mesh, int face, Array<int> & elems);


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   Array<int> ref_elems(1);
   ref_elems = 2;
   mesh.GeneralRefinement(ref_elems,-1,0);

   const Table & e2e = mesh.ElementToElementTable();

   e2e.Print();

   for (int i = 0; i<mesh.GetNEdges(); i++)
   {
      Array<int> elems;
      GetFaceElements(mesh,i,elems);
      Array<int> vert;
      mesh.GetEdgeVertices(i,vert);
      cout << "Face: " << i  <<": vertices:  (" << vert[0] <<"," << vert[1] <<
           "), elems: " ; elems.Print();
   }


   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "mesh\n" << mesh << flush;



   return 0;
}


void GetFaceElements(Mesh & mesh, int face, Array<int> & elems)
{
   // not fully correct yet
   int dim = mesh.Dimension();
   FiniteElementCollection * fec = nullptr;
   if (dim == 2)
   {
      fec = new ND_FECollection(1,dim);
   }
   else
   {
      fec = new RT_FECollection(0,dim);
   }
   FiniteElementSpace fespace(&mesh, fec);

   const SparseMatrix * P = fespace.GetConformingProlongation();
   const SparseMatrix * R = fespace.GetConformingRestriction();
   int el1, el2;
   if (P)
   {
      SparseMatrix * Pt = Transpose(*P);
      const int * col = P->GetRowColumns(face);
      int numslaves = Pt->RowSize(col[0]);
      const int * master = R->GetRowColumns(col[0]);

      if ((master[0] == face && numslaves == 1) ||
          master[0] != face)
      {
         mesh.GetFaceElements(face,&el1,&el2);
         if (el2 == -1 || el1 == -1)
         {
            // cout << "This is a boundary face," << endl;
            int el = (el1 == -1)? el2 : el1;
            elems.Append(el);
         }
         else
         {
            // cout << "This is not boundary face ..." << endl;
            elems.Append(el1);
            elems.Append(el2);
         }
      }
      else
      {
         MFEM_VERIFY(numslaves > 1, "Check numslaves");
         const int * slaves = Pt->GetRowColumns(col[0]);
         // cout << "numslaves = " << numslaves << endl;
         for (int j=0; j<numslaves; j++)
         {
            mesh.GetFaceElements(slaves[j],&el1,&el2);
            if (el1 != -1) { elems.Append(el1); }
            if (el2 != -1) { elems.Append(el2); }
         }
      }
   }
   else
   {
      mesh.GetFaceElements(face,&el1,&el2);
   }
   elems.Sort();
   elems.Unique();
}
