
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


void GetFaceElements(Mesh & mesh, int face, Array<int> & elems);




//         if master == iface and numslaves == 1:
//             # print("Face is neither a slave nor a master")
//             el1, el2 = mesh.GetFaceElements(iface)
//             if el2 == -1 or el1 == -1:
//                 # print("This is a boundary face, skipping ...")
//                 case = -1
//             else:
//                 # print("This is not boundary face ...")
//                 case = 0
//                 if (el1 == tel):
//                     neighbor_elems.Append(el2)
//                 elif (el2 == tel):
//                     neighbor_elems.Append(el1)
//                 else:
//                     assert False, "Check Target element 1"

//                 neighbor_elems.Append(tel)
//                 neighbor_face.Append(iface)
//                 neighbor_face.Append(iface)
//         elif master != iface:
//             el1, el2 = mesh.GetFaceElements(iface)
//             if el2 == -1 or el1 == -1:

//                 print("This is a boundary face, skipping ...This should not happen")
//                 case = -1 # this should not happen
//             else:
//                 case = 1
//                 # print("Face is slave with master face: " , master)
//                 if (el1 == tel):
//                     neighbor_elems.Append(el2)
//                 elif (el2 == tel):
//                     neighbor_elems.Append(el1)
//                 else:
//                     assert False, "Check Target element 2"

//                 neighbor_elems.Append(tel)
//                 neighbor_face.Append(master)
//                 neighbor_face.Append(iface)
//         else:
//             case = 2
//             # print("Face is master with slaves: ")
//             slavesdata = Pt.GetRowColumns(tarr[0])
//             slaves = mfem.intArray(numslaves)
//             slaves.Assign(slavesdata)
//             # print("numslaves = ", numslaves)
//             for j in range (numslaves):
//                 if (slaves[j] == iface):
//                     continue
//                 else:
//                     el1, el2 = mesh.GetFaceElements(slaves[j])
//                     if (el1 == tel):
//                         neighbor_elems.Append(el2)
//                     elif (el2 == tel):
//                         neighbor_elems.Append(el1)
//                     else:
//                         assert False, "Check Target element 1"

//                     neighbor_face.Append(slaves[j])

//             neighbor_elems.Append(tel)
//             neighbor_face.Append(iface)
//     else:
//         el1, el2 = mesh.GetFaceElements(iface)
//         if el2 == -1 or el1 == -1:
//             # print("This is a boundary face, skipping ...")
//             case = -1
//         else:
//             case = 0
//             if (el1 == tel):
//                 neighbor_elems.Append(el2)
//             elif (el2 == tel):
//                 neighbor_elems.Append(el1)
//             else:
//                 assert False, "Check Target element 1"
//             neighbor_elems.Append(tel)
//             neighbor_face.Append(iface)
//             neighbor_face.Append(iface)

//     if (case == -1):
//         assert (neighbor_elems.Size() == 0), "Wrong elem size for case = -1 "
//     elif (case == 0):
//         assert (neighbor_elems.Size() == 2), "Wrong elem size for case = 0 "
//     elif (case == 1):
//         assert (neighbor_elems.Size() == 2), "Wrong elem size for case = 1 "
//     else :
//         assert (neighbor_elems.Size() == 3), "Wrong elem size for case = 2 "

//     return neighbor_elems, neighbor_face, case





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
