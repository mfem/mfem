
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


void GetFaceElements(Mesh & mesh, int face, Array<int> & elems, bool vis=false);
void GetFaceElements2(Mesh & mesh, int face, Array<int> & elems);
void GetNCMeshAllFaceElements(Mesh & mesh, Array<Array<int>*> & FaceElements);
void GetNCMeshFaceElements(Mesh & mesh, int face, Array<int> & FaceElements);

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
   ref_elems = 5;
   mesh.GeneralRefinement(ref_elems,-1,0);

   // const Table & e2e = mesh.ElementToElementTable();

   // e2e.Print();

   // face 29: (6,11), elems: 2 7 8 9
   // face 10: (6,27), elems: 2 9
   // face 26: (27,34), elems: 7 9
   // face 28: (11,34), elems: 8 9

   Array<int> faces({0,10,26,28,29});

   // for (int i = 0; i<mesh.GetNEdges(); i++)
   // for (int i = 0; i<faces.Size(); i++)
   // {
   //    int face = faces[i];
   //    Array<int> elems;
   //    GetFaceElements2(mesh,face,elems);
   //    cout << "Elems 2 = " ; elems.Print();
   //    cout << endl;
   //    GetFaceElements(mesh,face,elems);
   //    // cout << "Elems 2 = " ; elems.Print();
   //    // Array<int> vert;
   //    // mesh.GetEdgeVertices(face,vert);
   //    // cout << "Face: " << face  <<": vertices:  (" << vert[0] <<"," << vert[1] <<
   //       //   "), elems: " ; elems.Print();
   // }

   Array<Array<int> *> FaceElements;

   GetNCMeshAllFaceElements(mesh,FaceElements);
   for (int i = 0; i<FaceElements.Size(); i++)
      // for (int i = 29; i<30; i++)
   {
      cout << "0: face: " << i << ", elements: "; FaceElements[i]->Print();
   }


   // for (int i = 0; i<mesh.GetNumFaces(); i++)
   for (int i = 0; i<mesh.GetNumFaces(); i++)
      // for (int i = 29; i<30; i++)
   {
      // cout << "1: face: " << i ;
      Array<int> elems;
      GetNCMeshFaceElements(mesh,i,elems);
      // cout << ", elements: "; elems.Print();
   }

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "mesh\n" << mesh << flush;

   return 0;
}


void GetFaceElements(Mesh & mesh, int face, Array<int> & elems, bool vis)
{
   int dim = mesh.Dimension();
   FiniteElementCollection * fec = nullptr;
   if (dim == 2)
   {
      // fec = new ND_FECollection(1,dim);
      fec = new RT_FECollection(0,dim);
   }
   else
   {
      // not yet tested in 3D
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

   if (vis)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream patch_sock(vishost, visport);
      L2_FECollection fec(1,mesh.Dimension());
      FiniteElementSpace fes(&mesh,&fec);
      GridFunction vis_gf(&fes);
      vis_gf = 0.0;
      Array<int> dofs;
      for (int i=0; i<elems.Size(); i++)
      {
         int el = elems[i];
         fes.GetElementDofs(el, dofs);
         vis_gf.SetSubVector(dofs,1.0);
      }
      patch_sock.precision(8);
      patch_sock << "solution\n" << mesh << vis_gf <<
                 "keys rRmjnppppp\n" << flush;
   }
}


void GetFaceElements2(Mesh & mesh, int face, Array<int> & elems)
{
   Table * f2e = mesh.GetFaceToElementTable();
   int n = f2e->RowSize(face);
   int * el = f2e->GetRow(face);
   elems.SetSize(n);
   for (int i = 0; i<n; i++)
   {
      elems[i] = el[i];
   }
   int inf1, inf2;
   int el1, el2;
   mesh.GetFaceElements(face, &el1, &el2);
   mesh.GetFaceInfos(face, &inf1, &inf2);
   // cout << "face: " << face << ", el1: " << el1 <<", el2: " << el2 << endl;
   // cout << "face: " << face << ", inf1: " << inf1 <<", inf2: " << inf2 << endl;
   const mfem::NCMesh::NCList & l = mesh.ncmesh->GetNCList(1);

   int k = 3;
   int faceId = l.masters[k].index;
   cout << "master "<< k <<" face no = " << faceId << endl;
   cout << "master "<< k <<" element = " << l.masters[k].element << endl;

   Array<int> vert;
   mesh.GetEdgeVertices(faceId,vert);
   cout << "Face: " << faceId  <<", vertices:  (" << vert[0] <<"," << vert[1] <<
        ") " << endl;
   // cout << "Face: " << l.masters[k].index  <<": vertices:  (" << vert[0] <<"," << vert[1] <<
   //   "), elems: " ; elems.Print();


   cout << "l master size = " << l.masters.Size() << endl;
   cout << "l slaves size = " << l.slaves.Size() << endl;
   cout << "slave begin = " << l.masters[k].slaves_begin << endl;
   cout << "slave end = " << l.masters[k].slaves_end << endl;
   cout << "slaves = " << endl;
   for (int i = l.masters[k].slaves_begin; i<l.masters[k].slaves_end ; i++)
   {
      int slaveId = l.slaves[i].index;
      mesh.GetEdgeVertices(slaveId,vert);

      cout << "slave = " << slaveId << ", vertices:  (" << vert[0] <<"," << vert[1] <<
           ") " << endl;
   }
}



void GetNCMeshAllFaceElements(Mesh & mesh, Array<Array<int>*> & FaceElements)
{
   int nfaces = mesh.GetNumFaces();
   FaceElements.SetSize(nfaces);

   for (int i = 0; i<nfaces; i++)
   {
      FaceElements[i] = new Array<int>();
   }
   int dim = mesh.Dimension();
   MFEM_VERIFY(dim>1, "1D meshes not supported");
   int entity = (dim == 2) ? 1 : 2;

   const mfem::NCMesh::NCList & l = mesh.ncmesh->GetNCList(entity);

   int nfaces_c = l.conforming.Size();
   // conforming faces
   int el1, el2;
   for (int i = 0; i<nfaces_c; i++)
   {
      int face = l.conforming[i].index;
      mesh.GetFaceElements(face, &el1, &el2);
      if (el1 != -1) { FaceElements[face]->Append(el1); }
      if (el2 != -1) { FaceElements[face]->Append(el2); }
   }
   // slaves
   int nfaces_s = l.slaves.Size();
   for (int i = 0; i<nfaces_s; i++)
   {
      int face = l.slaves[i].index;
      mesh.GetFaceElements(face, &el1, &el2);
      if (el1 != -1) { FaceElements[face]->Append(el1); }
      if (el2 != -1) { FaceElements[face]->Append(el2); }
   }

   // masters
   int nfaces_m = l.masters.Size();
   for (int i = 0; i<nfaces_m; i++)
   {
      int face = l.masters[i].index;

      // loop through its slaves
      for (int j = l.masters[i].slaves_begin; j<l.masters[i].slaves_end ; j++)
      {
         int face_s = l.slaves[j].index;

         mesh.GetFaceElements(face_s, &el1, &el2);
         if (el1 != -1) { FaceElements[face]->Append(el1); }
         if (el2 != -1) { FaceElements[face]->Append(el2); }
      }
      FaceElements[face]->Sort();
      FaceElements[face]->Unique();
   }

}



void GetNCMeshFaceElements(Mesh & mesh, int face, Array<int> & elems)
{

   int dim = mesh.Dimension();
   MFEM_VERIFY(dim>1, "1D meshes not supported");
   int entity = (dim == 2) ? 1 : 2;
   const mfem::NCMesh::NCList & l = mesh.ncmesh->GetNCList(entity);



   int inf1, inf2, ncface;

   mesh.GetFaceInfos(face, &inf1, &inf2, &ncface);
   cout << "face: " << face << endl;
   cout << "inf1: " << inf1 << endl;
   cout << "inf2: " << inf2 << endl;
   cout << "ncface: " << ncface << endl;


   // int type;
   // mfem::NCMesh::MeshId lm = l.LookUp(face,&type);


   // int el1, el2;
   // int nc_element;
   // switch (type)
   // { // case of conforming or slave
   // case 0:
   // case 2:
   //    mesh.GetFaceElements(face, &el1, &el2);
   //    if (el1 != -1) elems.Append(el1);
   //    if (el2 != -1) elems.Append(el2);
   //    break;
   //    // case of a master face
   // default:
   // nc_element = lm.element;
   // cout << "face :" << face << endl;
   // cout << " lm.index: " << lm.index << endl;
   // cout << " lm.element: " << lm.element << endl;
   // cout << " lm.local: " << (int)lm.local << endl;



   // for (int j = l.masters[nc_element].slaves_begin; j<l.masters[nc_element].slaves_end ; j++)
   // {
   //    int face_s = l.slaves[j].index;

   //    mesh.GetFaceElements(face_s, &el1, &el2);
   //    if (el1 != -1) elems.Append(el1);
   //    if (el2 != -1) elems.Append(el2);
   // }
   // elems.Sort();
   // elems.Unique();
   // break;
   // }
}