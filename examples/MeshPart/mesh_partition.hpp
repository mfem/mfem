#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;


class Subdomain
{
private:
   const Mesh *mesh0=nullptr;
   int dim, sdim;
   const FiniteElementSpace *fes0=nullptr;
   Mesh *mesh=nullptr; // Submesh 
   Mesh *bdr_mesh=nullptr; //bdr mesh 
   Mesh *surface_mesh=nullptr; //Surface mesh 
   FiniteElementSpace *fes=nullptr; // FE Space on the submesh
   FiniteElementSpace *bdr_fes=nullptr; // FE Space on the submesh
   FiniteElementSpace *surface_fes=nullptr; // FE Space on the submesh

   Array<int> element_map;
   Array<int> dof_map;
   SparseMatrix * P=nullptr;
   void BuildDofMap();
   void BuildProlongationMatrix();
   void BuildSubMesh(const Array<int> & elems);

   void BuildBdrSurfaceMesh(const Array<int> & bdr_elements);
   
   void BuildSurfaceMesh(const Array<int> & face_elements);

public:
   Subdomain(const Mesh & mesh_);
   Mesh * GetSubMesh(const Array<int> & elems)
   {
      if(!mesh) BuildSubMesh(elems);
      return mesh; 
   }
   Mesh * GetBdrSurfaceMesh(const Array<int> & bdr_elems)
   {
      if (!bdr_mesh) BuildBdrSurfaceMesh(bdr_elems);
      return bdr_mesh;
   }
   Mesh * GetSurfaceMesh(const Array<int> & face_elems)
   {
      if (!surface_mesh) BuildSurfaceMesh(face_elems);
      return surface_mesh;
   }

   void SetFESpace(const FiniteElementSpace & fes0_)
   {
      fes0 = &fes0_;
   }
   void GetElementMap(Array<int> & element_map_)
   {
      element_map_ = element_map;
   }
   void GetDofMap(Array<int> & dof_map_)
   {
      if (!dof_map.Size()) BuildDofMap();
      dof_map_ = dof_map;
   }
   SparseMatrix * GetProlonationMatrix()
   {
      if (!P) BuildProlongationMatrix();
      return P;
   }
   ~Subdomain(){ };
};

// void AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,int * ind);
// void GetNumVertices(int type, mfem::Element::Type & elem_type, int & nrvert);
// void PrintElementMap();