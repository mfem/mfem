#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;


class Subdomain
{
public:
   enum entity_type
   {
      volume, 
      bdr,
      face
   };
private:
   const Mesh *mesh0=nullptr;
   int dim, sdim;
   const FiniteElementSpace *fes0=nullptr;
   DenseMatrix vcoords;
   Mesh *mesh=nullptr; // Submesh 
   Mesh *bdr_mesh=nullptr; //bdr mesh 
   Mesh *surface_mesh=nullptr; //Surface mesh 
   FiniteElementSpace *fes=nullptr; // FE Space on the submesh
   FiniteElementSpace *bdr_fes=nullptr; // FE Space on the submesh
   FiniteElementSpace *surface_fes=nullptr; // FE Space on the submesh

   Array<int> element_map, bdr_element_map, face_element_map;
   Array<int> dof_map, bdr_dof_map, face_dof_map;
   SparseMatrix * P=nullptr; // element to element prolongation (dofs)
   SparseMatrix * Pb=nullptr; // bdr element to bdr element prolongation (dofs)
   SparseMatrix * Pf=nullptr; // face element to face element prolongation (dofs)
   void BuildDofMap(const entity_type & etype);
   void BuildProlongationMatrix(const entity_type & etype);
   void BuildSubMesh(const Array<int> & elems, const entity_type & etype);

public:
   Subdomain(const Mesh & mesh_);
   Mesh * GetSubMesh(const Array<int> & elems)
   {
      if(!mesh) BuildSubMesh(elems, entity_type::volume);
      return mesh; 
   }
   Mesh * GetBdrSurfaceMesh(const Array<int> & bdr_elems)
   {
      if (!bdr_mesh) BuildSubMesh(bdr_elems, entity_type::bdr);
      return bdr_mesh;
   }
   Mesh * GetSurfaceMesh(const Array<int> & face_elems)
   {
      if (!surface_mesh) BuildSubMesh(face_elems, entity_type::face);
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
   void GetBdrElementMap(Array<int> & bdr_element_map_)
   {
      bdr_element_map_ = bdr_element_map;
   }
   void GetFaceElementMap(Array<int> & face_element_map_)
   {
      face_element_map_ = face_element_map;
   }
   void GetDofMap(Array<int> & dof_map_)
   {
      if (!dof_map.Size()) BuildDofMap(entity_type::volume);
      dof_map_ = dof_map;
   }
   void GetBdrDofMap(Array<int> & bdr_dof_map_)
   {
      if (!bdr_dof_map.Size()) BuildDofMap(entity_type::bdr);
      bdr_dof_map_ = bdr_dof_map;
   }
   void GetFaceDofMap(Array<int> & face_dof_map_)
   {
      if (!face_dof_map.Size()) BuildDofMap(entity_type::face);
      face_dof_map_ = face_dof_map;
   }
   SparseMatrix * GetProlonationMatrix()
   {
      if (!P) BuildProlongationMatrix(entity_type::volume);
      return P;
   }
   SparseMatrix * GetBdrProlonationMatrix()
   {
      if (!Pb) BuildProlongationMatrix(entity_type::bdr);
      return Pb;
   }
   SparseMatrix * GetFaceProlonationMatrix()
   {
      if (!Pf) BuildProlongationMatrix(entity_type::face);
      return Pf;
   }
   FiniteElementSpace * GetSubFESpace(const entity_type & etype)
   {
      const FiniteElementCollection *fec = fes0->FEColl();
      switch (etype)
      {
      case 0: 
         if (!fes)
         {
            MFEM_VERIFY(mesh, "Submesh not built");
            BuildDofMap(etype);
         }
         return fes; 
         break;
      case 1: 
         if (!bdr_fes)
         {
            MFEM_VERIFY(bdr_mesh, "SubBdr mesh not built");
            BuildDofMap(etype);
         }
         return bdr_fes; 
         break;
      case 2: 
         if (!surface_fes)
         {
            MFEM_VERIFY(surface_mesh, "SubSurface mesh not built");
            BuildDofMap(etype);
         }
         return surface_fes; 
         break;
      default: 
         MFEM_ABORT("Wrong entity type"); 
         return nullptr; 
         break;
      }
   }
   ~Subdomain();
};

// void AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,int * ind);
// void GetNumVertices(int type, mfem::Element::Type & elem_type, int & nrvert);
// void PrintElementMap();