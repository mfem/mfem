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
      surface
   };
private:
   const Mesh *mesh0=nullptr;
   int dim, sdim;
   const FiniteElementSpace *fes0=nullptr;
   DenseMatrix vcoords;
   Mesh *mesh=nullptr; // Submesh 
   Mesh *surface_mesh=nullptr; //Surface mesh 
   FiniteElementSpace *fes=nullptr; // FE Space on the submesh
   FiniteElementSpace *surface_fes=nullptr; // FE Space on the submesh

   Array<int> element_map, surface_element_map;
   Array<int> dof_map, surface_dof_map;
   SparseMatrix * P=nullptr; 
   SparseMatrix * Pf=nullptr; 
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
   Mesh * GetSurfaceMesh(const Array<int> & surface_elems)
   {
      if (!surface_mesh) BuildSubMesh(surface_elems, entity_type::surface);
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
   void GetFaceElementMap(Array<int> & surface_element_map_)
   {
      surface_element_map_ = surface_element_map;
   }
   void GetDofMap(Array<int> & dof_map_)
   {
      if (!dof_map.Size()) BuildDofMap(entity_type::volume);
      dof_map_ = dof_map;
   }
   void GetSurfaceDofMap(Array<int> & surface_dof_map_)
   {
      if (!surface_dof_map.Size()) BuildDofMap(entity_type::surface);
      surface_dof_map_ = surface_dof_map;
   }
   SparseMatrix * GetProlonationMatrix()
   {
      if (!P) BuildProlongationMatrix(entity_type::volume);
      return P;
   }
   SparseMatrix * GetSurfaceProlonationMatrix()
   {
      if (!Pf) BuildProlongationMatrix(entity_type::surface);
      return Pf;
   }
   FiniteElementSpace * GetSubFESpace(const entity_type & etype)
   {
      switch (etype)
      {
      case 0: 
         if (!fes)
         {
            MFEM_VERIFY(mesh, "Volume mesh not built");
            BuildDofMap(etype);
         }
         return fes; 
         break;
      case 1: 
         if (!surface_fes)
         {
            MFEM_VERIFY(surface_mesh, "Surface mesh not built");
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