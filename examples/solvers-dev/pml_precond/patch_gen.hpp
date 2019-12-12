#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;


class mesh_partition // for now every vertex defines a patch 
{
   Mesh *mesh=nullptr;
   void AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,int * ind);
   void print_element_map();
   void save_mesh_partition();
public:
   int nrpatch;
   Array<Array<int>> element_map; // map local (patch) element to global (original mesh) element
   Array<Mesh *> patch_mesh;
   // constructor
   mesh_partition(Mesh * mesh_);
   ~mesh_partition();
};


class patch_assembly // for now every vertex defines a patch 
{
   FiniteElementSpace *fespace=nullptr;
   void print_patch_dof_map();
public:
   int nrpatch;
   Array<FiniteElementSpace *> patch_fespaces;
   Array<Array<int>> patch_dof_map;
   // constructor
   patch_assembly(FiniteElementSpace * fespace_);
   ~patch_assembly();
};

