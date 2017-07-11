// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of data type mesh

#include "mesh_headers.hpp"
#include "../fem/fem.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/text.hpp"
#include "mesh_pumi.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cmath>
#include <cstring>
#include <ctime>


using namespace std;

namespace mfem
{

PumiMesh::PumiMesh(apf::Mesh2* apf_mesh, int generate_edges, int refine,
        bool fix_orientation)
{
    SetEmpty();
    Load(apf_mesh, generate_edges, refine, fix_orientation);
}

Element *PumiMesh::ReadElement( apf::MeshEntity* Ent, const int geom, apf::Downward Verts, 
               const int Attr, apf::Numbering* vert_num)
{
    Element *el;
    int nv, *v;
    
    //Create element in MFEM
    el = NewElement(geom);
    nv = el->GetNVertices();
    v  = el->GetVertices();
    
    //Fill the connectivity
    for(int i = 0; i < nv; ++i)
    {
        v[i] = apf::getNumber(vert_num, Verts[i], 0, 0);
    }
    
    //Assign attribute
    el->SetAttribute(Attr);
    
    return el;
}

void PumiMesh::CountBoundaryEntity( apf::Mesh2* apf_mesh, const int BcDim, int &NumBc)
{
    apf::MeshEntity* ent;
    apf::MeshIterator* itr = apf_mesh->begin(BcDim);
    
    while((ent=apf_mesh->iterate(itr)))
    {
        apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
        if(apf_mesh->getModelType(mdEnt) == BcDim){ 
            NumBc++;
        }
    }
    apf_mesh->end(itr);
            
    //Check if any boundary is detected
    if (NumBc==0)
       MFEM_ABORT("In CountBoundaryEntity; no boundary is detected!"); 
}

void PumiMesh::Load(apf::Mesh2* apf_mesh, int generate_edges, int refine,
                bool fix_orientation)
{
  int  curved = 0, read_gf = 1;
   
  //Add a check on apf_mesh just in case
  Clear();

   //First number vertices
   apf::Field* apf_field_crd = apf_mesh->getCoordinateField();    
   apf::FieldShape* crd_shape = apf::getShape(apf_field_crd);
   apf::Numbering* v_num_loc = apf::createNumbering(apf_mesh, "VertexNumbering",
                             crd_shape, 1);
   //check if it is a curved mesh
   curved = (crd_shape->getOrder() > 1) ? 1 : 0; 
   
  //read mesh 
  ReadSCORECMesh(apf_mesh, v_num_loc, curved); 
  cout<< "After ReadSCORECMesh" <<endl;
   // at this point the following should be defined:
   //  1) Dim
   //  2) NumOfElements, elements
   //  3) NumOfBdrElements, boundary
   //  4) NumOfVertices, with allocated space in vertices
   //  5) curved
   //  5a) if curved == 0, vertices must be defined
   //  5b) if curved != 0 and read_gf != 0,
   //         'input' must point to a GridFunction
   //  5c) if curved != 0 and read_gf == 0,
   //         vertices and Nodes must be defined

   // FinalizeTopology() will:
   // - assume that generate_edges is true
   // - assume that refine is false
   // - does not check the orientation of regular and boundary elements
   FinalizeTopology();

   if (curved && read_gf)
   {
      // Check it to be only Quadratic if higher order
      cout << "Is Curved?: "<< curved << "\n" <<read_gf <<endl;
      Nodes = new GridFunctionPumi(this, apf_mesh, v_num_loc, crd_shape->getOrder());
      edge_vertex = NULL;
      own_nodes = 1;
      spaceDim = Nodes->VectorDim();
      //if (ncmesh) { ncmesh->spaceDim = spaceDim; }
      // Set the 'vertices' from the 'Nodes'
      for (int i = 0; i < spaceDim; i++)
      {
         Vector vert_val;
         Nodes->GetNodalValues(vert_val, i+1);
         for (int j = 0; j < NumOfVertices; j++)
         {
            vertices[j](i) = vert_val(j);
         }
      }
   }
  
   //Delete numbering
   apf::destroyNumbering(v_num_loc);  
   
   Finalize(refine, fix_orientation);
}

void PumiMesh::ReadSCORECMesh(apf::Mesh2* apf_mesh, apf::Numbering* v_num_loc, 
        const int curved)
{
    /*
     /Here fill the element table from SCOREC MESH
     /The vector of element pointers are generated with attr and connectivity
     */

   apf::MeshIterator* itr = apf_mesh->begin(0);
   apf::MeshEntity* ent;
   NumOfVertices = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
       //ids start from 0
       apf::number(v_num_loc, ent, 0, 0, NumOfVertices);
       NumOfVertices++;
   }
   apf_mesh->end(itr);
    
   Dim = apf_mesh->getDimension();
   NumOfElements = countOwned(apf_mesh,Dim);
   elements.SetSize(NumOfElements);

   //Get the attribute tag
   apf::MeshTag* attTag = apf_mesh->findTag("attribute");
   
   //read elements from SCOREC Mesh
   itr = apf_mesh->begin(Dim);
   unsigned int j=0;
   while ((ent = apf_mesh->iterate(itr)))
   {
       //Get vertices 
       apf::Downward verts;
       int num_vert =  apf_mesh->getDownward(ent,0,verts);    
       //Get attribute Tag vs Geometry
       int attr = 1;
       /*if (apf_mesh->hasTag(ent,atts)){
           attr = apf_mesh->getIntTag(ent,attTag,&attr);
       }*/
       apf::ModelEntity* me = apf_mesh->toModel(ent);
       attr = 1; //apf_mesh->getModelTag(me);
       int geom_type = apf_mesh->getType(ent); //Make sure this works!!!    
       elements[j] = ReadElement(ent, geom_type, verts, attr, v_num_loc);
       j++;
   }
  //End iterator
   apf_mesh->end(itr);
   
  //Read Boundaries from SCOREC Mesh
  //First we need to count them         
  int BCdim = Dim - 1;
  NumOfBdrElements = 0;
  CountBoundaryEntity(apf_mesh, BCdim, NumOfBdrElements);
  boundary.SetSize(NumOfBdrElements);
  j=0;

  //Read boundary from SCOREC mesh
  itr = apf_mesh->begin(BCdim);
  while ((ent = apf_mesh->iterate(itr)))
  {
      //check if this mesh entity is on the model boundary
      apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
      if (apf_mesh->getModelType(mdEnt) == BCdim)
      {
          apf::Downward verts;
          int num_verts = apf_mesh->getDownward(ent, 0, verts);
          int attr = 1 ;//apf_mesh->getModelTag(mdEnt);
          int geom_type = apf_mesh->getType(ent);
          boundary[j] = ReadElement( ent, geom_type, verts, attr, v_num_loc);
          j++;
      }
  }
  apf_mesh->end(itr);

  //Fill vertices
  vertices.SetSize(NumOfVertices);


  if (!curved)
  { 
    apf::MeshIterator* itr = apf_mesh->begin(0);
    spaceDim = Dim;

    while ((ent = apf_mesh->iterate(itr)))
    {
        unsigned int id = apf::getNumber(v_num_loc, ent, 0, 0);
        apf::Vector3 Crds;
        apf_mesh->getPoint(ent,0,Crds);
        
        for (unsigned int ii=0; ii<spaceDim; ii++)
            vertices[id](ii) = Crds[ii];
    }
    apf_mesh->end(itr); 

    // initialize vertex positions in NCMesh
    //if (ncmesh) { ncmesh->SetVertexPositions(vertices); }       
  }
}

}
