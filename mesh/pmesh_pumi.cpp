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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "mesh_headers.hpp"
#include "pmesh_pumi.hpp"
#include "../fem/fem.hpp"
#include "../general/sets.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/text.hpp"

#include <iostream>
using namespace std;

namespace mfem
{

Element *ParPumiMesh::ReadElement( apf::MeshEntity* Ent, const int geom, apf::Downward Verts, 
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
    

ParPumiMesh::ParPumiMesh(MPI_Comm comm, apf::Mesh2* apf_mesh)
{
   //set the comunicator for gtopo 
   gtopo.SetComm(comm);
   
   int i, j;
   Array<int> vert;

   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);
  
   Mesh::SetEmpty();
   //The ncmesh part is deleted
    
   Dim = apf_mesh->getDimension();
   spaceDim = Dim;//mesh.spaceDim;

   //Iterator to get type 
   apf::MeshIterator* itr = apf_mesh->begin(Dim); 
   BaseGeom = apf_mesh->getType( apf_mesh->iterate(itr) );
   apf_mesh->end(itr);
   
   itr = apf_mesh->begin(Dim - 1);
   BaseBdrGeom = apf_mesh->getType( apf_mesh->iterate(itr) );
   apf_mesh->end(itr);

   ncmesh = pncmesh = NULL;
   
   //Global numbering of vertices
   //This is necessary to build a local numbering that
   //has the same ordering in each process
   apf::FieldShape* v_shape = apf::getConstant(0);
   apf::Numbering* vLocNum = apf::createNumbering(apf_mesh, "AuxVertexNumbering", 
           v_shape, 1);
   //Number 
   itr = apf_mesh->begin(0);
   apf::MeshEntity* ent;
   int owned_num = 0;
   int all_num = 0;
   int shared_num = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
       all_num++;
       if (apf_mesh->isOwned(ent)){
         apf::number(vLocNum, ent, 0, 0, owned_num++);
       }  
       if (apf_mesh->isShared(ent))
           shared_num++;
   }
   apf_mesh->end(itr);  
   
   //make it global
   apf::GlobalNumbering* VertexNumbering = apf::makeGlobal(vLocNum, true);
   apf::synchronize(VertexNumbering);
   
   //Take this process global ids and sort
   Array<int> thisIds(all_num);
   Array<int> SharedVertIds(shared_num);
   itr = apf_mesh->begin(0);
   all_num = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
       unsigned int id = apf::getNumber(VertexNumbering, ent, 0, 0);
       thisIds[all_num++] = id;
   }
   apf_mesh->end(itr);
   thisIds.Sort();
   
   //Create local numbering that respects the global ordering
   apf::Field* apf_field_crd = apf_mesh->getCoordinateField();    
   apf::FieldShape* crd_shape = apf::getShape(apf_field_crd);   
   apf::Numbering* v_num_loc = apf::createNumbering(apf_mesh, "LocalVertexNumbering",
           crd_shape, 1);
    
   NumOfVertices = 0;
   shared_num = 0;
   itr = apf_mesh->begin(0);
   while ((ent = apf_mesh->iterate(itr)))
   {
       //Id from global numbering 
       unsigned int id = apf::getNumber(VertexNumbering, ent, 0, 0);
       //Find its position at sorted list 
       int ordered_id = thisIds.Find(id);
       //Assign as local number 
       apf::number(v_num_loc, ent, 0, 0, ordered_id);
       NumOfVertices++;
       
       //add to shared vetrtices list
       if (apf_mesh->isShared(ent))
           SharedVertIds[shared_num++] = ordered_id;
           
   }
   apf_mesh->end(itr);  
   SharedVertIds.Sort();
   apf::destroyGlobalNumbering(VertexNumbering);


   vertices.SetSize(NumOfVertices);
   //set vertices for non-curved mesh
   int curved = (crd_shape->getOrder() > 1) ? 1 : 0;  

   //if (!curved)
   //{ 
     itr = apf_mesh->begin(0);
     while ((ent = apf_mesh->iterate(itr)))
       {
           unsigned int id = apf::getNumber(v_num_loc, ent, 0, 0);
           apf::Vector3 Crds;
           apf_mesh->getPoint(ent,0,Crds);

           for (unsigned int ii=0; ii<spaceDim; ii++)
               vertices[id](ii) = Crds[ii];// !! I am assuming the ids are ordered and from 0
       }
       apf_mesh->end(itr);   
   //}   
   
   //Fill the elements
   NumOfElements = countOwned(apf_mesh,Dim);
   elements.SetSize(NumOfElements);

   //Get the attribute tag
   apf::MeshTag* attTag = apf_mesh->findTag("attribute");
   
   //read elements from SCOREC Mesh
   itr = apf_mesh->begin(Dim);
   j=0;
   while ((ent = apf_mesh->iterate(itr)))
   {
       //Get vertices 
       apf::Downward verts;
       int num_vert =  apf_mesh->getDownward(ent,0,verts);    
       //Get attribute Tag vs Geometry
       int attr = 1;
       /*if (apf_mesh->hasTag(ent,atts)){
          apf_mesh->getIntTag(ent,attTag,&attr);
       }*/
       //apf::ModelEntity* me = apf_mesh->toModel(ent);
       //attr = 1; //apf_mesh->getModelTag(me);
       
       int geom_type = BaseGeom;//apf_mesh->getType(ent); //Make sure this works!!!    
       elements[j] = ReadElement(ent, geom_type, verts, attr, v_num_loc);
       j++;
   }
  //End iterator
   apf_mesh->end(itr);  

   Table *edge_element = NULL;
   /*if (mesh.NURBSext)
   {
      activeBdrElem.SetSize(mesh.GetNBE());
      activeBdrElem = false;
   }*/
   
   //count number of boundaries by classification
   int BcDim = Dim - 1;
   itr = apf_mesh->begin(BcDim);
   NumOfBdrElements = 0; 
    
   while((ent=apf_mesh->iterate(itr)))
    {
        apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
        if(apf_mesh->getModelType(mdEnt) == BcDim){ 
            NumOfBdrElements++;
        }
    }
    apf_mesh->end(itr);   
 
   boundary.SetSize(NumOfBdrElements);
   int bdr_ctr=0;
   //Read boundary from SCOREC mesh
   itr = apf_mesh->begin(BcDim);
   while ((ent = apf_mesh->iterate(itr)))
   {
      //check if this mesh entity is on the model boundary
      apf::ModelEntity* mdEnt = apf_mesh->toModel(ent);
      if (apf_mesh->getModelType(mdEnt) == BcDim)
      {
        apf::Downward verts;
        int num_verts = apf_mesh->getDownward(ent, 0, verts);
        int attr = 1 ;//apf_mesh->getModelTag(mdEnt);
        /*if (apf_mesh->hasTag(ent,atts)){
            apf_mesh->getIntTag(ent,attTag,&attr);
          }*/        

        int geom_type = BaseBdrGeom;//apf_mesh->getType(ent);
        boundary[bdr_ctr] = ReadElement( ent, geom_type, verts, attr, v_num_loc);
        bdr_ctr++;
      }
   }
   apf_mesh->end(itr); 
   
   Mesh::SetMeshGen();
   Mesh::SetAttributes();
  
   // this is called by the default Mesh constructor
   Mesh::InitTables();
   bool refine = false;
   bool fix_orientation = true;  
   this->FinalizeTopology();     
   Mesh::Finalize(refine, fix_orientation);     
   if (Dim > 1)
   {
      el_to_edge = new Table;
      NumOfEdges = Mesh::GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   else
   {
      NumOfEdges = 0;
   }

   STable3D *faces_tbl = NULL;
   if (Dim == 3)
   {
      faces_tbl = GetElementToFaceTable(1);
   }
   else
   {
      NumOfFaces = 0;
   }

   GenerateFaces();
  
   ListOfIntegerSets  groups;
   IntegerSet         group;

   // the first group is the local one
   group.Recreate(1, &MyRank);
   groups.Insert(group);
   
#ifdef MFEM_DEBUG
   if (Dim < 3 && GetNFaces() != 0)
   {
      cerr << "ParMesh::ParMesh (proc " << MyRank << ") : "
           "(Dim < 3 && mesh.GetNFaces() != 0) is true!" << endl;
      mfem_error();
   }
#endif
   
   //determine shared faces
   int sface_counter = 0;
   Array<int> face_group(GetNFaces());
   apf::FieldShape* fc_shape =apf::getConstant(2);
   apf::Numbering* faceNum = apf::createNumbering(apf_mesh, "FaceNumbering",
                               fc_shape, 1);
   Array<int> SharedFaceIds;
   if (Dim > 2)
   {  
    //Number Faces
     apf::Numbering* AuxFaceNum = apf::numberOwnedDimension(apf_mesh, "AuxFaceNumbering", 2);
     apf::GlobalNumbering* GlobalFaceNum = apf::makeGlobal(AuxFaceNum, true);
     apf::synchronize(GlobalFaceNum);

     //Take this process global ids and sort
     Array<int> thisFaceIds(GetNFaces());

     itr = apf_mesh->begin(2);
     all_num = 0;
     shared_num = 0;
     while ((ent = apf_mesh->iterate(itr)))
     {
         unsigned int id = apf::getNumber(GlobalFaceNum, ent, 0, 0);
         thisFaceIds[all_num++] = id;
         if (apf_mesh->isShared(ent))
             shared_num++;
     }
     apf_mesh->end(itr);
     thisFaceIds.Sort();

     //Create local numbering that respects the global ordering
     SharedFaceIds.SetSize(shared_num);
     shared_num = 0;
     itr = apf_mesh->begin(2);
     while ((ent = apf_mesh->iterate(itr)))
     {
         //Id from global numbering 
         unsigned int id = apf::getNumber(GlobalFaceNum, ent, 0, 0);
         //Find its position at sorted list 
         int ordered_id = thisFaceIds.Find(id);
         //Assign as local number 
         apf::number(faceNum, ent, 0, 0, ordered_id);

         if (apf_mesh->isShared(ent))
             SharedFaceIds[shared_num++] = ordered_id;
     }
     apf_mesh->end(itr); 
     SharedFaceIds.Sort();  
     apf::destroyGlobalNumbering(GlobalFaceNum);
   
     itr = apf_mesh->begin(2);
     while ((ent = apf_mesh->iterate(itr)))
      {
        int faceId = apf::getNumber(faceNum, ent, 0, 0);
        face_group[faceId] = -1;
        if (apf_mesh->isShared(ent))
        { 
          //Number of adjacent element 
          int thisNumAdjs = 2;
          int eleRanks[thisNumAdjs];
           
          //Get the Ids
          apf::Parts res;
          apf_mesh->getResidence(ent, res);
          int kk = 0;
          for (std::set<int>::iterator itr = res.begin(); itr != res.end(); ++itr)
           {
               eleRanks[kk++] = *itr;
           }          
          
          group.Recreate(2, eleRanks);
          face_group[faceId] = groups.Insert(group) - 1;
          sface_counter++;
        }
      }
     apf_mesh->end(itr); 

   }

   // determine shared edges
   int sedge_counter = 0;
   if (!edge_element)
   {
       edge_element = new Table;
       if (Dim == 1)
       {
           edge_element->SetDims(0,0);
       }
       else
       {
           edge_element->SetSize(GetNEdges(), 1);         
       }
    }
   
   //Number Edges
   apf::Numbering* AuxEdgeNum = apf::numberOwnedDimension(apf_mesh, "EdgeNumbering", 1);
   apf::GlobalNumbering* GlobalEdgeNum = apf::makeGlobal(AuxEdgeNum, true);
   apf::synchronize(GlobalEdgeNum);
   
   //Take this process global ids and sort
   Array<int> thisEdgeIds(GetNEdges());

   itr = apf_mesh->begin(1);
   all_num = 0;
   shared_num = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
       unsigned int id = apf::getNumber(GlobalEdgeNum, ent, 0, 0);
       thisEdgeIds[all_num++] = id;
       if (apf_mesh->isShared(ent))
           shared_num++;
   }
   apf_mesh->end(itr);
   thisEdgeIds.Sort();
   
   //Create local numbering that respects the global ordering
   apf::FieldShape* ed_shape =apf::getConstant(1);
   apf::Numbering* edgeNum = apf::createNumbering(apf_mesh, "EdgeNumbering",
                             ed_shape, 1);
   
   Array<int> SharedEdgeIds(shared_num); 
   shared_num = 0;
   itr = apf_mesh->begin(1);
   while ((ent = apf_mesh->iterate(itr)))
   {
       //Id from global numbering 
       unsigned int id = apf::getNumber(GlobalEdgeNum, ent, 0, 0);
       //Find its position at sorted list 
       int ordered_id = thisEdgeIds.Find(id);
       //Assign as local number 
       apf::number(edgeNum, ent, 0, 0, ordered_id);
       
       if (apf_mesh->isShared(ent))
           SharedEdgeIds[shared_num++] = ordered_id;
   }
   apf_mesh->end(itr); 
   SharedEdgeIds.Sort();
   apf::destroyGlobalNumbering(GlobalEdgeNum);
   
   itr = apf_mesh->begin(1);
   i = 0;
   while ((ent = apf_mesh->iterate(itr)))
   {
       apf::Downward verts;
       int num_verts = apf_mesh->getDownward(ent,0,verts);
       int ed_ids[2];
       ed_ids[0] = apf::getNumber(v_num_loc, verts[0], 0, 0);
       ed_ids[1] = apf::getNumber(v_num_loc, verts[1], 0, 0);       

       int edId = apf::getNumber(edgeNum, ent, 0, 0);

       edge_element->GetRow(edId)[0] = -1;

       if (apf_mesh->isShared(ent))
       {
           sedge_counter++;
           
           //Number of adjacent element 
           apf::Parts res;
           apf_mesh->getResidence(ent, res);           
           int thisNumAdjs = res.size();
           int eleRanks[thisNumAdjs];
          
           //Get the Ids
           int kk = 0;
           for( std::set<int>::iterator itr = res.begin(); itr != res.end(); itr++)
           {
               eleRanks[kk++] = *itr;
           }

           //Generate the group
           group.Recreate(thisNumAdjs, eleRanks);
           edge_element->GetRow(edId)[0] = groups.Insert(group) - 1;      
           //edge_element->GetRow(i)[0] = groups.Insert(group) - 1;
           
       }
       i++;
   }
   apf_mesh->end(itr);  
          
   //determine shared vertices
   int svert_counter = 0;
   Table *vert_element = new Table;
   vert_element->SetSize(GetNV(), 1);
   
   itr = apf_mesh->begin(0);
   while ((ent = apf_mesh->iterate(itr)))
   {
       int vtId = apf::getNumber(v_num_loc, ent, 0, 0);
       vert_element->GetRow(vtId)[0] = -1;

       
       if (apf_mesh->isShared(ent))
       {
          svert_counter++;
          //Number of adjacent element 
          apf::Parts res;
          apf_mesh->getResidence(ent, res);
          int thisNumAdjs = res.size();          
          int eleRanks[thisNumAdjs];
           
          //Get the Ids
          int kk = 0;
          for (std::set<int>::iterator itr = res.begin(); itr != res.end(); itr++)
           {
               eleRanks[kk++] = *itr;
           }                 
          
          group.Recreate(thisNumAdjs, eleRanks);
          vert_element->GetRow(vtId)[0]= groups.Insert(group) - 1;
       }
   }
   apf_mesh->end(itr);
  
   // build group_sface
   group_sface.MakeI(groups.Size()-1);

   for (i = 0; i < face_group.Size(); i++)
   {
      if (face_group[i] >= 0)
      {
         group_sface.AddAColumnInRow(face_group[i]);
      }
   }
   
   group_sface.MakeJ();

   sface_counter = 0;
   for (i = 0; i < face_group.Size(); i++)
   {
      if (face_group[i] >= 0)
      {
         group_sface.AddConnection(face_group[i], sface_counter++);
      }
   }

   group_sface.ShiftUpI();

   // build group_sedge
   group_sedge.MakeI(groups.Size()-1);

   for (i = 0; i < edge_element->Size(); i++)
   {
      if (edge_element->GetRow(i)[0] >= 0)
      {
         group_sedge.AddAColumnInRow(edge_element->GetRow(i)[0]);
      }
   }
   
   group_sedge.MakeJ(); 
   
   sedge_counter = 0;
   for (i = 0; i < edge_element->Size(); i++)
   {
      if (edge_element->GetRow(i)[0] >= 0)
      {
         group_sedge.AddConnection(edge_element->GetRow(i)[0], sedge_counter++);
      }
   }

   group_sedge.ShiftUpI();      
   
   // build group_svert
   group_svert.MakeI(groups.Size()-1);

   for (i = 0; i < vert_element->Size(); i++)
   {
      if (vert_element->GetRow(i)[0] >= 0)
      {
         group_svert.AddAColumnInRow(vert_element->GetRow(i)[0]);
      }
   }

   group_svert.MakeJ();

   svert_counter = 0;
   for (i = 0; i < vert_element->Size(); i++)
   {
      if (vert_element->GetRow(i)[0] >= 0)
      {
         group_svert.AddConnection(vert_element->GetRow(i)[0], svert_counter++);
      }
   }
   group_svert.ShiftUpI();

   // build shared_faces and sface_lface
   shared_faces.SetSize(sface_counter);
   sface_lface. SetSize(sface_counter);
 
   if (Dim == 3)
   {
      sface_counter = 0;
      itr = apf_mesh->begin(2);
      while ((ent = apf_mesh->iterate(itr)))
      {
          if (apf_mesh->isShared(ent))
          {
              //Generate the face
              int fcId = apf::getNumber(faceNum, ent, 0, 0);
              int ctr = SharedFaceIds.Find(fcId);      

              apf::Downward verts;
              int num_vert =  apf_mesh->getDownward(ent,0,verts);
              int geom = BaseBdrGeom;
              int attr = 1;
              shared_faces[ctr] = ReadElement(ent, geom, verts, attr,
                      v_num_loc);
              
              int *v = shared_faces[ctr]->GetVertices();
              switch ( geom )
              {
                 case Element::TRIANGLE:
                    sface_lface[ctr] = (*faces_tbl)(v[0], v[1], v[2]);
                    //the marking for refinement is omitted. All done in PUMI
                    break;
                  case Element::QUADRILATERAL:
                    sface_lface[ctr] = 
                            (*faces_tbl)(v[0], v[1], v[2], v[3]);
                    break;
              }
              sface_counter++;
          }
      }
      apf_mesh->end(itr);
      delete faces_tbl;
   }
     
   // build shared_edges and sedge_ledge
   shared_edges.SetSize(sedge_counter);
   sedge_ledge. SetSize(sedge_counter);

   {
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);
    
      sedge_counter = 0;
      itr = apf_mesh->begin(1);
      while ((ent = apf_mesh->iterate(itr))) 
      {
          if (apf_mesh->isShared(ent))
          {
              int edId = apf::getNumber(edgeNum, ent, 0, 0);
              int ctr = SharedEdgeIds.Find(edId);
              apf::Downward verts;
              apf_mesh->getDownward(ent, 0, verts);
              int id1, id2;
              id1 = apf::getNumber(v_num_loc, verts[0], 0, 0);
              id2 = apf::getNumber(v_num_loc, verts[1], 0, 0);
              if (id1 > id2) swap(id1,id2);
          
              shared_edges[ctr] = new Segment(id1, id2, 1);
              if ((sedge_ledge[ctr] = v_to_v(id1,id2)) < 0)
              {
               cerr << "\n\n\n" << MyRank << ": ParMesh::ParMesh: "
                    << "ERROR in v_to_v\n\n" << endl;
               mfem_error();                  
              }
              
            sedge_counter++;
          }
      }
   }
   apf_mesh->end(itr);

   delete edge_element;     

   // build svert_lvert
   svert_lvert.SetSize(svert_counter);

   svert_counter = 0;
   itr = apf_mesh->begin(0);
   while ((ent = apf_mesh->iterate(itr)))
   {
       if (apf_mesh->isShared(ent))
       {   
           int vt_id = apf::getNumber(v_num_loc, ent, 0, 0);
           int ctr = SharedVertIds.Find(vt_id);
           svert_lvert[ctr] = vt_id;
           svert_counter++;
       }
   }
   apf_mesh->end(itr);
  
   delete vert_element;
 
   // build the group communication topology
   gtopo.Create(groups, 822); 
   

   if (curved) // curved mesh
   {
      GridFunctionPumi* auxNodes = new GridFunctionPumi(this, apf_mesh, v_num_loc,
              crd_shape->getOrder());
      Nodes = new ParGridFunction(this, auxNodes);     
      Nodes->SetData(auxNodes->GetData());      
      this->edge_vertex = NULL;
      own_nodes = 1;       
   }
   
   //pumi_ghost_delete(apf_mesh);
   apf::destroyNumbering(v_num_loc);
   apf::destroyNumbering(edgeNum);
   apf::destroyNumbering(faceNum);
   have_face_nbr_data = false;
}

}

#endif
