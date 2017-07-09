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

#include "pentsets.hpp"
#include "pmesh.hpp"

using namespace std;

namespace mfem
{

ParEntitySets::ParEntitySets(const ParEntitySets & ent_sets)
   : EntitySets(ent_sets),
     pmesh_(ent_sets.GetParMesh())
{
   MPI_Comm_size(pmesh_->GetComm(), &NRanks_);
   MPI_Comm_rank(pmesh_->GetComm(), &MyRank_);
}

ParEntitySets::ParEntitySets(ParMesh & pmesh, const EntitySets & ent_sets,
                             int * partitioning,
                             const Array<int> & vert_global_local)
   : EntitySets(ent_sets),
     pmesh_(&pmesh)
{
   // The copy constructor for EntitySets will initialize this object's
   // data with the correct set names, and numbers of sets.  However,
   // the set entries themselves will need to be recomputed based on
   // local numberings and the paritioning.
   //
   // The EntitySets object will be a copy of the serial object.  This
   // constructor will have to prune and renumber the data.  Once this
   // is done the mesh pointer stored in the EntitySets object can be
   // replaced with the local portion of the parallel mesh.

   MPI_Comm MyComm = pmesh_->GetComm();

   MPI_Comm_size(MyComm, &NRanks_);
   MPI_Comm_rank(MyComm, &MyRank_);

   int nelem = mesh_->GetNE();

   DSTable v_to_v(vert_global_local.Size());
   pmesh_->GetVertexToVertexTable(v_to_v);

   STable3D * faces_tbl = NULL;

   const Table * serial_edge_vertex = NULL;
   const Table * serial_face_vertex = NULL;

   if ( ent_sets.GetNumSets(EDGE) > 0 )
   {
      serial_edge_vertex = ent_sets.GetEdgeVertexTable();
   }
   if ( ent_sets.GetNumSets(FACE) > 0 )
   {
      serial_face_vertex = ent_sets.GetFaceVertexTable();
      faces_tbl = pmesh_->GetFacesTable();
   }

   Array<int> elem_global_local(nelem);
   elem_global_local = -1;
   int elem_counter = 0;
   for (int i=0; i<nelem; i++)
   {
      if ( partitioning[i] == MyRank_ )
      {
         elem_global_local[i] = elem_counter;
         elem_counter++;
      }
   }

   EntityType t;
   unsigned int ns;

   t = VERTEX;
   ns = ent_sets.GetNumSets(t);
   for (unsigned int s=0; s<ns; s++)
   {
      set<int>::iterator it;
      sets_[t][s].clear();
      for (it=ent_sets(t,s).begin(); it!=ent_sets(t,s).end(); it++)
      {
         int v0 = vert_global_local[*it];
         if ( v0 >= 0 )
         {
            sets_[t][s].insert(v0);
         }
      }
   }

   if ( pmesh_->Dimension() > 1 )
   {
      t = EDGE;
      ns = ent_sets.GetNumSets(t);
      for (unsigned int s=0; s<ns; s++)
      {
         set<int>::iterator it;
         sets_[t][s].clear();
         for (it=ent_sets(t,s).begin(); it!=ent_sets(t,s).end(); it++)
         {
            int old_edge = *it;
            const int *v = serial_edge_vertex->GetRow(old_edge);
            int v0 = vert_global_local[v[0]];
            int v1 = vert_global_local[v[1]];
            if ( v0 >= 0 && v1 >= 0 )
            {
               int new_edge = v_to_v(v0,v1);
               if ( new_edge >= 0 )
               {
                  sets_[t][s].insert(new_edge);
               }
            }
         }
      }
   }

   if ( pmesh_->Dimension() > 2 )
   {
      Array<int> v;
      t = FACE;
      ns = ent_sets.GetNumSets(t);
      for (unsigned int s=0; s<ns; s++)
      {
         set<int>::iterator it;
         sets_[t][s].clear();
         for (it=ent_sets(t,s).begin(); it!=ent_sets(t,s).end(); it++)
         {
            int old_face = *it;
            int numv = serial_face_vertex->RowSize(old_face);
            const int *v = serial_face_vertex->GetRow(old_face);
            if ( vert_global_local[v[0]] >= 0 &&
                 vert_global_local[v[1]] >= 0 &&
                 vert_global_local[v[2]] >= 0 )
            {
               int new_face = -1;
               if ( numv == 3 )
               {
                  new_face = (*faces_tbl)(vert_global_local[v[0]],
                                          vert_global_local[v[1]],
                                          vert_global_local[v[2]]);
               }
               else
               {
                  new_face = (*faces_tbl)(vert_global_local[v[0]],
                                          vert_global_local[v[1]],
                                          vert_global_local[v[2]],
                                          vert_global_local[v[3]]);
               }
               if ( new_face >= 0 )
               {
                  sets_[t][s].insert(new_face);
               }
            }
         }
      }
      delete faces_tbl;
   }

   t = ELEMENT;
   ns = ent_sets.GetNumSets(t);
   for (unsigned int s=0; s<ns; s++)
   {
      set<int>::iterator it;
      sets_[t][s].clear();
      for (it=ent_sets(t,s).begin(); it!=ent_sets(t,s).end(); it++)
      {
         if ( partitioning[*it] == MyRank_ )
         {
            sets_[t][s].insert(elem_global_local[*it]);
         }
      }
   }

   this->mesh_ = (Mesh*)this->pmesh_;

   this->CopyMeshTables();
}

ParEntitySets::~ParEntitySets()
{}

void
ParEntitySets::PrintSetInfo(std::ostream & output) const
{
   if ( MyRank_ == 0 &&
        ( GetNumSets(VERTEX) > 0 || GetNumSets(EDGE)    > 0 ||
          GetNumSets(FACE)   > 0 || GetNumSets(ELEMENT) > 0 ) )
   {
      output << "\nMFEM Parallel Entity Sets:\n";
   }
   this->PrintEntitySetInfo(output, VERTEX,  "Vertex");
   this->PrintEntitySetInfo(output, EDGE,    "Edge");
   this->PrintEntitySetInfo(output, FACE,    "Face");
   this->PrintEntitySetInfo(output, ELEMENT, "Element");
}

void
ParEntitySets::PrintEntitySetInfo(std::ostream & output, EntityType t,
                                  const string & ent_name) const
{
   if ( sets_[t].size() > 0 )
   {
      if ( MyRank_ == 0 )
      {
         output << "  " << ent_name
                << " Sets (Index, Set Name, Global Size):\n";
      }
      for (unsigned int s=0; s<sets_[t].size(); s++)
      {
         int loc_size = sets_[t][s].size();
         int glb_size = -1;
         MPI_Reduce(&loc_size, &glb_size, 1, MPI_INT, MPI_SUM, 0,
                    pmesh_->GetComm());
         if ( MyRank_ == 0 )
         {
            output << '\t' << s
                   << '\t' << set_names_[t][s]
                   << '\t' << glb_size
                   << '\n';
         }
      }
      if ( MyRank_ == 0 )
      {
         output << '\n';
      }
   }
}

} // namespace mfem

#endif // MFEM_USE_MPI
