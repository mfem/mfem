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


#include "entsets.hpp"
#include "mesh.hpp"

using namespace std;

namespace mfem
{

map<EntitySets::EntityType,string> EntitySets::EntityTypeNames =
   EntitySets::init_type_names();

map<EntitySets::EntityType,string> EntitySets::init_type_names()
{
   map<EntityType,string> m;

   m[INVALID] = "INVALID";
   m[VERTEX]  = "VERTEX";
   m[EDGE]    = "EDGE";
   m[FACE]    = "FACE";
   m[ELEMENT] = "ELEMENT";

   return m;
}

EntitySets::EntitySets(Mesh & mesh)
   : mesh_(&mesh),
     edge_vertex_(NULL),
     face_vertex_(NULL),
     face_edge_(NULL),
     NumOfVertices_(mesh.GetNV()),
     NumOfEdges_(mesh.GetNEdges()),
     NumOfElements_(mesh.GetNE()),
     sets_(4),
     set_names_(4),
     set_index_by_name_(4)
{
   cout << "Entering EntitySets(Mesh) c'tor" << endl;
   cout << "Leaving EntitySets(Mesh) c'tor" << endl;
}

EntitySets::EntitySets(const EntitySets & ent_sets)
   : mesh_(ent_sets.GetMesh()),
     edge_vertex_(NULL),
     face_vertex_(NULL),
     face_edge_(NULL),
     NumOfVertices_(mesh_->GetNV()),
     NumOfEdges_(mesh_->GetNEdges()),
     NumOfElements_(mesh_->GetNE()),
     sets_(4),
     set_names_(4),
     set_index_by_name_(4)
{
   cout << "Entering EntitySets copy c'tor" << endl;
   this->CopyEntitySets(ent_sets, VERTEX);
   this->CopyEntitySets(ent_sets, EDGE);
   this->CopyEntitySets(ent_sets, FACE);
   this->CopyEntitySets(ent_sets, ELEMENT);

   if ( ent_sets.GetEdgeVertexTable() )
   {
      edge_vertex_ = new Table(*ent_sets.GetEdgeVertexTable());
   }
   if ( ent_sets.GetFaceVertexTable() )
   {
      face_vertex_ = new Table(*ent_sets.GetFaceVertexTable());
   }
   if ( ent_sets.GetFaceEdgeTable() )
   {
      face_edge_ = new Table(*ent_sets.GetFaceEdgeTable());
   }
   cout << "Leaving EntitySets copy c'tor" << endl;
}

EntitySets::EntitySets(Mesh & mesh, NCMesh &ncmesh)
   : mesh_(&mesh),
     edge_vertex_(NULL),
     face_vertex_(NULL),
     face_edge_(NULL),
     NumOfVertices_(mesh_->GetNV()),
     NumOfEdges_(mesh_->GetNEdges()),
     NumOfElements_(mesh_->GetNE()),
     sets_(4),
     set_names_(4),
     set_index_by_name_(4)
{
   cout << "Entering EntitySets(Mesh, NCMesh) c'tor" << endl;
   this->BuildEntitySets(ncmesh, VERTEX);
   this->BuildEntitySets(ncmesh, EDGE);
   this->BuildEntitySets(ncmesh, FACE);
   this->BuildEntitySets(ncmesh, ELEMENT);
   cout << "Leaving EntitySets(Mesh, NCMesh) c'tor" << endl;
}

EntitySets::~EntitySets()
{
   cout << "Entering EntitySets d'tor" << endl;
   delete edge_vertex_;
   delete face_edge_;
   delete face_vertex_;
   cout << "Leaving EntitySets d'tor" << endl;
}

const string &
EntitySets::GetTypeName(EntityType t)
{
   return EntityTypeNames[t];
}

bool
EntitySets::SetExists(EntityType t, unsigned int s) const
{
   return s < sets_[t].size();
}

bool
EntitySets::SetExists(EntityType t, const string & s) const
{
   map<string,int>::const_iterator it = set_index_by_name_[t].find(s);

   return it != set_index_by_name_[t].end();
}

void
EntitySets::Load(istream &input)
{
   cout << "Entering EntitySets::Load" << endl;
   if (!input)
   {
      MFEM_ABORT("Input stream is not open");
   }

   string file_type;
   input >> ws;
   getline(input, file_type);
   filter_dos(file_type);

   bool mfem_sets_v10 = (file_type == "MFEM sets v1.0");
   if ( mfem_sets_v10 )
   {
      // string ident;
      //int Dim = -1;

      // read lines beginning with '#' (comments)
      // skip_comment_lines(input, '#');
      /*
      input >> ident >> ws; // 'dimension'

      MFEM_VERIFY(ident == "dimension", "invalid entity set file");
      input >> Dim >> ws;
      */
      int Dim = mesh_->Dimension();

      LoadEntitySets(input, VERTEX, "vertex_sets");

      if ( Dim > 1 )
      {
         LoadEntitySets(input, EDGE, "edge_sets");
      }
      if ( Dim > 2 )
      {
         LoadEntitySets(input, FACE, "face_sets");
      }

      LoadEntitySets(input, ELEMENT, "element_sets");
   }
   else
   {
      // Unknown input entity set format so this file has no entity sets
      return;
   }

   CopyMeshTables();
   cout << "Leaving EntitySets::Load" << endl;
}

void
EntitySets::LoadEntitySets(istream &input, EntityType t, const string & header)
{
   cout << "Entering EntitySets::LoadEntitySets(" << GetTypeName(t) << ")"
        << endl;
   string ident;
   int NumSets = -1;
   int NumEntities = -1;
   int g, v0, v1, v2, v3;

   DSTable  *v_to_v   = NULL;
   STable3D *face_tbl = NULL;

   skip_comment_lines(input, '#');
   input >> ident >> ws;

   cout << "ident: " << ident << endl;

   MFEM_VERIFY(ident == header, "invalid entity set file");
   input >> NumSets >> ws;
   cout << "NumSets: " << NumSets << endl;

   sets_[t].resize(NumSets);
   set_names_[t].resize(NumSets);
   cout << "resized" << endl;
   cout << "NumOfVertices: " << NumOfVertices_ << endl;
   if ( NumSets > 0 )
   {
      if ( t == EDGE )
      {
         v_to_v = new DSTable(NumOfVertices_);
         mesh_->GetVertexToVertexTable(*v_to_v);
      }
      else if ( t == FACE )
      {
         face_tbl = mesh_->GetFacesTable();
      }
   }
   cout << "reading sets" << endl;
   for (int i=0; i<NumSets; i++)
   {
      getline(input, ident);
      filter_dos(ident);
      cout << "set " << i << ", name: " << ident << flush;
      set_names_[t][i] = ident;
      set_index_by_name_[t][ident] = i;

      input >> NumEntities;
      cout << ", NumEntities: " << NumEntities << endl;
      for (int j=0; j<NumEntities; j++)
      {
         switch (t)
         {
            case VERTEX:
            case ELEMENT:
               // Read vertex or element index
               input >> v0;
               sets_[t][i].insert(v0);
               break;
            case EDGE:
               // Read two segment vertices
               input >> v0 >> v1;
               sets_[t][i].insert((*v_to_v)(v0,v1));
               break;
            case FACE:
               // Read geometry type
               input >> g;
               if ( g == 2 )
               {
                  // Read triangle vertices
                  input >> v0 >> v1 >> v2;
                  sets_[t][i].insert((*face_tbl)(v0, v1, v2));
               }
               else if ( g == 3)
               {
                  // Read quadrilateral vertices
                  input >> v0 >> v1 >> v2 >> v3;
                  sets_[t][i].insert((*face_tbl)(v0, v1, v2, v3));
               }
               else
               {
                  MFEM_ABORT("Unknown face geometry type format: \""
                             << g << "\"");
               }
               break;
            default:
               MFEM_ABORT("Unknown entity set type: \""
                          << GetTypeName(t) << "\"");
         }
         input >> ws;
      }
   }
   delete v_to_v;
   delete face_tbl;
   cout << "Leaving EntitySets::LoadEntitySets(" << GetTypeName(t) << ")"
        << endl;
}

void
EntitySets::Print(std::ostream &output) const
{
   output << "MFEM sets v1.0" << endl << endl;
   // output << "dimension" << endl << mesh_->Dimension() << endl << endl;

   this->PrintEntitySets(output, VERTEX, "vertex_sets");

   if ( mesh_->Dimension() > 1 )
   {
      this->PrintEdgeSets(output);
   }

   if ( mesh_->Dimension() > 2 )
   {
      this->PrintFaceSets(output);
   }

   this->PrintEntitySets(output, ELEMENT, "element_sets");
}

void
EntitySets::PrintEntitySets(std::ostream &output, EntityType t,
                            const std::string & header) const
{
   output << header << endl << sets_[t].size() << endl << endl;
   for (unsigned int s=0; s<sets_[t].size(); s++)
   {
      output << set_names_[t][s] << endl << sets_[t][s].size() << endl;

      set<int>::iterator it;
      unsigned int i = 0;
      for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++, i++)
      {
         output << *it;
         if ( i < sets_[t][s].size() - 1 )
         {
            output << " ";
         }
         else
         {
            output << endl << endl;
         }
      }
   }
}

void
EntitySets::PrintEdgeSets(std::ostream &output) const
{
   EntityType t = EDGE;

   output << "edge_sets" << endl << sets_[t].size() << endl << endl;
   for (unsigned int s=0; s<sets_[t].size(); s++)
   {
      output << set_names_[t][s] << endl << sets_[t][s].size() << endl;

      set<int>::const_iterator it;
      unsigned int i=0;
      for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++, i++)
      {
         int edge = *it;
         if ( edge < 0 )
         {
            output << "bad_edge_index:" << edge;
            if ( i < sets_[t][s].size() - 1 )
            {
               output << " ";
            }
            else
            {
               output << endl << endl;
            }
            continue;
         }
         if ( edge_vertex_ )
         {
            int *v = edge_vertex_->GetRow(edge);
            if ( v )
            {
               output << v[0] << " " << v[1];
            }
            else
            {
               output << "bad_vertex_info";
            }
         }
         else
         {
            output << edge;
         }
         if ( i < sets_[t][s].size() - 1 )
         {
            output << " ";
         }
         else
         {
            output << endl << endl;
         }
      }
   }
}

void
EntitySets::PrintFaceSets(std::ostream &output) const
{
   EntityType t = FACE;

   output << "face_sets" << endl << sets_[t].size() << endl << endl;
   for (unsigned int s=0; s<sets_[t].size(); s++)
   {
      output << set_names_[t][s] << endl << sets_[t][s].size() << endl;

      set<int>::const_iterator it;
      unsigned int i=0;
      for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++, i++)
      {
         int face = *it;
         if ( face < 0 )
         {
            output << "bad_face";
            if ( i < sets_[t][s].size() - 1 )
            {
               output << " ";
            }
            else
            {
               output << endl << endl;
            }
            continue;
         }
         if ( face_vertex_ )
         {
            int numv = face_vertex_->RowSize(face);
            int *v = face_vertex_->GetRow(face);
            output << numv - 1;
            for (int j=0; j<numv; j++)
            {
               output << " " << v[j];
            }
         }
         else
         {
            output << face;
         }
         output << endl;
      }
      output << endl;
   }
}

void
EntitySets::PrintSetInfo(std::ostream & output) const
{
   if ( GetNumSets(VERTEX) > 0 || GetNumSets(EDGE)    > 0 ||
        GetNumSets(FACE)   > 0 || GetNumSets(ELEMENT) > 0 )
   {
      output << "\nMFEM Entity Sets:\n";
   }
   else
   {
      output << "\nNo MFEM Entity Sets defined\n";
   }
   this->PrintEntitySetInfo(output, VERTEX,  "Vertex");
   this->PrintEntitySetInfo(output, EDGE,    "Edge");
   this->PrintEntitySetInfo(output, FACE,    "Face");
   this->PrintEntitySetInfo(output, ELEMENT, "Element");
}

void
EntitySets::PrintEntitySetInfo(std::ostream & output, EntityType t,
                               const string & ent_name) const
{
   if ( sets_[t].size() > 0 )
   {
      output << "  " << ent_name << " Sets (Index, Set Name, Size):\n";
      for (unsigned int s=0; s<sets_[t].size(); s++)
      {
         output << '\t' << s
                << '\t' << set_names_[t][s]
                << '\t' << sets_[t][s].size()
                << '\n';
      }
      output << '\n';
   }
}

void
EntitySets::CopyEntitySets(const EntitySets & ent_sets, EntityType t)
{
   unsigned int ns = ent_sets.GetNumSets(t);
   sets_[t].resize(ns);
   set_names_[t].resize(ns);
   for (unsigned int s=0; s<ns; s++)
   {
      set_names_[t][s] = ent_sets.GetSetName(t, s);
      set_index_by_name_[t][set_names_[t][s]] = s;

      set<int>::iterator it;
      for (it=ent_sets(t,s).begin(); it!=ent_sets(t,s).end(); it++)
      {
         sets_[t][s].insert(*it);
      }
   }
}

void
EntitySets::BuildEntitySets(NCMesh &ncmesh, EntityType t)
{
   cout << "BuildEntitySets for type " << GetTypeName(t) << endl;
   int es = ncmesh.ncent_sets->GetEntitySize(t);
   unsigned int ns = ncmesh.ncent_sets->GetNumSets(t);

   Array<int> inds(es);

   sets_[t].resize(ns);
   set_names_[t].resize(ns);
   for (unsigned int s=0; s<ns; s++)
   {
      int ni = ncmesh.ncent_sets->GetNumEntities(t, s);
      set_names_[t][s] = ncmesh.ncent_sets->GetSetName(t, s);
      set_index_by_name_[t][set_names_[t][s]] = s;
      std::cout << "processing set " << set_names_[t][s] << endl;
      switch (t)
      {
         case VERTEX:
            for (int i=0; i<ni; i++)
            {
               sets_[t][s].insert((*ncmesh.ncent_sets)(t, s, i));
            }
            break;
         case EDGE:
            for (int i=0; i<ni; i++)
            {
               ncmesh.ncent_sets->GetEntityIndex(t, s, i, inds);
               BlockArray<int> ind_coll;
               ncmesh.GetRefinedEdges(inds[0], inds[1],
                                      ind_coll);

               for (int j=0; j<ind_coll.Size(); j++)
               {
                  // sets_[t][s].insert(ind_coll[j]);
                  sets_[t][s].insert(ncmesh.nodes[ind_coll[j]].edge_index);
               }
            }
            break;
         case FACE:
            for (int i=0; i<ni; i++)
            {
               ncmesh.ncent_sets->GetEntityIndex(t, s, i, inds);
               BlockArray<int> ind_coll;
               ncmesh.GetRefinedFaces(inds[0], inds[1], inds[2], inds[3],
                                      ind_coll);

               for (int j=0; j<ind_coll.Size(); j++)
               {
                  sets_[t][s].insert(ncmesh.faces[ind_coll[j]].index);
               }
            }
            break;

         case ELEMENT:
            for (int i=0; i<ni; i++)
            {
               int elem = (*ncmesh.ncent_sets)(t, s, i);
               BlockArray<int> ind_coll;
               ncmesh.GetRefinedElements(elem, ind_coll);

               for (int j=0; j<ind_coll.Size(); j++)
               {
                  sets_[t][s].insert(ncmesh.elements[ind_coll[j]].index);
               }
            }
            break;
         default:
            MFEM_ABORT("Unknown entity set type: \"" << GetTypeName(t) << "\"");
      }
   }
   cout << "done BuildEntitySets for type " << GetTypeName(t) << endl;
}

unsigned int
EntitySets::GetNumSets(EntityType t) const
{
   MFEM_VERIFY( t >= VERTEX && t <= ELEMENT,
                "EntitySets Invalid entity type \"" << GetTypeName(t) << "\"");
   return sets_[t].size();
}

const string &
EntitySets::GetSetName(EntityType t, unsigned int s) const
{
   MFEM_VERIFY( t >= VERTEX && t <= ELEMENT,
                "EntitySets Invalid entity type \"" << GetTypeName(t) << "\"");
   MFEM_VERIFY(s < sets_[t].size(),"EntitySets:  set index out of range.");
   return set_names_[t][s];
}

int
EntitySets::GetSetIndex(EntityType t, const string & s) const
{
   MFEM_VERIFY( t >= VERTEX && t <= ELEMENT,
                "EntitySets Invalid entity type \"" << GetTypeName(t) << "\"");

   map<string,int>::const_iterator mit = set_index_by_name_[t].find(s);

   MFEM_VERIFY( mit != set_index_by_name_[t].end(),
                "EntitySets unrecognized set name \"" << s
                << "\" for entity type \"" << GetTypeName(t) << "\"" );

   return mit->second;
}

unsigned int
EntitySets::GetNumEntities(EntityType t, unsigned int s) const
{
   MFEM_VERIFY( t >= VERTEX && t <= ELEMENT,
                "EntitySets Invalid entity type \"" << GetTypeName(t) << "\"");
   MFEM_VERIFY(s < sets_[t].size(),"EntitySets:  set index out of range.");

   return sets_[t][s].size();
}

unsigned int
EntitySets::GetNumEntities(EntityType t, const std::string & s) const
{
   return this->GetNumEntities(t, this->GetSetIndex(t, s));
}

void
EntitySets::CopyMeshTables()
{
   if ( this->GetNumSets(EDGE) > 0 )
   {
      if ( edge_vertex_ ) { delete edge_vertex_; }
      if ( mesh_->edge_vertex )
      {
         edge_vertex_ = new Table(*mesh_->edge_vertex);
      }
      else
      {
         edge_vertex_ = mesh_->GetEdgeVertexTable();
         mesh_->edge_vertex = NULL;
      }
   }
   if ( this->GetNumSets(FACE) > 0 )
   {
      if ( face_vertex_ ) { delete face_vertex_; }
      if ( mesh_->face_vertex )
      {
         face_vertex_ = new Table(*mesh_->face_vertex);
      }
      else
      {
         face_vertex_ = mesh_->GetFaceVertexTable();
         mesh_->face_vertex = NULL;
      }
   }
   if ( this->GetNumSets(FACE) > 0 )
   {
      if ( face_edge_ ) { delete face_edge_; }
      if ( mesh_->face_edge )
      {
         face_edge_ = new Table(*mesh_->face_edge);
      }
      else
      {
         face_edge_ = mesh_->GetFaceEdgeTable();
         mesh_->face_edge = NULL;
      }
   }

   NumOfVertices_ = mesh_->GetNV();
   NumOfEdges_    = mesh_->GetNEdges();
   NumOfElements_ = mesh_->GetNE();
}

void
EntitySets::QuadUniformRefinement()
{
   // VERTEX Sets will be unchanged

   // EDGE Sets will need to be augmented
   if ( this->GetNumSets(EDGE) > 0 )
   {
      EntityType t = EDGE;
      DSTable * v_to_v = new DSTable(mesh_->GetNV());
      mesh_->GetVertexToVertexTable(*v_to_v);

      int oedge = NumOfVertices_;
      for (unsigned int s=0; s<this->GetNumSets(t); s++)
      {
         set<int>::iterator it;
         set<int> new_set;
         for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++)
         {
            int old_edge = *it;
            int *v = edge_vertex_->GetRow(old_edge);
            new_set.insert((*v_to_v)(v[0], oedge + old_edge));
            new_set.insert((*v_to_v)(v[1], oedge + old_edge));
         }
         sets_[t][s].clear();
         for (it=new_set.begin(); it!=new_set.end(); it++)
         {
            sets_[t][s].insert(*it);
         }
      }
      delete v_to_v;
   }

   // FACE Sets do not exist in 2D

   // ELEMENT Sets will need to be augmented
   EntityType t = ELEMENT;
   for (unsigned int s=0; s<this->GetNumSets(t); s++)
   {
      set<int>::iterator it;
      set<int> new_elems;
      for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++)
      {
         int e = *it;
         int j = NumOfElements_ + 3 * e;
         new_elems.insert(j + 0);
         new_elems.insert(j + 1);
         new_elems.insert(j + 2);
      }
      for (it=new_elems.begin(); it!=new_elems.end(); it++)
      {
         sets_[t][s].insert(*it);
      }
   }

   this->CopyMeshTables();
}

void
EntitySets::HexUniformRefinement()
{
   // VERTEX Sets will be unchanged

   // EDGE Sets will need to be augmented
   DSTable * v_to_v = NULL;
   if ( this->GetNumSets(EDGE) > 0 ||
        this->GetNumSets(FACE) > 0 )
   {
      v_to_v = new DSTable(mesh_->GetNV());
      mesh_->GetVertexToVertexTable(*v_to_v);
   }
   if ( this->GetNumSets(EDGE) > 0 )
   {
      EntityType t = EDGE;
      int oedge = NumOfVertices_;
      for (unsigned int s=0; s<this->GetNumSets(t); s++)
      {
         set<int>::iterator it;
         set<int> new_set;
         for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++)
         {
            int old_edge = *it;
            int *v = edge_vertex_->GetRow(old_edge);
            new_set.insert((*v_to_v)(v[0], oedge + old_edge));
            new_set.insert((*v_to_v)(v[1], oedge + old_edge));
         }
         sets_[t][s].clear();
         for (it=new_set.begin(); it!=new_set.end(); it++)
         {
            sets_[t][s].insert(*it);
         }
      }
   }
   delete v_to_v;

   // FACE Sets will need to be augmented
   if ( this->GetNumSets(FACE) > 0 )
   {
      EntityType t = FACE;
      STable3D * faces_tbl = mesh_->GetFacesTable();

      int oedge = NumOfVertices_;
      int oface = oedge + NumOfEdges_;

      for (unsigned int s=0; s<this->GetNumSets(t); s++)
      {
         set<int>::iterator it;
         set<int> new_set;
         for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++)
         {
            int old_face = *it;
            int * v = face_vertex_->GetRow(old_face);
            int * e = face_edge_->GetRow(old_face);

            for (int j=0; j<4; j++)
            {
               int v0 = v[j];
               int v3 = oface + old_face;
               for (int k=0; k<4; k++)
               {
                  int v1 = oedge + e[k];
                  int new_face = -1;
                  for (int l=1; l<4; l++)
                  {
                     int v2 = oedge + e[(k+l)%4];
                     new_face = (*faces_tbl)(v0,v1,v2,v3);
                     if ( new_face >= 0 )
                     {
                        new_set.insert(new_face);
                        break;
                     }
                  }
                  if ( new_face >= 0 )
                  {
                     break;
                  }
               }
            }
         }
         sets_[t][s].clear();
         for (it=new_set.begin(); it!=new_set.end(); it++)
         {
            sets_[t][s].insert(*it);
         }
      }
      delete faces_tbl;
   }

   // ELEMENT Sets will need to be augmented
   EntityType t = ELEMENT;
   for (unsigned int s=0; s<this->GetNumSets(t); s++)
   {
      set<int>::iterator it;
      set<int> new_elems;
      for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++)
      {
         int e = *it;
         for (int j=NumOfElements_ + 7 * e; j<NumOfElements_ + 7 * e + 7; j++)
         {
            new_elems.insert(j);
         }
      }
      for (it=new_elems.begin(); it!=new_elems.end(); it++)
      {
         sets_[t][s].insert(*it);
      }
   }

   this->CopyMeshTables();
}
void
EntitySets::UniformRefinement2D()
{
   // VERTEX Sets will be unchanged

   // EDGE Sets will need to be augmented
   if ( this->GetNumSets(EDGE) > 0 )
   {
      EntityType t = EDGE;
      DSTable * v_to_v = new DSTable(mesh_->GetNV());
      mesh_->GetVertexToVertexTable(*v_to_v);

      int oedge = NumOfVertices_;
      for (unsigned int s=0; s<this->GetNumSets(t); s++)
      {
         set<int>::iterator it;
         set<int> new_set;
         for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++)
         {
            int old_edge = *it;
            int *v = edge_vertex_->GetRow(old_edge);
            new_set.insert((*v_to_v)(v[0], oedge + old_edge));
            new_set.insert((*v_to_v)(v[1], oedge + old_edge));
         }
         sets_[t][s].clear();
         for (it=new_set.begin(); it!=new_set.end(); it++)
         {
            sets_[t][s].insert(*it);
         }
      }
      delete v_to_v;
   }

   // FACE Sets do not exist in 2D

   // ELEMENT Sets will need to be augmented
   EntityType t = ELEMENT;
   for (unsigned int s=0; s<this->GetNumSets(t); s++)
   {
      set<int>::iterator it;
      set<int> new_elems;
      for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++)
      {
         int e = *it;
         int j = NumOfElements_ + 3 * e;
         new_elems.insert(j + 0);
         new_elems.insert(j + 1);
         new_elems.insert(j + 2);
      }
      for (it=new_elems.begin(); it!=new_elems.end(); it++)
      {
         sets_[t][s].insert(*it);
      }
   }

   this->CopyMeshTables();
}

void
EntitySets::UniformRefinement3D()
{
   // VERTEX Sets will be unchanged

   // EDGE Sets will need to be augmented
   DSTable * v_to_v = NULL;
   if ( this->GetNumSets(EDGE) > 0 ||
        this->GetNumSets(FACE) > 0 )
   {
      v_to_v = new DSTable(mesh_->GetNV());
      mesh_->GetVertexToVertexTable(*v_to_v);
   }
   if ( this->GetNumSets(EDGE) > 0 )
   {
      EntityType t = EDGE;
      int oedge = NumOfVertices_;
      for (unsigned int s=0; s<this->GetNumSets(t); s++)
      {
         set<int>::iterator it;
         set<int> new_set;
         for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++)
         {
            int old_edge = *it;
            int *v = edge_vertex_->GetRow(old_edge);
            new_set.insert((*v_to_v)(v[0], oedge + old_edge));
            new_set.insert((*v_to_v)(v[1], oedge + old_edge));
         }
         sets_[t][s].clear();
         for (it=new_set.begin(); it!=new_set.end(); it++)
         {
            sets_[t][s].insert(*it);
         }
      }
   }
   delete v_to_v;

   // FACE Sets will need to be augmented
   if ( this->GetNumSets(FACE) > 0 )
   {
      EntityType t = FACE;
      STable3D * faces_tbl = mesh_->GetFacesTable();

      int oedge = NumOfVertices_;
      int oface = oedge + NumOfEdges_;

      for (unsigned int s=0; s<this->GetNumSets(t); s++)
      {
         set<int>::iterator it;
         set<int> new_set;
         for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++)
         {
            int old_face = *it;
            int * v = face_vertex_->GetRow(old_face);
            int * e = face_edge_->GetRow(old_face);

            for (int j=0; j<4; j++)
            {
               int v0 = v[j];
               int v3 = oface + old_face;
               for (int k=0; k<4; k++)
               {
                  int v1 = oedge + e[k];
                  int new_face = -1;
                  for (int l=1; l<4; l++)
                  {
                     int v2 = oedge + e[(k+l)%4];
                     new_face = (*faces_tbl)(v0,v1,v2,v3);
                     if ( new_face >= 0 )
                     {
                        new_set.insert(new_face);
                        break;
                     }
                  }
                  if ( new_face >= 0 )
                  {
                     break;
                  }
               }
            }
         }
         sets_[t][s].clear();
         for (it=new_set.begin(); it!=new_set.end(); it++)
         {
            sets_[t][s].insert(*it);
         }
      }
      delete faces_tbl;
   }

   // ELEMENT Sets will need to be augmented
   EntityType t = ELEMENT;
   for (unsigned int s=0; s<this->GetNumSets(t); s++)
   {
      set<int>::iterator it;
      set<int> new_elems;
      for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++)
      {
         int e = *it;
         for (int j=NumOfElements_ + 7 * e; j<NumOfElements_ + 7 * e + 7; j++)
         {
            new_elems.insert(j);
         }
      }
      for (it=new_elems.begin(); it!=new_elems.end(); it++)
      {
         sets_[t][s].insert(*it);
      }
   }

   this->CopyMeshTables();
}
/*
void
EntitySets::Prune(int nelems)
{
 EntityType t = ELEMENT;

 for (unsigned int s=0; s<sets_[t].size(); s++)
 {
    set<int>::const_iterator it;
    for (it=sets_[t][s].begin(); it!=sets_[t][s].end(); it++)
    {
       if ( *it >= nelems )
       {
          sets_[t][s].erase(it);
       }
    }
 }
}
*/
const int NCEntitySets::entity_size_[] = {1,2,4,1};

NCEntitySets::NCEntitySets(const EntitySets &ent_sets, NCMesh &ncmesh)
   : ncmesh_(&ncmesh),
     sets_(4),
     set_names_(4),
     set_index_by_name_(4)
{
   cout << "Entering NCEntitySets(Mesh) c'tor" << endl;

   EntitySets::EntityType t = EntitySets::INVALID;
   unsigned int ns = -1;

   std::cout << "Processing node sets" << std::endl;

   t = EntitySets::VERTEX;
   ns = ent_sets.GetNumSets(t);

   sets_[t].resize(ns);
   set_names_[t].resize(ns);

   for (unsigned int s=0; s<ns; s++)
   {
      unsigned int ni = ent_sets.GetNumEntities(t, s);

      sets_[t][s].resize(ni);
      set_names_[t][s] = ent_sets.GetSetName(t, s);
      set_index_by_name_[t][set_names_[t][s]] = s;

      set<int>::iterator it;
      int i = 0;
      for (it=ent_sets(t,s).begin();
           it!=ent_sets(t,s).end(); it++, i++)
      {
         sets_[t][s][i] = *it;
      }
   }

   std::cout << "Processing edge sets" << std::endl;
   if (ent_sets.edge_vertex_ == NULL)
   { std::cout << "edge_vertex table is NULL" << std::endl;}
   t = EntitySets::EDGE;
   ns = ent_sets.GetNumSets(t);

   sets_[t].resize(ns);
   set_names_[t].resize(ns);

   for (unsigned int s=0; s<ns; s++)
   {
      unsigned int ni = ent_sets.GetNumEntities(t, s);

      sets_[t][s].resize(2 * ni);
      set_names_[t][s] = ent_sets.GetSetName(t, s);
      set_index_by_name_[t][set_names_[t][s]] = s;

      set<int>::iterator it;
      int i = 0;
      for (it=ent_sets(t,s).begin();
           it!=ent_sets(t,s).end(); it++, i++)
      {
         int edge = *it;
         int *v = ent_sets.edge_vertex_->GetRow(edge);
         sets_[t][s][2 * i + 0] = v[0];
         sets_[t][s][2 * i + 1] = v[1];
      }
   }

   std::cout << "Processing face sets" << std::endl;

   t = EntitySets::FACE;
   ns = ent_sets.GetNumSets(t);

   sets_[t].resize(ns);
   set_names_[t].resize(ns);

   for (unsigned int s=0; s<ns; s++)
   {
      unsigned int ni = ent_sets.GetNumEntities(t, s);

      sets_[t][s].resize(4 * ni);
      set_names_[t][s] = ent_sets.GetSetName(t, s);
      set_index_by_name_[t][set_names_[t][s]] = s;

      set<int>::iterator it;
      int i = 0;
      for (it=ent_sets(t,s).begin();
           it!=ent_sets(t,s).end(); it++, i++)
      {
         int face = *it;
         int *v = ent_sets.face_vertex_->GetRow(face);
         int nv = ent_sets.face_vertex_->RowSize(face);
         sets_[t][s][4 * i + 0] = v[0];
         sets_[t][s][4 * i + 1] = v[1];
         sets_[t][s][4 * i + 2] = v[2];
         if ( nv > 3 )
         {
            sets_[t][s][4 * i + 3] = v[3];

            // Unfortunately the face_vertex table doesn not store
            // the topological order of the face vertices but we
            // need this to find refined faces.  To recover this
            // information we verify that edges connect vertex 0 to
            // vertices 1 and 3.  If not then we swap either 1 or 3
            // with vertex 2.
            int e01 = ncmesh.nodes.FindId(v[0], v[1]);
            int e03 = ncmesh.nodes.FindId(v[0], v[3]);

            if ( e01 < 0 )
            {
               Swap(sets_[t][s][4 * i + 1],sets_[t][s][4 * i + 2]);
            }
            else if ( e03 < 0 )
            {
               Swap(sets_[t][s][4 * i + 3],sets_[t][s][4 * i + 2]);
            }
         }
         else
         {
            sets_[t][s][4 * i + 3] = -1;
         }
      }
   }

   std::cout << "Processing element sets" << std::endl;

   t = EntitySets::ELEMENT;
   ns = ent_sets.GetNumSets(t);

   sets_[t].resize(ns);
   set_names_[t].resize(ns);

   for (unsigned int s=0; s<ns; s++)
   {
      unsigned int ni = ent_sets.GetNumEntities(t, s);

      sets_[t][s].resize(ni);
      set_names_[t][s] = ent_sets.GetSetName(t, s);
      set_index_by_name_[t][set_names_[t][s]] = s;

      set<int>::iterator it;
      int i = 0;
      for (it=ent_sets(t,s).begin();
           it!=ent_sets(t,s).end(); it++, i++)
      {
         sets_[t][s][i] = *it;
      }
   }
   cout << "Leaving NCEntitySets(Mesh) c'tor" << endl;
}

NCEntitySets::NCEntitySets(const NCEntitySets & ncent_sets)
   : ncmesh_(ncent_sets.ncmesh_),
     sets_(4),
     set_names_(4),
     set_index_by_name_(4)
{
   cout << "Entering NCEntitySets copy c'tor" << endl;
   this->CopyNCEntitySets(ncent_sets, EntitySets::VERTEX);
   this->CopyNCEntitySets(ncent_sets, EntitySets::EDGE);
   this->CopyNCEntitySets(ncent_sets, EntitySets::FACE);
   this->CopyNCEntitySets(ncent_sets, EntitySets::ELEMENT);
   cout << "Leaving NCEntitySets copy c'tor" << endl;
}

bool
NCEntitySets::SetExists(EntitySets::EntityType t, unsigned int s) const
{
   return s < sets_[t].size();
}

bool
NCEntitySets::SetExists(EntitySets::EntityType t, const string & s) const
{
   map<string,int>::const_iterator it = set_index_by_name_[t].find(s);

   return it != set_index_by_name_[t].end();
}

unsigned int
NCEntitySets::GetNumSets(EntitySets::EntityType t) const
{
   MFEM_ASSERT( t >= EntitySets::VERTEX && t <= EntitySets::ELEMENT,
                "NCEntitySets Invalid entity type \"" << t << "\"");
   return sets_[t].size();
}

int
NCEntitySets::GetEntitySize(EntitySets::EntityType t)
{
   MFEM_ASSERT( t >= EntitySets::VERTEX && t <= EntitySets::ELEMENT,
                "NCEntitySets Invalid entity type \"" << t << "\"");
   return entity_size_[t];
}

const string &
NCEntitySets::GetSetName(EntitySets::EntityType t, int s) const
{
   MFEM_ASSERT( t >= EntitySets::VERTEX && t <= EntitySets::ELEMENT,
                "NCEntitySets Invalid entity type \"" << t << "\"");
   return set_names_[t][s];
}

int
NCEntitySets::GetSetIndex(EntitySets::EntityType t, const string & s) const
{
   MFEM_ASSERT( t >= EntitySets::VERTEX && t <= EntitySets::ELEMENT,
                "NCEntitySets Invalid entity type \"" << t << "\"");

   map<string,int>::const_iterator mit = set_index_by_name_[t].find(s);

   MFEM_ASSERT( mit != set_index_by_name_[t].end(),
                "NCEntitySets unrecognized set name \"" << s
                << "\" for entity type \"" << t << "\"" );

   return mit->second;
}

unsigned int
NCEntitySets::GetNumEntities(EntitySets::EntityType t, int s) const
{
   MFEM_ASSERT( t >= EntitySets::VERTEX && t <= EntitySets::ELEMENT,
                "NCEntitySets Invalid entity type \"" << t << "\"");
   return sets_[t][s].size() / entity_size_[t];
}

unsigned int
NCEntitySets::GetNumEntities(EntitySets::EntityType t,
                             const std::string & s) const
{
   return this->GetNumEntities(t, this->GetSetIndex(t, s));
}

void
NCEntitySets::GetEntityIndex(EntitySets::EntityType t, int s, int i,
                             Array<int> & inds) const
{
   MFEM_ASSERT( t >= EntitySets::VERTEX && t <= EntitySets::ELEMENT,
                "NCEntitySets Invalid entity type \"" << t << "\"");

   inds.SetSize(entity_size_[t]);
   for (int j=0; j<entity_size_[t]; j++)
   {
      inds[j] = sets_[t][s][entity_size_[t] * i + j];
   }
}

void
NCEntitySets::GetEntityIndex(EntitySets::EntityType t, const string & s,
                             int i, Array<int> &inds) const
{
   return this->GetEntityIndex(t, this->GetSetIndex(t, s), i, inds);
}

void
NCEntitySets::CopyNCEntitySets(const NCEntitySets & ncent_sets,
                               EntitySets::EntityType t)
{
   unsigned int ns = ncent_sets.GetNumSets(t);
   sets_[t].resize(ns);
   set_names_[t].resize(ns);
   for (unsigned int s=0; s<ns; s++)
   {
      set_names_[t][s] = ncent_sets.GetSetName(t, s);
      set_index_by_name_[t][set_names_[t][s]] = s;

      sets_[t][s].resize(ncent_sets(t,s).size());
      for (unsigned int i=0; i<ncent_sets(t,s).size(); i++)
      {
         sets_[t][s][i] = ncent_sets(t,s,i);
      }
   }
}

}; // namespace mfem
