// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#include "entsets.hpp"
#include "mesh.hpp"
#include "../general/kdtree.hpp"

#ifdef MFEM_USE_MPI
#include "pmesh.hpp"
#endif

using namespace std;

namespace mfem
{

namespace
{

void GetPhysicalVertexCoordinates(Mesh &mesh, DenseMatrix &coordinates)
{
   coordinates.SetSize(mesh.SpaceDimension(), mesh.GetNV());
   if (!mesh.GetNodes())
   {
      for (int vertex = 0; vertex < mesh.GetNV(); vertex++)
      {
         coordinates.SetCol(vertex, mesh.GetVertex(vertex));
      }
      return;
   }

   Table *vertex_to_element = mesh.GetVertexToElementTable();
   Array<int> element_vertices;
   Vector point(mesh.SpaceDimension());
   for (int vertex = 0; vertex < mesh.GetNV(); vertex++)
   {
      MFEM_VERIFY(vertex_to_element->RowSize(vertex) > 0,
                  "A mesh vertex must belong to an element.");
      const int element = vertex_to_element->GetRow(vertex)[0];
      mesh.GetElementVertices(element, element_vertices);
      int local_vertex = -1;
      for (int i = 0; i < element_vertices.Size(); i++)
      {
         if (element_vertices[i] == vertex)
         {
            local_vertex = i;
            break;
         }
      }
      MFEM_VERIFY(local_vertex >= 0,
                  "Vertex-to-element connectivity is inconsistent.");

      const Geometry::Type geometry = mesh.GetElementBaseGeometry(element);
      const IntegrationRule *vertices = Geometries.GetVertices(geometry);
      mesh.GetElementTransformation(element)->Transform(
         vertices->IntPoint(local_vertex), point);
      coordinates.SetCol(vertex, point);
   }
   delete vertex_to_element;
}

template <int dimension>
void FindNearestVerticesKDTree(const DenseMatrix &coordinates,
                               const DenseMatrix &points,
                               Array<int> &vertices,
                               Array<real_t> &distances)
{
   KDTree<int,real_t,dimension> tree;
   for (int vertex = 0; vertex < coordinates.Width(); vertex++)
   {
      tree.AddPoint(coordinates.GetColumn(vertex), vertex);
   }
   tree.Sort();

   vertices.SetSize(points.Width());
   distances.SetSize(points.Width());
   for (int point = 0; point < points.Width(); point++)
   {
      typename KDTree<int,real_t,dimension>::PointND query(
         points.GetColumn(point));
      int nearest;
      real_t distance;
      tree.FindClosestPoint(query, nearest, distance);
      vertices[point] = nearest;
      distances[point] = distance;
   }
}

void FindNearestVertices(const DenseMatrix &coordinates,
                         const DenseMatrix &points,
                         Array<int> &vertices, Array<real_t> &distances)
{
   switch (coordinates.Height())
   {
      case 1:
         FindNearestVerticesKDTree<1>(coordinates, points, vertices, distances);
         break;
      case 2:
         FindNearestVerticesKDTree<2>(coordinates, points, vertices, distances);
         break;
      case 3:
         FindNearestVerticesKDTree<3>(coordinates, points, vertices, distances);
         break;
      default:
         MFEM_ABORT("Nearest-vertex search requires space dimension 1, 2, or 3.");
   }
}

void VerifyOrientedFaceSetMesh(const Mesh &mesh)
{
   MFEM_VERIFY(mesh.Dimension() == 3,
               "Oriented face sets require a three-dimensional mesh.");
   MFEM_VERIFY(mesh.Conforming(),
               "Oriented face sets do not yet support nonconforming meshes.");
#ifdef MFEM_USE_MPI
   MFEM_VERIFY(dynamic_cast<const ParMesh *>(&mesh) == nullptr,
               "Oriented face sets do not yet support parallel meshes.");
#endif
}

} // namespace

OrientedFaceSet::OrientedFaceSet(Mesh &mesh, const Array<int> &faces,
                                 const Array<int> &normal_signs)
   : mesh_(&mesh), faces_(faces), normal_signs_(normal_signs)
{
   MFEM_VERIFY(mesh.Dimension() == 3,
               "Oriented face sets require a three-dimensional mesh.");
   MFEM_VERIFY(faces_.Size() == normal_signs_.Size(),
               "Each oriented face must have a normal sign.");
   for (int i = 0; i < faces_.Size(); i++)
   {
      MFEM_VERIFY(faces_[i] >= 0 && faces_[i] < mesh.GetNumFaces(),
                  "Oriented face index is out of range.");
      MFEM_VERIFY(normal_signs_[i] == 1 || normal_signs_[i] == -1,
                  "Oriented face normal signs must be +1 or -1.");
   }
}

OrientedFaceSet::OrientedFaceSet(Mesh &mesh, const EntitySets &element_sets,
                                 int element_set)
   : mesh_(&mesh)
{
   VerifyOrientedFaceSetMesh(mesh);
   MFEM_VERIFY(element_sets.GetMesh() == &mesh,
               "The element set belongs to a different mesh.");
   MFEM_VERIFY(element_set >= 0 &&
               element_sets.SetExists(EntitySets::ELEMENT, element_set),
               "Invalid element-set index.");

   Array<int> marker(mesh.GetNE());
   marker = 0;
   for (const int element : element_sets(EntitySets::ELEMENT, element_set))
   {
      marker[element] = 1;
   }
   BuildExterior(marker);
}

OrientedFaceSet::OrientedFaceSet(Mesh &mesh, int attribute)
   : mesh_(&mesh)
{
   VerifyOrientedFaceSetMesh(mesh);
   Array<int> marker(mesh.GetNE());
   for (int element = 0; element < mesh.GetNE(); element++)
   {
      marker[element] = (mesh.GetAttribute(element) == attribute);
   }
   BuildExterior(marker);
}

OrientedFaceSet::OrientedFaceSet(Mesh &mesh, const EntitySets &element_sets,
                                 int from_set, int to_set)
   : mesh_(&mesh)
{
   VerifyOrientedFaceSetMesh(mesh);
   MFEM_VERIFY(element_sets.GetMesh() == &mesh,
               "The element sets belong to a different mesh.");
   MFEM_VERIFY(from_set >= 0 &&
               element_sets.SetExists(EntitySets::ELEMENT, from_set),
               "Invalid source element-set index.");
   MFEM_VERIFY(to_set >= 0 &&
               element_sets.SetExists(EntitySets::ELEMENT, to_set),
               "Invalid target element-set index.");

   Array<int> from_marker(mesh.GetNE());
   Array<int> to_marker(mesh.GetNE());
   from_marker = 0;
   to_marker = 0;
   for (const int element : element_sets(EntitySets::ELEMENT, from_set))
   {
      from_marker[element] = 1;
   }
   for (const int element : element_sets(EntitySets::ELEMENT, to_set))
   {
      to_marker[element] = 1;
   }
   BuildInterface(from_marker, to_marker);
}

OrientedFaceSet::OrientedFaceSet(Mesh &mesh, int from_attribute,
                                 int to_attribute)
   : mesh_(&mesh)
{
   VerifyOrientedFaceSetMesh(mesh);
   MFEM_VERIFY(from_attribute != to_attribute,
               "The two interface attributes must be different.");
   Array<int> from_marker(mesh.GetNE());
   Array<int> to_marker(mesh.GetNE());
   for (int element = 0; element < mesh.GetNE(); element++)
   {
      const int attribute = mesh.GetAttribute(element);
      from_marker[element] = (attribute == from_attribute);
      to_marker[element] = (attribute == to_attribute);
   }
   BuildInterface(from_marker, to_marker);
}

void OrientedFaceSet::BuildExterior(const Array<int> &element_marker)
{
   MFEM_VERIFY(element_marker.Size() == mesh_->GetNE(),
               "Invalid element marker.");
   for (int face = 0; face < mesh_->GetNumFaces(); face++)
   {
      int element1, element2;
      mesh_->GetFaceElements(face, &element1, &element2);
      const bool selected1 = element_marker[element1] != 0;
      const bool selected2 = element2 >= 0 && element_marker[element2] != 0;
      if (selected1 == selected2) { continue; }

      const int selected_element = selected1 ? element1 : element2;
      faces_.Append(face);
      normal_signs_.Append(GetOutwardNormalSign(face, selected_element));
   }
}

void OrientedFaceSet::BuildInterface(const Array<int> &from_marker,
                                     const Array<int> &to_marker)
{
   MFEM_VERIFY(from_marker.Size() == mesh_->GetNE() &&
               to_marker.Size() == mesh_->GetNE(), "Invalid element marker.");
   for (int element = 0; element < mesh_->GetNE(); element++)
   {
      MFEM_VERIFY(!(from_marker[element] && to_marker[element]),
                  "The two element regions must be disjoint.");
   }

   for (int face = 0; face < mesh_->GetNumFaces(); face++)
   {
      int element1, element2;
      mesh_->GetFaceElements(face, &element1, &element2);
      if (element2 < 0) { continue; }

      const bool from1 = from_marker[element1] != 0;
      const bool from2 = from_marker[element2] != 0;
      const bool to1 = to_marker[element1] != 0;
      const bool to2 = to_marker[element2] != 0;
      if (!((from1 && to2) || (from2 && to1))) { continue; }

      const int from_element = from1 ? element1 : element2;
      faces_.Append(face);
      normal_signs_.Append(GetOutwardNormalSign(face, from_element));
   }
}

int OrientedFaceSet::GetOutwardNormalSign(int face, int element) const
{
   ElementTransformation *face_trans = mesh_->GetFaceTransformation(face);
   const Geometry::Type geometry = mesh_->GetFaceGeometry(face);
   const IntegrationPoint &center = Geometries.GetCenter(geometry);
   Vector face_center(mesh_->SpaceDimension());
   Vector element_center(mesh_->SpaceDimension());
   face_trans->Transform(center, face_center);
   mesh_->GetElementCenter(element, element_center);
   face_center -= element_center;

   face_trans->SetIntPoint(&center);
   Vector normal(mesh_->SpaceDimension());
   CalcOrtho(face_trans->Jacobian(), normal);
   const real_t normal_direction = normal * face_center;
   MFEM_VERIFY(normal_direction != 0.0,
               "Unable to determine an oriented face normal.");
   return normal_direction > 0.0 ? 1 : -1;
}

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
   this->BuildEntitySets(ncmesh, VERTEX);
   this->BuildEntitySets(ncmesh, EDGE);
   this->BuildEntitySets(ncmesh, FACE);
   this->BuildEntitySets(ncmesh, ELEMENT);
}

EntitySets::~EntitySets()
{
   delete edge_vertex_;
   delete face_edge_;
   delete face_vertex_;
}

const string &
EntitySets::GetTypeName(EntityType t)
{
   return EntityTypeNames[t];
}

bool
EntitySets::SetExists(EntityType t, unsigned int s) const
{
   VerifyEntityType(t);
   return s < sets_[t].size();
}

bool
EntitySets::SetExists(EntityType t, const string & s) const
{
   VerifyEntityType(t);
   map<string,int>::const_iterator it = set_index_by_name_[t].find(s);

   return it != set_index_by_name_[t].end();
}

void
EntitySets::Load(istream &input)
{
   VerifyCanModify();
   if (!input)
   {
      MFEM_ABORT("Input stream is not open");
   }

   for (int t = VERTEX; t <= ELEMENT; t++)
   {
      sets_[t].clear();
      set_names_[t].clear();
      set_index_by_name_[t].clear();
   }

   string file_type;
   input >> ws;
   getline(input, file_type);
   filter_dos(file_type);

   bool mfem_sets_v10 = (file_type == "MFEM sets v1.0");
   if ( mfem_sets_v10 )
   {
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
}

void
EntitySets::LoadEntitySets(istream &input, EntityType t, const string & header)
{
   string ident;
   int NumSets = -1;
   int NumEntities = -1;
   int g, v0, v1, v2, v3;

   DSTable  *v_to_v   = NULL;
   STable3D *face_tbl = NULL;

   skip_comment_lines(input, '#');
   input >> ident >> ws;

   MFEM_VERIFY(ident == header, "invalid entity set file");
   input >> NumSets >> ws;
   MFEM_VERIFY(NumSets >= 0, "invalid number of entity sets");

   sets_[t].clear();
   set_names_[t].clear();
   set_index_by_name_[t].clear();
   sets_[t].resize(NumSets);
   set_names_[t].resize(NumSets);
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
   for (int i=0; i<NumSets; i++)
   {
      getline(input, ident);
      filter_dos(ident);
      MFEM_VERIFY(!ident.empty(), "EntitySets set names cannot be empty.");
      MFEM_VERIFY(set_index_by_name_[t].count(ident) == 0,
                  "EntitySets set name already exists.");
      set_names_[t][i] = ident;
      set_index_by_name_[t][ident] = i;

      input >> NumEntities;
      MFEM_VERIFY(input && NumEntities >= 0, "invalid number of set entities");
      for (int j=0; j<NumEntities; j++)
      {
         switch (t)
         {
            case VERTEX:
            case ELEMENT:
               // Read vertex or element index
               input >> v0;
               MFEM_VERIFY(input, "invalid entity set entry");
               VerifyEntityIndex(t, v0);
               sets_[t][i].insert(v0);
               break;
            case EDGE:
            {
               // Read two segment vertices
               input >> v0 >> v1;
               MFEM_VERIFY(input, "invalid entity set entry");
               VerifyVertexIndex(v0);
               VerifyVertexIndex(v1);
               const int edge = (*v_to_v)(v0, v1);
               MFEM_VERIFY(edge >= 0,
                           "The specified vertices do not define a mesh edge.");
               sets_[t][i].insert(edge);
               break;
            }
            case FACE:
            {
               // Read geometry type
               input >> g;
               if ( g == 2 )
               {
                  // Read triangle vertices
                  input >> v0 >> v1 >> v2;
                  MFEM_VERIFY(input, "invalid entity set entry");
                  VerifyVertexIndex(v0);
                  VerifyVertexIndex(v1);
                  VerifyVertexIndex(v2);
                  const int face = face_tbl->Index(v0, v1, v2);
                  MFEM_VERIFY(face >= 0,
                              "The specified vertices do not define a mesh face.");
                  sets_[t][i].insert(face);
               }
               else if ( g == 3)
               {
                  // Read quadrilateral vertices
                  input >> v0 >> v1 >> v2 >> v3;
                  MFEM_VERIFY(input, "invalid entity set entry");
                  VerifyVertexIndex(v0);
                  VerifyVertexIndex(v1);
                  VerifyVertexIndex(v2);
                  VerifyVertexIndex(v3);
                  const int face = (*face_tbl)(v0, v1, v2, v3);
                  MFEM_VERIFY(face >= 0,
                              "The specified vertices do not define a mesh face.");
                  sets_[t][i].insert(face);
               }
               else
               {
                  MFEM_ABORT("Unknown face geometry type format: \""
                             << g << "\"");
               }
               break;
            }
            default:
               MFEM_ABORT("Unknown entity set type: \""
                          << GetTypeName(t) << "\"");
         }
         input >> ws;
      }
   }
   delete v_to_v;
   delete face_tbl;
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
   int es = ncmesh.ncent_sets->GetEntitySize(t);
   unsigned int ns = ncmesh.ncent_sets->GetNumSets(t);

   Array<int> inds(es);
   auto root_node = [&ncmesh](int vertex)
   {
      return ncmesh.nodes.FindId(vertex, vertex);
   };

   sets_[t].resize(ns);
   set_names_[t].resize(ns);
   for (unsigned int s=0; s<ns; s++)
   {
      int ni = ncmesh.ncent_sets->GetNumEntities(t, s);
      set_names_[t][s] = ncmesh.ncent_sets->GetSetName(t, s);
      set_index_by_name_[t][set_names_[t][s]] = s;
      switch (t)
      {
         case VERTEX:
            for (int i=0; i<ni; i++)
            {
               const int node = root_node((*ncmesh.ncent_sets)(t, s, i));
               if (node < 0) { continue; }
               const int index = ncmesh.nodes[node].vert_index;
               if (index >= 0) { sets_[t][s].insert(index); }
            }
            break;
         case EDGE:
            for (int i=0; i<ni; i++)
            {
               ncmesh.ncent_sets->GetEntityIndex(t, s, i, inds);
               inds[0] = root_node(inds[0]);
               inds[1] = root_node(inds[1]);
               if (inds[0] < 0 || inds[1] < 0) { continue; }
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
               bool missing_root = false;
               for (int j = 0; j < 3; j++)
               {
                  inds[j] = root_node(inds[j]);
                  missing_root |= inds[j] < 0;
               }
               if (inds[3] >= 0)
               {
                  inds[3] = root_node(inds[3]);
                  missing_root |= inds[3] < 0;
               }
               if (missing_root) { continue; }
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

void EntitySets::VerifyEntityType(EntityType t) const
{
   MFEM_VERIFY(t >= VERTEX && t <= ELEMENT,
               "EntitySets invalid entity type \"" << GetTypeName(t) << "\"");
}

void EntitySets::VerifySetIndex(EntityType t, int set_index) const
{
   VerifyEntityType(t);
   MFEM_VERIFY(set_index >= 0 && set_index < (int)sets_[t].size(),
               "EntitySets set index out of range.");
}

void EntitySets::VerifyVertexIndex(int vertex_index) const
{
   MFEM_VERIFY(vertex_index >= 0 && vertex_index < mesh_->GetNV(),
               "EntitySets vertex index out of range.");
}

void EntitySets::VerifyEntityIndex(EntityType t, int entity_index) const
{
   VerifyEntityType(t);

   int num_entities = 0;
   switch (t)
   {
      case VERTEX:  num_entities = mesh_->GetNV(); break;
      case EDGE:
         MFEM_VERIFY(mesh_->Dimension() > 1,
                     "Edge sets require a mesh of dimension at least two.");
         num_entities = mesh_->GetNEdges();
         break;
      case FACE:
         MFEM_VERIFY(mesh_->Dimension() > 2,
                     "Face sets require a three-dimensional mesh.");
         num_entities = mesh_->GetNumFaces();
         break;
      case ELEMENT: num_entities = mesh_->GetNE(); break;
      default: MFEM_ABORT("invalid entity type");
   }

   MFEM_VERIFY(entity_index >= 0 && entity_index < num_entities,
               "EntitySets entity index out of range.");
}

void EntitySets::VerifyCanModify() const
{
   MFEM_VERIFY(!mesh_->ncmesh,
               "Entity sets must be defined before converting a mesh to a "
               "nonconforming mesh.");
}

int EntitySets::AddSet(EntityType t, const string &name)
{
   VerifyCanModify();
   VerifyEntityType(t);
   MFEM_VERIFY(!name.empty(), "EntitySets set names cannot be empty.");
   MFEM_VERIFY(name.find('\n') == string::npos && name.find('\r') == string::npos,
               "EntitySets set names cannot contain line breaks.");
   MFEM_VERIFY(!SetExists(t, name), "EntitySets set name already exists.");

   const int set_index = sets_[t].size();
   sets_[t].emplace_back();
   set_names_[t].push_back(name);
   set_index_by_name_[t][name] = set_index;
   return set_index;
}

void EntitySets::AddEntity(EntityType t, int set_index, int entity_index)
{
   VerifyCanModify();
   VerifySetIndex(t, set_index);
   VerifyEntityIndex(t, entity_index);
   sets_[t][set_index].insert(entity_index);

   if (t == EDGE || t == FACE) { CopyMeshTables(); }
}

void EntitySets::AddEntity(EntityType t, const string &set_name,
                           int entity_index)
{
   AddEntity(t, GetSetIndex(t, set_name), entity_index);
}

void EntitySets::AddEdge(int set_index, int v0, int v1)
{
   VerifyVertexIndex(v0);
   VerifyVertexIndex(v1);
   MFEM_VERIFY(v0 != v1, "An edge requires two distinct vertices.");

   DSTable v_to_v(mesh_->GetNV());
   mesh_->GetVertexToVertexTable(v_to_v);
   const int edge = v_to_v(v0, v1);
   MFEM_VERIFY(edge >= 0, "The specified vertices do not define a mesh edge.");
   AddEntity(EDGE, set_index, edge);
}

void EntitySets::AddEdge(const string &set_name, int v0, int v1)
{
   AddEdge(GetSetIndex(EDGE, set_name), v0, v1);
}

void EntitySets::AddTriFace(int set_index, int v0, int v1, int v2)
{
   VerifyVertexIndex(v0);
   VerifyVertexIndex(v1);
   VerifyVertexIndex(v2);
   MFEM_VERIFY(v0 != v1 && v0 != v2 && v1 != v2,
               "A triangular face requires three distinct vertices.");

   STable3D *faces = mesh_->GetFacesTable();
   const int face = faces->Index(v0, v1, v2);
   delete faces;
   MFEM_VERIFY(face >= 0, "The specified vertices do not define a mesh face.");
   AddEntity(FACE, set_index, face);
}

void EntitySets::AddTriFace(const string &set_name, int v0, int v1, int v2)
{
   AddTriFace(GetSetIndex(FACE, set_name), v0, v1, v2);
}

void EntitySets::AddQuadFace(int set_index, int v0, int v1, int v2, int v3)
{
   VerifyVertexIndex(v0);
   VerifyVertexIndex(v1);
   VerifyVertexIndex(v2);
   VerifyVertexIndex(v3);
   MFEM_VERIFY(v0 != v1 && v0 != v2 && v0 != v3 && v1 != v2 && v1 != v3 &&
               v2 != v3, "A quadrilateral face requires four distinct vertices.");

   STable3D *faces = mesh_->GetFacesTable();
   const int face = (*faces)(v0, v1, v2, v3);
   delete faces;
   MFEM_VERIFY(face >= 0, "The specified vertices do not define a mesh face.");
   AddEntity(FACE, set_index, face);
}

void EntitySets::AddQuadFace(const string &set_name, int v0, int v1, int v2,
                             int v3)
{
   AddQuadFace(GetSetIndex(FACE, set_name), v0, v1, v2, v3);
}

int EntitySets::AddElementSet(const string &name, const Array<int> &elements)
{
   const int set_index = AddSet(ELEMENT, name);
   for (const int element : elements)
   {
      AddEntity(ELEMENT, set_index, element);
   }
   return set_index;
}

int EntitySets::AddVertexSet(const string &name, const Array<int> &vertices)
{
   const int set_index = AddSet(VERTEX, name);
   for (const int vertex : vertices)
   {
      AddEntity(VERTEX, set_index, vertex);
   }
   return set_index;
}

int EntitySets::AddVertexSetByMarker(const string &name,
                                     const Array<int> &marker)
{
   MFEM_VERIFY(marker.Size() == mesh_->GetNV(),
               "Vertex marker size must equal the number of local vertices.");
   const int set_index = AddSet(VERTEX, name);
   for (int vertex = 0; vertex < marker.Size(); vertex++)
   {
      if (marker[vertex]) { sets_[VERTEX][set_index].insert(vertex); }
   }
   return set_index;
}

int EntitySets::AddVertexSetByPredicate(const string &name,
                                        const PointPredicate &predicate)
{
   MFEM_VERIFY(predicate, "EntitySets point predicate is empty.");
   const int set_index = AddSet(VERTEX, name);
   Vector point(mesh_->SpaceDimension());

   if (mesh_->GetNodes())
   {
      Table *vertex_to_element = mesh_->GetVertexToElementTable();
      Array<int> element_vertices;
      for (int vertex = 0; vertex < mesh_->GetNV(); vertex++)
      {
         const int *elements = vertex_to_element->GetRow(vertex);
         MFEM_VERIFY(vertex_to_element->RowSize(vertex) > 0,
                     "A mesh vertex must belong to an element.");
         const int element = elements[0];
         mesh_->GetElementVertices(element, element_vertices);
         int local_vertex = -1;
         for (int i = 0; i < element_vertices.Size(); i++)
         {
            if (element_vertices[i] == vertex)
            {
               local_vertex = i;
               break;
            }
         }
         MFEM_VERIFY(local_vertex >= 0,
                     "Vertex-to-element connectivity is inconsistent.");

         const Geometry::Type geometry = mesh_->GetElementBaseGeometry(element);
         const IntegrationRule *vertices = Geometries.GetVertices(geometry);
         mesh_->GetElementTransformation(element)->Transform(
            vertices->IntPoint(local_vertex), point);
         if (predicate(point)) { sets_[VERTEX][set_index].insert(vertex); }
      }
      delete vertex_to_element;
   }
   else
   {
      for (int vertex = 0; vertex < mesh_->GetNV(); vertex++)
      {
         const real_t *coordinates = mesh_->GetVertex(vertex);
         for (int d = 0; d < point.Size(); d++) { point[d] = coordinates[d]; }
         if (predicate(point)) { sets_[VERTEX][set_index].insert(vertex); }
      }
   }

   return set_index;
}

void EntitySets::FindNearestVertices(const DenseMatrix &points,
                                     Array<int> &vertices,
                                     Array<real_t> &distances) const
{
   MFEM_VERIFY(points.Height() == mesh_->SpaceDimension(),
               "Point matrix height must equal the mesh space dimension.");
   MFEM_VERIFY(mesh_->GetNV() > 0 || points.Width() == 0,
               "Cannot find a nearest vertex in an empty mesh.");

   DenseMatrix coordinates;
   GetPhysicalVertexCoordinates(*mesh_, coordinates);

#ifdef MFEM_USE_MPI
   const ParMesh *pmesh = dynamic_cast<const ParMesh *>(mesh_);
   Array<long long> vertex_ids(mesh_->GetNV());
   if (pmesh)
   {
      Array<HYPRE_BigInt> global_ids;
      pmesh->GetGlobalVertexIndices(global_ids);
      for (int vertex = 0; vertex < vertex_ids.Size(); vertex++)
      {
         vertex_ids[vertex] = static_cast<long long>(global_ids[vertex]);
      }
   }
#endif

   ::mfem::FindNearestVertices(coordinates, points, vertices, distances);

#ifdef MFEM_USE_MPI
   if (pmesh)
   {
      for (int point = 0; point < points.Width(); point++)
      {
         real_t global_distance;
         MPI_Allreduce(&distances[point], &global_distance, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh->GetComm());
         long long local_id = distances[point] == global_distance ?
                              vertex_ids[vertices[point]] :
                              std::numeric_limits<long long>::max();
         long long global_id;
         MPI_Allreduce(&local_id, &global_id, 1, MPI_LONG_LONG, MPI_MIN,
                       pmesh->GetComm());

         vertices[point] = -1;
         for (int vertex = 0; vertex < vertex_ids.Size(); vertex++)
         {
            if (vertex_ids[vertex] == global_id)
            {
               vertices[point] = vertex;
               break;
            }
         }
         distances[point] = global_distance;
      }
   }
#endif
}

int EntitySets::AddVertexSetByNearestPoints(
   const string &name, const DenseMatrix &points, real_t max_distance,
   Array<int> *vertices, Array<real_t> *distances)
{
   MFEM_VERIFY(max_distance >= 0.0,
               "Maximum nearest-vertex distance must be nonnegative.");
   Array<int> local_vertices;
   Array<real_t> local_distances;
   FindNearestVertices(points, local_vertices, local_distances);
   for (int point = 0; point < local_distances.Size(); point++)
   {
      MFEM_VERIFY(local_distances[point] <= max_distance,
                  "No mesh vertex is within the requested maximum distance.");
   }

   const int set_index = AddSet(VERTEX, name);
   for (const int vertex : local_vertices)
   {
      if (vertex >= 0) { AddEntity(VERTEX, set_index, vertex); }
   }
   if (vertices) { *vertices = local_vertices; }
   if (distances) { *distances = local_distances; }
   return set_index;
}

int EntitySets::AddElementSetByAttribute(const string &name, int attribute)
{
   const int set_index = AddSet(ELEMENT, name);
   for (int element = 0; element < mesh_->GetNE(); element++)
   {
      if (mesh_->GetElement(element)->GetAttribute() == attribute)
      {
         sets_[ELEMENT][set_index].insert(element);
      }
   }
   return set_index;
}

int EntitySets::AddElementSetByMarker(const string &name,
                                      const Array<int> &marker)
{
   MFEM_VERIFY(marker.Size() == mesh_->GetNE(),
               "Element marker size must equal the number of local elements.");
   const int set_index = AddSet(ELEMENT, name);
   for (int element = 0; element < marker.Size(); element++)
   {
      if (marker[element]) { sets_[ELEMENT][set_index].insert(element); }
   }
   return set_index;
}

int EntitySets::AddEdgeSetFromElementSet(const string &name, int element_set)
{
   MFEM_VERIFY(mesh_->Dimension() > 1,
               "Edge sets require a mesh of dimension at least two.");
   VerifySetIndex(ELEMENT, element_set);
   const int set_index = AddSet(EDGE, name);
   Array<int> selected_incidence(mesh_->GetNEdges());
   Array<int> unselected_incidence(mesh_->GetNEdges());
   selected_incidence = 0;
   unselected_incidence = 0;

   for (int element = 0; element < mesh_->GetNE(); element++)
   {
      Array<int> edges, orientations;
      mesh_->GetElementEdges(element, edges, orientations);
      Array<int> &incidence = sets_[ELEMENT][element_set].count(element) ?
                              selected_incidence : unselected_incidence;
      for (const int edge : edges) { incidence[edge]++; }
   }

   for (int edge = 0; edge < mesh_->GetNEdges(); edge++)
   {
      if (selected_incidence[edge] > 0 && unselected_incidence[edge] > 0)
      {
         sets_[EDGE][set_index].insert(edge);
      }
   }
   CopyMeshTables();
   return set_index;
}

int EntitySets::AddEdgeSetByBoundaryAttribute(const string &name, int attribute)
{
   MFEM_VERIFY(mesh_->Dimension() > 1,
               "Edge sets require a mesh of dimension at least two.");
   const int set_index = AddSet(EDGE, name);
   for (int boundary_element = 0; boundary_element < mesh_->GetNBE();
        boundary_element++)
   {
      if (mesh_->GetBdrAttribute(boundary_element) != attribute) { continue; }
      Array<int> edges, orientations;
      mesh_->GetBdrElementEdges(boundary_element, edges, orientations);
      for (const int edge : edges) { sets_[EDGE][set_index].insert(edge); }
   }
   CopyMeshTables();
   return set_index;
}

int EntitySets::AddEdgeSetByPredicate(const string &name,
                                      const PointPredicate &predicate,
                                      EdgeSetPointRule rule)
{
   MFEM_VERIFY(mesh_->Dimension() > 1,
               "Edge sets require a mesh of dimension at least two.");
   MFEM_VERIFY(predicate, "EntitySets point predicate is empty.");
   const int set_index = AddSet(EDGE, name);
   Vector point(mesh_->SpaceDimension());

   for (int edge = 0; edge < mesh_->GetNEdges(); edge++)
   {
      Array<int> vertices;
      mesh_->GetEdgeVertices(edge, vertices);
      const real_t *v0 = mesh_->GetVertex(vertices[0]);
      const real_t *v1 = mesh_->GetVertex(vertices[1]);
      bool selected = false;

      switch (rule)
      {
         case EdgeSetPointRule::CENTER:
            for (int d = 0; d < point.Size(); d++)
            {
               point[d] = 0.5 * (v0[d] + v1[d]);
            }
            selected = predicate(point);
            break;

         case EdgeSetPointRule::ALL_VERTICES:
            for (int d = 0; d < point.Size(); d++) { point[d] = v0[d]; }
            selected = predicate(point);
            if (selected)
            {
               for (int d = 0; d < point.Size(); d++) { point[d] = v1[d]; }
               selected = predicate(point);
            }
            break;

         default:
            MFEM_ABORT("Unknown edge-set point selection rule.");
      }

      if (selected) { sets_[EDGE][set_index].insert(edge); }
   }
   CopyMeshTables();
   return set_index;
}

int EntitySets::AddFaceSet(const string &name, const Array<int> &faces)
{
   MFEM_VERIFY(mesh_->Dimension() == 3,
               "Face sets require a three-dimensional mesh.");
   const int set_index = AddSet(FACE, name);
   for (const int face : faces)
   {
      AddEntity(FACE, set_index, face);
   }
   return set_index;
}

int EntitySets::AddFaceSetByMarker(const string &name,
                                   const Array<int> &marker)
{
   MFEM_VERIFY(mesh_->Dimension() == 3,
               "Face sets require a three-dimensional mesh.");
   MFEM_VERIFY(marker.Size() == mesh_->GetNumFaces(),
               "Face marker size must equal the number of local faces.");
   const int set_index = AddSet(FACE, name);
   for (int face = 0; face < marker.Size(); face++)
   {
      if (marker[face]) { sets_[FACE][set_index].insert(face); }
   }
   CopyMeshTables();
   return set_index;
}

int EntitySets::AddFaceSetFromElementSet(const string &name, int element_set)
{
   MFEM_VERIFY(mesh_->Dimension() == 3,
               "Face sets require a three-dimensional mesh.");
   VerifySetIndex(ELEMENT, element_set);
   const int set_index = AddSet(FACE, name);
   Array<int> selected_incidence(mesh_->GetNumFaces());
   Array<int> unselected_incidence(mesh_->GetNumFaces());
   selected_incidence = 0;
   unselected_incidence = 0;

   for (int element = 0; element < mesh_->GetNE(); element++)
   {
      Array<int> faces, orientations;
      mesh_->GetElementFaces(element, faces, orientations);
      Array<int> &incidence = sets_[ELEMENT][element_set].count(element) ?
                              selected_incidence : unselected_incidence;
      for (const int face : faces) { incidence[face]++; }
   }

   for (int face = 0; face < mesh_->GetNumFaces(); face++)
   {
      if (selected_incidence[face] > 0 && unselected_incidence[face] > 0)
      {
         sets_[FACE][set_index].insert(face);
      }
   }
   CopyMeshTables();
   return set_index;
}

int EntitySets::AddFaceSetByBoundaryAttribute(const string &name, int attribute)
{
   MFEM_VERIFY(mesh_->Dimension() == 3,
               "Face sets require a three-dimensional mesh.");
   const int set_index = AddSet(FACE, name);
   for (int boundary_element = 0; boundary_element < mesh_->GetNBE();
        boundary_element++)
   {
      if (mesh_->GetBdrAttribute(boundary_element) != attribute) { continue; }
      sets_[FACE][set_index].insert(
         mesh_->GetBdrElementFaceIndex(boundary_element));
   }
   CopyMeshTables();
   return set_index;
}

int EntitySets::AddFaceSetByPredicate(const string &name,
                                      const PointPredicate &predicate,
                                      FaceSetPointRule rule)
{
   MFEM_VERIFY(mesh_->Dimension() == 3,
               "Face sets require a three-dimensional mesh.");
   MFEM_VERIFY(predicate, "EntitySets point predicate is empty.");
   const int set_index = AddSet(FACE, name);
   Vector point(mesh_->SpaceDimension());

   for (int face = 0; face < mesh_->GetNumFaces(); face++)
   {
      ElementTransformation *trans = mesh_->GetFaceTransformation(face);
      const Geometry::Type geometry = mesh_->GetFaceGeometry(face);
      bool selected = false;

      switch (rule)
      {
         case FaceSetPointRule::CENTER:
            trans->Transform(Geometries.GetCenter(geometry), point);
            selected = predicate(point);
            break;

         case FaceSetPointRule::ALL_VERTICES:
         {
            const IntegrationRule *vertices = Geometries.GetVertices(geometry);
            selected = true;
            for (int vertex = 0; vertex < vertices->GetNPoints(); vertex++)
            {
               trans->Transform(vertices->IntPoint(vertex), point);
               if (!predicate(point))
               {
                  selected = false;
                  break;
               }
            }
            break;
         }

         default:
            MFEM_ABORT("Unknown face-set point selection rule.");
      }

      if (selected) { sets_[FACE][set_index].insert(face); }
   }
   CopyMeshTables();
   return set_index;
}

int EntitySets::AddElementSetByPredicate(const string &name,
                                         const PointPredicate &predicate,
                                         ElementSetPointRule rule)
{
   MFEM_VERIFY(predicate, "EntitySets point predicate is empty.");
   const int set_index = AddSet(ELEMENT, name);
   Vector point(mesh_->SpaceDimension());

   for (int element = 0; element < mesh_->GetNE(); element++)
   {
      ElementTransformation *trans = mesh_->GetElementTransformation(element);
      const Geometry::Type geometry = mesh_->GetElementBaseGeometry(element);
      bool selected = false;

      switch (rule)
      {
         case ElementSetPointRule::CENTER:
            trans->Transform(Geometries.GetCenter(geometry), point);
            selected = predicate(point);
            break;

         case ElementSetPointRule::ALL_VERTICES:
         {
            const IntegrationRule *vertices = Geometries.GetVertices(geometry);
            selected = true;
            for (int vertex = 0; vertex < vertices->GetNPoints(); vertex++)
            {
               trans->Transform(vertices->IntPoint(vertex), point);
               if (!predicate(point))
               {
                  selected = false;
                  break;
               }
            }
            break;
         }

         default:
            MFEM_ABORT("Unknown element-set point selection rule.");
      }

      if (selected) { sets_[ELEMENT][set_index].insert(element); }
   }

   return set_index;
}

int EntitySets::AddElementSetByLevelSet(const string &name,
                                        Coefficient &level_set,
                                        real_t tolerance,
                                        ElementSetPointRule rule)
{
   const int set_index = AddSet(ELEMENT, name);

   for (int element = 0; element < mesh_->GetNE(); element++)
   {
      ElementTransformation *trans = mesh_->GetElementTransformation(element);
      const Geometry::Type geometry = mesh_->GetElementBaseGeometry(element);
      bool selected = false;

      switch (rule)
      {
         case ElementSetPointRule::CENTER:
         {
            const IntegrationPoint &center = Geometries.GetCenter(geometry);
            trans->SetIntPoint(&center);
            selected = level_set.Eval(*trans, center) <= tolerance;
            break;
         }

         case ElementSetPointRule::ALL_VERTICES:
         {
            const IntegrationRule *vertices = Geometries.GetVertices(geometry);
            selected = true;
            for (int vertex = 0; vertex < vertices->GetNPoints(); vertex++)
            {
               const IntegrationPoint &ip = vertices->IntPoint(vertex);
               trans->SetIntPoint(&ip);
               if (level_set.Eval(*trans, ip) > tolerance)
               {
                  selected = false;
                  break;
               }
            }
            break;
         }

         default:
            MFEM_ABORT("Unknown element-set point selection rule.");
      }

      if (selected) { sets_[ELEMENT][set_index].insert(element); }
   }

   return set_index;
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
      set<int> new_elems;
      for (int i = 0; i < mesh_->CoarseFineTr.embeddings.Size(); i++)
      {
         const int parent = mesh_->CoarseFineTr.embeddings[i].parent;
         if (sets_[t][s].count(parent)) { new_elems.insert(i); }
      }
      sets_[t][s].swap(new_elems);
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
            const int nv = face_vertex_->RowSize(old_face);
            MFEM_VERIFY(nv == 3 || nv == 4, "invalid face geometry");

            int mid[4];
            for (int j = 0; j < nv; j++)
            {
               mid[j] = oedge + e[j];
            }

            // A face table does not preserve vertex orientation. Identify the
            // two incident old edges for each vertex instead of relying on an
            // ordering in face_vertex_.
            for (int j = 0; j < nv; j++)
            {
               int adjacent_mid[2];
               int nadjacent = 0;
               for (int k = 0; k < nv; k++)
               {
                  const int *ev = edge_vertex_->GetRow(e[k]);
                  if (ev[0] == v[j] || ev[1] == v[j])
                  {
                     adjacent_mid[nadjacent++] = mid[k];
                  }
               }
               MFEM_VERIFY(nadjacent == 2, "invalid face-edge connectivity");

               const int child = (nv == 3)
                                 ? faces_tbl->Index(v[j], adjacent_mid[0],
                                                    adjacent_mid[1])
                                 : (*faces_tbl)(v[j], adjacent_mid[0],
                                                oface + old_face,
                                                adjacent_mid[1]);
               MFEM_VERIFY(child >= 0, "refined face not found");
               new_set.insert(child);
            }

            if (nv == 3)
            {
               const int child = faces_tbl->Index(mid[0], mid[1], mid[2]);
               MFEM_VERIFY(child >= 0, "refined face not found");
               new_set.insert(child);
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
      set<int> new_elems;
      for (int i = 0; i < mesh_->CoarseFineTr.embeddings.Size(); i++)
      {
         const int parent = mesh_->CoarseFineTr.embeddings[i].parent;
         if (sets_[t][s].count(parent)) { new_elems.insert(i); }
      }
      sets_[t][s].swap(new_elems);
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

   // ELEMENT sets contain only current entities, so replace each selected
   // parent with all of its children. The embedding table handles every 2D
   // geometry and the numbering used by Mesh::UniformRefinement2D_base().
   EntityType t = ELEMENT;
   for (unsigned int s=0; s<this->GetNumSets(t); s++)
   {
      set<int> new_elems;
      for (int i = 0; i < mesh_->CoarseFineTr.embeddings.Size(); i++)
      {
         const int parent = mesh_->CoarseFineTr.embeddings[i].parent;
         if (sets_[t][s].count(parent)) { new_elems.insert(i); }
      }
      sets_[t][s].swap(new_elems);
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
            int *v = face_vertex_->GetRow(old_face);
            int *e = face_edge_->GetRow(old_face);
            const int nv = face_vertex_->RowSize(old_face);
            MFEM_VERIFY(nv == 3 || nv == 4, "invalid face geometry");

            int mid[4];
            for (int j = 0; j < nv; j++) { mid[j] = oedge + e[j]; }

            // The face tables are unordered. Determine a corner's two child
            // edges by incidence, which works for triangular and quad faces.
            for (int j = 0; j < nv; j++)
            {
               int adjacent_mid[2];
               int nadjacent = 0;
               for (int k = 0; k < nv; k++)
               {
                  const int *ev = edge_vertex_->GetRow(e[k]);
                  if (ev[0] == v[j] || ev[1] == v[j])
                  {
                     adjacent_mid[nadjacent++] = mid[k];
                  }
               }
               MFEM_VERIFY(nadjacent == 2, "invalid face-edge connectivity");

               const int child = (nv == 3)
                                 ? faces_tbl->Index(v[j], adjacent_mid[0],
                                                    adjacent_mid[1])
                                 : (*faces_tbl)(v[j], adjacent_mid[0],
                                                oface + old_face,
                                                adjacent_mid[1]);
               MFEM_VERIFY(child >= 0, "refined face not found");
               new_set.insert(child);
            }

            if (nv == 3)
            {
               const int child = faces_tbl->Index(mid[0], mid[1], mid[2]);
               MFEM_VERIFY(child >= 0, "refined face not found");
               new_set.insert(child);
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

   // See UniformRefinement2D(): the embedding table gives the exact children
   // for tetrahedra, pyramids, wedges, and hexahedra, including mixed meshes.
   EntityType t = ELEMENT;
   for (unsigned int s=0; s<this->GetNumSets(t); s++)
   {
      set<int> new_elems;
      for (int i = 0; i < mesh_->CoarseFineTr.embeddings.Size(); i++)
      {
         const int parent = mesh_->CoarseFineTr.embeddings[i].parent;
         if (sets_[t][s].count(parent)) { new_elems.insert(i); }
      }
      sets_[t][s].swap(new_elems);
   }

   this->CopyMeshTables();
}
const int NCEntitySets::entity_size_[] = {1,2,4,1};

NCEntitySets::NCEntitySets(const EntitySets &ent_sets, NCMesh &ncmesh)
   : ncmesh_(&ncmesh),
     sets_(4),
     set_names_(4),
     set_index_by_name_(4)
{
   EntitySets::EntityType t = EntitySets::INVALID;
   unsigned int ns = -1;

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
         // NCMesh root nodes retain the original Mesh vertex indices across
         // parallel rebalancing; HashTable node indices do not.
         sets_[t][s][i] = *it;
      }
   }

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
            int e01 = ncmesh.nodes.FindId(ncmesh.vertex_nodeId[v[0]],
                                          ncmesh.vertex_nodeId[v[1]]);
            int e03 = ncmesh.nodes.FindId(ncmesh.vertex_nodeId[v[0]],
                                          ncmesh.vertex_nodeId[v[3]]);

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
}

NCEntitySets::NCEntitySets(const NCEntitySets & ncent_sets)
   : ncmesh_(ncent_sets.ncmesh_),
     sets_(4),
     set_names_(4),
     set_index_by_name_(4)
{
   this->CopyNCEntitySets(ncent_sets, EntitySets::VERTEX);
   this->CopyNCEntitySets(ncent_sets, EntitySets::EDGE);
   this->CopyNCEntitySets(ncent_sets, EntitySets::FACE);
   this->CopyNCEntitySets(ncent_sets, EntitySets::ELEMENT);
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
