// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mesh_headers.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/text.hpp"

#include <string>
#include <cmath>
#include <map>

#include "ncmesh_tables.hpp"


namespace
{
/**
 * @brief Base case of convenience variadic max function.
 *
 * @tparam T Base type
 * @param arg Recursion base value
 * @return T value to max over
 */
template<typename T>
T max(T&& arg)
{
   return arg;
}
/**
 * @brief Convenience variadic max function.
 *
 * @tparam T Base Type
 * @tparam Ts Parameter pack of other types
 * @param arg Singular argument
 * @param args Pack of arguments
 * @return T maximum value
 */
template<typename T, typename... Ts>
T max(T arg, Ts... args)
{
   return std::max(std::forward<T>(arg), max(args...));
}
} // namespace
namespace mfem
{

NCMesh::GeomInfo NCMesh::GI[Geometry::NumGeom];

void NCMesh::GeomInfo::InitGeom(Geometry::Type geom)
{
   if (initialized) { return; }

   auto elem = [&]()
   {
      switch (geom)
      {
         case Geometry::CUBE: return std::unique_ptr<mfem::Element>(new Hexahedron);
         case Geometry::PRISM: return std::unique_ptr<mfem::Element>(new Wedge);
         case Geometry::TETRAHEDRON: return std::unique_ptr<mfem::Element>
                                               (new Tetrahedron);
         case Geometry::PYRAMID: return std::unique_ptr<mfem::Element>(new Pyramid);
         case Geometry::SQUARE: return std::unique_ptr<mfem::Element>(new Quadrilateral);
         case Geometry::TRIANGLE: return std::unique_ptr<mfem::Element>(new Triangle);
         case Geometry::SEGMENT: return std::unique_ptr<mfem::Element>(new Segment);
         default: MFEM_ABORT("unsupported geometry " << geom);
      }
   }();

   nv = elem->GetNVertices();
   ne = elem->GetNEdges();
   nf = elem->GetNFaces();
   for (int i = 0; i < ne; i++)
   {
      for (int j = 0; j < 2; j++)
      {
         edges[i][j] = elem->GetEdgeVertices(i)[j];
      }
   }
   for (int i = 0; i < nf; i++)
   {
      nfv[i] = elem->GetNFaceVertices(i);

      faces[i][3] = 7; // invalid node index for 3-node faces
      for (int j = 0; j < nfv[i]; j++)
      {
         faces[i][j] = elem->GetFaceVertices(i)[j];
      }
   }

   // in 1D/2D we pretend to have faces too, so we can use NCMesh::Face::elem[2]
   if (!nf)
   {
      if (ne)
      {
         for (int i = 0; i < ne; i++)
         {
            // make a degenerate face
            faces[i][0] = faces[i][1] = edges[i][0];
            faces[i][2] = faces[i][3] = edges[i][1];
            nfv[i] = 2;
         }
         nf = ne;
      }
      else
      {
         for (int i = 0; i < nv; i++)
         {
            // 1D degenerate face
            faces[i][0] = faces[i][1] = faces[i][2] = faces[i][3] = i;
            nfv[i] = 1;
         }
         nf = nv;
      }
   }

   initialized = true;
}

NCMesh::NCMesh(const Mesh *mesh)
   : shadow(1024, 2048)
{
   Dim = mesh->Dimension();
   spaceDim = mesh->SpaceDimension();
   MyRank = 0;
   Iso = true;
   Legacy = false;

   // create the NCMesh::Element struct for each Mesh element
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const mfem::Element *elem = mesh->GetElement(i);

      Geometry::Type geom = elem->GetGeometryType();
      CheckSupportedGeom(geom);
      GI[geom].InitGeom(geom);

      // if we have pyramids we will need tets after refinement
      if (geom == Geometry::PYRAMID)
      {
         GI[Geometry::TETRAHEDRON].InitGeom(Geometry::TETRAHEDRON);
      }

      // create NCMesh::Element for this mfem::Element
      int root_id = AddElement(geom, elem->GetAttribute());
      MFEM_ASSERT(root_id == i, "");
      Element &root_elem = elements[root_id];

      const int *v = elem->GetVertices();
      for (int j = 0; j < GI[geom].nv; j++)
      {
         int id = v[j];
         root_elem.node[j] = id;
         nodes.Alloc(id, id, id);
         // NOTE: top-level nodes are special: id == p1 == p2 == orig. vertex id
      }
   }

   // if the user initialized any hanging nodes with Mesh::AddVertexParents,
   // copy the hierarchy now
   if (mesh->tmp_vertex_parents.Size())
   {
      for (const auto &triple : mesh->tmp_vertex_parents)
      {
         nodes.Reparent(triple.one, triple.two, triple.three);
      }
   }

   // create edge nodes and faces
   nodes.UpdateUnused();
   for (int i = 0; i < elements.Size(); i++)
   {
      // increase reference count of all nodes the element is using (NOTE: this
      // will also create and reference all edge nodes and faces)
      ReferenceElement(i);

      // make links from faces back to the element
      RegisterFaces(i);
   }

   // store boundary element attributes
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const mfem::Element *be = mesh->GetBdrElement(i);
      const int *v = be->GetVertices();

      Face* face = NULL;
      switch (be->GetType())
      {
         case mfem::Element::QUADRILATERAL:
            face = faces.Find(v[0], v[1], v[2], v[3]);
            break;
         case mfem::Element::TRIANGLE:
            face = faces.Find(v[0], v[1], v[2]);
            break;
         case mfem::Element::SEGMENT:
            face = faces.Find(v[0], v[0], v[1], v[1]);
            break;
         case mfem::Element::POINT:
            face = faces.Find(v[0], v[0], v[0], v[0]);
            break;
         default:
            MFEM_ABORT("Unsupported boundary element geometry.");
      }

      MFEM_VERIFY(face, "Boundary face not found.");
      face->attribute = be->GetAttribute();
   }

   // copy top-level vertex coordinates (leave empty if the mesh is curved)
   if (!mesh->Nodes)
   {
      coordinates.SetSize(3*mesh->GetNV());
      for (int i = 0; i < mesh->GetNV(); i++)
      {
         std::memcpy(&coordinates[3*i], mesh->GetVertex(i), 3*sizeof(real_t));
      }
   }

   InitRootState(mesh->GetNE());
   InitGeomFlags();

   Update();
}

NCMesh::NCMesh(const NCMesh &other)
   : Dim(other.Dim)
   , spaceDim(other.spaceDim)
   , MyRank(other.MyRank)
   , Iso(other.Iso)
   , Geoms(other.Geoms)
   , Legacy(other.Legacy)
   , nodes(other.nodes)
   , faces(other.faces)
   , elements(other.elements)
   , free_element_ids(other.free_element_ids)
   , root_state(other.root_state)
   , coordinates(other.coordinates)
   , NEdges(other.NEdges)
   , NFaces(other.NFaces)
   , NGhostEdges(other.NGhostEdges)
   , NGhostFaces(other.NGhostFaces)
   , boundary_faces(other.boundary_faces)
   , face_geom(other.face_geom)
   , element_vertex(other.element_vertex)
   , shadow(1024, 2048)
{
   Update();
}

void NCMesh::InitGeomFlags()
{
   Geoms = 0;
   for (int i = 0; i < root_state.Size(); i++)
   {
      Geoms |= (1 << elements[i].Geom());
   }
}

void NCMesh::Update()
{
   UpdateLeafElements();
   UpdateVertices();

   vertex_list.Clear();
   face_list.Clear();
   edge_list.Clear();

   element_vertex.Clear();
}

NCMesh::~NCMesh()
{
#ifdef MFEM_DEBUG
   // sign off of all faces and nodes
   Array<int> elemFaces;
   for (int i = 0; i < elements.Size(); i++)
   {
      if (elements[i].IsLeaf())
      {
         elemFaces.SetSize(0);
         UnreferenceElement(i, elemFaces);
         DeleteUnusedFaces(elemFaces);
      }
   }
   // NOTE: in release mode, we just throw away all faces and nodes at once
#endif
}

NCMesh::Node::~Node()
{
   MFEM_ASSERT(!vert_refc && !edge_refc, "node was not unreferenced properly, "
               "vert_refc: " << (int) vert_refc << ", edge_refc: "
               << (int) edge_refc);
}

void NCMesh::ReparentNode(int node, int new_p1, int new_p2)
{
   Node &nd = nodes[node];
   int old_p1 = nd.p1, old_p2 = nd.p2;

   // assign new parents
   nodes.Reparent(node, new_p1, new_p2);

   MFEM_ASSERT(shadow.FindId(old_p1, old_p2) < 0,
               "shadow node already exists");

   // store old parent pair temporarily in 'shadow'
   int sh = shadow.GetId(old_p1, old_p2);
   shadow[sh].vert_index = node;
}

int NCMesh::FindMidEdgeNode(int node1, int node2) const
{
   int mid = nodes.FindId(node1, node2);
   if (mid < 0 && shadow.Size())
   {
      // if (anisotropic) refinement is underway, some nodes may temporarily be
      // available under alternate parents (see ReparentNode)
      mid = shadow.FindId(node1, node2);
      if (mid >= 0)
      {
         mid = shadow[mid].vert_index; // index of the original node
      }
   }
   return mid;
}

int NCMesh::GetMidEdgeNode(int node1, int node2)
{
   int mid = FindMidEdgeNode(node1, node2);
   if (mid < 0) { mid = nodes.GetId(node1, node2); } // create if not found
   return mid;
}

int NCMesh::GetMidFaceNode(int en1, int en2, int en3, int en4)
{
   // mid-face node can be created either from (en1, en3) or from (en2, en4)
   int midf = FindMidEdgeNode(en1, en3);
   if (midf >= 0) { return midf; }
   return nodes.GetId(en2, en4);
}

void NCMesh::ReferenceElement(int elem)
{
   const Element &el = elements[elem];
   const int* node = el.node;
   GeomInfo& gi = GI[el.Geom()];

   // reference all vertices
   for (int i = 0; i < gi.nv; i++)
   {
      nodes[node[i]].vert_refc++;
   }

   // reference all edges (possibly creating their nodes)
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      nodes.Get(node[ev[0]], node[ev[1]])->edge_refc++;
   }

   // get all faces (possibly creating them)
   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      faces.GetId(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]]);

      // NOTE: face->RegisterElement called separately to avoid having to store
      // 3 element indices  temporarily in the face when refining. See also
      // NCMesh::RegisterFaces.
   }
}

void NCMesh::UnreferenceElement(int elem, Array<int> &elemFaces)
{
   Element &el = elements[elem];
   int* node = el.node;
   GeomInfo& gi = GI[el.Geom()];

   // unreference all faces
   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      int face = faces.FindId(node[fv[0]], node[fv[1]],
                              node[fv[2]], node[fv[3]]);
      MFEM_ASSERT(face >= 0, "face not found.");
      faces[face].ForgetElement(elem);

      // NOTE: faces.Delete() called later to avoid destroying and recreating
      // faces during refinement, see NCMesh::DeleteUnusedFaces.
      elemFaces.Append(face);
   }

   // unreference all edges (possibly destroying them)
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      int enode = FindMidEdgeNode(node[ev[0]], node[ev[1]]);
      MFEM_ASSERT(enode >= 0, "edge not found.");
      MFEM_ASSERT(nodes.IdExists(enode), "edge does not exist.");
      if (!nodes[enode].UnrefEdge())
      {
         nodes.Delete(enode);
      }
   }

   // unreference all vertices (possibly destroying them)
   for (int i = 0; i < gi.nv; i++)
   {
      if (!nodes[node[i]].UnrefVertex())
      {
         nodes.Delete(node[i]);
      }
   }
}

void NCMesh::RegisterFaces(int elem, int* fattr)
{
   Element &el = elements[elem];
   GeomInfo &gi = GI[el.Geom()];

   for (int i = 0; i < gi.nf; i++)
   {
      Face* face = GetFace(el, i);
      MFEM_ASSERT(face, "face not found.");
      face->RegisterElement(elem);
      if (fattr) { face->attribute = fattr[i]; }
   }
}

void NCMesh::DeleteUnusedFaces(const Array<int> &elemFaces)
{
   for (int i = 0; i < elemFaces.Size(); i++)
   {
      if (faces[elemFaces[i]].Unused())
      {
         faces.Delete(elemFaces[i]);
      }
   }
}

void NCMesh::Face::RegisterElement(int e)
{
   if (elem[0] < 0) { elem[0] = e; }
   else if (elem[1] < 0) { elem[1] = e; }
   else { MFEM_ABORT("can't have 3 elements in Face::elem[]."); }
}

void NCMesh::Face::ForgetElement(int e)
{
   if (elem[0] == e) { elem[0] = -1; }
   else if (elem[1] == e) { elem[1] = -1; }
   else { MFEM_ABORT("element " << e << " not found in Face::elem[]."); }
}

NCMesh::Face* NCMesh::GetFace(Element &elem, int face_no)
{
   GeomInfo& gi = GI[(int) elem.geom];
   const int* fv = gi.faces[face_no];
   int* node = elem.node;
   return faces.Find(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]]);
}

int NCMesh::Face::GetSingleElement() const
{
   if (elem[0] >= 0)
   {
      MFEM_ASSERT(elem[1] < 0, "not a single element face.");
      return elem[0];
   }
   else
   {
      MFEM_ASSERT(elem[1] >= 0, "no elements in face.");
      return elem[1];
   }
}


//// Refinement ////////////////////////////////////////////////////////////////

NCMesh::Element::Element(Geometry::Type geom, int attr)
   : geom(geom), ref_type(0), tet_type(0), flag(0), index(-1)
   , rank(0), attribute(attr), parent(-1)
{
   for (int i = 0; i < MaxElemNodes; i++) { node[i] = -1; }
   for (int i = 0; i < MaxElemChildren; i++) { child[i] = -1; }

   // NOTE: in 2D the 8/10-element node/child arrays are not optimal, however,
   // testing shows we would only save 17% of the total NCMesh memory if
   // 4-element arrays were used (e.g. through templates); we thus prefer to
   // keep the code as simple as possible.
}

int NCMesh::NewHexahedron(int n0, int n1, int n2, int n3,
                          int n4, int n5, int n6, int n7,
                          int attr,
                          int fattr0, int fattr1, int fattr2,
                          int fattr3, int fattr4, int fattr5)
{
   // create new element, initialize nodes
   int new_id = AddElement(Geometry::CUBE, attr);
   Element &el = elements[new_id];

   el.node[0] = n0, el.node[1] = n1, el.node[2] = n2, el.node[3] = n3;
   el.node[4] = n4, el.node[5] = n5, el.node[6] = n6, el.node[7] = n7;

   // get faces and assign face attributes
   Face* f[MaxElemFaces];
   const GeomInfo &gi_hex = GI[Geometry::CUBE];
   for (int i = 0; i < gi_hex.nf; i++)
   {
      const int* fv = gi_hex.faces[i];
      f[i] = faces.Get(el.node[fv[0]], el.node[fv[1]],
                       el.node[fv[2]], el.node[fv[3]]);
   }

   f[0]->attribute = fattr0,  f[1]->attribute = fattr1;
   f[2]->attribute = fattr2,  f[3]->attribute = fattr3;
   f[4]->attribute = fattr4,  f[5]->attribute = fattr5;

   return new_id;
}

int NCMesh::NewWedge(int n0, int n1, int n2,
                     int n3, int n4, int n5,
                     int attr,
                     int fattr0, int fattr1,
                     int fattr2, int fattr3, int fattr4)
{
   // create new element, initialize nodes
   int new_id = AddElement(Geometry::PRISM, attr);
   Element &el = elements[new_id];

   el.node[0] = n0, el.node[1] = n1, el.node[2] = n2;
   el.node[3] = n3, el.node[4] = n4, el.node[5] = n5;

   // get faces and assign face attributes
   Face* f[5];
   const GeomInfo &gi_wedge = GI[Geometry::PRISM];
   for (int i = 0; i < gi_wedge.nf; i++)
   {
      const int* fv = gi_wedge.faces[i];
      f[i] = faces.Get(el.node[fv[0]], el.node[fv[1]],
                       el.node[fv[2]], el.node[fv[3]]);
   }

   f[0]->attribute = fattr0;
   f[1]->attribute = fattr1;
   f[2]->attribute = fattr2;
   f[3]->attribute = fattr3;
   f[4]->attribute = fattr4;

   return new_id;
}

int NCMesh::NewTetrahedron(int n0, int n1, int n2, int n3, int attr,
                           int fattr0, int fattr1, int fattr2, int fattr3)
{
   // create new element, initialize nodes
   int new_id = AddElement(Geometry::TETRAHEDRON, attr);
   Element &el = elements[new_id];

   el.node[0] = n0, el.node[1] = n1, el.node[2] = n2, el.node[3] = n3;

   // get faces and assign face attributes
   Face* f[4];
   const GeomInfo &gi_tet = GI[Geometry::TETRAHEDRON];
   for (int i = 0; i < gi_tet.nf; i++)
   {
      const int* fv = gi_tet.faces[i];
      f[i] = faces.Get(el.node[fv[0]], el.node[fv[1]], el.node[fv[2]]);
   }

   f[0]->attribute = fattr0;
   f[1]->attribute = fattr1;
   f[2]->attribute = fattr2;
   f[3]->attribute = fattr3;

   return new_id;
}
int NCMesh::NewPyramid(int n0, int n1, int n2, int n3, int n4, int attr,
                       int fattr0, int fattr1, int fattr2, int fattr3,
                       int fattr4)
{
   // create new element, initialize nodes
   int new_id = AddElement(Geometry::PYRAMID, attr);
   Element &el = elements[new_id];

   el.node[0] = n0, el.node[1] = n1, el.node[2] = n2, el.node[3] = n3;
   el.node[4] = n4;

   // get faces and assign face attributes
   Face* f[5];
   const GeomInfo &gi_pyr = GI[Geometry::PYRAMID];
   for (int i = 0; i < gi_pyr.nf; i++)
   {
      const int* fv = gi_pyr.faces[i];
      f[i] = faces.Get(el.node[fv[0]], el.node[fv[1]],
                       el.node[fv[2]], el.node[fv[3]]);
   }

   f[0]->attribute = fattr0;
   f[1]->attribute = fattr1;
   f[2]->attribute = fattr2;
   f[3]->attribute = fattr3;
   f[4]->attribute = fattr4;

   return new_id;
}

int NCMesh::NewQuadrilateral(int n0, int n1, int n2, int n3,
                             int attr,
                             int eattr0, int eattr1, int eattr2, int eattr3)
{
   // create new element, initialize nodes
   int new_id = AddElement(Geometry::SQUARE, attr);
   Element &el = elements[new_id];

   el.node[0] = n0, el.node[1] = n1, el.node[2] = n2, el.node[3] = n3;

   // get (degenerate) faces and assign face attributes
   Face* f[4];
   const GeomInfo &gi_quad = GI[Geometry::SQUARE];
   for (int i = 0; i < gi_quad.nf; i++)
   {
      const int* fv = gi_quad.faces[i];
      f[i] = faces.Get(el.node[fv[0]], el.node[fv[1]],
                       el.node[fv[2]], el.node[fv[3]]);
   }

   f[0]->attribute = eattr0,  f[1]->attribute = eattr1;
   f[2]->attribute = eattr2,  f[3]->attribute = eattr3;

   return new_id;
}

int NCMesh::NewTriangle(int n0, int n1, int n2,
                        int attr, int eattr0, int eattr1, int eattr2)
{
   // create new element, initialize nodes
   int new_id = AddElement(Geometry::TRIANGLE, attr);
   Element &el = elements[new_id];

   el.node[0] = n0, el.node[1] = n1, el.node[2] = n2;

   // get (degenerate) faces and assign face attributes
   Face* f[3];
   const GeomInfo &gi_tri = GI[Geometry::TRIANGLE];
   for (int i = 0; i < gi_tri.nf; i++)
   {
      const int* fv = gi_tri.faces[i];
      f[i] = faces.Get(el.node[fv[0]], el.node[fv[1]],
                       el.node[fv[2]], el.node[fv[3]]);
   }

   f[0]->attribute = eattr0;
   f[1]->attribute = eattr1;
   f[2]->attribute = eattr2;

   return new_id;
}

int NCMesh::NewSegment(int n0, int n1, int attr, int vattr1, int vattr2)
{
   // create new element, initialize nodes
   int new_id = AddElement(Geometry::SEGMENT, attr);
   Element &el = elements[new_id];
   el.node[0] = n0, el.node[1] = n1;

   // get (degenerate) faces and assign face attributes
   int v0 = el.node[0], v1 = el.node[1];
   faces.Get(v0, v0, v0, v0)->attribute = vattr1;
   faces.Get(v1, v1, v1, v1)->attribute = vattr2;

   return new_id;
}

inline bool CubeFaceLeft(int node, int* n)
{ return node == n[0] || node == n[3] || node == n[4] || node == n[7]; }

inline bool CubeFaceRight(int node, int* n)
{ return node == n[1] || node == n[2] || node == n[5] || node == n[6]; }

inline bool CubeFaceFront(int node, int* n)
{ return node == n[0] || node == n[1] || node == n[4] || node == n[5]; }

inline bool CubeFaceBack(int node, int* n)
{ return node == n[2] || node == n[3] || node == n[6] || node == n[7]; }

inline bool CubeFaceBottom(int node, int* n)
{ return node == n[0] || node == n[1] || node == n[2] || node == n[3]; }

inline bool CubeFaceTop(int node, int* n)
{ return node == n[4] || node == n[5] || node == n[6] || node == n[7]; }

inline bool PrismFaceBottom(int node, int* n)
{ return node == n[0] || node == n[1] || node == n[2]; }

inline bool PrismFaceTop(int node, int* n)
{ return node == n[3] || node == n[4] || node == n[5]; }


void NCMesh::ForceRefinement(int vn1, int vn2, int vn3, int vn4)
{
   // get the element this face belongs to
   Face* face = faces.Find(vn1, vn2, vn3, vn4);
   if (!face) { return; }

   int elem = face->GetSingleElement();
   Element &el = elements[elem];
   MFEM_ASSERT(!el.ref_type, "element already refined.");

   int* el_nodes = el.node;
   if (el.Geom() == Geometry::CUBE)
   {
      // schedule the right split depending on face orientation
      if ((CubeFaceLeft(vn1, el_nodes) && CubeFaceRight(vn2, el_nodes)) ||
          (CubeFaceLeft(vn2, el_nodes) && CubeFaceRight(vn1, el_nodes)))
      {
         ref_stack.Append(Refinement(elem, 1)); // X split
      }
      else if ((CubeFaceFront(vn1, el_nodes) && CubeFaceBack(vn2, el_nodes)) ||
               (CubeFaceFront(vn2, el_nodes) && CubeFaceBack(vn1, el_nodes)))
      {
         ref_stack.Append(Refinement(elem, 2)); // Y split
      }
      else if ((CubeFaceBottom(vn1, el_nodes) && CubeFaceTop(vn2, el_nodes)) ||
               (CubeFaceBottom(vn2, el_nodes) && CubeFaceTop(vn1, el_nodes)))
      {
         ref_stack.Append(Refinement(elem, 4)); // Z split
      }
      else
      {
         MFEM_ABORT("Inconsistent element/face structure.");
      }
   }
   else if (el.Geom() == Geometry::PRISM)
   {
      if ((PrismFaceTop(vn1, el_nodes) && PrismFaceBottom(vn4, el_nodes)) ||
          (PrismFaceTop(vn4, el_nodes) && PrismFaceBottom(vn1, el_nodes)))
      {
         ref_stack.Append(Refinement(elem, 3)); // XY split
      }
      else if ((PrismFaceTop(vn1, el_nodes) && PrismFaceBottom(vn2, el_nodes)) ||
               (PrismFaceTop(vn2, el_nodes) && PrismFaceBottom(vn1, el_nodes)))
      {
         ref_stack.Append(Refinement(elem, 4)); // Z split
      }
      else
      {
         MFEM_ABORT("Inconsistent element/face structure.");
      }
   }
   else
   {
      MFEM_ABORT("Unsupported geometry.")
   }
}


void NCMesh::FindEdgeElements(int vn1, int vn2, int vn3, int vn4,
                              Array<MeshId> &elem_edge) const
{
   // Assuming that f = (vn1, vn2, vn3, vn4) is a quad face and e = (vn1, vn4)
   // is its edge, this function finds the N elements sharing e, and returns the
   // N different MeshIds of the edge (i.e., different element-local pairs
   // describing the edge).

   int ev1 = vn1, ev2 = vn4;

   // follow face refinement towards 'vn1', get an existing face
   int split, mid[5];
   while ((split = QuadFaceSplitType(vn1, vn2, vn3, vn4, mid)) > 0)
   {
      if (split == 1) // vertical
      {
         vn2 = mid[0]; vn3 = mid[2];
      }
      else // horizontal
      {
         vn3 = mid[1]; vn4 = mid[3];
      }
   }

   const Face *face = faces.Find(vn1, vn2, vn3, vn4);
   MFEM_ASSERT(face != NULL, "Face not found: "
               << vn1 << ", " << vn2 << ", " << vn3 << ", " << vn4
               << " (edge " << ev1 << "-" << ev2 << ").");

   int elem = face->GetSingleElement();
   int local = find_node(elements[elem], vn1);

   Array<int> cousins;
   FindVertexCousins(elem, local, cousins);

   elem_edge.SetSize(0);
   for (int i = 0; i < cousins.Size(); i++)
   {
      local = find_element_edge(elements[cousins[i]], ev1, ev2, false);
      if (local > 0)
      {
         elem_edge.Append(MeshId(-1, cousins[i], local, Geometry::SEGMENT));
      }
   }
}


void NCMesh::CheckAnisoPrism(int vn1, int vn2, int vn3, int vn4,
                             const Refinement *refs, int nref)
{
   MeshId buf[4];
   Array<MeshId> eid(buf, 4);
   FindEdgeElements(vn1, vn2, vn3, vn4, eid);

   // see if there is an element that has not been force-refined yet
   for (int i = 0, j; i < eid.Size(); i++)
   {
      int elem = eid[i].element;
      for (j = 0; j < nref; j++)
      {
         if (refs[j].index == elem) { break; }
      }
      if (j == nref) // elem not found in refs[]
      {
         // schedule prism refinement along Z axis
         MFEM_ASSERT(elements[elem].Geom() == Geometry::PRISM, "");
         ref_stack.Append(Refinement(elem, 4));
      }
   }
}


void NCMesh::CheckAnisoFace(int vn1, int vn2, int vn3, int vn4,
                            int mid12, int mid34, int level)
{
   // When a face is getting split anisotropically (without loss of generality
   // we assume a "vertical" split here, see picture), it is important to make
   // sure that the mid-face vertex node (midf) has mid34 and mid12 as parents.
   // This is necessary for the face traversal algorithm and at places like
   // Refine() that assume the mid-edge nodes to be accessible through the right
   // parents. However, midf may already exist under the parents mid41 and
   // mid23. In that case we need to "reparent" midf, i.e., reinsert it to the
   // hash-table under the correct parents. This doesn't affect other nodes as
   // all IDs stay the same, only the face refinement "tree" is affected.
   //
   //                     vn4      mid34      vn3
   //                        *------*------*
   //                        |      |      |
   //                        |      |midf  |
   //                  mid41 *- - - *- - - * mid23
   //                        |      |      |
   //                        |      |      |
   //                        *------*------*
   //                    vn1      mid12      vn2
   //
   // This function is recursive, because the above applies to any node along
   // the middle vertical edge. The function calls itself again for the bottom
   // and upper half of the above picture.

   int mid23 = FindMidEdgeNode(vn2, vn3);
   int mid41 = FindMidEdgeNode(vn4, vn1);
   if (mid23 >= 0 && mid41 >= 0)
   {
      int midf = nodes.FindId(mid23, mid41);
      if (midf >= 0)
      {
         reparents.Append(Triple<int, int, int>(midf, mid12, mid34));

         int rs = ref_stack.Size();

         CheckAnisoFace(vn1, vn2, mid23, mid41, mid12, midf, level+1);
         CheckAnisoFace(mid41, mid23, vn3, vn4, midf, mid34, level+1);

         if (HavePrisms() && nodes[midf].HasEdge())
         {
            // Check if there is a prism with edge (mid23, mid41) that we may
            // have missed in 'CheckAnisoFace', and force-refine it if present.

            if (ref_stack.Size() > rs)
            {
               CheckAnisoPrism(mid23, vn3, vn4, mid41,
                               &ref_stack[rs], ref_stack.Size() - rs);
            }
            else
            {
               CheckAnisoPrism(mid23, vn3, vn4, mid41, NULL, 0);
            }
         }

         // perform the reparents all at once at the end
         if (level == 0)
         {
            for (int i = 0; i < reparents.Size(); i++)
            {
               const Triple<int, int, int> &tr = reparents[i];
               ReparentNode(tr.one, tr.two, tr.three);
            }
            reparents.DeleteAll();
         }
         return;
      }
   }

   // Also, this is the place where forced refinements begin. In the picture
   // above, edges mid12-midf and midf-mid34 should actually exist in the
   // neighboring elements, otherwise the mesh is inconsistent and needs to be
   // fixed. Example: suppose an element is being refined isotropically (!)
   // whose neighbors across some face look like this:
   //
   //                         *--------*--------*
   //                         |   d    |    e   |
   //                         *--------*--------*
   //                         |      c          |
   //                         *--------*--------*
   //                         |        |        |
   //                         |   a    |    b   |
   //                         |        |        |
   //                         *--------*--------*
   //
   // Element 'c' needs to be refined vertically for the mesh to remain valid.

   if (level > 0)
   {
      ForceRefinement(vn1, vn2, vn3, vn4);
   }
}

void NCMesh::CheckIsoFace(int vn1, int vn2, int vn3, int vn4,
                          int en1, int en2, int en3, int en4, int midf)
{
   if (!Iso)
   {
      /* If anisotropic refinements are present in the mesh, we need to check
         isotropically split faces as well, see second comment in CheckAnisoFace
         above. */

      CheckAnisoFace(vn1, vn2, en2, en4, en1, midf);
      CheckAnisoFace(en4, en2, vn3, vn4, midf, en3);
      CheckAnisoFace(vn4, vn1, en1, en3, en4, midf);
      CheckAnisoFace(en3, en1, vn2, vn3, midf, en2);
   }
}


void NCMesh::RefineElement(int elem, char ref_type)
{
   if (!ref_type) { return; }

   // handle elements that may have been (force-) refined already
   Element &el = elements[elem];
   if (el.ref_type)
   {
      char remaining = ref_type & ~el.ref_type;

      // do the remaining splits on the children
      for (int i = 0; i < MaxElemChildren; i++)
      {
         if (el.child[i] >= 0) { RefineElement(el.child[i], remaining); }
      }
      return;
   }

   /*mfem::out << "Refining element " << elem << " (" << el.node[0] << ", " <<
             el.node[1] << ", " << el.node[2] << ", " << el.node[3] << ", " <<
             el.node[4] << ", " << el.node[5] << ", " << el.node[6] << ", " <<
             el.node[7] << "), " << "ref_type " << int(ref_type) << std::endl;*/

   int* no = el.node;
   int attr = el.attribute;

   int child[MaxElemChildren];
   for (int i = 0; i < MaxElemChildren; i++) { child[i] = -1; }

   // get parent's face attributes
   int fa[MaxElemFaces];
   GeomInfo& gi = GI[el.Geom()];
   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      Face* face = faces.Find(no[fv[0]], no[fv[1]], no[fv[2]], no[fv[3]]);
      fa[i] = face->attribute;
   }

   // create child elements
   if (el.Geom() == Geometry::CUBE)
   {
      // Cube vertex numbering is assumed to be as follows:
      //
      //       7             6
      //        +-----------+                Faces: 0 bottom
      //       /|          /|                       1 front
      //    4 / |       5 / |                       2 right
      //     +-----------+  |                       3 back
      //     |  |        |  |                       4 left
      //     |  +--------|--+                       5 top
      //     | / 3       | / 2       Z Y
      //     |/          |/          |/
      //     +-----------+           *--X
      //    0             1

      if (ref_type == Refinement::X) // split along X axis
      {
         int mid01 = GetMidEdgeNode(no[0], no[1]);
         int mid23 = GetMidEdgeNode(no[2], no[3]);
         int mid45 = GetMidEdgeNode(no[4], no[5]);
         int mid67 = GetMidEdgeNode(no[6], no[7]);

         child[0] = NewHexahedron(no[0], mid01, mid23, no[3],
                                  no[4], mid45, mid67, no[7], attr,
                                  fa[0], fa[1], -1, fa[3], fa[4], fa[5]);

         child[1] = NewHexahedron(mid01, no[1], no[2], mid23,
                                  mid45, no[5], no[6], mid67, attr,
                                  fa[0], fa[1], fa[2], fa[3], -1, fa[5]);

         CheckAnisoFace(no[0], no[1], no[5], no[4], mid01, mid45);
         CheckAnisoFace(no[2], no[3], no[7], no[6], mid23, mid67);
         CheckAnisoFace(no[4], no[5], no[6], no[7], mid45, mid67);
         CheckAnisoFace(no[3], no[2], no[1], no[0], mid23, mid01);
      }
      else if (ref_type == Refinement::Y) // split along Y axis
      {
         int mid12 = GetMidEdgeNode(no[1], no[2]);
         int mid30 = GetMidEdgeNode(no[3], no[0]);
         int mid56 = GetMidEdgeNode(no[5], no[6]);
         int mid74 = GetMidEdgeNode(no[7], no[4]);

         child[0] = NewHexahedron(no[0], no[1], mid12, mid30,
                                  no[4], no[5], mid56, mid74, attr,
                                  fa[0], fa[1], fa[2], -1, fa[4], fa[5]);

         child[1] = NewHexahedron(mid30, mid12, no[2], no[3],
                                  mid74, mid56, no[6], no[7], attr,
                                  fa[0], -1, fa[2], fa[3], fa[4], fa[5]);

         CheckAnisoFace(no[1], no[2], no[6], no[5], mid12, mid56);
         CheckAnisoFace(no[3], no[0], no[4], no[7], mid30, mid74);
         CheckAnisoFace(no[5], no[6], no[7], no[4], mid56, mid74);
         CheckAnisoFace(no[0], no[3], no[2], no[1], mid30, mid12);
      }
      else if (ref_type == Refinement::Z) // split along Z axis
      {
         int mid04 = GetMidEdgeNode(no[0], no[4]);
         int mid15 = GetMidEdgeNode(no[1], no[5]);
         int mid26 = GetMidEdgeNode(no[2], no[6]);
         int mid37 = GetMidEdgeNode(no[3], no[7]);

         child[0] = NewHexahedron(no[0], no[1], no[2], no[3],
                                  mid04, mid15, mid26, mid37, attr,
                                  fa[0], fa[1], fa[2], fa[3], fa[4], -1);

         child[1] = NewHexahedron(mid04, mid15, mid26, mid37,
                                  no[4], no[5], no[6], no[7], attr,
                                  -1, fa[1], fa[2], fa[3], fa[4], fa[5]);

         CheckAnisoFace(no[4], no[0], no[1], no[5], mid04, mid15);
         CheckAnisoFace(no[5], no[1], no[2], no[6], mid15, mid26);
         CheckAnisoFace(no[6], no[2], no[3], no[7], mid26, mid37);
         CheckAnisoFace(no[7], no[3], no[0], no[4], mid37, mid04);
      }
      else if (ref_type == Refinement::XY) // XY split
      {
         int mid01 = GetMidEdgeNode(no[0], no[1]);
         int mid12 = GetMidEdgeNode(no[1], no[2]);
         int mid23 = GetMidEdgeNode(no[2], no[3]);
         int mid30 = GetMidEdgeNode(no[3], no[0]);

         int mid45 = GetMidEdgeNode(no[4], no[5]);
         int mid56 = GetMidEdgeNode(no[5], no[6]);
         int mid67 = GetMidEdgeNode(no[6], no[7]);
         int mid74 = GetMidEdgeNode(no[7], no[4]);

         int midf0 = GetMidFaceNode(mid23, mid12, mid01, mid30);
         int midf5 = GetMidFaceNode(mid45, mid56, mid67, mid74);

         child[0] = NewHexahedron(no[0], mid01, midf0, mid30,
                                  no[4], mid45, midf5, mid74, attr,
                                  fa[0], fa[1], -1, -1, fa[4], fa[5]);

         child[1] = NewHexahedron(mid01, no[1], mid12, midf0,
                                  mid45, no[5], mid56, midf5, attr,
                                  fa[0], fa[1], fa[2], -1, -1, fa[5]);

         child[2] = NewHexahedron(midf0, mid12, no[2], mid23,
                                  midf5, mid56, no[6], mid67, attr,
                                  fa[0], -1, fa[2], fa[3], -1, fa[5]);

         child[3] = NewHexahedron(mid30, midf0, mid23, no[3],
                                  mid74, midf5, mid67, no[7], attr,
                                  fa[0], -1, -1, fa[3], fa[4], fa[5]);

         CheckAnisoFace(no[0], no[1], no[5], no[4], mid01, mid45);
         CheckAnisoFace(no[1], no[2], no[6], no[5], mid12, mid56);
         CheckAnisoFace(no[2], no[3], no[7], no[6], mid23, mid67);
         CheckAnisoFace(no[3], no[0], no[4], no[7], mid30, mid74);

         CheckIsoFace(no[3], no[2], no[1], no[0], mid23, mid12, mid01, mid30, midf0);
         CheckIsoFace(no[4], no[5], no[6], no[7], mid45, mid56, mid67, mid74, midf5);
      }
      else if (ref_type == Refinement::XZ) // XZ split
      {
         int mid01 = GetMidEdgeNode(no[0], no[1]);
         int mid23 = GetMidEdgeNode(no[2], no[3]);
         int mid45 = GetMidEdgeNode(no[4], no[5]);
         int mid67 = GetMidEdgeNode(no[6], no[7]);

         int mid04 = GetMidEdgeNode(no[0], no[4]);
         int mid15 = GetMidEdgeNode(no[1], no[5]);
         int mid26 = GetMidEdgeNode(no[2], no[6]);
         int mid37 = GetMidEdgeNode(no[3], no[7]);

         int midf1 = GetMidFaceNode(mid01, mid15, mid45, mid04);
         int midf3 = GetMidFaceNode(mid23, mid37, mid67, mid26);

         child[0] = NewHexahedron(no[0], mid01, mid23, no[3],
                                  mid04, midf1, midf3, mid37, attr,
                                  fa[0], fa[1], -1, fa[3], fa[4], -1);

         child[1] = NewHexahedron(mid01, no[1], no[2], mid23,
                                  midf1, mid15, mid26, midf3, attr,
                                  fa[0], fa[1], fa[2], fa[3], -1, -1);

         child[2] = NewHexahedron(midf1, mid15, mid26, midf3,
                                  mid45, no[5], no[6], mid67, attr,
                                  -1, fa[1], fa[2], fa[3], -1, fa[5]);

         child[3] = NewHexahedron(mid04, midf1, midf3, mid37,
                                  no[4], mid45, mid67, no[7], attr,
                                  -1, fa[1], -1, fa[3], fa[4], fa[5]);

         CheckAnisoFace(no[3], no[2], no[1], no[0], mid23, mid01);
         CheckAnisoFace(no[2], no[6], no[5], no[1], mid26, mid15);
         CheckAnisoFace(no[6], no[7], no[4], no[5], mid67, mid45);
         CheckAnisoFace(no[7], no[3], no[0], no[4], mid37, mid04);

         CheckIsoFace(no[0], no[1], no[5], no[4], mid01, mid15, mid45, mid04, midf1);
         CheckIsoFace(no[2], no[3], no[7], no[6], mid23, mid37, mid67, mid26, midf3);
      }
      else if (ref_type == Refinement::YZ) // YZ split
      {
         int mid12 = GetMidEdgeNode(no[1], no[2]);
         int mid30 = GetMidEdgeNode(no[3], no[0]);
         int mid56 = GetMidEdgeNode(no[5], no[6]);
         int mid74 = GetMidEdgeNode(no[7], no[4]);

         int mid04 = GetMidEdgeNode(no[0], no[4]);
         int mid15 = GetMidEdgeNode(no[1], no[5]);
         int mid26 = GetMidEdgeNode(no[2], no[6]);
         int mid37 = GetMidEdgeNode(no[3], no[7]);

         int midf2 = GetMidFaceNode(mid12, mid26, mid56, mid15);
         int midf4 = GetMidFaceNode(mid30, mid04, mid74, mid37);

         child[0] = NewHexahedron(no[0], no[1], mid12, mid30,
                                  mid04, mid15, midf2, midf4, attr,
                                  fa[0], fa[1], fa[2], -1, fa[4], -1);

         child[1] = NewHexahedron(mid30, mid12, no[2], no[3],
                                  midf4, midf2, mid26, mid37, attr,
                                  fa[0], -1, fa[2], fa[3], fa[4], -1);

         child[2] = NewHexahedron(mid04, mid15, midf2, midf4,
                                  no[4], no[5], mid56, mid74, attr,
                                  -1, fa[1], fa[2], -1, fa[4], fa[5]);

         child[3] = NewHexahedron(midf4, midf2, mid26, mid37,
                                  mid74, mid56, no[6], no[7], attr,
                                  -1, -1, fa[2], fa[3], fa[4], fa[5]);

         CheckAnisoFace(no[4], no[0], no[1], no[5], mid04, mid15);
         CheckAnisoFace(no[0], no[3], no[2], no[1], mid30, mid12);
         CheckAnisoFace(no[3], no[7], no[6], no[2], mid37, mid26);
         CheckAnisoFace(no[7], no[4], no[5], no[6], mid74, mid56);

         CheckIsoFace(no[1], no[2], no[6], no[5], mid12, mid26, mid56, mid15, midf2);
         CheckIsoFace(no[3], no[0], no[4], no[7], mid30, mid04, mid74, mid37, midf4);
      }
      else if (ref_type == Refinement::XYZ) // full isotropic refinement
      {
         int mid01 = GetMidEdgeNode(no[0], no[1]);
         int mid12 = GetMidEdgeNode(no[1], no[2]);
         int mid23 = GetMidEdgeNode(no[2], no[3]);
         int mid30 = GetMidEdgeNode(no[3], no[0]);

         int mid45 = GetMidEdgeNode(no[4], no[5]);
         int mid56 = GetMidEdgeNode(no[5], no[6]);
         int mid67 = GetMidEdgeNode(no[6], no[7]);
         int mid74 = GetMidEdgeNode(no[7], no[4]);

         int mid04 = GetMidEdgeNode(no[0], no[4]);
         int mid15 = GetMidEdgeNode(no[1], no[5]);
         int mid26 = GetMidEdgeNode(no[2], no[6]);
         int mid37 = GetMidEdgeNode(no[3], no[7]);

         int midf0 = GetMidFaceNode(mid23, mid12, mid01, mid30);
         int midf1 = GetMidFaceNode(mid01, mid15, mid45, mid04);
         int midf2 = GetMidFaceNode(mid12, mid26, mid56, mid15);
         int midf3 = GetMidFaceNode(mid23, mid37, mid67, mid26);
         int midf4 = GetMidFaceNode(mid30, mid04, mid74, mid37);
         int midf5 = GetMidFaceNode(mid45, mid56, mid67, mid74);

         int midel = GetMidEdgeNode(midf1, midf3);

         child[0] = NewHexahedron(no[0], mid01, midf0, mid30,
                                  mid04, midf1, midel, midf4, attr,
                                  fa[0], fa[1], -1, -1, fa[4], -1);

         child[1] = NewHexahedron(mid01, no[1], mid12, midf0,
                                  midf1, mid15, midf2, midel, attr,
                                  fa[0], fa[1], fa[2], -1, -1, -1);

         child[2] = NewHexahedron(midf0, mid12, no[2], mid23,
                                  midel, midf2, mid26, midf3, attr,
                                  fa[0], -1, fa[2], fa[3], -1, -1);

         child[3] = NewHexahedron(mid30, midf0, mid23, no[3],
                                  midf4, midel, midf3, mid37, attr,
                                  fa[0], -1, -1, fa[3], fa[4], -1);

         child[4] = NewHexahedron(mid04, midf1, midel, midf4,
                                  no[4], mid45, midf5, mid74, attr,
                                  -1, fa[1], -1, -1, fa[4], fa[5]);

         child[5] = NewHexahedron(midf1, mid15, midf2, midel,
                                  mid45, no[5], mid56, midf5, attr,
                                  -1, fa[1], fa[2], -1, -1, fa[5]);

         child[6] = NewHexahedron(midel, midf2, mid26, midf3,
                                  midf5, mid56, no[6], mid67, attr,
                                  -1, -1, fa[2], fa[3], -1, fa[5]);

         child[7] = NewHexahedron(midf4, midel, midf3, mid37,
                                  mid74, midf5, mid67, no[7], attr,
                                  -1, -1, -1, fa[3], fa[4], fa[5]);

         CheckIsoFace(no[3], no[2], no[1], no[0], mid23, mid12, mid01, mid30, midf0);
         CheckIsoFace(no[0], no[1], no[5], no[4], mid01, mid15, mid45, mid04, midf1);
         CheckIsoFace(no[1], no[2], no[6], no[5], mid12, mid26, mid56, mid15, midf2);
         CheckIsoFace(no[2], no[3], no[7], no[6], mid23, mid37, mid67, mid26, midf3);
         CheckIsoFace(no[3], no[0], no[4], no[7], mid30, mid04, mid74, mid37, midf4);
         CheckIsoFace(no[4], no[5], no[6], no[7], mid45, mid56, mid67, mid74, midf5);
      }
      else
      {
         MFEM_ABORT("invalid refinement type.");
      }

      if (ref_type != Refinement::XYZ) { Iso = false; }
   }
   else if (el.Geom() == Geometry::PRISM)
   {
      // Wedge vertex numbering:
      //
      //          5
      //         _+_
      //       _/ | \_                    Faces: 0 bottom
      //    3 /   |   \ 4                        1 top
      //     +---------+                         2 front
      //     |    |    |                         3 right (1 2 5 4)
      //     |   _+_   |                         4 left (2 0 3 5)
      //     | _/ 2 \_ |           Z  Y
      //     |/       \|           | /
      //     +---------+           *--X
      //    0           1

      if (ref_type < 4) // XY refinement (split in 4 wedges)
      {
         ref_type = Refinement::XY; // for consistence

         int mid01 = GetMidEdgeNode(no[0], no[1]);
         int mid12 = GetMidEdgeNode(no[1], no[2]);
         int mid20 = GetMidEdgeNode(no[2], no[0]);

         int mid34 = GetMidEdgeNode(no[3], no[4]);
         int mid45 = GetMidEdgeNode(no[4], no[5]);
         int mid53 = GetMidEdgeNode(no[5], no[3]);

         child[0] = NewWedge(no[0], mid01, mid20,
                             no[3], mid34, mid53, attr,
                             fa[0], fa[1], fa[2], -1, fa[4]);

         child[1] = NewWedge(mid01, no[1], mid12,
                             mid34, no[4], mid45, attr,
                             fa[0], fa[1], fa[2], fa[3], -1);

         child[2] = NewWedge(mid20, mid12, no[2],
                             mid53, mid45, no[5], attr,
                             fa[0], fa[1], -1, fa[3], fa[4]);

         child[3] = NewWedge(mid12, mid20, mid01,
                             mid45, mid53, mid34, attr,
                             fa[0], fa[1], -1, -1, -1);

         CheckAnisoFace(no[0], no[1], no[4], no[3], mid01, mid34);
         CheckAnisoFace(no[1], no[2], no[5], no[4], mid12, mid45);
         CheckAnisoFace(no[2], no[0], no[3], no[5], mid20, mid53);
      }
      else if (ref_type == Refinement::Z) // Z refinement only (split in 2 wedges)
      {
         int mid03 = GetMidEdgeNode(no[0], no[3]);
         int mid14 = GetMidEdgeNode(no[1], no[4]);
         int mid25 = GetMidEdgeNode(no[2], no[5]);

         child[0] = NewWedge(no[0], no[1], no[2],
                             mid03, mid14, mid25, attr,
                             fa[0], -1, fa[2], fa[3], fa[4]);

         child[1] = NewWedge(mid03, mid14, mid25,
                             no[3], no[4], no[5], attr,
                             -1, fa[1], fa[2], fa[3], fa[4]);

         CheckAnisoFace(no[3], no[0], no[1], no[4], mid03, mid14);
         CheckAnisoFace(no[4], no[1], no[2], no[5], mid14, mid25);
         CheckAnisoFace(no[5], no[2], no[0], no[3], mid25, mid03);
      }
      else // ref_type > 4, full isotropic refinement (split in 8 wedges)
      {
         ref_type = Refinement::XYZ; // for consistence

         int mid01 = GetMidEdgeNode(no[0], no[1]);
         int mid12 = GetMidEdgeNode(no[1], no[2]);
         int mid20 = GetMidEdgeNode(no[2], no[0]);

         int mid34 = GetMidEdgeNode(no[3], no[4]);
         int mid45 = GetMidEdgeNode(no[4], no[5]);
         int mid53 = GetMidEdgeNode(no[5], no[3]);

         int mid03 = GetMidEdgeNode(no[0], no[3]);
         int mid14 = GetMidEdgeNode(no[1], no[4]);
         int mid25 = GetMidEdgeNode(no[2], no[5]);

         int midf2 = GetMidFaceNode(mid01, mid14, mid34, mid03);
         int midf3 = GetMidFaceNode(mid12, mid25, mid45, mid14);
         int midf4 = GetMidFaceNode(mid20, mid03, mid53, mid25);

         child[0] = NewWedge(no[0], mid01, mid20,
                             mid03, midf2, midf4, attr,
                             fa[0], -1, fa[2], -1, fa[4]);

         child[1] = NewWedge(mid01, no[1], mid12,
                             midf2, mid14, midf3, attr,
                             fa[0], -1, fa[2], fa[3], -1);

         child[2] = NewWedge(mid20, mid12, no[2],
                             midf4, midf3, mid25, attr,
                             fa[0], -1, -1, fa[3], fa[4]);

         child[3] = NewWedge(mid12, mid20, mid01,
                             midf3, midf4, midf2, attr,
                             fa[0], -1, -1, -1, -1);

         child[4] = NewWedge(mid03, midf2, midf4,
                             no[3], mid34, mid53, attr,
                             -1, fa[1], fa[2], -1, fa[4]);

         child[5] = NewWedge(midf2, mid14, midf3,
                             mid34, no[4], mid45, attr,
                             -1, fa[1], fa[2], fa[3], -1);

         child[6] = NewWedge(midf4, midf3, mid25,
                             mid53, mid45, no[5], attr,
                             -1, fa[1], -1, fa[3], fa[4]);

         child[7] = NewWedge(midf3, midf4, midf2,
                             mid45, mid53, mid34, attr,
                             -1, fa[1], -1, -1, -1);

         CheckIsoFace(no[0], no[1], no[4], no[3], mid01, mid14, mid34, mid03, midf2);
         CheckIsoFace(no[1], no[2], no[5], no[4], mid12, mid25, mid45, mid14, midf3);
         CheckIsoFace(no[2], no[0], no[3], no[5], mid20, mid03, mid53, mid25, midf4);
      }

      if (ref_type != Refinement::XYZ) { Iso = false; }
   }
   else if (el.Geom() == Geometry::TETRAHEDRON)
   {
      // Tetrahedron vertex numbering:
      //
      //    3
      //     +                         Faces: 0 back (1, 2, 3)
      //     |\\_                             1 left (0, 3, 2)
      //     ||  \_                           2 front (0, 1, 3)
      //     | \   \_                         3 bottom (0, 1, 2)
      //     |  +__  \_
      //     | /2  \__ \_       Z  Y
      //     |/       \__\      | /
      //     +------------+     *--X
      //    0              1

      ref_type = Refinement::XYZ; // for consistence

      int mid01 = GetMidEdgeNode(no[0], no[1]);
      int mid12 = GetMidEdgeNode(no[1], no[2]);
      int mid02 = GetMidEdgeNode(no[2], no[0]);

      int mid03 = GetMidEdgeNode(no[0], no[3]);
      int mid13 = GetMidEdgeNode(no[1], no[3]);
      int mid23 = GetMidEdgeNode(no[2], no[3]);

      child[0] = NewTetrahedron(no[0], mid01, mid02, mid03, attr,
                                -1, fa[1], fa[2], fa[3]);

      child[1] = NewTetrahedron(mid01, no[1], mid12, mid13, attr,
                                fa[0], -1, fa[2], fa[3]);

      child[2] = NewTetrahedron(mid02, mid12, no[2], mid23, attr,
                                fa[0], fa[1], -1, fa[3]);

      child[3] = NewTetrahedron(mid03, mid13, mid23, no[3], attr,
                                fa[0], fa[1], fa[2], -1);

      // There are three ways to split the inner octahedron. A good strategy is
      // to use the shortest diagonal. At the moment we don't have the geometric
      // information in this class to determine which diagonal is the shortest,
      // but it seems that with reasonable shapes of the coarse tets and MFEM's
      // default tet orientation, always using tet_type == 0 produces stable
      // refinements. Types 1 and 2 are unused for now.
      el.tet_type = 0;

      if (el.tet_type == 0) // shortest diagonal mid01--mid23
      {
         child[4] = NewTetrahedron(mid01, mid23, mid02, mid03, attr,
                                   fa[1], -1, -1, -1);

         child[5] = NewTetrahedron(mid01, mid23, mid03, mid13, attr,
                                   -1, fa[2], -1, -1);

         child[6] = NewTetrahedron(mid01, mid23, mid13, mid12, attr,
                                   fa[0], -1, -1, -1);

         child[7] = NewTetrahedron(mid01, mid23, mid12, mid02, attr,
                                   -1, fa[3], -1, -1);
      }
      else if (el.tet_type == 1) // shortest diagonal mid12--mid03
      {
         child[4] = NewTetrahedron(mid03, mid01, mid02, mid12, attr,
                                   fa[3], -1, -1, -1);

         child[5] = NewTetrahedron(mid03, mid02, mid23, mid12, attr,
                                   -1, -1, -1, fa[1]);

         child[6] = NewTetrahedron(mid03, mid23, mid13, mid12, attr,
                                   fa[0], -1, -1, -1);

         child[7] = NewTetrahedron(mid03, mid13, mid01, mid12, attr,
                                   -1, -1, -1, fa[2]);
      }
      else // el.tet_type == 2, shortest diagonal mid02--mid13
      {
         child[4] = NewTetrahedron(mid02, mid01, mid13, mid03, attr,
                                   fa[2], -1, -1, -1);

         child[5] = NewTetrahedron(mid02, mid03, mid13, mid23, attr,
                                   -1, -1, fa[1], -1);

         child[6] = NewTetrahedron(mid02, mid23, mid13, mid12, attr,
                                   fa[0], -1, -1, -1);

         child[7] = NewTetrahedron(mid02, mid12, mid13, mid01, attr,
                                   -1, -1, fa[3], -1);
      }
   }
   else if (el.Geom() == Geometry::PYRAMID)
   {
      // Pyramid vertex numbering:
      //
      //    4
      //     + \_                       Faces: 0 bottom (3,2,1,0)
      //     |\\_ \_                           1 front (0, 1, 4)
      //     ||  \_ \__                        2 right (1, 2, 4)
      //     | \   \_   \__                    3 back (2, 3, 4)
      //     |  +____\_ ____\                  4 left (3, 0, 4)
      //     | /3      \_   2   Z  Y
      //     |/          \ /    | /
      //     +------------+     *--X
      //    0              1

      ref_type = Refinement::XYZ; // for consistence

      int mid01 = GetMidEdgeNode(no[0], no[1]);
      int mid12 = GetMidEdgeNode(no[1], no[2]);
      int mid23 = GetMidEdgeNode(no[2], no[3]);
      int mid03 = GetMidEdgeNode(no[0], no[3]);
      int mid04 = GetMidEdgeNode(no[0], no[4]);
      int mid14 = GetMidEdgeNode(no[1], no[4]);
      int mid24 = GetMidEdgeNode(no[2], no[4]);
      int mid34 = GetMidEdgeNode(no[3], no[4]);
      int midf0 = GetMidFaceNode(mid23, mid12, mid01, mid03);

      child[0] = NewPyramid(no[0], mid01, midf0, mid03, mid04,
                            attr, fa[0], fa[1], -1, -1, fa[4]);

      child[1] = NewPyramid(mid01, no[1], mid12, midf0, mid14,
                            attr, fa[0], fa[1], fa[2], -1, -1);

      child[2] = NewPyramid(midf0, mid12, no[2], mid23, mid24,
                            attr, fa[0], -1, fa[2], fa[3], -1);

      child[3] = NewPyramid(mid03, midf0, mid23, no[3], mid34,
                            attr, fa[0], -1, -1, fa[3], fa[4]);

      child[4] = NewPyramid(mid24, mid14, mid04, mid34, midf0,
                            attr, -1, -1, -1, -1, -1);

      child[5] = NewPyramid(mid04, mid14, mid24, mid34, no[4],
                            attr, -1, fa[1], fa[2], fa[3], fa[4]);

      child[6] = NewTetrahedron(mid01, midf0, mid04, mid14,
                                attr, -1, -1, -1, fa[1]);

      child[7] = NewTetrahedron(midf0, mid14, mid12, mid24,
                                attr, -1, -1, fa[2], -1);

      child[8] = NewTetrahedron(midf0, mid23, mid34, mid24,
                                attr, -1, -1, fa[3], -1);

      child[9] = NewTetrahedron(mid03, mid04, midf0, mid34,
                                attr, -1, fa[4], -1, -1);

      CheckIsoFace(no[3], no[2], no[1], no[0], mid23, mid12, mid01, mid03, midf0);
   }
   else if (el.Geom() == Geometry::SQUARE)
   {
      ref_type &= 0x3; // ignore Z bit

      if (ref_type == Refinement::X) // X split
      {
         int mid01 = nodes.GetId(no[0], no[1]);
         int mid23 = nodes.GetId(no[2], no[3]);

         child[0] = NewQuadrilateral(no[0], mid01, mid23, no[3],
                                     attr, fa[0], -1, fa[2], fa[3]);

         child[1] = NewQuadrilateral(mid01, no[1], no[2], mid23,
                                     attr, fa[0], fa[1], fa[2], -1);
      }
      else if (ref_type == Refinement::Y) // Y split
      {
         int mid12 = nodes.GetId(no[1], no[2]);
         int mid30 = nodes.GetId(no[3], no[0]);

         child[0] = NewQuadrilateral(no[0], no[1], mid12, mid30,
                                     attr, fa[0], fa[1], -1, fa[3]);

         child[1] = NewQuadrilateral(mid30, mid12, no[2], no[3],
                                     attr, -1, fa[1], fa[2], fa[3]);
      }
      else if (ref_type == Refinement::XY) // iso split
      {
         int mid01 = nodes.GetId(no[0], no[1]);
         int mid12 = nodes.GetId(no[1], no[2]);
         int mid23 = nodes.GetId(no[2], no[3]);
         int mid30 = nodes.GetId(no[3], no[0]);

         int midel = nodes.GetId(mid01, mid23);

         child[0] = NewQuadrilateral(no[0], mid01, midel, mid30,
                                     attr, fa[0], -1, -1, fa[3]);

         child[1] = NewQuadrilateral(mid01, no[1], mid12, midel,
                                     attr, fa[0], fa[1], -1, -1);

         child[2] = NewQuadrilateral(midel, mid12, no[2], mid23,
                                     attr, -1, fa[1], fa[2], -1);

         child[3] = NewQuadrilateral(mid30, midel, mid23, no[3],
                                     attr, -1, -1, fa[2], fa[3]);
      }
      else
      {
         MFEM_ABORT("Invalid refinement type.");
      }

      if (ref_type != Refinement::XY) { Iso = false; }
   }
   else if (el.Geom() == Geometry::TRIANGLE)
   {
      ref_type = Refinement::XY; // for consistence

      // isotropic split - the only ref_type available for triangles
      int mid01 = nodes.GetId(no[0], no[1]);
      int mid12 = nodes.GetId(no[1], no[2]);
      int mid20 = nodes.GetId(no[2], no[0]);

      child[0] = NewTriangle(no[0], mid01, mid20, attr, fa[0], -1, fa[2]);
      child[1] = NewTriangle(mid01, no[1], mid12, attr, fa[0], fa[1], -1);
      child[2] = NewTriangle(mid20, mid12, no[2], attr, -1, fa[1], fa[2]);
      child[3] = NewTriangle(mid12, mid20, mid01, attr, -1, -1, -1);
   }
   else if (el.Geom() == Geometry::SEGMENT)
   {
      ref_type = Refinement::X; // for consistence

      int mid = nodes.GetId(no[0], no[1]);
      child[0] = NewSegment(no[0], mid, attr, fa[0], -1);
      child[1] = NewSegment(mid, no[1], attr, -1, fa[1]);
   }
   else
   {
      MFEM_ABORT("Unsupported element geometry.");
   }

   // start using the nodes of the children, create edges & faces
   for (int i = 0; i < MaxElemChildren && child[i] >= 0; i++)
   {
      ReferenceElement(child[i]);
   }

   int buf[MaxElemFaces];
   Array<int> parentFaces(buf, MaxElemFaces);
   parentFaces.SetSize(0);

   // sign off of all nodes of the parent, clean up unused nodes, but keep faces
   UnreferenceElement(elem, parentFaces);

   // register the children in their faces
   for (int i = 0; i < MaxElemChildren && child[i] >= 0; i++)
   {
      RegisterFaces(child[i]);
   }

   // clean up parent faces, if unused
   DeleteUnusedFaces(parentFaces);

   // make the children inherit our rank; set the parent element
   for (int i = 0; i < MaxElemChildren && child[i] >= 0; i++)
   {
      Element &ch = elements[child[i]];
      ch.rank = el.rank;
      ch.parent = elem;
   }

   // finish the refinement
   el.ref_type = ref_type;
   std::memcpy(el.child, child, sizeof(el.child));
}


void NCMesh::Refine(const Array<Refinement>& refinements)
{
   // push all refinements on the stack in reverse order
   ref_stack.Reserve(refinements.Size());
   for (int i = refinements.Size()-1; i >= 0; i--)
   {
      const Refinement& ref = refinements[i];
      ref_stack.Append(Refinement(leaf_elements[ref.index], ref.ref_type));
   }

   // keep refining as long as the stack contains something
   int nforced = 0;
   while (ref_stack.Size())
   {
      Refinement ref = ref_stack.Last();
      ref_stack.DeleteLast();

      int size = ref_stack.Size();
      RefineElement(ref.index, ref.ref_type);
      nforced += ref_stack.Size() - size;
   }

   /* TODO: the current algorithm of forced refinements is not optimal. As
      forced refinements spread through the mesh, some may not be necessary in
      the end, since the affected elements may still be scheduled for refinement
      that could stop the propagation. We should introduce the member
      Element::ref_pending that would show the intended refinement in the batch.
      A forced refinement would be combined with ref_pending to (possibly) stop
      the propagation earlier.

      Update: what about a FIFO instead of ref_stack? */

#if defined(MFEM_DEBUG) && !defined(MFEM_USE_MPI)
   mfem::out << "Refined " << refinements.Size() << " + " << nforced
             << " elements" << std::endl;
#else
   MFEM_CONTRACT_VAR(nforced);
#endif

   ref_stack.DeleteAll();
   shadow.DeleteAll();

   Update();
}


//// Derefinement //////////////////////////////////////////////////////////////

int NCMesh::RetrieveNode(const Element &el, int index)
{
   if (!el.ref_type) { return el.node[index]; }

   // need to retrieve node from a child element (there is always a child that
   // inherited the parent's corner under the same index)
   int ch;
   switch (el.Geom())
   {
      case Geometry::CUBE:
         ch = el.child[hex_deref_table[el.ref_type - 1][index]];
         break;

      case Geometry::PRISM:
         ch = prism_deref_table[el.ref_type - 1][index];
         MFEM_ASSERT(ch != -1, "");
         ch = el.child[ch];
         break;

      case Geometry::PYRAMID:
         ch = pyramid_deref_table[el.ref_type - 1][index];
         MFEM_ASSERT(ch != -1, "");
         ch = el.child[ch];
         break;

      case Geometry::SQUARE:
         ch = el.child[quad_deref_table[el.ref_type - 1][index]];
         break;

      case Geometry::TETRAHEDRON:
      case Geometry::TRIANGLE:
         ch = el.child[index];
         break;

      default:
         ch = 0; // suppress compiler warning
         MFEM_ABORT("Unsupported element geometry.");
   }
   return RetrieveNode(elements[ch], index);
}


void NCMesh::DerefineElement(int elem)
{
   Element &el = elements[elem];
   if (!el.ref_type) { return; }

   int child[MaxElemChildren];
   std::memcpy(child, el.child, sizeof(child));

   // first make sure that all children are leaves, derefine them if not
   for (int i = 0; i < MaxElemChildren && child[i] >= 0; i++)
   {
      if (elements[child[i]].ref_type)
      {
         DerefineElement(child[i]);
      }
   }

   int faces_attribute[MaxElemFaces];
   int ref_type_key = el.ref_type - 1;

   for (int i = 0; i < MaxElemNodes; i++) { el.node[i] = -1; }

   // retrieve original corner nodes and face attributes from the children
   if (el.Geom() == Geometry::CUBE)
   {
      // Sets corner nodes from childs
      constexpr int nb_cube_childs = 8;
      for (int i = 0; i < nb_cube_childs; i++)
      {
         const int child_local_index = hex_deref_table[ref_type_key][i];
         const int child_global_index = child[child_local_index];
         Element &ch = elements[child_global_index];
         el.node[i] = ch.node[i];
      }
      // Sets faces attributes from childs' faces
      constexpr int nb_cube_faces = 6;
      for (int i = 0; i < nb_cube_faces; i++)
      {
         const int child_local_index = hex_deref_table[ref_type_key]
                                       [i + nb_cube_childs];
         const int child_global_index = child[child_local_index];
         Element &ch = elements[child_global_index];
         const int* fv = GI[el.Geom()].faces[i];
         faces_attribute[i] = faces.Find(ch.node[fv[0]], ch.node[fv[1]],
                                         ch.node[fv[2]], ch.node[fv[3]])
                              ->attribute;
      }
   }
   else if (el.Geom() == Geometry::PRISM)
   {
      MFEM_ASSERT(prism_deref_table[ref_type_key][0] != -1,
                  "invalid prism refinement");
      constexpr int nb_prism_childs = 6;
      for (int i = 0; i < nb_prism_childs; i++)
      {
         const int child_local_index = prism_deref_table[ref_type_key][i];
         const int child_global_index = child[child_local_index];
         Element &ch = elements[child_global_index];
         el.node[i] = ch.node[i];
      }
      el.node[6] = el.node[7] = -1;

      constexpr int nb_prism_faces = 5;
      for (int i = 0; i < nb_prism_faces; i++)
      {
         const int child_local_index = prism_deref_table[ref_type_key]
                                       [i + nb_prism_childs];
         const int child_global_index = child[child_local_index];
         Element &ch = elements[child_global_index];
         const int* fv = GI[el.Geom()].faces[i];
         faces_attribute[i] = faces.Find(ch.node[fv[0]], ch.node[fv[1]],
                                         ch.node[fv[2]], ch.node[fv[3]])
                              ->attribute;
      }
   }
   else if (el.Geom() == Geometry::PYRAMID)
   {
      MFEM_ASSERT(pyramid_deref_table[ref_type_key][0] != -1,
                  "invalid pyramid refinement");
      constexpr int nb_pyramid_childs = 5;
      for (int i = 0; i < nb_pyramid_childs; i++)
      {
         const int child_local_index = pyramid_deref_table[ref_type_key][i];
         const int child_global_index = child[child_local_index];
         Element &ch = elements[child_global_index];
         el.node[i] = ch.node[i];
      }
      el.node[5] = el.node[6] = el.node[7] = -1;


      constexpr int nb_pyramid_faces = 5;
      for (int i = 0; i < nb_pyramid_faces; i++)
      {
         const int child_local_index = pyramid_deref_table[ref_type_key]
                                       [i + nb_pyramid_childs];
         const int child_global_index = child[child_local_index];
         Element &ch = elements[child_global_index];
         const int* fv = GI[el.Geom()].faces[i];
         faces_attribute[i] = faces.Find(ch.node[fv[0]], ch.node[fv[1]],
                                         ch.node[fv[2]], ch.node[fv[3]])
                              ->attribute;
      }
   }
   else if (el.Geom() == Geometry::TETRAHEDRON)
   {
      for (int i = 0; i < 4; i++)
      {
         Element& ch1 = elements[child[i]];
         Element& ch2 = elements[child[(i+1) & 0x3]];
         el.node[i] = ch1.node[i];
         const int* fv = GI[el.Geom()].faces[i];
         faces_attribute[i] = faces.Find(ch2.node[fv[0]], ch2.node[fv[1]],
                                         ch2.node[fv[2]], ch2.node[fv[3]])
                              ->attribute;
      }
   }
   else if (el.Geom() == Geometry::SQUARE)
   {
      constexpr int nb_square_childs = 4;
      for (int i = 0; i < nb_square_childs; i++)
      {
         const int child_local_index = quad_deref_table[ref_type_key][i];
         const int child_global_index = child[child_local_index];
         Element &ch = elements[child_global_index];
         el.node[i] = ch.node[i];
      }
      constexpr int nb_square_faces = 4;
      for (int i = 0; i < nb_square_faces; i++)
      {
         const int child_local_index = quad_deref_table[ref_type_key]
                                       [i + nb_square_childs];
         const int child_global_index = child[child_local_index];
         Element &ch = elements[child_global_index];
         const int* fv = GI[el.Geom()].faces[i];
         faces_attribute[i] = faces.Find(ch.node[fv[0]], ch.node[fv[1]],
                                         ch.node[fv[2]], ch.node[fv[3]])
                              ->attribute;
      }
   }
   else if (el.Geom() == Geometry::TRIANGLE)
   {
      constexpr int nb_triangle_childs = 3;
      for (int i = 0; i < nb_triangle_childs; i++)
      {
         Element& ch = elements[child[i]];
         el.node[i] = ch.node[i];
         const int* fv = GI[el.Geom()].faces[i];
         faces_attribute[i] = faces.Find(ch.node[fv[0]], ch.node[fv[1]],
                                         ch.node[fv[2]], ch.node[fv[3]])
                              ->attribute;
      }
   }
   else if (el.Geom() == Geometry::SEGMENT)
   {
      constexpr int nb_segment_childs = 2;
      for (int i = 0; i < nb_segment_childs; i++)
      {
         int ni = elements[child[i]].node[i];
         el.node[i] = ni;
         faces_attribute[i] = faces.Find(ni, ni, ni, ni)->attribute;
      }
   }
   else
   {
      MFEM_ABORT("Unsupported element geometry.");
   }

   // sign in to all nodes
   ReferenceElement(elem);

   int buf[MaxElemChildren*MaxElemFaces];
   Array<int> childFaces(buf, MaxElemChildren*MaxElemFaces);
   childFaces.SetSize(0);

   // delete children, determine rank
   el.rank = std::numeric_limits<int>::max();
   for (int i = 0; i < MaxElemChildren && child[i] >= 0; i++)
   {
      el.rank = std::min(el.rank, elements[child[i]].rank);
      UnreferenceElement(child[i], childFaces);
      FreeElement(child[i]);
   }

   RegisterFaces(elem, faces_attribute);

   // delete unused faces
   childFaces.Sort();
   childFaces.Unique();
   DeleteUnusedFaces(childFaces);

   el.ref_type = 0;
}


void NCMesh::CollectDerefinements(int elem, Array<Connection> &list)
{
   Element &el = elements[elem];
   if (!el.ref_type) { return; }

   int total = 0, ref = 0, ghost = 0;
   for (int i = 0; i < MaxElemChildren && el.child[i] >= 0; i++)
   {
      total++;
      Element &ch = elements[el.child[i]];
      if (ch.ref_type) { ref++; break; }
      if (IsGhost(ch)) { ghost++; }
   }

   if (!ref && ghost < total)
   {
      // can be derefined, add to list
      int next_row = list.Size() ? (list.Last().from + 1) : 0;
      for (int i = 0; i < MaxElemChildren && el.child[i] >= 0; i++)
      {
         Element &ch = elements[el.child[i]];
         list.Append(Connection(next_row, ch.index));
      }
   }
   else
   {
      for (int i = 0; i < MaxElemChildren && el.child[i] >= 0; i++)
      {
         CollectDerefinements(el.child[i], list);
      }
   }
}

const Table& NCMesh::GetDerefinementTable()
{
   Array<Connection> list;
   list.Reserve(leaf_elements.Size());

   for (int i = 0; i < root_state.Size(); i++)
   {
      CollectDerefinements(i, list);
   }

   int size = list.Size() ? (list.Last().from + 1) : 0;
   derefinements.MakeFromList(size, list);
   return derefinements;
}

void NCMesh::CheckDerefinementNCLevel(const Table &deref_table,
                                      Array<int> &level_ok, int max_nc_level)
{
   level_ok.SetSize(deref_table.Size());
   for (int i = 0; i < deref_table.Size(); i++)
   {
      const int* fine = deref_table.GetRow(i), size = deref_table.RowSize(i);
      Element &parent = elements[elements[leaf_elements[fine[0]]].parent];

      int ok = 1;
      for (int j = 0; j < size; j++)
      {
         int splits[3];
         CountSplits(leaf_elements[fine[j]], splits);

         for (int k = 0; k < Dim; k++)
         {
            if ((parent.ref_type & (1 << k)) &&
                splits[k] >= max_nc_level)
            {
               ok = 0; break;
            }
         }
         if (!ok) { break; }
      }
      level_ok[i] = ok;
   }
}

void NCMesh::Derefine(const Array<int> &derefs)
{
   MFEM_VERIFY(Dim < 3 || Iso,
               "derefinement of 3D anisotropic meshes not implemented yet.");

   InitDerefTransforms();

   Array<int> fine_coarse;
   leaf_elements.Copy(fine_coarse);

   // perform the derefinements
   for (int i = 0; i < derefs.Size(); i++)
   {
      int row = derefs[i];
      MFEM_VERIFY(row >= 0 && row < derefinements.Size(),
                  "invalid derefinement number.");

      const int* fine = derefinements.GetRow(row);
      int parent = elements[leaf_elements[fine[0]]].parent;

      // record the relation of the fine elements to their parent
      SetDerefMatrixCodes(parent, fine_coarse);

      DerefineElement(parent);
   }

   // update leaf_elements, Element::index etc.
   Update();

   // link old fine elements to the new coarse elements
   for (int i = 0; i < fine_coarse.Size(); i++)
   {
      transforms.embeddings[i].parent = elements[fine_coarse[i]].index;
   }
}

void NCMesh::InitDerefTransforms()
{
   int nfine = leaf_elements.Size();

   // this will tell GetDerefinementTransforms that transforms are not finished
   transforms.Clear();

   transforms.embeddings.SetSize(nfine);
   for (int i = 0; i < nfine; i++)
   {
      Embedding &emb = transforms.embeddings[i];
      emb.parent = -1;
      emb.matrix = 0;
      Element &el = elements[leaf_elements[i]];
      emb.geom = el.Geom();
      emb.ghost = IsGhost(el);
   }
}

void NCMesh::SetDerefMatrixCodes(int parent, Array<int> &fine_coarse)
{
   // encode the ref_type and child number for GetDerefinementTransforms()
   Element &prn = elements[parent];
   for (int i = 0; i < MaxElemChildren && prn.child[i] >= 0; i++)
   {
      Element &ch = elements[prn.child[i]];
      if (ch.index >= 0)
      {
         int code = (prn.ref_type << 4) | i;
         transforms.embeddings[ch.index].matrix = code;
         fine_coarse[ch.index] = parent;
      }
   }
}


//// Mesh Interface ////////////////////////////////////////////////////////////

void NCMesh::CollectLeafElements(int elem, int state, Array<int> &ghosts,
                                 int &counter)
{
   Element &el = elements[elem];
   if (!el.ref_type)
   {
      if (el.rank >= 0) // skip elements beyond the ghost layer in parallel
      {
         if (!IsGhost(el))
         {
            leaf_elements.Append(elem);
         }
         else
         {
            // in parallel (or in serial loading a parallel file), collect
            // elements of neighboring ranks in a separate array
            ghosts.Append(elem);
         }

         // assign the SFC index (temporarily, will be replaced by Mesh index)
         el.index = counter++;
      }
      else
      {
         // elements beyond the ghost layer are invalid and don't appear in
         // 'leaf_elements' (also for performance reasons)
         el.index = -1;
      }
   }
   else // Refined element
   {
      // in non-leaf elements, the 'rank' and 'index' members have no meaning
      el.rank = -1;
      el.index = -1;

      // recurse to subtrees; try to order leaf elements along a space-filling
      // curve by changing the order the children are visited at each level
      if (el.Geom() == Geometry::SQUARE && el.ref_type == Refinement::XY)
      {
         for (int i = 0; i < 4; i++)
         {
            int ch = quad_hilbert_child_order[state][i];
            int st = quad_hilbert_child_state[state][i];
            CollectLeafElements(el.child[ch], st, ghosts, counter);
         }
      }
      else if (el.Geom() == Geometry::CUBE && el.ref_type == Refinement::XYZ)
      {
         for (int i = 0; i < 8; i++)
         {
            int ch = hex_hilbert_child_order[state][i];
            int st = hex_hilbert_child_state[state][i];
            CollectLeafElements(el.child[ch], st, ghosts, counter);
         }
      }
      else // no space filling curve tables yet for remaining cases
      {
         for (int i = 0; i < MaxElemChildren; i++)
         {
            if (el.child[i] >= 0)
            {
               CollectLeafElements(el.child[i], state, ghosts, counter);
            }
         }
      }
   }
}

void NCMesh::UpdateLeafElements()
{
   Array<int> ghosts;

   // collect leaf elements in leaf_elements and ghosts elements in ghosts from
   // all roots
   leaf_elements.SetSize(0);
   for (int i = 0, counter = 0; i < root_state.Size(); i++)
   {
      CollectLeafElements(i, root_state[i], ghosts, counter);
   }

   NElements = leaf_elements.Size();
   NGhostElements = ghosts.Size();

   // append ghost elements at the end of 'leaf_element' (if any) and assign the
   // final (Mesh) indices of leaves
   leaf_elements.Append(ghosts);
   leaf_sfc_index.SetSize(leaf_elements.Size());
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element &el = elements[leaf_elements[i]];
      leaf_sfc_index[i] = el.index;
      el.index = i;
   }
}

void NCMesh::UpdateVertices()
{
#ifndef MFEM_NCMESH_OLD_VERTEX_ORDERING
   // This method assigns indices to vertices (Node::vert_index) that will be
   // seen by the Mesh class and the rest of MFEM. We must be careful to:
   //
   //   1. Stay compatible with the conforming code, which expects top-level
   //      (original) vertices to be indexed first, otherwise GridFunctions
   //      defined on a conforming mesh would no longer be valid when the mesh
   //      is converted to an NC mesh.
   //
   //   2. Make sure serial NCMesh is compatible with the parallel ParNCMesh, so
   //      it is possible to read parallel partial solutions in serial code
   //      (e.g., serial GLVis). This means handling ghost elements, if present.
   //
   //   3. Assign vertices in a globally consistent order for parallel meshes:
   //      if two vertices i,j are shared by two ranks r1,r2, and i<j on r1,
   //      then i<j on r2 as well. This is true for top-level vertices but also
   //      for the remaining shared vertices thanks to the globally consistent
   //      SFC ordering of the leaf elements. This property reduces
   //      communication and simplifies ParNCMesh.

   // STEP 1: begin by splitting vertices into 4 classes:
   //   - local top-level vertices (code -1)
   //   - local non-top level vertices (code -2)
   //   - ghost (non-local) vertices (code -3)
   //   - vertices beyond the ghost layer (code -4)

   for (auto & node : nodes)
   {
      node.vert_index = -4; // assume beyond ghost layer
   }

   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element &el = elements[leaf_elements[i]];
      for (int j = 0; j < GI[el.Geom()].nv; j++)
      {
         Node &nd = nodes[el.node[j]];
         if (el.rank == MyRank)
         {
            if (nd.p1 == nd.p2) // local top-level vertex
            {
               if (nd.vert_index < -1) { nd.vert_index = -1; }
            }
            else // local non-top-level vertex
            {
               if (nd.vert_index < -2) { nd.vert_index = -2; }
            }
         }
         else // ghost vertex
         {
            if (nd.vert_index < -3) { nd.vert_index = -3; }
         }
      }
   }

   // STEP 2: assign indices of top-level local vertices, in original order
   NVertices = 0;
   for (auto &node : nodes)
   {
      if (node.vert_index == -1)
      {
         node.vert_index = NVertices++;
      }
   }

   // STEP 3: go over all elements (local and ghost) in SFC order and assign
   // remaining local vertices in that order.
   Array<int> sfc_order(leaf_elements.Size());
   for (int i = 0; i < sfc_order.Size(); i++)
   {
      sfc_order[leaf_sfc_index[i]] = leaf_elements[i];
   }

   for (int i = 0; i < sfc_order.Size(); i++)
   {
      const Element &el = elements[sfc_order[i]];
      for (int j = 0; j < GI[el.Geom()].nv; j++)
      {
         Node &nd = nodes[el.node[j]];
         if (nd.vert_index == -2) { nd.vert_index = NVertices++; }
      }
   }

   // STEP 4: create the mapping from Mesh vertex index to NCMesh node index
   vertex_nodeId.SetSize(NVertices);
   for (auto node = nodes.begin(); node != nodes.end(); ++node)
   {
      if (node->HasVertex() && node->vert_index >= 0)
      {
         MFEM_ASSERT(node->vert_index < vertex_nodeId.Size(), "");
         vertex_nodeId[node->vert_index] = node.index();
      }
   }

   // STEP 5: assign remaining ghost vertices, ignore vertices beyond the ghost
   // layer
   NGhostVertices = 0;
   for (int i = 0; i < sfc_order.Size(); i++)
   {
      const Element &el = elements[sfc_order[i]];
      for (int j = 0; j < GI[el.Geom()].nv; j++)
      {
         Node &nd = nodes[el.node[j]];
         if (nd.vert_index == -3)
         {
            nd.vert_index = NVertices + NGhostVertices++;
         }
      }
   }

#else // old ordering for debugging/testing only
   bool parallel = false;
#ifdef MFEM_USE_MPI
   if (dynamic_cast<ParNCMesh*>(this)) { parallel = true; }
#endif

   if (!parallel)
   {
      NVertices = 0;
      for (auto node = nodes.begin(); node != nodes.end(); ++node)
      {
         if (node->HasVertex()) { node->vert_index = NVertices++; }
      }

      vertex_nodeId.SetSize(NVertices);

      NVertices = 0;
      for (auto node = nodes.begin(); node != nodes.end(); ++node)
      {
         if (node->HasVertex()) { vertex_nodeId[NVertices++] = node.index(); }
      }
   }
   else
   {
      for (auto node = nodes.begin(); node != nodes.end(); ++node)
      {
         if (node->HasVertex()) { node->vert_index = -1; }
      }

      NVertices = 0;
      for (int i = 0; i < leaf_elements.Size(); i++)
      {
         Element &el = elements[leaf_elements[i]];
         if (el.rank == MyRank)
         {
            for (int j = 0; j < GI[el.Geom()].nv; j++)
            {
               int &vindex = nodes[el.node[j]].vert_index;
               if (vindex < 0) { vindex = NVertices++; }
            }
         }
      }

      vertex_nodeId.SetSize(NVertices);
      for (auto &node : nodes)
      {
         if (node.HasVertex() && node.vert_index >= 0)
         {
            vertex_nodeId[node.vert_index] = node.index();
         }
      }

      NGhostVertices = 0;
      for (auto &node : nodes)
      {
         if (node.HasVertex() && node.vert_index < 0)
         {
            node.vert_index = NVertices + (NGhostVertices++);
         }
      }
   }
#endif
}

void NCMesh::InitRootState(int root_count)
{
   root_state.SetSize(root_count);
   root_state = 0;

   if (elements.Size() == 0) { return; }

   char* node_order;
   int nch;

   switch (elements[0].Geom()) // TODO: mixed meshes
   {
      case Geometry::SQUARE:
         nch = 4;
         node_order = (char*) quad_hilbert_child_order;
         break;

      case Geometry::CUBE:
         nch = 8;
         node_order = (char*) hex_hilbert_child_order;
         break;

      default:
         return; // do nothing, all states stay zero
   }

   int entry_node = -2;

   // process the root element sequence
   for (int i = 0; i < root_count; i++)
   {
      Element &el = elements[i];

      int v_in = FindNodeExt(el, entry_node, false);
      if (v_in < 0) { v_in = 0; }

      // determine which nodes are shared with the next element
      bool shared[MaxElemNodes];
      for (int ni = 0; ni < MaxElemNodes; ++ni) { shared[ni] = 0; }
      if (i+1 < root_count)
      {
         Element &next = elements[i+1];
         for (int j = 0; j < nch; j++)
         {
            int node = FindNodeExt(el, RetrieveNode(next, j), false);
            if (node >= 0) { shared[node] = true; }
         }
      }

      // select orientation that starts in v_in and exits in shared node
      int state = Dim*v_in;
      for (int j = 0; j < Dim; j++)
      {
         if (shared[(int) node_order[nch*(state + j) + nch-1]])
         {
            state += j;
            break;
         }
      }

      root_state[i] = state;

      entry_node = RetrieveNode(el, node_order[nch*state + nch-1]);
   }
}

mfem::Element* NCMesh::NewMeshElement(int geom) const
{
   switch (geom)
   {
      case Geometry::CUBE: return new mfem::Hexahedron;
      case Geometry::PRISM: return new mfem::Wedge;
      case Geometry::PYRAMID: return new mfem::Pyramid;
      case Geometry::TETRAHEDRON: return new mfem::Tetrahedron;
      case Geometry::SQUARE: return new mfem::Quadrilateral;
      case Geometry::TRIANGLE: return new mfem::Triangle;
   }
   MFEM_ABORT("invalid geometry");
   return NULL;
}

const real_t* NCMesh::CalcVertexPos(int node) const
{
   const Node &nd = nodes[node];
   if (nd.p1 == nd.p2) // top-level vertex
   {
      return &coordinates[3*nd.p1];
   }

   TmpVertex &tv = tmp_vertex[node];
   if (tv.valid) { return tv.pos; }

   MFEM_VERIFY(tv.visited == false, "cyclic vertex dependencies.");
   tv.visited = true;

   const real_t* pos1 = CalcVertexPos(nd.p1);
   const real_t* pos2 = CalcVertexPos(nd.p2);

   for (int i = 0; i < 3; i++)
   {
      tv.pos[i] = (pos1[i] + pos2[i]) * 0.5;
   }
   tv.valid = true;
   return tv.pos;
}

void NCMesh::GetMeshComponents(Mesh &mesh) const
{
   mesh.vertices.SetSize(vertex_nodeId.Size());
   if (coordinates.Size())
   {
      // calculate vertex positions from stored top-level vertex coordinates
      tmp_vertex = new TmpVertex[nodes.NumIds()];
      for (int i = 0; i < mesh.vertices.Size(); i++)
      {
         mesh.vertices[i].SetCoords(spaceDim, CalcVertexPos(vertex_nodeId[i]));
      }
      delete [] tmp_vertex;
   }
   // NOTE: if the mesh is curved ('coordinates' is empty), mesh.vertices are
   // left uninitialized here; they will be initialized later by the Mesh from
   // Nodes -- here we just make sure mesh.vertices has the correct size.

   for (auto &elem : mesh.elements)
   {
      mesh.FreeElement(elem);
   }
   mesh.elements.SetSize(0);

   for (auto &elem : mesh.boundary)
   {
      mesh.FreeElement(elem);
   }
   mesh.boundary.SetSize(0);

   // Save off boundary face vertices to make boundary elements later.
   std::map<int, mfem::Array<int>> unique_boundary_faces;

   // create an mfem::Element for each leaf Element
   for (int i = 0; i < NElements; i++)
   {
      const Element &nc_elem = elements[leaf_elements[i]];

      const int* node = nc_elem.node;
      GeomInfo& gi = GI[(int) nc_elem.geom];

      mfem::Element* elem = mesh.NewElement(nc_elem.geom);
      mesh.elements.Append(elem);

      elem->SetAttribute(nc_elem.attribute);
      for (int j = 0; j < gi.nv; j++)
      {
         elem->GetVertices()[j] = nodes[node[j]].vert_index;
      }

      // Loop over faces and collect those marked as boundaries
      for (int k = 0; k < gi.nf; ++k)
      {
         const int nfv = gi.nfv[k];
         const int * const fv = gi.faces[k];
         const auto id = faces.FindId(node[fv[0]], node[fv[1]], node[fv[2]],
                                      node[fv[3]]);
         if (id >= 0 && faces[id].Boundary())
         {
            const auto &face = faces[id];
            if (face.elem[0] >= 0 && face.elem[1] >= 0 &&
                nc_elem.rank != std::min(elements[face.elem[0]].rank,
                                         elements[face.elem[1]].rank))
            {
               // This is a conformal internal face, but this element is not the
               // lowest ranking attached processor, thus not the owner of the
               // face. Consequently, we do not add this face to avoid double
               // counting.
               continue;
            }

            // Add in all boundary faces that are actual boundaries or not
            // masters of another face. The fv[2] in the edge split is on
            // purpose. A point cannot have a split level, thus do not check for
            // master/slave relation.
            if ((nfv == 4 &&
                 !QuadFaceIsMaster(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]]))
                || (nfv == 3 && !TriFaceIsMaster(node[fv[0]], node[fv[1]], node[fv[2]]))
                || (nfv == 2 && EdgeSplitLevel(node[fv[0]], node[fv[2]]) == 0) || (nfv == 1))
            {
               // This face has no split faces below, it is conformal or a
               // slave.
               unique_boundary_faces[id].SetSize(nfv);
               for (int v = 0; v < nfv; ++v)
               {
                  // Using a map overwrites if a face is visited twice. The
                  // nfv==2 is necessary because faces of 2D are storing the
                  // second index in the 2 slot, not the 1 slot.
                  unique_boundary_faces[id][v] = nodes[node[fv[(nfv==2) ? 2*v : v]]].vert_index;
               }
            }
         }
      }
   }

   auto geom_from_nfv = [](int nfv)
   {
      switch (nfv)
      {
         case 1: return Geometry::POINT;
         case 2: return Geometry::SEGMENT;
         case 3: return Geometry::TRIANGLE;
         case 4: return Geometry::SQUARE;
      }
      return Geometry::INVALID;
   };

   for (const auto &fv : unique_boundary_faces)
   {
      const auto f = fv.first;
      const auto &v = fv.second;
      const auto &face = faces.At(f);

      auto geom = geom_from_nfv(v.Size());

      MFEM_ASSERT(geom != Geometry::INVALID,
                  "nfv: " << v.Size() <<
                  " does not match a valid face geometry: Quad, Tri, Segment, Point");

      // Add a new boundary element, with matching attribute and vertices
      mesh.boundary.Append(mesh.NewElement(geom));
      auto * const be = mesh.boundary.Last();
      be->SetAttribute(face.attribute);
      be->SetVertices(v);
   }
}


void NCMesh::OnMeshUpdated(Mesh *mesh)
{
   //// PART 1: pull indices of regular edges/faces from the Mesh

   NEdges = mesh->GetNEdges();
   NFaces = mesh->GetNumFaces();
   if (Dim < 2) { NFaces = 0; }
   // clear Node::edge_index and Face::index
   for (auto &node : nodes)
   {
      if (node.HasEdge()) { node.edge_index = -1; }
   }
   for (auto &face : faces)
   {
      face.index = -1;
   }

   // get edge enumeration from the Mesh
   Table *edge_vertex = mesh->GetEdgeVertexTable();
   for (int i = 0; i < edge_vertex->Size(); i++)
   {
      const int *ev = edge_vertex->GetRow(i);
      Node* node = nodes.Find(vertex_nodeId[ev[0]], vertex_nodeId[ev[1]]);
      MFEM_ASSERT(node && node->HasEdge(),
                  "edge (" << ev[0] << "," << ev[1] << ") not found, "
                  "node = " << node << " node->HasEdge() "
                  << (node != nullptr ? node->HasEdge() : false));
      node->edge_index = i;
   }

   // get face enumeration from the Mesh, initialize 'face_geom'
   face_geom.SetSize(NFaces);
   for (int i = 0; i < NFaces; i++)
   {
      const int* fv = mesh->GetFace(i)->GetVertices();
      const int nfv = mesh->GetFace(i)->GetNVertices();

      Face* face;
      if (Dim == 3)
      {
         if (nfv == 4)
         {
            face_geom[i] = Geometry::SQUARE;
            face = faces.Find(vertex_nodeId[fv[0]], vertex_nodeId[fv[1]],
                              vertex_nodeId[fv[2]], vertex_nodeId[fv[3]]);
         }
         else
         {
            MFEM_ASSERT(nfv == 3, "");
            face_geom[i] = Geometry::TRIANGLE;
            face = faces.Find(vertex_nodeId[fv[0]], vertex_nodeId[fv[1]],
                              vertex_nodeId[fv[2]]);
         }
      }
      else
      {
         MFEM_ASSERT(nfv == 2, "");
         face_geom[i] = Geometry::SEGMENT;
         int n0 = vertex_nodeId[fv[0]], n1 = vertex_nodeId[fv[1]];
         face = faces.Find(n0, n0, n1, n1); // look up degenerate face

#ifdef MFEM_DEBUG
         // (non-ghost) edge and face numbers must match in 2D
         const int *ev = edge_vertex->GetRow(i);
         MFEM_ASSERT((ev[0] == fv[0] && ev[1] == fv[1]) ||
                     (ev[1] == fv[0] && ev[0] == fv[1]), "");
#endif
      }

      MFEM_VERIFY(face, "face not found.");
      face->index = i;
   }

   //// PART 2: assign indices of ghost edges/faces, if any

   // count ghost edges and assign their indices
   NGhostEdges = 0;
   for (auto &node : nodes)
   {
      if (node.HasEdge() && node.edge_index < 0)
      {
         node.edge_index = NEdges + (NGhostEdges++);
      }
   }

   // count ghost faces
   NGhostFaces = 0;
   for (auto &face : faces)
   {
      if (face.index < 0) { NGhostFaces++; }
   }

   if (Dim == 2)
   {
      // in 2D we have fake faces because of DG
      MFEM_ASSERT(NFaces == NEdges, "");
      MFEM_ASSERT(NGhostFaces == NGhostEdges, "");
   }

   // resize face_geom (default_geom is for slave faces beyond the ghost layer)
   Geometry::Type default_geom = Geometry::SQUARE;
   face_geom.SetSize(NFaces + NGhostFaces, default_geom);

   // update 'face_geom' for ghost faces, assign ghost face indices
   int nghosts = 0;
   for (int i = 0; i < NGhostElements; i++)
   {
      Element &el = elements[leaf_elements[NElements + i]]; // ghost element
      GeomInfo &gi = GI[el.Geom()];

      for (int j = 0; j < gi.nf; j++)
      {
         const int *fv = gi.faces[j];
         const int fid = faces.FindId(el.node[fv[0]], el.node[fv[1]],
                                      el.node[fv[2]], el.node[fv[3]]);
         MFEM_ASSERT(fid >= 0, "face not found!");
         auto &face = faces[fid];

         if (face.index < 0)
         {
            face.index = NFaces + (nghosts++);
            // store the face geometry
            static const Geometry::Type types[5] =
            {
               Geometry::INVALID, Geometry::INVALID,
               Geometry::SEGMENT, Geometry::TRIANGLE, Geometry::SQUARE
            };
            face_geom[face.index] = types[gi.nfv[j]];
         }
      }
   }

   // assign valid indices also to faces beyond the ghost layer
   for (auto &face : faces)
   {
      if (face.index < 0) { face.index = NFaces + (nghosts++); }
   }
   MFEM_ASSERT(nghosts == NGhostFaces, "");
}


//// Face/edge lists ///////////////////////////////////////////////////////////

int NCMesh::QuadFaceSplitType(int v1, int v2, int v3, int v4,
                              int mid[5]) const
{
   MFEM_ASSERT(Dim >= 3, "");

   // find edge nodes
   int e1 = FindMidEdgeNode(v1, v2);
   int e2 = FindMidEdgeNode(v2, v3);
   int e3 = (e1 >= 0 && nodes[e1].HasVertex()) ? FindMidEdgeNode(v3, v4) : -1;
   int e4 = (e2 >= 0 && nodes[e2].HasVertex()) ? FindMidEdgeNode(v4, v1) : -1;

   // optional: return the mid-edge nodes if requested
   if (mid) { mid[0] = e1, mid[1] = e2, mid[2] = e3, mid[3] = e4; }

   // try to get a mid-face node, either by (e1, e3) or by (e2, e4)
   int midf1 = -1, midf2 = -1;
   if (e1 >= 0 && e3 >= 0) { midf1 = FindMidEdgeNode(e1, e3); }
   if (e2 >= 0 && e4 >= 0) { midf2 = FindMidEdgeNode(e2, e4); }

   // get proper node if shadow node exists
   if (midf1 >= 0 && midf1 == midf2)
   {
      const Node &nd = nodes[midf1];
      if (nd.p1 != e1 && nd.p2 != e1) { midf1 = -1; }
      if (nd.p1 != e2 && nd.p2 != e2) { midf2 = -1; }
   }

   // only one way to access the mid-face node must always exist
   MFEM_ASSERT(!(midf1 >= 0 && midf2 >= 0), "incorrectly split face!");

   if (midf1 < 0 && midf2 < 0) // face not split
   {
      if (mid) { mid[4] = -1; }
      return 0;
   }
   else if (midf1 >= 0) // face split "vertically"
   {
      if (mid) { mid[4] = midf1; }
      return 1;
   }
   else // face split "horizontally"
   {
      if (mid) { mid[4] = midf2; }
      return 2;
   }
}

bool NCMesh::TriFaceSplit(int v1, int v2, int v3, int mid[3]) const
{
   int e1 = nodes.FindId(v1, v2);
   if (e1 < 0 || !nodes[e1].HasVertex()) { return false; }

   int e2 = nodes.FindId(v2, v3);
   if (e2 < 0 || !nodes[e2].HasVertex()) { return false; }

   int e3 = nodes.FindId(v3, v1);
   if (e3 < 0 || !nodes[e3].HasVertex()) { return false; }

   if (mid) { mid[0] = e1, mid[1] = e2, mid[2] = e3; }

   // This is necessary but not sufficient to determine if a face has been
   // split. All edges might have been split due to edge attached faces being
   // refined. Need to check for existence of face made up of midpoints.
   return true;
}

bool contains_node(const std::array<int, 4> &nodes, int n)
{
   return std::find(nodes.begin(), nodes.end(), n) != nodes.end();
};

int NCMesh::ParentFaceNodes(std::array<int, 4> &face_nodes) const
{
   const bool is_tri = face_nodes[3] == -1;
   const bool is_segment = (face_nodes[0] == face_nodes[1] &&
                            face_nodes[2] == face_nodes[3]);
   const bool is_quad = *std::min_element(face_nodes.begin(),
                                          face_nodes.end()) >= 0;

   MFEM_ASSERT((is_tri && !is_segment && !is_quad)
               || (!is_tri && is_segment && !is_quad) || (!is_tri && !is_segment &&
                                                          is_quad), "Inconsistent node geometry");

   bool all_nodes_root = true;
   for (auto x : face_nodes)
   {
      all_nodes_root = all_nodes_root && (x < 0 || (nodes[x].p1 == nodes[x].p2));
   }
   // This face is a root face -> nothing to do.
   if (all_nodes_root) { return -1; }

   int child = -1; // The index into parent.child that this face corresponds to.
   auto parent_nodes = face_nodes;
   if (is_quad)
   {
      // Logic for coarsening anisotropic faces is more complex, needs
      // identification and handling of multiple "crux" points. Will require
      // inspection of edge nodes.
      MFEM_VERIFY(Iso,
                  "ParentFaceNodes does not support anisotropic refinement yet!");

      // Finds the first node whose parents aren't in the face_nodes. This is
      // also the index of the child location in the parent face. Treated
      // separately as ultimately multiple crux will need to be handled for
      // anisotropic faces.
      const auto crux = [&]()
      {
         for (int i = 0; i < static_cast<int>(face_nodes.size()); i++)
         {
            if ((!contains_node(face_nodes, nodes[face_nodes[i]].p1)
                 && !contains_node(face_nodes, nodes[face_nodes[i]].p2))
                || (nodes[face_nodes[i]].p1 == nodes[face_nodes[i]].p2) /* top level node */)
            {
               return i;
            }
         }
         return -1;
      }();
      MFEM_ASSERT(crux != -1, "A root face should have been returned early");

      // Loop over nodes, starting from diagonal to child, wrapping and skipping
      // child. This will visit the node opposite child twice, thereby
      // coarsening to the diagonally opposite. NOTE: This assumes that the
      // nodes for a square are numbered (0 -> 1 -> 2 -> 3 -> 0).
      for (int i = 0; i < static_cast<int>(face_nodes.size()) + 1; i++)
      {
         int ind = (crux + i + 2) %
                   4; // Start and end with coarsening of the diagonally opposite
         if (ind == crux) { continue; }
         auto &x = parent_nodes[ind];

         // Check against parent_nodes rather than face_nodes so on second lap
         // the node opposite crux will coarsen again to the diagonally across
         // in the parent face. A top level node has p1 == p2, thus these
         // modifications do nothing.
         if (contains_node(parent_nodes, nodes[x].p1))
         {
            MFEM_ASSERT(nodes[x].p2 == nodes[x].p1 ||
                        !contains_node(parent_nodes, nodes[x].p2), "!");
            x = nodes[x].p2;
         }
         else if (contains_node(parent_nodes, nodes[x].p2))
         {
            MFEM_ASSERT(nodes[x].p2 == nodes[x].p1 ||
                        !contains_node(parent_nodes, nodes[x].p1), "!");
            x = nodes[x].p1;
         }
         else { /* do nothing */ }
      }
   }
   else if (is_tri)
   {
      for (int i = 0; i < 3; i++)
      {
         auto x = face_nodes[i];
         if (x == -1) { continue; }
         if (contains_node(face_nodes, nodes[x].p1))
         {
            MFEM_ASSERT(nodes[x].p2 == nodes[x].p1 ||
                        !contains_node(face_nodes, nodes[x].p2), "!");
            parent_nodes[i] = nodes[x].p2;
         }
         else if (contains_node(face_nodes, nodes[x].p2))
         {
            MFEM_ASSERT(nodes[x].p2 == nodes[x].p1 ||
                        !contains_node(face_nodes, nodes[x].p1), "!");
            parent_nodes[i] = nodes[x].p1;
         }
         else { /* do nothing */ }
      }

      if (std::equal(face_nodes.begin(), face_nodes.end(), parent_nodes.begin()))
      {
         // Having excluded root faces, this must be an interior face. We need
         // to handle the special case of the interior face of the parent face.
         std::array<std::array<int, 2>, 6> parent_pairs;
         for (std::size_t i = 0; i < face_nodes.size() - 1; i++)
         {
            parent_pairs[i][0] = nodes[face_nodes[i]].p1;
            parent_pairs[i][1] = nodes[face_nodes[i]].p2;
         }
         // Each node gets mapped to the common node from its parents and the
         // predecessor node's parents.
         for (int i = 0; i < 3; i++)
         {
            // Parenting convention here assumes parent face has the SAME
            // orientation as the original. This is true on exterior boundaries,
            // but for an interior boundary the master face will have an
            // opposing orientation. TODO: Possibly fix for interior boundaries.
            const auto &prev = parent_pairs[(i - 1 + 3) % 3]; // (0 -> 2, 1 -> 0, 2 -> 1)
            const auto &next = parent_pairs[(i + 1 + 3) % 3]; // (0 -> 1, 1 -> 2, 2 -> 0)
            for (auto x : next)
            {
               if (std::find(prev.begin(), prev.end(), x) != prev.end()) { parent_nodes[i] = x; }
            }
         }
         child = 3; // The interior face is the final child.
      }
   }
   else if (is_segment)
   {
      // Given this isn't a root face, one node must be the parent of the other.
      if (face_nodes[0] == nodes[face_nodes[1]].p1)
      {
         face_nodes[1] = nodes[face_nodes[1]].p2;
      }
      else if (face_nodes[0] == nodes[face_nodes[1]].p2)
      {
         face_nodes[1] = nodes[face_nodes[1]].p1;
      }
      else if (face_nodes[1] == nodes[face_nodes[0]].p1)
      {
         face_nodes[0] = nodes[face_nodes[0]].p2;
      }
      else if (face_nodes[1] == nodes[face_nodes[0]].p2)
      {
         face_nodes[0] = nodes[face_nodes[0]].p1;
      }
      else
      {
         MFEM_ABORT("Internal logic error!");
      }
   }
   else
   {
      MFEM_ABORT("Unrecognized face geometry!");
   }
   for (int i = 0; i < 4 && face_nodes[i] >= 0; i++)
   {
      if (face_nodes[i] == parent_nodes[i])
      {
         MFEM_ASSERT(child == -1,
                     "This face cannot be more than one child of the parent face!");
         child = i;
      }
   }
   MFEM_ASSERT(child != -1, "Root elements must have exited early!");
   std::swap(face_nodes, parent_nodes);
   return child;
}

int NCMesh::find_node(const Element &el, int node)
{
   for (int i = 0; i < MaxElemNodes; i++)
   {
      if (el.node[i] == node) { return i; }
   }
   MFEM_ABORT("Node not found.");
   return -1;
}

int NCMesh::FindNodeExt(const Element &el, int node, bool abort)
{
   for (int i = 0; i < GI[el.Geom()].nv; i++)
   {
      if (RetrieveNode(el, i) == node) { return i; }
   }
   if (abort) { MFEM_ABORT("Node not found."); }
   return -1;
}

int NCMesh::find_element_edge(const Element &el, int vn0, int vn1, bool abort)
{
   MFEM_ASSERT(!el.ref_type, "");

   GeomInfo &gi = GI[el.Geom()];
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      int n0 = el.node[ev[0]];
      int n1 = el.node[ev[1]];
      if ((n0 == vn0 && n1 == vn1) ||
          (n0 == vn1 && n1 == vn0)) { return i; }
   }

   if (abort) { MFEM_ABORT("Edge (" << vn0 << ", " << vn1 << ") not found"); }
   return -1;
}

int NCMesh::find_local_face(int geom, int a, int b, int c)
{
   GeomInfo &gi = GI[geom];
   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      if ((a == fv[0] || a == fv[1] || a == fv[2] || a == fv[3]) &&
          (b == fv[0] || b == fv[1] || b == fv[2] || b == fv[3]) &&
          (c == fv[0] || c == fv[1] || c == fv[2] || c == fv[3]))
      {
         return i;
      }
   }
   MFEM_ABORT("Face not found.");
   return -1;
}

namespace
{
template <typename T> struct IntHash;
template <> struct IntHash<float>
{
   using int_type = uint32_t;
   static constexpr int_type initial_value = 0xc4a016dd; // random value;
};
template <> struct IntHash<double>
{
   using int_type = uint64_t;
   static constexpr int_type initial_value = 0xf9ca9ba106acbba9; // random value
};
}

/// Hash function for a PointMatrix, used in MatrixMap::map.
struct PointMatrixHash
{
   std::size_t operator()(const NCMesh::PointMatrix &pm) const
   {
      // This is a variation on "Hashing an array of floats" from here:
      // https://cs.stackexchange.com/questions/37952

      // Make sure (at compile time) that the types have compatible sizes
      static_assert(sizeof(IntHash<real_t>::int_type) == sizeof(real_t), "");
      // Suppress maybe unused warnings
      MFEM_CONTRACT_VAR(IntHash<float>::initial_value);
      MFEM_CONTRACT_VAR(IntHash<double>::initial_value);

      auto int_bit_cast = [](real_t val)
      {
         // std::memcpy is the proper way of doing type punning, see e.g.
         // https://gist.github.com/shafik/848ae25ee209f698763cffee272a58f8
         IntHash<real_t>::int_type int_val;
         std::memcpy(&int_val, &val, sizeof(real_t));
         return int_val;
      };

      IntHash<real_t>::int_type hash = IntHash<real_t>::initial_value;

      for (int i = 0; i < pm.np; i++)
      {
         for (int j = 0; j < pm.points[i].dim; j++)
         {
            // mix the doubles by adding their binary representations many times
            // over (note: 31 is 11111 in binary)
            real_t coord = pm.points[i].coord[j];
            hash = 31*hash + int_bit_cast(coord);
         }
      }
      return hash; // return the lowest bits of the huge sum
   }
};

/** Helper container to keep track of point matrices encountered during
 *  face/edge traversal and to assign unique indices to them.
 */
struct MatrixMap
{
   int GetIndex(const NCMesh::PointMatrix &pm)
   {
      int &index = map[pm];
      if (!index) { index = map.size(); }
      return index - 1;
   }

   void ExportMatrices(Array<DenseMatrix*> &point_matrices) const
   {
      point_matrices.SetSize(map.size());
      for (const auto &pair : map)
      {
         DenseMatrix* mat = new DenseMatrix();
         pair.first.GetMatrix(*mat);
         point_matrices[pair.second - 1] = mat;
      }
   }

   void DumpBucketSizes() const
   {
      for (unsigned i = 0; i < map.bucket_count(); i++)
      {
         mfem::out << map.bucket_size(i) << " ";
      }
   }

private:
   std::unordered_map<NCMesh::PointMatrix, int, PointMatrixHash> map;
};


int NCMesh::ReorderFacePointMat(int v0, int v1, int v2, int v3,
                                int elem, const PointMatrix &pm,
                                PointMatrix &reordered) const
{
   const Element &el = elements[elem];
   int master[4] =
   {
      find_node(el, v0), find_node(el, v1), find_node(el, v2),
      (v3 >= 0) ? find_node(el, v3) : -1
   };
   int nfv = (v3 >= 0) ? 4 : 3;

   int local = find_local_face(el.Geom(), master[0], master[1], master[2]);
   const int* fv = GI[el.Geom()].faces[local];

   reordered.np = pm.np;
   for (int i = 0, j; i < nfv; i++)
   {
      for (j = 0; j < nfv; j++)
      {
         if (fv[i] == master[j])
         {
            reordered.points[i] = pm.points[j];
            break;
         }
      }
      MFEM_ASSERT(j != nfv, "node not found.");
   }
   return local;
}

void NCMesh::TraverseQuadFace(int vn0, int vn1, int vn2, int vn3,
                              const PointMatrix& pm, int level,
                              Face* eface[4], MatrixMap &matrix_map)
{
   if (level > 0)
   {
      // check if we made it to a face that is not split further
      Face* fa = faces.Find(vn0, vn1, vn2, vn3);
      if (fa)
      {
         // we have a slave face, add it to the list
         int elem = fa->GetSingleElement();
         face_list.slaves.Append(
            Slave(fa->index, elem, -1, Geometry::SQUARE));
         Slave &sl = face_list.slaves.Last();

         // reorder the point matrix according to slave face orientation
         PointMatrix pm_r;
         sl.local = ReorderFacePointMat(vn0, vn1, vn2, vn3, elem, pm, pm_r);
         sl.matrix = matrix_map.GetIndex(pm_r);

         eface[0] = eface[2] = fa;
         eface[1] = eface[3] = fa;

         return;
      }
   }

   // we need to recurse deeper
   int mid[5];
   int split = QuadFaceSplitType(vn0, vn1, vn2, vn3, mid);

   Face *ef[2][4];
   if (split == 1) // "X" split face
   {
      Point pmid0(pm(0), pm(1)), pmid2(pm(2), pm(3));

      TraverseQuadFace(vn0, mid[0], mid[2], vn3,
                       PointMatrix(pm(0), pmid0, pmid2, pm(3)),
                       level+1, ef[0], matrix_map);

      TraverseQuadFace(mid[0], vn1, vn2, mid[2],
                       PointMatrix(pmid0, pm(1), pm(2), pmid2),
                       level+1, ef[1], matrix_map);

      eface[1] = ef[1][1];
      eface[3] = ef[0][3];
      eface[0] = eface[2] = NULL;
   }
   else if (split == 2) // "Y" split face
   {
      Point pmid1(pm(1), pm(2)), pmid3(pm(3), pm(0));

      TraverseQuadFace(vn0, vn1, mid[1], mid[3],
                       PointMatrix(pm(0), pm(1), pmid1, pmid3),
                       level+1, ef[0], matrix_map);

      TraverseQuadFace(mid[3], mid[1], vn2, vn3,
                       PointMatrix(pmid3, pmid1, pm(2), pm(3)),
                       level+1, ef[1], matrix_map);

      eface[0] = ef[0][0];
      eface[2] = ef[1][2];
      eface[1] = eface[3] = NULL;
   }

   // check for a prism edge constrained by the master face
   if (HavePrisms() && mid[4] >= 0)
   {
      Node& enode = nodes[mid[4]];
      if (enode.HasEdge())
      {
         // process the edge only if it's not shared by slave faces within this
         // master face (i.e. the edge is "hidden")
         const int fi[3][2] = {{0, 0}, {1, 3}, {2, 0}};
         if (!ef[0][fi[split][0]] && !ef[1][fi[split][1]])
         {
            MFEM_ASSERT(enode.edge_refc == 1, "");

            MeshId buf[4];
            Array<MeshId> eid(buf, 4);

            (split == 1) ? FindEdgeElements(mid[0], vn1, vn2, mid[2], eid)
            /*        */ : FindEdgeElements(mid[3], vn0, vn1, mid[1], eid);

            MFEM_ASSERT(eid.Size() > 0, "edge prism not found");
            MFEM_ASSERT(eid.Size() < 2, "non-unique edge prism");

            // create a slave face record with a degenerate point matrix
            face_list.slaves.Append(
               Slave(-1 - enode.edge_index,
                     eid[0].element, eid[0].local, Geometry::SQUARE));
            Slave &sl = face_list.slaves.Last();

            if (split == 1)
            {
               Point mid0(pm(0), pm(1)), mid2(pm(2), pm(3));
               int v1 = nodes[mid[0]].vert_index;
               int v2 = nodes[mid[2]].vert_index;
               sl.matrix =
                  matrix_map.GetIndex(
                     (v1 < v2) ? PointMatrix(mid0, mid2, mid2, mid0) :
                     /*       */ PointMatrix(mid2, mid0, mid0, mid2));
            }
            else
            {
               Point mid1(pm(1), pm(2)), mid3(pm(3), pm(0));
               int v1 = nodes[mid[1]].vert_index;
               int v2 = nodes[mid[3]].vert_index;
               sl.matrix =
                  matrix_map.GetIndex(
                     (v1 < v2) ? PointMatrix(mid1, mid3, mid3, mid1) :
                     /*       */ PointMatrix(mid3, mid1, mid1, mid3));
            }
         }
      }
   }
}

void NCMesh::TraverseTetEdge(int vn0, int vn1, const Point &p0, const Point &p1,
                             MatrixMap &matrix_map)
{
   int mid = nodes.FindId(vn0, vn1);
   if (mid < 0) { return; }

   const Node &nd = nodes[mid];
   if (nd.HasEdge())
   {
      // check if the edge is already a master in 'edge_list'
      const auto eid_and_type = edge_list.GetMeshIdAndType(nd.edge_index);
      if (eid_and_type.type == NCList::MeshIdType::MASTER
          || eid_and_type.type == NCList::MeshIdType::CONFORMING)
      {
         // in this case we need to add an edge-face constraint, because the
         // non-slave edge is really a (face-)slave itself.
         const MeshId &eid = *eid_and_type.id;
         face_list.slaves.Append(
            Slave(-1 - eid.index, eid.element, eid.local, Geometry::TRIANGLE));

         int v0index = nodes[vn0].vert_index;
         int v1index = nodes[vn1].vert_index;

         face_list.slaves.Last().matrix =
            matrix_map.GetIndex((v0index < v1index) ? PointMatrix(p0, p1, p0)
                                /*               */ : PointMatrix(p1, p0, p1));

         return; // no need to continue deeper
      }
   }

   // recurse deeper
   Point pmid(p0, p1);
   TraverseTetEdge(vn0, mid, p0, pmid, matrix_map);
   TraverseTetEdge(mid, vn1, pmid, p1, matrix_map);
}

NCMesh::TriFaceTraverseResults NCMesh::TraverseTriFace(int vn0, int vn1,
                                                       int vn2,
                                                       const PointMatrix& pm, int level,
                                                       MatrixMap &matrix_map)
{
   if (level > 0)
   {
      // check if we made it to a face that is not split further
      Face* fa = faces.Find(vn0, vn1, vn2);
      if (fa)
      {
         // we have a slave face, add it to the list
         int elem = fa->GetSingleElement();
         face_list.slaves.Append(
            Slave(fa->index, elem, -1, Geometry::TRIANGLE));
         Slave &sl = face_list.slaves.Last();

         // reorder the point matrix according to slave face orientation
         PointMatrix pm_r;
         sl.local = ReorderFacePointMat(vn0, vn1, vn2, -1, elem, pm, pm_r);
         sl.matrix = matrix_map.GetIndex(pm_r);

         return {true, elements[elem].rank != MyRank};
      }
   }

   int mid[3];
   if (TriFaceSplit(vn0, vn1, vn2, mid))
   {
      Point pmid0(pm(0), pm(1)), pmid1(pm(1), pm(2)), pmid2(pm(2), pm(0));
      TriFaceTraverseResults b[4];

      b[0] = TraverseTriFace(vn0, mid[0], mid[2],
                             PointMatrix(pm(0), pmid0, pmid2),
                             level+1, matrix_map);

      b[1] = TraverseTriFace(mid[0], vn1, mid[1],
                             PointMatrix(pmid0, pm(1), pmid1),
                             level+1, matrix_map);

      b[2] = TraverseTriFace(mid[2], mid[1], vn2,
                             PointMatrix(pmid2, pmid1, pm(2)),
                             level+1, matrix_map);

      b[3] = TraverseTriFace(mid[1], mid[2], mid[0],
                             PointMatrix(pmid1, pmid2, pmid0),
                             level+1, matrix_map);

      // Traverse possible tet edges constrained by the master face. This needs
      // to occur if none of these first NC level faces are split further, OR if
      // they are on different processors. The different processor constraint is
      // needed in the case of local elements constrained by this face via the
      // edge alone. Cannot know this a priori, so just constrain any edge
      // attached to two neighbors.
      if (HaveTets() && (!b[3].unsplit || b[3].ghost_neighbor))
      {
         // If the faces have no further splits, so would not be captured by
         // normal face relations, add possible edge constraints.
         if (!b[1].unsplit || b[1].ghost_neighbor) { TraverseTetEdge(mid[0],mid[1], pmid0,pmid1, matrix_map); }
         if (!b[2].unsplit || b[2].ghost_neighbor) { TraverseTetEdge(mid[1],mid[2], pmid1,pmid2, matrix_map); }
         if (!b[0].unsplit || b[0].ghost_neighbor) { TraverseTetEdge(mid[2],mid[0], pmid2,pmid0, matrix_map); }
      }
   }
   return {false, false};
}

void NCMesh::BuildFaceList()
{
   face_list.Clear();
   if (Dim < 3) { return; }

   if (HaveTets()) { GetEdgeList(); } // needed by TraverseTetEdge()

   boundary_faces.SetSize(0);

   Array<char> processed_faces(faces.NumIds());
   processed_faces = 0;

   MatrixMap matrix_maps[Geometry::NumGeom];

   // visit faces of leaf elements
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      int elem = leaf_elements[i];
      Element &el = elements[elem];
      MFEM_ASSERT(!el.ref_type, "not a leaf element.");

      GeomInfo& gi = GI[el.Geom()];
      for (int j = 0; j < gi.nf; j++)
      {
         // get nodes for this face
         int node[4];
         for (int k = 0; k < 4; k++)
         {
            node[k] = el.node[gi.faces[j][k]];
         }

         int face = faces.FindId(node[0], node[1], node[2], node[3]);
         MFEM_ASSERT(face >= 0, "face not found!");

         // tell ParNCMesh about the face
         ElementSharesFace(elem, j, face);

         // have we already processed this face? skip if yes
         if (processed_faces[face]) { continue; }
         processed_faces[face] = 1;

         int fgeom = (node[3] >= 0) ? Geometry::SQUARE : Geometry::TRIANGLE;

         Face &fa = faces[face];
         bool is_master = false;
         if (fa.elem[0] >= 0 && fa.elem[1] >= 0)
         {
            // this is a conforming face, add it to the list
            face_list.conforming.Append(MeshId(fa.index, elem, j, fgeom));
         }
         else
         {
            // this is either a master face or a slave face, but we can't tell
            // until we traverse the face refinement 'tree'...
            int sb = face_list.slaves.Size();
            if (fgeom == Geometry::SQUARE)
            {
               Face* dummy[4];
               TraverseQuadFace(node[0], node[1], node[2], node[3],
                                pm_quad_identity, 0, dummy, matrix_maps[fgeom]);
            }
            else
            {
               TraverseTriFace(node[0], node[1], node[2],
                               pm_tri_identity, 0, matrix_maps[fgeom]);
            }

            int se = face_list.slaves.Size();
            if (sb < se)
            {
               // found slaves, so this is a master face; add it to the list
               is_master = true;
               face_list.masters.Append(
                  Master(fa.index, elem, j, fgeom, sb, se));

               // also, set the master index for the slaves
               for (int ii = sb; ii < se; ii++)
               {
                  face_list.slaves[ii].master = fa.index;
               }
            }
         }

         // To support internal boundaries can only insert non-master faces.
         if (fa.Boundary() && !is_master) { boundary_faces.Append(face); }
      }
   }

   // export unique point matrices
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      matrix_maps[i].ExportMatrices(face_list.point_matrices[i]);
   }
}

void NCMesh::TraverseEdge(int vn0, int vn1, real_t t0, real_t t1, int flags,
                          int level, MatrixMap &matrix_map)
{
   int mid = nodes.FindId(vn0, vn1);
   if (mid < 0) { return; }

   Node &nd = nodes[mid];
   if (nd.HasEdge() && level > 0)
   {
      // we have a slave edge, add it to the list
      edge_list.slaves.Append(Slave(nd.edge_index, -1, -1, Geometry::SEGMENT));

      Slave &sl = edge_list.slaves.Last();
      sl.matrix = matrix_map.GetIndex(PointMatrix(Point(t0), Point(t1)));

      // handle slave edge orientation
      sl.edge_flags = flags;
      int v0index = nodes[vn0].vert_index;
      int v1index = nodes[vn1].vert_index;
      if (v0index > v1index) { sl.edge_flags |= 2; }
   }

   // recurse deeper
   real_t tmid = (t0 + t1) / 2;
   TraverseEdge(vn0, mid, t0, tmid, flags, level+1, matrix_map);
   TraverseEdge(mid, vn1, tmid, t1, flags, level+1, matrix_map);
}

void NCMesh::BuildEdgeList()
{
   edge_list.Clear();
   if (Dim < 3) { boundary_faces.SetSize(0); }

   Array<char> processed_edges(nodes.NumIds());
   processed_edges = 0;

   Array<int> edge_element(nodes.NumIds());
   Array<signed char> edge_local(nodes.NumIds());
   edge_local = -1;

   MatrixMap matrix_map;

   // visit edges of leaf elements
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      int elem = leaf_elements[i];
      Element &el = elements[elem];
      MFEM_ASSERT(!el.ref_type, "not a leaf element.");

      GeomInfo& gi = GI[el.Geom()];
      for (int j = 0; j < gi.ne; j++)
      {
         // get nodes for this edge
         const int* ev = gi.edges[j];
         int node[2] = { el.node[ev[0]], el.node[ev[1]] };

         int enode = nodes.FindId(node[0], node[1]);
         MFEM_ASSERT(enode >= 0, "edge node not found!");

         Node &nd = nodes[enode];
         MFEM_ASSERT(nd.HasEdge(), "edge not found!");

         // tell ParNCMesh about the edge
         ElementSharesEdge(elem, j, enode);

         // store element/local for later
         edge_element[nd.edge_index] = elem;
         edge_local[nd.edge_index] = j;

         // skip slave edges here, they will be reached from their masters
         if (GetEdgeMaster(enode) >= 0)
         {
            // (2D only, store internal boundary faces)
            if (Dim <= 2)
            {
               int face = faces.FindId(node[0], node[0], node[1], node[1]);
               MFEM_ASSERT(face >= 0, "face not found!");
               if (faces[face].Boundary()) { boundary_faces.Append(face); }
            }
            continue;
         }

         // have we already processed this edge? skip if yes
         if (processed_edges[enode]) { continue; }
         processed_edges[enode] = 1;

         // prepare edge interval for slave traversal, handle orientation
         real_t t0 = 0.0, t1 = 1.0;
         int v0index = nodes[node[0]].vert_index;
         int v1index = nodes[node[1]].vert_index;
         int flags = (v0index > v1index) ? 1 : 0;

         // try traversing the edge to find slave edges
         int sb = edge_list.slaves.Size();
         TraverseEdge(node[0], node[1], t0, t1, flags, 0, matrix_map);

         int se = edge_list.slaves.Size();
         if (sb < se)
         {
            // found slaves, this is a master face; add it to the list
            edge_list.masters.Append(
               Master(nd.edge_index, elem, j, Geometry::SEGMENT, sb, se));

            // also, set the master index for the slaves
            for (int ii = sb; ii < se; ii++)
            {
               edge_list.slaves[ii].master = nd.edge_index;
            }
         }
         else
         {
            // no slaves, this is a conforming edge
            edge_list.conforming.Append(MeshId(nd.edge_index, elem, j));
            // (2D only, store boundary faces)
            if (Dim <= 2)
            {
               int face = faces.FindId(node[0], node[0], node[1], node[1]);
               MFEM_ASSERT(face >= 0, "face not found!");
               if (faces[face].Boundary()) { boundary_faces.Append(face); }
            }
         }
      }
   }

   // fix up slave edge element/local
   for (int i = 0; i < edge_list.slaves.Size(); i++)
   {
      Slave &sl = edge_list.slaves[i];
      int local = edge_local[sl.index];
      if (local >= 0)
      {
         sl.local = local;
         sl.element = edge_element[sl.index];
      }
   }

   // export unique point matrices
   matrix_map.ExportMatrices(edge_list.point_matrices[Geometry::SEGMENT]);
}

void NCMesh::BuildVertexList()
{
   int total = NVertices + NGhostVertices;

   vertex_list.Clear();
   vertex_list.conforming.Reserve(total);

   Array<char> processed_vertices(total);
   processed_vertices = 0;

   // analogously to above, visit vertices of leaf elements
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      int elem = leaf_elements[i];
      Element &el = elements[elem];

      for (int j = 0; j < GI[el.Geom()].nv; j++)
      {
         int node = el.node[j];
         Node &nd = nodes[node];

         int index = nd.vert_index;
         if (index >= 0)
         {
            ElementSharesVertex(elem, j, node);

            if (processed_vertices[index]) { continue; }
            processed_vertices[index] = 1;

            vertex_list.conforming.Append(MeshId(index, elem, j));
         }
      }
   }
}

void NCMesh::NCList::OrientedPointMatrix(const Slave &slave,
                                         DenseMatrix &oriented_matrix) const
{
   oriented_matrix = *(point_matrices[slave.Geom()][slave.matrix]);

   if (slave.edge_flags)
   {
      MFEM_ASSERT(oriented_matrix.Height() == 1 &&
                  oriented_matrix.Width() == 2, "not an edge point matrix");

      if (slave.edge_flags & 1) // master inverted
      {
         oriented_matrix(0,0) = 1.0 - oriented_matrix(0,0);
         oriented_matrix(0,1) = 1.0 - oriented_matrix(0,1);
      }
      if (slave.edge_flags & 2) // slave inverted
      {
         std::swap(oriented_matrix(0,0), oriented_matrix(0,1));
      }
   }
}

void NCMesh::NCList::Clear()
{
   conforming.DeleteAll();
   masters.DeleteAll();
   slaves.DeleteAll();

   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      for (int j = 0; j < point_matrices[i].Size(); j++)
      {
         delete point_matrices[i][j];
      }
      point_matrices[i].DeleteAll();
   }

   inv_index.clear();
}

NCMesh::NCList::MeshIdAndType
NCMesh::NCList::GetMeshIdAndType(int index) const
{
   BuildIndex();
   const auto it = inv_index.find(index);
   auto ft = it != inv_index.end() ? it->second.first : MeshIdType::UNRECOGNIZED;
   switch (ft)
   {
      case MeshIdType::CONFORMING:
         return {&conforming[it->second.second], it->second.first};
      case MeshIdType::MASTER:
         return {&masters[it->second.second], it->second.first};
      case MeshIdType::SLAVE:
         return {&slaves[it->second.second], it->second.first};
      case MeshIdType::UNRECOGNIZED:
      default:
         return {nullptr, MeshIdType::UNRECOGNIZED};
   }
}

NCMesh::NCList::MeshIdType
NCMesh::NCList::GetMeshIdType(int index) const
{
   BuildIndex();
   auto it = inv_index.find(index);
   return (it != inv_index.end()) ? it->second.first : MeshIdType::UNRECOGNIZED;
}

bool
NCMesh::NCList::CheckMeshIdType(int index, MeshIdType ft) const
{
   return GetMeshIdType(index) == ft;
}

void
NCMesh::NCList::BuildIndex() const
{
   if (inv_index.size() == 0)
   {
      auto index_compare = [](const MeshId &a, const MeshId &b) { return a.index < b.index; };
      auto max_conforming = std::max_element(conforming.begin(), conforming.end(),
                                             index_compare);
      auto max_master = std::max_element(masters.begin(), masters.end(),
                                         index_compare);
      auto max_slave = std::max_element(slaves.begin(), slaves.end(), index_compare);

      int max_conforming_index = max_conforming != nullptr ? max_conforming->index :
                                 -1;
      int max_master_index = max_master != nullptr ? max_master->index : -1;
      int max_slave_index = max_slave != nullptr ? max_slave->index : -1;

      inv_index.reserve(max(max_conforming_index, max_master_index, max_slave_index,
                            0));
      for (int i = 0; i < conforming.Size(); i++)
      {
         inv_index.emplace(conforming[i].index, std::make_pair(MeshIdType::CONFORMING,
                                                               i));
      }
      for (int i = 0; i < masters.Size(); i++)
      {
         inv_index.emplace(masters[i].index, std::make_pair(MeshIdType::MASTER, i));
      }
      for (int i = 0; i < slaves.Size(); i++)
      {
         inv_index.emplace(slaves[i].index, std::make_pair(MeshIdType::SLAVE, i));
      }
   }
}

//// Neighbors /////////////////////////////////////////////////////////////////

void NCMesh::CollectEdgeVertices(int v0, int v1, Array<int> &indices)
{
   int mid = nodes.FindId(v0, v1);
   if (mid >= 0 && nodes[mid].HasVertex())
   {
      indices.Append(mid);

      CollectEdgeVertices(v0, mid, indices);
      CollectEdgeVertices(mid, v1, indices);
   }
}

void NCMesh::CollectTriFaceVertices(int v0, int v1, int v2, Array<int> &indices)
{
   int mid[3];
   if (TriFaceSplit(v0, v1, v2, mid))
   {
      for (int i = 0; i < 3; i++)
      {
         indices.Append(mid[i]);
      }

      CollectTriFaceVertices(v0, mid[0], mid[2], indices);
      CollectTriFaceVertices(mid[0], v1, mid[1], indices);
      CollectTriFaceVertices(mid[2], mid[1], v2, indices);
      CollectTriFaceVertices(mid[0], mid[1], mid[2], indices);

      if (HaveTets()) // possible edge-face contact
      {
         CollectEdgeVertices(mid[0], mid[1], indices);
         CollectEdgeVertices(mid[1], mid[2], indices);
         CollectEdgeVertices(mid[2], mid[0], indices);
      }
   }
}

void NCMesh::CollectQuadFaceVertices(int v0, int v1, int v2, int v3,
                                     Array<int> &indices)
{
   int mid[5];
   switch (QuadFaceSplitType(v0, v1, v2, v3, mid))
   {
      case 1:
         indices.Append(mid[0]);
         indices.Append(mid[2]);

         CollectQuadFaceVertices(v0, mid[0], mid[2], v3, indices);
         CollectQuadFaceVertices(mid[0], v1, v2, mid[2], indices);

         if (HavePrisms()) // possible edge-face contact
         {
            CollectEdgeVertices(mid[0], mid[2], indices);
         }
         break;

      case 2:
         indices.Append(mid[1]);
         indices.Append(mid[3]);

         CollectQuadFaceVertices(v0, v1, mid[1], mid[3], indices);
         CollectQuadFaceVertices(mid[3], mid[1], v2, v3, indices);

         if (HavePrisms()) // possible edge-face contact
         {
            CollectEdgeVertices(mid[1], mid[3], indices);
         }
         break;
   }
}

void NCMesh::BuildElementToVertexTable()
{
   int nrows = leaf_elements.Size();
   int* I = Memory<int>(nrows + 1);
   int** JJ = new int*[nrows];

   Array<int> indices;
   indices.Reserve(128);

   // collect vertices coinciding with each element, including hanging vertices
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      int elem = leaf_elements[i];
      Element &el = elements[elem];
      MFEM_ASSERT(!el.ref_type, "not a leaf element.");

      GeomInfo& gi = GI[el.Geom()];
      int* node = el.node;

      indices.SetSize(0);
      for (int j = 0; j < gi.ne; j++)
      {
         const int* ev = gi.edges[j];
         CollectEdgeVertices(node[ev[0]], node[ev[1]], indices);
      }

      if (Dim >= 3)
      {
         for (int j = 0; j < gi.nf; j++)
         {
            const int* fv = gi.faces[j];
            if (gi.nfv[j] == 4)
            {
               CollectQuadFaceVertices(node[fv[0]], node[fv[1]],
                                       node[fv[2]], node[fv[3]], indices);
            }
            else
            {
               CollectTriFaceVertices(node[fv[0]], node[fv[1]], node[fv[2]],
                                      indices);
            }
         }
      }

      // temporarily store one row of the table
      indices.Sort();
      indices.Unique();
      int size = indices.Size();
      I[i] = size;
      JJ[i] = new int[size];
      std::memcpy(JJ[i], indices.GetData(), size * sizeof(int));
   }

   // finalize the I array of the table
   int nnz = 0;
   for (int i = 0; i < nrows; i++)
   {
      int cnt = I[i];
      I[i] = nnz;
      nnz += cnt;
   }
   I[nrows] = nnz;

   // copy the temporarily stored rows into one J array
   int *J = Memory<int>(nnz);
   nnz = 0;
   for (int i = 0; i < nrows; i++)
   {
      int cnt = I[i+1] - I[i];
      std::memcpy(J+nnz, JJ[i], cnt * sizeof(int));
      delete [] JJ[i];
      nnz += cnt;
   }

   element_vertex.SetIJ(I, J, nrows);

   delete [] JJ;
}


void NCMesh::FindSetNeighbors(const Array<char> &elem_set,
                              Array<int> *neighbors,
                              Array<char> *neighbor_set)
{
   // If A is the element-to-vertex table (see 'element_vertex') listing all
   // vertices touching each element, including hanging vertices, then A*A^T is
   // the element-to-neighbor table. Multiplying the element set with A*A^T
   // gives the neighbor set. To save memory, this function only computes the
   // action of A*A^T, the product itself is not stored anywhere.

   // Optimization: the 'element_vertex' table does not store the obvious corner
   // nodes in it. The table is therefore empty for conforming meshes.

   UpdateElementToVertexTable();

   int nleaves = leaf_elements.Size();
   MFEM_VERIFY(elem_set.Size() == nleaves, "");
   MFEM_ASSERT(element_vertex.Size() == nleaves, "");

   // step 1: vertices = A^T * elem_set, i.e, find all vertices touching the
   // element set

   Array<char> vmark(nodes.NumIds());
   vmark = 0;

   for (int i = 0; i < nleaves; i++)
   {
      if (elem_set[i])
      {
         int *v = element_vertex.GetRow(i);
         int nv = element_vertex.RowSize(i);
         for (int j = 0; j < nv; j++)
         {
            vmark[v[j]] = 1;
         }

         Element &el = elements[leaf_elements[i]];
         nv = GI[el.Geom()].nv;
         for (int j = 0; j < nv; j++)
         {
            vmark[el.node[j]] = 1;
         }
      }
   }

   // step 2: neighbors = A * vertices, i.e., find all elements coinciding with
   // vertices from step 1; NOTE: in the result we don't include elements from
   // the original set

   if (neighbor_set)
   {
      neighbor_set->SetSize(nleaves);
      *neighbor_set = 0;
   }

   for (int i = 0; i < nleaves; i++)
   {
      if (!elem_set[i])
      {
         bool hit = false;

         int *v = element_vertex.GetRow(i);
         int nv = element_vertex.RowSize(i);
         for (int j = 0; j < nv; j++)
         {
            if (vmark[v[j]]) { hit = true; break; }
         }

         if (!hit)
         {
            Element &el = elements[leaf_elements[i]];
            nv = GI[el.Geom()].nv;
            for (int j = 0; j < nv; j++)
            {
               if (vmark[el.node[j]]) { hit = true; break; }
            }
         }

         if (hit)
         {
            if (neighbors) { neighbors->Append(leaf_elements[i]); }
            if (neighbor_set) { (*neighbor_set)[i] = 1; }
         }
      }
   }
}

static bool sorted_lists_intersect(const int* a, const int* b, int na, int nb)
{
   // pointers to "end" sentinel, not last entry. Not for dereferencing.
   const int * const a_end = a + na;
   const int * const b_end = b + nb;
   while (a != a_end && b != b_end)
   {
      if (*a < *b)
      {
         ++a;
      }
      else if (*b < *a)
      {
         ++b;
      }
      else
      {
         return true; // neither *a < *b nor *b < *a thus a == b
      }
   }
   return false; // no common element found
}


void NCMesh::FindNeighbors(int elem, Array<int> &neighbors,
                           const Array<int> *search_set)
{
   // TODO future: this function is inefficient. For a single element, an octree
   // neighbor search algorithm would be better. However, the octree neighbor
   // algorithm is hard to get right in the multi-octree case due to the
   // different orientations of the octrees (i.e., the root elements).

   UpdateElementToVertexTable();

   // sorted list of all vertex nodes touching 'elem'
   Array<int> vert;
   vert.Reserve(128);

   // support for non-leaf 'elem', collect vertices of all children
   Array<int> stack;
   stack.Reserve(64);
   stack.Append(elem);

   while (stack.Size())
   {
      Element &el = elements[stack.Last()];
      stack.DeleteLast();

      if (!el.ref_type)
      {
         int *v = element_vertex.GetRow(el.index);
         int nv = element_vertex.RowSize(el.index);
         for (int i = 0; i < nv; i++)
         {
            vert.Append(v[i]);
         }

         nv = GI[el.Geom()].nv;
         for (int i = 0; i < nv; i++)
         {
            vert.Append(el.node[i]);
         }
      }
      else
      {
         for (int i = 0; i < MaxElemChildren && el.child[i] >= 0; i++)
         {
            stack.Append(el.child[i]);
         }
      }
   }

   vert.Sort();
   vert.Unique();

   int *v1 = vert.GetData();
   int nv1 = vert.Size();

   if (!search_set) { search_set = &leaf_elements; }

   // test *all* potential neighbors from the search set
   for (int i = 0; i < search_set->Size(); i++)
   {
      int testme = (*search_set)[i];
      if (testme != elem)
      {
         Element &el = elements[testme];
         int *v2 = element_vertex.GetRow(el.index);
         int nv2 = element_vertex.RowSize(el.index);

         bool hit = sorted_lists_intersect(v1, v2, nv1, nv2);

         if (!hit)
         {
            int nv = GI[el.Geom()].nv;
            for (int j = 0; j < nv; j++)
            {
               hit = sorted_lists_intersect(&el.node[j], v1, 1, nv1);
               if (hit) { break; }
            }
         }

         if (hit) { neighbors.Append(testme); }
      }
   }
}

void NCMesh::NeighborExpand(const Array<int> &elems,
                            Array<int> &expanded,
                            const Array<int> *search_set)
{
   UpdateElementToVertexTable();

   Array<char> vmark(nodes.NumIds());
   vmark = 0;

   for (int i = 0; i < elems.Size(); i++)
   {
      Element &el = elements[elems[i]];

      int *v = element_vertex.GetRow(el.index);
      int nv = element_vertex.RowSize(el.index);
      for (int j = 0; j < nv; j++)
      {
         vmark[v[j]] = 1;
      }

      nv = GI[el.Geom()].nv;
      for (int j = 0; j < nv; j++)
      {
         vmark[el.node[j]] = 1;
      }
   }

   if (!search_set)
   {
      search_set = &leaf_elements;
   }

   expanded.SetSize(0);
   for (int i = 0; i < search_set->Size(); i++)
   {
      int testme = (*search_set)[i];
      Element &el = elements[testme];
      bool hit = false;

      int *v = element_vertex.GetRow(el.index);
      int nv = element_vertex.RowSize(el.index);
      for (int j = 0; j < nv; j++)
      {
         if (vmark[v[j]]) { hit = true; break; }
      }

      if (!hit)
      {
         nv = GI[el.Geom()].nv;
         for (int j = 0; j < nv; j++)
         {
            if (vmark[el.node[j]]) { hit = true; break; }
         }
      }

      if (hit) { expanded.Append(testme); }
   }
}

int NCMesh::GetVertexRootCoord(int elem, RefCoord coord[3]) const
{
   while (1)
   {
      const Element &el = elements[elem];
      if (el.parent < 0) { return elem; }

      const Element &pa = elements[el.parent];
      MFEM_ASSERT(pa.ref_type, "internal error");

      int ch = 0;
      while (ch < MaxElemChildren && pa.child[ch] != elem) { ch++; }
      MFEM_ASSERT(ch < MaxElemChildren, "internal error");

      MFEM_ASSERT(geom_parent[el.Geom()], "unsupported geometry");
      const RefTrf &tr = geom_parent[el.Geom()][(int) pa.ref_type][ch];
      tr.Apply(coord, coord);

      elem = el.parent;
   }
}

static bool RefPointInside(Geometry::Type geom, const RefCoord pt[3])
{
   switch (geom)
   {
      case Geometry::SQUARE:
         return (pt[0] >= 0) && (pt[0] <= T_ONE) &&
                (pt[1] >= 0) && (pt[1] <= T_ONE);

      case Geometry::CUBE:
         return (pt[0] >= 0) && (pt[0] <= T_ONE) &&
                (pt[1] >= 0) && (pt[1] <= T_ONE) &&
                (pt[2] >= 0) && (pt[2] <= T_ONE);

      case Geometry::TRIANGLE:
         return (pt[0] >= 0) && (pt[1] >= 0) && (pt[0] + pt[1] <= T_ONE);

      case Geometry::PRISM:
         return (pt[0] >= 0) && (pt[1] >= 0) && (pt[0] + pt[1] <= T_ONE) &&
                (pt[2] >= 0) && (pt[2] <= T_ONE);

      case Geometry::PYRAMID:
         return (pt[0] >= 0) && (pt[1] >= 0) && (pt[2] >= 0.0) &&
                (pt[0] + pt[2] <= T_ONE) && (pt[1] + pt[2] <= T_ONE) &&
                (pt[2] <= T_ONE);

      default:
         MFEM_ABORT("unsupported geometry");
         return false;
   }
}

void NCMesh::CollectIncidentElements(int elem, const RefCoord coord[3],
                                     Array<int> &list) const
{
   const Element &el = elements[elem];
   if (!el.ref_type)
   {
      list.Append(elem);
      return;
   }

   RefCoord tcoord[3];
   for (int ch = 0; ch < MaxElemChildren && el.child[ch] >= 0; ch++)
   {
      const RefTrf &tr = geom_child[el.Geom()][(int) el.ref_type][ch];
      tr.Apply(coord, tcoord);

      if (RefPointInside(el.Geom(), tcoord))
      {
         CollectIncidentElements(el.child[ch], tcoord, list);
      }
   }
}

void NCMesh::FindVertexCousins(int elem, int local, Array<int> &cousins) const
{
   const Element &el = elements[elem];

   RefCoord coord[3];
   MFEM_ASSERT(geom_corners[el.Geom()], "unsupported geometry");
   std::memcpy(coord, geom_corners[el.Geom()][local], sizeof(coord));

   int root = GetVertexRootCoord(elem, coord);

   cousins.SetSize(0);
   CollectIncidentElements(root, coord, cousins);
}


//// Coarse/fine transformations ///////////////////////////////////////////////

bool NCMesh::PointMatrix::operator==(const PointMatrix &pm) const
{
   MFEM_ASSERT(np == pm.np, "");
   for (int i = 0; i < np; i++)
   {
      MFEM_ASSERT(points[i].dim == pm.points[i].dim, "");
      for (int j = 0; j < points[i].dim; j++)
      {
         if (points[i].coord[j] != pm.points[i].coord[j]) { return false; }
      }
   }
   return true;
}

void NCMesh::PointMatrix::GetMatrix(DenseMatrix& point_matrix) const
{
   point_matrix.SetSize(points[0].dim, np);
   for (int i = 0; i < np; i++)
   {
      for (int j = 0; j < points[0].dim; j++)
      {
         point_matrix(j, i) = points[i].coord[j];
      }
   }
}

NCMesh::PointMatrix NCMesh::pm_seg_identity(
   Point(0), Point(1)
);
NCMesh::PointMatrix NCMesh::pm_tri_identity(
   Point(0, 0), Point(1, 0), Point(0, 1)
);
NCMesh::PointMatrix NCMesh::pm_quad_identity(
   Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)
);
NCMesh::PointMatrix NCMesh::pm_tet_identity(
   Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0), Point(0, 0, 1)
);
NCMesh::PointMatrix NCMesh::pm_prism_identity(
   Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0),
   Point(0, 0, 1), Point(1, 0, 1), Point(0, 1, 1)
);
NCMesh::PointMatrix NCMesh::pm_pyramid_identity(
   Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0),
   Point(0, 1, 0), Point(0, 0, 1)
);
NCMesh::PointMatrix NCMesh::pm_hex_identity(
   Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0),
   Point(0, 0, 1), Point(1, 0, 1), Point(1, 1, 1), Point(0, 1, 1)
);

const NCMesh::PointMatrix& NCMesh::GetGeomIdentity(Geometry::Type geom)
{
   switch (geom)
   {
      case Geometry::SEGMENT:     return pm_seg_identity;
      case Geometry::TRIANGLE:    return pm_tri_identity;
      case Geometry::SQUARE:      return pm_quad_identity;
      case Geometry::TETRAHEDRON: return pm_tet_identity;
      case Geometry::PRISM:       return pm_prism_identity;
      case Geometry::PYRAMID:     return pm_pyramid_identity;
      case Geometry::CUBE:        return pm_hex_identity;
      default:
         MFEM_ABORT("unsupported geometry " << geom);
         return pm_tri_identity;
   }
}

void NCMesh::GetPointMatrix(Geometry::Type geom, const char* ref_path,
                            DenseMatrix& matrix) const
{
   PointMatrix pm = GetGeomIdentity(geom);

   while (*ref_path)
   {
      int ref_type = *ref_path++;
      int child = *ref_path++;

      // TODO: do this with the new child transform tables

      if (geom == Geometry::CUBE)
      {
         if (ref_type == 1) // split along X axis
         {
            Point mid01(pm(0), pm(1)), mid23(pm(2), pm(3));
            Point mid67(pm(6), pm(7)), mid45(pm(4), pm(5));

            if (child == 0)
            {
               pm = PointMatrix(pm(0), mid01, mid23, pm(3),
                                pm(4), mid45, mid67, pm(7));
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid01, pm(1), pm(2), mid23,
                                mid45, pm(5), pm(6), mid67);
            }
         }
         else if (ref_type == 2) // split along Y axis
         {
            Point mid12(pm(1), pm(2)), mid30(pm(3), pm(0));
            Point mid56(pm(5), pm(6)), mid74(pm(7), pm(4));

            if (child == 0)
            {
               pm = PointMatrix(pm(0), pm(1), mid12, mid30,
                                pm(4), pm(5), mid56, mid74);
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid30, mid12, pm(2), pm(3),
                                mid74, mid56, pm(6), pm(7));
            }
         }
         else if (ref_type == 4) // split along Z axis
         {
            Point mid04(pm(0), pm(4)), mid15(pm(1), pm(5));
            Point mid26(pm(2), pm(6)), mid37(pm(3), pm(7));

            if (child == 0)
            {
               pm = PointMatrix(pm(0), pm(1), pm(2), pm(3),
                                mid04, mid15, mid26, mid37);
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid04, mid15, mid26, mid37,
                                pm(4), pm(5), pm(6), pm(7));
            }
         }
         else if (ref_type == 3) // XY split
         {
            Point mid01(pm(0), pm(1)), mid12(pm(1), pm(2));
            Point mid23(pm(2), pm(3)), mid30(pm(3), pm(0));
            Point mid45(pm(4), pm(5)), mid56(pm(5), pm(6));
            Point mid67(pm(6), pm(7)), mid74(pm(7), pm(4));

            Point midf0(mid23, mid12, mid01, mid30);
            Point midf5(mid45, mid56, mid67, mid74);

            if (child == 0)
            {
               pm = PointMatrix(pm(0), mid01, midf0, mid30,
                                pm(4), mid45, midf5, mid74);
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid01, pm(1), mid12, midf0,
                                mid45, pm(5), mid56, midf5);
            }
            else if (child == 2)
            {
               pm = PointMatrix(midf0, mid12, pm(2), mid23,
                                midf5, mid56, pm(6), mid67);
            }
            else if (child == 3)
            {
               pm = PointMatrix(mid30, midf0, mid23, pm(3),
                                mid74, midf5, mid67, pm(7));
            }
         }
         else if (ref_type == 5) // XZ split
         {
            Point mid01(pm(0), pm(1)), mid23(pm(2), pm(3));
            Point mid45(pm(4), pm(5)), mid67(pm(6), pm(7));
            Point mid04(pm(0), pm(4)), mid15(pm(1), pm(5));
            Point mid26(pm(2), pm(6)), mid37(pm(3), pm(7));

            Point midf1(mid01, mid15, mid45, mid04);
            Point midf3(mid23, mid37, mid67, mid26);

            if (child == 0)
            {
               pm = PointMatrix(pm(0), mid01, mid23, pm(3),
                                mid04, midf1, midf3, mid37);
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid01, pm(1), pm(2), mid23,
                                midf1, mid15, mid26, midf3);
            }
            else if (child == 2)
            {
               pm = PointMatrix(midf1, mid15, mid26, midf3,
                                mid45, pm(5), pm(6), mid67);
            }
            else if (child == 3)
            {
               pm = PointMatrix(mid04, midf1, midf3, mid37,
                                pm(4), mid45, mid67, pm(7));
            }
         }
         else if (ref_type == 6) // YZ split
         {
            Point mid12(pm(1), pm(2)), mid30(pm(3), pm(0));
            Point mid56(pm(5), pm(6)), mid74(pm(7), pm(4));
            Point mid04(pm(0), pm(4)), mid15(pm(1), pm(5));
            Point mid26(pm(2), pm(6)), mid37(pm(3), pm(7));

            Point midf2(mid12, mid26, mid56, mid15);
            Point midf4(mid30, mid04, mid74, mid37);

            if (child == 0)
            {
               pm = PointMatrix(pm(0), pm(1), mid12, mid30,
                                mid04, mid15, midf2, midf4);
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid30, mid12, pm(2), pm(3),
                                midf4, midf2, mid26, mid37);
            }
            else if (child == 2)
            {
               pm = PointMatrix(mid04, mid15, midf2, midf4,
                                pm(4), pm(5), mid56, mid74);
            }
            else if (child == 3)
            {
               pm = PointMatrix(midf4, midf2, mid26, mid37,
                                mid74, mid56, pm(6), pm(7));
            }
         }
         else if (ref_type == 7) // full isotropic refinement
         {
            Point mid01(pm(0), pm(1)), mid12(pm(1), pm(2));
            Point mid23(pm(2), pm(3)), mid30(pm(3), pm(0));
            Point mid45(pm(4), pm(5)), mid56(pm(5), pm(6));
            Point mid67(pm(6), pm(7)), mid74(pm(7), pm(4));
            Point mid04(pm(0), pm(4)), mid15(pm(1), pm(5));
            Point mid26(pm(2), pm(6)), mid37(pm(3), pm(7));

            Point midf0(mid23, mid12, mid01, mid30);
            Point midf1(mid01, mid15, mid45, mid04);
            Point midf2(mid12, mid26, mid56, mid15);
            Point midf3(mid23, mid37, mid67, mid26);
            Point midf4(mid30, mid04, mid74, mid37);
            Point midf5(mid45, mid56, mid67, mid74);

            Point midel(midf1, midf3);

            if (child == 0)
            {
               pm = PointMatrix(pm(0), mid01, midf0, mid30,
                                mid04, midf1, midel, midf4);
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid01, pm(1), mid12, midf0,
                                midf1, mid15, midf2, midel);
            }
            else if (child == 2)
            {
               pm = PointMatrix(midf0, mid12, pm(2), mid23,
                                midel, midf2, mid26, midf3);
            }
            else if (child == 3)
            {
               pm = PointMatrix(mid30, midf0, mid23, pm(3),
                                midf4, midel, midf3, mid37);
            }
            else if (child == 4)
            {
               pm = PointMatrix(mid04, midf1, midel, midf4,
                                pm(4), mid45, midf5, mid74);
            }
            else if (child == 5)
            {
               pm = PointMatrix(midf1, mid15, midf2, midel,
                                mid45, pm(5), mid56, midf5);
            }
            else if (child == 6)
            {
               pm = PointMatrix(midel, midf2, mid26, midf3,
                                midf5, mid56, pm(6), mid67);
            }
            else if (child == 7)
            {
               pm = PointMatrix(midf4, midel, midf3, mid37,
                                mid74, midf5, mid67, pm(7));
            }
         }
      }
      else if (geom == Geometry::PRISM)
      {
         if (ref_type < 4) // XY split
         {
            Point mid01(pm(0), pm(1)), mid12(pm(1), pm(2));
            Point mid20(pm(2), pm(0)), mid34(pm(3), pm(4));
            Point mid45(pm(4), pm(5)), mid53(pm(5), pm(3));

            if (child == 0)
            {
               pm = PointMatrix(pm(0), mid01, mid20, pm(3), mid34, mid53);
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid01, pm(1), mid12, mid34, pm(4), mid45);
            }
            else if (child == 2)
            {
               pm = PointMatrix(mid20, mid12, pm(2), mid53, mid45, pm(5));
            }
            else if (child == 3)
            {
               pm = PointMatrix(mid12, mid20, mid01, mid45, mid53, mid34);
            }
         }
         else if (ref_type == 4) // Z split
         {
            Point mid03(pm(0), pm(3)), mid14(pm(1), pm(4)), mid25(pm(2), pm(5));

            if (child == 0)
            {
               pm = PointMatrix(pm(0), pm(1), pm(2), mid03, mid14, mid25);
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid03, mid14, mid25, pm(3), pm(4), pm(5));
            }
         }
         else // ref_type > 4, iso split
         {
            Point mid01(pm(0), pm(1)), mid12(pm(1), pm(2)), mid20(pm(2), pm(0));
            Point mid34(pm(3), pm(4)), mid45(pm(4), pm(5)), mid53(pm(5), pm(3));
            Point mid03(pm(0), pm(3)), mid14(pm(1), pm(4)), mid25(pm(2), pm(5));

            Point midf2(mid01, mid14, mid34, mid03);
            Point midf3(mid12, mid25, mid45, mid14);
            Point midf4(mid20, mid03, mid53, mid25);

            if (child == 0)
            {
               pm = PointMatrix(pm(0), mid01, mid20, mid03, midf2, midf4);
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid01, pm(1), mid12, midf2, mid14, midf3);
            }
            else if (child == 2)
            {
               pm = PointMatrix(mid20, mid12, pm(2), midf4, midf3, mid25);
            }
            else if (child == 3)
            {
               pm = PointMatrix(mid12, mid20, mid01, midf3, midf4, midf2);
            }
            else if (child == 4)
            {
               pm = PointMatrix(mid03, midf2, midf4, pm(3), mid34, mid53);
            }
            else if (child == 5)
            {
               pm = PointMatrix(midf2, mid14, midf3, mid34, pm(4), mid45);
            }
            else if (child == 6)
            {
               pm = PointMatrix(midf4, midf3, mid25, mid53, mid45, pm(5));
            }
            else if (child == 7)
            {
               pm = PointMatrix(midf3, midf4, midf2, mid45, mid53, mid34);
            }
         }
      }
      else if (geom == Geometry::PYRAMID)
      {
         Point mid01(pm(0), pm(1)), mid23(pm(2), pm(3));
         Point mid03(pm(0), pm(3)), mid12(pm(1), pm(2));
         Point mid04(pm(0), pm(4)), mid14(pm(1), pm(4));
         Point mid24(pm(2), pm(4)), mid34(pm(3), pm(4));
         Point midf0(mid23, mid12, mid01, mid03);

         if (child == 0)        // Pyramid
         {
            pm = PointMatrix(pm(0), mid01, midf0, mid03, mid04);
         }
         else if (child == 1)   // Pyramid
         {
            pm = PointMatrix(mid01, pm(1), mid12, midf0, mid14);
         }
         else if (child == 2)   // Pyramid
         {
            pm = PointMatrix(midf0, mid12, pm(2), mid23, mid24);
         }
         else if (child == 3)   // Pyramid
         {
            pm = PointMatrix(mid03, midf0, mid23, pm(3), mid34);
         }
         else if (child == 4)   // Pyramid
         {
            pm = PointMatrix(mid24, mid14, mid04, mid34, midf0);
         }
         else if (child == 5)   // Pyramid
         {
            pm = PointMatrix(mid04, mid14, mid24, mid34, pm(4));
         }
         else if (child == 6)   // Tet
         {
            pm = PointMatrix(mid01, midf0, mid04, mid14);
         }
         else if (child == 7)   // Tet
         {
            pm = PointMatrix(midf0, mid14, mid12, mid24);
         }
         else if (child == 8)   // Tet
         {
            pm = PointMatrix(midf0, mid23, mid34, mid24);
         }
         else if (child == 9)   // Tet
         {
            pm = PointMatrix(mid03, mid04, midf0, mid34);
         }
      }
      else if (geom == Geometry::TETRAHEDRON)
      {
         Point mid01(pm(0), pm(1)), mid12(pm(1), pm(2)), mid02(pm(2), pm(0));
         Point mid03(pm(0), pm(3)), mid13(pm(1), pm(3)), mid23(pm(2), pm(3));

         if (child == 0)
         {
            pm = PointMatrix(pm(0), mid01, mid02, mid03);
         }
         else if (child == 1)
         {
            pm = PointMatrix(mid01, pm(1), mid12, mid13);
         }
         else if (child == 2)
         {
            pm = PointMatrix(mid02, mid12, pm(2), mid23);
         }
         else if (child == 3)
         {
            pm = PointMatrix(mid03, mid13, mid23, pm(3));
         }
         else if (child == 4)
         {
            pm = PointMatrix(mid01, mid23, mid02, mid03);
         }
         else if (child == 5)
         {
            pm = PointMatrix(mid01, mid23, mid03, mid13);
         }
         else if (child == 6)
         {
            pm = PointMatrix(mid01, mid23, mid13, mid12);
         }
         else if (child == 7)
         {
            pm = PointMatrix(mid01, mid23, mid12, mid02);
         }
      }
      else if (geom == Geometry::SQUARE)
      {
         if (ref_type == 1) // X split
         {
            Point mid01(pm(0), pm(1)), mid23(pm(2), pm(3));

            if (child == 0)
            {
               pm = PointMatrix(pm(0), mid01, mid23, pm(3));
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid01, pm(1), pm(2), mid23);
            }
         }
         else if (ref_type == 2) // Y split
         {
            Point mid12(pm(1), pm(2)), mid30(pm(3), pm(0));

            if (child == 0)
            {
               pm = PointMatrix(pm(0), pm(1), mid12, mid30);
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid30, mid12, pm(2), pm(3));
            }
         }
         else if (ref_type == 3) // iso split
         {
            Point mid01(pm(0), pm(1)), mid12(pm(1), pm(2));
            Point mid23(pm(2), pm(3)), mid30(pm(3), pm(0));
            Point midel(mid01, mid23);

            if (child == 0)
            {
               pm = PointMatrix(pm(0), mid01, midel, mid30);
            }
            else if (child == 1)
            {
               pm = PointMatrix(mid01, pm(1), mid12, midel);
            }
            else if (child == 2)
            {
               pm = PointMatrix(midel, mid12, pm(2), mid23);
            }
            else if (child == 3)
            {
               pm = PointMatrix(mid30, midel, mid23, pm(3));
            }
         }
      }
      else if (geom == Geometry::TRIANGLE)
      {
         Point mid01(pm(0), pm(1)), mid12(pm(1), pm(2)), mid20(pm(2), pm(0));

         if (child == 0)
         {
            pm = PointMatrix(pm(0), mid01, mid20);
         }
         else if (child == 1)
         {
            pm = PointMatrix(mid01, pm(1), mid12);
         }
         else if (child == 2)
         {
            pm = PointMatrix(mid20, mid12, pm(2));
         }
         else if (child == 3)
         {
            pm = PointMatrix(mid12, mid20, mid01);
         }
      }
      else if (geom == Geometry::SEGMENT)
      {
         Point mid01(pm(0), pm(1));

         if (child == 0)
         {
            pm = PointMatrix(pm(0), mid01);
         }
         else if (child == 1)
         {
            pm = PointMatrix(mid01, pm(1));
         }
      }
   }

   // write the points to the matrix
   for (int i = 0; i < pm.np; i++)
   {
      for (int j = 0; j < pm(i).dim; j++)
      {
         matrix(j, i) = pm(i).coord[j];
      }
   }
}

void NCMesh::MarkCoarseLevel()
{
   coarse_elements.SetSize(leaf_elements.Size());
   coarse_elements.SetSize(0);

   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      int elem = leaf_elements[i];
      if (!IsGhost(elements[elem])) { coarse_elements.Append(elem); }
   }

   transforms.embeddings.DeleteAll();
}

void NCMesh::TraverseRefinements(int elem, int coarse_index,
                                 std::string &ref_path, RefPathMap &map) const
{
   const Element &el = elements[elem];
   if (!el.ref_type)
   {
      int &matrix = map[ref_path];
      if (!matrix) { matrix = map.size(); }

      Embedding &emb = transforms.embeddings[el.index];
      emb.parent = coarse_index;
      emb.matrix = matrix - 1;
      emb.geom = el.Geom();
      emb.ghost = IsGhost(el);
   }
   else
   {
      MFEM_ASSERT(el.tet_type == 0, "not implemented");

      ref_path.push_back(el.ref_type);
      ref_path.push_back(0);

      for (int i = 0; i < MaxElemChildren; i++)
      {
         if (el.child[i] >= 0)
         {
            ref_path[ref_path.length()-1] = i;
            TraverseRefinements(el.child[i], coarse_index, ref_path, map);
         }
      }
      ref_path.resize(ref_path.length()-2);
   }
}

const CoarseFineTransformations& NCMesh::GetRefinementTransforms() const
{
   MFEM_VERIFY(coarse_elements.Size() || !leaf_elements.Size(),
               "GetRefinementTransforms() must be preceded by MarkCoarseLevel()"
               " and Refine().");

   if (!transforms.embeddings.Size())
   {
      transforms.Clear();
      transforms.embeddings.SetSize(NElements);

      std::string ref_path;
      ref_path.reserve(100);

      RefPathMap path_map[Geometry::NumGeom];
      for (int g = 0; g < Geometry::NumGeom; g++)
      {
         path_map[g][ref_path] = 1; // empty path == identity
      }

      int used_geoms = 0;
      for (int i = 0; i < coarse_elements.Size(); i++)
      {
         int geom = elements[coarse_elements[i]].geom;
         TraverseRefinements(coarse_elements[i], i, ref_path, path_map[geom]);
         used_geoms |= (1 << geom);
      }

      for (int g = 0; g < Geometry::NumGeom; g++)
      {
         if (used_geoms & (1 << g))
         {
            Geometry::Type geom = Geometry::Type(g);
            const PointMatrix &identity = GetGeomIdentity(geom);

            transforms.point_matrices[g]
            .SetSize(Dim, identity.np, path_map[g].size());

            // calculate the point matrices
            RefPathMap::iterator it;
            for (it = path_map[g].begin(); it != path_map[g].end(); ++it)
            {
               GetPointMatrix(geom, it->first.c_str(),
                              transforms.point_matrices[g](it->second-1));
            }
         }
      }
   }
   return transforms;
}

const CoarseFineTransformations& NCMesh::GetDerefinementTransforms() const
{
   MFEM_VERIFY(transforms.embeddings.Size() || !leaf_elements.Size(),
               "GetDerefinementTransforms() must be preceded by Derefine().");

   if (!transforms.IsInitialized())
   {
      std::map<int, int> mat_no[Geometry::NumGeom];
      for (int g = 0; g < Geometry::NumGeom; g++)
      {
         mat_no[g][0] = 1; // 0 == identity
      }

      // assign numbers to the different matrices used
      for (int i = 0; i < transforms.embeddings.Size(); i++)
      {
         Embedding &emb = transforms.embeddings[i];
         int code = emb.matrix; // see SetDerefMatrixCodes()
         if (code)
         {
            int &matrix = mat_no[emb.geom][code];
            if (!matrix) { matrix = mat_no[emb.geom].size(); }

            emb.matrix = matrix - 1;
         }
      }

      for (int g = 0; g < Geometry::NumGeom; g++)
      {
         if (Geoms & (1 << g))
         {
            Geometry::Type geom = Geometry::Type(g);
            const PointMatrix &identity = GetGeomIdentity(geom);

            transforms.point_matrices[geom]
            .SetSize(Dim, identity.np, mat_no[geom].size());

            // calculate point matrices
            for (auto it = mat_no[geom].begin(); it != mat_no[geom].end(); ++it)
            {
               char path[3] = { 0, 0, 0 };

               int code = it->first;
               if (code)
               {
                  path[0] = code >> 4;  // ref_type (see SetDerefMatrixCodes())
                  path[1] = code & 0xf; // child
               }

               GetPointMatrix(geom, path,
                              transforms.point_matrices[geom](it->second-1));
            }
         }
      }
   }
   return transforms;
}

void CoarseFineTransformations::MakeCoarseToFineTable(Table &coarse_to_fine,
                                                      bool want_ghosts) const
{
   Array<Connection> conn;
   conn.Reserve(embeddings.Size());

   int max_parent = -1;
   for (int i = 0; i < embeddings.Size(); i++)
   {
      const Embedding &emb = embeddings[i];
      if ((emb.parent >= 0) &&
          (!emb.ghost || want_ghosts))
      {
         conn.Append(Connection(emb.parent, i));
         max_parent = std::max(emb.parent, max_parent);
      }
   }

   conn.Sort(); // NOTE: unique is not necessary
   coarse_to_fine.MakeFromList(max_parent+1, conn);
}

void NCMesh::ClearTransforms()
{
   coarse_elements.DeleteAll();
   transforms.Clear();
}

void CoarseFineTransformations::Clear()
{
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      point_matrices[i].SetSize(0, 0, 0);
   }
   embeddings.DeleteAll();
}

bool CoarseFineTransformations::IsInitialized() const
{
   // return true if point matrices are present for any geometry
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      if (point_matrices[i].SizeK()) { return true; }
   }
   return false;
}

void Swap(CoarseFineTransformations &a, CoarseFineTransformations &b)
{
   for (int g = 0; g < Geometry::NumGeom; ++g)
   {
      a.point_matrices[g].Swap(b.point_matrices[g]);
   }
   Swap(a.embeddings, b.embeddings);
}


//// SFC Ordering //////////////////////////////////////////////////////////////

static int sgn(int x)
{
   return (x < 0) ? -1 : (x > 0) ? 1 : 0;
}

static void HilbertSfc2D(int x, int y, int ax, int ay, int bx, int by,
                         Array<int> &coords)
{
   int w = std::abs(ax + ay);
   int h = std::abs(bx + by);

   int dax = sgn(ax), day = sgn(ay); // unit major direction ("right")
   int dbx = sgn(bx), dby = sgn(by); // unit orthogonal direction ("up")

   if (h == 1) // trivial row fill
   {
      for (int i = 0; i < w; i++, x += dax, y += day)
      {
         coords.Append(x);
         coords.Append(y);
      }
      return;
   }
   if (w == 1) // trivial column fill
   {
      for (int i = 0; i < h; i++, x += dbx, y += dby)
      {
         coords.Append(x);
         coords.Append(y);
      }
      return;
   }

   int ax2 = ax/2, ay2 = ay/2;
   int bx2 = bx/2, by2 = by/2;

   int w2 = std::abs(ax2 + ay2);
   int h2 = std::abs(bx2 + by2);

   if (2*w > 3*h) // long case: split in two parts only
   {
      if ((w2 & 0x1) && (w > 2))
      {
         ax2 += dax, ay2 += day; // prefer even steps
      }

      HilbertSfc2D(x, y, ax2, ay2, bx, by, coords);
      HilbertSfc2D(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by, coords);
   }
   else // standard case: one step up, one long horizontal step, one step down
   {
      if ((h2 & 0x1) && (h > 2))
      {
         bx2 += dbx, by2 += dby; // prefer even steps
      }

      HilbertSfc2D(x, y, bx2, by2, ax2, ay2, coords);
      HilbertSfc2D(x+bx2, y+by2, ax, ay, bx-bx2, by-by2, coords);
      HilbertSfc2D(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby),
                   -bx2, -by2, -(ax-ax2), -(ay-ay2), coords);
   }
}

static void HilbertSfc3D(int x, int y, int z,
                         int ax, int ay, int az,
                         int bx, int by, int bz,
                         int cx, int cy, int cz,
                         Array<int> &coords)
{
   int w = std::abs(ax + ay + az);
   int h = std::abs(bx + by + bz);
   int d = std::abs(cx + cy + cz);

   int dax = sgn(ax), day = sgn(ay), daz = sgn(az); // unit major dir ("right")
   int dbx = sgn(bx), dby = sgn(by), dbz = sgn(bz); // unit ortho dir ("forward")
   int dcx = sgn(cx), dcy = sgn(cy), dcz = sgn(cz); // unit ortho dir ("up")

   // trivial row/column fills
   if (h == 1 && d == 1)
   {
      for (int i = 0; i < w; i++, x += dax, y += day, z += daz)
      {
         coords.Append(x);
         coords.Append(y);
         coords.Append(z);
      }
      return;
   }
   if (w == 1 && d == 1)
   {
      for (int i = 0; i < h; i++, x += dbx, y += dby, z += dbz)
      {
         coords.Append(x);
         coords.Append(y);
         coords.Append(z);
      }
      return;
   }
   if (w == 1 && h == 1)
   {
      for (int i = 0; i < d; i++, x += dcx, y += dcy, z += dcz)
      {
         coords.Append(x);
         coords.Append(y);
         coords.Append(z);
      }
      return;
   }

   int ax2 = ax/2, ay2 = ay/2, az2 = az/2;
   int bx2 = bx/2, by2 = by/2, bz2 = bz/2;
   int cx2 = cx/2, cy2 = cy/2, cz2 = cz/2;

   int w2 = std::abs(ax2 + ay2 + az2);
   int h2 = std::abs(bx2 + by2 + bz2);
   int d2 = std::abs(cx2 + cy2 + cz2);

   // prefer even steps
   if ((w2 & 0x1) && (w > 2))
   {
      ax2 += dax, ay2 += day, az2 += daz;
   }
   if ((h2 & 0x1) && (h > 2))
   {
      bx2 += dbx, by2 += dby, bz2 += dbz;
   }
   if ((d2 & 0x1) && (d > 2))
   {
      cx2 += dcx, cy2 += dcy, cz2 += dcz;
   }

   // wide case, split in w only
   if ((2*w > 3*h) && (2*w > 3*d))
   {
      HilbertSfc3D(x, y, z,
                   ax2, ay2, az2,
                   bx, by, bz,
                   cx, cy, cz, coords);

      HilbertSfc3D(x+ax2, y+ay2, z+az2,
                   ax-ax2, ay-ay2, az-az2,
                   bx, by, bz,
                   cx, cy, cz, coords);
   }
   // do not split in d
   else if (3*h > 4*d)
   {
      HilbertSfc3D(x, y, z,
                   bx2, by2, bz2,
                   cx, cy, cz,
                   ax2, ay2, az2, coords);

      HilbertSfc3D(x+bx2, y+by2, z+bz2,
                   ax, ay, az,
                   bx-bx2, by-by2, bz-bz2,
                   cx, cy, cz, coords);

      HilbertSfc3D(x+(ax-dax)+(bx2-dbx),
                   y+(ay-day)+(by2-dby),
                   z+(az-daz)+(bz2-dbz),
                   -bx2, -by2, -bz2,
                   cx, cy, cz,
                   -(ax-ax2), -(ay-ay2), -(az-az2), coords);
   }
   // do not split in h
   else if (3*d > 4*h)
   {
      HilbertSfc3D(x, y, z,
                   cx2, cy2, cz2,
                   ax2, ay2, az2,
                   bx, by, bz, coords);

      HilbertSfc3D(x+cx2, y+cy2, z+cz2,
                   ax, ay, az,
                   bx, by, bz,
                   cx-cx2, cy-cy2, cz-cz2, coords);

      HilbertSfc3D(x+(ax-dax)+(cx2-dcx),
                   y+(ay-day)+(cy2-dcy),
                   z+(az-daz)+(cz2-dcz),
                   -cx2, -cy2, -cz2,
                   -(ax-ax2), -(ay-ay2), -(az-az2),
                   bx, by, bz, coords);
   }
   // regular case, split in all w/h/d
   else
   {
      HilbertSfc3D(x, y, z,
                   bx2, by2, bz2,
                   cx2, cy2, cz2,
                   ax2, ay2, az2, coords);

      HilbertSfc3D(x+bx2, y+by2, z+bz2,
                   cx, cy, cz,
                   ax2, ay2, az2,
                   bx-bx2, by-by2, bz-bz2, coords);

      HilbertSfc3D(x+(bx2-dbx)+(cx-dcx),
                   y+(by2-dby)+(cy-dcy),
                   z+(bz2-dbz)+(cz-dcz),
                   ax, ay, az,
                   -bx2, -by2, -bz2,
                   -(cx-cx2), -(cy-cy2), -(cz-cz2), coords);

      HilbertSfc3D(x+(ax-dax)+bx2+(cx-dcx),
                   y+(ay-day)+by2+(cy-dcy),
                   z+(az-daz)+bz2+(cz-dcz),
                   -cx, -cy, -cz,
                   -(ax-ax2), -(ay-ay2), -(az-az2),
                   bx-bx2, by-by2, bz-bz2, coords);

      HilbertSfc3D(x+(ax-dax)+(bx2-dbx),
                   y+(ay-day)+(by2-dby),
                   z+(az-daz)+(bz2-dbz),
                   -bx2, -by2, -bz2,
                   cx2, cy2, cz2,
                   -(ax-ax2), -(ay-ay2), -(az-az2), coords);
   }
}

void NCMesh::GridSfcOrdering2D(int width, int height, Array<int> &coords)
{
   coords.SetSize(0);
   coords.Reserve(2*width*height);

   if (width >= height)
   {
      HilbertSfc2D(0, 0, width, 0, 0, height, coords);
   }
   else
   {
      HilbertSfc2D(0, 0, 0, height, width, 0, coords);
   }
}

void NCMesh::GridSfcOrdering3D(int width, int height, int depth,
                               Array<int> &coords)
{
   coords.SetSize(0);
   coords.Reserve(3*width*height*depth);

   if (width >= height && width >= depth)
   {
      HilbertSfc3D(0, 0, 0,
                   width, 0, 0,
                   0, height, 0,
                   0, 0, depth, coords);
   }
   else if (height >= width && height >= depth)
   {
      HilbertSfc3D(0, 0, 0,
                   0, height, 0,
                   width, 0, 0,
                   0, 0, depth, coords);
   }
   else // depth >= width && depth >= height
   {
      HilbertSfc3D(0, 0, 0,
                   0, 0, depth,
                   width, 0, 0,
                   0, height, 0, coords);
   }
}


//// Utility ///////////////////////////////////////////////////////////////////

void NCMesh::GetEdgeVertices(const MeshId &edge_id, int vert_index[2],
                             bool oriented) const
{
   const Element &el = elements[edge_id.element];
   const GeomInfo& gi = GI[el.Geom()];
   const int* ev = gi.edges[(int) edge_id.local];

   int n0 = el.node[ev[0]], n1 = el.node[ev[1]];
   if (n0 > n1) { std::swap(n0, n1); }

   vert_index[0] = nodes[n0].vert_index;
   vert_index[1] = nodes[n1].vert_index;

   if (oriented && vert_index[0] > vert_index[1])
   {
      std::swap(vert_index[0], vert_index[1]);
   }
}

int NCMesh::GetEdgeNCOrientation(const NCMesh::MeshId &edge_id) const
{
   const Element &el = elements[edge_id.element];
   const GeomInfo& gi = GI[el.Geom()];
   const int* ev = gi.edges[(int) edge_id.local];

   int v0 = nodes[el.node[ev[0]]].vert_index;
   int v1 = nodes[el.node[ev[1]]].vert_index;

   return ((v0 < v1 && ev[0] > ev[1]) || (v0 > v1 && ev[0] < ev[1])) ? -1 : 1;
}

int NCMesh::GetFaceVerticesEdges(const MeshId &face_id,
                                 int vert_index[4], int edge_index[4],
                                 int edge_orientation[4]) const
{
   MFEM_ASSERT(Dim >= 3, "");

   const Element &el = elements[face_id.element];
   const GeomInfo& gi = GI[el.Geom()];

   const int *fv = gi.faces[(int) face_id.local];
   const int nfv = gi.nfv[(int) face_id.local];

   vert_index[3] = edge_index[3] = -1;
   edge_orientation[3] = 0;

   for (int i = 0; i < nfv; i++)
   {
      vert_index[i] = nodes[el.node[fv[i]]].vert_index;
   }

   for (int i = 0; i < nfv; i++)
   {
      int j = i+1;
      if (j >= nfv) { j = 0; }

      int n1 = el.node[fv[i]];
      int n2 = el.node[fv[j]];

      const Node* en = nodes.Find(n1, n2);
      MFEM_ASSERT(en != NULL, "edge not found.");

      edge_index[i] = en->edge_index;
      edge_orientation[i] = (vert_index[i] < vert_index[j]) ? 1 : -1;
   }

   return nfv;
}

int NCMesh::GetEdgeMaster(int node) const
{
   MFEM_ASSERT(node >= 0, "edge node not found.");
   const Node &nd = nodes[node];

   int p1 = nd.p1, p2 = nd.p2;
   MFEM_ASSERT(p1 != p2, "invalid edge node.");

   const Node &n1 = nodes[p1], &n2 = nodes[p2];

   int n1p1 = n1.p1, n1p2 = n1.p2;
   int n2p1 = n2.p1, n2p2 = n2.p2;

   if ((n2p1 != n2p2) && (p1 == n2p1 || p1 == n2p2))
   {
      // n1 is parent of n2: (n1)--(nd)--(n2)------(*)
      if (n2.HasEdge()) { return p2; }
      else { return GetEdgeMaster(p2); }
   }

   if ((n1p1 != n1p2) && (p2 == n1p1 || p2 == n1p2))
   {
      // n2 is parent of n1: (n2)--(nd)--(n1)------(*)
      if (n1.HasEdge()) { return p1; }
      else { return GetEdgeMaster(p1); }
   }

   return -1;
}

int NCMesh::GetEdgeMaster(int v1, int v2) const
{
   int node = nodes.FindId(vertex_nodeId[v1], vertex_nodeId[v2]);
   MFEM_ASSERT(node >= 0 && nodes[node].HasEdge(), "(v1, v2) is not an edge.");

   int master = GetEdgeMaster(node);
   return (master >= 0) ? nodes[master].edge_index : -1;
}

int NCMesh::GetElementDepth(int i) const
{
   int elem = leaf_elements[i];
   int depth = 0, parent;
   while ((parent = elements[elem].parent) != -1)
   {
      elem = parent;
      depth++;
   }
   return depth;
}

int NCMesh::GetElementSizeReduction(int i) const
{
   int elem = leaf_elements[i];
   int parent, reduction = 1;
   while ((parent = elements[elem].parent) != -1)
   {
      if (elements[parent].ref_type & 1) { reduction *= 2; }
      if (elements[parent].ref_type & 2) { reduction *= 2; }
      if (elements[parent].ref_type & 4) { reduction *= 2; }
      elem = parent;
   }
   return reduction;
}

void NCMesh::GetElementFacesAttributes(int leaf_elem,
                                       Array<int> &face_indices,
                                       Array<int> &face_attribs) const
{
   const Element &el = elements[leaf_elements[leaf_elem]];
   const GeomInfo& gi = GI[el.Geom()];

   face_indices.SetSize(gi.nf);
   face_attribs.SetSize(gi.nf);

   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      const Face *face = faces.Find(el.node[fv[0]], el.node[fv[1]],
                                    el.node[fv[2]], el.node[fv[3]]);
      MFEM_ASSERT(face, "face not found");
      face_indices[i] = face->index;
      face_attribs[i] = face->attribute;
   }
}
void NCMesh::FindFaceNodes(int face, int node[4]) const
{
   auto tmp = FindFaceNodes(face);
   std::copy(tmp.begin(), tmp.end(), node);
}

std::array<int, 4> NCMesh::FindFaceNodes(int face) const
{
   return FindFaceNodes(faces[face]);
}

std::array<int, 4> NCMesh::FindFaceNodes(const Face &fa) const
{
   // Obtain face nodes from one of its elements (note that face->p1, p2, p3
   // cannot be used directly since they are not in order and p4 is missing).
   int elem = fa.elem[0];
   if (elem < 0) { elem = fa.elem[1]; }
   MFEM_ASSERT(elem >= 0, "Face has no elements?");

   const Element &el = elements[elem];
   int f = find_local_face(el.Geom(),
                           find_node(el, fa.p1),
                           find_node(el, fa.p2),
                           find_node(el, fa.p3));

   const int* fv = GI[el.Geom()].faces[f];
   std::array<int, 4> node;
   for (int i = 0; i < 4; i++)
   {
      node[i] = el.node[fv[i]];
   }
   return node;
}

void NCMesh::GetBoundaryClosure(const Array<int> &bdr_attr_is_ess,
                                Array<int> &bdr_vertices, Array<int> &bdr_edges,
                                Array<int> &bdr_faces)
{
   bdr_vertices.SetSize(0);
   bdr_edges.SetSize(0);
   bdr_faces.SetSize(0);

   if (Dim == 3)
   {
      GetFaceList(); // make sure 'boundary_faces' is up to date
      for (int f : boundary_faces)
      {
         if (bdr_attr_is_ess[faces[f].attribute - 1])
         {
            auto node = FindFaceNodes(f);
            int nfv = (node[3] < 0) ? 3 : 4;

            for (int j = 0; j < nfv; j++)
            {
               bdr_vertices.Append(nodes[node[j]].vert_index);

               int enode = nodes.FindId(node[j], node[(j+1) % nfv]);
               MFEM_ASSERT(enode >= 0 && nodes[enode].HasEdge(), "Edge not found.");
               bdr_edges.Append(nodes[enode].edge_index);

               while ((enode = GetEdgeMaster(enode)) >= 0)
               {
                  // append master edges that may not be accessible from any
                  // boundary element, this happens in 3D in re-entrant corners
                  bdr_edges.Append(nodes[enode].edge_index);
               }
            }

            // If the face is a slave face, collect its non-ghost master face
            const Face &face = faces[f];

            const auto id_and_type = GetFaceList().GetMeshIdAndType(face.index);
            if (id_and_type.type == NCList::MeshIdType::SLAVE)
            {
               // A slave face must mark its masters
               const auto &slave_face_id = static_cast<const Slave&>(*id_and_type.id);
               bdr_faces.Append(slave_face_id.master);
            }
         }
      }
   }
   else if (Dim == 2)
   {
      GetFaceList();
      GetEdgeList(); // make sure 'boundary_faces' is up to date

      for (int f : boundary_faces)
      {
         Face &face = faces[f];
         if (bdr_attr_is_ess[face.attribute - 1])
         {
            bdr_vertices.Append(nodes[face.p1].vert_index);
            bdr_vertices.Append(nodes[face.p3].vert_index);
         }

         const auto id_and_type = GetEdgeList().GetMeshIdAndType(face.index);
         if (id_and_type.type == NCList::MeshIdType::SLAVE)
         {
            // A slave face must mark its masters
            const auto &slave_edge_id = static_cast<const Slave&>(*id_and_type.id);
            bdr_edges.Append(slave_edge_id.master);
         }
      }
   }

   // Filter, sort and unique an array, so it contains only local unique values.
   auto FilterSortUnique = [](Array<int> &v, int N)
   {
      // Perform the O(N) filter before the O(NlogN) sort. begin -> it is only
      // entries < N.
      auto it = std::remove_if(v.begin(), v.end(), [N](int i) { return i >= N; });
      std::sort(v.begin(), it);
      v.SetSize(std::distance(v.begin(), std::unique(v.begin(), it)));
   };

   FilterSortUnique(bdr_vertices, NVertices);
   FilterSortUnique(bdr_edges, NEdges);
   FilterSortUnique(bdr_faces, NFaces);
}

int NCMesh::EdgeSplitLevel(int vn1, int vn2) const
{
   int mid = nodes.FindId(vn1, vn2);
   if (mid < 0 || !nodes[mid].HasVertex()) { return 0; }
   return 1 + std::max(EdgeSplitLevel(vn1, mid), EdgeSplitLevel(mid, vn2));
}

int NCMesh::TriFaceSplitLevel(int vn1, int vn2, int vn3) const
{
   int mid[3];
   if (TriFaceSplit(vn1, vn2, vn3, mid) &&
       faces.FindId(vn1, vn2, vn3) < 0)
   {
      return 1 + max(TriFaceSplitLevel(vn1, mid[0], mid[2]),
                     TriFaceSplitLevel(mid[0], vn2, mid[1]),
                     TriFaceSplitLevel(mid[2], mid[1], vn3),
                     TriFaceSplitLevel(mid[0], mid[1], mid[2]));
   }

   return 0; // not split
}

void NCMesh::QuadFaceSplitLevel(int vn1, int vn2, int vn3, int vn4,
                                int& h_level, int& v_level) const
{
   int hl1, hl2, vl1, vl2;
   int mid[5];

   switch (QuadFaceSplitType(vn1, vn2, vn3, vn4, mid))
   {
      case 0: // not split
         h_level = v_level = 0;
         break;

      case 1: // vertical
         QuadFaceSplitLevel(vn1, mid[0], mid[2], vn4, hl1, vl1);
         QuadFaceSplitLevel(mid[0], vn2, vn3, mid[2], hl2, vl2);
         h_level = std::max(hl1, hl2);
         v_level = std::max(vl1, vl2) + 1;
         break;

      default: // horizontal
         QuadFaceSplitLevel(vn1, vn2, mid[1], mid[3], hl1, vl1);
         QuadFaceSplitLevel(mid[3], mid[1], vn3, vn4, hl2, vl2);
         h_level = std::max(hl1, hl2) + 1;
         v_level = std::max(vl1, vl2);
   }
}

int NCMesh::QuadFaceSplitLevel(int vn1, int vn2, int vn3, int vn4) const
{
   int h_level, v_level;
   QuadFaceSplitLevel(vn1, vn2, vn3, vn4, h_level, v_level);
   return h_level + v_level;
}

void NCMesh::CountSplits(int elem, int splits[3]) const
{
   const Element &el = elements[elem];
   const int* node = el.node;
   GeomInfo& gi = GI[el.Geom()];

   int elevel[MaxElemEdges];
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      elevel[i] = EdgeSplitLevel(node[ev[0]], node[ev[1]]);
   }

   int flevel[MaxElemFaces][2];
   if (Dim >= 3)
   {
      for (int i = 0; i < gi.nf; i++)
      {
         const int* fv = gi.faces[i];
         if (gi.nfv[i] == 4)
         {
            QuadFaceSplitLevel(node[fv[0]], node[fv[1]],
                               node[fv[2]], node[fv[3]],
                               flevel[i][1], flevel[i][0]);
         }
         else
         {
            flevel[i][1] = 0;
            flevel[i][0] =
               TriFaceSplitLevel(node[fv[0]], node[fv[1]], node[fv[2]]);
         }
      }
   }

   if (el.Geom() == Geometry::CUBE)
   {
      splits[0] = max(flevel[0][0], flevel[1][0], flevel[3][0], flevel[5][0],
                      elevel[0], elevel[2], elevel[4], elevel[6]);

      splits[1] = max(flevel[0][1], flevel[2][0], flevel[4][0], flevel[5][1],
                      elevel[1], elevel[3], elevel[5], elevel[7]);

      splits[2] = max(flevel[1][1], flevel[2][1], flevel[3][1], flevel[4][1],
                      elevel[8], elevel[9], elevel[10], elevel[11]);
   }
   else if (el.Geom() == Geometry::PRISM)
   {
      splits[0] = splits[1] = max(flevel[0][0], flevel[1][0], 0,
                                  flevel[2][0], flevel[3][0], flevel[4][0],
                                  elevel[0], elevel[1], elevel[2],
                                  elevel[3], elevel[4], elevel[5]);

      splits[2] = max(flevel[2][1], flevel[3][1], flevel[4][1],
                      elevel[6], elevel[7], elevel[8]);
   }
   else if (el.Geom() == Geometry::PYRAMID)
   {
      splits[0] = max(flevel[0][0], flevel[1][0], 0,
                      flevel[2][0], flevel[3][0], flevel[4][0],
                      elevel[0], elevel[1], elevel[2],
                      elevel[3], elevel[4], elevel[5],
                      elevel[6], elevel[7]);

      splits[1] = splits[0];
      splits[2] = splits[0];
   }
   else if (el.Geom() == Geometry::TETRAHEDRON)
   {
      splits[0] = max(flevel[0][0], flevel[1][0], flevel[2][0], flevel[3][0],
                      elevel[0], elevel[1], elevel[2], elevel[3], elevel[4], elevel[5]);

      splits[1] = splits[0];
      splits[2] = splits[0];
   }
   else if (el.Geom() == Geometry::SQUARE)
   {
      splits[0] = max(elevel[0], elevel[2]);
      splits[1] = max(elevel[1], elevel[3]);
   }
   else if (el.Geom() == Geometry::TRIANGLE)
   {
      splits[0] = max(elevel[0], elevel[1], elevel[2]);
      splits[1] = splits[0];
   }
   else
   {
      MFEM_ABORT("Unsupported element geometry.");
   }
}

void NCMesh::GetLimitRefinements(Array<Refinement> &refinements, int max_level)
{
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      if (IsGhost(elements[leaf_elements[i]])) { break; } // TODO: NElements

      int splits[3];
      CountSplits(leaf_elements[i], splits);

      char ref_type = 0;
      for (int k = 0; k < Dim; k++)
      {
         if (splits[k] > max_level)
         {
            ref_type |= (1 << k);
         }
      }

      if (ref_type)
      {
         if (Iso)
         {
            // iso meshes should only be modified by iso refinements
            ref_type = 7;
         }
         refinements.Append(Refinement(i, ref_type));
      }
   }
}

void NCMesh::LimitNCLevel(int max_nc_level)
{
   MFEM_VERIFY(max_nc_level >= 1, "'max_nc_level' must be 1 or greater.");

   while (1)
   {
      Array<Refinement> refinements;
      GetLimitRefinements(refinements, max_nc_level);
      if (!refinements.Size()) { break; }
      Refine(refinements);
   }
}


//// I/O ///////////////////////////////////////////////////////////////////////

int NCMesh::PrintVertexParents(std::ostream *os) const
{
   if (!os)
   {
      // count vertex nodes with parents
      int nv = 0;
      for (auto node = nodes.cbegin(); node != nodes.cend(); ++node)
      {
         if (node->HasVertex() && node->p1 != node->p2) { nv++; }
      }
      return nv;
   }
   else
   {
      // print the relations
      for (auto node = nodes.cbegin(); node != nodes.cend(); ++node)
      {
         if (node->HasVertex() && node->p1 != node->p2)
         {
            MFEM_ASSERT(nodes[node->p1].HasVertex(), "");
            MFEM_ASSERT(nodes[node->p2].HasVertex(), "");

            (*os) << node.index() << " " << node->p1 << " " << node->p2 << "\n";
         }
      }
      return 0;
   }
}

void NCMesh::LoadVertexParents(std::istream &input)
{
   int nv;
   input >> nv;
   while (nv--)
   {
      int id, p1, p2;
      input >> id >> p1 >> p2;
      MFEM_VERIFY(input, "problem reading vertex parents.");

      MFEM_VERIFY(nodes.IdExists(id), "vertex " << id << " not found.");
      MFEM_VERIFY(nodes.IdExists(p1), "parent " << p1 << " not found.");
      MFEM_VERIFY(nodes.IdExists(p2), "parent " << p2 << " not found.");

      int check = nodes.FindId(p1, p2);
      MFEM_VERIFY(check < 0, "parents (" << p1 << ", " << p2 << ") already "
                  "assigned to node " << check);

      // assign new parents for the node
      nodes.Reparent(id, p1, p2);
   }
}

int NCMesh::PrintBoundary(std::ostream *os) const
{
   static const int nfv2geom[5] =
   {
      Geometry::INVALID, Geometry::POINT, Geometry::SEGMENT,
      Geometry::TRIANGLE, Geometry::SQUARE
   };
   int deg = (Dim == 2) ? 2 : 1; // for degenerate faces in 2D

   int count = 0;
   for (int i = 0; i < elements.Size(); i++)
   {
      const Element &el = elements[i];
      if (!el.IsLeaf()) { continue; }

      GeomInfo& gi = GI[el.Geom()];
      for (int k = 0; k < gi.nf; k++)
      {
         const int* fv = gi.faces[k];
         const int nfv = gi.nfv[k];
         const Face* face = faces.Find(el.node[fv[0]], el.node[fv[1]],
                                       el.node[fv[2]], el.node[fv[3]]);
         MFEM_ASSERT(face != NULL, "face not found");
         if (face->Boundary())
         {
            if (!os) { count++; continue; }

            (*os) << face->attribute << " " << nfv2geom[nfv];
            for (int j = 0; j < nfv; j++)
            {
               (*os) << " " << el.node[fv[j*deg]];
            }
            (*os) << "\n";
         }
      }
   }
   return count;
}

void NCMesh::LoadBoundary(std::istream &input)
{
   int nb, attr, geom;
   input >> nb;
   for (int i = 0; i < nb; i++)
   {
      input >> attr >> geom;

      int v1, v2, v3, v4;
      if (geom == Geometry::SQUARE)
      {
         input >> v1 >> v2 >> v3 >> v4;
         Face* face = faces.Get(v1, v2, v3, v4);
         face->attribute = attr;
      }
      else if (geom == Geometry::TRIANGLE)
      {
         input >> v1 >> v2 >> v3;
         Face* face = faces.Get(v1, v2, v3);
         face->attribute = attr;
      }
      else if (geom == Geometry::SEGMENT)
      {
         input >> v1 >> v2;
         Face* face = faces.Get(v1, v1, v2, v2);
         face->attribute = attr;
      }
      else if (geom == Geometry::POINT)
      {
         input >> v1;
         Face* face = faces.Get(v1, v1, v1, v1);
         face->attribute = attr;
      }
      else
      {
         MFEM_ABORT("unsupported boundary element geometry: " << geom);
      }
   }
}

void NCMesh::PrintCoordinates(std::ostream &os) const
{
   int nv = coordinates.Size()/3;
   os << nv << "\n";
   if (!nv) { return; }

   os << spaceDim << "\n";
   for (int i = 0; i < nv; i++)
   {
      os << coordinates[3*i];
      for (int j = 1; j < spaceDim; j++)
      {
         os << " " << coordinates[3*i + j];
      }
      os << "\n";
   }
}

void NCMesh::LoadCoordinates(std::istream &input)
{
   int nv;
   input >> nv;
   if (!nv) { return; }

   input >> spaceDim;

   coordinates.SetSize(nv * 3);
   coordinates = 0.0;

   for (int i = 0; i < nv; i++)
   {
      for (int j = 0; j < spaceDim; j++)
      {
         input >> coordinates[3*i + j];
         MFEM_VERIFY(input.good(), "unexpected EOF");
      }
   }
}

bool NCMesh::ZeroRootStates() const
{
   for (int i = 0; i < root_state.Size(); i++)
   {
      if (root_state[i]) { return false; }
   }
   return true;
}

void NCMesh::Print(std::ostream &os, const std::string &comments) const
{
   os << "MFEM NC mesh v1.0\n\n";

   if (!comments.empty()) { os << comments << "\n\n"; }

   os <<
      "# NCMesh supported geometry types:\n"
      "# SEGMENT     = 1\n"
      "# TRIANGLE    = 2\n"
      "# SQUARE      = 3\n"
      "# TETRAHEDRON = 4\n"
      "# CUBE        = 5\n"
      "# PRISM       = 6\n"
      "# PYRAMID     = 7\n";

   os << "\ndimension\n" << Dim << "\n";

#ifndef MFEM_USE_MPI
   if (MyRank != 0) // don't print this section in serial: default rank is 0
#endif
   {
      os << "\nrank\n" << MyRank << "\n";
   }

   os << "\n# rank attr geom ref_type nodes/children";
   os << "\nelements\n" << elements.Size() << "\n";

   for (int i = 0; i < elements.Size(); i++)
   {
      const Element &el = elements[i];
      os << el.rank << " " << el.attribute << " ";
      if (el.parent == -2) { os << "-1\n"; continue; } // unused element

      os << int(el.geom) << " " << int(el.ref_type);
      for (int j = 0; j < MaxElemNodes && el.node[j] >= 0; j++)
      {
         os << " " << el.node[j];
      }
      os << "\n";
   }

   int nb = PrintBoundary(NULL);
   if (nb)
   {
      os << "\n# attr geom nodes";
      os << "\nboundary\n" << nb << "\n";

      PrintBoundary(&os);
   }

   int nvp = PrintVertexParents(NULL);
   if (nvp)
   {
      os << "\n# vert_id p1 p2";
      os << "\nvertex_parents\n" << nvp << "\n";

      PrintVertexParents(&os);
   }

   if (!ZeroRootStates()) // root_state section is optional
   {
      os << "\n# root element orientation";
      os << "\nroot_state\n" << root_state.Size() << "\n";

      for (int i = 0; i < root_state.Size(); i++)
      {
         os << root_state[i] << "\n";
      }
   }

   if (coordinates.Size())
   {
      os << "\n# top-level node coordinates";
      os << "\ncoordinates\n";

      PrintCoordinates(os);
   }
   else
   {
      // 'nodes' will be printed one level up by Mesh::Printer()
   }
}

void NCMesh::InitRootElements()
{
   // initialize Element::parent
   for (int i = 0; i < elements.Size(); i++)
   {
      Element &el = elements[i];
      if (el.ref_type)
      {
         for (int j = 0; j < MaxElemChildren && el.child[j] >= 0; j++)
         {
            int child = el.child[j];
            MFEM_VERIFY(child < elements.Size(), "invalid mesh file: "
                        "element " << i << " references invalid child " << child);
            elements[child].parent = i;
         }
      }
   }

   // count the root elements
   int nroots = 0;
   for (const auto &e : elements)
      if (e.parent == -1)
      {
         ++nroots;
      }
   MFEM_VERIFY(nroots > 0 ||
               elements.Size() == 0,
               "invalid mesh file: no root elements in non-empty mesh found.");


   // check that only the first 'nroot' elements are roots (have no parent)
   for (int i = nroots; i < elements.Size(); i++)
   {
      MFEM_VERIFY(elements[i].parent != -1,
                  "invalid mesh file: only the first M elements can be roots. "
                  "Found element " << i << " with no parent.");
   }

   // default root state is zero
   root_state.SetSize(nroots);
   root_state = 0;
}

int NCMesh::CountTopLevelNodes() const
{
   int ntop = 0;
   for (auto node = nodes.cbegin(); node != nodes.cend(); ++node)
   {
      if (node->p1 == node->p2) { ntop = node.index() + 1; }
   }
   return ntop;
}

NCMesh::NCMesh(std::istream &input, int version, int &curved, int &is_nc)
   : spaceDim(0), MyRank(0), Iso(true), Legacy(false)
{
   is_nc = 1;
   if (version == 1) // old MFEM mesh v1.1 format
   {
      LoadLegacyFormat(input, curved, is_nc);
      Legacy = true;
      return;
   }

   MFEM_ASSERT(version == 10, "");
   std::string ident;
   int count;

   // Skip the version string
   skip_comment_lines(input, 'M');

   // load dimension
   skip_comment_lines(input, '#');
   input >> ident;
   MFEM_VERIFY(ident == "dimension", "Invalid mesh file: " << ident);
   input >> Dim;

   // load rank, if present
   skip_comment_lines(input, '#');
   input >> ident;
   if (ident == "rank")
   {
      input >> MyRank;
      MFEM_VERIFY(MyRank >= 0, "Invalid rank");

      skip_comment_lines(input, '#');
      input >> ident;
   }

   // load file SFC version, if present (for future changes to SFC ordering)
   if (ident == "sfc_version")
   {
      int sfc_version; // TODO future: store as class member
      input >> sfc_version;
      MFEM_VERIFY(sfc_version == 0,
                  "Unsupported mesh file SFC version (" << sfc_version << "). "
                  "Please update MFEM.");

      skip_comment_lines(input, '#');
      input >> ident;
   }

   // load elements
   MFEM_VERIFY(ident == "elements", "Invalid mesh file: " << ident);
   input >> count;
   for (int i = 0; i < count; i++)
   {
      int rank, attr, geom, ref_type;
      input >> rank >> attr >> geom;

      Geometry::Type type = Geometry::Type(geom);
      elements.Append(Element(type, attr));

      MFEM_ASSERT(elements.Size() == i+1, "");
      Element &el = elements[i];
      el.rank = rank;

      if (geom >= 0)
      {
         CheckSupportedGeom(type);
         GI[geom].InitGeom(type);

         input >> ref_type;
         MFEM_VERIFY(ref_type >= 0 && ref_type < 8, "");
         el.ref_type = ref_type;

         if (ref_type) // refined element
         {
            for (int j = 0; j < ref_type_num_children[ref_type]; j++)
            {
               input >> el.child[j];
            }
            if (Dim == 3 && ref_type != 7) { Iso = false; }
         }
         else // leaf element
         {
            for (int j = 0; j < GI[geom].nv; j++)
            {
               int id;
               input >> id;
               el.node[j] = id;
               nodes.Alloc(id, id, id);
               // NOTE: nodes that won't get parents assigned will stay hashed
               // with p1 == p2 == id (top-level nodes)
            }
         }
      }
      else
      {
         el.parent = -2; // mark as unused
         free_element_ids.Append(i);
      }
   }

   InitRootElements();
   InitGeomFlags();

   skip_comment_lines(input, '#');
   input >> ident;

   // load boundary
   if (ident == "boundary")
   {
      LoadBoundary(input);

      skip_comment_lines(input, '#');
      input >> ident;
   }

   // load vertex hierarchy
   if (ident == "vertex_parents")
   {
      LoadVertexParents(input);

      skip_comment_lines(input, '#');
      input >> ident;
   }

   // load root states
   if (ident == "root_state")
   {
      input >> count;
      MFEM_VERIFY(count <= root_state.Size(), "Too many root states");
      for (int i = 0; i < count; i++)
      {
         input >> root_state[i];
      }

      skip_comment_lines(input, '#');
      input >> ident;
   }

   // load coordinates or nodes
   if (ident == "coordinates")
   {
      LoadCoordinates(input);

      MFEM_VERIFY(coordinates.Size() >= 3*CountTopLevelNodes(),
                  "Invalid mesh file: not all top-level nodes are covered by "
                  "the 'coordinates' section of the mesh file: " << coordinates.Size() << ' ' <<
                  3*CountTopLevelNodes());
      curved = 0;
   }
   else if (ident == "nodes")
   {
      coordinates.SetSize(0); // this means the mesh is curved

      // prepare to read the nodes
      input >> std::ws;
      curved = 1;
   }
   else
   {
      MFEM_ABORT("Invalid mesh file: either 'coordinates' or "
                 "'nodes' must be present");
   }

   // create edge nodes and faces
   nodes.UpdateUnused();
   for (int i = 0; i < elements.Size(); i++)
   {
      if (elements[i].IsLeaf())
      {
         ReferenceElement(i);
         RegisterFaces(i);
      }
   }

   Update();
}

void NCMesh::CopyElements(int elem,
                          const BlockArray<Element> &tmp_elements)
{
   Element &el = elements[elem];
   if (el.ref_type)
   {
      for (int i = 0; i < MaxElemChildren && el.child[i] >= 0; i++)
      {
         int old_id = el.child[i];
         // here we know 'free_element_ids' is empty
         int new_id = elements.Append(tmp_elements[old_id]);
         el.child[i] = new_id;
         elements[new_id].parent = elem;
         CopyElements(new_id, tmp_elements);
      }
   }
}

void NCMesh::LoadCoarseElements(std::istream &input)
{
   int ne;
   input >> ne;

   bool iso = true;

   // load the coarse elements
   while (ne--)
   {
      int ref_type;
      input >> ref_type;

      int elem = AddElement(Geometry::INVALID, 0);
      Element &el = elements[elem];
      el.ref_type = ref_type;

      if (Dim == 3 && ref_type != 7) { iso = false; }

      // load child IDs and make parent-child links
      int nch = ref_type_num_children[ref_type];
      for (int i = 0, id; i < nch; i++)
      {
         input >> id;
         MFEM_VERIFY(id >= 0, "");
         MFEM_VERIFY(id < elements.Size(),
                     "coarse element cannot be referenced before it is "
                     "defined (id=" << id << ").");

         Element &child = elements[id];
         MFEM_VERIFY(child.parent == -1,
                     "element " << id << " cannot have two parents.");

         el.child[i] = id;
         child.parent = elem;

         if (!i) // copy geom and attribute from first child
         {
            el.geom = child.geom;
            el.attribute = child.attribute;
         }
      }
   }

   // prepare for reordering the elements
   BlockArray<Element> tmp_elements;
   elements.Swap(tmp_elements);

   // copy roots, they need to be at the beginning of 'elements'
   int root_count = 0;
   for (auto el = tmp_elements.begin(); el != tmp_elements.end(); ++el)
   {
      if (el->parent == -1)
      {
         elements.Append(*el); // same as AddElement()
         root_count++;
      }
   }

   // copy the rest of the hierarchy
   for (int i = 0; i < root_count; i++)
   {
      CopyElements(i, tmp_elements);
   }

   // set the Iso flag (must be false if there are 3D aniso refinements)
   Iso = iso;

   InitRootState(root_count);
}

void NCMesh::LoadLegacyFormat(std::istream &input, int &curved, int &is_nc)
{
   MFEM_ASSERT(elements.Size() == 0, "");
   MFEM_ASSERT(nodes.Size() == 0, "");
   MFEM_ASSERT(free_element_ids.Size() == 0, "");

   std::string ident;
   int count, attr, geom;

   // load dimension
   skip_comment_lines(input, '#');
   input >> ident;
   MFEM_VERIFY(ident == "dimension", "invalid mesh file");
   input >> Dim;

   // load elements
   skip_comment_lines(input, '#');
   input >> ident;
   MFEM_VERIFY(ident == "elements", "invalid mesh file");

   input >> count;
   for (int i = 0; i < count; i++)
   {
      input >> attr >> geom;

      Geometry::Type type = Geometry::Type(geom);
      CheckSupportedGeom(type);
      GI[geom].InitGeom(type);

      int eid = AddElement(type, attr);
      MFEM_ASSERT(eid == i, "");

      Element &el = elements[eid];
      for (int j = 0; j < GI[geom].nv; j++)
      {
         int id;
         input >> id;
         el.node[j] = id;
         nodes.Alloc(id, id, id); // see comment in NCMesh::NCMesh
      }
      el.index = i; // needed for file leaf order below
   }

   // load boundary
   skip_comment_lines(input, '#');
   input >> ident;
   MFEM_VERIFY(ident == "boundary", "invalid mesh file");

   LoadBoundary(input);

   // load vertex hierarchy
   skip_comment_lines(input, '#');
   input >> ident;
   if (ident == "vertex_parents")
   {
      LoadVertexParents(input);
      is_nc = 1;

      skip_comment_lines(input, '#');
      input >> ident;
   }
   else
   {
      // no "vertex_parents" section: this file needs to be treated as a
      // conforming mesh for complete backward compatibility with MFEM 4.2
      is_nc = 0;
   }

   // load element hierarchy
   if (ident == "coarse_elements")
   {
      LoadCoarseElements(input);

      skip_comment_lines(input, '#');
      input >> ident;
   }
   else
   {
      // no element hierarchy -> all elements are roots
      InitRootState(elements.Size());
   }
   InitGeomFlags();

   // load vertices
   MFEM_VERIFY(ident == "vertices", "invalid mesh file");
   int nvert;
   input >> nvert;
   input >> std::ws >> ident;
   if (ident != "nodes")
   {
      spaceDim = atoi(ident.c_str());

      coordinates.SetSize(3*nvert);
      coordinates = 0.0;

      for (int i = 0; i < nvert; i++)
      {
         for (int j = 0; j < spaceDim; j++)
         {
            input >> coordinates[3*i + j];
            MFEM_VERIFY(input.good(), "unexpected EOF");
         }
      }

      // truncate extra coordinates (legacy vertices section is longer)
      int ntop = CountTopLevelNodes();
      if (3*ntop < coordinates.Size())
      {
         coordinates.SetSize(3*ntop);
      }
   }
   else
   {
      coordinates.SetSize(0);

      // prepare to read the nodes
      input >> std::ws;
      curved = 1;
   }

   // create edge nodes and faces
   nodes.UpdateUnused();
   int leaf_count = 0;
   for (int i = 0; i < elements.Size(); i++)
   {
      if (elements[i].IsLeaf())
      {
         ReferenceElement(i);
         RegisterFaces(i);
         leaf_count++;
      }
   }

   // v1.1 honors file leaf order on load, prepare legacy 'leaf_elements'
   Array<int> file_leaf_elements(leaf_count);
   file_leaf_elements = -1;
   for (int i = 0; i < elements.Size(); i++)
   {
      if (elements[i].IsLeaf())
      {
         file_leaf_elements[elements[i].index] = i;
      }
   }
   MFEM_ASSERT(file_leaf_elements.Min() >= 0, "");

   Update();

   // force file leaf order
   Swap(leaf_elements, file_leaf_elements);

   // make sure Mesh::NVertices is equal to "nvert" from the file (in case of
   // unused vertices), see also GetMeshComponents
   if (nvert > vertex_nodeId.Size())
   {
      vertex_nodeId.SetSize(nvert, -1);
   }
}

void NCMesh::LegacyToNewVertexOrdering(Array<int> &order) const
{
   order.SetSize(NVertices);
   order = -1;

   int count = 0;
   for (auto node = nodes.cbegin(); node != nodes.cend(); ++node)
   {
      if (node->HasVertex())
      {
         MFEM_ASSERT(node.index() >= 0, "");
         MFEM_ASSERT(node.index() < order.Size(), "");
         MFEM_ASSERT(order[node.index()] == -1, "");

         order[node.index()] = node->vert_index;
         count++;
      }
   }
   MFEM_ASSERT(count == order.Size(), "");
   MFEM_CONTRACT_VAR(count);
}


////////////////////////////////////////////////////////////////////////////////

void NCMesh::Trim()
{
   vertex_list.Clear();
   face_list.Clear();
   edge_list.Clear();

   boundary_faces.DeleteAll();
   element_vertex.Clear();

   ClearTransforms();

   // TODO future: consider trimming unused blocks at the end of 'elements' and
   // maybe also of 'nodes' and 'faces'.
}

long NCMesh::NCList::MemoryUsage() const
{
   int pm_size = 0;
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      for (int j = 0; j < point_matrices[i].Size(); i++)
      {
         pm_size += point_matrices[i][j]->MemoryUsage();
      }
      pm_size += point_matrices[i].MemoryUsage();
   }

   return conforming.MemoryUsage() +
          masters.MemoryUsage() +
          slaves.MemoryUsage() +
          pm_size;
}

long CoarseFineTransformations::MemoryUsage() const
{
   long mem = embeddings.MemoryUsage();
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      mem += point_matrices[i].MemoryUsage();
   }
   return mem;
}

long NCMesh::MemoryUsage() const
{
   return nodes.MemoryUsage() +
          faces.MemoryUsage() +
          elements.MemoryUsage() +
          free_element_ids.MemoryUsage() +
          root_state.MemoryUsage() +
          coordinates.MemoryUsage() +
          leaf_elements.MemoryUsage() +
          leaf_sfc_index.MemoryUsage() +
          vertex_nodeId.MemoryUsage() +
          face_list.MemoryUsage() +
          edge_list.MemoryUsage() +
          vertex_list.MemoryUsage() +
          boundary_faces.MemoryUsage() +
          element_vertex.MemoryUsage() +
          ref_stack.MemoryUsage() +
          derefinements.MemoryUsage() +
          transforms.MemoryUsage() +
          coarse_elements.MemoryUsage() +
          sizeof(*this);
}

int NCMesh::PrintMemoryDetail() const
{
   nodes.PrintMemoryDetail(); mfem::out << " nodes\n";
   faces.PrintMemoryDetail(); mfem::out << " faces\n";

   mfem::out << elements.MemoryUsage() << " elements\n"
             << free_element_ids.MemoryUsage() << " free_element_ids\n"
             << root_state.MemoryUsage() << " root_state\n"
             << coordinates.MemoryUsage() << " top_vertex_pos\n"
             << leaf_elements.MemoryUsage() << " leaf_elements\n"
             << leaf_sfc_index.MemoryUsage() << " leaf_sfc_index\n"
             << vertex_nodeId.MemoryUsage() << " vertex_nodeId\n"
             << face_list.MemoryUsage() << " face_list\n"
             << edge_list.MemoryUsage() << " edge_list\n"
             << vertex_list.MemoryUsage() << " vertex_list\n"
             << boundary_faces.MemoryUsage() << " boundary_faces\n"
             << element_vertex.MemoryUsage() << " element_vertex\n"
             << ref_stack.MemoryUsage() << " ref_stack\n"
             << derefinements.MemoryUsage() << " derefinements\n"
             << transforms.MemoryUsage() << " transforms\n"
             << coarse_elements.MemoryUsage() << " coarse_elements\n"
             << sizeof(*this) << " NCMesh"
             << std::endl;

   return elements.Size() - free_element_ids.Size();
}

#ifdef MFEM_DEBUG
void NCMesh::DebugLeafOrder(std::ostream &os) const
{
   tmp_vertex = new TmpVertex[nodes.NumIds()];
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      const Element* elem = &elements[leaf_elements[i]];
      for (int j = 0; j < Dim; j++)
      {
         real_t sum = 0.0;
         int count = 0;
         for (int k = 0; k < MaxElemNodes; k++)
         {
            if (elem->node[k] >= 0)
            {
               sum += CalcVertexPos(elem->node[k])[j];
               count++;
            }
         }
         os << sum / count << " ";
      }
      os << "\n";
   }
   delete [] tmp_vertex;
}

void NCMesh::DebugDump(std::ostream &os) const
{
   // dump nodes
   tmp_vertex = new TmpVertex[nodes.NumIds()];
   os << nodes.Size() << "\n";
   for (auto node = nodes.cbegin(); node != nodes.cend(); ++node)
   {
      const real_t *pos = CalcVertexPos(node.index());
      os << node.index() << " "
         << pos[0] << " " << pos[1] << " " << pos[2] << " "
         << node->p1 << " " << node->p2 << " "
         << node->vert_index << " " << node->edge_index << " "
         << 0 << "\n";
   }
   delete [] tmp_vertex;
   os << "\n";

   // dump elements
   int nleaves = 0;
   for (int i = 0; i < elements.Size(); i++)
   {
      if (elements[i].IsLeaf()) { nleaves++; }
   }
   os << nleaves << "\n";
   for (int i = 0; i < elements.Size(); i++)
   {
      const Element &el = elements[i];
      if (el.IsLeaf())
      {
         const GeomInfo& gi = GI[el.Geom()];
         os << gi.nv << " ";
         for (int j = 0; j < gi.nv; j++)
         {
            os << el.node[j] << " ";
         }
         os << el.attribute << " " << el.rank << " " << i << "\n";
      }
   }
   os << "\n";

   // dump faces
   os << faces.Size() << "\n";
   for (const auto &face : faces)
   {
      int elem = face.elem[0];
      if (elem < 0) { elem = face.elem[1]; }
      MFEM_ASSERT(elem >= 0, "");
      const Element &el = elements[elem];

      int lf = find_local_face(el.Geom(),
                               find_node(el, face.p1),
                               find_node(el, face.p2),
                               find_node(el, face.p3));

      const int* fv = GI[el.Geom()].faces[lf];
      const int nfv = GI[el.Geom()].nfv[lf];

      os << nfv;
      for (int i = 0; i < nfv; i++)
      {
         os << " " << el.node[fv[i]];
      }
      //os << " # face " << face.index() << ", index " << face.index << "\n";
      os << "\n";
   }
}
#endif

} // namespace mfem
