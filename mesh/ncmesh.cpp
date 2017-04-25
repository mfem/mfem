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

#include "mesh_headers.hpp"
#include "../fem/fem.hpp"

#include <string>
#include <algorithm>
#include <cmath>
#include <climits> // INT_MAX

namespace mfem
{

NCMesh::GeomInfo NCMesh::GI[Geometry::NumGeom];

static NCMesh::GeomInfo& gi_hex  = NCMesh::GI[Geometry::CUBE];
static NCMesh::GeomInfo& gi_quad = NCMesh::GI[Geometry::SQUARE];
static NCMesh::GeomInfo& gi_tri  = NCMesh::GI[Geometry::TRIANGLE];

void NCMesh::GeomInfo::Initialize(const mfem::Element* elem)
{
   if (initialized) { return; }

   nv = elem->GetNVertices();
   ne = elem->GetNEdges();
   nf = elem->GetNFaces(nfv);

   for (int i = 0; i < ne; i++)
   {
      for (int j = 0; j < 2; j++)
      {
         edges[i][j] = elem->GetEdgeVertices(i)[j];
      }
   }
   for (int i = 0; i < nf; i++)
   {
      for (int j = 0; j < nfv; j++)
      {
         faces[i][j] = elem->GetFaceVertices(i)[j];
      }
   }

   // in 2D we pretend to have faces too, so we can use Face::elem[2]
   if (!nf)
   {
      for (int i = 0; i < ne; i++)
      {
         // make a degenerate face
         faces[i][0] = faces[i][1] = edges[i][0];
         faces[i][2] = faces[i][3] = edges[i][1];
      }
      nf = ne;
   }

   initialized = true;
}


NCMesh::NCMesh(const Mesh *mesh, std::istream *vertex_parents)
{
   Dim = mesh->Dimension();
   spaceDim = mesh->SpaceDimension();

   // assume the mesh is anisotropic if we're loading a file
   Iso = vertex_parents ? false : true;

   // examine elements and reserve the first node IDs for vertices
   // (note: 'mesh' may not have vertices defined yet, e.g., on load)
   int max_id = -1;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const mfem::Element *elem = mesh->GetElement(i);
      const int *v = elem->GetVertices(), nv = elem->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         max_id = std::max(max_id, v[j]);
      }
   }
   for (int id = 0; id <= max_id; id++)
   {
      // top-level nodes are special: id == p1 == p2 == orig. vertex id
      int node = nodes.GetId(id, id);
      MFEM_CONTRACT_VAR(node);
      MFEM_ASSERT(node == id, "");
   }

   // if a mesh file is being read, load the vertex hierarchy now;
   // 'vertex_parents' must be at the appropriate section in the mesh file
   if (vertex_parents)
   {
      LoadVertexParents(*vertex_parents);
   }
   else
   {
      top_vertex_pos.SetSize(3*mesh->GetNV());
      for (int i = 0; i < mesh->GetNV(); i++)
      {
         memcpy(&top_vertex_pos[3*i], mesh->GetVertex(i), 3*sizeof(double));
      }
   }

   // create the NCMesh::Element struct for each Mesh element
   root_count = mesh->GetNE();
   for (int i = 0; i < root_count; i++)
   {
      const mfem::Element *elem = mesh->GetElement(i);

      int geom = elem->GetGeometryType();
      if (geom != Geometry::TRIANGLE &&
          geom != Geometry::SQUARE &&
          geom != Geometry::CUBE)
      {
         MFEM_ABORT("only triangles, quads and hexes are supported by NCMesh.");
      }

      // initialize edge/face tables for this type of element
      GI[geom].Initialize(elem);

      // create our Element struct for this element
      int root_id = AddElement(Element(geom, elem->GetAttribute()));
      MFEM_ASSERT(root_id == i, "");
      Element &root_elem = elements[root_id];

      const int *v = elem->GetVertices();
      for (int j = 0; j < GI[geom].nv; j++)
      {
         root_elem.node[j] = v[j];
      }

      // increase reference count of all nodes the element is using
      // (NOTE: this will also create and reference all edge nodes and faces)
      RefElement(root_id);

      // make links from faces back to the element
      RegisterFaces(root_id);
   }

   // store boundary element attributes
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const mfem::Element *be = mesh->GetBdrElement(i);
      const int *v = be->GetVertices();

      if (be->GetType() == mfem::Element::QUADRILATERAL)
      {
         Face* face = faces.Find(v[0], v[1], v[2], v[3]);
         MFEM_VERIFY(face, "boundary face not found.");
         face->attribute = be->GetAttribute();
      }
      else if (be->GetType() == mfem::Element::SEGMENT)
      {
         Face* face = faces.Find(v[0], v[0], v[1], v[1]);
         MFEM_VERIFY(face, "boundary face not found.");
         face->attribute = be->GetAttribute();
      }
      else
      {
         MFEM_ABORT("only segment and quadrilateral boundary "
                    "elements are supported by NCMesh.");
      }
   }

   Update();
}

NCMesh::NCMesh(const NCMesh &other)
   : Dim(other.Dim)
   , spaceDim(other.spaceDim)
   , Iso(other.Iso)
   , nodes(other.nodes)
   , faces(other.faces)
   , elements(other.elements)
   , root_count(other.root_count)
{
   other.free_element_ids.Copy(free_element_ids);
   other.top_vertex_pos.Copy(top_vertex_pos);
   Update();
}

void NCMesh::Update()
{
   UpdateLeafElements();
   UpdateVertices();

   face_list.Clear();
   edge_list.Clear();

   element_vertex.Clear();
}

NCMesh::~NCMesh()
{
#ifdef MFEM_DEBUG
#ifdef MFEM_USE_MPI
   // in parallel, update 'leaf_elements'
   for (int i = 0; i < elements.Size(); i++)
   {
      elements[i].rank = 0; // make sure all leaves are in leaf_elements
   }
   UpdateLeafElements();
#endif

   // sign off of all faces and nodes
   Array<int> elemFaces;
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      elemFaces.SetSize(0);
      UnrefElement(leaf_elements[i], elemFaces);
      DeleteUnusedFaces(elemFaces);
   }
   // NOTE: in release mode, we just throw away all faces and nodes at once
#endif
}

NCMesh::Node::~Node()
{
   MFEM_ASSERT(!vert_refc && !edge_refc, "node was not unreffed properly, "
               "vert_refc: " << (int) vert_refc << ", edge_refc: "
               << (int) edge_refc);
}

void NCMesh::RefElement(int elem)
{
   Element &el = elements[elem];
   int* node = el.node;
   GeomInfo& gi = GI[(int) el.geom];

   // ref all vertices
   for (int i = 0; i < gi.nv; i++)
   {
      nodes[node[i]].vert_refc++;
   }

   // ref all edges (possibly creating their nodes)
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

      // NOTE: face->RegisterElement called separately to avoid having
      // to store 3 element indices  temporarily in the face when refining.
      // See also NCMesh::RegisterFaces.
   }
}

void NCMesh::UnrefElement(int elem, Array<int> &elemFaces)
{
   Element &el = elements[elem];
   int* node = el.node;
   GeomInfo& gi = GI[(int) el.geom];

   // unref all faces
   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      int face = faces.FindId(node[fv[0]], node[fv[1]],
                              node[fv[2]], node[fv[3]]);
      MFEM_ASSERT(face >= 0, "face not found.");
      faces[face].ForgetElement(elem);

      // NOTE: faces.Delete() called later to avoid destroying and
      // recreating faces during refinement, see NCMesh::DeleteUnusedFaces.
      elemFaces.Append(face);
   }

   // unref all edges (possibly destroying them)
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      int enode = FindAltParents(node[ev[0]], node[ev[1]]);
      MFEM_ASSERT(enode >= 0, "edge not found.");
      MFEM_ASSERT(nodes.IdExists(enode), "edge does not exist.");
      if (!nodes[enode].UnrefEdge())
      {
         nodes.Delete(enode);
      }
   }

   // unref all vertices (possibly destroying them)
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
   GeomInfo &gi = GI[(int) el.geom];

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

int NCMesh::FindAltParents(int node1, int node2)
{
   int mid = nodes.FindId(node1, node2);
   if (mid < 0 && Dim >= 3 && !Iso)
   {
      // In rare cases, a mid-face node exists under alternate parents a1, a2
      // (see picture) instead of the requested parents n1, n2. This is an
      // inconsistent situation that may exist temporarily as a result of
      // "nodes.Reparent" while doing anisotropic splits, before forced
      // refinements are all processed. This function attempts to retrieve such
      // a node. An extra twist is that w1 and w2 may themselves need to be
      // obtained using this very function.
      //
      //                 n1.p1      n1       n1.p2
      //                      *------*------*
      //                      |      |      |
      //                      |      |mid   |
      //                   a1 *------*------* a2
      //                      |      |      |
      //                      |      |      |
      //                      *------*------*
      //                 n2.p1      n2       n2.p2
      //
      // NOTE: this function would not be needed if the elements remembered
      // their edge nodes. We have however opted to save memory at the cost of
      // this computation, which is only necessary when forced refinements are
      // being done.

      Node &n1 = nodes[node1], &n2 = nodes[node2];

      int n1p1 = n1.p1, n1p2 = n1.p2;
      int n2p1 = n2.p1, n2p2 = n2.p2;

      if ((n1p1 != n1p2) && (n2p1 != n2p2)) // non-top-level nodes?
      {
         int a1 = FindAltParents(n1p1, n2p1);
         int a2 = (a1 >= 0) ? FindAltParents(n1p2, n2p2) : -1 /*optimization*/;

         if (a1 < 0 || a2 < 0)
         {
            // one more try may be needed as p1, p2 are unordered
            a1 = FindAltParents(n1p1, n2p2);
            a2 = (a1 >= 0) ? FindAltParents(n1p2, n2p1) : -1 /*optimization*/;
         }

         if (a1 >= 0 && a2 >= 0) // got both alternate parents?
         {
            mid = nodes.FindId(a1, a2);
         }
      }
   }
   return mid;
}


//// Refinement & Derefinement /////////////////////////////////////////////////

NCMesh::Element::Element(int geom, int attr)
   : geom(geom), ref_type(0), flag(0), index(-1), rank(0), attribute(attr)
   , parent(-1)
{
   for (int i = 0; i < 8; i++) { node[i] = -1; }

   // NOTE: in 2D the 8-element node/child arrays are not optimal, however,
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
   // create new unrefined element, initialize nodes
   int new_id = AddElement(Element(Geometry::CUBE, attr));
   Element &el = elements[new_id];

   el.node[0] = n0, el.node[1] = n1, el.node[2] = n2, el.node[3] = n3;
   el.node[4] = n4, el.node[5] = n5, el.node[6] = n6, el.node[7] = n7;

   // get faces and assign face attributes
   Face* f[6];
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

int NCMesh::NewQuadrilateral(int n0, int n1, int n2, int n3,
                             int attr,
                             int eattr0, int eattr1, int eattr2, int eattr3)
{
   // create new unrefined element, initialize nodes
   int new_id = AddElement(Element(Geometry::SQUARE, attr));
   Element &el = elements[new_id];

   el.node[0] = n0, el.node[1] = n1, el.node[2] = n2, el.node[3] = n3;

   // get (degenerate) faces and assign face attributes
   Face* f[4];
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
   // create new unrefined element, initialize nodes
   int new_id = AddElement(Element(Geometry::TRIANGLE, attr));
   Element &el = elements[new_id];
   el.node[0] = n0, el.node[1] = n1, el.node[2] = n2;

   // get (degenerate) faces and assign face attributes
   Face* f[3];
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

int NCMesh::GetMidEdgeNode(int vn1, int vn2)
{
   // in 3D we must be careful about getting the mid-edge node
   int mid = FindAltParents(vn1, vn2);
   if (mid < 0) { mid = nodes.GetId(vn1, vn2); } // create if not found
   return mid;
}

int NCMesh::GetMidFaceNode(int en1, int en2, int en3, int en4)
{
   // mid-face node can be created either from (en1, en3) or from (en2, en4)
   int midf = nodes.FindId(en1, en3);
   if (midf >= 0) { return midf; }
   return nodes.GetId(en2, en4);
}

//
inline bool NCMesh::NodeSetX1(int node, int* n)
{ return node == n[0] || node == n[3] || node == n[4] || node == n[7]; }

inline bool NCMesh::NodeSetX2(int node, int* n)
{ return node == n[1] || node == n[2] || node == n[5] || node == n[6]; }

inline bool NCMesh::NodeSetY1(int node, int* n)
{ return node == n[0] || node == n[1] || node == n[4] || node == n[5]; }

inline bool NCMesh::NodeSetY2(int node, int* n)
{ return node == n[2] || node == n[3] || node == n[6] || node == n[7]; }

inline bool NCMesh::NodeSetZ1(int node, int* n)
{ return node == n[0] || node == n[1] || node == n[2] || node == n[3]; }

inline bool NCMesh::NodeSetZ2(int node, int* n)
{ return node == n[4] || node == n[5] || node == n[6] || node == n[7]; }


void NCMesh::ForceRefinement(int vn1, int vn2, int vn3, int vn4)
{
   // get the element this face belongs to
   Face* face = faces.Find(vn1, vn2, vn3, vn4);
   if (!face) { return; }

   int elem = face->GetSingleElement();
   MFEM_ASSERT(!elements[elem].ref_type, "element already refined.");

   int* nodes = elements[elem].node;

   // schedule the right split depending on face orientation
   if ((NodeSetX1(vn1, nodes) && NodeSetX2(vn2, nodes)) ||
       (NodeSetX1(vn2, nodes) && NodeSetX2(vn1, nodes)))
   {
      ref_stack.Append(Refinement(elem, 1)); // X split
   }
   else if ((NodeSetY1(vn1, nodes) && NodeSetY2(vn2, nodes)) ||
            (NodeSetY1(vn2, nodes) && NodeSetY2(vn1, nodes)))
   {
      ref_stack.Append(Refinement(elem, 2)); // Y split
   }
   else if ((NodeSetZ1(vn1, nodes) && NodeSetZ2(vn2, nodes)) ||
            (NodeSetZ1(vn2, nodes) && NodeSetZ2(vn1, nodes)))
   {
      ref_stack.Append(Refinement(elem, 4)); // Z split
   }
   else
   {
      MFEM_ABORT("inconsistent element/face structure.");
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

   int mid23 = nodes.FindId(vn2, vn3);
   int mid41 = nodes.FindId(vn4, vn1);
   if (mid23 >= 0 && mid41 >= 0)
   {
      int midf = nodes.FindId(mid23, mid41);
      if (midf >= 0)
      {
         nodes.Reparent(midf, mid12, mid34);

         CheckAnisoFace(vn1, vn2, mid23, mid41, mid12, midf, level+1);
         CheckAnisoFace(mid41, mid23, vn3, vn4, midf, mid34, level+1);
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
         isotropically split faces as well, see second comment in
         CheckAnisoFace above. */

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
      for (int i = 0; i < 8; i++)
      {
         if (el.child[i] >= 0) { RefineElement(el.child[i], remaining); }
      }
      return;
   }

   int* no = el.node;
   int attr = el.attribute;

   int child[8];
   for (int i = 0; i < 8; i++) { child[i] = -1; }

   // get parent's face attributes
   int fa[6];
   GeomInfo& gi = GI[(int) el.geom];
   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      Face* face = faces.Find(no[fv[0]], no[fv[1]], no[fv[2]], no[fv[3]]);
      fa[i] = face->attribute;
   }

   // create child elements
   if (el.geom == Geometry::CUBE)
   {
      // Vertex numbering is assumed to be as follows:
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

      if (ref_type == 1) // split along X axis
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
      else if (ref_type == 2) // split along Y axis
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
      else if (ref_type == 4) // split along Z axis
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
      else if (ref_type == 3) // XY split
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
      else if (ref_type == 5) // XZ split
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
      else if (ref_type == 6) // YZ split
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
      else if (ref_type == 7) // full isotropic refinement
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

      if (ref_type != 7) { Iso = false; }
   }
   else if (el.geom == Geometry::SQUARE)
   {
      ref_type &= ~4; // ignore Z bit

      if (ref_type == 1) // X split
      {
         int mid01 = nodes.GetId(no[0], no[1]);
         int mid23 = nodes.GetId(no[2], no[3]);

         child[0] = NewQuadrilateral(no[0], mid01, mid23, no[3],
                                     attr, fa[0], -1, fa[2], fa[3]);

         child[1] = NewQuadrilateral(mid01, no[1], no[2], mid23,
                                     attr, fa[0], fa[1], fa[2], -1);
      }
      else if (ref_type == 2) // Y split
      {
         int mid12 = nodes.GetId(no[1], no[2]);
         int mid30 = nodes.GetId(no[3], no[0]);

         child[0] = NewQuadrilateral(no[0], no[1], mid12, mid30,
                                     attr, fa[0], fa[1], -1, fa[3]);

         child[1] = NewQuadrilateral(mid30, mid12, no[2], no[3],
                                     attr, -1, fa[1], fa[2], fa[3]);
      }
      else if (ref_type == 3) // iso split
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

      if (ref_type != 3) { Iso = false; }
   }
   else if (el.geom == Geometry::TRIANGLE)
   {
      ref_type = 3; // for consistence

      // isotropic split - the only ref_type available for triangles
      int mid01 = nodes.GetId(no[0], no[1]);
      int mid12 = nodes.GetId(no[1], no[2]);
      int mid20 = nodes.GetId(no[2], no[0]);

      child[0] = NewTriangle(no[0], mid01, mid20, attr, fa[0], -1, fa[2]);
      child[1] = NewTriangle(mid01, no[1], mid12, attr, fa[0], fa[1], -1);
      child[2] = NewTriangle(mid20, mid12, no[2], attr, -1, fa[1], fa[2]);
      child[3] = NewTriangle(mid01, mid12, mid20, attr, -1, -1, -1);
   }
   else
   {
      MFEM_ABORT("Unsupported element geometry.");
   }

   // start using the nodes of the children, create edges & faces
   for (int i = 0; i < 8 && child[i] >= 0; i++)
   {
      RefElement(child[i]);
   }

   int buf[6];
   Array<int> parentFaces(buf, 6);
   parentFaces.SetSize(0);

   // sign off of all nodes of the parent, clean up unused nodes, but keep faces
   UnrefElement(elem, parentFaces);

   // register the children in their faces
   for (int i = 0; i < 8 && child[i] >= 0; i++)
   {
      RegisterFaces(child[i]);
   }

   // clean up parent faces, if unused
   DeleteUnusedFaces(parentFaces);

   // make the children inherit our rank, set the parent element
   for (int i = 0; i < 8 && child[i] >= 0; i++)
   {
      Element &ch = elements[child[i]];
      ch.rank = el.rank;
      ch.parent = elem;
   }

   // finish the refinement
   el.ref_type = ref_type;
   memcpy(el.child, child, sizeof(el.child));
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
      forced refinements spread through the mesh, some may not be necessary
      in the end, since the affected elements may still be scheduled for
      refinement that could stop the propagation. We should introduce the
      member Element::ref_pending that would show the intended refinement in
      the batch. A forced refinement would be combined with ref_pending to
      (possibly) stop the propagation earlier.

      Update: what about a FIFO instead of ref_stack? */

#if defined(MFEM_DEBUG) && !defined(MFEM_USE_MPI)
   std::cout << "Refined " << refinements.Size() << " + " << nforced
             << " elements" << std::endl;
#endif
   ref_stack.DeleteAll();

   Update();
}


//// Derefinement //////////////////////////////////////////////////////////////

void NCMesh::DerefineElement(int elem)
{
   Element &el = elements[elem];
   if (!el.ref_type) { return; }

   int child[8];
   memcpy(child, el.child, sizeof(child));

   // first make sure that all children are leaves, derefine them if not
   for (int i = 0; i < 8 && child[i] >= 0; i++)
   {
      if (elements[child[i]].ref_type)
      {
         DerefineElement(child[i]);
      }
   }

   // retrieve original corner nodes and face attributes from the children
   int fa[6];
   if (el.geom == Geometry::CUBE)
   {
      const int table[7][8 + 6] =
      {
         { 0, 1, 1, 0, 0, 1, 1, 0, /**/ 1, 1, 1, 0, 0, 0 }, // 1 - X
         { 0, 0, 1, 1, 0, 0, 1, 1, /**/ 0, 0, 0, 1, 1, 1 }, // 2 - Y
         { 0, 1, 2, 3, 0, 1, 2, 3, /**/ 1, 1, 1, 3, 3, 3 }, // 3 - XY
         { 0, 0, 0, 0, 1, 1, 1, 1, /**/ 0, 0, 0, 1, 1, 1 }, // 4 - Z
         { 0, 1, 1, 0, 3, 2, 2, 3, /**/ 1, 1, 1, 3, 3, 3 }, // 5 - XZ
         { 0, 0, 1, 1, 2, 2, 3, 3, /**/ 0, 0, 0, 3, 3, 3 }, // 6 - YZ
         { 0, 1, 2, 3, 4, 5, 6, 7, /**/ 1, 1, 1, 7, 7, 7 }  // 7 - iso
      };
      for (int i = 0; i < 8; i++)
      {
         el.node[i] = elements[child[table[el.ref_type - 1][i]]].node[i];
      }
      for (int i = 0; i < 6; i++)
      {
         Element &ch = elements[child[table[el.ref_type - 1][i + 8]]];
         const int* fv = gi_hex.faces[i];
         fa[i] = faces.Find(ch.node[fv[0]], ch.node[fv[1]],
                            ch.node[fv[2]], ch.node[fv[3]])->attribute;
      }
   }
   else if (el.geom == Geometry::SQUARE)
   {
      const int table[3][4 + 4] =
      {
         { 0, 1, 1, 0, /**/ 1, 1, 0, 0 }, // 1 - X
         { 0, 0, 1, 1, /**/ 0, 0, 1, 1 }, // 2 - Y
         { 0, 1, 2, 3, /**/ 1, 1, 3, 3 }  // 3 - iso
      };
      for (int i = 0; i < 4; i++)
      {
         el.node[i] = elements[child[table[el.ref_type - 1][i]]].node[i];
      }
      for (int i = 0; i < 4; i++)
      {
         Element &ch = elements[child[table[el.ref_type - 1][i + 4]]];
         const int* fv = gi_quad.faces[i];
         fa[i] = faces.Find(ch.node[fv[0]], ch.node[fv[1]],
                            ch.node[fv[2]], ch.node[fv[3]])->attribute;
      }
   }
   else if (el.geom == Geometry::TRIANGLE)
   {
      for (int i = 0; i < 3; i++)
      {
         Element& ch = elements[child[i]];
         el.node[i] = ch.node[i];
         const int* fv = gi_tri.faces[i];
         fa[i] = faces.Find(ch.node[fv[0]], ch.node[fv[1]],
                            ch.node[fv[2]], ch.node[fv[3]])->attribute;
      }
   }
   else
   {
      MFEM_ABORT("Unsupported element geometry.");
   }

   // sign in to all nodes again
   RefElement(elem);

   int buf[8*6];
   Array<int> childFaces(buf, 8*6);
   childFaces.SetSize(0);

   // delete children, determine rank
   el.rank = INT_MAX;
   for (int i = 0; i < 8 && child[i] >= 0; i++)
   {
      el.rank = std::min(el.rank, elements[child[i]].rank);
      UnrefElement(child[i], childFaces);
      FreeElement(child[i]);
   }

   RegisterFaces(elem, fa);

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
   for (int i = 0; i < 8 && el.child[i] >= 0; i++)
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
      for (int i = 0; i < 8 && el.child[i] >= 0; i++)
      {
         Element &ch = elements[el.child[i]];
         list.Append(Connection(next_row, ch.index));
      }
   }
   else
   {
      for (int i = 0; i < 8 && el.child[i] >= 0; i++)
      {
         CollectDerefinements(el.child[i], list);
      }
   }
}

const Table& NCMesh::GetDerefinementTable()
{
   Array<Connection> list;
   list.Reserve(leaf_elements.Size());

   for (int i = 0; i < root_count; i++)
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

   transforms.embeddings.SetSize(nfine);
   for (int i = 0; i < nfine; i++)
   {
      transforms.embeddings[i].parent = -1;
      transforms.embeddings[i].matrix = 0;
   }

   // this will tell GetDerefinementTransforms that transforms are not finished
   transforms.point_matrices.SetSize(0, 0, 0);
}

void NCMesh::SetDerefMatrixCodes(int parent, Array<int> &fine_coarse)
{
   // encode the ref_type and child number for GetDerefinementTransforms()
   Element &prn = elements[parent];
   for (int i = 0; i < 8 && prn.child[i] >= 0; i++)
   {
      Element &ch = elements[prn.child[i]];
      if (ch.index >= 0)
      {
         int code = (prn.ref_type << 3) + i;
         transforms.embeddings[ch.index].matrix = code;
         fine_coarse[ch.index] = parent;
      }
   }
}


//// Mesh Interface ////////////////////////////////////////////////////////////

void NCMesh::UpdateVertices()
{
   // (overridden in ParNCMesh to assign special indices to ghost vertices)
   int num_vert = 0;
   for (node_iterator node = nodes.begin(); node != nodes.end(); ++node)
   {
      if (node->HasVertex()) { node->vert_index = num_vert++; }
   }

   vertex_nodeId.SetSize(num_vert);

   num_vert = 0;
   for (node_iterator node = nodes.begin(); node != nodes.end(); ++node)
   {
      if (node->HasVertex()) { vertex_nodeId[num_vert++] = node.index(); }
   }
}

static char quad_hilbert_child_order[8][4] =
{
   {0,1,2,3}, {0,3,2,1}, {1,2,3,0}, {1,0,3,2},
   {2,3,0,1}, {2,1,0,3}, {3,0,1,2}, {3,2,1,0}
};
static char quad_hilbert_child_state[8][4] =
{
   {1,0,0,5}, {0,1,1,4}, {3,2,2,7}, {2,3,3,6},
   {5,4,4,1}, {4,5,5,0}, {7,6,6,3}, {6,7,7,2}
};
static char hex_hilbert_child_order[24][8] =
{
   {0,1,2,3,7,6,5,4}, {0,3,7,4,5,6,2,1}, {0,4,5,1,2,6,7,3},
   {1,0,3,2,6,7,4,5}, {1,2,6,5,4,7,3,0}, {1,5,4,0,3,7,6,2},
   {2,1,5,6,7,4,0,3}, {2,3,0,1,5,4,7,6}, {2,6,7,3,0,4,5,1},
   {3,0,4,7,6,5,1,2}, {3,2,1,0,4,5,6,7}, {3,7,6,2,1,5,4,0},
   {4,0,1,5,6,2,3,7}, {4,5,6,7,3,2,1,0}, {4,7,3,0,1,2,6,5},
   {5,1,0,4,7,3,2,6}, {5,4,7,6,2,3,0,1}, {5,6,2,1,0,3,7,4},
   {6,2,3,7,4,0,1,5}, {6,5,1,2,3,0,4,7}, {6,7,4,5,1,0,3,2},
   {7,3,2,6,5,1,0,4}, {7,4,0,3,2,1,5,6}, {7,6,5,4,0,1,2,3}
};
static char hex_hilbert_child_state[24][8] =
{
   {1,2,2,7,7,21,21,17},     {2,0,0,22,22,16,16,8},    {0,1,1,15,15,6,6,23},
   {4,5,5,10,10,18,18,14},   {5,3,3,19,19,13,13,11},   {3,4,4,12,12,9,9,20},
   {8,7,7,17,17,23,23,2},    {6,8,8,0,0,15,15,22},     {7,6,6,21,21,1,1,16},
   {11,10,10,14,14,20,20,5}, {9,11,11,3,3,12,12,19},   {10,9,9,18,18,4,4,13},
   {13,14,14,5,5,19,19,10},  {14,12,12,20,20,11,11,4}, {12,13,13,9,9,3,3,18},
   {16,17,17,2,2,22,22,7},   {17,15,15,23,23,8,8,1},   {15,16,16,6,6,0,0,21},
   {20,19,19,11,11,14,14,3}, {18,20,20,4,4,10,10,12},  {19,18,18,13,13,5,5,9},
   {23,22,22,8,8,17,17,0},   {21,23,23,1,1,7,7,15},    {22,21,21,16,16,2,2,6}
};

void NCMesh::CollectLeafElements(int elem, int state)
{
   Element &el = elements[elem];
   if (!el.ref_type)
   {
      if (el.rank >= 0) // skip elements beyond ghost layer in parallel
      {
         leaf_elements.Append(elem);
      }
   }
   else
   {
      if (el.geom == Geometry::SQUARE && el.ref_type == 3)
      {
         for (int i = 0; i < 4; i++)
         {
            int ch = quad_hilbert_child_order[state][i];
            int st = quad_hilbert_child_state[state][i];
            CollectLeafElements(el.child[ch], st);
         }
      }
      else if (el.geom == Geometry::CUBE && el.ref_type == 7)
      {
         for (int i = 0; i < 8; i++)
         {
            int ch = hex_hilbert_child_order[state][i];
            int st = hex_hilbert_child_state[state][i];
            CollectLeafElements(el.child[ch], st);
         }
      }
      else
      {
         for (int i = 0; i < 8; i++)
         {
            if (el.child[i] >= 0) { CollectLeafElements(el.child[i], state); }
         }
      }
   }
   el.index = -1;
}

void NCMesh::UpdateLeafElements()
{
   // collect leaf elements from all roots
   leaf_elements.SetSize(0);
   for (int i = 0; i < root_count; i++)
   {
      CollectLeafElements(i, 0);
      // TODO: root state should not always be 0, we need a precomputed array
      // with root element states to ensure continuity where possible, also
      // optimized ordering of the root elements themselves (Gecko?)
   }
   AssignLeafIndices();
}

void NCMesh::AssignLeafIndices()
{
   // (overridden in ParNCMesh to handle ghost elements)
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      elements[leaf_elements[i]].index = i;
   }
}

mfem::Element* NCMesh::NewMeshElement(int geom) const
{
   switch (geom)
   {
      case Geometry::CUBE: return new mfem::Hexahedron;
      case Geometry::SQUARE: return new mfem::Quadrilateral;
      case Geometry::TRIANGLE: return new mfem::Triangle;
   }
   MFEM_ABORT("invalid geometry");
   return NULL;
}

const double* NCMesh::CalcVertexPos(int node) const
{
   const Node &nd = nodes[node];
   if (nd.p1 == nd.p2) // top-level vertex
   {
      return &top_vertex_pos[3*nd.p1];
   }

   TmpVertex &tv = tmp_vertex[nd.vert_index];
   if (tv.valid) { return tv.pos; }

   MFEM_VERIFY(tv.visited == false, "cyclic vertex dependencies.");
   tv.visited = true;

   const double* pos1 = CalcVertexPos(nd.p1);
   const double* pos2 = CalcVertexPos(nd.p2);

   for (int i = 0; i < 3; i++)
   {
      tv.pos[i] = (pos1[i] + pos2[i]) * 0.5;
   }
   tv.valid = true;
   return tv.pos;
}

void NCMesh::GetMeshComponents(Array<mfem::Vertex>& mvertices,
                               Array<mfem::Element*>& melements,
                               Array<mfem::Element*>& mboundary,
                               bool want_vertices) const
{
   if (want_vertices)
   {
      mvertices.SetSize(vertex_nodeId.Size());
      tmp_vertex = new TmpVertex[nodes.NumIds()];
      for (int i = 0; i < mvertices.Size(); i++)
      {
         mvertices[i].SetCoords(spaceDim, CalcVertexPos(vertex_nodeId[i]));
      }
      delete [] tmp_vertex;
   }

   melements.SetSize(leaf_elements.Size() - GetNumGhosts());
   melements.SetSize(0);

   mboundary.SetSize(0);

   // create an mfem::Element for each leaf Element
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      const Element &nc_elem = elements[leaf_elements[i]];
      if (IsGhost(nc_elem)) { continue; } // ParNCMesh

      const int* node = nc_elem.node;
      GeomInfo& gi = GI[(int) nc_elem.geom];

      mfem::Element* elem = NewMeshElement(nc_elem.geom);
      melements.Append(elem);

      elem->SetAttribute(nc_elem.attribute);
      for (int j = 0; j < gi.nv; j++)
      {
         elem->GetVertices()[j] = nodes[node[j]].vert_index;
      }

      // create boundary elements
      for (int k = 0; k < gi.nf; k++)
      {
         const int* fv = gi.faces[k];
         const Face* face = faces.Find(node[fv[0]], node[fv[1]],
                                       node[fv[2]], node[fv[3]]);
         if (face->Boundary())
         {
            if (nc_elem.geom == Geometry::CUBE)
            {
               Quadrilateral* quad = new Quadrilateral;
               quad->SetAttribute(face->attribute);
               for (int j = 0; j < 4; j++)
               {
                  quad->GetVertices()[j] = nodes[node[fv[j]]].vert_index;
               }
               mboundary.Append(quad);
            }
            else
            {
               Segment* segment = new Segment;
               segment->SetAttribute(face->attribute);
               for (int j = 0; j < 2; j++)
               {
                  segment->GetVertices()[j] = nodes[node[fv[2*j]]].vert_index;
               }
               mboundary.Append(segment);
            }
         }
      }
   }
}

void NCMesh::OnMeshUpdated(Mesh *mesh)
{
   Table *edge_vertex = mesh->GetEdgeVertexTable();

   // get edge enumeration from the Mesh
   for (int i = 0; i < edge_vertex->Size(); i++)
   {
      const int *ev = edge_vertex->GetRow(i);
      Node* node = nodes.Find(vertex_nodeId[ev[0]], vertex_nodeId[ev[1]]);

      MFEM_ASSERT(node && node->HasEdge(), "edge not found.");
      node->edge_index = i;
   }

   // get face enumeration from the Mesh
   for (int i = 0; i < mesh->GetNumFaces(); i++)
   {
      const int* fv = mesh->GetFace(i)->GetVertices();
      Face* face;
      if (Dim == 3)
      {
         MFEM_ASSERT(mesh->GetFace(i)->GetNVertices() == 4, "");
         face = faces.Find(vertex_nodeId[fv[0]], vertex_nodeId[fv[1]],
                           vertex_nodeId[fv[2]], vertex_nodeId[fv[3]]);
      }
      else
      {
         MFEM_ASSERT(mesh->GetFace(i)->GetNVertices() == 2, "");
         int n0 = vertex_nodeId[fv[0]], n1 = vertex_nodeId[fv[1]];
         face = faces.Find(n0, n0, n1, n1); // look up degenerate face

#ifdef MFEM_DEBUG
         // (non-ghost) edge and face numbers must match in 2D
         const int *ev = edge_vertex->GetRow(i);
         MFEM_ASSERT((ev[0] == fv[0] && ev[1] == fv[1]) ||
                     (ev[1] == fv[0] && ev[0] == fv[1]), "");
#endif
      }
      MFEM_ASSERT(face, "face not found.");
      face->index = i;
   }
}


//// Face/edge lists ///////////////////////////////////////////////////////////

int NCMesh::FaceSplitType(int v1, int v2, int v3, int v4,
                          int mid[4]) const
{
   MFEM_ASSERT(Dim >= 3, "");

   // find edge nodes
   int e1 = nodes.FindId(v1, v2);
   int e2 = nodes.FindId(v2, v3);
   int e3 = (e1 >= 0 && nodes[e1].HasVertex()) ? nodes.FindId(v3, v4) : -1;
   int e4 = (e2 >= 0 && nodes[e2].HasVertex()) ? nodes.FindId(v4, v1) : -1;

   // optional: return the mid-edge nodes if requested
   if (mid) { mid[0] = e1, mid[1] = e2, mid[2] = e3, mid[3] = e4; }

   // try to get a mid-face node, either by (e1, e3) or by (e2, e4)
   int midf1 = -1, midf2 = -1;
   if (e1 >= 0 && e3 >= 0) { midf1 = nodes.FindId(e1, e3); }
   if (e2 >= 0 && e4 >= 0) { midf2 = nodes.FindId(e2, e4); }

   // only one way to access the mid-face node must always exist
   MFEM_ASSERT(!(midf1 >= 0 && midf2 >= 0), "incorrectly split face!");

   if (midf1 < 0 && midf2 < 0) { return 0; }  // face not split
   else if (midf1 >= 0) { return 1; }  // face split "vertically"
   else { return 2; }  // face split "horizontally"
}

int NCMesh::find_node(const Element &el, int node)
{
   for (int i = 0; i < 8; i++)
   {
      if (el.node[i] == node) { return i; }
   }
   MFEM_ABORT("Node not found.");
   return -1;
}

int NCMesh::find_element_edge(const Element &el, int vn0, int vn1)
{
   MFEM_ASSERT(!el.ref_type, "");
   GeomInfo &gi = GI[(int) el.geom];
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      int n0 = el.node[ev[0]];
      int n1 = el.node[ev[1]];
      if ((n0 == vn0 && n1 == vn1) ||
          (n0 == vn1 && n1 == vn0)) { return i; }
   }
   MFEM_ABORT("Edge not found");
   return -1;
}

int NCMesh::find_hex_face(int a, int b, int c)
{
   for (int i = 0; i < 6; i++)
   {
      const int* fv = gi_hex.faces[i];
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

int NCMesh::ReorderFacePointMat(int v0, int v1, int v2, int v3,
                                int elem, DenseMatrix& mat) const
{
   const Element &el = elements[elem];
   int master[4] =
   {
      find_node(el, v0), find_node(el, v1),
      find_node(el, v2), find_node(el, v3)
   };

   int local = find_hex_face(master[0], master[1], master[2]);
   const int* fv = gi_hex.faces[local];

   DenseMatrix tmp(mat);
   for (int i = 0, j; i < 4; i++)
   {
      for (j = 0; j < 4; j++)
      {
         if (fv[i] == master[j])
         {
            // "pm.column(i) = tmp.column(j)"
            for (int k = 0; k < mat.Height(); k++)
            {
               mat(k,i) = tmp(k,j);
            }
            break;
         }
      }
      MFEM_ASSERT(j != 4, "node not found.");
   }
   return local;
}

void NCMesh::TraverseFace(int vn0, int vn1, int vn2, int vn3,
                          const PointMatrix& pm, int level)
{
   if (level > 0)
   {
      // check if we made it to a face that is not split further
      Face* fa = faces.Find(vn0, vn1, vn2, vn3);
      if (fa)
      {
         // we have a slave face, add it to the list
         int elem = fa->GetSingleElement();
         face_list.slaves.push_back(Slave(fa->index, elem, -1));
         DenseMatrix &mat = face_list.slaves.back().point_matrix;
         pm.GetMatrix(mat);

         // reorder the point matrix according to slave face orientation
         int local = ReorderFacePointMat(vn0, vn1, vn2, vn3, elem, mat);
         face_list.slaves.back().local = local;

         return;
      }
   }

   // we need to recurse deeper
   int mid[4];
   int split = FaceSplitType(vn0, vn1, vn2, vn3, mid);

   if (split == 1) // "X" split face
   {
      Point mid0(pm(0), pm(1)), mid2(pm(2), pm(3));

      TraverseFace(vn0, mid[0], mid[2], vn3,
                   PointMatrix(pm(0), mid0, mid2, pm(3)), level+1);

      TraverseFace(mid[0], vn1, vn2, mid[2],
                   PointMatrix(mid0, pm(1), pm(2), mid2), level+1);
   }
   else if (split == 2) // "Y" split face
   {
      Point mid1(pm(1), pm(2)), mid3(pm(3), pm(0));

      TraverseFace(vn0, vn1, mid[1], mid[3],
                   PointMatrix(pm(0), pm(1), mid1, mid3), level+1);

      TraverseFace(mid[3], mid[1], vn2, vn3,
                   PointMatrix(mid3, mid1, pm(2), pm(3)), level+1);
   }
}

void NCMesh::BuildFaceList()
{
   face_list.Clear();
   boundary_faces.SetSize(0);

   if (Dim < 3) { return; }

   Array<char> processed_faces(faces.NumIds());
   processed_faces = 0;

   // visit faces of leaf elements
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      int elem = leaf_elements[i];
      Element &el = elements[elem];
      MFEM_ASSERT(!el.ref_type, "not a leaf element.");

      GeomInfo& gi = GI[(int) el.geom];
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
         ElementSharesFace(elem, face);

         // have we already processed this face? skip if yes
         if (processed_faces[face]) { continue; }
         processed_faces[face] = 1;

         Face &fa = faces[face];
         if (fa.elem[0] >= 0 && fa.elem[1] >= 0)
         {
            // this is a conforming face, add it to the list
            face_list.conforming.push_back(MeshId(fa.index, elem, j));
         }
         else
         {
            PointMatrix pm(Point(0,0), Point(1,0), Point(1,1), Point(0,1));

            // this is either a master face or a slave face, but we can't
            // tell until we traverse the face refinement 'tree'...
            int sb = face_list.slaves.size();
            TraverseFace(node[0], node[1], node[2], node[3], pm, 0);

            int se = face_list.slaves.size();
            if (sb < se)
            {
               // found slaves, so this is a master face; add it to the list
               face_list.masters.push_back(Master(fa.index, elem, j, sb, se));

               // also, set the master index for the slaves
               for (int i = sb; i < se; i++)
               {
                  face_list.slaves[i].master = fa.index;
               }
            }
         }

         if (fa.Boundary()) { boundary_faces.Append(face); }
      }
   }
}

void NCMesh::TraverseEdge(int vn0, int vn1, double t0, double t1, int flags,
                          int level)
{
   int mid = nodes.FindId(vn0, vn1);
   if (mid < 0) { return; }

   Node &nd = nodes[mid];
   if (nd.HasEdge() && level > 0)
   {
      // we have a slave edge, add it to the list
      edge_list.slaves.push_back(Slave(nd.edge_index, -1, -1));
      Slave &sl = edge_list.slaves.back();

      sl.point_matrix.SetSize(1, 2);
      sl.point_matrix(0,0) = t0;
      sl.point_matrix(0,1) = t1;

      // handle slave edge orientation
      sl.edge_flags = flags;
      int v0index = nodes[vn0].vert_index;
      int v1index = nodes[vn1].vert_index;
      if (v0index > v1index) { sl.edge_flags |= 2; }

      // in 2D, get the element/local info from the degenerate face
      if (Dim == 2)
      {
         Face* fa = faces.Find(vn0, vn0, vn1, vn1);
         MFEM_ASSERT(fa != NULL, "");
         sl.element = fa->GetSingleElement();
         sl.local = find_element_edge(elements[sl.element], vn0, vn1);
      }
   }

   // recurse deeper
   double tmid = (t0 + t1) / 2;
   TraverseEdge(vn0, mid, t0, tmid, flags, level+1);
   TraverseEdge(mid, vn1, tmid, t1, flags, level+1);
}

void NCMesh::BuildEdgeList()
{
   edge_list.Clear();
   if (Dim <= 2)
   {
      boundary_faces.SetSize(0);
   }

   Array<char> processed_edges(nodes.NumIds());
   processed_edges = 0;

   // visit edges of leaf elements
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      int elem = leaf_elements[i];
      Element &el = elements[elem];
      MFEM_ASSERT(!el.ref_type, "not a leaf element.");

      GeomInfo& gi = GI[(int) el.geom];
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
         ElementSharesEdge(elem, enode);

         // (2D only, store boundary faces)
         if (Dim <= 2)
         {
            int face = faces.FindId(node[0], node[0], node[1], node[1]);
            MFEM_ASSERT(face >= 0, "face not found!");
            if (faces[face].Boundary()) { boundary_faces.Append(face); }
         }

         // skip slave edges here, they will be reached from their masters
         if (GetEdgeMaster(enode) >= 0) { continue; }

         // have we already processed this edge? skip if yes
         if (processed_edges[enode]) { continue; }
         processed_edges[enode] = 1;

         // prepare edge interval for slave traversal, handle orientation
         double t0 = 0.0, t1 = 1.0;
         int v0index = nodes[node[0]].vert_index;
         int v1index = nodes[node[1]].vert_index;
         int flags = (v0index > v1index) ? 1 : 0;

         // try traversing the edge to find slave edges
         int sb = edge_list.slaves.size();
         TraverseEdge(node[0], node[1], t0, t1, flags, 0);

         int se = edge_list.slaves.size();
         if (sb < se)
         {
            // found slaves, this is a master face; add it to the list
            edge_list.masters.push_back(Master(nd.edge_index, elem, j, sb, se));

            // also, set the master index for the slaves
            for (int i = sb; i < se; i++)
            {
               edge_list.slaves[i].master = nd.edge_index;
            }
         }
         else
         {
            // no slaves, this is a conforming edge
            edge_list.conforming.push_back(MeshId(nd.edge_index, elem, j));
         }
      }
   }
}

void NCMesh::Slave::OrientedPointMatrix(DenseMatrix &oriented_matrix) const
{
   oriented_matrix = point_matrix;

   if (edge_flags)
   {
      MFEM_ASSERT(oriented_matrix.Height() == 1 &&
                  oriented_matrix.Width() == 2, "not an edge point matrix");

      if (edge_flags & 1) // master inverted
      {
         oriented_matrix(0,0) = 1.0 - oriented_matrix(0,0);
         oriented_matrix(0,1) = 1.0 - oriented_matrix(0,1);
      }
      if (edge_flags & 2) // slave inverted
      {
         std::swap(oriented_matrix(0,0), oriented_matrix(0,1));
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

void NCMesh::CollectFaceVertices(int v0, int v1, int v2, int v3,
                                 Array<int> &indices)
{
   int mid[4];
   switch (FaceSplitType(v0, v1, v2, v3, mid))
   {
      case 1:
         indices.Append(mid[0]);
         indices.Append(mid[2]);

         CollectFaceVertices(v0, mid[0], mid[2], v3, indices);
         CollectFaceVertices(mid[0], v1, v2, mid[2], indices);
         break;

      case 2:
         indices.Append(mid[1]);
         indices.Append(mid[3]);

         CollectFaceVertices(v0, v1, mid[1], mid[3], indices);
         CollectFaceVertices(mid[3], mid[1], v2, v3, indices);
         break;
   }
}

void NCMesh::BuildElementToVertexTable()
{
   int nrows = leaf_elements.Size();
   int* I = new int[nrows + 1];
   int** JJ = new int*[nrows];

   Array<int> indices;
   indices.Reserve(128);

   // collect vertices coinciding with each element, including hanging vertices
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      int elem = leaf_elements[i];
      Element &el = elements[elem];
      MFEM_ASSERT(!el.ref_type, "not a leaf element.");

      GeomInfo& gi = GI[(int) el.geom];
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
            CollectFaceVertices(node[fv[0]], node[fv[1]],
                                node[fv[2]], node[fv[3]], indices);
         }
      }

      // temporarily store one row of the table
      indices.Sort();
      indices.Unique();
      int size = indices.Size();
      I[i] = size;
      JJ[i] = new int[size];
      memcpy(JJ[i], indices.GetData(), size * sizeof(int));
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
   int *J = new int[nnz];
   nnz = 0;
   for (int i = 0; i < nrows; i++)
   {
      int cnt = I[i+1] - I[i];
      memcpy(J+nnz, JJ[i], cnt * sizeof(int));
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

   // Optimization: the 'element_vertex' table does not store the obvious
   // corner nodes in it. The table is therefore empty for conforming meshes.

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
         nv = GI[(int) el.geom].nv;
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
            nv = GI[(int) el.geom].nv;
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
   if (!na || !nb) { return false; }
   int a_last = a[na-1], b_last = b[nb-1];
   if (*b < *a) { goto l2; }  // woo-hoo! I always wanted to use a goto! :)
l1:
   if (a_last < *b) { return false; }
   while (*a < *b) { a++; }
   if (*a == *b) { return true; }
l2:
   if (b_last < *a) { return false; }
   while (*b < *a) { b++; }
   if (*a == *b) { return true; }
   goto l1;
}

void NCMesh::FindNeighbors(int elem, Array<int> &neighbors,
                           const Array<int> *search_set)
{
   // TODO future: this function is inefficient. For a single element, an
   // octree neighbor search algorithm would be better. However, the octree
   // neighbor algorithm is hard to get right in the multi-octree case due to
   // the differrent orientations of the octrees (i.e., the root elements).

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

         nv = GI[(int) el.geom].nv;
         for (int i = 0; i < nv; i++)
         {
            vert.Append(el.node[i]);
         }
      }
      else
      {
         for (int i = 0; i < 8 && el.child[i] >= 0; i++)
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
            int nv = GI[(int) el.geom].nv;
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

      nv = GI[(int) el.geom].nv;
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
         nv = GI[(int) el.geom].nv;
         for (int j = 0; j < nv; j++)
         {
            if (vmark[el.node[j]]) { hit = true; break; }
         }
      }

      if (hit) { expanded.Append(testme); }
   }
}

#ifdef MFEM_DEBUG
void NCMesh::DebugNeighbors(Array<char> &elem_set)
{
   Array<int> neighbors;
   FindSetNeighbors(elem_set, &neighbors);

   for (int i = 0; i < neighbors.Size(); i++)
   {
      elem_set[elements[neighbors[i]].index] = 2;
   }
}
#endif


//// Coarse/fine transformations ///////////////////////////////////////////////

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

NCMesh::PointMatrix NCMesh::pm_tri_identity(
   Point(0, 0), Point(1, 0), Point(0, 1)
);
NCMesh::PointMatrix NCMesh::pm_quad_identity(
   Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)
);
NCMesh::PointMatrix NCMesh::pm_hex_identity(
   Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0),
   Point(0, 0, 1), Point(1, 0, 1), Point(1, 1, 1), Point(0, 1, 1)
);

const NCMesh::PointMatrix& NCMesh::GetGeomIdentity(int geom)
{
   switch (geom)
   {
      case Geometry::TRIANGLE: return pm_tri_identity;
      case Geometry::SQUARE:   return pm_quad_identity;
      case Geometry::CUBE:     return pm_hex_identity;
      default:
         MFEM_ABORT("unsupported geometry.");
         return pm_tri_identity;
   }
}

void NCMesh::GetPointMatrix(int geom, const char* ref_path, DenseMatrix& matrix)
{
   PointMatrix pm = GetGeomIdentity(geom);

   while (*ref_path)
   {
      int ref_type = *ref_path++;
      int child = *ref_path++;

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
            pm = PointMatrix(mid01, mid12, mid20);
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
                                 std::string &ref_path, RefPathMap &map)
{
   Element &el = elements[elem];
   if (!el.ref_type)
   {
      int &matrix = map[ref_path];
      if (!matrix) { matrix = map.size(); }

      Embedding &emb = transforms.embeddings[el.index];
      emb.parent = coarse_index;
      emb.matrix = matrix - 1;
   }
   else
   {
      ref_path.push_back(el.ref_type);
      ref_path.push_back(0);

      for (int i = 0; i < 8; i++)
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

const CoarseFineTransformations& NCMesh::GetRefinementTransforms()
{
   MFEM_VERIFY(coarse_elements.Size() || !leaf_elements.Size(),
               "GetRefinementTransforms() must be preceded by MarkCoarseLevel()"
               " and Refine().");

   if (!transforms.embeddings.Size())
   {
      transforms.embeddings.SetSize(leaf_elements.Size());

      std::string ref_path;
      ref_path.reserve(100);

      RefPathMap map;
      map[ref_path] = 1; // identity

      for (int i = 0; i < coarse_elements.Size(); i++)
      {
         TraverseRefinements(coarse_elements[i], i, ref_path, map);
      }

      MFEM_ASSERT(elements.Size() > free_element_ids.Size(), "");
      int geom = elements[0].geom;
      const PointMatrix &identity = GetGeomIdentity(geom);

      transforms.point_matrices.SetSize(Dim, identity.np, map.size());

      // calculate the point matrices
      for (RefPathMap::iterator it = map.begin(); it != map.end(); ++it)
      {
         GetPointMatrix(geom, it->first.c_str(),
                        transforms.point_matrices(it->second-1));
      }
   }
   return transforms;
}

const CoarseFineTransformations& NCMesh::GetDerefinementTransforms()
{
   MFEM_VERIFY(transforms.embeddings.Size() || !leaf_elements.Size(),
               "GetDerefinementTransforms() must be preceded by Derefine().");

   if (!transforms.point_matrices.SizeK())
   {
      std::map<int, int> mat_no;
      mat_no[0] = 1; // identity

      // assign numbers to the different matrices used
      for (int i = 0; i < transforms.embeddings.Size(); i++)
      {
         int code = transforms.embeddings[i].matrix;
         if (code)
         {
            int &matrix = mat_no[code];
            if (!matrix) { matrix = mat_no.size(); }
            transforms.embeddings[i].matrix = matrix - 1;
         }
      }

      MFEM_ASSERT(elements.Size() > free_element_ids.Size(), "");
      int geom = elements[0].geom;
      const PointMatrix &identity = GetGeomIdentity(geom);

      transforms.point_matrices.SetSize(Dim, identity.np, mat_no.size());

      std::map<int, int>::iterator it;
      for (it = mat_no.begin(); it != mat_no.end(); ++it)
      {
         char path[3];
         int code = it->first;
         path[0] = code >> 3; // ref_type (see SetDerefMatrixCodes())
         path[1] = code & 7;  // child
         path[2] = 0;

         GetPointMatrix(geom, path, transforms.point_matrices(it->second-1));
      }
   }
   return transforms;
}

void NCMesh::ClearTransforms()
{
   coarse_elements.DeleteAll();
   transforms.embeddings.DeleteAll();
   transforms.point_matrices.SetSize(0, 0, 0);
}


//// Utility ///////////////////////////////////////////////////////////////////

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
      // n1 is parent of n2:
      // (n1)--(nd)--(n2)------(*)
      if (n2.HasEdge()) { return p2; }
      else { return GetEdgeMaster(p2); }
   }

   if ((n1p1 != n1p2) && (p2 == n1p1 || p2 == n1p2))
   {
      // n2 is parent of n1:
      // (n2)--(nd)--(n1)------(*)
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

void NCMesh::FindFaceNodes(int face, int node[4])
{
   // Obtain face nodes from one of its elements (note that face->p1, p2, p3
   // cannot be used directly since they are not in order and p4 is missing).

   Face &fa = faces[face];

   int elem = fa.elem[0];
   if (elem < 0) { elem = fa.elem[1]; }
   MFEM_ASSERT(elem >= 0, "Face has no elements?");

   Element &el = elements[elem];
   int f = find_hex_face(find_node(el, fa.p1),
                         find_node(el, fa.p2),
                         find_node(el, fa.p3));

   const int* fv = GI[Geometry::CUBE].faces[f];
   for (int i = 0; i < 4; i++)
   {
      node[i] = el.node[fv[i]];
   }
}

void NCMesh::GetBoundaryClosure(const Array<int> &bdr_attr_is_ess,
                                Array<int> &bdr_vertices, Array<int> &bdr_edges)
{
   bdr_vertices.SetSize(0);
   bdr_edges.SetSize(0);

   if (Dim == 3)
   {
      GetFaceList(); // make sure 'boundary_faces' is up to date

      for (int i = 0; i < boundary_faces.Size(); i++)
      {
         int face = boundary_faces[i];
         if (bdr_attr_is_ess[faces[face].attribute - 1])
         {
            int node[4];
            FindFaceNodes(face, node);

            for (int j = 0; j < 4; j++)
            {
               bdr_vertices.Append(nodes[node[j]].vert_index);

               int enode = nodes.FindId(node[j], node[(j+1) % 4]);
               MFEM_ASSERT(enode >= 0 && nodes[enode].HasEdge(), "Edge not found.");
               bdr_edges.Append(nodes[enode].edge_index);

               while ((enode = GetEdgeMaster(enode)) >= 0)
               {
                  // append master edges that may not be accessible from any
                  // boundary element, this happens in 3D in re-entrant corners
                  bdr_edges.Append(nodes[enode].edge_index);
               }
            }
         }
      }
   }
   else if (Dim == 2)
   {
      GetEdgeList(); // make sure 'boundary_faces' is up to date

      for (int i = 0; i < boundary_faces.Size(); i++)
      {
         int face = boundary_faces[i];
         Face &fc = faces[face];
         if (bdr_attr_is_ess[fc.attribute - 1])
         {
            bdr_vertices.Append(nodes[fc.p1].vert_index);
            bdr_vertices.Append(nodes[fc.p3].vert_index);
         }
      }
   }

   bdr_vertices.Sort();
   bdr_vertices.Unique();

   bdr_edges.Sort();
   bdr_edges.Unique();
}

int NCMesh::EdgeSplitLevel(int vn1, int vn2) const
{
   int mid = nodes.FindId(vn1, vn2);
   if (mid < 0 || !nodes[mid].HasVertex()) { return 0; }
   return 1 + std::max(EdgeSplitLevel(vn1, mid), EdgeSplitLevel(mid, vn2));
}

void NCMesh::FaceSplitLevel(int vn1, int vn2, int vn3, int vn4,
                            int& h_level, int& v_level) const
{
   int hl1, hl2, vl1, vl2;
   int mid[4];

   switch (FaceSplitType(vn1, vn2, vn3, vn4, mid))
   {
      case 0: // not split
         h_level = v_level = 0;
         break;

      case 1: // vertical
         FaceSplitLevel(vn1, mid[0], mid[2], vn4, hl1, vl1);
         FaceSplitLevel(mid[0], vn2, vn3, mid[2], hl2, vl2);
         h_level = std::max(hl1, hl2);
         v_level = std::max(vl1, vl2) + 1;
         break;

      default: // horizontal
         FaceSplitLevel(vn1, vn2, mid[1], mid[3], hl1, vl1);
         FaceSplitLevel(mid[3], mid[1], vn3, vn4, hl2, vl2);
         h_level = std::max(hl1, hl2) + 1;
         v_level = std::max(vl1, vl2);
   }
}

static int max8(int a, int b, int c, int d, int e, int f, int g, int h)
{
   return std::max(std::max(std::max(a, b), std::max(c, d)),
                   std::max(std::max(e, f), std::max(g, h)));
}

void NCMesh::CountSplits(int elem, int splits[3]) const
{
   const Element &el = elements[elem];
   const int* node = el.node;
   GeomInfo& gi = GI[(int) el.geom];

   int elevel[12];
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      elevel[i] = EdgeSplitLevel(node[ev[0]], node[ev[1]]);
   }

   if (el.geom == Geometry::CUBE)
   {
      int flevel[6][2];
      for (int i = 0; i < gi.nf; i++)
      {
         const int* fv = gi.faces[i];
         FaceSplitLevel(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]],
                        flevel[i][1], flevel[i][0]);
      }

      splits[0] = max8(flevel[0][0], flevel[1][0], flevel[3][0], flevel[5][0],
                       elevel[0], elevel[2], elevel[4], elevel[6]);

      splits[1] = max8(flevel[0][1], flevel[2][0], flevel[4][0], flevel[5][1],
                       elevel[1], elevel[3], elevel[5], elevel[7]);

      splits[2] = max8(flevel[1][1], flevel[2][1], flevel[3][1], flevel[4][1],
                       elevel[8], elevel[9], elevel[10], elevel[11]);
   }
   else if (el.geom == Geometry::SQUARE)
   {
      splits[0] = std::max(elevel[0], elevel[2]);
      splits[1] = std::max(elevel[1], elevel[3]);
   }
   else if (el.geom == Geometry::TRIANGLE)
   {
      splits[0] = std::max(elevel[0], std::max(elevel[1], elevel[2]));
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
      if (IsGhost(elements[leaf_elements[i]])) { break; }

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

void NCMesh::PrintVertexParents(std::ostream &out) const
{
   // count vertices with parents
   int nv = 0;
   for (node_const_iterator node = nodes.cbegin(); node != nodes.cend(); ++node)
   {
      if (node->HasVertex() && node->p1 != node->p2) { nv++; }
   }
   out << nv << "\n";

   // print the relations
   for (node_const_iterator node = nodes.cbegin(); node != nodes.cend(); ++node)
   {
      if (node->HasVertex() && node->p1 != node->p2)
      {
         const Node &p1 = nodes[node->p1];
         const Node &p2 = nodes[node->p2];

         MFEM_ASSERT(p1.HasVertex(), "");
         MFEM_ASSERT(p2.HasVertex(), "");

         out << node->vert_index << " "
             << p1.vert_index << " " << p2.vert_index << "\n";
      }
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

      // assign new parents for the node
      nodes.Reparent(id, p1, p2);

      // NOTE: when loading an AMR mesh, node indices are guaranteed to have
      // the same indices as vertices, see NCMesh::NCMesh.
   }
}

void NCMesh::SetVertexPositions(const Array<mfem::Vertex> &mvertices)
{
   int num_top_level = 0;
   for (node_iterator node = nodes.begin(); node != nodes.end(); ++node)
   {
      if (node->p1 == node->p2) // see NCMesh::NCMesh
      {
         MFEM_VERIFY(node.index() == node->p1, "invalid top-level vertex.");
         MFEM_VERIFY(node->HasVertex(), "top-level vertex not found.");
         MFEM_VERIFY(node->vert_index == node->p1, "bad top-level vertex index");
         num_top_level = std::max(num_top_level, node->p1 + 1);
      }
   }

   top_vertex_pos.SetSize(3*num_top_level);
   for (int i = 0; i < num_top_level; i++)
   {
      memcpy(&top_vertex_pos[3*i], mvertices[i](), 3*sizeof(double));
   }
}

static int ref_type_num_children[8] = { 0, 2, 2, 4, 2, 4, 4, 8 };

int NCMesh::PrintElements(std::ostream &out, int elem, int &coarse_id) const
{
   const Element &el = elements[elem];
   if (el.ref_type)
   {
      int child_id[8], nch = 0;
      for (int i = 0; i < 8 && el.child[i] >= 0; i++)
      {
         child_id[nch++] = PrintElements(out, el.child[i], coarse_id);
      }
      MFEM_ASSERT(nch == ref_type_num_children[(int) el.ref_type], "");

      out << (int) el.ref_type;
      for (int i = 0; i < nch; i++)
      {
         out << " " << child_id[i];
      }
      out << "\n";
      return coarse_id++; // return new id for this coarse element
   }
   else
   {
      return el.index;
   }
}

void NCMesh::PrintCoarseElements(std::ostream &out) const
{
   // print the number of non-leaf elements
   out << (elements.Size() - free_element_ids.Size() - leaf_elements.Size())
       << "\n";

   // print the hierarchy recursively
   int coarse_id = leaf_elements.Size();
   for (int i = 0; i < root_count; i++)
   {
      PrintElements(out, i, coarse_id);
   }
}

void NCMesh::CopyElements(int elem,
                          const BlockArray<Element> &tmp_elements,
                          Array<int> &index_map)
{
   Element &el = elements[elem];
   if (el.ref_type)
   {
      for (int i = 0; i < 8 && el.child[i] >= 0; i++)
      {
         int old_id = el.child[i];
         // here, we do not use the content of 'free_element_ids', if any
         int new_id = elements.Append(tmp_elements[old_id]);
         index_map[old_id] = new_id;
         el.child[i] = new_id;
         elements[new_id].parent = elem;
         CopyElements(new_id, tmp_elements, index_map);
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

      int elem = AddElement(Element(0, 0));
      Element &el = elements[elem];
      el.ref_type = ref_type;

      if (Dim == 3 && ref_type != 7) { iso = false; }

      // load child IDs and make parent-child links
      int nch = ref_type_num_children[ref_type];
      for (int i = 0, id; i < nch; i++)
      {
         input >> id;
         MFEM_VERIFY(id >= 0, "");
         MFEM_VERIFY(id < leaf_elements.Size() ||
                     id < elements.Size()-free_element_ids.Size(),
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
   free_element_ids.SetSize(0);

   Array<int> index_map(tmp_elements.Size());
   index_map = -1;

   // copy roots, they need to be at the beginning of 'elements'
   root_count = 0;
   for (elem_iterator el = tmp_elements.begin(); el != tmp_elements.end(); ++el)
   {
      if (el->parent == -1)
      {
         int new_id = elements.Append(*el); // same as AddElement()
         index_map[el.index()] = new_id;
         root_count++;
      }
   }

   // copy the rest of the hierarchy
   for (int i = 0; i < root_count; i++)
   {
      CopyElements(i, tmp_elements, index_map);
   }

   // we also need to renumber element links in Face::elem[]
   for (face_iterator face = faces.begin(); face != faces.end(); ++face)
   {
      for (int i = 0; i < 2; i++)
      {
         if (face->elem[i] >= 0)
         {
            face->elem[i] = index_map[face->elem[i]];
            MFEM_ASSERT(face->elem[i] >= 0, "");
         }
      }
   }

   // set the Iso flag (must be false if there are 3D aniso refinements)
   Iso = iso;

   Update();
}

long NCMesh::NCList::MemoryUsage() const
{
   int pmsize = 0;
   if (slaves.size())
   {
      pmsize = slaves[0].point_matrix.MemoryUsage();
   }

   return conforming.capacity() * sizeof(MeshId) +
          masters.capacity() * sizeof(Master) +
          slaves.capacity() * sizeof(Slave) +
          slaves.size() * pmsize;
}

long CoarseFineTransformations::MemoryUsage() const
{
   return point_matrices.MemoryUsage() + embeddings.MemoryUsage();
}

long NCMesh::MemoryUsage() const
{
   return nodes.MemoryUsage() +
          faces.MemoryUsage() +
          elements.MemoryUsage() +
          free_element_ids.MemoryUsage() +
          top_vertex_pos.MemoryUsage() +
          leaf_elements.MemoryUsage() +
          vertex_nodeId.MemoryUsage() +
          face_list.MemoryUsage() +
          edge_list.MemoryUsage() +
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
   nodes.PrintMemoryDetail(); std::cout << " nodes\n";
   faces.PrintMemoryDetail(); std::cout << " faces\n";

   std::cout << elements.MemoryUsage() << " elements\n"
             << free_element_ids.MemoryUsage() << " free_element_ids\n"
             << top_vertex_pos.MemoryUsage() << " top_vertex_pos\n"
             << leaf_elements.MemoryUsage() << " leaf_elements\n"
             << vertex_nodeId.MemoryUsage() << " vertex_nodeId\n"
             << face_list.MemoryUsage() << " face_list\n"
             << edge_list.MemoryUsage() << " edge_list\n"
             << boundary_faces.MemoryUsage() << " boundary_faces\n"
             << element_vertex.MemoryUsage() << " element_vertex\n"
             << ref_stack.MemoryUsage() << " ref_stack\n"
             << coarse_elements.MemoryUsage() << " coarse_elements\n"
             << sizeof(*this) << " NCMesh" << std::endl;

   return elements.Size()-free_element_ids.Size();
}

void NCMesh::PrintStats(std::ostream &out) const
{
   static const double MiB = 1024.*1024.;
   out <<
       "NCMesh statistics:\n"
       "------------------\n"
       "   mesh and space dimensions : " << Dim << ", " << spaceDim << "\n"
       "   isotropic only            : " << (Iso ? "yes" : "no") << "\n"
       "   number of Nodes           : " << std::setw(9)
       << nodes.Size() << " +    [ " << std::setw(9)
       << nodes.MemoryUsage()/MiB << " MiB ]\n"
       "      free                     " << std::setw(9)
       << nodes.NumFreeIds() << "\n"
       "   number of Faces           : " << std::setw(9)
       << faces.Size() << " +    [ " << std::setw(9)
       << faces.MemoryUsage()/MiB << " MiB ]\n"
       "      free                     " << std::setw(9)
       << faces.NumFreeIds() << "\n"
       "   number of Elements        : " << std::setw(9)
       << elements.Size()-free_element_ids.Size() << " +    [ " << std::setw(9)
       << (elements.MemoryUsage() +
           free_element_ids.MemoryUsage())/MiB << " MiB ]\n"
       "      free                     " << std::setw(9)
       << free_element_ids.Size() << "\n"
       "   number of root elements   : " << std::setw(9) << root_count << "\n"
       "   number of leaf elements   : " << std::setw(9)
       << leaf_elements.Size() << "\n"
       "   number of vertices        : " << std::setw(9)
       << vertex_nodeId.Size() << "\n"
       "   number of faces           : " << std::setw(9)
       << face_list.TotalSize() << " =    [ " << std::setw(9)
       << face_list.MemoryUsage()/MiB << " MiB ]\n"
       "      conforming               " << std::setw(9)
       << face_list.conforming.size() << " +\n"
       "      master                   " << std::setw(9)
       << face_list.masters.size() << " +\n"
       "      slave                    " << std::setw(9)
       << face_list.slaves.size() << "\n"
       "   number of edges           : " << std::setw(9)
       << edge_list.TotalSize() << " =    [ " << std::setw(9)
       << edge_list.MemoryUsage()/MiB << " MiB ]\n"
       "      conforming               " << std::setw(9)
       << edge_list.conforming.size() << " +\n"
       "      master                   " << std::setw(9)
       << edge_list.masters.size() << " +\n"
       "      slave                    " << std::setw(9)
       << edge_list.slaves.size() << "\n"
       "   total memory              : " << std::setw(17)
       << "[ " << std::setw(9) << MemoryUsage()/MiB << " MiB ]\n"
       ;
}

#if 0//def MFEM_DEBUG
void NCMesh::DebugLeafOrder() const
{
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      for (int j = 0; j < Dim; j++)
      {
         double sum = 0.0;
         int count = 0;
         for (int k = 0; k < 8; k++)
         {
            if (elem->node[k])
            {
               sum += elem->node[k]->vertex->pos[j];
               count++;
            }
         }
         std::cout << sum / count << " ";
      }
      std::cout << "\n";
   }
}
#endif

} // namespace mfem
