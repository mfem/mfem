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

#include "../fem/fem.hpp"
#include "ncmesh.hpp"

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
   for (int i = 0; i <= max_id; i++)
   {
      // top-level nodes are special: id == p1 == p2 == orig. vertex id
      Node* node = nodes.Get(i, i);
      MFEM_CONTRACT_VAR(node);
      MFEM_ASSERT(node->id == i, "");
   }

   // if a mesh file is being read, load the vertex hierarchy now;
   // 'vertex_parents' must be at the appropriate section in the mesh file
   if (vertex_parents)
   {
      LoadVertexParents(*vertex_parents);
   }

   // create the NCMesh::Element struct for each Mesh element
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const mfem::Element *elem = mesh->GetElement(i);
      const int *v = elem->GetVertices();

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
      Element* nc_elem = new Element(geom, elem->GetAttribute());
      root_elements.Append(nc_elem);

      for (int j = 0; j < GI[geom].nv; j++)
      {
         Node* node = nodes.Peek(v[j]);
         if (!node->vertex)
         {
            if (v[j] < mesh->GetNV())
            {
               // create a vertex in the node and initialize its position
               const double* pos = mesh->GetVertex(v[j]);
               node->vertex = new Vertex(pos[0], pos[1], pos[2]);
            }
            else
            {
               // the mesh may not have vertex positions defined yet
               node->vertex = new Vertex(0.0, 0.0, 0.0);
            }
         }
         nc_elem->node[j] = node;
      }

      // increase reference count of all nodes the element is using
      // (NOTE: this will also create and reference all edge and face nodes)
      RefElementNodes(nc_elem);

      // make links from faces back to the element
      RegisterFaces(nc_elem);
   }

   // store boundary element attributes
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const mfem::Element *be = mesh->GetBdrElement(i);
      const int *v = be->GetVertices();

      Node* node[4];
      for (int i = 0; i < be->GetNVertices(); i++)
      {
         node[i] = nodes.Peek(v[i]);
         MFEM_VERIFY(node[i], "boundary elements inconsistent.");
      }

      if (be->GetType() == mfem::Element::QUADRILATERAL)
      {
         Face* face = faces.Peek(node[0], node[1], node[2], node[3]);
         MFEM_VERIFY(face, "boundary face not found.");
         face->attribute = be->GetAttribute();
      }
      else if (be->GetType() == mfem::Element::SEGMENT)
      {
         Edge* edge = nodes.Peek(node[0], node[1])->edge;
         MFEM_VERIFY(edge, "boundary edge not found.");
         edge->attribute = be->GetAttribute();
      }
      else
      {
         MFEM_ABORT("only segment and quadrilateral boundary "
                    "elements are supported by NCMesh.");
      }
   }

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

NCMesh::Element* NCMesh::CopyHierarchy(Element* elem)
{
   Element* new_elem = new Element(*elem);
   if (elem->ref_type)
   {
      for (int i = 0; i < 8; i++)
      {
         if (elem->child[i])
         {
            new_elem->child[i] = CopyHierarchy(elem->child[i]);
            new_elem->child[i]->parent = new_elem;
         }
      }
   }
   else
   {
      GeomInfo& gi = GI[(int) elem->geom];
      for (int i = 0; i < gi.nv; i++)
      {
         new_elem->node[i] = nodes.Peek(elem->node[i]->id);
      }
      RegisterFaces(new_elem);
   }
   return new_elem;
}

void NCMesh::DeleteHierarchy(Element* elem)
{
   if (elem->ref_type)
   {
      for (int i = 0; i < 8; i++)
      {
         if (elem->child[i]) { DeleteHierarchy(elem->child[i]); }
      }
   }
   else
   {
      UnrefElementNodes(elem);
   }
   delete elem;
}

NCMesh::NCMesh(const NCMesh &other)
   : Dim(other.Dim), spaceDim(other.spaceDim), Iso(other.Iso)
   , nodes(other.nodes), faces(other.faces)
{
   // NOTE: this copy constructor is used by ParNCMesh
   root_elements.SetSize(other.root_elements.Size());
   for (int i = 0; i < root_elements.Size(); i++)
   {
      root_elements[i] = CopyHierarchy(other.root_elements[i]);
   }

   Update();
}

NCMesh::~NCMesh()
{
   for (int i = 0; i < root_elements.Size(); i++)
   {
      DeleteHierarchy(root_elements[i]);
   }
}


//// Node and Face Memory Management ///////////////////////////////////////////

void NCMesh::Node::RefVertex()
{
   MFEM_ASSERT(vertex, "can't create vertex here.");
   vertex->Ref();
}

void NCMesh::Node::RefEdge()
{
   if (!edge) { edge = new Edge; }
   edge->Ref();
}

void NCMesh::Node::UnrefVertex(HashTable<Node> &nodes)
{
   MFEM_ASSERT(vertex, "cannot unref a nonexistent vertex.");
   if (!vertex->Unref()) { vertex = NULL; }
   if (!vertex && !edge) { nodes.Delete(this); }
}

void NCMesh::Node::UnrefEdge(Node *node, HashTable<Node> &nodes)
{
   MFEM_ASSERT(node, "node not found.");
   MFEM_ASSERT(node->edge, "cannot unref a nonexistent edge.");
   if (!node->edge->Unref()) { node->edge = NULL; }
   if (!node->vertex && !node->edge) { nodes.Delete(node); }
}

NCMesh::Node::Node(const Node& other)
{
   std::memcpy(this, &other, sizeof(*this));
   if (vertex) { vertex = new Vertex(*vertex); }
   if (edge) { edge = new Edge(*edge); }
}

NCMesh::Node::~Node()
{
   MFEM_ASSERT(!vertex && !edge, "node was not unreffed properly.");
   if (vertex) { delete vertex; }
   if (edge) { delete edge; }
}

void NCMesh::RefElementNodes(Element *elem)
{
   Node** node = elem->node;
   GeomInfo& gi = GI[(int) elem->geom];

   // ref all vertices
   for (int i = 0; i < gi.nv; i++)
   {
      node[i]->RefVertex();
   }

   // ref all edges (possibly creating them)
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      nodes.Get(node[ev[0]], node[ev[1]])->RefEdge();
   }

   // ref all faces (possibly creating them)
   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      faces.Get(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]])->Ref();
      // NOTE: face->RegisterElement called elsewhere to avoid having
      // to store 3 element pointers temporarily in the face when refining.
      // See also NCMesh::RegisterFaces.
   }
}

void NCMesh::UnrefElementNodes(Element *elem)
{
   Node** node = elem->node;
   GeomInfo& gi = GI[(int) elem->geom];

   // unref all faces (possibly destroying them)
   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      Face* face = faces.Peek(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]]);
      face->ForgetElement(elem);
      if (!face->Unref()) { faces.Delete(face); }
   }

   // unref all edges (possibly destroying them)
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      //nodes.Peek(node[ev[0]], node[ev[1]])->UnrefEdge(nodes); -- pre-aniso
      Node::UnrefEdge(PeekAltParents(node[ev[0]], node[ev[1]]), nodes);
   }

   // unref all vertices (possibly destroying them)
   for (int i = 0; i < gi.nv; i++)
   {
      elem->node[i]->UnrefVertex(nodes);
   }
}

NCMesh::Face::Face(int id)
   : Hashed4<Face>(id), attribute(-1), index(-1)
{
   elem[0] = elem[1] = NULL;
}

NCMesh::Face::Face(const Face& other)
{
   // (skip Hashed4 constructor)
   std::memcpy(this, &other, sizeof(*this));
   elem[0] = elem[1] = NULL;
}

void NCMesh::Face::RegisterElement(Element* e)
{
   if (elem[0] == NULL) { elem[0] = e; }
   else if (elem[1] == NULL) { elem[1] = e; }
   else { MFEM_ABORT("can't have 3 elements in Face::elem[]."); }
}

void NCMesh::Face::ForgetElement(Element* e)
{
   if (elem[0] == e) { elem[0] = NULL; }
   else if (elem[1] == e) { elem[1] = NULL; }
   else { MFEM_ABORT("element not found in Face::elem[]."); }
}

NCMesh::Face* NCMesh::GetFace(Element* elem, int face_no)
{
   GeomInfo& gi = GI[(int) elem->geom];
   const int* fv = gi.faces[face_no];
   Node** node = elem->node;
   return faces.Peek(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]]);
}

void NCMesh::RegisterFaces(Element* elem, int* fattr)
{
   GeomInfo& gi = GI[(int) elem->geom];
   for (int i = 0; i < gi.nf; i++)
   {
      Face* face = GetFace(elem, i);
      face->RegisterElement(elem);
      if (fattr) { face->attribute = fattr[i]; }
   }
}

NCMesh::Element* NCMesh::Face::GetSingleElement() const
{
   if (elem[0])
   {
      MFEM_ASSERT(!elem[1], "not a single element face.");
      return elem[0];
   }
   else
   {
      MFEM_ASSERT(elem[1], "no elements in face.");
      return elem[1];
   }
}

NCMesh::Node* NCMesh::PeekAltParents(Node* v1, Node* v2)
{
   Node* mid = nodes.Peek(v1, v2);
   if (!mid) // TODO: && !Iso ?
   {
      // In rare cases, a mid-face node exists under alternate parents w1, w2
      // (see picture) instead of the requested parents v1, v2. This is an
      // inconsistent situation that may exist temporarily as a result of
      // "nodes.Reparent" while doing anisotropic splits, before forced
      // refinements are all processed. This function attempts to retrieve such
      // a node. An extra twist is that w1 and w2 may themselves need to be
      // obtained using this very function.
      //
      //                v1->p1      v1       v1->p2
      //                      *------*------*
      //                      |      |      |
      //                      |      |mid   |
      //                   w1 *------*------* w2
      //                      |      |      |
      //                      |      |      |
      //                      *------*------*
      //                v2->p1      v2       v2->p2
      //
      // NOTE: this function would not be needed if the elements remembered
      // pointers to their edge nodes. We have however opted to save memory
      // at the cost of this computation, which is only necessary when forced
      // refinements are being done.

      if ((v1->p1 != v1->p2) && (v2->p1 != v2->p2)) // non-top-level nodes?
      {
         Node *v1p1 = nodes.Peek(v1->p1), *v1p2 = nodes.Peek(v1->p2);
         Node *v2p1 = nodes.Peek(v2->p1), *v2p2 = nodes.Peek(v2->p2);

         Node* w1 = PeekAltParents(v1p1, v2p1);
         Node* w2 = w1 ? PeekAltParents(v1p2, v2p2) : NULL /* optimization */;

         if (!w1 || !w2) // one more try may be needed as p1, p2 are unordered
         {
            w1 = PeekAltParents(v1p1, v2p2);
            w2 = w1 ? PeekAltParents(v1p2, v2p1) : NULL /* optimization */;
         }

         if (w1 && w2) // got both alternate parents?
         {
            mid = nodes.Peek(w1, w2);
         }
      }
   }
   return mid;
}


//// Refinement & Derefinement /////////////////////////////////////////////////

NCMesh::Element::Element(int geom, int attr)
   : geom(geom), ref_type(0), flag(0), index(-1), rank(0), attribute(attr)
   , parent(NULL)
{
   memset(node, 0, sizeof(node));

   // NOTE: in 2D the 8-element node/child arrays are not optimal, however,
   // testing shows we would only save 17% of the total NCMesh memory if
   // 4-element arrays were used (e.g. through templates); we thus prefer to
   // keep the code as simple as possible.
}

NCMesh::Element*
NCMesh::NewHexahedron(Node* n0, Node* n1, Node* n2, Node* n3,
                      Node* n4, Node* n5, Node* n6, Node* n7,
                      int attr,
                      int fattr0, int fattr1, int fattr2,
                      int fattr3, int fattr4, int fattr5)
{
   // create new unrefined element, initialize nodes
   Element* e = new Element(Geometry::CUBE, attr);
   e->node[0] = n0, e->node[1] = n1, e->node[2] = n2, e->node[3] = n3;
   e->node[4] = n4, e->node[5] = n5, e->node[6] = n6, e->node[7] = n7;

   // get face nodes and assign face attributes
   Face* f[6];
   for (int i = 0; i < gi_hex.nf; i++)
   {
      const int* fv = gi_hex.faces[i];
      f[i] = faces.Get(e->node[fv[0]], e->node[fv[1]],
                       e->node[fv[2]], e->node[fv[3]]);
   }

   f[0]->attribute = fattr0,  f[1]->attribute = fattr1;
   f[2]->attribute = fattr2,  f[3]->attribute = fattr3;
   f[4]->attribute = fattr4,  f[5]->attribute = fattr5;

   return e;
}

NCMesh::Element*
NCMesh::NewQuadrilateral(Node* n0, Node* n1, Node* n2, Node* n3,
                         int attr,
                         int eattr0, int eattr1, int eattr2, int eattr3)
{
   // create new unrefined element, initialize nodes
   Element* e = new Element(Geometry::SQUARE, attr);
   e->node[0] = n0, e->node[1] = n1, e->node[2] = n2, e->node[3] = n3;

   // get edge nodes and assign edge attributes
   Edge* edge[4];
   for (int i = 0; i < gi_quad.ne; i++)
   {
      const int* ev = gi_quad.edges[i];
      Node* node = nodes.Get(e->node[ev[0]], e->node[ev[1]]);
      if (!node->edge) { node->edge = new Edge; }
      edge[i] = node->edge;
   }

   edge[0]->attribute = eattr0;
   edge[1]->attribute = eattr1;
   edge[2]->attribute = eattr2;
   edge[3]->attribute = eattr3;

   return e;
}

NCMesh::Element*
NCMesh::NewTriangle(Node* n0, Node* n1, Node* n2,
                    int attr, int eattr0, int eattr1, int eattr2)
{
   // create new unrefined element, initialize nodes
   Element* e = new Element(Geometry::TRIANGLE, attr);
   e->node[0] = n0, e->node[1] = n1, e->node[2] = n2;

   // get edge nodes and assign edge attributes
   Edge* edge[3];
   for (int i = 0; i < gi_tri.ne; i++)
   {
      const int* ev = gi_tri.edges[i];
      Node* node = nodes.Get(e->node[ev[0]], e->node[ev[1]]);
      if (!node->edge) { node->edge = new Edge; }
      edge[i] = node->edge;
   }

   edge[0]->attribute = eattr0;
   edge[1]->attribute = eattr1;
   edge[2]->attribute = eattr2;

   return e;
}

NCMesh::Vertex* NCMesh::NewVertex(Node* v1, Node* v2)
{
   MFEM_ASSERT(v1->vertex && v2->vertex, "missing parent vertices.");

   // get the midpoint between v1 and v2
   Vertex* v = new Vertex;
   for (int i = 0; i < 3; i++)
   {
      v->pos[i] = (v1->vertex->pos[i] + v2->vertex->pos[i]) * 0.5;
   }

   return v;
}

NCMesh::Node* NCMesh::GetMidEdgeVertex(Node* v1, Node* v2)
{
   // in 3D we must be careful about getting the mid-edge node
   Node* mid = PeekAltParents(v1, v2);
   if (!mid) { mid = nodes.Get(v1, v2); }
   if (!mid->vertex) { mid->vertex = NewVertex(v1, v2); }
   return mid;
}

NCMesh::Node* NCMesh::GetMidEdgeVertexSimple(Node* v1, Node* v2)
{
   // simple version for 2D cases
   Node* mid = nodes.Get(v1, v2);
   if (!mid->vertex) { mid->vertex = NewVertex(v1, v2); }
   return mid;
}

NCMesh::Node*
NCMesh::GetMidFaceVertex(Node* e1, Node* e2, Node* e3, Node* e4)
{
   // mid-face node can be created either from (e1, e3) or from (e2, e4)
   Node* midf = nodes.Peek(e1, e3);
   if (midf)
   {
      if (!midf->vertex) { midf->vertex = NewVertex(e1, e3); }
      return midf;
   }
   else
   {
      midf = nodes.Get(e2, e4);
      if (!midf->vertex) { midf->vertex = NewVertex(e2, e4); }
      return midf;
   }
}

//
inline bool NCMesh::NodeSetX1(Node* node, Node** n)
{ return node == n[0] || node == n[3] || node == n[4] || node == n[7]; }

inline bool NCMesh::NodeSetX2(Node* node, Node** n)
{ return node == n[1] || node == n[2] || node == n[5] || node == n[6]; }

inline bool NCMesh::NodeSetY1(Node* node, Node** n)
{ return node == n[0] || node == n[1] || node == n[4] || node == n[5]; }

inline bool NCMesh::NodeSetY2(Node* node, Node** n)
{ return node == n[2] || node == n[3] || node == n[6] || node == n[7]; }

inline bool NCMesh::NodeSetZ1(Node* node, Node** n)
{ return node == n[0] || node == n[1] || node == n[2] || node == n[3]; }

inline bool NCMesh::NodeSetZ2(Node* node, Node** n)
{ return node == n[4] || node == n[5] || node == n[6] || node == n[7]; }


void NCMesh::ForceRefinement(Node* v1, Node* v2, Node* v3, Node* v4)
{
   // get the element this face belongs to
   Face* face = faces.Peek(v1, v2, v3, v4);
   if (!face) { return; }

   Element* elem = face->GetSingleElement();
   MFEM_ASSERT(!elem->ref_type, "element already refined.");

   Node** nodes = elem->node;

   // schedule the right split depending on face orientation
   if ((NodeSetX1(v1, nodes) && NodeSetX2(v2, nodes)) ||
       (NodeSetX1(v2, nodes) && NodeSetX2(v1, nodes)))
   {
      ref_stack.Append(ElemRefType(elem, 1)); // X split
   }
   else if ((NodeSetY1(v1, nodes) && NodeSetY2(v2, nodes)) ||
            (NodeSetY1(v2, nodes) && NodeSetY2(v1, nodes)))
   {
      ref_stack.Append(ElemRefType(elem, 2)); // Y split
   }
   else if ((NodeSetZ1(v1, nodes) && NodeSetZ2(v2, nodes)) ||
            (NodeSetZ1(v2, nodes) && NodeSetZ2(v1, nodes)))
   {
      ref_stack.Append(ElemRefType(elem, 4)); // Z split
   }
   else
   {
      MFEM_ABORT("inconsistent element/face structure.");
   }
}


void NCMesh::CheckAnisoFace(Node* v1, Node* v2, Node* v3, Node* v4,
                            Node* mid12, Node* mid34, int level)
{
   // When a face is getting split anisotropically (without loss of generality
   // we assume a "vertical" split here, see picture), it is important to make
   // sure that the mid-face vertex (midf) has mid34 and mid12 as parents.
   // This is necessary for the face traversal algorithm and at places like
   // Refine() that assume the mid-edge nodes to be accessible through the right
   // parents. However, midf may already exist under the parents mid41 and
   // mid23. In that case we need to "reparent" midf, i.e., reinsert it to the
   // hash-table under the correct parents. This doesn't affect other nodes as
   // all IDs stay the same, only the face refinement "tree" is affected.
   //
   //                      v4      mid34      v3
   //                        *------*------*
   //                        |      |      |
   //                        |      |midf  |
   //                  mid41 *- - - *- - - * mid23
   //                        |      |      |
   //                        |      |      |
   //                        *------*------*
   //                     v1      mid12      v2
   //
   // This function is recursive, because the above applies to any node along
   // the middle vertical edge. The function calls itself again for the bottom
   // and upper half of the above picture.

   Node* mid23 = nodes.Peek(v2, v3);
   Node* mid41 = nodes.Peek(v4, v1);
   if (mid23 && mid41)
   {
      Node* midf = nodes.Peek(mid23, mid41);
      if (midf)
      {
         nodes.Reparent(midf, mid12->id, mid34->id);

         CheckAnisoFace(v1, v2, mid23, mid41, mid12, midf, level+1);
         CheckAnisoFace(mid41, mid23, v3, v4, midf, mid34, level+1);
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
      ForceRefinement(v1, v2, v3, v4);
   }
}

void NCMesh::CheckIsoFace(Node* v1, Node* v2, Node* v3, Node* v4,
                          Node* e1, Node* e2, Node* e3, Node* e4, Node* midf)
{
   if (!Iso)
   {
      /* If anisotropic refinements are present in the mesh, we need to check
         isotropically split faces as well, see second comment in
         CheckAnisoFace above. */

      CheckAnisoFace(v1, v2, e2, e4, e1, midf);
      CheckAnisoFace(e4, e2, v3, v4, midf, e3);
      CheckAnisoFace(v4, v1, e1, e3, e4, midf);
      CheckAnisoFace(e3, e1, v2, v3, midf, e2);
   }
}


void NCMesh::RefineElement(Element* elem, char ref_type)
{
   if (!ref_type) { return; }

   // handle elements that may have been (force-) refined already
   if (elem->ref_type)
   {
      char remaining = ref_type & ~elem->ref_type;

      // do the remaining splits on the children
      for (int i = 0; i < 8; i++)
      {
         if (elem->child[i]) { RefineElement(elem->child[i], remaining); }
      }
      return;
   }

   Node** no = elem->node;
   int attr = elem->attribute;

   Element* child[8];
   memset(child, 0, sizeof(child));

   // create child elements
   if (elem->geom == Geometry::CUBE)
   {
      // get parent's face attributes
      int fa[6];
      for (int i = 0; i < gi_hex.nf; i++)
      {
         const int* fv = gi_hex.faces[i];
         Face* face = faces.Peek(no[fv[0]], no[fv[1]], no[fv[2]], no[fv[3]]);
         fa[i] = face->attribute;
      }

      // Vertex numbering is assumed to be as follows:
      //
      //       7              6
      //        +------------+                Faces: 0 bottom
      //       /|           /|                       1 front
      //    4 / |        5 / |                       2 right
      //     +------------+  |                       3 back
      //     |  |         |  |                       4 left
      //     |  +---------|--+                       5 top
      //     | / 3        | / 2       Z Y
      //     |/           |/          |/
      //     +------------+           *--X
      //    0              1

      if (ref_type == 1) // split along X axis
      {
         Node* mid01 = GetMidEdgeVertex(no[0], no[1]);
         Node* mid23 = GetMidEdgeVertex(no[2], no[3]);
         Node* mid67 = GetMidEdgeVertex(no[6], no[7]);
         Node* mid45 = GetMidEdgeVertex(no[4], no[5]);

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
         Node* mid12 = GetMidEdgeVertex(no[1], no[2]);
         Node* mid30 = GetMidEdgeVertex(no[3], no[0]);
         Node* mid56 = GetMidEdgeVertex(no[5], no[6]);
         Node* mid74 = GetMidEdgeVertex(no[7], no[4]);

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
         Node* mid04 = GetMidEdgeVertex(no[0], no[4]);
         Node* mid15 = GetMidEdgeVertex(no[1], no[5]);
         Node* mid26 = GetMidEdgeVertex(no[2], no[6]);
         Node* mid37 = GetMidEdgeVertex(no[3], no[7]);

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
         Node* mid01 = GetMidEdgeVertex(no[0], no[1]);
         Node* mid12 = GetMidEdgeVertex(no[1], no[2]);
         Node* mid23 = GetMidEdgeVertex(no[2], no[3]);
         Node* mid30 = GetMidEdgeVertex(no[3], no[0]);

         Node* mid45 = GetMidEdgeVertex(no[4], no[5]);
         Node* mid56 = GetMidEdgeVertex(no[5], no[6]);
         Node* mid67 = GetMidEdgeVertex(no[6], no[7]);
         Node* mid74 = GetMidEdgeVertex(no[7], no[4]);

         Node* midf0 = GetMidFaceVertex(mid23, mid12, mid01, mid30);
         Node* midf5 = GetMidFaceVertex(mid45, mid56, mid67, mid74);

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
         Node* mid01 = GetMidEdgeVertex(no[0], no[1]);
         Node* mid23 = GetMidEdgeVertex(no[2], no[3]);
         Node* mid45 = GetMidEdgeVertex(no[4], no[5]);
         Node* mid67 = GetMidEdgeVertex(no[6], no[7]);

         Node* mid04 = GetMidEdgeVertex(no[0], no[4]);
         Node* mid15 = GetMidEdgeVertex(no[1], no[5]);
         Node* mid26 = GetMidEdgeVertex(no[2], no[6]);
         Node* mid37 = GetMidEdgeVertex(no[3], no[7]);

         Node* midf1 = GetMidFaceVertex(mid01, mid15, mid45, mid04);
         Node* midf3 = GetMidFaceVertex(mid23, mid37, mid67, mid26);

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
         Node* mid12 = GetMidEdgeVertex(no[1], no[2]);
         Node* mid30 = GetMidEdgeVertex(no[3], no[0]);
         Node* mid56 = GetMidEdgeVertex(no[5], no[6]);
         Node* mid74 = GetMidEdgeVertex(no[7], no[4]);

         Node* mid04 = GetMidEdgeVertex(no[0], no[4]);
         Node* mid15 = GetMidEdgeVertex(no[1], no[5]);
         Node* mid26 = GetMidEdgeVertex(no[2], no[6]);
         Node* mid37 = GetMidEdgeVertex(no[3], no[7]);

         Node* midf2 = GetMidFaceVertex(mid12, mid26, mid56, mid15);
         Node* midf4 = GetMidFaceVertex(mid30, mid04, mid74, mid37);

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
         Node* mid01 = GetMidEdgeVertex(no[0], no[1]);
         Node* mid12 = GetMidEdgeVertex(no[1], no[2]);
         Node* mid23 = GetMidEdgeVertex(no[2], no[3]);
         Node* mid30 = GetMidEdgeVertex(no[3], no[0]);

         Node* mid45 = GetMidEdgeVertex(no[4], no[5]);
         Node* mid56 = GetMidEdgeVertex(no[5], no[6]);
         Node* mid67 = GetMidEdgeVertex(no[6], no[7]);
         Node* mid74 = GetMidEdgeVertex(no[7], no[4]);

         Node* mid04 = GetMidEdgeVertex(no[0], no[4]);
         Node* mid15 = GetMidEdgeVertex(no[1], no[5]);
         Node* mid26 = GetMidEdgeVertex(no[2], no[6]);
         Node* mid37 = GetMidEdgeVertex(no[3], no[7]);

         Node* midf0 = GetMidFaceVertex(mid23, mid12, mid01, mid30);
         Node* midf1 = GetMidFaceVertex(mid01, mid15, mid45, mid04);
         Node* midf2 = GetMidFaceVertex(mid12, mid26, mid56, mid15);
         Node* midf3 = GetMidFaceVertex(mid23, mid37, mid67, mid26);
         Node* midf4 = GetMidFaceVertex(mid30, mid04, mid74, mid37);
         Node* midf5 = GetMidFaceVertex(mid45, mid56, mid67, mid74);

         Node* midel = GetMidEdgeVertex(midf1, midf3);

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
   else if (elem->geom == Geometry::SQUARE)
   {
      // get parent's edge attributes
      int ea0 = nodes.Peek(no[0], no[1])->edge->attribute;
      int ea1 = nodes.Peek(no[1], no[2])->edge->attribute;
      int ea2 = nodes.Peek(no[2], no[3])->edge->attribute;
      int ea3 = nodes.Peek(no[3], no[0])->edge->attribute;

      ref_type &= ~4; // ignore Z bit

      if (ref_type == 1) // X split
      {
         Node* mid01 = GetMidEdgeVertexSimple(no[0], no[1]);
         Node* mid23 = GetMidEdgeVertexSimple(no[2], no[3]);

         child[0] = NewQuadrilateral(no[0], mid01, mid23, no[3],
                                     attr, ea0, -1, ea2, ea3);

         child[1] = NewQuadrilateral(mid01, no[1], no[2], mid23,
                                     attr, ea0, ea1, ea2, -1);
      }
      else if (ref_type == 2) // Y split
      {
         Node* mid12 = GetMidEdgeVertexSimple(no[1], no[2]);
         Node* mid30 = GetMidEdgeVertexSimple(no[3], no[0]);

         child[0] = NewQuadrilateral(no[0], no[1], mid12, mid30,
                                     attr, ea0, ea1, -1, ea3);

         child[1] = NewQuadrilateral(mid30, mid12, no[2], no[3],
                                     attr, -1, ea1, ea2, ea3);
      }
      else if (ref_type == 3) // iso split
      {
         Node* mid01 = GetMidEdgeVertexSimple(no[0], no[1]);
         Node* mid12 = GetMidEdgeVertexSimple(no[1], no[2]);
         Node* mid23 = GetMidEdgeVertexSimple(no[2], no[3]);
         Node* mid30 = GetMidEdgeVertexSimple(no[3], no[0]);

         Node* midel = GetMidEdgeVertexSimple(mid01, mid23);

         child[0] = NewQuadrilateral(no[0], mid01, midel, mid30,
                                     attr, ea0, -1, -1, ea3);

         child[1] = NewQuadrilateral(mid01, no[1], mid12, midel,
                                     attr, ea0, ea1, -1, -1);

         child[2] = NewQuadrilateral(midel, mid12, no[2], mid23,
                                     attr, -1, ea1, ea2, -1);

         child[3] = NewQuadrilateral(mid30, midel, mid23, no[3],
                                     attr, -1, -1, ea2, ea3);
      }
      else
      {
         MFEM_ABORT("Invalid refinement type.");
      }

      if (ref_type != 3) { Iso = false; }
   }
   else if (elem->geom == Geometry::TRIANGLE)
   {
      // get parent's edge attributes
      int ea0 = nodes.Peek(no[0], no[1])->edge->attribute;
      int ea1 = nodes.Peek(no[1], no[2])->edge->attribute;
      int ea2 = nodes.Peek(no[2], no[0])->edge->attribute;

      ref_type = 3; // for consistence

      // isotropic split - the only ref_type available for triangles
      Node* mid01 = GetMidEdgeVertexSimple(no[0], no[1]);
      Node* mid12 = GetMidEdgeVertexSimple(no[1], no[2]);
      Node* mid20 = GetMidEdgeVertexSimple(no[2], no[0]);

      child[0] = NewTriangle(no[0], mid01, mid20, attr, ea0, -1, ea2);
      child[1] = NewTriangle(mid01, no[1], mid12, attr, ea0, ea1, -1);
      child[2] = NewTriangle(mid20, mid12, no[2], attr, -1, ea1, ea2);
      child[3] = NewTriangle(mid01, mid12, mid20, attr, -1, -1, -1);
   }
   else
   {
      MFEM_ABORT("Unsupported element geometry.");
   }

   // start using the nodes of the children, create edges & faces
   for (int i = 0; i < 8; i++)
   {
      if (child[i]) { RefElementNodes(child[i]); }
   }

   // sign off of all nodes of the parent, clean up unused nodes
   UnrefElementNodes(elem);

   // register the children in their faces once the parent is out of the way
   for (int i = 0; i < 8; i++)
   {
      if (child[i]) { RegisterFaces(child[i]); }
   }

   // make the children inherit our rank, set the parent pointer
   for (int i = 0; i < 8; i++)
   {
      if (child[i])
      {
         child[i]->rank = elem->rank;
         child[i]->parent = elem;
      }
   }

   // finish the refinement
   elem->ref_type = ref_type;
   memcpy(elem->child, child, sizeof(elem->child));
}


void NCMesh::Refine(const Array<Refinement>& refinements)
{
   // push all refinements on the stack in reverse order
   for (int i = refinements.Size()-1; i >= 0; i--)
   {
      const Refinement& ref = refinements[i];
      ref_stack.Append(ElemRefType(leaf_elements[ref.index], ref.ref_type));
   }

   // keep refining as long as the stack contains something
   int nforced = 0;
   while (ref_stack.Size())
   {
      ElemRefType ref = ref_stack.Last();
      ref_stack.DeleteLast();

      int size = ref_stack.Size();
      RefineElement(ref.elem, ref.ref_type);
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

#ifdef MFEM_DEBUG
   std::cout << "Refined " << refinements.Size() << " + " << nforced
             << " elements" << std::endl;
#endif

   Update();
}


//// Derefinement //////////////////////////////////////////////////////////////

void NCMesh::DerefineElement(Element* elem)
{
   if (!elem->ref_type) { return; }

   Element* child[8];
   std::memcpy(child, elem->child, sizeof(child));

   // first make sure that all children are leaves, derefine them if not
   for (int i = 0; i < 8; i++)
   {
      if (child[i] && child[i]->ref_type)
      {
         DerefineElement(child[i]);
      }
   }

   // retrieve original corner nodes and face attributes from the children
   int fa[6];
   if (elem->geom == Geometry::CUBE)
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
         elem->node[i] = child[table[elem->ref_type - 1][i]]->node[i];
      }
      for (int i = 0; i < 6; i++)
      {
         Element* ch = child[table[elem->ref_type - 1][i + 8]];
         const int* fv = gi_hex.faces[i];
         fa[i] = faces.Peek(ch->node[fv[0]], ch->node[fv[1]],
                            ch->node[fv[2]], ch->node[fv[3]])->attribute;
      }
   }
   else if (elem->geom == Geometry::SQUARE)
   {
      const int table[3][4 + 4] =
      {
         { 0, 1, 1, 0, /**/ 1, 1, 0, 0 }, // 1 - X
         { 0, 0, 1, 1, /**/ 0, 0, 1, 1 }, // 2 - Y
         { 0, 1, 2, 3, /**/ 1, 1, 3, 3 }  // 3 - iso
      };
      for (int i = 0; i < 4; i++)
      {
         elem->node[i] = child[table[elem->ref_type - 1][i]]->node[i];
      }
      for (int i = 0; i < 4; i++)
      {
         Element* ch = child[table[elem->ref_type - 1][i + 4]];
         const int* ev = gi_quad.edges[i];
         fa[i] = nodes.Peek(ch->node[ev[0]], ch->node[ev[1]])->edge->attribute;
      }
   }
   else if (elem->geom == Geometry::TRIANGLE)
   {
      for (int i = 0; i < 3; i++)
      {
         Element* ch = child[i];
         elem->node[i] = child[i]->node[i];
         const int* ev = gi_tri.edges[i];
         fa[i] = nodes.Peek(ch->node[ev[0]], ch->node[ev[1]])->edge->attribute;
      }
   }
   else
   {
      MFEM_ABORT("Unsupported element geometry.");
   }

   // sign in to all nodes again
   RefElementNodes(elem);

   // delete children, determine rank
   elem->rank = INT_MAX;
   for (int i = 0; i < 8; i++)
   {
      if (child[i])
      {
         elem->rank = std::min(elem->rank, child[i]->rank);
         DeleteHierarchy(child[i]);
      }
   }

   RegisterFaces(elem, fa);

   // set edge attributes (2D)
   // TODO: Edge::attribute should be removed
   if (Dim < 3)
   {
      Node** node = elem->node;
      GeomInfo& gi = GI[(int) elem->geom];
      for (int i = 0; i < gi.ne; i++)
      {
         const int* ev = gi.edges[i];
         nodes.Peek(node[ev[0]], node[ev[1]])->edge->attribute = fa[i];
      }
   }

   elem->ref_type = 0;
}


void NCMesh::CollectDerefinements(Element* elem, Array<Connection> &list)
{
   if (!elem->ref_type) { return; }

   int total = 0, ref = 0, ghost = 0;
   for (int i = 0; i < 8; i++)
   {
      Element* ch = elem->child[i];
      if (ch)
      {
         total++;
         if (ch->ref_type) { ref++; break; }
         if (IsGhost(ch)) { ghost++; }
      }
   }

   if (!ref && ghost < total)
   {
      // can be derefined, add to list
      int next_row = list.Size() ? (list.Last().from + 1) : 0;
      for (int i = 0; i < 8; i++)
      {
         Element* ch = elem->child[i];
         if (ch) { list.Append(Connection(next_row, ch->index)); }
      }
   }
   else
   {
      for (int i = 0; i < 8; i++)
      {
         Element* ch = elem->child[i];
         if (ch) { CollectDerefinements(ch, list); }
      }
   }
}

const Table& NCMesh::GetDerefinementTable()
{
   Array<Connection> list;
   list.Reserve(leaf_elements.Size());

   for (int i = 0; i < root_elements.Size(); i++)
   {
      CollectDerefinements(root_elements[i], list);
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
      Element* parent = leaf_elements[fine[0]]->parent;

      int ok = 1;
      for (int j = 0; j < size; j++)
      {
         int splits[3];
         CountSplits(leaf_elements[fine[j]], splits);

         for (int k = 0; k < Dim; k++)
         {
            if ((parent->ref_type & (1 << k)) &&
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

   Array<Element*> coarse;
   leaf_elements.Copy(coarse);

   // perform the derefinements
   for (int i = 0; i < derefs.Size(); i++)
   {
      int row = derefs[i];
      MFEM_VERIFY(row >= 0 && row < derefinements.Size(),
                  "invalid derefinement number.");

      const int* fine = derefinements.GetRow(row);
      Element* parent = leaf_elements[fine[0]]->parent;

      // record the relation of the fine elements to their parent
      SetDerefMatrixCodes(parent, coarse);

      DerefineElement(parent);
   }

   // update leaf_elements, Element::index etc.
   Update();

   // link old fine elements to the new coarse elements
   for (int i = 0; i < coarse.Size(); i++)
   {
      transforms.embeddings[i].parent = coarse[i]->index;
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

void NCMesh::SetDerefMatrixCodes(Element* parent, Array<Element*> &coarse)
{
   // encode the ref_type and child number for GetDerefinementTransforms()
   for (int i = 0; i < 8; i++)
   {
      Element* ch = parent->child[i];
      if (ch && ch->index >= 0)
      {
         int code = (parent->ref_type << 3) + i;
         transforms.embeddings[ch->index].matrix = code;
         coarse[ch->index] = parent;
      }
   }
}


//// Mesh Interface ////////////////////////////////////////////////////////////

void NCMesh::UpdateVertices()
{
   // (overridden in ParNCMesh to assign special indices to ghost vertices)
   int num_vert = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->vertex) { it->vertex->index = num_vert++; }
   }

   vertex_nodeId.SetSize(num_vert);

   num_vert = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->vertex) { vertex_nodeId[num_vert++] = it->id; }
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

void NCMesh::CollectLeafElements(Element* elem, int state)
{
   if (!elem->ref_type)
   {
      if (elem->rank >= 0) // skip elements beyond ghost layer in parallel
      {
         leaf_elements.Append(elem);
      }
   }
   else
   {
      if (elem->geom == Geometry::SQUARE && elem->ref_type == 3)
      {
         for (int i = 0; i < 4; i++)
         {
            int ch = quad_hilbert_child_order[state][i];
            int st = quad_hilbert_child_state[state][i];
            CollectLeafElements(elem->child[ch], st);
         }
      }
      else if (elem->geom == Geometry::CUBE && elem->ref_type == 7)
      {
         for (int i = 0; i < 8; i++)
         {
            int ch = hex_hilbert_child_order[state][i];
            int st = hex_hilbert_child_state[state][i];
            CollectLeafElements(elem->child[ch], st);
         }
      }
      else
      {
         for (int i = 0; i < 8; i++)
         {
            if (elem->child[i]) { CollectLeafElements(elem->child[i], state); }
         }
      }
   }
   elem->index = -1;
}

void NCMesh::UpdateLeafElements()
{
   // collect leaf elements from all roots
   leaf_elements.SetSize(0);
   for (int i = 0; i < root_elements.Size(); i++)
   {
      CollectLeafElements(root_elements[i], 0);
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
      leaf_elements[i]->index = i;
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

void NCMesh::GetMeshComponents(Array<mfem::Vertex>& vertices,
                               Array<mfem::Element*>& elements,
                               Array<mfem::Element*>& boundary) const
{
   // copy vertex coordinates
   vertices.SetSize(vertex_nodeId.Size());
   for (int i = 0; i < vertices.Size(); i++)
   {
      Node* node = nodes.Peek(vertex_nodeId[i]);
      vertices[i].SetCoords(node->vertex->pos);
   }

   elements.SetSize(leaf_elements.Size() - GetNumGhosts());
   elements.SetSize(0);

   boundary.SetSize(0);

   // create an mfem::Element for each leaf Element
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* nc_elem = leaf_elements[i];
      if (IsGhost(nc_elem)) { continue; } // ParNCMesh

      Node** node = nc_elem->node;
      GeomInfo& gi = GI[(int) nc_elem->geom];

      mfem::Element* elem = NewMeshElement(nc_elem->geom);
      elements.Append(elem);

      elem->SetAttribute(nc_elem->attribute);
      for (int j = 0; j < gi.nv; j++)
      {
         elem->GetVertices()[j] = node[j]->vertex->index;
      }

      // create boundary elements
      if (nc_elem->geom == Geometry::CUBE)
      {
         for (int k = 0; k < gi.nf; k++)
         {
            const int* fv = gi.faces[k];
            Face* face = faces.Peek(node[fv[0]], node[fv[1]],
                                    node[fv[2]], node[fv[3]]);
            if (face->Boundary())
            {
               Quadrilateral* quad = new Quadrilateral;
               quad->SetAttribute(face->attribute);
               for (int j = 0; j < 4; j++)
               {
                  quad->GetVertices()[j] = node[fv[j]]->vertex->index;
               }
               boundary.Append(quad);
            }
         }
      }
      else // quad & triangle boundary elements
      {
         for (int k = 0; k < gi.ne; k++)
         {
            const int* ev = gi.edges[k];
            Edge* edge = nodes.Peek(node[ev[0]], node[ev[1]])->edge;
            if (edge->Boundary())
            {
               Segment* segment = new Segment;
               segment->SetAttribute(edge->attribute);
               for (int j = 0; j < 2; j++)
               {
                  segment->GetVertices()[j] = node[ev[j]]->vertex->index;
               }
               boundary.Append(segment);
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
      Node* node = nodes.Peek(vertex_nodeId[ev[0]], vertex_nodeId[ev[1]]);

      MFEM_ASSERT(node && node->edge, "edge not found.");
      node->edge->index = i;
   }

   // get face enumeration from the Mesh
   for (int i = 0; i < mesh->GetNumFaces(); i++)
   {
      const int* fv = mesh->GetFace(i)->GetVertices();
      Face* face;
      if (Dim == 3)
      {
         MFEM_ASSERT(mesh->GetFace(i)->GetNVertices() == 4, "");
         face = faces.Peek(vertex_nodeId[fv[0]], vertex_nodeId[fv[1]],
                           vertex_nodeId[fv[2]], vertex_nodeId[fv[3]]);
      }
      else
      {
         MFEM_ASSERT(mesh->GetFace(i)->GetNVertices() == 2, "");
         int n0 = vertex_nodeId[fv[0]], n1 = vertex_nodeId[fv[1]];
         face = faces.Peek(n0, n0, n1, n1); // look up degenerate face

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

int NCMesh::FaceSplitType(Node* v1, Node* v2, Node* v3, Node* v4,
                          Node* mid[4]) const
{
   MFEM_ASSERT(Dim >= 3, "");

   // find edge nodes
   Node* e1 = nodes.Peek(v1, v2);
   Node* e2 = nodes.Peek(v2, v3);
   Node* e3 = e1 ? nodes.Peek(v3, v4) : NULL;  // TODO: e1 && e1->vertex ?
   Node* e4 = e2 ? nodes.Peek(v4, v1) : NULL;

   // optional: return the mid-edge nodes if requested
   if (mid) { mid[0] = e1, mid[1] = e2, mid[2] = e3, mid[3] = e4; }

   // try to get a mid-face node, either by (e1, e3) or by (e2, e4)
   Node *midf1 = NULL, *midf2 = NULL;
   if (e1 && e3) { midf1 = nodes.Peek(e1, e3); }
   if (e2 && e4) { midf2 = nodes.Peek(e2, e4); }

   // only one way to access the mid-face node must always exist
   MFEM_ASSERT(!(midf1 && midf2), "incorrectly split face!");

   if (!midf1 && !midf2) { return 0; }  // face not split

   if (midf1) { return 1; }  // face split "vertically"
   else { return 2; }  // face split "horizontally"
}

int NCMesh::find_node(Element* elem, Node* node)
{
   for (int i = 0; i < 8; i++)
      if (elem->node[i] == node) { return i; }

   MFEM_ABORT("Node not found.");
   return -1;
}

int NCMesh::find_node(Element* elem, int node_id)
{
   for (int i = 0; i < 8; i++)
      if (elem->node[i]->id == node_id) { return i; }

   MFEM_ABORT("Node not found.");
   return -1;
}

int NCMesh::find_element_edge(Element* elem, int v0, int v1)
{
   MFEM_ASSERT(!elem->ref_type, "");
   GeomInfo &gi = GI[(int) elem->geom];
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      int n0 = elem->node[ev[0]]->id;
      int n1 = elem->node[ev[1]]->id;
      if ((n0 == v0 && n1 == v1) ||
          (n0 == v1 && n1 == v0)) { return i; }
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

int NCMesh::ReorderFacePointMat(Node* v0, Node* v1, Node* v2, Node* v3,
                                Element* elem, DenseMatrix& mat) const
{
   int master[4] =
   {
      find_node(elem, v0), find_node(elem, v1),
      find_node(elem, v2), find_node(elem, v3)
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

void NCMesh::TraverseFace(Node* v0, Node* v1, Node* v2, Node* v3,
                          const PointMatrix& pm, int level)
{
   if (level > 0)
   {
      // check if we made it to a face that is not split further
      Face* face = faces.Peek(v0, v1, v2, v3);
      if (face)
      {
         // we have a slave face, add it to the list
         Element* elem = face->GetSingleElement();
         face_list.slaves.push_back(Slave(face->index, elem, -1));
         DenseMatrix &mat = face_list.slaves.back().point_matrix;
         pm.GetMatrix(mat);

         // reorder the point matrix according to slave face orientation
         int local = ReorderFacePointMat(v0, v1, v2, v3, elem, mat);
         face_list.slaves.back().local = local;

         return;
      }
   }

   // we need to recurse deeper
   Node* mid[4];
   int split = FaceSplitType(v0, v1, v2, v3, mid);

   if (split == 1) // "X" split face
   {
      Point mid0(pm(0), pm(1)), mid2(pm(2), pm(3));

      TraverseFace(v0, mid[0], mid[2], v3,
                   PointMatrix(pm(0), mid0, mid2, pm(3)), level+1);

      TraverseFace(mid[0], v1, v2, mid[2],
                   PointMatrix(mid0, pm(1), pm(2), mid2), level+1);
   }
   else if (split == 2) // "Y" split face
   {
      Point mid1(pm(1), pm(2)), mid3(pm(3), pm(0));

      TraverseFace(v0, v1, mid[1], mid[3],
                   PointMatrix(pm(0), pm(1), mid1, mid3), level+1);

      TraverseFace(mid[3], mid[1], v2, v3,
                   PointMatrix(mid3, mid1, pm(2), pm(3)), level+1);
   }
}

void NCMesh::BuildFaceList()
{
   face_list.Clear();
   boundary_faces.SetSize(0);

   if (Dim < 3) { return; }

   // visit faces of leaf elements
   Array<char> processed_faces; // TODO: size
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      MFEM_ASSERT(!elem->ref_type, "not a leaf element.");

      GeomInfo& gi = GI[(int) elem->geom];
      for (int j = 0; j < gi.nf; j++)
      {
         // get nodes for this face
         Node* node[4];
         for (int k = 0; k < 4; k++)
         {
            node[k] = elem->node[gi.faces[j][k]];
         }

         Face* face = faces.Peek(node[0], node[1], node[2], node[3]);
         MFEM_ASSERT(face, "face not found!");

         // tell ParNCMesh about the face
         ElementSharesFace(elem, face);

         // have we already processed this face? skip if yes
         int index = face->index;
         if (index >= processed_faces.Size())
         {
            processed_faces.SetSize(index + 1, 0);
         }
         else if (processed_faces[index])
         {
            continue;
         }
         processed_faces[index] = 1;

         if (face->ref_count == 2)
         {
            // this is a conforming face, add it to the list
            face_list.conforming.push_back(MeshId(index, elem, j));
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
               face_list.masters.push_back(Master(index, elem, j, sb, se));

               // also, set the master index for the slaves
               for (int i = sb; i < se; i++)
               {
                  face_list.slaves[i].master = index;
               }
            }
         }

         if (face->Boundary()) { boundary_faces.Append(face); }
      }
   }
}

void NCMesh::TraverseEdge(Node* v0, Node* v1, double t0, double t1, int flags,
                          int level)
{
   Node* mid = nodes.Peek(v0, v1);
   if (!mid) { return; }

   if (mid->edge && level > 0)
   {
      // we have a slave edge, add it to the list
      edge_list.slaves.push_back(Slave(mid->edge->index, NULL, -1));
      Slave &sl = edge_list.slaves.back();

      sl.point_matrix.SetSize(1, 2);
      sl.point_matrix(0,0) = t0;
      sl.point_matrix(0,1) = t1;

      // handle slave edge orientation
      sl.edge_flags = flags;
      if (v0->vertex->index > v1->vertex->index) { sl.edge_flags |= 2; }

      // in 2D, get the element/local info from the degenerate face
      if (Dim == 2)
      {
         Face* face = faces.Peek(v0, v0, v1, v1);
         MFEM_ASSERT(face != NULL, "");
         sl.element = face->GetSingleElement();
         sl.local = find_element_edge(sl.element, v0->id, v1->id);
      }
   }

   // recurse deeper
   double tmid = (t0 + t1) / 2;
   TraverseEdge(v0, mid, t0, tmid, flags, level+1);
   TraverseEdge(mid, v1, tmid, t1, flags, level+1);
}

void NCMesh::BuildEdgeList()
{
   edge_list.Clear();
   boundary_edges.SetSize(0);

   // visit edges of leaf elements
   Array<char> processed_edges; // TODO: size
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      MFEM_ASSERT(!elem->ref_type, "not a leaf element.");

      GeomInfo& gi = GI[(int) elem->geom];
      for (int j = 0; j < gi.ne; j++)
      {
         // get nodes for this edge
         const int* ev = gi.edges[j];
         Node* node[2] = { elem->node[ev[0]], elem->node[ev[1]] };

         Node* edge = nodes.Peek(node[0], node[1]);
         MFEM_ASSERT(edge && edge->edge, "edge not found!");

         // tell ParNCMesh about the edge
         ElementSharesEdge(elem, edge->edge);

         // (2D only, store boundary edges)
         if (edge->edge->Boundary()) { boundary_edges.Append(edge); }

         // skip slave edges here, they will be reached from their masters
         if (GetEdgeMaster(edge)) { continue; }

         // have we already processed this edge? skip if yes
         int index = edge->edge->index;
         if (index >= processed_edges.Size())
         {
            processed_edges.SetSize(index + 1, 0);
         }
         else if (processed_edges[index])
         {
            continue;
         }
         processed_edges[index] = 1;

         // prepare edge interval for slave traversal, handle orientation
         double t0 = 0.0, t1 = 1.0;
         int flags = (node[0]->vertex->index > node[1]->vertex->index) ? 1 : 0;

         // try traversing the edge to find slave edges
         int sb = edge_list.slaves.size();
         TraverseEdge(node[0], node[1], t0, t1, flags, 0);

         int se = edge_list.slaves.size();
         if (sb < se)
         {
            // found slaves, this is a master face; add it to the list
            edge_list.masters.push_back(Master(index, elem, j, sb, se));

            // also, set the master index for the slaves
            for (int i = sb; i < se; i++)
            {
               edge_list.slaves[i].master = index;
            }
         }
         else
         {
            // no slaves, this is a conforming edge
            edge_list.conforming.push_back(MeshId(index, elem, j));
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

void NCMesh::CollectEdgeVertices(Node* v0, Node* v1, Array<int> &indices)
{
   Node* mid = nodes.Peek(v0, v1);
   if (mid && mid->vertex)
   {
      indices.Append(mid->vertex->index);

      CollectEdgeVertices(v0, mid, indices);
      CollectEdgeVertices(mid, v1, indices);
   }
}

void NCMesh::CollectFaceVertices(Node* v0, Node* v1, Node* v2, Node* v3,
                                 Array<int> &indices)
{
   Node* mid[4];
   switch (FaceSplitType(v0, v1, v2, v3, mid))
   {
      case 1:
         indices.Append(mid[0]->vertex->index);
         indices.Append(mid[2]->vertex->index);

         CollectFaceVertices(v0, mid[0], mid[2], v3, indices);
         CollectFaceVertices(mid[0], v1, v2, mid[2], indices);
         break;

      case 2:
         indices.Append(mid[1]->vertex->index);
         indices.Append(mid[3]->vertex->index);

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
      Element* elem = leaf_elements[i];
      MFEM_ASSERT(!elem->ref_type, "not a leaf element.");

      GeomInfo& gi = GI[(int) elem->geom];
      Node** node = elem->node;

      indices.SetSize(0);
      for (int j = 0; j < gi.nv; j++)
      {
         indices.Append(node[j]->vertex->index);
      }

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
   num_vertices = element_vertex.Width();

   delete [] JJ;
}


void NCMesh::FindSetNeighbors(const Array<char> &elem_set,
                              Array<Element*> *neighbors,
                              Array<char> *neighbor_set)
{
   // If A is the element-to-vertex table (see 'element_vertex') listing all
   // vertices touching each element, including hanging vertices, then A*A^T is
   // the element-to-neighbor table. Multiplying the element set with A*A^T
   // gives the neighbor set. To save memory, this function only computes the
   // action of A*A^T, the product itself is not stored anywhere.

   // TODO: we should further optimize the 'element_vertex' table by not storing
   // the obvious corner vertices in it. The table would therefore be empty
   // for conforming meshes and very cheap for NC meshes.

   UpdateElementToVertexTable();

   int nleaves = leaf_elements.Size();
   MFEM_VERIFY(elem_set.Size() == nleaves, "");
   MFEM_ASSERT(element_vertex.Size() == nleaves, "");

   // step 1: vertices = A^T * elem_set, i.e, find all vertices touching the
   // element set

   Array<char> vertices(num_vertices);
   vertices = 0;

   for (int i = 0; i < nleaves; i++)
   {
      if (elem_set[i])
      {
         int *v = element_vertex.GetRow(i);
         int nv = element_vertex.RowSize(i);
         for (int j = 0; j < nv; j++)
         {
            vertices[v[j]] = 1;
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
         int *v = element_vertex.GetRow(i);
         int nv = element_vertex.RowSize(i);
         for (int j = 0; j < nv; j++)
         {
            if (vertices[v[j]])
            {
               if (neighbors) { neighbors->Append(leaf_elements[i]); }
               if (neighbor_set) { (*neighbor_set)[i] = 1; }
               break;
            }
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

void NCMesh::FindNeighbors(const Element* elem,
                           Array<Element*> &neighbors,
                           const Array<Element*> *search_set)
{
   UpdateElementToVertexTable();

   Array<int> vert, tmp;

   int *v1, nv1;
   if (!elem->ref_type)
   {
      v1 = element_vertex.GetRow(elem->index);
      nv1 = element_vertex.RowSize(elem->index);
   }
   else // support for non-leaf 'elem', collect vertices of all children
   {
      Array<const Element*> stack;
      stack.Reserve(32);
      stack.Append(elem);

      while (stack.Size())
      {
         const Element* e = stack.Last();
         stack.DeleteLast();
         if (!e->ref_type)
         {
            element_vertex.GetRow(e->index, tmp);
            vert.Append(tmp);
         }
         else
         {
            for (int i = 0; i < 8; i++)
            {
               if (e->child[i]) { stack.Append(e->child[i]); }
            }
         }
      }
      vert.Sort();
      vert.Unique();

      v1 = vert.GetData();
      nv1 = vert.Size();
   }

   if (!search_set) { search_set = &leaf_elements; }

   for (int i = 0; i < search_set->Size(); i++)
   {
      Element* testme = (*search_set)[i];
      if (testme != elem)
      {
         int *v2 = element_vertex.GetRow(testme->index);
         int nv2 = element_vertex.RowSize(testme->index);

         if (sorted_lists_intersect(v1, v2, nv1, nv2))
         {
            neighbors.Append(testme);
         }
      }
   }
}

void NCMesh::NeighborExpand(const Array<Element*> &elements,
                            Array<Element*> &expanded,
                            const Array<Element*> *search_set)
{
   UpdateElementToVertexTable();

   Array<char> vertices(num_vertices);
   vertices = 0;

   for (int i = 0; i < elements.Size(); i++)
   {
      int index = elements[i]->index;
      int *v = element_vertex.GetRow(index);
      int nv = element_vertex.RowSize(index);
      for (int j = 0; j < nv; j++)
      {
         vertices[v[j]] = 1;
      }
   }

   if (!search_set)
   {
      search_set = &leaf_elements;
   }

   expanded.SetSize(0);
   for (int i = 0; i < search_set->Size(); i++)
   {
      Element* testme = (*search_set)[i];
      int *v = element_vertex.GetRow(testme->index);
      int nv = element_vertex.RowSize(testme->index);
      for (int j = 0; j < nv; j++)
      {
         if (vertices[v[j]])
         {
            expanded.Append(testme);
            break;
         }
      }
   }
}

#ifdef MFEM_DEBUG
void NCMesh::DebugNeighbors(Array<char> &elem_set)
{
   Array<Element*> neighbors;
   FindSetNeighbors(elem_set, &neighbors);

   for (int i = 0; i < neighbors.Size(); i++)
   {
      elem_set[neighbors[i]->index] = 2;
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
         MFEM_ABORT("unsupported geometry."); throw;
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
      Element* e = leaf_elements[i];
      if (!IsGhost(e)) { coarse_elements.Append(e); }
   }

   transforms.embeddings.DeleteAll();
}

void NCMesh::TraverseRefinements(Element* elem, int coarse_index,
                                 std::string &ref_path, RefPathMap &map)
{
   if (!elem->ref_type)
   {
      int &matrix = map[ref_path];
      if (!matrix) { matrix = map.size(); }

      Embedding &emb = transforms.embeddings[elem->index];
      emb.parent = coarse_index;
      emb.matrix = matrix - 1;
   }
   else
   {
      ref_path.push_back(elem->ref_type);
      ref_path.push_back(0);

      for (int i = 0; i < 8; i++)
      {
         if (elem->child[i])
         {
            ref_path[ref_path.length()-1] = i;
            TraverseRefinements(elem->child[i], coarse_index, ref_path, map);
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

      MFEM_ASSERT(root_elements.Size(), "");
      int geom = root_elements[0]->geom;
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

      MFEM_ASSERT(root_elements.Size(), "");
      int geom = root_elements[0]->geom;
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

NCMesh::Node* NCMesh::GetEdgeMaster(Node* node) const
{
   MFEM_ASSERT(node != NULL && node->p1 != node->p2, "Invalid edge node.");

   Node *n1 = nodes.Peek(node->p1);
   Node *n2 = nodes.Peek(node->p2);

   if ((n2->p1 != n2->p2) && (n1->id == n2->p1 || n1->id == n2->p2))
   {
      // n1 is parent of n2:
      // (n1)--(n)--(n2)------(*)  or  (*)------(n2)--(n)--(n1)
      if (n2->edge) { return n2; }
      else { return GetEdgeMaster(n2); }
   }

   if ((n1->p1 != n1->p2) && (n2->id == n1->p1 || n2->id == n1->p2))
   {
      // n2 is parent of n1:
      // (n2)--(n)--(n1)------(*)  or  (*)------(n1)--(n)--(n2)
      if (n1->edge) { return n1; }
      else { return GetEdgeMaster(n1); }
   }

   return NULL;
}

int NCMesh::GetEdgeMaster(int v1, int v2) const
{
   Node* node = nodes.Peek(vertex_nodeId[v1], vertex_nodeId[v2]);
   MFEM_ASSERT(node->edge != NULL, "(v1, v2) is not an edge.");

   Node* master = GetEdgeMaster(node);
   return master ? master->edge->index : -1;
}

int NCMesh::GetElementDepth(int i) const
{
   Element* elem = leaf_elements[i];
   int depth = 0;
   while (elem->parent)
   {
      elem = elem->parent;
      depth++;
   }
   return depth;
}

void NCMesh::find_face_nodes(const Face *face, Node* node[4])
{
   // Obtain face nodes from one of its elements (note that face->p1, p2, p3
   // cannot be used directly since they are not in order and p4 is missing).

   Element* elem = face->elem[0];
   if (!elem) { elem = face->elem[1]; }
   MFEM_ASSERT(elem, "Face has no elements?");

   int f = find_hex_face(find_node(elem, face->p1),
                         find_node(elem, face->p2),
                         find_node(elem, face->p3));

   const int* fv = GI[Geometry::CUBE].faces[f];
   for (int i = 0; i < 4; i++)
   {
      node[i] = elem->node[fv[i]];
   }
}

void NCMesh::GetBoundaryClosure(const Array<int> &bdr_attr_is_ess,
                                Array<int> &bdr_vertices, Array<int> &bdr_edges)
{
   bdr_vertices.SetSize(0);
   bdr_edges.SetSize(0);

   if (Dim == 3)
   {
      GetFaceList();
      for (int i = 0; i < boundary_faces.Size(); i++)
      {
         Face* face = boundary_faces[i];
         if (bdr_attr_is_ess[face->attribute - 1])
         {
            Node* node[4];
            find_face_nodes(face, node);

            for (int j = 0; j < 4; j++)
            {
               bdr_vertices.Append(node[j]->vertex->index);

               Node* edge = nodes.Peek(node[j], node[(j+1) % 4]);
               MFEM_ASSERT(edge && edge->edge, "Edge not found.");
               bdr_edges.Append(edge->edge->index);

               while ((edge = GetEdgeMaster(edge)) != NULL)
               {
                  // append master edges that may not be accessible from any
                  // boundary element, this happens in 3D in re-entrant corners
                  bdr_edges.Append(edge->edge->index);
               }
            }
         }
      }
   }
   else if (Dim == 2)
   {
      GetEdgeList();
      for (int i = 0; i < boundary_edges.Size(); i++)
      {
         Node* edge = boundary_edges[i];
         if (bdr_attr_is_ess[edge->edge->attribute - 1])
         {
            bdr_vertices.Append(nodes.Peek(edge->p1)->vertex->index);
            bdr_vertices.Append(nodes.Peek(edge->p2)->vertex->index);
         }
      }
   }

   bdr_vertices.Sort();
   bdr_vertices.Unique();

   bdr_edges.Sort();
   bdr_edges.Unique();
}

int NCMesh::EdgeSplitLevel(Node *v1, Node *v2) const
{
   Node* mid = nodes.Peek(v1, v2);
   if (!mid || !mid->vertex) { return 0; }
   return 1 + std::max(EdgeSplitLevel(v1, mid), EdgeSplitLevel(mid, v2));
}

void NCMesh::FaceSplitLevel(Node* v1, Node* v2, Node* v3, Node* v4,
                            int& h_level, int& v_level) const
{
   int hl1, hl2, vl1, vl2;
   Node* mid[4];

   switch (FaceSplitType(v1, v2, v3, v4, mid))
   {
      case 0: // not split
         h_level = v_level = 0;
         break;

      case 1: // vertical
         FaceSplitLevel(v1, mid[0], mid[2], v4, hl1, vl1);
         FaceSplitLevel(mid[0], v2, v3, mid[2], hl2, vl2);
         h_level = std::max(hl1, hl2);
         v_level = std::max(vl1, vl2) + 1;
         break;

      default: // horizontal
         FaceSplitLevel(v1, v2, mid[1], mid[3], hl1, vl1);
         FaceSplitLevel(mid[3], mid[1], v3, v4, hl2, vl2);
         h_level = std::max(hl1, hl2) + 1;
         v_level = std::max(vl1, vl2);
   }
}

static int max8(int a, int b, int c, int d, int e, int f, int g, int h)
{
   return std::max(std::max(std::max(a, b), std::max(c, d)),
                   std::max(std::max(e, f), std::max(g, h)));
}

void NCMesh::CountSplits(Element* elem, int splits[3]) const
{
   Node** node = elem->node;
   GeomInfo& gi = GI[(int) elem->geom];

   int elevel[12];
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      elevel[i] = EdgeSplitLevel(node[ev[0]], node[ev[1]]);
   }

   if (elem->geom == Geometry::CUBE)
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
   else if (elem->geom == Geometry::SQUARE)
   {
      splits[0] = std::max(elevel[0], elevel[2]);
      splits[1] = std::max(elevel[1], elevel[3]);
   }
   else if (elem->geom == Geometry::TRIANGLE)
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
      if (IsGhost(leaf_elements[i])) { break; }

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
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->vertex && it->p1 != it->p2) { nv++; }
   }
   out << nv << "\n";

   // print the relations
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->vertex && it->p1 != it->p2)
      {
         Node *p1 = nodes.Peek(it->p1);
         Node *p2 = nodes.Peek(it->p2);

         MFEM_ASSERT(p1 && p1->vertex, "");
         MFEM_ASSERT(p2 && p2->vertex, "");

         out << it->vertex->index << " "
             << p1->vertex->index << " "
             << p2->vertex->index << "\n";
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

      Node* node = nodes.Peek(id);
      MFEM_VERIFY(node, "vertex " << id << " not found.");
      MFEM_VERIFY(nodes.Peek(p1), "parent " << p1 << " not found.");
      MFEM_VERIFY(nodes.Peek(p2), "parent " << p2 << " not found.");

      // assign new parents for the node
      nodes.Reparent(node, p1, p2);
   }
}

void NCMesh::SetVertexPositions(const Array<mfem::Vertex> &vertices)
{
   for (int i = 0; i < vertices.Size(); i++)
   {
      Node* node = nodes.Peek(i);
      MFEM_ASSERT(node && node->vertex, "");

      const double* pos = vertices[i]();
      memcpy(node->vertex->pos, pos, sizeof(node->vertex->pos));
   }
}

static int ref_type_num_children[8] = {0, 2, 2, 4, 2, 4, 4, 8 };

int NCMesh::PrintElements(std::ostream &out, Element* elem,
                          int &coarse_id) const
{
   if (elem->ref_type)
   {
      int child_id[8], nch = 0;
      for (int i = 0; i < 8; i++)
      {
         if (elem->child[i])
         {
            child_id[nch++] = PrintElements(out, elem->child[i], coarse_id);
         }
      }
      MFEM_ASSERT(nch == ref_type_num_children[(int) elem->ref_type], "");

      out << (int) elem->ref_type;
      for (int i = 0; i < nch; i++)
      {
         out << " " << child_id[i];
      }
      out << "\n";
      return coarse_id++; // return new id for this coarse element
   }
   else
   {
      return elem->index;
   }
}

void NCMesh::PrintCoarseElements(std::ostream &out) const
{
   // print the number of non-leaf elements
   int ne = 0;
   for (int i = 0; i < root_elements.Size(); i++)
   {
      ne += CountElements(root_elements[i]);
   }
   out << (ne - leaf_elements.Size()) << "\n";

   // print the hierarchy recursively
   int coarse_id = leaf_elements.Size();
   for (int i = 0; i < root_elements.Size(); i++)
   {
      PrintElements(out, root_elements[i], coarse_id);
   }
}

void NCMesh::LoadCoarseElements(std::istream &input)
{
   int ne;
   input >> ne;

   Array<Element*> coarse, leaves;
   coarse.Reserve(ne);
   leaf_elements.Copy(leaves);
   int nleaf = leaves.Size();
   bool iso = true;

   // load the coarse elements
   while (ne--)
   {
      int ref_type;
      input >> ref_type;

      Element* elem = new Element(0, 0);
      elem->ref_type = ref_type;

      if (Dim == 3 && ref_type != 7) { iso = false; }

      // load child IDs and convert to Element*
      int nch = ref_type_num_children[ref_type];
      for (int i = 0, id; i < nch; i++)
      {
         input >> id;
         MFEM_VERIFY(id >= 0, "");
         MFEM_VERIFY(id < nleaf || id - nleaf < coarse.Size(),
                     "coarse element cannot be referenced before it is "
                     "defined (id=" << id << ").");

         Element* &child = (id < nleaf) ? leaves[id] : coarse[id - nleaf];

         MFEM_VERIFY(child, "element " << id << " cannot have two parents.");
         elem->child[i] = child;
         child->parent = elem;
         child = NULL; // make sure the child can't be used again

         if (!i) // copy geom and attribute from first child
         {
            elem->geom = elem->child[i]->geom;
            elem->attribute = elem->child[i]->attribute;
         }
      }

      // keep a list of coarse elements (and their IDs, implicitly)
      coarse.Append(elem);
   }

   // elements that have no parents are the original 'root_elements'
   root_elements.SetSize(0);
   for (int i = 0; i < coarse.Size(); i++)
   {
      if (coarse[i]) { root_elements.Append(coarse[i]); }
   }
   for (int i = 0; i < leaves.Size(); i++)
   {
      if (leaves[i]) { root_elements.Append(leaves[i]); }
   }

   // set the Iso flag (must be false if there are 3D aniso refinements)
   Iso = iso;
}

int NCMesh::CountElements(Element* elem) const
{
   int n = 1;
   if (elem->ref_type)
   {
      for (int i = 0; i < 8; i++)
      {
         if (elem->child[i]) { n += CountElements(elem->child[i]); }
      }
   }
   return n;
}

long NCMesh::NCList::MemoryUsage() const
{
   int pmsize = 0;
   if (slaves.size())
   {
      const DenseMatrix &pm = slaves[0].point_matrix;
      pmsize = pm.Width() * pm.Height() * sizeof(double);
   }

   return conforming.capacity() * sizeof(MeshId) +
          masters.capacity() * sizeof(Master) +
          slaves.capacity() * sizeof(Slave) +
          slaves.size() * pmsize;
}

void NCMesh::CountObjects(int &nelem, int &nvert, int &nedges) const
{
   nelem = nvert = nedges = 0;
   for (int i = 0; i < root_elements.Size(); i++)
   {
      nelem += CountElements(root_elements[i]);
   }
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->vertex) { nvert++; }
      if (it->edge) { nedges++; }
   }
}

long NCMesh::MemoryUsage() const
{
   int nelem, nvert, nedges;
   CountObjects(nelem, nvert, nedges);

   return nelem * sizeof(Element) +
          nvert * sizeof(Vertex) +
          nedges * sizeof(Edge) +
          nodes.MemoryUsage() +
          faces.MemoryUsage() +
          root_elements.MemoryUsage() +
          leaf_elements.MemoryUsage() +
          vertex_nodeId.MemoryUsage() +
          face_list.MemoryUsage() +
          edge_list.MemoryUsage() +
          boundary_faces.MemoryUsage() +
          boundary_edges.MemoryUsage() +
          element_vertex.MemoryUsage() +
          ref_stack.MemoryUsage() +
          coarse_elements.MemoryUsage() +
          sizeof(*this);
}

void NCMesh::PrintMemoryDetail() const
{
   int nelem, nvert, nedges;
   CountObjects(nelem, nvert, nedges);

   std::cout << nelem * sizeof(Element) << " elements\n"
             << nvert * sizeof(Vertex) << " vertices\n"
             << nedges * sizeof(Edge) << " edges\n";

   nodes.PrintMemoryDetail(); std::cout << " nodes\n";
   faces.PrintMemoryDetail(); std::cout << " faces\n";

   std::cout << root_elements.MemoryUsage() << " root_elements\n"
             << leaf_elements.MemoryUsage() << " leaf_elements\n"
             << vertex_nodeId.MemoryUsage() << " vertex_nodeId\n"
             << face_list.MemoryUsage() << " face_list\n"
             << edge_list.MemoryUsage() << " edge_list\n"
             << boundary_faces.MemoryUsage() << " boundary_faces\n"
             << boundary_edges.MemoryUsage() << " boundary_edges\n"
             << element_vertex.MemoryUsage() << " element_vertex\n"
             << ref_stack.MemoryUsage() << " ref_stack\n"
             << coarse_elements.MemoryUsage() << " coarse_elements\n"
             << sizeof(*this) << " NCMesh" << std::endl;
}

#ifdef MFEM_DEBUG
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
