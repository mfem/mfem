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
//
//   ------------------------------------------------------------------------
//   Reflector Miniapp: Reflect a mesh about a plane
//   ------------------------------------------------------------------------
//
// This miniapp reflects a 3D mesh about a plane defined by a point and a
// normal vector. Element and boundary element attributes are copied from the
// corresponding elements in the original mesh, except for boundary elements on
// the plane of reflection.
//
// Compile with: make reflector
//
// Sample runs:  reflector -m ../../data/pipe-nurbs.mesh -n '0 0 1'
//               reflector -m ../../data/fichera.mesh -o '1 0 0' -n '1 0 0'


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <array>

using namespace std;
using namespace mfem;


void ReflectPoint(Vector & p, Vector const& origin, Vector const& normal)
{
   Vector diff(3);
   Vector proj(3);
   subtract(p, origin, diff);
   const real_t ip = diff * normal;

   diff = normal;
   diff *= -2.0 * ip;

   add(p, diff, p);
}

class ReflectedCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient * a;
   const Vector origin, normal;
   Mesh *meshOrig;

   // Map from reflected to original mesh elements
   std::vector<int> *r2o;

   std::vector<std::vector<int>> *perm;

public:
   ReflectedCoefficient(VectorCoefficient &A, Vector const& origin_,
                        Vector const& normal_, std::vector<int> *r2o_,
                        Mesh *mesh, std::vector<std::vector<int>> *refPerm) :
      VectorCoefficient(3), a(&A), origin(origin_), normal(normal_),
      meshOrig(mesh), r2o(r2o_), perm(refPerm)
   { }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   using VectorCoefficient::Eval;
};

void ReflectedCoefficient::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   real_t x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   const int elem = T.ElementNo;
   const bool reflected = (*r2o)[elem] < 0;
   const int originalElem = reflected ? -1 - (*r2o)[elem] : (*r2o)[elem];

   ElementTransformation *T_orig = meshOrig->GetElementTransformation(
                                      originalElem);

   Vector rp(transip);

   if (reflected)
   {
      ReflectPoint(rp, origin, normal);
   }

   IntegrationPoint ipo;

   a->Eval(V, *T_orig, ip);

   if (reflected)
   {
      // Map from ip in reflected elements to ip in initial elements, in mesh
      // `reflected`. y = Ax + b in reference spaces, where A has 9 entries,
      // and b has 3, totaling 12 unknowns. The number of data points is
      // 3 * 8 = 24, so it is overdetermined. We use 4 points out of 8, (0,0,0),
      // (1,0,0), (0,1,0), (0,0,1). For x=(0,0,0), b = y, and the other choices
      // give the columns of A.

      // Permutation p is such that hex_reflected[i] = hex_init[p[i]]
      const std::vector<int>& p = (*perm)[elem];

      // ip is on reflected hex. We map from the reflected hex to the initial
      // hex, in reference space. Thus we use y = Ax + b, where x is in the
      // reflected reference space, and y is in the initial hex reference space.

      const IntegrationRule *ir = Geometries.GetVertices(Geometry::CUBE);

      Vector b(3);
      b[0] = (*ir)[p[0]].x;
      b[1] = (*ir)[p[0]].y;
      b[2] = (*ir)[p[0]].z;

      DenseMatrix A(3);

      // Vertex 1 is x=(1,0,0), so Ax is the first column of A.
      A(0,0) = (*ir)[p[1]].x - b[0];
      A(1,0) = (*ir)[p[1]].y - b[1];
      A(2,0) = (*ir)[p[1]].z - b[2];

      // Vertex 3 is x=(0,1,0), so Ax is the second column of A.
      A(0,1) = (*ir)[p[3]].x - b[0];
      A(1,1) = (*ir)[p[3]].y - b[1];
      A(2,1) = (*ir)[p[3]].z - b[2];

      // Vertex 4 is x=(0,0,1), so Ax is the third column of A.
      A(0,2) = (*ir)[p[4]].x - b[0];
      A(1,2) = (*ir)[p[4]].y - b[1];
      A(2,2) = (*ir)[p[4]].z - b[2];

      Vector r(3);
      Vector y(3);

      r[0] = ip.x;
      r[1] = ip.y;
      r[2] = ip.z;

      A.Mult(r, y);
      y += b;

      ipo.x = y[0];
      ipo.y = y[1];
      ipo.z = y[2];

      a->Eval(V, *T_orig, ipo);
   }

   if (reflected)
   {
      ReflectPoint(V, origin, normal);
   }
}

// Find perm such that h1[i] = h2[perm[i]]
void GetHexPermutation(Array<int> const& h1, Array<int> const& h2,
                       std::vector<int> & perm)
{
   std::map<int, int> h2inv;
   const int n = perm.size();

   for (int i=0; i<n; ++i)
   {
      h2inv[h2[i]] = i;
   }

   for (int i=0; i<n; ++i)
   {
      perm[i] = h2inv[h1[i]];
   }
}

// This class facilitates constructing a hexahedral mesh one element at a time,
// using AddElement. The hexahedron input to AddElement is specified by vertices
// without requiring consistent global orientations. The mesh can be constructed
// simply by calling AddVertex for all vertices and then AddElement for all
// hexahedra. The mesh is not owned by this class and should be deleted outside
// this class.
class HexMeshBuilder
{
public:
   HexMeshBuilder(int nv, int ne) : v2f(nv)
   {
      mesh = new Mesh(3, nv, ne);
      f2v =
      {
         {  {0,3,2,1},
            {4,5,6,7},
            {0,1,5,4},
            {3,7,6,2},
            {0,4,7,3},
            {1,2,6,5}
         }
      };

      e2v =
      {
         {  {0,1},
            {1,2},
            {2,3},
            {0,3},
            {4,5},
            {5,6},
            {6,7},
            {4,7},
            {0,4},
            {1,5},
            {2,6},
            {3,7}
         }
      };
   }

   /** @brief Add a single vertex to the mesh, specified by 3 coordinates. */
   int AddVertex(const real_t *coords) const
   {
      return mesh->AddVertex(coords);
   }

   /** @brief Add a single hexahedral element to the mesh, specified by 8
       vertices. */
   /** If reorder is false, the ordering in @vertices is used, so it must be
       known in advance to have consistent orientations. Otherwise, a new
       ordering will be found to ensure consistent orientations in the mesh.
    */
   int AddElement(Array<int> const& vertices, const bool reorder);

   Mesh *mesh;
   std::vector<std::vector<int>> refPerm;

private:
   std::vector<std::vector<int>> faces;
   std::vector<std::set<int>> v2f;
   std::vector<std::vector<int>> f2e;
   std::array<std::array<int, 4>, 6> f2v;
   std::array<std::array<int, 2>, 12> e2v;

   int FindFourthVertexOnFace(Array<int> const& hex,
                              std::vector<int> const& v3) const;

   void ReorderHex(Array<int> & hex) const;
   void ReverseHexX(Array<int> & hex) const;
   void ReverseHexY(Array<int> & hex) const;
   void ReverseHexZ(Array<int> & hex) const;
   void ReverseHexFace(Array<int> & hex, const int face) const;
   int FindHexFace(Array<int> const& hex, std::vector<int> const& face) const;

   bool ReorderHex_faceOrientations(Array<int> & hex) const;
   void SaveHexFaces(const int elem, Array<int> const& hex);
};

int HexMeshBuilder::AddElement(Array<int> const& vertices, const bool reorder)
{
   MFEM_ASSERT(vertices.Size() == 8, "Hexahedron must have 8 vertices");

   Array<int> rvert(vertices);
   if (reorder)
   {
      ReorderHex(rvert);  // First reorder to set (0,0,0) and (1,1,1) vertices.

      // Now reorder to get consistent face orientations.
      bool reordered = true;
      int iter = 0;
      do
      {
         reordered = ReorderHex_faceOrientations(rvert);
         iter++;
         MFEM_VERIFY(iter < 5, "");
      }
      while (reordered);

      std::vector<int> perm_e(8);
      GetHexPermutation(rvert, vertices, perm_e);
      refPerm.push_back(perm_e);
   }
   else
   {
      refPerm.push_back(std::vector<int> {0, 1, 2, 3, 4, 5, 6, 7});
   }

   SaveHexFaces(mesh->GetNE(), rvert);

   Element * nel = mesh->NewElement(Geometry::Type::CUBE);

   nel->SetVertices(rvert);
   return mesh->AddElement(nel);
}

int HexMeshBuilder::FindFourthVertexOnFace(Array<int> const& hex,
                                           std::vector<int> const& v3) const
{
   int f0 = -1;
   for (int f=0; f<6; ++f)
   {
      bool all3found = true;

      for (int i=0; i<3; ++i)
      {
         // Check whether v3[i] is in face f
         bool found = false;

         for (int j=0; j<4; ++j)
         {
            if (hex[f2v[f][j]] == v3[i])
            {
               found = true;
            }
         }

         if (!found)
         {
            all3found = false;
            break;
         }
      }

      if (all3found)
      {
         MFEM_ASSERT(f0 == -1, "");
         f0 = f;
      }
   }

   MFEM_VERIFY(f0 >= 0, "");

   // Find the vertex of f0 not in v3
   int v = -1;

   for (int j=0; j<4; ++j)
   {
      bool found = false;
      for (int i=0; i<3; ++i)
      {
         // Check whether v3[i] is in face f
         if (hex[f2v[f0][j]] == v3[i])
         {
            found = true;
         }
      }

      if (!found)
      {
         MFEM_ASSERT(v == -1, "");
         v = hex[f2v[f0][j]];
      }
   }

   MFEM_VERIFY(v >= 0, "");

   return v;
}

void HexMeshBuilder::ReorderHex(Array<int> & hex) const
{
   MFEM_VERIFY(hex.Size() == 8, "hex");

   Array<int> h(hex);

   const int v0 = hex.Min();

   std::map<int, int> v2hex0;

   for (int i=0; i<hex.Size(); ++i)
   {
      v2hex0[hex[i]] = i;
   }

   // Find the 3 vertices sharing an edge with v0.
   std::vector<int> v0e;
   for (int e=0; e<12; ++e)
   {
      if (v0 == hex[e2v[e][0]] || v0 == hex[e2v[e][1]])
      {
         v0e.push_back(e);
      }
   }

   MFEM_VERIFY(v0e.size() == 3, "");

   std::vector<int> v0n;  // Neighbors of v0
   for (auto e : v0e)
   {
      if (v0 == hex[e2v[e][0]])
      {
         v0n.push_back(hex[e2v[e][1]]);
      }
      else
      {
         v0n.push_back(hex[e2v[e][0]]);
      }
   }

   MFEM_VERIFY(v0n.size() == 3, "");

   sort(v0n.begin(), v0n.end());

   h[0] = v0;
   h[1] = v0n[0];
   h[3] = v0n[1];
   h[4] = v0n[2];

   // Set h[2] by finding the face containing h[0], h[1], h[3]
   std::vector<int> v3(3);
   v3[0] = h[0];
   v3[1] = h[1];
   v3[2] = h[3];
   h[2] = FindFourthVertexOnFace(hex, v3);

   // Set h[5] based on h[0], h[1], h[4]
   v3[2] = h[4];
   h[5] = FindFourthVertexOnFace(hex, v3);

   // Set h[7] based on h[0], h[3], h[4]
   v3[1] = h[3];
   h[7] = FindFourthVertexOnFace(hex, v3);

   // Set h[6] based on h[1], h[2], h[5]
   v3[0] = h[1];
   v3[1] = h[2];
   v3[2] = h[5];
   h[6] = FindFourthVertexOnFace(hex, v3);

   hex = h;
}

void HexMeshBuilder::ReverseHexZ(Array<int> & hex) const
{
   // faces {0,1,2,3} and {4,5,6,7} are reversed to become
   // {0,3,2,1} and {4,7,6,5}
   // This is accomplished by swapping vertices 1 and 3, and vertices 5 and 7.
   int s = hex[1];
   hex[1] = hex[3];
   hex[3] = s;

   s = hex[5];
   hex[5] = hex[7];
   hex[7] = s;
}

void HexMeshBuilder::ReverseHexY(Array<int> & hex) const
{
   // faces {0,1,5,4} and {3,2,6,7} are reversed to become
   // {0,4,5,1} and {3,7,6,2}
   // This is accomplished by swapping vertices 1 and 4, and vertices 2 and 7.
   int s = hex[1];
   hex[1] = hex[4];
   hex[4] = s;

   s = hex[2];
   hex[2] = hex[7];
   hex[7] = s;
}

void HexMeshBuilder::ReverseHexX(Array<int> & hex) const
{
   // faces {0,3,7,4} and {1,2,6,5} are reversed to become
   // {0,4,7,3} and {1,5,6,2}
   // This is accomplished by swapping vertices 3 and 4, and vertices 2 and 5.
   int s = hex[4];
   hex[4] = hex[3];
   hex[3] = s;

   s = hex[5];
   hex[5] = hex[2];
   hex[2] = s;
}

// Reverse face orientations without changing reference vertices 0 or 6.
void HexMeshBuilder::ReverseHexFace(Array<int> & hex, const int face) const
{
   const int f = 2 * (face / 2);  // f is in {0, 2, 4}

   switch (f)
   {
      case 0:
         ReverseHexZ(hex);
         break;
      case 2:
         ReverseHexY(hex);
         break;
      default:  // case 4
         ReverseHexX(hex);
   }
}

int HexMeshBuilder::FindHexFace(Array<int> const& hex,
                                std::vector<int> const& face) const
{
   int localFace = -1;
   for (int f=0; f<6; ++f)
   {
      std::vector<int> fv(4);
      for (int i=0; i<4; ++i)
      {
         fv[i] = hex[f2v[f][i]];
      }

      sort(fv.begin(), fv.end());

      if (fv == face)
      {
         MFEM_VERIFY(localFace == -1, "");
         localFace = f;
      }
   }

   MFEM_VERIFY(localFace >= 0, "");

   return localFace;
}

bool HexMeshBuilder::ReorderHex_faceOrientations(Array<int> & hex) const
{
   std::vector<int> localFacesFound, globalFacesFound;
   for (int f=0; f<6; ++f)
   {
      std::vector<int> fv(4);
      for (int i=0; i<4; ++i)
      {
         fv[i] = hex[f2v[f][i]];
      }

      sort(fv.begin(), fv.end());

      const int vmin = fv[0];
      int globalFace = -1;
      for (auto gf : v2f[vmin])
      {
         if (fv == faces[gf])
         {
            globalFace = gf;
         }
      }

      if (globalFace >= 0)
      {
         globalFacesFound.push_back(globalFace);
         localFacesFound.push_back(f);
      }
   }

   const int numFoundFaces = globalFacesFound.size();

   for (int ff=0; ff<numFoundFaces; ++ff)
   {
      const int globalFace = globalFacesFound[ff];
      const int localFace = localFacesFound[ff];

      MFEM_VERIFY(f2e[globalFace].size() == 1, "");
      const int neighborElem = f2e[globalFace][0];

      Array<int> neighborElemVert;
      mesh->GetElementVertices(neighborElem, neighborElemVert);
      const int neighborLocalFace = FindHexFace(neighborElemVert, faces[globalFace]);

      std::vector<int> fv(4);
      std::vector<int> nv(4);
      for (int i=0; i<4; ++i)
      {
         fv[i] = hex[f2v[localFace][i]];
         nv[i] = neighborElemVert[f2v[neighborLocalFace][i]];
      }

      // As in Mesh::GetQuadOrientation, check whether fv and nv are oriented
      // in the same direction.

      int id0;
      for (id0 = 0; id0 < 4; id0++)
      {
         if (fv[id0] == nv[0])
         {
            break;
         }
      }

      MFEM_VERIFY(id0 < 4, "");

      bool same = (fv[(id0+1) % 4] == nv[1]);
      if (same)
      {
         // Orientation should not be the same, so reverse the orientation of
         // face localFace and its opposite face in hex.
         ReverseHexFace(hex, localFace);
         return true;
      }
   }

   return false;
}

void HexMeshBuilder::SaveHexFaces(const int elem, Array<int> const& hex)
{
   for (int f=0; f<6; ++f)
   {
      std::vector<int> fv(4);
      for (int i=0; i<4; ++i)
      {
         fv[i] = hex[f2v[f][i]];
      }

      sort(fv.begin(), fv.end());

      const int vmin = fv[0];
      int globalFace = -1;
      for (auto gf : v2f[vmin])
      {
         if (fv == faces[gf])
         {
            globalFace = gf;
         }
      }

      if (globalFace == -1)
      {
         // Face not found, so add it.
         faces.push_back(fv);
         globalFace = faces.size() - 1;

         std::vector<int> firstElem = {elem};
         f2e.push_back(firstElem);
      }
      else
      {
         // Face found, so add elem to f2e
         MFEM_VERIFY(f2e[globalFace].size() == 1 &&
                     f2e[globalFace][0] != elem, "");
         f2e[globalFace].push_back(elem);
      }

      MFEM_VERIFY(faces.size() == f2e.size(), "");
      v2f[vmin].insert(globalFace);
   }
}

real_t GetElementEdgeMin(Mesh const& mesh, const int elem)
{
   Array<int> edges, cor;
   mesh.GetElementEdges(elem, edges, cor);

   real_t diam = -1.0;
   for (auto e : edges)
   {
      Array<int> vert;
      mesh.GetEdgeVertices(e, vert);
      const real_t *v0 = mesh.GetVertex(vert[0]);
      const real_t *v1 = mesh.GetVertex(vert[1]);

      real_t L = 0.0;
      for (int i=0; i<3; ++i)
      {
         L += (v0[i] - v1[i]) * (v0[i] - v1[i]);
      }

      L = sqrt(L);

      if (diam < 0.0 || L < diam) { diam = L; }
   }

   return diam;
}

void FindElementsTouchingPlane(Mesh const& mesh, Vector const& origin,
                               Vector const& normal, std::vector<int> & el)
{
   const real_t relTol = 1.0e-6;
   Vector diff(3);

   for (int e=0; e<mesh.GetNE(); ++e)
   {
      const real_t diam = GetElementEdgeMin(mesh, e);
      Array<int> vert;
      mesh.GetElementVertices(e, vert);

      bool onplane = false;
      for (auto v : vert)
      {
         const real_t *vcrd = mesh.GetVertex(v);
         for (int i=0; i<3; ++i)
         {
            diff[i] = vcrd[i] - origin[i];
         }

         if (std::abs(diff * normal) < relTol * diam)
         {
            onplane = true;
         }
      }

      if (onplane) { el.push_back(e); }
   }
}

// Order the elements in layers, starting at the plane of reflection.
bool GetMeshElementOrder(Mesh const& mesh, Vector const& origin,
                         Vector const& normal, std::vector<int> & elOrder)
{
   const int ne = mesh.GetNE();
   elOrder.assign(ne, -1);

   std::vector<bool> elementMarked;

   elementMarked.assign(ne, false);

   std::vector<int> layer;
   FindElementsTouchingPlane(mesh, origin, normal, layer);

   if (layer.size() == 0)
   {
      // If the mesh does not touch the plane, any ordering will work.
      for (int i=0; i<ne; ++i)
      {
         elOrder[i] = i;
      }

      return false;
   }

   int cnt = 0;
   while (cnt < ne)
   {
      for (auto e : layer)
      {
         elOrder[cnt] = e;
         cnt++;
         elementMarked[e] = true;
      }

      if (cnt == ne) { break; }

      std::set<int> layerNext;
      for (auto e : layer)
      {
         Array<int> nghb = mesh.FindFaceNeighbors(e);
         for (auto n : nghb)
         {
            if (!elementMarked[n]) { layerNext.insert(n); }
         }
      }

      MFEM_VERIFY(layerNext.size() > 0, "");

      layer.clear();
      layer.reserve(layerNext.size());
      for (auto e : layerNext)
      {
         layer.push_back(e);
      }
   }

   MFEM_VERIFY(cnt == ne, "");

   return true;
}

Mesh* ReflectHighOrderMesh(Mesh & mesh, Vector origin, Vector normal)
{
   MFEM_VERIFY(mesh.Dimension() == 3, "Only 3D meshes can be reflected");

   // Find the minimum edge length, to use for a relative tolerance.
   real_t minLength = 0.0;
   for (int i=0; i<mesh.GetNE(); i++)
   {
      Array<int> vert;
      mesh.GetEdgeVertices(i, vert);
      const Vector v0(mesh.GetVertex(vert[0]), 3);
      const Vector v1(mesh.GetVertex(vert[1]), 3);
      Vector diff(3);
      subtract(v0, v1, diff);
      const real_t length = diff.Norml2();
      if (i == 0 || length < minLength)
      {
         minLength = length;
      }
   }

   const real_t relTol = 1.0e-6;

   // Find vertices in reflection plane.
   std::set<int> planeVertices;
   for (int i=0; i<mesh.GetNV(); i++)
   {
      Vector v(mesh.GetVertex(i), 3);
      Vector diff(3);
      subtract(v, origin, diff);
      const real_t ip = diff * normal;
      if (std::abs(ip) < relTol * minLength)
      {
         planeVertices.insert(i);
      }
   }

   const int nv = mesh.GetNV();
   const int ne = mesh.GetNE();

   std::vector<int> r2o;

   const int nv_reflected = (2*nv) - planeVertices.size();

   HexMeshBuilder builder(nv_reflected, 2*ne);

   r2o.assign(2*ne, -2-ne);  // Initialize to invalid value.

   std::vector<int> v2r;
   v2r.assign(mesh.GetNV(), -1);

   // Copy vertices
   for (int v=0; v<mesh.GetNV(); v++)
   {
      builder.AddVertex(mesh.GetVertex(v));
   }

   for (int v=0; v<mesh.GetNV(); v++)
   {
      // Check whether vertex v is in the plane
      if (planeVertices.find(v) == planeVertices.end())
      {
         // For vertices not in plane, reflect and add.
         Vector vr(3);
         for (int i=0; i<3; ++i)
         {
            vr[i] = mesh.GetVertex(v)[i];
         }

         ReflectPoint(vr, origin, normal);

         v2r[v] = builder.AddVertex(vr.GetData());
      }
   }

   std::vector<int> elOrder;
   const bool onPlane = GetMeshElementOrder(mesh, origin, normal, elOrder);

   for (int eidx=0; eidx<mesh.GetNE(); eidx++)
   {
      const int e = elOrder[eidx];

      // Copy the original element
      Array<int> elvert;
      mesh.GetElementVertices(e, elvert);

      MFEM_VERIFY(elvert.Size() == 8, "Only hexahedral elements are supported");

      const int copiedElem = builder.AddElement(elvert, false);
      r2o[copiedElem] = e;

      // Add the new reflected element
      Array<int> rvert(elvert.Size());
      for (int i=0; i<elvert.Size(); ++i)
      {
         const int v = elvert[i];
         rvert[i] = (v2r[v] == -1) ? v : v2r[v];
      }

      const int newElem = builder.AddElement(rvert, onPlane);
      r2o[newElem] = -1 - e;
   }

   Mesh *reflected = builder.mesh;

   // Set attributes
   MFEM_VERIFY((int) r2o.size() == reflected->GetNE(), "");
   for (int i = 0; i < (int) r2o.size(); ++i)
   {
      const int e = (r2o[i] >= 0) ? r2o[i] : -1 - r2o[i];
      reflected->SetAttribute(i, mesh.GetAttribute(e));
   }

   // In order to set boundary attributes, first set a map from original mesh
   // boundary elements to reflected mesh boundary elements, by using the vertex
   // map v2r. Note that for v < mesh.GetNV(), vertex v of `mesh` coincides with
   // vertex v of `reflected`, and if that vertex is not in the reflection
   // plane, v2r[v] >= mesh.GetNV() is the index of the vertex in `reflected`
   // that is its reflection.

   // Identify each quadrilateral boundary element with the unique pair of
   // vertices (v1, v2) such that v1 is the minimum vertex index in the
   // quadrilateral, and v2 is diagonally opposite v1.

   std::map<std::pair<int, int>, int> mapBE;
   for (int i=0; i<mesh.GetNBE(); ++i)
   {
      const Element *be = mesh.GetBdrElement(i);
      Array<int> v;
      be->GetVertices(v);
      MFEM_VERIFY(v.Size() == 4, "Boundary elements must be quadrilateral");

      const int v1 = v.Min();
      int v1i = -1;
      for (int j=0; j<v.Size(); ++j)
      {
         if (v[j] == v1)
         {
            v1i = j;
         }
      }

      const int v2 = v[(v1i + 2) % 4];

      mapBE[std::pair<int, int>(v1, v2)] = i;

      // Find the indices of vertices in `reflected` of the reflected quadrilateral.
      Array<int> rv(4);
      int rv1 = -1;  // Find the minimum reflected vertex index.
      int rv1i = -1;
      bool inPlane = true;
      for (int j=0; j<v.Size(); ++j)
      {
         rv[j] = (v2r[v[j]] == -1) ? v[j] : v2r[v[j]];

         if (v2r[v[j]] != -1)
         {
            inPlane = false;
         }

         if (rv1 == -1 || rv[j] < rv1)
         {
            rv1 = rv[j];
            rv1i = j;
         }
      }

      // Note that in-plane boundary elements are skipped.
      if (!inPlane)
      {
         const int rv2 = rv[(rv1i + 2) % 4];

         mapBE[std::pair<int, int>(rv1, rv2)] = i;

         mfem::Swap(rv[0], rv[2]);  // Fix the orientation

         const Geometry::Type orig_geom = mesh.GetBdrElementGeometry(i);
         Element *rbe = reflected->NewElement(orig_geom);
         rbe->SetVertices(v);
         reflected->AddBdrElement(rbe);

         rbe = reflected->NewElement(orig_geom);
         rbe->SetVertices(rv);
         reflected->AddBdrElement(rbe);
      }
   }

   for (int i=0; i<reflected->GetNBE(); ++i)
   {
      Element *be = reflected->GetBdrElement(i);
      Array<int> rv;
      be->GetVertices(rv);
      MFEM_VERIFY(rv.Size() == 4, "Boundary elements must be quadrilateral");

      // Reflected boundary element i is identified with vertices

      const int v1 = rv.Min();
      int v1i = -1;
      for (int j=0; j<rv.Size(); ++j)
      {
         if (rv[j] == v1)
         {
            v1i = j;
         }
      }

      const int v2 = rv[(v1i + 2) % 4];

      const int originalBE = mapBE[std::pair<int, int>(v1, v2)];
      const int originalAttribute = mesh.GetBdrAttribute(originalBE);
      reflected->SetBdrAttribute(i, originalAttribute);
   }

   reflected->FinalizeTopology();
   reflected->Finalize();
   reflected->RemoveUnusedVertices();

   if (mesh.GetNodes())
   {
      // Extract Nodes GridFunction and determine its type
      const GridFunction * Nodes = mesh.GetNodes();
      const FiniteElementSpace * fes = Nodes->FESpace();

      Ordering::Type ordering = fes->GetOrdering();
      int order = fes->FEColl()->GetOrder();
      int sdim = mesh.SpaceDimension();
      bool discont =
         dynamic_cast<const L2_FECollection*>(fes->FEColl()) != NULL;

      // Set curvature of the same type as original mesh
      reflected->SetCurvature(order, discont, sdim, ordering);

      GridFunction * reflected_nodes = reflected->GetNodes();
      GridFunction newReflectedNodes(*reflected_nodes);

      VectorGridFunctionCoefficient nodesCoef(Nodes);

      ReflectedCoefficient rc(nodesCoef, origin, normal, &r2o, &mesh,
                              &builder.refPerm);

      newReflectedNodes.ProjectCoefficient(rc);
      *reflected_nodes = newReflectedNodes;
   }

   return reflected;
}

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "../../data/pipe-nurbs.mesh";
   bool visualization = 1;
   Vector normal(3);
   Vector origin(3);

   normal = 0.0;
   normal[2] = 1.0;
   int visport = 19916;
   origin = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&normal, "-n", "--normal",
                  "Normal vector of plane.");
   args.AddOption(&origin, "-o", "--origin",
                  "A point in the plane.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   MFEM_VERIFY(std::abs(normal.Norml2() - 1.0) < 1.0e-14, "");

   Mesh mesh(mesh_file, 0, 0);

   Mesh *reflected = ReflectHighOrderMesh(mesh, origin, normal);

   // Save the final mesh
   ofstream mesh_ofs("reflected.mesh");
   mesh_ofs.precision(8);
   reflected->Print(mesh_ofs);

   if (visualization)
   {
      // GLVis server to visualize to
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *reflected << flush;
   }

   delete reflected;

   return 0;
}
