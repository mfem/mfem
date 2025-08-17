// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "ncnurbs.hpp"

namespace mfem
{

using namespace std;

// Helper functions for NC-NURBS

void GetShiftedGridPoints2D(int m, int n, int i, int j, int signedShift,
                            int& sm, int& sn, int& si, int& sj);

void GetInverseShiftedDimensions2D(int signedShift, int sm, int sn, int &m,
                                   int &n);

int GetFaceOrientation(const Mesh *mesh, const int face,
                       const std::array<int, 4> &verts);

bool Reorder2D(int ori, std::array<int, 2> &s0);

std::pair<int, int> QuadrupleToPair(const std::array<int, 4> &q);

NCNURBSExtension::NCNURBSExtension(std::istream &input, bool spacing)
{
   // Read topology
   patchTopo = new Mesh;
   patchTopo->LoadNonconformingPatchTopo(input, edge_to_ukv);
   nonconformingPT = true;

   Load(input, spacing);
}

NCNURBSExtension::NCNURBSExtension(const NCNURBSExtension &orig)
   : NURBSExtension(orig),
     aux_e_meshOffsets(orig.aux_e_meshOffsets),
     aux_f_meshOffsets(orig.aux_f_meshOffsets),
     aux_e_spaceOffsets(orig.aux_e_spaceOffsets),
     aux_f_spaceOffsets(orig.aux_f_spaceOffsets),
     auxEdges(orig.auxEdges),
     auxFaces(orig.auxFaces),
     auxef(orig.auxef)
{ }

void NCNURBSExtension::GetMasterEdgeEntities(
   int edge, Array<int> &edgeV, Array<int> &edgeE, Array<int> &edgeVki)
{
   const int mid = masterEdgeToId.at(edge);
   const std::size_t nes = masterEdgeInfo[mid].slaves.size();
   MFEM_ASSERT(masterEdgeInfo[mid].vertices.size() + 1 == nes, "");

   // Vertices in masterEdgeVerts[mid] are ordered starting
   // from the master edge endpoint with lower vertex index.

   Array<int> everts;
   patchTopo->GetEdgeVertices(edge, everts);

   edgeV.Append(everts[0]);
   edgeVki.Append(0);

   MFEM_ASSERT(masterEdgeInfo[mid].vertices.size() ==
               masterEdgeInfo[mid].ks.size(), "");
   for (std::size_t i=0; i<masterEdgeInfo[mid].vertices.size(); ++i)
   {
      edgeV.Append(masterEdgeInfo[mid].vertices[i]);
      edgeVki.Append(masterEdgeInfo[mid].ks[i]);
   }

   const int nelem = KnotVec(edge)->GetNE();

   edgeV.Append(everts[1]);
   edgeVki.Append(nelem);

   for (std::size_t i=0; i<nes; ++i)
   {
      const int edge_i = slaveEdges[masterEdgeInfo[mid].slaves[i]];
      edgeE.Append(edge_i);

      Array<int> sverts(2);
      if (edge_i >= 0) // If a slave edge
      {
         patchTopo->GetEdgeVertices(edge_i, sverts);
      }
      else
      {
         const int auxEdge = -1 - edge_i;
         GetAuxEdgeVertices(auxEdge, sverts);
      }

      MFEM_ASSERT((sverts[0] == edgeV[i] &&
                   sverts[1] == edgeV[i+1]) ||
                  (sverts[1] == edgeV[i] &&
                   sverts[0] == edgeV[i+1]), "");
   }
}

void NCNURBSExtension::FindAdditionalFacesSA(
   std::map<std::pair<int, int>, int> &v2f,
   std::set<int> &addParentFaces,
   std::vector<FacePairInfo> &facePairs)
{
   for (int f=0; f<patchTopo->GetNFaces(); ++f)
   {
      if (masterFaces.find(f) != masterFaces.end())
      {
         continue; // Already a master face
      }

      Array<int> edges, ori, verts;
      patchTopo->GetFaceEdges(f, edges, ori);
      patchTopo->GetFaceVertices(f, verts);

      MFEM_ASSERT(edges.Size() == 4 && verts.Size() == 4, "");

      const int fn1 = KnotVec(edges[0])->GetNE();
      const int fn2 = KnotVec(edges[1])->GetNE();

      // Loop over the 2 pairs of opposite sides
      for (int p=0; p<2; ++p) // Pair p
      {
         std::array<int, 2> oppEdges;
         const int sideEdge0 = edges[1 - p];
         bool bothMaster = true;
         for (int s=0; s<2; ++s)
         {
            oppEdges[s] = edges[p + 2*s];

            bool isTrueMasterEdge = false;
            if (masterEdges.count(oppEdges[s]) > 0)
            {
               const int mid = masterEdgeToId.at(oppEdges[s]);
               if (masterEdgeInfo[mid].slaves.size() != 0) { isTrueMasterEdge = true; }
            }

            if (!isTrueMasterEdge) { bothMaster = false; }
         }

         if (!bothMaster) { continue; }

         // Possibly define auxiliary and/or slave faces on this face

         // Check for auxiliary and slave edges
         std::vector<Array<int>> sideAuxEdges(2);
         std::vector<Array<int>> sideSlaveEdges(2);
         for (int s=0; s<2; ++s)
         {
            const int mid = masterEdgeToId.at(oppEdges[s]);
            for (auto edge : masterEdgeInfo[mid].slaves)
            {
               if (edge < 0)
               {
                  sideAuxEdges[s].Append(-1 - edge);
               }
               else
               {
                  sideSlaveEdges[s].Append(edge);
               }
            }
         }

         const bool hasAux = sideAuxEdges[0].Size() > 0;
         const bool hasSlave = sideSlaveEdges[0].Size() > 0;

         // Find patchTopo vertices in the interior of each side
         if (hasAux || hasSlave)
         {
            std::vector<Array<int>> edgeV(2);
            std::vector<Array<int>> edgeE(2);
            std::vector<Array<int>> edgeVki(2);

            for (int s=0; s<2; ++s)
            {
               GetMasterEdgeEntities(oppEdges[s], edgeV[s], edgeE[s], edgeVki[s]);
            }

            // Check whether the number and types of edges on opposite
            // sides match. If not, skip this pair of sides.
            if (edgeE[0].Size() != edgeE[1].Size()) { continue; }

            const int nes = edgeE[0].Size();

            {
               bool matching = true;
               for (int i=0; i<nes; ++i)
               {
                  if ((edgeE[0][i] >= 0) != (edgeE[1][i] >= 0))
                  {
                     matching = false;
                     break;
                  }
               }

               if (!matching) { continue; }
            }

            // Check whether edgeV[s] are in the same order or reversed, for
            // s=0,1.
            bool rev = true;
            {
               Array<int> sideVerts0;
               patchTopo->GetEdgeVertices(sideEdge0, sideVerts0);
               sideVerts0.Sort();

               std::array<bool, 2> found{false, false};
               Array<int> ep(2);
               for (int e=0; e<2; ++e) // Loop over ends
               {
                  for (int s=0; s<2; ++s) // Loop over sides
                  {
                     ep[s] = edgeV[s][e * (edgeV[s].Size() - 1)];

                     for (int i=0; i<2; ++i)
                     {
                        if (ep[s] == sideVerts0[i])
                        {
                           found[i] = true;
                        }
                     }
                  }

                  ep.Sort();
                  if (ep == sideVerts0)
                  {
                     rev = false;
                  }
               }

               MFEM_ASSERT(found[0] && found[1], "");
            }

            // Find auxiliary or slave subfaces of face f.
            // Note that there may be no master faces in patchTopo->ncmesh.

            for (int i=0; i<2; ++i)
            {
               MFEM_ASSERT(edgeV[i].Size() == nes + 1, "");
            }

            for (int e=0; e<nes; ++e) // Loop over edges
            {
               std::array<int, 4> fverts{edgeV[0][e], edgeV[0][e + 1],
                                         edgeV[1][rev ? nes - e - 1 : e + 1],
                                         edgeV[1][rev ? nes - e : e]};

               // Get indices with respect to the edge.
               const int eki = edgeVki[0][e];
               const int eki1 = edgeVki[0][e + 1];

               const int e1ki = edgeVki[1][rev ? nes - e : e];
               const int e1ki1 = edgeVki[1][rev ? nes - e - 1 : e + 1];

               // ori_f is the signed shift such that fverts[abs1(ori_f)] is
               // closest to vertex 0, verts[0], of parent face f, and the
               // relative direction of the ordering is encoded in the sign.
               int ori_f = 0;

               // Set the 2D knot-span indices of the 4 vertices in fverts,
               // ordered with respect to the face f.
               Array2D<int> fki(4,2);
               {
                  // Use eki to get the 2D face knot-span index.

                  // Number of elements on the edge.
                  const int eNE = edgeVki[0][edgeVki[0].Size() - 1];

                  if (p == 0)
                  {
                     MFEM_ASSERT(edgeV[0][0] == verts[0] ||
                                 edgeV[0][edgeV[0].Size() - 1] == verts[0],
                                 "");
                     MFEM_ASSERT(edgeV[1][0] == verts[3] ||
                                 edgeV[1][edgeV[0].Size() - 1] == verts[3],
                                 "");
                     MFEM_ASSERT(eNE == fn1, "");
                     MFEM_ASSERT(edgeVki[1][edgeVki[1].Size() - 1] == fn1,
                                 "");

                     const bool rev0 = edgeV[0][0] != verts[0];
                     fki(0,0) = rev0 ? eNE - eki : eki;
                     fki(0,1) = 0;

                     fki(1,0) = rev0 ? eNE - eki1 : eki1;
                     fki(1,1) = 0;

                     if (rev0) { ori_f = -2; }

                     // Other side
                     const bool rev1 = edgeV[1][0] != verts[3];

                     fki(2,0) = rev1 ? eNE - e1ki1 : e1ki1;
                     fki(2,1) = fn2;

                     fki(3,0) = rev1 ? eNE - e1ki : e1ki;
                     fki(3,1) = fn2;

                     MFEM_ASSERT(fki(0,0) == fki(3,0) &&
                                 fki(1,0) == fki(2,0), "");
                  }
                  else
                  {
                     MFEM_ASSERT(edgeV[0][0] == verts[1] ||
                                 edgeV[0][edgeV[0].Size() - 1] == verts[1],
                                 "");
                     MFEM_ASSERT(edgeV[1][0] == verts[0] ||
                                 edgeV[1][edgeV[0].Size() - 1] == verts[0],
                                 "");
                     MFEM_ASSERT(eNE == fn2, "");
                     MFEM_ASSERT(edgeVki[1][edgeVki[1].Size() - 1] == fn2,
                                 "");

                     const bool rev0 = edgeV[0][0] != verts[1];
                     fki(0,0) = fn1;
                     fki(0,1) = rev0 ? eNE - eki : eki;

                     fki(1,0) = fn1;
                     fki(1,1) = rev0 ? eNE - eki1 : eki1;

                     if (rev0)
                     {
                        ori_f = -3;
                     }
                     else
                     {
                        ori_f = 3;
                     }

                     // Other side
                     const bool rev1 = edgeV[1][0] != verts[0];

                     fki(2,0) = 0;
                     fki(2,1) = rev1 ? fn2 - e1ki1 : e1ki1;

                     fki(3,0) = 0;
                     fki(3,1) = rev1 ? fn2 - e1ki : e1ki;

                     MFEM_ASSERT(fki(0,1) == fki(3,1) &&
                                 fki(1,1) == fki(2,1), "");
                  }
               }

               // Returns the vertex with minimum knot-span indices.
               auto VertexMinKI = [&fki]()
               {
                  int id = -1;
                  {
                     std::array<int, 2> kiMin;
                     for (int j=0; j<2; ++j)
                     {
                        kiMin[j] = fki(0,j);
                        for (int i=1; i<4; ++i)
                        {
                           if (fki(i,j) < kiMin[j]) { kiMin[j] = fki(i,j); }
                        }
                     }

                     for (int i=0; i<4; ++i)
                     {
                        if (fki(i,0) == kiMin[0] && fki(i,1) == kiMin[1])
                        {
                           MFEM_ASSERT(id == -1, "");
                           id = i;
                        }
                     }
                  }

                  MFEM_ASSERT(id >= 0, "");
                  return id;
               };

               const std::pair<int, int> vpair = QuadrupleToPair(fverts);
               if (edgeE[0][e] >= 0)
               {
                  const bool vPairTopo = v2f.count(vpair) > 0;
                  if (!vPairTopo) { continue; }

                  const int sface = v2f.at(vpair);
                  addParentFaces.insert(f);

                  // Set facePairs

                  // Find the vertex with minimum knot-span indices.
                  const int vMinID = VertexMinKI();

                  std::array<int, 4> fvertsMasterOrdering;
                  for (int i=0; i<4; ++i)
                  {
                     if (ori_f >= 0)
                     {
                        fvertsMasterOrdering[i] = fverts[(vMinID + i) % 4];
                     }
                     else
                     {
                        fvertsMasterOrdering[i] = fverts[(vMinID + 4 - i) % 4];
                     }
                  }

                  const int ori_sface = GetFaceOrientation(patchTopo, sface,
                                                           fvertsMasterOrdering);

                  facePairs.emplace_back(FacePairInfo{fverts[vMinID], f,
                  SlaveFaceInfo{sface, ori_sface,
                     {fki(vMinID,0), fki(vMinID,1)},
                     {
                        fki((vMinID + 2) % 4,0) - fki(vMinID,0),
                        fki((vMinID + 2) % 4,1) - fki(vMinID,1)
                     }}});
               }
               else  // Auxiliary face
               {
                  const int afid = auxv2f.count(vpair) > 0 ?
                                   auxv2f.at(vpair) : -1;

                  addParentFaces.insert(f);

                  const int vMinID = VertexMinKI();

                  if (afid >= 0)
                  {
                     // Find orientation of ordered vertices for this face,
                     // in fvertsOrdered, w.r.t. the auxFaces ordering.
                     std::array<int, 4> fvertsOrdered, afverts;

                     int ori_f2 = -1;
                     for (int i=0; i<4; ++i)
                     {
                        afverts[i] = auxFaces[afid].v[i];
                        if (ori_f >= 0)
                        {
                           fvertsOrdered[i] = fverts[(vMinID + i) % 4];
                        }
                        else
                        {
                           fvertsOrdered[i] = fverts[(vMinID + 4 - i) % 4];
                        }

                        if (fvertsOrdered[i] == afverts[0]) { ori_f2 = i; }
                     }

                     MFEM_ASSERT(ori_f2 >= 0, "");

                     if (fvertsOrdered[(ori_f2 + 1) % 4] != afverts[1])
                     {
                        for (int j=0; j<4; ++j)
                        {
                           MFEM_ASSERT(fvertsOrdered[(ori_f2 + 4 - j) % 4]
                                       == afverts[j], "");
                        }

                        ori_f2 = -1 - ori_f2;
                     }
                     else
                     {
                        for (int j=0; j<4; ++j)
                        {
                           MFEM_ASSERT(fvertsOrdered[(ori_f2 + j) % 4]
                                       == afverts[j], "");
                        }
                     }

                     facePairs.emplace_back(FacePairInfo{fverts[vMinID], f,
                     SlaveFaceInfo{-1 - afid, ori_f2,
                        {fki(vMinID,0), fki(vMinID,1)},
                        {
                           fki((vMinID + 2) % 4,0) - fki(vMinID,0),
                           fki((vMinID + 2) % 4,1) - fki(vMinID,1)
                        }}});
                  }
                  else
                  {
                     // Create a new auxiliary face.
                     const int auxFaceId = auxFaces.size();
                     // Find the knot-span indices of the vertices in fverts,
                     // with respect to the parent face.
                     AuxiliaryFace auxFace;
                     for (int i=0; i<4; ++i)
                     {
                        if (ori_f >= 0)
                        {
                           auxFace.v[i] = fverts[(vMinID + i) % 4];
                        }
                        else
                        {
                           auxFace.v[i] = fverts[(vMinID + 4 - i) % 4];
                        }
                     }

                     // Orientation is defined as 0 for a new auxiliary face.
                     ori_f = 0;

                     auxFace.parent = f;
                     auxFace.ori = ori_f;
                     for (int i=0; i<2; ++i)
                     {
                        auxFace.ksi0[i] = fki(vMinID,i);
                        auxFace.ksi1[i] = fki((vMinID + 2) % 4,i);
                     }

                     auxv2f[vpair] = auxFaces.size();
                     auxFaces.push_back(auxFace);

                     facePairs.emplace_back(FacePairInfo{fverts[vMinID], f,
                     SlaveFaceInfo{-1 - auxFaceId, ori_f,
                        {fki(vMinID,0), fki(vMinID,1)},
                        {
                           fki((vMinID + 2) % 4,0) - fki(vMinID,0),
                           fki((vMinID + 2) % 4,1) - fki(vMinID,1)
                        }}});
                  }
               }
            }
         }
      } // Pair (p) loop
   } // f
}

void NCNURBSExtension::ProcessFacePairs(int start, int midStart,
                                        const std::vector<std::array<int, 2>> &parentSize,
                                        std::vector<int> &parentVerts,
                                        const std::vector<FacePairInfo> &facePairs)
{
   const int nfpairs = facePairs.size();
   const bool is3D = Dimension() == 3;
   MFEM_VERIFY(nfpairs > 0 || !is3D, "");
   int midPrev = -1;
   int orientation = 0;
   for (int q=start; q<nfpairs; ++q)
   {
      // We assume that j is the fast index in (i,j).
      // Note that facePairs is set by ProcessVertexToKnot3D.
      const int i = facePairs[q].info.ksi[0];
      const int j = facePairs[q].info.ksi[1];
      const int nfe1 = facePairs[q].info.ne[0]; // Number of elements, direction 1
      const int nfe2 = facePairs[q].info.ne[1]; // Number of elements, direction 2
      const int v0 = facePairs[q].v0; // Bottom-left corner vertex of child face
      const int childFace = facePairs[q].info.index;
      const int parentFace = facePairs[q].parent;
      const int cpori =
         facePairs[q].info.ori; // Orientation for childFace w.r.t. parentFace
      const int mid = masterFaceToId.at(parentFace);

      // Ignore data about master faces already processed.
      if (mid < midStart) { continue; }

      MFEM_ASSERT(0 <= i && i < parentSize[mid][0] && 0 <= j &&
                  j < parentSize[mid][1], "");
      if (mid != midPrev) // Next parent face
      {
         std::array<int, 4> pv;
         for (int k=0; k<4; ++k) { pv[k] = parentVerts[(4*mid) + k]; }
         const int ori = GetFaceOrientation(patchTopo, parentFace, pv);
         // Ori is the signed shift such that pv[abs1(ori)] is vertex 0 of
         // parentFace, and the relative direction of the ordering is encoded in
         // the sign.
         if (q > start && midPrev >= 0)
         {
            // For the previous parentFace, use previous orientation to reorder
            // masterFaceSlaves, masterFaceSlaveCorners, masterFaceSizes.
            std::array<int, 2> s0;
            masterFaceInfo[midPrev].rev = Reorder2D(orientation, s0);
            masterFaceInfo[midPrev].s0 = s0;
         }

         orientation = ori;
         midPrev = mid;
      } // next parent face

      slaveFaces.emplace_back(SlaveFaceInfo{childFace, cpori, {i, j},
         {nfe1, nfe2}});

      const int si = slaveFaces.size() - 1;
      masterFaceInfo[mid].slaves.push_back(si);
      masterFaceInfo[mid].slaveCorners.push_back(v0);
      masterFaceInfo[mid].ne[0] = parentSize[mid][0];
      masterFaceInfo[mid].ne[1] = parentSize[mid][1];
   } // Loop (q) over facePairs

   if (midPrev >= 0)
   {
      std::array<int, 2> s0;
      masterFaceInfo[midPrev].rev = Reorder2D(orientation, s0);
      masterFaceInfo[midPrev].s0 = s0;
   }
}

void NCNURBSExtension::GetAuxEdgeVertices(int auxEdge, Array<int> &verts) const
{
   verts.SetSize(2);
   for (int i=0; i<2; ++i) { verts[i] = auxEdges[auxEdge].v[i]; }
}

void NCNURBSExtension::GetAuxFaceVertices(int auxFace, Array<int> &verts) const
{
   verts.SetSize(4);
   for (int i=0; i<4; ++i) { verts[i] = auxFaces[auxFace].v[i]; }
}

void NCNURBSExtension::GetAuxFaceEdges(int auxFace, Array<int> &edges) const
{
   edges.SetSize(4);
   Array<int> verts(2);
   for (int i=0; i<4; ++i)
   {
      for (int j=0; j<2; ++j) { verts[j] = auxFaces[auxFace].v[(i + j) % 4]; }

      verts.Sort();
      const std::pair<int, int> edge_v(verts[0], verts[1]);
      // Note that v2e is a map only for conforming patchTopo->ncmesh edges.
      // Auxiliary edges are in auxv2e, not in v2e.
      if (v2e.count(edge_v) > 0)
      {
         edges[i] = v2e.at(edge_v); // patchTopo edge
      }
      else // Auxiliary edge
      {
         edges[i] = -1 - auxv2e.at(edge_v);
      }
   }
}

// Negative indices are for array `b`. Nonnegative indices are for array `a`,
// except for index `a.Size()`, corresponding to `b[0]`.
int OffsetHelper(int i, int j, const Array<int> &a, const Array<int> &b)
{
   if (i < 0)
   {
      return b[-1 - i + j];
   }
   else if (i + j < a.Size())
   {
      return a[i + j];
   }
   else
   {
      return b[0];
   }
}

int NCNURBSExtension::GetEdgeOffset(bool dof, int edge, int increment) const
{
   return OffsetHelper(edge, increment, dof ? e_spaceOffsets : e_meshOffsets,
                       dof ? aux_e_spaceOffsets : aux_e_meshOffsets);
}

int NCNURBSExtension::GetFaceOffset(bool dof, int face, int increment) const
{
   return OffsetHelper(face, increment, dof ? f_spaceOffsets : f_meshOffsets,
                       dof ? aux_f_spaceOffsets : aux_f_meshOffsets);
}

void NCNURBSExtension::GetMasterEdgeDofs(bool dof, int me,
                                         Array<int> &dofs) const
{
   MFEM_ASSERT(masterEdges.count(me) > 0, "Not a master edge");
   const int mid = masterEdgeToId.at(me);

   MFEM_ASSERT(masterEdgeInfo[mid].vertices.size() ==
               masterEdgeInfo[mid].slaves.size() - 1, "");

   const Array<int>& v_offsets = dof ? v_spaceOffsets : v_meshOffsets;
   const std::size_t nes = masterEdgeInfo[mid].slaves.size();
   for (std::size_t s=0; s<nes; ++s)
   {
      const int slaveId = slaveEdges[masterEdgeInfo[mid].slaves[s]];

      Array<int> svert;
      if (slaveId >= 0)
      {
         patchTopo->GetEdgeVertices(slaveId, svert);
      }
      else // Auxiliary edge
      {
         GetAuxEdgeVertices(-1 - slaveId, svert);
      }

      bool reverse = false;
      if (nes > 1)
      {
         const int mev = masterEdgeInfo[mid].vertices[std::max((int) s - 1,0)];
         MFEM_ASSERT(mev == svert[0] || mev == svert[1], "");
         if (s == 0)
         {
            // In this case, mev is the second vertex of the edge.
            if (svert[0] == mev) { reverse = true; }
         }
         else
         {
            // In this case, mev is the first vertex of the edge.
            if (svert[1] == mev) { reverse = true; }
         }
      }

      const int eos = GetEdgeOffset(dof, slaveId, 0);
      const int eos1 = GetEdgeOffset(dof, slaveId, 1);
      const int nvs = eos1 - eos;
      MFEM_ASSERT(nvs >= 0, "");

      // Add all slave edge vertices/DOFs

      Array<int> sdofs(nvs);
      for (int j=0; j<nvs; ++j) { sdofs[j] = reverse ? eos1 - 1 - j : eos + j; }

      dofs.Append(sdofs);

      if (s < masterEdgeInfo[mid].slaves.size() - 1)
      {
         // Add interior vertex DOF
         dofs.Append(v_offsets[masterEdgeInfo[mid].vertices[s]]);
      }
   }
}

// Set masterDofs.
void NURBSPatchMap::SetMasterEdges(bool dof, const KnotVector *kv[])
{
   edgeMaster.SetSize(edges.Size());
   edgeMasterOffset.SetSize(edges.Size());
   masterDofs.SetSize(0);

   int mos = 0;
   for (int i=0; i<edges.Size(); ++i)
   {
      edgeMaster[i] = Ext->IsMasterEdge(edges[i]);
      edgeMasterOffset[i] = mos;

      if (edgeMaster[i])
      {
         Array<int> mdof;
         Ext->GetMasterEdgeDofs(dof, edges[i], mdof);
         masterDofs.Append(mdof);
         mos += mdof.Size();
      }
   }
}

void NCNURBSExtension::GetFaceOrdering(int sf, int n1, int n2, int v0,
                                       int e1, int e2, Array<int> &perm) const
{
   perm.SetSize(n1 * n2);

   // The ordering of entities in the face is based on the vertices.

   Array<int> faceEdges, ori, evert, e2vert, vert;
   patchTopo->GetFaceEdges(sf, faceEdges, ori);
   patchTopo->GetFaceVertices(sf, vert);
   patchTopo->GetEdgeVertices(faceEdges[e1], evert);
   MFEM_ASSERT(evert[0] == v0 || evert[1] == v0, "");

   bool d[2];
   d[0] = (evert[0] == v0);

   const int v10 = d[0] ? evert[1] : evert[0];

   // The face has {fn1,fn2} interior entities, with ordering based on `vert`.
   // Now we find these sizes by first finding the edge with vertices [v0, v10].
   int e0 = -1;
   for (int i=0; i<4; ++i)
   {
      patchTopo->GetEdgeVertices(faceEdges[i], evert);
      if ((evert[0] == v0 && evert[1] == v10) ||
          (evert[1] == v0 && evert[0] == v10)) { e0 = i; }
   }

   MFEM_ASSERT(e0 >= 0, "");

   const bool tr = e0 % 2 == 1; // True means (fn1,fn2) == (n2,n1)

   patchTopo->GetEdgeVertices(faceEdges[e2], evert);
   MFEM_ASSERT(evert[0] == v10 || evert[1] == v10, "");
   d[1] = (evert[0] == v10);

   const int v11 = d[1] ? evert[1] : evert[0];

   int v01 = -1;
   for (int i=0; i<4; ++i)
   {
      if (vert[i] != v0 && vert[i] != v10 && vert[i] != v11) { v01 = vert[i]; }
   }

   MFEM_ASSERT(v01 >= 0 && v01 == vert.Sum() - v0 - v10 - v11, "");

   // Translate indices [v0, v10, v11, v01] to pairs of indices in {0,1}.
   constexpr char ipair[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
   int f00[2];

   int allv[4] = {v0, v10, v11, v01};
   int locv[4];
   for (int i=0; i<4; ++i)
   {
      locv[i] = -1;
      for (int j=0; j<4; ++j)
      {
         if (vert[j] == allv[i])
         {
            locv[i] = j;
         }
      }

      MFEM_ASSERT(locv[i] >= 0, "");
   }

   for (int i=0; i<2; ++i) { f00[i] = ipair[locv[0]][i]; }

   const int i0 = f00[0];
   const int j0 = f00[1];

   for (int i=0; i<n1; ++i)
      for (int j=0; j<n2; ++j)
      {
         // Entity perm[i] of the face should be entity i in the master face
         // ordering. The master face ordering varies faster in the direction
         // from v0 to v10, and slower in the direction from v10 to v11, or
         // equivalently, from v0 to v01.
         if (tr)
         {
            const int fi = i0 == 0 ? j : n2 - 1 - j;
            const int fj = j0 == 0 ? i : n1 - 1 - i;
            const int p = fi + (fj * n2); // Index in the slave face ordering
            const int m = i + (j * n1); // Index in the master face ordering
            perm[m] = p;
         }
         else
         {
            const int fi = i0 == 0 ? i : n1 - 1 - i;
            const int fj = j0 == 0 ? j : n2 - 1 - j;
            const int p = fi + (fj * n1); // Index in the slave face ordering
            const int m = i + (j * n1); // Index in the master face ordering
            perm[m] = p;
         }
      }
}

// Set an integer, with a check that it is uninitialized (-1) or unchanged.
bool ConsistentlySetEntry(int v, int &e)
{
   const bool consistent = e == -1 || e == v;
   e = v;
   return consistent;
}

// Reorder a 2D array to start at a corner given by (i0,j0) in {0,1}^2.
void ReorderArray2D(int i0, int j0, const Array2D<int> &a,
                    Array2D<int> &b)
{
   const int m = a.NumRows();
   const int n = a.NumCols();

   b.SetSize(m, n);

   const int s0 = i0 == 0 ? 1 : -1;
   const int s1 = j0 == 0 ? 1 : -1;
   for (int i=0; i<m; ++i)
   {
      const int ia = (i0 * (m - 1)) + (s0 * i);
      for (int j=0; j<n; ++j)
      {
         const int ja = (j0 * (n - 1)) + (s1 * j);
         b(i, j) = a(ia, ja);
      }
   }
}

// Set a quadrilateral vertex index permutation for a given orientation.
void GetVertexOrdering(int ori, std::array<int, 4> &perm)
{
   const int oriAbs = ori < 0 ? -1 - ori : ori;

   for (int i=0; i<4; ++i)
   {
      if (ori < 0)
      {
         perm[i] = (oriAbs - i + 4) % 4;
      }
      else
      {
         perm[i] = (ori + i) % 4;
      }
   }
}

// Append master face DOFs to masterDofs.
void NURBSPatchMap::SetMasterFaces(bool dof)
{
   faceMaster.SetSize(faces.Size());
   faceMasterOffset.SetSize(faces.Size());

   // The loop over master edges is already done by SetMasterEdges, and now we
   // append face DOFs to masterDofs.

   int mos = masterDofs.Size();
   for (int i=0; i<faces.Size(); ++i)
   {
      faceMaster[i] = Ext->IsMasterFace(faces[i]);
      faceMasterOffset[i] = mos;

      if (!faceMaster[i]) { continue; }

      Array2D<int> mdof;
      Ext->GetMasterFaceDofs(dof, faces[i], mdof);
      if (mdof.NumRows() == 0)
      {
         faceMaster[i] = false;
         continue;
      }

      for (int j=0; j<mdof.NumCols(); ++j)
         for (int k=0; k<mdof.NumRows(); ++k)
         {
            masterDofs.Append(mdof(k,j));
         }

      mos += mdof.NumRows() * mdof.NumCols();
   } // loop (i) over faces
}

void NCNURBSExtension::GetMasterFaceDofs(bool dof, int mf,
                                         Array2D<int> &dofs) const
{
   const int mid = masterFaceToId.at(mf);
   const bool rev = masterFaceInfo[mid].rev;
   const int s0i = masterFaceInfo[mid].s0[0];
   const int s0j = masterFaceInfo[mid].s0[1];
   const int n1orig = masterFaceInfo[mid].ne[0];
   const int n2orig = masterFaceInfo[mid].ne[1];

   // Skip master faces with no slave faces (only having slave edges).
   if (n1orig == 0 && n2orig == 0) { return; }

   const int n1 = rev ? n2orig : n1orig;
   const int n2 = rev ? n1orig : n2orig;

   MFEM_ASSERT((n1 > 1 || n2 > 1) &&
               n1 * n2 >= (int) masterFaceInfo[mid].slaves.size(),
               "Inconsistent number of faces");

   int fcnt = 0;
   for (auto slaveId : masterFaceInfo[mid].slaves)
   {
      fcnt += slaveFaces[slaveId].ne[0] * slaveFaces[slaveId].ne[1];
   }

   MFEM_VERIFY(fcnt == n1 * n2, "");
   MFEM_VERIFY((int) masterFaceInfo[mid].slaveCorners.size() <= n1 * n2, "");

   // Set an array of vertices or DOFs for the interior of this master face.
   // Set master face entity dimensions.
   int mnf1, mnf2;
   Array<int> medges;
   {
      Array<int> mori;
      patchTopo->GetFaceEdges(mf, medges, mori);
   }

   MFEM_ASSERT(medges.Size() == 4, "");

   if (dof)
   {
      mnf1 = KnotVec(medges[0])->GetNCP() - 2;
      mnf2 = KnotVec(medges[1])->GetNCP() - 2;
   }
   else
   {
      mnf1 = KnotVec(medges[0])->GetNE() - 1;
      mnf2 = KnotVec(medges[1])->GetNE() - 1;
   }

   // Set dimensions for a single mesh edge.
   const int sne1 = (mnf1 - n1 + 1) / n1;
   const int sne2 = (mnf2 - n2 + 1) / n2;

   MFEM_ASSERT(sne1 * n1 == mnf1 - n1 + 1, "");
   MFEM_ASSERT(sne2 * n2 == mnf2 - n2 + 1, "");

   const Array<int> &v_offsets = dof ? v_spaceOffsets : v_meshOffsets;

   Array2D<int> mdof(mnf1, mnf2);
   mdof = -1;

   bool consistent = true;

   for (std::size_t s=0; s<masterFaceInfo[mid].slaves.size(); ++s)
   {
      const int sId = masterFaceInfo[mid].slaves[s];
      const int slaveId = slaveFaces[sId].index;
      const int v0 = masterFaceInfo[mid].slaveCorners[s];
      const int ori = slaveFaces[sId].ori;
      // ori gives the orientation and index of cv matching the first vertex of
      // childFace, where cv is the array of slave face vertices in CCW order
      // with respect to the master face.

      const int sI = slaveFaces[sId].ksi[0];
      const int sJ = slaveFaces[sId].ksi[1];
      const int ne1 = slaveFaces[sId].ne[0];
      const int ne2 = slaveFaces[sId].ne[1];

      const int fos = GetFaceOffset(dof, slaveId, 0);
      const int fos1 = GetFaceOffset(dof, slaveId, 1);
      const int nvs = fos1 - fos;

      // These offsets are for the interior entities of the face. To get the
      // lower edge, subtract 1.
      const int os1 = (sI * sne1) + sI;
      const int os2 = (sJ * sne2) + sJ;

      std::array<int, 4> orderedVertices;
      std::array<bool, 4> edgeBdry;
      std::set<int> vbdry;
      int nf1, nf2;
      auto SetEdgeEntries = [&](int eidx, int edge, const Array<int> &evert,
                                int &vstart)
      {
         const bool reverse = (vstart == evert[1]);
         const int vend = evert.Sum() - vstart;
         vstart = vend;

         // Skip edges on the boundary of the master face.
         if (edgeBdry[eidx])
         {
            for (auto v : evert) { vbdry.insert(v); }
            return;
         }
         const bool horizontal = (eidx % 2 == 0);
         const int nf_e = horizontal ? nf1 : nf2;

         // Edge entities
         const int eos = GetEdgeOffset(dof, edge, 0);
#ifdef MFEM_DEBUG
         const int eos1 = GetEdgeOffset(dof, edge, 1);
#endif
         const bool edgeIsMaster = masterEdges.count(edge) > 0;

         Array<int> edofs;
         if (edgeIsMaster)
         {
            // This edge is a slave edge and a master edge. Instead of
            // getting DOFs from e_offsets, take them from the slave edges
            // of this edge.
            GetMasterEdgeDofs(dof, edge, edofs);
         }
         else
         {
            MFEM_ASSERT(eos1 - eos == nf_e, "");

            edofs.SetSize(nf_e);
            for (int j=0; j<nf_e; ++j)
            {
               edofs[j] = eos + j;
            }
         }

         MFEM_ASSERT(edofs.Size() == nf_e, "");

         for (int j=0; j<nf_e; ++j)
         {
            int m1, m2;
            if (eidx == 0)
            {
               m1 = os1 + j;
               m2 = os2 - 1;
            }
            else if (eidx == 1)
            {
               m1 = os1 + nf1;
               m2 = os2 + j;
            }
            else if (eidx == 2)
            {
               m1 = os1 + nf1 - 1 - j;
               m2 = os2 + nf2;
            }
            else
            {
               m1 = os1 - 1;
               m2 = os2 + nf2 - 1 - j;
            }

            if (!ConsistentlySetEntry(reverse ? edofs[nf_e - 1 - j]
                                      : edofs[j], mdof(m1, m2)))
            {
               consistent = false;
            }
         }
      };

      if (slaveId < 0)
      {
         // Auxiliary face
         const int auxFace = -1 - slaveId;

         // Set slave face entity dimensions.
         if (dof)
         {
            nf1 = (sne1 * ne1) + ne1 - 1;
            nf2 = (sne2 * ne2) + ne2 - 1;
         }
         else
         {
            nf1 = auxFaces[auxFace].ksi1[0] - auxFaces[auxFace].ksi0[0] - 1;
            nf2 = auxFaces[auxFace].ksi1[1] - auxFaces[auxFace].ksi0[1] - 1;
         }

         MFEM_VERIFY(sne1 * ne1 == nf1 - ne1 + 1 &&
                     sne2 * ne2 == nf2 - ne2 + 1 &&
                     nvs == nf1 * nf2, "");

         // If ori >= 0, then vertex ori of the aux face is closest to vertex 0
         // of the parent face. If ori < 0, then the orientations of the aux face
         // and parent face are reversed.

         // NOTE: When an aux face is first defined, it is constructed with
         // orientation 0 w.r.t. its parent face. However, it can be part of
         // another parent face. In this case, the original aux face index is
         // used, but it is paired with a different parent face, with
         // possibly different orientation w.r.t. that parent face. In
         // NURBSPatchMap::SetMasterFaces, the DOFs are set on the parent
         // face in its knot-span indices, with an aux face of possibly nonzero
         // orientation. When the orientation is nonzero, the aux face DOFs
         // must be reordered for the parent face, based on orientation.

         int onf1, onf2;
         GetInverseShiftedDimensions2D(ori, nf1, nf2, onf1, onf2);
         for (int k=0; k<onf2; ++k)
            for (int j=0; j<onf1; ++j)
            {
               int sm, sn, sj, sk;
               GetShiftedGridPoints2D(onf1, onf2, j, k, ori, sm, sn, sj, sk);
               const int q = j + (k * onf1);
               if (!ConsistentlySetEntry(fos + q, mdof(os1 + sj, os2 + sk)))
               {
                  consistent = false;
               }
            }

         // Set entries on edges of this face, if interior to the master face.

         // Horizontal edges
         edgeBdry[0] = sJ == 0;
         edgeBdry[2] = sJ + ne2 == n2;

         // Vertical edges
         edgeBdry[1] = sI + ne1 == n1;
         edgeBdry[3] = sI == 0;

         Array<int> faceEdges;
         GetAuxFaceEdges(auxFace, faceEdges);

         std::array<int, 4> perm;
         GetVertexOrdering(ori, perm);

         int vstart = v0;
         for (int eidx=0; eidx<4; ++eidx)
         {
            orderedVertices[eidx] = vstart;
            MFEM_ASSERT(orderedVertices[eidx] ==
                        auxFaces[auxFace].v[perm[eidx]], "");
            const int eperm = ori < 0 ? (perm[eidx] - 1 + 4) % 4 : perm[eidx];
            const int edge = faceEdges[eperm];
            Array<int> evert;
            if (edge >= 0)
            {
               patchTopo->GetEdgeVertices(edge, evert);
            }
            else
            {
               const int auxEdge = -1 - edge;
               GetAuxEdgeVertices(auxEdge, evert);
            }
            MFEM_ASSERT(evert[0] == vstart || evert[1] == vstart, "");
            SetEdgeEntries(eidx, edge, evert, vstart);
         } // eidx
      }
      else // slaveId >= 0
      {
         // Determine which slave face edges are in the first and second
         // dimensions of the master face, by using ori.
         int e1 = -1, e2 = -1;
         {
            const int aori = ori < 0 ? -1 - ori : ori;
            if (aori % 2 == 0)
            {
               e1 = 0;
               e2 = 1;
            }
            else
            {
               e1 = 1;
               e2 = 0;
            }

            if (ori < 0)
            {
               // Swap e1, e2
               const int sw = e1;
               e1 = e2;
               e2 = sw;
            }
         }

         // Now, e1 is one of the two horizontal edges in the master face
         // directions. If it does not touch v0, then take the other horizontal
         // edge. Do the same for e2.

         Array<int> sedges;
         {
            Array<int> sori;
            patchTopo->GetFaceEdges(slaveId, sedges, sori);
         }

         int v1 = -1;
         {
            Array<int> evert;
            patchTopo->GetEdgeVertices(sedges[e1], evert);
            if (evert.Find(v0) == -1)
            {
               e1 += 2;
               patchTopo->GetEdgeVertices(sedges[e1], evert);
            }

            const int idv0 = evert.Find(v0);
            MFEM_ASSERT(idv0 >= 0, "");
            v1 = evert[1 - idv0];

            patchTopo->GetEdgeVertices(sedges[e2], evert);
            if (evert.Find(v1) == -1)
            {
               e2 += 2;
               patchTopo->GetEdgeVertices(sedges[e2], evert);
            }

            MFEM_ASSERT(evert.Find(v1) >= 0, "");
         }

         // Set slave face entity dimensions.
         if (dof)
         {
            nf1 = KnotVec(sedges[e1])->GetNCP() - 2;
            nf2 = KnotVec(sedges[e2])->GetNCP() - 2;
         }
         else
         {
            nf1 = KnotVec(sedges[e1])->GetNE() - 1;
            nf2 = KnotVec(sedges[e2])->GetNE() - 1;
         }

         MFEM_ASSERT(sne1 * ne1 == nf1 - ne1 + 1, "");
         MFEM_ASSERT(sne2 * ne2 == nf2 - ne2 + 1, "");

         MFEM_ASSERT(nvs == nf1 * nf2, "");

         // Find the DOFs of the slave face ordered for the master face. We know
         // that e1 and e2 are the local indices of the slave face edges on the
         // bottom and right side, with respect to the master face directions.
         Array<int> perm;
         GetFaceOrdering(slaveId, nf1, nf2, v0, e1, e2, perm);

         for (int k=0; k<nf2; ++k)
            for (int j=0; j<nf1; ++j)
            {
               const int q = j + (k * nf1);
               if (!ConsistentlySetEntry(fos + perm[q],
                                         mdof(os1 + j, os2 + k)))
               {
                  consistent = false;
               }
            }

         // Set entries on edges of this face, if interior to the master face.
         std::array<int, 4> edgeOrder{e1, e2, (e1 + 2) % 4, (e2 + 2) % 4};

         // Horizontal edges
         edgeBdry[0] = sJ == 0;
         edgeBdry[2] = sJ + ne2 == n2;

         // Vertical edges
         edgeBdry[1] = sI + ne1 == n1;
         edgeBdry[3] = sI == 0;

         int vstart = v0;
         for (int eidx=0; eidx<4; ++eidx)
         {
            orderedVertices[eidx] = vstart;
            const int edge = sedges[edgeOrder[eidx]];
            Array<int> evert;
            patchTopo->GetEdgeVertices(edge, evert);
            SetEdgeEntries(eidx, edge, evert, vstart);
         } // eidx
      }

      // Set entries at vertices of this face, if interior to the master face.
      for (int vidx=0; vidx<4; ++vidx)
      {
         const int v = orderedVertices[vidx];
         if (vbdry.count(v) == 0) // If not on the master face boundary.
         {
            int m1, m2;
            if (vidx == 0)
            {
               m1 = os1 - 1;
               m2 = os2 - 1;
            }
            else if (vidx == 1)
            {
               m1 = os1 + nf1;
               m2 = os2 - 1;
            }
            else if (vidx == 2)
            {
               m1 = os1 + nf1;
               m2 = os2 + nf2;
            }
            else
            {
               m1 = os1 - 1;
               m2 = os2 + nf2;
            }

            if (!ConsistentlySetEntry(v_offsets[v], mdof(m1, m2)))
            {
               consistent = false;
            }
         }
      } // vidx
   } // Loop (s) over slave faces.

   // Let `ori` be the signed shift such that pv[abs1(ori)] is vertex 0 of
   // parentFace, and the relative direction of the ordering is encoded in the
   // sign. Here, pv means parent vertices, as ordered in the mesh file. Then
   // Reorder2D takes `ori` and computes (s0i, s0j) as the corresponding integer
   // coordinates in {0,1}x{0,1}. Thus reference vertex (s0i, s0j) of pv (parent
   // vertices in mesh file) is vertex (0, 0) of parentFace. Currently, mdof is
   // in the ordering of pv, and the entries are now reordered, according to
   // parentFace vertex ordering, for appending to masterDofs. That means the
   // first entry appended to masterDofs should be the entry of mdof
   // corresponding to (s0i, s0j).

   ReorderArray2D(s0i, s0j, mdof, dofs);
   const bool all_set = dofs.NumRows() * dofs.NumCols() == 0 || dofs.Min() >= 0;
   MFEM_VERIFY(all_set && consistent, "");
}

int NURBSPatchMap::GetMasterEdgeDof(const int e, const int i) const
{
   const int os = edgeMasterOffset[e];
   return masterDofs[os + i];
}

int NURBSPatchMap::GetMasterFaceDof(const int f, const int i) const
{
   const int os = faceMasterOffset[f];
   return masterDofs[os + i];
}

void NCNURBSExtension::ProcessVertexToKnot2D(const VertexToKnotSpan &v2k,
                                             std::set<int> &reversedParents,
                                             std::vector<EdgePairInfo> &edgePairs)
{
   auxEdges.clear();
   auxv2e.clear();

   const int nv2k = v2k.Size();

   int prevParent = -1;
   int prevV = -1;
   int prevKI = -1;
   for (int i=0; i<nv2k; ++i)
   {
      int tv, ks;
      std::array<int, 2> pv;
      v2k.GetVertex2D(i, tv, ks, pv);

      // Given that the parent Mesh is not yet constructed, and all we have at
      // this point is patchTopo->ncmesh, we should only define master/slave
      // edges by indices in patchTopo->ncmesh, as done in the case of nonempty
      // nce.masters. Now find the edge in patchTopo->ncmesh with vertices
      // (pv[0], pv[1]), and define it as a master edge.

      const std::pair<int, int> parentPair(pv[0] < pv[1] ? pv[0] : pv[1],
                                           pv[0] < pv[1] ? pv[1] : pv[0]);

      MFEM_ASSERT(v2e.count(parentPair) > 0, "Vertex pair not found");
      const int parentEdge = v2e[parentPair];
      masterEdges.insert(parentEdge);

      const int kv = KnotInd(parentEdge);
      parentToKV[parentPair] = std::array<int, 2> {kv, -1};

      const bool rev = pv[1] < pv[0];
      if (rev) { reversedParents.insert(parentEdge); }

      // Note that the logic here assumes that the "vertex_to_knotspan" data in
      // the mesh file has vertices in order of ascending knotIndex.

      const bool newParentEdge = (prevParent != parentEdge);
      const int v0 = newParentEdge ? pv[0] : prevV;

      if (ks == 1) { MFEM_ASSERT(newParentEdge, ""); }

      // Find the edge in patchTopo->ncmesh with vertices (v0, tv), and define
      // it as a slave edge.

      const std::pair<int, int> childPair(v0 < tv ? v0 : tv, v0 < tv ? tv : v0);
      const bool childPairTopo = v2e.count(childPair) > 0;
      if (!childPairTopo)
      {
         // Check whether childPair is in auxEdges.
         if (auxv2e.count(childPair) == 0)
         {
            // Create a new auxiliary edge
            auxv2e[childPair] = auxEdges.size();
            auxEdges.emplace_back(AuxiliaryEdge{pv[0] < pv[1] ?
                                                parentEdge : -1 - parentEdge,
            {childPair.first, childPair.second},
            {newParentEdge ? 0 : prevKI, ks}});
         }
      }

      const int childEdge = childPairTopo ? v2e[childPair] : -1 - auxv2e[childPair];

      // Check whether this is the final vertex in this parent edge. Note that
      // the logic for comparing (pv[0],pv[1]) to the next parents assumes the
      // ordering does not change, which is ensured by the assumption that the
      // knot-span index is increasing.
      bool finalVertex = (i == nv2k-1);
      if (i < nv2k-1)
      {
         int tv_next, ks_next;
         std::array<int, 2> pv_next;
         v2k.GetVertex2D(i + 1, tv_next, ks_next, pv_next);
         if (pv_next[0] != pv[0] || pv_next[1] != pv[1])
         {
            finalVertex = true;
         }
      }

      edgePairs.emplace_back(tv, ks, childEdge, parentEdge);

      if (finalVertex)
      {
         // Also find the edge with vertices (tv, pv[1]), and define it as a
         // slave edge.
         const std::pair<int, int> finalChildPair(tv < pv[1] ? tv : pv[1],
                                                  tv < pv[1] ? pv[1] : tv);
         const bool finalChildPairTopo = v2e.count(finalChildPair) > 0;
         if (!finalChildPairTopo)
         {
            // Check whether finalChildPair is in auxEdges.
            if (auxv2e.count(finalChildPair) == 0)
            {
               // Create a new auxiliary edge
               auxv2e[finalChildPair] = auxEdges.size();

               // -1 denotes `ne` at endpoint
               auxEdges.emplace_back(AuxiliaryEdge{pv[0] < pv[1] ?
                                                   -1 - parentEdge : parentEdge,
               {finalChildPair.first, finalChildPair.second},
               {ks, -1}});
            }
         }

         const int finalChildEdge = finalChildPairTopo ? v2e[finalChildPair] :
                                    -1 - auxv2e[finalChildPair];
         edgePairs.emplace_back(-1, -1, finalChildEdge, parentEdge);
      }

      prevV = tv;
      prevKI = ks;
      prevParent = parentEdge;
   } // loop over vertices in vertex_to_knotspan
}

void NCNURBSExtension::ProcessVertexToKnot3D(
   const VertexToKnotSpan &v2k,
   const std::map<std::pair<int, int>, int> &v2f,
   std::vector<std::array<int, 2>> &parentSize,
   std::vector<EdgePairInfo> &edgePairs,
   std::vector<FacePairInfo> &facePairs,
   std::vector<int> &parentFaces,
   std::vector<int> &parentVerts)
{
   auxEdges.clear();
   auxFaces.clear();
   auxv2e.clear();
   auxv2f.clear();

   const int nv2k = v2k.Size();

   // Note that the logic here assumes that the "vertex_to_knotspan" data in the
   // mesh file has vertices in order of ascending (k1,k2), with k2 being the
   // fast variable, and with corners skipped.

   // Find parentOffset, which stores the indices in v2k at which parent faces
   // start.
   int prevParent = -1;
   std::vector<int> parentOffset;
   std::vector<bool> parentV2Kedge;
   int n1 = 0;
   int n2 = 0;
   int n1min = 0;
   int n2min = 0;
   for (int i = 0; i < nv2k; ++i)
   {
      int tv;
      std::array<int, 2> ks;
      std::array<int, 4> pv;
      v2k.GetVertex3D(i, tv, ks, pv);

      // The face with vertices (pv[0], pv[1], pv[2], pv[3]) is defined as a
      // parent face.
      const std::pair<int, int> parentPair = QuadrupleToPair(pv);
      const int parentFace = v2f.at(parentPair);
      const bool newParentFace = (prevParent != parentFace);
      if (newParentFace)
      {
         parentOffset.push_back(i);
         parentFaces.push_back(parentFace);

         // Find the knotvectors for the first two edges this face.
         {
            Array<int> edges, ori, verts;
            patchTopo->GetFaceEdges(parentFace, edges, ori);
            patchTopo->GetFaceVertices(parentFace, verts);

            std::array<int,2> kv = {-1, -1};
            for (int e=0; e<2; ++e)
            {
               // Find the edge with vertices pv[e] and pv[e+1].
               for (auto edge : edges)
               {
                  Array<int> evert;
                  patchTopo->GetEdgeVertices(edge, evert);
                  const bool matching = (evert[0] == pv[e] && evert[1] == pv[e+1]) ||
                                        (evert[1] == pv[e] && evert[0] == pv[e+1]);
                  if (matching) { kv[e] = KnotInd(edge); }
               }

               MFEM_ASSERT(kv[e] >= 0, "");
            }

            parentToKV[parentPair] = std::array<int, 2> {kv[0], kv[1]};
         }

         if (i > 0)
         {
            // In the case of only 1 element in the 1-direction, it is assumed
            // that the 2-direction has more than 1 element, so there are
            // knot-spans (0, ki2) and (1, ki2) for 0 < ki2 < n2. This will
            // result in n1 = 0, which should be 1. Also, n2 will be 1 less than
            // it should be. Similarly for the situation with directions
            // reversed.
            const int n1range = n1 - n1min;
            const int n2range = n2 - n2min;
            parentV2Kedge.push_back(n1range == 0 || n2range == 0);
         }

         auto getEdgeNE = [&](int d)
         {
            Array<int> ev(2);
            for (int j=0; j<2; ++j) { ev[j] = pv[j + d]; }
            ev.Sort();
            return KnotVecNE(v2e.at(std::pair<int, int>(ev[0], ev[1])));
         };
         parentSize.emplace_back(std::array<int, 2> {getEdgeNE(0), getEdgeNE(1)});

         n1 = ks[0]; // Finding max of ks[0]
         n2 = ks[1]; // Finding max of ks[1]

         n1min = n1;
         n2min = n2;
      }
      else
      {
         n1 = std::max(n1, ks[0]); // Finding max of ks[0]
         n2 = std::max(n2, ks[1]); // Finding max of ks[1]

         n1min = std::min(n1min, ks[0]);
         n2min = std::min(n2min, ks[1]);
      }

      prevParent = parentFace;
   }

   {
      const int n1range = n1 - n1min;
      const int n2range = n2 - n2min;
      parentV2Kedge.push_back(n1range == 0 || n2range == 0);
   }

   const int numParents = parentOffset.size();
   parentOffset.push_back(nv2k);

   std::set<int> visitedParentEdges;
   std::map<int, int> edgePairOS;
   bool consistent = true;

   for (int parent = 0; parent < numParents; ++parent)
   {
      const int parentFace = parentFaces[parent];

      int parentEdges[4];
      bool parentEdgeRev[4];

      int tvi;
      std::array<int, 2> ks;
      std::array<int, 4> pv;
      v2k.GetVertex3D(parentOffset[parent], tvi, ks, pv);

      // Set all 4 edges of the parent face as master edges.
      {
         Array<int> ev(2);
         for (int i=0; i<4; ++i)
         {
            for (int j=0; j<2; ++j)
            {
               ev[j] = pv[(i + j) % 4];
            }

            const bool reverse = (ev[1] < ev[0]);
            parentEdgeRev[i] = reverse;

            ev.Sort();

            const std::pair<int, int> edge_i(ev[0], ev[1]);

            const int parentEdge = v2e.at(edge_i);
            masterEdges.insert(parentEdge);
            parentEdges[i] = parentEdge;
         }
      }

      n1 = parentSize[parent][0];
      n2 = parentSize[parent][1];
      Array2D<int> gridVertex(n1 + 1, n2 + 1);
      gridVertex = -1;

      gridVertex(0,0) = pv[0];
      gridVertex(n1,0) = pv[1];
      gridVertex(n1,n2) = pv[2];
      gridVertex(0,n2) = pv[3];

      for (int i=0; i<4; ++i) { parentVerts.push_back(pv[i]); }

      int r1min = -1;
      int r1max = -1;
      int r2min = -1;
      int r2max = -1;

      for (int i = parentOffset[parent]; i < parentOffset[parent + 1]; ++i)
      {
         v2k.GetVertex3D(i, tvi, ks, pv);
         gridVertex(ks[0], ks[1]) = tvi;
         if (i == parentOffset[parent])
         {
            // Initialize min/max
            r1min = ks[0];
            r1max = ks[0];

            r2min = ks[1];
            r2max = ks[1];
         }
         else
         {
            r1min = std::min(r1min, ks[0]);
            r1max = std::max(r1max, ks[0]);

            r2min = std::min(r2min, ks[1]);
            r2max = std::max(r2max, ks[1]);
         }
      } // loop over vertices in v2k

      MFEM_ASSERT((r1max - r1min + 1) * (r2max - r2min + 1) >=
                  parentOffset[parent + 1] - parentOffset[parent], "");

      std::array<int,2> kvi;
      if (kvf.size() > 0)
      {
         const std::pair<int, int> parentPair = v2k.GetVertexParentPair(
                                                   parentOffset[parent]);
         std::array<int, 2> kv = parentToKV.at(parentPair);
         for (int i=0; i<2; ++i) { kvi[i] = kv[i]; }
      }

      // Default refinement factor
      const int rf = ref_factors.Size() == 3 ? ref_factors[0] : 1;

      int n1orig = kvf_coarse.size() > 0 ? kvf_coarse[kvi[0]].Size() : n1 / rf;
      int n2orig = kvf_coarse.size() > 0 ? kvf_coarse[kvi[1]].Size() : n2 / rf;

      if (kvf.size() > 0 && kvf_coarse.size() == 0)
      {
         n1orig = kvf[kvi[0]].Size();
         n2orig = kvf[kvi[1]].Size();
      }

      if (kvf.size() > 0)
      {
         MFEM_ASSERT(kvf[kvi[0]].Sum() == n1 && kvf[kvi[1]].Sum() == n2, "");
      }

      std::vector<Array<int>> cgrid(2);
      std::array<int,2> n_orig = {n1orig, n2orig};
      for (int dir=0; dir<2; ++dir)
      {
         cgrid[dir].SetSize(n_orig[dir] + 1);
         cgrid[dir][0] = 0;
         for (int ii = 0; ii < n_orig[dir]; ++ii)
         {
            const int iir = parentEdgeRev[dir] ? n_orig[dir] - 1 - ii : ii;

            int d = 1; // refinement factor

            if (kvf_coarse.size() > 0)
            {
               d = kvf_coarse[kvi[dir]][iir];
            }
            else if (kvf.size() > 0)
            {
               d = kvf[kvi[dir]][iir];
            }

            cgrid[dir][ii + 1] = cgrid[dir][ii] + d;
         }
      }

      MFEM_ASSERT(cgrid[0][n_orig[0]] == n1 && cgrid[1][n_orig[1]] == n2, "");

      bool allset = true;
      bool hasSlaveFaces = false;
      bool hasAuxFace = false;

      for (int ii=0; ii<=n_orig[0]; ++ii)
      {
         const int i = cgrid[0][ii];
         for (int jj=0; jj<=n_orig[1]; ++jj)
         {
            const int j = cgrid[1][jj];
            if (gridVertex(i,j) < 0)
            {
               allset = false;
            }
            else if (0 < i && i < n1 && 0 < j && j < n2)
            {
               hasSlaveFaces = true;
            }
         }
      }

      auto SetFacePairOnGridRange = [&](int i0, int i1, int j0, int j1)
      {
         std::array<int, 4> cv{gridVertex(i0, j0), gridVertex(i1, j0),
                               gridVertex(i1, j1), gridVertex(i0, j1)};
         const std::pair<int, int> childPair = QuadrupleToPair(cv);

         // min(cv) may be negative, if gridVertex is not set everywhere.
         if (childPair.first < 0) { return; }

         const int d0 = i1 - i0;
         const int d1 = j1 - j0;

         const bool childPairTopo = v2f.count(childPair) > 0;
         if (childPairTopo)
         {
            const int childFace = v2f.at(childPair);
            const int ori = GetFaceOrientation(patchTopo, childFace, cv);
            // ori gives the orientation and index of cv matching the first
            // vertex of childFace.
            facePairs.emplace_back(
               FacePairInfo{cv[0], parentFace, SlaveFaceInfo{childFace,
                                                             ori, {i0, j0}, {d0, d1}}});
         }
         else
         {
            // Check whether the parent face is on the boundary.
            const Mesh::FaceInformation faceInfo = patchTopo->GetFaceInformation(
                                                      parentFace);
            const bool bdryParentFace = faceInfo.IsBoundary();

            if (!allset && !bdryParentFace)
            {
               hasAuxFace = true;
               // Check whether childPair is in auxFaces.
               if (auxv2f.count(childPair) == 0)
               {
                  // Create a new auxiliary face
                  auxv2f[childPair] = auxFaces.size();
                  AuxiliaryFace auxFace;
                  for (int k=0; k<4; ++k) { auxFace.v[k] = cv[k]; }

                  auxFace.parent = parentFace;
                  // Orientation is defined as 0 for a new auxiliary face.
                  auxFace.ori = 0;
                  auxFace.ksi0[0] = i0;
                  auxFace.ksi0[1] = j0;
                  auxFace.ksi1[0] = i1;
                  auxFace.ksi1[1] = j1;

                  auxFaces.push_back(auxFace);
                  facePairs.emplace_back(
                     FacePairInfo{cv[0], parentFace,
                                  SlaveFaceInfo{-1 - auxv2f[childPair],
                                                0, {i0, j0}, {d0, d1}}});
               }
            }
         }
      };

      // Loop over child faces and set facePairs, as well as auxiliary faces.
      for (int ii=0; ii<n_orig[0]; ++ii)
      {
         const int i = cgrid[0][ii];
         const int i1 = cgrid[0][ii + 1];
         for (int jj=0; jj<n_orig[1]; ++jj)
         {
            const int j = cgrid[1][jj];
            const int j1 = cgrid[1][jj + 1];
            SetFacePairOnGridRange(i, i1, j, j1);
         }
      }

      // Loop over child boundary edges and set edgePairs.
      for (int dir=1; dir<=2; ++dir)
      {
         const int ne = dir == 1 ? n1 : n2;
         for (int s=0; s<2; ++s) // Loop over 2 sides for this direction.
         {
            const int parentEdge = parentEdges[dir == 1 ? 2*s : (2*s) + 1];
            const bool reverse_p = parentEdgeRev[dir == 1 ? 2*s : (2*s) + 1];
            // Sides with s=1 are reversed in defining parentEdgeRev.
            const bool reverse = s == 0 ? reverse_p : !reverse_p;
            const bool parentVisited = visitedParentEdges.count(parentEdge) > 0;

            if (!parentVisited)
            {
               edgePairOS[parentEdge] = edgePairs.size();
               edgePairs.resize(edgePairs.size() + ne);
            }

            int tvprev = -1;
            int kiprev = -1;
            bool lagTV = false;
            int firstEdge = -1;
            int os_e = 0;

            // Loop edges in direction `dir`
            for (int e_orig = 0; e_orig < n_orig[dir-1]; ++e_orig)
            {
               const int e_i = os_e;
               const int e_orig_rev = reverse ? n_orig[dir-1] - 1 - e_orig : e_orig;
               int de = rf;
               if (kvf_coarse.size() > 0)
               {
                  de = kvf_coarse[kvi[dir - 1]][e_orig_rev];
               }
               else if (kvf.size() > 0)
               {
                  de = kvf[kvi[dir - 1]][e_orig_rev];
               }

               os_e += de;

               // For both directions, side s=0 has increasing indices and s=1
               // has decreasing indices.

               const int i0 = e_i;
               const int i1 = e_i + de;

               // Edge index with respect to the master edge.
               const int e_idx = reverse ? ne - e_i - de : e_i;

               Array<int> cv(2);
               if (dir == 1)
               {
                  cv[0] = gridVertex(i0,s*n2);
                  cv[1] = gridVertex(i1,s*n2);
               }
               else
               {
                  cv[0] = gridVertex((1-s)*n1, i0);
                  cv[1] = gridVertex((1-s)*n1, i1);
               }

               const int cv0 = cv[0];
               int tv_int = -1; // Top-vertex interior to the master edge
               int ki = -1; // Knot-span index of tv_int, w.r.t. the master edge

               if (lagTV)
               {
                  tv_int = tvprev;
                  ki = kiprev;
               }

               if (tvprev == -1)
               {
                  kiprev = (i0 == 0 || i0 == ne) ? i1 : i0;
                  // Top-vertex interior to the master edge
                  tvprev = (i0 == 0 || i0 == ne) ? cv[1] : cv[0];
               }
               else if (e_i < ne - 1) // Don't set to the endpoint
               {
                  kiprev = (tvprev == cv[0]) ? i1 : i0;
                  // Next interior vertex along the master edge
                  tvprev = (tvprev == cv[0]) ? cv[1] : cv[0];
               }

               if (!lagTV)
               {
                  tv_int = tvprev;
                  ki = kiprev;
               }

               cv.Sort();
               if (cv[0] < 0) // may occur if gridVertex is not set everywhere.
               {
                  continue;
               }
               else if (firstEdge == -1)
               {
                  firstEdge = e_i;
                  if (e_i > 0)
                  {
                     tv_int = cv0;
                     ki = i0;
                     tvprev = cv.Sum() - cv0; // cv1
                     kiprev = i1;
                     lagTV = true;
                  }
               }

               const int tv = (e_idx == ne - de) ? -1 : tv_int;
               const int tvki = (e_idx == ne - de) ? -1 : (reverse ? ne - ki : ki);

               const std::pair<int, int> edge_i(cv[0], cv[1]);
               const int childEdge = v2e.at(edge_i);

               if (tv == -1) { lagTV = true; }

               if (!parentVisited)
               {
                  // edgePairs is ordered starting from the vertex of lower index.
                  edgePairs[edgePairOS[parentEdge] + e_idx].Set(tv, tvki,
                                                                childEdge,
                                                                parentEdge);
               }
               else
               {
                  // Consistency check
                  const int os = edgePairOS[parentEdge];
                  if (edgePairs[os + e_idx].child != childEdge ||
                      edgePairs[os + e_idx].parent != parentEdge)
                  {
                     consistent = false;
                  }
               }
            }

            visitedParentEdges.insert(parentEdge);
         }
      } // dir

      // Set auxiliary and patch-slave faces outside the set gridVertex. Here,
      // patch-slave refers to a slave face that is a face of a neighboring
      // patch and may contain multiple mesh faces. In general, there can be at
      // most 8 = 3^2 - 1 such faces.

      std::array<int, 4> gv1 = {0, r1min, r1max, n1};
      std::array<int, 4> gv2 = {0, r2min, r2max, n2};

      if (hasSlaveFaces && !allset)
      {
         for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j)
            {
               // Skip the middle, which is covered by gridVertex.
               if (i == 1 && j == 1) { continue; }

               // Skip degenerate faces
               if (gv1[i] == gv1[i+1] || gv2[j] == gv2[j+1]) { continue; }

               // Define auxiliary face (gv1[i], gv1[i+1]) x (gv2[j], gv2[j+1])
               SetFacePairOnGridRange(gv1[i], gv1[i+1], gv2[j], gv2[j+1]);
            }
      }

      // Set auxiliary edges outside the gridVertex, on the boundary of the
      // parent face. Note that auxiliary edges cannot simply be found as edges
      // of auxiliary faces, because the faces above are defined only on parent
      // faces listed in V2K data. Other auxiliary faces will be defined in
      // FindAdditionalFacesSA by using auxiliary edges found below.

      // Auxiliary edges in first and second directions
      for (int d=0; d<2; ++d)
      {
         for (int i=0; i<3; ++i)
         {
            if (i == 1) // Skip the middle, which is covered by gridVertex.
            {
               continue;
            }

            for (int s=0; s<2; ++s) // Loop over 2 sides in this direction
            {
               // Set gv1 (d == 0) or gv2 (d == 1) for this edge.
               // Set range of set knot-span indices for this edge.
               int rmin = -1;
               int rmax = -1;
               const int n_d = d == 0 ? n1 : n2;
               for (int j=1; j<n_d; ++j)
               {
                  const int tv = d == 0 ? gridVertex(j,s*n2) :
                                 gridVertex((1-s)*n1,j);
                  if (tv >= 0)
                  {
                     if (rmin == -1)
                     {
                        // Initialize range
                        rmin = j;
                        rmax = j;
                     }
                     else
                     {
                        rmin = std::min(rmin, j);
                        rmax = std::max(rmax, j);
                     }
                  }
               }

               if (rmax == -1)
               {
                  // No vertices set in gridVertex on the interior of this edge.
                  continue;
               }

               if (d == 0)
               {
                  gv1[1] = rmin;
                  gv1[2] = rmax;
               }
               else
               {
                  gv2[1] = rmin;
                  gv2[2] = rmax;
               }

               const int pid = d == 0 ? 2*s : (2*s) + 1; // Parent index
               const bool reverse_p = parentEdgeRev[pid];
               // Sides with s=1 are reversed in defining parentEdgeRev.
               const bool reverse = s == 0 ? reverse_p : !reverse_p;

               // Define an auxiliary edge (gv1[i], gv1[i+1])
               Array<int> cv(2);
               std::array<int, 2> ki;

               if (d == 0)
               {
                  cv[0] = gridVertex(gv1[i],s*n2);
                  cv[1] = gridVertex(gv1[i+1],s*n2);

                  ki[0] = gv1[i];
                  ki[1] = gv1[i+1];
               }
               else
               {
                  cv[0] = gridVertex((1-s)*n1,gv2[i]);
                  cv[1] = gridVertex((1-s)*n1,gv2[i+1]);

                  ki[0] = gv2[i];
                  ki[1] = gv2[i+1];
               }

               if (cv[0] == cv[1])
               {
                  continue;
               }

               // Top-vertex interior to the master edge.
               const int tv = i == 0 ? cv[1] : cv[0];
               const int tvki_f = i == 0 ? ki[1] : ki[0]; // face index
               const int tvki = reverse ? n_d - tvki_f : tvki_f; // edge index

               cv.Sort();
               MFEM_ASSERT(cv[0] >= 0, "");

               const std::pair<int, int> childPair(cv[0], cv[1]);
               const bool childPairTopo = v2e.count(childPair) > 0;
               if (!childPairTopo)
               {
                  const int pv0 = d == 0 ? gridVertex(0,s*n2) :
                                  gridVertex((1-s)*n1,0);
                  const int pv1 = d == 0 ? gridVertex(n1,s*n2) :
                                  gridVertex((1-s)*n1,n2);
                  const std::pair<int, int> parentPair(pv0 < pv1 ? pv0 : pv1,
                                                       pv0 < pv1 ? pv1 : pv0);
                  const int parentEdge = v2e.at(parentPair);
                  MFEM_ASSERT(parentEdges[pid] == parentEdge, "");

                  // Check whether childPair is in auxEdges.
                  if (auxv2e.count(childPair) == 0)
                  {
                     const int knotIndex0 = (d == 0) ? gv1[i] : gv2[i];
                     const int knotIndex1 = (d == 0) ? gv1[i+1] : gv2[i+1];

                     // Create a new auxiliary edge
                     auxv2e[childPair] = auxEdges.size();
                     auxEdges.emplace_back(AuxiliaryEdge{pv0 < pv1 ?
                                                         parentEdge :
                                                         -1 - parentEdge,
                     {childPair.first, childPair.second},
                     {knotIndex0, knotIndex1}});
                  }

                  const bool start = (i == 0 && !reverse) || (i != 0 && reverse);
                  int end_idx = kvf_coarse.size() > 0 ?
                                (start ? 0 : kvf_coarse[kvi[d]].Size() - 1) : 0;
                  int de = kvf_coarse.size() > 0 ? kvf_coarse[kvi[d]][end_idx] : rf;
                  if (kvf.size() > 0 && kvf_coarse.size() == 0)
                  {
                     end_idx = start ? 0 : kvf[kvi[d]].Size() - 1;
                     de = kvf[kvi[d]][end_idx];
                  }

                  const int e_idx_i = i == 0 ? 0 : n_d - de;
                  const int e_idx = reverse ? n_d - de - e_idx_i : e_idx_i;

                  const EdgePairInfo ep_e((e_idx == n_d - de) ? -1 : tv,
                                          (e_idx == n_d - de) ? -1 : tvki,
                                          -1 - auxv2e[childPair], parentEdge);

                  const bool unset = !edgePairs[edgePairOS[parentEdge] + e_idx].isSet;
                  if (unset)
                  {
                     edgePairs[edgePairOS[parentEdge] + e_idx] = ep_e;
                  }
                  else
                  {
                     // Verify matching
                     MFEM_ASSERT(edgePairs[edgePairOS[parentEdge] + e_idx] == ep_e, "");
                  }
               }
               else  // childPairTopo == true, so this edge is a slave edge.
               {
                  const int childEdge = v2e.at(childPair);

                  const int pv0 = d == 0 ? gridVertex(0,s*n2) : gridVertex((1-s)*n1,0);
                  const int pv1 = d == 0 ? gridVertex(n1,s*n2) : gridVertex((1-s)*n1,n2);
                  const std::pair<int, int> parentPair(pv0 < pv1 ? pv0 : pv1,
                                                       pv0 < pv1 ? pv1 : pv0);
                  const int parentEdge = v2e.at(parentPair);
                  MFEM_ASSERT(parentEdges[pid] == parentEdge, "");

                  const bool start = (i == 0 && !reverse) || (i != 0 && reverse);
                  int end_idx = kvf_coarse.size() > 0 ?
                                (start ? 0 : kvf_coarse[kvi[d]].Size() - 1) : 0;
                  int de = kvf_coarse.size() > 0 ? kvf_coarse[kvi[d]][end_idx] : rf;
                  if (kvf.size() > 0 && kvf_coarse.size() == 0)
                  {
                     end_idx = start ? 0 : kvf[kvi[d]].Size() - 1;
                     de = kvf[kvi[d]][end_idx];
                  }

                  const int e_idx_i = i == 0 ? 0 : n_d - de;
                  const int e_idx = reverse ? n_d - de - e_idx_i : e_idx_i;

                  const int tv_e = (e_idx == n_d - de) ? -1 : tv;
                  const int tv_ki = (e_idx == n_d - de) ? -1 : tvki;

                  const EdgePairInfo ep_e(tv_e, tv_ki, childEdge, parentEdge);

#ifdef MFEM_DEBUG
                  const bool unset =
                     !edgePairs[edgePairOS[parentEdge] + e_idx].isSet;
                  const bool matching =
                     edgePairs[edgePairOS[parentEdge] + e_idx] == ep_e;
                  MFEM_ASSERT(unset || matching, "");
#endif

                  edgePairs[edgePairOS[parentEdge] + e_idx] = ep_e;
               }
            }
         }
      }

      if (hasSlaveFaces || hasAuxFace) { masterFaces.insert(parentFace); }
   } // loop over parents

   MFEM_VERIFY(consistent, "");
}

void NCNURBSExtension::GetAuxFaceToPatchTable(Array2D<int> &auxface2patch)
{
   auxface2patch.SetSize(auxFaces.size(), 2);

   if (auxFaces.size() == 0) { return; }

   auxface2patch = -1;

   const int dim = Dimension();

   bool consistent = true;

   for (int p=0; p<num_structured_patches; ++p)
   {
      Array<int> faces, orient;
      if (dim == 2) { patchTopo->GetElementEdges(p, faces, orient); }
      else { patchTopo->GetElementFaces(p, faces, orient); }

      for (auto face : faces)
      {
         const bool isMaster = dim == 2 ? masterEdgeToId.count(face) > 0 :
                               masterFaceToId.count(face) > 0;
         if (isMaster) // If a master face
         {
            const int mid = dim == 2 ? masterEdgeToId.at(face) :
                            masterFaceToId.at(face);
            const std::vector<int> &slaves = dim == 2 ? masterEdgeInfo[mid].slaves :
                                             masterFaceInfo[mid].slaves;
            for (auto s : slaves)
            {
               if (s < 0)
               {
                  // Auxiliary face.
                  const int aux = -1 - s;
                  if (auxface2patch(aux, 0) >= 0)
                  {
                     if (auxface2patch(aux, 1) != -1) { consistent = false; }
                     auxface2patch(aux, 1) = p;
                  }
                  else
                  {
                     auxface2patch(aux, 0) = p;
                  }
               }
            }
         }
      }
   }

   MFEM_VERIFY(consistent, "");
}

void NCNURBSExtension::GetSlaveFaceToPatchTable(Array2D<int> &sface2patch)
{
   const int dim = Dimension();
   const int numUnique = dim == 2 ? slaveEdgesUnique.Size() :
                         slaveFacesUnique.Size();
   sface2patch.SetSize(numUnique, 2);

   if (numUnique == 0) { return; }

   sface2patch = -1;

   bool consistent = true;

   for (int p=0; p<num_structured_patches; ++p)
   {
      Array<int> faces, orient;
      if (dim == 2) { patchTopo->GetElementEdges(p, faces, orient); }
      else { patchTopo->GetElementFaces(p, faces, orient); }

      for (auto face : faces)
      {
         const bool isMaster = dim == 2 ? masterEdgeToId.count(face) > 0 :
                               masterFaceToId.count(face) > 0;
         if (isMaster) // If a master face
         {
            const int mid = dim == 2 ? masterEdgeToId.at(face) :
                            masterFaceToId.at(face);
            const std::vector<int> &slaves = dim == 2 ? masterEdgeInfo[mid].slaves :
                                             masterFaceInfo[mid].slaves;
            for (auto id : slaves)
            {
               if (id >= 0)
               {
                  const int s = dim == 2 ? slaveEdges[id] : slaveFaces[id].index;
                  const int u = dim == 2 ? slaveEdgesToUnique[s] :
                                slaveFacesToUnique[s];
                  if (sface2patch(u, 0) >= 0)
                  {
                     if (sface2patch(u, 1) != -1) { consistent = false; }
                     sface2patch(u, 1) = p;
                  }
                  else
                  {
                     sface2patch(u, 0) = p;
                  }
               }
            }
         }
      }
   }

   MFEM_VERIFY(consistent, "");
}

void RemapKnotIndex(bool rev, const Array<int> &rf, int &k)
{
   const int ne = rf.Size();
   const int k0 = k;
   k = 0;
   for (int p=0; p<k0; ++p)
   {
      const int rp = rev ? ne - 1 - p : p;
      k += rf[rp];
   }
}

void NCNURBSExtension::UpdateAuxiliaryKnotSpans(const Array<int> &rf)
{
   for (auto auxEdge : auxEdges)
   {
      const int p = auxEdge.parent;
      const int parent = p < 0 ? -1 - p : p;
      const int kv = KnotInd(parent);
      for (int i=0; i<2; ++i)
      {
         RemapKnotIndex(false, kvf[kv], auxEdge.ksi[i]);
      }
   }

   for (auto auxFace : auxFaces)
   {
      Array<int> pv;
      std::array<int, 4> quad;
      patchTopo->GetFaceVertices(auxFace.parent, pv);
      MFEM_ASSERT(pv.Size() == 4, "");
      for (int i=0; i<4; ++i) { quad[i] = pv[i]; }
      // The face with vertices (pv0, pv1, pv2, pv3) is defined as a parent face.
      const std::pair<int, int> parentPair = QuadrupleToPair(quad);
      const std::array<int, 2> kv = parentToKV.at(parentPair);

      RemapKnotIndex(false, kvf[kv[0]], auxFace.ksi0[0]);
      RemapKnotIndex(false, kvf[kv[0]], auxFace.ksi1[0]);

      RemapKnotIndex(false, kvf[kv[1]], auxFace.ksi0[1]);
      RemapKnotIndex(false, kvf[kv[1]], auxFace.ksi1[1]);
   }
}

const int NURBSExtension::unsetFactor;

void NCNURBSExtension::LoadFactorsForKV(const std::string &filename)
{
   if (kvf_coarse.size() == 0) { kvf_coarse = kvf; }
   if (kvf.size() == 0) { kvf.resize(NumOfKnotVectors); }

   for (int kv=0; kv<NumOfKnotVectors; ++kv)
   {
      kvf[kv].SetSize(knotVectors[kv]->GetNE());
      kvf[kv] = unsetFactor;
   }

   if (filename.empty()) { return; }

   ifstream f(filename);
   int nkv;
   f >> nkv;

   for (int i=0; i<nkv; ++i)
   {
      int kv, nf, rf;
      f >> kv >> nf;
      MFEM_ASSERT(nf == 1, ""); // TODO: support input of multiple factors.

      kvf[kv] = unsetFactor;
      for (int j=0; j<nf; ++j) { f >> rf; }

      for (int j=0; j<kvf[kv].Size(); ++j) { kvf[kv][j] = rf; }
   }

   f.close();
}

int NCNURBSExtension::AuxiliaryEdgeNE(int aux_edge)
{
   const int signedParentEdge = auxEdges[aux_edge].parent;
   const int ki0 = auxEdges[aux_edge].ksi[0];
   const int ki1raw = auxEdges[aux_edge].ksi[1];
   int ki1 = ki1raw;
   if (ki1raw == -1)
   {
      const bool rev = signedParentEdge < 0;
      const int parentEdge = rev ? -1 - signedParentEdge : signedParentEdge;
      ki1 = KnotVec(parentEdge)->GetNE();
   }

   return ki1 - ki0;
}

// parentVerts are ordered with ascending knot-span index in parent edge, with
// knots from lower edge endpoint vertex to higher.
void NCNURBSExtension::SlaveEdgeToParent(int se, int parent,
                                         const Array<int> &os,
                                         const std::vector<int> &parentVerts,
                                         Array<int> &edges)
{
   Array<int> sev(2);
   if (se < 0) // Auxiliary edge
   {
      for (int i=0; i<2; ++i) { sev[i] = auxEdges[-1 - se].v[i]; }
   }
   else
   {
      patchTopo->GetEdgeVertices(se, sev);
   }

   // Number of slave and auxiliary edges, not mesh edges
   const int nedge = parentVerts.size() + 1;
   MFEM_ASSERT((int) parentVerts.size() + 2 == os.Size(), "");

   Array<int> parentEndpoints;
   patchTopo->GetEdgeVertices(parent, parentEndpoints);

   bool found = false;
   for (int i=0; i<nedge; ++i)
   {
      const bool first = (i == 0);
      const bool last = (i == nedge - 1);
      const int v0 = first ? parentEndpoints[0] : parentVerts[i - 1];
      const int v1 = last ? parentEndpoints[1] : parentVerts[i];
      if (sev[0] == v0 && sev[1] == v1)
      {
         found = true;
         MFEM_ASSERT(edges.Size() == os[i + 1] - os[i], "");
         for (int j=0; j<edges.Size(); ++j) { edges[j] = os[i] + j; }
      }
      else if (sev[0] == v1 && sev[1] == v0)
      {
         found = true;
         MFEM_ASSERT(edges.Size() == os[i + 1] - os[i], "");
         for (int j=0; j<edges.Size(); ++j) { edges[j] = os[i + 1] - 1 - j; }
      }
   }

   MFEM_VERIFY(found, "");
}

void NCNURBSExtension::GetMasterEdgePieceOffsets(int mid, Array<int> &os)
{
   const int np = masterEdgeInfo[mid].slaves.size();
   MFEM_VERIFY(np > 0, "");
   os.SetSize(np + 1);
   os[0] = 0;

   for (int i=0; i<np; ++i)
   {
      const int p = masterEdgeInfo[mid].slaves[i];
      const int s = slaveEdges[p];
      int nes = 0;
      if (s >= 0)
      {
         nes = knotVectors[KnotInd(s)]->GetNE();
      }
      else
      {
         nes = AuxiliaryEdgeNE(-1 - s);
      }

      os[i+1] = os[i] + nes;
   }
}

Array<int> CoarseToFineFactors(const Array<int> &rf)
{
   Array<int> frf(rf.Sum());
   int os = 0;
   for (auto f : rf)
   {
      for (int i=0; i<f; ++i)
      {
         frf[os + i] = f;
      }

      os += f;
   }

   return frf;
}

int NCNURBSExtension::SetPatchFactors(int p)
{
   constexpr char dirEdges3D[3][4] = {{0, 2, 4, 6}, {1, 3, 5, 7}, {8, 9, 10, 11}};
   constexpr char dirEdges2D[2][2] = {{0, 2}, {1, 3}};

   Array<int> edges, oedges;
   patchTopo->GetElementEdges(p, edges, oedges);

   const int dim = Dimension();
   const int nedge = dim == 3 ? 4 : 2;

   int dirSet = 0;

   bool partialChange = false;
   bool consistent = true;

   auto SetFactorsDirection = [&](int j, int os_final, int af, Array<int> &rf)
   {
      if (af == unsetFactor) { return; }
      if (rf.Size() == 0)
      {
         rf.SetSize(os_final);
         rf = unsetFactor;
      }
      if (rf[j] != unsetFactor && af != rf[j]) { consistent = false; }
      rf[j] = af;
   };

   auto SetFactorsEdge = [&](int j, int rf, Array<int> &pf)
   {
      if (rf == unsetFactor) { return; }
      if (pf[j] != unsetFactor && pf[j] != rf) { consistent = false; }
      pf[j] = rf;
   };

   auto LoopEdgesForDirection = [&](int d, Array<int> &rf, bool first)
   {
      for (int i=0; i<nedge; ++i)
      {
         const int edgeIndex = dim == 3 ? dirEdges3D[d][i] : dirEdges2D[d][i];
         const int edge = edges[edgeIndex];
         const bool isMaster = IsMasterEdge(edge);
         const int kv = KnotInd(edge);
         const bool rev = KnotSign(edge) < 0;

         if (first)
         {
            const int nfe = knotVectors[kv]->GetNE();
            const bool fullSize = kvf.size() > 0 && kvf[kv].Size() == nfe;
            const Array<int> rf_i = fullSize ? kvf[kv] :
                                    CoarseToFineFactors(kvf[kv]);
            if (rf_i.Size() > 0 && rf.Size() == 0) { rf = rf_i; }
         }
         else
         {
            if (kvf[kv] != rf) { partialChange = true; }
            kvf[kv] = rf;
         }

         if (isMaster)
         {
            // Check whether slave edges have factors set.
            const int mid = masterEdgeToId.at(edge);
            const int numPieces = masterEdgeInfo[mid].slaves.size();
            Array<int> os;
            GetMasterEdgePieceOffsets(mid, os);

            for (int piece=0; piece<numPieces; ++piece)
            {
               const int e = masterEdgeInfo[mid].slaves[piece];
               const int s = slaveEdges[e];
               Array<int> parentEdges;
               Array<int> *pf; // Refinement factors for this piece

               if (s >= 0) // Slave edge
               {
                  const int kvs = KnotInd(s);
                  parentEdges.SetSize(kvf[kvs].Size());
                  pf = &kvf[kvs];
               }
               else // Aux edge
               {
                  const int aux_edge = -1 - s;
                  if (auxef[aux_edge].Size() == 0)
                  {
                     auxef[aux_edge].SetSize(AuxiliaryEdgeNE(aux_edge));
                     auxef[aux_edge] = unsetFactor;
                  }
                  parentEdges.SetSize(auxef[aux_edge].Size());
                  pf = &auxef[aux_edge];
               }

               if (first && parentEdges.Size() == 0) { continue; }
               SlaveEdgeToParent(s, edge, os, masterEdgeInfo[mid].vertices, parentEdges);
               MFEM_ASSERT(parentEdges.Size() == os[piece + 1] - os[piece], "");

               for (int j = os[piece]; j < os[piece + 1]; ++j)
               {
                  const int jj = parentEdges[j - os[piece]];
                  const int jr = rev ? rf.Size() - 1 - jj : jj;
                  if (first)
                  {
                     SetFactorsDirection(jr, os[numPieces],
                                         (*pf)[j - os[piece]], rf);
                  }
                  else
                  {
                     SetFactorsEdge(j - os[piece], rf[jr], *pf);
                  }
               }
            }
         }
      }
   };

   for (int d=0; d<dim; ++d)
   {
      // Find the array of factors for direction d
      Array<int> rf;
      LoopEdgesForDirection(d, rf, true);
      if (rf.Size() == 0) { continue; } // This direction is unset

      // Set the same factor for all knotvectors in direction d.
      LoopEdgesForDirection(d, rf, false);
      if (rf.Min() > unsetFactor) { dirSet += pow(2, d); }
   }

   MFEM_VERIFY(consistent, "");
   return partialChange ? -1 - dirSet : dirSet;
}

void NCNURBSExtension::PropagateFactorsForKV(int rf_default)
{
   const int dim = Dimension();
   if (dim == 1 || num_structured_patches < 1)
   {
      for (size_t i=0; i<kvf.size(); ++i)
      {
         kvf[i] = rf_default;
      }
      return;
   }

   // Note that a slave edge can be a patchTopo edge (nonnegative index) or an
   // AuxiliaryEdge (negative index). A slave edge can be contained in multiple
   // overlapping master edges.

   // First, set slaveEdgesUnique.
   {
      slaveEdgesUnique.SetSize(0);
      slaveEdgesUnique.Reserve(slaveEdges.size());
      for (auto s : slaveEdges)
      {
         if (slaveEdgesToUnique.count(s) == 0)
         {
            slaveEdgesToUnique[s] = slaveEdgesUnique.Size();
            slaveEdgesUnique.Append(s);
         }
      }
   }

   // Set slaveFacesUnique.
   {
      slaveFacesUnique.SetSize(0);
      slaveFacesUnique.Reserve(slaveFaces.size());
      for (auto s : slaveFaces)
      {
         if (slaveFacesToUnique.count(s.index) == 0)
         {
            slaveFacesToUnique[s.index] = slaveFacesUnique.Size();
            slaveFacesUnique.Append(s.index);
         }
      }
   }

   // Initialize a set of patches to visit, using face-neighbors of the first
   // patch.
   const Table *face2elem = patchTopo->GetFaceToElementTable();

   Array2D<int> auxface2patch, sface2patch;
   GetAuxFaceToPatchTable(auxface2patch);
   GetSlaveFaceToPatchTable(sface2patch);

   Array<int> faces, orient;

   auto faceNeighbors = [&](int p, std::set<int> &nghb)
   {
      if (dim == 2) { patchTopo->GetElementEdges(p, faces, orient); }
      else { patchTopo->GetElementFaces(p, faces, orient); }

      for (auto face : faces)
      {
         Array<int> row;
         face2elem->GetRow(face, row);

         const bool isSlave = dim == 2 ? slaveEdgesToUnique.count(
                                 face) > 0 : slaveFacesToUnique.count(face) > 0;
         if (isSlave)
         {
            const int u = dim == 2 ? slaveEdgesToUnique[face] :
                          slaveFacesToUnique[face];
            for (int i=0; i<2; ++i)
            {
               const int elem = sface2patch(u, i);
               if (elem >= 0 && elem != p) { nghb.insert(elem); }
            }
         }

         for (auto elem : row) { nghb.insert(elem); }
      }
   };

   auto masterFaceNeighbors = [&](int p, std::set<int> &nghb)
   {
      if (dim == 2) { patchTopo->GetElementEdges(p, faces, orient); }
      else { patchTopo->GetElementFaces(p, faces, orient); }

      for (auto face : faces)
      {
         const bool isMaster = dim == 2 ? masterEdgeToId.count(face) > 0 :
                               masterFaceToId.count(face) > 0;
         if (isMaster) // If a master face
         {
            const int mid = dim == 2 ? masterEdgeToId.at(face) :
                            masterFaceToId.at(face);
            const std::vector<int> &slaves =
               dim == 2 ? masterEdgeInfo[mid].slaves : masterFaceInfo[mid].slaves;
            for (auto s : slaves)
            {
               if (s < 0)
               {
                  // Auxiliary face.
                  const int aux = -1 - s;
                  for (int i=0; i<2; ++i)
                  {
                     const int patch = auxface2patch(aux, i);
                     if (patch >= 0) { nghb.insert(patch); }
                  }
               }
               else
               {
                  // Slave face in patchTopo.
                  Array<int> row;
                  face2elem->GetRow(s, row);
                  for (auto elem : row) { nghb.insert(elem); }
               }
            }
         }
      }
   };

   const int npatchall = patches.Size();
   Array<int> patchState(npatchall);
   patchState = 0;

   auxef.resize(auxEdges.size());

   std::set<int> nextPatches, unchanged;
   const int dirAllSet = dim == 3 ? 7 : 3;
   int lastChanged = 0;
   int iter = 0;
   bool done = false;
   while (iter < 100 && !done)
   {
      // Start each iteration at the patch last changed
      nextPatches.clear();
      nextPatches.insert(lastChanged);

      std::set<int> visited; // Visit each patch only once per iteration
      iter++;

      while (nextPatches.size() > 0)
      {
         const int p = *nextPatches.begin();
         nextPatches.erase(p);

         visited.insert(p);

         const int dirSetSigned = SetPatchFactors(p);
         const bool partialChange = dirSetSigned < 0;
         const int dirSet = partialChange ? -1 - dirSetSigned : dirSetSigned;
         const bool changed = (patchState[p] != dirSet) || partialChange;
         patchState[p] = dirSet;

         // Find neighbors of patch p
         std::set<int> neighbors;

         // First, find neighbors sharing a conforming face, via face2elem.
         faceNeighbors(p, neighbors);

         // Second, find neighbors sharing a slave/auxiliary face in patchTopo.
         masterFaceNeighbors(p, neighbors);

         // Add neighbors not done to nextPatches. Note that a patch can be
         // added to nextPatches on multiple iterations, to propagate factors in
         // different directions, on multiple sweeps.

         for (auto n : neighbors)
         {
            if (n < npatchall && n != p && patchState[n] != dirAllSet &&
                visited.count(n) == 0)
            {
               nextPatches.insert(n);
            }
         }

         if (changed)
         {
            unchanged.erase(p);
            lastChanged = p;
         }
         else
         {
            unchanged.insert(p);
         }

         if (unchanged.size() == (size_t) npatchall)
         {
            // Make another pass through all patches to check for changes
            for (int i=0; i<npatchall; ++i)
            {
               const int dirSetSigned_i = SetPatchFactors(i);
               const bool partialChange_i = dirSetSigned_i < 0;
               const int dirSet_i = partialChange_i ? -1 - dirSetSigned_i :
                                    dirSetSigned_i;
               const bool changed_i = (patchState[i] != dirSet_i) ||
                                      partialChange_i;
               patchState[p] = dirSet_i;
               if (changed_i)
               {
                  unchanged.erase(i);
                  lastChanged = i;
               }
            }
         }

         if (unchanged.size() == (size_t) npatchall)
         {
            done = true;
            break;
         }
      }
   }

   // For any unset entries of kvf, set to default refinement factor rf_default.
   for (size_t i=0; i<kvf.size(); ++i)
   {
      if (kvf[i].Size() == 0)
      {
         kvf[i].SetSize(knotVectors[i]->GetNE());
         kvf[i] = rf_default;
      }
      else
      {
         for (int j=0; j<kvf[i].Size(); ++j)
         {
            if (kvf[i][j] == unsetFactor) { kvf[i][j] = rf_default; }
         }
      }

      if (knotVectors[i]->spacing)
      {
         PiecewiseSpacingFunction *pws = dynamic_cast<PiecewiseSpacingFunction*>
                                         (knotVectors[i]->spacing.get());
         if (pws)
         {
            Array<int> pwn = pws->RelativePieceSizes();
            const bool rev = pws->GetReverse();
            const int np = pwn.Size();
            const int f = kvf[i].Size() / pwn.Sum();
            MFEM_ASSERT(kvf[i].Size() == f * pwn.Sum(), "");

            Array<int> os(np + 1);
            os[0] = 0;
            for (int j=1; j<np+1; ++j)
            {
               const int jp = rev ? np - j : j - 1;
               os[j] = os[j-1] + (f * pwn[jp]);
            }

            Array<int> pwf(np);
            for (int j=0; j<np; ++j)
            {
               pwf[j] = kvf[i][os[j]];
               for (int r=os[j]+1; r<os[j+1]; ++r)
               {
                  MFEM_ASSERT(kvf[i][r] == pwf[j], "");
               }
            }

            pws->ScalePartition(pwf, true);
         }
      }
   }
}

void UpdateFactors(Array<int> &f)
{
   Array<int> rf(f.Sum());

   int os = 0;
   for (int i=0; i<f.Size(); ++i)
   {
      const int f_i = f[i];
      for (int j=0; j<f_i; ++j) { rf[os + j] = f_i; }

      os += f_i;
   }

   MFEM_ASSERT(os == rf.Size(), "");

   f = rf;
}

void NCNURBSExtension::RefineWithKVFactors(int rf,
                                           const std::string &kvf_filename,
                                           bool coarsened)
{
   if (ref_factors.Size() > 0)
   {
      MFEM_VERIFY(ref_factors.Size() == Dimension(), "");
      for (int i=0; i<ref_factors.Size(); ++i) { ref_factors[i] *= rf; }
   }
   else
   {
      ref_factors.SetSize(Dimension());
      ref_factors = rf;
   }

   LoadFactorsForKV(kvf_filename);
   PropagateFactorsForKV(rf);

   Refine(coarsened);
}

void NCNURBSExtension::ReadCoarsePatchCP(std::istream &input)
{
   input >> num_structured_patches;

   const int maxOrder = mOrders.Max();

   // For degree maxOrder, there are 2*(maxOrder + 1) knots for a single
   // element, and the number of control points in each dimension is
   // 2*(maxOrder + 1) - maxOrder - 1
   const int ncp1D = maxOrder + 1;
   const int ncp = pow(ncp1D, Dimension());

   patchCP.SetSize(num_structured_patches, ncp, Dimension());
   for (int p=0; p<num_structured_patches; ++p)
      for (int i=0; i<ncp; ++i)
         for (int j=0; j<Dimension(); ++j) { input >> patchCP(p, i, j); }
}

void NCNURBSExtension::PrintCoarsePatches(std::ostream &os)
{
   const int maxOrder = mOrders.Max();
   const int patchCP_size1 = patchCP.GetSize1();
   MFEM_VERIFY(patchCP_size1 == num_structured_patches || patchCP_size1 == 0,
               "");

   if (patchCP_size1 == 0) { return; }

   // For degree maxOrder, there are 2*(maxOrder + 1) knots for a single element,
   // and the number of control points in each dimension is
   // 2*(maxOrder + 1) - maxOrder - 1
   const int ncp1D = maxOrder + 1;
   const int ncp = pow(ncp1D, Dimension());

   os << "\npatch_cp\n" << num_structured_patches << "\n";
   for (int p=0; p<num_structured_patches; ++p)
   {
      for (int i=0; i<ncp; ++i)
      {
         os << patchCP(p, i, 0);
         for (int j=1; j<Dimension(); ++j)
         {
            os << ' ' << patchCP(p, i, j);
         }
         os << '\n';
      }
   }
}

void ApplyFineToCoarse(const Array<int> &f, Array<int> &c)
{
   MFEM_ASSERT(f.Size() == c.Sum(), "");
   bool consistent = true;
   int os = 0;
   for (int j=0; j<c.Size(); ++j)
   {
      const int cf = c[j];
      const int ff = f[os];
      for (int i=0; i<cf; ++i)
      {
         if (f[os + i] != ff) { consistent = false; }
      }

      c[j] *= ff;

      os += cf;
   }

   MFEM_VERIFY(consistent, "");
}

void NCNURBSExtension::UpdateCoarseKVF()
{
   if (kvf_coarse.size() == 0) { return; }
   for (int k=0; k<NumOfKnotVectors; ++k)
   {
      ApplyFineToCoarse(kvf[k], kvf_coarse[k]);
   }
}

int GetFaceOrientation(const Mesh *mesh, const int face,
                       const std::array<int, 4> &verts)
{
   Array<int> fverts;
   mesh->GetFaceVertices(face, fverts);
   MFEM_ASSERT(fverts.Size() == 4, "");

   // Verify that verts and fvert have the same entries as sets, by deep-copying
   // and sorting.
   {
      Array<int> s1(4);
      Array<int> s2(fverts);

      for (int i=0; i<4; ++i) { s1[i] = verts[i]; }

      s1.Sort(); s2.Sort();
      MFEM_ASSERT(s1 == s2, "");
   }

   // Find the shift of the first vertex.
   int s = -1;
   for (int i=0; i<4; ++i)
   {
      if (verts[i] == fverts[0]) { s = i; }
   }

   // Check whether ordering is reversed.
   const bool rev = verts[(s + 1) % 4] != fverts[1];
   if (rev) { s = -1 - s; } // Reversed order is encoded by the sign.
   return s;
}

// The 2D array `a` is of size n1*n2, with index j + n2*i corresponding to (i,j)
// with the fast index j, for 0 <= i < n1 and 0 <= j < n2. We assume that j is
// the fast index in (i,j). The orientation is encoded by ori, defining a shift
// and relative direction, such that a quad face F1, on which the ordering of
// `a` is based, has vertex with index `shift` matching vertex 0 of the new quad
// face F2, on which the new ordering of `a` should be based. For more details,
// see GetFaceOrientation.
bool Reorder2D(int ori, std::array<int, 2> &s0)
{
   const int shift = ori < 0 ? -1 - ori : ori;

   // Shift is an F1 index in the counter-clockwise ordering of 4 quad vertices.
   // Now find the (i,j) indices of this index, with i,j in {0,1}.
   const int s0i = (shift == 0 || shift == 3) ? 0 : 1;
   const int s0j = (shift < 2) ? 0 : 1;

   s0[0] = s0i;
   s0[1] = s0j;

   // Determine whether the dimensions of F1 and F2 are reversed. Do this by
   // finding the (i,j) indices of s1, which is the next vertex on F1.
   const int shift1 = ori < 0 ? shift - 1: shift + 1;
   const int s1 = (shift1 + 4) % 4;
   const int s1i = (s1 == 0 || s1 == 3) ? 0 : 1;
   const bool dimReverse = s0i == s1i;

   return dimReverse;
}

void GetInverseShiftedDimensions2D(int signedShift, int sm, int sn, int &m,
                                   int &n)
{
   const bool rev = (signedShift < 0);
   const int shift = rev ? -1 - signedShift : signedShift;
   MFEM_ASSERT(0 <= shift && shift < 4, "");

   // We consider 8 cases for the possible values of rev and shift.
   if (rev)
   {
      if (shift == 0)
      {
         // New: 3 2  Old: 1 2
         //      0 1       0 3
         n = sm;
         m = sn;
      }
      else if (shift == 1)
      {
         // New: 3 2  Old: 2 3
         //      0 1       1 0
         m = sm;
         n = sn;
      }
      else if (shift == 2)
      {
         // New: 3 2  Old: 3 0
         //      0 1       2 1
         n = sm;
         m = sn;
      }
      else // shift == 3
      {
         // New: 3 2  Old: 0 1
         //      0 1       3 2
         m = sm;
         n = sn;
      }
   }
   else
   {
      if (shift == 0)
      {
         // New: 3 2  Old: 3 2
         //      0 1       0 1
         m = sm;
         n = sn;
      }
      else if (shift == 1)
      {
         // New: 3 2  Old: 0 3
         //      0 1       1 2
         n = sm;
         m = sn;
      }
      else if (shift == 2)
      {
         // New: 3 2  Old: 1 0
         //      0 1       2 3
         m = sm;
         n = sn;
      }
      else // shift == 3
      {
         // New: 3 2  Old: 2 1
         //      0 1       3 0
         n = sm;
         m = sn;
      }
   }
}

void GetShiftedGridPoints2D(int m, int n, int i, int j, int signedShift,
                            int& sm, int& sn, int& si, int& sj)
{
   const bool rev = (signedShift < 0);
   const int shift = rev ? -1 - signedShift : signedShift;
   MFEM_ASSERT(0 <= shift && shift < 4, "");

   // (0,0) <= (i,j) < (m,n) are old indices, and old vertex [shift] maps
   // to new vertex 0 in counter-clockwise quad ordering.

   // We consider 8 cases for the possible values of rev and shift.
   if (rev)
   {
      if (shift == 0)
      {
         // New: 3 2  Old: 1 2
         //      0 1       0 3
         sm = n;
         sn = m;

         si = j;
         sj = i;
      }
      else if (shift == 1)
      {
         // New: 3 2  Old: 2 3
         //      0 1       1 0
         sm = m;
         sn = n;

         si = m - 1 - i;
         sj = j;
      }
      else if (shift == 2)
      {
         // New: 3 2  Old: 3 0
         //      0 1       2 1
         sm = n;
         sn = m;

         si = n - 1 - j;
         sj = m - 1 - i;
      }
      else // shift == 3
      {
         // New: 3 2  Old: 0 1
         //      0 1       3 2
         sm = m;
         sn = n;

         si = i;
         sj = n - 1 - j;
      }
   }
   else
   {
      if (shift == 0)
      {
         // New: 3 2  Old: 3 2
         //      0 1       0 1
         sm = m;
         sn = n;

         si = i;
         sj = j;
      }
      else if (shift == 1)
      {
         // New: 3 2  Old: 0 3
         //      0 1       1 2
         sm = n;
         sn = m;

         si = j;
         sj = m - 1 - i;
      }
      else if (shift == 2)
      {
         // New: 3 2  Old: 1 0
         //      0 1       2 3
         sm = m;
         sn = n;

         si = m - 1 - i;
         sj = n - 1 - j;
      }
      else // shift == 3
      {
         // New: 3 2  Old: 2 1
         //      0 1       3 0
         sm = n;
         sn = m;

         si = n - 1 - j;
         sj = i;
      }
   }
}

// Given a quadruple in q, return the pair (q_i, q_j), where q_i is the minimum
// entry of q, and q_j is the entry two indices away from q_i. When q contains
// indices of the vertices of a quadrilateral, the returned pair represents the
// unique diagonal touching the vertex of minimum index, which is a more concise
// way of representing the quadrilateral, facilitating the search of faces.
std::pair<int, int> QuadrupleToPair(const std::array<int, 4> &q)
{
   const auto qmin = std::min_element(q.begin(), q.end());
   const int idmin = std::distance(q.begin(), qmin);
   return std::pair<int, int>(q[idmin], q[(idmin + 2) % 4]);
}

void VertexToKnotSpan::SetSize(int dimension, int numVertices)
{
   dim = dimension;
   MFEM_ASSERT((dim == 2 || dim == 3) && numVertices > 0, "Invalid size");
   data.SetSize(numVertices, dim == 3 ? 7 : 4);
}

void VertexToKnotSpan::SetVertex2D(int index, int v, int ks,
                                   const std::array<int, 2> &pv)
{
   data(index,0) = v;
   data(index,1) = ks;
   data(index,2) = pv[0];
   data(index,3) = pv[1];
}

void VertexToKnotSpan::SetVertex3D(int index, int v,
                                   const std::array<int, 2> &ks,
                                   const std::array<int, 4> &pv)
{
   data(index,0) = v;
   data(index,1) = ks[0];
   data(index,2) = ks[1];
   data(index,3) = pv[0];
   data(index,4) = pv[1];
   data(index,5) = pv[2];
   data(index,6) = pv[3];
}

void VertexToKnotSpan::SetKnotSpan2D(int index, int ks)
{
   data(index,1) = ks;
}

void VertexToKnotSpan::SetKnotSpans3D(int index, const std::array<int, 2> &ks)
{
   data(index,1) = ks[0];
   data(index,2) = ks[1];
}

void VertexToKnotSpan::GetVertex2D(int index, int &v, int &ks,
                                   std::array<int, 2> &pv) const
{
   v = data(index,0);
   ks = data(index,1);
   pv[0] = data(index,2);
   pv[1] = data(index,3);
}

void VertexToKnotSpan::GetVertex3D(int index, int &v, std::array<int, 2> &ks,
                                   std::array<int, 4> &pv) const
{
   v = data(index,0);
   ks[0] = data(index,1);
   ks[1] = data(index,2);
   pv[0] = data(index,3);
   pv[1] = data(index,4);
   pv[2] = data(index,5);
   pv[3] = data(index,6);
}

void VertexToKnotSpan::Print(std::ostream &os) const
{
   const int nv = data.NumRows();
   const int m = data.NumCols();
   os << nv << "\n";
   for (int i = 0; i < nv; i++)
   {
      os << data(i,0);
      for (int j = 1; j < m; j++)
      {
         os << " " << data(i,j);
      }
      os << "\n";
   }
}

std::pair<int, int> VertexToKnotSpan::GetVertexParentPair(int index) const
{
   if (dim == 3)
   {
      std::array<int, 4> pv;
      for (int i=0; i<4; ++i) { pv[i] = data(index, 3 + i); }
      // The face with vertices (pv[0], pv[1], pv[2], pv[3]) is defined as a
      // parent face.
      return QuadrupleToPair(pv);
   }

   int c0 = data(index, 2);
   int c1 = data(index, 3);
   if (c0 > c1) { std::swap(c0, c1); }
   return std::pair<int, int>(c0, c1);
}

void NCNURBSExtension::UniformRefinement(const Array<int> &rf)
{
   MFEM_VERIFY(!nonconformingPT,
               "NURBS NC-patch meshes cannot use this method of refinement");

   if (ref_factors.Size())
   {
      MFEM_VERIFY(ref_factors.Size() == rf.Size(), "");
      for (int i=0; i<rf.Size(); ++i) { ref_factors[i] *= rf[i]; }
   }
   else
   {
      ref_factors = rf;
   }

   Refine(false, &rf);
}

void NCNURBSExtension::Refine(bool coarsened, const Array<int> *rf)
{
   const int maxOrder = mOrders.Max();
   const int dim = Dimension();

   for (int p = 0; p < patches.Size(); p++)
   {
      if (nonconformingPT)
      {
         std::vector<Array<int>> prf(dim);
         Array<KnotVector*> pkv(dim);
         Array<int> edges, orient;
         patchTopo->GetElementEdges(p, edges, orient);

         if (dim == 3)
         {
            constexpr char e3[3] = {0, 3, 8};
            for (int i=0; i<3; ++i)
            {
               prf[i] = kvf[KnotInd(edges[e3[i]])];
               pkv[i] = knotVectors[KnotInd(edges[e3[i]])];
            }
         }
         else
         {
            MFEM_VERIFY(dim == 2, "");
            for (int i=0; i<2; ++i)
            {
               prf[i] = kvf[KnotInd(edges[i])];
               pkv[i] = knotVectors[KnotInd(edges[i])];
            }
         }

         if (p >= num_structured_patches)
         {
            for (int i=0; i<dim; ++i)
            {
               // Collapse prf[i] to a single factor
               MFEM_VERIFY(prf[i].IsConstant(), "");
               prf[i].SetSize(1);
            }
         }

         patches[p]->UpdateSpacingPartitions(pkv);
         patches[p]->UniformRefinement(prf, coarsened, maxOrder);
      }
      else
      {
         patches[p]->UniformRefinement(*rf);
      }
   }

   if (nonconformingPT)
   {
      patchTopo->ncmesh->RefineVertexToKnotSpan(kvf, knotVectors, parentToKV);
      UpdateAuxiliaryKnotSpans(ref_factors);
      UpdateCoarseKVF();
   }
}

void NCNURBSExtension::SetDofToPatch()
{
   dof2patch.SetSize(NumOfDofs);
   dof2patch = -1;

   const int dim = Dimension();
   if (dim == 1) { return; }

   Array<int> edges, faces, orient;
   const int np = patchTopo->GetNE();

   for (int p = 0; p < np; p++)
   {
      patchTopo->GetElementEdges(p, edges, orient);
      for (auto e : edges)
      {
         if (masterEdges.count(e) > 0)
         {
            Array<int> mdof;
            GetMasterEdgeDofs(true, e, mdof);
            for (auto dof : mdof) { dof2patch[dof] = p; }
         }
      }

      if (dim == 3)
      {
         patchTopo->GetElementFaces(p, faces, orient);

         for (auto f : faces)
         {
            if (masterFaces.count(f) > 0)
            {
               Array2D<int> mdof;
               GetMasterFaceDofs(true, f, mdof);
               for (int j=0; j<mdof.NumCols(); ++j)
                  for (int k=0; k<mdof.NumRows(); ++k)
                  {
                     dof2patch[mdof(k,j)] = p;
                  }
            }
         }
      }
   }
}

// This function assumes a uniform number of control points per element in kv.
int GetNCPperEdge(const KnotVector *kv)
{
   const int ne = kv->GetNE();

   // Total number of CP on edge, excluding vertex CP.
   const int totalEdgeCP = kv->GetNCP() - 2 - ne + 1;
   const int perEdgeCP = totalEdgeCP / ne;

   MFEM_VERIFY(perEdgeCP * ne == totalEdgeCP, "");

   return perEdgeCP;
}

void NCNURBSExtension::GenerateOffsets()
{
   const int nv = patchTopo->GetNV();
   const int ne = patchTopo->GetNEdges();
   const int nf = patchTopo->GetNFaces();
   const int np = patchTopo->GetNE();
   int meshCounter, spaceCounter, dim = Dimension();

   std::set<int> reversedParents;
   if (patchTopo->ncmesh)
   {
      // Note that master or slave entities exist only for a mesh with
      // vertex_parents, not for the vertex_to_knotspan case. Currently, a mesh
      // is not allowed to have both cases, see the MFEM_VERIFY below.

      const NCMesh::NCList& nce = patchTopo->ncmesh->GetNCList(1);
      const NCMesh::NCList& ncf = patchTopo->ncmesh->GetNCList(2);

      masterEdges.clear();
      masterFaces.clear();
      slaveEdges.clear();
      slaveFaces.clear();
      masterEdgeToId.clear();
      masterFaceToId.clear();

      MFEM_VERIFY(nce.masters.Size() > 0 ||
                  patchTopo->ncmesh->GetVertexToKnotSpan().Size() > 0, "");
      MFEM_VERIFY(!(nce.masters.Size() > 0 &&
                    patchTopo->ncmesh->GetVertexToKnotSpan().Size() > 0), "");

      std::vector<EdgePairInfo> edgePairs;
      std::vector<FacePairInfo> facePairs;
      std::vector<int> parentFaces, parentVerts;
      std::vector<std::array<int, 2>> parentSize;

      const bool is3D = dim == 3;

      std::map<std::pair<int, int>, int> v2f;

      if (patchTopo->ncmesh->GetVertexToKnotSpan().Size() > 0)
      {
         // Intersections of master edges may not be edges in patchTopo->ncmesh,
         // so we represent them in auxEdges, to account for their vertices and
         // DOFs.
         {
            int vert_index[2];
            const NCMesh::NCList& EL = patchTopo->ncmesh->GetEdgeList();
            for (auto edgeID : EL.conforming)
            {
               patchTopo->ncmesh->GetEdgeVertices(edgeID, vert_index);
               v2e[std::pair<int, int> (vert_index[0], vert_index[1])] = edgeID.index;
            }
         }

         if (is3D)
         {
            Array<int> vert;
            for (int i=0; i<patchTopo->GetNumFaces(); ++i)
            {
               patchTopo->GetFaceVertices(i, vert);
               const int vmin = vert.Min();
               const int idmin = vert.Find(vmin);
               v2f[std::pair<int, int> (vert[idmin], vert[(idmin + 2) % 4])] = i;
            }
         }

         const VertexToKnotSpan &v2k = patchTopo->ncmesh->GetVertexToKnotSpan();

         if (is3D)
            ProcessVertexToKnot3D(v2k, v2f, parentSize, edgePairs,
                                  facePairs, parentFaces, parentVerts);
         else
         {
            ProcessVertexToKnot2D(v2k, reversedParents, edgePairs);
         }
      } // if using vertex_to_knotspan

      const int numMasters = is3D ? ncf.masters.Size() : nce.masters.Size();

      if (is3D)
      {
         for (auto masterFace : ncf.masters)
         {
            masterFaces.insert(masterFace.index);
         }
      }

      for (auto masterEdge : nce.masters)
      {
         masterEdges.insert(masterEdge.index);
      }

      masterEdgeIndex.SetSize(masterEdges.size());
      int cnt = 0;
      for (auto medge : masterEdges)
      {
         masterEdgeIndex[cnt] = medge;
         masterEdgeToId[medge] = cnt;
         cnt++;
      }
      MFEM_VERIFY(cnt == masterEdgeIndex.Size(), "");

      Array<int> masterFaceIndex(parentFaces.size());

      // Note that masterFaces is a subset of parentFaces.
      MFEM_VERIFY(masterFaces.size() <= parentFaces.size(), "");

      cnt = 0;
      for (auto mface : parentFaces)
      {
         masterFaceIndex[cnt] = mface;
         masterFaceToId[mface] = cnt;
         cnt++;
      }

      MFEM_VERIFY(cnt == masterFaceIndex.Size(), "");

      masterEdgeInfo.clear();
      masterEdgeInfo.resize(masterEdgeIndex.Size());

      masterFaceInfo.clear();
      masterFaceInfo.resize(masterFaceIndex.Size());

      if (patchTopo->ncmesh->GetVertexToKnotSpan().Size() > 0)
      {
         // Note that this is used in 2D and 3D.
         const int npairs = edgePairs.size();

         for (int i=0; i<npairs; ++i)
         {
            if (!edgePairs[i].isSet) { continue; }

            const int v = edgePairs[i].v;
            const int s = edgePairs[i].child;
            const int m = edgePairs[i].parent;
            const int ksi = edgePairs[i].ksi;

            slaveEdges.push_back(s);

            const int mid = masterEdgeToId[m];
            const int si = slaveEdges.size() - 1;
            masterEdgeInfo[mid].slaves.push_back(si);
            if (v >= 0)
            {
               masterEdgeInfo[mid].vertices.push_back(v);
               masterEdgeInfo[mid].ks.push_back(ksi);
            }
         }

         ProcessFacePairs(0, 0, parentSize, parentVerts, facePairs);
      }

      for (int i=0; i<nce.slaves.Size(); ++i)
      {
         const NCMesh::Slave& slaveEdge = nce.slaves[i];
         int vert_index[2];
         patchTopo->ncmesh->GetEdgeVertices(slaveEdge, vert_index);
         slaveEdges.push_back(slaveEdge.index);

         const int mid = masterEdgeToId[slaveEdge.master];
         masterEdgeInfo[mid].slaves.push_back(i);
      }

      if (!is3D)
      {
         for (int m=0; m<numMasters; ++m)
         {
            // Order the slaves of each master edge, from the first to second
            // vertex of the master edge.
            const int numSlaves = masterEdgeInfo[m].slaves.size();
            MFEM_ASSERT(numSlaves > 0, "");
            int mvert[2];
            int svert[2];
            patchTopo->ncmesh->GetEdgeVertices(nce.masters[m], mvert);

            std::vector<int> orderedSlaves(numSlaves);
            std::set<int> used;

            int vi = mvert[0];
            for (int s=0; s<numSlaves; ++s)
            {
               // Find the slave edge containing vertex vi.
               // This has quadratic complexity, but numSlaves is small.
               orderedSlaves[s] = -1;
               for (int t=0; t<numSlaves; ++t)
               {
                  const int sid = masterEdgeInfo[m].slaves[t];
                  if (used.count(sid) > 0) { continue; }
                  patchTopo->ncmesh->GetEdgeVertices(nce.slaves[sid], svert);
                  if (svert[0] == vi || svert[1] == vi)
                  {
                     orderedSlaves[s] = sid;
                     used.insert(sid);
                     break;
                  }
               }

               MFEM_ASSERT(orderedSlaves[s] >= 0, "");

               // Update vi to the next vertex
               vi = (svert[0] == vi) ? svert[1] : svert[0];

               if (s < numSlaves - 1)
               {
                  masterEdgeInfo[m].vertices.push_back(vi);
                  masterEdgeInfo[m].ks.push_back(-1); // Used only in 3D.
               }
            }

            masterEdgeInfo[m].slaves = orderedSlaves;
         } // m
      }

      if (is3D)
      {
         // Remove edges from masterEdges if they do not have any slave edges.
         std::vector<int> falseMasterEdges;
         for (auto me : masterEdges)
         {
            const int mid = masterEdgeToId.at(me);
            if (masterEdgeInfo[mid].slaves.size() <= 1)
            {
               falseMasterEdges.push_back(me);
            }
         }

         for (auto me : falseMasterEdges) { masterEdges.erase(me); }

         // Find slave and auxiliary faces not yet defined.
         const int nfp0 = facePairs.size();
         std::set<int> addParentFaces;
         FindAdditionalFacesSA(v2f, addParentFaces, facePairs);

         cnt = parentFaces.size();
         const int npf0 = cnt;

         for (auto pf : addParentFaces)
         {
            if (masterFaces.count(pf) == 0)
            {
               masterFaces.insert(pf);
               masterFaceIndex.Append(pf);

               masterFaceToId[pf] = cnt;
               cnt++;

               {
                  Array<int> edges, ori, verts;
                  patchTopo->GetFaceEdges(pf, edges, ori);
                  patchTopo->GetFaceVertices(pf, verts);
                  MFEM_ASSERT(edges.Size() == 4 && verts.Size() == 4, "");

                  parentSize.emplace_back(std::array<int, 2>
                  {
                     KnotVec(edges[0])->GetNE(),
                     KnotVec(edges[1])->GetNE()
                  });

                  masterFaceInfo.push_back(
                     MasterFaceInfo(KnotVec(edges[0])->GetNE(),
                                    KnotVec(edges[1])->GetNE()));

                  for (int i=0; i<4; ++i) { parentVerts.push_back(verts[i]); }
               }
            }
         }

         MFEM_VERIFY(cnt == masterFaceIndex.Size(), "");

         ProcessFacePairs(nfp0, npf0, parentSize, parentVerts, facePairs);
      }
   }

   for (auto rp : reversedParents)
   {
      masterEdgeInfo[masterEdgeToId[rp]].Reverse();
   }

   Array<int> edges, orient;

   v_meshOffsets.SetSize(nv);
   e_meshOffsets.SetSize(ne);
   f_meshOffsets.SetSize(nf);
   p_meshOffsets.SetSize(np);

   v_spaceOffsets.SetSize(nv);
   e_spaceOffsets.SetSize(ne);
   f_spaceOffsets.SetSize(nf);
   p_spaceOffsets.SetSize(np);

   // Get vertex offsets
   for (meshCounter = 0; meshCounter < nv; meshCounter++)
   {
      v_meshOffsets[meshCounter]  = meshCounter;
      v_spaceOffsets[meshCounter] = meshCounter;
   }
   spaceCounter = meshCounter;

   // Get edge offsets
   for (int e = 0; e < ne; e++)
   {
      e_meshOffsets[e]  = meshCounter;
      e_spaceOffsets[e] = spaceCounter;

      if (masterEdges.count(e) == 0)  // If not a master edge
      {
         meshCounter  += KnotVec(e)->GetNE() - 1;
         spaceCounter += KnotVec(e)->GetNCP() - 2;
      }
   }

   const int nauxe = auxEdges.size();
   aux_e_meshOffsets.SetSize(nauxe + 1);
   aux_e_spaceOffsets.SetSize(nauxe + 1);
   for (int e = 0; e < nauxe; e++)
   {
      aux_e_meshOffsets[e] = meshCounter;
      aux_e_spaceOffsets[e] = spaceCounter;

      // Find the number of elements and CP in this auxiliary edge, which is
      // defined only on part of the master edge knotvector.
      const int signedParentEdge = auxEdges[e].parent;
      const int ki0 = auxEdges[e].ksi[0];
      const int ki1raw = auxEdges[e].ksi[1];
      const bool rev = signedParentEdge < 0;
      const int parentEdge = rev ? -1 - signedParentEdge : signedParentEdge;
      const int masterNE = KnotVec(parentEdge)->GetNE();
      const int ki1 = ki1raw == -1 ? masterNE : ki1raw;
      const int perEdgeCP = GetNCPperEdge(KnotVec(e));
      const int auxne = ki1 - ki0;
      MFEM_ASSERT(auxne > 0, "");
      meshCounter += auxne - 1;
      spaceCounter += (auxne * perEdgeCP) + auxne - 1;
   }

   aux_e_meshOffsets[nauxe] = meshCounter;
   aux_e_spaceOffsets[nauxe] = spaceCounter;

   // Get face offsets
   for (int f = 0; f < nf; f++)
   {
      f_meshOffsets[f]  = meshCounter;
      f_spaceOffsets[f] = spaceCounter;

      if (masterFaces.count(f) == 0)  // If not a master face
      {
         patchTopo->GetFaceEdges(f, edges, orient);

         meshCounter +=
            (KnotVec(edges[0])->GetNE() - 1) *
            (KnotVec(edges[1])->GetNE() - 1);
         spaceCounter +=
            (KnotVec(edges[0])->GetNCP() - 2) *
            (KnotVec(edges[1])->GetNCP() - 2);
      }
   }

   const int nauxf = auxFaces.size();
   aux_f_meshOffsets.SetSize(nauxf + 1);
   aux_f_spaceOffsets.SetSize(nauxf + 1);
   for (int f = 0; f < nauxf; f++)
   {
      aux_f_meshOffsets[f] = meshCounter;
      aux_f_spaceOffsets[f] = spaceCounter;

      const int parentFace = auxFaces[f].parent;
      patchTopo->GetFaceEdges(parentFace, edges, orient);

      // Number of control points per edge, in first and second directions.
      const int perEdgeCP0 = GetNCPperEdge(KnotVec(edges[0]));
      const int perEdgeCP1 = GetNCPperEdge(KnotVec(edges[1]));

      const int auxne0 = auxFaces[f].ksi1[0] - auxFaces[f].ksi0[0];
      const int auxne1 = auxFaces[f].ksi1[1] - auxFaces[f].ksi0[1];
      meshCounter += (auxne0 - 1) * (auxne1 - 1);
      spaceCounter += ((auxne0 * perEdgeCP0) + auxne0 - 1) *
                      ((auxne1 * perEdgeCP1) + auxne1 - 1);
   }

   aux_f_meshOffsets[nauxf] = meshCounter;
   aux_f_spaceOffsets[nauxf] = spaceCounter;

   // Get patch offsets
   GetPatchOffsets(meshCounter, spaceCounter);

   NumOfVertices = meshCounter;
   NumOfDofs     = spaceCounter;

   SetDofToPatch();
}

} // namespace mfem
