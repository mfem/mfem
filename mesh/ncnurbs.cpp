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

#include "nurbs.hpp"

namespace mfem
{

using namespace std;

// Helper functions for NC-NURBS
void GetShiftedGridPoints2D(int m, int n, int i, int j, int signedShift,
                            int& sm, int& sn, int& si, int& sj);

void GetInverseShiftedDimensions2D(int signedShift, int sm, int sn, int &m,
                                   int &n);

int GetFaceOrientation(const Mesh *mesh, const int face,
                       const Array<int> & verts);

bool Reorder2D(int n1, int n2, int ori, std::vector<int> & s0);

void NURBSExtension::FindAdditionalSlaveAndAuxiliaryFaces(
   std::map<std::pair<int, int>, int> & v2f,
   std::set<int> & addParentFaces,
   std::vector<FacePairInfo> & facePairs)
{
   for (int f=0; f<patchTopo->GetNFaces(); ++f)
   {
      if (masterFaces.find(f) != masterFaces.end())
      {
         continue;   // Already a master face
      }

      Array<int> edges, ori, verts;
      patchTopo->GetFaceEdges(f, edges, ori);
      patchTopo->GetFaceVertices(f, verts);

      MFEM_VERIFY(edges.Size() == 4 && verts.Size() == 4, "");

      const int fn1 = KnotVec(edges[0])->GetNE();
      const int fn2 = KnotVec(edges[1])->GetNE();

      // Loop over the 2 pairs of opposite sides
      for (int p=0; p<2; ++p)  // Pair p
      {
         Array<int> oppEdges(2);
         const int sideEdge0 = edges[1 - p];
         bool bothMaster = true;
         for (int s=0; s<2; ++s)
         {
            oppEdges[s] = edges[p + 2*s];

            bool isTrueMasterEdge = false;
            if (masterEdges.count(oppEdges[s]) > 0)
            {
               const int mid = masterEdgeToId.at(oppEdges[s]);
               if (masterEdgeSlaves[mid].size() != 0) { isTrueMasterEdge = true; }
            }

            if (!isTrueMasterEdge) { bothMaster = false; }
         }

         if (bothMaster)
         {
            // Possibly define auxiliary and/or slave faces on this face

            // Check for auxiliary and slave edges
            std::vector<Array<int>> sideAuxEdges(2);
            std::vector<Array<int>> sideSlaveEdges(2);
            for (int s=0; s<2; ++s)
            {
               const int mid = masterEdgeToId.at(oppEdges[s]);
               for (auto edge : masterEdgeSlaves[mid])
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

               Array2D<int> epv(2,2);

               for (int s=0; s<2; ++s)
               {
                  const int mid = masterEdgeToId.at(oppEdges[s]);
                  const std::size_t nes = masterEdgeSlaves[mid].size();
                  MFEM_VERIFY(masterEdgeVerts[mid].size() + 1 == nes, "");

                  // Vertices in masterEdgeVerts[mid] are ordered starting
                  // from the master edge endpoint with lower vertex index.

                  Array<int> everts;
                  patchTopo->GetEdgeVertices(oppEdges[s], everts);

                  for (int i=0; i<2; ++i)
                  {
                     const int v = everts[i];
                     // Find the index of vertex v in verts.
                     const int vid = verts.Find(v);
                     MFEM_ASSERT(vid >= 0, "");
                     epv(s,i) = vid;
                  }

                  edgeV[s].Append(everts[0]);
                  edgeVki[s].Append(0);

                  MFEM_VERIFY(masterEdgeVerts[mid].size() == masterEdgeKI[mid].size(), "");
                  for (std::size_t i=0; i<masterEdgeVerts[mid].size(); ++i)
                  {
                     edgeV[s].Append(masterEdgeVerts[mid][i]);
                     edgeVki[s].Append(masterEdgeKI[mid][i]);
                  }

                  const int nelem = KnotVec(oppEdges[s])->GetNE();

                  edgeV[s].Append(everts[1]);
                  edgeVki[s].Append(nelem);

                  for (std::size_t i=0; i<nes; ++i)
                  {
                     const int edge = slaveEdges[masterEdgeSlaves[mid][i]];
                     edgeE[s].Append(edge);

                     Array<int> sverts(2);
                     if (edge >= 0)  // If a slave edge
                     {
                        patchTopo->GetEdgeVertices(edge, sverts);
                     }
                     else
                     {
                        const int auxEdge = -1 - edge;
                        GetAuxEdgeVertices(auxEdge, sverts);
                     }

                     MFEM_VERIFY((sverts[0] == edgeV[s][i] && sverts[1] == edgeV[s][i+1]) ||
                                 (sverts[1] == edgeV[s][i] && sverts[0] == edgeV[s][i+1]), "");
                  }
               }

               // Check whether the number and types of edges on opposite
               // sides match. If not, skip this pair of sides.
               if (edgeE[0].Size() != edgeE[1].Size())
               {
                  continue;
               }

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

                  if (!matching)
                  {
                     continue;
                  }
               }

               // Check whether edgeV[s] are in the same order or reversed, for s=0,1.
               bool rev = true;
               {
                  Array<int> sideVerts0;
                  patchTopo->GetEdgeVertices(sideEdge0, sideVerts0);
                  sideVerts0.Sort();

                  Array<bool> found(2);
                  found = false;

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

                  MFEM_VERIFY(found[0] && found[1], "");
               }

               // Find auxiliary or slave subfaces of face f.
               // Note that there may be no master faces in patchTopo->ncmesh.

               for (int i=0; i<2; ++i)
               {
                  MFEM_VERIFY(edgeV[i].Size() == nes + 1, "");
               }

               for (int e=0; e<nes; ++e) // Loop over edges
               {
                  Array<int> fverts(4);
                  fverts[0] = edgeV[0][e];
                  fverts[1] = edgeV[0][e + 1];

                  fverts[2] = edgeV[1][rev ? nes - e - 1 : e + 1];
                  fverts[3] = edgeV[1][rev ? nes - e : e];

                  const int eki = edgeVki[0][e];  // Knot index w.r.t. the edge.
                  const int eki1 = edgeVki[0][e + 1];  // Knot index w.r.t. the edge.

                  const int e1ki = edgeVki[1][rev ? nes - e : e];  // Knot index w.r.t. the edge.
                  const int e1ki1 = edgeVki[1][rev ? nes - e - 1 : e +
                                               1];  // Knot index w.r.t. the edge.

                  // ori_f is the signed shift such that fverts[abs1(ori_f)] is
                  // closest to vertex 0, verts[0], of parent face f, and the
                  // relative direction of the ordering is encoded in the sign.
                  // TODO: this is not used, since we put fverts into auxFaces with reordering to match
                  // the orientation of the master face, effectively making ori_f = 0.
                  int ori_f = 0;

                  // Set the 2D knot indices of the 4 vertices in fverts, with respect to the face f.
                  Array2D<int> fki(4,2);
                  {
                     // Use epv and eki to get the 2D face knot index.

                     // Number of elements on the edge.
                     const int eNE = edgeVki[0][edgeVki[0].Size() - 1];

                     if (p == 0)
                     {
                        MFEM_ASSERT(edgeV[0][0] == verts[0] ||
                                    edgeV[0][edgeV[0].Size() - 1] == verts[0], "");
                        MFEM_ASSERT(edgeV[1][0] == verts[3] ||
                                    edgeV[1][edgeV[0].Size() - 1] == verts[3], "");
                        MFEM_ASSERT(eNE == fn1, "");
                        MFEM_ASSERT(edgeVki[1][edgeVki[1].Size() - 1] == fn1, "");

                        const bool rev0 = edgeV[0][0] != verts[0];
                        fki(0,0) = rev0 ? eNE - eki : eki;
                        fki(0,1) = 0;

                        fki(1,0) = rev0 ? eNE - eki1 : eki1;
                        fki(1,1) = 0;

                        if (rev0)
                        {
                           ori_f = -2;
                        }

                        // Other side
                        const bool rev1 = edgeV[1][0] != verts[3];

                        fki(2,0) = rev1 ? eNE - e1ki1 : e1ki1;
                        fki(2,1) = fn2;

                        fki(3,0) = rev1 ? eNE - e1ki : e1ki;
                        fki(3,1) = fn2;

                        MFEM_ASSERT(fki(0,0) == fki(3,0) && fki(1,0) == fki(2,0), "");
                     }
                     else
                     {
                        MFEM_ASSERT(edgeV[0][0] == verts[1] ||
                                    edgeV[0][edgeV[0].Size() - 1] == verts[1], "");
                        MFEM_ASSERT(edgeV[1][0] == verts[0] ||
                                    edgeV[1][edgeV[0].Size() - 1] == verts[0], "");
                        MFEM_ASSERT(eNE == fn2, "");
                        MFEM_ASSERT(edgeVki[1][edgeVki[1].Size() - 1] == fn2, "");

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

                        MFEM_ASSERT(fki(0,1) == fki(3,1) && fki(1,1) == fki(2,1), "");
                     }
                  }

                  const int vmin = fverts.Min();
                  const int idmin = fverts.Find(vmin);

                  if (edgeE[0][e] >= 0)
                  {
                     std::pair<int, int> vpair(fverts[idmin], fverts[(idmin + 2) % 4]);
                     const bool vPairTopo = v2f.count(vpair) > 0;
                     if (!vPairTopo)
                     {
                        continue;
                     }

                     const int sface = v2f.at(std::pair<int, int> (fverts[idmin],
                                                                   fverts[(idmin + 2) % 4]));
                     addParentFaces.insert(f);

                     // Set facePairs

                     // TODO: refactor this repeated code to find vMinID

                     // Find the vertex with minimum knot indices.
                     int vMinID = -1;
                     {
                        Array<int> kiMin(2);
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
                              MFEM_ASSERT(vMinID == -1, "");
                              vMinID = i;
                           }
                        }
                     }

                     MFEM_ASSERT(vMinID >= 0, "");

                     Array<int> fvertsMasterOrdering(4);
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

                     facePairs.emplace_back(FacePairInfo{fverts[vMinID], sface,
                                                         f, ori_sface,
                     {fki(vMinID,0), fki(vMinID,1)},
                     {
                        fki((vMinID + 2) % 4,0) - fki(vMinID,0),
                        fki((vMinID + 2) % 4,1) - fki(vMinID,1)
                     }});
                  }
                  else  // Auxiliary face
                  {
                     // TODO: use a fast, efficient look-up, using something like auxv2f.
                     const int nauxf = auxFaces.size();
                     Array<int> afverts(4);

                     Array<int> fvertsSorted(fverts);
                     fvertsSorted.Sort();

                     int afid = -1;
                     for (int af = 0; af < nauxf; ++af)
                     {
                        for (int i=0; i<4; ++i)
                        {
                           afverts[i] = auxFaces[af].v[i];
                        }

                        afverts.Sort();

                        if (afverts == fvertsSorted)
                        {
                           MFEM_VERIFY(afid == -1, "");
                           afid = af;
                        }
                     }

                     addParentFaces.insert(f);

                     // Find the vertex with minimum knot indices.
                     int vMinID = -1;
                     {
                        Array<int> kiMin(2);
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
                              MFEM_ASSERT(vMinID == -1, "");
                              vMinID = i;
                           }
                        }
                     }

                     MFEM_ASSERT(vMinID >= 0, "");

                     if (afid >= 0)
                     {
                        // Find orientation of ordered vertices for this face, in fvertsSorted, w.r.t. the auxFaces ordering.

                        for (int i=0; i<4; ++i)
                        {
                           afverts[i] = auxFaces[afid].v[i];
                        }

                        int ori_f2 = -1;  // TODO: better name?
                        for (int i=0; i<4; ++i)
                        {
                           if (ori_f >= 0)
                           {
                              fvertsSorted[i] = fverts[(vMinID + i) % 4];
                           }
                           else
                           {
                              fvertsSorted[i] = fverts[(vMinID + 4 - i) % 4];
                           }

                           if (fvertsSorted[i] == afverts[0])
                           {
                              ori_f2 = i;
                           }
                        }

                        MFEM_ASSERT(ori_f2 >= 0, "");

                        if (fvertsSorted[(ori_f2 + 1) % 4] != afverts[1])
                        {
                           for (int j=0; j<4; ++j)
                           {
                              MFEM_ASSERT(fvertsSorted[(ori_f2 + 4 - j) % 4] == afverts[j], "");
                           }

                           ori_f2 = -1 - ori_f2;
                        }
                        else
                        {
                           for (int j=0; j<4; ++j)
                           {
                              MFEM_ASSERT(fvertsSorted[(ori_f2 + j) % 4] == afverts[j], "");
                           }
                        }

                        facePairs.emplace_back(FacePairInfo{fverts[vMinID], -1 - afid,
                                                            f, ori_f2,
                        {fki(vMinID,0), fki(vMinID,1)},
                        {
                           fki((vMinID + 2) % 4,0) - fki(vMinID,0),
                           fki((vMinID + 2) % 4,1) - fki(vMinID,1)
                        }});
                     }
                     else
                     {
                        // Create a new auxiliary face.
                        const int auxFaceId = auxFaces.size();
                        // Find the knot indices of the vertices in fverts,
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

                        // Orientation is 0 for a new auxiliary face, by construction.
                        ori_f = 0;

                        auxFace.parent = f;
                        auxFace.ori = ori_f;
                        for (int i=0; i<2; ++i)
                        {
                           auxFace.ki0[i] = fki(vMinID,i);
                           auxFace.ki1[i] = fki((vMinID + 2) % 4,i);
                        }

                        auxFaces.push_back(auxFace);

                        facePairs.emplace_back(FacePairInfo{fverts[vMinID], -1 - auxFaceId,
                                                            f, ori_f,
                        {fki(vMinID,0), fki(vMinID,1)},
                        {
                           fki((vMinID + 2) % 4,0) - fki(vMinID,0),
                           fki((vMinID + 2) % 4,1) - fki(vMinID,1)
                        }});
                     }
                  }
               }
            }
         }  // if (bothMaster)
      }  // Pair (p) loop
   }  // f
}

void NURBSExtension::ProcessFacePairs(int start, int midStart,
                                      const std::vector<int> & parentN1,
                                      const std::vector<int> & parentN2,
                                      std::vector<int> & parentVerts,
                                      const std::vector<FacePairInfo> & facePairs)
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
      const int i = facePairs[q].ki[0];
      const int j = facePairs[q].ki[1];
      const int nfe1 = facePairs[q].ne[0];  // Number of elements, direction 1
      const int nfe2 = facePairs[q].ne[1];  // Number of elements, direction 2
      const int v0 = facePairs[q].v0;  // Bottom-left corner vertex of child face
      const int childFace = facePairs[q].child;
      const int parentFace = facePairs[q].parent;
      const int cpori =
         facePairs[q].ori;  // Orientation for childFace w.r.t. parentFace
      const int mid = masterFaceToId.at(parentFace);

      if (mid < midStart)
      {
         // Ignore data about master faces already processed.
         // TODO: instead of just ignoring, should we verify that the new data matches the old?
         continue;
      }

      MFEM_VERIFY(0 <= i && i < parentN1[mid], "");
      MFEM_VERIFY(0 <= j && j < parentN2[mid], "");
      if (mid != midPrev)  // Next parent face
      {
         Array<int> pv(parentVerts.data() + (4*mid), 4);
         const int ori = GetFaceOrientation(patchTopo, parentFace, pv);
         // Ori is the signed shift such that pv[abs1(ori)] is vertex 0 of parentFace,
         // and the relative direction of the ordering is encoded in the sign.

         if (q > start && midPrev >= 0)
         {
            // For the previous parentFace, use previous orientation
            // to reorder masterFaceSlaves, masterFaceSlaveCorners,
            // masterFaceSizes, masterFaceVerts.
            std::vector<int> s0(2);
            const bool rev = Reorder2D(parentN1[midPrev], parentN2[midPrev],
                                       orientation, s0);
            masterFaceS0[midPrev] = s0;
            masterFaceRev[midPrev] = rev;
            Reorder2D(parentN1[midPrev], parentN2[midPrev],
                      orientation, s0);
         }

         orientation = ori;

         midPrev = mid;
      }  // next parent face

      slaveFaces.push_back(childFace);
      slaveFaceI.push_back(i);
      slaveFaceJ.push_back(j);
      slaveFaceNE1.push_back(nfe1);
      slaveFaceNE2.push_back(nfe2);
      slaveFaceOri.push_back(cpori);

      const int si = slaveFaces.size() - 1;
      masterFaceSlaves[mid].push_back(si);
      masterFaceSlaveCorners[mid].push_back(v0);

      masterFaceSizes[mid][0] = parentN1[mid];
      masterFaceSizes[mid][1] = parentN2[mid];

      if (i < parentN1[mid] - 1 && j < parentN2[mid] - 1)
      {
         // Find the interior vertex associated with this child face.

         // For left-most faces, use right side of face, else left side.
         const int vi0 = (i == 0) ? 1 : 0;

         // For bottom-most faces, use top side of face, else bottom side.
         const int vi1 = (j == 0) ? 1 : 0;

         // Get the face vertex at position (vi0, vi1) of the quadrilateral child face.

         int qid[2][2] = {{0, 3}, {1, 2}};

         const int vid = qid[vi0][vi1];

         // Find the index of vertex v0, which is at the bottom-left corner.
         Array<int> vert;
         if (childFace >= 0)
         {
            patchTopo->GetFaceVertices(childFace, vert);
         }
         else
         {
            // Auxiliary Face
            GetAuxFaceVertices(-1 - childFace, vert);
         }

         MFEM_VERIFY(vert.Size() == 4, "TODO: remove this obvious check");
         const int v0id = vert.Find(v0);
         MFEM_VERIFY(v0id >= 0, "");

         // Set the interior vertex associated with this child face.
         const int vint = vert[(vid - v0id + 4) % 4];
         masterFaceVerts[mid].push_back(vint);  // TODO: not used?
      }
   }  // Loop (q) over facePairs

   // TODO: restructure the above loop (q) over nfpairs to avoid this copying of code for Reorder2D.
   if (midPrev >= 0)
   {
      std::vector<int> s0(2);
      const bool rev = Reorder2D(parentN1[midPrev], parentN2[midPrev],
                                 orientation, s0);

      masterFaceS0[midPrev] = s0;
      masterFaceRev[midPrev] = rev;
      Reorder2D(parentN1[midPrev], parentN2[midPrev],
                orientation, s0);
   }
}

void NURBSExtension::GetAuxEdgeVertices(int auxEdge, Array<int> &verts) const
{
   verts.SetSize(2);
   for (int i=0; i<2; ++i)
   {
      verts[i] = auxEdges[auxEdge].v[i];
   }
}

void NURBSExtension::GetAuxFaceVertices(int auxFace, Array<int> & verts) const
{
   verts.SetSize(4);
   for (int i=0; i<4; ++i)
   {
      verts[i] = auxFaces[auxFace].v[i];
   }
}

// Note that v2e is a map only for conforming patchTopo->ncmesh edges.
// Auxiliary edges are not included.
void NURBSExtension::GetAuxFaceEdges(int auxFace,
                                     Array<int> & edges) const
{
   edges.SetSize(4);
   Array<int> verts(2);
   for (int i=0; i<4; ++i)
   {
      for (int j=0; j<2; ++j)
      {
         verts[j] = auxFaces[auxFace].v[(i + j) % 4];
      }

      verts.Sort();

      const std::pair<int, int> edge_v(verts[0], verts[1]);
      if (v2e.count(edge_v) > 0)
      {
         edges[i] = v2e.at(edge_v);  // patchTopo edge
      }
      else
      {
         // Auxiliary edge

         // TODO: use a map like auxv2e in ProcessVertexToKnot3D for
         // efficient look-up of auxiliary edges.
         int auxEdge = -1;
         const int nauxe = auxEdges.size();
         Array<int> aeverts(2);
         for (int ae = 0; ae < nauxe; ++ae)
         {
            for (int j=0; j<2; ++j)
            {
               aeverts[j] = auxEdges[ae].v[j];
            }

            aeverts.Sort();

            if (aeverts == verts)
            {
               MFEM_ASSERT(auxEdge == -1, "");
               auxEdge = ae;
            }
         }

         MFEM_ASSERT(auxEdge >= 0, "");
         edges[i] = -1 - auxEdge;
      }
   }
}

void NURBSPatchMap::GetMasterEdgeDofs(int edge,
                                      const Array<int>& v_offsets,
                                      const Array<int>& e_offsets,
                                      const Array<int>& aux_e_offsets,
                                      Array<int> & dofs)
{
   {
      const bool isMaster = Ext->masterEdges.count(edge) > 0;
      MFEM_VERIFY(isMaster, "");
   }

   const int mid = Ext->masterEdgeToId.at(edge);
   MFEM_ASSERT(mid >= 0, "Master edge index not found");

   MFEM_VERIFY(Ext->masterEdgeVerts[mid].size() ==
               Ext->masterEdgeSlaves[mid].size() - 1, "");

   const std::size_t nes = Ext->masterEdgeSlaves[mid].size();
   for (std::size_t s=0; s<nes; ++s)
   {
      const int slaveId = Ext->slaveEdges[Ext->masterEdgeSlaves[mid][s]];

      Array<int> svert;
      if (slaveId >= 0)
      {
         Ext->patchTopo->GetEdgeVertices(slaveId, svert);
      }
      else
      {
         // Auxiliary edge
         Ext->GetAuxEdgeVertices(-1 - slaveId, svert);
      }

      bool reverse = false;
      if (nes > 1)
      {
         const int mev = Ext->masterEdgeVerts[mid][std::max((int) s - 1,0)];
         MFEM_VERIFY(mev == svert[0] || mev == svert[1], "");
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

      const int eos = slaveId >= 0 ? e_offsets[slaveId] : aux_e_offsets[-1 - slaveId];

      // TODO: in 3D, the next offset would be f_offsets[0], not
      // p_offsets[0]. This needs to be generalized in an elegant way.
      // How about appending the next offset to the end of e_offsets?
      // Would increasing the size of e_offsets by 1 break something?

      const int eos1 = slaveId >= 0 ? (slaveId + 1 < e_offsets.Size() ?
                                       e_offsets[slaveId + 1] : aux_e_offsets[0]) :
                       aux_e_offsets[-slaveId];

      const int nvs = eos1 - eos;

      MFEM_VERIFY(nvs >= 0, "");

      // Add all slave edge vertices/DOFs

      Array<int> sdofs(nvs);
      for (int j=0; j<nvs; ++j)
      {
         sdofs[j] = reverse ? eos1 - 1 - j : eos + j;
      }

      dofs.Append(sdofs);

      if (s < Ext->masterEdgeSlaves[mid].size() - 1)
      {
         // Add interior vertex DOF
         dofs.Append(v_offsets[Ext->masterEdgeVerts[mid][s]]);
      }
   }
}

// This sets masterDofs but does not change e_meshOffsets.
void NURBSPatchMap::SetMasterEdges(bool dof, const KnotVector *kv[])
{
   const Array<int>& v_offsets = dof ? Ext->v_spaceOffsets : Ext->v_meshOffsets;
   const Array<int>& e_offsets = dof ? Ext->e_spaceOffsets : Ext->e_meshOffsets;

   const Array<int>& aux_e_offsets = dof ? Ext->aux_e_spaceOffsets :
                                     Ext->aux_e_meshOffsets;

   edgeMaster.SetSize(edges.Size());
   edgeMasterOffset.SetSize(edges.Size());
   masterDofs.SetSize(0);

   int mos = 0;
   for (int i=0; i<edges.Size(); ++i)
   {
      edgeMaster[i] = Ext->masterEdges.count(edges[i]) > 0;
      edgeMasterOffset[i] = mos;

      if (edgeMaster[i])
      {
         const int mid = Ext->masterEdgeToId.at(edges[i]);
         MFEM_VERIFY(mid >= 0, "Master edge index not found");
         MFEM_VERIFY(Ext->masterEdgeSlaves[mid].size() != 0,
                     "False master edges should have been removed");

         Array<int> mdof;
         GetMasterEdgeDofs(edges[i], v_offsets, e_offsets, aux_e_offsets, mdof);
         masterDofs.Append(mdof);

         mos += mdof.Size();
      }
   }
}

// The input is assumed to be such that the face of patchTopo has {n1,n2}
// interior entities in master face directions {1,2}; v0 is the bottom-left
// vertex with respect to the master face directions; edges {e1,e2} are local
// edges of the face on the bottom and right side (master face directions). We
// find the permutation perm of face interior entities such that entity perm[i]
// of the face should be entity i in the master face ordering.
// Note that, in the above comments, it is irrelevant whether entities are interior.
void NURBSPatchMap::GetFaceOrdering(int face, int n1, int n2, int v0,
                                    int e1, int e2, Array<int> & perm)
{
   perm.SetSize(n1 * n2);

   // The ordering of entities in the face is based on the vertices.

   Array<int> faceEdges, ori, evert, e2vert, vert;
   Ext->patchTopo->GetFaceEdges(face, faceEdges, ori);
   Ext->patchTopo->GetFaceVertices(face, vert);

   MFEM_VERIFY(vert.Size() == 4, "");
   int v0id = -1;
   for (int i=0; i<4; ++i)
   {
      if (vert[i] == v0)
      {
         v0id = i;
      }
   }

   MFEM_VERIFY(v0id >= 0, "");

   Ext->patchTopo->GetEdgeVertices(faceEdges[e1], evert);
   MFEM_VERIFY(evert[0] == v0 || evert[1] == v0, "");

   bool d[2];
   d[0] = (evert[0] == v0);

   const int v10 = d[0] ? evert[1] : evert[0];

   // The face has {fn1,fn2} interior entities, with ordering based on `vert`.
   // Now, we find these sizes, by first finding the edge with vertices [v0, v10].
   int e0 = -1;
   for (int i=0; i<4; ++i)
   {
      Ext->patchTopo->GetEdgeVertices(faceEdges[i], evert);
      if ((evert[0] == v0 && evert[1] == v10) ||
          (evert[1] == v0 && evert[0] == v10))
      {
         e0 = i;
      }
   }

   MFEM_VERIFY(e0 >= 0, "");

   const bool tr = e0 % 2 == 1;  // True means (fn1,fn2) == (n2,n1)

   Ext->patchTopo->GetEdgeVertices(faceEdges[e2], evert);
   MFEM_VERIFY(evert[0] == v10 || evert[1] == v10, "");
   d[1] = (evert[0] == v10);

   const int v11 = d[1] ? evert[1] : evert[0];

   int v01 = -1;
   for (int i=0; i<4; ++i)
   {
      if (vert[i] != v0 && vert[i] != v10 && vert[i] != v11)
      {
         v01 = vert[i];
      }
   }

   MFEM_VERIFY(v01 >= 0 && v01 == vert.Sum() - v0 - v10 - v11, "");

   // Translate indices [v0, v10, v11, v01] to pairs of indices in {0,1}.
   const int ipair[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
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

      MFEM_VERIFY(locv[i] >= 0, "");
   }

   for (int i=0; i<2; ++i)
   {
      f00[i] = ipair[locv[0]][i];
   }

   const int i0 = f00[0];
   const int j0 = f00[1];

   for (int i=0; i<n1; ++i)
      for (int j=0; j<n2; ++j)
      {
         // Entity perm[i] of the face should be entity i in the master face ordering.
         // The master face ordering varies faster in the direction from v0 to v10,
         // and slower in the direction from v10 to v11, or equivalently, from v0 to v01.
         if (tr)
         {
            const int fi = i0 == 0 ? j : n2 - 1 - j;
            const int fj = j0 == 0 ? i : n1 - 1 - i;
            const int p = fi + (fj * n2);  // Index in face ordering
            const int m = i + (j * n1);  // Index in the master face ordering
            perm[m] = p;
         }
         else
         {
            const int fi = i0 == 0 ? i : n1 - 1 - i;
            const int fj = j0 == 0 ? j : n2 - 1 - j;
            const int p = fi + (fj * n1);  // Index in face ordering
            const int m = i + (j * n1);  // Index in the master face ordering
            perm[m] = p;
         }
      }
}

bool ConsistentlySetEntry(int v, int & e)
{
   const bool consistent = e == -1 || e == v;
   e = v;
   return consistent;
}

void ReorderArray2D(bool rev, int i0, int j0, const Array2D<int> & a,
                    Array2D<int> & b)
{
   const int ma = a.NumRows();
   const int na = a.NumCols();

   const int mb = rev ? na : ma;
   const int nb = rev ? ma : na;

   b.SetSize(mb, nb);

   if (rev)
   {
      MFEM_ABORT("TODO");
   }
   else
   {
      const int s0 = i0 == 0 ? 1 : -1;
      const int s1 = j0 == 0 ? 1 : -1;
      for (int i=0; i<mb; ++i)
      {
         const int ia = (i0 * (ma - 1)) + (s0 * i);
         for (int j=0; j<nb; ++j)
         {
            const int ja = (j0 * (na - 1)) + (s1 * j);
            b(i, j) = a(ia, ja);
         }
      }
   }
}

void GetVertexOrdering(int ori, Array<int> & perm)
{
   perm.SetSize(4);
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

// This sets masterDofs but does not change f_meshOffsets.
void NURBSPatchMap::SetMasterFaces(bool dof)
{
   const Array<int>& v_offsets = dof ? Ext->v_spaceOffsets : Ext->v_meshOffsets;
   const Array<int>& e_offsets = dof ? Ext->e_spaceOffsets : Ext->e_meshOffsets;
   const Array<int>& f_offsets = dof ? Ext->f_spaceOffsets : Ext->f_meshOffsets;

   const Array<int>& aux_e_offsets = dof ? Ext->aux_e_spaceOffsets :
                                     Ext->aux_e_meshOffsets;
   const Array<int>& aux_f_offsets = dof ? Ext->aux_f_spaceOffsets :
                                     Ext->aux_f_meshOffsets;

   faceMaster.SetSize(faces.Size());
   faceMasterOffset.SetSize(faces.Size());

   // The loop over master edges is already done by SetMasterEdges, and now we
   // append face DOFs to masterDofs.

   int mos = masterDofs.Size();
   for (int i=0; i<faces.Size(); ++i)
   {
      faceMaster[i] = Ext->masterFaces.count(faces[i]) > 0;
      faceMasterOffset[i] = mos;

      if (!faceMaster[i]) { continue; }

      const int mid = Ext->masterFaceToId.at(faces[i]);
      MFEM_ASSERT(mid >= 0, "Master face index not found");

      const bool rev = Ext->masterFaceRev[mid];
      const int s0i = Ext->masterFaceS0[mid][0];
      const int s0j = Ext->masterFaceS0[mid][1];

      const int n1orig = Ext->masterFaceSizes[mid][0];
      const int n2orig = Ext->masterFaceSizes[mid][1];

      // TODO: use a better way to mark master faces without slave faces, instead
      // of just failing to set Ext->masterFaceSizes[mid].
      // Skip master faces with no slave faces (only having slave edges).
      if (n1orig == 0 && n2orig == 0)
      {
         faceMaster[i] = false;
         continue;
      }

      const int n1 = rev ? n2orig : n1orig;
      const int n2 = rev ? n1orig : n2orig;

      MFEM_VERIFY(n1 > 1 || n2 > 1, "");
      MFEM_VERIFY(n1 * n2 >= (int) Ext->masterFaceSlaves[mid].size(),
                  "Inconsistent number of faces");

      MFEM_VERIFY((n1 - 1) * (n2 - 1) >= (int) Ext->masterFaceVerts[mid].size(),
                  "Inconsistent number of vertices");

      // TODO: define masterFaceSlaveCorners at master face corners omitted from V2K?

      int fcnt = 0;
      for (auto slaveId : Ext->masterFaceSlaves[mid])
      {
         if (slaveId < 0)
         {
            // Auxiliary face
            MFEM_ABORT("TODO: aux face implementation is not done");
         }
         else
         {
            fcnt += Ext->slaveFaceNE1[slaveId] * Ext->slaveFaceNE2[slaveId];
         }
      }

      MFEM_VERIFY(fcnt == n1 * n2, "");

      MFEM_VERIFY((int) Ext->masterFaceSlaveCorners[mid].size() <= n1*n2, "");

      // Set an array of vertices or DOFs for the interior of this master face, `faces[i]`.
      // Set master face entity dimensions.
      int mnf1, mnf2;
      Array<int> medges;
      {
         Array<int> mori;
         Ext->patchTopo->GetFaceEdges(faces[i], medges, mori);
      }

      MFEM_ASSERT(medges.Size() == 4, "");

      if (dof)
      {
         mnf1 = Ext->KnotVec(medges[0])->GetNCP() - 2;
         mnf2 = Ext->KnotVec(medges[1])->GetNCP() - 2;
      }
      else
      {
         mnf1 = Ext->KnotVec(medges[0])->GetNE() - 1;
         mnf2 = Ext->KnotVec(medges[1])->GetNE() - 1;
      }

      // Set dimensions for a single mesh edge.
      const int sne1 = (mnf1 - n1 + 1) / n1;
      const int sne2 = (mnf2 - n2 + 1) / n2;

      MFEM_ASSERT(sne1 * n1 == mnf1 - n1 + 1, "");
      MFEM_ASSERT(sne2 * n2 == mnf2 - n2 + 1, "");

      Array2D<int> mdof(mnf1, mnf2);
      mdof = -1;

      bool consistent = true;

      for (std::size_t s=0; s<Ext->masterFaceSlaves[mid].size(); ++s)
      {
         const int sId = Ext->masterFaceSlaves[mid][s];
         const int slaveId = Ext->slaveFaces[sId];
         const int v0 = Ext->masterFaceSlaveCorners[mid][s];
         const int ori = Ext->slaveFaceOri[sId];
         // ori gives the orientation and index of cv matching the first
         // vertex of childFace, where cv is the array of slave face
         // vertices in CCW w.r.t. to the master face.

         const int sI = Ext->slaveFaceI[sId];
         const int sJ = Ext->slaveFaceJ[sId];
         const int ne1 = Ext->slaveFaceNE1[sId];
         const int ne2 = Ext->slaveFaceNE2[sId];

         const int fos = slaveId >= 0 ? f_offsets[slaveId] : aux_f_offsets[-1 - slaveId];
         const int fos1 = slaveId >= 0 ? (slaveId + 1 < f_offsets.Size() ?
                                          f_offsets[slaveId + 1] : aux_f_offsets[0]) :
                          aux_f_offsets[-slaveId];

         const int nvs = fos1 - fos;

         if (slaveId < 0)
         {
            // Auxiliary face
            const int auxFace = -1 - slaveId;

            // If ori >= 0, then vertex ori of the aux face is closest to vertex 0 of
            // the parent face. If ori < 0, then the orientations of the aux face and parent face
            // are reversed.

            // TODO: eliminate auxFaceOri? Isn't it always equal to Ext->slaveFaceOri[sId]?
            // Actually, I think auxFaces has ori defined as 0 for its original parent face,
            // and ori here is Ext->slaveFaceOri, which comes from facePairs, which can be
            // w.r.t. a different parent face. This means auxFaceOri is always 0 and is not
            // useful, so that only ori should be used.
            //MFEM_VERIFY(auxFaceOri == ori, "");

            // TODO: refactor the duplicated code in this case and the other case, slaveId >= 0.

            // Set slave face entity dimensions.
            int nf1, nf2;
            if (dof)
            {
               nf1 = (sne1 * ne1) + ne1 - 1;
               nf2 = (sne2 * ne2) + ne2 - 1;
            }
            else
            {
               nf1 = Ext->auxFaces[auxFace].ki1[0] - Ext->auxFaces[auxFace].ki0[0] - 1;
               nf2 = Ext->auxFaces[auxFace].ki1[1] - Ext->auxFaces[auxFace].ki0[1] - 1;
            }

            MFEM_ASSERT(sne1 * ne1 == nf1 - ne1 + 1, "");
            MFEM_ASSERT(sne2 * ne2 == nf2 - ne2 + 1, "");

            MFEM_VERIFY(nvs == nf1 * nf2, "");

            /*
               NOTE: When an aux face is first defined, it is constructed
            with orientation 0 w.r.t. its parent face. However, it can be
            part of another parent face. In this case, the original aux
            face index is used, but it is paired with a different parent
            face, with possibly different orientation w.r.t. that parent
            face. In NURBSPatchMap::SetMasterFaces, the DOFs are set on
            the parent face in its knot indices, with an aux face of
            possibly nonzero orientation. When the orientation is nonzero,
            the aux face DOFs must be reordered for the parent face, based
            on orientation.
             */

            // These offsets are for the interior entities of the face. To get the lower edge, subtract 1.
            const int os1 = (sI * sne1) + sI;
            const int os2 = (sJ * sne2) + sJ;

            // TODO: is there a function to permute 2D indices based on orientation? GetShiftedGridPoints2D
            // TODO: the problem with using GetShiftedGridPoints2D is that we have (nf1, nf2) for the shifted (new)
            // face, not for the original (old) face.
            int onf1, onf2;
            GetInverseShiftedDimensions2D(ori, nf1, nf2, onf1, onf2);
            for (int k=0; k<onf2; ++k)
               for (int j=0; j<onf1; ++j)
               {
                  int sm, sn, sj, sk;
                  GetShiftedGridPoints2D(onf1, onf2, j, k, ori, sm, sn, sj, sk);
                  MFEM_VERIFY(sm == nf1 && sn == nf2, "TODO: remove this?");

                  const int q = j + (k * onf1);
                  if (!ConsistentlySetEntry(fos + q, mdof(os1 + sj, os2 + sk)))
                  {
                     consistent = false;
                  }
               }

            // Set entries on edges of this face, if they are interior to the master face.
            std::vector<bool> edgeBdry(4);

            // Horizontal edges
            edgeBdry[0] = sJ == 0;
            edgeBdry[2] = sJ + ne2 == n2;

            // Vertical edges
            edgeBdry[1] = sI + ne1 == n1;
            edgeBdry[3] = sI == 0;

            Array<int> faceEdges;
            Ext->GetAuxFaceEdges(auxFace, faceEdges);

            Array<int> orderedVertices(4);
            Array<int> perm(4);
            GetVertexOrdering(ori, perm);

            std::set<int> vbdry;

            int vstart = v0;
            for (int eidx=0; eidx<4; ++eidx)
            {
               orderedVertices[eidx] = vstart;
               // TODO: eliminate orderedVertices if it is redundant.
               MFEM_VERIFY(orderedVertices[eidx] == Ext->auxFaces[auxFace].v[perm[eidx]], "");

               const int eperm = ori < 0 ? (perm[eidx] - 1 + 4) % 4 : perm[eidx];
               const int edge = faceEdges[eperm];

               Array<int> evert;

               if (edge >= 0)
               {
                  Ext->patchTopo->GetEdgeVertices(edge, evert);
               }
               else
               {
                  const int auxEdge = -1 - edge;
                  Ext->GetAuxEdgeVertices(auxEdge, evert);
               }

               MFEM_VERIFY(evert[0] == vstart || evert[1] == vstart, "");

               const bool reverse = (vstart == evert[1]);
               const int vend = evert.Sum() - vstart;
               vstart = vend;

               // Skip edges on the boundary of the master face.
               if (edgeBdry[eidx])
               {
                  for (auto v : evert)
                  {
                     vbdry.insert(v);
                  }

                  continue;
               }
               const bool horizontal = (eidx % 2 == 0);
               const int nf_e = horizontal ? nf1 : nf2;

               // Edge entities
               const int eos = edge >= 0 ? e_offsets[edge] : aux_e_offsets[-1 - edge];
               const int eos1 = edge >= 0 ? ((edge + 1 < e_offsets.Size()) ?
                                             e_offsets[edge + 1] : aux_e_offsets[0]) : aux_e_offsets[-edge];

               const bool edgeIsMaster = Ext->masterEdges.count(edge) > 0;

               Array<int> edofs;
               if (edgeIsMaster)
               {
                  // This edge is a slave edge and a master edge.
                  // Instead of getting DOFs from e_offsets, take them
                  // from the slave edges of this edge.
                  GetMasterEdgeDofs(edge, v_offsets, e_offsets,
                                    aux_e_offsets, edofs);
               }
               else
               {
                  MFEM_VERIFY(eos1 - eos == nf_e, "");

                  edofs.SetSize(nf_e);
                  for (int j=0; j<nf_e; ++j)
                  {
                     edofs[j] = eos + j;
                  }
               }

               MFEM_VERIFY(edofs.Size() == nf_e, "");

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
            }  // eidx

            // Set entries at vertices of this face, if they are interior to the master face.
            for (int vidx=0; vidx<4; ++vidx)
            {
               const int v = orderedVertices[vidx];
               if (vbdry.count(v) == 0)  // If not on the boundary of the master face.
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
            }  // vidx
         }
         else
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

            // Now, e1 is one of the two horizontal edges in the master face directions.
            // If it does not touch v0, then take the other horizontal edge.
            // Do the same for e2.

            Array<int> sedges;
            {
               Array<int> sori;
               Ext->patchTopo->GetFaceEdges(slaveId, sedges, sori);
               MFEM_VERIFY(sedges.Size() == 4, "TODO: remove this obvious check");
            }

            int v1 = -1;
            {
               Array<int> evert;
               Ext->patchTopo->GetEdgeVertices(sedges[e1], evert);
               if (evert.Find(v0) == -1)
               {
                  e1 += 2;
                  Ext->patchTopo->GetEdgeVertices(sedges[e1], evert);
               }

               const int idv0 = evert.Find(v0);
               MFEM_ASSERT(idv0 >= 0, "");
               v1 = evert[1 - idv0];

               Ext->patchTopo->GetEdgeVertices(sedges[e2], evert);
               if (evert.Find(v1) == -1)
               {
                  e2 += 2;
                  Ext->patchTopo->GetEdgeVertices(sedges[e2], evert);
               }

               MFEM_ASSERT(evert.Find(v1) >= 0, "");
            }

            // Set slave face entity dimensions.
            int nf1, nf2;
            if (dof)
            {
               nf1 = Ext->KnotVec(sedges[e1])->GetNCP() - 2;
               nf2 = Ext->KnotVec(sedges[e2])->GetNCP() - 2;
            }
            else
            {
               nf1 = Ext->KnotVec(sedges[e1])->GetNE() - 1;
               nf2 = Ext->KnotVec(sedges[e2])->GetNE() - 1;
            }

            MFEM_ASSERT(sne1 * ne1 == nf1 - ne1 + 1, "");
            MFEM_ASSERT(sne2 * ne2 == nf2 - ne2 + 1, "");

            MFEM_VERIFY(nvs == nf1 * nf2, "");

            // Find the DOFs of the slave face ordered for the master
            // face. We know that e1 and e2 are the local indices of
            // the slave face edges on the bottom and right side, with
            // respect to the master face directions.
            Array<int> perm;
            GetFaceOrdering(slaveId, nf1, nf2, v0, e1, e2, perm);

            // These offsets are for the interior entities of the face.
            // To get the lower edge, subtract 1.
            const int os1 = (sI * sne1) + sI;
            const int os2 = (sJ * sne2) + sJ;

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

            // Set entries on edges of this face, if they are interior to the master face.
            std::vector<int> edgeOrder{e1, e2, (e1 + 2) % 4, (e2 + 2) % 4};
            std::vector<bool> edgeBdry(4);

            // Horizontal edges
            edgeBdry[0] = sJ == 0;
            edgeBdry[2] = sJ + ne2 == n2;

            // Vertical edges
            edgeBdry[1] = sI + ne1 == n1;
            edgeBdry[3] = sI == 0;

            Array<int> orderedVertices(4);

            std::set<int> vbdry;

            int vstart = v0;
            for (int eidx=0; eidx<4; ++eidx)
            {
               orderedVertices[eidx] = vstart;

               const int edge = sedges[edgeOrder[eidx]];

               Array<int> evert;
               Ext->patchTopo->GetEdgeVertices(edge, evert);

               const bool reverse = (vstart == evert[1]);

               const int vend = evert.Sum() - vstart;
               vstart = vend;

               // Skip edges on the boundary of the master face.
               if (edgeBdry[eidx])
               {
                  for (auto v : evert)
                  {
                     vbdry.insert(v);
                  }

                  continue;
               }
               const bool horizontal = (eidx % 2 == 0);
               const int nf_e = horizontal ? nf1 : nf2;

               // Edge entities
               const int eos = e_offsets[edge];
               const int eos1 = (edge + 1 < e_offsets.Size()) ?
                                e_offsets[edge + 1] : aux_e_offsets[0];

               const bool edgeIsMaster = Ext->masterEdges.count(edge) > 0;

               Array<int> edofs;
               if (edgeIsMaster)
               {
                  // TODO: shouldn't the DOFs from this master edge be obtainable from SetMasterEdges?
                  // I guess SetMasterEdges is only for the patch edges, not master edges on the interior
                  // of a patch face.
                  // However, can we reuse/refactor code from SetMasterEdges?
                  GetMasterEdgeDofs(edge, v_offsets, e_offsets, aux_e_offsets, edofs);
               }
               else
               {
                  MFEM_VERIFY(eos1 - eos == nf_e, "");

                  edofs.SetSize(nf_e);
                  for (int j=0; j<nf_e; ++j)
                  {
                     edofs[j] = eos + j;
                  }
               }

               MFEM_VERIFY(edofs.Size() == nf_e, "");

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
            }  // eidx

            // Set entries at vertices of this face, if they are interior to the master face.
            for (int vidx=0; vidx<4; ++vidx)
            {
               const int v = orderedVertices[vidx];
               if (vbdry.count(v) == 0)  // If not on the boundary of the master face.
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
            }  // vidx
         }
      }  // Loop (s) over slave faces.

      // Let `ori` be the signed shift such that pv[abs1(ori)] is vertex 0 of parentFace,
      // and the relative direction of the ordering is encoded in the sign.
      // Here, pv means parent vertices, as ordered in the mesh file.
      // Then Reorder2D takes `ori` and computes (s0i, s0j) as the
      // corresponding integer coordinates in {0,1}x{0,1}. Thus reference
      // vertex (s0i, s0j) of pv (parent vertices in mesh file) is vertex
      // (0, 0) of parentFace.
      // Currently, mdof is in the ordering of pv, and the entries are now reordered,
      // according to parentFace vertex ordering, for appending to masterDofs.
      // That means the first entry appended to masterDofs should be the entry of
      // mdof corresponding to (s0i, s0j).

      Array2D<int> mdof_reordered;
      ReorderArray2D(rev, s0i, s0j, mdof, mdof_reordered);

      bool allset = true;
      for (int j=0; j<mdof_reordered.NumCols(); ++j)
         for (int k=0; k<mdof_reordered.NumRows(); ++k)
         {
            if (mdof_reordered(k,j) < 0)
            {
               allset = false;
            }

            masterDofs.Append(mdof_reordered(k,j));
         }

      MFEM_VERIFY(allset && consistent, "");

      mos += mnf1 * mnf2;
   }  // loop (i) over faces
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

void NURBSExtension::ProcessVertexToKnot2D(const Array2D<int> & v2k,
                                           std::set<int> & reversedParents,
                                           std::vector<EdgePairInfo> & edgePairs)
{
   auxEdges.clear();

   std::map<std::pair<int, int>, int> auxv2e;

   const int nv2k = v2k.NumRows();
   MFEM_VERIFY(4 == v2k.NumCols(), "");

   int prevParent = -1;
   int prevV = -1;
   int prevKI = -1;
   for (int i=0; i<nv2k; ++i)
   {
      const int tv = v2k(i,0);
      const int knotIndex = v2k(i,1);
      const int pv0 = v2k(i,2);
      const int pv1 = v2k(i,3);

      // Given that the parent Mesh is not yet constructed, and all we have at
      // this point is patchTopo->ncmesh, we should only define master/slave
      // edges by indices in patchTopo->ncmesh, as done in the case of nonempty
      // nce.masters. Now find the edge in patchTopo->ncmesh with vertices
      // (pv0, pv1), and define it as a master edge.

      const std::pair<int, int> parentPair(pv0 < pv1 ? pv0 : pv1,
                                           pv0 < pv1 ? pv1 : pv0);

      MFEM_VERIFY(v2e.count(parentPair) > 0, "Vertex pair not found");

      const int parentEdge = v2e[parentPair];
      masterEdges.insert(parentEdge);

      if (pv1 < pv0)
      {
         reversedParents.insert(parentEdge);
      }

      // Note that the logic here assumes that the "vertex_to_knot" data in the
      // mesh file has vertices in order of ascending knotIndex.

      const bool newParentEdge = (prevParent != parentEdge);
      const int v0 = newParentEdge ? pv0 : prevV;

      if (knotIndex == 1)
      {
         MFEM_VERIFY(newParentEdge, "");
      }

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
            auxEdges.emplace_back(AuxiliaryEdge{pv0 < pv1 ? parentEdge : -1 - parentEdge,
               {childPair.first, childPair.second},
               {newParentEdge ? 0 : prevKI, knotIndex}});
         }
      }

      const int childEdge = childPairTopo ? v2e[childPair] : -1 - auxv2e[childPair];

      // Check whether this is the final vertex in this parent edge.
      // TODO: this logic for comparing (pv0,pv1) to the next parents assumes
      // the ordering won't change. If the next v2k entry has (pv1,pv0), this
      // would cause a bug. An improvement in the implementation should avoid
      // this issue. Or is it not possible to change the order, since the knot
      // index is assumed to increase from pv0 to pv1?
      const bool finalVertex = (i == nv2k-1) || (v2k(i+1,2) != pv0) ||
                               (v2k(i+1,3) != pv1);

      edgePairs.emplace_back(tv, knotIndex, childEdge, parentEdge);

      if (finalVertex)
      {
         // Also find the edge with vertices (tv, pv1), and define it as a slave
         // edge.
         const std::pair<int, int> finalChildPair(tv < pv1 ? tv : pv1,
                                                  tv < pv1 ? pv1 : tv);
         const bool finalChildPairTopo = v2e.count(finalChildPair) > 0;
         if (!finalChildPairTopo)
         {
            // Check whether finalChildPair is in auxEdges.
            if (auxv2e.count(finalChildPair) == 0)
            {
               // Create a new auxiliary edge
               auxv2e[finalChildPair] = auxEdges.size();

               // -1 denotes `ne` at endpoint
               auxEdges.emplace_back(AuxiliaryEdge{pv0 < pv1 ? -1 - parentEdge : parentEdge,
                  {finalChildPair.first, finalChildPair.second},
                  {knotIndex, -1}});
            }
         }

         const int finalChildEdge = finalChildPairTopo ? v2e[finalChildPair] :
                                    -1 - auxv2e[finalChildPair];
         edgePairs.emplace_back(-1, -1, finalChildEdge, parentEdge);
      }

      prevV = tv;
      prevKI = knotIndex;
      prevParent = parentEdge;
   }  // loop over vertices in vertex_to_knot
}

// TODO: better code design.
void NURBSExtension::ProcessVertexToKnot3D(Array2D<int> const& v2k,
                                           const std::map<std::pair<int, int>, int> & v2f,
                                           std::vector<int> & parentN1,
                                           std::vector<int> & parentN2,
                                           std::vector<EdgePairInfo> & edgePairs,
                                           std::vector<FacePairInfo> & facePairs,
                                           std::vector<int> & parentFaces,
                                           std::vector<int> & parentVerts)
{
   auxEdges.clear();
   auxFaces.clear();

   std::map<std::pair<int, int>, int> auxv2e, auxv2f;

   // Each entry of v2k has the following 7 entries: tv, ki1, ki2, p0, p1, p2, p3
   constexpr int np = 7;  // Number of integers for each entry in v2k.

   const int nv2k = v2k.NumRows();
   MFEM_VERIFY(np == v2k.NumCols(), "");

   // Note that the logic here assumes that the "vertex_to_knot" data
   // in the mesh file has vertices in order of ascending (k1,k2), with k2
   // being the fast variable, and with corners skipped.
   // TODO: is this assumption still required and valid?

   // Find parentOffset, which stores the indices in v2k at which parent faces start.
   int prevParent = -1;
   std::vector<int> parentOffset;
   std::vector<bool> parentV2Kedge;
   int n1 = 0;
   int n2 = 0;
   int n1min = 0;
   int n2min = 0;
   for (int i = 0; i < nv2k; ++i)
   {
      const int ki1 = v2k(i,1);
      const int ki2 = v2k(i,2);

      std::vector<int> pv(4);
      for (int j=0; j<4; ++j)
      {
         pv[j] = v2k(i, 3 + j);
      }

      // The face with vertices (pv0, pv1, pv2, pv3) is defined as a parent face.
      const auto pvmin = std::min_element(pv.begin(), pv.end());
      const int idmin = std::distance(pv.begin(), pvmin);
      const int c0 = pv[idmin];  // First corner
      const int c1 = pv[(idmin + 2) % 4];  // Opposite corner

      const std::pair<int, int> parentPair(c0, c1);

      const int parentFace = v2f.at(parentPair);
      const bool newParentFace = (prevParent != parentFace);
      if (newParentFace)
      {
         parentOffset.push_back(i);
         parentFaces.push_back(parentFace);

         if (i > 0)
         {
            // TODO: change the comments. The usage of "knot" is wrong.

            // In the case of only 1 element in the 1-direction, it is assumed that
            // the 2-direction has more than 1 element, so there are knots (0, ki2)
            // and (1, ki2) for 0 < ki2 < n2. This will result in n1 = 0, which
            // should be 1. Also, n2 will be 1 less than it should be.
            // Similarly for the situation with directions reversed.
            // TODO: fix/test this in the 1-element case.

            const int n1range = n1 - n1min;
            const int n2range = n2 - n2min;
            parentV2Kedge.push_back(n1range == 0 || n2range == 0);
         }

         // TODO: refactor!
         {
            Array<int> ev(2);
            for (int j=0; j<2; ++j)
            {
               ev[j] = pv[j];
            }

            ev.Sort();
            const std::pair<int, int> edge_v(ev[0], ev[1]);
            const int edge = v2e.at(edge_v);

            n1 = KnotVecNE(edge);
            parentN1.push_back(n1);
         }

         {
            Array<int> ev(2);
            for (int j=0; j<2; ++j)
            {
               ev[j] = pv[j+1];
            }

            ev.Sort();
            const std::pair<int, int> edge_v(ev[0], ev[1]);
            const int edge = v2e.at(edge_v);

            n2 = KnotVecNE(edge);
            parentN2.push_back(n2);
         }

         n1 = ki1;  // Finding max of ki1
         n2 = ki2;  // Finding max of ki2

         n1min = n1;
         n2min = n2;
      }
      else
      {
         n1 = std::max(n1, ki1);  // Finding max of ki1
         n2 = std::max(n2, ki2);  // Finding max of ki2

         n1min = std::min(n1min, ki1);
         n2min = std::min(n2min, ki2);
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

      // Set all 4 edges of the parent face as master edges.
      {
         Array<int> ev(2);
         for (int i=0; i<4; ++i)
         {
            for (int j=0; j<2; ++j)
            {
               ev[j] = v2k(parentOffset[parent], 3 + ((i + j) % 4));
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

      n1 = parentN1[parent];
      n2 = parentN2[parent];
      Array2D<int> gridVertex(n1 + 1, n2 + 1);

      for (int i=0; i<=n1; ++i)
         for (int j=0; j<=n2; ++j)
         {
            gridVertex(i,j) = -1;
         }

      gridVertex(0,0) = v2k(parentOffset[parent],3);
      gridVertex(n1,0) = v2k(parentOffset[parent],4);
      gridVertex(n1,n2) = v2k(parentOffset[parent],5);
      gridVertex(0,n2) = v2k(parentOffset[parent],6);

      for (int i=0; i<4; ++i)
      {
         parentVerts.push_back(v2k(parentOffset[parent],3 + i));
      }

      int r1min = -1;
      int r1max = -1;
      int r2min = -1;
      int r2max = -1;

      for (int i = parentOffset[parent]; i < parentOffset[parent + 1]; ++i)
      {
         const int tv = v2k(i,0);
         const int ki1 = v2k(i,1);
         const int ki2 = v2k(i,2);

         gridVertex(ki1, ki2) = tv;

         if (i == parentOffset[parent])
         {
            // Initialize min/max
            r1min = ki1;
            r1max = ki1;

            r2min = ki2;
            r2max = ki2;
         }
         else
         {
            r1min = std::min(r1min, ki1);
            r1max = std::max(r1max, ki1);

            r2min = std::min(r2min, ki2);
            r2max = std::max(r2max, ki2);
         }
      } // loop over vertices in v2k

      const int n1set = r1max - r1min + 1;
      const int n2set = r2max - r2min + 1;

      MFEM_VERIFY(n1set * n2set >= parentOffset[parent + 1] - parentOffset[parent],
                  "");

      std::array<int, 2> rf;
      if (ref_factors.Size() == 3)
      {
         rf[0] = ref_factors[0];  // TODO: map the 3D entries of ref_factors to the 2 dimensions of this face
         rf[1] = ref_factors[0];  // TODO: map the 3D entries of ref_factors to the 2 dimensions of this face
      }
      else
      {
         for (int i=0; i<2; ++i)
         {
            rf[i] = 1;
         }
      }

      bool allset = true;
      bool hasSlaveFaces = false;
      bool hasAuxFace = false;
      for (int i=0; i<=n1; ++i)
         for (int j=0; j<=n2; ++j)
         {
            const bool origGrid = (i % rf[0] == 0) && (j % rf[1] == 0);
            if (!origGrid)
            {
               continue;
            }

            if (gridVertex(i,j) < 0)
            {
               allset = false;
            }
            else if (0 < i && i < n1 && 0 < j && j < n2)
            {
               hasSlaveFaces = true;
            }
         }

      const int d0 = rf[0];
      const int d1 = rf[1];

      // Loop over child faces and set facePairs, as well as auxiliary faces as needed.
      // TODO: just loop over (r1min, r1max) and (r2min, r2max)?
      const int n1orig = n1 / d0;
      const int n2orig = n2 / d1;

      MFEM_VERIFY(d0 * n1orig == n1 && d1 * n2orig == n2, "");

      for (int ii=0; ii<n1orig; ++ii)
         for (int jj=0; jj<n2orig; ++jj)
         {
            const int i = ii * d0;
            const int j = jj * d1;

            std::vector<int> cv(4);
            cv[0] = gridVertex(i, j);
            cv[1] = gridVertex(i + d0, j);
            cv[2] = gridVertex(i + d0, j + d1);
            cv[3] = gridVertex(i, j + d1);

            const auto cvmin = std::min_element(cv.begin(), cv.end());
            const int idmin = std::distance(cv.begin(), cvmin);
            const int c0 = cv[idmin];  // First corner
            const int c1 = cv[(idmin + 2) % 4];  // Opposite corner

            if (c0 < 0)  // This may occur, if gridVertex is not set everywhere.
            {
               continue;
            }

            const std::pair<int, int> childPair(c0 < c1 ? c0 : c1, c0 < c1 ? c1 : c0);
            const bool childPairTopo = v2f.count(childPair) > 0;
            if (childPairTopo)
            {
               const int childFace = v2f.at(childPair);
               Array<int> acv(cv.data(), 4);
               const int ori = GetFaceOrientation(patchTopo, childFace, acv);
               // ori gives the orientation and index of cv matching the first vertex of childFace.

               facePairs.emplace_back(FacePairInfo{cv[0], childFace, parentFace, ori,
                  {i, j}, {d0, d1}});
            }
            else
            {
               // Check whether the parent faces is on the boundary.
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
                     for (int k=0; k<4; ++k)
                     {
                        auxFace.v[k] = cv[k];
                     }

                     auxFace.parent = parentFace;
                     // Orientation is 0 for a new auxiliary face, by construction.
                     auxFace.ori = 0;

                     auxFace.ki0[0] = i;
                     auxFace.ki0[1] = j;
                     auxFace.ki1[0] = i + d0;
                     auxFace.ki1[1] = j + d1;

                     auxFaces.push_back(auxFace);

                     // Orientation is 0 for a new auxiliary face, by construction.
                     facePairs.emplace_back(FacePairInfo{cv[0], -1 - auxv2f[childPair], parentFace, 0,
                        {i, j}, {d0, d1}});
                  }
               }
            }
         }

      // Loop over child boundary edges and set edgePairs.
      for (int dir=1; dir<=2; ++dir)
      {
         const int ne = dir == 1 ? n1 : n2;
         for (int s=0; s<2; ++s)  // Loop over 2 sides for this direction.
         {
            const int parentEdge = parentEdges[dir == 1 ? 2*s : (2*s) + 1];
            const bool reverse_p = parentEdgeRev[dir == 1 ? 2*s : (2*s) + 1];
            const bool reverse = s == 0 ? reverse_p :
                                 !reverse_p;  // Sides with s=1 are reversed in defining parentEdgeRev.

            const bool parentVisited = visitedParentEdges.count(parentEdge) > 0;

            if (!parentVisited)
            {
               edgePairOS[parentEdge] = edgePairs.size();
               edgePairs.resize(edgePairs.size() + ne);
            }

            int tvprev = -1;  // TODO: change name
            int kiprev = -1;  // TODO: change name
            bool lagTV = false;

            int firstEdge = -1;

            const int de = d0;  // TODO: set this correctly for the direction of the edge.
            const int ne_orig = ne / de;
            MFEM_VERIFY(de * ne_orig == ne, "");

            for (int e_orig=0; e_orig<ne_orig; ++e_orig)  // edges in direction `dir`
            {
               const int e_i = de * e_orig;

               // For both directions, side s=0 has increasing indices and
               // s=1 has decreasing indices.

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

               int tv_int = -1;  // Top-vertex interior to the master edge
               int ki = -1;  // Knot index of tv_int, with respect to the master edge

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
               else if (e_i < ne - 1)  // Don't set to the endpoint
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

               if (cv[0] < 0)  // This may occur if gridVertex is not set everywhere.
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
                     tvprev = cv.Sum() - cv0;  // cv1
                     kiprev = i1;
                     lagTV = true;
                  }
               }

               const int tv = (e_idx == ne - de) ? -1 : tv_int;
               const int tvki = (e_idx == ne - de) ? -1 : (reverse ? ne - ki : ki);

               if (tv == -1)
               {
                  lagTV = true;
               }

               const std::pair<int, int> edge_i(cv[0], cv[1]);

               const int childEdge = v2e.at(edge_i);

               if (!parentVisited)
               {
                  // edgePairs is ordered starting from the vertex of lower index.
                  edgePairs[edgePairOS[parentEdge] + e_idx].Set(tv, tvki, childEdge, parentEdge);
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

      // Set auxiliary and patch-slave faces outside the set gridVertex.
      // Here, patch-slave refers to a slave face that is a face of a
      // neighboring patch and may contain multiple mesh faces.
      // In general, there can be at most 8 = 3^2 - 1 such faces.

      Array<int> gv1(4);
      Array<int> gv2(4);

      gv1[0] = 0;
      gv1[1] = r1min;
      gv1[2] = r1max;
      gv1[3] = n1;

      gv2[0] = 0;
      gv2[1] = r2min;
      gv2[2] = r2max;
      gv2[3] = n2;

      if (hasSlaveFaces && !allset)
      {
         for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j)
            {
               if (i == 1 && j == 1)  // Skip the middle, which is covered by gridVertex.
               {
                  continue;
               }

               // Define an auxiliary face (gv1[i], gv1[i+1]) x (gv2[j], gv2[j+1])

               // Skip degenerate faces
               if (gv1[i] == gv1[i+1] || gv2[j] == gv2[j+1])
               {
                  continue;
               }

               // TODO: the following code is copied and modified from above. Refactor into a function?
               std::vector<int> cv(4);
               std::vector<int> cv0(4);  // TODO: remove?
               std::vector<int> cv1(4);  // TODO: remove?

               cv[0] = gridVertex(gv1[i],gv2[j]);
               cv[1] = gridVertex(gv1[i+1],gv2[j]);
               cv[2] = gridVertex(gv1[i+1],gv2[j+1]);
               cv[3] = gridVertex(gv1[i],gv2[j+1]);

               cv0[0] = gv1[i];
               cv0[1] = gv1[i+1];
               cv0[2] = gv1[i+1];
               cv0[3] = gv1[i];

               cv1[0] = gv2[j];
               cv1[1] = gv2[j];
               cv1[2] = gv2[j+1];
               cv1[3] = gv2[j+1];

               const auto cvmin = std::min_element(cv.begin(), cv.end());
               const int idmin = std::distance(cv.begin(), cvmin);
               const int c0 = cv[idmin];  // First corner
               const int c1 = cv[(idmin + 2) % 4];  // Opposite corner

               MFEM_ASSERT(c0 >= 0, "");

               const std::pair<int, int> childPair(c0, c1);
               const bool childPairTopo = v2f.count(childPair) > 0;
               if (childPairTopo)
               {
                  const int childFace = v2f.at(childPair);
                  Array<int> acv(cv.data(), 4);

                  // ori gives the orientation and index of cv matching the
                  // first vertex of childFace.
                  const int ori = GetFaceOrientation(patchTopo, childFace, acv);
                  facePairs.emplace_back(FacePairInfo{cv[0], childFace, parentFace, ori,
                     {gv1[i], gv2[j]}, {gv1[i+1] - gv1[i], gv2[j+1] - gv2[j]}});
               }
               else
               {
                  hasAuxFace = true;

                  // Check whether childPair is in auxFaces.
                  if (auxv2f.count(childPair) == 0)
                  {
                     // Create a new auxiliary face
                     auxv2f[childPair] = auxFaces.size();
                     AuxiliaryFace auxFace;
                     for (int k=0; k<4; ++k)
                     {
                        auxFace.v[k] = cv[k];
                     }

                     auxFace.parent = parentFace;
                     // Orientation is 0 for a new auxiliary face, by construction.
                     auxFace.ori = 0;

                     auxFace.ki0[0] = gv1[i];
                     auxFace.ki0[1] = gv2[j];
                     auxFace.ki1[0] = gv1[i+1];
                     auxFace.ki1[1] = gv2[j+1];

                     auxFaces.push_back(auxFace);

                     // Orientation is 0 for a new auxiliary face, by construction.
                     facePairs.emplace_back(FacePairInfo{cv[0], -1 - auxv2f[childPair], parentFace, 0,
                        {gv1[i], gv2[j]}, {gv1[i+1] - gv1[i], gv2[j+1] - gv2[j]}});

                     // TODO: is it redundant to put parentFace in both
                     // facePairs and auxFaces?
                     // TODO: is it redundant to put gv1[i+{0,1}], gv2[j+{0,1}] in both
                     // facePairs and auxFaces?
                     // TODO: can facePairs and auxFaces be combined into
                     // one array of a struct?
                  }
               }
            }
      }

      // Set auxiliary edges outside the gridVertex, on the boundary of the parent face.
      // Note that auxiliary edges cannot simply be found as edges of auxiliary faces, because
      // the faces above are defined only on parent faces listed in V2K data. Other auxiliary
      // faces will be defined in FindAdditionalSlaveAndAuxiliaryFaces by using auxiliary edges found below.

      // Auxiliary edges in first and second directions
      for (int d=0; d<2; ++d)
      {
         const int de = d0;  // TODO: set this correctly for the direction of the edge.

         // TODO: interchange i- and s- loops?
         for (int i=0; i<3; ++i)
         {
            if (i == 1)  // Skip the middle, which is covered by gridVertex.
            {
               continue;
            }

            for (int s=0; s<2; ++s)  // Loop over 2 sides in this direction
            {
               // Set gv1 (d == 0) or gv2 (d == 1) for this edge.
               // Set range of set knot indices for this edge.
               int rmin = -1;
               int rmax = -1;
               const int n_d = d == 0 ? n1 : n2;
               for (int j=1; j<n_d; ++j)
               {
                  const int tv = d == 0 ? gridVertex(j,s*n2) : gridVertex((1-s)*n1,j);
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

               if (rmax == -1)  // No vertices set in gridVertex on the interior of this edge.
               {
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
               Array<int> ki(2);

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
               const int tvki_f = i == 0 ? ki[1] : ki[0];  // Knot index w.r.t. face
               const int tvki = reverse ? n_d - tvki_f : tvki_f;  // Knot index w.r.t. edge

               cv.Sort();

               MFEM_ASSERT(cv[0] >= 0, "");

               const std::pair<int, int> childPair(cv[0], cv[1]);
               const bool childPairTopo = v2e.count(childPair) > 0;
               if (!childPairTopo)
               {
                  const int pv0 = d == 0 ? gridVertex(0,s*n2) : gridVertex((1-s)*n1,0);
                  const int pv1 = d == 0 ? gridVertex(n1,s*n2) : gridVertex((1-s)*n1,n2);

                  const std::pair<int, int> parentPair(pv0 < pv1 ? pv0 : pv1,
                                                       pv0 < pv1 ? pv1 : pv0);

                  const int parentEdge = v2e.at(parentPair);

                  MFEM_VERIFY(parentEdges[pid] == parentEdge, "");

                  // Check whether childPair is in auxEdges.
                  if (auxv2e.count(childPair) == 0)
                  {
                     const int knotIndex0 = (d == 0) ? gv1[i] : gv2[i];
                     const int knotIndex1 = (d == 0) ? gv1[i+1] : gv2[i+1];

                     // Create a new auxiliary edge
                     auxv2e[childPair] = auxEdges.size();
                     auxEdges.emplace_back(AuxiliaryEdge{pv0 < pv1 ? parentEdge : -1 - parentEdge,
                        {childPair.first, childPair.second},
                        {knotIndex0, knotIndex1}});
                  }

                  // TODO: ne is identical to n_d.
                  const int ne = d == 0 ? n1 : n2;
                  const int e_idx_i = i == 0 ? 0 : ne - de;
                  const int e_idx = reverse ? ne - de - e_idx_i : e_idx_i;

                  const EdgePairInfo ep_e((e_idx == ne - de) ? -1 : tv,
                                          (e_idx == ne - de) ? -1 : tvki,
                                          -1 - auxv2e[childPair], parentEdge);

                  const bool unset = !edgePairs[edgePairOS[parentEdge] + e_idx].isSet;
                  if (unset)
                  {
                     edgePairs[edgePairOS[parentEdge] + e_idx] = ep_e;
                  }
                  else
                  {
                     // Verify matching
                     MFEM_VERIFY(edgePairs[edgePairOS[parentEdge] + e_idx] == ep_e, "");
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

                  MFEM_VERIFY(parentEdges[pid] == parentEdge, "");

                  // TODO: ne is identical to n_d.
                  const int ne = d == 0 ? n1 : n2;

                  const int e_idx_i = i == 0 ? 0 : ne - de;
                  const int e_idx = reverse ? ne - de - e_idx_i : e_idx_i;

                  const int tv_e = (e_idx == ne - de) ? -1 : tv;
                  const int tv_ki = (e_idx == ne - de) ? -1 : tvki;  // TODO: better naming

                  const EdgePairInfo ep_e(tv_e, tv_ki, childEdge, parentEdge);

                  const bool unset = !edgePairs[edgePairOS[parentEdge] + e_idx].isSet;
                  const bool matching = edgePairs[edgePairOS[parentEdge] + e_idx] == ep_e;
                  MFEM_VERIFY(unset || matching, "");

                  edgePairs[edgePairOS[parentEdge] + e_idx] = ep_e;
               }
            }
         }
      }

      if (hasSlaveFaces || hasAuxFace)
      {
         masterFaces.insert(parentFace);
      }
   } // loop over parents

   MFEM_VERIFY(consistent, "");
}

int GetFaceOrientation(const Mesh *mesh, const int face,
                       const Array<int> & verts)
{
   Array<int> fverts;
   mesh->GetFaceVertices(face, fverts);

   MFEM_VERIFY(verts.Size() == 4 && fverts.Size() == 4, "");

   // Verify that verts and fvert have the same entries as sets, by deep-copying and sorting.
   {
      Array<int> s1(verts);
      Array<int> s2(fverts);

      s1.Sort(); s2.Sort();
      MFEM_VERIFY(s1 == s2, "");
   }

   // Find the shift of the first vertex.
   int s = -1;
   for (int i=0; i<4; ++i)
   {
      if (verts[i] == fverts[0]) { s = i; }
   }

   // Check whether ordering is reversed.
   const bool rev = verts[(s + 1) % 4] != fverts[1];

   if (rev) { s = -1 - s; }  // Reversed order is encoded by the sign.

   // Sanity check (TODO: remove this)
   for (int i=0; i<4; ++i)
   {
      const int j = s < 0 ? (-1 - s) - i : i + s;
      MFEM_VERIFY(verts[(j + 4) % 4] == fverts[i], "");
   }

   return s;
}

// The 2D array `a` is of size n1*n2, with index
// j + n2*i corresponding to (i,j) with the fast index j,
// for 0 <= i < n1 and 0 <= j < n2.
// We assume that j is the fast index in (i,j).
// The orientation is encoded by ori, defining a shift and relative
// direction, such that a quad face F1, on which the ordering of `a` is based,
// has vertex with index `shift` matching vertex 0 of the new quad face F2,
// on which the new ordering of `a` should be based.
// For more details, see GetFaceOrientation.
bool Reorder2D(int n1, int n2, int ori, std::vector<int> & s0)
{
   const bool noReorder = false;
   if (noReorder)
   {
      s0[0] = 0;
      s0[1] = 0;
      return false;
   }

   const int shift = ori < 0 ? -1 - ori : ori;

   // Shift is an F1 index in the counter-clockwise ordering of 4 quad vertices.
   // Now find the (i,j) indices of this index, with i,j in {0,1}.
   const int s0i = (shift == 0 || shift == 3) ? 0 : 1;
   const int s0j = (shift < 2) ? 0 : 1;

   s0[0] = s0i;
   s0[1] = s0j;

   // Determine whether the dimensions of F1 and F2 are reversed.
   // Do this by finding the (i,j) indices of s1, which is the next vertex on F1.
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
   MFEM_VERIFY(0 <= shift && shift < 4, "");
   // TODO: condense this somehow?

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

// TODO: reuse this in other places, to reduce code duplication.
void GetShiftedGridPoints2D(int m, int n, int i, int j, int signedShift,
                            int& sm, int& sn, int& si, int& sj)
{
   const bool rev = (signedShift < 0);
   const int shift = rev ? -1 - signedShift : signedShift;
   MFEM_VERIFY(0 <= shift && shift < 4, "");

   // (0,0) <= (i,j) < (m,n) are old indices, and old vertex [shift] maps
   // to new vertex 0 in counter-clockwise quad ordering.

   // TODO: condense this somehow?

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

}
