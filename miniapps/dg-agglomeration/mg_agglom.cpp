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

#include "mg_agglom.hpp"
#include "partition.hpp"

using namespace std;

namespace mfem
{

std::vector<std::vector<int>> Agglomerate(Mesh &mesh)
{
   const int ne = mesh.GetNE();
   const int ncoarse = 4;

   const int num_partitions = std::ceil(std::log(ne)/std::log(ncoarse));

   std::vector<std::vector<int>> E(num_partitions);

   // Recursive METIS partitioning to create 'E' data

   DG_FECollection fec(0, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction p(&fes);
   p = 0;
   Array<int> partitioning = PartitionMesh(mesh, ncoarse);
   for (int i = 0; i < p.Size(); ++i)
   {
      p[i] = partitioning[i];
   }
   for (int i = 0; i < ncoarse; ++i)
   {
      E[0].push_back(0);
   }
   std::vector<std::vector<int>> macro_elements(ncoarse);
   for (int k = 0; k < ne; ++k)
   {
      const int i = partitioning[k];
      macro_elements[i].push_back(k);
   }
   for (int j = 1; j < num_partitions; ++j)
   {
      std::vector<std::vector<int>> macro_elements(E[j-1].size());
      for (int i = 0; i < p.Size(); ++i)
      {
         const int k = p[i];
         macro_elements[k].push_back(i);
      }
      int num_total_parts = 0;
      for (int e = 0; e < E[j-1].size(); ++e)
      {
         const int num_el_part = macro_elements[e].size();
         Array<int> subset(num_el_part);
         for (int i=0; i<num_el_part; i++) {subset[i] = macro_elements[e][i];}
         Array<int> partitioning = PartitionMesh(mesh, ncoarse, subset);
         int num_actual_parts = 0;
         for (int ip = 0; ip < partitioning.Size(); ++ip)
         {
            const int i = partitioning[ip];
            num_actual_parts = (i > num_actual_parts) ? i : num_actual_parts;
            p[subset[ip]] = i + num_total_parts;
         }
         for (int k = 0; k <= num_actual_parts; ++k) {E[j].push_back(e);}
         num_total_parts = num_total_parts + num_actual_parts + 1;
      }
   }

   return E;
}

AgglomerationMultigrid::AgglomerationMultigrid(
   FiniteElementSpace &fes, SparseMatrix &Af)
{
   MFEM_VERIFY(fes.GetMaxElementOrder() == 1, "Only linear elements supported.");
   Mesh* mesh = fes.GetMesh();
   // Create the mesh hierarchy
   auto E = Agglomerate(*fes.GetMesh());
   int num_levels = E.size()+1;

   // Populate the arrays: operators, smoothers, ownedOperators, ownedSmoothers
   // from the MultigridBase class. (All smoothers are owned, all operators
   // except the finest are owned).
   operators.SetSize(num_levels);
   smoothers.SetSize(num_levels);
   ownedOperators.SetSize(num_levels);
   ownedSmoothers.SetSize(num_levels);
   prolongations.SetSize(num_levels-1);
   ownedProlongations.SetSize(num_levels-1);
   //Set the ownership
   for (int l = 0; l < num_levels-1; ++l)
   {
      ownedOperators[l] = true;
      ownedSmoothers[l] = true;
      ownedProlongations[l] = true;
   }
   ownedOperators[num_levels-1] = false;
   ownedSmoothers[num_levels-1] = true;

   // Populate the arrays: prolongations, ownedProlongations from the Multigrid
   // class. All prolongations are owned.
   // Create the prolongations using 'E' using the SparseMatrix class

   // Make the final prolongation
   operators[num_levels - 1] = &Af;
   SparseMatrix p_level = SparseMatrix(4*E[num_levels-2].size(),
                                       4*E[num_levels-3].size());
   GridFunction verts(&fes);
   mesh->GetNodes(verts);
   int nnodes = verts.Size()/2;
   for (int r = 0; r < E[num_levels-2].size(); ++r)
   {
      int c = E[num_levels-2][r];
      double x0 = verts(r*4); double x1 = verts(r*4+1); double x2 = verts(r*4+2);
      double x3 = verts(r*4+3);
      double y0 = verts(nnodes+ r*4); double y1 = verts(nnodes + (r*4+1));
      double y2 = verts(nnodes + r*4+2); double y3 = verts(nnodes+r*4+3);
      //column 1 of block
      p_level.Set(4*r, 4*c, 1); p_level.Set(4*r+1, 4*c, 1);
      p_level.Set(4*r+1, 4*c, 1); p_level.Set(4*r+1, 4*c, 1);
      //column 2 of block
      p_level.Set(4*r, 4*c+1, x0); p_level.Set(4*r+1, 4*c+1, x1);
      p_level.Set(4*r+1, 4*c+1, x2); p_level.Set(4*r+1, 4*c+1, x3);
      //column 3 of block
      p_level.Set(4*r, 4*c+2, y0); p_level.Set(4*r+1, 4*c+2, y1);
      p_level.Set(4*r+1, 4*c+2, y2); p_level.Set(4*r+1, 4*c+2, y3);
      //column 3 of block
      p_level.Set(4*r, 4*c+3, x0*y0); p_level.Set(4*r+1, 4*c+3, x1*y1);
      p_level.Set(4*r+1, 4*c+3, x2*y2); p_level.Set(4*r+1, 4*c+3, x3*y3);
   }
   SparseMatrix* PT = Transpose(p_level);
   RAPOperator Ac(*PT, *operators[num_levels-1], p_level);
   operators[num_levels-2] = &Ac;
   //SparseMatrix* p_level_p = &p_level;
   prolongations[num_levels-2] = &p_level;

   //Make the rest of the prolongations
   for (int l = num_levels-3; l > 0; --l)
   {
      SparseMatrix p_level = SparseMatrix(4*E[l].size(), 4*E[l-1].size());
      for (int r = 1; r < E[l].size(); ++r)
      {
         int c = E[l][r];
         p_level.Set(4*r, 4*c, 1);
         p_level.Set(4*r+1, 4*c+1, 1);
         p_level.Set(4*r+2, 4*c+2, 1);
         p_level.Set(4*r+3, 4*c+3, 1);
      }
      SparseMatrix* PT = Transpose(p_level);
      RAPOperator Ac(*PT, *operators[l+1], p_level);
      operators[l] = &Ac;
      //SparseMatrix* p_level_p = &p_level;
      prolongations[l] = &p_level;
   }

   //Make First Prolongation
   SparseMatrix p_levelc = SparseMatrix(4*E[0].size(), 4);
   for (int r = 0; r < 4; ++r)
   {
      p_levelc.Set(r, r, 1);
      p_levelc.Set(r+4, r, 1);
      p_levelc.Set(r+8, r, 1);
      p_levelc.Set(r+12, r, 1);
   }
   SparseMatrix* PTc = Transpose(p_levelc);
   RAPOperator Acc(*PTc, *operators[1], p_levelc);
   operators[0] = &Acc;
   //SparseMatrix* p_level_p = &p_level;
   prolongations[0] = &p_levelc;

   // Create the smoothers (using BlockILU for now) remember block size is num degrees of freedom per element
   for (int l=0; l < num_levels; ++l)
   {
      BlockILU smooth_l(*operators[l], 4);
      smoothers[l] = &smooth_l;
   }
}

} // namespace mfem
