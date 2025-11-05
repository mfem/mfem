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
   const int ncoarse = 2;

   const int num_partitions = std::ceil(std::log(ne)/std::log(ncoarse));

   std::cout << "number of fine elements: " << ne << std::endl;
   std::cout << "number of elements per macro elements: " << ncoarse << std::endl;
   std::cout << "number of times we partition: " << num_partitions << std::endl;

   // E is a data structure such that E_ij = index of the parent element to element j on level i.
   // i goes from coarsest to finest, so i = 0 corresponds to the coarsest level.
   // For example, E_0j = 0, since for any element j on the second-coarsest level, the parent element is 0.

   std::vector<std::vector<int>> E(num_partitions);

   // Recursive METIS partitioning to create 'E' data

   DG_FECollection fec(0, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   // Partition the coarsest mesh.
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

   // macro_elements is a data structure which, for each macro element idx i, gives the indices of all the fine mesh elements which belong to it
   std::vector<std::vector<int>> macro_elements(ncoarse);
   for (int k = 0; k < ne; ++k)
   {
      const int i = partitioning[k];
      macro_elements[i].push_back(k);
   }

   // Iterate through each level, and populate E. 
   int j = 1;
   while (E[j-1].size() != ne)
   {
      // If j is >= num_partitions, but we still have not fully refined the mesh, resize E
      if(j >= num_partitions){E.resize(E.size()+1);}

      // populate macro-elements
      std::vector<std::vector<int>> macro_elements(E[j-1].size());
      for (int i = 0; i < p.Size(); ++i)
      {
         const int k = p[i];
         macro_elements[k].push_back(i);
      }

      // for each macro_element, partition it
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
      j = j+1;
   }

   return E;
}

SparseMatrix *CreateNodalProlongation(
   const std::vector<std::vector<int>> &E, FiniteElementSpace &fes)
{
   const int n = E.size();
   int nr = 4*E[n-1].size();
   int nc = (n == 1) ? 4:4*E[n-2].size();
   //int nc = 4*E[n-2].size();
   SparseMatrix *P = new SparseMatrix(nr, nc);

   std::cout << "nodal prolongation rows: " << nr << std::endl;
   std::cout << "nodal prolongation columns: " << nc << std::endl;

   Mesh &mesh = *fes.GetMesh();
   FiniteElementSpace nodal_fes(&mesh, fes.FEColl(), 2);
   GridFunction nodes(&nodal_fes);
   mesh.GetNodes(nodes);
   int nnodes = nodes.Size()/mesh.Dimension();
   for (int r = 0; r < E[n-1].size(); ++r)
   {
      int c = E[n-1][r];
      double x0 = nodes(r*4); double x1 = nodes(r*4+1); double x2 = nodes(r*4+2);
      double x3 = nodes(r*4+3);
      double y0 = nodes(nnodes+ r*4); double y1 = nodes(nnodes + (r*4+1));
      double y2 = nodes(nnodes + r*4+2); double y3 = nodes(nnodes+r*4+3);
      //column 1 of block
      P->Set(4*r, 4*c, 1); P->Set(4*r+1, 4*c, 1);
      P->Set(4*r+2, 4*c, 1); P->Set(4*r+3, 4*c, 1);
      //column 2 of block
      P->Set(4*r, 4*c+1, x0); P->Set(4*r+1, 4*c+1, x1);
      P->Set(4*r+2, 4*c+1, x2); P->Set(4*r+3, 4*c+1, x3);
      //column 3 of block
      P->Set(4*r, 4*c+2, y0); P->Set(4*r+1, 4*c+2, y1);
      P->Set(4*r+2, 4*c+2, y2); P->Set(4*r+3, 4*c+2, y3);
      //column 3 of block
      P->Set(4*r, 4*c+3, x0*y0); P->Set(4*r+1, 4*c+3, x1*y1);
      P->Set(4*r+2, 4*c+3, x2*y2); P->Set(4*r+3, 4*c+3, x3*y3);
   }
   P->Finalize();
   return P;
}

SparseMatrix *CreateInclusionProlongation(
   int l, const std::vector<std::vector<int>> &E)
{
   int nc = (l != 0) ? 4*E[l-1].size() : 4;
   //int nc = 4*E[l].size();
   int nr = 4*E[l].size();
   std::cout << "inclusion prolongation rows: " << nr << std::endl;
   std::cout << "inclusion prolongation columns: " << nc << std::endl;
   SparseMatrix *P = new SparseMatrix(nr, nc);
   for (int e = 0; e < E[l].size(); ++e)
   {
      int c = E[l][e];
      for (int i = 0; i < 4; ++i)
      {
         P->Set(4*e + i, 4*c + i, 1);
      }
   }
   P->Finalize();
   return P;
}

AgglomerationMultigrid::AgglomerationMultigrid(
   FiniteElementSpace &fes, SparseMatrix &Af)
{
   MFEM_VERIFY(fes.GetMaxElementOrder() == 1, "Only linear elements supported.");
   // Create the mesh hierarchy
   auto E = Agglomerate(*fes.GetMesh());
   int num_levels_s = E.size() + 1;
   int num_levels = 2;

   std::cout << "num levels: " << num_levels << std::endl;

   // Populate the arrays: operators, smoothers, ownedOperators, ownedSmoothers
   // from the MultigridBase class. (All smoothers are owned, all operators
   // except the finest are owned).

   operators.SetSize(num_levels);
   smoothers.SetSize(num_levels);
   ownedOperators.SetSize(num_levels);
   ownedSmoothers.SetSize(num_levels);
   prolongations.SetSize(num_levels-1);
   ownedProlongations.SetSize(num_levels-1);

   //Used for making the smoother
   real_t alpha = 1.0;
   Array<Operator*> sm_operators;
   sm_operators.SetSize(num_levels);

   //Set the ownership
   for (int l = 0; l < num_levels-1; ++l)
   {
      ownedOperators[l] = true;
      ownedSmoothers[l] = true;
      ownedProlongations[l] = true;
   }
   ownedOperators[num_levels-1] = false;
   ownedSmoothers[num_levels-1] = true;

   operators[num_levels - 1] = &Af;

   // Populate the arrays: prolongations, ownedProlongations from the Multigrid
   // class. All prolongations are owned.
   // Create the prolongations using 'E' using the SparseMatrix class
   int k = num_levels_s-2;
   for (int l = num_levels - 2; l >= 0; --l)
   {
      SparseMatrix *P;
      if (l < num_levels - 2)
      {
         P = CreateInclusionProlongation(k, E);
      }
      else
      {
         P = CreateNodalProlongation(E, fes);
      }

      SparseMatrix &A_prev = static_cast<SparseMatrix&>(*operators[l + 1]);

      unique_ptr<SparseMatrix> AP(mfem::Mult(A_prev, *P));
      unique_ptr<SparseMatrix> Pt(Transpose(*P));
      operators[l] = mfem::Mult(*Pt, *AP);
      //sm_operators[l] = &(*(mfem::Mult(*Pt, *AP))*=alpha);
      prolongations[l] = P;
      k = k-1;
   }

   // Create the smoothers (using BlockILU for now) remember block size is num degrees of freedom per element
   
   SparseMatrix &Ac = static_cast<SparseMatrix&>(*operators[0]);
   smoothers[0] = new UMFPackSolver(Ac);
   for (int l=1; l < num_levels; ++l)
   {
      smoothers[l] = new BlockILU(*operators[l], 4);

   }
}


} // namespace mfem
