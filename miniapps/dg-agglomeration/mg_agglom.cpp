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

// implement with gauss siedel
// or implement with block jacobi with l1 scaling 



#include "mg_agglom.hpp"
#include "partition.hpp"

using namespace std;

namespace mfem
{

std::vector<std::vector<int>> Agglomerate(Mesh &mesh, int ncoarse)
{
   const int ne = mesh.GetNE();

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

   // Iterate through each level, and populate E. 
   int j = 1;
   while (E[j-1].size() != ne)
   {
      // If j is >= num_partitions, but we still have not fully refined the mesh, resize E
      if(j >= num_partitions){E.resize(E.size()+1);}

      // macro_elements is a data structure which, for each macro element idx i, gives the indices of all the fine mesh elements which belong to it
      // populate macro-elements (this is done twice on first run)
      std::vector<std::vector<int>> macro_elements(E[j-1].size());
      for (int i = 0; i < p.Size(); ++i)
      {
         const int k = p[i];
         macro_elements[k].push_back(i);
      }

      // for each macro_element, partition it
      int total_new_macros = 0;
      for (int e = 0; e < E[j-1].size(); ++e)
      {
         const int num_fine_in_macro = macro_elements[e].size(); 
         if (num_fine_in_macro == 1)
         {
            p[macro_elements[e][0]] = total_new_macros;
            total_new_macros += 1;
            E[j].push_back(e);
         }
         else
         {
            Array<int> subset(num_fine_in_macro);
            for (int i=0; i<num_fine_in_macro; i++) {subset[i] = macro_elements[e][i];}
            Array<int> partitioning = PartitionMesh(mesh, ncoarse, subset);
            int new_macros_in_subset = 0;
            for (int ip = 0; ip < partitioning.Size(); ++ip)
            {
               const int i = partitioning[ip];
               new_macros_in_subset = (i > new_macros_in_subset) ? i : new_macros_in_subset;
               p[subset[ip]] = i + total_new_macros;
            }
            for (int k = 0; k <= new_macros_in_subset; ++k) {E[j].push_back(e);}
            total_new_macros += new_macros_in_subset + 1;
         }
      }
      j = j+1;
   }

   return E;
}

SparseMatrix *CreateNodalProlongation(
   const std::vector<std::vector<int>> &E, FiniteElementSpace &fes)
{
   Mesh &mesh = *fes.GetMesh();
   int dim = mesh.Dimension();
   FiniteElementSpace nodal_fes(&mesh, fes.FEColl(), dim);
   GridFunction nodes(&nodal_fes);
   mesh.GetNodes(nodes);
   mfem::Geometry::Type geo_type = mesh.GetElementBaseGeometry(0); // Assumes that all elements are of the same geom type
   int nnodes = nodes.Size()/dim;
   const int n = E.size();
   if (geo_type == mfem::Geometry::SQUARE) 
   {
      int d =  4;
      int nr = d*E[n-1].size();
      int nc = (n == 1) ? d:d*E[n-2].size();
      SparseMatrix *P = new SparseMatrix(nr, nc);

      std::cout << "nodal prolongation rows: " << nr << std::endl;
      std::cout << "nodal prolongation columns: " << nc << std::endl;

      for (int r = 0; r < E[n-1].size(); ++r)
      {
         int c = E[n-1][r];

         for (int i = 0; i<d; ++i)
         {
            double x = nodes(r*d+i); double y = nodes(nnodes + r*d + i);
            P->Set(d*r+i, d*c, 1);
            P->Set(d*r+i, d*c+1, x);
            P->Set(d*r+i, d*c+2, y);
            P->Set(d*r+i, d*c+3, x*y);
         }
      }
      P->Finalize();
      return P;
   }
   else if (geo_type == mfem::Geometry::CUBE)
   {
      int d =  8;
      int nr = d*E[n-1].size();
      int nc = (n == 1) ? d:d*E[n-2].size();
      SparseMatrix *P = new SparseMatrix(nr, nc);

      std::cout << "nodal prolongation rows: " << nr << std::endl;
      std::cout << "nodal prolongation columns: " << nc << std::endl;

      for (int r = 0; r < E[n-1].size(); ++r)
      {
         int c = E[n-1][r];

         for (int i = 0; i<d; ++i)
         {
            double x = nodes(r*d+i); double y = nodes(nnodes + r*d + i); double z = nodes(2*nnodes + r*d + i);
            P->Set(d*r+i, d*c, 1);
            P->Set(d*r+i, d*c+1, x);
            P->Set(d*r+i, d*c+2, y);
            P->Set(d*r+i, d*c+3, z);
            P->Set(d*r+i, d*c+4, x*y);
            P->Set(d*r+i, d*c+5, x*z);
            P->Set(d*r+i, d*c+6, y*z);
            P->Set(d*r+i, d*c+7, x*y*z);
         }
      }
      P->Finalize();
      return P;
   }
   else if (geo_type == mfem::Geometry::TRIANGLE)
   {
      int d = 3;
      int nr = d*E[n-1].size();
      int nc = (n == 1) ? d:d*E[n-2].size();
      SparseMatrix *P = new SparseMatrix(nr, nc);

      std::cout << "nodal prolongation rows: " << nr << std::endl;
      std::cout << "nodal prolongation columns: " << nc << std::endl;

      for (int r = 0; r < E[n-1].size(); ++r)
      {
         int c = E[n-1][r];

         for (int i = 0; i<d; ++i)
         {
            double x = nodes(r*d+i); double y = nodes(nnodes + r*d + i);
            P->Set(d*r+i, d*c, x);
            P->Set(d*r+i, d*c+1, y);
            P->Set(d*r+i, d*c+2, 1-x-y);
         }
      }
      P->Finalize();
      return P;
   }
   else if (geo_type == mfem::Geometry::TETRAHEDRON)
   {
      int d = 4;
      int nr = d*E[n-1].size();
      int nc = (n == 1) ? d:d*E[n-2].size();
      SparseMatrix *P = new SparseMatrix(nr, nc);

      std::cout << "nodal prolongation rows: " << nr << std::endl;
      std::cout << "nodal prolongation columns: " << nc << std::endl;

      for (int r = 0; r < E[n-1].size(); ++r)
      {
         int c = E[n-1][r];

         for (int i = 0; i<d; ++i)
         {
            double x = nodes(r*d+i); double y = nodes(nnodes + r*d + i); double z = nodes(2*nnodes + r*d + i);
            P->Set(d*r+i, d*c, x);
            P->Set(d*r+i, d*c+1, y);
            P->Set(d*r+i, d*c+2, z);
            P->Set(d*r+i, d*c+3, 1-x-y-z);
         }
      }
      P->Finalize();
      return P;
   }
   else
   {
      MFEM_ABORT("Unsupported element type for agglomeration.");
   }
}

SparseMatrix *CreateInclusionProlongation(
   int l, const std::vector<std::vector<int>> &E, int d)
{
   int nc = (l != 0) ? d*E[l-1].size() : d;
   int nr = d*E[l].size();
   std::cout << "inclusion prolongation rows: " << nr << std::endl;
   std::cout << "inclusion prolongation columns: " << nc << std::endl;
   SparseMatrix *P = new SparseMatrix(nr, nc);
   for (int e = 0; e < E[l].size(); ++e)
   {
      int c = E[l][e];
      for (int i = 0; i < d; ++i)
      {
         P->Set(d*e + i, d*c + i, 1);
      }
   }
   P->Finalize();
   return P;
}

AgglomerationMultigrid::AgglomerationMultigrid(
   FiniteElementSpace &fes, SparseMatrix &Af, int ncoarse, int num_levels, int smoother_choice, bool paraview_vis)
{
   MFEM_VERIFY(fes.GetMaxElementOrder() == 1, "Only linear elements supported.");
   // Create the mesh hierarchy
   auto E = Agglomerate(*fes.GetMesh(), ncoarse);
   int num_levels_s = E.size() + 1;
   Mesh &mesh = *fes.GetMesh();
   mfem::Geometry::Type geo_type = mesh.GetElementBaseGeometry(0);
   int d = 4;
   if (geo_type == mfem::Geometry::SQUARE || geo_type == mfem::Geometry::TETRAHEDRON)
   {
      d = 4;
   }
   else if (geo_type == mfem::Geometry::CUBE) 
   {
      d = 8;
   }
   else if (geo_type == mfem::Geometry::TRIANGLE)
   {
      d = 3;
   }
   else
   {
      MFEM_ABORT("Unsupported element type for agglomeration.");
   }
   int dim = mesh.Dimension();
   if (paraview_vis)
   {
      L2_FECollection l2_fec(0, mesh.Dimension());
      FiniteElementSpace l2_fes(&mesh, &l2_fec);
      GridFunction p_gf(&l2_fes);
      ParaViewDataCollection pv("Agglomeration", &mesh);
      pv.SetPrefixPath("ParaView");
      pv.RegisterField("p", &p_gf);
      const int ne = mesh.GetNE();
      vector<vector<int>> E2(E.size());
      E2.back() = E.back();
      for (int i = E.size() - 2; i >= 0; --i)
      {
         E2[i].resize(ne);
         for (int e = 0; e < ne; ++e)
         {
            const int m_e = E2[i+1][e];
            E2[i][e] = E[i][m_e];
         }
      }
      for (int i = 0; i < E.size(); ++i)
      {
         for (int e = 0; e < ne; ++e)
         {
            p_gf[e] = E2[i][e];
         }
         pv.SetCycle(i);
         pv.SetTime(i);
         pv.Save();
      }
      for (int e = 0; e < ne; ++e)
      {
         p_gf[e] = e;
         pv.SetCycle(num_levels_s - 1);
         pv.SetTime(num_levels_s - 1);
         pv.Save();
      }
   }

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
         P = CreateInclusionProlongation(k, E, d);
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

   // Create the smoothers remember block size is num degrees of freedom per element
   
   SparseMatrix &Ac = static_cast<SparseMatrix&>(*operators[0]);
   smoothers[0] = new UMFPackSolver(Ac);
   for (int l=1; l < num_levels; l++)
   {
      if (smoother_choice == 0)
      {
         smoothers[l] = new BlockGS(*operators[l], 4);
      }
      else if (smoother_choice == 1)
      {
         smoothers[l] = new Blockl1Jacobi(*operators[l], 4);
      }
      else if (smoother_choice == 2)
      {
         smoothers[l] = new BlockILU(*operators[l], 4);
      }
      else
      {
         MFEM_ABORT("Unknown Smoother.")
      }
   }
}


} // namespace mfem

