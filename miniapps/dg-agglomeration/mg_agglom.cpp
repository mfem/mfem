
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

void GetMeshSubsetBoundingBox(const mfem::Mesh &mesh, const std::vector<int> &E2l, int c, mfem::Vector &min_coords, mfem::Vector &max_coords) {
   int dim = mesh.SpaceDimension();
   min_coords.SetSize(dim);
   max_coords.SetSize(dim);

   // Initialize min/max with extreme values
   for (int i = 0; i < dim; ++i) {
       min_coords(i) = infinity();
       max_coords(i) = -infinity();
   }

   for (int i = 0; i < E2l.size(); ++i) {
       if (E2l[i] == c) {
           mfem::Array<int> vert_indices;
           mesh.GetElementVertices(i, vert_indices);
           for (int j = 0; j < vert_indices.Size(); ++j) {
               const double* coords = mesh.GetVertex(vert_indices[j]);
               for (int k = 0; k < dim; ++k) {
                   if (coords[k] < min_coords(k)) {min_coords(k) = coords[k];};
                   if (coords[k] > max_coords(k)) {max_coords(k) = coords[k];};
               }
           }
       }
   }
}

SparseMatrix *CreateNodalProlongation(
   const std::vector<std::vector<int>> &E, const std::vector<std::vector<int>> &E2, FiniteElementSpace &fes)
{
   Mesh &mesh = *fes.GetMesh();
   int ne = mesh.GetNE();
   int dim = mesh.Dimension();
   int d = (dim == 2) ? 4 : 8;
   FiniteElementSpace nodal_fes(&mesh, fes.FEColl(), dim);
   GridFunction nodes(&nodal_fes);
   mesh.GetNodes(nodes);
   int nnodes = nodes.Size()/dim;
   int nnodes_per_el = nnodes/ne; // Assumes same number of nodes per element on fine mesh
   const int n = E.size();
   int nc = d*E[n-2].size();
   int nr = nnodes_per_el*E[n-1].size();
   std::cout << "inclusion prolongation rows: " << nr << std::endl;
   std::cout << "inclusion prolongation columns: " << nc << std::endl;
   SparseMatrix *P = new SparseMatrix(nr, nc);
   for (int e = 0; e < E[n-1].size(); ++e)
   {
      int c = E[n-1][e];
      Vector bb_min_coarse; Vector bb_max_coarse;
      GetMeshSubsetBoundingBox(mesh, E2[n-1], c, bb_min_coarse, bb_max_coarse);
      if (dim == 2)
      {
         Mesh ref_mesh("../../data/ref-square.mesh");
         L2_FECollection ref_fec(1, 2);
         FiniteElementSpace ref_fes(&ref_mesh, &ref_fec); 
         const FiniteElement* rfe = ref_fes.GetFE(0); 
         for (int i = 0; i < nnodes_per_el; ++i)
         {
            double x_phys = nodes(e*nnodes_per_el + i); double y_phys = nodes(nnodes + e*nnodes_per_el + i);
            double x_ref = (x_phys - bb_min_coarse(0)) / (bb_max_coarse(0) - bb_min_coarse(0));
            double y_ref = (y_phys - bb_min_coarse(1)) / (bb_max_coarse(1) - bb_min_coarse(1));
            IntegrationPoint ip;
            ip.Set2w(x_ref, y_ref, 1);
            Vector shape_vec(d);
            rfe->CalcShape(ip, shape_vec);
            for (int k = 0; k < d; k++)
            {
               P -> Set(nnodes_per_el*e + i, d*c + k, shape_vec(k));
            }
         }
      }
      else
      {
         Mesh ref_mesh("../../data/ref-cube.mesh");
         L2_FECollection ref_fec(1, 3);
         FiniteElementSpace ref_fes(&ref_mesh, &ref_fec); 
         const FiniteElement* rfe = ref_fes.GetFE(0); 
         for (int i = 0; i < nnodes_per_el; ++i)
         {
            double x_phys = nodes(e*nnodes_per_el + i); double y_phys = nodes(nnodes + e*nnodes_per_el + i);
            double z_phys = nodes(2*nnodes + e*nnodes_per_el + i);
            double x_ref = (x_phys - bb_min_coarse(0)) / (bb_max_coarse(0) - bb_min_coarse(0));
            double y_ref = (y_phys - bb_min_coarse(1)) / (bb_max_coarse(1) - bb_min_coarse(1));
            double z_ref = (z_phys - bb_min_coarse(2)) / (bb_max_coarse(2) - bb_min_coarse(2));
            IntegrationPoint ip;
            ip.Set(x_ref, y_ref, z_ref, 1);
            Vector shape_vec(d);
            rfe->CalcShape(ip, shape_vec);
            for (int k = 0; k < d; k++)
            {
               P -> Set(nnodes_per_el*e + i, d*c + k, shape_vec(k));
            }
         }
      }
   }
   P -> Finalize();
   return P;
}

SparseMatrix *CreateInclusionProlongation(
   int l, const std::vector<std::vector<int>> &E, const std::vector<std::vector<int>> &E2, FiniteElementSpace &fes)
{
   Mesh &mesh = *fes.GetMesh();
   int dim = mesh.Dimension();
   FiniteElementSpace nodal_fes(&mesh, fes.FEColl(), dim);
   int d = (dim == 2) ? 4 : 8;
   GridFunction nodes(&nodal_fes);
   if (l == 0)
   {
      int nc = d;
      int nr = d*E[l].size();
      std::cout << "inclusion prolongation rows: " << nr << std::endl;
      std::cout << "inclusion prolongation columns: " << nc << std::endl;
      SparseMatrix *P = new SparseMatrix(nr, nc);
      Vector bb_min_coarse; Vector bb_max_coarse;
      mesh.GetBoundingBox(bb_min_coarse, bb_max_coarse, 1);
      for (int e = 0; e < E[l].size(); ++e)
      {
         Vector bb_min_fine; Vector bb_max_fine;
         GetMeshSubsetBoundingBox(mesh, E2[l+1], e, bb_min_fine, bb_max_fine);
         Vector ref_v_0(3);
         ref_v_0(0) = (bb_min_fine(0) - bb_min_coarse(0))/(bb_max_coarse(0) - bb_min_coarse(0));
         ref_v_0(1) = (bb_min_fine(1) - bb_min_coarse(1))/(bb_max_coarse(1) - bb_min_coarse(1));
         Vector ref_v_1(3); 
         ref_v_1 (0) = (bb_max_fine(0) - bb_min_coarse(0))/(bb_max_coarse(0) - bb_min_coarse(0));
         ref_v_1 (1) = (bb_max_fine(1) - bb_min_coarse(1))/(bb_max_coarse(1) - bb_min_coarse(1));
         if (dim == 2)
         {
            Mesh ref_mesh("../../data/ref-square.mesh");
            L2_FECollection ref_fec(1, 2);
            FiniteElementSpace ref_fes(&ref_mesh, &ref_fec); 
            const FiniteElement* rfe = ref_fes.GetFE(0); 
            double placeholder = ref_v_0(1);
            ref_v_0(1) = ref_v_1(0);
            ref_v_1(0) = placeholder;
            for (int i = 0; i <= 1; i++)
            {
               for (int j = 0; j <= 1; j++)
               {
                  IntegrationPoint ip;
                  int P_idx = 2*i + j;
                  const real_t x = ref_v_0(i);
                  const real_t y = ref_v_1(j);
                  ip.Set2w(x, y, 1);
                  Vector shape_vec(4);
                  rfe->CalcShape(ip, shape_vec);
                  for (int k = 0; k < d; k++)
                  {
                     P -> Set(d*e + P_idx, k, shape_vec(k));
                  }
               }
            }
         }
         else
         {
            Mesh ref_mesh("../../data/ref-cube.mesh");
            L2_FECollection ref_fec(1, 3);
            FiniteElementSpace ref_fes(&ref_mesh, &ref_fec); 
            const FiniteElement* rfe = ref_fes.GetFE(0); 
            ref_v_0(2) = (bb_min_fine(2) - bb_min_coarse(2))/(bb_max_coarse(2) - bb_min_coarse(2));
            ref_v_1(2) = (bb_max_fine(2) - bb_min_coarse(2))/(bb_max_coarse(2) - bb_min_coarse(2));
            Vector tr_v_0(2); Vector tr_v_1(2); Vector tr_v_2(2);
            tr_v_0(0) = ref_v_0(0); tr_v_0(1) = ref_v_1(0);
            tr_v_1(0) = ref_v_0(1); tr_v_1(1) = ref_v_1(1);
            tr_v_2(0) = ref_v_0(2); tr_v_2(1) = ref_v_1(2);
            for (int i = 0; i<= 1; i++)
            {
               for (int j = 0; j <= 1; j++)
               {
                  for (int k = 0; k<=1; k++)
                  {
                     IntegrationPoint ip;
                     int P_idx = 4*i + 2*j + k;
                     const real_t x = tr_v_0(i);
                     const real_t y = tr_v_1(j);
                     const real_t z = tr_v_2(k);
                     ip.Set(x, y, z, 1);
                     Vector shape_vec(d);
                     rfe->CalcShape(ip, shape_vec);
                     for (int k = 0; k < d; k++)
                     {
                        P -> Set(d*e + P_idx, k, shape_vec(k));
                     }
                  }
               }
            }
         }
      }
      P->Finalize();
      return P;
   }
   else
   {
      int nc = d*E[l-1].size();
      int nr = d*E[l].size();
      std::cout << "inclusion prolongation rows: " << nr << std::endl;
      std::cout << "inclusion prolongation columns: " << nc << std::endl;
      std::cout << "l = " << l << std::endl;
      SparseMatrix *P = new SparseMatrix(nr, nc);
      for (int e = 0; e < E[l].size(); ++e)
      {
         int c = E[l][e];
         Vector bb_min_coarse; Vector bb_max_coarse;
         GetMeshSubsetBoundingBox(mesh, E2[l], c, bb_min_coarse, bb_max_coarse);
         Vector bb_min_fine; Vector bb_max_fine;
         GetMeshSubsetBoundingBox(mesh, E2[l+1], e, bb_min_fine, bb_max_fine);
         Vector ref_v_0(3);
         ref_v_0(0) = (bb_min_fine(0) - bb_min_coarse(0))/(bb_max_coarse(0) - bb_min_coarse(0));
         ref_v_0(1) = (bb_min_fine(1) - bb_min_coarse(1))/(bb_max_coarse(1) - bb_min_coarse(1));
         Vector ref_v_1(3); 
         ref_v_1 (0) = (bb_max_fine(0) - bb_min_coarse(0))/(bb_max_coarse(0) - bb_min_coarse(0));
         ref_v_1 (1) = (bb_max_fine(1) - bb_min_coarse(1))/(bb_max_coarse(1) - bb_min_coarse(1));
         if (dim == 2)
         {
            Mesh ref_mesh("../../data/ref-square.mesh");
            L2_FECollection ref_fec(1, 2);
            FiniteElementSpace ref_fes(&ref_mesh, &ref_fec); 
            const FiniteElement* rfe = ref_fes.GetFE(0); 
            double placeholder = ref_v_0(1);
            ref_v_0(1) = ref_v_1(0);
            ref_v_1(0) = placeholder;
            for (int i = 0; i <= 1; i++)
            {
               for (int j = 0; j <= 1; j++)
               {
                  IntegrationPoint ip;
                  int P_idx = 2*i + j;
                  const real_t x = ref_v_0(i);
                  const real_t y = ref_v_1(j);
                  ip.Set2w(x, y, 1);
                  Vector shape_vec(d);
                  rfe->CalcShape(ip, shape_vec);
                  for (int k = 0; k < d; k++)
                  {
                     P -> Set(d*e + P_idx, d*c + k, shape_vec(k));
                  }
               }
            }
         }
         else
         {
            Mesh ref_mesh("../../data/ref-cube.mesh");
            L2_FECollection ref_fec(1, 3);
            FiniteElementSpace ref_fes(&ref_mesh, &ref_fec); 
            const FiniteElement* rfe = ref_fes.GetFE(0); 
            ref_v_0(2) = (bb_min_fine(2) - bb_min_coarse(2))/(bb_max_coarse(2) - bb_min_coarse(2));
            ref_v_1(2) = (bb_max_fine(2) - bb_min_coarse(2))/(bb_max_coarse(2) - bb_min_coarse(2));
            Vector tr_v_0(2); Vector tr_v_1(2); Vector tr_v_2(2);
            tr_v_0(0) = ref_v_0(0); tr_v_0(1) = ref_v_1(0);
            tr_v_1(0) = ref_v_0(1); tr_v_1(1) = ref_v_1(1);
            tr_v_2(0) = ref_v_0(2); tr_v_2(1) = ref_v_1(2);
            for (int i = 0; i<= 1; i++)
            {
               for (int j = 0; j <= 1; j++)
               {
                  for (int k = 0; k<=1; k++)
                  {
                     IntegrationPoint ip;
                     int P_idx = 4*i + 2*j + k;
                     const real_t x = tr_v_0(i);
                     const real_t y = tr_v_1(j);
                     const real_t z = tr_v_2(k);
                     ip.Set(x, y, z, 1);
                     Vector shape_vec(d);
                     rfe->CalcShape(ip, shape_vec);
                     for (int k = 0; k < d; k++)
                     {
                        P -> Set(d*e + P_idx, d*c + k, shape_vec(k));
                     }
                  }
               }
            }
         }
      }
      P->Finalize();
      return P;
   }
}

AgglomerationMultigrid::AgglomerationMultigrid(
   FiniteElementSpace &fes, SparseMatrix &Af, int ncoarse, int num_levels, int smoother_choice, bool paraview_vis)
{
   MFEM_VERIFY(fes.GetMaxElementOrder() == 1, "Only linear elements supported.");
   // Create the mesh hierarchy
   auto E = Agglomerate(*fes.GetMesh(), ncoarse);
   int num_levels_s = E.size() + 1;
   Mesh &mesh = *fes.GetMesh();
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
   vector<int> E2_finest_level(ne);
   for (int i = 0; i < ne; ++i) 
   {
      E2_finest_level[i] = i; 
   }
   E2.push_back(E2_finest_level);
   int dim = mesh.Dimension();
   if (paraview_vis)
   {
      L2_FECollection l2_fec(0, mesh.Dimension());
      FiniteElementSpace l2_fes(&mesh, &l2_fec);
      GridFunction p_gf(&l2_fes);
      ParaViewDataCollection pv("Agglomeration", &mesh);
      pv.SetPrefixPath("ParaView");
      pv.RegisterField("p", &p_gf);
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
         P = CreateInclusionProlongation(k, E, E2, fes);
      }
      else
      {
         P = CreateNodalProlongation(E, E2, fes);
         //P = CreateInclusionProlongation(k, E, E2, fes);
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
   // mfem::out << "Sparse Matrix A:\n";
   // Ac.Print(mfem::out);
   std::cout << "test"  << std::endl;
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



