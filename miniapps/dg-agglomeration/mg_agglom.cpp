
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

TruncatedMultigrid::TruncatedMultigrid(const AgglomerationMultigrid &other)
{
   MFEM_VERIFY(other.NumLevels() >= 2, "");
   const int nlevels = other.NumLevels() - 1;

   operators = other.operators;
   operators.DeleteLast();
   ownedOperators.SetSize(nlevels, false);

   smoothers = other.smoothers;
   smoothers.DeleteLast();
   ownedSmoothers.SetSize(nlevels, false);

   prolongations = other.prolongations;
   prolongations.DeleteLast();
   ownedProlongations.SetSize(nlevels - 1, false);
}

const SparseMatrix& AgglomerationMultigrid::GetFinestProlongation() const
{
   return static_cast<SparseMatrix&>(*prolongations.Last());

}

std::vector<std::vector<int>> Agglomerate(Mesh &mesh, int ncoarse, int num_levels)
{
   const int ne = mesh.GetNE();

   const int ne_coarsest = ne / pow(ncoarse, num_levels-1);

   // const int num_partitions = std::ceil(std::log(ne)/std::log(ncoarse));

   std::cout << "number of fine elements: " << ne << std::endl;
   std::cout << "number of elements per macro elements: " << ncoarse << std::endl;
   std::cout << "number of coarsest level elements: " << ne_coarsest << std::endl;

   // E is a data structure such that E_ij = index of the parent element to element j on level i.
   // i goes from coarsest to finest, so i = 0 corresponds to the coarsest level.
   // For example, E_0j = 0, since for any element j on the second-coarsest level, the parent element is 0.

   std::vector<std::vector<int>> E(num_levels+1);

   // Recursive METIS partitioning to create 'E' data

   DG_FECollection fec(0, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   // Partition the coarsest mesh.
   GridFunction p(&fes);
   p = 0;
   Array<int> partitioning = PartitionMesh(mesh, ne_coarsest);
   for (int i = 0; i < p.Size(); ++i)
   {
      p[i] = partitioning[i];
   }
   for (int i = 0; i < ne_coarsest; ++i)
   {
      E[0].push_back(0);
   }

   // Iterate through each level, and populate E. 
   int j = 1;
   int ej = E[j-1].size();
   while (ej != ne)
   {
      // If j is >= num_partitions, but we still have not fully refined the mesh, resize E
      if(j >= num_levels+1){E.resize(E.size()+1);}

      // macro_elements is a data structure which, for each macro element idx i, gives the indices of all the fine mesh elements which belong to it
      std::vector<std::vector<int>> macro_elements(E[j-1].size());
      for (int i = 0; i < p.Size(); ++i)
      {
         const int k = p[i];
         macro_elements[k].push_back(i);
      }

      // for each macro_element, partition it
      int total_new_macros = 0;
      for (int e = 0; e < ej; ++e)
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
      ej = E[j-1].size();
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
   int E2l_size = E2l.size();
   for (int i = 0; i < E2l_size; ++i) {
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
   int vdim = fes.GetVDim();
   int ne = mesh.GetNE();
   int dim = mesh.Dimension();
   FiniteElementSpace nodal_fes(&mesh, fes.FEColl(), dim);
   GridFunction nodes(&nodal_fes);
   mesh.GetNodes(nodes);
   int nnodes = nodes.Size()/dim*vdim;
   int nodes_per_dim = nodes.Size()/dim;
   const int n = E.size();
   const int p = fes.GetOrder(0);
   int d = (dim == 2) ? (p+1)*(p+2)/2 : (p+1)*(p+2)*(p+3)/6;
   int ncc = d*E[n-2].size();
   int nc = ncc*vdim;
   int nr = nnodes;
   std::cout << "nodal prolongation rows: " << nr << std::endl;
   std::cout << "nodal prolongation columns: " << nc << std::endl;
   SparseMatrix *P = new SparseMatrix(nr, nc);
   for (int e = 0; e < ne; ++e)
   {
      int c = E[n-1][e];
      Vector bb_min_coarse; Vector bb_max_coarse;
      GetMeshSubsetBoundingBox(mesh, E2[n-1], c, bb_min_coarse, bb_max_coarse);
      Array<int> local_element_dof_indices;
      fes.GetElementDofs(e, local_element_dof_indices);
      int num_el_dofs = local_element_dof_indices.Size();
      for (int i = 0; i < num_el_dofs; ++i)
      {
         int dof_idx = local_element_dof_indices[i];
         double x_phys = nodes(dof_idx); double y_phys = nodes(nodes_per_dim + dof_idx);
         double x_ref = (x_phys - bb_min_coarse(0)) / (bb_max_coarse(0) - bb_min_coarse(0));
         double y_ref = (y_phys - bb_min_coarse(1)) / (bb_max_coarse(1) - bb_min_coarse(1));
         IntegrationPoint ip;
         Vector shape_vec(d);
         if (dim == 3){
            L2_TetrahedronElement rfe(p);
            double z_phys = nodes(2*nodes_per_dim + dof_idx);
            double z_ref = (z_phys - bb_min_coarse(2)) / (bb_max_coarse(2) - bb_min_coarse(2));
            ip.Set(x_ref, y_ref, z_ref, 1);
            rfe.CalcShape(ip, shape_vec);
         }
         else
         {
            L2_TriangleElement rfe(p);
            ip.Set2w(x_ref, y_ref, 1);
            rfe.CalcShape(ip, shape_vec);
         }
         for (int k = 0; k < d; k++)
         {
            for(int vd=0; vd < vdim; vd++)
            {
               P -> Set(fes.DofToVDof(dof_idx, vd), (ncc*vd) + d*c + k, shape_vec(k));
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
   int vdim = fes.GetVDim();
   int dim = mesh.Dimension();
   FiniteElementSpace nodal_fes(&mesh, fes.FEColl(), dim);
   const int p = fes.GetOrder(0);
   int d = (dim == 2) ? (p+1)*(p+2)/2 : (p+1)*(p+2)*(p+3)/6;
   GridFunction nodes(&nodal_fes);
   int ncc = (l == 0) ? d: d*E[l-1].size();
   int nc = vdim*ncc;
   int nrr = d*E[l].size();
   int nr = vdim*nrr;
   std::cout << "inclusion prolongation rows: " << nr << std::endl;
   std::cout << "inclusion prolongation columns: " << nc << std::endl;
   SparseMatrix *P = new SparseMatrix(nr, nc);
   int el_size = E[l].size();
   for (int e = 0; e < el_size; ++e)
    {
      int c = E[l][e];
      Vector bb_min_coarse; Vector bb_max_coarse;
      GetMeshSubsetBoundingBox(mesh, E2[l], c, bb_min_coarse, bb_max_coarse);
      Vector bb_min_fine; Vector bb_max_fine;
      GetMeshSubsetBoundingBox(mesh, E2[l+1], e, bb_min_fine, bb_max_fine);
      if (dim == 2)
      {
         L2_TriangleElement rfe(p);
         const IntegrationRule rfe_nodes = rfe.GetNodes();
         for (int i = 0; i < rfe_nodes.Size(); i++)
         {
            IntegrationPoint ip = rfe_nodes.IntPoint(i);
            Vector small_bb_map(2);
            small_bb_map(0) = (bb_max_fine(0) - bb_min_fine(0))*ip.x + bb_min_fine(0);
            small_bb_map(1) = (bb_max_fine(1) - bb_min_fine(1))*ip.y + bb_min_fine(1);
            Vector ref_coord_big(2); 
            ref_coord_big(0) = (small_bb_map(0) - bb_min_coarse(0))/(bb_max_coarse(0) - bb_min_coarse(0));
            ref_coord_big(1) = (small_bb_map(1) - bb_min_coarse(1))/(bb_max_coarse(1) - bb_min_coarse(1));
            IntegrationPoint ip2;
            ip2.Set2w(ref_coord_big(0), ref_coord_big(1), 1);
            Vector shape_vec(4);
            rfe.CalcShape(ip2, shape_vec);
            for (int k = 0; k < d; k++)
            {
               for (int vd = 0; vd < vdim; vd++)
               {
                  P -> Set((nrr*vd) + d*e + i, (ncc*vd) + d*c + k, shape_vec(k));
               }
            } 
         }
      }
      else
      {
         L2_TetrahedronElement rfe(p);
         const IntegrationRule rfe_nodes = rfe.GetNodes();
         for (int i = 0; i < rfe_nodes.Size(); i++)
         {
            IntegrationPoint ip = rfe_nodes.IntPoint(i);
            Vector small_bb_map(3);
            small_bb_map(0) = (bb_max_fine(0) - bb_min_fine(0))*ip.x + bb_min_fine(0);
            small_bb_map(1) = (bb_max_fine(1) - bb_min_fine(1))*ip.y + bb_min_fine(1);
            small_bb_map(2) = (bb_max_fine(2) - bb_min_fine(2))*ip.z + bb_min_fine(2);
            Vector ref_coord_big(3); 
            ref_coord_big(0) = (small_bb_map(0) - bb_min_coarse(0))/(bb_max_coarse(0) - bb_min_coarse(0));
            ref_coord_big(1) = (small_bb_map(1) - bb_min_coarse(1))/(bb_max_coarse(1) - bb_min_coarse(1));
            ref_coord_big(2) = (small_bb_map(2) - bb_min_coarse(2))/(bb_max_coarse(2) - bb_min_coarse(2));
            IntegrationPoint ip2;
            ip2.Set3(ref_coord_big(0), ref_coord_big(1), ref_coord_big(2));
            Vector shape_vec(8);
            rfe.CalcShape(ip2, shape_vec);
            for (int k = 0; k < d; k++)
            {
               for (int vd = 0; vd < vdim; vd++)
               {
                  P -> Set((nrr*vd) + d*e + i, (ncc*vd) + d*c + k, shape_vec(k));
               }
            } 
         }
      }
   }
   P->Finalize();
   return P;
}

AgglomerationMultigrid::AgglomerationMultigrid(
   FiniteElementSpace &fes, SparseMatrix &Af, int ncoarse, int num_levels, int smoother_choice, bool paraview_vis)
{
   Mesh &mesh = *fes.GetMesh();
   const int ne = mesh.GetNE();

   // Create the mesh hierarchy
   // E2 is a data structure such that E2_ij = gives the level i index for fine element j
   auto E = Agglomerate(*fes.GetMesh(), ncoarse, num_levels);
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

   // output a paraview visualization of the partition, if desired
   if (paraview_vis)
   {
      L2_FECollection l2_fec(0, mesh.Dimension());
      FiniteElementSpace l2_fes(&mesh, &l2_fec);
      GridFunction p_gf(&l2_fes);
      ParaViewDataCollection pv("Agglomeration", &mesh);
      pv.SetPrefixPath("ParaView");
      pv.RegisterField("p", &p_gf);
      int E_size = E.size();
      for (int i = 0; i < E_size; ++i)
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
         pv.SetCycle(E.size());
         pv.SetTime(E.size());
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
   operators[num_levels - 1] = &Af;
   int k = num_levels;
   for (int l = num_levels - 2; l >= 0; --l)
   {
      SparseMatrix *P;
      if (l < num_levels - 2)
      {
         P = CreateInclusionProlongation(k, E, E2, fes);
         SparseMatrix &A_prev = static_cast<SparseMatrix&>(*operators[l + 1]);
         unique_ptr<SparseMatrix> AP(mfem::Mult(A_prev, *P));
         unique_ptr<SparseMatrix> Pt(Transpose(*P));
         operators[l] = mfem::Mult(*Pt, *AP);
         prolongations[l] = P;
      }
      else
      {
         P = CreateNodalProlongation(E, E2, fes);
         SparseMatrix &A_prev = static_cast<SparseMatrix&>(*operators[l + 1]);
         std::cout << "num cols Af = " << A_prev.NumCols() << std::endl;
         unique_ptr<SparseMatrix> AP(mfem::Mult(A_prev, *P));
         unique_ptr<SparseMatrix> Pt(Transpose(*P));
         operators[l] = mfem::Mult(*Pt, *AP);
         prolongations[l] = P;
      }
      k = k-1;
   }

   // Create the smoothers remember block size is num degrees of freedom per element
   SparseMatrix &Ac = static_cast<SparseMatrix&>(*operators[0]);
   smoothers[0] = new UMFPackSolver(Ac);
   int block_size = 3;
   for (int l=1; l < num_levels; l++)
   {
      if (smoother_choice == 0)
      {
         smoothers[l] = new BlockGS(*operators[l], block_size);
      }
      else if (smoother_choice == 1)
      {
         smoothers[l] = new Blockl1Jacobi(*operators[l], block_size);
      }
      else if (smoother_choice == 2)
      {
         real_t alpha = 3.0;
         ScaledOperator scaledop(operators[l], alpha);
         // note - need to set alpha value
         //sm_operators[l] = &(*(mfem::Mult(*Pt, *AP))*=alpha);
         smoothers[l] = new BlockILU(scaledop, block_size);
      }
      else
      {
         MFEM_ABORT("Unknown Smoother.")
      }
   }
}

BlockGS *AdaptiveSmoother(Operator &op, SparseMatrix &A, int blocksize, Vector &x_random, DenseMatrix &B)
{
   // SparseMatrix &A = static_cast<SparseMatrix&>(op);
   Vector PinvAx = x_random;
   BlockGS *smoother = new BlockGS(op, blocksize);
   for(int i = 0; i < 3; i++)
   {
      Vector Ax(PinvAx.Size());
      A.Mult(PinvAx, Ax);
      smoother->Mult(Ax, PinvAx);
      PinvAx /= PinvAx.Norml2();
   }
   real_t rho = PinvAx.Norml2();
   real_t w = 4.0/3.0/rho;
   smoother->SetDamping(1.0 / w);
   real_t tol = 1 + 0.03;
   int num_B_cols = B.Width();
   int num_B_rows = B.Height();
   for(int j = 0; j < num_B_cols; j++)
   {
      Vector b(num_B_rows);
      Vector b_prev(num_B_rows);
      Vector Ab(num_B_rows);
      B.GetColumn(j, b);

      real_t norm_prev = b.Norml2();
      real_t ratio;
      do
      {
         b_prev = b;
         A.Mult(b, Ab);
         smoother->Mult(Ab, b);
         b *= -1.0;
         b += b_prev;

         real_t norm = b.Norml2();
         ratio = norm_prev / norm;
         norm_prev = norm;
      }
      while(ratio > tol);

      for(int i = 0; i < num_B_rows; i++){B(i, j) = b(i);}
   }
   return smoother;
}

SmoothedAggregationGMG::SmoothedAggregationGMG(FiniteElementSpace &fes, SparseMatrix &Af, int ncoarse, int num_levels)
{
   Mesh &mesh = *fes.GetMesh();
   int vdim = fes.GetVDim();
   int ne = mesh.GetNE();
   int dim = mesh.Dimension();
   int n_cut = (dim == 2) ? 2*2 - 1 : 2*2*2 - 2;
   FiniteElementSpace nodal_fes(&mesh, fes.FEColl(), dim);
   GridFunction nodes(&nodal_fes);
   mesh.GetNodes(nodes);
   int nnodes = nodes.Size()/dim*vdim;

   // Create the mesh hierarchy
   auto E = Agglomerate(*fes.GetMesh(), ncoarse, num_levels);

   // Construct matrix B1
   DenseMatrix B0(nnodes, 16); // make number of columns a median
   std::random_device rd;
   std::mt19937 gen(rd()); 
   std::normal_distribution<double> dist(0.0, 1.0);
   for (int i = 0; i < B0.Height(); i++)
   {
      for (int j = 0; j < B0.Width(); j++)
      {
         B0(i,j) = dist(gen);
      }
   }

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
   operators[num_levels - 1] = &Af;
   DenseMatrix B = B0;
   for (int k = num_levels - 2; k >= 0; --k)
   {
      // Estimate Spactra Norm 
      // generate random vector x
      SparseMatrix &A_prev = static_cast<SparseMatrix&>(*operators[k + 1]);
      Vector x_random(A_prev.Height());
      for (int i = 0; i < x_random.Size(); i++){x_random(i) = dist(gen);}
      x_random /= x_random.Norml2();
      BlockGS *A_tilde = AdaptiveSmoother(*operators[k + 1], A_prev, 4, x_random, B);

      //Get the Block Offsets
      Array<int> row_offsets(E[k+1].size()+1);
      Array<int> col_offsets(E[k+1].size()+1);
      row_offsets = 0;
      col_offsets = 0;
      for (int j = 0; j < E[k+1].size(); j++)
      {
         int z = E[k][j];
         row_offsets[z+1] += n_cut;
         col_offsets[z+1] += 1;
      }
      Array<int> part_sizes_dof = row_offsets;
      Array<int> part_sizes_el = col_offsets;
      row_offsets.PartialSum();
      // col_offsets /= n_cut;
      col_offsets.PartialSum(); 

      BlockMatrix P(row_offsets, col_offsets);

      // Get submatrix from B
      for (int i = 0; i < E[k].size(); i++)
      {
         DenseMatrix B_sub(part_sizes_dof[i], B.Width());
         for (int j = 0; j < E[k+1].size(); j++)
         {
            if (E[k][j] == i)
            {
               int row_idx = j;
               for (int nn = 0; nn < n_cut; nn++)
               {
                  for(int c = 0; c <  B.Width(); c++)
                  {
                     B_sub(j+nn, c) = B(row_idx + nn, c);
                  }
               }
            }
         }
         DenseMatrixSVD Bsvd(B_sub, 'A', 'A'); 
         DenseMatrix U = Bsvd.LeftSingularvectors(); 
         SparseMatrix U_sub(part_sizes_dof[i], part_sizes_el[i]);
         for(int ri = 0; ri < part_sizes_dof[i]; ri++)
         {
            for (int ci = 0; ci < part_sizes_el[i]; ci++)
            {
               U_sub.Set(ri, ci,  U(ri, ci));
            }
         }
         U_sub.Finalize();
         P.SetBlock(i, i, &U_sub);
      }
      prolongations[k] = &P;
      SparseMatrix *mono_P = P.CreateMonolithic();
      unique_ptr<SparseMatrix> AP(mfem::Mult(A_prev, *mono_P));
      unique_ptr<SparseMatrix> Pt(Transpose(*mono_P));
      operators[k] = mfem::Mult(*Pt, *AP);
      if (k == 0)
      {
         SparseMatrix &Ac = static_cast<SparseMatrix&>(*operators[0]);
         smoothers[0] = new UMFPackSolver(Ac);
      }
      else
      {
         smoothers[k] = A_tilde;
      }
      B = *mfem::Mult(*Pt, B);
   }
}
} // namespace mfem



