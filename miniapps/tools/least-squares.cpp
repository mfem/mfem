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

#include "mfem.hpp"

using namespace mfem;

namespace reconstruction
{

/// Enumerator for solver types
enum SolverType
{
   direct,
   cg,
   bicgstab,
   minres,
   num_solvers  // last
};

/** Class for an asymmetric (out-of-element) mass integrator $a_K(u,v) := (E(u),v)_K$,
 *  where $E$ is an extension of $u$ (with original domain $\hat{K}$) to $K$.
 *
 *  @note Currently we don't need to derive from MassIntegrator, as GetRule is public.
 *  This could be it's own standalone function.
 *  @dev Kernel implementation: requires kernel_dispatch.hpp. MFEM_THREAD_SAFE...
 */
class AsymmetricMassIntegrator : public MassIntegrator
{
private:
   Vector ngh_shape, self_shape;
   Vector physical_ngh_ip;
   DenseMatrix physical_ngh_pts;
   IntegrationPoint ngh_ip;
   int ngh_ndof, self_ndof;

protected:
   const mfem::FiniteElementSpace *tr_fes, *te_fes;

public:

   AsymmetricMassIntegrator() {};

   /// Assembles element mass matrix, extending @a ngh_fe to @a self_fe.
   void AsymmetricElementMatrix(const FiniteElement &ngh_fe,
                                const FiniteElement &self_fe,
                                ElementTransformation &ngh_tr,
                                ElementTransformation &self_tr,
                                DenseMatrix &el_mat);
};

void AsymmetricMassIntegrator::AsymmetricElementMatrix(const FiniteElement
                                                       &ngh_fe,
                                                       const FiniteElement &self_fe,
                                                       ElementTransformation &ngh_tr,
                                                       ElementTransformation &self_tr,
                                                       DenseMatrix &elmat)
{
   ngh_ndof = ngh_fe.GetDof();
   self_ndof = self_fe.GetDof();

   self_shape.SetSize(self_ndof);
   ngh_shape.SetSize(ngh_ndof);
   elmat.SetSize(self_ndof, ngh_ndof);
   elmat = 0.0;

   /* Notes:
    * Let T_K : ref_K -> K from reference to physical
    * Let phi_i_K a shape function on the element K
    * and phi_i a shape function on the reference element ref_K
    * Let N(K) be an adjacent element to K
    *
    * Then phi_i_K(x) = phi_i(inv_T_K(x)), x in Omega.
    * Moreover,
    * int_N(K) phi_i_K(x) v(x) dx
    * = int_N(K) phi_i(inv_T_K(x)) v(x) dx
    * = int_ref_K phi_i(inv_T_K( T_N(K)(y) )) v( T_N(K)(y) ) |Jac T_N(K)| dy.
    */

   const int int_order = self_fe.GetOrder() + ngh_fe.GetOrder() + self_tr.OrderW();
   const IntegrationRule &ir = IntRules.Get(self_fe.GetGeomType(), int_order);

   ngh_tr.Transform(ir, physical_ngh_pts);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      physical_ngh_pts.GetColumn(i, physical_ngh_ip);

      // Pullback physical neighboring integration points to
      // "extended" reference element under self_tr
      InverseElementTransformation inv_tr(&self_tr);
      inv_tr.SetPhysicalRelTol(1e-16);
      inv_tr.SetMaxIter(50);
      inv_tr.SetSolverType(InverseElementTransformation::SolverType::Newton);
      inv_tr.SetInitialGuessType(
         InverseElementTransformation::InitGuessType::ClosestPhysNode);
      inv_tr.Transform(physical_ngh_ip, ngh_ip);

      ngh_tr.SetIntPoint(&ngh_ip);
      self_tr.SetIntPoint(&ip);

      // Compute shape functions on self_fe
      ngh_fe.CalcPhysShape(ngh_tr, ngh_shape);
      self_fe.CalcPhysShape(self_tr, self_shape);

      self_shape *= self_tr.Weight() * ip.weight;
      AddMultVWt(self_shape, ngh_shape, elmat);
   }
}

/// @brief Dense small least squares solver
/// @dev This grew a lot, might good to derive from Solver...
void LSSolver(Solver& solver, const DenseMatrix& A, const Vector& b,
              Vector& x, real_t shift = 0.0)
{
   x.SetSize(A.Width());
   x = 0.0;

   Vector Atb(A.Width());
   A.MultTranspose(b, Atb);

   if (dynamic_cast<IterativeSolver*>(&solver))
   {
      // Don't compute the products
      TransposeOperator At(&A);
      ProductOperator AtA(&At, &A, false, false);
      IdentityOperator I(AtA.Height());
      SumOperator AtA_reg(&AtA, 1.0, &I, shift, false, false);
      solver.SetOperator(AtA_reg);
      solver.Mult(Atb, x); // TODO(Gabriel): Lazy fix, Operators are scoped...
   }
   else if (dynamic_cast<DenseMatrixInverse*>(&solver))
   {
      DenseMatrix AtA_reg(A.Width());
      Vector col_i(A.Height()), col_j(A.Height());
      for (int i = 0; i < A.Width(); i++)
      {
         A.GetColumn(i, col_i);
         for (int j = 0; j < A.Width(); j++)
         {
            A.GetColumn(j, col_j);
            AtA_reg(i,j) = col_i * col_j + (i==j)*shift;
         }
      }
      solver.SetOperator(AtA_reg);
      solver.Mult(Atb, x); // TODO(Gabriel): Lazy fix, redundant...
   }
}

/// @brief Dense small least squares solver, with constrains @a C with value @a c
void LSSolver(Solver& solver, const DenseMatrix& A, const DenseMatrix& C,
              const Vector& b, const Vector& c, Vector& x, Vector& y,
              real_t shift = 0.0)
{
   TransposeOperator At(&A);
   ProductOperator AtA(&At, &A, false, false);
   IdentityOperator I(AtA.Height());
   SumOperator AtA_reg(&AtA, 1.0, &I, shift, false, false);

   TransposeOperator Ct(&C);

   // Block matrix
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = AtA_reg.Width();
   offsets[2] = AtA_reg.Width() + Ct.Width();

   BlockOperator block_mat(offsets);
   block_mat.SetBlock(0, 0, &AtA_reg);
   block_mat.SetBlock(0, 1, &Ct);
   block_mat.SetBlock(1, 0, const_cast<DenseMatrix*>(&C));

   // Block vectors
   BlockVector rhs(offsets), z(offsets);
   rhs = 0.0;
   z = 0.0;

   Vector Atb(A.Width());
   A.MultTranspose(b, Atb);

   rhs.SetVector(Atb, offsets[0]);
   rhs.SetVector(c, offsets[1]);

   solver.SetOperator(block_mat);
   solver.Mult(rhs, z);

   x.SetSize(A.Width());
   y.SetSize(C.Width());

   z.GetBlockView(0,x);
   z.GetBlockView(1,y);
}

void CheckLSSolver(const DenseMatrix& A, const Vector& b, const Vector& x)
{
   Vector res(b), sol(A.Width());
   A.AddMult(x,res,-1.0);
   A.MultTranspose(res, sol);
   mfem::out << "\nNorm of the residual of LS problem: " << sol.Norml2() <<
             std::endl;
}

/// @brief Get the L1-Jacobi row-subs for the @b normal matrix associated with @a A
void GetNormalL1Diag(const DenseMatrix& A, Vector& l1_diag)
{
   l1_diag.SetSize(A.Height());
   Vector col_j(A.Height()), col_k(A.Height());
   real_t sums;
   for (int i = 0; i < A.Height(); i++)
   {
      sums = 0.0;
      for (int j = 0; j < A.Width(); j++)
      {
         A.GetColumn(j, col_j);
         for (int k = 0; k < A.Width(); k++) { A.GetColumn(k, col_k); }
         sums += std::abs(col_j * col_k);
      }
      l1_diag(i) = sums;
   }
}

/// @brief Saturates a neighborhood with hopes to well-pose the least squares problem.
/// @dev Make friends with @a MCMesh
void SaturateNeighborhood(NCMesh& mesh, const int element_idx,
                          const int target_ndofs, const int contributed_ndofs,
                          Array<int>& neighbors)
{
   Array<int> temp;
   mesh.FindNeighbors(element_idx, neighbors);
   neighbors.Append(element_idx);
   while (neighbors.Size() * contributed_ndofs < target_ndofs)
   {
      mesh.NeighborExpand(neighbors, temp);
      neighbors = temp;
   }
   neighbors.Unique();
}

} // end namespace reconstruction

using namespace reconstruction;

int main(int argc, char* argv[])
{
   Mpi::Init(argc, argv);

   // Default command-line options
   int ser_ref_levels = 0;
   int par_ref_levels = 0;

   int order_original = 3;
   int order_averages = 0;
   int order_reconstruction = 1;

   SolverType solver_type = direct;
   real_t solver_reg = 0.0;
   real_t solver_rtol = 1.0e-30;
   real_t solver_atol = 0.0;
   int solver_maxiter = 1000;
   // TODO(Gabriel): Not implemented yet
   // int solver_plevel = 3;

   bool preserve_volumes = false; // TODO(Gabriel): Why this is not working?
   bool save_to_file = false;

   // Parse options
   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of serial refinement steps.");
   args.AddOption(&par_ref_levels, "-rp", "--refine",
                  "Number of parallel refinement steps.");

   args.AddOption(&order_original, "-O", "--order-original",
                  "Order original broken space.");
   // TODO(Gabriel): Not implemented yet
   // args.AddOption(&order_averages, "-A", "--order-averages",
   //                "Order averaged broken space.");
   args.AddOption(&order_reconstruction, "-R", "--order-reconstruction",
                  "Order of reconstruction broken space.");

   args.AddOption(&preserve_volumes, "-V", "--preserve-volumes", "-no-V",
                  "--no-preserve-volumes", "Preserve averages (volumes) by"
                  " solving a constrained least squares problem");

   args.AddOption((int*)&solver_type, "-S", "--solver",
                  "Solvers to be considered:"
                  "\n\t0: Direct Solver"
                  "\n\t1: CG - Conjugate Gradient"
                  "\n\t2: BiCGSTAB -Biconjugate gradient stabilized"
                  "\n\t3: MINRES - Minimal residual");
   args.AddOption(&solver_reg, "-Sreg", "--solver-reg",
                  "Add regularization term to the least squares problem");
   args.AddOption(&solver_rtol, "-Srtol", "--solver-rtol",
                  "Relative tolerance for the iterative solver");
   args.AddOption(&solver_atol, "-Satol", "--solver-atol",
                  "Absolute tolerance for the iterative solver");
   args.AddOption(&solver_maxiter, "-Smi", "--solver-maxiter",
                  "Maximum number of iterations for the solver");
   // TODO(Gabriel): Not implemented yet
   // args.AddOption(&solver_plevel, "-Sp", "--solver-print",
   //                "Print level for the iterative solver");

   args.AddOption(&save_to_file, "-s", "--save", "-no-s",
                  "--no-save", "Show or not show approximation error.");

   args.ParseCheck();
   MFEM_VERIFY((ser_ref_levels >= 0) && (par_ref_levels >= 0), "")
   MFEM_VERIFY((solver_reg>=0.0), "")
   MFEM_VERIFY(order_original > order_averages, "")
   MFEM_VERIFY((0 <= solver_type) && (solver_type < num_solvers),
               "invalid solver type: " << solver_type);

   if (Mpi::Root())
   {
      mfem::out << "Number of serial refs.:  " << ser_ref_levels << "\n";
      mfem::out << "Number of parallel refs: " << par_ref_levels << "\n";
      mfem::out << "Original order:          " << order_original << "\n";
      mfem::out << "Original averages:       " << order_averages << "\n";
      mfem::out << "Original reconstruction: " << order_reconstruction << "\n";
   }

   // Mesh
   const int num_x = 2;
   const int num_y = 2;
   Mesh serial_mesh = Mesh::MakeCartesian2D(num_x, num_y, Element::QUADRILATERAL);
   for (int i = 0; i < ser_ref_levels; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   for (int i = 0; i < par_ref_levels; ++i) { mesh.UniformRefinement(); }
   NCMesh nc_mesh(static_cast<Mesh*>(&mesh));

   // target function u(x,y)
   const int k_x = 1;
   const int k_y = 2;
   std::function<real_t(const Vector &)> u_function =
      [=](const Vector& x)
   {
      return std::cos(2*M_PI*k_x * x(0)) * std::sin(2*M_PI*k_y * x(1));
   };
   FunctionCoefficient u_coefficient(u_function);

   // Broken spaces
   L2_FECollection fec_original(order_original, mesh.Dimension());
   L2_FECollection fec_averages(order_averages, mesh.Dimension());
   L2_FECollection fec_reconstruction(order_reconstruction, mesh.Dimension());

   ParFiniteElementSpace fes_original(&mesh, &fec_original);
   ParFiniteElementSpace fes_averages(&mesh, &fec_averages);
   ParFiniteElementSpace fes_reconstruction(&mesh, &fec_reconstruction);

   ParGridFunction u_original(&fes_original);
   ParGridFunction u_averages(&fes_averages);
   ParGridFunction u_rec_avg(&fes_averages);
   ParGridFunction diff(&fes_averages);
   ParGridFunction u_reconstruction(&fes_reconstruction);

   u_original.ProjectCoefficient(u_coefficient);
   u_original.GetElementAverages(u_averages);

   // Declare mass integrator
   AsymmetricMassIntegrator mass;

   // Compute local volumes
   ConstantCoefficient zeros(0.0);
   ParGridFunction ones(&fes_averages);

   ones = 1.0;
   Vector volumes(mesh.GetNE());
   ones.ComputeElementL1Errors(zeros, volumes);

   // Solver choice
   // TODO(Gabriel): Support more solvers?
   // TODO(Gabriel): MPI_Comm  constructor for MPI...
   // TODO(Gabriel): Support more PCs?
   /* Notes:
    * OperatorJacobiSmoother will call AssembleDiagonal on
    * Operator, even if the Setup function will be called
    * later on. DenseMatrix does not have this method
    * implemented.
    */
   Solver *small_solver = nullptr;
   switch (solver_type)
   {
      case bicgstab:
         small_solver = new BiCGSTABSolver();
         break;
      case cg:
         small_solver = new CGSolver();
         break;
      case minres:
         small_solver = new MINRESSolver();
         break;
      case direct:
      default:
         small_solver = new DenseMatrixInverse();
   }

   // Solver setting
   small_solver->iterative_mode = false;
   auto it_solver = dynamic_cast<IterativeSolver*>(small_solver);
   if (it_solver)
   {
      // TODO(Gabriel): Not implemented yet
      IterativeSolver::PrintLevel print_level;
      print_level.All();

      it_solver->SetRelTol(solver_rtol);
      it_solver->SetAbsTol(solver_atol);
      it_solver->SetMaxIter(solver_maxiter);
      it_solver->SetPrintLevel(print_level);
   }

   // L1-Jacobi Preconditioner
   Vector l1_diag;

   // Neighboring elements and DoFs
   Array<int> ngh_e, temp_ngh, local_dofs;
   auto ngh_tr = new IsoparametricTransformation;
   auto self_tr = new IsoparametricTransformation;

   // TODO(Gabriel): Loop too big!
   for (int e_idx = 0; e_idx < mesh.GetNE(); e_idx++ )
   {
      const int fe_rec_e_ndof = fes_reconstruction.GetFE(e_idx)->GetDof();
      const int fe_avg_e_ndof = fes_averages.GetFE(e_idx)->GetDof();

      SaturateNeighborhood(nc_mesh, e_idx, fe_rec_e_ndof, fe_avg_e_ndof, ngh_e);
      if (preserve_volumes) { ngh_e.DeleteFirst(e_idx); }

      // Define small matrix
      const int num_ngh_e = ngh_e.Size();
      DenseMatrix local_mass_mat(num_ngh_e, fe_rec_e_ndof);

      for (int i = 0; i < num_ngh_e; i++)
      {
         const int ngh_e_idx = ngh_e[i];

         fes_reconstruction.GetElementTransformation(ngh_e_idx, ngh_tr);
         fes_averages.GetElementTransformation(e_idx, self_tr);

         DenseMatrix ngh_elem_mat;
         mass.AsymmetricElementMatrix(*fes_reconstruction.GetFE(ngh_e_idx),
                                      *fes_averages.GetFE(e_idx),
                                      *ngh_tr, *self_tr, ngh_elem_mat);

         if (ngh_elem_mat.Height()!=1) { mfem_error("High order case for source space not implemented yet!"); }
         Vector ngh_vec;
         ngh_elem_mat.GetRow(0, ngh_vec);
         local_mass_mat.SetRow(i, ngh_vec);
      }

      // Get local volumes and scale patch matrix
      Vector local_volumes(num_ngh_e);
      volumes.GetSubVector(ngh_e, local_volumes);
      local_mass_mat.InvLeftScaling(local_volumes);
      // End define small matrix

      // Define L1-Jacobi PC (diagonal vector) for A^T A
      GetNormalL1Diag(local_mass_mat, l1_diag);

      // Solve
      Vector local_u_avg, local_u_rec;
      u_averages.GetSubVector(ngh_e, local_u_avg);

      // Apply preconditioner
      local_mass_mat.InvLeftScaling(l1_diag);
      local_u_avg /= l1_diag;

      if (preserve_volumes)
      {
         Vector _mult, exact_average_e(1), self_volume(1);
         exact_average_e = u_averages(e_idx);
         self_volume = volumes(e_idx);

         DenseMatrix self_avg_mat;
         mass.AsymmetricElementMatrix(*fes_reconstruction.GetFE(e_idx),
                                      *fes_averages.GetFE(e_idx),
                                      *self_tr, *self_tr, self_avg_mat);
         self_avg_mat.InvLeftScaling(self_volume);

         LSSolver(*small_solver, local_mass_mat, self_avg_mat,
                  local_u_avg, exact_average_e,
                  local_u_rec, _mult, solver_reg);
      }
      else
      {
         LSSolver(*small_solver, local_mass_mat, local_u_avg, local_u_rec, solver_reg);
      }

      // Check solver
      if (it_solver && !it_solver->GetConverged()) { mfem_error("\n\tIterative solver failed to converge!"); }
      CheckLSSolver(local_mass_mat, local_u_avg, local_u_rec);

      // Integrate into global solution
      fes_reconstruction.GetElementDofs(e_idx, local_dofs);
      u_reconstruction.SetSubVector(local_dofs, local_u_rec);

      ngh_e.DeleteAll();
      temp_ngh.DeleteAll();
      local_dofs.DeleteAll();
   }
   u_reconstruction.GetElementAverages(u_rec_avg);

   char vishost[] = "localhost";
   int visport = 20000;
   socketstream glvis_original(vishost, visport);
   socketstream glvis_averages(vishost, visport);
   socketstream glvis_rec_avg(vishost, visport);
   socketstream glvis_reconstruction(vishost, visport);
   if (glvis_original && glvis_averages && glvis_reconstruction)
   {
      //glvis_original.precision(8);
      glvis_original << "parallel " << mesh.GetNRanks()
                     << " " << mesh.GetMyRank() << "\n"
                     << "solution\n" << mesh << u_original
                     << "window_title 'original'\n" << std::flush;
      MPI_Barrier(mesh.GetComm());
      //glvis_averages.precision(8);
      glvis_averages << "parallel " << mesh.GetNRanks()
                     << " " << mesh.GetMyRank() << "\n"
                     << "solution\n" << mesh << u_averages
                     << "window_title 'averages'\n" << std::flush;
      MPI_Barrier(mesh.GetComm());
      //glvis_reconstruction.precision(8);
      glvis_reconstruction << "parallel " << mesh.GetNRanks()
                           << " " << mesh.GetMyRank() << "\n"
                           << "solution\n" << mesh << u_reconstruction
                           << "window_title 'reconstruction'\n" << std::flush;
      MPI_Barrier(mesh.GetComm());
      //glvis_reconstruction.precision(8);
      glvis_rec_avg << "parallel " << mesh.GetNRanks()
                    << " " << mesh.GetMyRank() << "\n"
                    << "solution\n" << mesh << u_rec_avg
                    << "window_title 'rec average'\n" << std::flush;
   }

   // Error studies
   real_t error = u_reconstruction.ComputeL2Error(u_coefficient);
   subtract(u_rec_avg, u_averages, diff);
   real_t error_avg = diff.ComputeL2Error(zeros);

   if (Mpi::Root())
   {
      mfem::out << "\n|| Rec(u_h) - u ||_{L^2} = " << error << "\n" << std::endl;
      mfem::out << "\n|| Avg(Rec(u_h)) - u_h ||_{ell^2} = " << error_avg << "\n" <<
                std::endl;
   }

   if (save_to_file && Mpi::Root())
   {
      Vector el_error(mesh.GetNE());
      ones.ComputeElementLpErrors(2.0, zeros, el_error);
      real_t hmax = el_error.Max();

      std::ofstream file;
      file.open("convergence.csv", std::ios::out | std::ios::app);
      if (!file.is_open())
      {
         mfem_error("Failed to open");
      }
      file << std::scientific << std::setprecision(16);
      file << error
           << "," << fes_averages.GetNConformingDofs()
           << "," << fes_reconstruction.GetNConformingDofs()
           << "," << hmax
           << "," << mesh.GetNE() << std::endl;
      file.close();
   }

   if (small_solver) { delete small_solver; }
   delete ngh_tr;
   delete self_tr;

   Mpi::Finalize();
   return 0;
}
