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
 *  @todo Kernel implementation: requires kernel_dispatch.hpp. MFEM_THREAD_SAFE...
 */
class AsymmetricMassIntegrator : public MassIntegrator
{
private:
   Vector ngh_shape, self_shape;
   Vector phys_neighbor_ip;
   DenseMatrix phys_neighbor_pts;
   IntegrationPoint ngh_ip;
   int neighbor_ndof, e_ndof;

protected:
   const mfem::FiniteElementSpace *tr_fes, *te_fes;

public:

   AsymmetricMassIntegrator() {};

   /// Assembles element mass matrix, extending @a neighbor_fe to @a e_fe.
   void AsymmetricElementMatrix(const FiniteElement &neighbor_fe,
                                const FiniteElement &e_fe,
                                ElementTransformation &neighbor_trans,
                                ElementTransformation &e_trans,
                                DenseMatrix &el_mat,
                                int _max_iter = 1000,
                                real_t _rtol = 1e-15);

   /// Assembles element mass matrix, computing @a neighbor_fe in @a e_trans.
   void AsymmetricElementMatrix(const FiniteElement &neighbor_fe,
                                ElementTransformation &neighbor_trans,
                                ElementTransformation &e_trans,
                                DenseMatrix &el_mat,
                                int _max_iter = 1000,
                                real_t _rtol = 1e-15);
};

void AsymmetricMassIntegrator::AsymmetricElementMatrix(const FiniteElement
                                                       &neighbor_fe,
                                                       const FiniteElement &e_fe,
                                                       ElementTransformation &neighbor_trans,
                                                       ElementTransformation &e_trans,
                                                       DenseMatrix &elmat,
                                                       int _max_iter,
                                                       real_t _rtol)
{
   neighbor_ndof = neighbor_fe.GetDof();
   e_ndof = e_fe.GetDof();

   self_shape.SetSize(e_ndof);
   ngh_shape.SetSize(neighbor_ndof);
   elmat.SetSize(e_ndof, neighbor_ndof);
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

   const int int_order = e_fe.GetOrder() + neighbor_fe.GetOrder() +
                         e_trans.OrderW();
   const IntegrationRule &ir = IntRules.Get(e_fe.GetGeomType(), int_order);

   neighbor_trans.Transform(ir, phys_neighbor_pts);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      phys_neighbor_pts.GetColumn(i, phys_neighbor_ip);

      // Pullback physical neighboring integration points to
      // "extended" reference element under e_trans
      using InvElTr = mfem::InverseElementTransformation;
      InverseElementTransformation inv_tr(&e_trans);
      inv_tr.SetPhysicalRelTol(_rtol);
      inv_tr.SetMaxIter(_max_iter);
      inv_tr.SetSolverType(InvElTr::SolverType::Newton);
      inv_tr.SetPrintLevel(0); // TODO(Gabriel): Debug
      inv_tr.SetInitialGuessType(InvElTr::InitGuessType::ClosestPhysNode);
      MFEM_VERIFY(inv_tr.Transform(phys_neighbor_ip,
                                   ngh_ip) != InvElTr::TransformResult::Unknown, "");

      neighbor_trans.SetIntPoint(&ngh_ip);
      e_trans.SetIntPoint(&ip);

      // Compute shape functions on e_fe
      neighbor_fe.CalcPhysShape(neighbor_trans, ngh_shape);
      e_fe.CalcPhysShape(e_trans, self_shape);

      self_shape *= e_trans.Weight() * ip.weight;
      AddMultVWt(self_shape, ngh_shape, elmat);
   }
}

void AsymmetricMassIntegrator::AsymmetricElementMatrix(const FiniteElement
                                                       &neighbor_fe,
                                                       ElementTransformation &neighbor_trans,
                                                       ElementTransformation &e_trans,
                                                       DenseMatrix &elmat,
                                                       int _max_iter,
                                                       real_t _rtol)
{
   neighbor_ndof = neighbor_fe.GetDof();

   ngh_shape.SetSize(neighbor_ndof);
   elmat.SetSize(neighbor_ndof);
   elmat = 0.0;

   const int int_order = 2*neighbor_fe.GetOrder() + e_trans.OrderW();
   const IntegrationRule &ir = IntRules.Get(neighbor_fe.GetGeomType(), int_order);

   neighbor_trans.Transform(ir, phys_neighbor_pts);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      phys_neighbor_pts.GetColumn(i, phys_neighbor_ip);

      // Pullback physical neighboring integration points to
      // "extended" reference element under e_trans
      using InvElTr = mfem::InverseElementTransformation;
      InverseElementTransformation inv_tr(&e_trans);
      inv_tr.SetPhysicalRelTol(_rtol);
      inv_tr.SetMaxIter(_max_iter);
      inv_tr.SetSolverType(InvElTr::SolverType::Newton);
      inv_tr.SetPrintLevel(0); // TODO(Gabriel): Debug
      inv_tr.SetInitialGuessType(InvElTr::InitGuessType::ClosestPhysNode);
      MFEM_VERIFY(inv_tr.Transform(phys_neighbor_ip,
                                   ngh_ip) != InvElTr::TransformResult::Unknown, "");

      neighbor_trans.SetIntPoint(&ngh_ip);
      e_trans.SetIntPoint(&ip);

      neighbor_fe.CalcPhysShape(neighbor_trans, ngh_shape);
      AddMult_a_VVt(e_trans.Weight()*ip.weight, ngh_shape, elmat);
   }
}

/// @brief Dense small least squares solver
/// @todo This grew a lot, might good to derive from Solver...
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
/// @todo Make friends with @a MCMesh
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
   int solver_plevel = -1;

   real_t newton_rtol = 1.0e-15;
   int newton_maxiter = 100;

   bool preserve_volumes = false; // TODO(Gabriel): Why this is not working?
   bool save_to_file = false;

   bool visualization = true;
   int visport = 19916;

   // Parse options
   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of serial refinement steps.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
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
   args.AddOption(&solver_plevel, "-Sp", "--solver-print",
                  "Print level for the iterative solver:"
                  "\n\t0: All"
                  "\n\t1: First and last with warnings"
                  "\n\t2: Errors"
                  "\n\tdefault: None"
                  "\nA negative value deactivates LS check");

   args.AddOption(&newton_rtol, "-Nrtol", "--newton-rtol",
                  "Relative tolerance for the Newton solver (q-points)");
   args.AddOption(&newton_maxiter, "-Nmi", "--newton-maxiter",
                  "Maximum number of iterations for the Newton solver (q-points)");

   args.AddOption(&save_to_file, "-s", "--save", "-no-s",
                  "--no-save", "Show or not show approximation error.");

   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");

   args.ParseCheck();
   MFEM_VERIFY((ser_ref_levels >= 0) && (par_ref_levels >= 0), "")
   MFEM_VERIFY((solver_reg>=0.0), "")
   MFEM_VERIFY(order_original > order_averages, "")
   MFEM_VERIFY((0 <= solver_type) && (solver_type < num_solvers),
               "invalid solver type: " << solver_type);

   if (Mpi::Root())
   {
      mfem::out << "Number of serial refs.:  " << ser_ref_levels << "\n"
                << "Number of parallel refs: " << par_ref_levels << "\n"
                << "Original order:          " << order_original << "\n"
                << "Original averages:       " << order_averages << "\n"
                << "Original reconstruction: " << order_reconstruction << "\n"
                << "Newton relative tol:     " << newton_rtol << "\n"
                << "Newton max. num. iter.:  " << newton_maxiter << std::endl;
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
   const real_t k_x = 0.5;
   const real_t k_y = 1.0;
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

   // Auxiliary constant coefficients
   ConstantCoefficient ccf_zeros(0.0);
   ConstantCoefficient ccf_ones(1.0);

   // Auxiliary partition of unity
   ParGridFunction punity_avg(&fes_averages);
   ParGridFunction punity_rec(&fes_reconstruction);

   punity_rec.ProjectCoefficient(ccf_ones);
   punity_avg.ProjectCoefficient(ccf_ones);

   // Compute volumes
   Vector volumes(mesh.GetNE());
   punity_avg.ComputeElementL1Errors(ccf_zeros, volumes);

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
   Solver *solver = nullptr;
   switch (solver_type)
   {
      case bicgstab:
         solver = new BiCGSTABSolver();
         break;
      case cg:
         solver = new CGSolver();
         break;
      case minres:
         solver = new MINRESSolver();
         break;
      case direct:
      default:
         solver = new DenseMatrixInverse();
   }

   // Solver setting
   solver->iterative_mode = false;
   auto it_solver = dynamic_cast<IterativeSolver*>(solver);
   if (it_solver)
   {
      IterativeSolver::PrintLevel print_level;
      switch (solver_plevel)
      {
         case 0:
            print_level.All();
            break;
         case 1:
            print_level.FirstAndLast();
            print_level.Warnings();
         case 2:
            print_level.Errors();
            break;
         default:
            print_level.None();
      }

      it_solver->SetRelTol(solver_rtol);
      it_solver->SetAbsTol(solver_atol);
      it_solver->SetMaxIter(solver_maxiter);
      it_solver->SetPrintLevel(print_level);
   }

   // TODO(Gabriel): L1-Jacobi Preconditioner
   // Vector l1_diag;

   // Neighboring elements and DoFs
   Array<int> neighbors_e, e_dofs;
   auto neighbor_trans = new IsoparametricTransformation;
   auto e_trans = new IsoparametricTransformation;

   // TODO(Gabriel): Loop too big!
   for (int e_idx = 0; e_idx < mesh.GetNE(); e_idx++ )
   {
      const int fe_rec_e_ndof = fes_reconstruction.GetFE(e_idx)->GetDof();
      const int fe_avg_e_ndof = fes_averages.GetFE(e_idx)->GetDof();

      SaturateNeighborhood(nc_mesh, e_idx, fe_rec_e_ndof, fe_avg_e_ndof, neighbors_e);
      if (preserve_volumes) { neighbors_e.DeleteFirst(e_idx); }

      // BEGIN definition small matrix
      const int num_neighbors = neighbors_e.Size();
      DenseMatrix fe_rec_to_neighbors_mat(num_neighbors, fe_rec_e_ndof);

      for (int i = 0; i < num_neighbors; i++)
      {
         const int neighbor_idx = neighbors_e[i];

         fes_reconstruction.GetElementTransformation(neighbor_idx, neighbor_trans);
         fes_averages.GetElementTransformation(e_idx, e_trans);

         DenseMatrix e_to_neighbor_mat;
         mass.AsymmetricElementMatrix(*fes_reconstruction.GetFE(neighbor_idx),
                                      *fes_averages.GetFE(e_idx),
                                      *neighbor_trans, *e_trans,
                                      e_to_neighbor_mat,
                                      newton_maxiter, newton_rtol);

         if (e_to_neighbor_mat.Height()!=1) { mfem_error("High order case for source space not implemented yet!"); }
         Vector neighbor_idx_row;
         e_to_neighbor_mat.GetRow(0, neighbor_idx_row);
         fe_rec_to_neighbors_mat.SetRow(i, neighbor_idx_row);
      }

      // Get neighbors volumes and scale patch matrix
      Vector neighbors_volumes(num_neighbors);
      volumes.GetSubVector(neighbors_e, neighbors_volumes);
      fe_rec_to_neighbors_mat.InvLeftScaling(neighbors_volumes);
      // END definition small matrix

      // TODO(Gabriel): Cannot use here if need original matrix
      // Define L1-Jacobi PC (diagonal vector) for A^T A
      // GetNormalL1Diag(fe_rec_to_neighbors_mat, l1_diag);

      // Get local averages and local ones
      Vector u_avg_neighbors, u_rec_e, punity_e;
      fes_reconstruction.GetElementDofs(e_idx, e_dofs);
      u_averages.GetSubVector(neighbors_e, u_avg_neighbors);
      punity_rec.GetSubVector(e_dofs, punity_e);

      // Apply preconditioner
      // fe_rec_to_neighbors_mat.InvLeftScaling(l1_diag);
      // u_avg_neighbors /= l1_diag;

      // Solve
      if (preserve_volumes)
      {
         real_t u_e_avg = u_averages(e_idx);

         Vector shape_avg_e, local_ones(num_neighbors);
         local_ones = 1.0;
         DenseMatrix temp_mat;
         mass.AsymmetricElementMatrix(*fes_reconstruction.GetFE(e_idx),
                                      *fes_averages.GetFE(e_idx),
                                      *e_trans, *e_trans, temp_mat);
         temp_mat *= -1.0/volumes(e_idx);
         temp_mat.GetRow(0, shape_avg_e);

         // Set up A - 1 x shape_avgs
         AddMultVWt(local_ones, shape_avg_e, fe_rec_to_neighbors_mat);
         // Set up u avgs minus average on e
         add(1.0, u_avg_neighbors, -u_e_avg, local_ones, u_avg_neighbors);

         LSSolver(*solver, fe_rec_to_neighbors_mat, u_avg_neighbors, u_rec_e,
                  solver_reg);

         // Add the average to the final solution
         add(1.0, u_rec_e, u_e_avg, punity_e, u_rec_e);
      }
      else
      {
         LSSolver(*solver, fe_rec_to_neighbors_mat, u_avg_neighbors, u_rec_e,
                  solver_reg);
      }

      // Check solver
      if (it_solver && !it_solver->GetConverged()) { mfem_error("\n\tIterative solver failed to converge!"); }
      if (solver_plevel >= 0) { CheckLSSolver(fe_rec_to_neighbors_mat, u_avg_neighbors, u_rec_e); }

      // Integrate into global solution
      u_reconstruction.SetSubVector(e_dofs, u_rec_e);

      neighbors_e.DeleteAll();
      e_dofs.DeleteAll();
   }
   u_reconstruction.GetElementAverages(u_rec_avg);

   char vishost[] = "localhost";
   socketstream glvis_original(vishost, visport);
   socketstream glvis_averages(vishost, visport);
   socketstream glvis_rec_avg(vishost, visport);
   socketstream glvis_reconstruction(vishost, visport);

   if (glvis_original &&
       glvis_averages &&
       glvis_rec_avg &&
       glvis_reconstruction &&
       visualization)
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
   else if (visualization)
   {
      MFEM_WARNING("Cannot connect to glvis server, disabling visualization.")
   }

   // Error studies
   real_t error = u_reconstruction.ComputeL2Error(u_coefficient);
   subtract(u_rec_avg, u_averages, diff);
   real_t error_avg = diff.ComputeL2Error(ccf_zeros);

   if (Mpi::Root())
   {
      mfem::out << "\n|| Rec(u_h) - u ||_{L^2} = " << error << "\n" << std::endl;
      mfem::out << "\n|| Avg(Rec(u_h)) - u_h ||_{L^2} = " << error_avg << "\n" <<
                std::endl;
   }

   if (save_to_file && Mpi::Root())
   {
      Vector el_error(mesh.GetNE());
      punity_avg.ComputeElementLpErrors(2.0, ccf_zeros, el_error);
      real_t hmax = el_error.Max();

      std::ofstream file;
      file.open("convergence.csv", std::ios::out | std::ios::app);
      if (!file.is_open()) { mfem_error("Failed to open file"); }
      file << std::scientific << std::setprecision(16);
      file << error
           << "," << fes_averages.GetNConformingDofs()
           << "," << fes_reconstruction.GetNConformingDofs()
           << "," << hmax
           << "," << mesh.GetNE() << std::endl;
      file.close();
   }

   if (solver) { delete solver; }
   delete neighbor_trans;
   delete e_trans;

   Mpi::Finalize();
   return 0;
}
