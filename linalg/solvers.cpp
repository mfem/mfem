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

#include "linalg.hpp"
#include "lapack.hpp"
#include "../general/annotation.hpp"
#include "../general/forall.hpp"
#include "../general/globals.hpp"
#include "../fem/bilinearform.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <set>

namespace mfem
{

using namespace std;

IterativeSolver::IterativeSolver()
   : Solver(0, true)
{
   oper = NULL;
   prec = NULL;
   max_iter = 10;
   rel_tol = abs_tol = 0.0;
#ifdef MFEM_USE_MPI
   dot_prod_type = 0;
#endif
}

#ifdef MFEM_USE_MPI

IterativeSolver::IterativeSolver(MPI_Comm comm_)
   : Solver(0, true)
{
   oper = NULL;
   prec = NULL;
   max_iter = 10;
   rel_tol = abs_tol = 0.0;
   dot_prod_type = 1;
   comm = comm_;
}

#endif // MFEM_USE_MPI

real_t IterativeSolver::Dot(const Vector &x, const Vector &y) const
{
#ifndef MFEM_USE_MPI
   return (x * y);
#else
   if (dot_prod_type == 0)
   {
      return (x * y);
   }
   else
   {
      return InnerProduct(comm, x, y);
   }
#endif
}

void IterativeSolver::SetPrintLevel(int print_lvl)
{
   print_options = FromLegacyPrintLevel(print_lvl);
   int print_level_ = print_lvl;

#ifdef MFEM_USE_MPI
   if (dot_prod_type != 0)
   {
      int rank;
      MPI_Comm_rank(comm, &rank);
      if (rank != 0) // Suppress output.
      {
         print_level_ = -1;
         print_options = PrintLevel().None();
      }
   }
#endif

   print_level = print_level_;
}

void IterativeSolver::SetPrintLevel(PrintLevel options)
{
   print_options = options;

   int derived_print_level = GuessLegacyPrintLevel(options);

#ifdef MFEM_USE_MPI
   if (dot_prod_type != 0)
   {
      int rank;
      MPI_Comm_rank(comm, &rank);
      if (rank != 0)
      {
         derived_print_level = -1;
         print_options = PrintLevel().None();
      }
   }
#endif

   print_level = derived_print_level;
}

IterativeSolver::PrintLevel IterativeSolver::FromLegacyPrintLevel(
   int print_level_)
{
#ifdef MFEM_USE_MPI
   int rank = 0;
   if (comm != MPI_COMM_NULL)
   {
      MPI_Comm_rank(comm, &rank);
   }
#endif

   switch (print_level_)
   {
      case -1:
         return PrintLevel();
      case 0:
         return PrintLevel().Errors().Warnings();
      case 1:
         return PrintLevel().Errors().Warnings().Iterations();
      case 2:
         return PrintLevel().Errors().Warnings().Summary();
      case 3:
         return PrintLevel().Errors().Warnings().FirstAndLast();
      default:
#ifdef MFEM_USE_MPI
         if (rank == 0)
#endif
         {
            MFEM_WARNING("Unknown print level " << print_level_ <<
                         ". Defaulting to level 0.");
         }
         return PrintLevel().Errors().Warnings();
   }
}

int IterativeSolver::GuessLegacyPrintLevel(PrintLevel print_options_)
{
   if (print_options_.iterations)
   {
      return 1;
   }
   else if (print_options_.first_and_last)
   {
      return 3;
   }
   else if (print_options_.summary)
   {
      return 2;
   }
   else if (print_options_.errors && print_options_.warnings)
   {
      return 0;
   }
   else
   {
      return -1;
   }
}

void IterativeSolver::SetPreconditioner(Solver &pr)
{
   prec = &pr;
   prec->iterative_mode = false;
}

void IterativeSolver::SetOperator(const Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   if (prec)
   {
      prec->SetOperator(*oper);
   }
}

void IterativeSolver::Monitor(int it, real_t norm, const Vector& r,
                              const Vector& x, bool final) const
{
   if (monitor != nullptr)
   {
      monitor->MonitorResidual(it, norm, r, final);
      monitor->MonitorSolution(it, norm, x, final);
   }
}

OperatorJacobiSmoother::OperatorJacobiSmoother(const real_t dmpng)
   : damping(dmpng),
     ess_tdof_list(nullptr),
     oper(nullptr),
     allow_updates(true)
{ }

OperatorJacobiSmoother::OperatorJacobiSmoother(const BilinearForm &a,
                                               const Array<int> &ess_tdofs,
                                               const real_t dmpng)
   :
   Solver(a.FESpace()->GetTrueVSize()),
   dinv(height),
   damping(dmpng),
   ess_tdof_list(&ess_tdofs),
   residual(height),
   allow_updates(false)
{
   Vector &diag(residual);
   a.AssembleDiagonal(diag);
   // 'a' cannot be used for iterative_mode == true because its size may be
   // different.
   oper = nullptr;
   Setup(diag);
}

OperatorJacobiSmoother::OperatorJacobiSmoother(const Vector &d,
                                               const Array<int> &ess_tdofs,
                                               const real_t dmpng)
   :
   Solver(d.Size()),
   dinv(height),
   damping(dmpng),
   ess_tdof_list(&ess_tdofs),
   residual(height),
   oper(NULL),
   allow_updates(false)
{
   Setup(d);
}

void OperatorJacobiSmoother::SetOperator(const Operator &op)
{
   if (!allow_updates)
   {
      // original behavior of this method
      oper = &op; return;
   }

   // Treat (Par)BilinearForm objects as a special case since their
   // AssembleDiagonal method returns the true-dof diagonal whereas the form
   // itself may act as an ldof operator. This is for compatibility with the
   // constructor that takes a BilinearForm parameter.
   const BilinearForm *blf = dynamic_cast<const BilinearForm *>(&op);
   if (blf)
   {
      // 'a' cannot be used for iterative_mode == true because its size may be
      // different.
      oper = nullptr;
      height = width = blf->FESpace()->GetTrueVSize();
   }
   else
   {
      oper = &op;
      height = op.Height();
      width = op.Width();
      MFEM_VERIFY(height == width, "not a square matrix!");
      // ess_tdof_list is only used with BilinearForm
      ess_tdof_list = nullptr;
   }
   dinv.SetSize(height);
   residual.SetSize(height);
   Vector &diag(residual);
   op.AssembleDiagonal(diag);
   Setup(diag);
}

void OperatorJacobiSmoother::Setup(const Vector &diag)
{
   residual.UseDevice(true);
   const real_t delta = damping;
   auto D = diag.Read();
   auto DI = dinv.Write();
   const bool use_abs_diag_ = use_abs_diag;
   mfem::forall(height, [=] MFEM_HOST_DEVICE (int i)
   {
      if (D[i] == 0.0)
      {
         MFEM_ABORT_KERNEL("Zero diagonal entry in OperatorJacobiSmoother");
      }
      if (!use_abs_diag_) { DI[i] = delta / D[i]; }
      else                { DI[i] = delta / std::abs(D[i]); }
   });
   if (ess_tdof_list && ess_tdof_list->Size() > 0)
   {
      auto I = ess_tdof_list->Read();
      mfem::forall(ess_tdof_list->Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         DI[I[i]] = delta;
      });
   }
}

void OperatorJacobiSmoother::Mult(const Vector &x, Vector &y) const
{
   // For empty MPI ranks, height may be 0:
   // MFEM_VERIFY(Height() > 0, "The diagonal hasn't been computed.");
   MFEM_VERIFY(x.Size() == Width(), "invalid input vector");
   MFEM_VERIFY(y.Size() == Height(), "invalid output vector");

   if (iterative_mode)
   {
      MFEM_VERIFY(oper, "iterative_mode == true requires the forward operator");
      oper->Mult(y, residual);  // r = A y
      subtract(x, residual, residual); // r = x - A y
   }
   else
   {
      residual = x;
      y.UseDevice(true);
      y = 0.0;
   }
   auto DI = dinv.Read();
   auto R = residual.Read();
   auto Y = y.ReadWrite();
   mfem::forall(height, [=] MFEM_HOST_DEVICE (int i)
   {
      Y[i] += DI[i] * R[i];
   });
}

OperatorChebyshevSmoother::OperatorChebyshevSmoother(const Operator &oper_,
                                                     const Vector &d,
                                                     const Array<int>& ess_tdofs,
                                                     int order_, real_t max_eig_estimate_)
   :
   Solver(d.Size()),
   order(order_),
   max_eig_estimate(max_eig_estimate_),
   N(d.Size()),
   dinv(N),
   diag(d),
   coeffs(order),
   ess_tdof_list(ess_tdofs),
   residual(N),
   oper(&oper_) { Setup(); }

#ifdef MFEM_USE_MPI
OperatorChebyshevSmoother::OperatorChebyshevSmoother(const Operator &oper_,
                                                     const Vector &d,
                                                     const Array<int>& ess_tdofs,
                                                     int order_, MPI_Comm comm, int power_iterations, real_t power_tolerance)
#else
OperatorChebyshevSmoother::OperatorChebyshevSmoother(const Operator &oper_,
                                                     const Vector &d,
                                                     const Array<int>& ess_tdofs,
                                                     int order_, int power_iterations, real_t power_tolerance)
#endif
   : Solver(d.Size()),
     order(order_),
     N(d.Size()),
     dinv(N),
     diag(d),
     coeffs(order),
     ess_tdof_list(ess_tdofs),
     residual(N),
     oper(&oper_)
{
   OperatorJacobiSmoother invDiagOperator(diag, ess_tdofs, 1.0);
   ProductOperator diagPrecond(&invDiagOperator, oper, false, false);

#ifdef MFEM_USE_MPI
   PowerMethod powerMethod(comm);
#else
   PowerMethod powerMethod;
#endif
   Vector ev(oper->Width());
   max_eig_estimate = powerMethod.EstimateLargestEigenvalue(diagPrecond, ev,
                                                            power_iterations, power_tolerance);

   Setup();
}

OperatorChebyshevSmoother::OperatorChebyshevSmoother(const Operator* oper_,
                                                     const Vector &d,
                                                     const Array<int>& ess_tdofs,
                                                     int order_, real_t max_eig_estimate_)
   : OperatorChebyshevSmoother(*oper_, d, ess_tdofs, order_, max_eig_estimate_) { }

#ifdef MFEM_USE_MPI
OperatorChebyshevSmoother::OperatorChebyshevSmoother(const Operator* oper_,
                                                     const Vector &d,
                                                     const Array<int>& ess_tdofs,
                                                     int order_, MPI_Comm comm, int power_iterations, real_t power_tolerance)
   : OperatorChebyshevSmoother(*oper_, d, ess_tdofs, order_, comm,
                               power_iterations, power_tolerance) { }
#else
OperatorChebyshevSmoother::OperatorChebyshevSmoother(const Operator* oper_,
                                                     const Vector &d,
                                                     const Array<int>& ess_tdofs,
                                                     int order_, int power_iterations, real_t power_tolerance)
   : OperatorChebyshevSmoother(*oper_, d, ess_tdofs, order_, power_iterations,
                               power_tolerance) { }
#endif

void OperatorChebyshevSmoother::Setup()
{
   // Invert diagonal
   residual.UseDevice(true);
   auto D = diag.Read();
   auto X = dinv.Write();
   mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { X[i] = 1.0 / D[i]; });
   auto I = ess_tdof_list.Read();
   mfem::forall(ess_tdof_list.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      X[I[i]] = 1.0;
   });

   // Set up Chebyshev coefficients
   // For reference, see e.g., Parallel multigrid smoothing: polynomial versus
   // Gauss-Seidel by Adams et al.
   real_t upper_bound = 1.2 * max_eig_estimate;
   real_t lower_bound = 0.3 * max_eig_estimate;
   real_t theta = 0.5 * (upper_bound + lower_bound);
   real_t delta = 0.5 * (upper_bound - lower_bound);

   switch (order-1)
   {
      case 0:
      {
         coeffs[0] = 1.0 / theta;
         break;
      }
      case 1:
      {
         real_t tmp_0 = 1.0/(pow(delta, 2) - 2*pow(theta, 2));
         coeffs[0] = -4*theta*tmp_0;
         coeffs[1] = 2*tmp_0;
         break;
      }
      case 2:
      {
         real_t tmp_0 = 3*pow(delta, 2);
         real_t tmp_1 = pow(theta, 2);
         real_t tmp_2 = 1.0/(-4*pow(theta, 3) + theta*tmp_0);
         coeffs[0] = tmp_2*(tmp_0 - 12*tmp_1);
         coeffs[1] = 12/(tmp_0 - 4*tmp_1);
         coeffs[2] = -4*tmp_2;
         break;
      }
      case 3:
      {
         real_t tmp_0 = pow(delta, 2);
         real_t tmp_1 = pow(theta, 2);
         real_t tmp_2 = 8*tmp_0;
         real_t tmp_3 = 1.0/(pow(delta, 4) + 8*pow(theta, 4) - tmp_1*tmp_2);
         coeffs[0] = tmp_3*(32*pow(theta, 3) - 16*theta*tmp_0);
         coeffs[1] = tmp_3*(-48*tmp_1 + tmp_2);
         coeffs[2] = 32*theta*tmp_3;
         coeffs[3] = -8*tmp_3;
         break;
      }
      case 4:
      {
         real_t tmp_0 = 5*pow(delta, 4);
         real_t tmp_1 = pow(theta, 4);
         real_t tmp_2 = pow(theta, 2);
         real_t tmp_3 = pow(delta, 2);
         real_t tmp_4 = 60*tmp_3;
         real_t tmp_5 = 20*tmp_3;
         real_t tmp_6 = 1.0/(16*pow(theta, 5) - pow(theta, 3)*tmp_5 + theta*tmp_0);
         real_t tmp_7 = 160*tmp_2;
         real_t tmp_8 = 1.0/(tmp_0 + 16*tmp_1 - tmp_2*tmp_5);
         coeffs[0] = tmp_6*(tmp_0 + 80*tmp_1 - tmp_2*tmp_4);
         coeffs[1] = tmp_8*(tmp_4 - tmp_7);
         coeffs[2] = tmp_6*(-tmp_5 + tmp_7);
         coeffs[3] = -80*tmp_8;
         coeffs[4] = 16*tmp_6;
         break;
      }
      default:
         MFEM_ABORT("Chebyshev smoother not implemented for order = " << order);
   }
}

void OperatorChebyshevSmoother::Mult(const Vector& x, Vector &y) const
{
   if (iterative_mode)
   {
      MFEM_ABORT("Chebyshev smoother not implemented for iterative mode");
   }

   if (!oper)
   {
      MFEM_ABORT("Chebyshev smoother requires operator");
   }

   residual = x;
   helperVector.SetSize(x.Size());

   y.UseDevice(true);
   y = 0.0;

   for (int k = 0; k < order; ++k)
   {
      // Apply
      if (k > 0)
      {
         oper->Mult(residual, helperVector);
         residual = helperVector;
      }

      // Scale residual by inverse diagonal
      const int n = N;
      auto Dinv = dinv.Read();
      auto R = residual.ReadWrite();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { R[i] *= Dinv[i]; });

      // Add weighted contribution to y
      auto Y = y.ReadWrite();
      auto C = coeffs.Read();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) { Y[i] += C[k] * R[i]; });
   }
}

void SLISolver::UpdateVectors()
{
   r.SetSize(width);
   z.SetSize(width);
}

void SLISolver::Mult(const Vector &b, Vector &x) const
{
   int i;

   // Optimized preconditioned SLI with fixed number of iterations and given
   // initial guess
   if (rel_tol == 0.0 && iterative_mode && prec)
   {
      for (i = 0; i < max_iter; i++)
      {
         oper->Mult(x, r);  // r = A x
         subtract(b, r, r); // r = b - A x
         prec->Mult(r, z);  // z = B r
         add(x, 1.0, z, x); // x = x + B (b - A x)
      }
      converged = true;
      final_iter = i;
      return;
   }

   // Optimized preconditioned SLI with fixed number of iterations and zero
   // initial guess
   if (rel_tol == 0.0 && !iterative_mode && prec)
   {
      prec->Mult(b, x);     // x = B b (initial guess 0)
      for (i = 1; i < max_iter; i++)
      {
         oper->Mult(x, r);  // r = A x
         subtract(b, r, r); // r = b - A x
         prec->Mult(r, z);  // z = B r
         add(x, 1.0, z, x); // x = x + B (b - A x)
      }
      converged = true;
      final_iter = i;
      return;
   }

   // General version of SLI with a relative tolerance, optional preconditioner
   // and optional initial guess
   real_t r0, nom, nom0, nomold = 1, cf;

   if (iterative_mode)
   {
      oper->Mult(x, r);
      subtract(b, r, r); // r = b - A x
   }
   else
   {
      r = b;
      x = 0.0;
   }

   if (prec)
   {
      prec->Mult(r, z); // z = B r
      nom0 = nom = sqrt(Dot(z, z));
   }
   else
   {
      nom0 = nom = sqrt(Dot(r, r));
   }
   initial_norm = nom0;

   if (print_options.iterations || print_options.first_and_last)
   {
      mfem::out << "   Iteration : " << setw(3) << right << 0 << "  ||Br|| = "
                << nom << (print_options.first_and_last ? " ..." : "") << '\n';
   }

   r0 = std::max(nom*rel_tol, abs_tol);
   if (nom <= r0)
   {
      converged = true;
      final_iter = 0;
      final_norm = nom;
      return;
   }

   // start iteration
   converged = false;
   final_iter = max_iter;
   for (i = 1; true; )
   {
      if (prec) //  x = x + B (b - A x)
      {
         add(x, 1.0, z, x);
      }
      else
      {
         add(x, 1.0, r, x);
      }

      oper->Mult(x, r);
      subtract(b, r, r); // r = b - A x

      if (prec)
      {
         prec->Mult(r, z); //  z = B r
         nom = sqrt(Dot(z, z));
      }
      else
      {
         nom = sqrt(Dot(r, r));
      }

      cf = nom/nomold;
      nomold = nom;

      bool done = false;
      if (nom < r0)
      {
         converged = true;
         final_iter = i;
         done = true;
      }

      if (++i > max_iter)
      {
         done = true;
      }

      if (print_options.iterations || (done && print_options.first_and_last))
      {
         mfem::out << "   Iteration : " << setw(3) << right << (i-1)
                   << "  ||Br|| = " << setw(11) << left << nom
                   << "\tConv. rate: " << cf << '\n';
      }

      if (done) { break; }
   }

   if (print_options.summary || (print_options.warnings && !converged))
   {
      const auto rf = pow (nom/nom0, 1.0/final_iter);
      mfem::out << "SLI: Number of iterations: " << final_iter << '\n'
                << "Conv. rate: " << cf << '\n'
                << "Average reduction factor: "<< rf << '\n';
   }
   if (print_options.warnings && !converged)
   {
      mfem::out << "SLI: No convergence!" << '\n';
   }

   final_norm = nom;
}

void SLI(const Operator &A, const Vector &b, Vector &x,
         int print_iter, int max_num_iter,
         real_t RTOLERANCE, real_t ATOLERANCE)
{
   MFEM_PERF_FUNCTION;

   SLISolver sli;
   sli.SetPrintLevel(print_iter);
   sli.SetMaxIter(max_num_iter);
   sli.SetRelTol(sqrt(RTOLERANCE));
   sli.SetAbsTol(sqrt(ATOLERANCE));
   sli.SetOperator(A);
   sli.Mult(b, x);
}

void SLI(const Operator &A, Solver &B, const Vector &b, Vector &x,
         int print_iter, int max_num_iter,
         real_t RTOLERANCE, real_t ATOLERANCE)
{
   MFEM_PERF_FUNCTION;

   SLISolver sli;
   sli.SetPrintLevel(print_iter);
   sli.SetMaxIter(max_num_iter);
   sli.SetRelTol(sqrt(RTOLERANCE));
   sli.SetAbsTol(sqrt(ATOLERANCE));
   sli.SetOperator(A);
   sli.SetPreconditioner(B);
   sli.Mult(b, x);
}


void CGSolver::UpdateVectors()
{
   MemoryType mt = GetMemoryType(oper->GetMemoryClass());

   r.SetSize(width, mt); r.UseDevice(true);
   d.SetSize(width, mt); d.UseDevice(true);
   z.SetSize(width, mt); z.UseDevice(true);
}

void CGSolver::Mult(const Vector &b, Vector &x) const
{
   int i;
   real_t r0, den, nom, nom0, betanom, alpha, beta;

   x.UseDevice(true);
   if (iterative_mode)
   {
      oper->Mult(x, r);
      subtract(b, r, r); // r = b - A x
   }
   else
   {
      r = b;
      x = 0.0;
   }

   if (prec)
   {
      prec->Mult(r, z); // z = B r
      d = z;
   }
   else
   {
      d = r;
   }
   nom0 = nom = Dot(d, r);
   if (nom0 >= 0.0) { initial_norm = sqrt(nom0); }
   MFEM_VERIFY(IsFinite(nom), "nom = " << nom);
   if (print_options.iterations || print_options.first_and_last)
   {
      mfem::out << "   Iteration : " << setw(3) << 0 << "  (B r, r) = "
                << nom << (print_options.first_and_last ? " ...\n" : "\n");
   }
   Monitor(0, nom, r, x);

   if (nom < 0.0)
   {
      if (print_options.warnings)
      {
         mfem::out << "PCG: The preconditioner is not positive definite. (Br, r) = "
                   << nom << '\n';
      }
      converged = false;
      final_iter = 0;
      initial_norm = nom;
      final_norm = nom;
      return;
   }
   r0 = std::max(nom*rel_tol*rel_tol, abs_tol*abs_tol);
   if (nom <= r0)
   {
      converged = true;
      final_iter = 0;
      final_norm = sqrt(nom);
      return;
   }

   oper->Mult(d, z);  // z = A d
   den = Dot(z, d);
   MFEM_VERIFY(IsFinite(den), "den = " << den);
   if (den <= 0.0)
   {
      if (Dot(d, d) > 0.0 && print_options.warnings)
      {
         mfem::out << "PCG: The operator is not positive definite. (Ad, d) = "
                   << den << '\n';
      }
      if (den == 0.0)
      {
         converged = false;
         final_iter = 0;
         final_norm = sqrt(nom);
         return;
      }
   }

   // start iteration
   converged = false;
   final_iter = max_iter;
   for (i = 1; true; )
   {
      alpha = nom/den;
      add(x,  alpha, d, x);     //  x = x + alpha d
      add(r, -alpha, z, r);     //  r = r - alpha A d

      if (prec)
      {
         prec->Mult(r, z);      //  z = B r
         betanom = Dot(r, z);
      }
      else
      {
         betanom = Dot(r, r);
      }
      MFEM_VERIFY(IsFinite(betanom), "betanom = " << betanom);
      if (betanom < 0.0)
      {
         if (print_options.warnings)
         {
            mfem::out << "PCG: The preconditioner is not positive definite. (Br, r) = "
                      << betanom << '\n';
         }
         converged = false;
         final_iter = i;
         break;
      }

      if (print_options.iterations)
      {
         mfem::out << "   Iteration : " << setw(3) << i << "  (B r, r) = "
                   << betanom << std::endl;
      }

      Monitor(i, betanom, r, x);

      if (betanom <= r0)
      {
         converged = true;
         final_iter = i;
         break;
      }

      if (++i > max_iter)
      {
         break;
      }

      beta = betanom/nom;
      if (prec)
      {
         add(z, beta, d, d);   //  d = z + beta d
      }
      else
      {
         add(r, beta, d, d);
      }
      oper->Mult(d, z);       //  z = A d
      den = Dot(d, z);
      MFEM_VERIFY(IsFinite(den), "den = " << den);
      if (den <= 0.0)
      {
         if (Dot(d, d) > 0.0 && print_options.warnings)
         {
            mfem::out << "PCG: The operator is not positive definite. (Ad, d) = "
                      << den << '\n';
         }
         if (den == 0.0)
         {
            final_iter = i;
            break;
         }
      }
      nom = betanom;
   }
   if (print_options.first_and_last && !print_options.iterations)
   {
      mfem::out << "   Iteration : " << setw(3) << final_iter << "  (B r, r) = "
                << betanom << '\n';
   }
   if (print_options.summary || (print_options.warnings && !converged))
   {
      mfem::out << "PCG: Number of iterations: " << final_iter << '\n';
   }
   if (print_options.summary || print_options.iterations ||
       print_options.first_and_last)
   {
      const auto arf = pow (betanom/nom0, 0.5/final_iter);
      mfem::out << "Average reduction factor = " << arf << '\n';
   }
   if (print_options.warnings && !converged)
   {
      mfem::out << "PCG: No convergence!" << '\n';
   }

   final_norm = sqrt(betanom);

   Monitor(final_iter, final_norm, r, x, true);
}

void CG(const Operator &A, const Vector &b, Vector &x,
        int print_iter, int max_num_iter,
        real_t RTOLERANCE, real_t ATOLERANCE)
{
   MFEM_PERF_FUNCTION;

   CGSolver cg;
   cg.SetPrintLevel(print_iter);
   cg.SetMaxIter(max_num_iter);
   cg.SetRelTol(sqrt(RTOLERANCE));
   cg.SetAbsTol(sqrt(ATOLERANCE));
   cg.SetOperator(A);
   cg.Mult(b, x);
}

void PCG(const Operator &A, Solver &B, const Vector &b, Vector &x,
         int print_iter, int max_num_iter,
         real_t RTOLERANCE, real_t ATOLERANCE)
{
   MFEM_PERF_FUNCTION;

   CGSolver pcg;
   pcg.SetPrintLevel(print_iter);
   pcg.SetMaxIter(max_num_iter);
   pcg.SetRelTol(sqrt(RTOLERANCE));
   pcg.SetAbsTol(sqrt(ATOLERANCE));
   pcg.SetOperator(A);
   pcg.SetPreconditioner(B);
   pcg.Mult(b, x);
}


inline void GeneratePlaneRotation(real_t &dx, real_t &dy,
                                  real_t &cs, real_t &sn)
{
   if (dy == 0.0)
   {
      cs = 1.0;
      sn = 0.0;
   }
   else if (fabs(dy) > fabs(dx))
   {
      real_t temp = dx / dy;
      sn = 1.0 / sqrt( 1.0 + temp*temp );
      cs = temp * sn;
   }
   else
   {
      real_t temp = dy / dx;
      cs = 1.0 / sqrt( 1.0 + temp*temp );
      sn = temp * cs;
   }
}

inline void ApplyPlaneRotation(real_t &dx, real_t &dy, real_t &cs, real_t &sn)
{
   real_t temp = cs * dx + sn * dy;
   dy = -sn * dx + cs * dy;
   dx = temp;
}

inline void Update(Vector &x, int k, DenseMatrix &h, Vector &s,
                   Array<Vector*> &v)
{
   Vector y(s);

   // Backsolve:
   for (int i = k; i >= 0; i--)
   {
      y(i) /= h(i,i);
      for (int j = i - 1; j >= 0; j--)
      {
         y(j) -= h(j,i) * y(i);
      }
   }

   for (int j = 0; j <= k; j++)
   {
      x.Add(y(j), *v[j]);
   }
}

void GMRESSolver::Mult(const Vector &b, Vector &x) const
{
   // Generalized Minimum Residual method following the algorithm
   // on p. 20 of the SIAM Templates book.

   int n = width;

   DenseMatrix H(m+1, m);
   Vector s(m+1), cs(m+1), sn(m+1);
   Vector r(n), w(n);
   Array<Vector *> v;

   int i, j, k;

   if (iterative_mode)
   {
      oper->Mult(x, r);
   }
   else
   {
      x = 0.0;
   }

   if (prec)
   {
      if (iterative_mode)
      {
         subtract(b, r, w);
         prec->Mult(w, r);    // r = M (b - A x)
      }
      else
      {
         prec->Mult(b, r);
      }
   }
   else
   {
      if (iterative_mode)
      {
         subtract(b, r, r);
      }
      else
      {
         r = b;
      }
   }
   real_t beta = initial_norm = Norm(r);  // beta = ||r||
   MFEM_VERIFY(IsFinite(beta), "beta = " << beta);

   final_norm = std::max(rel_tol*beta, abs_tol);

   if (beta <= final_norm)
   {
      final_norm = beta;
      final_iter = 0;
      converged = true;
      j = 0;
      goto finish;
   }

   if (print_options.iterations || print_options.first_and_last)
   {
      mfem::out << "   Pass : " << setw(2) << 1
                << "   Iteration : " << setw(3) << 0
                << "  ||B r|| = " << beta
                << (print_options.first_and_last ? " ...\n" : "\n");
   }

   Monitor(0, beta, r, x);

   v.SetSize(m+1, NULL);

   for (j = 1; j <= max_iter; )
   {
      if (v[0] == NULL) { v[0] = new Vector(n); }
      v[0]->Set(1.0/beta, r);
      s = 0.0; s(0) = beta;

      for (i = 0; i < m && j <= max_iter; i++, j++)
      {
         if (prec)
         {
            oper->Mult(*v[i], r);
            prec->Mult(r, w);        // w = M A v[i]
         }
         else
         {
            oper->Mult(*v[i], w);
         }

         for (k = 0; k <= i; k++)
         {
            H(k,i) = Dot(w, *v[k]);  // H(k,i) = w * v[k]
            w.Add(-H(k,i), *v[k]);   // w -= H(k,i) * v[k]
         }

         H(i+1,i) = Norm(w);           // H(i+1,i) = ||w||
         MFEM_VERIFY(IsFinite(H(i+1,i)), "Norm(w) = " << H(i+1,i));
         if (v[i+1] == NULL) { v[i+1] = new Vector(n); }
         v[i+1]->Set(1.0/H(i+1,i), w); // v[i+1] = w / H(i+1,i)

         for (k = 0; k < i; k++)
         {
            ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k));
         }

         GeneratePlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));

         const real_t resid = fabs(s(i+1));
         MFEM_VERIFY(IsFinite(resid), "resid = " << resid);

         if (resid <= final_norm)
         {
            Update(x, i, H, s, v);
            final_norm = resid;
            final_iter = j;
            converged = true;
            goto finish;
         }

         if (print_options.iterations)
         {
            mfem::out << "   Pass : " << setw(2) << (j-1)/m+1
                      << "   Iteration : " << setw(3) << j
                      << "  ||B r|| = " << resid << '\n';
         }

         Monitor(j, resid, r, x);
      }

      if (print_options.iterations && j <= max_iter)
      {
         mfem::out << "Restarting..." << '\n';
      }

      Update(x, i-1, H, s, v);

      oper->Mult(x, r);
      if (prec)
      {
         subtract(b, r, w);
         prec->Mult(w, r);    // r = M (b - A x)
      }
      else
      {
         subtract(b, r, r);
      }
      beta = Norm(r);         // beta = ||r||
      MFEM_VERIFY(IsFinite(beta), "beta = " << beta);
      if (beta <= final_norm)
      {
         final_norm = beta;
         final_iter = j;
         converged = true;
         goto finish;
      }
   }

   final_norm = beta;
   final_iter = max_iter;
   converged = false;

finish:
   if ((print_options.iterations && converged) || print_options.first_and_last)
   {
      mfem::out << "   Pass : " << setw(2) << (j-1)/m+1
                << "   Iteration : " << setw(3) << final_iter
                << "  ||B r|| = " << final_norm << '\n';
   }
   if (print_options.summary || (print_options.warnings && !converged))
   {
      mfem::out << "GMRES: Number of iterations: " << final_iter << '\n';
   }
   if (print_options.warnings && !converged)
   {
      mfem::out << "GMRES: No convergence!\n";
   }

   Monitor(final_iter, final_norm, r, x, true);

   for (i = 0; i < v.Size(); i++)
   {
      delete v[i];
   }
}

void FGMRESSolver::Mult(const Vector &b, Vector &x) const
{
   DenseMatrix H(m+1,m);
   Vector s(m+1), cs(m+1), sn(m+1);
   Vector r(b.Size());

   int i, j, k;

   if (iterative_mode)
   {
      oper->Mult(x, r);
      subtract(b,r,r);
   }
   else
   {
      x = 0.;
      r = b;
   }
   real_t beta = initial_norm = Norm(r);  // beta = ||r||
   MFEM_VERIFY(IsFinite(beta), "beta = " << beta);

   final_norm = std::max(rel_tol*beta, abs_tol);

   converged = false;

   if (beta <= final_norm)
   {
      final_norm = beta;
      final_iter = 0;
      converged = true;
      return;
   }

   if (print_options.iterations || print_options.first_and_last)
   {
      mfem::out << "   Pass : " << setw(2) << 1
                << "   Iteration : " << setw(3) << 0
                << "  || r || = " << beta
                << (print_options.first_and_last ? " ...\n" : "\n");
   }

   Monitor(0, beta, r, x);

   Array<Vector*> v(m+1);
   Array<Vector*> z(m+1);
   for (i= 0; i<=m; i++)
   {
      v[i] = NULL;
      z[i] = NULL;
   }

   j = 1;
   while (j <= max_iter)
   {
      if (v[0] == NULL) { v[0] = new Vector(b.Size()); }
      (*v[0]) = 0.0;
      v[0] -> Add (1.0/beta, r);   // v[0] = r / ||r||
      s = 0.0; s(0) = beta;

      for (i = 0; i < m && j <= max_iter; i++, j++)
      {

         if (z[i] == NULL) { z[i] = new Vector(b.Size()); }
         (*z[i]) = 0.0;

         if (prec)
         {
            prec->Mult(*v[i], *z[i]);
         }
         else
         {
            (*z[i]) = (*v[i]);
         }
         oper->Mult(*z[i], r);

         for (k = 0; k <= i; k++)
         {
            H(k,i) = Dot( r, *v[k]); // H(k,i) = r * v[k]
            r.Add(-H(k,i), (*v[k])); // r -= H(k,i) * v[k]
         }

         H(i+1,i)  = Norm(r);       // H(i+1,i) = ||r||
         if (v[i+1] == NULL) { v[i+1] = new Vector(b.Size()); }
         (*v[i+1]) = 0.0;
         v[i+1] -> Add (1.0/H(i+1,i), r); // v[i+1] = r / H(i+1,i)

         for (k = 0; k < i; k++)
         {
            ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k));
         }

         GeneratePlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));

         const real_t resid = fabs(s(i+1));
         MFEM_VERIFY(IsFinite(resid), "resid = " << resid);
         if (print_options.iterations || (print_options.first_and_last &&
                                          resid <= final_norm))
         {
            mfem::out << "   Pass : " << setw(2) << (j-1)/m+1
                      << "   Iteration : " << setw(3) << j
                      << "  || r || = " << resid << endl;
         }
         Monitor(j, resid, r, x, resid <= final_norm);

         if (resid <= final_norm)
         {
            Update(x, i, H, s, z);
            final_norm = resid;
            final_iter = j;
            converged = true;

            if (print_options.summary)
            {
               mfem::out << "FGMRES: Number of iterations: " << final_iter << '\n';
            }

            for (i= 0; i<=m; i++)
            {
               if (v[i]) { delete v[i]; }
               if (z[i]) { delete z[i]; }
            }
            return;
         }
      }

      if (print_options.iterations)
      {
         mfem::out << "Restarting..." << endl;
      }

      Update(x, i-1, H, s, z);

      oper->Mult(x, r);
      subtract(b,r,r);
      beta = Norm(r);
      MFEM_VERIFY(IsFinite(beta), "beta = " << beta);
      if (beta <= final_norm)
      {
         converged = true;

         break;
      }
   }

   // Clean buffers up
   for (i = 0; i <= m; i++)
   {
      if (v[i]) { delete v[i]; }
      if (z[i]) { delete z[i]; }
   }

   final_norm = beta;
   final_iter = converged ? j : max_iter;

   // Note: j is off by one when we arrive here
   if (!print_options.iterations && print_options.first_and_last)
   {
      mfem::out << "   Pass : " << setw(2) << (j-1)/m+1
                << "   Iteration : " << setw(3) << j-1
                << "  || r || = " << final_norm << endl;
   }
   if (print_options.summary || (print_options.warnings && !converged))
   {
      mfem::out << "FGMRES: Number of iterations: " << final_iter << '\n';
   }
   if (print_options.warnings && !converged)
   {
      mfem::out << "FGMRES: No convergence!\n";
   }
}


int GMRES(const Operator &A, Vector &x, const Vector &b, Solver &M,
          int &max_iter, int m, real_t &tol, real_t atol, int printit)
{
   MFEM_PERF_FUNCTION;

   GMRESSolver gmres;
   gmres.SetPrintLevel(printit);
   gmres.SetMaxIter(max_iter);
   gmres.SetKDim(m);
   gmres.SetRelTol(sqrt(tol));
   gmres.SetAbsTol(sqrt(atol));
   gmres.SetOperator(A);
   gmres.SetPreconditioner(M);
   gmres.Mult(b, x);
   max_iter = gmres.GetNumIterations();
   tol = gmres.GetFinalNorm()*gmres.GetFinalNorm();
   return gmres.GetConverged();
}

void GMRES(const Operator &A, Solver &B, const Vector &b, Vector &x,
           int print_iter, int max_num_iter, int m, real_t rtol, real_t atol)
{
   GMRES(A, x, b, B, max_num_iter, m, rtol, atol, print_iter);
}


void BiCGSTABSolver::UpdateVectors()
{
   p.SetSize(width);
   phat.SetSize(width);
   s.SetSize(width);
   shat.SetSize(width);
   t.SetSize(width);
   v.SetSize(width);
   r.SetSize(width);
   rtilde.SetSize(width);
}

void BiCGSTABSolver::Mult(const Vector &b, Vector &x) const
{
   // BiConjugate Gradient Stabilized method following the algorithm
   // on p. 27 of the SIAM Templates book.

   int i;
   real_t resid, tol_goal;
   real_t rho_1, rho_2=1.0, alpha=1.0, beta, omega=1.0;

   if (iterative_mode)
   {
      oper->Mult(x, r);
      subtract(b, r, r); // r = b - A x
   }
   else
   {
      x = 0.0;
      r = b;
   }
   rtilde = r;

   resid = initial_norm = Norm(r);
   MFEM_VERIFY(IsFinite(resid), "resid = " << resid);
   if (print_options.iterations || print_options.first_and_last)
   {
      mfem::out << "   Iteration : " << setw(3) << 0
                << "   ||r|| = " << resid << (print_options.first_and_last ? " ...\n" : "\n");
   }

   Monitor(0, resid, r, x);

   tol_goal = std::max(resid*rel_tol, abs_tol);

   if (resid <= tol_goal)
   {
      final_norm = resid;
      final_iter = 0;
      converged = true;
      return;
   }

   for (i = 1; i <= max_iter; i++)
   {
      rho_1 = Dot(rtilde, r);
      if (rho_1 == 0)
      {
         if (print_options.iterations || print_options.first_and_last)
         {
            mfem::out << "   Iteration : " << setw(3) << i
                      << "   ||r|| = " << resid << '\n';
         }

         Monitor(i, resid, r, x);

         final_norm = resid;
         final_iter = i;
         converged = false;
         if (print_options.summary || (print_options.warnings && !converged))
         {
            mfem::out << "BiCGStab: Number of iterations: " << final_iter << '\n';
         }
         if (print_options.warnings)
         {
            mfem::out << "BiCGStab: No convergence!\n";
         }
         return;
      }
      if (i == 1)
      {
         p = r;
      }
      else
      {
         beta = (rho_1/rho_2) * (alpha/omega);
         add(p, -omega, v, p);  //  p = p - omega * v
         add(r, beta, p, p);    //  p = r + beta * p
      }
      if (prec)
      {
         prec->Mult(p, phat);   //  phat = M^{-1} * p
      }
      else
      {
         phat = p;
      }
      oper->Mult(phat, v);     //  v = A * phat
      alpha = rho_1 / Dot(rtilde, v);
      add(r, -alpha, v, s); //  s = r - alpha * v
      resid = Norm(s);
      MFEM_VERIFY(IsFinite(resid), "resid = " << resid);
      if (resid < tol_goal)
      {
         x.Add(alpha, phat);  //  x = x + alpha * phat
         if (print_options.iterations || print_options.first_and_last)
         {
            mfem::out << "   Iteration : " << setw(3) << i
                      << "   ||s|| = " << resid << '\n';
         }
         final_norm = resid;
         final_iter = i;
         converged = true;
         if (print_options.summary || (print_options.warnings && !converged))
         {
            mfem::out << "BiCGStab: Number of iterations: " << final_iter << '\n';
         }
         return;
      }
      if (print_options.iterations)
      {
         mfem::out << "   Iteration : " << setw(3) << i
                   << "   ||s|| = " << resid;
      }
      Monitor(i, resid, r, x);
      if (prec)
      {
         prec->Mult(s, shat);  //  shat = M^{-1} * s
      }
      else
      {
         shat = s;
      }
      oper->Mult(shat, t);     //  t = A * shat
      omega = Dot(t, s) / Dot(t, t);
      x.Add(alpha, phat);   //  x += alpha * phat
      x.Add(omega, shat);   //  x += omega * shat
      add(s, -omega, t, r); //  r = s - omega * t

      rho_2 = rho_1;
      resid = Norm(r);
      MFEM_VERIFY(IsFinite(resid), "resid = " << resid);
      if (print_options.iterations)
      {
         mfem::out << "   ||r|| = " << resid << '\n';
      }
      Monitor(i, resid, r, x);
      if (resid < tol_goal)
      {
         final_norm = resid;
         final_iter = i;
         converged = true;
         if (!print_options.iterations && print_options.first_and_last)
         {
            mfem::out << "   Iteration : " << setw(3) << i
                      << "   ||r|| = " << resid << '\n';
         }
         if (print_options.summary || (print_options.warnings && !converged))
         {
            mfem::out << "BiCGStab: Number of iterations: " << final_iter << '\n';
         }
         return;
      }
      if (omega == 0)
      {
         final_norm = resid;
         final_iter = i;
         converged = false;
         if (!print_options.iterations && print_options.first_and_last)
         {
            mfem::out << "   Iteration : " << setw(3) << i
                      << "   ||r|| = " << resid << '\n';
         }
         if (print_options.summary || (print_options.warnings && !converged))
         {
            mfem::out << "BiCGStab: Number of iterations: " << final_iter << '\n';
         }
         if (print_options.warnings)
         {
            mfem::out << "BiCGStab: No convergence!\n";
         }
         return;
      }
   }

   final_norm = resid;
   final_iter = max_iter;
   converged = false;

   if (!print_options.iterations && print_options.first_and_last)
   {
      mfem::out << "   Iteration : " << setw(3) << final_iter
                << "   ||r|| = " << resid << '\n';
   }
   if (print_options.summary || (print_options.warnings && !converged))
   {
      mfem::out << "BiCGStab: Number of iterations: " << final_iter << '\n';
   }
   if (print_options.warnings)
   {
      mfem::out << "BiCGStab: No convergence!\n";
   }
}

int BiCGSTAB(const Operator &A, Vector &x, const Vector &b, Solver &M,
             int &max_iter, real_t &tol, real_t atol, int printit)
{
   BiCGSTABSolver bicgstab;
   bicgstab.SetPrintLevel(printit);
   bicgstab.SetMaxIter(max_iter);
   bicgstab.SetRelTol(sqrt(tol));
   bicgstab.SetAbsTol(sqrt(atol));
   bicgstab.SetOperator(A);
   bicgstab.SetPreconditioner(M);
   bicgstab.Mult(b, x);
   max_iter = bicgstab.GetNumIterations();
   tol = bicgstab.GetFinalNorm()*bicgstab.GetFinalNorm();
   return bicgstab.GetConverged();
}

void BiCGSTAB(const Operator &A, Solver &B, const Vector &b, Vector &x,
              int print_iter, int max_num_iter, real_t rtol, real_t atol)
{
   BiCGSTAB(A, x, b, B, max_num_iter, rtol, atol, print_iter);
}


void MINRESSolver::SetOperator(const Operator &op)
{
   IterativeSolver::SetOperator(op);
   v0.SetSize(width);
   v1.SetSize(width);
   w0.SetSize(width);
   w1.SetSize(width);
   q.SetSize(width);
   if (prec)
   {
      u1.SetSize(width);
   }

   v0.UseDevice(true);
   v1.UseDevice(true);
   w0.UseDevice(true);
   w1.UseDevice(true);
   q.UseDevice(true);
   u1.UseDevice(true);
}

void MINRESSolver::Mult(const Vector &b, Vector &x) const
{
   // Based on the MINRES algorithm on p. 86, Fig. 6.9 in
   // "Iterative Krylov Methods for Large Linear Systems",
   // by Henk A. van der Vorst, 2003.
   // Extended to support an SPD preconditioner.

   b.UseDevice(true);
   x.UseDevice(true);

   int it;
   real_t beta, eta, gamma0, gamma1, sigma0, sigma1;
   real_t alpha, delta, rho1, rho2, rho3, norm_goal;
   Vector *z = (prec) ? &u1 : &v1;

   converged = true;

   if (!iterative_mode)
   {
      v1 = b;
      x = 0.;
   }
   else
   {
      oper->Mult(x, v1);
      subtract(b, v1, v1);
   }

   if (prec)
   {
      prec->Mult(v1, u1);
   }
   eta = beta = initial_norm = sqrt(Dot(*z, v1));
   MFEM_VERIFY(IsFinite(eta), "eta = " << eta);
   gamma0 = gamma1 = 1.;
   sigma0 = sigma1 = 0.;

   norm_goal = std::max(rel_tol*eta, abs_tol);

   if (eta <= norm_goal)
   {
      it = 0;
      goto loop_end;
   }

   if (print_options.iterations || print_options.first_and_last)
   {
      mfem::out << "MINRES: iteration " << setw(3) << 0 << ": ||r||_B = "
                << eta << (print_options.first_and_last ? " ..." : "") << '\n';
   }
   Monitor(0, eta, *z, x);

   for (it = 1; it <= max_iter; it++)
   {
      v1 /= beta;
      if (prec)
      {
         u1 /= beta;
      }
      oper->Mult(*z, q);
      alpha = Dot(*z, q);
      MFEM_VERIFY(IsFinite(alpha), "alpha = " << alpha);
      if (it > 1) // (v0 == 0) for (it == 1)
      {
         q.Add(-beta, v0);
      }
      add(q, -alpha, v1, v0);

      delta = gamma1*alpha - gamma0*sigma1*beta;
      rho3 = sigma0*beta;
      rho2 = sigma1*alpha + gamma0*gamma1*beta;
      if (!prec)
      {
         beta = Norm(v0);
      }
      else
      {
         prec->Mult(v0, q);
         beta = sqrt(Dot(v0, q));
      }
      MFEM_VERIFY(IsFinite(beta), "beta = " << beta);
      rho1 = std::hypot(delta, beta);

      if (it == 1)
      {
         w0.Set(1./rho1, *z);   // (w0 == 0) and (w1 == 0)
      }
      else if (it == 2)
      {
         add(1./rho1, *z, -rho2/rho1, w1, w0);   // (w0 == 0)
      }
      else
      {
         add(-rho3/rho1, w0, -rho2/rho1, w1, w0);
         w0.Add(1./rho1, *z);
      }

      gamma0 = gamma1;
      gamma1 = delta/rho1;

      x.Add(gamma1*eta, w0);

      sigma0 = sigma1;
      sigma1 = beta/rho1;

      eta = -sigma1*eta;
      MFEM_VERIFY(IsFinite(eta), "eta = " << eta);

      if (fabs(eta) <= norm_goal)
      {
         goto loop_end;
      }

      if (print_options.iterations)
      {
         mfem::out << "MINRES: iteration " << setw(3) << it << ": ||r||_B = "
                   << fabs(eta) << '\n';
      }
      Monitor(it, fabs(eta), *z, x);

      if (prec)
      {
         Swap(u1, q);
      }
      Swap(v0, v1);
      Swap(w0, w1);
   }
   converged = false;
   it--;

loop_end:
   final_iter = it;
   final_norm = fabs(eta);

   if (print_options.iterations || print_options.first_and_last)
   {
      mfem::out << "MINRES: iteration " << setw(3) << it << ": ||r||_B = "
                << fabs(eta) << '\n';
   }

   if (print_options.summary || (!converged && print_options.warnings))
   {
      mfem::out << "MINRES: Number of iterations: " << setw(3) << final_iter << '\n';
   }

   Monitor(final_iter, final_norm, *z, x, true);

   // if (print_options.iteration_details || (!converged && print_options.errors))
   // {
   //    oper->Mult(x, v1);
   //    subtract(b, v1, v1);
   //    if (prec)
   //    {
   //       prec->Mult(v1, u1);
   //    }
   //    eta = sqrt(Dot(*z, v1));
   //    mfem::out << "MINRES: iteration " << setw(3) << it << '\n'
   //              << "   ||r||_B = " << eta << " (re-computed)" << '\n';
   // }

   if (!converged && (print_options.warnings))
   {
      mfem::out << "MINRES: No convergence!\n";
   }
}

void MINRES(const Operator &A, const Vector &b, Vector &x, int print_it,
            int max_it, real_t rtol, real_t atol)
{
   MFEM_PERF_FUNCTION;

   MINRESSolver minres;
   minres.SetPrintLevel(print_it);
   minres.SetMaxIter(max_it);
   minres.SetRelTol(sqrt(rtol));
   minres.SetAbsTol(sqrt(atol));
   minres.SetOperator(A);
   minres.Mult(b, x);
}

void MINRES(const Operator &A, Solver &B, const Vector &b, Vector &x,
            int print_it, int max_it, real_t rtol, real_t atol)
{
   MINRESSolver minres;
   minres.SetPrintLevel(print_it);
   minres.SetMaxIter(max_it);
   minres.SetRelTol(sqrt(rtol));
   minres.SetAbsTol(sqrt(atol));
   minres.SetOperator(A);
   minres.SetPreconditioner(B);
   minres.Mult(b, x);
}


void NewtonSolver::SetOperator(const Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   MFEM_VERIFY(height == width, "square Operator is required.");

   r.SetSize(width);
   c.SetSize(width);
}

void NewtonSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_VERIFY(prec != NULL, "the Solver is not set (use SetSolver).");

   int it;
   real_t norm0, norm, norm_goal;
   const bool have_b = (b.Size() == Height());

   if (!iterative_mode)
   {
      x = 0.0;
   }

   ProcessNewState(x);

   oper->Mult(x, r);
   if (have_b)
   {
      r -= b;
   }

   norm0 = norm = initial_norm = Norm(r);
   if (print_options.first_and_last && !print_options.iterations)
   {
      mfem::out << "Newton iteration " << setw(2) << 0
                << " : ||r|| = " << norm << "...\n";
   }
   norm_goal = std::max(rel_tol*norm, abs_tol);

   prec->iterative_mode = false;

   // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
   for (it = 0; true; it++)
   {
      MFEM_VERIFY(IsFinite(norm), "norm = " << norm);
      if (print_options.iterations)
      {
         mfem::out << "Newton iteration " << setw(2) << it
                   << " : ||r|| = " << norm;
         if (it > 0)
         {
            mfem::out << ", ||r||/||r_0|| = " << norm/norm0;
         }
         mfem::out << '\n';
      }
      Monitor(it, norm, r, x);

      if (norm <= norm_goal)
      {
         converged = true;
         break;
      }

      if (it >= max_iter)
      {
         converged = false;
         break;
      }

      grad = &oper->GetGradient(x);
      prec->SetOperator(*grad);

      if (lin_rtol_type)
      {
         AdaptiveLinRtolPreSolve(x, it, norm);
      }

      prec->Mult(r, c); // c = [DF(x_i)]^{-1} [F(x_i)-b]

      if (lin_rtol_type)
      {
         AdaptiveLinRtolPostSolve(c, r, it, norm);
      }

      const real_t c_scale = ComputeScalingFactor(x, b);
      if (c_scale == 0.0)
      {
         converged = false;
         break;
      }
      add(x, -c_scale, c, x);

      ProcessNewState(x);

      oper->Mult(x, r);
      if (have_b)
      {
         r -= b;
      }
      norm = Norm(r);
   }

   final_iter = it;
   final_norm = norm;

   if (print_options.summary || (!converged && print_options.warnings) ||
       print_options.first_and_last)
   {
      mfem::out << "Newton: Number of iterations: " << final_iter << '\n'
                << "   ||r|| = " << final_norm
                << ",  ||r||/||r_0|| = " << final_norm/norm0 << '\n';
   }
   if (!converged && (print_options.summary || print_options.warnings))
   {
      mfem::out << "Newton: No convergence!\n";
   }
}

void NewtonSolver::SetAdaptiveLinRtol(const int type,
                                      const real_t rtol0,
                                      const real_t rtol_max,
                                      const real_t alpha_,
                                      const real_t gamma_)
{
   lin_rtol_type = type;
   lin_rtol0 = rtol0;
   lin_rtol_max = rtol_max;
   this->alpha = alpha_;
   this->gamma = gamma_;
}

void NewtonSolver::AdaptiveLinRtolPreSolve(const Vector &x,
                                           const int it,
                                           const real_t fnorm) const
{
   // Assume that when adaptive linear solver relative tolerance is activated,
   // we are working with an iterative solver.
   auto iterative_solver = static_cast<IterativeSolver *>(prec);
   // Adaptive linear solver relative tolerance
   real_t eta;
   // Safeguard threshold
   real_t sg_threshold = 0.1;

   if (it == 0)
   {
      eta = lin_rtol0;
   }
   else
   {
      if (lin_rtol_type == 1)
      {
         // eta = gamma * abs(||F(x1)|| - ||F(x0) + DF(x0) s0||) / ||F(x0)||
         eta = gamma * abs(fnorm - lnorm_last) / fnorm_last;
      }
      else if (lin_rtol_type == 2)
      {
         // eta = gamma * (||F(x1)|| / ||F(x0)||)^alpha
         eta = gamma * pow(fnorm / fnorm_last, alpha);
      }
      else
      {
         MFEM_ABORT("Unknown adaptive linear solver rtol version");
      }

      // Safeguard rtol from "oversolving" ?!
      const real_t sg_eta = gamma * pow(eta_last, alpha);
      if (sg_eta > sg_threshold) { eta = std::max(eta, sg_eta); }
   }

   eta = std::min(eta, lin_rtol_max);
   iterative_solver->SetRelTol(eta);
   eta_last = eta;
   if (print_options.iterations)
   {
      mfem::out << "Eisenstat-Walker rtol = " << eta << "\n";
   }
}

void NewtonSolver::AdaptiveLinRtolPostSolve(const Vector &x,
                                            const Vector &b,
                                            const int it,
                                            const real_t fnorm) const
{
   fnorm_last = fnorm;

   // If version 1 is chosen, the true linear residual norm has to be computed
   // and in most cases we can only retrieve the preconditioned linear residual
   // norm.
   if (lin_rtol_type == 1)
   {
      // lnorm_last = ||F(x0) + DF(x0) s0||
      Vector linres(x.Size());
      grad->Mult(x, linres);
      linres -= b;
      lnorm_last = Norm(linres);
   }
}

void LBFGSSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(oper != NULL, "the Operator is not set (use SetOperator).");

   // Quadrature points that are checked for negative Jacobians etc.
   Vector sk, rk, yk, rho, alpha;

   // r - r_{k+1}, c - descent direction
   sk.SetSize(width);    // x_{k+1}-x_k
   rk.SetSize(width);    // nabla(f(x_{k}))
   yk.SetSize(width);    // r_{k+1}-r_{k}
   rho.SetSize(m);       // 1/(dot(yk,sk)
   alpha.SetSize(m);     // rhok*sk'*c
   int last_saved_id = -1;

   int it;
   real_t norm0, norm, norm_goal;
   const bool have_b = (b.Size() == Height());

   if (!iterative_mode)
   {
      x = 0.0;
   }

   ProcessNewState(x);

   // r = F(x)-b
   oper->Mult(x, r);
   if (have_b) { r -= b; }

   c = r;           // initial descent direction

   norm0 = norm = initial_norm = Norm(r);
   if (print_options.first_and_last && !print_options.iterations)
   {
      mfem::out << "LBFGS iteration " << setw(2) << 0
                << " : ||r|| = " << norm << "...\n";
   }
   norm_goal = std::max(rel_tol*norm, abs_tol);
   for (it = 0; true; it++)
   {
      MFEM_VERIFY(IsFinite(norm), "norm = " << norm);
      if (print_options.iterations)
      {
         mfem::out << "LBFGS iteration " <<  it
                   << " : ||r|| = " << norm;
         if (it > 0)
         {
            mfem::out << ", ||r||/||r_0|| = " << norm/norm0;
         }
         mfem::out << '\n';
      }

      if (norm <= norm_goal)
      {
         converged = true;
         break;
      }

      if (it >= max_iter)
      {
         converged = false;
         break;
      }

      rk = r;
      const real_t c_scale = ComputeScalingFactor(x, b);
      if (c_scale == 0.0)
      {
         converged = false;
         break;
      }
      add(x, -c_scale, c, x); // x_{k+1} = x_k - c_scale*c

      ProcessNewState(x);

      oper->Mult(x, r);
      if (have_b)
      {
         r -= b;
      }

      // LBFGS - construct descent direction
      subtract(r, rk, yk);   // yk = r_{k+1} - r_{k}
      sk = c; sk *= -c_scale; //sk = x_{k+1} - x_{k} = -c_scale*c
      const real_t gamma = Dot(sk, yk)/Dot(yk, yk);

      // Save last m vectors
      last_saved_id = (last_saved_id == m-1) ? 0 : last_saved_id+1;
      *skArray[last_saved_id] = sk;
      *ykArray[last_saved_id] = yk;

      c = r;
      for (int i = last_saved_id; i > -1; i--)
      {
         rho(i) = 1.0/Dot((*skArray[i]),(*ykArray[i]));
         alpha(i) = rho(i)*Dot((*skArray[i]),c);
         add(c, -alpha(i), (*ykArray[i]), c);
      }
      if (it > m-1)
      {
         for (int i = m-1; i > last_saved_id; i--)
         {
            rho(i) = 1./Dot((*skArray[i]), (*ykArray[i]));
            alpha(i) = rho(i)*Dot((*skArray[i]),c);
            add(c, -alpha(i), (*ykArray[i]), c);
         }
      }

      c *= gamma;   // scale search direction
      if (it > m-1)
      {
         for (int i = last_saved_id+1; i < m ; i++)
         {
            real_t betai = rho(i)*Dot((*ykArray[i]), c);
            add(c, alpha(i)-betai, (*skArray[i]), c);
         }
      }
      for (int i = 0; i < last_saved_id+1 ; i++)
      {
         real_t betai = rho(i)*Dot((*ykArray[i]), c);
         add(c, alpha(i)-betai, (*skArray[i]), c);
      }

      norm = Norm(r);
   }

   final_iter = it;
   final_norm = norm;

   if (print_options.summary || (!converged && print_options.warnings) ||
       print_options.first_and_last)
   {
      mfem::out << "LBFGS: Number of iterations: " << final_iter << '\n'
                << "   ||r|| = " << final_norm
                << ",  ||r||/||r_0|| = " << final_norm/norm0 << '\n';
   }
   if (print_options.summary || (!converged && print_options.warnings))
   {
      mfem::out << "LBFGS: No convergence!\n";
   }
}

int aGMRES(const Operator &A, Vector &x, const Vector &b,
           const Operator &M, int &max_iter,
           int m_max, int m_min, int m_step, real_t cf,
           real_t &tol, real_t &atol, int printit)
{
   int n = A.Width();

   int m = m_max;

   DenseMatrix H(m+1,m);
   Vector s(m+1), cs(m+1), sn(m+1);
   Vector w(n), av(n);

   real_t r1, resid;
   int i, j, k;

   M.Mult(b,w);
   real_t normb = w.Norml2(); // normb = ||M b||
   if (normb == 0.0)
   {
      normb = 1;
   }

   Vector r(n);
   A.Mult(x, r);
   subtract(b,r,w);
   M.Mult(w, r);           // r = M (b - A x)
   real_t beta = r.Norml2();  // beta = ||r||

   resid = beta / normb;

   if (resid * resid <= tol)
   {
      tol = resid * resid;
      max_iter = 0;
      return 0;
   }

   if (printit)
   {
      mfem::out << "   Pass : " << setw(2) << 1
                << "   Iteration : " << setw(3) << 0
                << "  (r, r) = " << beta*beta << '\n';
   }

   tol *= (normb*normb);
   tol = (atol > tol) ? atol : tol;

   m = m_max;
   Array<Vector *> v(m+1);
   for (i= 0; i<=m; i++)
   {
      v[i] = new Vector(n);
      (*v[i]) = 0.0;
   }

   j = 1;
   while (j <= max_iter)
   {
      (*v[0]) = 0.0;
      v[0] -> Add (1.0/beta, r);   // v[0] = r / ||r||
      s = 0.0; s(0) = beta;

      r1 = beta;

      for (i = 0; i < m && j <= max_iter; i++)
      {
         A.Mult((*v[i]),av);
         M.Mult(av,w);              // w = M A v[i]

         for (k = 0; k <= i; k++)
         {
            H(k,i) = w * (*v[k]);    // H(k,i) = w * v[k]
            w.Add(-H(k,i), (*v[k])); // w -= H(k,i) * v[k]
         }

         H(i+1,i)  = w.Norml2();     // H(i+1,i) = ||w||
         (*v[i+1]) = 0.0;
         v[i+1] -> Add (1.0/H(i+1,i), w); // v[i+1] = w / H(i+1,i)

         for (k = 0; k < i; k++)
         {
            ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k));
         }

         GeneratePlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));

         resid = fabs(s(i+1));
         if (printit)
         {
            mfem::out << "   Pass : " << setw(2) << j
                      << "   Iteration : " << setw(3) << i+1
                      << "  (r, r) = " << resid*resid << '\n';
         }

         if ( resid*resid < tol)
         {
            Update(x, i, H, s, v);
            tol = resid * resid;
            max_iter = j;
            for (i= 0; i<=m; i++)
            {
               delete v[i];
            }
            return 0;
         }
      }

      if (printit)
      {
         mfem::out << "Restarting..." << '\n';
      }

      Update(x, i-1, H, s, v);

      A.Mult(x, r);
      subtract(b,r,w);
      M.Mult(w, r);           // r = M (b - A x)
      beta = r.Norml2();      // beta = ||r||
      if ( resid*resid < tol)
      {
         tol = resid * resid;
         max_iter = j;
         for (i= 0; i<=m; i++)
         {
            delete v[i];
         }
         return 0;
      }

      if (beta/r1 > cf)
      {
         if (m - m_step >= m_min)
         {
            m -= m_step;
         }
         else
         {
            m = m_max;
         }
      }

      j++;
   }

   tol = resid * resid;
   for (i= 0; i<=m; i++)
   {
      delete v[i];
   }
   return 1;
}

OptimizationProblem::OptimizationProblem(const int insize,
                                         const Operator *C_,
                                         const Operator *D_)
   : C(C_), D(D_), c_e(NULL), d_lo(NULL), d_hi(NULL), x_lo(NULL), x_hi(NULL),
     input_size(insize)
{
   if (C) { MFEM_VERIFY(C->Width() == input_size, "Wrong width of C."); }
   if (D) { MFEM_VERIFY(D->Width() == input_size, "Wrong width of D."); }
}

void OptimizationProblem::SetEqualityConstraint(const Vector &c)
{
   MFEM_VERIFY(C, "The C operator is unspecified -- can't set constraints.");
   MFEM_VERIFY(c.Size() == C->Height(), "Wrong size of the constraint.");

   c_e = &c;
}

void OptimizationProblem::SetInequalityConstraint(const Vector &dl,
                                                  const Vector &dh)
{
   MFEM_VERIFY(D, "The D operator is unspecified -- can't set constraints.");
   MFEM_VERIFY(dl.Size() == D->Height() && dh.Size() == D->Height(),
               "Wrong size of the constraint.");

   d_lo = &dl; d_hi = &dh;
}

void OptimizationProblem::SetSolutionBounds(const Vector &xl, const Vector &xh)
{
   MFEM_VERIFY(xl.Size() == input_size && xh.Size() == input_size,
               "Wrong size of the constraint.");

   x_lo = &xl; x_hi = &xh;
}

int OptimizationProblem::GetNumConstraints() const
{
   int m = 0;
   if (C) { m += C->Height(); }
   if (D) { m += D->Height(); }
   return m;
}

void SLBQPOptimizer::SetOptimizationProblem(const OptimizationProblem &prob)
{
   if (print_options.warnings)
   {
      MFEM_WARNING("Objective functional is ignored as SLBQP always minimizes"
                   "the l2 norm of (x - x_target).");
   }
   MFEM_VERIFY(prob.GetC(), "Linear constraint is not set.");
   MFEM_VERIFY(prob.GetC()->Height() == 1, "Solver expects scalar constraint.");

   problem = &prob;
}

void SLBQPOptimizer::SetBounds(const Vector &lo_, const Vector &hi_)
{
   lo.SetDataAndSize(lo_.GetData(), lo_.Size());
   hi.SetDataAndSize(hi_.GetData(), hi_.Size());
}

void SLBQPOptimizer::SetLinearConstraint(const Vector &w_, real_t a_)
{
   w.SetDataAndSize(w_.GetData(), w_.Size());
   a = a_;
}

inline void SLBQPOptimizer::print_iteration(int it, real_t r, real_t l) const
{
   if (print_options.iterations || (print_options.first_and_last && it == 0))
   {
      mfem::out << "SLBQP iteration " << it << ": residual = " << r
                << ", lambda = " << l << '\n';
   }
}

void SLBQPOptimizer::Mult(const Vector& xt, Vector& x) const
{
   // Based on code provided by Denis Ridzal, dridzal@sandia.gov.
   // Algorithm adapted from Dai and Fletcher, "New Algorithms for
   // Singly Linearly Constrained Quadratic Programs Subject to Lower
   // and Upper Bounds", Numerical Analysis Report NA/216, 2003.

   // Set some algorithm-specific constants and temporaries.
   int nclip   = 0;
   real_t l    = 0;
   real_t llow = 0;
   real_t lupp = 0;
   real_t lnew = 0;
   real_t dl   = 2;
   real_t r    = 0;
   real_t rlow = 0;
   real_t rupp = 0;
   real_t s    = 0;

   const real_t smin = 0.1;

   const real_t tol = max(abs_tol, rel_tol*a);

   // *** Start bracketing phase of SLBQP ***
   if (print_options.iterations)
   {
      mfem::out << "SLBQP bracketing phase" << '\n';
   }

   // Solve QP with fixed Lagrange multiplier
   r = initial_norm = solve(l,xt,x,nclip);
   print_iteration(nclip, r, l);


   // If x=xt was already within bounds and satisfies the linear
   // constraint, then we already have the solution.
   if (fabs(r) <= tol)
   {
      converged = true;
      goto slbqp_done;
   }

   if (r < 0)
   {
      llow = l;  rlow = r;  l = l + dl;

      // Solve QP with fixed Lagrange multiplier
      r = solve(l,xt,x,nclip);
      print_iteration(nclip, r, l);

      while ((r < 0) && (nclip < max_iter))
      {
         llow = l;
         s = rlow/r - 1.0;
         if (s < smin) { s = smin; }
         dl = dl + dl/s;
         l = l + dl;

         // Solve QP with fixed Lagrange multiplier
         r = solve(l,xt,x,nclip);
         print_iteration(nclip, r, l);
      }

      lupp = l;  rupp = r;
   }
   else
   {
      lupp = l;  rupp = r;  l = l - dl;

      // Solve QP with fixed Lagrange multiplier
      r = solve(l,xt,x,nclip);
      print_iteration(nclip, r, l);

      while ((r > 0) && (nclip < max_iter))
      {
         lupp = l;
         s = rupp/r - 1.0;
         if (s < smin) { s = smin; }
         dl = dl + dl/s;
         l = l - dl;

         // Solve QP with fixed Lagrange multiplier
         r = solve(l,xt,x,nclip);
         print_iteration(nclip, r, l);
      }

      llow = l;  rlow = r;
   }

   // *** Stop bracketing phase of SLBQP ***


   // *** Start secant phase of SLBQP ***
   if (print_options.iterations)
   {
      mfem::out << "SLBQP secant phase" << '\n';
   }

   s = 1.0 - rlow/rupp;  dl = dl/s;  l = lupp - dl;

   // Solve QP with fixed Lagrange multiplier
   r = solve(l,xt,x,nclip);
   print_iteration(nclip, r, l);

   while ( (fabs(r) > tol) && (nclip < max_iter) )
   {
      if (r > 0)
      {
         if (s <= 2.0)
         {
            lupp = l;  rupp = r;  s = 1.0 - rlow/rupp;
            dl = (lupp - llow)/s;  l = lupp - dl;
         }
         else
         {
            s = rupp/r - 1.0;
            if (s < smin) { s = smin; }
            dl = (lupp - l)/s;
            lnew = 0.75*llow + 0.25*l;
            if (lnew < l-dl) { lnew = l-dl; }
            lupp = l;  rupp = r;  l = lnew;
            s = (lupp - llow)/(lupp - l);
         }

      }
      else
      {
         if (s >= 2.0)
         {
            llow = l;  rlow = r;  s = 1.0 - rlow/rupp;
            dl = (lupp - llow)/s;  l = lupp - dl;
         }
         else
         {
            s = rlow/r - 1.0;
            if (s < smin) { s = smin; }
            dl = (l - llow)/s;
            lnew = 0.75*lupp + 0.25*l;
            if (lnew < l+dl) { lnew = l+dl; }
            llow = l;  rlow = r; l = lnew;
            s = (lupp - llow)/(lupp - l);
         }
      }

      // Solve QP with fixed Lagrange multiplier
      r = solve(l,xt,x,nclip);
      print_iteration(nclip, r, l);
   }

   // *** Stop secant phase of SLBQP ***
   converged = (fabs(r) <= tol);

slbqp_done:

   final_iter = nclip;
   final_norm = r;

   if (print_options.summary || (!converged && print_options.warnings) ||
       print_options.first_and_last)
   {
      mfem::out << "SLBQP: Number of iterations: " << final_iter << '\n'
                << "   lambda = " << l << '\n'
                << "   ||r||  = " << final_norm << '\n';
   }
   if (!converged && print_options.warnings)
   {
      mfem::out << "SLBQP: No convergence!" << '\n';
   }
}

struct WeightMinHeap
{
   const std::vector<real_t> &w;
   std::vector<size_t> c;
   std::vector<int> loc;

   WeightMinHeap(const std::vector<real_t> &w_) : w(w_)
   {
      c.reserve(w.size());
      loc.resize(w.size());
      for (size_t i=0; i<w.size(); ++i) { push(i); }
   }

   size_t percolate_up(size_t pos, real_t val)
   {
      for (; pos > 0 && w[c[(pos-1)/2]] > val; pos = (pos-1)/2)
      {
         c[pos] = c[(pos-1)/2];
         loc[c[(pos-1)/2]] = static_cast<int>(pos);
      }
      return pos;
   }

   size_t percolate_down(size_t pos, real_t val)
   {
      while (2*pos+1 < c.size())
      {
         size_t left = 2*pos+1;
         size_t right = left+1;
         size_t tgt;
         if (right < c.size() && w[c[right]] < w[c[left]]) { tgt = right; }
         else { tgt = left; }
         if (w[c[tgt]] < val)
         {
            c[pos] = c[tgt];
            loc[c[tgt]] = static_cast<int>(pos);
            pos = tgt;
         }
         else
         {
            break;
         }
      }
      return pos;
   }

   void push(size_t i)
   {
      real_t val = w[i];
      c.push_back(0);
      size_t pos = c.size()-1;
      pos = percolate_up(pos, val);
      c[pos] = i;
      loc[i] = static_cast<int>(pos);
   }

   int pop()
   {
      size_t i = c[0];
      size_t j = c.back();
      c.pop_back();
      // Mark as removed
      loc[i] = -1;
      if (c.empty()) { return static_cast<int>(i); }
      real_t val = w[j];
      size_t pos = 0;
      pos = percolate_down(pos, val);
      c[pos] = j;
      loc[j] = static_cast<int>(pos);
      return static_cast<int>(i);
   }

   void update(size_t i)
   {
      size_t pos = loc[i];
      real_t val = w[i];
      pos = percolate_up(pos, val);
      pos = percolate_down(pos, val);
      c[pos] = i;
      loc[i] = static_cast<int>(pos);
   }

   bool picked(size_t i)
   {
      return loc[i] < 0;
   }
};

void MinimumDiscardedFillOrdering(SparseMatrix &C, Array<int> &p)
{
   int n = C.Width();
   // Scale rows by reciprocal of diagonal and take absolute value
   Vector D;
   C.GetDiag(D);
   int *I = C.GetI();
   int *J = C.GetJ();
   real_t *V = C.GetData();
   for (int i=0; i<n; ++i)
   {
      for (int j=I[i]; j<I[i+1]; ++j)
      {
         V[j] = abs(V[j]/D[i]);
      }
   }

   std::vector<real_t> w(n, 0.0);
   for (int k=0; k<n; ++k)
   {
      // Find all neighbors i of k
      for (int ii=I[k]; ii<I[k+1]; ++ii)
      {
         int i = J[ii];
         // Find value of (i,k)
         real_t C_ik = 0.0;
         for (int kk=I[i]; kk<I[i+1]; ++kk)
         {
            if (J[kk] == k)
            {
               C_ik = V[kk];
               break;
            }
         }
         for (int jj=I[k]; jj<I[k+1]; ++jj)
         {
            int j = J[jj];
            if (j == k) { continue; }
            real_t C_kj = V[jj];
            bool ij_exists = false;
            for (int jj2=I[i]; jj2<I[i+1]; ++jj2)
            {
               if (J[jj2] == j)
               {
                  ij_exists = true;
                  break;
               }
            }
            if (!ij_exists) { w[k] += pow(C_ik*C_kj,2); }
         }
      }
      w[k] = sqrt(w[k]);
   }

   WeightMinHeap w_heap(w);

   // Compute ordering
   p.SetSize(n);
   for (int ii=0; ii<n; ++ii)
   {
      int pi = w_heap.pop();
      p[ii] = pi;
      w[pi] = -1;
      for (int kk=I[pi]; kk<I[pi+1]; ++kk)
      {
         int k = J[kk];
         if (w_heap.picked(k)) { continue; }
         // Recompute weight
         w[k] = 0.0;
         // Find all neighbors i of k
         for (int ii2=I[k]; ii2<I[k+1]; ++ii2)
         {
            int i = J[ii2];
            if (w_heap.picked(i)) { continue; }
            // Find value of (i,k)
            real_t C_ik = 0.0;
            for (int kk2=I[i]; kk2<I[i+1]; ++kk2)
            {
               if (J[kk2] == k)
               {
                  C_ik = V[kk2];
                  break;
               }
            }
            for (int jj=I[k]; jj<I[k+1]; ++jj)
            {
               int j = J[jj];
               if (j == k || w_heap.picked(j)) { continue; }
               real_t C_kj = V[jj];
               bool ij_exists = false;
               for (int jj2=I[i]; jj2<I[i+1]; ++jj2)
               {
                  if (J[jj2] == j)
                  {
                     ij_exists = true;
                     break;
                  }
               }
               if (!ij_exists) { w[k] += pow(C_ik*C_kj,2); }
            }
         }
         w[k] = sqrt(w[k]);
         w_heap.update(k);
      }
   }
}

BlockILU::BlockILU(int block_size_,
                   Reordering reordering_,
                   int k_fill_)
   : Solver(0),
     block_size(block_size_),
     k_fill(k_fill_),
     reordering(reordering_)
{ }

BlockILU::BlockILU(const Operator &op,
                   int block_size_,
                   Reordering reordering_,
                   int k_fill_)
   : BlockILU(block_size_, reordering_, k_fill_)
{
   SetOperator(op);
}

void BlockILU::SetOperator(const Operator &op)
{
   const SparseMatrix *A = NULL;
#ifdef MFEM_USE_MPI
   const HypreParMatrix *A_par = dynamic_cast<const HypreParMatrix *>(&op);
   SparseMatrix A_par_diag;
   if (A_par != NULL)
   {
      A_par->GetDiag(A_par_diag);
      A = &A_par_diag;
   }
#endif
   if (A == NULL)
   {
      A = dynamic_cast<const SparseMatrix *>(&op);
      if (A == NULL)
      {
         MFEM_ABORT("BlockILU must be created with a SparseMatrix or HypreParMatrix");
      }
   }
   height = op.Height();
   width = op.Width();
   MFEM_VERIFY(A->Finalized(), "Matrix must be finalized.");
   CreateBlockPattern(*A);
   Factorize();
}

void BlockILU::CreateBlockPattern(const SparseMatrix &A)
{
   MFEM_VERIFY(k_fill == 0, "Only block ILU(0) is currently supported.");
   if (A.Height() % block_size != 0)
   {
      MFEM_ABORT("BlockILU: block size must evenly divide the matrix size");
   }

   int nrows = A.Height();
   const int *I = A.GetI();
   const int *J = A.GetJ();
   const real_t *V = A.GetData();
   int nnz = 0;
   int nblockrows = nrows / block_size;

   std::vector<std::set<int>> unique_block_cols(nblockrows);

   for (int iblock = 0; iblock < nblockrows; ++iblock)
   {
      for (int bi = 0; bi < block_size; ++bi)
      {
         int i = iblock * block_size + bi;
         for (int k = I[i]; k < I[i + 1]; ++k)
         {
            unique_block_cols[iblock].insert(J[k] / block_size);
         }
      }
      nnz += static_cast<int>(unique_block_cols[iblock].size());
   }

   if (reordering != Reordering::NONE)
   {
      SparseMatrix C(nblockrows, nblockrows);
      for (int iblock = 0; iblock < nblockrows; ++iblock)
      {
         for (int jblock : unique_block_cols[iblock])
         {
            for (int bi = 0; bi < block_size; ++bi)
            {
               int i = iblock * block_size + bi;
               for (int k = I[i]; k < I[i + 1]; ++k)
               {
                  int j = J[k];
                  if (j >= jblock * block_size && j < (jblock + 1) * block_size)
                  {
                     C.Add(iblock, jblock, V[k]*V[k]);
                  }
               }
            }
         }
      }
      C.Finalize(false);
      real_t *CV = C.GetData();
      for (int i=0; i<C.NumNonZeroElems(); ++i)
      {
         CV[i] = sqrt(CV[i]);
      }

      switch (reordering)
      {
         case Reordering::MINIMUM_DISCARDED_FILL:
            MinimumDiscardedFillOrdering(C, P);
            break;
         default:
            MFEM_ABORT("BlockILU: unknown reordering")
      }
   }
   else
   {
      // No reordering: permutation is identity
      P.SetSize(nblockrows);
      for (int i=0; i<nblockrows; ++i)
      {
         P[i] = i;
      }
   }

   // Compute inverse permutation
   Pinv.SetSize(nblockrows);
   for (int i=0; i<nblockrows; ++i)
   {
      Pinv[P[i]] = i;
   }

   // Permute columns
   std::vector<std::vector<int>> unique_block_cols_perminv(nblockrows);
   for (int i=0; i<nblockrows; ++i)
   {
      std::vector<int> &cols = unique_block_cols_perminv[i];
      for (int j : unique_block_cols[P[i]])
      {
         cols.push_back(Pinv[j]);
      }
      std::sort(cols.begin(), cols.end());
   }

   ID.SetSize(nblockrows);
   IB.SetSize(nblockrows + 1);
   IB[0] = 0;
   JB.SetSize(nnz);
   AB.SetSize(block_size, block_size, nnz);
   DB.SetSize(block_size, block_size, nblockrows);
   AB = 0.0;
   DB = 0.0;
   ipiv.SetSize(block_size*nblockrows);
   int counter = 0;

   for (int iblock = 0; iblock < nblockrows; ++iblock)
   {
      int iblock_perm = P[iblock];
      for (int jblock : unique_block_cols_perminv[iblock])
      {
         int jblock_perm = P[jblock];
         if (iblock == jblock)
         {
            ID[iblock] = counter;
         }
         JB[counter] = jblock;
         for (int bi = 0; bi < block_size; ++bi)
         {
            int i = iblock_perm*block_size + bi;
            for (int k = I[i]; k < I[i + 1]; ++k)
            {
               int j = J[k];
               if (j >= jblock_perm*block_size && j < (jblock_perm + 1)*block_size)
               {
                  int bj = j - jblock_perm*block_size;
                  real_t val = V[k];
                  AB(bi, bj, counter) = val;
                  // Extract the diagonal
                  if (iblock == jblock)
                  {
                     DB(bi, bj, iblock) = val;
                  }
               }
            }
         }
         ++counter;
      }
      IB[iblock + 1] = counter;
   }
}

void BlockILU::Factorize()
{
   int nblockrows = Height()/block_size;

   // Precompute LU factorization of diagonal blocks
   for (int i=0; i<nblockrows; ++i)
   {
      LUFactors factorization(DB.GetData(i), &ipiv[i*block_size]);
      factorization.Factor(block_size);
   }

   // Note: we use UseExternalData to extract submatrices from the tensor AB
   // instead of the DenseTensor call operator, because the call operator does
   // not allow for two simultaneous submatrix views into the same tensor
   DenseMatrix A_ik, A_ij, A_kj;
   // Loop over block rows (starting with second block row)
   for (int i=1; i<nblockrows; ++i)
   {
      // Find all nonzeros to the left of the diagonal in row i
      for (int kk=IB[i]; kk<IB[i+1]; ++kk)
      {
         int k = JB[kk];
         // Make sure we're still to the left of the diagonal
         if (k == i) { break; }
         if (k > i)
         {
            MFEM_ABORT("Matrix must be sorted with nonzero diagonal");
         }
         LUFactors A_kk_inv(DB.GetData(k), &ipiv[k*block_size]);
         A_ik.UseExternalData(&AB(0,0,kk), block_size, block_size);
         // A_ik = A_ik * A_kk^{-1}
         A_kk_inv.RightSolve(block_size, block_size, A_ik.GetData());
         // Modify everything to the right of k in row i
         for (int jj=kk+1; jj<IB[i+1]; ++jj)
         {
            int j = JB[jj];
            if (j <= k) { continue; } // Superfluous because JB is sorted?
            A_ij.UseExternalData(&AB(0,0,jj), block_size, block_size);
            for (int ll=IB[k]; ll<IB[k+1]; ++ll)
            {
               int l = JB[ll];
               if (l == j)
               {
                  A_kj.UseExternalData(&AB(0,0,ll), block_size, block_size);
                  // A_ij = A_ij - A_ik*A_kj;
                  AddMult_a(-1.0, A_ik, A_kj, A_ij);
                  // If we need to, update diagonal factorization
                  if (j == i)
                  {
                     DB(i) = A_ij;
                     LUFactors factorization(DB.GetData(i), &ipiv[i*block_size]);
                     factorization.Factor(block_size);
                  }
                  break;
               }
            }
         }
      }
   }
}

void BlockILU::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(height > 0, "BlockILU(0) preconditioner is not constructed");
   int nblockrows = Height()/block_size;
   y.SetSize(Height());

   DenseMatrix B;
   Vector yi, yj, xi, xj;
   Vector tmp(block_size);
   // Forward substitute to solve Ly = b
   // Implicitly, L has identity on the diagonal
   y = 0.0;
   for (int i=0; i<nblockrows; ++i)
   {
      yi.SetDataAndSize(&y[i*block_size], block_size);
      for (int ib=0; ib<block_size; ++ib)
      {
         yi[ib] = b[ib + P[i]*block_size];
      }
      for (int k=IB[i]; k<ID[i]; ++k)
      {
         int j = JB[k];
         const DenseMatrix &L_ij = AB(k);
         yj.SetDataAndSize(&y[j*block_size], block_size);
         // y_i = y_i - L_ij*y_j
         L_ij.AddMult_a(-1.0, yj, yi);
      }
   }
   // Backward substitution to solve Ux = y
   for (int i=nblockrows-1; i >= 0; --i)
   {
      xi.SetDataAndSize(&x[P[i]*block_size], block_size);
      for (int ib=0; ib<block_size; ++ib)
      {
         xi[ib] = y[ib + i*block_size];
      }
      for (int k=ID[i]+1; k<IB[i+1]; ++k)
      {
         int j = JB[k];
         const DenseMatrix &U_ij = AB(k);
         xj.SetDataAndSize(&x[P[j]*block_size], block_size);
         // x_i = x_i - U_ij*x_j
         U_ij.AddMult_a(-1.0, xj, xi);
      }
      LUFactors A_ii_inv(&DB(0,0,i), &ipiv[i*block_size]);
      // x_i = D_ii^{-1} x_i
      A_ii_inv.Solve(block_size, 1, xi.GetData());
   }
}


void ResidualBCMonitor::MonitorResidual(
   int it, real_t norm, const Vector &r, bool final)
{
   if (!ess_dofs_list) { return; }

   real_t bc_norm_squared = 0.0;
   r.HostRead();
   ess_dofs_list->HostRead();
   for (int i = 0; i < ess_dofs_list->Size(); i++)
   {
      const real_t r_entry = r((*ess_dofs_list)[i]);
      bc_norm_squared += r_entry*r_entry;
   }
   bool print = true;
#ifdef MFEM_USE_MPI
   MPI_Comm comm = iter_solver->GetComm();
   if (comm != MPI_COMM_NULL)
   {
      double glob_bc_norm_squared = 0.0;
      MPI_Reduce(&bc_norm_squared, &glob_bc_norm_squared, 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, 0, comm);
      bc_norm_squared = glob_bc_norm_squared;
      int rank;
      MPI_Comm_rank(comm, &rank);
      print = (rank == 0);
   }
#endif
   if ((it == 0 || final || bc_norm_squared > 0.0) && print)
   {
      mfem::out << "      ResidualBCMonitor : b.c. residual norm = "
                << sqrt(bc_norm_squared) << endl;
   }
}


#ifdef MFEM_USE_SUITESPARSE

void UMFPackSolver::Init()
{
   mat = NULL;
   Numeric = NULL;
   AI = AJ = NULL;
   if (!use_long_ints)
   {
      umfpack_di_defaults(Control);
   }
   else
   {
      umfpack_dl_defaults(Control);
   }
}

void UMFPackSolver::SetOperator(const Operator &op)
{
   void *Symbolic;

   if (Numeric)
   {
      if (!use_long_ints)
      {
         umfpack_di_free_numeric(&Numeric);
      }
      else
      {
         umfpack_dl_free_numeric(&Numeric);
      }
   }

   mat = const_cast<SparseMatrix *>(dynamic_cast<const SparseMatrix *>(&op));
   MFEM_VERIFY(mat, "not a SparseMatrix");

   // UMFPack requires that the column-indices in mat corresponding to each
   // row be sorted.
   // Generally, this will modify the ordering of the entries of mat.
   mat->SortColumnIndices();

   height = mat->Height();
   width = mat->Width();
   MFEM_VERIFY(width == height, "not a square matrix");

   const int * Ap = mat->HostReadI();
   const int * Ai = mat->HostReadJ();
   const real_t * Ax = mat->HostReadData();

   if (!use_long_ints)
   {
      int status = umfpack_di_symbolic(width, width, Ap, Ai, Ax, &Symbolic,
                                       Control, Info);
      if (status < 0)
      {
         umfpack_di_report_info(Control, Info);
         umfpack_di_report_status(Control, status);
         mfem_error("UMFPackSolver::SetOperator :"
                    " umfpack_di_symbolic() failed!");
      }

      status = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric,
                                  Control, Info);
      if (status < 0)
      {
         umfpack_di_report_info(Control, Info);
         umfpack_di_report_status(Control, status);
         mfem_error("UMFPackSolver::SetOperator :"
                    " umfpack_di_numeric() failed!");
      }
      umfpack_di_free_symbolic(&Symbolic);
   }
   else
   {
      SuiteSparse_long status;

      delete [] AJ;
      delete [] AI;
      AI = new SuiteSparse_long[width + 1];
      AJ = new SuiteSparse_long[Ap[width]];
      for (int i = 0; i <= width; i++)
      {
         AI[i] = (SuiteSparse_long)(Ap[i]);
      }
      for (int i = 0; i < Ap[width]; i++)
      {
         AJ[i] = (SuiteSparse_long)(Ai[i]);
      }

      status = umfpack_dl_symbolic(width, width, AI, AJ, Ax, &Symbolic,
                                   Control, Info);
      if (status < 0)
      {
         umfpack_dl_report_info(Control, Info);
         umfpack_dl_report_status(Control, status);
         mfem_error("UMFPackSolver::SetOperator :"
                    " umfpack_dl_symbolic() failed!");
      }

      status = umfpack_dl_numeric(AI, AJ, Ax, Symbolic, &Numeric,
                                  Control, Info);
      if (status < 0)
      {
         umfpack_dl_report_info(Control, Info);
         umfpack_dl_report_status(Control, status);
         mfem_error("UMFPackSolver::SetOperator :"
                    " umfpack_dl_numeric() failed!");
      }
      umfpack_dl_free_symbolic(&Symbolic);
   }
}

void UMFPackSolver::Mult(const Vector &b, Vector &x) const
{
   if (mat == NULL)
      mfem_error("UMFPackSolver::Mult : matrix is not set!"
                 " Call SetOperator first!");
   b.HostRead();
   x.HostReadWrite();
   if (!use_long_ints)
   {
      int status =
         umfpack_di_solve(UMFPACK_At, mat->HostReadI(), mat->HostReadJ(),
                          mat->HostReadData(), x.HostWrite(), b.HostRead(),
                          Numeric, Control, Info);
      umfpack_di_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_di_report_status(Control, status);
         mfem_error("UMFPackSolver::Mult : umfpack_di_solve() failed!");
      }
   }
   else
   {
      SuiteSparse_long status =
         umfpack_dl_solve(UMFPACK_At, AI, AJ, mat->HostReadData(),
                          x.HostWrite(), b.HostRead(), Numeric, Control,
                          Info);
      umfpack_dl_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_dl_report_status(Control, status);
         mfem_error("UMFPackSolver::Mult : umfpack_dl_solve() failed!");
      }
   }
}

void UMFPackSolver::MultTranspose(const Vector &b, Vector &x) const
{
   if (mat == NULL)
      mfem_error("UMFPackSolver::MultTranspose : matrix is not set!"
                 " Call SetOperator first!");
   b.HostRead();
   x.HostReadWrite();
   if (!use_long_ints)
   {
      int status =
         umfpack_di_solve(UMFPACK_A, mat->HostReadI(), mat->HostReadJ(),
                          mat->HostReadData(), x.HostWrite(), b.HostRead(),
                          Numeric, Control, Info);
      umfpack_di_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_di_report_status(Control, status);
         mfem_error("UMFPackSolver::MultTranspose :"
                    " umfpack_di_solve() failed!");
      }
   }
   else
   {
      SuiteSparse_long status =
         umfpack_dl_solve(UMFPACK_A, AI, AJ, mat->HostReadData(),
                          x.HostWrite(), b.HostRead(), Numeric, Control,
                          Info);
      umfpack_dl_report_info(Control, Info);
      if (status < 0)
      {
         umfpack_dl_report_status(Control, status);
         mfem_error("UMFPackSolver::MultTranspose :"
                    " umfpack_dl_solve() failed!");
      }
   }
}

UMFPackSolver::~UMFPackSolver()
{
   delete [] AJ;
   delete [] AI;
   if (Numeric)
   {
      if (!use_long_ints)
      {
         umfpack_di_free_numeric(&Numeric);
      }
      else
      {
         umfpack_dl_free_numeric(&Numeric);
      }
   }
}

void KLUSolver::Init()
{
   klu_defaults(&Common);
}

void KLUSolver::SetOperator(const Operator &op)
{
   if (Numeric)
   {
      MFEM_VERIFY(Symbolic != 0,
                  "Had Numeric pointer in KLU, but not Symbolic");
      klu_free_symbolic(&Symbolic, &Common);
      Symbolic = 0;
      klu_free_numeric(&Numeric, &Common);
      Numeric = 0;
   }

   mat = const_cast<SparseMatrix *>(dynamic_cast<const SparseMatrix *>(&op));
   MFEM_VERIFY(mat != NULL, "not a SparseMatrix");

   // KLU requires that the column-indices in mat corresponding to each row be
   // sorted.  Generally, this will modify the ordering of the entries of mat.
   mat->SortColumnIndices();

   height = mat->Height();
   width = mat->Width();
   MFEM_VERIFY(width == height, "not a square matrix");

   int * Ap = mat->GetI();
   int * Ai = mat->GetJ();
   real_t * Ax = mat->GetData();

   Symbolic = klu_analyze( height, Ap, Ai, &Common);
   Numeric = klu_factor(Ap, Ai, Ax, Symbolic, &Common);
}

void KLUSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(mat != NULL,
               "KLUSolver::Mult : matrix is not set!  Call SetOperator first!");

   int n = mat->Height();
   int numRhs = 1;
   // Copy B into X, so we can pass it in and overwrite it.
   x = b;
   // Solve the transpose, since KLU thinks the matrix is compressed column
   // format.
   klu_tsolve( Symbolic, Numeric, n, numRhs, x.GetData(), &Common);
}

void KLUSolver::MultTranspose(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(mat != NULL,
               "KLUSolver::Mult : matrix is not set!  Call SetOperator first!");

   int n = mat->Height();
   int numRhs = 1;
   // Copy B into X, so we can pass it in and overwrite it.
   x = b;
   // Solve the regular matrix, not the transpose, since KLU thinks the matrix
   // is compressed column format.
   klu_solve( Symbolic, Numeric, n, numRhs, x.GetData(), &Common);
}

KLUSolver::~KLUSolver()
{
   klu_free_symbolic (&Symbolic, &Common) ;
   klu_free_numeric (&Numeric, &Common) ;
   Symbolic = 0;
   Numeric = 0;
}

#endif // MFEM_USE_SUITESPARSE

DirectSubBlockSolver::DirectSubBlockSolver(const SparseMatrix &A,
                                           const SparseMatrix &block_dof_)
   : Solver(A.NumRows()), block_dof(const_cast<SparseMatrix&>(block_dof_)),
     block_solvers(new DenseMatrixInverse[block_dof.NumRows()])
{
   DenseMatrix sub_A;
   for (int i = 0; i < block_dof.NumRows(); ++i)
   {
      local_dofs.MakeRef(block_dof.GetRowColumns(i), block_dof.RowSize(i));
      sub_A.SetSize(local_dofs.Size());
      A.GetSubMatrix(local_dofs, local_dofs, sub_A);
      block_solvers[i].SetOperator(sub_A);
   }
}

void DirectSubBlockSolver::Mult(const Vector &x, Vector &y) const
{
   y.SetSize(x.Size());
   y = 0.0;

   for (int i = 0; i < block_dof.NumRows(); ++i)
   {
      local_dofs.MakeRef(block_dof.GetRowColumns(i), block_dof.RowSize(i));
      x.GetSubVector(local_dofs, sub_rhs);
      sub_sol.SetSize(local_dofs.Size());
      block_solvers[i].Mult(sub_rhs, sub_sol);
      y.AddElementVector(local_dofs, sub_sol);
   }
}

void ProductSolver::Mult(const Vector & x, Vector & y) const
{
   y.SetSize(x.Size());
   y = 0.0;
   S0->Mult(x, y);

   Vector z(x.Size());
   z = 0.0;
   A->Mult(y, z);
   add(-1.0, z, 1.0, x, z); // z = (I - A * S0) x

   Vector S1z(x.Size());
   S1z = 0.0;
   S1->Mult(z, S1z);
   y += S1z;
}

void ProductSolver::MultTranspose(const Vector & x, Vector & y) const
{
   y.SetSize(x.Size());
   y = 0.0;
   S1->MultTranspose(x, y);

   Vector z(x.Size());
   z = 0.0;
   A->MultTranspose(y, z);
   add(-1.0, z, 1.0, x, z); // z = (I - A^T * S1^T) x

   Vector S0Tz(x.Size());
   S0Tz = 0.0;
   S0->MultTranspose(z, S0Tz);
   y += S0Tz;
}

OrthoSolver::OrthoSolver()
   : Solver(0, false), global_size(-1)
#ifdef MFEM_USE_MPI
   , parallel(false)
#endif
{ }

#ifdef MFEM_USE_MPI
OrthoSolver::OrthoSolver(MPI_Comm mycomm_)
   : Solver(0, false), mycomm(mycomm_), global_size(-1), parallel(true) { }
#endif

void OrthoSolver::SetSolver(Solver &s)
{
   solver = &s;
   height = s.Height();
   width = s.Width();
   MFEM_VERIFY(height == width, "Solver must be a square Operator!");
   global_size = -1; // lazy evaluated
}

void OrthoSolver::SetOperator(const Operator &op)
{
   MFEM_VERIFY(solver, "Solver hasn't been set, call SetSolver() first.");
   solver->SetOperator(op);
   height = solver->Height();
   width = solver->Width();
   MFEM_VERIFY(height == width, "Solver must be a square Operator!");
   global_size = -1; // lazy evaluated
}

void OrthoSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(solver, "Solver hasn't been set, call SetSolver() first.");
   MFEM_VERIFY(height == solver->Height(),
               "solver was modified externally! call SetSolver() again!");
   MFEM_VERIFY(height == b.Size(), "incompatible input Vector size!");
   MFEM_VERIFY(height == x.Size(), "incompatible output Vector size!");

   // Orthogonalize input
   Orthogonalize(b, b_ortho);

   // Propagate iterative_mode to the solver:
   solver->iterative_mode = iterative_mode;

   // Apply the Solver
   solver->Mult(b_ortho, x);

   // Orthogonalize output
   Orthogonalize(x, x);
}

void OrthoSolver::Orthogonalize(const Vector &v, Vector &v_ortho) const
{
   if (global_size == -1)
   {
      global_size = height;
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         MPI_Allreduce(MPI_IN_PLACE, &global_size, 1, HYPRE_MPI_BIG_INT,
                       MPI_SUM, mycomm);
      }
#endif
   }

   // TODO: GPU/device implementation

   real_t global_sum = v.Sum();

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      MPI_Allreduce(MPI_IN_PLACE, &global_sum, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, mycomm);
   }
#endif

   real_t ratio = global_sum / static_cast<real_t>(global_size);
   v_ortho.SetSize(v.Size());
   v.HostRead();
   v_ortho.HostWrite();
   for (int i = 0; i < v_ortho.Size(); ++i)
   {
      v_ortho(i) = v(i) - ratio;
   }
}

#ifdef MFEM_USE_MPI
AuxSpaceSmoother::AuxSpaceSmoother(const HypreParMatrix &op,
                                   HypreParMatrix *aux_map,
                                   bool op_is_symmetric,
                                   bool own_aux_map)
   : Solver(op.NumRows()), aux_map_(aux_map, own_aux_map)
{
   aux_system_.Reset(RAP(&op, aux_map));
   aux_system_.As<HypreParMatrix>()->EliminateZeroRows();
   aux_smoother_.Reset(new HypreSmoother(*aux_system_.As<HypreParMatrix>()));
   aux_smoother_.As<HypreSmoother>()->SetOperatorSymmetry(op_is_symmetric);
}

void AuxSpaceSmoother::Mult(const Vector &x, Vector &y, bool transpose) const
{
   Vector aux_rhs(aux_map_->NumCols());
   aux_map_->MultTranspose(x, aux_rhs);

   Vector aux_sol(aux_rhs.Size());
   if (transpose)
   {
      aux_smoother_->MultTranspose(aux_rhs, aux_sol);
   }
   else
   {
      aux_smoother_->Mult(aux_rhs, aux_sol);
   }

   y.SetSize(aux_map_->NumRows());
   aux_map_->Mult(aux_sol, y);
}
#endif // MFEM_USE_MPI

#ifdef MFEM_USE_LAPACK

NNLSSolver::NNLSSolver()
   : Solver(0), mat(nullptr), const_tol_(1.0e-14), min_nnz_(0),
     max_nnz_(0), verbosity_(0), res_change_termination_tol_(1.0e-4),
     zero_tol_(1.0e-14), rhs_delta_(1.0e-11), n_outer_(100000),
     n_inner_(100000), nStallCheck_(100), normalize_(true),
     NNLS_qrres_on_(false), qr_residual_mode_(QRresidualMode::hybrid)
{}

void NNLSSolver::SetOperator(const Operator &op)
{
   mat = dynamic_cast<const DenseMatrix*>(&op);
   MFEM_VERIFY(mat, "NNLSSolver operator must be of type DenseMatrix");

   // The size of this operator is that of the transpose of op.
   height = op.Width();
   width = op.Height();

   row_scaling_.SetSize(mat->NumRows());
   row_scaling_ = 1.0;
}

void NNLSSolver::SetQRResidualMode(const QRresidualMode qr_residual_mode)
{
   qr_residual_mode_ = qr_residual_mode;
   if (qr_residual_mode_ == QRresidualMode::on)
   {
      NNLS_qrres_on_ = true;
   }
}

void NNLSSolver::NormalizeConstraints(Vector& rhs_lb, Vector& rhs_ub) const
{
   // Scale everything so that rescaled half gap is the same for all constraints
   const int m = mat->NumRows();

   MFEM_VERIFY(rhs_lb.Size() == m && rhs_ub.Size() == m, "");

   Vector rhs_avg = rhs_ub;
   rhs_avg += rhs_lb;
   rhs_avg *= 0.5;

   Vector rhs_halfgap = rhs_ub;
   rhs_halfgap -= rhs_lb;
   rhs_halfgap *= 0.5;

   Vector rhs_avg_glob = rhs_avg;
   Vector rhs_halfgap_glob = rhs_halfgap;
   Vector halfgap_target(m);
   halfgap_target = 1.0e3 * const_tol_;

   row_scaling_.SetSize(m);

   for (int i=0; i<m; ++i)
   {
      const real_t s = halfgap_target(i) / rhs_halfgap_glob(i);
      row_scaling_[i] = s;

      rhs_lb(i) = (rhs_avg(i) * s) - halfgap_target(i);
      rhs_ub(i) = (rhs_avg(i) * s) + halfgap_target(i);
   }
}

void NNLSSolver::Mult(const Vector &w, Vector &sol) const
{
   MFEM_VERIFY(mat, "NNLSSolver operator must be of type DenseMatrix");
   Vector rhs_ub(mat->NumRows());
   mat->Mult(w, rhs_ub);
   rhs_ub *= row_scaling_;

   Vector rhs_lb(rhs_ub);
   Vector rhs_Gw(rhs_ub);

   for (int i=0; i<rhs_ub.Size(); ++i)
   {
      rhs_lb(i) -= rhs_delta_;
      rhs_ub(i) += rhs_delta_;
   }

   if (normalize_) { NormalizeConstraints(rhs_lb, rhs_ub); }
   Solve(rhs_lb, rhs_ub, sol);

   if (verbosity_ > 1)
   {
      int nnz = 0;
      for (int i=0; i<sol.Size(); ++i)
      {
         if (sol(i) != 0.0)
         {
            nnz++;
         }
      }

      mfem::out << "Number of nonzeros in NNLSSolver solution: " << nnz
                << ", out of " << sol.Size() << endl;

      // Check residual of NNLS solution
      Vector res(mat->NumRows());
      mat->Mult(sol, res);
      res *= row_scaling_;

      const real_t normGsol = res.Norml2();
      const real_t normRHS = rhs_Gw.Norml2();

      res -= rhs_Gw;
      const real_t relNorm = res.Norml2() / std::max(normGsol, normRHS);
      mfem::out << "Relative residual norm for NNLSSolver solution of Gs = Gw: "
                << relNorm << endl;
   }
}

void NNLSSolver::Solve(const Vector& rhs_lb, const Vector& rhs_ub,
                       Vector& soln) const
{
   int m = mat->NumRows();
   int n = mat->NumCols();

   MFEM_VERIFY(rhs_lb.Size() == m && rhs_lb.Size() == m && soln.Size() == n, "");
   MFEM_VERIFY(n >= m, "NNLSSolver system cannot be over-determined.");

   if (max_nnz_ == 0)
   {
      max_nnz_ = mat->NumCols();
   }

   // Prepare right hand side
   Vector rhs_avg(rhs_ub);
   rhs_avg += rhs_lb;
   rhs_avg *= 0.5;

   Vector rhs_halfgap(rhs_ub);
   rhs_halfgap -= rhs_lb;
   rhs_halfgap *= 0.5;

   Vector rhs_avg_glob(rhs_avg);
   Vector rhs_halfgap_glob(rhs_halfgap);

   int ione = 1;
   real_t fone = 1.0;

   char lside = 'L';
   char trans = 'T';
   char notrans = 'N';

   std::vector<unsigned int> nz_ind(m);
   Vector res_glob(m);
   Vector mu(n);
   Vector mu2(n);
   int n_nz_ind = 0;
   int n_glob = 0;
   int m_update;
   int min_nnz_cap = std::min(static_cast<int>(min_nnz_), std::min(m,n));
   int info;
   std::vector<real_t> l2_res_hist;
   std::vector<unsigned int> stalled_indices;
   int stalledFlag = 0;
   int num_stalled = 0;
   int nz_ind_zero = 0;

   Vector soln_nz_glob(m);
   Vector soln_nz_glob_up(m);

   // The following matrices are stored in column-major format as Vectors
   Vector mat_0_data(m * n);
   Vector mat_qr_data(m * n);
   Vector submat_data(m * n);

   Vector tau(n);
   Vector sub_tau = tau;
   Vector vec1(m);

   // Temporary work arrays
   int lwork;
   std::vector<real_t> work;
   int n_outer_iter = 0;
   int n_total_inner_iter = 0;
   int i_qr_start;
   int n_update;
   // 0 = converged; 1 = maximum iterations reached;
   // 2 = NNLS stalled (no change in residual for many iterations)
   int exit_flag = 1;

   res_glob = rhs_avg_glob;
   Vector qt_rhs_glob = rhs_avg_glob;
   Vector qqt_rhs_glob = qt_rhs_glob;
   Vector sub_qt = rhs_avg_glob;

   // Compute threshold tolerance for the Lagrange multiplier mu
   real_t mu_tol = 0.0;

   {
      Vector rhs_scaled(rhs_halfgap_glob);
      Vector tmp(n);
      rhs_scaled *= row_scaling_;
      mat->MultTranspose(rhs_scaled, tmp);

      mu_tol = 1.0e-15 * tmp.Max();
   }

   real_t rmax = 0.0;
   real_t mumax = 0.0;

   for (int oiter = 0; oiter < n_outer_; ++oiter)
   {
      stalledFlag = 0;

      rmax = fabs(res_glob(0)) - rhs_halfgap_glob(0);
      for (int i=1; i<m; ++i)
      {
         rmax = std::max(rmax, fabs(res_glob(i)) - rhs_halfgap_glob(i));
      }

      l2_res_hist.push_back(res_glob.Norml2());

      if (verbosity_ > 1)
      {
         mfem::out << "NNLS " << oiter << " " << n_total_inner_iter << " " << m
                   << " " << n << " " << n_glob << " " << rmax << " "
                   << l2_res_hist[oiter] << endl;
      }
      if (rmax <= const_tol_ && n_glob >= min_nnz_cap)
      {
         if (verbosity_ > 1)
         {
            mfem::out << "NNLS target tolerance met" << endl;
         }
         exit_flag = 0;
         break;
      }

      if (n_glob >= max_nnz_)
      {
         if (verbosity_ > 1)
         {
            mfem::out << "NNLS target nnz met" << endl;
         }
         exit_flag = 0;
         break;
      }

      if (n_glob >= m)
      {
         if (verbosity_ > 1)
         {
            mfem::out << "NNLS system is square... exiting" << endl;
         }
         exit_flag = 3;
         break;
      }

      // Check for stall after the first nStallCheck iterations
      if (oiter > nStallCheck_)
      {
         real_t mean0 = 0.0;
         real_t mean1 = 0.0;
         for (int i=0; i<nStallCheck_/2; ++i)
         {
            mean0 += l2_res_hist[oiter - i];
            mean1 += l2_res_hist[oiter - (nStallCheck_) - i];
         }

         real_t mean_res_change = (mean1 / mean0) - 1.0;
         if (std::abs(mean_res_change) < res_change_termination_tol_)
         {
            if (verbosity_ > 1)
            {
               mfem::out << "NNLSSolver stall detected... exiting" << endl;
            }
            exit_flag = 2;
            break;
         }
      }

      // Find the next index
      res_glob *= row_scaling_;
      mat->MultTranspose(res_glob, mu);

      for (int i = 0; i < n_nz_ind; ++i)
      {
         mu(nz_ind[i]) = 0.0;
      }
      for (unsigned int i = 0; i < stalled_indices.size(); ++i)
      {
         mu(stalled_indices[i]) = 0.0;
      }

      mumax = mu.Max();

      if (mumax < mu_tol)
      {
         num_stalled = stalled_indices.size();
         if (num_stalled > 0)
         {
            if (verbosity_ > 0)
            {
               mfem::out << "NNLS Lagrange multiplier is below the minimum "
                         << "threshold: mumax = " << mumax << ", mutol = "
                         << mu_tol << "\n" << " Resetting stalled indices "
                         << "vector of size " << num_stalled << "\n";
            }
            stalled_indices.resize(0);

            mat->MultTranspose(res_glob, mu);

            for (int i = 0; i < n_nz_ind; ++i)
            {
               mu(nz_ind[i]) = 0.0;
            }

            mumax = mu.Max();
         }
      }

      int imax = 0;
      {
         real_t tmax = mu(0);
         for (int i=1; i<n; ++i)
         {
            if (mu(i) > tmax)
            {
               tmax = mu(i);
               imax = i;
            }
         }
      }

      // Record the local value of the next index
      nz_ind[n_nz_ind] = imax;
      ++n_nz_ind;

      if (verbosity_ > 2)
      {
         mfem::out << "Found next index: " << imax << " " << mumax << endl;
      }

      for (int i=0; i<m; ++i)
      {
         mat_0_data(i + (n_glob*m)) = (*mat)(i,imax) * row_scaling_[i];
         mat_qr_data(i + (n_glob*m)) = mat_0_data(i + (n_glob*m));
      }

      i_qr_start = n_glob;
      ++n_glob; // Increment the size of the global matrix

      if (verbosity_ > 2)
      {
         mfem::out << "Updated matrix with new index" << endl;
      }

      for (int iiter = 0; iiter < n_inner_; ++iiter)
      {
         ++n_total_inner_iter;

         // Initialize
         const bool incremental_update = true;
         n_update = n_glob - i_qr_start;
         m_update = m - i_qr_start;
         if (incremental_update)
         {
            // Apply Householder reflectors to compute Q^T new_cols
            lwork = -1;
            work.resize(10);

            MFEM_LAPACK_PREFIX(ormqr_)(&lside, &trans, &m, &n_update,
                                       &i_qr_start, mat_qr_data.GetData(), &m,
                                       tau.GetData(),
                                       mat_qr_data.GetData() + (i_qr_start * m),
                                       &m, work.data(), &lwork, &info);
            MFEM_VERIFY(info == 0, ""); // Q^T A update work calculation failed
            lwork = static_cast<int>(work[0]);
            work.resize(lwork);
            MFEM_LAPACK_PREFIX(ormqr_)(&lside, &trans, &m, &n_update,
                                       &i_qr_start, mat_qr_data.GetData(), &m,
                                       tau.GetData(),
                                       mat_qr_data.GetData() + (i_qr_start * m),
                                       &m, work.data(), &lwork, &info);
            MFEM_VERIFY(info == 0, ""); // Q^T A update failed
            // Compute QR factorization of the submatrix
            lwork = -1;
            work.resize(10);

            // Copy m_update-by-n_update submatrix of mat_qr_data,
            // starting at (i_qr_start, i_qr_start)
            for (int i=0; i<m_update; ++i)
               for (int j=0; j<n_update; ++j)
               {
                  submat_data[i + (j * m_update)] =
                     mat_qr_data[i + i_qr_start + ((j + i_qr_start) * m)];
               }

            // Copy tau subvector of length n_update, starting at i_qr_start
            for (int j=0; j<n_update; ++j)
            {
               sub_tau[j] = tau[i_qr_start + j];
            }

            MFEM_LAPACK_PREFIX(geqrf_)(&m_update, &n_update, submat_data.GetData(),
                                       &m_update, sub_tau.GetData(), work.data(),
                                       &lwork, &info);
            MFEM_VERIFY(info == 0, ""); // QR update factorization work calc
            lwork = static_cast<int>(work[0]);
            if (lwork == 0) { lwork = 1; }
            work.resize(lwork);
            MFEM_LAPACK_PREFIX(geqrf_)(&m_update, &n_update, submat_data.GetData(),
                                       &m_update, sub_tau.GetData(), work.data(),
                                       &lwork, &info);
            MFEM_VERIFY(info == 0, ""); // QR update factorization failed

            // Copy result back
            for (int i=0; i<m_update; ++i)
               for (int j=0; j<n_update; ++j)
               {
                  mat_qr_data[i + i_qr_start + ((j + i_qr_start)* m)] =
                     submat_data[i + (j * m_update)];
               }

            for (int j=0; j<n_update; ++j)
            {
               tau[i_qr_start + j] = sub_tau[j];
            }
         }
         else
         {
            // Copy everything to mat_qr then do full QR
            for (int i=0; i<m; ++i)
               for (int j=0; j<n_glob; ++j)
               {
                  mat_qr_data(i + (j*m)) = mat_0_data(i + (j*m));
               }

            // Compute qr factorization (first find the size of work and then
            // perform qr)
            lwork = -1;
            work.resize(10);
            MFEM_LAPACK_PREFIX(geqrf_)(&m, &n_glob, mat_qr_data.GetData(), &m,
                                       tau.GetData(), work.data(), &lwork, &info);
            MFEM_VERIFY(info == 0, ""); // QR factorization work calculation
            lwork = static_cast<int>(work[0]);
            work.resize(lwork);
            MFEM_LAPACK_PREFIX(geqrf_)(&m, &n_glob, mat_qr_data.GetData(), &m,
                                       tau.GetData(), work.data(), &lwork, &info);
            MFEM_VERIFY(info == 0, ""); // QR factorization failed
         }

         if (verbosity_ > 2)
         {
            mfem::out << "Updated QR " << iiter << endl;
         }

         // Apply Householder reflectors to compute Q^T b
         if (incremental_update && iiter == 0)
         {
            lwork = -1;
            work.resize(10);

            // Copy submatrix of mat_qr_data starting at
            //   (i_qr_start, i_qr_start), of size m_update-by-1
            // Copy submatrix of qt_rhs_glob starting at (i_qr_start, 0),
            //   of size m_update-by-1

            for (int i=0; i<m_update; ++i)
            {
               submat_data[i] = mat_qr_data[i + i_qr_start + (i_qr_start * m)];
               sub_qt[i] = qt_rhs_glob[i + i_qr_start];
            }

            sub_tau[0] = tau[i_qr_start];

            MFEM_LAPACK_PREFIX(ormqr_)(&lside, &trans, &m_update, &ione, &ione,
                                       submat_data.GetData(), &m_update,
                                       sub_tau.GetData(), sub_qt.GetData(),
                                       &m_update, work.data(), &lwork, &info);
            MFEM_VERIFY(info == 0, ""); // H_last y work calculation failed
            lwork = static_cast<int>(work[0]);
            work.resize(lwork);
            MFEM_LAPACK_PREFIX(ormqr_)(&lside, &trans, &m_update, &ione, &ione,
                                       submat_data.GetData(), &m_update,
                                       sub_tau.GetData(), sub_qt.GetData(),
                                       &m_update, work.data(), &lwork, &info);
            MFEM_VERIFY(info == 0, ""); // H_last y failed
            // Copy result back
            for (int i=0; i<m_update; ++i)
            {
               qt_rhs_glob[i + i_qr_start] = sub_qt[i];
            }
         }
         else
         {
            // Compute Q^T b from scratch
            qt_rhs_glob = rhs_avg_glob;
            lwork = -1;
            work.resize(10);
            MFEM_LAPACK_PREFIX(ormqr_)(&lside, &trans, &m, &ione, &n_glob,
                                       mat_qr_data.GetData(), &m, tau.GetData(),
                                       qt_rhs_glob.GetData(), &m,
                                       work.data(), &lwork, &info);
            MFEM_VERIFY(info == 0, ""); // Q^T b work calculation failed
            lwork = static_cast<int>(work[0]);
            work.resize(lwork);
            MFEM_LAPACK_PREFIX(ormqr_)(&lside, &trans, &m, &ione, &n_glob,
                                       mat_qr_data.GetData(), &m, tau.GetData(),
                                       qt_rhs_glob.GetData(), &m,
                                       work.data(), &lwork, &info);
            MFEM_VERIFY(info == 0, ""); // Q^T b failed
         }

         if (verbosity_ > 2)
         {
            mfem::out << "Updated rhs " << iiter << endl;
         }

         // Apply R^{-1}; first n_glob entries of vec1 are overwritten
         char upper = 'U';
         char nounit = 'N';
         vec1 = qt_rhs_glob;
         MFEM_LAPACK_PREFIX(trsm_)(&lside, &upper, &notrans, &nounit,
                                   &n_glob, &ione, &fone,
                                   mat_qr_data.GetData(), &m,
                                   vec1.GetData(), &n_glob);

         if (verbosity_ > 2)
         {
            mfem::out << "Solved triangular system " << iiter << endl;
         }

         // Check if all entries are positive
         int pos_ibool = 0;
         real_t smin = n_glob > 0 ? vec1(0) : 0.0;
         for (int i=0; i<n_glob; ++i)
         {
            soln_nz_glob_up(i) = vec1(i);
            smin = std::min(smin, soln_nz_glob_up(i));
         }

         if (smin > zero_tol_)
         {
            pos_ibool = 1;
            for (int i=0; i<n_glob; ++i)
            {
               soln_nz_glob(i) = soln_nz_glob_up(i);
            }
         }

         if (pos_ibool == 1)
         {
            break;
         }

         if (verbosity_ > 2)
         {
            mfem::out << "Start pruning " << iiter << endl;
            for (int i = 0; i < n_glob; ++i)
            {
               if (soln_nz_glob_up(i) <= zero_tol_)
               {
                  mfem::out << i << " " << n_glob << " " << soln_nz_glob_up(i) << endl;
               }
            }
         }

         if (soln_nz_glob_up(n_glob - 1) <= zero_tol_)
         {
            stalledFlag = 1;
            if (verbosity_ > 2)
            {
               if (qr_residual_mode_ == QRresidualMode::hybrid)
               {
                  mfem::out << "Detected stall due to adding and removing same "
                            << "column. Switching to QR residual calculation "
                            << "method." << endl;
               }
               else
               {
                  mfem::out << "Detected stall due to adding and removing same"
                            << " column. Exiting now." << endl;
               }
            }
         }

         if (stalledFlag == 1 && qr_residual_mode_ == QRresidualMode::hybrid)
         {
            NNLS_qrres_on_ = true;
            break;
         }

         real_t alpha = numeric_limits<real_t>::max();

         // Find maximum permissible step
         for (int i = 0; i < n_glob; ++i)
         {
            if (soln_nz_glob_up(i) <= zero_tol_)
            {
               alpha = std::min(alpha, soln_nz_glob(i)/(soln_nz_glob(i) - soln_nz_glob_up(i)));
            }
         }
         // Update solution
         smin = 0.0;
         for (int i = 0; i < n_glob; ++i)
         {
            soln_nz_glob(i) += alpha*(soln_nz_glob_up(i) - soln_nz_glob(i));
            if (i == 0 || soln_nz_glob(i) < smin)
            {
               smin = soln_nz_glob(i);
            }
         }

         while (smin > zero_tol_)
         {
            // This means there was a rounding error, as we should have
            // a zero element by definition. Recalculate alpha based on
            // the index that corresponds to the element that should be
            // zero.

            int index_min = 0;
            smin = soln_nz_glob(0);
            for (int i = 1; i < n_glob; ++i)
            {
               if (soln_nz_glob(i) < smin)
               {
                  smin = soln_nz_glob(i);
                  index_min = i;
               }
            }

            alpha = soln_nz_glob(index_min)/(soln_nz_glob(index_min)
                                             - soln_nz_glob_up(index_min));

            // Reupdate solution
            for (int i = 0; i < n_glob; ++i)
            {
               soln_nz_glob(i) += alpha*(soln_nz_glob_up(i) - soln_nz_glob(i));
            }
         }

         // Clean up zeroed entry
         i_qr_start = n_glob+1;
         while (true)
         {
            // Check if there is a zero entry
            int zero_ibool;

            smin = n_glob > 0 ? soln_nz_glob(0) : 0.0;
            for (int i=1; i<n_glob; ++i)
            {
               smin = std::min(smin, soln_nz_glob(i));
            }

            if (smin < zero_tol_)
            {
               zero_ibool = 1;
            }
            else
            {
               zero_ibool = 0;
            }

            if (zero_ibool == 0)   // Break if there is no more zero entry
            {
               break;
            }

            int ind_zero = -1; // Index where the first zero is encountered
            nz_ind_zero = 0;

            // Identify global index of the zeroed element
            for (int i = 0; i < n_glob; ++i)
            {
               if (soln_nz_glob(i) < zero_tol_)
               {
                  ind_zero = i;
                  break;
               }
            }
            MFEM_VERIFY(ind_zero != -1, "");
            // Identify the local index for nz_ind to which the zeroed entry
            // belongs
            for (int i = 0; i < ind_zero; ++i)
            {
               ++nz_ind_zero;
            }

            {
               // Copy mat_0.cols[ind_zero+1,n_glob) to mat_qr.cols[ind_zero,n_glob-1)
               for (int i=0; i<m; ++i)
                  for (int j=ind_zero; j<n_glob-1; ++j)
                  {
                     mat_qr_data(i + (j*m)) = mat_0_data(i + ((j+1)*m));
                  }

               // Copy mat_qr.cols[ind_zero,n_glob-1) to
               // mat_0.cols[ind_zero,n_glob-1)
               for (int i=0; i<m; ++i)
                  for (int j=ind_zero; j<n_glob-1; ++j)
                  {
                     mat_0_data(i + (j*m)) = mat_qr_data(i + (j*m));
                  }
            }

            // Remove the zeroed entry from the local matrix index
            for (int i = nz_ind_zero; i < n_nz_ind-1; ++i)
            {
               nz_ind[i] = nz_ind[i+1];
            }
            --n_nz_ind;

            // Shift soln_nz_glob and proc_index
            for (int i = ind_zero; i < n_glob-1; ++i)
            {
               soln_nz_glob(i) = soln_nz_glob(i+1);
            }

            i_qr_start = std::min(i_qr_start, ind_zero);
            --n_glob;
         } // End of pruning loop

         if (verbosity_ > 2)
         {
            mfem::out << "Finished pruning " << iiter << endl;
         }
      } // End of inner loop

      // Check if we have stalled
      if (stalledFlag == 1)
      {
         --n_glob;
         --n_nz_ind;
         num_stalled = stalled_indices.size();
         stalled_indices.resize(num_stalled + 1);
         stalled_indices[num_stalled] = imax;
         if (verbosity_ > 2)
         {
            mfem::out << "Adding index " << imax << " to stalled index list "
                      << "of size " << num_stalled << endl;
         }
      }

      // Compute residual
      if (!NNLS_qrres_on_)
      {
         res_glob = rhs_avg_glob;
         real_t fmone = -1.0;
         MFEM_LAPACK_PREFIX(gemv_)(&notrans, &m, &n_glob, &fmone,
                                   mat_0_data.GetData(), &m,
                                   soln_nz_glob.GetData(), &ione, &fone,
                                   res_glob.GetData(), &ione);
      }
      else
      {
         // Compute residual using res = b - Q*Q^T*b, where Q is from an
         // economical QR decomposition
         lwork = -1;
         work.resize(10);
         qqt_rhs_glob = 0.0;
         for (int i=0; i<n_glob; ++i)
         {
            qqt_rhs_glob(i) = qt_rhs_glob(i);
         }

         MFEM_LAPACK_PREFIX(ormqr_)(&lside, &notrans, &m, &ione, &n_glob,
                                    mat_qr_data.GetData(), &m,
                                    tau.GetData(), qqt_rhs_glob.GetData(), &m,
                                    work.data(), &lwork, &info);

         MFEM_VERIFY(info == 0, ""); // Q Q^T b work calculation failed.
         lwork = static_cast<int>(work[0]);
         work.resize(lwork);
         MFEM_LAPACK_PREFIX(ormqr_)(&lside, &notrans, &m, &ione, &n_glob,
                                    mat_qr_data.GetData(), &m,
                                    tau.GetData(), qqt_rhs_glob.GetData(), &m,
                                    work.data(), &lwork, &info);
         MFEM_VERIFY(info == 0, ""); // Q Q^T b calculation failed.
         res_glob = rhs_avg_glob;
         res_glob -= qqt_rhs_glob;
      }

      if (verbosity_ > 2)
      {
         mfem::out << "Computed residual" << endl;
      }

      ++n_outer_iter;
   } // End of outer loop

   // Insert the solutions
   MFEM_VERIFY(n_glob == n_nz_ind, "");
   soln = 0.0;
   for (int i = 0; i < n_glob; ++i)
   {
      soln(nz_ind[i]) = soln_nz_glob(i);
   }

   if (verbosity_ > 0)
   {
      mfem::out << "NNLS solver: m = " << m << ", n = " << n
                << ", outer_iter = " << n_outer_iter << ", inner_iter = "
                << n_total_inner_iter;

      if (exit_flag == 0)
      {
         mfem::out << ": converged" << endl;
      }
      else
      {
         mfem::out << endl << "Warning, NNLS convergence stalled: "
                   << (exit_flag == 2) << endl;
         mfem::out << "resErr = " << rmax << " vs tol = " << const_tol_
                   << "; mumax = " << mumax << " vs tol = " << mu_tol << endl;
      }
   }
}
#endif // MFEM_USE_LAPACK

}
