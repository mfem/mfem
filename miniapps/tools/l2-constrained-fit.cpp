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
//
//   ------------------------------------------------------------------------
//   Parallel Miniapp: Element-local constrained L2 fit in a DG/L2 space
//   ------------------------------------------------------------------------
//
// This miniapp computes, element by element, the constrained fit
//
//   minimize ||E - E*||_{L2(K)}
//
// subject to either
//
//   1. E >= 0 at the element DOFs, or
//   2. E >= 0 at the element DOFs and L E >= 0,
//
// where L is the lower-bound matrix produced by PLBound for the chosen
// tensor-product L2 basis.
//
// The analytic target E* is a simple trigonometric field. The unconstrained
// target in the discrete space is the elementwise L2 projection of E*. The
// constrained correction is then the projection of the local coefficient vector
// onto the feasible set in the element mass norm.
//
// The local QP is solved without external libraries using either cyclic dual
// coordinate descent on the convex dual problem or a dense primal active-set
// method.
//
// Compile with: make l2-constrained-fit
//
// Sample runs:
//   mpirun -np 4 l2-constrained-fit
//   mpirun -np 4 l2-constrained-fit -o 4 -r 2 -no-vis
//   mpirun -np 4 l2-constrained-fit -no-plb
//   mpirun -np 4 l2-constrained-fit -m ../../data/periodic-square.mesh -no-vis

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace mfem;
using namespace std;

namespace
{

int func_type = 0;

struct LocalSolveInfo
{
   int iterations = 0;
   real_t min_constraint = 0.0;
   bool converged = false;
};

enum LocalSolverType
{
   COORDINATE_DESCENT = 0,
   ACTIVE_SET = 1
};

const char *GetFunctionDescription(const int type)
{
   switch (type)
   {
      case 0:
         return "mixed-sign trigonometric sin-cos field";
      case 1:
         return "2D solid-body-rotation benchmark profile";
      case 2:
         return "positive high-power sine bubble";
      case 3:
         return "positive sharp front with transverse modulation";
      default:
         return "unknown";
   }
}

const char *GetLocalSolverDescription(const int type)
{
   switch (type)
   {
      case COORDINATE_DESCENT:
         return "dual cyclic coordinate descent";
      case ACTIVE_SET:
         return "dense primal active-set";
      default:
         return "unknown";
   }
}

real_t ExactFunction(const Vector &x)
{
   constexpr real_t pi = 3.1415926535897932384626433832795028841971694;
   constexpr real_t two_pi = 6.2831853071795864769252867665590057683943388;

   const real_t x0 = x(0);
   const real_t x1 = (x.Size() > 1) ? x(1) : 0.0;
   const real_t x2 = (x.Size() > 2) ? x(2) : 0.0;

   switch (func_type)
   {
      case 0:
      {
         real_t value = std::sin(two_pi * x0);
         for (int d = 1; d < x.Size(); d++)
         {
            value *= std::cos(two_pi * x(d));
         }
         return value;
      }
      case 1:
      {
         const real_t r_hump = std::hypot(x0 - 0.25, x1 - 0.50);
         if (r_hump <= 0.15)
         {
            return 0.25 + 0.25 * std::cos(pi * r_hump / 0.15);
         }

         const real_t r_cone = std::hypot(x0 - 0.50, x1 - 0.25);
         if (r_cone <= 0.15)
         {
            return 1.0 - r_cone / 0.15;
         }

         const real_t r_slot = std::hypot(x0 - 0.50, x1 - 0.75);
         if (r_slot <= 0.15 &&
             (std::abs(x0 - 0.50) >= 0.025 || x1 >= 0.85))
         {
            return 1.0;
         }

         return 0.0;
      }
      case 2:
      {
         real_t value = std::sin(pi * x0);
         for (int d = 1; d < x.Size(); d++)
         {
            value *= std::sin(pi * x(d));
         }
         value = std::max<real_t>(0.0, value);
         return std::pow(value, 8.0);
      }
      case 3:
      {
         const real_t cross =
            (x.Size() > 1) ? 0.12 * std::sin(two_pi * x1) : 0.0;
         const real_t front = 0.5 * (1.0 + std::tanh(30.0 * (0.38 - x0 + cross)));
         real_t mod = 1.0;
         if (x.Size() > 1)
         {
            mod *= 0.15 + 0.85 * std::pow(std::sin(pi * x1), 4.0);
         }
         if (x.Size() > 2)
         {
            mod *= 0.20 + 0.80 * std::pow(std::sin(pi * x2), 4.0);
         }
         return front * mod;
      }
      default:
         MFEM_ABORT("Unknown func_type.");
   }

   return 0.0;
}

void VisualizeField(ParMesh &pmesh, ParGridFunction &field,
                    const char *title, int pos_x, int pos_y)
{
   socketstream sock;
   if (pmesh.GetMyRank() == 0)
   {
      sock.open("localhost", 19916);
      sock << "solution\n";
   }
   pmesh.PrintAsOne(sock);
   field.SaveAsOne(sock);
   if (pmesh.GetMyRank() == 0)
   {
      sock << "window_title '" << title << "'\n"
           << "window_geometry "
           << pos_x << " " << pos_y << " " << 400 << " " << 400 << "\n"
           << "keys jRmclA" << endl;
   }
}

void BuildConstraintMatrix(const int ndof,
                           const DenseMatrix &L,
                           const bool use_plb_constraint,
                           const DenseMatrix &U,
                           const bool use_upper_constraint,
                           const real_t upper_bound,
                           DenseMatrix &A,
                           Vector &constraint_rhs)
{
   const int num_plb_rows = use_plb_constraint ? L.Height() : 0;
   const int num_upper_rows = use_upper_constraint ? U.Height() : 0;
   A.SetSize(ndof + num_plb_rows + num_upper_rows, ndof);
   A = 0.0;
   constraint_rhs.SetSize(A.Height());
   constraint_rhs = 0.0;

   for (int i = 0; i < ndof; i++)
   {
      A(i, i) = 1.0;
   }

   for (int i = 0; i < num_plb_rows; i++)
   {
      for (int j = 0; j < ndof; j++)
      {
         A(ndof + i, j) = L(i, j);
      }
   }

   for (int i = 0; i < num_upper_rows; i++)
   {
      for (int j = 0; j < ndof; j++)
      {
         A(ndof + num_plb_rows + i, j) = -U(i, j);
      }
      constraint_rhs(ndof + num_plb_rows + i) = -upper_bound;
   }
}

real_t ComputeMinLinearConstraint(const DenseMatrix &A, const Vector &x)
{
   if (A.Height() == 0)
   {
      return numeric_limits<real_t>::infinity();
   }

   Vector values(A.Height());
   A.Mult(x, values);
   return values.Min();
}

real_t ComputeMaxLinearConstraint(const DenseMatrix &A, const Vector &x)
{
   if (A.Height() == 0)
   {
      return -numeric_limits<real_t>::infinity();
   }

   Vector values(A.Height());
   A.Mult(x, values);
   return values.Max();
}

real_t ComputeMinAffineConstraint(const DenseMatrix &A,
                                  const Vector &constraint_rhs,
                                  const Vector &x)
{
   if (A.Height() == 0)
   {
      return numeric_limits<real_t>::infinity();
   }

   Vector values(A.Height());
   A.Mult(x, values);
   values -= constraint_rhs;
   return values.Min();
}

void BuildConstraintData(const DenseMatrixInverse &M_inv,
                         const DenseMatrix &A,
                         vector<Vector> &rows,
                         vector<Vector> &minv_rows)
{
   const int num_constraints = A.Height();
   const int ndof = A.Width();

   rows.resize(num_constraints);
   minv_rows.resize(num_constraints);

   for (int i = 0; i < num_constraints; i++)
   {
      rows[i].SetSize(ndof);
      minv_rows[i].SetSize(ndof);
      A.GetRow(i, rows[i]);
      M_inv.Mult(rows[i], minv_rows[i]);
   }
}

void BuildWorkingSetSchur(const vector<int> &working_set,
                          const vector<Vector> &rows,
                          const vector<Vector> &minv_rows,
                          DenseMatrix &schur)
{
   const int wsize = static_cast<int>(working_set.size());
   schur.SetSize(wsize);
   for (int i = 0; i < wsize; i++)
   {
      for (int j = 0; j < wsize; j++)
      {
         schur(i, j) = rows[working_set[i]] * minv_rows[working_set[j]];
      }
   }
}

void SolveWorkingSetSystem(const vector<int> &working_set,
                           const vector<Vector> &rows,
                           const vector<Vector> &minv_rows,
                           const Vector &primal_minus_target,
                           Vector &multipliers)
{
   const int wsize = static_cast<int>(working_set.size());
   multipliers.SetSize(wsize);
   if (wsize == 0)
   {
      multipliers = 0.0;
      return;
   }

   DenseMatrix schur;
   Vector rhs(wsize);
   BuildWorkingSetSchur(working_set, rows, minv_rows, schur);

   for (int i = 0; i < wsize; i++)
   {
      rhs(i) = rows[working_set[i]] * primal_minus_target;
   }

   DenseMatrixInverse schur_inv(schur, false);
   schur_inv.Mult(rhs, multipliers);
}

bool IsIndependentConstraint(const vector<int> &working_set,
                             const int candidate,
                             const vector<Vector> &rows,
                             const vector<Vector> &minv_rows,
                             const real_t dep_tol)
{
   const real_t self_inner = rows[candidate] * minv_rows[candidate];
   if (working_set.empty())
   {
      return self_inner > dep_tol;
   }

   DenseMatrix schur;
   Vector coupling(static_cast<int>(working_set.size()));
   Vector coefficients;
   BuildWorkingSetSchur(working_set, rows, minv_rows, schur);

   for (int i = 0; i < coupling.Size(); i++)
   {
      coupling(i) = rows[working_set[i]] * minv_rows[candidate];
   }

   DenseMatrixInverse schur_inv(schur, false);
   schur_inv.Mult(coupling, coefficients);

   const real_t schur_complement = self_inner - (coupling * coefficients);
   return schur_complement > dep_tol;
}

LocalSolveInfo SolveLocalConstrainedFitCoordinateDescent(
   const DenseMatrixInverse &M_inv,
   const Vector &target,
   const DenseMatrix &A,
   const Vector &constraint_rhs,
   const int max_iterations,
   const real_t abs_tol,
   const real_t rel_tol,
   Vector &x)
{
   const int num_constraints = A.Height();
   const int ndof = A.Width();

   MFEM_VERIFY(target.Size() == ndof, "Target size mismatch.");
   MFEM_VERIFY(constraint_rhs.Size() == num_constraints,
               "Constraint rhs size mismatch.");

   if (num_constraints == 0)
   {
      x = target;
      return LocalSolveInfo();
   }

   vector<Vector> rows(num_constraints), minv_rows(num_constraints);
   DenseMatrix S(num_constraints);
   Vector lambda(num_constraints), grad(num_constraints);
   vector<int> work_indices;
   lambda = 0.0;

   BuildConstraintData(M_inv, A, rows, minv_rows);

   for (int i = 0; i < num_constraints; i++)
   {
      grad(i) = rows[i] * target - constraint_rhs(i);
      for (int j = 0; j < num_constraints; j++)
      {
         S(j, i) = rows[j] * minv_rows[i];
      }
      MFEM_VERIFY(S(i, i) > 0.0, "Constraint matrix contains a zero row.");
   }

   const real_t grad_scale = std::max<real_t>(1.0, grad.Normlinf());
   const real_t tol = std::max(abs_tol, rel_tol * grad_scale);
   const real_t work_tol = 0.0;
   const int initial_full_sweeps = 2;

   auto BuildWorkIndices = [&](const bool force_full)
   {
      work_indices.clear();
      if (force_full)
      {
         work_indices.resize(num_constraints);
         for (int i = 0; i < num_constraints; i++)
         {
            work_indices[i] = i;
         }
         return;
      }

      for (int i = 0; i < num_constraints; i++)
      {
         if (lambda(i) > 0.0 || grad(i) < work_tol)
         {
            work_indices.push_back(i);
         }
      }
   };

   LocalSolveInfo info;
   info.min_constraint = grad.Min();
   BuildWorkIndices(true);

   for (int sweep = 0; sweep < max_iterations; sweep++)
   {
      real_t max_update = 0.0;

      if (work_indices.empty())
      {
         info.converged = true;
         break;
      }

      for (const int i : work_indices)
      {
         const real_t new_lambda =
            std::max<real_t>(0.0, lambda(i) - grad(i) / S(i, i));
         const real_t delta = new_lambda - lambda(i);

         if (delta == 0.0) { continue; }

         lambda(i) = new_lambda;
         max_update = std::max(max_update, std::abs(delta));

         for (int j = 0; j < num_constraints; j++)
         {
            grad(j) += delta * S(j, i);
         }
      }

      info.iterations = sweep + 1;
      info.min_constraint = grad.Min();
      const real_t max_violation = std::max<real_t>(0.0, -info.min_constraint);

      if (max_violation <= tol && max_update <= tol)
      {
         info.converged = true;
         break;
      }

      BuildWorkIndices(sweep + 1 < initial_full_sweeps);
   }

   x = target;
   for (int i = 0; i < num_constraints; i++)
   {
      if (lambda(i) != 0.0)
      {
         x.Add(lambda(i), minv_rows[i]);
      }
   }

   return info;
}

LocalSolveInfo SolveLocalConstrainedFitActiveSet(
   const DenseMatrixInverse &M_inv,
   const Vector &target,
   const DenseMatrix &A,
   const Vector &constraint_rhs,
   const int max_iterations,
   const real_t abs_tol,
   const real_t rel_tol,
   Vector &x)
{
   const int num_constraints = A.Height();
   const int ndof = A.Width();

   MFEM_VERIFY(target.Size() == ndof, "Target size mismatch.");
   MFEM_VERIFY(constraint_rhs.Size() == num_constraints,
               "Constraint rhs size mismatch.");

   if (num_constraints == 0)
   {
      x = target;
      return LocalSolveInfo();
   }

   vector<Vector> rows, minv_rows;
   BuildConstraintData(M_inv, A, rows, minv_rows);

   x.SetSize(ndof);
   x = 0.0;

   const real_t initial_min_constraint =
      ComputeMinAffineConstraint(A, constraint_rhs, x);
   MFEM_VERIFY(initial_min_constraint >= -abs_tol,
               "The active-set solver requires a feasible zero initial guess.");

   real_t scale = std::max<real_t>(1.0, target.Normlinf());
   for (int i = 0; i < num_constraints; i++)
   {
      scale = std::max(scale, std::abs(constraint_rhs(i)));
   }
   const real_t tol = std::max(abs_tol, rel_tol * scale);
   const real_t dep_tol =
      std::max<real_t>(10.0 * abs_tol,
                       100.0 * numeric_limits<real_t>::epsilon());

   vector<int> working_set;
   vector<char> in_working_set(num_constraints, 0);
   Vector primal_minus_target(ndof), step(ndof), multipliers;

   for (int i = 0; i < num_constraints; i++)
   {
      const real_t zero_slack = -constraint_rhs(i);
      if (std::abs(zero_slack) <= tol &&
          IsIndependentConstraint(working_set, i, rows, minv_rows, dep_tol))
      {
         in_working_set[i] = 1;
         working_set.push_back(i);
      }
   }

   LocalSolveInfo info;
   info.min_constraint = initial_min_constraint;

   for (int iter = 0; iter < max_iterations; iter++)
   {
      info.iterations = iter + 1;

      primal_minus_target = x;
      primal_minus_target -= target;

      step = target;
      step -= x;

      if (!working_set.empty())
      {
         SolveWorkingSetSystem(working_set, rows, minv_rows,
                               primal_minus_target, multipliers);
         for (int i = 0; i < multipliers.Size(); i++)
         {
            step.Add(multipliers(i), minv_rows[working_set[i]]);
         }
      }

      if (step.Normlinf() <= tol)
      {
         if (working_set.empty())
         {
            info.converged = true;
            break;
         }

         SolveWorkingSetSystem(working_set, rows, minv_rows,
                               primal_minus_target, multipliers);

         int drop_index = -1;
         real_t most_negative_multiplier = -tol;
         for (int i = 0; i < multipliers.Size(); i++)
         {
            if (multipliers(i) < most_negative_multiplier)
            {
               most_negative_multiplier = multipliers(i);
               drop_index = i;
            }
         }

         if (drop_index < 0)
         {
            info.converged = true;
            break;
         }

         in_working_set[working_set[drop_index]] = 0;
         working_set.erase(working_set.begin() + drop_index);
         continue;
      }

      real_t alpha = 1.0;
      int blocking_constraint = -1;

      for (int i = 0; i < num_constraints; i++)
      {
         if (in_working_set[i]) { continue; }

         const real_t row_step = rows[i] * step;
         if (row_step >= -tol) { continue; }

         const real_t slack = rows[i] * x - constraint_rhs(i);
         const real_t candidate_alpha = slack / (-row_step);
         if (candidate_alpha < alpha &&
             IsIndependentConstraint(working_set, i, rows, minv_rows, dep_tol))
         {
            alpha = std::max<real_t>(0.0, candidate_alpha);
            blocking_constraint = i;
         }
      }

      x.Add(alpha, step);

      if (blocking_constraint >= 0 && alpha < 1.0 - tol)
      {
         in_working_set[blocking_constraint] = 1;
         working_set.push_back(blocking_constraint);
      }

      info.min_constraint = ComputeMinAffineConstraint(A, constraint_rhs, x);
   }

   info.min_constraint = ComputeMinAffineConstraint(A, constraint_rhs, x);
   return info;
}

LocalSolveInfo SolveLocalConstrainedFit(const int solver_type,
                                        const DenseMatrixInverse &M_inv,
                                        const Vector &target,
                                        const DenseMatrix &A,
                                        const Vector &constraint_rhs,
                                        const int max_iterations,
                                        const real_t abs_tol,
                                        const real_t rel_tol,
                                        Vector &x)
{
   switch (solver_type)
   {
      case COORDINATE_DESCENT:
         return SolveLocalConstrainedFitCoordinateDescent(
                   M_inv, target, A, constraint_rhs, max_iterations,
                   abs_tol, rel_tol, x);
      case ACTIVE_SET:
         return SolveLocalConstrainedFitActiveSet(
                   M_inv, target, A, constraint_rhs, max_iterations,
                   abs_tol, rel_tol, x);
      default:
         MFEM_ABORT("Unknown local solver type.");
   }
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "";
   int dim = 2;
   int nx = 8;
   int ny = 8;
   int nz = 8;
   int ref_levels = 0;
   int order = 3;
   func_type = 0;
   int ncp = -1;
   int cp_type = 0;
   int rhs_qorder = -1;
   int local_solver_type = COORDINATE_DESCENT;
   int max_sweeps = 4000;
   real_t abs_tol = 1.0e-12;
   real_t rel_tol = 1.0e-10;
   bool use_plb_constraint = true;
   bool use_upper_constraint = false;
   real_t upper_bound = 1.0;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. If empty, build a Cartesian tensor mesh.");
   args.AddOption(&dim, "-dim", "--dimension",
                  "Dimension for the built-in Cartesian mesh (2 or 3).");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in x for the built-in mesh.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in y for the built-in mesh.");
   args.AddOption(&nz, "-nz", "--num-elements-z",
                  "Number of elements in z for the built-in mesh.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform serial refinements.");
   args.AddOption(&order, "-o", "--order",
                  "L2 finite element order.");
   args.AddOption(&func_type, "-ftype", "--function-type",
                  "Analytic target function type:\n"
                  "\t0: mixed-sign trigonometric sin-cos field\n"
                  "\t1: 2D solid-body-rotation benchmark profile\n"
                  "\t2: positive high-power sine bubble\n"
                  "\t3: positive sharp front with transverse modulation");
   args.AddOption(&ncp, "-ncp", "--num-control-points",
                  "Number of PLBound control points in 1D. Use -1 for auto.");
   args.AddOption(&cp_type, "-cpt", "--control-point-type",
                  "PLBound control point type: 0 = GL+endpoints, 1 = Chebyshev.");
   args.AddOption(&rhs_qorder, "-qo", "--quadrature-order",
                  "Quadrature order for the analytic RHS projection. Use -1 for"
                  " the default 2*order+8.");
   args.AddOption(&local_solver_type, "-ls", "--local-solver",
                  "Local QP solver type:\n"
                  "\t0: dual cyclic coordinate descent\n"
                  "\t1: dense primal active-set");
   args.AddOption(&max_sweeps, "-ms", "--max-sweeps",
                  "Maximum number of local solver iterations. For coordinate "
                  "descent this is the number of sweeps.");
   args.AddOption(&abs_tol, "-atol", "--absolute-tolerance",
                  "Absolute tolerance for the local solver.");
   args.AddOption(&rel_tol, "-rtol", "--relative-tolerance",
                  "Relative tolerance for the local solver.");
   args.AddOption(&use_plb_constraint, "-plb", "--use-plb-constraint",
                  "-no-plb", "--nodal-only",
                  "Use both E >= 0 and L E >= 0, or only E >= 0.");
   args.AddOption(&use_upper_constraint, "-ub", "--use-upper-bound-constraint",
                  "-no-ub", "--no-upper-bound-constraint",
                  "Enable or disable the PLBound upper constraint U E <= upper.");
   args.AddOption(&upper_bound, "-upper", "--upper-bound-value",
                  "Upper bound value used when -ub is enabled.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
      cout << "Selected function: " << func_type << " ("
           << GetFunctionDescription(func_type) << ")\n";
      cout << "Selected local solver: " << local_solver_type << " ("
           << GetLocalSolverDescription(local_solver_type) << ")\n";
   }

   MFEM_VERIFY(local_solver_type == COORDINATE_DESCENT ||
               local_solver_type == ACTIVE_SET,
               "Unsupported local solver type.");
   MFEM_VERIFY(!use_upper_constraint || upper_bound >= 0.0,
               "The constraints E >= 0 and U E <= upper are infeasible for "
               "upper < 0.");

   Mesh serial_mesh;
   if (std::string(mesh_file).empty())
   {
      MFEM_VERIFY(dim == 2 || dim == 3,
                  "The built-in mesh path supports only dim = 2 or 3.");
      serial_mesh = (dim == 2)
                    ? Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                            true, 1.0, 1.0)
                    : Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON,
                                            1.0, 1.0, 1.0);
   }
   else
   {
      serial_mesh = Mesh(mesh_file, 1, 1);
      dim = serial_mesh.Dimension();
   }

   for (int l = 0; l < ref_levels; l++)
   {
      serial_mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   L2_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec);

   MFEM_VERIFY(fes.GetVDim() == 1, "Scalar L2 space expected.");
   MFEM_VERIFY(!fes.IsVariableOrder(), "Variable-order spaces are not supported.");

   const FiniteElement *typical_fe = fes.GetTypicalFE();
   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement *>(typical_fe);
   MFEM_VERIFY(tbe != NULL,
               "This miniapp requires a tensor-product L2 finite element.");

   const int ndof = typical_fe->GetDof();
   if (rhs_qorder < 0)
   {
      rhs_qorder = 2 * order + 8;
   }

   FunctionCoefficient exact_coeff(ExactFunction);
   PLBound plb(&fes, ncp, cp_type);
   DenseMatrix L = plb.GetLowerBoundMatrix(dim, &fes);
   DenseMatrix U = plb.GetUpperBoundMatrix(dim, &fes);
   DenseMatrix A;
   Vector constraint_rhs;
   BuildConstraintMatrix(ndof, L, use_plb_constraint, U, use_upper_constraint,
                         upper_bound, A, constraint_rhs);

   ParGridFunction exact_vis_gf(&fes);
   ParGridFunction target_gf(&fes);
   ParGridFunction constrained_gf(&fes);
   exact_vis_gf.ProjectCoefficient(exact_coeff);
   target_gf = 0.0;
   constrained_gf = 0.0;

   MassIntegrator mass_integrator;
   DomainLFIntegrator rhs_integrator(
      exact_coeff, &IntRules.Get(typical_fe->GetGeomType(), rhs_qorder));

   real_t local_min_target_dof = numeric_limits<real_t>::infinity();
   real_t local_min_target_plb = numeric_limits<real_t>::infinity();
   real_t local_min_constrained_dof = numeric_limits<real_t>::infinity();
   real_t local_min_constrained_plb = numeric_limits<real_t>::infinity();
   real_t local_max_target_dof = -numeric_limits<real_t>::infinity();
   real_t local_max_constrained_dof = -numeric_limits<real_t>::infinity();
   real_t local_max_target_pub = -numeric_limits<real_t>::infinity();
   real_t local_max_constrained_pub = -numeric_limits<real_t>::infinity();
   int local_max_used_sweeps = 0;
   int local_num_failed = 0;

   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      const FiniteElement &fe = *fes.GetFE(e);
      ElementTransformation &Tr = *fes.GetElementTransformation(e);

      DenseMatrix Mloc;
      Vector rhs(ndof), target_loc(ndof), constrained_loc(ndof);
      Array<int> dofs;

      mass_integrator.AssembleElementMatrix(fe, Tr, Mloc);
      rhs_integrator.AssembleRHSElementVect(fe, Tr, rhs);

      DenseMatrixInverse M_inv(Mloc, true);
      M_inv.Mult(rhs, target_loc);

      const LocalSolveInfo info =
         SolveLocalConstrainedFit(local_solver_type, M_inv, target_loc, A,
                                  constraint_rhs, max_sweeps,
                                  abs_tol, rel_tol, constrained_loc);

      fes.GetElementDofs(e, dofs);
      target_gf.SetSubVector(dofs, target_loc);
      constrained_gf.SetSubVector(dofs, constrained_loc);

      local_min_target_dof = std::min(local_min_target_dof, target_loc.Min());
      local_max_target_dof = std::max(local_max_target_dof, target_loc.Max());
      local_min_constrained_dof =
         std::min(local_min_constrained_dof, constrained_loc.Min());
      local_max_constrained_dof =
         std::max(local_max_constrained_dof, constrained_loc.Max());

      if (L.Height() > 0)
      {
         local_min_target_plb = std::min(local_min_target_plb,
                                         ComputeMinLinearConstraint(L, target_loc));
         local_min_constrained_plb =
            std::min(local_min_constrained_plb,
                     ComputeMinLinearConstraint(L, constrained_loc));
      }

      if (U.Height() > 0)
      {
         local_max_target_pub = std::max(local_max_target_pub,
                                         ComputeMaxLinearConstraint(U, target_loc));
         local_max_constrained_pub =
            std::max(local_max_constrained_pub,
                     ComputeMaxLinearConstraint(U, constrained_loc));
      }

      local_max_used_sweeps = std::max(local_max_used_sweeps, info.iterations);
      local_num_failed += info.converged ? 0 : 1;
   }

   real_t global_min_target_dof = local_min_target_dof;
   real_t global_min_target_plb = local_min_target_plb;
   real_t global_min_constrained_dof = local_min_constrained_dof;
   real_t global_min_constrained_plb = local_min_constrained_plb;
   real_t global_max_target_dof = local_max_target_dof;
   real_t global_max_constrained_dof = local_max_constrained_dof;
   real_t global_max_target_pub = local_max_target_pub;
   real_t global_max_constrained_pub = local_max_constrained_pub;
   int global_max_used_sweeps = local_max_used_sweeps;
   int global_num_failed = local_num_failed;

   MPI_Allreduce(MPI_IN_PLACE, &global_min_target_dof, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &global_min_target_plb, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &global_min_constrained_dof, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &global_min_constrained_plb, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &global_max_target_dof, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &global_max_constrained_dof, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &global_max_target_pub, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &global_max_constrained_pub, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &global_max_used_sweeps, 1,
                 MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &global_num_failed, 1,
                 MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   const real_t target_l2_error = target_gf.ComputeL2Error(exact_coeff);
   const real_t constrained_l2_error = constrained_gf.ComputeL2Error(exact_coeff);
   const real_t final_plb_violation =
      std::max<real_t>(0.0, -global_min_constrained_plb);
   const real_t final_upper_violation =
      std::max<real_t>(0.0, global_max_constrained_pub - upper_bound);
   const HYPRE_BigInt global_true_vsize = fes.GlobalTrueVSize();

   if (Mpi::Root())
   {
      cout << "\nConstraint mode: "
           << (use_plb_constraint ? "E >= 0 and L E >= 0" : "E >= 0 only")
           << "\n";
      cout << "Local solver: "
           << GetLocalSolverDescription(local_solver_type) << "\n";
      cout << "Upper constraint mode: "
           << (use_upper_constraint ? "U E <= upper" : "disabled") << "\n";
      if (use_upper_constraint)
      {
         cout << "Upper bound value: " << upper_bound << "\n";
      }
      cout << "Global number of true DOFs: " << global_true_vsize << "\n";
      cout << "Element local dofs: " << ndof << "\n";
      cout << "PLBound control points in 1D: " << plb.GetNControlPoints() << "\n";
      cout << "Rows in L: " << L.Height() << "\n";
      cout << "Rows in U: " << U.Height() << "\n";
      cout << "Rows in local constraint matrix A: " << A.Height() << "\n";
      cout << "Unconstrained L2 projection error: " << target_l2_error << "\n";
      cout << "Constrained fit L2 error: " << constrained_l2_error << "\n";
      cout << "Minimum target DOF value: " << global_min_target_dof << "\n";
      cout << "Minimum constrained DOF value: "
           << global_min_constrained_dof << "\n";
      cout << "Maximum target DOF value: " << global_max_target_dof << "\n";
      cout << "Maximum constrained DOF value: "
           << global_max_constrained_dof << "\n";
      cout << "Minimum target L*E value: " << global_min_target_plb << "\n";
      cout << "Final min(L*e_K) over mesh: "
           << global_min_constrained_plb << "\n";
      cout << "Final PLBound constraint violation max(0, -min(L*e_K)): "
           << final_plb_violation << "\n";
      if (use_upper_constraint)
      {
         cout << "Maximum target U*E value: " << global_max_target_pub << "\n";
         cout << "Final max(U*e_K) over mesh: "
              << global_max_constrained_pub << "\n";
         cout << "Final upper constraint violation max(0, max(U*e_K)-upper): "
              << final_upper_violation << "\n";
      }
      cout << "Maximum local solver iterations used: "
           << global_max_used_sweeps << "\n";
      cout << "Number of locally unconverged solves: " << global_num_failed
           << "\n";
   }

   if (visualization)
   {
      char title0[] = "Original analytic field";
      char title1[] = "Optimized constrained fit";
      VisualizeField(pmesh, exact_vis_gf, title0, 0, 0);
      VisualizeField(pmesh, constrained_gf, title1, 420, 0);
   }

   return 0;
}
