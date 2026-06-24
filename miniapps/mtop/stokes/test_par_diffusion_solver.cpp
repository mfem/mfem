// test_par_diffusion_solver.cpp
//
// Standalone parallel MFEM driver for ParDiffusionSolver.  The main test solves
// a nonzero Dirichlet manufactured problem on [0,1]^d:
//
//     -div(kappa grad u) = f,
//                     u = u_exact on all boundary attributes,
//
// with u_exact(x) = exp(sum_i x_i) and f = -kappa*d*u_exact for constant kappa.
// The driver also clears the coefficient-valued boundary conditions, adds zero
// Dirichlet data with the constant-value overload, and solves the zero problem.

#include "mfem.hpp"
#include "par_diffusion_solver.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

using namespace mfem;
using namespace std;

namespace
{
/// Diffusion coefficient used by the manufactured RHS callback.
real_t g_kappa = 1.0;

/// Spatial dimension used by the manufactured RHS callback.
int g_dim = 2;

/**
 * @brief Evaluate the Dirichlet manufactured exact solution.
 *
 * The function is @c exp(sum_i x_i) on the unit box.  It gives nonzero,
 * spatially varying values on most boundary attributes, which exercises the
 * coefficient-valued boundary-condition interface.
 *
 * @param x Physical coordinates at which to evaluate the function.
 * @return Exact scalar solution value.
 */
real_t ExactDirichletSolution(const Vector &x)
{
   real_t s = 0.0;
   for (int d = 0; d < x.Size(); d++)
   {
      s += x(d);
   }
   return exp(s);
}

/**
 * @brief Evaluate the RHS for the manufactured Dirichlet problem.
 *
 * For @c u=exp(sum_i x_i) and constant @c kappa,
 * @c -div(kappa grad u) = -kappa*d*u.
 *
 * @param x Physical coordinates at which to evaluate the RHS.
 * @return Scalar RHS value.
 */
real_t DirichletRightHandSide(const Vector &x)
{
   return -g_kappa*g_dim*ExactDirichletSolution(x);
}

/**
 * @brief Compute the parallel Euclidean norm of a true-DOF vector.
 *
 * The helper uses MFEM's MPI-reduced true-vector inner product.
 *
 * @param comm MPI communicator for the vector distribution.
 * @param v True-DOF vector.
 * @return Parallel l2 norm of @a v.
 */
real_t ParNorml2(MPI_Comm comm, const Vector &v)
{
   return sqrt(InnerProduct(comm, v, v));
}

/**
 * @brief Fill an array with the boundary attributes present on a ParMesh.
 *
 * The returned array stores the actual one-based MFEM boundary attribute ids,
 * not a dense marker.  It is convenient for applying a boundary condition to
 * every boundary attribute without assuming a particular Cartesian attribute
 * ordering.
 *
 * @param pmesh Parallel mesh whose boundary attributes are requested.
 * @param attrs Output array of present boundary attribute ids.
 */
void GetBoundaryAttributes(const ParMesh &pmesh, Array<int> &attrs)
{
   attrs.SetSize(pmesh.bdr_attributes.Size());
   for (int i = 0; i < pmesh.bdr_attributes.Size(); i++)
   {
      attrs[i] = pmesh.bdr_attributes[i];
   }
}

/**
 * @brief Add the same coefficient-valued Dirichlet data to all attributes.
 *
 * This helper demonstrates the @c AddBoundaryCondition(id, Coefficient&)
 * overload.  It records all boundary conditions but does not call
 * @c solver.Assemble(); the caller does that once after all additions.
 *
 * @param attrs One-based boundary attribute ids.
 * @param coef Coefficient to impose on every listed boundary attribute.
 * @param solver Diffusion solver to modify.
 */
void AddCoefficientBCOnAllAttributes(const Array<int> &attrs,
                                     Coefficient &coef,
                                     ParDiffusionSolver &solver)
{
   for (int i = 0; i < attrs.Size(); i++)
   {
      solver.AddBoundaryCondition(attrs[i], coef);
   }
}

/**
 * @brief Add the same constant Dirichlet value to all attributes.
 *
 * This helper demonstrates the @c AddBoundaryCondition(id, real_t) overload.
 * It records all boundary conditions but leaves assembly to the caller.
 *
 * @param attrs One-based boundary attribute ids.
 * @param value Constant boundary value.
 * @param solver Diffusion solver to modify.
 */
void AddConstantBCOnAllAttributes(const Array<int> &attrs,
                                  real_t value,
                                  ParDiffusionSolver &solver)
{
   for (int i = 0; i < attrs.Size(); i++)
   {
      solver.AddBoundaryCondition(attrs[i], value);
   }
}

/**
 * @brief Save MFEM mesh/grid-function files from one parallel solve.
 *
 * Each MPI rank writes its local mesh and fields with a rank suffix.
 *
 * @param pmesh Parallel mesh to save.
 * @param uh Numerical solution field.
 * @param uex Projected exact solution field.
 * @param error Error field @c uh-uex.
 * @param prefix Filename prefix describing the solve.
 * @param myid MPI rank id.
 */
void SaveMFEMFiles(ParMesh &pmesh,
                   ParGridFunction &uh,
                   ParGridFunction &uex,
                   ParGridFunction &error,
                   const char *prefix,
                   int myid)
{
   ostringstream mesh_name, sol_name, exact_name, error_name;
   mesh_name << prefix << "_mesh." << setfill('0') << setw(6) << myid;
   sol_name << prefix << "_sol." << setfill('0') << setw(6) << myid;
   exact_name << prefix << "_exact." << setfill('0') << setw(6) << myid;
   error_name << prefix << "_error." << setfill('0') << setw(6) << myid;

   ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   pmesh.Print(mesh_ofs);

   ofstream sol_ofs(sol_name.str().c_str());
   sol_ofs.precision(16);
   uh.Save(sol_ofs);

   ofstream exact_ofs(exact_name.str().c_str());
   exact_ofs.precision(16);
   uex.Save(exact_ofs);

   ofstream error_ofs(error_name.str().c_str());
   error_ofs.precision(16);
   error.Save(error_ofs);
}

/**
 * @brief Save one ParaView data collection.
 *
 * The collection contains @c solution, @c exact, and @c error fields.
 *
 * @param pmesh Parallel mesh associated with the fields.
 * @param uh Numerical solution field.
 * @param uex Projected exact solution field.
 * @param error Error field @c uh-uex.
 * @param collection_name ParaView collection name.
 * @param prefix Output directory prefix.
 * @param binary Use binary VTK files when true; ASCII when false.
 * @param high_order Preserve high-order field/mesh output when true.
 * @param levels_of_detail ParaView levels of detail.
 * @param cycle Output cycle number.
 * @param time Output time value.
 */
void SaveParaView(ParMesh &pmesh,
                  ParGridFunction &uh,
                  ParGridFunction &uex,
                  ParGridFunction &error,
                  const char *collection_name,
                  const char *prefix,
                  bool binary,
                  bool high_order,
                  int levels_of_detail,
                  int cycle,
                  real_t time)
{
   ParaViewDataCollection pvdc(collection_name, &pmesh);
   pvdc.SetPrefixPath(prefix);
   pvdc.RegisterField("solution", &uh);
   pvdc.RegisterField("exact", &uex);
   pvdc.RegisterField("error", &error);
   pvdc.SetLevelsOfDetail(levels_of_detail);
   pvdc.SetDataFormat(binary ? VTKFormat::BINARY : VTKFormat::ASCII);
   pvdc.SetHighOrderOutput(high_order);
   pvdc.SetCycle(cycle);
   pvdc.SetTime(time);
   pvdc.Save();
}
}

/**
 * @brief Run the parallel Dirichlet diffusion solver test.
 *
 * The driver performs two solves with the same solver instance:
 *
 * 1. coefficient-valued Dirichlet data from the manufactured exact solution;
 * 2. after @c ClearBoundaryConditions(), constant zero Dirichlet data with a
 *    zero RHS.
 *
 * The first solve checks L2 error and the eliminated-system residual.  The
 * second solve checks that clearing and constant boundary conditions produce a
 * near-zero solution.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument array.
 * @return 0 if all checks pass, 1 for option parsing failure, 2 for invalid
 *         input options, or 3 if a numerical check fails.
 */
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   MPI_Comm comm = MPI_COMM_WORLD;
   const int myid = Mpi::WorldRank();

   int dim = 2;
   int nx = 8;
   int order = 2;
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   real_t kappa_value = 1.0;
   real_t cg_rel_tol = 1e-12;
   int cg_max_iter = 500;
   int cg_print = 0;
   real_t l2_tol = 1e-3;
   real_t residual_tol = 1e-8;
   real_t zero_solution_tol = 1e-10;
   bool save = false;
   bool paraview = false;
   const char *paraview_prefix = "ParaView";
   const char *paraview_name = "dirichlet_diffusion";
   bool paraview_binary = true;
   bool paraview_high_order = true;
   int paraview_lod = -1;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension",
                  "Dimension of the unit box test mesh: 2 or 3.");
   args.AddOption(&nx, "-n", "--num-elements",
                  "Number of elements per direction before refinement.");
   args.AddOption(&order, "-o", "--order",
                  "H1 finite element order.");
   args.AddOption(&ser_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of uniform serial refinements before partitioning.");
   args.AddOption(&par_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of uniform parallel refinements after partitioning.");
   args.AddOption(&kappa_value, "-k", "--kappa",
                  "Constant diffusion coefficient.");
   args.AddOption(&cg_rel_tol, "-rtol", "--relative-tolerance",
                  "CG relative tolerance.");
   args.AddOption(&cg_max_iter, "-mi", "--max-iterations",
                  "Maximum CG iterations.");
   args.AddOption(&cg_print, "-pl", "--print-level",
                  "CG print level.");
   args.AddOption(&l2_tol, "-l2tol", "--l2-tolerance",
                  "Pass/fail tolerance for the manufactured L2 error.");
   args.AddOption(&residual_tol, "-restol", "--residual-tolerance",
                  "Pass/fail tolerance for the relative eliminated residual.");
   args.AddOption(&zero_solution_tol, "-ztol", "--zero-solution-tolerance",
                  "Pass/fail tolerance for the zero-Dirichlet zero-RHS test.");
   args.AddOption(&save, "-s", "--save", "-no-s", "--no-save",
                  "Save parallel mesh and solution files in MFEM format.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Save ParaView/PVTU output.");
   args.AddOption(&paraview_prefix, "-pv-prefix", "--paraview-prefix",
                  "Directory prefix for ParaView output.");
   args.AddOption(&paraview_name, "-pv-name", "--paraview-name",
                  "Base ParaView data collection name.");
   args.AddOption(&paraview_binary, "-pvbin", "--paraview-binary",
                  "-pvtxt", "--paraview-ascii",
                  "Use binary or ASCII ParaView output.");
   args.AddOption(&paraview_high_order, "-pvho", "--paraview-high-order",
                  "-no-pvho", "--no-paraview-high-order",
                  "Use high-order ParaView output.");
   args.AddOption(&paraview_lod, "-pv-lod", "--paraview-levels-of-detail",
                  "ParaView output refinement level. Use -1 to match FE order.");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }

   if (dim != 2 && dim != 3)
   {
      if (myid == 0) { cerr << "ERROR: -dim must be 2 or 3.\n"; }
      return 2;
   }
   if (nx < 1 || order < 1)
   {
      if (myid == 0) { cerr << "ERROR: -n and -o must be positive.\n"; }
      return 2;
   }
   if (kappa_value <= 0.0)
   {
      if (myid == 0) { cerr << "ERROR: -k must be positive.\n"; }
      return 2;
   }
   if (paraview_lod == 0 || paraview_lod < -1)
   {
      if (myid == 0)
      {
         cerr << "ERROR: -pv-lod must be positive, or -1 to match FE order.\n";
      }
      return 2;
   }

   if (myid == 0) { args.PrintOptions(cout); }

   g_dim = dim;
   g_kappa = kappa_value;

   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL,
                                   true, 1.0, 1.0);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON,
                                   1.0, 1.0, 1.0);
   }

   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(comm, mesh);
   mesh.Clear();

   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec);

   Array<int> boundary_attributes;
   GetBoundaryAttributes(pmesh, boundary_attributes);
   if (boundary_attributes.Size() == 0)
   {
      if (myid == 0) { cerr << "ERROR: test mesh has no boundary attributes.\n"; }
      return 2;
   }

   {
      long int gd=fes.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Global true dofs: " << gd << '\n';
         cout << "Boundary attributes:";
      }
      for (int i = 0; i < boundary_attributes.Size(); i++)
      {
         if (myid == 0) { cout << ' ' << boundary_attributes[i];}
      }

      if (myid == 0) { cout << '\n';}
      
   }

   ConstantCoefficient kappa(kappa_value);
   FunctionCoefficient ucoef(ExactDirichletSolution);
   FunctionCoefficient fcoef(DirichletRightHandSide);
   ConstantCoefficient zero_coef(0.0);

   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(fcoef));
   b.Assemble();

   Vector B; B.SetSize(fes.GetTrueVSize());
   b.ParallelAssemble(B);

   ParDiffusionSolver solver(fes, kappa, cg_rel_tol, cg_max_iter, cg_print);

   AddCoefficientBCOnAllAttributes(boundary_attributes, ucoef, solver);
   solver.Assemble();

   Vector X;
   solver.Mult(B, X);
   const int manufactured_ess_tdofs = solver.GetEssentialTrueDofs().Size();

   Vector B_system;
   solver.FormSystemRHS(B, B_system);

   Vector residual(B.Size());
   solver.GetSystemMatrix().Mult(X, residual);
   residual -= B_system;

   const real_t system_rhs_norm = ParNorml2(comm, B_system);
   const real_t residual_norm = ParNorml2(comm, residual);
   const real_t relative_residual = residual_norm/max(system_rhs_norm, real_t(1.0));

   ParGridFunction uh(&fes);
   uh.SetFromTrueDofs(X);

   ParGridFunction uex_gf(&fes);
   uex_gf.ProjectCoefficient(ucoef);

   ParGridFunction error_gf(&fes);
   error_gf = uh;
   error_gf -= uex_gf;

   const real_t l2_error = uh.ComputeL2Error(ucoef);

   const bool pass_l2 = (l2_error <= l2_tol);
   const bool pass_residual = (relative_residual <= residual_tol);

   solver.ClearBoundaryConditions();
   AddConstantBCOnAllAttributes(boundary_attributes, 0.0, solver);
   solver.Assemble();

   Vector B_zero(B.Size());
   B_zero = 0.0;

   Vector X_zero;
   solver.Mult(B_zero, X_zero);

   const real_t zero_solution_norm = ParNorml2(comm, X_zero);
   const bool pass_zero = (zero_solution_norm <= zero_solution_tol);

   ParGridFunction zero_uh(&fes);
   zero_uh.SetFromTrueDofs(X_zero);

   ParGridFunction zero_exact_gf(&fes);
   zero_exact_gf.ProjectCoefficient(zero_coef);

   ParGridFunction zero_error_gf(&fes);
   zero_error_gf = zero_uh;
   zero_error_gf -= zero_exact_gf;

   auto nbc=solver.GetNumBoundaryConditions();

   if (myid == 0)
   {
      cout << setprecision(16);
      cout << "\nDirichlet diffusion manufactured test\n"
           << "  recorded coefficient BC count     = " << boundary_attributes.Size() << '\n'
           << "  assembled essential true dofs     = " << manufactured_ess_tdofs << '\n'
           << "  ||B_system||_2                    = " << system_rhs_norm << '\n'
           << "  ||A_system X - B_system||_2       = " << residual_norm << '\n'
           << "  relative eliminated residual      = " << relative_residual << '\n'
           << "  L2 error                          = " << l2_error << '\n'
           << "\nClear + constant-zero BC test\n"
           << "  recorded constant BC count        = " << nbc << '\n'
           << "  ||X_zero||_2                      = " << zero_solution_norm << '\n'
           << "\nChecks\n"
           << "  L2 error                          " << (pass_l2 ? "PASS" : "FAIL") << '\n'
           << "  eliminated residual               " << (pass_residual ? "PASS" : "FAIL") << '\n'
           << "  clear + constant BC               " << (pass_zero ? "PASS" : "FAIL") << '\n';
   }

   if (save)
   {
      SaveMFEMFiles(pmesh, uh, uex_gf, error_gf, "dirichlet", myid);
      SaveMFEMFiles(pmesh, zero_uh, zero_exact_gf, zero_error_gf, "zero_bc", myid);
   }

   if (paraview)
   {
      const int levels_of_detail = (paraview_lod < 0) ? order : paraview_lod;

      ostringstream dirichlet_name;
      dirichlet_name << paraview_name << "_manufactured";
      SaveParaView(pmesh, uh, uex_gf, error_gf,
                   dirichlet_name.str().c_str(), paraview_prefix,
                   paraview_binary, paraview_high_order,
                   levels_of_detail, 0, 0.0);

      ostringstream zero_name;
      zero_name << paraview_name << "_zero_bc";
      SaveParaView(pmesh, zero_uh, zero_exact_gf, zero_error_gf,
                   zero_name.str().c_str(), paraview_prefix,
                   paraview_binary, paraview_high_order,
                   levels_of_detail, 1, 1.0);

      if (myid == 0)
      {
         cout << "\nSaved ParaView output under: " << paraview_prefix << '\n';
      }
   }

   return (pass_l2 && pass_residual && pass_zero) ? 0 : 3;
}
