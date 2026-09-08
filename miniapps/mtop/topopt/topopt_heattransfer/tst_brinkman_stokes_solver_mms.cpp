// MMS test for Brinkman-penalized StokesSolver.
//
// The manufactured problem solves
//
//   -div(nu grad u) + alpha u + grad p = f,   div u = 0,
//
// with Taylor-Hood elements and velocity Dirichlet data on all boundaries.  No
// pressure Dirichlet data is set, so the pressure error is measured after
// removing the algebraic true-dof mean of the pressure difference.

#include "diffusion_mass_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

static int GlobalMax(MPI_Comm comm, int value)
{
   int global = 0;
   MPI_Allreduce(&value, &global, 1, MPI_INT, MPI_MAX, comm);
   return global;
}

static void ExactVelocity(const Vector &x, Vector &u)
{
   const real_t pi = 4.0*std::atan(1.0);
   const int dim = x.Size();
   u.SetSize(dim);
   const real_t sx = std::sin(pi*x(0));
   const real_t sy = std::sin(pi*x(1));
   const real_t s2x = std::sin(2.0*pi*x(0));
   const real_t s2y = std::sin(2.0*pi*x(1));
   if (dim == 2)
   {
      u(0) = sx*sx*s2y;
      u(1) = -sy*sy*s2x;
   }
   else
   {
      const real_t sz = std::sin(pi*x(2));
      u(0) = pi*sx*sx*s2y*sz*sz;
      u(1) = -pi*s2x*sy*sy*sz*sz;
      u(2) = 0.0;
   }
}

static real_t ExactPressure(const Vector &x)
{
   const real_t pi = 4.0*std::atan(1.0);
   real_t p = 1.0;
   for (int d = 0; d < x.Size(); d++) { p *= std::sin(pi*x(d)); }
   return p;
}

static real_t Viscosity(const Vector &x, real_t viscosity, bool variable)
{
   if (!variable) { return viscosity; }
   const real_t pi = 4.0*std::atan(1.0);
   real_t oscillation = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      oscillation *= std::sin(2.0*pi*x(d));
   }
   return viscosity*(1.0 + 0.25*oscillation);
}

static real_t Brinkman(const Vector &x, real_t alpha,
                       real_t alpha_min, real_t alpha_max,
                       bool variable)
{
   if (!variable) { return alpha; }
   const real_t pi = 4.0*std::atan(1.0);
   real_t wave = 0.0;
   for (int d = 0; d < x.Size(); d++)
   {
      wave += std::cos(6.0*pi*x(d)) + 0.5*std::sin(8.0*pi*x(d));
   }
   wave /= std::sqrt(1.25)*x.Size();
   wave = std::max(real_t(-1.0), std::min(real_t(1.0), wave));
   const real_t theta = 0.5*(wave + 1.0);
   return alpha_min*std::pow(alpha_max/alpha_min, theta);
}

static void GradViscosity(const Vector &x, Vector &grad_nu,
                          real_t viscosity, bool variable)
{
   const int dim = x.Size();
   grad_nu.SetSize(dim);
   grad_nu = 0.0;
   if (!variable) { return; }
   const real_t pi = 4.0*std::atan(1.0);
   for (int d = 0; d < dim; d++)
   {
      real_t product = 1.0;
      for (int e = 0; e < dim; e++)
      {
         product *= (e == d) ? std::cos(2.0*pi*x(e))
                             : std::sin(2.0*pi*x(e));
      }
      grad_nu(d) = 0.5*pi*viscosity*product;
   }
}

static void StokesAcceleration(const Vector &x, Vector &f, real_t viscosity,
                               bool variable_viscosity)
{
   const real_t pi = 4.0*std::atan(1.0);
   const int dim = x.Size();
   f.SetSize(dim);
   const real_t sx = std::sin(pi*x(0));
   const real_t sy = std::sin(pi*x(1));
   const real_t cx = std::cos(pi*x(0));
   const real_t cy = std::cos(pi*x(1));
   const real_t s2x = std::sin(2.0*pi*x(0));
   const real_t s2y = std::sin(2.0*pi*x(1));
   const real_t c2x = std::cos(2.0*pi*x(0));
   const real_t c2y = std::cos(2.0*pi*x(1));
   Vector grad_nu;
   GradViscosity(x, grad_nu, viscosity, variable_viscosity);
   const real_t nu = Viscosity(x, viscosity, variable_viscosity);

   if (dim == 2)
   {
      const real_t dpx = pi*cx*sy;
      const real_t dpy = pi*sx*cy;
      const real_t lap_u0 = 2.0*pi*pi*c2x*s2y
                          - 4.0*pi*pi*sx*sx*s2y;
      const real_t lap_u1 = -2.0*pi*pi*c2y*s2x
                            + 4.0*pi*pi*sy*sy*s2x;
      const real_t du0dx = pi*s2x*s2y;
      const real_t du0dy = 2.0*pi*sx*sx*c2y;
      const real_t du1dx = -2.0*pi*sy*sy*c2x;
      const real_t du1dy = -pi*s2y*s2x;
      f(0) = -nu*lap_u0 - (grad_nu(0)*du0dx + grad_nu(1)*du0dy) + dpx;
      f(1) = -nu*lap_u1 - (grad_nu(0)*du1dx + grad_nu(1)*du1dy) + dpy;
   }
   else
   {
      const real_t sz = std::sin(pi*x(2));
      const real_t cz = std::cos(pi*x(2));
      const real_t s2z = std::sin(2.0*pi*x(2));
      const real_t c2z = std::cos(2.0*pi*x(2));
      const real_t dpx = pi*cx*sy*sz;
      const real_t dpy = pi*sx*cy*sz;
      const real_t dpz = pi*sx*sy*cz;
      const real_t lap_u0 =
         pi*pi*pi*(2.0*c2x*s2y*sz*sz
                   - 4.0*sx*sx*s2y*sz*sz
                   + 2.0*sx*sx*s2y*c2z);
      const real_t lap_u1 =
         pi*pi*pi*(4.0*s2x*sy*sy*sz*sz
                   - 2.0*s2x*c2y*sz*sz
                   - 2.0*s2x*sy*sy*c2z);
      const real_t du0dx = pi*pi*s2x*s2y*sz*sz;
      const real_t du0dy = 2.0*pi*pi*sx*sx*c2y*sz*sz;
      const real_t du0dz = pi*pi*sx*sx*s2y*s2z;
      const real_t du1dx = -2.0*pi*pi*c2x*sy*sy*sz*sz;
      const real_t du1dy = -pi*pi*s2x*s2y*sz*sz;
      const real_t du1dz = -pi*pi*s2x*sy*sy*s2z;
      f(0) = -nu*lap_u0
             - (grad_nu(0)*du0dx + grad_nu(1)*du0dy + grad_nu(2)*du0dz)
             + dpx;
      f(1) = -nu*lap_u1
             - (grad_nu(0)*du1dx + grad_nu(1)*du1dy + grad_nu(2)*du1dz)
             + dpy;
      f(2) = dpz;
   }
}

static void BrinkmanAcceleration(const Vector &x, Vector &f,
                                 real_t viscosity, bool variable_viscosity,
                                 real_t alpha, real_t alpha_min,
                                 real_t alpha_max,
                                 bool variable_alpha)
{
   StokesAcceleration(x, f, viscosity, variable_viscosity);
   Vector u;
   ExactVelocity(x, u);
   f.Add(Brinkman(x, alpha, alpha_min, alpha_max, variable_alpha), u);
}

static void ProjectCoefficientToQuadrature(ParMesh &pmesh,
                                           const QuadratureSpace &qspace,
                                           Coefficient &coefficient,
                                           QuadratureFunction &qf)
{
   qf.SetVDim(1);
   real_t *values = qf.HostWrite();
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      ElementTransformation *T = pmesh.GetElementTransformation(e);
      const IntegrationRule &ir = qspace.GetIntRule(e);
      const int offset = qspace.Offset(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         values[offset + q] = coefficient.Eval(*T, ip);
      }
   }
}

static void RemovePressureMean(ParFiniteElementSpace &pspace,
                               const ParGridFunction &exact,
                               ParGridFunction &pressure)
{
   Vector p_true, exact_true;
   pressure.GetTrueDofs(p_true);
   exact.GetTrueDofs(exact_true);
   p_true -= exact_true;
   real_t local_sum = p_true.Sum();
   real_t global_sum = 0.0;
   MPI_Allreduce(&local_sum, &global_sum, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 pspace.GetComm());
   real_t local_size = static_cast<real_t>(p_true.Size());
   real_t global_size = 0.0;
   MPI_Allreduce(&local_size, &global_size, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 pspace.GetComm());
   if (global_size == 0.0) { return; }
   pressure.GetTrueDofs(p_true);
   p_true -= global_sum/global_size;
   pressure.SetFromTrueDofs(p_true);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   const char *mesh_file = "";
   const char *krylov = "minres";
   const char *velocity_prec = "amg";
   const char *pressure_prec = "cc";
   const char *cc_diffusion_solver = "gmres";
   const char *lsc_velocity_operator = "assembled";
   const char *lsc_diagonal_operator = "match";
   const char *lsc_q_preconditioner = "operator-jacobi";
   int dim = 2;
   int order = 3;
   int nx = 4;
   int ny = 4;
   int nz = 4;
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   int print_level = -1;
   int max_iter = 500;
   int velocity_pc_cg_max_iter = 50;
   int pressure_pc_cg_max_iter = 80;
   int kdim = 50;
   bool paraview = true;
   bool variable_viscosity = false;
   bool variable_brinkman = true;
   bool velocity_amg_elasticity_near_nullspace = false;
   real_t viscosity = 1.0;
   real_t brinkman = 10.0;
   real_t brinkman_min = 1.0e-6;
   real_t brinkman_max = 1.0e5;
   real_t rel_tol = 1.0e-10;
   real_t abs_tol = 0.0;
   real_t velocity_pc_cg_rel_tol = 1.0e-8;
   real_t velocity_pc_cg_abs_tol = 0.0;
   real_t pressure_pc_cg_rel_tol = 1.0e-8;
   real_t pressure_pc_cg_abs_tol = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. Empty string creates a Cartesian mesh.");
   args.AddOption(&dim, "-dim", "--dimension",
                  "Dimension of the generated Cartesian mesh: 2 or 3.");
   args.AddOption(&order, "-o", "--order",
                  "Velocity finite element order. Pressure uses order-1.");
   args.AddOption(&nx, "-nx", "--num-elements-x", "Number of elements in x.");
   args.AddOption(&ny, "-ny", "--num-elements-y", "Number of elements in y.");
   args.AddOption(&nz, "-nz", "--num-elements-z",
                  "Number of elements in z for 3D generated meshes.");
   args.AddOption(&ser_ref_levels, "-srl", "--ser-ref-levels",
                  "Number of serial refinements.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of parallel refinements.");
   args.AddOption(&viscosity, "-nu", "--viscosity",
                  "Base viscosity used in the manufactured RHS.");
   args.AddOption(&variable_viscosity, "-vnu", "--variable-viscosity",
                  "-const-nu", "--constant-viscosity",
                  "Use a spatially varying viscosity coefficient.");
   args.AddOption(&brinkman, "-alpha", "--brinkman-penalization",
                  "Base Brinkman penalization coefficient.");
   args.AddOption(&brinkman_min, "-alpha-min", "--minimum-brinkman",
                  "Minimum value for variable Brinkman penalization.");
   args.AddOption(&brinkman_max, "-alpha-max", "--maximum-brinkman",
                  "Maximum value for variable Brinkman penalization.");
   args.AddOption(&variable_brinkman, "-valpha", "--variable-brinkman",
                  "-const-alpha", "--constant-brinkman",
                  "Use a spatially varying Brinkman coefficient.");
   args.AddOption(&krylov, "-ks", "--krylov-solver",
                  "Krylov solver: minres or gmres.");
   args.AddOption(&velocity_prec, "-vp", "--velocity-preconditioner",
                  "Velocity preconditioner: amg or cg.");
   args.AddOption(&velocity_amg_elasticity_near_nullspace,
                  "-vel-amg-elast-ns",
                  "--velocity-amg-elasticity-near-nullspace",
                  "-no-vel-amg-elast-ns",
                  "--no-velocity-amg-elasticity-near-nullspace",
                  "Use elasticity rigid-body modes in velocity AMG.");
   args.AddOption(&pressure_prec, "-pp", "--pressure-preconditioner",
                  "Pressure preconditioner: cc, lsc, diag, cg, or amg.");
   args.AddOption(&cc_diffusion_solver, "-ccds",
                  "--cc-diffusion-solver",
                  "Krylov method for the CC pressure diffusion solve: "
                  "gmres or cg.");
   args.AddOption(&lsc_velocity_operator, "-lsc-vo",
                  "--lsc-velocity-operator",
                  "Velocity operator used by LSC: pa or assembled.");
   args.AddOption(&lsc_diagonal_operator, "-lsc-do",
                  "--lsc-diagonal-operator",
                  "Velocity operator used for the LSC diagonal: "
                  "match, pa, or assembled.");
   args.AddOption(&lsc_q_preconditioner, "-lsc-qp",
                  "--lsc-q-preconditioner",
                  "Preconditioner for the inner LSC Q solve: "
                  "operator-jacobi, jacobi, or amg.");
   args.AddOption(&rel_tol, "-rtol", "--relative-tolerance",
                  "Relative tolerance.");
   args.AddOption(&abs_tol, "-atol", "--absolute-tolerance",
                  "Absolute tolerance.");
   args.AddOption(&max_iter, "-mi", "--max-iterations",
                  "Maximum outer iterations.");
   args.AddOption(&velocity_pc_cg_rel_tol, "-vpc-rtol",
                  "--velocity-preconditioner-cg-relative-tolerance",
                  "Relative tolerance for the velocity CG preconditioner solve.");
   args.AddOption(&velocity_pc_cg_abs_tol, "-vpc-atol",
                  "--velocity-preconditioner-cg-absolute-tolerance",
                  "Absolute tolerance for the velocity CG preconditioner solve.");
   args.AddOption(&velocity_pc_cg_max_iter, "-vpc-mi",
                  "--velocity-preconditioner-cg-max-iterations",
                  "Maximum iterations for the velocity CG preconditioner solve.");
   args.AddOption(&pressure_pc_cg_rel_tol, "-ppc-rtol",
                  "--pressure-preconditioner-cg-relative-tolerance",
                  "Relative tolerance for pressure inner solves.");
   args.AddOption(&pressure_pc_cg_abs_tol, "-ppc-atol",
                  "--pressure-preconditioner-cg-absolute-tolerance",
                  "Absolute tolerance for pressure inner solves.");
   args.AddOption(&pressure_pc_cg_max_iter, "-ppc-mi",
                  "--pressure-preconditioner-cg-max-iterations",
                  "Maximum iterations for pressure inner solves.");
   args.AddOption(&kdim, "-kdim", "--gmres-dim", "GMRES restart dimension.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Linear solver print level.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable ParaView output.");
   args.ParseCheck();

   MFEM_VERIFY(dim == 2 || dim == 3, "Expected dimension 2 or 3.");
   MFEM_VERIFY(order >= 2, "Taylor-Hood MMS test requires velocity order >= 2.");
   MFEM_VERIFY(nx > 0 && ny > 0 && nz > 0, "Expected positive mesh dimensions.");
   MFEM_VERIFY(brinkman > 0.0, "Brinkman penalization must be positive.");
   MFEM_VERIFY(brinkman_min > 0.0 && brinkman_max > brinkman_min,
               "Expected 0 < minimum Brinkman < maximum Brinkman.");

   Device device(device_config);
   device.Print();

   std::unique_ptr<Mesh> mesh;
   if (std::strlen(mesh_file) > 0)
   {
      mesh.reset(new Mesh(mesh_file, 1, 1));
      dim = mesh->Dimension();
      MFEM_VERIFY(dim == 2 || dim == 3, "This MMS test supports 2D or 3D.");
   }
   else if (dim == 2)
   {
      mesh.reset(new Mesh(Mesh::MakeCartesian2D(
         nx, ny, Element::QUADRILATERAL, true, 1.0, 1.0)));
   }
   else
   {
      mesh.reset(new Mesh(Mesh::MakeCartesian3D(
         nx, ny, nz, Element::HEXAHEDRON, 1.0, 1.0, 1.0)));
   }
   for (int l = 0; l < ser_ref_levels; l++) { mesh->UniformRefinement(); }

   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   mesh->Clear();
   for (int l = 0; l < par_ref_levels; l++) { pmesh.UniformRefinement(); }

   const int pressure_order = order - 1;
   H1_FECollection vfec(order, dim, BasisType::GaussLobatto);
   H1_FECollection pfec(pressure_order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace vspace(&pmesh, &vfec, dim, Ordering::byNODES);
   ParFiniteElementSpace pspace(&pmesh, &pfec);
   const int max_attr = GlobalMax(pmesh.GetComm(),
                                  pmesh.attributes.Size()
                                  ? pmesh.attributes.Max() : 0);
   const int max_bdr_attr = GlobalMax(pmesh.GetComm(),
                                      pmesh.bdr_attributes.Size()
                                      ? pmesh.bdr_attributes.Max() : 0);

   BrinkmanStokesSolver solver(vspace, pspace);
   if (!std::strcmp(krylov, "minres"))
   {
      solver.SetSolverType(StokesSolver::KrylovSolver::MINRES);
   }
   else if (!std::strcmp(krylov, "gmres"))
   {
      solver.SetSolverType(StokesSolver::KrylovSolver::GMRES);
   }
   else { MFEM_ABORT("Unknown Krylov solver. Use 'minres' or 'gmres'."); }

   if (!std::strcmp(velocity_prec, "amg"))
   {
      solver.SetVelocityPreconditionerType(
         StokesSolver::VelocityPreconditioner::AMG);
   }
   else if (!std::strcmp(velocity_prec, "cg"))
   {
      solver.SetVelocityPreconditionerType(
         StokesSolver::VelocityPreconditioner::CG);
   }
   else { MFEM_ABORT("Unknown velocity preconditioner. Use 'amg' or 'cg'."); }

   if (!std::strcmp(pressure_prec, "cc"))
   {
      solver.SetPressurePreconditionerType(
         StokesSolver::PressurePreconditioner::CAHOUET_CHABARD);
   }
   else if (!std::strcmp(pressure_prec, "lsc"))
   {
      solver.SetPressurePreconditionerType(
         StokesSolver::PressurePreconditioner::LSC);
   }
   else if (!std::strcmp(pressure_prec, "diag"))
   {
      solver.SetPressurePreconditionerType(
         StokesSolver::PressurePreconditioner::DIAGONAL_MASS);
   }
   else if (!std::strcmp(pressure_prec, "cg"))
   {
      solver.SetPressurePreconditionerType(
         StokesSolver::PressurePreconditioner::CG);
   }
   else if (!std::strcmp(pressure_prec, "amg"))
   {
      solver.SetPressurePreconditionerType(
         StokesSolver::PressurePreconditioner::AMG);
   }
   else
   {
      MFEM_ABORT("Unknown pressure preconditioner. Use 'cc', 'lsc', "
                 "'diag', 'cg', or 'amg'.");
   }

   if (!std::strcmp(cc_diffusion_solver, "gmres"))
   {
      solver.SetCCDiffusionSolverType(
         StokesSolver::CCDiffusionSolver::GMRES);
   }
   else if (!std::strcmp(cc_diffusion_solver, "cg"))
   {
      solver.SetCCDiffusionSolverType(
         StokesSolver::CCDiffusionSolver::CG);
   }
   else
   {
      MFEM_ABORT("Unknown CC diffusion solver. Use 'gmres' or 'cg'.");
   }
   if (!std::strcmp(lsc_velocity_operator, "pa"))
   {
      solver.SetLSCVelocityOperatorType(
         StokesSolver::LSCVelocityOperator::PA);
   }
   else if (!std::strcmp(lsc_velocity_operator, "assembled"))
   {
      solver.SetLSCVelocityOperatorType(
         StokesSolver::LSCVelocityOperator::ASSEMBLED);
   }
   else
   {
      MFEM_ABORT("Unknown LSC velocity operator. Use 'pa' or 'assembled'.");
   }
   if (!std::strcmp(lsc_diagonal_operator, "match"))
   {
      solver.SetLSCDiagonalOperatorType(
         StokesSolver::LSCDiagonalOperator::MATCH_VELOCITY);
   }
   else if (!std::strcmp(lsc_diagonal_operator, "pa"))
   {
      solver.SetLSCDiagonalOperatorType(
         StokesSolver::LSCDiagonalOperator::PA);
   }
   else if (!std::strcmp(lsc_diagonal_operator, "assembled"))
   {
      solver.SetLSCDiagonalOperatorType(
         StokesSolver::LSCDiagonalOperator::ASSEMBLED);
   }
   else
   {
      MFEM_ABORT("Unknown LSC diagonal operator. Use 'match', 'pa', "
                 "or 'assembled'.");
   }
   if (!std::strcmp(lsc_q_preconditioner, "operator-jacobi"))
   {
      solver.SetLSCQPreconditionerType(
         StokesSolver::LSCQPreconditioner::OPERATOR_JACOBI);
   }
   else if (!std::strcmp(lsc_q_preconditioner, "jacobi"))
   {
      solver.SetLSCQPreconditionerType(
         StokesSolver::LSCQPreconditioner::JACOBI);
   }
   else if (!std::strcmp(lsc_q_preconditioner, "amg"))
   {
      solver.SetLSCQPreconditionerType(
         StokesSolver::LSCQPreconditioner::AMG);
   }
   else
   {
      MFEM_ABORT("Unknown LSC Q preconditioner. Use 'operator-jacobi', "
                 "'jacobi', or 'amg'.");
   }

   solver.SetRelTol(rel_tol);
   solver.SetAbsTol(abs_tol);
   solver.SetMaxIter(max_iter);
   solver.SetVelocityAMGElasticityNearNullspace(
      velocity_amg_elasticity_near_nullspace);
   solver.SetVelocityPreconditionerCGRelTol(velocity_pc_cg_rel_tol);
   solver.SetVelocityPreconditionerCGAbsTol(velocity_pc_cg_abs_tol);
   solver.SetVelocityPreconditionerCGMaxIter(velocity_pc_cg_max_iter);
   solver.SetPressurePreconditionerCGRelTol(pressure_pc_cg_rel_tol);
   solver.SetPressurePreconditionerCGAbsTol(pressure_pc_cg_abs_tol);
   solver.SetPressurePreconditionerCGMaxIter(pressure_pc_cg_max_iter);
   solver.SetKDim(kdim);
   solver.SetPrintLevel(print_level);

   FunctionCoefficient viscosity_coeff(
      [viscosity, variable_viscosity](const Vector &x)
      {
         return Viscosity(x, viscosity, variable_viscosity);
      });
   FunctionCoefficient brinkman_coeff(
      [brinkman, brinkman_min, brinkman_max, variable_brinkman](const Vector &x)
      {
         return Brinkman(x, brinkman, brinkman_min, brinkman_max,
                         variable_brinkman);
      });
   solver.SetViscosity(viscosity_coeff);
   solver.SetBrinkmanPenalization(brinkman_coeff);

   VectorFunctionCoefficient exact_u(dim, ExactVelocity);
   FunctionCoefficient exact_p(ExactPressure);
   VectorFunctionCoefficient accel(dim,
      [viscosity, variable_viscosity, brinkman, brinkman_min, brinkman_max,
       variable_brinkman]
      (const Vector &x, Vector &f)
      {
         BrinkmanAcceleration(x, f, viscosity, variable_viscosity,
                              brinkman, brinkman_min, brinkman_max,
                              variable_brinkman);
      });
   for (int attr = 1; attr <= max_attr; attr++)
   {
      solver.Acceleration().Add(attr, accel);
   }
   for (int attr = 1; attr <= max_bdr_attr; attr++)
   {
      solver.VelocityBoundary().Add(attr, exact_u);
   }

   BlockVector x(solver.GetBlockOffsets());
   solver.Solve(x);

   ParGridFunction uh(&vspace), ph(&pspace);
   uh.SetFromTrueDofs(x.GetBlock(0));
   ph.SetFromTrueDofs(x.GetBlock(1));
   ParGridFunction u_exact(&vspace), p_exact(&pspace);
   u_exact.ProjectCoefficient(exact_u);
   p_exact.ProjectCoefficient(exact_p);
   RemovePressureMean(pspace, p_exact, ph);
   ParFiniteElementSpace coefficient_space(&pmesh, &vfec);
   ParGridFunction viscosity_gf(&coefficient_space);
   ParGridFunction brinkman_gf(&coefficient_space);
   viscosity_gf.ProjectCoefficient(viscosity_coeff);
   brinkman_gf.ProjectCoefficient(brinkman_coeff);
   QuadratureSpace coefficient_qspace(&pmesh, 2*order);
   QuadratureFunction viscosity_qf(coefficient_qspace);
   QuadratureFunction brinkman_qf(coefficient_qspace);
   ProjectCoefficientToQuadrature(pmesh, coefficient_qspace, viscosity_coeff,
                                  viscosity_qf);
   ProjectCoefficientToQuadrature(pmesh, coefficient_qspace, brinkman_coeff,
                                  brinkman_qf);

   const real_t u_error = uh.ComputeL2Error(exact_u);
   const real_t p_error = ph.ComputeL2Error(exact_p);

   if (Mpi::Root())
   {
      cout << "Brinkman-Stokes MMS test\n"
           << "  dim=" << dim
           << " order=" << order
           << " pressure_order=" << pressure_order
           << " velocity_true_size=" << vspace.GlobalTrueVSize()
           << " pressure_true_size=" << pspace.GlobalTrueVSize()
           << " krylov=" << krylov
           << " velocity_preconditioner=" << velocity_prec
           << " pressure_preconditioner=" << pressure_prec
           << " cc_diffusion_solver=" << cc_diffusion_solver
           << " viscosity=" << viscosity
           << " variable_viscosity=" << variable_viscosity
           << " brinkman=" << brinkman
           << " brinkman_min=" << brinkman_min
           << " brinkman_max=" << brinkman_max
           << " variable_brinkman=" << variable_brinkman << '\n'
           << "  ||u-uh||_L2=" << setprecision(12) << u_error
           << "  ||p-ph||_L2=" << p_error << endl;
   }

   if (paraview)
   {
      ParaViewDataCollection pvdc("BrinkmanStokesSolverMMS", &pmesh);
      pvdc.SetPrefixPath("ParaView");
      pvdc.RegisterField("velocity", &uh);
      pvdc.RegisterField("pressure", &ph);
      pvdc.RegisterField("exact_velocity", &u_exact);
      pvdc.RegisterField("exact_pressure", &p_exact);
      pvdc.RegisterField("viscosity_projected", &viscosity_gf);
      pvdc.RegisterField("brinkman_penalization_projected", &brinkman_gf);
      pvdc.RegisterQField("viscosity", &viscosity_qf);
      pvdc.RegisterQField("brinkman_penalization", &brinkman_qf);
      pvdc.SetLevelsOfDetail(order);
      pvdc.SetDataFormat(VTKFormat::BINARY);
      pvdc.SetHighOrderOutput(true);
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      pvdc.Save();
   }

   return 0;
}
