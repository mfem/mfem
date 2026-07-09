// MMS test for DiffusionMassGeometricMultigrid with an LOR hierarchy.
//
// This example first creates the requested high-order tensor-product mesh, then
// replaces it by a low-order refined (LOR) mesh using the high-order polynomial
// degree as the local refinement factor.  The multigrid hierarchy is built from
// order-1 H1 spaces on the LOR mesh and its uniform refinements.
//
// The PDE and manufactured solution match tst_diffusion_mass_gmg_mms.cpp:
//     -div(a grad u) + m u = f,  u = product_i sin(pi x_i),
//     f = (d a pi^2 + m) u.

#include "diffusion_mass_gmg.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace mfem;

static real_t ExactSolution(const Vector &x)
{
   const real_t pi = 4.0*std::atan(1.0);
   real_t value = 1.0;
   for (int i = 0; i < x.Size(); i++)
   {
      value *= std::sin(pi*x(i));
   }
   return value;
}

static real_t DiffusionMassRHS(const Vector &x, int dim, real_t diffusion,
                               real_t mass)
{
   const real_t pi = 4.0*std::atan(1.0);
   return (dim*diffusion*pi*pi + mass)*ExactSolution(x);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   int dim = 2;
   int order = 4;
   int nx = 1;
   int ny = 1;
   int nz = 1;
   int ser_ref_levels = 0;
   int par_ref_levels = 3;
   int basis_lor = BasisType::GaussLobatto;
   int pre_smooth = 1;
   int post_smooth = 1;
   int max_iter = 200;
   int gmres_kdim = 50;
   int print_level = 1;
   bool paraview = true;
   const char *coarse_solver = "jacobi";
   real_t diffusion = 1.0;
   real_t mass = 1.0;
   real_t jacobi_damping = 1.0;
   real_t rel_tol = 1e-12;
   real_t abs_tol = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&dim, "-dim", "--dimension",
                  "Problem dimension: 2 or 3.");
   args.AddOption(&order, "-o", "--order",
                  "High-order degree used as the LOR refinement factor.");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of high-order coarse elements in the x direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of high-order coarse elements in the y direction.");
   args.AddOption(&nz, "-nz", "--num-elements-z",
                  "Number of high-order coarse elements in the z direction.");
   args.AddOption(&ser_ref_levels, "-srl", "--ser-ref-levels",
                  "Number of serial uniform refinements before LOR.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of parallel uniform refinement levels after LOR.");
   args.AddOption(&diffusion, "-dc", "--diffusion-coefficient",
                  "Constant diffusion coefficient.");
   args.AddOption(&mass, "-mc", "--mass-coefficient",
                  "Constant mass coefficient.");
   args.AddOption(&jacobi_damping, "-jd", "--jacobi-damping",
                  "Damping parameter for the Jacobi smoother.");
   args.AddOption(&coarse_solver, "-cs", "--coarse-solver",
                  "Coarse solver: jacobi or amg.");
   args.AddOption(&pre_smooth, "-pre", "--pre-smoothing-steps",
                  "Number of pre-smoothing steps in each multigrid cycle.");
   args.AddOption(&post_smooth, "-post", "--post-smoothing-steps",
                  "Number of post-smoothing steps in each multigrid cycle.");
   args.AddOption(&rel_tol, "-rtol", "--relative-tolerance",
                  "CG relative tolerance.");
   args.AddOption(&abs_tol, "-atol", "--absolute-tolerance",
                  "CG absolute tolerance.");
   args.AddOption(&max_iter, "-mi", "--max-iterations",
                  "Maximum number of outer solver iterations.");
   args.AddOption(&gmres_kdim, "-kdim", "--gmres-dim",
                  "GMRES restart dimension used with AMG coarse solves.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "CG print level.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.ParseCheck();

   MFEM_VERIFY(dim == 2 || dim == 3, "Expected dimension 2 or 3.");
   MFEM_VERIFY(order >= 1, "Expected LOR refinement factor >= 1.");
   MFEM_VERIFY(nx > 0 && ny > 0 && nz > 0,
               "Expected positive mesh dimensions.");
   MFEM_VERIFY(par_ref_levels >= 0 && ser_ref_levels >= 0,
               "Refinement levels must be nonnegative.");
   MFEM_VERIFY(diffusion > 0.0, "Diffusion coefficient must be positive.");
   MFEM_VERIFY(mass >= 0.0, "Mass coefficient must be nonnegative.");

   DiffusionMassGeometricMultigrid::CoarseSolver coarse_solver_type;
   if (!strcmp(coarse_solver, "jacobi"))
   {
      coarse_solver_type =
         DiffusionMassGeometricMultigrid::CoarseSolver::SymmetrizedJacobi;
   }
   else if (!strcmp(coarse_solver, "amg"))
   {
      coarse_solver_type =
         DiffusionMassGeometricMultigrid::CoarseSolver::BoomerAMG;
   }
   else
   {
      MFEM_ABORT("Unknown coarse solver. Use 'jacobi' or 'amg'.");
   }

   Device device(device_config);
   device.Print();

   Mesh mesh = (dim == 2)
             ? Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                     true, 1.0, 1.0)
             : Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON,
                                     1.0, 1.0, 1.0, true);
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh ho_pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   ParMesh *lor_pmesh = new ParMesh;
   *lor_pmesh = ParMesh::MakeRefined(ho_pmesh, order, basis_lor);

   H1_FECollection fec_lor(1, lor_pmesh->Dimension());
   ParFiniteElementSpace *coarse_fespace =
      new ParFiniteElementSpace(lor_pmesh, &fec_lor);
   ParFiniteElementSpaceHierarchy hierarchy(lor_pmesh, coarse_fespace,
                                            true, true);
   for (int l = 0; l < par_ref_levels; l++)
   {
      hierarchy.AddUniformlyRefinedLevel(1, Ordering::byVDIM);
   }

   ParFiniteElementSpace &fespace = hierarchy.GetFinestFESpace();
   Array<int> ess_bdr(lor_pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   ConstantCoefficient diffusion_coefficient(diffusion);
   ConstantCoefficient mass_coefficient(mass);
   DiffusionMassGeometricMultigrid mg(hierarchy, ess_bdr,
                                      diffusion_coefficient,
                                      mass_coefficient,
                                      jacobi_damping,
                                      coarse_solver_type);
   mg.SetCycleType(MultigridBase::CycleType::VCYCLE, pre_smooth, post_smooth);

   FunctionCoefficient exact_coefficient(ExactSolution);
   FunctionCoefficient rhs_coefficient(
      [dim, diffusion, mass](const Vector &x)
      {
         return DiffusionMassRHS(x, dim, diffusion, mass);
      });

   ParGridFunction x(&fespace);
   x = 0.0;
   x.ProjectBdrCoefficient(exact_coefficient, ess_bdr);

   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(rhs_coefficient));
   b.Assemble();

   OperatorHandle A;
   Vector X, B;
   mg.FormFineLinearSystem(x, b, A, X, B);

   const bool use_gmres =
      coarse_solver_type == DiffusionMassGeometricMultigrid::CoarseSolver::BoomerAMG;
   if (use_gmres)
   {
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetKDim(gmres_kdim);
      gmres.SetRelTol(rel_tol);
      gmres.SetAbsTol(abs_tol);
      gmres.SetMaxIter(max_iter);
      gmres.SetPrintLevel(print_level);
      gmres.SetPreconditioner(mg);
      gmres.SetOperator(*A);
      gmres.Mult(B, X);
   }
   else
   {
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(rel_tol);
      cg.SetAbsTol(abs_tol);
      cg.SetMaxIter(max_iter);
      cg.SetPrintLevel(print_level);
      cg.SetPreconditioner(mg);
      cg.SetOperator(*A);
      cg.Mult(B, X);
   }

   mg.RecoverFineFEMSolution(X, b, x);

   const real_t l2_error = x.ComputeL2Error(exact_coefficient);
   ParGridFunction exact(&fespace);
   exact.ProjectCoefficient(exact_coefficient);
   ConstantCoefficient zero(0.0);
   const real_t exact_l2 = exact.ComputeL2Error(zero);

   if (Mpi::Root())
   {
      cout << "Diffusion+mass GMG LOR MMS test\n"
           << "  dim=" << dim
           << " ho_order=" << order
           << " lor_order=1"
           << " levels=" << hierarchy.GetNumLevels()
           << " finest true size=" << fespace.GetTrueVSize()
           << " diffusion=" << diffusion
           << " mass=" << mass
           << " coarse_solver=" << coarse_solver
           << " outer_solver=" << (use_gmres ? "gmres" : "cg")
           << " jacobi_damping=" << jacobi_damping << '\n'
           << "  L2 error=" << setprecision(12) << l2_error << '\n'
           << "  relative L2 error=" << l2_error/exact_l2 << endl;
   }

   if (paraview)
   {
      ParaViewDataCollection pvdc("DiffusionMassGMGLORMMS", lor_pmesh);
      pvdc.SetPrefixPath("ParaView");
      pvdc.RegisterField("solution", &x);
      pvdc.RegisterField("exact", &exact);
      pvdc.SetLevelsOfDetail(1);
      pvdc.SetDataFormat(VTKFormat::BINARY);
      pvdc.SetHighOrderOutput(false);
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      pvdc.Save();
   }

   return 0;
}
