// MMS test for DiffusionMassSolver.
//
// The test creates a user-provided H1 finite element space and passes it to the
// solver.  The operator is partial assembly.  Boundary conditions are installed
// by boundary attribute ID through solver.Boundary().Add(id, coefficient).  For
// order > 1, the solver builds an LOR+BoomerAMG preconditioner internally; for
// order 1, it uses BoomerAMG directly on an assembled operator.
// In ParGridFunction coefficient mode, order > 1 also tests the solver's
// transfer of HO coefficient true-dof values to LOR coefficient grid functions
// for preconditioner assembly.
//
// Manufactured problem:
//     -div(a grad u) + m u = f,
//     u = product_i sin(pi x_i),
//     f = (d a pi^2 + m) u.
// On arbitrary meshes, the same manufactured fields are evaluated in physical
// coordinates, so exact boundary data remains consistent with the chosen mesh.

#include "diffusion_mass_solver.hpp"

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

static real_t RHSValue(const Vector &x, int dim, real_t diffusion, real_t mass)
{
   const real_t pi = 4.0*std::atan(1.0);
   return (dim*diffusion*pi*pi + mass)*ExactSolution(x);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   const char *mesh_file = "";
   int dim = 2;
   int order = 3;
   int nx = 4;
   int ny = 4;
   int nz = 4;
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   int print_level = -1;
   int max_iter = 0;
   bool paraview = true;
   bool use_qf_coefficients = false;
   bool use_pgf_coefficients = false;
   real_t diffusion = 1.0;
   real_t mass = 1.0;
   real_t rel_tol = 1.0e-12;
   real_t abs_tol = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. Empty string creates a Cartesian mesh.");
   args.AddOption(&dim, "-dim", "--dimension",
                  "Problem dimension for generated Cartesian meshes.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order.");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in the x direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in the y direction.");
   args.AddOption(&nz, "-nz", "--num-elements-z",
                  "Number of elements in the z direction.");
   args.AddOption(&ser_ref_levels, "-srl", "--ser-ref-levels",
                  "Number of serial uniform refinements.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of parallel uniform refinements.");
   args.AddOption(&diffusion, "-dc", "--diffusion-coefficient",
                  "Constant diffusion coefficient.");
   args.AddOption(&mass, "-mc", "--mass-coefficient",
                  "Constant mass coefficient.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Print level for the linear solver and AMG.");
   args.AddOption(&rel_tol, "-rtol", "--relative-tolerance",
                  "Relative tolerance for the linear solver.");
   args.AddOption(&abs_tol, "-atol", "--absolute-tolerance",
                  "Absolute tolerance for the linear solver.");
   args.AddOption(&max_iter, "-mi", "--max-iterations",
                  "Maximum number of linear solver iterations. A value <= 0 "
                  "uses the solver default.");
   args.AddOption(&use_qf_coefficients, "-qf", "--quadrature-coefficients",
                  "-no-qf", "--no-quadrature-coefficients",
                  "Use QuadratureFunction coefficients for mass and diffusion.");
   args.AddOption(&use_pgf_coefficients, "-pgf",
                  "--pargridfunction-coefficients",
                  "-no-pgf", "--no-pargridfunction-coefficients",
                  "Use ParGridFunction coefficients for diffusion, mass, and RHS.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.ParseCheck();

   MFEM_VERIFY(order >= 1, "Expected finite element order >= 1.");
   MFEM_VERIFY(!(use_qf_coefficients && use_pgf_coefficients),
               "Use either QuadratureFunction or ParGridFunction coefficients.");
   MFEM_VERIFY(nx > 0 && ny > 0 && nz > 0,
               "Expected positive mesh dimensions.");

   Device device(device_config);
   device.Print();

   std::unique_ptr<Mesh> mesh;
   if (std::strlen(mesh_file) > 0)
   {
      mesh.reset(new Mesh(mesh_file, 1, 1));
      dim = mesh->Dimension();
   }
   else
   {
      MFEM_VERIFY(dim == 2 || dim == 3, "Expected dimension 2 or 3.");
      if (dim == 2)
      {
         mesh.reset(new Mesh(Mesh::MakeCartesian2D(
            nx, ny, Element::QUADRILATERAL, true, 1.0, 1.0)));
      }
      else
      {
         mesh.reset(new Mesh(Mesh::MakeCartesian3D(
            nx, ny, nz, Element::HEXAHEDRON, 1.0, 1.0, 1.0, true)));
      }
   }
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   mesh->Clear();
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   H1_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   const int max_attr = GlobalMax(pmesh.GetComm(),
                                  pmesh.attributes.Size()
                                  ? pmesh.attributes.Max() : 0);
   const int max_bdr_attr = GlobalMax(pmesh.GetComm(),
                                      pmesh.bdr_attributes.Size()
                                      ? pmesh.bdr_attributes.Max() : 0);
   const HYPRE_BigInt global_true_size = fespace.GlobalTrueVSize();

   QuadratureSpace qspace(&pmesh, 2*order);
   DiffusionMassSolver solver(fespace);
   solver.SetPrintLevel(print_level);
   solver.SetRelTol(rel_tol);
   solver.SetAbsTol(abs_tol);
   if (max_iter > 0)
   {
      solver.SetMaxIter(max_iter);
   }
   if (use_qf_coefficients)
   {
      auto diffusion_qf = std::make_shared<QuadratureFunction>(qspace);
      auto mass_qf = std::make_shared<QuadratureFunction>(qspace);
      *diffusion_qf = diffusion;
      *mass_qf = mass;
      solver.SetDiffusionCoefficient(diffusion_qf);
      solver.SetMassCoefficient(mass_qf);
   }
   else
   {
      if (use_pgf_coefficients)
      {
         auto diffusion_gf = std::make_shared<ParGridFunction>(&fespace);
         auto mass_gf = std::make_shared<ParGridFunction>(&fespace);
         *diffusion_gf = diffusion;
         *mass_gf = mass;
         solver.SetDiffusionCoefficient(diffusion_gf);
         solver.SetMassCoefficient(mass_gf);
      }
      else
      {
         solver.SetDiffusionCoefficient(diffusion);
         solver.SetMassCoefficient(std::make_shared<ConstantCoefficient>(mass));
      }
   }

   auto rhs = std::make_shared<FunctionCoefficient>(
      [dim, diffusion, mass](const Vector &x)
      {
         return RHSValue(x, dim, diffusion, mass);
      });
   if (use_pgf_coefficients)
   {
      auto rhs_gf = std::make_shared<ParGridFunction>(&fespace);
      rhs_gf->ProjectCoefficient(*rhs);
      for (int attr = 1; attr <= max_attr; attr++)
      {
         solver.RHS().Add(attr, rhs_gf);
      }
   }
   else
   {
      for (int attr = 1; attr <= max_attr; attr++)
      {
         solver.RHS().Add(attr, rhs);
      }
   }

   auto exact = std::make_shared<FunctionCoefficient>(ExactSolution);
   for (int attr = 1; attr <= max_bdr_attr; attr++)
   {
      solver.Boundary().Add(attr, exact);
   }

   ParGridFunction x(&fespace);
   solver.Solve(x);

   const real_t l2_error = x.ComputeL2Error(*exact);
   ParGridFunction exact_gf(&fespace);
   exact_gf.ProjectCoefficient(*exact);
   ConstantCoefficient zero(0.0);
   const real_t exact_l2 = exact_gf.ComputeL2Error(zero);

   if (Mpi::Root())
   {
      cout << "DiffusionMassSolver MMS test\n"
           << "  dim=" << dim
           << " order=" << order
           << " global true size=" << global_true_size
           << " diffusion=" << diffusion
           << " mass=" << mass
           << " coefficient_storage="
           << (use_qf_coefficients ? "quadrature-function" :
               (use_pgf_coefficients ? "par-grid-function" : "coefficient"))
           << " preconditioner=" << (order > 1 ? "LOR+AMG" : "AMG")
           << '\n'
           << "  L2 error=" << setprecision(12) << l2_error << '\n'
           << "  relative L2 error=" << l2_error/exact_l2 << endl;
   }

   if (paraview)
   {
      ParaViewDataCollection pvdc("DiffusionMassSolverMMS", &pmesh);
      pvdc.SetPrefixPath("ParaView");
      pvdc.RegisterField("solution", &x);
      pvdc.RegisterField("exact", &exact_gf);
      pvdc.SetLevelsOfDetail(order);
      pvdc.SetDataFormat(VTKFormat::BINARY);
      pvdc.SetHighOrderOutput(true);
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      pvdc.Save();
   }

   return 0;
}
