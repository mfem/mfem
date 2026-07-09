// Riesz map test.
//
// This miniapp verifies the L2 Riesz map and its inverse on a user-selected
// parallel finite element space.  The forward Riesz map takes a primal true
// vector u and returns the dual true vector M u, where M is the mass operator.
// The inverse Riesz map solves M u = b with CG and a diagonal preconditioner.
//
// The test projects a smooth field into the selected FE space, applies the
// forward map, applies the inverse map, and reports the relative true-vector
// recovery error.  ParaView output stores the primal field, the dual vector
// represented in the same basis for inspection, the recovered primal field, and
// the recovery error.

#include "diffusion_mass_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

using namespace std;
using namespace mfem;

static real_t ExactField(const Vector &x)
{
   const real_t pi = 4.0*std::atan(1.0);
   real_t value = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      value += std::sin((d + 1)*pi*x(d));
   }
   return value;
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   const char *mesh_file = "";
   const char *space_type = "h1";
   int dim = 2;
   int order = 2;
   int nx = 8;
   int ny = 8;
   int nz = 4;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int print_level = -1;
   int max_iter = 200;
   real_t rel_tol = 1.0e-12;
   real_t abs_tol = 0.0;
   bool paraview = true;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. Empty string creates a Cartesian mesh.");
   args.AddOption(&space_type, "-s", "--space",
                  "Finite element space: h1 or l2.");
   args.AddOption(&dim, "-dim", "--dimension",
                  "Problem dimension for generated Cartesian meshes.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order.");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in x.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in y.");
   args.AddOption(&nz, "-nz", "--num-elements-z",
                  "Number of elements in z.");
   args.AddOption(&ser_ref_levels, "-srl", "--ser-ref-levels",
                  "Number of serial uniform refinements.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of parallel uniform refinements.");
   args.AddOption(&rel_tol, "-rtol", "--relative-tolerance",
                  "Relative tolerance for the inverse Riesz CG solve.");
   args.AddOption(&abs_tol, "-atol", "--absolute-tolerance",
                  "Absolute tolerance for the inverse Riesz CG solve.");
   args.AddOption(&max_iter, "-mi", "--max-iterations",
                  "Maximum iterations for the inverse Riesz CG solve.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Print level for the inverse Riesz CG solve.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.ParseCheck();

   const std::string space(space_type);
   MFEM_VERIFY(space == "h1" || space == "l2",
               "Unknown FE space. Expected h1 or l2.");
   MFEM_VERIFY(order >= 0, "Expected finite element order >= 0.");
   MFEM_VERIFY(nx > 0 && ny > 0 && nz > 0,
               "Expected positive mesh dimensions.");
   MFEM_VERIFY(rel_tol >= 0.0, "Expected nonnegative relative tolerance.");
   MFEM_VERIFY(abs_tol >= 0.0, "Expected nonnegative absolute tolerance.");
   MFEM_VERIFY(max_iter > 0, "Expected positive maximum iteration count.");

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

   auto pmesh = std::make_shared<ParMesh>(MPI_COMM_WORLD, *mesh);
   mesh->Clear();
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   std::shared_ptr<FiniteElementCollection> fec;
   if (space == "h1")
   {
      MFEM_VERIFY(order >= 1, "H1 space requires order >= 1.");
      fec = std::make_shared<H1_FECollection>(order, dim,
                                              BasisType::GaussLobatto);
   }
   else
   {
      fec = std::make_shared<L2_FECollection>(order, dim);
   }
   auto fespace = std::make_shared<ParFiniteElementSpace>(pmesh.get(),
                                                          fec.get());

   RieszMapOperator riesz(fespace);
   InverseRieszMapOperator inverse_riesz(fespace);
   inverse_riesz.SetRelTol(rel_tol);
   inverse_riesz.SetAbsTol(abs_tol);
   inverse_riesz.SetMaxIter(max_iter);
   inverse_riesz.SetPrintLevel(print_level);

   FunctionCoefficient exact_coeff(ExactField);
   ParGridFunction primal_gf(fespace.get());
   primal_gf.ProjectCoefficient(exact_coeff);

   Vector primal_true;
   primal_gf.GetTrueDofs(primal_true);

   Vector dual_true;
   riesz.Mult(primal_true, dual_true);

   Vector recovered_true;
   inverse_riesz.Mult(dual_true, recovered_true);

   Vector transpose_recovered_true;
   inverse_riesz.MultTranspose(dual_true, transpose_recovered_true);

   Vector error_true(primal_true.Size());
   add(recovered_true, -1.0, primal_true, error_true);
   const real_t error_norm =
      std::sqrt(InnerProduct(fespace->GetComm(), error_true, error_true));
   const real_t primal_norm =
      std::sqrt(InnerProduct(fespace->GetComm(), primal_true, primal_true));
   const real_t relative_error = error_norm/primal_norm;

   Vector transpose_error_true(primal_true.Size());
   add(transpose_recovered_true, -1.0, primal_true, transpose_error_true);
   const real_t transpose_error_norm =
      std::sqrt(InnerProduct(fespace->GetComm(), transpose_error_true,
                             transpose_error_true));

   const HYPRE_BigInt global_true_size = fespace->GlobalTrueVSize();

   if (Mpi::Root())
   {
      cout << "Riesz map test\n"
           << "  dim=" << dim
           << " order=" << order
           << " space=" << space
           << " true_size=" << global_true_size
           << '\n'
           << "  absolute recovery error=" << setprecision(12) << error_norm
           << '\n'
           << "  relative recovery error=" << relative_error << '\n'
           << "  transpose inverse absolute error="
           << transpose_error_norm << endl;
   }

   ParGridFunction dual_gf(fespace.get());
   ParGridFunction recovered_gf(fespace.get());
   ParGridFunction error_gf(fespace.get());
   dual_gf.SetFromTrueDofs(dual_true);
   recovered_gf.SetFromTrueDofs(recovered_true);
   error_gf.SetFromTrueDofs(error_true);

   if (paraview)
   {
      ParaViewDataCollection pvdc("RieszMap", pmesh.get());
      pvdc.SetPrefixPath("ParaView");
      pvdc.RegisterField("primal", &primal_gf);
      pvdc.RegisterField("dual", &dual_gf);
      pvdc.RegisterField("recovered", &recovered_gf);
      pvdc.RegisterField("error", &error_gf);
      pvdc.SetLevelsOfDetail(std::max(order, 1));
      pvdc.SetDataFormat(VTKFormat::BINARY);
      pvdc.SetHighOrderOutput(true);
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      pvdc.Save();
   }

   return 0;
}
