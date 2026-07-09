// PDE filter test for the Lazarov-Sigmund topology optimization filter.
//
// This miniapp exercises the composed PDEFilter operator:
//
//     rho_f = A^{-1} M rho,
//     A = -R^2 Delta + I,    R = r_min/(2 sqrt(3)).
//
// Here rho is an input density true vector, M is the mass map from the selected
// input space to the H1 filtered space, and A is represented by
// DiffusionMassSolver with mass coefficient one and diffusion coefficient R^2.
// The input space can be H1 or discontinuous L2/DG.  Natural Neumann conditions
// are the topology-optimization default, but the test can also impose
// homogeneous Dirichlet or unit Dirichlet data on all boundary attributes with
// `--boundary-condition zero|one`.
//
// The test constructs a black/white high-contrast input density field with
// values exactly 0 or 1, filters it, and evaluates
//
//     J(rho_f) = 0.5 int rho_f^2 dx.
//
// The filtered-space gradient is M_f rho_f, where M_f is the filtered-space mass
// matrix.  PDEFilter::MultTranspose() propagates that gradient back to the
// input true vector:
//
//     dJ/drho = M^T A^{-T} M_f rho_f.
//
// The Taylor remainder check perturbs the input by eps*p and prints
//
//     |J(rho + eps p) - J(rho) - eps <dJ/drho, p>|,
//
// which should converge at second order.  The observed order from consecutive
// halvings of eps is printed and compared with `--min-order`.
//
// ParaView output documents the result with fields:
// input, filtered, filtered_gradient, input_gradient, and direction.

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

static real_t InputField(const Vector &x)
{
   // Black/white checkerboard input.  The discontinuous high-contrast pattern
   // makes the smoothing effect of the PDE filter easy to inspect.
   const real_t scale = 8.0;
   int parity = int(std::floor(scale*x(0)));
   if (x.Size() > 1)
   {
      parity += int(std::floor(scale*x(1)));
   }
   if (x.Size() > 2)
   {
      parity += int(std::floor(scale*x(2)));
   }
   return (parity % 2 == 0) ? 1.0 : 0.0;
}

static real_t PerturbationField(const Vector &x)
{
   // Smooth deterministic Taylor direction.  It is normalized in true-vector
   // Euclidean norm before use.
   const real_t pi = 4.0*std::atan(1.0);
   real_t value = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      value *= std::sin(pi*(d + 1)*x(d));
   }
   return value;
}

static int GlobalMax(MPI_Comm comm, int value)
{
   // Attribute arrays are local to each rank.  The test installs BCs on all
   // global boundary attributes, so the maximum has to be collective.
   int global = 0;
   MPI_Allreduce(&value, &global, 1, MPI_INT, MPI_MAX, comm);
   return global;
}

static real_t Functional(MPI_Comm comm, TrueMassMapOperator &mass,
                         const Vector &filtered_true,
                         Vector &filtered_gradient)
{
   // J = 0.5 (rho_f, rho_f)_L2.  The mass map also returns the exact gradient
   // with respect to filtered true dofs for this quadratic functional.
   mass.Mult(filtered_true, filtered_gradient);
   return 0.5*InnerProduct(comm, filtered_true, filtered_gradient);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   const char *mesh_file = "";
   int dim = 2;
   int order = 2;
   int nx = 16;
   int ny = 16;
   int nz = 8;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int print_level = -1;
   const char *input_space_type = "h1";
   const char *bc_type = "neumann";
   bool paraview = true;
   real_t filter_radius = 0.05;
   real_t taylor_eps = 1.0e-2;
   int taylor_steps = 6;
   real_t min_expected_order = 1.8;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use. Empty string creates a Cartesian mesh.");
   args.AddOption(&dim, "-dim", "--dimension",
                  "Problem dimension for generated Cartesian meshes.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order for input and filtered spaces.");
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
   args.AddOption(&filter_radius, "-r", "--filter-radius",
                  "Lazarov-Sigmund filter radius r_min.");
   args.AddOption(&taylor_eps, "-eps", "--taylor-epsilon",
                  "Initial Taylor perturbation scale.");
   args.AddOption(&taylor_steps, "-ts", "--taylor-steps",
                  "Number of Taylor halving steps.");
   args.AddOption(&min_expected_order, "-mo", "--min-order",
                  "Minimum acceptable observed Taylor remainder order.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Print level for the internal diffusion-mass solver.");
   args.AddOption(&input_space_type, "-is", "--input-space",
                  "Input FE space: h1 or l2.");
   args.AddOption(&bc_type, "-bc", "--boundary-condition",
                  "Filter BC: neumann, zero, or one.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.ParseCheck();

   MFEM_VERIFY(order >= 1, "Expected order >= 1.");
   MFEM_VERIFY(nx > 0 && ny > 0 && nz > 0,
               "Expected positive mesh dimensions.");
   MFEM_VERIFY(filter_radius >= 0.0, "Expected nonnegative filter radius.");
   MFEM_VERIFY(taylor_eps > 0.0, "Expected positive Taylor epsilon.");
   MFEM_VERIFY(taylor_steps > 1, "Expected at least two Taylor steps.");
   MFEM_VERIFY(min_expected_order > 0.0, "Expected positive minimum order.");
   const std::string input_space_name(input_space_type);
   MFEM_VERIFY(input_space_name == "h1" || input_space_name == "l2",
               "Unknown input space. Expected h1 or l2.");
   const std::string bc(bc_type);
   MFEM_VERIFY(bc == "neumann" || bc == "zero" || bc == "one",
               "Unknown BC type. Expected neumann, zero, or one.");

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

   auto filtered_fec = std::make_shared<H1_FECollection>(
      order, dim, BasisType::GaussLobatto);
   auto filtered_fespace = std::make_shared<ParFiniteElementSpace>(
      pmesh.get(), filtered_fec.get());

   std::shared_ptr<FiniteElementCollection> input_fec;
   if (input_space_name == "l2")
   {
      input_fec = std::make_shared<L2_FECollection>(order, dim);
   }
   else
   {
      input_fec = filtered_fec;
   }
   auto input_fespace = std::make_shared<ParFiniteElementSpace>(
      pmesh.get(), input_fec.get());

   PDEFilter filter(input_fespace, filtered_fespace);
   filter.SetFilterRadius(filter_radius);
   filter.GetSolver().SetPrintLevel(print_level);

   // Configure the PDE boundary condition.  Neumann is represented by an empty
   // essential boundary set; zero and one install Dirichlet data on all boundary
   // attributes of the parallel mesh.
   if (bc != "neumann")
   {
      const int max_bdr_attr = GlobalMax(pmesh->GetComm(),
                                         pmesh->bdr_attributes.Size()
                                         ? pmesh->bdr_attributes.Max() : 0);
      for (int attr = 1; attr <= max_bdr_attr; attr++)
      {
         if (bc == "zero")
         {
            filter.GetSolver().AddBoundaryID(attr);
         }
         else
         {
            filter.GetSolver().Boundary().Add(attr, 1.0);
         }
      }
   }

   TrueMassMapOperator filtered_mass(filtered_fespace, filtered_fespace);

   // Project the black/white input and a smooth perturbation into the same FE
   // space used by the filter input. This can be a discontinuous L2 space.
   FunctionCoefficient input_coeff(InputField);
   FunctionCoefficient perturb_coeff(PerturbationField);
   ParGridFunction input_gf(input_fespace.get());
   ParGridFunction perturb_gf(input_fespace.get());
   input_gf.ProjectCoefficient(input_coeff);
   perturb_gf.ProjectCoefficient(perturb_coeff);

   Vector input_true, perturb_true;
   input_gf.GetTrueDofs(input_true);
   perturb_gf.GetTrueDofs(perturb_true);
   const real_t perturb_norm =
      std::sqrt(InnerProduct(input_fespace->GetComm(), perturb_true,
                             perturb_true));
   perturb_true /= perturb_norm;
   perturb_gf.SetFromTrueDofs(perturb_true);

   // Forward filter application and objective/gradient evaluation.
   Vector filtered_true;
   filter.Mult(input_true, filtered_true);

   Vector filtered_gradient;
   const real_t J0 = Functional(filtered_fespace->GetComm(), filtered_mass,
                                filtered_true, filtered_gradient);

   Vector input_gradient;
   filter.MultTranspose(filtered_gradient, input_gradient);
   const real_t directional_derivative =
      InnerProduct(input_fespace->GetComm(), input_gradient, perturb_true);
   const HYPRE_BigInt global_input_true_size =
      input_fespace->GlobalTrueVSize();
   const HYPRE_BigInt global_filtered_true_size =
      filtered_fespace->GlobalTrueVSize();

   ParGridFunction filtered_gf(filtered_fespace.get());
   ParGridFunction filtered_gradient_gf(filtered_fespace.get());
   ParGridFunction input_gradient_gf(input_fespace.get());
   filtered_gf.SetFromTrueDofs(filtered_true);
   filtered_gradient_gf.SetFromTrueDofs(filtered_gradient);
   input_gradient_gf.SetFromTrueDofs(input_gradient);

   if (Mpi::Root())
   {
      cout << "PDE filter Taylor test\n"
           << "  dim=" << dim
           << " order=" << order
           << " input_space=" << input_space_name
           << " input_true_size=" << global_input_true_size
           << " filtered_true_size=" << global_filtered_true_size
           << " filter_radius=" << filter.GetFilterRadius()
           << " diffusion=" << filter.GetDiffusionCoefficient()
           << " bc=" << bc
           << '\n'
           << "  J0=" << setprecision(12) << J0
           << " dJ[p]=" << directional_derivative << '\n'
           << "  eps              remainder        remainder/eps^2 observed_order\n";
   }

   Vector trial_input(input_true.Size());
   Vector trial_filtered;
   Vector trial_gradient;
   Array<real_t> remainders(taylor_steps);
   Array<real_t> orders(taylor_steps);
   orders = 0.0;

   // First-order Taylor remainder.  For a correct adjoint gradient, the
   // remainder should be O(eps^2), so halving eps should reduce it by about 4.
   for (int i = 0; i < taylor_steps; i++)
   {
      const real_t eps = taylor_eps/std::pow(real_t(2.0), i);
      add(input_true, eps, perturb_true, trial_input);
      filter.Mult(trial_input, trial_filtered);
      const real_t J = Functional(filtered_fespace->GetComm(), filtered_mass,
                                  trial_filtered, trial_gradient);
      const real_t remainder = J - J0 - eps*directional_derivative;
      remainders[i] = std::abs(remainder);
      if (i > 0 && remainders[i - 1] > 0.0 && remainders[i] > 0.0)
      {
         orders[i] = std::log(remainders[i - 1]/remainders[i])/std::log(2.0);
      }
      if (Mpi::Root())
      {
         cout << "  " << setw(14) << setprecision(6) << eps
              << "  " << setw(14) << setprecision(6) << remainders[i]
              << "  " << setw(14) << setprecision(6)
              << remainders[i]/(eps*eps)
              << "  " << setw(14) << setprecision(6)
              << (i > 0 ? orders[i] : 0.0) << '\n';
      }
   }

   real_t local_min_order = infinity();
   for (int i = std::max(1, taylor_steps/2); i < taylor_steps; i++)
   {
      if (orders[i] > 0.0)
      {
         local_min_order = std::min(local_min_order, orders[i]);
      }
   }
   if (local_min_order == infinity()) { local_min_order = 0.0; }
   real_t global_min_order = 0.0;
   MPI_Allreduce(&local_min_order, &global_min_order, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, filtered_fespace->GetComm());

   // The status is intentionally checked collectively.  If the observed order
   // is too low on any rank, the test fails.
   const bool taylor_ok = global_min_order >= min_expected_order;
   if (Mpi::Root())
   {
      cout << "  observed minimum order=" << setprecision(6)
           << global_min_order
           << " expected>=" << min_expected_order
           << " status=" << (taylor_ok ? "OK" : "FAILED") << endl;
   }
   MFEM_VERIFY(taylor_ok, "PDE filter Taylor remainder test failed.");

   if (paraview)
   {
      // Store all fields needed to inspect the filter and adjoint result.
      ParaViewDataCollection pvdc("PDEFilterTaylor", pmesh.get());
      pvdc.SetPrefixPath("ParaView");
      pvdc.RegisterField("input", &input_gf);
      pvdc.RegisterField("filtered", &filtered_gf);
      pvdc.RegisterField("filtered_gradient", &filtered_gradient_gf);
      pvdc.RegisterField("input_gradient", &input_gradient_gf);
      pvdc.RegisterField("direction", &perturb_gf);
      pvdc.SetLevelsOfDetail(order);
      pvdc.SetDataFormat(VTKFormat::BINARY);
      pvdc.SetHighOrderOutput(true);
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      pvdc.Save();
   }

   return 0;
}
