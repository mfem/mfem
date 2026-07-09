// Quadrature-input PDE filter test.
//
// This miniapp verifies QuadraturePDEFilter, the Lazarov-Sigmund PDE filter
// variant whose design/input variables are scalar QuadratureFunction entries
// instead of FE true dofs:
//
//     rho_f = A^{-1} M_q rho_q,
//     A = -R^2 Delta + I,    R = r_min/(2 sqrt(3)).
//
// M_q assembles the filtered-space RHS directly from quadrature-point values.
// Its transpose returns gradients with respect to the scalar quadrature entries,
// including the same quadrature weights and geometric factors.  The test uses a
// black/white high-contrast quadrature input, evaluates
//
//     J(rho_f) = 0.5 int rho_f^2 dx,
//
// and checks the adjoint gradient with a Taylor remainder test.  ParaView output
// stores the filtered FE fields and DG0 element-average views of the quadrature
// input, quadrature gradient, and perturbation direction.

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
   const real_t scale = 8.0;
   int parity = int(std::floor(scale*x(0)));
   if (x.Size() > 1) { parity += int(std::floor(scale*x(1))); }
   if (x.Size() > 2) { parity += int(std::floor(scale*x(2))); }
   return (parity % 2 == 0) ? 1.0 : 0.0;
}

static real_t PerturbationField(const Vector &x)
{
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
   int global = 0;
   MPI_Allreduce(&value, &global, 1, MPI_INT, MPI_MAX, comm);
   return global;
}

static void ProjectToQuadrature(ParMesh &pmesh,
                                const QuadratureSpace &qspace,
                                real_t (*function)(const Vector &),
                                Vector &qvalues)
{
   qvalues.SetSize(qspace.GetSize());
   real_t *values = qvalues.HostWrite();
   Vector x;
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      ElementTransformation *T = pmesh.GetElementTransformation(e);
      const IntegrationRule &ir = qspace.GetIntRule(e);
      const int offset = qspace.Offset(e);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->Transform(ip, x);
         values[offset + q] = function(x);
      }
   }
}

static void QuadratureToDG0(ParMesh &pmesh,
                            const QuadratureSpace &qspace,
                            const Vector &qvalues,
                            ParGridFunction &dg0)
{
   dg0 = 0.0;
   const real_t *values = qvalues.HostRead();
   Array<int> vdofs;
   Vector one_value(1);
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      ElementTransformation *T = pmesh.GetElementTransformation(e);
      const IntegrationRule &ir = qspace.GetIntRule(e);
      const int offset = qspace.Offset(e);
      real_t integral = 0.0;
      real_t volume = 0.0;
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         const real_t weight = ip.weight*T->Weight();
         integral += weight*values[offset + q];
         volume += weight;
      }
      one_value(0) = integral/volume;
      dg0.FESpace()->GetElementVDofs(e, vdofs);
      dg0.SetSubVector(vdofs, one_value);
   }
}

static real_t Functional(MPI_Comm comm, TrueMassMapOperator &mass,
                         const Vector &filtered_true,
                         Vector &filtered_gradient)
{
   mass.Mult(filtered_true, filtered_gradient);
   return 0.5*InnerProduct(comm, filtered_true, filtered_gradient);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   const char *mesh_file = "";
   const char *bc_type = "neumann";
   int dim = 2;
   int order = 2;
   int q_order = -1;
   int nx = 16;
   int ny = 16;
   int nz = 8;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int print_level = -1;
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
                  "Filtered H1 finite element order.");
   args.AddOption(&q_order, "-qo", "--quadrature-order",
                  "Quadrature order for the scalar input. Negative uses 2*order.");
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
   args.AddOption(&bc_type, "-bc", "--boundary-condition",
                  "Filter BC: neumann, zero, or one.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.ParseCheck();

   MFEM_VERIFY(order >= 1, "Expected order >= 1.");
   if (q_order < 0) { q_order = 2*order; }
   MFEM_VERIFY(q_order >= 0, "Expected quadrature order >= 0.");
   MFEM_VERIFY(nx > 0 && ny > 0 && nz > 0,
               "Expected positive mesh dimensions.");
   MFEM_VERIFY(filter_radius >= 0.0, "Expected nonnegative filter radius.");
   MFEM_VERIFY(taylor_eps > 0.0, "Expected positive Taylor epsilon.");
   MFEM_VERIFY(taylor_steps > 1, "Expected at least two Taylor steps.");
   MFEM_VERIFY(min_expected_order > 0.0, "Expected positive minimum order.");
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
   auto qspace = std::make_shared<QuadratureSpace>(pmesh.get(), q_order);

   QuadraturePDEFilter filter(qspace, filtered_fespace);
   filter.SetFilterRadius(filter_radius);
   filter.GetSolver().SetPrintLevel(print_level);

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

   Vector input_q, perturb_q;
   ProjectToQuadrature(*pmesh, *qspace, InputField, input_q);
   ProjectToQuadrature(*pmesh, *qspace, PerturbationField, perturb_q);

   const real_t perturb_norm =
      std::sqrt(InnerProduct(pmesh->GetComm(), perturb_q, perturb_q));
   perturb_q /= perturb_norm;

   TrueMassMapOperator filtered_mass(filtered_fespace, filtered_fespace);

   Vector filtered_true;
   filter.Mult(input_q, filtered_true);

   Vector filtered_gradient;
   const real_t J0 = Functional(filtered_fespace->GetComm(), filtered_mass,
                                filtered_true, filtered_gradient);

   Vector input_q_gradient;
   filter.MultTranspose(filtered_gradient, input_q_gradient);
   const real_t directional_derivative =
      InnerProduct(pmesh->GetComm(), input_q_gradient, perturb_q);

   const HYPRE_BigInt global_filtered_true_size =
      filtered_fespace->GlobalTrueVSize();

   ParGridFunction filtered_gf(filtered_fespace.get());
   ParGridFunction filtered_gradient_gf(filtered_fespace.get());
   filtered_gf.SetFromTrueDofs(filtered_true);
   filtered_gradient_gf.SetFromTrueDofs(filtered_gradient);

   if (Mpi::Root())
   {
      cout << "Quadrature PDE filter Taylor test\n"
           << "  dim=" << dim
           << " order=" << order
           << " quadrature_order=" << q_order
           << " local_quadrature_size=" << qspace->GetSize()
           << " filtered_true_size=" << global_filtered_true_size
           << " filter_radius=" << filter.GetFilterRadius()
           << " diffusion=" << filter.GetDiffusionCoefficient()
           << " bc=" << bc
           << '\n'
           << "  J0=" << setprecision(12) << J0
           << " dJ[p]=" << directional_derivative << '\n'
           << "  eps              remainder        remainder/eps^2 observed_order\n";
   }

   Vector trial_input(input_q.Size());
   Vector trial_filtered;
   Vector trial_gradient;
   Array<real_t> remainders(taylor_steps);
   Array<real_t> orders(taylor_steps);
   orders = 0.0;

   for (int i = 0; i < taylor_steps; i++)
   {
      const real_t eps = taylor_eps/std::pow(real_t(2.0), i);
      add(input_q, eps, perturb_q, trial_input);
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

   real_t observed_order = 0.0;
   int observed_count = 0;
   for (int i = 1; i < taylor_steps; i++)
   {
      if (orders[i] > 0.0)
      {
         observed_order += orders[i];
         observed_count++;
      }
   }
   observed_order = observed_count ? observed_order/observed_count : 0.0;
   const bool taylor_ok = observed_order >= min_expected_order;

   if (Mpi::Root())
   {
      cout << "  average observed order=" << setprecision(6)
           << observed_order
           << " min_required=" << min_expected_order
           << " status=" << (taylor_ok ? "OK" : "FAILED") << endl;
   }

   if (paraview)
   {
      L2_FECollection dg0_fec(0, dim);
      ParFiniteElementSpace dg0_fes(pmesh.get(), &dg0_fec);
      ParGridFunction input_avg(&dg0_fes);
      ParGridFunction input_gradient_avg(&dg0_fes);
      ParGridFunction direction_avg(&dg0_fes);
      QuadratureToDG0(*pmesh, *qspace, input_q, input_avg);
      QuadratureToDG0(*pmesh, *qspace, input_q_gradient, input_gradient_avg);
      QuadratureToDG0(*pmesh, *qspace, perturb_q, direction_avg);

      ParaViewDataCollection pvdc("QuadraturePDEFilter", pmesh.get());
      pvdc.SetPrefixPath("ParaView");
      pvdc.RegisterField("input_q_average", &input_avg);
      pvdc.RegisterField("filtered", &filtered_gf);
      pvdc.RegisterField("filtered_gradient", &filtered_gradient_gf);
      pvdc.RegisterField("input_q_gradient_average", &input_gradient_avg);
      pvdc.RegisterField("direction_q_average", &direction_avg);
      pvdc.SetLevelsOfDetail(std::max(order, 1));
      pvdc.SetDataFormat(VTKFormat::BINARY);
      pvdc.SetHighOrderOutput(true);
      pvdc.SetCycle(0);
      pvdc.SetTime(0.0);
      pvdc.Save();
   }

   return taylor_ok ? 0 : 1;
}
