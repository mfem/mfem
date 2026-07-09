// DiffusionSolver compliance-gradient Taylor test.
//
// The design/input field is supplied either as a ParGridFunction true vector or
// as scalar QuadratureFunction values.  A PDE filter maps that input to an H1
// field rho_f, and the diffusion coefficient is
//
//     a = a_min + (1 - a_min) rho_f.
//
// The state equation is solved with DiffusionSolver,
//
//     -div(a grad u) = f,     u = 0 on the boundary,
//
// and the compliance functional is J(a) = f^T u.  The derivative with respect
// to the filtered coefficient is
//
//     dJ/drho_f = -(1 - a_min) int |grad u|^2 delta rho_f dx,
//
// which is propagated back to the input with the adjoint PDE filter.  The test
// verifies this chained gradient with a Taylor remainder check.

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

class StateEnergyDensityCoefficient : public Coefficient
{
public:
   StateEnergyDensityCoefficient(const ParGridFunction &state, real_t scale)
      : state_(state), scale_(scale)
   {
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      state_.GetGradient(T, grad_);
      return scale_*InnerProduct(grad_, grad_);
   }

private:
   const ParGridFunction &state_;
   real_t scale_;
   Vector grad_;
};

static real_t InputField(const Vector &x)
{
   const real_t pi = 4.0*std::atan(1.0);
   real_t wave = 1.0;
   for (int d = 0; d < x.Size(); d++)
   {
      wave *= std::sin(pi*x(d));
   }
   return 0.5 + 0.25*wave;
}

static real_t PerturbationField(const Vector &x)
{
   const real_t pi = 4.0*std::atan(1.0);
   real_t value = 0.35;
   for (int d = 0; d < x.Size(); d++)
   {
      value += std::cos(pi*(5 + 2*d)*x(d));
      value += 0.5*std::sin(pi*(4 + d)*x(d));
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

static void AddHomogeneousBoundary(DiffusionSolver &solver, ParMesh &pmesh)
{
   const int max_bdr_attr = GlobalMax(pmesh.GetComm(),
                                      pmesh.bdr_attributes.Size()
                                      ? pmesh.bdr_attributes.Max() : 0);
   for (int attr = 1; attr <= max_bdr_attr; attr++)
   {
      solver.AddBoundaryID(attr);
   }
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   const char *mesh_file = "";
   const char *input_type = "pgf";
   int dim = 2;
   int order = 3;
   int q_order = -1;
   int nx = 8;
   int ny = 8;
   int nz = 4;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int filter_boundary_attr = 1;
   int print_level = -1;
   int max_iter = 500;
   bool paraview = true;
   real_t filter_radius = 0.08;
   real_t min_diffusion = 1.0e-2;
   real_t source_value = 1.0;
   real_t rel_tol = 1.0e-12;
   real_t abs_tol = 0.0;
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
                  "H1 finite element order for filtered coefficient and state.");
   args.AddOption(&q_order, "-qo", "--quadrature-order",
                  "Quadrature order for qf input. Negative uses 2*order.");
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
   args.AddOption(&input_type, "-it", "--input-type",
                  "Input field representation: pgf or qf.");
   args.AddOption(&filter_radius, "-r", "--filter-radius",
                  "PDE filter radius r_min.");
   args.AddOption(&filter_boundary_attr, "-fba", "--filter-boundary-attr",
                  "Boundary attribute where the PDE filter is set to 1.0. "
                  "Use 0 to leave the filter with natural BCs.");
   args.AddOption(&min_diffusion, "-amin", "--minimum-diffusion",
                  "Lower bound in a = a_min + (1-a_min) rho_f.");
   args.AddOption(&source_value, "-src", "--source-value",
                  "Constant source value in the state equation.");
   args.AddOption(&rel_tol, "-rtol", "--relative-tolerance",
                  "Relative tolerance for DiffusionSolver.");
   args.AddOption(&abs_tol, "-atol", "--absolute-tolerance",
                  "Absolute tolerance for DiffusionSolver.");
   args.AddOption(&max_iter, "-mi", "--max-iterations",
                  "Maximum iterations for DiffusionSolver.");
   args.AddOption(&taylor_eps, "-eps", "--taylor-epsilon",
                  "Initial Taylor perturbation scale.");
   args.AddOption(&taylor_steps, "-ts", "--taylor-steps",
                  "Number of Taylor halving steps.");
   args.AddOption(&min_expected_order, "-mo", "--min-order",
                  "Minimum acceptable observed Taylor remainder order.");
   args.AddOption(&print_level, "-pl", "--print-level",
                  "Print level for filter and diffusion solvers.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.ParseCheck();

   if (q_order < 0) { q_order = 2*order; }
   const std::string input_name(input_type);
   MFEM_VERIFY(input_name == "pgf" || input_name == "qf",
               "Unknown input type. Expected pgf or qf.");
   MFEM_VERIFY(order >= 1, "Expected order >= 1.");
   MFEM_VERIFY(q_order >= 0, "Expected quadrature order >= 0.");
   MFEM_VERIFY(nx > 0 && ny > 0 && nz > 0,
               "Expected positive mesh dimensions.");
   MFEM_VERIFY(filter_radius >= 0.0, "Expected nonnegative filter radius.");
   MFEM_VERIFY(filter_boundary_attr >= 0,
               "Expected nonnegative filter boundary attribute.");
   MFEM_VERIFY(min_diffusion > 0.0 && min_diffusion < 1.0,
               "Expected minimum diffusion in (0, 1).");
   MFEM_VERIFY(max_iter > 0, "Expected positive maximum iteration count.");
   MFEM_VERIFY(taylor_eps > 0.0, "Expected positive Taylor epsilon.");
   MFEM_VERIFY(taylor_steps > 1, "Expected at least two Taylor steps.");
   MFEM_VERIFY(min_expected_order > 0.0, "Expected positive minimum order.");

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

   auto fec = std::make_shared<H1_FECollection>(
      order, dim, BasisType::GaussLobatto);
   auto fespace = std::make_shared<ParFiniteElementSpace>(
      pmesh.get(), fec.get());
   auto qspace = std::make_shared<QuadratureSpace>(pmesh.get(), q_order);

   std::unique_ptr<PDEFilter> pgf_filter;
   std::unique_ptr<QuadraturePDEFilter> qf_filter;
   if (input_name == "pgf")
   {
      pgf_filter.reset(new PDEFilter(fespace, fespace));
      pgf_filter->SetFilterRadius(filter_radius);
      pgf_filter->GetSolver().SetPrintLevel(print_level);
      if (filter_boundary_attr > 0)
      {
         pgf_filter->GetSolver().Boundary().Add(filter_boundary_attr, 1.0);
      }
   }
   else
   {
      qf_filter.reset(new QuadraturePDEFilter(qspace, fespace));
      qf_filter->SetFilterRadius(filter_radius);
      qf_filter->GetSolver().SetPrintLevel(print_level);
      if (filter_boundary_attr > 0)
      {
         qf_filter->GetSolver().Boundary().Add(filter_boundary_attr, 1.0);
      }
   }

   DiffusionSolver state_solver(fespace);
   state_solver.SetPrintLevel(print_level);
   state_solver.SetRelTol(rel_tol);
   state_solver.SetAbsTol(abs_tol);
   state_solver.SetMaxIter(max_iter);
   AddHomogeneousBoundary(state_solver, *pmesh);

   ConstantCoefficient source_coeff(source_value);
   ParLinearForm rhs_form(fespace.get());
   rhs_form.AddDomainIntegrator(new DomainLFIntegrator(
                                   source_coeff,
                                   &state_solver.GetIntegrationRule(
                                      pmesh->GetElementGeometry(0))));
   rhs_form.Assemble();
   Vector rhs_true;
   rhs_true.SetSize(fespace->GetTrueVSize());
   rhs_form.ParallelAssemble(rhs_true);

   FunctionCoefficient input_coeff(InputField);
   FunctionCoefficient direction_coeff(PerturbationField);
   ParGridFunction input_gf(fespace.get());
   ParGridFunction direction_gf(fespace.get());
   Vector input, direction;
   if (input_name == "pgf")
   {
      input_gf.ProjectCoefficient(input_coeff);
      direction_gf.ProjectCoefficient(direction_coeff);
      input_gf.GetTrueDofs(input);
      direction_gf.GetTrueDofs(direction);
   }
   else
   {
      ProjectToQuadrature(*pmesh, *qspace, InputField, input);
      ProjectToQuadrature(*pmesh, *qspace, PerturbationField, direction);
   }

   const real_t direction_norm =
      std::sqrt(InnerProduct(pmesh->GetComm(), direction, direction));
   MFEM_VERIFY(direction_norm > 0.0, "Taylor direction has zero norm.");
   direction /= direction_norm;
   if (input_name == "pgf")
   {
      direction_gf.SetFromTrueDofs(direction);
   }

   auto diffusion_gf = std::make_shared<ParGridFunction>(fespace.get());
   ParGridFunction filtered_gf(fespace.get());
   ParGridFunction state_gf(fespace.get());
   Vector filtered_true;
   Vector state_true;
   Vector filtered_gradient;
   Vector input_gradient;

   auto ApplyFilter = [&](const Vector &x, Vector &filtered)
   {
      if (input_name == "pgf")
      {
         pgf_filter->Mult(x, filtered);
      }
      else
      {
         qf_filter->Mult(x, filtered);
      }
   };

   auto ApplyFilterTranspose = [&](const Vector &filtered_bar,
                                   Vector &input_bar)
   {
      if (input_name == "pgf")
      {
         pgf_filter->MultTranspose(filtered_bar, input_bar);
      }
      else
      {
         qf_filter->MultTranspose(filtered_bar, input_bar);
      }
   };

   auto Evaluate = [&](const Vector &x, Vector *gradient) -> real_t
   {
      ApplyFilter(x, filtered_true);
      filtered_gf.SetFromTrueDofs(filtered_true);

      *diffusion_gf = filtered_gf;
      *diffusion_gf *= (1.0 - min_diffusion);
      *diffusion_gf += min_diffusion;

      state_solver.SetDiffusionCoefficient(diffusion_gf);
      state_solver.Mult(rhs_true, state_true);
      state_gf.SetFromTrueDofs(state_true);

      const real_t compliance =
         InnerProduct(fespace->GetComm(), rhs_true, state_true);

      if (gradient)
      {
         StateEnergyDensityCoefficient energy(
            state_gf, -(1.0 - min_diffusion));
         ParLinearForm filtered_grad_form(fespace.get());
         filtered_grad_form.AddDomainIntegrator(new DomainLFIntegrator(
                                                   energy,
                                                   &state_solver
                                                   .GetIntegrationRule(
                                                      pmesh
                                                      ->GetElementGeometry(0))));
         filtered_grad_form.Assemble();
         filtered_gradient.SetSize(fespace->GetTrueVSize());
         filtered_grad_form.ParallelAssemble(filtered_gradient);
         ApplyFilterTranspose(filtered_gradient, *gradient);
      }

      return compliance;
   };

   const real_t J0 = Evaluate(input, &input_gradient);
   const real_t directional_derivative =
      InnerProduct(pmesh->GetComm(), input_gradient, direction);

   if (Mpi::Root())
   {
      const HYPRE_BigInt global_state_size = fespace->GlobalTrueVSize();
      cout << "Diffusion compliance gradient Taylor test\n"
           << "  dim=" << dim
           << " order=" << order
           << " input_type=" << input_name
           << " input_size="
           << (input_name == "pgf" ? global_state_size : qspace->GetSize())
           << " state_true_size=" << global_state_size
           << " filter_radius=" << filter_radius
           << " filter_boundary_attr=" << filter_boundary_attr
           << " min_diffusion=" << min_diffusion
           << '\n'
           << "  J0=" << setprecision(12) << J0
           << " dJ[p]=" << directional_derivative << '\n'
           << "  eps              remainder        remainder/eps^2 observed_order\n";
   }

   Vector trial_input(input.Size());
   Array<real_t> remainders(taylor_steps);
   Array<real_t> orders(taylor_steps);
   orders = 0.0;
   for (int i = 0; i < taylor_steps; i++)
   {
      const real_t eps = taylor_eps/std::pow(real_t(2.0), i);
      add(input, eps, direction, trial_input);
      const real_t J = Evaluate(trial_input, nullptr);
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
   MPI_Allreduce(&local_min_order, &global_min_order, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh->GetComm());

   const bool taylor_ok = global_min_order >= min_expected_order;
   if (Mpi::Root())
   {
      cout << "  observed minimum order=" << setprecision(6)
           << global_min_order
           << " expected>=" << min_expected_order
           << " status=" << (taylor_ok ? "OK" : "FAILED") << endl;
   }
   MFEM_VERIFY(taylor_ok,
               "Diffusion compliance gradient Taylor remainder test failed.");

   if (paraview)
   {
      ParaViewDataCollection pvdc("DiffusionComplianceGradient", pmesh.get());
      pvdc.SetPrefixPath("ParaView");
      pvdc.RegisterField("filtered_density", &filtered_gf);
      pvdc.RegisterField("diffusion_coefficient", diffusion_gf.get());
      pvdc.RegisterField("state", &state_gf);
      if (input_name == "pgf")
      {
         ParGridFunction input_gradient_gf(fespace.get());
         input_gradient_gf.SetFromTrueDofs(input_gradient);
         pvdc.RegisterField("input", &input_gf);
         pvdc.RegisterField("input_gradient", &input_gradient_gf);
         pvdc.RegisterField("direction", &direction_gf);
         pvdc.SetLevelsOfDetail(order);
         pvdc.SetDataFormat(VTKFormat::BINARY);
         pvdc.SetHighOrderOutput(true);
         pvdc.SetCycle(0);
         pvdc.SetTime(0.0);
         pvdc.Save();
      }
      else
      {
         L2_FECollection dg0_fec(0, dim);
         ParFiniteElementSpace dg0_fes(pmesh.get(), &dg0_fec);
         ParGridFunction input_avg(&dg0_fes);
         ParGridFunction gradient_avg(&dg0_fes);
         ParGridFunction direction_avg(&dg0_fes);
         QuadratureToDG0(*pmesh, *qspace, input, input_avg);
         QuadratureToDG0(*pmesh, *qspace, input_gradient, gradient_avg);
         QuadratureToDG0(*pmesh, *qspace, direction, direction_avg);
         pvdc.RegisterField("input_q_average", &input_avg);
         pvdc.RegisterField("input_q_gradient_average", &gradient_avg);
         pvdc.RegisterField("direction_q_average", &direction_avg);
         pvdc.SetLevelsOfDetail(order);
         pvdc.SetDataFormat(VTKFormat::BINARY);
         pvdc.SetHighOrderOutput(true);
         pvdc.SetCycle(0);
         pvdc.SetTime(0.0);
         pvdc.Save();
      }
   }

   return 0;
}
