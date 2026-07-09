#include "frac_noise.hpp"
#include "periodic_fraclap_coefficients.hpp"
#include "spde_solver.hpp"

using namespace std;
using namespace mfem;

static Mesh MakePeriodicUnitSquareMesh(int nx, int ny)
{
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                     false, 1.0, 1.0, false);

   std::vector<Vector> translations =
   {
      Vector({1.0, 0.0}),
      Vector({0.0, 1.0})
   };

   return Mesh::MakePeriodic(mesh, mesh.CreatePeriodicVertexMapping(translations));
}

static real_t ShiftedFracLapU2D(const Vector &X, real_t exponent)
{
   using periodic_fraclap::pi;

   const real_t x = X(0);
   const real_t y = X(1);
   const real_t t = 2.0*pi;
   const real_t k1 = t;
   const real_t k2 = t*std::sqrt(2.0);

   return 3.0
        + std::pow(1.0 + k1*k1, exponent)*std::sin(t*x)
        + 2.0*std::pow(1.0 + k1*k1, exponent)*std::cos(t*y)
        + 0.5*std::pow(1.0 + k2*k2, exponent)*std::cos(t*(x + y));
}

static real_t L2Norm(ParGridFunction &x)
{
   ConstantCoefficient zero(0.0);
   return x.ComputeL2Error(zero);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   int order = 2;
   int nx = 8;
   int ny = 8;
   int par_ref_levels = 2;
   int ser_ref_levels = 0;
   int smoother_applications = 1;
   bool paraview = true;
   real_t s = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in the x direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in the y direction.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&ser_ref_levels, "-srl", "--ser-ref-levels",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&smoother_applications, "-ns",
                  "--num-smoother-applications",
                  "Number of smoother applications on every level.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.AddOption(&s, "-s", "--s",
                  "Multilevel fractional parameter; tests exponent 1-s.");
   args.ParseCheck();

   MFEM_VERIFY(nx >= 3 && ny >= 3,
               "Fully periodic quadrilateral meshes require nx, ny >= 3.");
   MFEM_VERIFY(s >= 0.0 && s < 0.5, "Expected s in [0,0.5).");
   MFEM_VERIFY(smoother_applications >= 1,
               "Expected at least one smoother application.");

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh mesh = MakePeriodicUnitSquareMesh(nx, ny);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2, "This manufactured test is 2D only.");

   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   const real_t exponent = 1.0 - s;
   const real_t frac_order = 2.0*exponent;
   const real_t nu = 2.0*exponent - dim/2.0;
   const real_t corr_len = std::sqrt(2.0*nu);

   if (Mpi::Root())
   {
      cout << "Manufactured solution on periodic (0,1)x(0,1)" << endl;
      cout << "  exponent=" << exponent
           << " frac_order=" << frac_order
           << " nu=" << nu
           << " l=" << corr_len
           << " smoother_applications=" << smoother_applications << endl;
   }

   FracRandomFieldGenerator multilevel(*pmesh, par_ref_levels, order, 1.0, s,
                                       smoother_applications);
   FracRandomFieldGeneratorSPDE multilevel_spde(*pmesh, par_ref_levels,
                                                order, 1.0, s,
                                                smoother_applications);
   ParFiniteElementSpace &fes = multilevel.GetFinestFESpace();

   FunctionCoefficient exact_coeff(periodic_fraclap::U2D);
   FunctionCoefficient rhs_coeff(
      [exponent](const Vector &X) { return ShiftedFracLapU2D(X, exponent); });
   FunctionCoefficient pure_fraclap_rhs_coeff(
      [frac_order](const Vector &X)
      {
         return periodic_fraclap::FracLapU2D(X, frac_order);
      });

   ParGridFunction exact(&fes);
   exact.ProjectCoefficient(exact_coeff);

   ParGridFunction rhs_gf(&fes);
   rhs_gf.ProjectCoefficient(rhs_coeff);

   ParGridFunction pure_fraclap_rhs_gf(&fes);
   pure_fraclap_rhs_gf.ProjectCoefficient(pure_fraclap_rhs_coeff);

   ParLinearForm rhs_spde(&fes);
   rhs_spde.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   rhs_spde.Assemble();

   ParGridFunction spde_sol(&fes);
   spde_sol = 0.0;
   {
      spde::Boundary periodic_bc;
      spde::SPDESolver spde_solver(nu, periodic_bc, &fes,
                                   corr_len, corr_len, 1.0);
      spde_solver.SetPrintLevel(0);
      spde_solver.Solve(rhs_spde, spde_sol);
   }

   ParLinearForm rhs_mg_lf(&fes);
   rhs_mg_lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   rhs_mg_lf.Assemble();
   unique_ptr<HypreParVector> rhs_mg(rhs_mg_lf.ParallelAssemble());
   rhs_mg->UseDevice(true);

   Vector mg_true;
   mg_true.SetSize(fes.GetTrueVSize());
   multilevel.Mult(*rhs_mg, mg_true);

   ParGridFunction multilevel_sol(&fes);
   multilevel_sol.SetFromTrueDofs(mg_true);

   ParFiniteElementSpace &fes_spde = multilevel_spde.GetFinestFESpace();
   ParLinearForm rhs_mg_spde_lf(&fes_spde);
   rhs_mg_spde_lf.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   rhs_mg_spde_lf.Assemble();
   unique_ptr<HypreParVector> rhs_mg_spde(rhs_mg_spde_lf.ParallelAssemble());
   rhs_mg_spde->UseDevice(true);

   Vector mg_spde_true;
   mg_spde_true.SetSize(fes_spde.GetTrueVSize());
   multilevel_spde.Mult(*rhs_mg_spde, mg_spde_true);

   ParGridFunction multilevel_spde_sol(&fes_spde);
   multilevel_spde_sol.SetFromTrueDofs(mg_spde_true);

   ParGridFunction exact_spde(&fes_spde);
   exact_spde.ProjectCoefficient(exact_coeff);

   ParGridFunction spde_err(&fes);
   spde_err = spde_sol;
   spde_err -= exact;

   ParGridFunction multilevel_err(&fes);
   multilevel_err = multilevel_sol;
   multilevel_err -= exact;

   ParGridFunction multilevel_spde_err(&fes_spde);
   multilevel_spde_err = multilevel_spde_sol;
   multilevel_spde_err -= exact_spde;

   const real_t exact_l2 = L2Norm(exact);
   const real_t spde_l2 = L2Norm(spde_err);
   const real_t multilevel_l2 = L2Norm(multilevel_err);
   const real_t multilevel_spde_l2 = L2Norm(multilevel_spde_err);

   if (Mpi::Root())
   {
      cout << "Errors:" << endl;
      cout << "  SPDE absolute L2       = " << spde_l2 << endl;
      cout << "  SPDE relative L2       = " << spde_l2/exact_l2 << endl;
      cout << "  multilevel absolute L2 = " << multilevel_l2 << endl;
      cout << "  multilevel relative L2 = " << multilevel_l2/exact_l2 << endl;
      cout << "  multilevel+SPDE coarse absolute L2 = "
           << multilevel_spde_l2 << endl;
      cout << "  multilevel+SPDE coarse relative L2 = "
           << multilevel_spde_l2/exact_l2 << endl;
   }

   if (paraview)
   {
      ParaViewDataCollection paraview_dc("periodic_fraclap_mms",
                                         fes.GetParMesh());
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("exact", &exact);
      paraview_dc.RegisterField("shifted_rhs", &rhs_gf);
      paraview_dc.RegisterField("pure_fraclap_rhs", &pure_fraclap_rhs_gf);
      paraview_dc.RegisterField("spde", &spde_sol);
      paraview_dc.RegisterField("multilevel", &multilevel_sol);
      paraview_dc.RegisterField("spde_error", &spde_err);
      paraview_dc.RegisterField("multilevel_error", &multilevel_err);
      paraview_dc.Save();

      ParaViewDataCollection paraview_dc_spde(
         "periodic_fraclap_mms_spde_coarse", fes_spde.GetParMesh());
      paraview_dc_spde.SetPrefixPath("ParaView");
      paraview_dc_spde.SetLevelsOfDetail(order);
      paraview_dc_spde.SetDataFormat(VTKFormat::BINARY);
      paraview_dc_spde.SetHighOrderOutput(true);
      paraview_dc_spde.SetCycle(0);
      paraview_dc_spde.SetTime(0.0);
      paraview_dc_spde.RegisterField("exact", &exact_spde);
      paraview_dc_spde.RegisterField("multilevel_spde", &multilevel_spde_sol);
      paraview_dc_spde.RegisterField("multilevel_spde_error",
                                     &multilevel_spde_err);
      paraview_dc_spde.Save();
   }

   delete pmesh;
   return EXIT_SUCCESS;
}
