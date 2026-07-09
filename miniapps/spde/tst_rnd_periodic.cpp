#include "frac_noise.hpp"
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

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   int order = 1;
   int nx = 8;
   int ny = 8;
   int par_ref_levels = 4;
   int ser_ref_levels = 0;
   bool paraview = false;
   bool visualization = true;
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
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or not Paraview visualization");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&s, "-s", "--s", "Fractional exponent s in [0,0.5).");
   args.ParseCheck();

   MFEM_VERIFY(nx >= 3 && ny >= 3,
               "Fully periodic quadrilateral meshes require nx, ny >= 3.");

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh mesh = MakePeriodicUnitSquareMesh(nx, ny);
   const int dim = mesh.Dimension();
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   if (Mpi::Root())
   {
      cout << "Periodic unit-square mesh: "
           << mesh.GetNE() << " elements, "
           << mesh.GetNV() << " vertices" << endl;
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   real_t nu = (4.0*(1.0 - s) - dim)/2.0;
   real_t sca = pow((2.0*nu), (2.0*nu + dim)/4.0);
   if (Mpi::Root())
   {
      cout << "nu=" << nu << " sca=" << sca << " 1/sca=" << 1.0/sca << endl;
   }

   FracRandomFieldGenerator *frac_rng =
      new FracRandomFieldGenerator(*pmesh, par_ref_levels, order, 1.0, s);

   ParFiniteElementSpace &fes = frac_rng->GetFinestFESpace();
   HYPRE_BigInt size = fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   int myrank;
   MPI_Comm_rank(pmesh->GetComm(), &myrank);

   unique_ptr<ParLinearForm> b(new ParLinearForm(&fes));
   LinearFormIntegrator *rint =
      new WhiteGaussianNoiseDomainLFIntegrator(pmesh->GetComm(),
                                               7497 + 17*myrank);
   b->AddDomainIntegrator(rint);
   b->Assemble();
   unique_ptr<HypreParVector> r(b->ParallelAssemble());
   r->UseDevice(true);

   Vector rfv;
   rfv.SetSize(fes.GetTrueVSize());
   frac_rng->Mult(*r, rfv);

   ParGridFunction spderf(&fes);
   spderf.UseDevice(true);
   spderf.SetTrueVector();
   spderf.GetTrueVector().UseDevice(true);

   {
      spde::Boundary spde_bc;
      spde::SPDESolver spde(nu, spde_bc, &fes, 1.0, 1.0, 1.0);

      ParLinearForm rhs(&fes);
      rhs.AddDomainIntegrator(
         new WhiteGaussianNoiseDomainLFIntegrator(pmesh->GetComm(),
                                                  7497 + 17*myrank));
      rhs.Assemble();

      spde.Solve(rhs, spderf);
   }

   ParGridFunction diffrf(&fes);
   diffrf = 0.0;
   diffrf.SetTrueVector();
   {
      mfem::ParBilinearForm bf(&fes);
      bf.AddDomainIntegrator(new DiffusionIntegrator());
      ConstantCoefficient mc(1.0);
      bf.AddDomainIntegrator(new MassIntegrator(mc));
      bf.Assemble();
      bf.Finalize();
      HypreParMatrix *A = bf.ParallelAssemble();

      HypreBoomerAMG amg;
      amg.SetPrintLevel(0);
      amg.SetOperator(*A);
      unique_ptr<CGSolver> solver(new CGSolver(pmesh->GetComm()));
      solver->SetPrintLevel(0);
      solver->SetOperator(*A);
      solver->SetRelTol(1e-12);
      solver->SetMaxIter(500);
      solver->SetAbsTol(1e-14);
      solver->SetPreconditioner(amg);

      solver->Mult(*r, diffrf.GetTrueVector());
      diffrf.SetFromTrueVector();

      delete A;
   }

   ParGridFunction x(&fes);
   x.UseDevice(true);
   x.SetTrueVector();
   x.GetTrueVector().UseDevice(true);
   x.SetFromTrueDofs(rfv);

   ParaViewDataCollection paraview_dc("rnd_periodic", fes.GetParMesh());
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("rnd", &x);
   paraview_dc.RegisterField("spde_rf", &spderf);
   paraview_dc.RegisterField("diffrf", &diffrf);
   paraview_dc.Save();

   delete frac_rng;
   delete pmesh;
   return EXIT_SUCCESS;
}
