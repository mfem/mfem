#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>

#include "lor_mms.hpp"

using namespace std;
using namespace mfem;

struct Opts
{
   const char *mesh_file = "../../data/star.mesh";
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   int order = 3;
   const char *fe = "h";
};

int Run(const Opts &opts)
{
   bool H1 = false, ND = false, RT = false, L2 = false;
   if (string(opts.fe) == "h") { H1 = true; }
   else if (string(opts.fe) == "n") { ND = true; }
   else if (string(opts.fe) == "r") { RT = true; }
   else if (string(opts.fe) == "l") { L2 = true; }
   else { MFEM_ABORT("Bad FE type. Must be 'h', 'n', 'r', or 'l'."); }

   const int order = opts.order;
   const real_t kappa = (order+1)*(order+1); // Penalty used for DG discretizations

   Mesh serial_mesh(opts.mesh_file, 1, 1);
   const int dim = serial_mesh.Dimension();
   const int sdim = serial_mesh.SpaceDimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Mesh dimension must be 2 or 3.");
   MFEM_VERIFY(!L2 || dim == sdim, "DG surface meshes not supported.");
   for (int l = 0; l < opts.ser_ref_levels; l++) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   for (int l = 0; l < opts.par_ref_levels; l++) { mesh.UniformRefinement(); }
   serial_mesh.Clear();

   if (mesh.ncmesh && (RT || ND))
   { MFEM_ABORT("LOR AMS and ADS solvers are not supported with AMR meshes."); }

   FunctionCoefficient f_coeff(f(1.0)), u_coeff(u);
   VectorFunctionCoefficient f_vec_coeff(sdim, f_vec(RT)),
                             u_vec_coeff(sdim, u_vec);

   int b1 = BasisType::GaussLobatto, b2 = BasisType::IntegratedGLL;
   unique_ptr<FiniteElementCollection> fec;
   if (H1) { fec.reset(new H1_FECollection(order, dim, b1)); }
   else if (ND) { fec.reset(new ND_FECollection(order, dim, b1, b2)); }
   else if (RT) { fec.reset(new RT_FECollection(order-1, dim, b1, b2)); }
   else { fec.reset(new L2_FECollection(order, dim, b1)); }

   ParFiniteElementSpace fes(&mesh, fec.get());

   // fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);

   {
      MFEM_PERF_SCOPE("Ensure Nodes");
      mesh.EnsureNodes();
   }
   // {
   //    auto &ir = DiffusionIntegrator::GetRule(*fes.GetFE(0), *fes.GetFE(0));
   //    mesh.GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
   // }

   HYPRE_Int ndofs = fes.GlobalTrueVSize();
   if (Mpi::Root()) { cout << "Number of DOFs: " << ndofs << endl; }

   Array<int> ess_dofs;
   // In DG, boundary conditions are enforced weakly, so no essential DOFs.
   if (!L2) { fes.GetBoundaryTrueDofs(ess_dofs); }

   ParBilinearForm a(&fes);
   if (H1 || L2)
   {
      // a.AddDomainIntegrator(new MassIntegrator);
      a.AddDomainIntegrator(new DiffusionIntegrator);
   }
   else
   {
      a.AddDomainIntegrator(new VectorFEMassIntegrator);
   }

   if (ND) { a.AddDomainIntegrator(new CurlCurlIntegrator); }
   else if (RT) { a.AddDomainIntegrator(new DivDivIntegrator); }
   else if (L2)
   {
      a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(-1.0, kappa));
      a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(-1.0, kappa));
   }
   // Partial assembly not currently supported for DG or for surface meshes with
   // vector finite elements (ND or RT).
   if (!L2 && (H1 || sdim == dim)) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   a.Assemble();
   // a.Assemble();

   ParLinearForm b(&fes);
   if (H1 || L2) { b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff)); }
   else { b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_vec_coeff)); }
   if (L2)
   {
      // DG boundary conditions are enforced weakly with this integrator.
      b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(u_coeff, -1.0, kappa));
   }
   b.Assemble();

   ParGridFunction x(&fes);
   if (H1 || L2) { x.ProjectCoefficient(u_coeff);}
   else { x.ProjectCoefficient(u_vec_coeff); }

   Vector X, B;
   OperatorHandle A;
   a.FormLinearSystem(ess_dofs, x, b, A, X, B);

   unique_ptr<Solver> solv_lor;

   if (H1 || L2)
   {
      auto solv = new LORSolver<HypreBoomerAMG>(a, ess_dofs);
      solv->GetSolver().SetPrintLevel(0);
      solv->GetSolver().Setup(B, X);
      solv_lor.reset(solv);
   }
   else if (RT && dim == 3)
   {
      solv_lor.reset(new LORSolver<HypreADS>(a, ess_dofs));
   }
   else
   {
      solv_lor.reset(new LORSolver<HypreAMS>(a, ess_dofs));
   }

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(*solv_lor);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);

   return 0;
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   Opts opts;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&opts.mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&opts.ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&opts.par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&opts.order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&opts.fe, "-fe", "--fe-type",
                  "FE type. h for H1, n for Hcurl, r for Hdiv, l for L2");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   MFEM_PERF_SYNC(true);

   {
      MFEM_PERF_SCOPE("Temporary allocations");
      Vector tmp1(1024 * 1024 * 1024);
      tmp1.ReadWrite();
      Vector tmp2(1024 * 1024 * 1024);
      tmp2.ReadWrite();
   }
   {
      MFEM_PERF_SCOPE("Hypre allocations");
      double *tmp1 = mfem_hypre_CTAlloc(double, 1024 * 1024 * 1024);
      double *tmp2 = mfem_hypre_CTAlloc(double, 1024 * 1024 * 1024);
      mfem_hypre_TFree(tmp2);
      mfem_hypre_TFree(tmp1);
   }

   MFEM_PERF_DISABLE;
   Run(opts);

   MFEM_PERF_ENABLE;
   Run(opts);
}
