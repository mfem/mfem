// Compile with: make TODO
//
// Sample runs: mpirun -np 4 ...
//
// Description:

#include "lpq-jacobi.hpp"

using namespace std;
using namespace mfem;
using namespace lpq_jacobi;


int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   string mesh_file = "meshes/icf.mesh";
   // System properties
   int order = 1;
   SolverType solver_type = sli;
   IntegratorType integrator_type = mass;
   // Number of refinements
   int refine_serial = 1;
   int refine_parallel = 1;
   // Preconditioner parameters
   double p_order = 1.0;
   double q_order = 0.0;
   // Solver parameters
   double rel_tol = 1e-10;
   double max_iter = 3000;
   // Kershaw Transformation
   double eps_y = 0.0;
   double eps_z = 0.0;
   // TODO(Gabriel): To be added later
   // const char *device_config = "cpu";
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   // TODO(Gabriel): To be added later ... " or -1 for isoparametric space.");
   args.AddOption((int*)&solver_type, "-s", "--solver",
                  "Solvers to be considered:"
                  "\n\t0: Stationary Linear Iteration"
                  "\n\t1: Preconditioned Conjugate Gradient"
                  "\n\tTODO");
   args.AddOption((int*)&integrator_type, "-i", "--integrator",
                  "Integrators to be considered:"
                  "\n\t0: MassIntegrator"
                  "\n\t1: DiffusionIntegrator"
                  "\n\t2: ElasticityIntegrator"
                  "\n\t3: CurlCurlIntegrator + VectorFEMassIntegrator"
                  "\n\tTODO");
   args.AddOption(&refine_serial, "-rs", "--refine-serial",
                  "Number of serial refinements");
   args.AddOption(&refine_parallel, "-rp", "--refine-parallel",
                  "Number of parallel refinements");
   args.AddOption(&p_order, "-p", "--p-order",
                  "P-order for L(p,q)-Jacobi preconditioner");
   args.AddOption(&q_order, "-q", "--q-order",
                  "Q-order for L(p,q)-Jacobi preconditioner");
   args.AddOption(&rel_tol, "-t", "--tolerance",
                  "Relative tolerance for the iterative solver");
   args.AddOption(&max_iter, "-ni", "--iterations",
                  "Maximum number of iterations");
   args.AddOption(&eps_y, "-Ky", "--Kershaw-y",
                  "Kershaw transform factor, eps_y in (0,1]");
   args.AddOption(&eps_z, "-Kz", "--Kershaw-z",
                  "Kershaw transform factor, eps_z in (0,1]");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   MFEM_ASSERT(p_order > 0.0, "p needs to be positive");
   MFEM_ASSERT((0 <= solver_type) && (solver_type < num_solvers), "");
   MFEM_ASSERT((0 <= integrator_type) && (integrator_type < num_integrators), "");
   MFEM_ASSERT(0.0 < eps_y <= 1.0, "eps_y in (0,1]");
   MFEM_ASSERT(0.0 < eps_z <= 1.0, "eps_z in (0,1]");

   kappa = freq * M_PI;

   // TODO(Gabriel): To be restructured
   ostringstream file_name;
   {
      string base_name = mesh_file.substr(mesh_file.find_last_of("/\\") + 1);
      base_name = base_name.substr(0, base_name.find_last_of('.'));

      file_name << base_name << "-o" << order << "-i" << integrator_type <<
                "-s" << solver_type << fixed << setprecision(4) << "-p" <<
                (int) (p_order*1000) << "-q" << (int) (q_order*1000) << ".csv";
   }

   // TODO(Gabriel): To be added later...
   // Device device(device_config);
   // if (Mpi::Root()) { device.Print(); }

   Mesh *serial_mesh = new Mesh(mesh_file);
   for (int ls = 0; ls < refine_serial; ls++)
   {
      serial_mesh->UniformRefinement();
   }

   ParMesh *mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   for (int lp = 0; lp < refine_parallel; lp++)
   {
      mesh->UniformRefinement();
   }

   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   dim = mesh->Dimension();
   space_dim = mesh->SpaceDimension();
   switch (integrator_type)
   {
      case mass: case diffusion:
         fec = new H1_FECollection(order, dim);
         fespace = new ParFiniteElementSpace(mesh, fec);
         break;
      case elasticity:
         fec = new H1_FECollection(order, dim);
         fespace = new ParFiniteElementSpace(mesh, fec, dim);
         break;
      case maxwell:
         fec = new ND_FECollection(order, dim);
         fespace = new ParFiniteElementSpace(mesh, fec);
         break;
      default:
         mfem_error("Invalid integrator type! Check FiniteElementCollection");
   }

   HYPRE_BigInt sys_size = fespace->GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << "Number of unknowns: " << sys_size << endl;
   }

   // Define (all) essential boundary attributes
   // Get the essential true DoFs list
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   Array<int> ess_tdof_list;
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ParBilinearForm *a = new ParBilinearForm(fespace);
   ParLinearForm *b = new ParLinearForm(fespace);

   // These pointers are owned by the forms
   LinearFormIntegrator *lfi = nullptr;
   BilinearFormIntegrator *bfi = nullptr;
   // Required for a static_cast
   SumIntegrator *sum_bfi = nullptr;

   // These pointers are not owned by the integrators
   FunctionCoefficient *scalar_u = nullptr;
   FunctionCoefficient *scalar_f = nullptr;
   VectorFunctionCoefficient *vector_u = nullptr;
   VectorFunctionCoefficient *vector_f = nullptr;

   ConstantCoefficient one(1.0);

   // These variables will define the linear system
   ParGridFunction x(fespace);
   HypreParMatrix A;
   Vector B, X;

   x = 0.0;

   switch (integrator_type)
   {
      case mass:
         scalar_u = new FunctionCoefficient(diffusion_solution);
         lfi = new DomainLFIntegrator(*scalar_u);
         bfi = new MassIntegrator();
         x.ProjectBdrCoefficient(*scalar_u, ess_bdr);
         break;
      case diffusion:
         scalar_u = new FunctionCoefficient(diffusion_solution);
         scalar_f = new FunctionCoefficient(diffusion_source);
         lfi = new DomainLFIntegrator(*scalar_f);
         bfi = new DiffusionIntegrator();
         x.ProjectBdrCoefficient(*scalar_u, ess_bdr);
         break;
      case elasticity:
         vector_u = new VectorFunctionCoefficient(space_dim, elasticity_solution);
         vector_f = new VectorFunctionCoefficient(space_dim, elasticity_source);
         lfi = new VectorDomainLFIntegrator(*vector_f);
         bfi = new ElasticityIntegrator(one, one);
         x.ProjectBdrCoefficient(*vector_u, ess_bdr);
         break;
      case maxwell:
         vector_u = new VectorFunctionCoefficient(space_dim, maxwell_solution);
         vector_f = new VectorFunctionCoefficient(space_dim, maxwell_source);
         lfi = new VectorFEDomainLFIntegrator(*vector_f);
         bfi = new SumIntegrator();
         sum_bfi = static_cast<SumIntegrator*>(bfi);
         sum_bfi->AddIntegrator(new CurlCurlIntegrator(one));
         sum_bfi->AddIntegrator(new VectorFEMassIntegrator(one));
         x.ProjectBdrCoefficient(*vector_u, ess_bdr);
         break;
      default:
         mfem_error("Invalid integrator type! Check ParLinearForm");
   }

   a->AddDomainIntegrator(bfi);
   a->Assemble();
   b->AddDomainIntegrator(lfi);
   b->Assemble();

   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // D_{p,q} = diag( D^{1+q-p} |A|^p D^{-q} 1) , where D = diag(A)
   auto lpq_jacobi = new OperatorLpqJacobiSmoother(A, ess_tdof_list, p_order,
                                                   q_order);

   Solver *solver = nullptr;
   // DataMonitor monitor(file_name.str(), NDIGITS);
   switch (solver_type)
   {
      case sli:
         solver = new SLISolver(MPI_COMM_WORLD);
         break;
      case cg:
         solver = new CGSolver(MPI_COMM_WORLD);
         break;
      default:
         mfem_error("Invalid solver type!");
   }

   IterativeSolver *it_solver = dynamic_cast<IterativeSolver *>(solver);
   if (it_solver)
   {
      it_solver->SetRelTol(rel_tol);
      it_solver->SetMaxIter(max_iter);
      it_solver->SetPrintLevel(1);
      it_solver->SetPreconditioner(*lpq_jacobi);
      // it_solver->SetMonitor(monitor);
   }
   solver->SetOperator(A);
   solver->Mult(B, X);

   a->RecoverFEMSolution(X, *b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   delete solver;
   delete lpq_jacobi;
   delete a;
   delete b;
   if (scalar_u) { delete scalar_u; }
   if (scalar_f) { delete scalar_f; }
   if (vector_u) { delete vector_u; }
   if (vector_f) { delete vector_f; }
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
