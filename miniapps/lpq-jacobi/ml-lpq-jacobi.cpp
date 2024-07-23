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
   /// 1. Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   /// 2. Parse command line options.
   string mesh_file = "meshes/icf.mesh";
   // System properties
   SolverType solver_type = sli;
   IntegratorType integrator_type = mass;
   // Number of refinements
   int refine_serial = 1;
   int refine_parallel = 1;
   // Number of geometric and order levels
   int geometric_levels = 1;
   int order_levels = 1;
   // Preconditioner parameters
   double p_order = 1.0;
   double q_order = 0.0;
   // Solver parameters
   double rel_tol = 1e-10;
   double max_iter = 3000;
   // Kershaw Transformation
   double eps_y = 0.0;
   double eps_z = 0.0;
   // Other options
   // TODO(Gabriel): To add device support
   // const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&geometric_levels, "-gl", "--geometric-levels",
                  "Number of geometric refinements (levels) done prior to order refinements.");
   args.AddOption(&order_levels, "-ol", "--order-levels",
                  "Number of order refinements (levels). "
                  "Finest level in the hierarchy has order 2^{or}.");
   args.AddOption((int*)&solver_type, "-s", "--solver",
                  "Solvers to be considered:"
                  "\n\t0: Stationary Linear Iteration"
                  "\n\t1: Preconditioned Conjugate Gradient");
   args.AddOption((int*)&integrator_type, "-i", "--integrator",
                  "Integrators to be considered:"
                  "\n\t0: MassIntegrator"
                  "\n\t1: DiffusionIntegrator"
                  "\n\t2: ElasticityIntegrator"
                  "\n\t3: CurlCurlIntegrator + VectorFEMassIntegrator");
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
   MFEM_ASSERT(geometric_levels >= 0, "geometric_level needs to be non-negative");
   MFEM_ASSERT(order_levels >= 0, "order_level needs to be non-negative");
   MFEM_ASSERT((0 <= integrator_type) && (integrator_type < num_integrators), "");
   MFEM_ASSERT(0.0 < eps_y <= 1.0, "eps_y in (0,1]");
   MFEM_ASSERT(0.0 < eps_z <= 1.0, "eps_z in (0,1]");

   kappa = freq * M_PI;

   // TODO(Gabriel): To be restructured
   ostringstream file_name;
   {
      string base_name = mesh_file.substr(mesh_file.find_last_of("/\\") + 1);
      base_name = base_name.substr(0, base_name.find_last_of('.'));

      file_name << base_name << "-i" << integrator_type <<
                fixed << setprecision(4) << "-p" <<
                (int) (p_order*1000) << "-q" << (int) (q_order*1000) << ".csv";
   }

   // TODO(Gabriel): To be added later...
   // Device device(device_config);
   // if (myid == 0) { device.Print(); }

   /// 3. Read the serial mesh from the given mesh file.
   ///    For convinience, the meshes are available in
   ///    ./meshes, and the number of serial and parallel
   ///    refinements are user-defined.
   Mesh *serial_mesh = new Mesh(mesh_file);
   for (int ls = 0; ls < refine_serial; ls++)
   {
      serial_mesh->UniformRefinement();
   }

   /// 4. Define a parallel mesh by a partitioning of the serial mesh.
   ///    Number of parallel refinements given by the user. If defined,
   ///    apply Kershaw transformation.
   ParMesh *mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   for (int lp = 0; lp < refine_parallel; lp++)
   {
      mesh->UniformRefinement();
   }

   dim = mesh->Dimension();
   space_dim = mesh->SpaceDimension();

   bool cond_z = (dim < 3)?true:(eps_z != 0); // lazy check
   if (eps_y != 0.0 && cond_z)
   {
      if (dim < 3) { eps_z = 0.0; }
      common::KershawTransformation kershawT(dim, eps_y, eps_z);
      mesh->Transform(kershawT);
   }

   /// 5. Define a finite element space on the mesh. We use different spaces
   ///    and collections for different systems.
   ///    - H1-conforming Lagrange elements for the H1-mass matrix and the
   ///      diffusion problem.
   ///    - Vector H1-conforming Lagrange elements for the elasticity problem.
   ///    - H(curl)-conforming Nedelec elements for the definite Maxwell problem.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *coarse_fes;
   switch (integrator_type)
   {
      case mass: case diffusion:
         fec = new H1_FECollection(1, dim);
         coarse_fes = new ParFiniteElementSpace(mesh, fec);
         break;
      case elasticity:
         fec = new H1_FECollection(1, dim);
         coarse_fes= new ParFiniteElementSpace(mesh, fec, dim);
         break;
      case maxwell:
         fec = new ND_FECollection(1, dim);
         coarse_fes= new ParFiniteElementSpace(mesh, fec);
         break;
      default:
         mfem_error("Invalid integrator type! Check FiniteElementCollection");
   }

   /// 6. Define a finite element space hierarchy for the multigrid solver.
   ///    Define a FEC array for the order-refinement levels. Add the refinements
   ///    to the hierarchy.
   Array<FiniteElementCollection*> fec_array;
   fec_array.Append(fec);
   // Transfer ownership of mesh and coarse_fes to fes_hierarchy
   ParFiniteElementSpaceHierarchy* fes_hierarchy = new
   ParFiniteElementSpaceHierarchy(mesh, coarse_fes, true, true);

   for (int lg = 0; lg < geometric_levels; ++lg)
   {
      fes_hierarchy->AddUniformlyRefinedLevel();
   }
   for (int lo = 0; lo < order_levels; ++lo)
   {
      switch (integrator_type)
      {
         case mass: case diffusion: case elasticity:
            fec_array.Append(new H1_FECollection(std::pow(2, lo + 1), dim));
            break;
         case maxwell:
            fec_array.Append(new ND_FECollection(std::pow(2, lo + 1), dim));
            break;
         default:
            mfem_error("Invalid integrator type! "
                       "Check FiniteElementCollection for order refinements...");
      }
      fes_hierarchy->AddOrderRefinedLevel(fec_array.Last());
   }

   HYPRE_BigInt sys_size = fes_hierarchy->GetFinestFESpace().GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << "Number of unknowns: " << sys_size << endl;
   }

   /// 7. Extract the list of the essential boundary DoFs. We mark all boundary
   ///    attibutes as essential. GeneralGeometricMultigrid will determine
   ///    the DoFs per level.
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   /// 8. Define the linear system. Set up the linear form b(.).
   ///    The linear form has the standar form (f,v).
   ///    Define the matrices and vectors associated to the forms, and project
   ///    the required boundary data into the GridFunction solution.
   ParLinearForm *b = new ParLinearForm(&fes_hierarchy->GetFinestFESpace());
   LinearFormIntegrator *lfi = nullptr;

   // These pointers are not owned by the integrators
   FunctionCoefficient *scalar_u = nullptr;
   FunctionCoefficient *scalar_f = nullptr;
   VectorFunctionCoefficient *vector_u = nullptr;
   VectorFunctionCoefficient *vector_f = nullptr;

   ConstantCoefficient one(1.0);

   // These variables will define the linear system
   ParGridFunction x(&fes_hierarchy->GetFinestFESpace());
   OperatorPtr A(Operator::Type::Hypre_ParCSR);
   Vector B, X;

   x = 0.0;

   switch (integrator_type)
   {
      case mass:
         scalar_u = new FunctionCoefficient(diffusion_solution);
         lfi = new DomainLFIntegrator(*scalar_u);
         x.ProjectBdrCoefficient(*scalar_u, ess_bdr);
         break;
      case diffusion:
         scalar_u = new FunctionCoefficient(diffusion_solution);
         scalar_f = new FunctionCoefficient(diffusion_source);
         lfi = new DomainLFIntegrator(*scalar_f);
         x.ProjectBdrCoefficient(*scalar_u, ess_bdr);
         break;
      case elasticity:
         vector_u = new VectorFunctionCoefficient(space_dim, elasticity_solution);
         vector_f = new VectorFunctionCoefficient(space_dim, elasticity_source);
         lfi = new VectorDomainLFIntegrator(*vector_f);
         x.ProjectBdrCoefficient(*vector_u, ess_bdr);
         break;
      case maxwell:
         vector_u = new VectorFunctionCoefficient(space_dim, maxwell_solution);
         vector_f = new VectorFunctionCoefficient(space_dim, maxwell_source);
         lfi = new VectorFEDomainLFIntegrator(*vector_f);
         x.ProjectBdrCoefficientTangent(*vector_u, ess_bdr);
         break;
      default:
         mfem_error("Invalid integrator type! Check ParLinearForm");
   }
   b->AddDomainIntegrator(lfi);
   b->Assemble();

   /// 9. Define a geometric multigrid solver. The bilinear form
   ///    a(.,.) is assembled internally. Set up the type of cycles
   ///    and form the linear system.
   GeneralGeometricMultigrid* mg = new GeneralGeometricMultigrid(*fes_hierarchy,
                                                                 ess_bdr,
                                                                 integrator_type,
                                                                 solver_type,
                                                                 p_order,
                                                                 q_order);
   mg->SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);
   mg->FormFineLinearSystem(x, *b, A, X, B);

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
   solver->SetOperator(*A.Ptr());

   IterativeSolver *it_solver = dynamic_cast<IterativeSolver *>(solver);
   if (it_solver)
   {
      it_solver->SetRelTol(rel_tol);
      it_solver->SetMaxIter(max_iter);
      it_solver->SetPrintLevel(1);
      // it_solver->SetMonitor(monitor);
      it_solver->SetPreconditioner(*mg);
   }
   solver->Mult(B, X);

   /// 10. Recover the solution x as a grid function. Send the data by socket
   ///     to a GLVis server.
   mg->RecoverFineFEMSolution(X, *b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *fes_hierarchy->GetFinestFESpace().GetParMesh()
               << x << flush;
   }

   /// 11. Free the memory used
   delete mg;
   delete solver;
   delete b;
   if (scalar_u) { delete scalar_u; }
   if (scalar_f) { delete scalar_f; }
   if (vector_u) { delete vector_u; }
   if (vector_f) { delete vector_f; }
   delete fes_hierarchy;
   return 0;
}
