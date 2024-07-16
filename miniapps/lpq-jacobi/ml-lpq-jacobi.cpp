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
   Mpi::Init();
   Hypre::Init();

   // TODO(Gabriel): simpler default mesh to be defined by the type if left blank
   string mesh_file = "meshes/icf.mesh";
   // System properties
   int num_levels = 1;
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
   // TODO(Gabriel): To be added later
   // const char *device_config = "cpu";
   // bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&geometric_levels, "-gl", "--geometric-levels",
                  "Number of geometric refinements (levels) done prior to order refinements.");
   args.AddOption(&order_levels, "-ol", "--order-levels",
                  "Number of order refinements (levels). "
                  "Finest level in the hierarchy has order 2^{or}.");
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
   // args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
   //                "--no-visualization",
   //                "Enable or disable GLVis visualization.");
   args.ParseCheck();

   MFEM_ASSERT(p_order > 0.0, "p needs to be positive");
   MFEM_ASSERT(geometric_levels >= 0, "geometric_level needs to be non-negative");
   MFEM_ASSERT(order_levels >= 0, "order_level needs to be non-negative");
   // MFEM_ASSERT(geometric_levels + order_levels > 0, ""); // TODO(Gabriel): Do i require this?
   MFEM_ASSERT((0 <= integrator_type) && (integrator_type < num_integrators), "");
   MFEM_ASSERT(0.0 < eps_y <= 1.0, "eps_y in (0,1]");
   MFEM_ASSERT(0.0 < eps_z <= 1.0, "eps_z in (0,1]");

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

   // Create fec
   // Create coarse_fes
   // Create array fec_array
   // append first fec
   // Create fes_hierarchy (transfer mesh and coarse_fes)
   // Add mesh refinement in a loop (geom)
   // Add order refinement in loop (creates)
   //   Create fec of higher order and same dimension and append to fec_array
   //   Add order refined level
   int dim = mesh->Dimension();
   FiniteElementCollection *fec;
   ParFiniteElementSpace *coarse_fes;
   Array<FiniteElementCollection*> fec_array;
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
   fec_array.Append(fec);
   // Transfer ownership of mesh and coarse_fes to fes_hierarchy
   ParFiniteElementSpaceHierarchy* fes_hierarchy = new
   ParFiniteElementSpaceHierarchy(
      mesh, coarse_fes, true, true);
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

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   switch (integrator_type)
   {
      case mass: case diffusion: case maxwell:
         ess_bdr = 1;
         break;
      case elasticity:
         ess_bdr = 0;
         ess_bdr[0] = 1;
         break;
      default:
         mfem_error("Invalid integrator type! Check GetEssentialTrueDofs");
   }

   fes_hierarchy->GetFinestFESpace().GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ParLinearForm *b = new ParLinearForm(&fes_hierarchy->GetFinestFESpace());
   ConstantCoefficient one(1.0);
   VectorArrayCoefficient *f = nullptr;
   switch (integrator_type)
   {
      case mass: case diffusion:
         b->AddDomainIntegrator(new DomainLFIntegrator(one));
         break;
      case elasticity:
         f = new VectorArrayCoefficient(dim);
         for (int i = 0; i < dim; i++)
         {
            f->Set(i, &one);
         }
         b->AddDomainIntegrator(new VectorDomainLFIntegrator(*f));
         break;
      case maxwell:
         f = new VectorArrayCoefficient(dim);
         for (int i = 0; i < dim; i++)
         {
            f->Set(i, &one);
         }
         b->AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(*f));
         break;
      default:
         mfem_error("Invalid integrator type! Check ParLinearForm");
   }
   b->Assemble();

   // ParBilinearForm *a = new ParBilinearForm(fespace);
   // switch (integrator_type)
   // {
   //    case mass:
   //       a->AddDomainIntegrator(new MassIntegrator);
   //       break;
   //    case diffusion:
   //       a->AddDomainIntegrator(new DiffusionIntegrator);
   //       break;
   //    case elasticity:
   //       // TODO(Gabriel): Add lambda/mu
   //       a->AddDomainIntegrator(new ElasticityIntegrator(one, one));
   //       break;
   //    case maxwell:
   //       // TODO(Gabriel): Add muinv/sigma
   //       a->AddDomainIntegrator(new CurlCurlIntegrator(one));
   //       a->AddDomainIntegrator(new VectorFEMassIntegrator(one));
   //       break;
   //    default:
   //       mfem_error("Invalid integrator type! Check ParBilinearForm");
   // }
   // a->Assemble();

   ParGridFunction x(&fes_hierarchy->GetFinestFESpace());
   HypreParMatrix A;
   Vector B, X;

   x = 0.0;

   // a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // // D_{p,q} = diag( D^{1+q-p} |A|^p D^{-q} 1) , where D = diag(A)
   // // TODO(Chak): Make into one function!
   // Vector right(A.Height());
   // Vector temp(A.Height());
   // Vector left(A.Height());

   // // D^{-q} 1
   // right = 1.0;
   // if (q_order !=0)
   // {
   //    A.GetDiag(right);
   //    right.PowerAbs(-q_order);
   // }

   // // |A|^p D^{-q} 1
   // temp = 0.0;
   // A.PowAbsMult(p_order, 1.0, right, 0.0, temp);

   // // D^{1+q-p} |A|^p D^{-q} 1
   // left = temp;
   // if (1.0 + q_order - p_order != 0.0)
   // {
   //    A.GetDiag(left);
   //    left.PowerAbs(1.0 + q_order - p_order);
   //    left *= temp;
   // }

   // // diag(...)
   // auto lpq_jacobi = new OperatorJacobiSmoother(left, ess_tdof_list);
   // TODO(Chak): Make into one function! Into a class!

   // Solver *solver = nullptr;
   // DataMonitor monitor(file_name.str(), NDIGITS);
   // IterativeSolver *it_solver = dynamic_cast<IterativeSolver *>(solver);
   // if (it_solver)
   // {
   //    it_solver->SetRelTol(rel_tol);
   //    it_solver->SetMaxIter(max_iter);
   //    it_solver->SetPrintLevel(1);
   //    it_solver->SetPreconditioner(*lpq_jacobi);
   //    it_solver->SetMonitor(monitor);
   // }
   // solver->SetOperator(A);
   // solver->Mult(B, X);

   // if (visualization)
   // {
   //    a->RecoverFEMSolution(X, *b, x);
   //    x.Save("sol");
   //    mesh->Save("mesh");
   // }

   // delete solver;
   // delete lpq_jacobi;
   // delete a;
   delete b;
   // if (f){ delete f; }
   delete fes_hierarchy;
   delete mesh;

   return 0;
}
