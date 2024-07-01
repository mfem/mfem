// TODO(Gabriel): To decide if this should be an example or not
//
// Compile with: make TODO
//
// Sample runs: mpirun -np 4 ...
//
// Description:

#include "mfem.hpp"
#include "../miniapps/common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

enum SolverType
{
   sli,
   cg,
   num_solvers,  // last
};

enum IntegratorType
{
   mass,
   diffusion,
   elasticity,
   num_integrators,  // last
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   // TODO(Gabriel): simpler default mesh...
   string mesh_file = "../data/star.mesh";
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
   // bool visualization = true;

   // Create name for data_file
   // TODO(Gabriel): To put this on its own scope?

   // Extract the base name from the path
   // string base_name = mesh_file.substr(mesh_file.find_last_of("/\\") + 1);
   // base_name = base_name.substr(0, base_name.find_last_of('.'));

   // // Create an output string using ostringstream
   // std::ostringstream oss;
   // oss << base_name << "-n" << n;

   // std::string output = oss.str();

   // // Print the result
   // std::cout << "Output: " << output << std::endl;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   // TODO(Gabriel): TO be added later
   // " or -1 for isoparametric space.");
   args.AddOption((int*)&solver_type, "-s", "--solver",
                  "Solvers to be considered:"
                  "\n\t0: Stationary Linear Iteration"
                  "\n\t1: Preconditioned Conjugate Gradient"
                  "\n\tTODO");
   args.AddOption((int*)&integrator_type, "-i", "--integrator",
                  "Integrators to be considered:"
                  "\n\t0: MassIntegrator"
                  "\n\t1: DiffusionIntegrator"
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
   args.ParseCheck();

   MFEM_ASSERT(p_order > 0.0, "p needs to be positive");
   MFEM_ASSERT((0 <= solver_type) && (solver_type < num_solvers), "");
   MFEM_ASSERT((0 <= integrator_type) && (integrator_type < num_integrators), "");
   MFEM_ASSERT(0.0 < eps_y <= 1.0, "eps_y in (0,1]");
   MFEM_ASSERT(0.0 < eps_z <= 1.0, "eps_z in (0,1]");

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

   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   int dim = mesh->Dimension();
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
      default:
         mfem_error("Invalid integrator type!");
   }

   HYPRE_BigInt sys_size = fespace->GlobalTrueVSize();
   //if (id == 0)
   if (Mpi::Root())
   {
      mfem::out << "Number of unknowns: " << sys_size << endl;
   }

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   switch (integrator_type)
   {
      case mass: case diffusion:
         ess_bdr = 1;
         break;
      case elasticity:
         ess_bdr = 0;
         ess_bdr[0] = 1;
         break;
      default:
         mfem_error("Invalid integrator type!");
   }
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ParLinearForm *b = new ParLinearForm(fespace);
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
         b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*f));
         break;
      default:
         mfem_error("Invalid integrator type!");
   }
   b->Assemble();

   ParBilinearForm *a = new ParBilinearForm(fespace);
   switch (integrator_type)
   {
      case mass:
         a->AddDomainIntegrator(new MassIntegrator);
         break;
      case diffusion:
         a->AddDomainIntegrator(new DiffusionIntegrator);
         break;
      case elasticity:
         // TODO(Gabriel): Add lambda/mu
         a->AddDomainIntegrator(new ElasticityIntegrator(one, one));
         break;
      default:
         mfem_error("Invalid integrator type!");
   }
   a->Assemble();

   ParGridFunction x(fespace);
   HypreParMatrix A;
   Vector B, X;

   x = -1.0;

   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // D_{p,q} = diag( D^{1+q-p} |A|^p D^{-q} 1) , where D = diag(A)
   Vector right(A.Height());
   Vector temp(A.Height());
   Vector left(A.Height());

   // D^{-q} 1
   right = 1.0;
   if (q_order !=0)
   {
      A.GetDiag(right);
      right.PowerAbs(-q_order);
   }

   // |A|^p D^{-q} 1
   temp = 0.0;
   A.PowAbsMult(p_order, 1.0, right, 0.0, temp);

   // D^{1+q-p} |A|^p D^{-q} 1
   left = temp;
   if (1.0 + q_order - p_order != 0.0)
   {
      A.GetDiag(left);
      left.PowerAbs(1.0 + q_order - p_order);
      left *= temp;
   }

   // diag(...)
   auto lpq_jacobi = new OperatorJacobiSmoother(left, ess_tdof_list);

   Solver *solver = nullptr;
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
   }
   solver->SetOperator(A);
   solver->Mult(B, X);

   a->RecoverFEMSolution(X, *b, x);
   x.Save("sol");
   mesh->Save("mesh");

   delete solver;
   delete lpq_jacobi;
   delete a;
   delete b;
   // if (f){ delete f; }
   delete fespace;
   delete fec;
   delete mesh;
}
