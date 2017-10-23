#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

/** After spatial discretization, the rubber model can be written as:
 *     0=H(x)
 *  where x is the block vector representing the deformation and pressure
 *  and H(x) is the nonlinear incompressible neo-Hookean operator. */
class RubberOperator : public Operator
{
protected:
   Array<ParFiniteElementSpace *> spaces;

   ParBlockNonlinearForm *Hform;
   mutable BlockOperator *Jacobian;
   const BlockVector *x;

   /// Newton solver for the hyperelastic operator
   NewtonSolver newton_solver;
   /// Solver for the Jacobian solve in the Newton method
   mutable Solver *J_solver;
   /// Preconditioner for the Jacobian
   mutable Solver *J_prec;
   mutable HypreBoomerAMG *invK, *invS;
   mutable HypreParMatrix *K, *B, *KinvBt, *S;
   mutable HypreParVector *Kd;

   Coefficient &mu;

   Array<int> &block_trueOffsets;

public:
   RubberOperator(Array<ParFiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr,
                  Array<int> &block_trueOffsets, double rel_tol, double abs_tol, int iter,
                  Coefficient &mu);

   /// Required to use the native newton solver
   virtual Operator &GetGradientSolver(const Vector &xp) const;
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Driver for the newton solver
   void Solve(Vector &xp) const;

   virtual ~RubberOperator();
};

void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name = NULL,
               bool init_vis = false);

void ReferenceConfiguration(const Vector &x, Vector &y);
void InitialDeformation(const Vector &x, Vector &y);


int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   const char *mesh_file = "../data/beam-hex.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 2;
   bool visualization = true;
   double newton_rel_tol = 1.0e-6;
   double newton_abs_tol = 1.0e-8;
   int newton_iter = 500;
   double mu = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&newton_rel_tol, "-rel", "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol, "-abs", "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter, "-it", "--newton-iterations",
                  "Maximum iterations for the Newton solve.");
   args.AddOption(&mu, "-mu", "--shear-modulus",
                  "Shear modulus for the neo-Hookean material.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Open the mesh
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      }
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   ParMesh *pmesh = NULL;

   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   delete mesh;
   int dim = pmesh->Dimension();

   ConstantCoefficient c_mu(mu);

   // Definie the finite element spaces for displacement and pressure (Stokes elements)
   H1_FECollection quad_coll(order, dim);
   H1_FECollection lin_coll(order-1, dim);

   ParFiniteElementSpace R_space(pmesh, &quad_coll, dim);
   ParFiniteElementSpace W_space(pmesh, &lin_coll);

   Array<ParFiniteElementSpace *> spaces(2);
   spaces[0] = &R_space;
   spaces[1] = &W_space;

   HYPRE_Int glob_R_size = R_space.GlobalTrueVSize();
   HYPRE_Int glob_W_size = W_space.GlobalTrueVSize();

   // Define the Dirichlet conditions (set to boundary attribute 1)
   Array<Array<int> *> ess_bdr(2);

   Array<int> ess_bdr_u(R_space.GetMesh()->bdr_attributes.Max());
   Array<int> ess_bdr_p(W_space.GetMesh()->bdr_attributes.Max());

   ess_bdr_p = 0;
   ess_bdr_u = 0;
   ess_bdr_u[0] = 1;
   ess_bdr_u[1] = 1;

   ess_bdr[0] = &ess_bdr_u;
   ess_bdr[1] = &ess_bdr_p;

   // Print the mesh statistics
   if (myid == 0)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(u) = " << glob_R_size << "\n";
      std::cout << "dim(p) = " << glob_W_size << "\n";
      std::cout << "dim(u+p) = " << glob_R_size + glob_W_size << "\n";
      std::cout << "***********************************************************\n";
   }

   // Define the block structure of the solution vector (u then p)
   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = R_space.GetVSize();
   block_offsets[2] = W_space.GetVSize();
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(3);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = R_space.TrueVSize();
   block_trueOffsets[2] = W_space.TrueVSize();
   block_trueOffsets.PartialSum();

   BlockVector xp(block_trueOffsets);

   // Define grid functions for the current configuration, reference configuration,
   // final deformation, and pressure
   ParGridFunction x_gf(&R_space);
   ParGridFunction x_ref(&R_space);
   ParGridFunction x_def(&R_space);
   ParGridFunction p_gf(&W_space);

   // Project the initial and reference configuration functions onto the appropriate grid functions
   VectorFunctionCoefficient deform(dim, InitialDeformation);
   VectorFunctionCoefficient refconfig(dim, ReferenceConfiguration);

   x_gf.ProjectCoefficient(deform);
   x_ref.ProjectCoefficient(refconfig);

   // Set up the block solution vectors
   x_gf.GetTrueDofs(xp.GetBlock(0));
   p_gf.GetTrueDofs(xp.GetBlock(1));

   // Initialize the incompressible neo-Hookean operator
   RubberOperator oper(spaces, ess_bdr, block_trueOffsets,
                       newton_rel_tol, newton_abs_tol, newton_iter, c_mu);

   // Solve the Newton system
   oper.Solve(xp);

   // Distribute the ghost dofs
   x_gf.Distribute(xp.GetBlock(0));
   p_gf.Distribute(xp.GetBlock(1));

   // Set the final deformation
   subtract(x_gf, x_ref, x_def);

   // Visualize the results if requested
   socketstream vis_u, vis_p;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_u.open(vishost, visport);
      vis_u.precision(8);
      visualize(vis_u, pmesh, &x_gf, &x_def, "Deformation", true);
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      vis_p.open(vishost, visport);
      vis_p.precision(8);
      visualize(vis_p, pmesh, &x_gf, &p_gf, "Pressure", true);
   }

   // Save the displaced mesh, the final deformation, and the pressure
   {
      GridFunction *nodes = &x_gf;
      int owns_nodes = 0;
      pmesh->SwapNodes(nodes, owns_nodes);

      ostringstream mesh_name, pressure_name, deformation_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      pressure_name << "pressure." << setfill('0') << setw(6) << myid;
      deformation_name << "deformation." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream pressure_ofs(pressure_name.str().c_str());
      pressure_ofs.precision(8);
      p_gf.Save(pressure_ofs);

      ofstream deformation_ofs(deformation_name.str().c_str());
      deformation_ofs.precision(8);
      x_def.Save(deformation_ofs);
   }


   // Free the used memory.
   delete pmesh;

   MPI_Finalize();

   return 0;
}

RubberOperator::RubberOperator(Array<ParFiniteElementSpace *> &fes,
                               Array<Array<int> *> &ess_bdr,
                               Array<int> &trueOffsets,
                               double rel_tol,
                               double abs_tol,
                               int iter,
                               Coefficient &c_mu)
   : Operator(fes[0]->TrueVSize() + fes[1]->TrueVSize()),
     newton_solver(fes[0]->GetComm(), true), mu(c_mu), block_trueOffsets(trueOffsets)
{
   Array<Vector *> rhs(2);

   rhs[0] = NULL;
   rhs[1] = NULL;

   fes.Copy(spaces);

   // Define the mixed nonlinear form
   Hform = new ParBlockNonlinearForm(spaces);

   // Add the passive stress integrator
   Hform->AddDomainIntegrator(new IncompressibleNeoHookeanIntegrator(mu));

   // Set the essential boundary conditions
   Hform->SetEssentialBC(ess_bdr, rhs);
   // Set the newton solve parameters
   newton_solver.iterative_mode = true;

   newton_solver.SetOperator(*this);
   newton_solver.SetPrintLevel(1);
   newton_solver.SetRelTol(rel_tol);
   newton_solver.SetAbsTol(abs_tol);
   newton_solver.SetMaxIter(iter);

   J_solver = NULL;
   J_prec = NULL;
   invK = NULL;
   invS = NULL;

}

// Solve the Newton system
void RubberOperator::Solve(Vector &xp) const
{
   Vector zero;
   newton_solver.Mult(zero, xp);
   MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");
}

// compute: y = H(x,p)
void RubberOperator::Mult(const Vector &k, Vector &y) const
{
   Hform->Mult(k, y);

}

// Compute the Jacobian from the nonlinear form
Operator &RubberOperator::GetGradientSolver(const Vector &xp) const
{

   Jacobian = &Hform->GetGradient(xp);

   if (J_solver != NULL) {
      delete J_solver;
      delete invK;
      delete invS;
      delete KinvBt, Kd, S;
   }


   K = (HypreParMatrix*) &Jacobian->GetBlock(0,0);
   B = (HypreParMatrix*) &Jacobian->GetBlock(1,0);

   *B *= -1.0; 

   KinvBt = B->Transpose();
   Kd = new HypreParVector(MPI_COMM_WORLD, K->GetGlobalNumRows(),
                           K->GetRowStarts());
   K->GetDiag(*Kd);

   KinvBt->InvScaleRows(*Kd);
   S = ParMult(B, KinvBt);

   invK = new HypreBoomerAMG(*K);
   invS = new HypreBoomerAMG(*S);

   invK->SetPrintLevel(0);
   invS->SetPrintLevel(0);

   BlockDiagonalPreconditioner *blockPr = new BlockDiagonalPreconditioner(block_trueOffsets);
   blockPr->SetDiagonalBlock(0, invK);
   blockPr->SetDiagonalBlock(1, invS);

   MINRESSolver *solver = new MINRESSolver(MPI_COMM_WORLD);
   solver->SetAbsTol(1.0e-4);
   solver->SetRelTol(1.0e-4);
   solver->SetMaxIter(100000);
   solver->SetOperator(*Jacobian);
   solver->SetPreconditioner(*blockPr);
   solver->SetPrintLevel(0);

   J_solver = solver;

   return *J_solver;
}

RubberOperator::~RubberOperator()
{
   delete J_solver;
   if (J_prec != NULL)
   {
      delete J_prec;
   }
}

// In line visualization
void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name, bool init_vis)
{
   if (!out)
   {
      return;
   }

   GridFunction *nodes = deformed_nodes;
   int owns_nodes = 0;

   mesh->SwapNodes(nodes, owns_nodes);

   out << "parallel " << mesh->GetNRanks() << " " << mesh->GetMyRank() << "\n";
   out << "solution\n" << *mesh << *field;

   mesh->SwapNodes(nodes, owns_nodes);

   if (init_vis)
   {
      out << "window_size 800 800\n";
      out << "window_title '" << field_name << "'\n";
      if (mesh->SpaceDimension() == 2)
      {
         out << "view 0 0\n"; // view from top
         out << "keys jl\n";  // turn off perspective and light
      }
      out << "keys cm\n";         // show colorbar and mesh
      out << "autoscale value\n"; // update value-range; keep mesh-extents fixed
      out << "pause\n";
   }
   out << flush;
}

void ReferenceConfiguration(const Vector &x, Vector &y)
{
   // set the reference, stress
   // free, configuration
   y = x;
}


void InitialDeformation(const Vector &x, Vector &y)
{
   // set the initial configuration. Having this different from the
   // reference configuration can help convergence
   y = x;
   y[1] = x[1] + 0.25*x[0];
}
