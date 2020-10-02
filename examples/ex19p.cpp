//                       MFEM Example 19 - Parallel Version
//
// Compile with: make ex19p
//
// Sample runs:
//    mpirun -np 2 ex19p -m ../data/beam-quad.mesh
//    mpirun -np 2 ex19p -m ../data/beam-tri.mesh
//    mpirun -np 2 ex19p -m ../data/beam-hex.mesh
//    mpirun -np 2 ex19p -m ../data/beam-tet.mesh
//    mpirun -np 2 ex19p -m ../data/beam-wedge.mesh
//    mpirun -np 2 ex19p -m ../data/beam-quad-amr.mesh
//
// Description:  This examples solves a quasi-static incompressible nonlinear
//               elasticity problem of the form 0 = H(x), where H is an
//               incompressible hyperelastic model and x is a block state vector
//               containing displacement and pressure variables. The geometry of
//               the domain is assumed to be as follows:
//
//                                 +---------------------+
//                    boundary --->|                     |<--- boundary
//                    attribute 1  |                     |     attribute 2
//                    (fixed)      +---------------------+     (fixed, nonzero)
//
//               The example demonstrates the use of block nonlinear operators
//               (the class RubberOperator defining H(x)) as well as a nonlinear
//               Newton solver for the quasi-static problem. Each Newton step
//               requires the inversion of a Jacobian matrix, which is done
//               through a (preconditioned) inner solver. The specialized block
//               preconditioner is implemented as a user-defined solver.
//
//               We recommend viewing examples 2, 5, and 10 before viewing this
//               example.

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

class GeneralResidualMonitor : public IterativeSolverMonitor
{
public:
   GeneralResidualMonitor(MPI_Comm comm, const std::string& prefix_,
                          int print_lvl)
      : prefix(prefix_)
   {
#ifndef MFEM_USE_MPI
      print_level = print_lvl;
#else
      int rank;
      MPI_Comm_rank(comm, &rank);
      if (rank == 0)
      {
         print_level = print_lvl;
      }
      else
      {
         print_level = -1;
      }
#endif
   }

   virtual void MonitorResidual(int it, double norm, const Vector &r, bool final);

private:
   const std::string prefix;
   int print_level;
   mutable double norm0;
};

void GeneralResidualMonitor::MonitorResidual(int it, double norm,
                                             const Vector &r, bool final)
{
   if (print_level == 1 || (print_level == 3 && (final || it == 0)))
   {
      mfem::out << prefix << " iteration " << setw(2) << it
                << " : ||r|| = " << norm;
      if (it > 0)
      {
         mfem::out << ",  ||r||/||r_0|| = " << norm/norm0;
      }
      else
      {
         norm0 = norm;
      }
      mfem::out << '\n';
   }
}

// Custom block preconditioner for the Jacobian of the incompressible nonlinear
// elasticity operator. It has the form
//
// P^-1 = [ K^-1 0 ][ I -B^T ][ I  0           ]
//        [ 0    I ][ 0  I   ][ 0 -\gamma S^-1 ]
//
// where the original Jacobian has the form
//
// J = [ K B^T ]
//     [ B 0   ]
//
// and K^-1 is an approximation of the inverse of the displacement part of the
// Jacobian and S^-1 is an approximation of the inverse of the Schur
// complement S = B K^-1 B^T. The Schur complement is approximated using
// a mass matrix of the pressure variables.
class JacobianPreconditioner : public Solver
{
protected:
   // Finite element spaces for setting up preconditioner blocks
   Array<ParFiniteElementSpace *> spaces;

   // Offsets for extracting block vector segments
   Array<int> &block_trueOffsets;

   // Jacobian for block access
   BlockOperator *jacobian;

   // Scaling factor for the pressure mass matrix in the block preconditioner
   double gamma;

   // Objects for the block preconditioner application
   Operator *pressure_mass;
   Solver *mass_pcg;
   Solver *mass_prec;
   Solver *stiff_pcg;
   Solver *stiff_prec;

public:
   JacobianPreconditioner(Array<ParFiniteElementSpace *> &fes,
                          Operator &mass, Array<int> &offsets);

   virtual void Mult(const Vector &k, Vector &y) const;
   virtual void SetOperator(const Operator &op);

   virtual ~JacobianPreconditioner();
};

// After spatial discretization, the rubber model can be written as:
//     0 = H(x)
// where x is the block vector representing the deformation and pressure and
// H(x) is the nonlinear incompressible neo-Hookean operator.
class RubberOperator : public Operator
{
protected:
   // Finite element spaces
   Array<ParFiniteElementSpace *> spaces;

   // Block nonlinear form
   ParBlockNonlinearForm *Hform;

   // Pressure mass matrix for the preconditioner
   Operator *pressure_mass;

   // Newton solver for the hyperelastic operator
   NewtonSolver newton_solver;
   GeneralResidualMonitor newton_monitor;

   // Solver for the Jacobian solve in the Newton method
   Solver *j_solver;
   GeneralResidualMonitor j_monitor;

   // Preconditioner for the Jacobian
   Solver *j_prec;

   // Shear modulus coefficient
   Coefficient &mu;

   // Block offsets for variable access
   Array<int> &block_trueOffsets;

public:
   RubberOperator(Array<ParFiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr,
                  Array<int> &block_trueOffsets, double rel_tol, double abs_tol,
                  int iter, Coefficient &mu);

   // Required to use the native newton solver
   virtual Operator &GetGradient(const Vector &xp) const;
   virtual void Mult(const Vector &k, Vector &y) const;

   // Driver for the newton solver
   void Solve(Vector &xp) const;

   virtual ~RubberOperator();
};

// Visualization driver
void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name = NULL,
               bool init_vis = false);

// Configuration definition functions
void ReferenceConfiguration(const Vector &x, Vector &y);
void InitialDeformation(const Vector &x, Vector &y);


int main(int argc, char *argv[])
{
   // 1. Initialize MPI
   MPI_Session mpi;
   const int myid = mpi.WorldRank();

   // 2. Parse command-line options
   const char *mesh_file = "../data/beam-tet.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 2;
   bool visualization = true;
   double newton_rel_tol = 1e-4;
   double newton_abs_tol = 1e-6;
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
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define the shear modulus for the incompressible Neo-Hookean material
   ConstantCoefficient c_mu(mu);

   // 7. Define the finite element spaces for displacement and pressure
   //    (Taylor-Hood elements). By default, the displacement (u/x) is a second
   //    order vector field, while the pressure (p) is a linear scalar function.
   H1_FECollection quad_coll(order, dim);
   H1_FECollection lin_coll(order-1, dim);

   ParFiniteElementSpace R_space(pmesh, &quad_coll, dim, Ordering::byVDIM);
   ParFiniteElementSpace W_space(pmesh, &lin_coll);

   Array<ParFiniteElementSpace *> spaces(2);
   spaces[0] = &R_space;
   spaces[1] = &W_space;

   HYPRE_Int glob_R_size = R_space.GlobalTrueVSize();
   HYPRE_Int glob_W_size = W_space.GlobalTrueVSize();

   // 8. Define the Dirichlet conditions (set to boundary attribute 1 and 2)
   Array<Array<int> *> ess_bdr(2);

   Array<int> ess_bdr_u(R_space.GetMesh()->bdr_attributes.Max());
   Array<int> ess_bdr_p(W_space.GetMesh()->bdr_attributes.Max());

   ess_bdr_p = 0;
   ess_bdr_u = 0;
   ess_bdr_u[0] = 1;
   ess_bdr_u[1] = 1;

   ess_bdr[0] = &ess_bdr_u;
   ess_bdr[1] = &ess_bdr_p;

   // 9. Print the mesh statistics
   if (myid == 0)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(u) = " << glob_R_size << "\n";
      std::cout << "dim(p) = " << glob_W_size << "\n";
      std::cout << "dim(u+p) = " << glob_R_size + glob_W_size << "\n";
      std::cout << "***********************************************************\n";
   }

   // 10. Define the block structure of the solution vector (u then p)
   Array<int> block_trueOffsets(3);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = R_space.TrueVSize();
   block_trueOffsets[2] = W_space.TrueVSize();
   block_trueOffsets.PartialSum();

   BlockVector xp(block_trueOffsets);

   // 11. Define grid functions for the current configuration, reference
   //     configuration, final deformation, and pressure
   ParGridFunction x_gf(&R_space);
   ParGridFunction x_ref(&R_space);
   ParGridFunction x_def(&R_space);
   ParGridFunction p_gf(&W_space);

   VectorFunctionCoefficient deform(dim, InitialDeformation);
   VectorFunctionCoefficient refconfig(dim, ReferenceConfiguration);

   x_gf.ProjectCoefficient(deform);
   x_ref.ProjectCoefficient(refconfig);
   p_gf = 0.0;

   // 12. Set up the block solution vectors
   x_gf.GetTrueDofs(xp.GetBlock(0));
   p_gf.GetTrueDofs(xp.GetBlock(1));

   // 13. Initialize the incompressible neo-Hookean operator
   RubberOperator oper(spaces, ess_bdr, block_trueOffsets,
                       newton_rel_tol, newton_abs_tol, newton_iter, c_mu);

   // 14. Solve the Newton system
   oper.Solve(xp);

   // 15. Distribute the shared degrees of freedom
   x_gf.Distribute(xp.GetBlock(0));
   p_gf.Distribute(xp.GetBlock(1));

   // 16. Compute the final deformation
   subtract(x_gf, x_ref, x_def);

   // 17. Visualize the results if requested
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

   // 18. Save the displaced mesh, the final deformation, and the pressure
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

   // 19. Free the used memory
   delete pmesh;

   return 0;
}


JacobianPreconditioner::JacobianPreconditioner(Array<ParFiniteElementSpace *>
                                               &fes,
                                               Operator &mass,
                                               Array<int> &offsets)
   : Solver(offsets[2]), block_trueOffsets(offsets), pressure_mass(&mass)
{
   fes.Copy(spaces);

   gamma = 0.00001;

   // The mass matrix and preconditioner do not change every Newton cycle, so
   // we only need to define them once
   HypreBoomerAMG *mass_prec_amg = new HypreBoomerAMG();
   mass_prec_amg->SetPrintLevel(0);

   mass_prec = mass_prec_amg;

   CGSolver *mass_pcg_iter = new CGSolver(spaces[0]->GetComm());
   mass_pcg_iter->SetRelTol(1e-12);
   mass_pcg_iter->SetAbsTol(1e-12);
   mass_pcg_iter->SetMaxIter(200);
   mass_pcg_iter->SetPrintLevel(0);
   mass_pcg_iter->SetPreconditioner(*mass_prec);
   mass_pcg_iter->SetOperator(*pressure_mass);
   mass_pcg_iter->iterative_mode = false;

   mass_pcg = mass_pcg_iter;

   // The stiffness matrix does change every Newton cycle, so we will define it
   // during SetOperator
   stiff_pcg = NULL;
   stiff_prec = NULL;
}

void JacobianPreconditioner::Mult(const Vector &k, Vector &y) const
{
   // Extract the blocks from the input and output vectors
   Vector disp_in(k.GetData() + block_trueOffsets[0],
                  block_trueOffsets[1]-block_trueOffsets[0]);
   Vector pres_in(k.GetData() + block_trueOffsets[1],
                  block_trueOffsets[2]-block_trueOffsets[1]);

   Vector disp_out(y.GetData() + block_trueOffsets[0],
                   block_trueOffsets[1]-block_trueOffsets[0]);
   Vector pres_out(y.GetData() + block_trueOffsets[1],
                   block_trueOffsets[2]-block_trueOffsets[1]);

   Vector temp(block_trueOffsets[1]-block_trueOffsets[0]);
   Vector temp2(block_trueOffsets[1]-block_trueOffsets[0]);

   // Perform the block elimination for the preconditioner
   mass_pcg->Mult(pres_in, pres_out);
   pres_out *= -gamma;

   jacobian->GetBlock(0,1).Mult(pres_out, temp);
   subtract(disp_in, temp, temp2);

   stiff_pcg->Mult(temp2, disp_out);
}

void JacobianPreconditioner::SetOperator(const Operator &op)
{
   jacobian = (BlockOperator *) &op;

   // Initialize the stiffness preconditioner and solver
   if (stiff_prec == NULL)
   {
      HypreBoomerAMG *stiff_prec_amg = new HypreBoomerAMG();
      stiff_prec_amg->SetPrintLevel(0);

      if (!spaces[0]->GetParMesh()->Nonconforming())
      {
         stiff_prec_amg->SetElasticityOptions(spaces[0]);
      }

      stiff_prec = stiff_prec_amg;

      GMRESSolver *stiff_pcg_iter = new GMRESSolver(spaces[0]->GetComm());
      stiff_pcg_iter->SetRelTol(1e-8);
      stiff_pcg_iter->SetAbsTol(1e-8);
      stiff_pcg_iter->SetMaxIter(200);
      stiff_pcg_iter->SetPrintLevel(0);
      stiff_pcg_iter->SetPreconditioner(*stiff_prec);
      stiff_pcg_iter->iterative_mode = false;

      stiff_pcg = stiff_pcg_iter;
   }

   // At each Newton cycle, compute the new stiffness AMG preconditioner by
   // updating the iterative solver which, in turn, updates its preconditioner
   stiff_pcg->SetOperator(jacobian->GetBlock(0,0));
}

JacobianPreconditioner::~JacobianPreconditioner()
{
   delete mass_pcg;
   delete mass_prec;
   delete stiff_prec;
   delete stiff_pcg;
}


RubberOperator::RubberOperator(Array<ParFiniteElementSpace *> &fes,
                               Array<Array<int> *> &ess_bdr,
                               Array<int> &trueOffsets,
                               double rel_tol,
                               double abs_tol,
                               int iter,
                               Coefficient &c_mu)
   : Operator(fes[0]->TrueVSize() + fes[1]->TrueVSize()),
     newton_solver(fes[0]->GetComm()),
     newton_monitor(fes[0]->GetComm(), "Newton", 1),
     j_monitor(fes[0]->GetComm(), "  GMRES", 3),
     mu(c_mu), block_trueOffsets(trueOffsets)
{
   Array<Vector *> rhs(2);
   rhs = NULL; // Set all entries in the array

   fes.Copy(spaces);

   // Define the block nonlinear form
   Hform = new ParBlockNonlinearForm(spaces);

   // Add the incompressible neo-Hookean integrator
   Hform->AddDomainIntegrator(new IncompressibleNeoHookeanIntegrator(mu));

   // Set the essential boundary conditions
   Hform->SetEssentialBC(ess_bdr, rhs);

   // Compute the pressure mass stiffness matrix
   ParBilinearForm *a = new ParBilinearForm(spaces[1]);
   ConstantCoefficient one(1.0);
   OperatorHandle mass(Operator::Hypre_ParCSR);
   a->AddDomainIntegrator(new MassIntegrator(one));
   a->Assemble();
   a->Finalize();
   a->ParallelAssemble(mass);
   delete a;

   mass.SetOperatorOwner(false);
   pressure_mass = mass.Ptr();

   // Initialize the Jacobian preconditioner
   JacobianPreconditioner *jac_prec =
      new JacobianPreconditioner(fes, *pressure_mass, block_trueOffsets);
   j_prec = jac_prec;

   // Set up the Jacobian solver
   GMRESSolver *j_gmres = new GMRESSolver(spaces[0]->GetComm());
   j_gmres->iterative_mode = false;
   j_gmres->SetRelTol(1e-12);
   j_gmres->SetAbsTol(1e-12);
   j_gmres->SetMaxIter(300);
   j_gmres->SetPrintLevel(-1);
   j_gmres->SetMonitor(j_monitor);
   j_gmres->SetPreconditioner(*j_prec);
   j_solver = j_gmres;

   // Set the newton solve parameters
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*j_solver);
   newton_solver.SetOperator(*this);
   newton_solver.SetPrintLevel(-1);
   newton_solver.SetMonitor(newton_monitor);
   newton_solver.SetRelTol(rel_tol);
   newton_solver.SetAbsTol(abs_tol);
   newton_solver.SetMaxIter(iter);
}

// Solve the Newton system
void RubberOperator::Solve(Vector &xp) const
{
   Vector zero;
   newton_solver.Mult(zero, xp);
   MFEM_VERIFY(newton_solver.GetConverged(),
               "Newton Solver did not converge.");
}

// compute: y = H(x,p)
void RubberOperator::Mult(const Vector &k, Vector &y) const
{
   Hform->Mult(k, y);
}

// Compute the Jacobian from the nonlinear form
Operator &RubberOperator::GetGradient(const Vector &xp) const
{
   return Hform->GetGradient(xp);
}

RubberOperator::~RubberOperator()
{
   delete Hform;
   delete pressure_mass;
   delete j_solver;
   delete j_prec;
}


// Inline visualization
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
         out << "keys jlA\n"; // turn off perspective and light, +anti-aliasing
      }
      out << "keys cmA\n";        // show colorbar and mesh, +anti-aliasing
      out << "autoscale value\n"; // update value-range; keep mesh-extents fixed
   }
   out << flush;
}

void ReferenceConfiguration(const Vector &x, Vector &y)
{
   // Set the reference, stress free, configuration
   y = x;
}

void InitialDeformation(const Vector &x, Vector &y)
{
   // Set the initial configuration. Having this different from the reference
   // configuration can help convergence
   y = x;
   y[1] = x[1] + 0.25*x[0];
}
