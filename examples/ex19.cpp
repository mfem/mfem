//                                MFEM Example 19
//
// Compile with: make ex19
//
// Sample runs:
//    ex19 -m ../data/beam-quad.mesh
//    ex19 -m ../data/beam-tri.mesh
//    ex19 -m ../data/beam-hex.mesh
//    ex19 -m ../data/beam-tet.mesh
//    ex19 -m ../data/beam-wedge.mesh
//    ex19 -m ../data/beam-quad-amr.mesh
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
   GeneralResidualMonitor(const std::string& prefix_, int print_lvl)
      : prefix(prefix_)
   {
      print_level = print_lvl;
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
   Array<FiniteElementSpace *> spaces;

   // Offsets for extracting block vector segments
   Array<int> &block_trueOffsets;

   // Jacobian for block access
   BlockOperator *jacobian;

   // Scaling factor for the pressure mass matrix in the block preconditioner
   double gamma;

   // Objects for the block preconditioner application
   SparseMatrix *pressure_mass;
   Solver *mass_pcg;
   Solver *mass_prec;
   Solver *stiff_pcg;
   Solver *stiff_prec;

public:
   JacobianPreconditioner(Array<FiniteElementSpace *> &fes,
                          SparseMatrix &mass, Array<int> &offsets);

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
   Array<FiniteElementSpace *> spaces;

   // Block nonlinear form
   BlockNonlinearForm *Hform;

   // Pressure mass matrix for the preconditioner
   SparseMatrix *pressure_mass;

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
   RubberOperator(Array<FiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr,
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
void visualize(ostream &out, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name = NULL,
               bool init_vis = false);

// Configuration definition functions
void ReferenceConfiguration(const Vector &x, Vector &y);
void InitialDeformation(const Vector &x, Vector &y);


int main(int argc, char *argv[])
{
   // 1. Parse command-line options
   const char *mesh_file = "../data/beam-tet.mesh";
   int ref_levels = 0;
   int order = 2;
   bool visualization = true;
   double newton_rel_tol = 1e-4;
   double newton_abs_tol = 1e-6;
   int newton_iter = 500;
   double mu = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
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
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 4. Define the shear modulus for the incompressible Neo-Hookean material
   ConstantCoefficient c_mu(mu);

   // 5. Define the finite element spaces for displacement and pressure
   //    (Taylor-Hood elements). By default, the displacement (u/x) is a second
   //    order vector field, while the pressure (p) is a linear scalar function.
   H1_FECollection quad_coll(order, dim);
   H1_FECollection lin_coll(order-1, dim);

   FiniteElementSpace R_space(mesh, &quad_coll, dim, Ordering::byVDIM);
   FiniteElementSpace W_space(mesh, &lin_coll);

   Array<FiniteElementSpace *> spaces(2);
   spaces[0] = &R_space;
   spaces[1] = &W_space;

   int R_size = R_space.GetTrueVSize();
   int W_size = W_space.GetTrueVSize();

   // 6. Define the Dirichlet conditions (set to boundary attribute 1 and 2)
   Array<Array<int> *> ess_bdr(2);

   Array<int> ess_bdr_u(R_space.GetMesh()->bdr_attributes.Max());
   Array<int> ess_bdr_p(W_space.GetMesh()->bdr_attributes.Max());

   ess_bdr_p = 0;
   ess_bdr_u = 0;
   ess_bdr_u[0] = 1;
   ess_bdr_u[1] = 1;

   ess_bdr[0] = &ess_bdr_u;
   ess_bdr[1] = &ess_bdr_p;

   // 7. Print the mesh statistics
   std::cout << "***********************************************************\n";
   std::cout << "dim(u) = " << R_size << "\n";
   std::cout << "dim(p) = " << W_size << "\n";
   std::cout << "dim(u+p) = " << R_size + W_size << "\n";
   std::cout << "***********************************************************\n";

   // 8. Define the block structure of the solution vector (u then p)
   Array<int> block_trueOffsets(3);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = R_space.GetTrueVSize();
   block_trueOffsets[2] = W_space.GetTrueVSize();
   block_trueOffsets.PartialSum();

   BlockVector xp(block_trueOffsets);

   // 9. Define grid functions for the current configuration, reference
   //    configuration, final deformation, and pressure
   GridFunction x_gf(&R_space);
   GridFunction x_ref(&R_space);
   GridFunction x_def(&R_space);
   GridFunction p_gf(&W_space);

   x_gf.MakeTRef(&R_space, xp.GetBlock(0), 0);
   p_gf.MakeTRef(&W_space, xp.GetBlock(1), 0);

   VectorFunctionCoefficient deform(dim, InitialDeformation);
   VectorFunctionCoefficient refconfig(dim, ReferenceConfiguration);

   x_gf.ProjectCoefficient(deform);
   x_ref.ProjectCoefficient(refconfig);
   p_gf = 0.0;

   x_gf.SetTrueVector();
   p_gf.SetTrueVector();

   // 10. Initialize the incompressible neo-Hookean operator
   RubberOperator oper(spaces, ess_bdr, block_trueOffsets,
                       newton_rel_tol, newton_abs_tol, newton_iter, c_mu);

   // 11. Solve the Newton system
   oper.Solve(xp);

   // 12. Compute the final deformation
   x_gf.SetFromTrueVector();
   p_gf.SetFromTrueVector();
   subtract(x_gf, x_ref, x_def);

   // 13. Visualize the results if requested
   socketstream vis_u, vis_p;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_u.open(vishost, visport);
      vis_u.precision(8);
      visualize(vis_u, mesh, &x_gf, &x_def, "Deformation", true);
      vis_p.open(vishost, visport);
      vis_p.precision(8);
      visualize(vis_p, mesh, &x_gf, &p_gf, "Pressure", true);
   }

   // 14. Save the displaced mesh, the final deformation, and the pressure
   {
      GridFunction *nodes = &x_gf;
      int owns_nodes = 0;
      mesh->SwapNodes(nodes, owns_nodes);

      ofstream mesh_ofs("deformed.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream pressure_ofs("pressure.sol");
      pressure_ofs.precision(8);
      p_gf.Save(pressure_ofs);

      ofstream deformation_ofs("deformation.sol");
      deformation_ofs.precision(8);
      x_def.Save(deformation_ofs);
   }

   // 15. Free the used memory
   delete mesh;

   return 0;
}


JacobianPreconditioner::JacobianPreconditioner(Array<FiniteElementSpace *> &fes,
                                               SparseMatrix &mass,
                                               Array<int> &offsets)
   : Solver(offsets[2]), block_trueOffsets(offsets), pressure_mass(&mass)
{
   fes.Copy(spaces);

   gamma = 0.00001;

   // The mass matrix and preconditioner do not change every Newton cycle, so we
   // only need to define them once
   GSSmoother *mass_prec_gs = new GSSmoother(*pressure_mass);

   mass_prec = mass_prec_gs;

   CGSolver *mass_pcg_iter = new CGSolver();
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
      GSSmoother *stiff_prec_gs = new GSSmoother();

      stiff_prec = stiff_prec_gs;

      GMRESSolver *stiff_pcg_iter = new GMRESSolver();
      stiff_pcg_iter->SetRelTol(1e-8);
      stiff_pcg_iter->SetAbsTol(1e-8);
      stiff_pcg_iter->SetMaxIter(200);
      stiff_pcg_iter->SetPrintLevel(0);
      stiff_pcg_iter->SetPreconditioner(*stiff_prec);
      stiff_pcg_iter->iterative_mode = false;

      stiff_pcg = stiff_pcg_iter;
   }

   // At each Newton cycle, compute the new stiffness preconditioner by updating
   // the iterative solver which, in turn, updates its preconditioner
   stiff_pcg->SetOperator(jacobian->GetBlock(0,0));
}

JacobianPreconditioner::~JacobianPreconditioner()
{
   delete mass_pcg;
   delete mass_prec;
   delete stiff_prec;
   delete stiff_pcg;
}


RubberOperator::RubberOperator(Array<FiniteElementSpace *> &fes,
                               Array<Array<int> *> &ess_bdr,
                               Array<int> &offsets,
                               double rel_tol,
                               double abs_tol,
                               int iter,
                               Coefficient &c_mu)
   : Operator(fes[0]->GetTrueVSize() + fes[1]->GetTrueVSize()),
     newton_solver(), newton_monitor("Newton", 1),
     j_monitor("  GMRES", 3), mu(c_mu), block_trueOffsets(offsets)
{
   Array<Vector *> rhs(2);
   rhs = NULL; // Set all entries in the array

   fes.Copy(spaces);

   // Define the block nonlinear form
   Hform = new BlockNonlinearForm(spaces);

   // Add the incompressible neo-Hookean integrator
   Hform->AddDomainIntegrator(new IncompressibleNeoHookeanIntegrator(mu));

   // Set the essential boundary conditions
   Hform->SetEssentialBC(ess_bdr, rhs);

   // Compute the pressure mass stiffness matrix
   BilinearForm *a = new BilinearForm(spaces[1]);
   ConstantCoefficient one(1.0);
   a->AddDomainIntegrator(new MassIntegrator(one));
   a->Assemble();
   a->Finalize();

   OperatorPtr op;
   Array<int> p_ess_tdofs;
   a->FormSystemMatrix(p_ess_tdofs, op);
   pressure_mass = a->LoseMat();
   delete a;

   // Initialize the Jacobian preconditioner
   JacobianPreconditioner *jac_prec =
      new JacobianPreconditioner(fes, *pressure_mass, block_trueOffsets);
   j_prec = jac_prec;

   // Set up the Jacobian solver
   GMRESSolver *j_gmres = new GMRESSolver();
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
void visualize(ostream &out, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name, bool init_vis)
{
   if (!out)
   {
      return;
   }

   GridFunction *nodes = deformed_nodes;
   int owns_nodes = 0;

   mesh->SwapNodes(nodes, owns_nodes);

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
