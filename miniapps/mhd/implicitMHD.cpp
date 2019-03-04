//                                MFEM modified from Example 10
//
// Compile with: make implicitMHD
//
// Sample runs:
//    implicitMHD -m ../../data/beam-quad.mesh -s 3 -r 2 -o 2 -dt 3
//
// Description:  it solves a time dependent reduced resistive MHD problem 
//  10/30/2018 -QT

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;


double alpha; //a global value of magnetude for the pertubation
double Lx;  //size of x domain

/** After spatial discretization, the reduced MHD model can be written as a
 *  system of ODEs:
 *     dPsi/dt = M^{-1}*F1,
 *     dw  /dt = M^{-1}*F2,
 *  coupled with two linear systems
 *     j   = M^{-1}*K*Psi 
 *     Phi = K^{-1}*M*w
 *
 *  Class ImplicitMHDOperator represents the right-hand side of the above
 *  system of ODEs. */
class ImplicitMHDOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;

   BilinearForm M, K;
   NonlinearForm Nv, Nb;
   double viscosity, resistivity;
   ImplicitMHDModel *model;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver K_solver; // Krylov solver for inverting the stiffness matrix K
   DSmoother K_prec;  // Preconditioner for the stiffness matrix K

   //mutable Vector z; // auxiliary vector  not needed -QT

public:
   ReducedMHDOperator(FiniteElementSpace &f, Array<int> &ess_bdr,
                        double visc, double resi, double K);

   // Compute the right-hand side of the ODE system.
   void Mult(const Vector &vx, Vector &dvx_dt) const;

   virtual ~ImplicitMHDOperator();
};


void InitialDeformation(const Vector &x, Vector &y);

void InitialCurrent(const Vector &x, Vector &v);

void visualize(ostream &out, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name = NULL,
               bool init_vis = false);


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";  //a square mesh
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 1;
   double t_final = 1.0;
   double dt = 0.0001;
   double visc = 0.0;
   double resi = 0.0;
   alpha = 0.001; 

   bool visualization = true;
   int vis_steps = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  " only FE supported 13 - RK3 SSP, 14 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visc, "-visc", "--viscosity",
                  "Viscosity coefficient.");
   args.AddOption(&resi, "-resi", "--resistivity",
                  "Resistivity coefficient.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
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

   // 3. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
     // Explicit methods
     case 1: ode_solver = new ForwardEulerSolver; break;
     //case 2: ode_solver = new PD1Solver; break; //first order predictor-corrector
     default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the vector finite element spaces representing 
   //  [Psi, Phi, w, j]
   // in block vector bv, with offsets given by the
   //    fe_offset array.
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll); 

   int fe_size = fespace.GetTrueVSize();
   cout << "Number of scalar unknowns: " << fe_size << endl;
   Array<int> fe_offset(5);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;
   fe_offset[3] = 3*fe_size;
   fe_offset[4] = 4*fe_size;

   BlockVector bv(fe_offset);
   GridFunction psi, phi, w, j;
   psi.MakeTRef(&fespace, bv.GetBlock(0), 0);
   phi.MakeTRef(&fespace, bv.GetBlock(1), 0);
     w.MakeTRef(&fespace, bv.GetBlock(2), 0);
     j.MakeTRef(&fespace, bv.GetBlock(3), 0);

   // 6. Set the initial conditions, and the boundary conditions
   phi=0.0;

//   FunctionCoefficient phiInit(InitialPhi);
//   phi.ProjectCoefficient(phiInit);
//   phi.SetTrueVector();

   FunctionCoefficient psiInit(InitialPsi);
   psi.ProjectCoefficient(psiInit);
   psi.SetTrueVector();

   w=0.0;

//   FunctionCoefficient   wInit(InitialW);
//   w.ProjectCoefficient(wInit);
//   w.SetTrueVector();

   FunctionCoefficient jInit(InitialJ);
   j.ProjectCoefficient(jInit);
   j.SetTrueVector();

   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed

   // 7. Initialize the hyperelastic operator, the GLVis visualization and print
   //    the initial energies.
   ReducedMHDOperator oper(fespace, ess_bdr, visc, mu, K);

   socketstream vis_v, vis_w;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_v.open(vishost, visport);
      vis_v.precision(8);
      v.SetFromTrueVector(); x.SetFromTrueVector();
      visualize(vis_v, mesh, &x, &v, "Velocity", true);
      vis_w.open(vishost, visport);
      if (vis_w)
      {
         oper.GetElasticEnergyDensity(x, w);
         vis_w.precision(8);
         visualize(vis_w, mesh, &x, &w, "Elastic energy density", true);
      }
   }

   double ee0 = oper.ElasticEnergy(x.GetTrueVector());
   double ke0 = oper.KineticEnergy(v.GetTrueVector());
   cout << "initial elastic energy (EE) = " << ee0 << endl;
   cout << "initial kinetic energy (KE) = " << ke0 << endl;
   cout << "initial   total energy (TE) = " << (ee0 + ke0) << endl;

   double t = 0.0;
   oper.SetTime(t);
   ode_solver->Init(oper);

   // 8. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(vx, t, dt_real);

      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         double ee = oper.ElasticEnergy(x.GetTrueVector());
         double ke = oper.KineticEnergy(v.GetTrueVector());

         cout << "step " << ti << ", t = " << t << ", EE = " << ee << ", KE = "
              << ke << ", ΔTE = " << (ee+ke)-(ee0+ke0) << endl;

         if (visualization)
         {
            v.SetFromTrueVector(); x.SetFromTrueVector();
            visualize(vis_v, mesh, &x, &v);
            if (vis_w)
            {
               oper.GetElasticEnergyDensity(x, w);
               visualize(vis_w, mesh, &x, &w);
            }
         }
      }
   }

   // 9. Save the displaced mesh, the velocity and elastic energy.
   {
      v.SetFromTrueVector(); x.SetFromTrueVector();
      GridFunction *nodes = &x;
      int owns_nodes = 0;
      mesh->SwapNodes(nodes, owns_nodes);
      ofstream mesh_ofs("deformed.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      mesh->SwapNodes(nodes, owns_nodes);
      ofstream velo_ofs("velocity.sol");
      velo_ofs.precision(8);
      v.Save(velo_ofs);
      ofstream ee_ofs("elastic_energy.sol");
      ee_ofs.precision(8);
      oper.GetElasticEnergyDensity(x, w);
      w.Save(ee_ofs);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete mesh;

   return 0;
}


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
         out << "keys jl\n";  // turn off perspective and light
      }
      out << "keys cm\n";         // show colorbar and mesh
      out << "autoscale value\n"; // update value-range; keep mesh-extents fixed
      out << "pause\n";
   }
   out << flush;
}


ReducedSystemOperator::ReducedSystemOperator(
   BilinearForm *M_, BilinearForm *S_, NonlinearForm *H_)
   : Operator(M_->Height()), M(M_), S(S_), H(H_), Jacobian(NULL),
     dt(0.0), v(NULL), x(NULL), w(height), z(height)
{ }

void ReducedSystemOperator::SetParameters(double dt_, const Vector *v_,
                                          const Vector *x_)
{
   dt = dt_;  v = v_;  x = x_;
}

void ReducedSystemOperator::Mult(const Vector &k, Vector &y) const
{
   // compute: y = H(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
   add(*v, dt, k, w);
   add(*x, dt, w, z);
   H->Mult(z, y);
   M->AddMult(k, y);
   S->AddMult(w, y);
}

Operator &ReducedSystemOperator::GetGradient(const Vector &k) const
{
   delete Jacobian;
   Jacobian = Add(1.0, M->SpMat(), dt, S->SpMat());
   add(*v, dt, k, w);
   add(*x, dt, w, z);
   SparseMatrix *grad_H = dynamic_cast<SparseMatrix *>(&H->GetGradient(z));
   Jacobian->Add(dt*dt, *grad_H);
   return *Jacobian;
}

ReducedSystemOperator::~ReducedSystemOperator()
{
   delete Jacobian;
}


ReducedMHDOperator::ReducedMHDOperator(FiniteElementSpace &f,
                                           Array<int> &ess_bdr, double visc,
                                           double mu, double K)
   : TimeDependentOperator(2*f.GetTrueVSize(), 0.0), fespace(f),
     M(&fespace), S(&fespace), H(&fespace),
     viscosity(visc), z(height/2)
{
   const double rel_tol = 1e-8;
   const int skip_zero_entries = 0;

   const double ref_density = 1.0; // density in the reference configuration
   ConstantCoefficient rho0(ref_density);
   M.AddDomainIntegrator(new VectorMassIntegrator(rho0));
   M.Assemble(skip_zero_entries);
   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   SparseMatrix tmp;
   M.FormSystemMatrix(ess_tdof_list, tmp);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M.SpMat());

   model = new NeoHookeanModel(mu, K);
   H.AddDomainIntegrator(new HyperelasticNLFIntegrator(model));
   H.SetEssentialTrueDofs(ess_tdof_list);

   ConstantCoefficient visc_coeff(nu);
   S.AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
   S.Assemble(skip_zero_entries);
   S.FormSystemMatrix(ess_tdof_list, tmp);

   reduced_oper = new ReducedSystemOperator(&M, &S, &H);

#ifndef MFEM_USE_SUITESPARSE
   J_prec = new DSmoother(1);
   MINRESSolver *J_minres = new MINRESSolver;
   J_minres->SetRelTol(rel_tol);
   J_minres->SetAbsTol(0.0);
   J_minres->SetMaxIter(300);
   J_minres->SetPrintLevel(-1);
   J_minres->SetPreconditioner(*J_prec);
   J_solver = J_minres;
#else
   J_solver = new UMFPackSolver;
   J_prec = NULL;
#endif

   newton_solver.iterative_mode = false;
   newton_solver.SetSolver(*J_solver);
   newton_solver.SetOperator(*reduced_oper);
   newton_solver.SetPrintLevel(1); // print Newton iterations
   newton_solver.SetRelTol(rel_tol);
   newton_solver.SetAbsTol(0.0);
   newton_solver.SetMaxIter(10);
}

void ReducedMHDOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
   // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   H.Mult(x, z);
   if (nu != 0.0)
   {
      S.AddMult(v, z);
   }
   z.Neg(); // z = -z
   M_solver.Mult(z, dv_dt);

   dx_dt = v;
}

void ReducedMHDOperator::ImplicitSolve(const double dt,
                                         const Vector &vx, Vector &dvx_dt)
{
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   // By eliminating kx from the coupled system:
   //    kv = -M^{-1}*[H(x + dt*kx) + S*(v + dt*kv)]
   //    kx = v + dt*kv
   // we reduce it to a nonlinear equation for kv, represented by the
   // reduced_oper. This equation is solved with the newton_solver
   // object (using J_solver and J_prec internally).
   reduced_oper->SetParameters(dt, &v, &x);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   newton_solver.Mult(zero, dv_dt);
   MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");
   add(v, dt, dv_dt, dx_dt);
}

void ReducedMHDOperator::GetElasticEnergyDensity(
   const GridFunction &x, GridFunction &w) const
{
   ElasticEnergyCoefficient w_coeff(*model, x);
   w.ProjectCoefficient(w_coeff);
}

ReducedMHDOperator::~ReducedMHDOperator()
{
   delete J_solver;
   delete J_prec;
   delete reduced_oper;
   delete model;
}


void InitialJ(const Vector &x, double &j)
{
   j =-M_PI*M_PI*(1.0+4.0/Lx/Lx)*alpha*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}

void InitialPsi(const Vector &x, double &psi)
{
   psi = alpha*sin(M_PI*x(1))*cos(2.0*M_PI/Lx*x(0));
}
