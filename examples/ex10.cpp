//                                MFEM Example 10
//
// Compile with: make ex10
//
// Sample runs:
//    ex10 -m ../data/beam-quad.mesh -s 3 -r 2 -o 2 -dt 3
//    ex10 -m ../data/beam-tri.mesh -s 3 -r 2 -o 2 -dt 3
//    ex10 -m ../data/beam-hex.mesh -s 2 -r 1 -o 2 -dt 3
//    ex10 -m ../data/beam-tet.mesh -s 2 -r 1 -o 2 -dt 3
//    ex10 -m ../data/beam-wedge.mesh -s 2 -r 1 -o 2 -dt 3
//    ex10 -m ../data/beam-quad.mesh -s 14 -r 2 -o 2 -dt 0.03 -vs 20
//    ex10 -m ../data/beam-hex.mesh -s 14 -r 1 -o 2 -dt 0.05 -vs 20
//    ex10 -m ../data/beam-quad-amr.mesh -s 3 -r 2 -o 2 -dt 3
//
// Description:  This examples solves a time dependent nonlinear elasticity
//               problem of the form dv/dt = H(x) + S v, dx/dt = v, where H is a
//               hyperelastic model and S is a viscosity operator of Laplacian
//               type. The geometry of the domain is assumed to be as follows:
//
//                                 +---------------------+
//                    boundary --->|                     |
//                    attribute 1  |                     |
//                    (fixed)      +---------------------+
//
//               The example demonstrates the use of nonlinear operators (the
//               class HyperelasticOperator defining H(x)), as well as their
//               implicit time integration using a Newton method for solving an
//               associated reduced backward-Euler type nonlinear equation
//               (class ReducedSystemOperator). Each Newton step requires the
//               inversion of a Jacobian matrix, which is done through a
//               (preconditioned) inner solver. Note that implementing the
//               method HyperelasticOperator::ImplicitSolve is the only
//               requirement for high-order implicit (SDIRK) time integration.
//
//               We recommend viewing examples 2 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

class ReducedSystemOperator;

/** After spatial discretization, the hyperelastic model can be written as a
 *  system of ODEs:
 *     dv/dt = -M^{-1}*(H(x) + S*v)
 *     dx/dt = v,
 *  where x is the vector representing the deformation, v is the velocity field,
 *  M is the mass matrix, S is the viscosity matrix, and H(x) is the nonlinear
 *  hyperelastic operator.
 *
 *  Class HyperelasticOperator represents the right-hand side of the above
 *  system of ODEs. */
class HyperelasticOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;

   BilinearForm M, S;
   NonlinearForm H;
   double viscosity;
   HyperelasticModel *model;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   /** Nonlinear operator defining the reduced backward Euler equation for the
       velocity. Used in the implementation of method ImplicitSolve. */
   ReducedSystemOperator *reduced_oper;

   /// Newton solver for the reduced backward Euler equation
   NewtonSolver newton_solver;

   /// Solver for the Jacobian solve in the Newton method
   Solver *J_solver;
   /// Preconditioner for the Jacobian solve in the Newton method
   Solver *J_prec;

   mutable Vector z; // auxiliary vector

public:
   HyperelasticOperator(FiniteElementSpace &f, Array<int> &ess_bdr,
                        double visc, double mu, double K);

   /// Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;
   /** Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   double ElasticEnergy(const Vector &x) const;
   double KineticEnergy(const Vector &v) const;
   void GetElasticEnergyDensity(const GridFunction &x, GridFunction &w) const;

   virtual ~HyperelasticOperator();
};

/** Nonlinear operator of the form:
    k --> (M + dt*S)*k + H(x + dt*v + dt^2*k) + S*v,
    where M and S are given BilinearForms, H is a given NonlinearForm, v and x
    are given vectors, and dt is a scalar. */
class ReducedSystemOperator : public Operator
{
private:
   BilinearForm *M, *S;
   NonlinearForm *H;
   mutable SparseMatrix *Jacobian;
   double dt;
   const Vector *v, *x;
   mutable Vector w, z;

public:
   ReducedSystemOperator(BilinearForm *M_, BilinearForm *S_, NonlinearForm *H_);

   /// Set current dt, v, x values - needed to compute action and Jacobian.
   void SetParameters(double dt_, const Vector *v_, const Vector *x_);

   /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &k) const;

   virtual ~ReducedSystemOperator();
};


/** Function representing the elastic energy density for the given hyperelastic
    model+deformation. Used in HyperelasticOperator::GetElasticEnergyDensity. */
class ElasticEnergyCoefficient : public Coefficient
{
private:
   HyperelasticModel  &model;
   const GridFunction &x;
   DenseMatrix         J;

public:
   ElasticEnergyCoefficient(HyperelasticModel &m, const GridFunction &x_)
      : model(m), x(x_) { }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   virtual ~ElasticEnergyCoefficient() { }
};

void InitialDeformation(const Vector &x, Vector &y);

void InitialVelocity(const Vector &x, Vector &v);

void visualize(ostream &out, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name = NULL,
               bool init_vis = false);


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/beam-quad.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 3;
   double t_final = 300.0;
   double dt = 3.0;
   double visc = 1e-2;
   double mu = 0.25;
   double K = 5.0;
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
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "            11 - Forward Euler, 12 - RK2,\n\t"
                  "            13 - RK3 SSP, 14 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visc, "-v", "--viscosity",
                  "Viscosity coefficient.");
   args.AddOption(&mu, "-mu", "--shear-modulus",
                  "Shear modulus in the Neo-Hookean hyperelastic model.");
   args.AddOption(&K, "-K", "--bulk-modulus",
                  "Bulk modulus in the Neo-Hookean hyperelastic model.");
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
      // Implicit L-stable methods
      case 1: ode_solver = new BackwardEulerSolver; break;
      case 2: ode_solver = new SDIRK23Solver(2); break;
      case 3: ode_solver = new SDIRK33Solver; break;
      // Explicit methods
      case 11: ode_solver = new ForwardEulerSolver; break;
      case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 13: ode_solver = new RK3SSPSolver; break;
      case 14: ode_solver = new RK4Solver; break;
      case 15: ode_solver = new GeneralizedAlphaSolver(0.5); break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
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

   // 5. Define the vector finite element spaces representing the mesh
   //    deformation x, the velocity v, and the initial configuration, x_ref.
   //    Define also the elastic energy density, w, which is in a discontinuous
   //    higher-order space. Since x and v are integrated in time as a system,
   //    we group them together in block vector vx, with offsets given by the
   //    fe_offset array.
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll, dim);

   int fe_size = fespace.GetTrueVSize();
   cout << "Number of velocity/deformation unknowns: " << fe_size << endl;
   Array<int> fe_offset(3);
   fe_offset[0] = 0;
   fe_offset[1] = fe_size;
   fe_offset[2] = 2*fe_size;

   BlockVector vx(fe_offset);
   GridFunction v, x;
   v.MakeTRef(&fespace, vx.GetBlock(0), 0);
   x.MakeTRef(&fespace, vx.GetBlock(1), 0);

   GridFunction x_ref(&fespace);
   mesh->GetNodes(x_ref);

   L2_FECollection w_fec(order + 1, dim);
   FiniteElementSpace w_fespace(mesh, &w_fec);
   GridFunction w(&w_fespace);

   // 6. Set the initial conditions for v and x, and the boundary conditions on
   //    a beam-like mesh (see description above).
   VectorFunctionCoefficient velo(dim, InitialVelocity);
   v.ProjectCoefficient(velo);
   v.SetTrueVector();
   VectorFunctionCoefficient deform(dim, InitialDeformation);
   x.ProjectCoefficient(deform);
   x.SetTrueVector();

   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed

   // 7. Initialize the hyperelastic operator, the GLVis visualization and print
   //    the initial energies.
   HyperelasticOperator oper(fespace, ess_bdr, visc, mu, K);

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


HyperelasticOperator::HyperelasticOperator(FiniteElementSpace &f,
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

   ConstantCoefficient visc_coeff(viscosity);
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

void HyperelasticOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
   // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   H.Mult(x, z);
   if (viscosity != 0.0)
   {
      S.AddMult(v, z);
   }
   z.Neg(); // z = -z
   M_solver.Mult(z, dv_dt);

   dx_dt = v;
}

void HyperelasticOperator::ImplicitSolve(const double dt,
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

double HyperelasticOperator::ElasticEnergy(const Vector &x) const
{
   return H.GetEnergy(x);
}

double HyperelasticOperator::KineticEnergy(const Vector &v) const
{
   return 0.5*M.InnerProduct(v, v);
}

void HyperelasticOperator::GetElasticEnergyDensity(
   const GridFunction &x, GridFunction &w) const
{
   ElasticEnergyCoefficient w_coeff(*model, x);
   w.ProjectCoefficient(w_coeff);
}

HyperelasticOperator::~HyperelasticOperator()
{
   delete J_solver;
   delete J_prec;
   delete reduced_oper;
   delete model;
}


double ElasticEnergyCoefficient::Eval(ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   model.SetTransformation(T);
   x.GetVectorGradient(T, J);
   // return model.EvalW(J);  // in reference configuration
   return model.EvalW(J)/J.Det(); // in deformed configuration
}


void InitialDeformation(const Vector &x, Vector &y)
{
   // set the initial configuration to be the same as the reference, stress
   // free, configuration
   y = x;
}

void InitialVelocity(const Vector &x, Vector &v)
{
   const int dim = x.Size();
   const double s = 0.1/64.;

   v = 0.0;
   v(dim-1) = s*x(0)*x(0)*(8.0-x(0));
   v(0) = -s*x(0)*x(0);
}
