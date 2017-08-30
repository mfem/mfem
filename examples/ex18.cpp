//                                MFEM Example 18
//
// Compile with: make ex18
//
// Sample runs:
//
//       ex18 -p 1 -o 1 -r 2 -tf 2 -c 0.2 -s 1
//       ex18 -p 1 -o 3 -r 1 -tf 2 -c 0.3 -s 3
//       ex18 -p 1 -o 5 -r 1 -tf 2 -c 0.2 -s 4
//       ex18 -p 1 -o 5 -r 0 -tf 2 -c 0.2 -s 6
//       ex18 -p 2 -o 1 -r 2 -tf 2 -c 0.3 -s 3
//       ex18 -p 2 -o 3 -r 2 -tf 2 -c 0.4 -s 4
//       ex18 -p 2 -o 5 -r 1 -tf 2 -c 0.3 -s 4
//       ex18 -p 2 -o 5 -r 1 -tf 2 -c 0.4 -s 6
//
// Description: This example code solves the compressible Euler system
//              of equations, a model nonlinear hyperbolic PDE, with a
//              discontinuous Galerkin (DG) formulation.
//
//              Specifically, it solves for an exact solution of the
//              equations whereby a vortex is transported by a uniform
//              flow. Since all boundaries are periodic here, the
//              method's accuracy can be assessed by measuring the
//              difference between the solution and the initial
//              condition at a later time when the vortex returns to
//              its initial location.
//
//              Note that as the order of the spatial discretization
//              increases, the timestep must become smaller. This
//              example currently uses a simple estimate derived by
//              Cockburn and Shu for the 1D RKDG method. An additional
//              factor is given by passing the --cfl or -c flag.
//
//              Since the solution is a vector grid function,
//              components need to be visualized separately in GLvis
//              using the -gc flag to select the component.
//

#include "mfem.hpp"
#include <fstream>
#include <string>
#include <iostream>

using namespace std;
using namespace mfem;

// Choice for the problem setup. See the u0_function for details.
int problem;

// Equation constant parameters.
const int num_equations = 4;
const double specific_heat_ratio = 1.4;
const double gas_constant = 1.0;

// Maximum char speed (updated by integrators)
double max_char_speed;

// Initial condition
void u0_function(const Vector &x, Vector &u0);

// Time-dependent operator for the right-hand side of the ODE
// representing the DG weak form.
class FE_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M;
   Operator &A;
   DSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(SparseMatrix &M_, Operator &A_);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};

// Implements a simple Rusanov flux
class RiemannSolver
{
private:
   Vector flux1;
   Vector flux2;

public:
   RiemannSolver();
   double Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux);
};

// Element term
class DomainIntegrator : public NonlinearFormIntegrator
{
private:
   Vector shape;
   Vector funval;
   DenseMatrix flux;
   DenseMatrix dshapedr;
   DenseMatrix dshapedx;
   DenseMatrix elfun_mat;
   DenseMatrix elvect_mat;

public:
   DomainIntegrator(const int dim);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);
};

// Interior face term
class FaceIntegrator : public NonlinearFormIntegrator
{
private:
   RiemannSolver rsolver;
   Vector shape1;
   Vector shape2;
   Vector funval1;
   Vector funval2;
   Vector nor;
   Vector fluxN;
   DenseMatrix elfun1_mat;
   DenseMatrix elfun2_mat;
   DenseMatrix elvect1_mat;
   DenseMatrix elvect2_mat;
   IntegrationPoint eip1;
   IntegrationPoint eip2;

public:
   FaceIntegrator(RiemannSolver &rsolver_, const int dim);

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 1;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ref_levels = 1;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 2;
   double dt = 0.01;
   double cfl = 0.3;
   bool visualization = true;
   int vis_steps = 200;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number multiplier (negative means use constant dt specified).");
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

   // 2. Read the mesh from the given mesh file. This example requires
   // a periodic mesh to function correctly.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   MFEM_ASSERT(dim == 2, "Need a two-dimensional mesh for the problem definition");

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
   case 1: ode_solver = new ForwardEulerSolver; break;
   case 2: ode_solver = new RK2Solver(1.0); break;
   case 3: ode_solver = new RK3SSPSolver; break;
   case 4: ode_solver = new RK4Solver; break;
   case 6: ode_solver = new RK6Solver; break;
   default:
      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   // NOTE: The Euler functions below assume byVDIM ordering
   FiniteElementSpace fes(mesh, &fec, num_equations, Ordering::byVDIM);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up the nonlinear form corresponding to the DG
   //    discretization of the flux divergence, and assemble the
   //    corresponding mass matrix.

   // TODO: Wait for the pull request by Neumueller to be merged then
   //       switch the VectorMassIntegrator constructor

   BilinearForm M(&fes);
   Vector cc(num_equations);
   for (int i = 0; i < cc.Size(); i++) cc(i) = 1.0;
   VectorConstantCoefficient vc(cc);
   M.AddDomainIntegrator(new VectorMassIntegrator(vc));
   M.Assemble();
   M.Finalize();

   NonlinearForm A(&fes);
   A.AddDomainIntegrator(new DomainIntegrator(dim));

   RiemannSolver rsolver;
   A.AddInteriorFaceIntegrator(new FaceIntegrator(rsolver, dim));

   // 7. Define the initial conditions, save the corresponding mesh
   //    and grid functions to a file. Note again that the file can be
   //    opened with GLvis with the -gc option.
   VectorFunctionCoefficient u0(num_equations, u0_function);
   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   {
      ofstream omesh("vortex.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);

      ofstream osol("vortex-init.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution euler(M.SpMat(), A);

   // Determine the minimum element size

   // TODO: Need the radius of the smallest circle that encompasses
   // the element, which this method does not quite provide.
   double hmin;
   if (cfl > 0)
   {
      hmin = mesh->GetElementSize(0, 1);
      for (int i = 1; i < mesh->GetNE(); i++)
      {
         hmin = min(mesh->GetElementSize(i, 1), hmin);
      }
   }

   double t = 0.0;
   euler.SetTime(t);
   ode_solver->Init(euler);

   if (cfl > 0)
   {
      // Find a safe dt, using a temporary vector.
      Vector z(u.Size());
      max_char_speed = 0.;
      A.Mult(u, z);
      dt = cfl * hmin / max_char_speed / (2*order+1);
   }

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(u, t, dt_real);
      if (cfl > 0)
      {
         dt = cfl * hmin / max_char_speed / (2*order+1);
      }
      ti++;

      done = (t >= t_final - 1e-8*dt);
      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         // Write out the grid function, since GLvis cannot yet
         // visualize vector grid functions over the socket.
         ofstream osol(string("vortex-") + to_string(ti) + string(".gf"));
         osol.precision(precision);
         u.Save(osol);
      }
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m vortex.mesh -g vortex-final.gf -gc 1".
   {
      ofstream osol("vortex-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // 10. Compute the solution error.
   VectorFunctionCoefficient coeff(num_equations, u0_function);
   const double error = u.ComputeLpError(2, coeff);
   cout << "Solution error: " << error << endl;

   // Free the used memory.
   delete ode_solver;

   return 0;
}

// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(SparseMatrix &M_, Operator &A_)
   : TimeDependentOperator(M_.Size()), M(M_), A(A_), z(M_.Size())
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // Reset wavespeed calculation before step
   max_char_speed = 0.;
   A.Mult(x, z);
   M_solver.Mult(z, y);
}

// Initial condition
void u0_function(const Vector &x, Vector &y)
{
   const int dim = x.Size();
   MFEM_ASSERT(dim == 2, "");

   double radius = 0, Minf = 0, beta = 0;
   if (problem == 1)
   {
      // "Fast vortex"
      radius = 0.2;
      Minf = 0.5;
      beta = 1. / 5.;
   }
   else if (problem == 2)
   {
      // "Slow vortex"
      radius = 0.2;
      Minf = 0.05;
      beta = 1. / 50.;
   }
   else
   {
      mfem_error("Cannot recognize problem."
                 "Options are: 1 - slow vortex, 2 - fast vortex");
   }

   const double xc = 0.0, yc = 0.0;

   // Nice units
   const double vel_inf = 1.;
   const double den_inf = 1.;

   // Derive remainder of background state from this and Minf
   const double pres_inf = (den_inf / specific_heat_ratio) * (vel_inf / Minf) * (vel_inf / Minf);
   const double temp_inf = pres_inf / (den_inf * gas_constant);

   double r2rad = 0.0;
   r2rad += (x(0) - xc) * (x(0) - xc);
   r2rad += (x(1) - yc) * (x(1) - yc);
   r2rad /= (radius * radius);

   const double shrinv1 = 1.0 / (specific_heat_ratio - 1.);

   const double velX = vel_inf * (1 - beta * (x(1) - yc) / radius * exp(-0.5 * r2rad));
   const double velY = vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
   const double vel2 = velX * velX + velY * velY;

   const double specific_heat = gas_constant * specific_heat_ratio * shrinv1;
   const double temp = temp_inf - 0.5 * (vel_inf * beta) * (vel_inf * beta) / specific_heat * exp(-r2rad);

   const double den = den_inf * pow(temp/temp_inf, shrinv1);
   const double pres = den * gas_constant * temp;
   const double energy = shrinv1 * pres / den + 0.5 * vel2;

   y(0) = den;
   y(1) = den * velX;
   y(2) = den * velY;
   y(3) = den * energy;
}

inline double ComputePressure(const Vector &state, int dim)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   double den_vel2 = 0;
   for (int d = 0; d < dim; d++) den_vel2 += den_vel(d) * den_vel(d);
   den_vel2 /= den;

   return (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);
}

// Physicality check (at end)
bool StateIsPhysical(const Vector &state, const int dim);

void ComputeFlux(const Vector &state, int dim, DenseMatrix &flux)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   MFEM_ASSERT(StateIsPhysical(state, dim), "");

   const double pres = ComputePressure(state, dim);

   for (int d = 0; d < dim; d++)
   {
      flux(0,d) = den_vel(d);
      for (int i = 0; i < dim; i++)
         flux(1+i,d) = den_vel(i) * den_vel(d) / den;
      flux(1+d,d) += pres;
   }

   const double H = (den_energy + pres) / den;
   for (int d = 0; d < dim; d++)
   {
      flux(1+dim,d) = den_vel(d) * H;
   }
}

void ComputeFluxDotN(const Vector &state, const Vector &nor,
                     Vector &fluxN)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   MFEM_ASSERT(StateIsPhysical(state, dim), "");

   const double pres = ComputePressure(state, dim);

   double den_velN = 0;
   for (int d = 0; d < dim; d++) den_velN += den_vel(d) * nor(d);

   fluxN(0) = den_velN;
   for (int d = 0; d < dim; d++)
   {
      fluxN(1+d) = den_velN * den_vel(d) / den + pres * nor(d);
   }

   const double H = (den_energy + pres) / den;
   fluxN(1 + dim) = den_velN * H;
}

inline double ComputeMaxCharSpeed(const Vector &state, const int dim)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);

   double den_vel2 = 0;
   for (int d = 0; d < dim; d++) den_vel2 += den_vel(d) * den_vel(d);
   den_vel2 /= den;

   const double pres = ComputePressure(state, dim);
   const double sound = sqrt(specific_heat_ratio * pres / den);
   const double vel = sqrt(den_vel2 / den);

   return vel + sound;
}

RiemannSolver::RiemannSolver() :
   flux1(num_equations),
   flux2(num_equations) { }

double RiemannSolver::Eval(const Vector &state1, const Vector &state2,
                           const Vector &nor, Vector &flux)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();

   MFEM_ASSERT(StateIsPhysical(state1, dim), "");
   MFEM_ASSERT(StateIsPhysical(state2, dim), "");

   const double maxE1 = ComputeMaxCharSpeed(state1, dim);
   const double maxE2 = ComputeMaxCharSpeed(state2, dim);

   const double maxE = max(maxE1, maxE2);

   ComputeFluxDotN(state1, nor, flux1);
   ComputeFluxDotN(state2, nor, flux2);

   double normag = 0;
   for (int i = 0; i < dim; i++)
   {
      normag += nor(i) * nor(i);
   }
   normag = sqrt(normag);

   for (int i = 0; i < num_equations; i++)
   {
      flux(i) = 0.5 * (flux1(i) + flux2(i))
         - 0.5 * maxE * (state2(i) - state1(i)) * normag;
   }

   return maxE;
}

DomainIntegrator::DomainIntegrator(const int dim) :
   funval(num_equations),
   flux(num_equations, dim) { }

void DomainIntegrator::AssembleElementVector(const FiniteElement &el,
                                             ElementTransformation &Tr,
                                             const Vector &elfun, Vector &elvect)
{
   const int dof = el.GetDof();
   const int dim = el.GetDim();

   shape.SetSize(dof);
   dshapedr.SetSize(dof, dim);
   dshapedx.SetSize(dof, dim);

   elvect.SetSize(dof * num_equations);
   elvect = 0.0;

   elfun_mat.UseExternalData(elfun.GetData(), dof, num_equations);
   elvect_mat.UseExternalData(elvect.GetData(), dof, num_equations);

   const int intorder = 2 * el.GetOrder() + 1;
   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Interpolate elfun at the point
      el.CalcShape(ip, shape);
      elfun_mat.MultTranspose(shape, funval);

      // Compute the physical gradients of the test functions
      Tr.SetIntPoint(&ip);
      el.CalcDShape(ip, dshapedr);
      Mult(dshapedr, Tr.AdjugateJacobian(), dshapedx);

      // Update max char speed
      const double mcs = ComputeMaxCharSpeed(funval, dim);
      if (mcs > max_char_speed) max_char_speed = mcs;

      // Compute the flux
      ComputeFlux(funval, dim, flux);
      flux *= ip.weight;

      // Multiply and add to output
      AddMultABt(dshapedx, flux, elvect_mat);
   }
}

FaceIntegrator::FaceIntegrator(RiemannSolver &rsolver_, const int dim) :
   rsolver(rsolver_),
   funval1(num_equations),
   funval2(num_equations),
   nor(dim),
   fluxN(num_equations) { }

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect)
{
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

   shape1.SetSize(dof1);
   shape2.SetSize(dof2);

   elvect.SetSize((dof1 + dof2) * num_equations);
   elvect = 0.0;

   elfun1_mat.UseExternalData(elfun.GetData(), dof1, num_equations);
   elfun2_mat.UseExternalData(elfun.GetData() + dof1 * num_equations, dof2,
                              num_equations);

   elvect1_mat.UseExternalData(elvect.GetData(), dof1, num_equations);
   elvect2_mat.UseExternalData(elvect.GetData() + dof1 * num_equations, dof2,
                              num_equations);

   const int order = std::max(el1.GetOrder(), el2.GetOrder());
   const int intorder = 2 * order + 2;
   const IntegrationRule *ir = &IntRules.Get(Tr.FaceGeom, intorder);
   IntegrationPoint eip1, eip2;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.Loc1.Transform(ip, eip1);
      Tr.Loc2.Transform(ip, eip2);

      // Calculate basis functions on both elements at the face
      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, funval1);
      elfun2_mat.MultTranspose(shape2, funval2);

      Tr.Face->SetIntPoint(&ip);

      // Get the normal vector and the flux on the face
      CalcOrtho(Tr.Face->Jacobian(), nor);
      const double mcs = rsolver.Eval(funval1, funval2, nor, fluxN);

      // Update max char speed
      if (mcs > max_char_speed) max_char_speed = mcs;

      fluxN *= ip.weight;
      for (int k = 0; k < num_equations; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) -= fluxN(k) * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) += fluxN(k) * shape2(s);
         }
      }
   }
}

// Check that the state is physical - enabled in debug mode
bool StateIsPhysical(const Vector &state, const int dim)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   if (den < 0)
   {
      cout << "Negative density: ";
      for (int i = 0; i < state.Size(); i++)
         cout << state(i) << " ";
      cout << endl;
      return false;
   }
   if (den_energy <= 0)
   {
      cout << "Negative energy: ";
      for (int i = 0; i < state.Size(); i++)
         cout << state(i) << " ";
      cout << endl;
      return false;
   }

   double den_vel2 = 0;
   for (int i = 0; i < dim; i++) den_vel2 += den_vel(i) * den_vel(i);
   den_vel2 /= den;

   const double pres = (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);

   if (pres <= 0)
   {
      cout << "Negative pressure: " << pres << ", state: ";
      for (int i = 0; i < state.Size(); i++)
         cout << state(i) << " ";
      cout << endl;
      return false;
   }
   return true;
}
