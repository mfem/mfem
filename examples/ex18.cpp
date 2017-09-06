//                                MFEM Example 18
//
// Compile with: make ex18
//
// Sample runs: TBD
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
#include <sstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Choice for the problem setup. See the u0_function for details.
int problem;

// Equation constant parameters.
const int num_equation = 4;
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
   const int dim;

   FiniteElementSpace &vfes;
   Operator &A;
   SparseMatrix &Aflux;
   DenseTensor Me_inv;

   mutable Vector state;
   mutable DenseMatrix f;
   mutable DenseTensor flux;
   mutable Vector z;

   void GetFlux(const DenseMatrix &state, DenseTensor &flux) const;

public:
   FE_Evolution(FiniteElementSpace &_vfes, Operator &_A,
                SparseMatrix &_Aflux);

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


// Constant mixed bilinear form multiplying the flux grid function
class DomainIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape;
   DenseMatrix flux;
   DenseMatrix dshapedr;
   DenseMatrix dshapedx;

public:
   DomainIntegrator(const int dim);

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Tr,
                                       DenseMatrix &elmat);
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
   bool visualization = false;
   int vis_steps = 50;

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
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

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
      mesh.UniformRefinement();
   }

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   FiniteElementSpace dfes(&mesh, &fec, dim, Ordering::byNODES);
   FiniteElementSpace vfes(&mesh, &fec, num_equation, Ordering::byNODES);
   cout << "Number of unknowns: " << vfes.GetVSize() << endl;

   // Much of this example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

   // 6. Define the initial conditions, save the corresponding mesh
   //    and grid functions to a file. Note again that the file can be
   //    opened with GLvis with the -gc option.
   Array<int> offsets(num_equation + 1);
   for (int k = 0; k <= num_equation; k++) { offsets[k] = k * vfes.GetNDofs(); }
   BlockVector u(offsets);

   // Density grid function for visualization.
   GridFunction den(&fes, u.GetBlock(0));

   // Initialize the block vector state: define a vector finite
   // element space for all equations and initialize together.
   VectorFunctionCoefficient u0(num_equation, u0_function);
   GridFunction sol(&vfes, u.GetData());
   sol.ProjectCoefficient(u0);

   // Output the initial solution.
   {
      ofstream mesh_ofs("vortex.mesh");
      mesh_ofs.precision(precision);
      mesh_ofs << mesh;

      for (int k = 0; k < num_equation; k++)
      {
         GridFunction uk(&fes, u.GetBlock(k));
         ostringstream sol_name;
         sol_name << "vortex-" << k << "-init.gf";
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 7. Set up the nonlinear form corresponding to the DG
   //    discretization of the flux divergence, and assemble the
   //    corresponding mass matrix.
   MixedBilinearForm Aflux(&dfes, &fes);
   Aflux.AddDomainIntegrator(new DomainIntegrator(dim));
   Aflux.Assemble();

   NonlinearForm A(&vfes);
   RiemannSolver rsolver;
   A.AddInteriorFaceIntegrator(new FaceIntegrator(rsolver, dim));

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution euler(vfes, A, Aflux.SpMat());

   // Visualize the density
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << mesh << den;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // Determine the minimum element size.
   double hmin;
   if (cfl > 0)
   {
      hmin = mesh.GetElementSize(0, 1);
      for (int i = 1; i < mesh.GetNE(); i++)
      {
         hmin = min(mesh.GetElementSize(i, 1), hmin);
      }
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

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

   // Integrate in time.
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
      }

      if (visualization)
      {
         sout << "solution\n" << mesh << den << flush;
      }
   }

   tic_toc.Stop();
   cout << " done, " << tic_toc.RealTime() << "s." << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m vortex.mesh -g vortex-final.gf -gc 1".
   // Output the initial solution
   for (int k = 0; k < num_equation; k++)
   {
      GridFunction uk(&fes, u.GetBlock(k));
      ostringstream sol_name;
      sol_name << "vortex-" << k << "-final.gf";
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << uk;
   }

   // 10. Compute the L2 solution error summed for all components.
   if (t_final == 2.0)
   {
      const double error = sol.ComputeLpError(2, u0);
      cout << "Solution error: " << error << endl;
   }

   // Free the used memory.
   delete ode_solver;

   return 0;
}

// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(FiniteElementSpace &_vfes,
                           Operator &_A, SparseMatrix &_Aflux)
   : TimeDependentOperator(_A.Height()),
     dim(_vfes.GetFE(0)->GetDim()),
     vfes(_vfes),
     A(_A),
     Aflux(_Aflux),
     Me_inv(vfes.GetFE(0)->GetDof(), vfes.GetFE(0)->GetDof(), vfes.GetNE()),
     state(num_equation),
     f(num_equation, dim),
     flux(vfes.GetNDofs(), dim, num_equation),
     z(A.Height())
{
   // Standard local assembly and inversion for energy mass matrices.
   const int dof = vfes.GetFE(0)->GetDof();
   DenseMatrix Me(dof);
   DenseMatrixInverse inv(&Me);
   MassIntegrator mi;
   for (int i = 0; i < vfes.GetNE(); i++)
   {
      mi.AssembleElementMatrix(*vfes.GetFE(i), *vfes.GetElementTransformation(i), Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv(i));
   }
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   max_char_speed = 0.;

   // 1. Create the vector z with the face terms -<F.n(u), [w]>.
   A.Mult(x, z);

   // 2. Add the element terms.
   // i.  computing the flux approximately as a grid function by
   //     interpolating at the solution nodes.
   // ii. multiplying this grid function by a (constant) mixed
   //     bilinear form for each of the num_equation, computing
   //     (F(u), grad(w)) for each equation.

   DenseMatrix xmat(x.GetData(), vfes.GetNDofs(), num_equation);
   GetFlux(xmat, flux);

   for (int k = 0; k < num_equation; k++)
   {
      Vector fk(flux(k).GetData(), dim * vfes.GetNDofs());
      Vector zk(z.GetData() + k * vfes.GetNDofs(), vfes.GetNDofs());
      Aflux.AddMult(fk, zk);
   }

   // 3. Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> vdofs;
   const int dof = vfes.GetFE(0)->GetDof();
   DenseMatrix zmat, ymat(dof, num_equation);

   for (int k = 0; k < num_equation; k++)
   {
      for (int i = 0; i < vfes.GetNE(); i++)
      {
         // Return the vdofs ordered byNODES
         vfes.GetElementVDofs(i, vdofs);
         z.GetSubVector(vdofs, zval);
         zmat.UseExternalData(zval.GetData(), dof, num_equation);
         mfem::Mult(Me_inv(i), zmat, ymat);
         y.SetSubVector(vdofs, ymat.GetData());
      }
   }
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

// Pressure (EOS) computation
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

// Compute the vector flux F(u)
void ComputeFlux(const Vector &state, int dim, DenseMatrix &flux)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   MFEM_ASSERT(StateIsPhysical(state, dim), "");

   const double pres = ComputePressure(state, dim);

   for (int d = 0; d < dim; d++)
   {
      flux(0, d) = den_vel(d);
      for (int i = 0; i < dim; i++)
         flux(1+i, d) = den_vel(i) * den_vel(d) / den;
      flux(1+d, d) += pres;
   }

   const double H = (den_energy + pres) / den;
   for (int d = 0; d < dim; d++)
   {
      flux(1+dim, d) = den_vel(d) * H;
   }
}

// Compute the scalar F(u).n
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

// Compute the maximum characteristic speed.
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
   flux1(num_equation),
   flux2(num_equation) { }

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

   for (int i = 0; i < num_equation; i++)
   {
      flux(i) = 0.5 * (flux1(i) + flux2(i))
         - 0.5 * maxE * (state2(i) - state1(i)) * normag;
   }

   return maxE;
}

// Compute the flux at solution nodes.
void FE_Evolution::GetFlux(const DenseMatrix &x, DenseTensor &flux) const
{
   const int dof = flux.SizeI();
   const int dim = flux.SizeJ();

   for (int i = 0; i < dof; i++)
   {
      for (int k = 0; k < num_equation; k++) state(k) = x(i, k);
      ComputeFlux(state, dim, f);

      for (int d = 0; d < dim; d++)
      {
         for (int k = 0; k < num_equation; k++)
         {
            flux(i, d, k) = f(k, d);
         }
      }
   }
}

DomainIntegrator::DomainIntegrator(const int dim) : flux(num_equation, dim) { }

void DomainIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                              const FiniteElement &test_fe,
                                              ElementTransformation &Tr,
                                              DenseMatrix &elmat)
{
   // Assemble the form (vec(v), grad(w))

   // Trial space = vector L2 space (mesh dim)
   // Test space = scalar L2 space

   const int dof_trial = trial_fe.GetDof();
   const int dof_test = test_fe.GetDof();
   const int dim = trial_fe.GetDim();

   shape.SetSize(dof_trial);
   dshapedr.SetSize(dof_test, dim);
   dshapedx.SetSize(dof_test, dim);

   elmat.SetSize(dof_test, dof_trial * dim);
   elmat = 0.0;

   const int maxorder = max(trial_fe.GetOrder(), test_fe.GetOrder());
   const int intorder = 2 * maxorder;
   const IntegrationRule *ir = &IntRules.Get(trial_fe.GetGeomType(), intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Calculate the shape functions
      trial_fe.CalcShape(ip, shape);
      shape *= ip.weight;

      // Compute the physical gradients of the test functions
      Tr.SetIntPoint(&ip);
      test_fe.CalcDShape(ip, dshapedr);
      Mult(dshapedr, Tr.AdjugateJacobian(), dshapedx);

      for (int d = 0; d < dim; d++)
      {
         for (int j = 0; j < dof_test; j++)
         {
            for (int k = 0; k < dof_trial; k++)
            {
               elmat(j, k + d * dof_trial) += shape(k) * dshapedx(j, d);
            }
         }
      }
   }
}


FaceIntegrator::FaceIntegrator(RiemannSolver &rsolver_, const int dim) :
   rsolver(rsolver_),
   funval1(num_equation),
   funval2(num_equation),
   nor(dim),
   fluxN(num_equation) { }

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect)
{
   // Compute the term <F.n(u),[w]> on the interior faces.
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

   shape1.SetSize(dof1);
   shape2.SetSize(dof2);

   elvect.SetSize((dof1 + dof2) * num_equation);
   elvect = 0.0;

   elfun1_mat.UseExternalData(elfun.GetData(), dof1, num_equation);
   elfun2_mat.UseExternalData(elfun.GetData() + dof1 * num_equation, dof2,
                              num_equation);

   elvect1_mat.UseExternalData(elvect.GetData(), dof1, num_equation);
   elvect2_mat.UseExternalData(elvect.GetData() + dof1 * num_equation, dof2,
                              num_equation);

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
      for (int k = 0; k < num_equation; k++)
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
