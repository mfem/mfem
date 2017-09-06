//                                MFEM Example 18
//
// Compile with: make ex18
//
// Sample runs: TODO
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
   ParFiniteElementSpace &fes;
   FiniteElementSpace &fes_dim;
   Operator &A;
   SparseMatrix &Aflux;
   DenseTensor Me_inv;

   mutable Vector state;
   mutable DenseMatrix f;
   mutable Vector flux;
   mutable Vector z;

   void GetFlux(const Vector &x, Vector &y) const;

public:
   FE_Evolution(ParFiniteElementSpace &_fes, FiniteElementSpace &_fes_dim,
                Operator &_A, SparseMatrix &_Aflux);

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
   // 1. Initialize MPI.
   MPI_Session mpi(argc, argv);

   // 2. Parse command-line options.
   problem = 1;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ser_ref_levels = -1;
   int par_ref_levels = 1;
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
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly before parallel"
                  " partitioning, -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly after parallel"
                  " partitioning.");
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
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (mpi.Root()) { args.PrintOptions(cout); }

   // 2. Read the mesh from the given mesh file. This example requires
   // a periodic mesh to function correctly.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   MFEM_ASSERT(dim == 2, "Need a two-dimensional mesh for the problem definition");

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   if (ser_ref_levels < 0)
   {
      ser_ref_levels = (int)floor(log(1000./mesh.GetNE())/log(2.)/dim);
   }
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Distribute and and refine.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   // 4. Define the ODE solver used for time integration. Several explicit
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

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec, num_equations, Ordering::byNODES);

   if (mpi.Root())
   {
      cout << "Number of unknowns: " << fes.GetVSize() << endl;
   }

   // 6. Set up the nonlinear form corresponding to the DG
   //    discretization of the flux divergence, and assemble the
   //    corresponding mass matrix.

   // NOTE: The mixed bilinear form does not rely on neighbor data, so
   // it does not need parallel communication
   FiniteElementSpace fes_dim(&pmesh, &fec, dim, Ordering::byNODES);
   FiniteElementSpace fes_1(&pmesh, &fec, 1, Ordering::byNODES);
   MixedBilinearForm Aflux(&fes_dim, &fes_1);

   Aflux.AddDomainIntegrator(new DomainIntegrator(dim));
   Aflux.Assemble();

   RiemannSolver rsolver;
   ParNonlinearForm A(&fes);
   A.AddInteriorFaceIntegrator(new FaceIntegrator(rsolver, dim));

   // 7. Define the initial conditions, save the corresponding mesh
   //    and grid functions to a file. Note again that the file can be
   //    opened with GLvis with the -gc option.
   VectorFunctionCoefficient u0(num_equations, u0_function);
   ParGridFunction u(&fes);
   u.ProjectCoefficient(u0);
   HypreParVector &U = *u.GetTrueDofs();

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "vortex." << setfill('0') << setw(6) << mpi.WorldRank();
      sol_name << "vortex-init." << setfill('0') << setw(6) << mpi.WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << pmesh;

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << u;
   }

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution euler(fes, fes_dim, A, Aflux.SpMat());

   // Determine the minimum element size
   double hmin;
   if (cfl > 0)
   {
      double my_hmin = pmesh.GetElementSize(0, 1);
      for (int i = 1; i < pmesh.GetNE(); i++)
      {
         my_hmin = min(pmesh.GetElementSize(i, 1), my_hmin);
      }
      // Reduce to find the global minimum element size
      MPI_Allreduce(&my_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
   }

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
      A.Mult(U, z);
      // Reduce to find the global maximum wave speed
      {
         double all_max_char_speed = max_char_speed;
         MPI_Allreduce(&max_char_speed, &all_max_char_speed,
                       1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         max_char_speed = all_max_char_speed;
      }

      dt = cfl * hmin / max_char_speed / (2*order+1);
   }

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(U, t, dt_real);
      if (cfl > 0)
      {
         // Reduce to find the global maximum wave speed
         {
            double all_max_char_speed = max_char_speed;
            MPI_Allreduce(&max_char_speed, &all_max_char_speed,
                          1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
            max_char_speed = all_max_char_speed;
         }
         dt = cfl * hmin / max_char_speed / (2*order+1);
      }
      ti++;

      done = (t >= t_final - 1e-8*dt);
      if (done || ti % vis_steps == 0)
      {
         if (mpi.Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }

         // TODO: Enable visualization in parallel
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   tic_toc.Stop();
   if (mpi.Root())
   {
      cout << " done, " << tic_toc.RealTime() << "s." << endl;
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m vortex.mesh -g vortex-final.gf -gc 1".
   {
      ostringstream sol_name;
      sol_name << "vortex-final." << setfill('0') << setw(6) << mpi.WorldRank();

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << u;
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
FE_Evolution::FE_Evolution(ParFiniteElementSpace &_fes,
                           FiniteElementSpace &_fes_dim,
                           Operator &_A, SparseMatrix &_Aflux)
   : TimeDependentOperator(_A.Height()),
     fes(_fes),
     fes_dim(_fes_dim),
     A(_A),
     Aflux(_Aflux),
     Me_inv(fes.GetFE(0)->GetDof(), fes.GetFE(0)->GetDof(), fes.GetNE()),
     state(num_equations),
     f(num_equations, fes.GetFE(0)->GetDim()),
     flux(fes.GetVSize() * fes.GetFE(0)->GetDim()),
     z(A.Height())
{
   // Standard local assembly and inversion for energy mass matrices.
   const int dof = fes.GetFE(0)->GetDof();
   DenseMatrix Me(dof);
   DenseMatrixInverse inv(&Me);
   MassIntegrator mi;
   for (int i = 0; i < fes.GetNE(); i++)
   {
      mi.AssembleElementMatrix(*fes.GetFE(i), *fes.GetElementTransformation(i), Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv(i));
   }
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   max_char_speed = 0.;

   // 1. Create the vector z with the face terms -<F.n(u), [w]>.
   // This is the only place communication needs to happen.
   A.Mult(x, z);

   // 2. Add the element terms.
   // i.  computing the flux approximately as a grid function by
   //     interpolating at the solution nodes.
   // ii. multiplying this grid function by a (constant) mixed
   //     bilinear form for each of the num_equations, computing
   //     (F(u), grad(w)) for each equation.

   // NOTE: fes_dim and fes (and by extension the gf flux) must be
   // byNODES for this trick to work
   MFEM_ASSERT(fes_dim.GetOrdering() == Ordering::byNODES, "");
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");
   GetFlux(x, flux);
   for (int k = 0; k < num_equations; k++)
   {
      Vector fk(flux.GetData() + k * fes_dim.GetVSize(), fes_dim.GetVSize());
      Vector zk(z.GetData() + k * fes.GetNDofs(), fes_dim.GetNDofs());
      Aflux.AddMult(fk, zk);
   }

   // 3. Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> vdofs;
   const int dofs = fes.GetFE(0)->GetDof();
   DenseMatrix zmat, ymat(dofs, num_equations);

   for (int i = 0; i < fes.GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      fes.GetElementVDofs(i, vdofs);
      z.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), dofs, num_equations);
      mfem::Mult(Me_inv(i), zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
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

// Compute the flux at solution nodes.
void FE_Evolution::GetFlux(const Vector &x, Vector &flux) const
{
   // NOTE: fes must be byNODES for this logic to work.
   const int dof0 = fes.GetNDofs();
   const int dim  = fes.GetFE(0)->GetDim();

   for (int i = 0; i < dof0; i++)
   {
      for (int k = 0; k < num_equations; k++) state(k) = x(i + k * dof0);

      ComputeFlux(state, dim, f);

      for (int d = 0; d < dim; d++)
      {
         for (int k = 0; k < num_equations; k++)
         {
            flux(i + d * dof0 + k * dof0 * dim) = f(k, d);
         }
      }
   }
}

DomainIntegrator::DomainIntegrator(const int dim) : flux(num_equations, dim) { }

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
   funval1(num_equations),
   funval2(num_equations),
   nor(dim),
   fluxN(num_equations) { }

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
      // cout << mcs << " " << max_char_speed << endl;
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
