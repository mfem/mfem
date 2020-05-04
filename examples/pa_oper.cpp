//                                MFEM Example 9
//
// Compile with: make serial_nogpu
//
// Description:  This code solves the time-dependent advection-diffusion
//               equation:
//               \frac(\partial u}{\partial t}
//                                 = \mathbf{a} \cdot \Nabla u - \nu \Nabla^2 u
//               where a is a given advection velocity, \nu is the diffusion
//               parameter, and u0(x) = u(0,x) is a given initial condition.
//
//               The demonstrates explicit time marching with H1 elements of
//               arbitrary order. Periodic boundary conditions are used through
//               periodic meshes. GLVis can be used for visualization of a
//               time-evolving solution.

#include <fstream>
#include <iostream>
#include <algorithm>

#include "mfem.hpp"
#include "mpi.h"

using namespace std;
using namespace mfem;

/** A time-dependent operator for the right-hand side of the ODE. The weak
    form of du/dt = -a.grad(u) + nu Delta(u) is M du/dt = K u + b, where M and
    K are the mass and advection-diffusion matrices, and b describes the flow
    on the boundary. This can be written as a general ODE,
    du/dt = M^{-1} (K u + b), and this class is used to evaluate the right-hand
    side. */
class AdvectionDiffusionEvolution : public mfem::TimeDependentOperator
{
public:
   /// \param[in] M - bilinear form for mass matrix
   /// \param[in] K - bilinear form for stiffness matrix
   /// \param[in] b - load vector
   AdvectionDiffusionEvolution(mfem::BilinearForm &M, mfem::BilinearForm &K,
                               const mfem::Vector &b);

   /// Perform the action of the operator: y = k = f(x, t), where k solves
   /// Compute k = M^-1(Kx + l)
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Solve the implicit equation: k = f(x + dt k, t), for the unknown k at
   /// the current time t. 
   void ImplicitSolve(const double dt, const mfem::Vector &x,
                      mfem::Vector &k) override;

   virtual ~AdvectionDiffusionEvolution();

private:
   mfem::BilinearForm &M, &K;
   const mfem::Vector &b;
   /// solver for inverting mass matrix for explicit time-marching
   std::unique_ptr<mfem::Solver> M_prec;
   mfem::CGSolver M_solver;
   /// solver for implicit time-marching
   mfem::GSSmoother prec;
   mfem::GMRESSolver linear_solver;
   mfem::NewtonSolver newton;

   mutable mfem::Vector z;

   /// pointer-to-implementation idiom
   /// Hides implementation details of this operator
   class SystemOperator;
   /// Operator that combines the linear spatial discretization with
   /// the load vector into one operator used for implicit solves
   std::unique_ptr<SystemOperator> combined_oper;

   /// sets the state and dt for the combined operator
   /// \param[in] dt - time increment
   /// \param[in] x - the current state
   void setOperParameters(double dt, const mfem::Vector *x);

};

class PAJacobianOperator : public mfem::Operator
{
public:
   PAJacobianOperator(mfem::ParBilinearForm &_mass,
                      mfem::ParBilinearForm &_stiff);

   /// Compute r = J@k = M@k + dt*K@k
   /// \param[in] k - dx/dt
   /// \param[out] r - J@k = M@k + dt*K@k
   void Mult(const mfem::Vector &k, mfem::Vector &r) const override;
      
   /// Set current dt values - needed to compute action of Jacobian.
   void setParameters(double dt);

private:
   mfem::ParBilinearForm &mass;
   mfem::ParBilinearForm &stiff;

   double dt;
};

class ParSystemOperator : public mfem::Operator
{
public:
   /// Nonlinear operator of the form that combines the mass, res, stiff,
   /// and load elements for implicit/explicit ODE integration
   /// \param[in] ess_bdr - array of boundaries attributes marked essential
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] res - nonlinear residual operator (not owned)
   /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
   /// \param[in] load - load vector (not owned)
   /// \param[in] a - used to move the spatial residual to the rhs
   ParSystemOperator(mfem::ParBilinearForm &_mass,
                     mfem::ParBilinearForm &_stiff);

   /// Compute r = M@k + K@(x+dt*k)
   /// (with `@` denoting matrix-vector multiplication)
   /// \param[in] k - dx/dt 
   /// \param[out] r - the residual
   /// \note the signs on each operator must be accounted for elsewhere
   void Mult(const mfem::Vector &k, mfem::Vector &r) const override;

   /// Compute J = M + dt * K
   /// \param[in] k - dx/dt 
   mfem::Operator &GetGradient(const mfem::Vector &k) const override;

   /// Set current dt and x values - needed to compute action and Jacobian.
   void setParameters(double _dt, const mfem::Vector *_x);

   ~ParSystemOperator();

private:
   mfem::ParBilinearForm &mass;
   mfem::ParBilinearForm &stiff;
   mutable mfem::HypreParMatrix *jacobian, *stiff_jacobian;

   double dt;
   const mfem::Vector *x;

   mutable mfem::Vector work, work2;

   std::unique_ptr<PAJacobianOperator> pa_jac;

};

/** A time-dependent operator for the right-hand side of the ODE. The weak
    form of du/dt = -a.grad(u) + nu Delta(u) is M du/dt = K u + b, where M and
    K are the mass and advection-diffusion matrices, and b describes the flow
    on the boundary. This can be written as a general ODE,
    du/dt = M^{-1} (K u + b), and this class is used to evaluate the right-hand
    side. */
class ParAdvectionDiffusionEvolution : public mfem::TimeDependentOperator
{
public:
   /// \param[in] M - parallel bilinear form for mass matrix
   /// \param[in] K - parallel bilinear form for stiffness matrix
   ParAdvectionDiffusionEvolution(mfem::ParBilinearForm &M,
                                  mfem::ParBilinearForm &K);

   /// Perform the action of the operator: y = k = f(x, t), where k solves
   /// Compute k = M^-1(Kx + l)
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Solve the implicit equation: k = f(x + dt k, t), for the unknown k at
   /// the current time t. 
   void ImplicitSolve(const double dt, const mfem::Vector &x,
                      mfem::Vector &k) override;

   virtual ~ParAdvectionDiffusionEvolution();

private:
   mfem::OperatorHandle M_;
   mfem::ParBilinearForm &M, &K;
   /// solver for inverting mass matrix for explicit time-marching
   std::unique_ptr<mfem::Solver> M_prec;
   mfem::CGSolver M_solver;
   /// solver for implicit time-marching
   mfem::Solver *prec;
   mfem::GMRESSolver linear_solver;
   mfem::NewtonSolver newton;

   mfem::Vector diag;
   mutable mfem::Vector z, work, work2;

   /// pointer-to-implementation idiom
   /// Hides implementation details of this operator
   /// Operator that combines the linear spatial discretization with
   /// the load vector into one operator used for implicit solves
   std::unique_ptr<ParSystemOperator> combined_oper;

   /// sets the state and dt for the combined operator
   /// \param[in] dt - time increment
   /// \param[in] x - the current state
   void setOperParameters(double dt, const mfem::Vector *x);

};

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &X, Vector &v);

// Initial condition
double u0_function(const Vector &X);

// Inflow boundary condition
double inflow_function(const Vector &X, const double t);

// Mesh bounding box
Vector bb_min, bb_max;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   problem = 3;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 3;
   const char *device_config = "cpu";
   int ode_solver_type = 22;
   double t_final = 3 * 2*M_PI;
   double dt = 0.01;
   bool glvis = false;
   bool paraview = false;
   int vis_steps = 5;

   double nu_val = 0.001;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&glvis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&nu_val, "-nu", "--nu-value",
                  "Value for \nu, the parameter that controls diffusion.");
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
      std::cout << "Num ranks: " << num_procs << "\n";
      args.PrintOptions(cout);
   }
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle geometrically periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter. If the mesh is of NURBS type, we convert it
   //    to a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 6. Define the parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 7. Define the finite element space of the given
   //    polynomial order on the refined mesh.
   H1_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   // 8. Set up and assemble the bilinear and linear forms corresponding to the
   //    CG discretization.
   /// negative to move the diffusion terms to the right side
   ConstantCoefficient nu(-nu_val);
   ConstantCoefficient one(1.0);
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient u0(u0_function);

   ParBilinearForm *m_pa = new ParBilinearForm(fes);
   ParBilinearForm *k_pa = new ParBilinearForm(fes);
   m_pa->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   k_pa->SetAssemblyLevel(AssemblyLevel::PARTIAL);

   /// create mass matrix
   m_pa->AddDomainIntegrator(new MassIntegrator(one));
   /// add advection terms to stiffness matrix
   k_pa->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   /// add diffusion terms to stiffness matrix
   k_pa->AddDomainIntegrator(new DiffusionIntegrator(nu));

   m_pa->Assemble();
   int skip_zeros = 0;
   k_pa->Assemble(skip_zeros);
   m_pa->Finalize();
   k_pa->Finalize(skip_zeros);

   ParBilinearForm *m = new ParBilinearForm(fes);
   ParBilinearForm *k = new ParBilinearForm(fes);
   /// create mass matrix
   m->AddDomainIntegrator(new MassIntegrator);
   /// add advection terms to stiffness matrix
   k->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   /// add diffusion terms to stiffness matrix
   k->AddDomainIntegrator(new DiffusionIntegrator(nu));

   m->Assemble();
   k->Assemble(skip_zeros);
   m->Finalize();
   k->Finalize(skip_zeros);


   ParGridFunction *u = new ParGridFunction(fes);
   u->UseDevice(true);
   u->ProjectCoefficient(u0);


   HypreParVector *U = u->GetTrueDofs();

   ParSystemOperator pso(*m, *k);
   ParSystemOperator pso_pa(*m_pa, *k_pa);

   pso.setParameters(dt, U);
   pso_pa.setParameters(dt, U);

   MPI_Barrier(MPI_COMM_WORLD);
   mfem::Vector pso_r(U->Size());
   double t1 = MPI_Wtime();
   pso.Mult(*U, pso_r);
   double t2 = MPI_Wtime();
   double fa_mult_time = t2 - t1;
   double average_fa_mult_time;
   MPI_Reduce(&fa_mult_time, &average_fa_mult_time, 1,
              MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if (myid == 0)
      std::cout << "FA Mult time: " << average_fa_mult_time / num_procs << endl;

   MPI_Barrier(MPI_COMM_WORLD);
   mfem::Vector pso_pa_r(U->Size());
   double t3 = MPI_Wtime();
   pso_pa.Mult(*U, pso_pa_r);
   double t4 = MPI_Wtime();
   double pa_mult_time = t4 - t3;
   double average_pa_mult_time;
   MPI_Reduce(&pa_mult_time, &average_pa_mult_time, 1,
              MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if (myid == 0)
      std::cout << "FA Mult time: " << average_pa_mult_time / num_procs << endl;
   
   double local_mult_speedup = (t2-t1) / (t4-t3);
   double global_mult_speedup;
   MPI_Reduce(&local_mult_speedup, &global_mult_speedup, 1,
              MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   if (myid == 0)
      std::cout << "PA mult speedup: " << global_mult_speedup / num_procs << endl;

   mfem::Vector diff_r(pso_pa_r);
   diff_r -= pso_r;
   // std::cout << "r diff: " << diff_r.Norml2() << std::endl;

   mfem::Operator &pso_jac = pso.GetGradient(*U);
   mfem::Operator &pso_pa_jac = pso_pa.GetGradient(*U);

   MPI_Barrier(MPI_COMM_WORLD);
   mfem::Vector pso_jac_r(U->Size());
   double t5 = MPI_Wtime();
   pso_jac.Mult(*U, pso_jac_r);
   double t6 = MPI_Wtime();
   double fa_jac_mult_time = t6-t5;
   double average_fa_jac_time;
   MPI_Reduce(&fa_jac_mult_time, &average_fa_jac_time, 1,
              MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if (myid == 0)
      std::cout << "FA Jac Mult time: " << average_fa_jac_time / num_procs << endl;

   MPI_Barrier(MPI_COMM_WORLD);
   mfem::Vector pso_pa_jac_r(U->Size());
   double t7 = MPI_Wtime();
   pso_pa_jac.Mult(*U, pso_pa_jac_r);
   double t8 = MPI_Wtime();
   double pa_jac_mult_time = t8-t7;
   double average_pa_jac_time;
   MPI_Reduce(&pa_jac_mult_time, &average_pa_jac_time, 1,
              MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if (myid == 0)
      std::cout << "PA Jac Mult time: " << average_pa_jac_time / num_procs << endl;

   double local_jac_speedup = (t6-t5) / (t8-t7);
   double global_jac_speedup;
   MPI_Reduce(&local_jac_speedup, &global_jac_speedup, 1,
              MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   if (myid == 0)
      std::cout << "PA Jac mult speedup: " << global_jac_speedup / num_procs << endl;

   // 13. Free the used memory.
   delete U;
   delete u;
   delete k;
   delete m;
   delete fes;
   delete pmesh;

   MPI_Finalize();
   return 0;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 3:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      {
         // Clockwise rotation in 2D around the origin
         const double w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 0:
      {
         // Clockwise twisting rotation in 2D around the origin
         const double w = M_PI/2;
         double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
   }
}

// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x, const double t)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}

class AdvectionDiffusionEvolution::SystemOperator : public mfem::Operator
{
public:
   /// Nonlinear operator of the form that combines the mass, res, stiff,
   /// and load elements for implicit/explicit ODE integration
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] res - nonlinear residual operator (not owned)
   /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
   /// \param[in] load - load vector (not owned)
   /// \param[in] a - used to move the spatial residual to the rhs
   SystemOperator(BilinearForm &_mass, BilinearForm &_stiff,
                  const mfem::Vector &b)
      : Operator(_mass.Height()), mass(_mass), stiff(_stiff),
        load(b), Jacobian(NULL), dt(0.0), x(NULL), work(height)
      { }

   /// Compute r = M@k + K@(x+dt*k) + l
   /// (with `@` denoting matrix-vector multiplication)
   /// \param[in] k - dx/dt 
   /// \param[out] r - the residual
   /// \note the signs on each operator must be accounted for elsewhere
   void Mult(const mfem::Vector &k, mfem::Vector &r) const override
   {
      /// work = x+dt*k = x+dt*dx/dt = x+dx
      add(1.0, *x, dt, k, work);
      r = 0.0;
      stiff.AddMult(work, r);
      r += load;
      mass.AddMult(k, r, -1.0);
   }

   /// Compute J = M + dt * K
   /// \param[in] k - dx/dt 
   mfem::Operator &GetGradient(const mfem::Vector &k) const override
   {
      delete Jacobian;
      Jacobian = Add(-1.0, mass.SpMat(), dt, stiff.SpMat());
      return *Jacobian;
   }

   /// Set current dt and x values - needed to compute action and Jacobian.
   void setParameters(double _dt, const mfem::Vector *_x)
   {
      dt = _dt;
      x = _x;
   };

   ~SystemOperator() {delete Jacobian;};

private:
   BilinearForm &mass;
   BilinearForm &stiff;
   const mfem::Vector &load;
   mutable mfem::SparseMatrix *Jacobian;

   double dt;
   const mfem::Vector *x;

   mutable mfem::Vector work, work2;

};

AdvectionDiffusionEvolution::AdvectionDiffusionEvolution(
      BilinearForm &_M, BilinearForm &_K, const Vector &_b)
   : TimeDependentOperator(_M.Height()), M(_M), K(_K), b(_b),
     z(_M.Height())
{
   bool pa = M.GetAssemblyLevel() == AssemblyLevel::PARTIAL;
   Array<int> ess_tdof_list;
   if (pa)
   {
      M_prec.reset(new OperatorJacobiSmoother(M, ess_tdof_list));
      M_solver.SetOperator(M);
   }
   else
   {
      M_prec.reset(new DSmoother(M.SpMat()));
      M_solver.SetOperator(M.SpMat());
   }

   combined_oper.reset(new SystemOperator(_M, _K, _b));

   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);

   linear_solver.iterative_mode = true;
   linear_solver.SetRelTol(1e-12);
   linear_solver.SetAbsTol(0.0);
   linear_solver.SetMaxIter(100);
   linear_solver.SetPrintLevel(0);
   linear_solver.SetPreconditioner(prec);

   newton.iterative_mode = false;
   newton.SetRelTol(1e-9);
   newton.SetAbsTol(0.0);
   newton.SetMaxIter(100);
   newton.SetPrintLevel(-1);
   newton.SetSolver(linear_solver);
   newton.SetOperator(*combined_oper);
}

void AdvectionDiffusionEvolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

void AdvectionDiffusionEvolution::ImplicitSolve(const double dt,
                                                const Vector &x,
                                                Vector &k)
{
   setOperParameters(dt, &x);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   newton.Mult(zero, k);
   MFEM_VERIFY(newton.GetConverged(), "Newton solver did not converge!");
}

void AdvectionDiffusionEvolution::setOperParameters(double dt,
                                                    const mfem::Vector *x)
{
   combined_oper->setParameters(dt, x);
}

AdvectionDiffusionEvolution::~AdvectionDiffusionEvolution() {}


PAJacobianOperator::PAJacobianOperator(ParBilinearForm &_mass, ParBilinearForm &_stiff)
 : Operator(_mass.ParFESpace()->GetTrueVSize()), mass(_mass), stiff(_stiff),
   dt(0.0) { }


void PAJacobianOperator::Mult(const mfem::Vector &k, mfem::Vector &r) const
{
   r.UseDevice(true);
   r = 0.0;
   stiff.TrueAddMult(k, r, dt);
   mass.TrueAddMult(k, r, -1.0);
}
   
void PAJacobianOperator::setParameters(const double _dt)
{
   dt = _dt;
};

ParSystemOperator::ParSystemOperator(ParBilinearForm &_mass, ParBilinearForm &_stiff)
   : Operator(_mass.ParFESpace()->GetTrueVSize()), mass(_mass), stiff(_stiff),
   jacobian(NULL), stiff_jacobian(NULL), dt(0.0), x(NULL),
   work(height)
{
   pa_jac.reset(new PAJacobianOperator(mass, stiff));
}

/// Compute r = M@k + K@(x+dt*k)
/// (with `@` denoting matrix-vector multiplication)
/// \param[in] k - dx/dt 
/// \param[out] r - the residual
/// \note the signs on each operator must be accounted for elsewhere
void ParSystemOperator::Mult(const mfem::Vector &k, mfem::Vector &r) const
{
   r = 0.0;
   work.UseDevice(true);
   work = 0.0;
   /// work = x+dt*k = x+dt*dx/dt = x+dx
   if (x)
   {
      add(1.0, *x, dt, k, work);
   }

   stiff.TrueAddMult(work, r);
   mass.TrueAddMult(k, r, -1.0);
}

/// Compute J = M + dt * K
/// \param[in] k - dx/dt 
mfem::Operator &ParSystemOperator::GetGradient(const mfem::Vector &k) const
{
   bool mass_pa = mass.GetAssemblyLevel() == AssemblyLevel::PARTIAL;
   bool stiff_pa = stiff.GetAssemblyLevel() == AssemblyLevel::PARTIAL;
   
   if (mass_pa && stiff_pa)
   {
      return *pa_jac.get();
   }
   else
   {
      delete stiff_jacobian;
      delete jacobian;
      jacobian = mass.ParallelAssemble();
      *jacobian *= -1.0; //alpha;
      stiff_jacobian = stiff.ParallelAssemble();
      jacobian->Add(dt, *stiff_jacobian);
      return *jacobian;
   }
}

/// Set current dt and x values - needed to compute action and Jacobian.
void ParSystemOperator::setParameters(const double _dt, const mfem::Vector *_x)
{
   dt = _dt;
   x = _x;
   pa_jac->setParameters(_dt);
};

ParSystemOperator::~ParSystemOperator()
{
   delete jacobian;
   delete stiff_jacobian;
};

ParAdvectionDiffusionEvolution::ParAdvectionDiffusionEvolution(
      ParBilinearForm &_M, ParBilinearForm &_K)
   : TimeDependentOperator(_M.ParFESpace()->GetTrueVSize()), M(_M), K(_K), z(_M.Height())
{
   bool mass_pa = M.GetAssemblyLevel() == AssemblyLevel::PARTIAL;
   bool stiff_pa = K.GetAssemblyLevel() == AssemblyLevel::PARTIAL;

   Array<int> ess_tdof_list;
   M_solver = CGSolver(MPI_COMM_WORLD);
   if (mass_pa)
   {
      M_prec.reset(new OperatorJacobiSmoother(M, ess_tdof_list));
      M_solver.SetOperator(M);
   }
   else
   {
      M_.Reset(_M.ParallelAssemble(), true);

      // M_prec.reset(new HypreSmoother());
      // M_solver.SetOperator(M.As<HypreParMatrix>());
      HypreParMatrix &M_mat = *M_.As<HypreParMatrix>();
      // HypreParMatrix &K_mat = *K.As<HypreParMatrix>();
      M_prec.reset(new HypreSmoother(M_mat, HypreSmoother::Jacobi));
   }

   combined_oper.reset(new ParSystemOperator(_M, _K));

   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);

   if (mass_pa && stiff_pa)
   {
      diag.UseDevice(true);
      diag.SetSize(M.ParFESpace()->GetTrueVSize());
      diag = 0.0;
      work.UseDevice(true);
      work2.UseDevice(true);
      work.SetSize(M.ParFESpace()->GetTrueVSize());
      work2.SetSize(M.ParFESpace()->GetTrueVSize());
      work = 0.0;
      work2 = 0.0;
      M.AssembleDiagonal(work);

      ParBilinearForm k(M.ParFESpace());
      ConstantCoefficient nu(-0.01);
      k.AddDomainIntegrator(new mfem::DiffusionIntegrator(nu));
      k.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      k.Assemble(0);
      k.Finalize(0);
      k.AssembleDiagonal(work2);

      double dt = 0.1;
      add(-1.0, work, dt, work2, diag);

      prec = new OperatorChebyshevSmoother(combined_oper.get(), diag,
                                           ess_tdof_list, 5,
                                           M.ParFESpace()->GetComm());
   }
   else
   {
      prec = new HypreSmoother();
   }
   
   linear_solver = GMRESSolver(MPI_COMM_WORLD);
   linear_solver.iterative_mode = true;
   linear_solver.SetRelTol(1e-12);
   linear_solver.SetAbsTol(0.0);
   linear_solver.SetMaxIter(2000);
   linear_solver.SetPrintLevel(0);
   linear_solver.SetPreconditioner(*prec);
   linear_solver.SetKDim(2000);

   newton.iterative_mode = true;
   newton.SetRelTol(1e-9);
   newton.SetAbsTol(0.0);
   newton.SetMaxIter(10);
   newton.SetPrintLevel(-1);
   newton.SetSolver(linear_solver);
   newton.SetOperator(*combined_oper);
}

void ParAdvectionDiffusionEvolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   M_solver.Mult(z, y);
}

void ParAdvectionDiffusionEvolution::ImplicitSolve(const double dt,
                                                   const Vector &x,
                                                   Vector &k)
{
   setOperParameters(dt, &x);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   newton.Mult(zero, k);
   MFEM_VERIFY(newton.GetConverged(), "Newton solver did not converge!");
}

void ParAdvectionDiffusionEvolution::setOperParameters(const double dt,
                                                       const mfem::Vector *x)
{
   combined_oper->setParameters(dt, x);
}

ParAdvectionDiffusionEvolution::~ParAdvectionDiffusionEvolution() {delete prec;}
