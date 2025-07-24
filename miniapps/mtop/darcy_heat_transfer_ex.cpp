//                               MFEM Darcy Test Run
//
// Compile with: make darcy_heat_transfer_ex
//
//
// Description:  This code performs the forward and backward adjoint solve for advection diffusion, where the velocity field is given by Darcy 


#include "mfem.hpp"
#include <fstream>
#include <iostream>


using namespace std;
using namespace mfem;


// *****Funtion definitions for the Advection-Diffusion solve******

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double theta0_function(const Vector &x);

// true solution
real_t theta_exact(const Vector &x, real_t t);

// rhs
double forcing_function(const Vector &x, real_t t);

// inflow
double inflow_function(const Vector &x);

real_t f_natural(const Vector & x);

// Mesh bounding box
Vector bb_min, bb_max;


/** A time-dependent operator for the right-hand side of the ODE for use with
    explicit ODE solvers. The DG weak form of du/dt = div(D grad(u))-v.grad(u) is
    M du/dt = - S u + K u + b, where M, S, and K are the mass,
    stiffness, and advection matrices, and b describes sources and the flow on
    the boundary.
    This can be written as a general ODE,
    du/dt = M^{-1} (-S u + K u + b), and this class is used to compute the RHS
    and perform the solve for du/dt. */
class EX_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &S, &K;
   const Vector &b;

   DSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

   void initA(double dt);

public:
   EX_Evolution(SparseMatrix &_M, SparseMatrix &_S, SparseMatrix &_K,
                const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~EX_Evolution() {}
};

/** A time-dependent operator for the right-hand side of the ODE for use with
    implicit ODE solvers. The DG weak form of du/dt = div(D grad(u))-v.grad(u) is
    [M + dt (S - K)] du/dt = - S u + K u + b, where M, S, and K are the mass,
    stiffness, and advection matrices, and b describes sources and the flow on
    the boundary.
    This can be written as a general ODE,
    du/dt = A^{-1} (-S u + K u + b) with A = [M + dt (S - K)], and this class is
    used to perform the fully implicit solve for du/dt. */
class IM_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &S, &K;
   SparseMatrix *A;
   const Vector &b;

   DSmoother M_prec;
   CGSolver M_solver;

   DSmoother *A_prec;
   GMRESSolver *A_solver;
   double dt;

   mutable Vector z;

   void initA(double dt);

public:
   IM_Evolution(SparseMatrix &_M, SparseMatrix &_S, SparseMatrix &_K,
                const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y);

   virtual ~IM_Evolution() { delete A_solver; delete A_prec; delete A; }
};

/** A time-dependent operator for the right-hand side of the ODE for use with
    IMEX (Implicit-Explicit) ODE solvers. The DG weak form of
    du/dt = div(D grad(u))-v.grad(u) is
    [M + dt S] du/dt = - S u + K u + b, where M, S, and K are the mass,
    stiffness, and advection matrices, and b describes sources and the flow on
    the boundary.
    This can be written as a general ODE,
    du/dt = A^{-1} (-S u + K u + b) with A = [M + dt (S - K)], and this class is
    used to perform the implicit or explicit solve for du/dt. */
class IMEX_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &S, &K;
   SparseMatrix *A;
   const Vector &b;

   DSmoother M_prec;
   CGSolver M_solver;

   DSmoother *A_prec;
   CGSolver *A_solver;
   double dt;

   mutable Vector z;

   void initA(double dt);

public:
   IMEX_Evolution(SparseMatrix &_M, SparseMatrix &_S, SparseMatrix &_K,
                  const Vector &_b);

   virtual void ExplicitMult(const Vector &x, Vector &y) const;
   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y);

   virtual ~IMEX_Evolution() { delete A_solver; delete A_prec; delete A; }
};

// *****Define the analytical solution and forcing terms / boundary conditions for Darcy*****
void uFun_ex(const Vector & x, Vector & u);
real_t pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
real_t gFun(const Vector & x);

int main(int argc, char *argv[]){
    // 1. Parse command-line options.
   const char *mesh_file = "square-extended.mesh"; //reference square, but extended to be [-1, 1] x [-1, 1]
   int order_darcy = 1;
   int ref_levels = 2;
   int order_ad = 3;
   int ode_solver_type = 4;
   double t_final = 10.0;
   double d_coef = 0.1;
   double dt = 0.01;
   double sigma = -1.0;
   double kappa = -1.0;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 5;
   bool paraview = false;
   int precision = 16;
   const char *device_config = "cpu";
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order_darcy, "-od", "--order_darcy",
                  "Order (degree) of the finite elements for darcy solve.");
    args.AddOption(&order_ad, "-oad", "--order_ad",
                  "Order (degree) of the finite elements for advection diffusion.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler, 2 - RK4, 3 - Implicit Euler,"
                  " 4 - IMEX Implicit-Explicit Euler, 5 - IMEX RK2.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&d_coef, "-d", "--diff-coef",
                  "Diffusion coefficient.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
    args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order_ad+1)*(order_ad+1);
   }
   args.PrintOptions(cout);
   Device device(device_config);
   device.Print();

   // 2. Define the ODE solver used for time integration. Several explicit, implicit and IMEX
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   ODESolver *ode_solver_adj = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; ode_solver_adj = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK4Solver; ode_solver_adj = new RK4Solver; break;
      case 3: ode_solver = new BackwardEulerSolver; ode_solver_adj = new BackwardEulerSolver; break;
      case 4: ode_solver = new IMEX_BE_FE; ode_solver_adj = new IMEX_BE_FE; break;
      case 5: ode_solver = new IMEXRK2; ode_solver_adj = new IMEXRK2; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 3. Read the mesh from the given mesh file. 
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

    // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter. 
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   mesh.GetBoundingBox(bb_min, bb_max, max(order_ad, 1));

   // ********DARCY SOLVE
   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order_darcy, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order_darcy, dim));

   FiniteElementSpace *R_space = new FiniteElementSpace(&mesh, hdiv_coll);
   FiniteElementSpace *W_space = new FiniteElementSpace(&mesh, l2_coll);

   // 6. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = R_space->GetVSize();
   block_offsets[2] = W_space->GetVSize();
   block_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // 7. Define the coefficients, analytical solution, and rhs of the Darcy PDE.
   ConstantCoefficient k(1.0);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   FunctionCoefficient fnatcoeff(f_natural);
   FunctionCoefficient gcoeff(gFun);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);

   // 8. Allocate memory for solution and rhs of Darcy
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   LinearForm *fform(new LinearForm);
   fform->Update(R_space, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   LinearForm *gform(new LinearForm);
   gform->Update(W_space, rhs.GetBlock(1), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform->Assemble();
   gform->SyncAliasMemory(rhs);

   // 9. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   BilinearForm *mVarf(new BilinearForm(R_space));
   mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
   mVarf->Assemble();
   MixedBilinearForm *bVarf(new MixedBilinearForm(R_space, W_space));
   bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf->Assemble();
   mVarf->Finalize();
   bVarf->Finalize();
   BlockOperator darcyOp(block_offsets);
   TransposeOperator *Bt = NULL;
   SparseMatrix &M(mVarf->SpMat());
   SparseMatrix &B(bVarf->SpMat());
   B *= -1.;
   Bt = new TransposeOperator(&B);
   darcyOp.SetBlock(0,0, &M);
   darcyOp.SetBlock(0,1, Bt);
   darcyOp.SetBlock(1,0, &B);

   // 10. Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     pressure Schur Complement
   SparseMatrix *MinvBt = NULL;
   Vector Md(mVarf->Height());

   BlockDiagonalPreconditioner darcyPrec(block_offsets);
   Solver *invM, *invS;
   SparseMatrix *S = NULL;
   // SparseMatrix &M(mVarf->SpMat());
   M.GetDiag(Md);
   Md.HostReadWrite();
   // SparseMatrix &B(bVarf->SpMat());
   MinvBt = Transpose(B);
   for (int i = 0; i < Md.Size(); i++)
   {
      MinvBt->ScaleRow(i, 1./Md(i));
   }
   S = Mult(B, *MinvBt);
   invM = new DSmoother(M);
   #ifndef MFEM_USE_SUITESPARSE
     invS = new GSSmoother(*S);
   #else
     invS = new UMFPackSolver(*S);
   #endif
   invM->iterative_mode = false;
   invS->iterative_mode = false;

   darcyPrec.SetDiagonalBlock(0, invM);
   darcyPrec.SetDiagonalBlock(1, invS);

   // 11. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.
   int maxIter(1000);
   real_t rtol(1.e-6);
   real_t atol(1.e-10);

   MINRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(darcyOp);
   solver.SetPreconditioner(darcyPrec);
   solver.SetPrintLevel(1);
   x = 0.0;
   solver.Mult(rhs, x);

   if (solver.GetConverged())
   {
      std::cout << "MINRES converged in " << solver.GetNumIterations()
                << " iterations with a residual norm of "
                << solver.GetFinalNorm() << ".\n";
   }
   else
   {
      std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                << " iterations. Residual norm is " << solver.GetFinalNorm()
                << ".\n";
   }

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u, p;
   u.MakeRef(R_space, x.GetBlock(0), 0);
   p.MakeRef(W_space, x.GetBlock(1), 0);

   int order_quad = max(2, 2*order_darcy+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   real_t err_u  = u.ComputeL2Error(ucoeff, irs);
   real_t norm_u = ComputeLpNorm(2., ucoeff, mesh, irs);
   real_t err_p  = p.ComputeL2Error(pcoeff, irs);
   real_t norm_p = ComputeLpNorm(2., pcoeff, mesh, irs);

   std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
   std::cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";

   if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream u_sock(vishost, visport);
        u_sock.precision(8);
        u_sock << "solution\n" << mesh << u << "window_title 'Velocity'" << endl;
        socketstream p_sock(vishost, visport);
        p_sock.precision(8);
        p_sock << "solution\n" << mesh << p << "window_title 'Pressure'" << endl;
    }


   // ******Forward Advection-Diffusion solve
   // 13. Define the DG finite element space on the
   //    refined mesh of the given polynomial order.
   DG_FECollection fec(order_ad, dim);
   FiniteElementSpace fes(&mesh, &fec);
   int num_dofs = fes.GetNDofs();

   cout << "Number of unknowns (advection diffusion problem): " << fes.GetVSize() << endl;

   // 14. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the DG discretization. The
   //    DGTraceIntegrator involves integrals over mesh interior faces.
   ConstantCoefficient diff_coef(d_coef); 
   const GridFunction* u_pointer = &u;
   VectorGridFunctionCoefficient velocity(u_pointer);
   FunctionCoefficient theta0(theta0_function);
   FunctionCoefficient forcing(forcing_function);
   // FunctionCoefficient theta_exact_coeff(theta_exact);
   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);

   BilinearForm s(&fes);
   s.AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
   s.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma, kappa));
   s.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma, kappa));

   BilinearForm a(&fes);
   a.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   a.AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, -1.0));
   a.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, -1.0));


   LinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(forcing));
   //b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(U, diff_coef, sigma, kappa));

   int skip_zeros = 0;
   m.Assemble(skip_zeros);
   m.Finalize(skip_zeros);
   s.Assemble(skip_zeros);
   s.Finalize(skip_zeros);
   a.Assemble(skip_zeros);
   a.Finalize(skip_zeros);
   b.Assemble();

   // 15. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction theta(&fes);
   theta.ProjectCoefficient(theta0);
   ParaViewDataCollection *pd_forward = NULL;
   if (paraview)
   {
      pd_forward = new ParaViewDataCollection("darcy-adv-diff-forward", &mesh);
      pd_forward->SetPrefixPath("ParaView");
      pd_forward->RegisterField("solution_forward", &theta);
      pd_forward->SetLevelsOfDetail(order_ad);
      pd_forward->SetDataFormat(VTKFormat::BINARY);
      pd_forward->SetHighOrderOutput(true);
      pd_forward->SetCycle(0);
      pd_forward->SetTime(0.0);
      pd_forward->Save();
   }



   // 16. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   TimeDependentOperator *adv = NULL;
   if (ode_solver_type < 3)
   {
      adv = new EX_Evolution(m.SpMat(), s.SpMat(), a.SpMat(), b);
   }
   else if (ode_solver_type == 3)
   {
      adv = new IM_Evolution(m.SpMat(), s.SpMat(), a.SpMat(), b);
   }
   else
   {
      adv = new IMEX_Evolution(m.SpMat(), s.SpMat(), a.SpMat(), b);
   }

   double t = 0.0;
   adv->SetTime(t);
   ode_solver->Init(*adv);

   int n_steps = (int)ceil(t_final / dt);
   double dt_real = t_final / n_steps;
   // Vector err_vec(n_steps-1);

   std::vector<GridFunction> theta_gf_vector;
   theta_gf_vector.push_back(theta);

   for (int ti = 0; ti < n_steps; ti++)
   {
      ode_solver->Step(theta, t, dt_real);
      theta_gf_vector.push_back(theta);
    //   theta_exact_coeff.SetTime(t);
    //   double loc_err = theta.ComputeL2Error(theta_exact_coeff);
    //   err_vec(ti) = loc_err;
    //   cout << "\n|| E_h - E ||_{L^2} = " << loc_err << '\n' << endl;
      if (ti % vis_steps == 0 || ti == n_steps -1)
      {
        cout << "time step: " << ti << ", time: " << t << endl;
        if (paraview)
         {
            pd_forward->SetCycle(ti);
            pd_forward->SetTime(t);
            pd_forward->Save();
         }
      }
   }

//    real_t err_norm = err_vec.Norml2();
//    cout << "\n Error = " << err_norm << '\n' << endl;

   // ******Backward Advection-Diffusion solve
      // 17. Define the DG finite element space on the
   //    refined mesh of the given polynomial order.
   DG_FECollection fec_adjoint(order_ad, dim);
   FiniteElementSpace fes_adjoint(&mesh, &fec_adjoint);

   // 18. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the DG discretization. The
   //    DGTraceIntegrator involves integrals over mesh interior faces.
   ConstantCoefficient zero(0.0);
   ConstantCoefficient diff_coef_adj(-d_coef); 
   GridFunctionCoefficient theta_coeff(&(theta_gf_vector[n_steps-1]));
   FunctionCoefficient forcing_adj(forcing_function); //zero for now

   // FunctionCoefficient theta_exact_coeff(theta_exact);
   BilinearForm m_adj(&fes_adjoint);
   m_adj.AddDomainIntegrator(new MassIntegrator);

   BilinearForm s_adj(&fes_adjoint);
   s_adj.AddDomainIntegrator(new DiffusionIntegrator(diff_coef_adj));
   s_adj.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coef_adj, sigma, kappa));
   s_adj.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coef_adj, sigma, kappa));

   BilinearForm a_adj(&fes_adjoint);
   a_adj.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   a_adj.AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, -1.0));
   a_adj.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, -1.0));


   LinearForm b_adj(&fes_adjoint);
   b_adj.AddDomainIntegrator(new DomainLFIntegrator(theta_coeff));
   //b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(zero, diff_coef, sigma, kappa));

   // int skip_zeros = 0;
   m_adj.Assemble(skip_zeros);
   m_adj.Finalize(skip_zeros);
   s_adj.Assemble(skip_zeros);
   s_adj.Finalize(skip_zeros);
   a_adj.Assemble(skip_zeros);
   a_adj.Finalize(skip_zeros);
   b_adj.Assemble();

   // 19. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction lam(&fes_adjoint);
   lam.ProjectCoefficient(zero);
   ParaViewDataCollection *pd_backward = NULL;
   if (paraview)
   {
      pd_backward = new ParaViewDataCollection("darcy-adv-diff-backward", &mesh);
      pd_backward->SetPrefixPath("ParaView");
      pd_backward->RegisterField("solution-backward", &lam);
      pd_backward->SetLevelsOfDetail(order_ad);
      pd_backward->SetDataFormat(VTKFormat::BINARY);
      pd_backward->SetHighOrderOutput(true);
      pd_backward->SetCycle(0);
      pd_backward->SetTime(t_final);
      pd_backward->Save();
   }

   // 20. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   TimeDependentOperator *adv_adj = NULL;
   if (ode_solver_type < 3)
   {
      adv_adj = new EX_Evolution(m_adj.SpMat(), s_adj.SpMat(), a_adj.SpMat(), b_adj);
   }
   else if (ode_solver_type == 3)
   {
      adv_adj = new IM_Evolution(m_adj.SpMat(), s_adj.SpMat(), a_adj.SpMat(), b_adj);
   }
   else
   {
      adv_adj = new IMEX_Evolution(m_adj.SpMat(), s_adj.SpMat(), a_adj.SpMat(), b_adj);
   }

   double t_adj = t_final;
   adv_adj->SetTime(t_adj);
   ode_solver_adj->Init(*adv_adj);

   // int n_steps = (int)ceil(t_final / dt);
   double dt_real_adj = -dt;
   std::cout << "dt back = " << dt_real_adj << std::endl;
   //Vector err_vec(n_steps-1);

   for (int ti = 0; ti < n_steps; ti++)
   {
      ode_solver_adj->Step(lam, t_adj, dt_real_adj);
      Vector lam_vals(num_dofs);
      Vector theta_values(num_dofs);
      const GridFunction* theta_gf = theta_coeff.GetGridFunction();
      theta_gf->GetTrueDofs(theta_values);
      lam.GetTrueDofs(lam_vals);
      // if(ti == n_steps - 1){cout << "lam_val: " << lam_vals(12) << endl; cout << "theta_val: " << theta_values(12) << endl;}
      theta_coeff = *(new GridFunctionCoefficient(&(theta_gf_vector[n_steps - ti - 1])));
      b_adj = *(new LinearForm(&fes_adjoint));
      b_adj.AddDomainIntegrator(new DomainLFIntegrator(theta_coeff));
      b_adj.Assemble();
      if (ti % vis_steps == 0 || ti == n_steps - 1) 
      {
        cout << "time step: " << ti << ", time: " << t_adj << endl;
        if (paraview) 
         {
            pd_backward->SetCycle(ti); 
            pd_backward->SetTime(t_adj);
            pd_backward->Save();
         }
      }
   }

//    real_t err_norm = err_vec.Norml2();
//    cout << "\n Error = " << err_norm << '\n' << endl;

   // 21. Free the used memory.
   delete ode_solver;
   delete adv;
   delete adv_adj;
   delete fform;
   delete gform;
   delete invM;
   delete invS;
   delete S;
   delete Bt;
   delete MinvBt;
   delete mVarf;
   delete bVarf;
   delete W_space;
   delete R_space;
   delete l2_coll;
   delete hdiv_coll;
   // delete &b_adj;
   // delete &theta_coeff;

   return 0;
}

void uFun_ex(const Vector & x, Vector & u)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(0.0);
   if (x.Size() == 3)
   {
      zi = x(2);
   }

   u(0) = - exp(xi)*sin(yi)*cos(zi);
   u(1) = - exp(xi)*cos(yi)*cos(zi);

   if (x.Size() == 3)
   {
      u(2) = exp(xi)*sin(yi)*sin(zi); 
   }
}

// Change if needed
real_t pFun_ex(const Vector & x)
{
   real_t xi(x(0));
   real_t yi(x(1));
   real_t zi(0.0);

   if (x.Size() == 3)
   {
      zi = x(2);
   }

   return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const Vector & x, Vector & f)
{
   f = 0.0;
}

real_t gFun(const Vector & x)
{
   if (x.Size() == 3)
   {
      return -pFun_ex(x);
   }
   else
   {
      return 0;
   }
}

real_t f_natural(const Vector & x)
{
   return (-pFun_ex(x));
}



// Implementation of class EX_Evolution
EX_Evolution::EX_Evolution(SparseMatrix &_M, SparseMatrix &_S,
                           SparseMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Height()),
     M(_M), S(_S), K(_K), b(_b), z(_M.Height())
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void EX_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (-S x + K x + b)
   K.Mult(x, z);
   S.AddMult(x, z, -1.0);
   z += b;
   M_solver.Mult(z, y);
}

// Implementation of class IM_Evolution
IM_Evolution::IM_Evolution(SparseMatrix &_M, SparseMatrix &_S,
                           SparseMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Height()),
     M(_M), S(_S), K(_K), A(NULL), b(_b),
     A_prec(NULL), A_solver(NULL), dt(-1.0), z(_M.Height())
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void IM_Evolution::initA(double _dt)
{
   if (fabs(dt - _dt) > 1e-4 * _dt)
   {
      delete A_solver;
      delete A_prec;
      delete A;

      SparseMatrix * SK = Add(1.0, S, -1.0, K);
      A = Add(1.0, M, _dt, *SK);
      delete SK;
      dt = _dt;

      A_prec = new DSmoother(*A);
      A_solver = new GMRESSolver;
      A_solver->SetOperator(*A);
      A_solver->SetPreconditioner(*A_prec);

      A_solver->iterative_mode = false;
      A_solver->SetRelTol(1e-9);
      A_solver->SetAbsTol(0.0);
      A_solver->SetMaxIter(100);
      A_solver->SetPrintLevel(0);
   }
}

void IM_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (-S x + K x + b)
   K.Mult(x, z);
   S.AddMult(x, z, -1.0);
   z += b;
   M_solver.Mult(z, y);
}

void IM_Evolution::ImplicitSolve(const double _dt, const Vector &x, Vector &y)
{
   this->initA(_dt);

   // y = (M + dt S - dt K)^{-1} (-S x + K x + b)
   K.Mult(x, z);
   S.AddMult(x, z, -1.0);
   z += b;
   A_solver->Mult(z, y);
}

// Implementation of class IMEX_Evolution
IMEX_Evolution::IMEX_Evolution(SparseMatrix &_M, SparseMatrix &_S,
                               SparseMatrix &_K, const Vector &_b)
   : TimeDependentOperator(_M.Height()),
     M(_M), S(_S), K(_K), A(NULL), b(_b),
     A_prec(NULL), A_solver(NULL), dt(-1.0), z(_M.Height())
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void IMEX_Evolution::initA(double _dt)
{
   if (fabs(dt - _dt) > 1e-4 * _dt)
   {
      delete A_solver;
      delete A_prec;
      delete A;

      A = Add(_dt, S, 1.0, M); // A = M + dt * S
      dt = _dt;

      A_prec = new DSmoother(*A);
      A_solver = new CGSolver;
      A_solver->SetOperator(*A);
      A_solver->SetPreconditioner(*A_prec);

      A_solver->iterative_mode = false;
      A_solver->SetRelTol(1e-9);
      A_solver->SetAbsTol(0.0);
      A_solver->SetMaxIter(100);
      A_solver->SetPrintLevel(0);
   }
}

void IMEX_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (-S x + K x + b)
   K.Mult(x, z);
   S.AddMult(x, z, -1.0);
   z += b;
   M_solver.Mult(z, y);
}

void IMEX_Evolution::ExplicitMult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

void IMEX_Evolution::ImplicitSolve(const double _dt, const Vector &x, Vector &y)
{
   this->initA(_dt);
   // y = (M + dt S)^{-1} (-S x + b)
   S.Mult(x, z);
   z *= -1.0;
   z += b;
   A_solver->Mult(z, y);
}

// Initial condition
double theta0_function(const Vector &x)
{
   int dim = x.Size();
   // map to the reference [-1,1] domain
   Vector X(dim);
   // for (int i = 0; i < dim; i++)
   // {
   //    double center = (bb_min[i] + bb_max[i]) * 0.5;
   //    X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   // }

    double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
    if (dim == 3)
    {
        const double s = (1. + 0.25*cos(2*M_PI*x(2)));
        rx *= s;
        ry *= s;
    }
    return ( erfc(w*(x(0)-cx-rx))*erfc(-w*(x(0)-cx+rx))*erfc(w*(x(1)-cy-ry))*erfc(-w*(x(1)-cy+ry)) )/16;
}

//forcing term 
real_t forcing_function(const Vector &x, real_t t)
{
   int dim = x.Size();
   //map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]); 
   }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   return 0.0;
}