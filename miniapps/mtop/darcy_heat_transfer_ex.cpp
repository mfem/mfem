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

class DG_Solver : public Solver
{
private:
   SparseMatrix &M, &K, &S, A;
   CGSolver linear_solver;
   BlockILU prec;
   real_t dt;
public:
   DG_Solver(SparseMatrix &M_, SparseMatrix &K_, SparseMatrix &S_,
             const FiniteElementSpace &fes)
      : M(M_),
        K(K_),
        S(S_),
        prec(fes.GetTypicalFE()->GetDof(),
             BlockILU::Reordering::MINIMUM_DISCARDED_FILL),
        dt(1.0)
   {
      linear_solver.iterative_mode = false;
      linear_solver.SetRelTol(1e-9);
      linear_solver.SetAbsTol(0.0);
      linear_solver.SetMaxIter(100);
      linear_solver.SetPrintLevel(0);
      linear_solver.SetPreconditioner(prec);
   }

   void SetTimeStep(real_t dt_)
   {
      if (dt_ != dt)
      {
         dt = dt_;
         // Form operator A = M + dt*S
         A = S;
         A *= dt;
         A += M;

         // this will also call SetOperator on the preconditioner
         linear_solver.SetOperator(A);
      }
   }

   void SetOperator(const Operator &op) override
   {
      linear_solver.SetOperator(op);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      linear_solver.Mult(x, y);
   }
};


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of the advection-diffusion equation is (M + dt S) du/dt = Su - K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. In the case of IMEX evolution, the diffusion term is treated
    implicitly, and the advection term is treated explicitly.  */
class IMEX_Evolution : public SplitTimeDependentOperator
{
private:
   BilinearForm &M, &K, &S;
   const Vector &b;
   unique_ptr<Solver> M_prec;
   CGSolver M_solver;
   unique_ptr<DG_Solver> dg_solver;

   mutable Vector z;

public:
   IMEX_Evolution(BilinearForm &M_, BilinearForm &K_, BilinearForm &S_,
                  const Vector &b_);

   void Mult1(const Vector &x, Vector &y) const;
   void ImplicitSolve2(const real_t dt, const Vector &x, Vector &k) override;
};

// *****Define the analytical solution and forcing terms / boundary conditions for Darcy*****
void uFun_ex(const Vector & x, Vector & u);
real_t pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
real_t gFun(const Vector & x);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file =
      "square-extended.mesh"; //reference square, but extended to be [-1, 1] x [-1, 1]
   int order_darcy = 1;
   int ref_levels = 2;
   int order_ad = 3;
   int ode_solver_type = 55;
   double t_final = 10.0;
   double d_coef = 0.01;
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
                  "55 - Forward Backward Euler, 56 - IMEXRK2(2,2,2), 57 - IMEXRK2(2,3,2), 58 - IMEX_DIRK_RK3\n");
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
   unique_ptr<SplitODESolver> ode_solver = SplitODESolver::Select(ode_solver_type);
   unique_ptr<SplitODESolver> ode_solver_adj = SplitODESolver::Select(
                                                  ode_solver_type);
   // 3. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++) {mesh.UniformRefinement();}
   if (mesh.NURBSext) {mesh.SetCurvature(max(order_ad, 1));}
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
   ConstantCoefficient one(1.0);

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
   mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(one));
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
   DG_FECollection fec(order_ad, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);
   int num_dofs = fes.GetNDofs();

   cout << "Number of unknowns (advection diffusion problem): " << fes.GetVSize()
        << endl;

   // 14. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the DG discretization. The
   //    DGTraceIntegrator involves integrals over mesh interior faces.
   const GridFunction* u_pointer = &u;
   VectorGridFunctionCoefficient velocity(u_pointer);
   FunctionCoefficient inflow(inflow_function);
   ConstantCoefficient diff_coef(d_coef);

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);

   BilinearForm k(&fes);
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k.AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity,
                                                                    -1.0));
   k.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, -1.0));

   BilinearForm s(&fes);
   s.AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
   s.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma, kappa));
   s.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma, kappa));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, velocity, -1.0));
   //b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(U, diff_coef, sigma, kappa));

   int skip_zeros = 0;
   m.Assemble(skip_zeros);
   k.Assemble(skip_zeros);
   s.Assemble(skip_zeros);
   b.Assemble();

   m.Finalize(skip_zeros);
   k.Finalize(skip_zeros);
   s.Finalize(skip_zeros);

   // 15. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   FunctionCoefficient theta0(theta0_function);
   GridFunction theta(&fes);
   theta.ProjectCoefficient(theta0);

   // Set up visualization, if desired.
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
   IMEX_Evolution adv(m, k, s, b);

   real_t t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   int n_steps = (int)ceil(t_final / dt);
   double dt_real = t_final / n_steps;
   // Vector err_vec(n_steps-1);

   std::vector<GridFunction> theta_gf_vector;
   theta_gf_vector.push_back(theta);

   for (int ti = 0; ti < n_steps; ti++)
   {
      ode_solver->Step(theta, t, dt_real);
      theta_gf_vector.push_back(theta);
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

   // ******Backward Advection-Diffusion solve
   // 17. Define the DG finite element space on the
   //    refined mesh of the given polynomial order.
   DG_FECollection fec_adjoint(order_ad, dim);
   FiniteElementSpace fes_adjoint(&mesh, &fec_adjoint);

   // 18. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the DG discretization. The
   //    DGTraceIntegrator involves integrals over mesh interior faces.
   ConstantCoefficient zero(0.0);
   GridFunctionCoefficient theta_coeff(&(theta_gf_vector[n_steps-1]));
   FunctionCoefficient inflow_adj(inflow_function); //zero for now
   ConstantCoefficient diff_coef_adj(-d_coef);

   // FunctionCoefficient theta_exact_coeff(theta_exact);
   BilinearForm m_adj(&fes_adjoint);
   m_adj.AddDomainIntegrator(new MassIntegrator);

   BilinearForm k_adj(&fes_adjoint);
   k_adj.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k_adj.AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity,
                                                                        -1.0));
   k_adj.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity,
                                                                   -1.0));

   BilinearForm s_adj(&fes_adjoint);
   s_adj.AddDomainIntegrator(new DiffusionIntegrator(diff_coef_adj));
   s_adj.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coef_adj, sigma,
                                                             kappa));
   s_adj.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coef_adj, sigma,
                                                        kappa));

   LinearForm b_adj(&fes_adjoint);
   b_adj.AddDomainIntegrator(new DomainLFIntegrator(theta_coeff));
   //b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(zero, diff_coef, sigma, kappa));

   //int skip_zeros = 0;
   m_adj.Assemble(skip_zeros);
   m_adj.Finalize(skip_zeros);
   k_adj.Assemble(skip_zeros);
   k_adj.Finalize(skip_zeros);
   s_adj.Assemble(skip_zeros);
   s_adj.Finalize(skip_zeros);
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
   IMEX_Evolution adv_adj(m_adj, k_adj, s_adj, b_adj);

   real_t t_adj = t_final;
   adv_adj.SetTime(t_adj);
   ode_solver_adj->Init(adv_adj);

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
      theta_coeff = *(new GridFunctionCoefficient(&(theta_gf_vector[n_steps - ti -
                                                                            1])));
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

   // 21. Free the used memory.
   // delete &ode_solver;
   // delete &adv;
   // delete &adv_adj;
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

// Implementation of class IMEX_Evolution
IMEX_Evolution::IMEX_Evolution(BilinearForm &M_, BilinearForm &K_,
                               BilinearForm &S_, const Vector &b_)
   : SplitTimeDependentOperator(M_.FESpace()->GetTrueVSize()),
     M(M_), K(K_), S(S_), b(b_), z(height)
{
   Array<int> ess_tdof_list;
   if (M.GetAssemblyLevel() == AssemblyLevel::LEGACY)
   {
      M_prec = make_unique<DSmoother>(M.SpMat());
      M_solver.SetOperator(M.SpMat());
      dg_solver = make_unique<DG_Solver>(M.SpMat(), K.SpMat(), S.SpMat(),
                                         *M.FESpace());
   }
   else
   {
      M_prec = make_unique<OperatorJacobiSmoother>(M, ess_tdof_list);
      M_solver.SetOperator(M);
      dg_solver = NULL;
   }
   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void IMEX_Evolution::Mult1(const Vector &x, Vector &y) const
{
   // Perform the explicit step
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

void IMEX_Evolution::ImplicitSolve2(const real_t dt, const Vector &x, Vector &k)
{
   // Perform the implicit step
   // solve for k, k = -(M+dt S)^{-1} S x
   MFEM_VERIFY(dg_solver != NULL,
               "Implicit time integration is not supported with partial assembly");
   S.Mult(x, z);
   z*= -1.0;
   dg_solver->SetTimeStep(dt);
   dg_solver->Mult(z, k);
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
   return ( erfc(w*(x(0)-cx-rx))*erfc(-w*(x(0)-cx+rx))*erfc(w*(x(1)-cy-ry))*erfc(
               -w*(x(1)-cy+ry)) )/16;
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