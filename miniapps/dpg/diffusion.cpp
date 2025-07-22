// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//                       MFEM Ultraweak DPG example for diffusion
//
// Compile with: make diffusion
//
// sample runs
//   diffusion -m ../../data/star.mesh -o 3 -ref 1 -do 1 -prob 1 -sc
//   diffusion -m ../../data/inline-tri.mesh -o 2 -ref 2 -do 1 -prob 0
//   diffusion -m ../../data/inline-quad.mesh -o 4 -ref 1 -do 2 -prob 0 -sc
//   diffusion -m ../../data/inline-tet.mesh -o 3 -ref 0 -do 1 -prob 1 -sc

// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Poisson problem

//       - Δ u = f,   in Ω
//           u = u₀, on ∂Ω

// It solves two kinds of problems
// a) f = 1 and u₀  = 0 (like ex1)
// b) A manufactured solution problem where u_exact = sin(π * (x + y + z)).
//    This example computes and prints out convergence rates for the L2 error.

// The DPG UW deals with the First Order System
//   ∇ u - σ = 0, in Ω
// - ∇⋅σ     = f, in Ω
//        u  = u₀, in ∂Ω

// Ultraweak-DPG is obtained by integration by parts of both equations and the
// introduction of trace unknowns on the mesh skeleton
//
// u ∈ L²(Ω), σ ∈ (L²(Ω))ᵈⁱᵐ
// û ∈ H^1/2(Γₕ), σ̂ ∈ H^-1/2(Γₕ)
// -(u , ∇⋅τ) - (σ , τ) + < û, τ⋅n> = 0,      ∀ τ ∈ H(div,Ω)
//  (σ , ∇ v) + < σ̂, v  >           = (f,v)   ∀ v ∈ H¹(Ω)
//                                û = u₀      on ∂Ω

// Note:
// û := u and σ̂ := -σ on the mesh skeleton
//
// -------------------------------------------------------------
// |   |     u     |     σ     |    û      |    σ̂    |  RHS    |
// -------------------------------------------------------------
// | τ | -(u,∇⋅τ)  |  -(σ,τ)   | < û, τ⋅n> |         |    0    |
// |   |           |           |           |         |         |
// | v |           |  (σ,∇ v)  |           |  <σ̂,v>  |  (f,v)  |

// where (τ,v) ∈  H(div,Ω) × H^1(Ω)

// Here we use the "space-induced" test norm i.e.,
//
// ||(t,v)||²_H(div)×H¹ := ||t||² + ||∇⋅t||² + ||v||² + ||∇v||²

// For more information see https://doi.org/10.1007/978-3-319-01818-8_6

#include "mfem.hpp"
#include "util/weakform.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

#define USE_DIRECT_SOLVER

using namespace mfem;
using namespace mfem::common;

// Define the analytical solution and forcing terms / boundary conditions
typedef std::function<real_t(const Vector &, real_t)> TFunc;
typedef std::function<void(const Vector &, Vector &)> VecFunc;
typedef std::function<void(const Vector &, real_t, Vector &)> VecTFunc;
typedef std::function<void(const Vector &, DenseMatrix &)> MatFunc;

enum Problem
{
   Original = -1,
   SteadyDiffusion = 1,
   DiffusionRing,
   DiffusionRingGauss,
   BoundaryLayer,
   SteadyPeak,
   SteadyVaryingAngle,
   Sovinec,
   Umansky,
};

struct ProblemParams
{
   int nx, ny;
   real_t x0, y0, sx, sy;
   int order;
   real_t k, ks, ka;
   real_t t_0;
   //real_t a;
};

MatFunc GetKFun(Problem prob, const ProblemParams &params);
TFunc GetTFun(Problem prob, const ProblemParams &params);
VecTFunc GetQFun(Problem prob, const ProblemParams &params);
TFunc GetFFun(Problem prob, const ProblemParams &params);

enum prob_type
{
   manufactured,
   general
};

enum class discret_type
{
   DPG,
   RTDG,
   BRTDG,
   LDG,
};

prob_type prob;

real_t exact_u(const Vector & X);
void exact_gradu(const Vector & X, Vector &gradu);
real_t exact_laplacian_u(const Vector & X);
void exact_sigma(const Vector & X, Vector & sigma);
real_t exact_hatu(const Vector & X);
void exact_hatsigma(const Vector & X, Vector & hatsigma);
real_t f_exact(const Vector & X);

class C1Coeff : public Coefficient
{
   Mesh &mesh;
   MatrixCoefficient &eps;
public:
   C1Coeff(Mesh &m, MatrixCoefficient &eps_)
      : mesh(m), eps(eps_) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};

class C2Coeff : public VectorCoefficient
{
   Mesh &mesh;
   MatrixCoefficient &eps;
public:
   C2Coeff(Mesh &m, MatrixCoefficient &eps_)
      : VectorCoefficient(m.Dimension()), mesh(m), eps(eps_) { }

   void Eval(Vector &v, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int delta_order = 1;
   int ref = 0;
   bool visualization = true;
   int iprob = (int) prob_type::general;
   int idisc = (int) discret_type::DPG;
   int iproblem = (int) Problem::Original;
   ProblemParams pars;
   pars.nx = 0;
   pars.ny = 0;
   pars.x0 = 0.;
   pars.y0 = 0.;
   pars.sx = 1.;
   pars.sy = 1.;
   pars.order = 1;
   const int &order = pars.order;
   pars.k = 1.;
   pars.ks = 1.;
   pars.ka = 0.;
   //pars.a = 0.;
   real_t td = 0.5;
   int visport = 19916;
   bool static_cond = false;
   bool reduction = false;
   bool hybridization = false;
   bool analytic = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&pars.nx, "-nx", "--ncells-x",
                  "Number of cells in x.");
   args.AddOption(&pars.ny, "-ny", "--ncells-y",
                  "Number of cells in y.");
   args.AddOption(&pars.sx, "-sx", "--size-x",
                  "Size along x axis.");
   args.AddOption(&pars.sy, "-sy", "--size-y",
                  "Size along y axis.");
   args.AddOption(&pars.order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&ref, "-ref", "--num_refinements",
                  "Number of uniform refinements");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: manufactured, 1: general");
   args.AddOption(&idisc, "-disc", "--discretization", "Discretization"
                  " 0: DPG, 1: RTDG, 2: BRTDG, 3: LDG");
   args.AddOption(&iproblem, "-p", "--problem",
                  "Problem to solve:\n\t\t"
                  "1=sine diffusion\n\t\t"
                  "2=diffusion ring\n\t\t"
                  "3=diffusion ring - Gauss source\n\t\t"
                  "4=boundary layer\n\t\t"
                  "5=steady peak\n\t\t"
                  "6=steady varying angle\n\t\t"
                  "7=Sovinec\n\t\t");
   args.AddOption(&pars.k, "-k", "--kappa",
                  "Heat conductivity");
   args.AddOption(&pars.ks, "-ks", "--kappa_sym",
                  "Symmetric anisotropy of the heat conductivity tensor");
   args.AddOption(&pars.ka, "-ka", "--kappa_anti",
                  "Antisymmetric anisotropy of the heat conductivity tensor");
   //args.AddOption(&pars.a, "-a", "--heat_capacity",
   //               "Heat capacity coefficient (0=indefinite problem)");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&analytic, "-anal", "--analytic", "-no-anal",
                  "--no-analytic",
                  "Enable or disable analytic solution.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   if (iprob > 1) { iprob = 1; }
   prob = (prob_type)iprob;

   discret_type disc = (discret_type) idisc;
   Problem problem = (Problem)iproblem;
   if (problem != Problem::Original) { prob = prob_type::manufactured; }

   if (pars.ny <= 0)
   {
      pars.ny = pars.nx;
   }

   Mesh mesh;
   if (pars.nx <= 0)
   {
      mesh = std::move(Mesh(mesh_file, 1, 1));

      Vector x_min(2), x_max(2);
      mesh.GetBoundingBox(x_min, x_max);
      pars.x0 = x_min(0);
      pars.y0 = x_min(1);
      pars.sx = x_max(0) - x_min(0);
      pars.sy = x_max(1) - x_min(1);
   }
   else
   {
      mesh = std::move(Mesh::MakeCartesian2D(pars.nx, pars.ny,
                                             Element::QUADRILATERAL, false,
                                             pars.sx, pars.sy));
   }
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim > 1, "Dimension = 1 is not supported in this example");

   // Define spaces
   enum TrialSpace
   {
      u_space        = 0,
      sigma_space    = 1,
      hatu_space     = 2,
      hatsigma_space = 3
   };
   enum TestSpace
   {
      tau_space = 0,
      v_space   = 1
   };

   // L2 space for u
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *u_fes = new FiniteElementSpace(&mesh,u_fec);

   // Vector L2 space for σ
   FiniteElementCollection *sigma_fec{};
   FiniteElementSpace *sigma_fes{};

   switch (disc)
   {
      case discret_type::DPG:
         sigma_fec = new L2_FECollection(order-1, dim);
         sigma_fes = new FiniteElementSpace(&mesh, sigma_fec, dim);
         break;
      case discret_type::RTDG:
         sigma_fec = new RT_FECollection(order-1, dim);
         sigma_fes = new FiniteElementSpace(&mesh, sigma_fec);
         break;
      case discret_type::BRTDG:
         sigma_fec = new BrokenRT_FECollection(order-1, dim);
         sigma_fes = new FiniteElementSpace(&mesh, sigma_fec);
         break;
      case discret_type::LDG:
         sigma_fec = new L2_FECollection(order-1, dim, BasisType::GaussLobatto);
         sigma_fes = new FiniteElementSpace(&mesh,sigma_fec, dim);
         break;
      default:
         MFEM_ABORT("Unknown discretization!");
   }

   FiniteElementCollection *hatu_fec{}, *hatsigma_fec{};
   FiniteElementSpace *hatu_fes{}, *hatsigma_fes{};
   FiniteElementCollection *tau_fec{}, *v_fec{};

   if (disc == discret_type::DPG)
   {
      // H^1/2 space for û
      hatu_fec = new H1_Trace_FECollection(order,dim);
      hatu_fes = new FiniteElementSpace(&mesh,hatu_fec);

      // H^-1/2 space for σ̂
      hatsigma_fec = new RT_Trace_FECollection(order-1,dim);
      hatsigma_fes = new FiniteElementSpace(&mesh,hatsigma_fec);

      // test space fe collections
      int test_order = order+delta_order;
      tau_fec = new RT_FECollection(test_order-1, dim);
      v_fec = new H1_FECollection(test_order, dim);
   }
   else if (hybridization)
   {
      // trace space for û
      hatu_fec = new DG_Interface_FECollection(order-1, dim);
      hatu_fes = new FiniteElementSpace(&mesh, hatu_fec);
   }

   Array<FiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;

   trial_fes.Append(u_fes);
   trial_fes.Append(sigma_fes);
   if (hatu_fes) { trial_fes.Append(hatu_fes); }
   if (hatsigma_fes) { trial_fes.Append(hatsigma_fes); }
   test_fec.Append(tau_fec);
   test_fec.Append(v_fec);

   // Required coefficients for the weak formulation
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);

   pars.t_0 = 1.; //base temperature

   auto kFun = GetKFun(problem, pars);
   MatrixFunctionCoefficient eps(dim, kFun);
   InverseMatrixCoefficient eps1(eps);
   ScalarMatrixProductCoefficient negeps1(-1., eps1);
   auto fFun = GetFFun(problem, pars);
   FunctionCoefficient f(fFun); // rhs for the manufactured solution problem
   SumCoefficient negf(0., f, 1., -1.);

   C1Coeff c1_coeff(mesh, eps);
   C2Coeff c2_coeff(mesh, eps);

   // Required coefficients for the exact solution case
   auto TFun = GetTFun(problem, pars);
   FunctionCoefficient uex(TFun);
   auto QFun = GetQFun(problem, pars);
   VectorFunctionCoefficient sigmaex(dim, QFun);
   FunctionCoefficient hatuex(TFun);

   // Essential boundaries
   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   // (Bi)linear forms
   DPGWeakForm * a_dpg{};
   DarcyForm * a_darcy{};
   BilinearForm *a_sigma{}, *a_u{};
   MixedBilinearForm *a_div{};
   LinearForm *b_sigma{}, *b_u{};

   if (disc == discret_type::DPG)
   {
      // Define the DPG weak formulation
      a_dpg = new DPGWeakForm(trial_fes, test_fec);

      //  -(u,∇⋅τ)
      a_dpg->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),
                                TrialSpace::u_space,TestSpace::tau_space);

      // -(σ,τ)
      a_dpg->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(
                                                           negeps1)), TrialSpace::sigma_space, TestSpace::tau_space);

      // (σ,∇ v)
      a_dpg->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),
                                TrialSpace::sigma_space,TestSpace::v_space);

      //  <û,τ⋅n>
      a_dpg->AddTrialIntegrator(new NormalTraceIntegrator,
                                TrialSpace::hatu_space,TestSpace::tau_space);

      // -<σ̂,v> (sign is included in σ̂)
      a_dpg->AddTrialIntegrator(new TraceIntegrator,
                                TrialSpace::hatsigma_space, TestSpace::v_space);

      // test integrators (space-induced norm for H(div) × H1)
      // (∇⋅τ,∇⋅δτ)
      a_dpg->AddTestIntegrator(new DivDivIntegrator(one),
                               TestSpace::tau_space, TestSpace::tau_space);
      // c2 (τ,δτ)
      a_dpg->AddTestIntegrator(new VectorFEMassIntegrator(c2_coeff),
                               TestSpace::tau_space, TestSpace::tau_space);
      // ε (∇v,∇δv)
      a_dpg->AddTestIntegrator(new DiffusionIntegrator(eps),
                               TestSpace::v_space, TestSpace::v_space);
      // c1 (v,δv)
      a_dpg->AddTestIntegrator(new MassIntegrator(c1_coeff),
                               TestSpace::v_space, TestSpace::v_space);

      // RHS
      if (prob == prob_type::manufactured)
      {
         a_dpg->AddDomainLFIntegrator(new DomainLFIntegrator(f),TestSpace::v_space);
      }
      else
      {
         a_dpg->AddDomainLFIntegrator(new DomainLFIntegrator(one),TestSpace::v_space);
      }
   }
   else
   {
      // Define the RT/LDG formulation
      a_darcy = new DarcyForm(sigma_fes, u_fes);

      a_sigma = a_darcy->GetFluxMassForm();
      a_div = a_darcy->GetFluxDivForm();
      if (disc == discret_type::LDG)
      {
         a_u = a_darcy->GetPotentialMassForm();

         a_sigma->AddDomainIntegrator(new VectorMassIntegrator(eps1));
         a_div->AddDomainIntegrator(new VectorDivergenceIntegrator(negone));
         a_div->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                             new DGNormalTraceIntegrator(+1.)));
         a_u->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(eps, td));
      }
      else
      {
         a_sigma->AddDomainIntegrator(new VectorFEMassIntegrator(eps1));
         a_div->AddDomainIntegrator(new VectorFEDivergenceIntegrator(negone));
         if (disc == discret_type::BRTDG)
         {
            a_div->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                                new DGNormalTraceIntegrator(+1.)));
         }
      }

      //RHS
      b_sigma = a_darcy->GetFluxRHS();

      if (prob == prob_type::manufactured)
      {
         if (disc == discret_type::LDG)
         {
            b_sigma->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(uex),
                                          ess_bdr);
         }
         else if (disc == discret_type::BRTDG)
         {
            b_sigma->AddBdrFaceIntegrator(new VectorFEBoundaryFluxLFIntegrator(uex),
                                          ess_bdr);
         }
         else
         {
            b_sigma->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(uex),
                                           ess_bdr);
         }
      }

      b_u = a_darcy->GetPotentialRHS();

      if (prob == prob_type::manufactured)
      {
         b_u->AddDomainIntegrator(new DomainLFIntegrator(negf));
      }
      else
      {
         b_u->AddDomainIntegrator(new DomainLFIntegrator(negone));
      }
   }

   // GridFunction for Dirichlet bdr data
   GridFunction hatu_gf;

   // Visualization streams
   socketstream u_out, uex_out;
   socketstream sigma_out, sigmaex_out;

   if (prob == prob_type::manufactured)
   {
      std::cout << "\n  Ref |"
                << " Total dofs |"
                << " Systm dofs |"
                << "  L2 Error  |"
                << "  Rate  |"
                << "  Iters |"
                << " Totl time |"
                << " Mult time |"
                << std::endl;
      std::cout << std::string(88,'-')
                << std::endl;
   }

   real_t err0 = 0.;
   int dof0 = 0;

   //set hybridization / assembly level
   if (a_dpg && static_cond) { a_dpg->EnableStaticCondensation(); }

   StopWatch chrono_total, chrono_mult;

   for (int it = 0; it<=ref; it++)
   {
      chrono_total.Clear();
      chrono_total.Start();

      Array<int> ess_tdof_list;

      if (a_dpg)
      {
         a_dpg->Assemble();

         if (ess_bdr.Size())
         {
            hatu_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
         }

         // shift the ess_tdofs
         for (int i = 0; i < ess_tdof_list.Size(); i++)
         {
            ess_tdof_list[i] += u_fes->GetTrueVSize() + sigma_fes->GetTrueVSize();
         }
      }
      else
      {
         if (hybridization)
         {
            a_darcy->EnableHybridization(hatu_fes,
                                         new NormalTraceJumpIntegrator(-1.),
                                         ess_tdof_list);
         }
         else if (reduction)
         {
            if (disc != discret_type::BRTDG && disc != discret_type::LDG)
            {
               std::cerr << "Reduction not possible with continuous elements!" << std::endl;
               return 1;
            }
            a_darcy->EnableFluxReduction();
         }

         a_darcy->Assemble();
      }

      Array<int> offsets;
      if (a_dpg)
      {
         offsets.SetSize(5);
         offsets[0] = 0;
         offsets[1] = u_fes->GetVSize();
         offsets[2] = sigma_fes->GetVSize();
         offsets[3] = hatu_fes->GetVSize();
         offsets[4] = hatsigma_fes->GetVSize();
         offsets.PartialSum();
      }
      else
      {
         offsets.MakeRef(a_darcy->GetOffsets());
      }

      BlockVector x(offsets);
      x = 0.0;
      if (a_dpg && prob == prob_type::manufactured)
      {
         hatu_gf.MakeRef(hatu_fes,x.GetBlock(2),0);
         hatu_gf.ProjectBdrCoefficient(hatuex,ess_bdr);
      }

      OperatorPtr Ah;
      Vector X,B;

      if (a_dpg)
      {
         a_dpg->FormLinearSystem(ess_tdof_list,x,Ah,X,B);
      }
      else
      {
         a_darcy->FormLinearSystem(ess_tdof_list, x, Ah, X, B);
      }

      // Construct preconditioners

      std::unique_ptr<Solver> prec;
      std::unique_ptr<SparseMatrix> A_mono, MinvBt, S;
      Vector Md;

      if (a_dpg)
      {
         BlockMatrix * A = Ah.As<BlockMatrix>();
#if !defined(MFEM_USE_SUITESPARSE) or !defined(USE_DIRECT_SOLVER)
         auto *M = new BlockDiagonalPreconditioner(A->RowOffsets());
         M->owns_blocks = 1;
         for (int i=0; i<A->NumRowBlocks(); i++)
         {
            M->SetDiagonalBlock(i,new GSSmoother(A->GetBlock(i,i)));
         }
         prec.reset(M);
#else
         A_mono.reset(A->CreateMonolithic());
         prec.reset(new UMFPackSolver(*A_mono));
#endif
      }
      else
      {
         if (hybridization || reduction)
         {
#if !defined(MFEM_USE_SUITESPARSE) or !defined(USE_DIRECT_SOLVER)
            prec.reset(new GSSmoother(*Ah.As<SparseMatrix>()));
#else
            prec.reset(new UMFPackSolver(*Ah.As<SparseMatrix>()));
#endif
         }
         else
         {
            auto *darcy_prec = new BlockDiagonalPreconditioner(offsets);
            darcy_prec->owns_blocks = 1;

            SparseMatrix &M(a_sigma->SpMat());
            M.GetDiag(Md);
            Md.HostReadWrite();

            SparseMatrix &B(a_div->SpMat());
            MinvBt.reset(Transpose(B));

            for (int i = 0; i < Md.Size(); i++)
            {
               MinvBt->ScaleRow(i, 1./Md(i));
            }

            S.reset(Mult(B, *MinvBt));
            if (a_u)
            {
               SparseMatrix &Mtm(a_u->SpMat());
               SparseMatrix *Snew = Add(Mtm, *S);
               S.reset(Snew);
            }

            darcy_prec->SetDiagonalBlock(0, new DSmoother(M));
            darcy_prec->SetDiagonalBlock(1, new GSSmoother(*S));

            prec.reset(darcy_prec);
         }
      }

      // Construct solver

      std::unique_ptr<IterativeSolver> solver;
#if !defined(MFEM_USE_SUITESPARSE) or !defined(USE_DIRECT_SOLVER)
      if (a_dpg)
      {
         solver.reset(new CGSolver());
      }
      else
#endif
      {
         solver.reset(new GMRESSolver());
      }

      solver->SetRelTol(1e-10);
      solver->SetMaxIter(20000);
      solver->SetPrintLevel(prob== prob_type::general ? 3 : 0);
      solver->SetOperator(*Ah);
      solver->SetPreconditioner(*prec);

      chrono_mult.Clear();
      chrono_mult.Start();

      solver->Mult(B, X);

      chrono_mult.Stop();

      if (a_dpg)
      {
         a_dpg->RecoverFEMSolution(X,x);
      }
      else
      {
         a_darcy->RecoverFEMSolution(X, x);
      }

      GridFunction u_gf, sigma_gf;
      if (a_dpg)
      {
         u_gf.MakeRef(u_fes,x.GetBlock(0),0);
         sigma_gf.MakeRef(sigma_fes,x.GetBlock(1),0);
      }
      else
      {
         sigma_gf.MakeRef(sigma_fes,x.GetBlock(0),0);
         u_gf.MakeRef(u_fes,x.GetBlock(1),0);
      }

      chrono_total.Stop();

      if (prob == prob_type::manufactured)
      {
         const int l2dofs = u_fes->GetVSize() + sigma_fes->GetVSize();
         const int sysdofs = Ah->Height();
         real_t u_err = u_gf.ComputeL2Error(uex);
         real_t sigma_err = sigma_gf.ComputeL2Error(sigmaex);
         real_t L2Error = sqrt(u_err*u_err + sigma_err*sigma_err);
         real_t rate_err = (it) ? dim*log(err0/L2Error)/log((real_t)dof0/l2dofs) : 0.0;
         err0 = L2Error;
         dof0 = l2dofs;

         std::ios oldState(nullptr);
         oldState.copyfmt(std::cout);
         std::cout << std::right << std::setw(5) << it << " | "
                   << std::setw(10) <<  l2dofs << " | "
                   << std::setw(10) <<  sysdofs << " | "
                   << std::setprecision(3)
                   << std::setw(10) << std::scientific <<  err0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_err << " | "
                   << std::setw(6) << std::fixed << solver->GetNumIterations() << " | "
                   << std::setw(9) << std::scientific << chrono_total.RealTime() << " | "
                   << std::setw(9) << std::scientific << chrono_mult.RealTime() << " | "
                   << std::endl;
         std::cout.copyfmt(oldState);
      }

      if (visualization)
      {
         const char * keys = (it == 0 && dim == 2) ? "jlRcm\n" : nullptr;
         char vishost[] = "localhost";
         VisualizeField(u_out,vishost, visport, u_gf,
                        "Numerical u", 0,0, 500, 500, keys);
         VisualizeField(sigma_out,vishost, visport, sigma_gf,
                        "Numerical flux", 500,0,500, 500, keys);
         if (analytic)
         {
            GridFunction uex_gf(u_fes);
            uex_gf.ProjectCoefficient(uex);
            VisualizeField(uex_out, vishost, visport, uex_gf,
                           "Numerical u (analytic)", 1000,0, 500, 500, keys);

            GridFunction sigmaex_gf(sigma_fes);
            sigmaex_gf.ProjectCoefficient(sigmaex);
            VisualizeField(sigmaex_out, vishost, visport, sigmaex_gf,
                           "Numerical flux (analytic)", 1500,0,500, 500, keys);
         }
      }

      if (it == ref) { break; }

      mesh.UniformRefinement();
      pars.nx *= 2;
      pars.ny *= 2;
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      if (a_dpg)
      {
         a_dpg->Update();
      }
      else
      {
         a_darcy->Update();
      }
   }

   delete a_dpg;
   delete a_darcy;
   delete tau_fec;
   delete v_fec;
   delete hatsigma_fes;
   delete hatsigma_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete sigma_fec;
   delete sigma_fes;
   delete u_fec;
   delete u_fes;

   return 0;
}

real_t exact_u(const Vector & X)
{
   real_t alpha = M_PI * (X.Sum());
   return sin(alpha);
}

void exact_gradu(const Vector & X, Vector & du)
{
   du.SetSize(X.Size());
   real_t alpha = M_PI * (X.Sum());
   du.SetSize(X.Size());
   for (int i = 0; i<du.Size(); i++)
   {
      du[i] = M_PI * cos(alpha);
   }
}

real_t exact_laplacian_u(const Vector & X)
{
   real_t alpha = M_PI * (X.Sum());
   real_t u = sin(alpha);
   return - M_PI*M_PI * u * X.Size();
}

void exact_sigma(const Vector & X, Vector & sigma)
{
   // σ = ∇ u
   exact_gradu(X,sigma);
}

real_t exact_hatu(const Vector & X)
{
   return exact_u(X);
}

void exact_hatsigma(const Vector & X, Vector & hatsigma)
{
   exact_sigma(X,hatsigma);
   hatsigma *= -1.;
}

real_t f_exact(const Vector & X)
{
   return -exact_laplacian_u(X);
}

MatFunc GetKFun(Problem prob, const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   const real_t &ka = params.ka;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   switch (prob)
   {
      case Problem::Original:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            kappa.Diag(k, ndim);
         };
      case Problem::SteadyDiffusion:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            kappa.Diag(k, ndim);
            kappa(0,0) *= ks;
            kappa(0,1) = +ka * k;
            kappa(1,0) = -ka * k;
            if (ndim > 2)
            {
               kappa(0,2) = +ka * k;
               kappa(2,0) = -ka * k;
            }
         };
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::SteadyVaryingAngle:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            b = 0.;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            b(0) = (r>0.)?(-dx(1) / r):(1.);
            b(1) = (r>0.)?(+dx(0) / r):(0.);

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
      case Problem::Sovinec:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            b = 0.;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            //const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            const real_t psi_x = M_PI * sin(M_PI * dx(0)) * cos(M_PI * dx(1));
            const real_t psi_y = M_PI * cos(M_PI * dx(0)) * sin(M_PI * dx(1));
            const real_t psi_norm = hypot(psi_x, psi_y);
            if (psi_norm > 0.)
            {
               b(0) = -psi_y / psi_norm;
               b(1) = +psi_x / psi_norm;
            }
            else
            {
               b = 0.;
            }

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
      case Problem::Umansky:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            const real_t s = hypot(sx, sy);
            b(0) = +sx / s;
            b(1) = +sy / s;

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
   }
   return MatFunc();
}

TFunc GetTFun(Problem prob, const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &t_0 = params.t_0;
   //const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;
   const real_t &order = params.order;

   auto kFun = GetKFun(prob, params);

   switch (prob)
   {
      case Problem::Original:
         return [=](const Vector &x, real_t t) -> real_t
         {
            return exact_u(x);
         };
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const int ndim = x.Size();
            real_t t0 = t_0 * sin(M_PI*x(0)) * sin(M_PI*x(1));
            if (ndim > 2)
            {
               t0 *= sin(M_PI*x(2));
            }

            return t0;

            /*if (a <= 0.) { return t0; }

            Vector ddT((ndim<=2)?(2):(4));
            ddT(0) = -t_0 * M_PI*M_PI * sin(M_PI*x(0)) * sin(M_PI*x(1));//xx,yy
            ddT(1) = +t_0 * M_PI*M_PI * cos(M_PI*x(0)) * cos(M_PI*x(1));//xy
            if (ndim > 2)
            {
               ddT(0) *= sin(M_PI*x(2));//xx,yy,zz
               ddT(1) *= sin(M_PI*x(2));//xy
               //xz
               ddT(2) = +t_0 * M_PI*M_PI * cos(M_PI*x(0)) * sin(M_PI*x(1)) * cos(M_PI*x(2));
               //yz
               ddT(3) = +t_0 * M_PI*M_PI * sin(M_PI*x(0)) * cos(M_PI*x(1)) * cos(M_PI*x(2));

            }

            DenseMatrix kappa;
            kFun(x, kappa);

            real_t div = -(kappa(0,0) + kappa(1,1)) * ddT(0) - (kappa(0,1) + kappa(1,0)) * ddT(1);
            if (ndim > 2)
            {
               div += -kappa(2,2) * ddT(0) - (kappa(0,2) + kappa(2,0)) * ddT(2)
               - (kappa(1,2) + kappa(2,1)) * ddT(3);
            }
            return t0 - div / a * t;*/
         };
      case Problem::DiffusionRing:
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.25;
            constexpr real_t r1 = 0.35;
            constexpr real_t dr01 = 0.025;
            constexpr real_t theta0 = 11./12. * M_PI;
            constexpr real_t dtheta0 = 1./48. * M_PI;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t theta = fabs(atan2(dx(1), dx(0)));

            if (r < r0 - dr01 || r > r1 + dr01 || theta < theta0 - dtheta0)
            {
               return 0.;
            }

            const real_t dr = std::min(r - r0 + dr01, r1 + dr01 - r) / dr01;
            const real_t dth = (theta - theta0 + dtheta0) / dtheta0;
            return std::min(1., dr) * std::min(1., dth) * t_0;
         };
      case Problem::DiffusionRingGauss:
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.025;
            constexpr real_t x0 = 0.15;

            const real_t dx_l = x(0) - x0;
            const real_t dx_r = x(0) - (1. - x0);
            const real_t dy = x(1) - 0.5;
            const real_t r_l = hypot(dx_l, dy);
            const real_t r_r = hypot(dx_r, dy);

            return - exp(- r_l*r_l/(r0*r0)) + exp(- r_r*r_r/(r0*r0));
         };
      case Problem::BoundaryLayer:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers andMathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t k_para = M_PI*M_PI * k * ks;
            const real_t k_perp = k;
            const real_t k_frac = sqrt(k_para/k_perp);
            const real_t denom = 1. + exp(-k_frac);
            const real_t e_down = exp(-k_frac * x(1));
            const real_t e_up = exp(- k_frac * (1. - x(1)));
            return - (e_down + e_up) / denom * sin(M_PI * x(0));
         };
      case Problem::SteadyPeak:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATIONMETHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            return x(0)*x(1) * pow(arg, s);
         };
      case Problem::SteadyVaryingAngle:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATIONMETHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t t) -> real_t
         {
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            return 1. - r*r*r;
         };
      case Problem::Sovinec:
         // C. R. Sovinec et al., Nonlinear magnetohydrodynamics simulation
         // using high-order finite elements. Journal of Computational Physics,
         // 195, pp. 355–386 (2004).
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t &kappa_perp = k * ks;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            return psi / kappa_perp;
         };
      case Problem::Umansky:
         // M. V. Umansky, M. S. Day and T. D. Rognlien, On Numerical Solution
         // of Strongly Anisotropic Diffusion Equation on Misaligned Grids,
         // Numerical Heat Transfer, Part B: Fundamentals, 47(6), pp. 533-554
         // (2005).
         // Adopted from plasma-dev:miniapps/plasma/transport2d.cpp
         return [=,&params](const Vector &x, real_t t) -> real_t
         {
            const real_t hx = sx / params.nx;
            const real_t hy = sy / params.ny;
            if (x(0) < hx && x(1) < hy)
            {
               return 0.5 * (1.0 -
                             std::pow(1.0 - x(0) / hx, order) +
                             std::pow(1.0 - x(1) / hy, order));
            }
            else if (x(0) > sx - hx && x(1) > sy - hy)
            {
               return 0.5 * (1.0 +
                             std::pow(1.0 - (sx - x(0)) / hx, order) -
                             std::pow(1.0 - (sy - x(1)) / hy, order));
            }
            // else if (x_[0] > Lx_ - hx_ || x_[1] < hy_)
            else if (hx * (x(1) + hy) < hy * x(0))
            {
               return 1.0;
            }
            // else if (x_[0] < hx_ || x_[1] > Ly_ - hy_)
            else if (hx * x(1) > hy * (x(0) + hx))
            {
               return 0.0;
            }
            else
            {
               return 0.5 * (1.0 + std::tanh(M_LN10 * (x(0) / hx - x(1) / hy)));
            }
         };
   }
   return TFunc();
}

VecTFunc GetQFun(Problem prob, const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &t_0 = params.t_0;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   auto kFun = GetKFun(prob, params);

   switch (prob)
   {
      case Problem::Original:
         return [=](const Vector &x, real_t, Vector &v)
         {
            exact_sigma(x, v);
         };
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            Vector gT(vdim);
            gT = 0.;
            gT(0) = t_0 * M_PI * cos(M_PI*x(0)) * sin(M_PI*x(1));
            gT(1) = t_0 * M_PI * sin(M_PI*x(0)) * cos(M_PI*x(1));
            if (vdim > 2)
            {
               gT(0) *= sin(M_PI*x(2));
               gT(1) *= sin(M_PI*x(2));
               gT(2) = t_0 * M_PI * sin(M_PI*x(0)) * sin(M_PI*x(1)) * cos(M_PI*x(2));
            }

            DenseMatrix kappa;
            kFun(x, kappa);

            if (vdim <= 2)
            {
               v(0) = -kappa(0,0) * gT(0) -kappa(0,1) * gT(1);
               v(1) = -kappa(1,0) * gT(0) -kappa(1,1) * gT(1);
            }
            else
            {
               kappa.Mult(gT, v);
               v.Neg();
            }
            v.Neg();
         };
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::Umansky:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            v = 0.;
         };
      case Problem::BoundaryLayer:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            DenseMatrix kappa;
            kFun(x, kappa);
            const real_t k_para = M_PI*M_PI * kappa(0,0);
            const real_t k_perp = kappa(1,1);
            const real_t k_frac = sqrt(k_para/k_perp);

            const real_t denom = 1. + exp(-k_frac);
            const real_t e_down = exp(-k_frac * x(1));
            const real_t e_up = exp(- k_frac * (1. - x(1)));
            const real_t T_x = - (e_down + e_up) / denom * M_PI * cos(M_PI * x(0));
            const real_t T_y = k_frac * (e_down - e_up) / denom * sin(M_PI * x(0));
            v(0) = +kappa(0,0) * T_x;
            v(1) = +kappa(1,1) * T_y;
         };
      case Problem::SteadyPeak:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            DenseMatrix kappa;
            kFun(x, kappa);
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_x = M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_y = M_PI * cos(M_PI * x(1)) * sin(M_PI * x(0));
            const real_t T_x = x(1) * pow(arg, s-1) * (arg + x(0) * s * arg_x);
            const real_t T_y = x(0) * pow(arg, s-1) * (arg + x(1) * s * arg_y);
            v(0) = +kappa(0,0) * T_x;
            v(1) = +kappa(1,1) * T_y;
         };
      case Problem::SteadyVaryingAngle:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            const real_t kappa_r = k * ks;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t T_r = - 3. * r;
            v(0) = +kappa_r * T_r * dx(0);
            v(1) = +kappa_r * T_r * dx(1);
         };
      case Problem::Sovinec:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            v(0) = -M_PI * sin(M_PI * dx(0)) * cos(M_PI * dx(1));
            v(1) = -M_PI * cos(M_PI * dx(0)) * sin(M_PI * dx(1));
         };
   }
   return VecTFunc();
}

TFunc GetFFun(Problem prob, const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   //const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   auto TFun = GetTFun(prob, params);
   auto kFun = GetKFun(prob, params);

   switch (prob)
   {
      case Problem::Original:
         return [=](const Vector &x, real_t) -> real_t
         {
            return f_exact(x);
         };
      case Problem::SteadyDiffusion:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t T = TFun(x, 0);
            //return -((a > 0.)?(a):(1.)) * T;
            return T;
         };
      case Problem::BoundaryLayer:
      case Problem::Umansky:
         return [=](const Vector &x, real_t) -> real_t
         {
            return 0.;
         };
      case Problem::SteadyPeak:
         return [=](const Vector &x, real_t) -> real_t
         {
            DenseMatrix kappa;
            kFun(x, kappa);
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_x = M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_y = M_PI * cos(M_PI * x(1)) * sin(M_PI * x(0));
            const real_t T_xx = x(1) * pow(arg, s-2) * (2.*s * arg_x * arg + x(0) * s * ((s-1) * arg_x*arg_x - M_PI*M_PI * arg*arg));
            const real_t T_yy = x(0) * pow(arg, s-2) * (2.*s * arg_y * arg + x(1) * s * ((s-1) * arg_y*arg_y - M_PI*M_PI * arg*arg));
            return -(kappa(0,0) * T_xx + kappa(1,1) * T_yy);
         };
      case Problem::SteadyVaryingAngle:
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t kappa_r = ks * k;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t T_rr = - 9. * r;
            return -kappa_r * T_rr;
         };
      case Problem::Sovinec:
         return [=](const Vector &x, real_t) -> real_t
         {
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            return 2.*M_PI*M_PI * psi;
         };
   }
   return TFunc();
}

real_t C1Coeff::Eval(ElementTransformation &Tr, const IntegrationPoint &ip)
{
   DenseMatrix eps_i;
   Vector eps_id;

   real_t volume = mesh.GetElementVolume(Tr.ElementNo);
   eps.Eval(eps_i, Tr, ip);
   eps_i.GetDiag(eps_id);
   real_t c1 = std::min(eps_id.Min()/volume, (real_t) 1.);

   return c1;
}

void C2Coeff::Eval(Vector &v, ElementTransformation &Tr,
                   const IntegrationPoint &ip)
{
   v.SetSize(vdim);

   DenseMatrix eps_i;
   Vector eps_id;

   real_t volume = mesh.GetElementVolume(Tr.ElementNo);
   eps.Eval(eps_i, Tr, ip);
   eps_i.GetDiag(eps_id);
   v(0) = std::min(1./eps_id(0), 1./volume);
   v(1) = std::min(1./eps_id(1), 1./volume);
}
