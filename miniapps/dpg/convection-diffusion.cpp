// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
//              MFEM Ultraweak DPG example for convection-diffusion
//
// Compile with: make convection-diffusion
//
// sample runs
//  convection-diffusion -m ../../data/star.mesh -o 2 -ref 2 -theta 0.0 -eps 1e-1 -beta '2 3'
//  convection-diffusion -m ../../data/beam-hex.mesh -o 2 -ref 2 -theta 0.0 -eps 1e0 -beta '1 0 2'
//  convection-diffusion -m ../../data/inline-tri.mesh -o 3 -ref 2 -theta 0.0 -eps 1e-2 -beta '4 2' -sc

// AMR runs
//  convection-diffusion -o 3 -ref 5 -prob 1 -eps 1e-1 -theta 0.75
//  convection-diffusion -o 2 -ref 9 -prob 1 -eps 1e-2 -theta 0.75
//  convection-diffusion -o 3 -ref 9 -prob 1 -eps 1e-3 -theta 0.75 -sc

// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the convection-diffusion problem

//     - εΔu + ∇⋅(βu) = f,   in Ω
//                  u = u_0, on ∂Ω

// It solves the following kinds of problems
// (a) A manufactured solution where u_exact = sin(π * (x + y + z)).
// (b) The 2D Erickson-Johnson problem

// The DPG UW deals with the First Order System
//     - ∇⋅σ + ∇⋅(βu) = f,   in Ω
//        1/ε σ - ∇u  = 0,   in Ω
//                  u = u_0, on ∂Ω

// Ultraweak-DPG is obtained by integration by parts of both equations and the
// introduction of trace unknowns on the mesh skeleton
//
// u ∈ L²(Ω), σ ∈ (L²(Ω))ᵈⁱᵐ
// û ∈ H^1/2, σ̂ ∈ H^-1/2
// -(βu , ∇v)  + (σ , ∇v)     + < f̂ ,  v  > = (f,v),   ∀ v ∈ H¹(Ω)
//   (u , ∇⋅τ) + 1/ε (σ , τ)  + < û , τ⋅n > = 0,       ∀ τ ∈ H(div,Ω)
//                                        û = u_0  on ∂Ω

// Note:
// f̂ := βu - σ, û := -u on the mesh skeleton

// -------------------------------------------------------------
// |   |     u     |     σ     |   û       |     f̂    |  RHS    |
// -------------------------------------------------------------
// | v |-(βu , ∇v) | (σ , ∇v)  |           | < f̂ ,v > |  (f,v)  |
// |   |           |           |           |          |         |
// | τ | (u ,∇⋅τ)  | 1/ε(σ , τ)|  <û,τ⋅n>  |          |    0    |

// where (v,τ) ∈  H¹(Ωₕ) × H(div,Ωₕ)

// For more information see https://doi.org/10.1016/j.camwa.2013.06.010

#include "mfem.hpp"
#include "util/weakform.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>


using namespace mfem;
using namespace mfem::common;

enum prob_type
{
   manufactured,
   EJ // see https://doi.org/10.1016/j.camwa.2013.06.010
};

prob_type prob;
Vector beta;
real_t epsilon;

real_t exact_u(const Vector & X);
void exact_gradu(const Vector & X, Vector & du);
real_t exact_laplacian_u(const Vector & X);
void exact_sigma(const Vector & X, Vector & sigma);
real_t exact_hatu(const Vector & X);
void exact_hatf(const Vector & X, Vector & hatf);
real_t f_exact(const Vector & X);
void setup_test_norm_coeffs(GridFunction & c1_gf, GridFunction & c2_gf);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   int ref = 1;
   bool visualization = true;
   int iprob = 0;
   real_t theta = 0.0;
   bool static_cond = false;
   int visport = 19916;
   epsilon = 1e0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&delta_order, "-do", "--delta-order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&epsilon, "-eps", "--epsilon",
                  "Epsilon coefficient");
   args.AddOption(&ref, "-ref", "--num-refinements",
                  "Number of uniform refinements");
   args.AddOption(&theta, "-theta", "--theta",
                  "Theta parameter for AMR");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: manufactured, 1: Erickson-Johnson ");
   args.AddOption(&beta, "-beta", "--beta",
                  "Vector Coefficient beta");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }

   if (iprob > 1) { iprob = 1; }
   prob = (prob_type)iprob;

   if (prob == prob_type::EJ)
   {
      mesh_file = "../../data/inline-quad.mesh";
   }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   MFEM_VERIFY(dim > 1, "Dimension = 1 is not supported in this example");

   if (beta.Size() == 0)
   {
      beta.SetSize(dim);
      beta = 0.0;
      beta[0] = 1.;
   }

   args.PrintOptions(std::cout);

   // Define spaces
   enum TrialSpace
   {
      u_space     = 0,
      sigma_space = 1,
      hatu_space  = 2,
      hatf_space  = 3
   };
   enum TestSpace
   {
      v_space   = 0,
      tau_space = 1
   };
   // L2 space for u
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *u_fes = new FiniteElementSpace(&mesh,u_fec);

   // Vector L2 space for σ
   FiniteElementCollection *sigma_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *sigma_fes = new FiniteElementSpace(&mesh,sigma_fec, dim);

   // H^1/2 space for û
   FiniteElementCollection * hatu_fec = new H1_Trace_FECollection(order,dim);
   FiniteElementSpace *hatu_fes = new FiniteElementSpace(&mesh,hatu_fec);

   // H^-1/2 space for σ̂
   FiniteElementCollection * hatf_fec = new RT_Trace_FECollection(order-1,dim);
   FiniteElementSpace *hatf_fes = new FiniteElementSpace(&mesh,hatf_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * v_fec = new H1_FECollection(test_order, dim);
   FiniteElementCollection * tau_fec = new RT_FECollection(test_order-1, dim);

   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient eps(epsilon);
   ConstantCoefficient eps1(1./epsilon);
   ConstantCoefficient negeps1(-1./epsilon);
   ConstantCoefficient eps2(1/(epsilon*epsilon));

   ConstantCoefficient negeps(-epsilon);
   VectorConstantCoefficient betacoeff(beta);
   Vector negbeta = beta; negbeta.Neg();
   DenseMatrix bbt(beta.Size());
   MultVVt(beta, bbt);
   MatrixConstantCoefficient bbtcoeff(bbt);
   VectorConstantCoefficient negbetacoeff(negbeta);

   Array<FiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;

   trial_fes.Append(u_fes);
   trial_fes.Append(sigma_fes);
   trial_fes.Append(hatu_fes);
   trial_fes.Append(hatf_fes);
   test_fec.Append(v_fec);
   test_fec.Append(tau_fec);

   FiniteElementCollection *coeff_fec = new L2_FECollection(0,dim);
   FiniteElementSpace *coeff_fes = new FiniteElementSpace(&mesh,coeff_fec);
   GridFunction c1_gf, c2_gf;
   GridFunctionCoefficient c1_coeff(&c1_gf);
   GridFunctionCoefficient c2_coeff(&c2_gf);

   DPGWeakForm * a = new DPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices(true); // needed for residual calculation

   //-(βu , ∇v)
   a->AddTrialIntegrator(new MixedScalarWeakDivergenceIntegrator(betacoeff),
                         TrialSpace::u_space, TestSpace::v_space);

   // (σ,∇ v)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),
                         TrialSpace::sigma_space, TestSpace::v_space);

   // (u ,∇⋅τ)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(negone),
                         TrialSpace::u_space, TestSpace::tau_space);

   // 1/ε (σ,τ)
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(eps1)),
                         TrialSpace::sigma_space, TestSpace::tau_space);

   //  <û,τ⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,
                         TrialSpace::hatu_space, TestSpace::tau_space);

   // <f̂ ,v>
   a->AddTrialIntegrator(new TraceIntegrator,
                         TrialSpace::hatf_space, TestSpace::v_space);

   // mesh dependent test norm
   c1_gf.SetSpace(coeff_fes);
   c2_gf.SetSpace(coeff_fes);
   setup_test_norm_coeffs(c1_gf,c2_gf);

   // c1 (v,δv)
   a->AddTestIntegrator(new MassIntegrator(c1_coeff),
                        TestSpace::v_space, TestSpace::v_space);
   // ε (∇v,∇δv)
   a->AddTestIntegrator(new DiffusionIntegrator(eps),
                        TestSpace::v_space, TestSpace::v_space);
   // (β⋅∇v, β⋅∇δv)
   a->AddTestIntegrator(new DiffusionIntegrator(bbtcoeff),
                        TestSpace::v_space, TestSpace::v_space);
   // c2 (τ,δτ)
   a->AddTestIntegrator(new VectorFEMassIntegrator(c2_coeff),
                        TestSpace::tau_space, TestSpace::tau_space);
   // (∇⋅τ,∇⋅δτ)
   a->AddTestIntegrator(new DivDivIntegrator(one),
                        TestSpace::tau_space, TestSpace::tau_space);

   FunctionCoefficient f(f_exact);
   a->AddDomainLFIntegrator(new DomainLFIntegrator(f),TestSpace::v_space);

   FunctionCoefficient hatuex(exact_hatu);
   VectorFunctionCoefficient hatfex(dim,exact_hatf);
   Array<int> elements_to_refine;
   FunctionCoefficient uex(exact_u);
   VectorFunctionCoefficient sigmaex(dim,exact_sigma);

   GridFunction hatu_gf, hatf_gf;

   socketstream u_out;
   socketstream sigma_out;

   real_t res0 = 0.;
   real_t err0 = 0.;
   int dof0 = 0; // init to suppress gcc warning
   std::cout << "\n  Ref |"
             << "    Dofs    |"
             << "  L2 Error  |"
             << "  Rate  |"
             << "  Residual  |"
             << "  Rate  |" << std::endl;
   std::cout << std::string(64,'-') << std::endl;

   if (static_cond) { a->EnableStaticCondensation(); }
   for (int it = 0; it<=ref; it++)
   {
      a->Assemble();

      Array<int> ess_tdof_list_uhat;
      Array<int> ess_tdof_list_fhat;
      Array<int> ess_bdr_uhat;
      Array<int> ess_bdr_fhat;
      if (mesh.bdr_attributes.Size())
      {
         ess_bdr_uhat.SetSize(mesh.bdr_attributes.Max());
         ess_bdr_fhat.SetSize(mesh.bdr_attributes.Max());
         ess_bdr_uhat = 1; ess_bdr_fhat = 0;
         if (prob == prob_type::EJ)
         {
            ess_bdr_uhat = 0;
            ess_bdr_fhat = 1;
            ess_bdr_uhat[1] = 1;
            ess_bdr_fhat[1] = 0;
         }
         hatu_fes->GetEssentialTrueDofs(ess_bdr_uhat, ess_tdof_list_uhat);
         hatf_fes->GetEssentialTrueDofs(ess_bdr_fhat, ess_tdof_list_fhat);
      }

      // shift the ess_tdofs
      int n = ess_tdof_list_uhat.Size();
      int m = ess_tdof_list_fhat.Size();
      Array<int> ess_tdof_list(n+m);
      for (int j = 0; j < n; j++)
      {
         ess_tdof_list[j] = ess_tdof_list_uhat[j]
                            + u_fes->GetTrueVSize()
                            + sigma_fes->GetTrueVSize();
      }
      for (int j = 0; j < m; j++)
      {
         ess_tdof_list[j+n] = ess_tdof_list_fhat[j]
                              + u_fes->GetTrueVSize()
                              + sigma_fes->GetTrueVSize()
                              + hatu_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         offsets[i+1] = trial_fes[i]->GetVSize();
         dofs += trial_fes[i]->GetTrueVSize();
      }
      offsets.PartialSum();

      BlockVector x(offsets); x = 0.0;
      hatu_gf.MakeRef(hatu_fes,x.GetBlock(2),0);
      hatf_gf.MakeRef(hatf_fes,x.GetBlock(3),0);
      hatu_gf.ProjectBdrCoefficient(hatuex,ess_bdr_uhat);
      hatf_gf.ProjectBdrCoefficientNormal(hatfex,ess_bdr_fhat);

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);

      BlockMatrix * A = Ah.As<BlockMatrix>();

      BlockDiagonalPreconditioner M(A->RowOffsets());
      M.owns_blocks = 1;
      for (int i = 0 ; i < A->NumRowBlocks(); i++)
      {
         M.SetDiagonalBlock(i,new DSmoother(A->GetBlock(i,i)));
      }

      CGSolver cg;
      cg.SetRelTol(1e-8);
      cg.SetMaxIter(20000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(M);
      cg.SetOperator(*A);
      cg.Mult(B, X);

      a->RecoverFEMSolution(X,x);

      GridFunction u_gf, sigma_gf;
      u_gf.MakeRef(u_fes,x.GetBlock(0),0);
      sigma_gf.MakeRef(sigma_fes,x.GetBlock(1),0);

      real_t u_err = u_gf.ComputeL2Error(uex);
      real_t sigma_err = sigma_gf.ComputeL2Error(sigmaex);
      real_t L2Error = sqrt(u_err*u_err + sigma_err*sigma_err);

      Vector & residuals = a->ComputeResidual(x);
      real_t residual = residuals.Norml2();

      real_t rate_err = (it) ? dim*log(err0/L2Error)/log((real_t)dof0/dofs) : 0.0;
      real_t rate_res = (it) ? dim*log(res0/residual)/log((real_t)dof0/dofs) : 0.0;

      err0 = L2Error;
      res0 = residual;
      dof0 = dofs;

      std::ios oldState(nullptr);
      oldState.copyfmt(std::cout);
      std::cout << std::right << std::setw(5) << it << " | "
                << std::setw(10) <<  dof0 << " | "
                << std::setprecision(3)
                << std::setw(10) << std::scientific <<  err0 << " | "
                << std::setprecision(2)
                << std::setw(6) << std::fixed << rate_err << " | "
                << std::setprecision(3)
                << std::setw(10) << std::scientific <<  res0 << " | "
                << std::setprecision(2)
                << std::setw(6) << std::fixed << rate_res << " | "
                << std::endl;
      std::cout.copyfmt(oldState);

      if (visualization)
      {
         const char * keys = (it == 0 && dim == 2) ? "jRcm\n" : nullptr;
         char vishost[] = "localhost";
         VisualizeField(u_out,vishost, visport, u_gf,
                        "Numerical u", 0,0, 500, 500, keys);
         VisualizeField(sigma_out,vishost, visport, sigma_gf,
                        "Numerical flux", 501,0,500, 500, keys);
      }

      if (it == ref)
      {
         break;
      }

      elements_to_refine.SetSize(0);
      real_t max_resid = residuals.Max();
      for (int iel = 0; iel<mesh.GetNE(); iel++)
      {
         if (residuals[iel] > theta * max_resid)
         {
            elements_to_refine.Append(iel);
         }
      }

      mesh.GeneralRefinement(elements_to_refine,1,1);
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();

      coeff_fes->Update();
      c1_gf.Update();
      c2_gf.Update();
      setup_test_norm_coeffs(c1_gf,c2_gf);
   }

   delete coeff_fes;
   delete coeff_fec;
   delete a;
   delete tau_fec;
   delete v_fec;
   delete hatf_fes;
   delete hatf_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete sigma_fes;
   delete sigma_fec;
   delete u_fec;
   delete u_fes;

   return 0;
}

real_t exact_u(const Vector & X)
{
   real_t x = X[0];
   real_t y = X[1];
   real_t z = 0.;
   if (X.Size() == 3) { z = X[2]; }
   switch (prob)
   {
      case EJ:
      {
         real_t alpha = sqrt(1. + 4. * epsilon * epsilon * M_PI * M_PI);
         real_t r1 = (1. + alpha) / (2.*epsilon);
         real_t r2 = (1. - alpha) / (2.*epsilon);
         real_t denom = exp(-r2) - exp(-r1);

         real_t g1 = exp(r2*(x-1.));
         real_t g2 = exp(r1*(x-1.));
         real_t g = g1-g2;

         return g * cos(M_PI * y)/denom;
      }
      break;
      default:
      {
         real_t alpha = M_PI * (x + y + z);
         return sin(alpha);
      }
      break;
   }
}

void exact_gradu(const Vector & X, Vector & du)
{
   real_t x = X[0];
   real_t y = X[1];
   real_t z = 0.;
   if (X.Size() == 3) { z = X[2]; }
   du.SetSize(X.Size());
   du = 0.;
   switch (prob)
   {
      case EJ:
      {
         real_t alpha = sqrt(1. + 4. * epsilon * epsilon * M_PI * M_PI);
         real_t r1 = (1. + alpha) / (2.*epsilon);
         real_t r2 = (1. - alpha) / (2.*epsilon);
         real_t denom = exp(-r2) - exp(-r1);

         real_t g1 = exp(r2*(x-1.));
         real_t g1_x = r2*g1;
         real_t g2 = exp(r1*(x-1.));
         real_t g2_x = r1*g2;
         real_t g = g1-g2;
         real_t g_x = g1_x - g2_x;

         du[0] = g_x * cos(M_PI * y)/denom;
         du[1] = -M_PI * g * sin(M_PI*y)/denom;
      }
      break;
      default:
      {
         real_t alpha = M_PI * (x + y + z);
         du.SetSize(X.Size());
         for (int i = 0; i<du.Size(); i++)
         {
            du[i] = M_PI * cos(alpha);
         }
      }
      break;
   }
}

real_t exact_laplacian_u(const Vector & X)
{
   real_t x = X[0];
   real_t y = X[1];
   real_t z = 0.;
   if (X.Size() == 3) { z = X[2]; }

   switch (prob)
   {
      case EJ:
      {
         real_t alpha = sqrt(1. + 4. * epsilon * epsilon * M_PI * M_PI);
         real_t r1 = (1. + alpha) / (2.*epsilon);
         real_t r2 = (1. - alpha) / (2.*epsilon);
         real_t denom = exp(-r2) - exp(-r1);

         real_t g1 = exp(r2*(x-1.));
         real_t g1_x = r2*g1;
         real_t g1_xx = r2*g1_x;
         real_t g2 = exp(r1*(x-1.));
         real_t g2_x = r1*g2;
         real_t g2_xx = r1*g2_x;
         real_t g = g1-g2;
         real_t g_xx = g1_xx - g2_xx;

         real_t u = g * cos(M_PI * y)/denom;
         real_t u_xx = g_xx * cos(M_PI * y)/denom;
         real_t u_yy = -M_PI * M_PI * u;
         return u_xx + u_yy;
      }
      break;
      default:
      {
         real_t alpha = M_PI * (x + y + z);
         real_t u = sin(alpha);
         return -M_PI*M_PI * u * X.Size();
      }
      break;
   }
}

void exact_sigma(const Vector & X, Vector & sigma)
{
   // σ = ε ∇ u
   exact_gradu(X,sigma);
   sigma *= epsilon;
}

real_t exact_hatu(const Vector & X)
{
   return -exact_u(X);
}

void exact_hatf(const Vector & X, Vector & hatf)
{
   Vector sigma;
   exact_sigma(X,sigma);
   real_t u = exact_u(X);
   hatf.SetSize(X.Size());
   for (int i = 0; i<hatf.Size(); i++)
   {
      hatf[i] = beta[i] * u - sigma[i];
   }
}

real_t f_exact(const Vector & X)
{
   // f = - εΔu + ∇⋅(βu)
   Vector du;
   exact_gradu(X,du);
   real_t d2u = exact_laplacian_u(X);

   real_t s = 0;
   for (int i = 0; i<du.Size(); i++)
   {
      s += beta[i] * du[i];
   }
   return -epsilon * d2u + s;
}

void setup_test_norm_coeffs(GridFunction & c1_gf, GridFunction & c2_gf)
{
   Array<int> vdofs;
   FiniteElementSpace * fes = c1_gf.FESpace();
   Mesh * mesh = fes->GetMesh();
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      real_t volume = mesh->GetElementVolume(i);
      real_t c1 = std::min(epsilon/volume, (real_t) 1.);
      real_t c2 = std::min(1./epsilon, 1./volume);
      fes->GetElementDofs(i,vdofs);
      c1_gf.SetSubVector(vdofs,c1);
      c2_gf.SetSubVector(vdofs,c2);
   }
}
