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
//                   MFEM Ultraweak DPG Maxwell parallel example
//
// Compile with: make lh-eld-dpg
//
// mpirun -np 8 ./lh-eld-dpg -o 3 -paraview -pr 0
// mpirun -np 8 ./lh-eld-dpg -o 3 -paraview -pr 1 -sc

// Electron Landau Damping
// Strong formulation:
//     ∇×(1/μ₀∇×E) - ω² ϵ₀ ϵ E + i ω²ϵ₀(J₁ + J₂) = 0,   in Ω
//                   - Δ∥ J₁ + c₁ J₁ - c₁ P(r) E∥ = 0,   in Ω     
//                   - Δ∥ J₂ + c₂ J₂ + c₂ P(r) E∥ = 0,   in Ω 
//                                            E×n = E₀,  on ∂Ω
//                                            J₁  = 0,  on ∂Ω
//                                            J₂  = 0,  on ∂Ω
// The DPG UW deals with the First Order System
//   i ω μ₀ H + ∇ × E                  = 0,   in Ω
// -i ω ϵ₀ϵ E + ∇ × H - ω ϵ₀ (J₁ + J₂) = 0,   in Ω
//         - Δ∥ J₁ + c₁ J₁ - c₁ P(r) E∥ = 0,   in Ω     
//         - Δ∥ J₂ + c₂ J₂ + c₂ P(r) E∥ = 0,   in Ω 
//                                 E×n = E₀,  on ∂Ω
//                                 J₁  = 0,  on ∂Ω
//                                 J₂  = 0,  on ∂Ω


// in 2D
// E is vector valued and H is scalar.
//      (∇ × E, δE) = (E, ∇ × δE ) + < n × E , δE >
// or (∇ ⋅ AE , δE) = (AE, ∇ δE ) + < AE ⋅ n, δE >
// where A = [0 1; -1 0];

// E ∈ (L²(Ω))² , H ∈ L²(Ω), J ∈ (H¹(Ω))²
// Ê ∈ H^-1/2(Γₕ), Ĥ ∈ H^1/2(Γₕ)
//     iωμ₀ (H,δE) + (E,∇×δE) + < AÊ, δE > = 0,      ∀ δE ∈ H¹(Ω)
//  -i ωϵ₀ϵ (E,δH) + (H,∇×δH) + < Ĥ, δH×n > - ωϵ₀(J₁ + J₂,δH) = 0,  ∀ δH ∈ H(curl,Ω)
// ( (b⋅∇)J₁,(b⋅∇) δJ₁ ) + c₁ (J₁,δJ₁) - c₁ (P(r) b⊗b E, δJ₁) = 0,  ∀ δJ₁ ∈ (H¹(Ω))²
// ( (b⋅∇)J₂,(b⋅∇) δJ₂ ) + c₂ (J₂,δJ₂) + c₂ (P(r) b⊗b E, δJ₂) = 0,  ∀ δJ₁ ∈ (H¹(Ω))²
//                                                                  Ê = E₀, on ∂Ω
//                                                            J₁ = J₂ = 0,  on ∂Ω
// ----------------------------------------------------------------------------------------------
// |   |      E       |     H    |        J₁        |        J₂        |   Ê   |   Ĥ    |  RHS  |
// ----------------------------------------------------------------------------------------------
// |δE |  (E,∇ × δE)  |iωμ₀(H,δE)|                  |                  | <Ê,δE>|        |   0   |  
// |   |              |          |                  |                  |       |        |       |  
// |δH | -iωϵ₀ϵ(E,δH) | (H,∇×δH) |   -ωϵ₀ (J₁,δH)   |  -ωϵ₀ (J₂,δH)    |       |<Ĥ,δH×n>|   0   |  
// |   |              |          |                  |                  |       |        |       |  
// |δJ₁|-c₁(P(r)E,δJ₁)|          |((b⋅∇)J₁,(b⋅∇)δJ₁)|                  |       |        |   0   |  
// |   |              |          |     + c₁ (J₁,δJ₁)|                  |       |        |       |    
// |δJ₂| c₂(P(r)E,δJ₂)|          |                  |((b⋅∇)J₂,(b⋅∇)δJ₂)|       |        |   0   |  
// |   |              |          |                  |     + c₂ (J₂,δJ₂)|       |        |       |    
// where (δE,δH,δJ₁,δJ₂) ∈  H¹(Ω) × H(curl,Ω) × (H¹(Ω))² × (H¹(Ω))² 



#include "mfem.hpp"
#include "../util/pcomplexweakform.hpp"
#include "../util/pcomplexblockform.hpp"
#include "../../common/mfem-common.hpp"
#include "../util/blockcomplexhypremat.hpp"
#include "../util/maxwell_utils.hpp"
#include "utils/lh_utils.hpp"
#include "../util/utils.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "data/LH_hot.msh";
   int order = 1;
   int delta_order = 1;
   int par_ref_levels = 0;
   int ser_ref_levels = 0;

   real_t rnum=1.5;
   real_t mu = 1.257;
   real_t eps0 = 8.8541878128;
   real_t cfactor = 1e-6;
   real_t balance_scale = 1.0;
   bool enable_balance_scale = false;

   bool visualization = false;
   bool paraview = false;
   bool debug = false;
   bool mumps_solver = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&ser_ref_levels, "-sr", "--serial-refinement_levels",
                  "Number of serial refinement levels.");                  
   args.AddOption(&par_ref_levels, "-pr", "--parallel-refinement_levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&a0, "-a0", "--a0", "P(r) first parameter.");
   args.AddOption(&a1, "-a1", "--a1", "P(r) second parameter.");
   args.AddOption(&delta, "-delta", "--delta", "stability parameter.");
   args.AddOption(&mumps_solver, "-mumps", "--mumps", "-no-mumps",
                  "--no-mumps",
                  "Enable or disable MUMPS solver.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&debug, "-debug", "--debug", "-no-debug",
                  "--no-debug",
                  "Enable or disable debug mode (delta = 0.01 and no coupling).");                  
   args.AddOption(&enable_balance_scale, "-ebs", "--enable-balance-scale", "-no-ebs",
                  "--no-ebs",
                  "Enable or disable balance scale.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   // number of diffusion equations
   int ndiffusionequations = 2; 

   Vector cvals(ndiffusionequations);
   Vector csigns(ndiffusionequations);
   cvals(0)  = 25e6;  cvals(1)  = 1e6;
   csigns(0) = -1.0;  csigns(1) = 1.0;
   cvals *= cfactor; // scale the coefficients

   real_t omega = 2.*M_PI*rnum;
   int test_order = order+delta_order;

   balance_scale = (enable_balance_scale) ? omega * eps0  : 1.0;

   if (!debug) 
   {
      delta = 0.0; // disable delta if electron Landau damping is enabled
      if (Mpi::Root())
      {
         cout << "Electron Landau damping enabled, delta set to 0.0." << endl;
      }
   }    

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2, "Dimension != 2 is not supported in this example");

   for (int i = 0; i < ser_ref_levels; i++)
   {
      mesh.UniformRefinement();
   }

   // mesh.RemoveInternalBoundaries();

   Array<int> int_bdr_attr;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      if (mesh.FaceIsInterior(mesh.GetBdrElementFaceIndex(i)))
      {
         int_bdr_attr.Append(mesh.GetBdrAttribute(i));
      }
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   for (int i = 0; i < par_ref_levels; i++)
   {
      pmesh.UniformRefinement();
   }  

   int nattr = (pmesh.attributes.Size()) ? pmesh.attributes.Max() : 0;
   Array<int> attr(nattr);
   for (int i = 0; i<nattr; i++) { attr[i] = i+1; }

   // Define coefficients
   ConstantCoefficient muinv(1./mu);
   ConstantCoefficient one_cf(1.0);
   //  ωμ₀
   ConstantCoefficient omegamu_cf(omega*mu);
   // -ω μ₀  
   ConstantCoefficient negomegamu_cf(-omega*mu);
   // -ωϵ₀
   real_t scale = (debug) ? 0.0 : 1.0;
   ConstantCoefficient negomegeps0_cf(-omega*eps0 * scale);
   ConstantCoefficient balancescaled_negomegeps0_cf( -omega*eps0/balance_scale * scale);

   // μ₀² ω²
   ConstantCoefficient mu2omeg2_cf((mu*mu*omega*omega));

   Vector zero(dim); zero = 0.0;
   Vector one_x(dim); one_x = 0.0; one_x(0) = 1.0;
   Vector negone_x(dim); negone_x = 0.0; negone_x(0) = -1.0;
   VectorConstantCoefficient zero_vcf(zero);
   VectorConstantCoefficient one_x_cf(one_x);
   VectorConstantCoefficient negone_x_cf(negone_x);

   DenseMatrix Mone(dim); 
   Mone = 0.0; Mone(0,0) = Mone(1,1) = 1.0;
   MatrixConstantCoefficient Mone_cf(Mone);
   DenseMatrix Mzero(dim); Mzero = 0.0;
   MatrixConstantCoefficient Mzero_cf(Mzero);

   Array<MatrixCoefficient*> coefs_r(nattr);
   Array<MatrixCoefficient*> coefs_i(nattr);
   for (int i = 0; i < nattr-1; ++i)
   {
      coefs_r[i] = &Mone_cf;
      coefs_i[i] = &Mzero_cf;
   }

   // S(r) 
   FunctionCoefficient S_cf_r(sfunc_r), S_cf_i(sfunc_i);
   // P(r) 
   FunctionCoefficient P_cf_r(pfunc_r), P_cf_i(pfunc_i); 

   VectorFunctionCoefficient b_cf(dim,bfunc);// b
   ScalarVectorProductCoefficient scaled_b_cf(sqrt(cfactor), b_cf);

   MatrixFunctionCoefficient bb_cf(dim,bcrossb); // b⊗b
   MatrixSumCoefficient oneminusbb(Mone_cf, bb_cf, 1.0, -1.0); // 1 - b⊗b

   // S(r) (I - b⊗b)
   ScalarMatrixProductCoefficient Soneminusbb_r(S_cf_r, oneminusbb), Soneminusbb_i(S_cf_i, oneminusbb); 

   // P(r) b⊗b 
   ScalarMatrixProductCoefficient P_cf_bb_r(P_cf_r, bb_cf), P_cf_bb_i(P_cf_i, bb_cf); 

   // ε = S(r) (I - b⊗b) + P(r) b⊗b 
   MatrixSumCoefficient eps_r(Soneminusbb_r, P_cf_bb_r, 1.0, 1.0); 
   MatrixSumCoefficient eps_i(Soneminusbb_i, P_cf_bb_i, 1.0, 1.0); 

   coefs_r[nattr-1] = &eps_r;
   coefs_i[nattr-1] = &eps_i;

   // for (int i = 0; i < nattr-1; ++i)
   // {
   //    coefs_r[i] = &eps_r;
   //    coefs_i[i] = &eps_i;
   // }

   PWMatrixCoefficient eps_cf_r(dim, attr, coefs_r);
   PWMatrixCoefficient eps_cf_i(dim, attr, coefs_i);

   ConstantCoefficient eps0omeg(omega * eps0);
   ConstantCoefficient negeps0omeg(-omega * eps0);

   // ω ϵ₀ ϵᵣ 
   ScalarMatrixProductCoefficient eps0omeg_eps_r(eps0omeg, eps_cf_r);
   // ω ϵ₀ ϵᵢ
   ScalarMatrixProductCoefficient eps0omeg_eps_i(eps0omeg, eps_cf_i);
   // -ω ϵ₀ ϵᵣ 
   ScalarMatrixProductCoefficient negeps0omeg_eps_r(negeps0omeg, eps_cf_r);
   // -ω ϵ₀ ϵᵢ
   ScalarMatrixProductCoefficient negeps0omeg_eps_i(eps0omeg, eps_cf_i);

   // A = [0 1; -1 0]
   DenseMatrix rot_mat(2);
   rot_mat(0,0) = 0.; rot_mat(0,1) = 1.;
   rot_mat(1,0) = -1.; rot_mat(1,1) = 0.;
   MatrixConstantCoefficient rot(rot_mat);

   // ω ϵ₀ ϵᵣ A
   MatrixProductCoefficient eps0omeg_eps_r_rot(eps0omeg_eps_r, rot);
   // ω ϵ₀ ϵᵢ A
   MatrixProductCoefficient eps0omeg_eps_i_rot(eps0omeg_eps_i, rot);
   // -ω ϵ₀ ϵᵣ A
   MatrixProductCoefficient negeps0omeg_eps_r_rot(negeps0omeg_eps_r, rot);
   // -ω ϵ₀ ϵᵢ A
   MatrixProductCoefficient negeps0omeg_eps_i_rot(negeps0omeg_eps_i, rot);

   // (ωϵ₀ϵ)(ωϵ₀ϵ)^*  (δH, δH)
   TransposeMatrixCoefficient eps0omeg_eps_r_t(eps0omeg_eps_r);
   TransposeMatrixCoefficient eps0omeg_eps_i_t(eps0omeg_eps_i);
   MatrixProductCoefficient MrMrt_cf(eps0omeg_eps_r, eps0omeg_eps_r_t);
   MatrixProductCoefficient MiMit_cf(eps0omeg_eps_i, eps0omeg_eps_i_t);
   MatrixProductCoefficient MiMrt_cf(eps0omeg_eps_i, eps0omeg_eps_r_t);
   MatrixProductCoefficient MrMit_cf(eps0omeg_eps_r, eps0omeg_eps_i_t);

   // (MᵣMᵣᵗ + MᵢMᵢᵗ) + i (MᵢMᵣᵗ - MᵣMᵢᵗ)
   MatrixSumCoefficient Mreal_cf(MrMrt_cf,MiMit_cf);
   MatrixSumCoefficient Mimag_cf(MiMrt_cf,MrMit_cf,1.0,-1.0);

   // if ELD
   Array<Vector *> c_arrays(ndiffusionequations);
   Array<PWConstCoefficient *> pw_c_coeffs(ndiffusionequations);
   Array<MatrixCoefficient *> cPrbb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> cPibb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> signedcPrbb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> signedcPibb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> balancescaled_signedcPrbb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> balancescaled_signedcPibb_cf(ndiffusionequations);
   
   Vector temp(nattr); temp=0.0;
   Array<ConstantCoefficient *> c_coeffs(ndiffusionequations);
   for (int i = 0; i<ndiffusionequations; i++)
   {
      temp[nattr-1] = cvals(i);
      // temp = cvals(i);
      pw_c_coeffs[i] = new PWConstCoefficient(temp);
      c_coeffs[i] = new ConstantCoefficient(cvals(i));
      cPrbb_cf[i] = new ScalarMatrixProductCoefficient(*pw_c_coeffs[i], P_cf_bb_r);
      cPibb_cf[i] = new ScalarMatrixProductCoefficient(*pw_c_coeffs[i], P_cf_bb_i);
      signedcPrbb_cf[i] = new ScalarMatrixProductCoefficient(csigns[i], *cPrbb_cf[i]);
      signedcPibb_cf[i] = new ScalarMatrixProductCoefficient(csigns[i], *cPibb_cf[i]);
      balancescaled_signedcPrbb_cf[i] = new ScalarMatrixProductCoefficient(balance_scale,*signedcPrbb_cf[i]);
      balancescaled_signedcPibb_cf[i] = new ScalarMatrixProductCoefficient(balance_scale,*signedcPibb_cf[i]);
   }   

   // ----------------------------------------------
   // DPG UW formulation for Maxwell equations
   // ----------------------------------------------
   // Define DPG spaces for the Maxwell equations
   // trial spaces for E, H, Ê, Ĥ
   Array<FiniteElementCollection *> dpg_trial_fecols;
   Array<FiniteElementCollection *> dpg_test_fecols;
   Array<ParFiniteElementSpace *> dpg_pfes;

   // Vector L2 space for E
   dpg_trial_fecols.Append(new L2_FECollection(order-1, dim));
   dpg_pfes.Append(new ParFiniteElementSpace(&pmesh, dpg_trial_fecols.Last(), dim));
   // Scalar L2 space for H
   dpg_trial_fecols.Append(new L2_FECollection(order-1, dim));
   dpg_pfes.Append(new ParFiniteElementSpace(&pmesh, dpg_trial_fecols.Last()));

   // Trial trace space for Ê 
   dpg_trial_fecols.Append(new RT_Trace_FECollection(order-1, dim));
   dpg_pfes.Append(new ParFiniteElementSpace(&pmesh, dpg_trial_fecols.Last()));
   // Trial trace space for Ĥ 
   dpg_trial_fecols.Append(new H1_Trace_FECollection(order, dim));
   dpg_pfes.Append(new ParFiniteElementSpace(&pmesh, dpg_trial_fecols.Last()));

   Array<HYPRE_BigInt> dpg_tdofs(dpg_pfes.Size());
   for (int i = 0; i < dpg_pfes.Size(); ++i)
   {
      dpg_tdofs[i] = dpg_pfes[i]->GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << "DPG ParFiniteElementSpace " << i << " has " << dpg_tdofs[i]
              << " true dofs." << endl;
      }
   }
   if (Mpi::Root())
   {
      cout << "Total number of DPG true dofs: " << dpg_tdofs.Sum() << endl;
   }

   // test spaces for E and H
   dpg_test_fecols.Append(new H1_FECollection(test_order, dim));
   dpg_test_fecols.Append(new ND_FECollection(test_order, dim));

   ParComplexDPGWeakForm * a_dpg = new ParComplexDPGWeakForm(dpg_pfes,dpg_test_fecols);

   // (E,∇ × δE)
   a_dpg->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one_cf)),
                         nullptr,0, 0);
   //  -i ω ϵ₀ (ϵE,δH) = - i ω ϵ₀(ϵᵣ + i ϵᵢ E, δH)
   //                  =  (ω ϵ₀ ϵᵢ E, δH) + i (-ω ϵ₀ϵᵣ E, δH)
   a_dpg->AddTrialIntegrator(
      new TransposeIntegrator(new VectorFEMassIntegrator(eps0omeg_eps_i)), 
      new TransposeIntegrator(new VectorFEMassIntegrator(negeps0omeg_eps_r)),
      0, 1);
   // iωμ₀(H,δE)
   a_dpg->AddTrialIntegrator(nullptr, new MixedScalarMassIntegrator(omegamu_cf), 1, 0);
   // a_dpg->AddTrialIntegrator(nullptr, new MassIntegrator(omegamu_cf), 1, 0);
   // (H,∇ × δH)                         
   a_dpg->AddTrialIntegrator(
      new TransposeIntegrator(new MixedCurlIntegrator(one_cf)), nullptr, 1, 1);

   // Trace integrators               
   //  <Ê,δE>
   a_dpg->AddTrialIntegrator(new TraceIntegrator,nullptr, 2, 0);
   // <Ĥ,δH × n>
   a_dpg->AddTrialIntegrator(new TangentTraceIntegrator,nullptr, 3, 1);

   // test integrators
   // (∇δE,∇δE)
   a_dpg->AddTestIntegrator(new DiffusionIntegrator(one_cf),nullptr, 0, 0);
   // (δE,δE)
   a_dpg->AddTestIntegrator(new MassIntegrator(one_cf),nullptr, 0, 0);
   // μ₀² ω² (δE,δE)
   a_dpg->AddTestIntegrator(new MassIntegrator(mu2omeg2_cf),nullptr,0, 0);
   // -i ω μ₀ (δE,∇ × δH) = i (δE, -ω μ₀ ∇ × δ H)
   a_dpg->AddTestIntegrator(nullptr,
         new TransposeIntegrator(new MixedCurlIntegrator(negomegamu_cf)),0, 1);
   // -i ω ϵ₀ϵ(∇ × δE, δH) = -i (ωϵ₀(ϵᵣ+iϵᵢ) A ∇ δE,δE), A = [0 1; -1 0]
   //                       =  (ω ϵ₀ ϵᵢ A ∇ δE,δE) + i (-ω ϵ₀ ϵᵣ A ∇ δE,δE)
   a_dpg->AddTestIntegrator(new MixedVectorGradientIntegrator(eps0omeg_eps_i_rot),
                        new MixedVectorGradientIntegrator(negeps0omeg_eps_r_rot),0, 1);
   // i ω μ₀ (∇ × δH ,δE) = i (ω μ₀ ∇ × δH, δE )
   a_dpg->AddTestIntegrator(nullptr,new MixedCurlIntegrator(omegamu_cf),
                        1, 0);
   // i ω ϵ₀ϵ̄ (δH, ∇ × δE ) = i (ω ϵ₀(ϵᵣ -i ϵᵢ) δH, A ∇ δE) 
   //                        = ( δH, ω ϵ₀ ϵᵢ A ∇ δE) + i (δH, ω ϵ₀ ϵᵣ A ∇ δE)
   a_dpg->AddTestIntegrator(
      new TransposeIntegrator(new MixedVectorGradientIntegrator(eps0omeg_eps_i_rot)),
      new TransposeIntegrator(new MixedVectorGradientIntegrator(eps0omeg_eps_r_rot)),1, 0);
   // (ωϵ₀ϵ)(ωϵ₀ϵ)^*  (δH, δH)
   // (MᵣMᵣᵗ + MᵢMᵢᵗ) + i (MᵢMᵣᵗ - MᵣMᵢᵗ)
   a_dpg->AddTestIntegrator(new VectorFEMassIntegrator(Mreal_cf),
                        new VectorFEMassIntegrator(Mimag_cf),1, 1);
   // (∇×δH ,∇×δH)
   a_dpg->AddTestIntegrator(new CurlCurlIntegrator(one_cf),nullptr,1,1);
   // (δH,δH)
   a_dpg->AddTestIntegrator(new VectorFEMassIntegrator(one_cf),nullptr,1,1);

   a_dpg->Assemble();

   // ----------------------------------------------
   // Define FEM spaces for the diffusion equations
   // ----------------------------------------------

   Array<FiniteElementCollection *> fem_fecols;
   Array<ParFiniteElementSpace *> fem_pfes;
   for (int i = 0; i < ndiffusionequations; ++i)
   {
      fem_fecols.Append(new H1_FECollection(order, dim));
      fem_pfes.Append(new ParFiniteElementSpace(&pmesh, fem_fecols.Last(), dim));
   }

   Array<HYPRE_BigInt> fem_tdofs(fem_pfes.Size());
   for (int i = 0; i < fem_pfes.Size(); ++i)
   {
      fem_tdofs[i] = fem_pfes[i]->GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << "FEM ParFiniteElementSpace " << i << " has " << fem_tdofs[i]
              << " true dofs." << endl;
      }
   }

   if (Mpi::Root())
   {
      cout << "Total number of FEM true dofs: " << fem_tdofs.Sum() << endl;
   }
   
   ParComplexBlockForm *a_fem = new ParComplexBlockForm(fem_pfes);
   for (int i = 0; i<ndiffusionequations; i++)
   {
      // ( (b⋅∇)Jᵢ  , (b⋅∇) Gᵢ)
      a_fem->AddDomainIntegrator(new DirectionalVectorDiffusionIntegrator(scaled_b_cf), nullptr, i, i);
      // cᵢ (Jᵢ , Gᵢ)
      a_fem->AddDomainIntegrator(new VectorMassIntegrator(*pw_c_coeffs[i]), nullptr, i, i);
   }
   a_fem->Assemble();

   // ----------------------------------------------
   // Cross term DPG-FEM coupling
   // ----------------------------------------------
   //  -ωϵ₀ (J₁,δH)
   //  -ωϵ₀ (J₂,δH)
   ParMixedBilinearForm * a_J1H = new ParMixedBilinearForm(fem_pfes[0], dpg_pfes[1]);
   ParMixedBilinearForm * a_J2H = new ParMixedBilinearForm(fem_pfes[1], dpg_pfes[1]);

   ParMixedBilinearForm * a_J1H_i = new ParMixedBilinearForm(fem_pfes[0], dpg_pfes[1]);
   ParMixedBilinearForm * a_J2H_i = new ParMixedBilinearForm(fem_pfes[1], dpg_pfes[1]);
   ConstantCoefficient dummy_cf(0.0);
   a_J1H_i->AddDomainIntegrator(new VectorMassIntegrator(dummy_cf));
   a_J2H_i->AddDomainIntegrator(new VectorMassIntegrator(dummy_cf));
   a_J1H_i->Assemble(0);
   a_J2H_i->Assemble(0);

   // a_J1H->AddDomainIntegrator(new VectorMassIntegrator(negomegeps0_cf));
   // a_J2H->AddDomainIntegrator(new VectorMassIntegrator(negomegeps0_cf));

   a_J1H->AddDomainIntegrator(new VectorMassIntegrator(balancescaled_negomegeps0_cf));
   a_J2H->AddDomainIntegrator(new VectorMassIntegrator(balancescaled_negomegeps0_cf));


   
   a_J1H->Assemble(0);
   a_J2H->Assemble(0);


   // ±cᵢ(P(r) (b ⊗ b) E, δJᵢ))
   ParMixedBilinearForm * a_EJ1 = new ParMixedBilinearForm(dpg_pfes[0], fem_pfes[0]);
   ParMixedBilinearForm * a_EJ2 = new ParMixedBilinearForm(dpg_pfes[0], fem_pfes[1]);
   ParMixedBilinearForm * a_EJ1_i = new ParMixedBilinearForm(dpg_pfes[0], fem_pfes[0]);
   ParMixedBilinearForm * a_EJ2_i = new ParMixedBilinearForm(dpg_pfes[0], fem_pfes[1]);
   // a_EJ1_i->AddDomainIntegrator(new VectorMassIntegrator(*signedcPibb_cf[0]));
   // a_EJ2_i->AddDomainIntegrator(new VectorMassIntegrator(*signedcPibb_cf[1]));

   a_EJ1_i->AddDomainIntegrator(new VectorMassIntegrator(*balancescaled_signedcPibb_cf[0]));
   a_EJ2_i->AddDomainIntegrator(new VectorMassIntegrator(*balancescaled_signedcPibb_cf[1]));

   a_EJ1_i->Assemble(0);
   a_EJ2_i->Assemble(0);
   // a_EJ1->AddDomainIntegrator(new VectorMassIntegrator(*signedcPrbb_cf[0]));
   // a_EJ2->AddDomainIntegrator(new VectorMassIntegrator(*signedcPrbb_cf[1]));

   a_EJ1->AddDomainIntegrator(new VectorMassIntegrator(*balancescaled_signedcPrbb_cf[0]));
   a_EJ2->AddDomainIntegrator(new VectorMassIntegrator(*balancescaled_signedcPrbb_cf[1]));


   a_EJ1->Assemble(0);
   a_EJ2->Assemble(0);


   for (int i = 0; i<ndiffusionequations; i++)
   {
      delete pw_c_coeffs[i];
      delete c_coeffs[i];
      delete cPrbb_cf[i]; 
      delete cPibb_cf[i]; 
      delete signedcPrbb_cf[i]; 
      delete signedcPibb_cf[i]; 
   }

   // Assemble all the system and get the global matrix and right-hand side
   Array<ParFiniteElementSpace *> all_pfes;
   all_pfes.Append(dpg_pfes);
   all_pfes.Append(fem_pfes);

   int npfes = all_pfes.Size();
   Array<int> all_offsets(npfes + 1); all_offsets[0] = 0;
   Array<int> all_toffsets(npfes + 1); all_toffsets[0] = 0;
   for (int i = 0; i < npfes; ++i)
   {
      all_offsets[i+1] = all_pfes[i]->GetVSize();
      all_toffsets[i+1] = all_pfes[i]->GetTrueVSize();
   }
   all_offsets.PartialSum();
   all_toffsets.PartialSum();

   Array<int> empty;
   OperatorPtr Ah_dpg, Ah_fem, Ah_J1H, Ah_J2H, Ah_EJ1, Ah_EJ2;
   OperatorPtr Ah_J1H_i, Ah_J2H_i, Ah_EJ1_i, Ah_EJ2_i;
   // 4x4 upper left block (E, H, Ê, Ĥ)
   a_dpg->FormSystemMatrix(empty,Ah_dpg); 
   // 2x2 lower right block (J₁, J₂)
   a_fem->FormSystemMatrix(empty,Ah_fem);
   // cross term A₁₄  
   a_J1H->FormRectangularSystemMatrix(empty, empty, Ah_J1H);
   a_J1H_i->FormRectangularSystemMatrix(empty, empty, Ah_J1H_i);
   // cross term A₂₄
   a_J2H->FormRectangularSystemMatrix(empty, empty, Ah_J2H);
   a_J2H_i->FormRectangularSystemMatrix(empty, empty, Ah_J2H_i);
   // cross term A₄₀ 
   a_EJ1->FormRectangularSystemMatrix(empty, empty, Ah_EJ1);
   a_EJ1_i->FormRectangularSystemMatrix(empty, empty, Ah_EJ1_i);
   // cross term A₅₀  
   a_EJ2->FormRectangularSystemMatrix(empty, empty, Ah_EJ2);
   a_EJ2_i->FormRectangularSystemMatrix(empty, empty, Ah_EJ2_i);


   // put all the operators into a block operator
   BlockOperator A_r(all_toffsets);
   BlockOperator A_i(all_toffsets);

   ComplexOperator * Ac_dpg = Ah_dpg.As<ComplexOperator>();
   ComplexOperator * Ac_fem = Ah_fem.As<ComplexOperator>();

   BlockOperator * BlockAdpg_r = dynamic_cast<BlockOperator *>(&Ac_dpg->real());
   BlockOperator * BlockAdpg_i = dynamic_cast<BlockOperator *>(&Ac_dpg->imag());
   BlockOperator * BlockAfem_r = dynamic_cast<BlockOperator *>(&Ac_fem->real());
   BlockOperator * BlockAfem_i = dynamic_cast<BlockOperator *>(&Ac_fem->imag());


   for (int i = 0; i < dpg_pfes.Size(); ++i)
   {
      for (int j = 0; j < dpg_pfes.Size(); ++j)
      {
         A_r.SetBlock(i, j, &BlockAdpg_r->GetBlock(i, j));
         A_i.SetBlock(i, j, &BlockAdpg_i->GetBlock(i, j));
      }
   }
   for (int i = 0; i < fem_pfes.Size(); ++i)
   {
      for (int j = 0; j < fem_pfes.Size(); ++j)
      {
         A_r.SetBlock(dpg_pfes.Size() + i, dpg_pfes.Size() + j, &BlockAfem_r->GetBlock(i, j));
         A_i.SetBlock(dpg_pfes.Size() + i, dpg_pfes.Size() + j, &BlockAfem_i->GetBlock(i, j));
      }
   }  

   A_r.SetBlock(1, 4, Ah_J1H.Ptr());
   A_r.SetBlock(1, 5, Ah_J2H.Ptr());
   A_r.SetBlock(4, 0, Ah_EJ1.Ptr());
   A_r.SetBlock(5, 0, Ah_EJ2.Ptr());

   A_i.SetBlock(1, 4, Ah_J1H_i.Ptr());
   A_i.SetBlock(1, 5, Ah_J2H_i.Ptr());
   A_i.SetBlock(4, 0, Ah_EJ1_i.Ptr());
   A_i.SetBlock(5, 0, Ah_EJ2_i.Ptr());


   ComplexOperator * A = new ComplexOperator(&A_r, &A_i, false, false);

   if (Mpi::Root())
   {
      mfem::out << "Complex Operator A finished successfully." << endl;
   }

   socketstream E_out_r;

   Vector x(2*all_offsets.Last());
   x = 0.;
   
   Array<ParGridFunction *> pgf_r(npfes);
   Array<ParGridFunction *> pgf_i(npfes);

   for (int i = 0; i < npfes; ++i)
   {
      pgf_r[i] = new ParGridFunction(all_pfes[i], x, all_offsets[i]);
      pgf_i[i] = new ParGridFunction(all_pfes[i], x, all_offsets.Last() + all_offsets[i]);
   }

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);
   ParGridFunction E_par_r(&L2_fes);
   ParGridFunction E_par_i(&L2_fes);

   ParaViewDataCollection * paraview_dc = nullptr;

   std::string output_dir = "ParaView/UW-FEM/" + GetTimestamp();

   if (paraview)
   {
      if (Mpi::Root()) { WriteParametersToFile(args, output_dir); }
      std::ostringstream paraview_file_name;
      std::string filename = GetFilename(mesh_file);
      paraview_file_name << filename
                         << "_par_ref_" << par_ref_levels
                         << "_order_" << order
                         << "_eld_1" ;
      paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &pmesh);
      paraview_dc->SetPrefixPath(output_dir);
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",pgf_r[0]);
      paraview_dc->RegisterField("E_i",pgf_i[0]);
      paraview_dc->RegisterField("E_par_r",&E_par_r);
      paraview_dc->RegisterField("E_par_i",&E_par_i);
      paraview_dc->RegisterField("H_r",pgf_r[1]);
      paraview_dc->RegisterField("H_i",pgf_i[1]);      
      paraview_dc->RegisterField("Jh_1_r",pgf_r[4]);
      paraview_dc->RegisterField("Jh_1_i",pgf_i[4]);
      paraview_dc->RegisterField("Jh_2_r",pgf_r[5]);
      paraview_dc->RegisterField("Jh_2_i",pgf_i[5]);
   }

   Array<int> ess_tdof_list;
   Array<int> ess_tdof_listJ;
   Array<int> ess_bdr;
   Array<int> one_r_bdr;
   Array<int> one_i_bdr;
   Array<int> negone_r_bdr;
   Array<int> negone_i_bdr;
   
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      one_r_bdr.SetSize(pmesh.bdr_attributes.Max());
      one_i_bdr.SetSize(pmesh.bdr_attributes.Max());
      negone_r_bdr.SetSize(pmesh.bdr_attributes.Max());
      negone_i_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;

      // remove internal boundaries
      for (int i = 0; i<int_bdr_attr.Size(); i++)
      {
         ess_bdr[int_bdr_attr[i]-1] = 0;
      }
      all_pfes[2]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += all_toffsets[2];
      }
      ess_bdr = 1;
      for (int i = 0; i<ndiffusionequations;i++)
      {
         ess_tdof_listJ.SetSize(0);
         all_pfes[i+4]->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ);
         for (int j = 0; j < ess_tdof_listJ.Size(); j++)
         {
            ess_tdof_listJ[j] += all_toffsets[i+4];
         }
         ess_tdof_list.Append(ess_tdof_listJ);
      }

      one_r_bdr = 0;  one_i_bdr = 0;
      negone_r_bdr = 0;  negone_i_bdr = 0;
      // attr = 30,2 (real)
      one_r_bdr[30-1] = 1;  one_r_bdr[2-1] = 1;
      // attr = 26,6 (imag)
      one_i_bdr[26-1] = 1;  one_i_bdr[6-1] = 1;
      // attr = 22,10 (real)
      negone_r_bdr[22-1] = 1; negone_r_bdr[10-1] = 1;
      // attr = 18,14 (imag)
      negone_i_bdr[18-1] = 1; negone_i_bdr[14-1] = 1;
   }
   

   // rotate the vector
   // (x,y) -> (y,-x)
   Vector rot_one_x(dim); rot_one_x = 0.0; rot_one_x(1) = -1.0;
   Vector rot_negone_x(dim); rot_negone_x = 0.0; rot_negone_x(1) = 1.0;
   VectorConstantCoefficient rot_one_x_cf(rot_one_x);
   VectorConstantCoefficient rot_negone_x_cf(rot_negone_x);

   pgf_r[2]->ProjectBdrCoefficientNormal(rot_one_x_cf, one_r_bdr);
   pgf_r[2]->ProjectBdrCoefficientNormal(rot_negone_x_cf, negone_r_bdr);
   pgf_i[2]->ProjectBdrCoefficientNormal(rot_one_x_cf, one_i_bdr);
   pgf_i[2]->ProjectBdrCoefficientNormal(rot_negone_x_cf, negone_i_bdr);

   if (Mpi::Root())
   {
      mfem::out << "Boundary conditions finished." << endl;
   }

   BlockOperator * P = new BlockOperator(all_offsets, all_toffsets);
   BlockMatrix * R = new BlockMatrix(all_toffsets, all_offsets);
   P->owns_blocks = 0;
   R->owns_blocks = 0;

   for (int i = 0; i < npfes; i++)
   {
      HypreParMatrix * P_ = all_pfes[i]->Dof_TrueDof_Matrix();
      P->SetBlock(i,i,P_);
      const SparseMatrix * R_ = all_pfes[i]->GetRestrictionMatrix();
      R->SetBlock(i, i, const_cast<SparseMatrix*>(R_));
   }

   if (Mpi::Root())
   {
      mfem::out << "Build prolongation finished" << endl;
   }

   int n = P->Width();
   Vector B(2*n); B = 0.0;

   Vector X(2*n);
   Vector X_r(X, 0, n);
   Vector X_i(X, n, n);

   Vector x_r(x, 0, x.Size()/2);
   Vector x_i(x, x.Size()/2, x.Size()/2);

   R->Mult(x_r, X_r);
   R->Mult(x_i, X_i);

   ParBlockComplexSystem aa(A);
   A = aa.EliminateBC(ess_tdof_list, X, B);

   if (Mpi::Root())
   {
      mfem::out << "Eliminate BC finished successfully." << endl;
   }


   BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&A->real());
   BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&A->imag());

   int nblocks = BlockA_r->NumRowBlocks();
   
   Array2D<const HypreParMatrix*> A_r_matrices(nblocks, nblocks);
   Array2D<const HypreParMatrix*> A_i_matrices(nblocks, nblocks);
   for (int i = 0; i < nblocks; i++)
   {
      for (int j = 0; j < nblocks; j++)
      {
         if (!BlockA_r->IsZeroBlock(i,j))
         {
            A_r_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_r->GetBlock(i,j));
         }
         else
         {
            A_r_matrices(i,j) = nullptr;
         }
         if (!BlockA_i->IsZeroBlock(i,j))
         {
            A_i_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_i->GetBlock(i,j));
         }
         else
         {
            A_i_matrices(i,j) = nullptr;
         }
      }
   }

   HypreParMatrix * Ahr = HypreParMatrixFromBlocks(A_r_matrices);
   HypreParMatrix * Ahi = HypreParMatrixFromBlocks(A_i_matrices);

   ComplexHypreParMatrix * Ahc_hypre =
      new ComplexHypreParMatrix(Ahr, Ahi,false, false);

   if (Mpi::Root())
   {
      mfem::out << "Getting ready for solve." << endl;
   }
   HypreParMatrix *Ah = Ahc_hypre->GetSystemMatrix();
   

#ifdef MFEM_USE_MUMPS
   if (mumps_solver)
   {
      auto solver = new MUMPSSolver(MPI_COMM_WORLD);
      solver->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      solver->SetPrintLevel(1);
      solver->SetOperator(*Ah);
      solver->Mult(B,X);
      delete Ah;
      delete solver;
   }
#else
   if (mumps_solver)
   {
      MFEM_WARNING("MFEM compiled without mumps. Switching to an iterative solver");
   }
   mumps_solver = false;
#endif
   int num_iter = -1;

   Array<int> tdof_offsets(nblocks*2+1);
   tdof_offsets[0] = 0;
   for (int i=0; i<nblocks; i++)
   {
      tdof_offsets[i+1] = A_r_matrices(i,i)->Height();
      tdof_offsets[nblocks+i+1] = tdof_offsets[i+1];
   }
   tdof_offsets.PartialSum();

   if (!mumps_solver)
   {

      BlockDiagonalPreconditioner M(tdof_offsets);

      // HypreBoomerAMG * solver_E = new HypreBoomerAMG((HypreParMatrix &)
                                                   // BlockA_r->GetBlock(0,0));
      // solver_E->SetPrintLevel(0);
      // solver_E->SetSystemsOptions(dim);
      MUMPSSolver * solver_E = new MUMPSSolver(MPI_COMM_WORLD);
      solver_E->SetOperator((HypreParMatrix &)BlockA_r->GetBlock(0,0));

      // HypreBoomerAMG * solver_H = new HypreBoomerAMG((HypreParMatrix &)
                                                // BlockA_r->GetBlock(1,1));
      // solver_H->SetPrintLevel(0);
      MUMPSSolver * solver_H = new MUMPSSolver(MPI_COMM_WORLD);
      solver_H->SetOperator((HypreParMatrix &)BlockA_r->GetBlock(1,1));

      M.SetDiagonalBlock(0,solver_E);
      M.SetDiagonalBlock(1,solver_H);
      M.SetDiagonalBlock(nblocks,solver_E);
      M.SetDiagonalBlock(nblocks+1,solver_H);

      // HypreAMS * solver_hatE = 
      // new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(2,2), dpg_pfes[2]);

      MUMPSSolver * solver_hatE = new MUMPSSolver(MPI_COMM_WORLD);
      solver_hatE->SetOperator((HypreParMatrix &)BlockA_r->GetBlock(2,2));

      // HypreBoomerAMG * solver_hatH = new HypreBoomerAMG((HypreParMatrix &)
               //   BlockA_r->GetBlock(3,3));
      // solver_hatE->SetPrintLevel(0);
      // solver_hatH->SetPrintLevel(0);
      // solver_hatH->SetRelaxType(88);

      MUMPSSolver * solver_hatH = new MUMPSSolver(MPI_COMM_WORLD);
      solver_hatH->SetOperator((HypreParMatrix &)BlockA_r->GetBlock(3,3));


      M.SetDiagonalBlock(2,solver_hatE);
      M.SetDiagonalBlock(3,solver_hatH);
      M.SetDiagonalBlock(2+nblocks,solver_hatE);
      M.SetDiagonalBlock(3+nblocks,solver_hatH);


      // HypreBoomerAMG * solver_J1 = new HypreBoomerAMG((HypreParMatrix &)
                                                // BlockA_r->GetBlock(4,4));
      // solver_J1->SetPrintLevel(0);
      // solver_J1->SetSystemsOptions(dim);
      // solver_J1->SetRelaxType(88);
      MUMPSSolver * solver_J1 = new MUMPSSolver(MPI_COMM_WORLD);
      solver_J1->SetOperator((HypreParMatrix &)BlockA_r->GetBlock(4,4));

      // HypreBoomerAMG * solver_J2 = new HypreBoomerAMG((HypreParMatrix &)
                                                // BlockA_r->GetBlock(5,5));
      // solver_J2->SetPrintLevel(0);
      // solver_J2->SetSystemsOptions(dim);
      // solver_J2->SetRelaxType(88);

      MUMPSSolver * solver_J2 = new MUMPSSolver(MPI_COMM_WORLD);
      solver_J2->SetOperator((HypreParMatrix &)BlockA_r->GetBlock(5,5));


      M.SetDiagonalBlock(4,solver_J1);
      M.SetDiagonalBlock(5,solver_J2);
      M.SetDiagonalBlock(4+nblocks,solver_J1);
      M.SetDiagonalBlock(5+nblocks,solver_J2);                                                

      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetRelTol(1e-10);
      gmres.SetMaxIter(1000);
      gmres.SetPrintLevel(1);
      gmres.SetPreconditioner(M);
      gmres.SetOperator(*A);
      gmres.Mult(B, X);
   }


   n = P->Height();
   int m = P->Width();

   x_r.MakeRef(x, 0, n);
   x_i.MakeRef(x, n, n);

   X_r.MakeRef(X, 0, m);
   X_i.MakeRef(X, m, m);

   P->Mult(X_r, x_r);
   P->Mult(X_i, x_i);


   for (int i = 0; i < npfes; ++i)
   {
      pgf_r[i]->MakeRef(all_pfes[i], x, all_offsets[i]);
      pgf_i[i]->MakeRef(all_pfes[i], x, all_offsets.Last() + all_offsets[i]);
   }
   
   ParallelECoefficient par_e_r(pgf_r[0]);
   ParallelECoefficient par_e_i(pgf_i[0]);
   E_par_r.ProjectCoefficient(par_e_r);
   E_par_i.ProjectCoefficient(par_e_i);
   
   if (visualization)
   {
      const char * keys = nullptr;
      char vishost[] = "localhost";
      int  visport   = 19916;
      common::VisualizeField(E_out_r,vishost, visport, *pgf_r[0],
                             "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
   }

   if (paraview)
   {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime((real_t)0);
      paraview_dc->Save();
      delete paraview_dc;
   }


   delete a_fem;
   for (int i = 0; i < fem_fecols.Size(); ++i)
   {
      delete fem_fecols[i];
      delete fem_pfes[i];
   }


   delete a_dpg;
   for (int i = 0; i < dpg_trial_fecols.Size(); ++i)
   {
      delete dpg_trial_fecols[i];
      delete dpg_pfes[i];
   }
   for (int i = 0; i< dpg_test_fecols.Size(); ++i)
   {
      delete dpg_test_fecols[i];
   }

   return 0;
}
