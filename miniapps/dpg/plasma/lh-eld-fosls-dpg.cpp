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
// Compile with: make lh-eld-fosls-dpg
//
// mpirun -np 8 ./lh-eld-fosls-dpg -o 4 -paraview -eld -m data/quad.msh -ebs -sc

// Electron Landau Damping
// Strong formulation:
//     ∇×(1/μ₀∇×E) - ω² ϵ₀ ϵ E + i ω²ϵ₀(J₁ + J₂) = 0,   in Ω
//               λ₁   (- Δ∥ J₁ + c₁ J₁ - c₁ P(r) E∥) = 0,   in Ω     
//               λ₂   (- Δ∥ J₂ + c₂ J₂ + c₂ P(r) E∥) = 0,   in Ω 
//                                            E×n = E₀,  on ∂Ω
//                                            J₁  = 0,  on ∂Ω
//                                            J₂  = 0,  on ∂Ω
// The DPG-FOSLS deals with the First Order System
//  i ω μ₀ H  + ∇ × E                  = 0,   in Ω
// -i ω ϵ₀ϵ E + ∇ × H - ω ϵ₀ (J₁ + J₂) = 0,   in Ω
// - λ₁ b⋅∇Q₁ + λ₁ c₁ J₁ - λ₁ c₁ P B E = 0,   in Ω     
//                   λ₁ Q₁ + λ₁ b⋅∇ J₁ = 0,   in Ω
// - λ₂ b⋅∇Q₂ + λ₂ c₂ J₂ + λ₂ c₂ P B E = 0,   in Ω
//                   λ₂ Q₂ + λ₂ b⋅∇ J₂ = 0,   in Ω
//                                 E×n = E₀,  on ∂Ω
//                                 J₁  = 0,  on ∂Ω
//                                 J₂  = 0,  on ∂Ω


// in 2D  E is vector valued and H is scalar and
//     ∇ × E = ∇ ⋅ AE  where A = [0 1; -1 0];

// E ∈ H(curl,Ω) , H ∈ H¹(Ω), Jᵢ, Qᵢ ∈ (H¹(Ω))²
//  ( iωμ₀ H, F ) + ( ∇ × E, F) = 0,                      ∀ F  ∈  L²(Ω)
// -(iωϵ₀ϵ E, R ) + ( ∇ × H, R) - ω ϵ₀ (J₁ + J₂, R) = 0,  ∀ R  ∈ (L²(Ω))²
// λ₁ (b ⋅ ∇Q₁, K₁) + λ₁ c₁(J₁, K₁) - λ₁ c₁ (P B E, K₁) = 0,      ∀ K₁ ∈ (L²(Ω))²
// λ₁ (Q₁, L₁) + λ₁ (b ⋅ ∇ J₁, L₁) = 0,                        ∀ L₁ ∈ (L²(Ω))²
// λ₂ (b ⋅ ∇Q₂, K₂) + λ₂ c₂(J₂, K₂) + λ₂ c₂ (P B E, K₂) = 0,      ∀ K₂ ∈ (L²(Ω))²
// λ₂ (Q₂, L₂) + λ₂ (b ⋅ ∇ J₂, L₂) = 0,                        ∀ L₂ ∈ (L²(Ω))²
//                                    E = E₀, on ∂Ω
//                              J₁ = J₂ = 0,  on ∂Ω
// ----------------------------------------------------------------------------------------------------------------
// |   |      E       |      H    |     J₁      |     Q₁     |       J₂   |      Q₂    |   RHS  |
// ----------------------------------------------------------------------------------------------------------------
// | F |  (∇ × E, F)  | (iωμ₀H,F) |             |            |            |            |        |        |   0   |  
// |   |              |           |             |            |            |            |        |        |       |  
// | R | -iωϵ₀ϵ(E,R)  |(∇ × H, R) | -ωϵ₀(J₁,R)  |            | -ωϵ₀(J₂, R)|            |        |        |   0   |  
// |   |              |           |             |            |            |            |        |        |       |  
// |K₁ | -λ₁c₁(PBE,K₁)|           | λ₁c₁(J₁, K₁)|(λ₁b⋅∇Q₁,K₁)|            |            |        |        |   0   |  
// |   |              |           |             |            |            |            |        |        |       |    
// |L₁ |              |           | λ₁(b⋅∇J₁,L₁)| λ₁(Q₁, L₁) |            |            |        |        |   0   |  
// |   |              |           |             |            |            |            |        |        |       |    
// |K₂ | λ₂c₂(PBE, K₂)|           |             |            |λ₂c₂(J₂, K₂)|λ₂(b⋅∇Q₂,K₂)|        |        |   0   |  
// |   |              |           |             |            |            |            |        |        |       |    
// |L₂ |              |           |             |            |λ₂(b⋅∇J₂,L₂)| λ₂(Q₂, L₂) |        |        |   0   |  
// |   |              |           |             |            |            |            |        |        |       |    
// where (F,R,K₁,L₁,K₂,L₂) ∈  L²(Ω) × (L²(Ω))² × (L²(Ω))² × (L²(Ω))² × (L²(Ω))² × (L²(Ω))² 



#include "mfem.hpp"
#include "../util/pcomplexweakform.hpp"
#include "../../common/mfem-common.hpp"
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
   Hypre::Init();

   const char *mesh_file = "data/LH_hot.msh";
   int order = 1;
   int delta_order = 1;
   int par_ref_levels = 0;
   int ser_ref_levels = 0;

   // real_t rnum=1.5e9;
   // real_t mu = 1.257e-6;
   // real_t eps0 = 8.8541878128e-12;

   real_t rnum=1.5;
   real_t mu = 1.257;
   real_t eps0 = 8.8541878128;
   real_t balance_scale = 1.0;
   bool enable_balance_scale = false;

   bool eld = false; // enable/disable electron Landau damping 
   real_t delta_prec = 0.0;
   real_t lambda1 = 1.0;
   real_t lambda2 = 1.0;

   bool static_cond = false;
   bool visualization = false;
   bool paraview = false;
   bool debug = false;
   bool mumps_solver = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&delta_order, "-do", "--delta-order",
                  "Finite element order for the test space");                  
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
   args.AddOption(&lambda1, "-l1", "--lambda1", "Lambda 1 scaling parameter.");
   args.AddOption(&lambda2, "-l2", "--lambda2", "Lambda 2 scaling parameter.");
   args.AddOption(&delta_prec, "-dp", "--delta-prec", "stability parameter for the preconditioner.");
   args.AddOption(&eld, "-eld", "--eld", "-no-eld",
                  "--no-eld",
                  "Enable or disable electron Landau damping.");
   args.AddOption(&mumps_solver, "-mumps", "--mumps", "-no-mumps",
                  "--no-mumps",
                  "Enable or disable MUMPS solver.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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
   int ndiffusionequations = (eld) ?  2 : 0; 

   Vector cvals(ndiffusionequations);
   Vector csigns(ndiffusionequations);
   if (eld)
   {
      cvals(0)  = 25e6;  cvals(1)  = 1e6;
      csigns(0) = -1.0;  csigns(1) = 1.0;
   }
   cvals(0) *= lambda1;
   cvals(1) *= lambda2;
   real_t omega = 2.*M_PI*rnum;
   int test_order = order+delta_order;

   balance_scale = (enable_balance_scale) ? eps0 * omega * omega : 1.0;

   if (eld && !debug) 
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
   real_t scale = (debug) ? 0.0 : 1.0;

   // Define coefficients
   ConstantCoefficient one_cf(1.0);
   //  ωμ₀
   ConstantCoefficient omegamu_cf(omega*mu);
   // -ωϵ₀
   ConstantCoefficient negomegeps0_cf(-omega*eps0 * scale);
   ConstantCoefficient balancescaled_negomegeps0_cf( -omega*eps0/balance_scale * scale);

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

   // b
   VectorFunctionCoefficient b_cf(dim,bfunc);
   ScalarVectorProductCoefficient scaled1_b_cf(sqrt(lambda1), b_cf);
   ScalarVectorProductCoefficient scaled2_b_cf(sqrt(lambda2), b_cf);

   // b⊗b
   MatrixFunctionCoefficient bb_cf(dim,bcrossb); 
   // I - b⊗b
   MatrixSumCoefficient oneminusbb(Mone_cf, bb_cf, 1.0, -1.0); 
   // S(r) (I - b⊗b)
   ScalarMatrixProductCoefficient Soneminusbb_r(S_cf_r, oneminusbb), Soneminusbb_i(S_cf_i, oneminusbb); 
   // P(r) b⊗b 
   ScalarMatrixProductCoefficient P_cf_bb_r(P_cf_r, bb_cf), P_cf_bb_i(P_cf_i, bb_cf); 

   // ε = S(r) (I - b⊗b) + P(r) b⊗b 
   MatrixSumCoefficient eps_r(Soneminusbb_r, P_cf_bb_r, 1.0, 1.0); 
   MatrixSumCoefficient eps_i(Soneminusbb_i, P_cf_bb_i, 1.0, 1.0); 

   coefs_r[nattr-1] = &eps_r;
   coefs_i[nattr-1] = &eps_i;

   PWMatrixCoefficient eps_cf_r(dim, attr, coefs_r);
   PWMatrixCoefficient eps_cf_i(dim, attr, coefs_i);

   ConstantCoefficient eps0omeg(omega * eps0);
   ConstantCoefficient negeps0omeg(-omega * eps0);


   // - iωϵ₀ϵ = ωϵ₀ϵᵢ + i (-ωϵ₀ϵᵣ)
   // ω ϵ₀ ϵᵢ
   ScalarMatrixProductCoefficient eps0omeg_eps_i(eps0omeg, eps_cf_i);
   // -ω ϵ₀ ϵᵣ 
   ScalarMatrixProductCoefficient negeps0omeg_eps_r(negeps0omeg, eps_cf_r);

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


   std::vector<std::string> variables = {" E  ", " H  ", " J₁ ", " J₂ ", " Q₁ ", " Q₂ "};
   std::vector<std::string> spaces = {"H(curl,Ω) ", "   H¹(Ω)  ", 
                                      "  (H¹(Ω))ᵈ", "  (H¹(Ω))ᵈ", 
                                      "  (H¹(Ω))ᵈ", "  (H¹(Ω))ᵈ"};

   // Define the spaces
   Array<FiniteElementCollection *> trial_fecols;
   Array<FiniteElementCollection *> test_fecols;
   Array<ParFiniteElementSpace *> pfes;

   // H(curl) space for E
   trial_fecols.Append(new ND_FECollection(order, dim));
   pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last()));
   // Scalar H¹ space for H
   trial_fecols.Append(new H1_FECollection(order, dim));
   pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last()));

   // Vector H¹ spaces for Jᵢ and Qᵢ
   for (int i = 0; i < 2*ndiffusionequations; i++)
   {
      trial_fecols.Append(new H1_FECollection(order, dim));
      pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last(), dim));
   }

   Array<HYPRE_BigInt> tdofs(pfes.Size());
   for (int i = 0; i < pfes.Size(); ++i)
   {
      tdofs[i] = pfes[i]->GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << "ParFiniteElementSpace " << spaces[i] << "  for " << variables[i] << " has " << tdofs[i]
              << " true dofs." << endl;
      }
   }
   if (Mpi::Root())
   {
      cout << "Total number of true dofs: " << tdofs.Sum() << endl;
   }

   // test spaces for F and G
   test_fecols.Append(new L2_FECollection(test_order, dim));
   test_fecols.Append(new L2_FECollection(test_order, dim));
   // Test spaces Kᵢ and Lᵢ 
   for (int i = 0; i < 2*ndiffusionequations; i++)
   {
      test_fecols.Append(new L2_FECollection(test_order, dim));
   }

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(pfes,test_fecols);
   for (int i = 0; i < test_fecols.Size(); i++)
   {  // all but the first  space are vector valued
      if (i > 0) { a->SetTestFECollVdim(i,dim); }
   }

   // (∇ × E, F)
   a->AddTrialIntegrator(new MixedCurlIntegrator(one_cf), nullptr, 0, 0);

   // (i ω μ₀ H, F)
   a->AddTrialIntegrator(nullptr, new MixedScalarMassIntegrator(omegamu_cf), 1, 0);

   // -i ω ϵ₀ϵ (E, R) = ω ϵ₀ ϵᵢ (E, G) + i (-ω ϵ₀ ϵᵣ E, R)
   a->AddTrialIntegrator(new VectorFEMassIntegrator(eps0omeg_eps_i),
                         new VectorFEMassIntegrator(negeps0omeg_eps_r), 0, 1);

   // (∇ × H, R)
   a->AddTrialIntegrator(new MixedCurlIntegrator(one_cf), nullptr, 1,1);
   
   if (eld)
   {
      // - ω ϵ₀ (J₁, R)
      a->AddTrialIntegrator(new VectorMassIntegrator(balancescaled_negomegeps0_cf), nullptr, 2, 1);
      // - ω ϵ₀ (J₂, R)
      a->AddTrialIntegrator(new VectorMassIntegrator(balancescaled_negomegeps0_cf), nullptr, 3, 1);
      // -c₁(PBE, K₁)
      a->AddTrialIntegrator(new VectorFEMassIntegrator(*balancescaled_signedcPrbb_cf[0]),
                            new VectorFEMassIntegrator(*balancescaled_signedcPibb_cf[0]), 0, 2);

      // c₁(J₁, K₁)
      a->AddTrialIntegrator(new VectorMassIntegrator(*pw_c_coeffs[0]), nullptr, 2, 2);

      // (b⋅∇Q₁, K₁)
      a->AddTrialIntegrator(new DirectionalVectorGradientIntegrator(scaled1_b_cf), nullptr, 3, 2);

      // (b⋅∇J₁, L₁)
      a->AddTrialIntegrator(new DirectionalVectorDiffusionIntegrator(scaled1_b_cf), nullptr, 2, 3);

      // (Q₁, L₁)
      a->AddTrialIntegrator(new VectorMassIntegrator(one_cf), nullptr, 3, 3);

      // c₂(PBE, K₂)
      a->AddTrialIntegrator(new VectorFEMassIntegrator(*balancescaled_signedcPrbb_cf[1]),
                            new VectorFEMassIntegrator(*balancescaled_signedcPibb_cf[1]), 0, 4);

      // c₂(J₂, K₂)
      a->AddTrialIntegrator(new VectorMassIntegrator(*pw_c_coeffs[1]), nullptr, 4, 4);
   
      // (b⋅∇Q₂, K₂)
      a->AddTrialIntegrator(new DirectionalVectorGradientIntegrator(scaled2_b_cf), nullptr, 5, 4);

      // (b⋅∇J₂, L₂)
      a->AddTrialIntegrator(new DirectionalVectorDiffusionIntegrator(scaled2_b_cf), nullptr, 4, 5);

      // (Q₂, L₂)
      a->AddTrialIntegrator(new VectorMassIntegrator(one_cf), nullptr, 5, 5);
   }
   // test integrators for test norm: ||v||² = ||F||² + ||R||² + ||K₁||² + ||L₁||² + ||K₂||² + ||L₂||² 
   // (F,δF), F, δF ∈  L²(Ω)
   a->AddTestIntegrator(new MassIntegrator(one_cf),nullptr, 0, 0);
   // (R,δR), R, δR ∈ (L²(Ω))²
   a->AddTestIntegrator(new VectorMassIntegrator(one_cf),nullptr, 1, 1);
   
   if (eld)
   {
      // (K₁,δK₁), K₁, δK₁ ∈ (L²(Ω))²
      a->AddTestIntegrator(new VectorMassIntegrator(one_cf),nullptr, 2, 2);
      // (L₁,δL₁), L₁, δL₁ ∈ (L²(Ω))²
      a->AddTestIntegrator(new VectorMassIntegrator(one_cf),nullptr, 3, 3);
      // (K₂,δK₂), K₂, δK₂ ∈ (L²(Ω))²
      a->AddTestIntegrator(new VectorMassIntegrator(one_cf),nullptr, 4, 4);
      // (L₂,δL₂), L₂, δL₂ ∈ (L²(Ω))²
      a->AddTestIntegrator(new VectorMassIntegrator(one_cf),nullptr, 5, 5);        
   }

   a->Assemble(0);

   for (int i = 0; i<ndiffusionequations; i++)
   {
      delete pw_c_coeffs[i];
      delete c_coeffs[i];
      delete cPrbb_cf[i]; 
      delete cPibb_cf[i]; 
      delete signedcPrbb_cf[i]; 
      delete signedcPibb_cf[i]; 
   }

   socketstream E_out_r;

   int npfes = pfes.Size();
   Array<int> offsets(npfes+1);  offsets[0] = 0;
   Array<int> toffsets(npfes+1); toffsets[0] = 0;
   for (int i = 0; i<npfes; i++)
   {
      offsets[i+1] = pfes[i]->GetVSize();
      toffsets[i+1] = pfes[i]->TrueVSize();
   }
   offsets.PartialSum();
   toffsets.PartialSum();

   Vector x(2*offsets.Last());
   x = 0.;
   
   Array<ParGridFunction *> pgf_r(npfes);
   Array<ParGridFunction *> pgf_i(npfes);

   for (int i = 0; i < npfes; ++i)
   {
      pgf_r[i] = new ParGridFunction(pfes[i], x, offsets[i]);
      pgf_i[i] = new ParGridFunction(pfes[i], x, offsets.Last() + offsets[i]);
   }

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);
   ParGridFunction E_par_r(&L2_fes);
   ParGridFunction E_par_i(&L2_fes);

   ParaViewDataCollection * paraview_dc = nullptr;

   std::string output_dir = "ParaView/FOSLS/" + GetTimestamp();

   if (paraview)
   {
      if (Mpi::Root()) { WriteParametersToFile(args, output_dir); }
      std::ostringstream paraview_file_name;
      std::string filename = GetFilename(mesh_file);
      paraview_file_name << filename
                         << "_par_ref_" << par_ref_levels
                         << "_order_" << order
                         << "_eld_" << eld;
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
      if (eld)
      {
         paraview_dc->RegisterField("J1_r",pgf_r[2]);
         paraview_dc->RegisterField("J1_i",pgf_i[2]);
         paraview_dc->RegisterField("Q1_r",pgf_r[3]);
         paraview_dc->RegisterField("Q1_i",pgf_i[3]);
         paraview_dc->RegisterField("J2_r",pgf_r[4]);
         paraview_dc->RegisterField("J2_i",pgf_i[4]);
         paraview_dc->RegisterField("Q2_r",pgf_r[5]);
         paraview_dc->RegisterField("Q2_i",pgf_i[5]);
      }
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

      pfes[0]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      ess_bdr=1;
      for (int i = 0; i<ndiffusionequations;i++)
      {
         ess_tdof_listJ.SetSize(0);
         pfes[2*i+2]->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ); // J₁ (2), J₂ (4)
         for (int j = 0; j < ess_tdof_listJ.Size(); j++)
         {
            ess_tdof_listJ[j] += toffsets[2*i+2]; 
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
   

   pgf_r[0]->ProjectBdrCoefficientTangent(one_x_cf, one_r_bdr);
   pgf_r[0]->ProjectBdrCoefficientTangent(negone_x_cf, negone_r_bdr);
   pgf_i[0]->ProjectBdrCoefficientTangent(one_x_cf, one_i_bdr);
   pgf_i[0]->ProjectBdrCoefficientTangent(negone_x_cf, negone_i_bdr);

   OperatorPtr Ah;
   Vector X,B;
   a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);
   ComplexOperator * Ahc = Ah.As<ComplexOperator>();

   bool direct_solve = true;

   if (direct_solve)
   {
      BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
      BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

      int nblocks = BlockA_r->NumRowBlocks();
      if(Mpi::Root())
      {
         mfem::out << "Number of blocks: " << nblocks << std::endl;
      }
      Array2D<const HypreParMatrix*> A_r_matrices(nblocks, nblocks);
      Array2D<const HypreParMatrix*> A_i_matrices(nblocks, nblocks);
      for (int i = 0; i < nblocks; i++)
      {
         for (int j = 0; j < nblocks; j++)
         {
            A_r_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_r->GetBlock(i,j));
            A_i_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_i->GetBlock(i,j));
         }
      }
      HypreParMatrix * Ahr = HypreParMatrixFromBlocks(A_r_matrices);
      HypreParMatrix * Ahi = HypreParMatrixFromBlocks(A_i_matrices);

      ComplexHypreParMatrix * Ahc_hypre =
         new ComplexHypreParMatrix(Ahr, Ahi,true, true);

      ComplexMUMPSSolver cmumps(MPI_COMM_WORLD);
      cmumps.SetPrintLevel(0);
      cmumps.SetOperator(*Ahc_hypre);
      cmumps.Mult(B,X);
      delete Ahc_hypre;   

   }

   a->RecoverFEMSolution(X, x);

   for (int i = 0; i < npfes; ++i)
   {
      pgf_r[i]->MakeRef(pfes[i], x, offsets[i]);
      pgf_i[i]->MakeRef(pfes[i], x, offsets.Last() + offsets[i]);
   }
   
   ParallelECoefficient par_e_r(pgf_r[0]);
   ParallelECoefficient par_e_i(pgf_i[0]);
   E_par_r.ProjectCoefficient(par_e_r);
   E_par_i.ProjectCoefficient(par_e_i);
   
   // // rescale the J solutions
   for (int i = 0; i < 2*ndiffusionequations; ++i)
   {
      (*pgf_r[2+i]) /= balance_scale;
      (*pgf_i[2+i]) /= balance_scale;
   }

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


   delete a;
   for (int i = 0; i < trial_fecols.Size(); ++i)
   {
      delete trial_fecols[i];
      delete pfes[i];
   }
   for (int i = 0; i< test_fecols.Size(); ++i)
   {
      delete test_fecols[i];
   }

   return 0;
}
