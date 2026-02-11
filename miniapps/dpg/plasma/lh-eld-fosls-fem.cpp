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
//                   MFEM FOSLS-FEM Maxwell parallel example
//
// Compile with: make lh-eld-fosls-fem
//
// mpirun -np 8 ./lh-eld-fosls-fem -o 4 -paraview -m data/quad.msh -ebs -sc
// Electron Landau Damping
// Strong formulation:
//     ∇×(1/μ₀∇×E) - ω² ϵ₀ ϵ E + i ω²ϵ₀(J₁ + J₂) = 0,   in Ω
//     subject to the constraints
//                   - Δ∥ J₁ + c₁ J₁ - c₁ P(r) E∥ = 0,   in Ω     
//                   - Δ∥ J₂ + c₂ J₂ + c₂ P(r) E∥ = 0,   in Ω 
//                                            E×n = E₀,  on ∂Ω
//                                            J₁  = 0,  on ∂Ω
//                                            J₂  = 0,  on ∂Ω
// The DPG-FOSLS deals with the First Order System
//  i ω μ₀ H  + ∇ × E                  = 0,   in Ω
// -i ω ϵ₀ϵ E + ∇ × H - ω ϵ₀ (J₁ + J₂) = 0,   in Ω
// subject to the constraints
//         - Δ∥ J₁ + c₁ J₁ - c₁ P(r) E∥ = 0,   in Ω     
//         - Δ∥ J₂ + c₂ J₂ + c₂ P(r) E∥ = 0,   in Ω 
//                                 E×n = E₀,  on ∂Ω
//                                 J₁  = 0,  on ∂Ω
//                                 J₂  = 0,  on ∂Ω


// in 2D  E is vector valued and H is scalar and
//     ∇ × E = ∇ ⋅ AE  where A = [0 1; -1 0];

// E ∈ H(curl,Ω) , H ∈ H¹(Ω), Jᵢ, Qᵢ ∈ (H¹(Ω))²
// minimize the FOSLS functional:
//  ( iωμ₀ H, F ) + ( ∇ × E, F) = 0,                      ∀ F  ∈  L²(Ω)
// -(iωϵ₀ϵ E, R ) + ( ∇ × H, R) - ω ϵ₀ (J₁ + J₂, R) = 0,  ∀ R  ∈ (L²(Ω))²
// subject to the constraints
//  (b⋅∇J₁, b ⋅ ∇K₁) + c₁(J₁, K₁) - c₁ (P B E, K₁) = 0,      ∀ K₁ ∈ (H¹(Ω))²
//  (b⋅∇J₂, b ⋅ ∇K₂) + c₂(J₂, K₂) + c₂ (P B E, K₂) = 0,      ∀ K₂ ∈ (H¹(Ω))²
//                                    E = E₀, on ∂Ω
//                              J₁ = J₂ = 0,  on ∂Ω


// We formulate the problem as constrained minimization
// i.e, we minimize ||A U - F|| subject to B U = 0
// where U = [E, H, J₁, J₂]ᵀ
// A and B are given by:
// -----------------------------------------------

// A (FOSLS)       
// ----------------------------------------------------------
// |   |      E       |      H    |     J₁      |     J₂    |       
// ----------------------------------------------------------
// | F |  (∇ × E, F)  | (iωμ₀H,F) |             |           |   
// |   |              |           |             |           |   
// | R | -iωϵ₀ϵ(E,R)  |(∇ × H, R) | -ωϵ₀(J₁,R)  |-ωϵ₀(J₂, R)|   

// B (Constraints)
// --------------------------------------------------------------------------------
// |   |      E       |                 J₁           |                J₂          |       
// --------------------------------------------------------------------------------
// |K₁ | -c₁(PBE,K₁)  | (b⋅∇J₁, b⋅∇K₁) + c₁(J₁, K₁)  |                            |       
// |   |              |                              |                            |       
// |K₂ |  c₂(PBE, K₂) |                              |(b⋅∇J₂, b⋅∇K₂) + c₂(J₂, K₂) |       

// The saddle point system is then given by:
// | A   B̄ᵀ | |U| = |F|
// | B   0  | |λ|   |0|


#include "mfem.hpp"
#include "../util/pcomplexweakform.hpp"
#include "../util/pcomplexblockform.hpp"
#include "../util/blockcomplexhypremat.hpp"
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
   real_t cfactor = 1e-6;
   real_t balance_scale = 1.0;
   bool enable_balance_scale = false;

   real_t delta_prec = 0.0;


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
   args.AddOption(&delta_prec, "-dp", "--delta-prec", "stability parameter for the preconditioner.");
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
   int ndiffusionequations = 2; 

   Vector cvals(ndiffusionequations);
   Vector csigns(ndiffusionequations);
   cvals(0)  = 25e6;  cvals(1)  = 1e6;
   csigns(0) = -1.0;  csigns(1) = 1.0;

   cvals *= cfactor; // scale the coefficients
   real_t omega = 2.*M_PI*rnum;
   int test_order = order+delta_order;

   balance_scale = (enable_balance_scale) ? eps0 : 1.0;

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
   ScalarVectorProductCoefficient scaled_b_cf(sqrt(cfactor), b_cf);
   ConstantCoefficient diff_coeff(cfactor);

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


   std::vector<std::string> fosls_variables = {" E  ", " H  ", " J₁ ", " J₂ "};
   std::vector<std::string> fosls_spaces = {"H(curl,Ω) ", "   H¹(Ω)  ", 
                                            "  (H¹(Ω))ᵈ", "  (H¹(Ω))ᵈ"}; 
   std::vector<std::string> constraints_variables = {" Q₁ ", " Q₂ "};
   std::vector<std::string> constraints_spaces = {"  (H¹(Ω))ᵈ", "  (H¹(Ω))ᵈ"};                                             

   // Define the spaces
   Array<FiniteElementCollection *> fosls_trial_fecols;
   Array<FiniteElementCollection *> fosls_test_fecols;
   Array<ParFiniteElementSpace *> fosls_pfes;
   Array<FiniteElementCollection *> constraints_fecols;
   Array<ParFiniteElementSpace *> constraints_pfes;

   // H(curl) space for E
   fosls_trial_fecols.Append(new ND_FECollection(order, dim));
   fosls_pfes.Append(new ParFiniteElementSpace(&pmesh, fosls_trial_fecols.Last()));
   // Scalar H¹ space for H
   fosls_trial_fecols.Append(new H1_FECollection(order, dim));
   fosls_pfes.Append(new ParFiniteElementSpace(&pmesh, fosls_trial_fecols.Last()));
   // Vector H¹ spaces for Jᵢ
   for (int i = 0; i < ndiffusionequations; i++)
   {
      fosls_trial_fecols.Append(new H1_FECollection(order, dim));
      fosls_pfes.Append(new ParFiniteElementSpace(&pmesh, fosls_trial_fecols.Last(), dim));
      constraints_fecols.Append(new H1_FECollection(order, dim));
      constraints_pfes.Append(new ParFiniteElementSpace(&pmesh, constraints_fecols.Last(), dim));
   }

   Array<HYPRE_BigInt> tdofs(fosls_pfes.Size());
   Array<HYPRE_BigInt> constraints_tdofs(constraints_pfes.Size());
   for (int i = 0; i < fosls_pfes.Size(); ++i)
   {
      tdofs[i] = fosls_pfes[i]->GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << " FOSLS      FE Space " << fosls_spaces[i] << "  for " << fosls_variables[i] << " has " << tdofs[i]
              << " true dofs." << endl;
      }
   }
   for (int i = 0; i < constraints_pfes.Size(); ++i)
   {
      constraints_tdofs[i] = constraints_pfes[i]->GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << " Constraint FE Space " << constraints_spaces[i] << "  for " << constraints_variables[i] << " has " << constraints_tdofs[i]
              << " true dofs." << endl;
      }
   }
   if (Mpi::Root())
   {
      cout << "Total number of true dofs: " << tdofs.Sum() << endl;
   }

   // test spaces for F and R
   fosls_test_fecols.Append(new L2_FECollection(test_order, dim));
   fosls_test_fecols.Append(new L2_FECollection(test_order, dim));

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(fosls_pfes,fosls_test_fecols);
   a->SetTestFECollVdim(1,dim); 

   // (∇ × E, F)
   a->AddTrialIntegrator(new MixedCurlIntegrator(one_cf), nullptr, 0, 0);

   // (i ω μ₀ H, F)
   a->AddTrialIntegrator(nullptr, new MixedScalarMassIntegrator(omegamu_cf), 1, 0);

   // -i ω ϵ₀ϵ (E, R) = ω ϵ₀ ϵᵢ (E, G) + i (-ω ϵ₀ ϵᵣ E, R)
   a->AddTrialIntegrator(new VectorFEMassIntegrator(eps0omeg_eps_i),
                         new VectorFEMassIntegrator(negeps0omeg_eps_r), 0, 1);

   // (∇ × H, R)
   a->AddTrialIntegrator(new MixedCurlIntegrator(one_cf), nullptr, 1,1);
   
   // - ω ϵ₀ (J₁, R)
   a->AddTrialIntegrator(new VectorMassIntegrator(balancescaled_negomegeps0_cf), nullptr, 2, 1);
   // - ω ϵ₀ (J₂, R)
   a->AddTrialIntegrator(new VectorMassIntegrator(balancescaled_negomegeps0_cf), nullptr, 3, 1);

   // test integrators for test norm: ||v||² = ||F||² + ||R||² + ||K₁||² + ||L₁||² + ||K₂||² + ||L₂||² 
   // (F,δF), F, δF ∈  L²(Ω)
   a->AddTestIntegrator(new MassIntegrator(one_cf),nullptr, 0, 0);
   // (R,δR), R, δR ∈ (L²(Ω))²
   a->AddTestIntegrator(new VectorMassIntegrator(one_cf),nullptr, 1, 1);

   a->Assemble(0);


   // Constraint matrix B
   ParMixedBilinearForm bEK1_r(fosls_pfes[0], constraints_pfes[0]);
   ParMixedBilinearForm bEK1_i(fosls_pfes[0], constraints_pfes[0]);

   // - c₁ (P B E, K₁)
   bEK1_r.AddDomainIntegrator(new VectorFEMassIntegrator(*balancescaled_signedcPrbb_cf[0]));
   bEK1_i.AddDomainIntegrator(new VectorFEMassIntegrator(*balancescaled_signedcPibb_cf[0]));

   bEK1_r.Assemble();
   bEK1_i.Assemble();

   ParMixedBilinearForm bK2H_r(constraints_pfes[1], fosls_pfes[1]);
   bK2H_r.Assemble();

   ParMixedBilinearForm bEK2_r(fosls_pfes[0], constraints_pfes[1]);
   ParMixedBilinearForm bEK2_i(fosls_pfes[0], constraints_pfes[1]);
   // c₂(PBE, K₂)
   bEK2_r.AddDomainIntegrator(new VectorFEMassIntegrator(*balancescaled_signedcPrbb_cf[1]));    
   bEK2_i.AddDomainIntegrator(new VectorFEMassIntegrator(*balancescaled_signedcPibb_cf[1]));    

   bEK2_r.Assemble();
   bEK2_i.Assemble();

   // (b⋅∇J₁, b⋅∇K₁) + c₁(J₁, K₁) 
   ParBilinearForm bJ1K1(constraints_pfes[0]);
   bJ1K1.AddDomainIntegrator(new DirectionalVectorDiffusionIntegrator(scaled_b_cf));
   bJ1K1.AddDomainIntegrator(new VectorMassIntegrator(*pw_c_coeffs[0]));
   bJ1K1.Assemble();

   ParBilinearForm bJ2K2(constraints_pfes[1]);
   bJ2K2.AddDomainIntegrator(new DirectionalVectorDiffusionIntegrator(scaled_b_cf));
   bJ2K2.AddDomainIntegrator(new VectorMassIntegrator(*pw_c_coeffs[1]));
   bJ2K2.Assemble();


   int fosls_npfes = fosls_pfes.Size();
   Array<int> fosls_offsets(fosls_npfes+1);  fosls_offsets[0] = 0;
   Array<int> fosls_toffsets(fosls_npfes+1); fosls_toffsets[0] = 0;
   for (int i = 0; i<fosls_npfes; i++)
   {
      fosls_offsets[i+1] = fosls_pfes[i]->GetVSize();
      fosls_toffsets[i+1] = fosls_pfes[i]->TrueVSize();
   }
   fosls_offsets.PartialSum();
   fosls_toffsets.PartialSum();

   Array<int> empty;
   OperatorPtr fosls_Aop;

   a->FormSystemMatrix(empty, fosls_Aop);
   ComplexOperator * fosls_Ac = fosls_Aop.As<ComplexOperator>();

   OperatorPtr constraintOp_EK1_r, constraintOp_EK1_i;
   bEK1_r.FormRectangularSystemMatrix(empty, empty, constraintOp_EK1_r);
   bEK1_i.FormRectangularSystemMatrix(empty, empty, constraintOp_EK1_i);

   OperatorPtr constraintOp_EK2_r, constraintOp_EK2_i;
   bEK2_r.FormRectangularSystemMatrix(empty, empty, constraintOp_EK2_r);
   bEK2_i.FormRectangularSystemMatrix(empty, empty, constraintOp_EK2_i);

   OperatorPtr constraintOp_J1K1;
   bJ1K1.FormSystemMatrix(empty, constraintOp_J1K1);
   OperatorPtr constraintOp_J2K2;
   bJ2K2.FormSystemMatrix(empty, constraintOp_J2K2);




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
   all_pfes.Append(fosls_pfes);
   all_pfes.Append(constraints_pfes);

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


   // put all the operators into a block operator
   BlockOperator A_r(all_toffsets);
   BlockOperator A_i(all_toffsets);

   BlockOperator * BlockAfosls_r = dynamic_cast<BlockOperator *>(&fosls_Ac->real());
   BlockOperator * BlockAfosls_i = dynamic_cast<BlockOperator *>(&fosls_Ac->imag());


   for (int i = 0; i < fosls_pfes.Size(); ++i)
   {
      for (int j = 0; j < fosls_pfes.Size(); ++j)
      {
         A_r.SetBlock(i, j, &BlockAfosls_r->GetBlock(i, j));
         A_i.SetBlock(i, j, &BlockAfosls_i->GetBlock(i, j));
      }
   }


   A_r.SetBlock(fosls_pfes.Size()+0, 0, constraintOp_EK1_r.Ptr());
   A_i.SetBlock(fosls_pfes.Size()+0, 0, constraintOp_EK1_i.Ptr());

   A_r.SetBlock(fosls_pfes.Size()+1, 0, constraintOp_EK2_r.Ptr());
   A_i.SetBlock(fosls_pfes.Size()+1, 0, constraintOp_EK2_i.Ptr());

   A_r.SetBlock(fosls_pfes.Size()+0, 2, constraintOp_J1K1.Ptr());
   A_r.SetBlock(fosls_pfes.Size()+1, 3, constraintOp_J2K2.Ptr());

   // Construct the adjoint operator B̄ᵀ
   HypreParMatrix * constraintOp_EK1_r_t = constraintOp_EK1_r.As<HypreParMatrix>()->Transpose();
   HypreParMatrix * constraintOp_EK1_i_t = constraintOp_EK1_i.As<HypreParMatrix>()->Transpose();
   HypreParMatrix * constraintOp_EK2_r_t = constraintOp_EK2_r.As<HypreParMatrix>()->Transpose();
   HypreParMatrix * constraintOp_EK2_i_t = constraintOp_EK2_i.As<HypreParMatrix>()->Transpose();
   HypreParMatrix * constraintOp_J1K1_t = constraintOp_J1K1.As<HypreParMatrix>()->Transpose();
   HypreParMatrix * constraintOp_J2K2_t = constraintOp_J2K2.As<HypreParMatrix>()->Transpose();

   // We also need to scale the imaginary part of B̄ᵀ by -1;
   *constraintOp_EK1_i_t *= -1.0;
   *constraintOp_EK2_i_t *= -1.0;

   A_r.SetBlock(0, fosls_pfes.Size()+0, constraintOp_EK1_r_t);
   A_i.SetBlock(0, fosls_pfes.Size()+0, constraintOp_EK1_i_t);    
   A_r.SetBlock(0, fosls_pfes.Size()+1, constraintOp_EK2_r_t);
   A_i.SetBlock(0, fosls_pfes.Size()+1, constraintOp_EK2_i_t);
   A_r.SetBlock(2, fosls_pfes.Size()+0, constraintOp_J1K1_t);
   A_r.SetBlock(3, fosls_pfes.Size()+1, constraintOp_J2K2_t);

   //    // dammies for other coupling terms in B
   // ParMixedBilinearForm bHK1_r(fosls_pfes[1], constraints_pfes[0]); bHK1_r.Assemble();
   // ParMixedBilinearForm bHK1_i(fosls_pfes[1], constraints_pfes[0]); bHK1_i.Assemble();
   // OperatorPtr constraintOp_HK1_r, constraintOp_HK1_i;
   // bHK1_r.FormRectangularSystemMatrix(empty, empty, constraintOp_HK1_r);
   // bHK1_i.FormRectangularSystemMatrix(empty, empty, constraintOp_HK1_i);
   // A_r.SetBlock(fosls_pfes.Size()+0, 1, constraintOp_HK1_r.Ptr());
   // A_i.SetBlock(fosls_pfes.Size()+0, 1, constraintOp_HK1_i.Ptr());
   
   // ParMixedBilinearForm bHK2_r(fosls_pfes[1], constraints_pfes[1]); bHK2_r.Assemble(); 
   // ParMixedBilinearForm bHK2_i(fosls_pfes[1], constraints_pfes[1]); bHK2_i.Assemble();
   
   // ParMixedBilinearForm bK1H_r(constraints_pfes[0], fosls_pfes[1]); bK1H_r.Assemble();
   // ParMixedBilinearForm bK1H_i(constraints_pfes[0], fosls_pfes[1]); bK1H_i.Assemble();
   // ParMixedBilinearForm bK2H_r(fosls_pfes[1], constraints_pfes[1]); bK2H_r.Assemble();
   // ParMixedBilinearForm bK2H_i(fosls_pfes[1], constraints_pfes[0]); bK2H_i.Assemble();



   // ParBilinearForm bJ1K2(constraints_pfes[0]);




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
   std::string output_dir = "ParaView/FOSLS-FEM/" + GetTimestamp();


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
      paraview_dc->RegisterField("Er",pgf_r[0]);
      paraview_dc->RegisterField("Ei",pgf_i[0]);
      paraview_dc->RegisterField("E_par_r",&E_par_r);
      paraview_dc->RegisterField("E_par_i",&E_par_i);
      paraview_dc->RegisterField("Hr",pgf_r[1]);
      paraview_dc->RegisterField("Hi",pgf_i[1]);      
      paraview_dc->RegisterField("J1r",pgf_r[2]);
      paraview_dc->RegisterField("J1i",pgf_i[2]);
      paraview_dc->RegisterField("J2r",pgf_r[3]);
      paraview_dc->RegisterField("J2i",pgf_i[3]);
      paraview_dc->RegisterField("Q1r",pgf_r[4]);
      paraview_dc->RegisterField("Q1i",pgf_i[4]);
      paraview_dc->RegisterField("Q2r",pgf_r[5]);
      paraview_dc->RegisterField("Q2i",pgf_i[5]);
   }
   Array<int> ess_tdof_list;
   Array<int> ess_tdof_listJ;
   Array<int> ess_tdof_listQ;
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
      all_pfes[0]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += all_toffsets[0];
      }
      ess_bdr = 1;
      for (int i = 0; i<ndiffusionequations;i++)
      {
         ess_tdof_listJ.SetSize(0);
         ess_tdof_listQ.SetSize(0);
         all_pfes[i+2]->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ);
         all_pfes[i+4]->GetEssentialTrueDofs(ess_bdr, ess_tdof_listQ);
         for (int j = 0; j < ess_tdof_listJ.Size(); j++)
         {
            ess_tdof_listJ[j] += all_toffsets[i+2];
         }
         for (int j = 0; j < ess_tdof_listQ.Size(); j++)
         {
            ess_tdof_listQ[j] += all_toffsets[i+4];
         }
         ess_tdof_list.Append(ess_tdof_listJ);
         ess_tdof_list.Append(ess_tdof_listQ);
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
   ess_tdof_list.Print(mfem::out);

   if (Mpi::Root())
   {
      mfem::out << "Eliminate BC finished successfully." << endl;
   }

   bool direct_solve = true;

   if (direct_solve)
   {
      BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&A->real());
      BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&A->imag());

      int nblocks = BlockA_r->NumRowBlocks();
      Array2D<const HypreParMatrix*> A_r_matrices(nblocks, nblocks);
      Array2D<const HypreParMatrix*> A_i_matrices(nblocks, nblocks);
      for (int i = 0; i < nblocks; i++)
      {
         for (int j = 0; j < nblocks; j++)
         {
            if (BlockA_r->IsZeroBlock(i,j))
            {
               A_r_matrices(i,j) = nullptr;
            }
            else
            {
               A_r_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_r->GetBlock(i,j));
            }
            if (BlockA_i->IsZeroBlock(i,j))
            {
               A_i_matrices(i,j) = nullptr;
            }
            else
            {
               A_i_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_i->GetBlock(i,j));
            }   
         }
      }
      HypreParMatrix * Ahr = HypreParMatrixFromBlocks(A_r_matrices);
      HypreParMatrix * Ahi = HypreParMatrixFromBlocks(A_i_matrices);

      ComplexHypreParMatrix * Ahc_hypre =
         new ComplexHypreParMatrix(Ahr, Ahi,true, true);

      ComplexMUMPSSolver cmumps(MPI_COMM_WORLD);
      cmumps.SetPrintLevel(1);
      cmumps.SetOperator(*Ahc_hypre);

      if (Mpi::Root())
      {
         mfem::out << "Setup CMUMPS solver finished successfully." << endl;
      }

      cmumps.Mult(B,X);
      delete Ahc_hypre;   
   }
   else
   {
      MFEM_ABORT("to be implemented: iterative solver with preconditioner");
   }


   if (Mpi::Root())
   {
      mfem::out << "Solve finished successfully." << endl;
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



   return 0;


   


   // a->RecoverFEMSolution(X, x);

   // for (int i = 0; i < npfes; ++i)
   // {
   //    pgf_r[i]->MakeRef(pfes[i], x, offsets[i]);
   //    pgf_i[i]->MakeRef(pfes[i], x, offsets.Last() + offsets[i]);
   // }
   
   // ParallelECoefficient par_e_r(pgf_r[0]);
   // ParallelECoefficient par_e_i(pgf_i[0]);
   // E_par_r.ProjectCoefficient(par_e_r);
   // E_par_i.ProjectCoefficient(par_e_i);
   
   // // // rescale the J solutions
   // for (int i = 0; i < 2*ndiffusionequations; ++i)
   // {
   //    (*pgf_r[2+i]) /= balance_scale;
   //    (*pgf_i[2+i]) /= balance_scale;
   // }

   // if (visualization)
   // {
   //    const char * keys = nullptr;
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    common::VisualizeField(E_out_r,vishost, visport, *pgf_r[0],
   //                           "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
   // }

   // if (paraview)
   // {
   //    paraview_dc->SetCycle(0);
   //    paraview_dc->SetTime((real_t)0);
   //    paraview_dc->Save();
   //    delete paraview_dc;
   // }


   // delete a;
   // for (int i = 0; i < trial_fecols.Size(); ++i)
   // {
   //    delete trial_fecols[i];
   //    delete pfes[i];
   // }
   // for (int i = 0; i< test_fecols.Size(); ++i)
   // {
   //    delete test_fecols[i];
   // }

   // return 0;
}
