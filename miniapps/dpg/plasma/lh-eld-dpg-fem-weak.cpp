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
//                   MFEM Ultraweak DPG Maxwell parallel example
//
// Compile with: make lh-eld-weak
//
// mpirun -np 8 ./lh-eld-weak -o 3 -paraview -pr 0
// mpirun -np 8 ./lh-eld-weak -o 3 -paraview -pr 1 -sc

// Electron Landau Damping

// Weak Coupling Formulation:
// Strong formulation:
//     ∇×(1/μ₀∇×E) - ω² ϵ₀ ϵ E = - i ω²ϵ₀(J₁ + J₂),   in Ω
//                         E×n = E₀,  on ∂Ω
//             - Δ∥ J₁ + c₁ J₁ =  c₁ P(r) E∥,   in Ω     
//             - Δ∥ J₂ + c₂ J₂ = -c₂ P(r) E∥,   in Ω 
//                         J₁  = 0,  on ∂Ω
//                         J₂  = 0,  on ∂Ω
// The DPG UW deals with the First Order System
//  i ω μ₀ H  + ∇ × E = 0,   in Ω
// -i ω ϵ₀ϵ E + ∇ × H = ω ϵ₀ (J₁ + J₂),   in Ω
//                E×n = E₀,  on ∂Ω

//     - Δ∥ J₁ + c₁ J₁ =  c₁ P(r) E∥,   in Ω     
//     - Δ∥ J₂ + c₂ J₂ = -c₂ P(r) E∥,   in Ω 
//                 J₁  = 0,  on ∂Ω
//                 J₂  = 0,  on ∂Ω


// in 2D
// E is vector valued and H is scalar.
//      (∇ × E, δE) = (E, ∇ × δE ) + < n × E , δE >
// or (∇ ⋅ AE , δE) = (AE, ∇ δE ) + < AE ⋅ n, δE >
// where A = [0 1; -1 0];

// E ∈ (L²(Ω))² , H ∈ L²(Ω), J ∈ (H¹(Ω))²
// Ê ∈ H^-1/2(Γₕ), Ĥ ∈ H^1/2(Γₕ), Ĵ₁, Ĵ₂ ∈ (H^-1/2(Γₕ))²
//     iωμ₀ (H,δE) + (E,∇×δE) + < AÊ, δE > = 0,      ∀ δE ∈ H¹(Ω)
//  -i ωϵ₀ϵ (E,δH) + (H,∇×δH) + < Ĥ, δH×n > = ω ϵ₀ (J₁ + J₂,δH),  ∀ δH ∈ H(curl,Ω)
//                                        Ê = E₀, on ∂Ω


// ( (b⋅∇)J₁,(b⋅∇) δJ₁ ) + c₁ (J₁,δJ₁) - c₁ (P(r) b⊗b E, δJ₁) = 0,  ∀ δJ₁ ∈ (H¹(Ω))²
// ( (b⋅∇)J₂,(b⋅∇) δJ₂ ) + c₂ (J₂,δJ₂) + c₂ (P(r) b⊗b E, δJ₂) = 0,  ∀ δJ₂ ∈ (H¹(Ω))²
//                                                    J₁ = J₂ = 0,  on ∂Ω
// -----------------------------------------------------------------
// |  |    E       |     H    |   Ê  |    Ĥ   |          RHS        |
// -----------------------------------------------------------------
// |δE| (E,∇ × δE) |iωμ₀(H,δE)|<Ê,δE>|        |          0          |  
// |  |            |          |      |        |                     |  
// |δH|-iωϵ₀ϵ(E,δH)| (H,∇×δH) |      |<Ĥ,δH×n>|ωϵ₀(J₁,δH)+ωϵ₀(J₂,δH)|  

// -------------------------------------------------------------------------------
// |               J₁                |                 J₂           |      RHS    |
// -------------------------------------------------------------------------------
// |δJ₁|((b⋅∇)J₁,(b⋅∇)δJ₁)+c₁(J₁,δJ₁)|                              |c₁(P(r)E,δJ₁)|  
// |   |                             |                              |             |   
// |δJ₂|                             |((b⋅∇)J₂,(b⋅∇)δJ₂)+ c₂(J₂,δJ₂)|c₂(P(r)E,δJ₂)|  
// where (δE,δH,δJ₁,δJ₂) ∈  H¹(Ω) × H(curl,Ω) × (H¹(Ω))² × (H¹(Ω))² 

#include "mfem.hpp"
#include "../util/pcomplexweakform.hpp"
#include "../util/pcomplexblockform.hpp"
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

   bool static_cond = false;
   bool visualization = false;
   bool paraview = false;
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
   args.AddOption(&delta, "-delta", "--delta", "stability parameter.");
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

   balance_scale = (enable_balance_scale) ? omega * eps0 * omega : 1.0;

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
   // ωϵ₀
   ConstantCoefficient omegeps0_cf(omega*eps0);
   ConstantCoefficient balancescaled_omegeps0_cf(omega*eps0/balance_scale);

   ConstantCoefficient negomegeps0_cf(-omega*eps0);
   ConstantCoefficient balancescaled_negomegeps0_cf( -omega*eps0/balance_scale);
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
   ConstantCoefficient diff_coeff(cfactor);

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

   // Define the spaces
   Array<FiniteElementCollection *> dpg_trial_fecols;
   Array<FiniteElementCollection *> dpg_test_fecols;
   Array<ParFiniteElementSpace *> dpg_pfes;

   Array<FiniteElementCollection *> fem_trial_fecols;
   Array<ParFiniteElementSpace *> fem_pfes;


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

   // test spaces for E and H
   dpg_test_fecols.Append(new H1_FECollection(test_order, dim));
   dpg_test_fecols.Append(new ND_FECollection(test_order, dim));


   // Vector H1 spaces for Jᵢ 
   for (int i = 0; i < ndiffusionequations; i++)
   {
      fem_trial_fecols.Append(new H1_FECollection(order, dim));
      fem_pfes.Append(new ParFiniteElementSpace(&pmesh, fem_trial_fecols.Last(), dim));
   }


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
      cout << "Total number of DPG true dofs for Maxwell:    " << dpg_tdofs.Sum() << endl;
      cout << "Total number of FEM true dofs for VDiffusion: " << fem_tdofs.Sum() << endl;
   }

   int ndpg_pfes = dpg_pfes.Size();
   int nfem_pfes = fem_pfes.Size();

   Array<int> dpg_offsets(ndpg_pfes+1);  dpg_offsets[0] = 0;
   Array<int> dpg_toffsets(ndpg_pfes+1); dpg_toffsets[0] = 0;
   for (int i = 0; i<ndpg_pfes; i++)
   {
      dpg_offsets[i+1] = dpg_pfes[i]->GetVSize();
      dpg_toffsets[i+1] = dpg_pfes[i]->TrueVSize();
   }
   dpg_offsets.PartialSum();
   dpg_toffsets.PartialSum();

   Array<int> fem_offsets(nfem_pfes+1);  fem_offsets[0] = 0;
   Array<int> fem_toffsets(nfem_pfes+1); fem_toffsets[0] = 0;
   for (int i = 0; i<nfem_pfes; i++)
   {
      fem_offsets[i+1] = fem_pfes[i]->GetVSize();
      fem_toffsets[i+1] = fem_pfes[i]->TrueVSize();
   }
   fem_offsets.PartialSum();
   fem_toffsets.PartialSum();

   Vector dpg_x(2*dpg_offsets.Last());
   dpg_x = 0.;
   Vector fem_x(2*fem_offsets.Last());
   fem_x = 0.;

   Vector fem_b(2*fem_offsets.Last());
   fem_b = 0.;

   Array<ParGridFunction *> dpg_pgf_r(ndpg_pfes);
   Array<ParGridFunction *> dpg_pgf_i(ndpg_pfes);
   Array<ParGridFunction *> fem_pgf_r(nfem_pfes);
   Array<ParGridFunction *> fem_pgf_i(nfem_pfes);

   for (int i = 0; i < ndpg_pfes; ++i)
   {
      dpg_pgf_r[i] = new ParGridFunction(dpg_pfes[i], dpg_x, dpg_offsets[i]);
      dpg_pgf_i[i] = new ParGridFunction(dpg_pfes[i], dpg_x, dpg_offsets.Last() + dpg_offsets[i]);
   }
   for (int i = 0; i < nfem_pfes; ++i)
   {
      fem_pgf_r[i] = new ParGridFunction(fem_pfes[i], fem_x, fem_offsets[i]);
      fem_pgf_i[i] = new ParGridFunction(fem_pfes[i], fem_x, fem_offsets.Last() + fem_offsets[i]);
   }

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);
   ParGridFunction E_par_r(&L2_fes);
   ParGridFunction E_par_i(&L2_fes);

   ParaViewDataCollection * paraview_dc = nullptr;
   std::string output_dir = "ParaView/UW/WeakCoupling/" + GetTimestamp();

   if (paraview)
   {
      if (Mpi::Root()) { WriteParametersToFile(args, output_dir); }
      std::ostringstream paraview_file_name;
      std::string filename = GetFilename(mesh_file);
      paraview_file_name << filename
                         << "_par_ref_" << par_ref_levels
                         << "_order_" << order;
      paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &pmesh);
      paraview_dc->SetPrefixPath(output_dir);
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",dpg_pgf_r[0]);
      paraview_dc->RegisterField("E_i",dpg_pgf_i[0]);
      paraview_dc->RegisterField("E_par_r",&E_par_r);
      paraview_dc->RegisterField("E_par_i",&E_par_i);
      paraview_dc->RegisterField("H_r",dpg_pgf_r[1]);
      paraview_dc->RegisterField("H_i",dpg_pgf_i[1]);      
      paraview_dc->RegisterField("Jh_1_r",fem_pgf_r[0]);
      paraview_dc->RegisterField("Jh_1_i",fem_pgf_i[0]);
      paraview_dc->RegisterField("Jh_2_r",fem_pgf_r[1]);
      paraview_dc->RegisterField("Jh_2_i",fem_pgf_i[1]);
      paraview_dc->Save();
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

   int max_fixed_point_iter = 10;
   // loop through fixed point iterations
   for (int k = 0; k<max_fixed_point_iter; k++)
   {
      dpg_x = 0.0;

      // reset GridFunctions
      for (int i = 0; i < ndpg_pfes; ++i)
      {
         *dpg_pgf_r[i] = 0.0;
         *dpg_pgf_i[i] = 0.0;
      }

      delta = (k == 0) ? 0.01 : 0.0;
      // delta = 0.01 * pow(0.5, k);
      ParComplexDPGWeakForm a_dpg(dpg_pfes,dpg_test_fecols);
      if (static_cond) { a_dpg.EnableStaticCondensation(); }

      // (E,∇ × δE)
      a_dpg.AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one_cf)),
                               nullptr,0, 0);
      //  -i ω ϵ₀ (ϵE,δH) = - i ω ϵ₀(ϵᵣ + i ϵᵢ E, δH)
      //                  =  (ω ϵ₀ ϵᵢ E, δH) + i (-ω ϵ₀ϵᵣ E, δH)
      a_dpg.AddTrialIntegrator(
         new TransposeIntegrator(new VectorFEMassIntegrator(eps0omeg_eps_i)), 
         new TransposeIntegrator(new VectorFEMassIntegrator(negeps0omeg_eps_r)),
                                 0,1);
      // iωμ₀(H,δE) 
      a_dpg.AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(omegamu_cf),1, 0);
      // (H,∇ × δH)                         
      a_dpg.AddTrialIntegrator(
         new TransposeIntegrator(new MixedCurlIntegrator(one_cf)), nullptr,1, 1);

      // Trace integrators               
      //  <Ê,δE>
      a_dpg.AddTrialIntegrator(new TraceIntegrator,nullptr,2,0);
      // <Ĥ,δH × n>
      a_dpg.AddTrialIntegrator(new TangentTraceIntegrator,nullptr,3,1);
   
      // test integrators
      // (∇δE,∇δE)
      a_dpg.AddTestIntegrator(new DiffusionIntegrator(one_cf),nullptr,0,0);
      // (δE,δE)
      a_dpg.AddTestIntegrator(new MassIntegrator(one_cf),nullptr,0,0);
      // μ₀² ω² (δE,δE)
      a_dpg.AddTestIntegrator(new MassIntegrator(mu2omeg2_cf),nullptr,0, 0);
      // -i ω μ₀ (δE,∇ × δH) = i (δE, -ω μ₀ ∇ × δ H)
      a_dpg.AddTestIntegrator(nullptr,
         new TransposeIntegrator(new MixedCurlIntegrator(negomegamu_cf)),0, 1);
      // -i ω ϵ₀ϵ(∇ × δE, δH) = -i (ωϵ₀(ϵᵣ+iϵᵢ) A ∇ δE,δE), A = [0 1; -1 0]
      //                       =  (ω ϵ₀ ϵᵢ A ∇ δE,δE) + i (-ω ϵ₀ ϵᵣ A ∇ δE,δE)
      a_dpg.AddTestIntegrator(new MixedVectorGradientIntegrator(eps0omeg_eps_i_rot),
                              new MixedVectorGradientIntegrator(negeps0omeg_eps_r_rot),0, 1);
      // i ω μ₀ (∇ × δH ,δE) = i (ω μ₀ ∇ × δH, δE )
      a_dpg.AddTestIntegrator(nullptr,new MixedCurlIntegrator(omegamu_cf),1,0);
      // i ω ϵ₀ϵ̄ (δH, ∇ × δE ) = i (ω ϵ₀(ϵᵣ -i ϵᵢ) δH, A ∇ δE) 
      //                       = ( δH, ω ϵ₀ ϵᵢ A ∇ δE) + i (δH, ω ϵ₀ ϵᵣ A ∇ δE)
      a_dpg.AddTestIntegrator(
      new TransposeIntegrator(new MixedVectorGradientIntegrator(eps0omeg_eps_i_rot)),
      new TransposeIntegrator(new MixedVectorGradientIntegrator(eps0omeg_eps_r_rot)),1, 0);
      // (ωϵ₀ϵ)(ωϵ₀ϵ)^*  (δH, δH)
      // (MᵣMᵣᵗ + MᵢMᵢᵗ) + i (MᵢMᵣᵗ - MᵣMᵢᵗ)
      a_dpg.AddTestIntegrator(new VectorFEMassIntegrator(Mreal_cf),
                              new VectorFEMassIntegrator(Mimag_cf),1, 1);
      // (∇×δH ,∇×δH)
      a_dpg.AddTestIntegrator(new CurlCurlIntegrator(one_cf),nullptr,1,1);
      // (δH,δH)
      a_dpg.AddTestIntegrator(new VectorFEMassIntegrator(one_cf),nullptr,1,1);   


      VectorGridFunctionCoefficient J1_cf_r(fem_pgf_r[0]);
      VectorGridFunctionCoefficient J1_cf_i(fem_pgf_i[0]);
      VectorGridFunctionCoefficient J2_cf_r(fem_pgf_r[1]);
      VectorGridFunctionCoefficient J2_cf_i(fem_pgf_i[1]);

      // ωϵ₀ (Jᵢ ,δH)
      ScalarVectorProductCoefficient balance_scaled_J1_cf_r(balancescaled_omegeps0_cf, J1_cf_r);
      ScalarVectorProductCoefficient balance_scaled_J1_cf_i(balancescaled_omegeps0_cf, J1_cf_i);
      ScalarVectorProductCoefficient balance_scaled_J2_cf_r(balancescaled_omegeps0_cf, J2_cf_r);
      ScalarVectorProductCoefficient balance_scaled_J2_cf_i(balancescaled_omegeps0_cf, J2_cf_i);

      // DPG RHS   
      // ωϵ₀∑(Jᵢ ,δH)
      mfem::out << fem_pgf_r[0]->Norml2() << " "
                << fem_pgf_i[0]->Norml2() << " "
                << fem_pgf_r[1]->Norml2() << " "
                << fem_pgf_i[1]->Norml2() << endl;
      a_dpg.AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(balance_scaled_J1_cf_r),
                                  new VectorFEDomainLFIntegrator(balance_scaled_J1_cf_i),1);
      a_dpg.AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(balance_scaled_J2_cf_r),
                                 new VectorFEDomainLFIntegrator(balance_scaled_J2_cf_i),1);

      a_dpg.Assemble();


      // remove internal boundaries
      ess_bdr = 1;
      for (int i = 0; i<int_bdr_attr.Size(); i++)
      {
         ess_bdr[int_bdr_attr[i]-1] = 0;
      }

      dpg_pfes[2]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += dpg_toffsets[2];
      }

      dpg_pgf_r[2]->ProjectBdrCoefficientNormal(rot_one_x_cf, one_r_bdr);
      dpg_pgf_r[2]->ProjectBdrCoefficientNormal(rot_negone_x_cf, negone_r_bdr);
      dpg_pgf_i[2]->ProjectBdrCoefficientNormal(rot_one_x_cf, one_i_bdr);
      dpg_pgf_i[2]->ProjectBdrCoefficientNormal(rot_negone_x_cf, negone_i_bdr);

      OperatorPtr dpg_Ah;
      Vector dpg_X,dpg_B;
      a_dpg.FormLinearSystem(ess_tdof_list,dpg_x,dpg_Ah, dpg_X,dpg_B);

      ComplexOperator * dpg_Ahc = dpg_Ah.As<ComplexOperator>();

      BlockOperator * dpg_BlockA_r = dynamic_cast<BlockOperator *>(&dpg_Ahc->real());
      BlockOperator * dpg_BlockA_i = dynamic_cast<BlockOperator *>(&dpg_Ahc->imag());

      int dpg_nblocks = dpg_BlockA_r->NumRowBlocks();
   
      Array2D<const HypreParMatrix*> dpg_A_r_matrices(dpg_nblocks, dpg_nblocks);
      Array2D<const HypreParMatrix*> dpg_A_i_matrices(dpg_nblocks, dpg_nblocks);
      for (int i = 0; i < dpg_nblocks; i++)
      {
         for (int j = 0; j < dpg_nblocks; j++)
         {
            dpg_A_r_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&dpg_BlockA_r->GetBlock(i,j));
            dpg_A_i_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&dpg_BlockA_i->GetBlock(i,j));
         }
      }

      HypreParMatrix * dpg_Ahr = HypreParMatrixFromBlocks(dpg_A_r_matrices);
      HypreParMatrix * dpg_Ahi = HypreParMatrixFromBlocks(dpg_A_i_matrices);

      ComplexHypreParMatrix * dpg_Ahc_hypre =
         new ComplexHypreParMatrix(dpg_Ahr, dpg_Ahi,false, false);

      if (Mpi::Root())
      {
         mfem::out << "DPG Assembly finished successfully." << endl;
      }   

      Array<int> dpg_tdof_offsets(2*dpg_nblocks+1);
      int trace_idx_offset = (static_cond) ? 0 : 2;
      dpg_tdof_offsets[0] = 0;
      for (int i=0; i<dpg_nblocks; i++)
      {
         dpg_tdof_offsets[i+1] = dpg_A_r_matrices(i,i)->Height();
         dpg_tdof_offsets[dpg_nblocks+i+1] = dpg_tdof_offsets[i+1];
      }
      dpg_tdof_offsets.PartialSum();

      BlockDiagonalPreconditioner dpg_M(dpg_tdof_offsets);

      if (!static_cond)
      {
         HypreBoomerAMG * solver_E = new HypreBoomerAMG((HypreParMatrix &)
                                               dpg_BlockA_r->GetBlock(0,0));
         solver_E->SetPrintLevel(0);
         solver_E->SetSystemsOptions(dim);
         HypreBoomerAMG * solver_H = new HypreBoomerAMG((HypreParMatrix &)
                                               dpg_BlockA_r->GetBlock(1,1));
         solver_H->SetPrintLevel(0);
         dpg_M.SetDiagonalBlock(0,solver_E);
         dpg_M.SetDiagonalBlock(1,solver_H);
         dpg_M.SetDiagonalBlock(dpg_nblocks,solver_E);
         dpg_M.SetDiagonalBlock(dpg_nblocks+1,solver_H);
      }
      HypreAMS * solver_hatE =  new HypreAMS(
         (HypreParMatrix &)dpg_BlockA_r->GetBlock(trace_idx_offset, trace_idx_offset), 
         dpg_pfes[2]);
      HypreBoomerAMG * solver_hatH = new HypreBoomerAMG((HypreParMatrix &)
                 dpg_BlockA_r->GetBlock(trace_idx_offset+1,trace_idx_offset+1));
      solver_hatE->SetPrintLevel(0);
      solver_hatH->SetPrintLevel(0);
      solver_hatH->SetRelaxType(88);

      dpg_M.SetDiagonalBlock(trace_idx_offset,solver_hatE);
      dpg_M.SetDiagonalBlock(trace_idx_offset+1,solver_hatH);
      dpg_M.SetDiagonalBlock(trace_idx_offset+dpg_nblocks,solver_hatE);
      dpg_M.SetDiagonalBlock(trace_idx_offset+dpg_nblocks+1,solver_hatH);

      CGSolver dpg_cg(MPI_COMM_WORLD);
      dpg_cg.SetRelTol(1e-10);
      dpg_cg.SetMaxIter(2000);
      dpg_cg.SetPrintLevel(1);
      dpg_cg.SetPreconditioner(dpg_M);
      dpg_cg.SetOperator(*dpg_Ahc_hypre);
      dpg_cg.Mult(dpg_B, dpg_X);


      a_dpg.RecoverFEMSolution(dpg_X, dpg_x);

      for (int i = 0; i < ndpg_pfes; ++i)
      {
         dpg_pgf_r[i]->MakeRef(dpg_pfes[i], dpg_x, dpg_offsets[i]);
         dpg_pgf_i[i]->MakeRef(dpg_pfes[i], dpg_x, dpg_offsets.Last() + dpg_offsets[i]);
      }
   

      fem_x = 0.0;
      fem_b = 0.0;
      for (int i = 0; i < nfem_pfes; ++i)
      {
         *fem_pgf_r[i] = 0.0;
         *fem_pgf_i[i] = 0.0;
      }

      // Diffusion equations
      ParComplexBlockForm a_fem(fem_pfes);
      for (int i = 0; i < ndiffusionequations; i++)
      {
         // ((b⋅∇)Jᵢ, (b⋅∇) δJᵢ)
         a_fem.AddDomainIntegrator(new DirectionalVectorDiffusionIntegrator(scaled_b_cf), 
                                   nullptr,i,i);
         // cᵢ(Jᵢ, δJᵢ)
         a_fem.AddDomainIntegrator(new VectorMassIntegrator(*pw_c_coeffs[i]), 
                                   nullptr,i,i);
      }

      // FEM RHS
      // ±cᵢ(P(r) (b ⊗ b) E, δJᵢ)  
      real_t * fembdata = fem_b.GetData();
      ParLinearForm b_J1_r(fem_pfes[0],fembdata);
      ParLinearForm b_J2_r(fem_pfes[1],fembdata + fem_offsets[1]);

      ParLinearForm b_J1_i(fem_pfes[0],fembdata + fem_offsets.Last());
      ParLinearForm b_J2_i(fem_pfes[1],fembdata + fem_offsets.Last() + fem_offsets[1]);
      
      VectorGridFunctionCoefficient E_r_cf(dpg_pgf_r[0]);
      VectorGridFunctionCoefficient E_i_cf(dpg_pgf_i[0]);

      // (a + i b) * (c + i d) = (ac - bd) + i (ad + bc)
      MatrixVectorProductCoefficient c1_cf_rr(*balancescaled_signedcPrbb_cf[0], E_r_cf);
      MatrixVectorProductCoefficient c1_cf_ii(*balancescaled_signedcPibb_cf[0], E_i_cf);
      MatrixVectorProductCoefficient c1_cf_ri(*balancescaled_signedcPrbb_cf[0], E_i_cf);
      MatrixVectorProductCoefficient c1_cf_ir(*balancescaled_signedcPibb_cf[0], E_r_cf);

      VectorSumCoefficient c1_cf_r(c1_cf_rr, c1_cf_ii, 1.0, -1.0);
      VectorSumCoefficient c1_cf_i(c1_cf_ri, c1_cf_ir, 1.0,  1.0);

      MatrixVectorProductCoefficient c2_cf_rr(*balancescaled_signedcPrbb_cf[1], E_r_cf);
      MatrixVectorProductCoefficient c2_cf_ii(*balancescaled_signedcPibb_cf[1], E_i_cf);
      MatrixVectorProductCoefficient c2_cf_ri(*balancescaled_signedcPrbb_cf[1], E_i_cf);
      MatrixVectorProductCoefficient c2_cf_ir(*balancescaled_signedcPibb_cf[1], E_r_cf); 

      VectorSumCoefficient c2_cf_r(c2_cf_rr, c2_cf_ii, 1.0, -1.0);
      VectorSumCoefficient c2_cf_i(c2_cf_ri, c2_cf_ir, 1.0,  1.0);


      b_J1_r.AddDomainIntegrator(new VectorDomainLFIntegrator(c1_cf_r));
      b_J1_i.AddDomainIntegrator(new VectorDomainLFIntegrator(c1_cf_i));
      b_J2_r.AddDomainIntegrator(new VectorDomainLFIntegrator(c2_cf_r));
      b_J2_i.AddDomainIntegrator(new VectorDomainLFIntegrator(c2_cf_i));
      b_J1_r.Assemble();
      b_J1_i.Assemble();
      b_J2_r.Assemble();
      b_J2_i.Assemble();


      a_fem.Assemble();


      ess_bdr=1;
      ess_tdof_list.SetSize(0);
      for (int i = 0; i<ndiffusionequations;i++)
      {
         ess_tdof_listJ.SetSize(0);
         fem_pfes[i]->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ);
         for (int j = 0; j < ess_tdof_listJ.Size(); j++)
         {
            ess_tdof_listJ[j] += fem_toffsets[i];
         }
         ess_tdof_list.Append(ess_tdof_listJ);
      }

      OperatorPtr fem_Ah;
      Vector fem_X,fem_B;

      a_fem.FormLinearSystem(ess_tdof_list,fem_x,fem_b,fem_Ah, fem_X,fem_B);


      ComplexOperator * fem_Ahc = fem_Ah.As<ComplexOperator>();

      BlockOperator * fem_BlockA_r = dynamic_cast<BlockOperator *>(&fem_Ahc->real());
      BlockOperator * fem_BlockA_i = dynamic_cast<BlockOperator *>(&fem_Ahc->imag());

      int fem_nblocks = fem_BlockA_r->NumRowBlocks();

      Array2D<const HypreParMatrix*> fem_A_r_matrices(fem_nblocks, fem_nblocks);
      Array2D<const HypreParMatrix*> fem_A_i_matrices(fem_nblocks, fem_nblocks);
      for (int i = 0; i < fem_nblocks; i++)
      {
         for (int j = 0; j < fem_nblocks; j++)
         {
            fem_A_r_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&fem_BlockA_r->GetBlock(i,j));
            fem_A_i_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&fem_BlockA_i->GetBlock(i,j));
         }
      }

      HypreParMatrix * fem_Ahr = HypreParMatrixFromBlocks(fem_A_r_matrices);
      HypreParMatrix * fem_Ahi = HypreParMatrixFromBlocks(fem_A_i_matrices);

      ComplexHypreParMatrix * fem_Ahc_hypre =
         new ComplexHypreParMatrix(fem_Ahr, fem_Ahi,false, false);

      if (Mpi::Root())
      {
         mfem::out << "FEM Assembly finished successfully." << endl;
      }   


      Array<int> fem_tdof_offsets(2*fem_nblocks+1);
      fem_tdof_offsets[0] = 0;
      for (int i=0; i<fem_nblocks; i++)
      {
         fem_tdof_offsets[i+1] = fem_A_r_matrices(i,i)->Height();
         fem_tdof_offsets[fem_nblocks+i+1] = fem_tdof_offsets[i+1];
      }
      fem_tdof_offsets.PartialSum();

      BlockDiagonalPreconditioner fem_M(fem_tdof_offsets);

      HypreBoomerAMG * solver_J1 = new HypreBoomerAMG((HypreParMatrix &)
                                   fem_BlockA_r->GetBlock(0,0));
      solver_J1->SetPrintLevel(0);
      solver_J1->SetSystemsOptions(dim);
      HypreBoomerAMG * solver_J2 = new HypreBoomerAMG((HypreParMatrix &)
                                   fem_BlockA_r->GetBlock(1,1));
      solver_J2->SetPrintLevel(0);
      fem_M.SetDiagonalBlock(0,solver_J1);
      fem_M.SetDiagonalBlock(1,solver_J2);
      fem_M.SetDiagonalBlock(fem_nblocks,solver_J1);
      fem_M.SetDiagonalBlock(fem_nblocks+1,solver_J2);


      CGSolver fem_cg(MPI_COMM_WORLD);
      fem_cg.SetRelTol(1e-10);
      fem_cg.SetMaxIter(1000);
      fem_cg.SetPrintLevel(1);
      fem_cg.SetPreconditioner(fem_M);
      fem_cg.SetOperator(*fem_Ahc_hypre);
      fem_cg.Mult(fem_B, fem_X);

      a_fem.RecoverFEMSolution(fem_X, fem_x);

      for (int i = 0; i < nfem_pfes; ++i)
      {
         fem_pgf_r[i]->MakeRef(fem_pfes[i], fem_x, fem_offsets[i]);
         fem_pgf_i[i]->MakeRef(fem_pfes[i], fem_x, fem_offsets.Last() + fem_offsets[i]);
      }


      ParallelECoefficient par_e_r(dpg_pgf_r[0]);
      ParallelECoefficient par_e_i(dpg_pgf_i[0]);
      E_par_r.ProjectCoefficient(par_e_r);
      E_par_i.ProjectCoefficient(par_e_i);      


      // rescale the J solutions before saving
      for (int i = 0; i < ndiffusionequations; ++i)
      {
         (*fem_pgf_r[i]) /= balance_scale;
         (*fem_pgf_i[i]) /= balance_scale;
      }

      if (paraview)
      {
         paraview_dc->SetCycle(k);
         paraview_dc->SetTime((real_t)k);
         paraview_dc->Save();
      }

      // rescale back J after saving
      for (int i = 0; i < ndiffusionequations; ++i)
      {
         (*fem_pgf_r[i]) *= balance_scale;
         (*fem_pgf_i[i]) *= balance_scale;
      }
   }

   for (int i = 0; i<ndiffusionequations; i++)
   {
      delete pw_c_coeffs[i];
      delete c_coeffs[i];
      delete cPrbb_cf[i]; 
      delete cPibb_cf[i]; 
      delete signedcPrbb_cf[i]; 
      delete signedcPibb_cf[i]; 
   }




      
      

   // }
   

   


   

//       if (eld)
//       {
//          int j = (static_cond) ? 0 : 2;
//          HypreBoomerAMG * solver_J1 = new HypreBoomerAMG((HypreParMatrix &)
//                           BlockA_r->GetBlock(j,j));
//          solver_J1->SetPrintLevel(0);
//          solver_J1->SetSystemsOptions(dim);
//          M.SetDiagonalBlock(j,solver_J1);
//          M.SetDiagonalBlock(nblocks+j,solver_J1);
//          HypreBoomerAMG * solver_J2 = new HypreBoomerAMG((HypreParMatrix &)
//                                BlockA_r->GetBlock(j+1,j+1));
//          solver_J2->SetPrintLevel(0);
//          solver_J2->SetSystemsOptions(dim);
//          M.SetDiagonalBlock(j+1,solver_J2);
//          M.SetDiagonalBlock(nblocks+j+1,solver_J2);

//          MUMPSSolver * solver_hatJ1 = new MUMPSSolver((HypreParMatrix &)
//                   BlockA_r->GetBlock(trace_idx_offset+2,trace_idx_offset+2));


//          // HypreBoomerAMG * solver_hatJ1 = new HypreBoomerAMG((HypreParMatrix &)
//                   // BlockA_r->GetBlock(trace_idx_offset+2,trace_idx_offset+2));
//          // solver_hatJ1->SetSystemsOptions(dim);
//          // solver_hatJ1->SetRelaxType(88);
//          solver_hatJ1->SetPrintLevel(0);
//          M.SetDiagonalBlock(trace_idx_offset+2,solver_hatJ1);
//          M.SetDiagonalBlock(trace_idx_offset+nblocks+2,solver_hatJ1);

//          MUMPSSolver * solver_hatJ2 = new MUMPSSolver((HypreParMatrix &)
//                      BlockA_r->GetBlock(trace_idx_offset+3,trace_idx_offset+3));
//          // HypreBoomerAMG * solver_hatJ2 = new HypreBoomerAMG((HypreParMatrix &)
//                      // BlockA_r->GetBlock(trace_idx_offset+3,trace_idx_offset+3));
//          // solver_hatJ2->SetSystemsOptions(dim);
//          // solver_hatJ2->SetRelaxType(88);
//          solver_hatJ2->SetPrintLevel(0);
//          M.SetDiagonalBlock(trace_idx_offset+3,solver_hatJ2);
//          M.SetDiagonalBlock(trace_idx_offset+nblocks+3,solver_hatJ2);
//       }
//       CGSolver cg(MPI_COMM_WORLD);
//       cg.SetRelTol(1e-10);
//       cg.SetMaxIter(5000);
//       cg.SetPrintLevel(1);
//       cg.SetPreconditioner(M);
//       cg.SetOperator(*A);
//       cg.Mult(B, X);
//    }

//    a->RecoverFEMSolution(X, x);

//    for (int i = 0; i < npfes; ++i)
//    {
//       pgf_r[i]->MakeRef(pfes[i], x, offsets[i]);
//       pgf_i[i]->MakeRef(pfes[i], x, offsets.Last() + offsets[i]);
//    }
   
//    ParallelECoefficient par_e_r(pgf_r[0]);
//    ParallelECoefficient par_e_i(pgf_i[0]);
//    E_par_r.ProjectCoefficient(par_e_r);
//    E_par_i.ProjectCoefficient(par_e_i);
   
//    // rescale the J solutions
//    for (int i = 0; i < ndiffusionequations; ++i)
//    {
//       (*pgf_r[2+i]) /= balance_scale;
//       (*pgf_i[2+i]) /= balance_scale;
//    }



   for (int i = 0; i < dpg_trial_fecols.Size(); ++i)
   {
      delete dpg_trial_fecols[i];
      delete dpg_pfes[i];
   }
   for (int i = 0; i< dpg_test_fecols.Size(); ++i)
   {
      delete dpg_test_fecols[i];
   }
   for (int i = 0; i < fem_trial_fecols.Size(); ++i)
   {
      delete fem_trial_fecols[i];
      delete fem_pfes[i];
   }

   if (paraview_dc) 
   { 
      delete paraview_dc; 
   }

   return 0;

}
