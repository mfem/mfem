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
//                   MFEM Maxwell-Vdiffusion coupling parallel example
//
// Compile with: make lh-eld-weak
//
// mpirun -np 8 ./lh-eld-weak -o 3 -paraview -pr 0
// mpirun -np 8 ./lh-eld-weak -o 3 -paraview -pr 1 -sc

// Electron Landau Damping

// Weak Coupling Formulation:
// Strong formulation:
//     ∇×(1/μ₀∇×E) - ω² ϵ₀ ϵ E = i ω²ϵ₀(J₁ + J₂),   in Ω
//                         E×n = E₀,  on ∂Ω
//             - Δ∥ J₁ + c₁ J₁ =  c₁ P(r) E∥,   in Ω     
//             - Δ∥ J₂ + c₂ J₂ = -c₂ P(r) E∥,   in Ω 
//                         J₁  = 0,  on ∂Ω
//                         J₂  = 0,  on ∂Ω
//   Find E ∈ H(curl,Ω), J₁  ∈ H¹(Ω), J₂  ∈ H¹(Ω) such that
//   (1/μ₀ ∇×E, ∇ × F) - ω² ϵ₀ (ϵᵣ E, F) = -i ω²ϵ₀(J₁ + J₂, F),  ∀ F ∈ H(curl,Ω)

// ( (b⋅∇)J₁,(b⋅∇) δJ₁ ) + c₁ (J₁,δJ₁) =  c₁(P(r) b⊗b E, δJ₁),  ∀ δJ₁ ∈ (H¹(Ω))²
// ( (b⋅∇)J₂,(b⋅∇) δJ₂ ) + c₂ (J₂,δJ₂) = -c₂(P(r) b⊗b E, δJ₂),  ∀ δJ₂ ∈ (H¹(Ω))²
//                                                    J₁ = J₂ = 0,  on ∂Ω
// --------------------------------------------------------------------------------
// |               J₁                |                 J₂           |     RHS      |
// --------------------------------------------------------------------------------
// |δJ₁|((b⋅∇)J₁,(b⋅∇)δJ₁)+c₁(J₁,δJ₁)|                              | c₁(P(r)E,δJ₁)|  
// |   |                             |                              |              |   
// |δJ₂|                             |((b⋅∇)J₂,(b⋅∇)δJ₂)+ c₂(J₂,δJ₂)|-c₂(P(r)E,δJ₂)|  
// where (δE,δH,δJ₁,δJ₂) ∈  H¹(Ω) × H(curl,Ω) × (H¹(Ω))² × (H¹(Ω))² 



#include "mfem.hpp"
#include "../util/pcomplexweakform.hpp"
#include "../util/pcomplexblockform.hpp"
#include "../../common/mfem-common.hpp"
#include "../util/maxwell_utils.hpp"
#include "../util/utils.hpp"
#include "utils/lh_utils.hpp"
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
   int par_ref_levels = 0;
   int ser_ref_levels = 0;

   real_t rnum=1.5;
   real_t mu = 1.257;
   real_t eps0 = 8.8541878128;
   real_t cfactor = 1e-6;

   bool static_cond = false;
   bool visualization = false;
   bool paraview = false;
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
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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
   csigns(0) = 1.0;  csigns(1) = -1.0;
   cvals *= cfactor; // scale the coefficients
   real_t omega = 2.*M_PI*rnum;

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

   // Define coefficients
   ConstantCoefficient muinv(1./mu);
   ConstantCoefficient one_cf(1.0);

   ConstantCoefficient negomegeps0_cf(-omega*eps0);

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

   PWMatrixCoefficient eps_cf_r(dim, attr, coefs_r);
   PWMatrixCoefficient eps_cf_i(dim, attr, coefs_i);

   ConstantCoefficient eps0omeg(omega * eps0);
   ConstantCoefficient negeps0omeg(-omega * eps0);
   ConstantCoefficient negeps0omeg2(-omega * omega * eps0);
   ConstantCoefficient eps0omeg2(omega * omega * eps0);

   ScalarMatrixProductCoefficient m_cf_r(negeps0omeg2, eps_cf_r);
   ScalarMatrixProductCoefficient m_cf_i(negeps0omeg2, eps_cf_i);

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

   Array<Vector *> c_arrays(ndiffusionequations);
   Array<PWConstCoefficient *> pw_c_coeffs(ndiffusionequations);
   Array<MatrixCoefficient *> cPrbb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> cPibb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> signedcPrbb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> signedcPibb_cf(ndiffusionequations);

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
   }

   // Define the spaces
   Array<FiniteElementCollection *> fem_fecols;
   Array<ParFiniteElementSpace *> fem_pfes;


   ND_FECollection *nd_fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *nd_pfes = new ParFiniteElementSpace(&pmesh, nd_fec);

   // Vector H1 spaces for Jᵢ 
   for (int i = 0; i < ndiffusionequations; i++)
   {
      fem_fecols.Append(new H1_FECollection(order, dim));
      fem_pfes.Append(new ParFiniteElementSpace(&pmesh, fem_fecols.Last(), dim));
   }


   HYPRE_BigInt nd_tdofs = nd_pfes->GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "ND ParFiniteElementSpace has " << nd_tdofs
           << " true dofs." << endl;
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
      cout << "Total number of ND true dofs for Maxwell:     " << nd_tdofs << endl;
      cout << "Total number of FEM true dofs for VDiffusion: " << fem_tdofs.Sum() << endl;
   }

   int nfem_pfes = fem_pfes.Size();
   Array<int> fem_offsets(nfem_pfes+1);  fem_offsets[0] = 0;
   Array<int> fem_toffsets(nfem_pfes+1); fem_toffsets[0] = 0;
   for (int i = 0; i<nfem_pfes; i++)
   {
      fem_offsets[i+1] = fem_pfes[i]->GetVSize();
      fem_toffsets[i+1] = fem_pfes[i]->TrueVSize();
   }
   fem_offsets.PartialSum();
   fem_toffsets.PartialSum();

   Vector nd_x(2*nd_pfes->GetVSize());
   nd_x = 0.;
   Vector fem_x(2*fem_offsets.Last());
   fem_x = 0.;

   Vector fem_b(2*fem_offsets.Last());
   fem_b = 0.;

   ParGridFunction * nd_pgf_r = new ParGridFunction(nd_pfes, nd_x, 0);
   ParGridFunction * nd_pgf_i = new ParGridFunction(nd_pfes, nd_x, nd_x.Size()/2);
   Array<ParGridFunction *> fem_pgf_r(nfem_pfes);
   Array<ParGridFunction *> fem_pgf_i(nfem_pfes);

   Array<ParGridFunction *> old_fem_pgf_r(nfem_pfes);
   Array<ParGridFunction *> old_fem_pgf_i(nfem_pfes);

   for (int i = 0; i < nfem_pfes; ++i)
   {
      fem_pgf_r[i] = new ParGridFunction(fem_pfes[i], fem_x, fem_offsets[i]);
      fem_pgf_i[i] = new ParGridFunction(fem_pfes[i], fem_x, fem_offsets.Last() + fem_offsets[i]);
      old_fem_pgf_r[i] = new ParGridFunction(fem_pfes[i]); *old_fem_pgf_r[i] = 0.0;
      old_fem_pgf_i[i] = new ParGridFunction(fem_pfes[i]); *old_fem_pgf_i[i] = 0.0;
   }

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);
   ParGridFunction E_par_r(&L2_fes);
   ParGridFunction E_par_i(&L2_fes);

   ParaViewDataCollection * paraview_dc = nullptr;
   std::string output_dir = "ParaView/FEM/WeakCoupling/" + GetTimestamp();

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
      paraview_dc->RegisterField("E_r",nd_pgf_r);
      paraview_dc->RegisterField("E_i",nd_pgf_i);
      paraview_dc->RegisterField("E_par_r",&E_par_r);
      paraview_dc->RegisterField("E_par_i",&E_par_i);
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


   int max_fixed_point_iter = 50;
   // loop through fixed point iterations

   for (int k = 0; k<max_fixed_point_iter; k++)
   {
      nd_x = 0.0;

      // reset GridFunctions
      *nd_pgf_r = 0.0;
      *nd_pgf_i = 0.0;

      delta = (k == 0) ? 0.01 : 0.01;
      // delta = 0.01 * pow(0.5, 0.0);
      delta = 0.01 * pow(0.9, k);
      ParSesquilinearForm a_nd(nd_pfes);

      // (1/μ₀ ∇×E, ∇ × F)
      a_nd.AddDomainIntegrator(new CurlCurlIntegrator(muinv), nullptr);
      // - ω² ϵ₀ (ϵᵣ E, F)
      a_nd.AddDomainIntegrator(new VectorFEMassIntegrator(m_cf_r),
                          new VectorFEMassIntegrator(m_cf_i));
      a_nd.Assemble();

      VectorGridFunctionCoefficient J1_cf_r(fem_pgf_r[0]);
      VectorGridFunctionCoefficient J1_cf_i(fem_pgf_i[0]);
      VectorGridFunctionCoefficient J2_cf_r(fem_pgf_r[1]);
      VectorGridFunctionCoefficient J2_cf_i(fem_pgf_i[1]);

      // -iω²ϵ₀ (Jᵣ + i Jᵢ ,F) = ω² ϵ₀ (Jᵢ - i Jᵣ,F)   
      ScalarVectorProductCoefficient negomeg2eps0_J1_cf_i(negeps0omeg2, J1_cf_i);
      ScalarVectorProductCoefficient negomeg2eps0_J2_cf_i(negeps0omeg2, J2_cf_i);

      ScalarVectorProductCoefficient omeg2eps0_J1_cf_i(eps0omeg2, J1_cf_i);
      ScalarVectorProductCoefficient omeg2eps0_J2_cf_i(eps0omeg2, J2_cf_i);

      ScalarVectorProductCoefficient negomeg2eps0_J1_cf_r(negeps0omeg2, J1_cf_r);
      ScalarVectorProductCoefficient negomeg2eps0_J2_cf_r(negeps0omeg2, J2_cf_r);


      ScalarVectorProductCoefficient omeg2eps0_J1_cf_r(eps0omeg2, J1_cf_r);
      ScalarVectorProductCoefficient omeg2eps0_J2_cf_r(eps0omeg2, J2_cf_r);


      ParComplexLinearForm nd_b(nd_pfes);

      // ω² ϵ₀ (Jᵢ - i Jᵣ,F) = (ω² ϵ₀ Jᵢ, F) + i (-ω² ϵ₀Jᵢ,F)  
      // nd_b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(omeg2eps0_J1_cf_i),
      //                          new VectorFEDomainLFIntegrator(negomeg2eps0_J1_cf_r));
      // nd_b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(omeg2eps0_J2_cf_i),
      //                          new VectorFEDomainLFIntegrator(negomeg2eps0_J2_cf_r));    
      

      ScalarVectorProductCoefficient omeg2_eps0_J1_cf_i(eps0*omega*omega, J1_cf_i);
      ScalarVectorProductCoefficient omeg2_eps0_J2_cf_i(eps0*omega*omega, J2_cf_i);

      ScalarVectorProductCoefficient negomeg2_eps0_J1_cf_r(-eps0*omega*omega, J1_cf_r);
      ScalarVectorProductCoefficient negomeg2_eps0_J2_cf_r(-eps0*omega*omega, J2_cf_r);
      
      nd_b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(omeg2_eps0_J1_cf_i),
                               new VectorFEDomainLFIntegrator(negomeg2_eps0_J1_cf_r));
      nd_b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(omeg2_eps0_J2_cf_i),
                               new VectorFEDomainLFIntegrator(negomeg2_eps0_J2_cf_r));  

      // nd_b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(negomeg2eps0_J1_cf_i), nullptr);
      // ConstantCoefficient temp(1.0);
      // nd_b.AddDomainIntegrator(nullptr,new VectorFEDomainLFIntegrator(J1_cf_r));
      // nd_b.AddDomainIntegrator(nullptr,new VectorFEDomainLFIntegrator(omeg2eps0_J2_cf_r));
      

      nd_b.Assemble();

      // remove internal boundaries
      ess_bdr = 1;
      for (int i = 0; i<int_bdr_attr.Size(); i++)
      {
         ess_bdr[int_bdr_attr[i]-1] = 0;
      }

      nd_pfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // if (k == 0)
      // {
      nd_pgf_r->ProjectBdrCoefficientTangent(one_x_cf, one_r_bdr);
      nd_pgf_r->ProjectBdrCoefficientTangent(negone_x_cf, negone_r_bdr);
      nd_pgf_i->ProjectBdrCoefficientTangent(one_x_cf, one_i_bdr);
      nd_pgf_i->ProjectBdrCoefficientTangent(negone_x_cf, negone_i_bdr);
      // }

      OperatorPtr nd_Ah;
      Vector nd_X,nd_B;
      a_nd.FormLinearSystem(ess_tdof_list,nd_x,nd_b,nd_Ah, nd_X,nd_B);


      HypreParMatrix *nd_A = nd_Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();

      MUMPSSolver nd_mumps(MPI_COMM_WORLD);
      nd_mumps.SetPrintLevel(0);
      nd_mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      nd_mumps.SetOperator(*nd_A);
      nd_mumps.Mult(nd_B, nd_X);

      a_nd.RecoverFEMSolution(nd_X, nd_b, nd_x);

      nd_pgf_r->MakeRef(nd_pfes, nd_x, 0);
      nd_pgf_i->MakeRef(nd_pfes, nd_x, nd_x.Size()/2);


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
      
      VectorGridFunctionCoefficient E_r_cf(nd_pgf_r);
      VectorGridFunctionCoefficient E_i_cf(nd_pgf_i);

      // (a + i b) * (c + i d) = (ac - bd) + i (ad + bc)
      MatrixVectorProductCoefficient c1_cf_rr(*signedcPrbb_cf[0], E_r_cf);
      MatrixVectorProductCoefficient c1_cf_ii(*signedcPibb_cf[0], E_i_cf);
      MatrixVectorProductCoefficient c1_cf_ri(*signedcPrbb_cf[0], E_i_cf);
      MatrixVectorProductCoefficient c1_cf_ir(*signedcPibb_cf[0], E_r_cf);

      VectorSumCoefficient c1_cf_r(c1_cf_rr, c1_cf_ii, 1.0, -1.0);
      VectorSumCoefficient c1_cf_i(c1_cf_ri, c1_cf_ir, 1.0,  1.0);

      MatrixVectorProductCoefficient c2_cf_rr(*signedcPrbb_cf[1], E_r_cf);
      MatrixVectorProductCoefficient c2_cf_ii(*signedcPibb_cf[1], E_i_cf);
      MatrixVectorProductCoefficient c2_cf_ri(*signedcPrbb_cf[1], E_i_cf);
      MatrixVectorProductCoefficient c2_cf_ir(*signedcPibb_cf[1], E_r_cf); 

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
      fem_cg.SetRelTol(1e-16);
      fem_cg.SetMaxIter(1000);
      fem_cg.SetPrintLevel(1);
      fem_cg.SetPreconditioner(fem_M);
      fem_cg.SetOperator(*fem_Ahc_hypre);
      fem_cg.Mult(fem_B, fem_X);

      a_fem.RecoverFEMSolution(fem_X, fem_x);


      real_t alpha = (k == 0) ? 1.0 : 0.01;

      for (int i = 0; i < nfem_pfes; ++i)
      {
         fem_pgf_r[i]->MakeRef(fem_pfes[i], fem_x, fem_offsets[i]);
         fem_pgf_i[i]->MakeRef(fem_pfes[i], fem_x, fem_offsets.Last() + fem_offsets[i]);

         (*fem_pgf_r[i])*= alpha;
         (*fem_pgf_i[i])*= alpha;
         fem_pgf_r[i]->Add(1.0-alpha, *old_fem_pgf_r[i]);
         fem_pgf_i[i]->Add(1.0-alpha, *old_fem_pgf_i[i]);
         *old_fem_pgf_r[i] = *fem_pgf_r[i];
         *old_fem_pgf_i[i] = *fem_pgf_i[i];
      }


      ParallelECoefficient par_e_r(nd_pgf_r);
      ParallelECoefficient par_e_i(nd_pgf_i);
      E_par_r.ProjectCoefficient(par_e_r);
      E_par_i.ProjectCoefficient(par_e_i);      

      if (paraview)
      {
         paraview_dc->SetCycle(k);
         paraview_dc->SetTime((real_t)k);
         paraview_dc->Save();
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



   delete nd_fec;
   delete nd_pfes;
   for (int i = 0; i < fem_fecols.Size(); ++i)
   {
      delete fem_fecols[i];
      delete fem_pfes[i];
   }

   if (paraview_dc) 
   { 
      delete paraview_dc; 
   }

   return 0;

}
