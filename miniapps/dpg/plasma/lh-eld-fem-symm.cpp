//                   MFEM FEM parallel example
//

// Electron Landau Damping


//    ∇×(1/μ₀∇×E) - ω² ϵ₀ ϵᵣ E + i ω²ϵ₀(J₁ + J₂) = 0,   in Ω
//                   - Δ∥ J₁ + c₁ J₁ - c₁ P(r) E∥ = 0,   in Ω     
//                   - Δ∥ J₂ + c₂ J₂ + c₂ P(r) E∥ = 0,   in Ω 


// Strong formulation:
//    ∇×(1/μ₀∇×E) - ω² ϵ₀ ϵᵣ E + 1/2 i ω²ϵ₀(J₁ + J₂) - c₁ P(r)(b ⊗ b) J₃ + c₂ P(r)(b ⊗ b) J₄ = 0,   in Ω
//                   - Δ∥ J₁ + c₁ J₁ - c₁ P(r) (b ⊗ b) E = 0,   in Ω     
//                   - Δ∥ J₂ + c₂ J₂ + c₂ P(r) (b ⊗ b) E = 0,   in Ω 
//                   - Δ∥ J₃ + c₁ J₃ + 1/2 i ω² ϵ₀ E = 0,  in Ω
//                   - Δ∥ J₄ + c₂ J₄ + 1/2 i ω² ϵ₀ E = 0,  in Ω
//
//                                           E×n = E₀,  on ∂Ω
//                                           J₁  = 0,  on ∂Ω
//                                           J₂  = 0,  on ∂Ω
// weak formulation:
//   Find E ∈ H(curl,Ω), J₁  ∈ H¹(Ω), J₂  ∈ H¹(Ω),  J₃ ∈ H¹(Ω), J₄  ∈ H¹(Ω) such that
//   (1/μ₀ ∇×E, ∇ × F) - ω² ϵ₀ (ϵᵣ E, F) + i ω²ϵ₀(J₁ + J₂, F) - (c₁ P(r)(b ⊗ b) J₃, F) + (c₂ P(r)(b ⊗ b) J₄, F) = 0,  ∀ F ∈ H(curl,Ω)
//        ( (b⋅∇)J₁ , (b⋅∇) G) + c₁ (J₁ , G) - c₁ (P(r) (b ⊗ b) E, G) = 0,  ∀ G ∈ (H¹(Ω))ᵈ
//        ( (b⋅∇)J₂ , (b⋅∇) H) + c₂ (J₂ , H) + c₂ (P(r) (b ⊗ b) E, H) = 0,  ∀ H ∈ (H¹(Ω))ᵈ
//        ( (b⋅∇)J₃ , (b⋅∇) I) + c₁ (J₃ , I) - i ω² ϵ₀ (E, I) = 0,  ∀ I ∈ (H¹(Ω))ᵈ
//        ( (b⋅∇)J₄ , (b⋅∇) K) + c₂ (J₄ , K) - i ω² ϵ₀ (E, K) = 0,  ∀ K ∈ (H¹(Ω))ᵈ

//     |             E             |                J₁             |               J₂             |         J₃        |         J₄      | 
// -----------------------------------------------------------------------------------------------------------
// δE  |(1/μ₀∇×E,∇×δE)-ω²ϵ₀(ϵᵣE,δE)|          i ω²ϵ₀(J₁,δE)        |        i ω²ϵ₀(J₂,δE)         |-(c₁P(r)(b⊗b)J₃,δE)|c₂P(r)(b⊗b)J₄,δE)|    
// δJ₁ |                           | ((b⋅∇)J₁,(b⋅∇)δJ₁)+c₁(J₁,δJ₁) |                              |                   |    |       |
// δJ₂ |                           |                               |((b⋅∇)J₂,(b⋅∇)δJ₂)+ c₂(J₂,δJ₂)|       |       |  
// δJ₃ |           |        |       |       |       |  
// δJ₄ |           |        |       |       |       |  



#include "mfem.hpp"
#include "../util/pcomplexweakform.hpp"
#include "../util/pcomplexblockform.hpp"
#include "../util/blockcomplexhypremat.hpp"
#include "../util/utils.hpp"
#include "../util/maxwell_utils.hpp"
#include "utils/lh_utils.hpp"
#include "../../common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <filesystem>

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "data/LH_hot.msh";

   int order = 2;
   int par_ref_levels = 0;
   int ser_ref_levels = 0;
   bool visualization = false;
   // real_t rnum=4.6e9;
   // real_t mu = 1.257e-6;
   // real_t eps0 = 8.8541878128e-12*factor;
   real_t rnum=1.5e9;
   real_t mu = 1.257e-6;
   real_t eps0 = 8.8541878128e-12;
   bool eld = true; // enable/disable electron Landau damping 

   bool paraview = false;
   bool debug = false;
   bool mumps_solver = true;
   real_t delta_prec = 0.01;

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
   args.AddOption(&delta_prec, "-dp", "--delta-prec", "stability parameter for the preconditioner.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&eld, "-eld", "--eld", "-no-eld",
                  "--no-eld",
                  "Enable or disable electron Landau damping.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&mumps_solver, "-mumps", "--mumps", "-no-mumps",
                  "--no-mumps",
                  "Enable or disable MUMPS solver.");
   args.AddOption(&debug, "-debug", "--debug", "-no-debug",
                  "--no-debug",
                  "Enable or disable debug mode (delta = 0.01 and no coupling).");                  
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   
   // number of diffusion equations
   int ndiffusionequations = (eld) ?  2 : 0; 

   Vector cvals(ndiffusionequations);
   Vector csigns(ndiffusionequations);
   // real_t cfactor = 1e-6;
   real_t cfactor = 1.0;
   if (eld)
   {
      cvals(0)  = 25e6;  cvals(1)  = 1e6;
      csigns(0) = -1.0;  csigns(1) = 1.0;
   }
   cvals *= cfactor; // scale the coefficients
   
   real_t omega = 2.*M_PI*rnum;
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

   for (int i = 0; i < ser_ref_levels; i++)
   {
      mesh.UniformRefinement();
   }

   mesh.RemoveInternalBoundaries();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   int nattr = (pmesh.attributes.Size()) ? pmesh.attributes.Max() : 0;
   Array<int> attr(nattr);
   for (int i = 0; i<nattr; i++) { attr[i] = i+1; }
   
   for (int i = 0; i<par_ref_levels; i++)
   {
      pmesh.UniformRefinement();
   }

   // Define the coefficients 
   ConstantCoefficient muinv(1./mu);

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
   ScalarVectorProductCoefficient scaledb_cf(sqrt(cfactor), b_cf); 
   MatrixFunctionCoefficient bb_cf(dim,bcrossb); // b⊗b

   MatrixSumCoefficient oneminusbb(Mone_cf, bb_cf, 1.0, -1.0); // 1 - b⊗b

   // S(r) (I - b⊗b)
   ScalarMatrixProductCoefficient Soneminusbb_r(S_cf_r, oneminusbb), Soneminusbb_i(S_cf_i, oneminusbb); 

   // P(r) b⊗b 
   ScalarMatrixProductCoefficient P_cf_bb_r(P_cf_r, bb_cf), P_cf_bb_i(P_cf_i, bb_cf); 

   // εᵣ = S(r) (I - b⊗b) + P(r) b⊗b 
   MatrixSumCoefficient eps_r(Soneminusbb_r, P_cf_bb_r, 1.0, 1.0); 
   MatrixSumCoefficient eps_i(Soneminusbb_i, P_cf_bb_i, 1.0, 1.0); 

   coefs_r[nattr-1] = &eps_r;
   coefs_i[nattr-1] = &eps_i;

   PWMatrixCoefficient eps_cf_r(dim, attr, coefs_r);
   PWMatrixCoefficient eps_cf_i(dim, attr, coefs_i);

   real_t scale = (debug) ? 0.0 : 1.0;
   ConstantCoefficient eps0omeg2(eps0 * omega * omega * scale);
   ConstantCoefficient negeps0omeg2(-eps0 * omega * omega);

   ScalarMatrixProductCoefficient m_cf_r(negeps0omeg2, eps_cf_r);
   ScalarMatrixProductCoefficient m_cf_i(negeps0omeg2, eps_cf_i);

   // if ELD
   Array<Vector *> c_arrays(ndiffusionequations);
   Array<PWConstCoefficient *> pw_c_coeffs(ndiffusionequations);
   Array<MatrixCoefficient *> cPrbb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> cPibb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> signedcPrbb_cf(ndiffusionequations);
   Array<MatrixCoefficient *> signedcPibb_cf(ndiffusionequations);
   Vector temp(nattr); temp=0.0;
   for (int i = 0; i<ndiffusionequations; i++)
   {
      temp[nattr-1] = cvals(i);
      pw_c_coeffs[i] = new PWConstCoefficient(temp);
      cPrbb_cf[i] = new ScalarMatrixProductCoefficient(*pw_c_coeffs[i], P_cf_bb_r);
      cPibb_cf[i] = new ScalarMatrixProductCoefficient(*pw_c_coeffs[i], P_cf_bb_i);
      signedcPrbb_cf[i] = new ScalarMatrixProductCoefficient(csigns[i], *cPrbb_cf[i]);
      signedcPibb_cf[i] = new ScalarMatrixProductCoefficient(csigns[i], *cPibb_cf[i]);
   }

   Array<FiniteElementCollection *> fecols;
   Array<ParFiniteElementSpace *> pfes;
   fecols.Append(new ND_FECollection(order, dim));
   pfes.Append(new ParFiniteElementSpace(&pmesh, fecols[0]));
   if (eld)
   {
      for (int i = 0; i < 2*ndiffusionequations; ++i)
      {
         fecols.Append(new H1_FECollection(order, dim));
         pfes.Append(new ParFiniteElementSpace(&pmesh, fecols[i+1], dim));
      }
   }

   Array<HYPRE_BigInt> tdofs(pfes.Size());
   for (int i = 0; i < pfes.Size(); ++i)
   {
      tdofs[i] = pfes[i]->GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << "ParFiniteElementSpace " << i << " has " << tdofs[i]
              << " true dofs." << endl;
      }
   }
   if (Mpi::Root())
   {
      cout << "Total number of true dofs: " << tdofs.Sum() << endl;
   }

   ParComplexBlockForm *a = new ParComplexBlockForm(pfes);
   // (1/μ₀ ∇×E, ∇ × F)
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv), nullptr, 0, 0);
   // - ω² ϵ₀ (ϵᵣ E, F)
   a->AddDomainIntegrator(new VectorFEMassIntegrator(m_cf_r),
                          new VectorFEMassIntegrator(m_cf_i), 0, 0);
   ConstantCoefficient halfeps0omeg2(0.5*eps0 * omega * omega);    
   ConstantCoefficient neghalfeps0omeg2(-0.5*eps0 * omega * omega);                   
   if (eld)
   {
      for (int i = 0; i<ndiffusionequations; i++)
      {
         //  1/2 i ω²ϵ₀((J₁+J₂),F)
         a->AddDomainIntegrator(nullptr, new TransposeIntegrator(new VectorFEMassIntegrator(halfeps0omeg2)), i+1, 0);
         //  ( (b⋅∇)J₁ , (b⋅∇) G)
         a->AddDomainIntegrator(new DirectionalVectorDiffusionIntegrator(scaledb_cf), nullptr, i+1, i+1);
         // cᵢ (J₁ , G)
         a->AddDomainIntegrator(new VectorMassIntegrator(*pw_c_coeffs[i]), nullptr, i+1, i+1);
         // ±cᵢ(P(r) (b ⊗ b) E, G)
         a->AddDomainIntegrator(new VectorFEMassIntegrator(*signedcPrbb_cf[i]), 
                                new VectorFEMassIntegrator(*signedcPibb_cf[i]), 0, i+1);
         // 
         // ±(cᵢ P(r)(b ⊗ b) Jᵢ₊₂,F)
         a->AddDomainIntegrator(new TransposeIntegrator(
                                new VectorFEMassIntegrator(*signedcPrbb_cf[i])), 
                                new TransposeIntegrator(
                                new VectorFEMassIntegrator(*signedcPibb_cf[i])), i+3, 0);
      
      // ( (b⋅∇)J₃ , (b⋅∇) I) + c₁ (J₃ , I) - i ω² ϵ₀ (E, I) = 0,  ∀ I ∈ (H¹(Ω))ᵈ
      // ( (b⋅∇)J₄ , (b⋅∇) K) + c₂ (J₄ , K) - i ω² ϵ₀ (E, K) = 0,  ∀ K ∈ (H¹(Ω))ᵈ
         a->AddDomainIntegrator(new DirectionalVectorDiffusionIntegrator(scaledb_cf), nullptr, i+3, i+3);
         a->AddDomainIntegrator(new VectorMassIntegrator(*pw_c_coeffs[i]), nullptr, i+3, i+3);

         // + i ω² ϵ₀ (E, I)                        
         a->AddDomainIntegrator(nullptr, new VectorFEMassIntegrator(halfeps0omeg2), 0, i+3);                         
                                 
      }
   }

   a->Assemble();

   for (int i = 0; i<ndiffusionequations; i++)
   {
      delete pw_c_coeffs[i];
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

   ParComplexGridFunction maxwell_pgf(pfes[0]); maxwell_pgf = 0.0;

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);
   ParGridFunction E_par_r(&L2_fes);
   ParGridFunction E_par_i(&L2_fes);

   ParaViewDataCollection * paraview_dc = nullptr;

   std::string output_dir = "ParaView/FEM/" + GetTimestamp();

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
      paraview_dc->RegisterField("E_r",pgf_r[0]);
      paraview_dc->RegisterField("E_i",pgf_i[0]);
      paraview_dc->RegisterField("MaxwellE_r",&maxwell_pgf.real());
      paraview_dc->RegisterField("MaxwellE_i",&maxwell_pgf.imag());
      paraview_dc->RegisterField("E_par_r",&E_par_r);
      paraview_dc->RegisterField("E_par_i",&E_par_i);
      if (eld)
      {
         paraview_dc->RegisterField("Jh_1_r",pgf_r[1]);
         paraview_dc->RegisterField("Jh_1_i",pgf_i[1]);
         paraview_dc->RegisterField("Jh_2_r",pgf_r[2]);
         paraview_dc->RegisterField("Jh_2_i",pgf_i[2]);
         paraview_dc->RegisterField("Jh_3_r",pgf_r[3]);
         paraview_dc->RegisterField("Jh_3_i",pgf_i[3]);
         paraview_dc->RegisterField("Jh_4_r",pgf_r[4]);
         paraview_dc->RegisterField("Jh_4_i",pgf_i[4]);
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

      pfes[0]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      for (int i = 0; i<2*ndiffusionequations;i++)
      {
         ess_tdof_listJ.SetSize(0);
         pfes[i+1]->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJ);
         for (int j = 0; j < ess_tdof_listJ.Size(); j++)
         {
            ess_tdof_listJ[j] += toffsets[i+1];
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
   Vector B, X;

   Vector b(x.Size()); b = 0.0;

   a->FormLinearSystem(ess_tdof_list, x, b, Ah, X, B);
   ComplexOperator * Ahc = Ah.As<ComplexOperator>();


   BlockOperator * Block_r = dynamic_cast<BlockOperator *>(&Ahc->real());
   BlockOperator * Block_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

   int nblocks = Block_r->NumRowBlocks();
   Array2D<const HypreParMatrix*> r_matrices(nblocks, nblocks);
   Array2D<const HypreParMatrix*> i_matrices(nblocks, nblocks);
   for (int i = 0; i < nblocks; i++)
   {
      for (int j = 0; j < nblocks; j++)
      {
         r_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&Block_r->GetBlock(i,j));
         i_matrices(i,j) = dynamic_cast<HypreParMatrix*>(&Block_i->GetBlock(i,j));
      }
   }
   HypreParMatrix * hr = HypreParMatrixFromBlocks(r_matrices);
   HypreParMatrix * hi = HypreParMatrixFromBlocks(i_matrices);

   ComplexHypreParMatrix * hc_hypre =
      new ComplexHypreParMatrix(hr, hi,false, false);


   auto P = new ComplexMUMPSSolver(MPI_COMM_WORLD);
   P->SetPrintLevel(1);
   P->SetOperator(*hc_hypre);
   P->Mult(B, X);

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
   for (int i = 0; i < fecols.Size(); ++i)
   {
      delete fecols[i];
      delete pfes[i];
      delete pgf_r[i];
      delete pgf_i[i];
   }

   return 0;

}


