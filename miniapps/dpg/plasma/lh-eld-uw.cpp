//                   MFEM UW parallel example
//

// Electron Landau Damping
//
//        ∇×(1/μ₀∇×E) - ω² ϵ₀ ϵᵣ E + i ω²ϵ₀(J₁ + J₂) = 0,   in Ω
//                   - (b⋅∇)²J₁ + c₁ J₁ - c₁ P b⊗b E = 0,   in Ω     
//                   - (b⋅∇)²J₂ + c₂ J₂ + c₂ P b⊗b E = 0,   in Ω 
//                                               E×n = E₀,  on ∂Ω
//                                               J₁  = 0,   on ∂Ω
//                                               J₂  = 0,   on ∂Ω
// First order system:
//     i ω μ₀ H + ∇ × E                  = 0, in Ω
// -i ω ϵ₀ ϵᵣ E + ∇ × H - ω ϵ₀ (J₁ + J₂) = 0, in Ω 
//            b ⋅ ∇Q₁ + c₁ J₁ - c₁ P B E = 0, in Ω
//                          Q₁ + b ⋅ ∇J₁ = 0, in Ω 
//            b ⋅ ∇Q₂ + c₂ J₂ + c₂ P B E = 0, in Ω  
//                          Q₂ + b ⋅ ∇J₂ = 0, in Ω
// where B = b ⊗ b. Note that in 2D H is scalar and ∇ × E = ∇ ⋅ (R E) where R = [0 1; -1 0]
//
// Define the group variables 
// then the strong formulation reads:
// (A u, v) = (i ω μ₀ H, F) + (∇ × E, F) 
//          - (i ω ϵ₀ ϵᵣ E, W) + (∇ W, R H) - (ω ϵ₀ (J₁ + J₂), W)
//          + (b ⋅ ∇Q₁, K₁) + (c₁ J₁, K₁) - (c₁ P B E, K₁) + (Q₁, L₁) + (b ⋅ ∇J₁, L₁)
//          + (b ⋅ ∇Q₂, K₂) + (c₂ J₂, K₂) + (c₂ P B E, K₂) + (Q₂, L₂) + (b ⋅ ∇J₂, L₂) 
//
// and the adjoint operator is defined by:
// (u, A^⋆ v) = (E, ∇×F + i ω ϵ₀ ϵ⋆ᵣ W - c₁ P̄ B K₁ + c₂ P̄ B K₂)
//           + (H, -i ω μ₀ F + ∇ × W)
//           + (J₁, -ω ϵ₀ W + c₁ K₁ - b ⋅ ∇ L₁)
//           + (Q₁, - b ⋅ ∇ K₁ + L₁)
//           + (J₂, -ω ϵ₀ W + c₂ K₂ - b ⋅ ∇ L₂)
//           + (Q₂, - b ⋅ ∇ K₂ + L₂)
// 
// Define the group trial field and trace variables 
// u:=(E,H,J₁,Q₁,J₂,Q₂), û:=(Ê, Ĥ, Ĵ₁, Q̂₁, Ĵ₂, Q̂₂) and 
// the test variable  v:=(F,W,K₁,L₁,K₂,L₂) 
// The ultraweak variational formulation is then given by:
// Find u ∈ U=((L²(Ω))² × L²(Ω) × (L²(Ω))² × (L²(Ω))² × (L²(Ω))² × (L²(Ω))²),
// û ∈ Û=( H⁻¹/²(Γₕ) × H¹/²(Γₕ) × (H⁻¹/²(Γₕ))² × (H⁻¹/²(Γₕ))² × (H⁻¹/²(Γₕ))²× (H⁻¹/²(Γₕ))² )
//  such that ∀ v ∈ V=(H¹(Ω) × H(curl,Ω) × (H¹(Ω))² × (H¹(Ω))² × (H¹(Ω))² × (H¹(Ω))²)
//  (u, A^⋆ v) + b̂(û, v) = 0,
// or equivalently
// 
// (i ω μ₀ H, F) + (E, ∇×F) + < R Ê, F > = 0,      ∀ F ∈ H¹(Ω)
// -i ω ϵ₀ ϵᵣ (E, W) + (H, ∇ ×  W) + < Ĥ, W × n > - ω ϵ₀ (J₁ + J₂, W) = 0,  ∀ W ∈ H(curl,Ω)
// -(Q, b ⋅ ∇ K₁) + <Q̂₁, K₁> + (c₁ J₁, K₁) - c₁(P B E, K₁) = 0,  ∀ K₁ ∈ (H¹(Ω))²
//  (Q₁, L₁) - (J₁, b ⋅ ∇L₁) + <Ĵ₁, L₁> = 0,  ∀ L₁ ∈ (H¹(Ω))²
// -(Q, b⋅ ∇ K₂) + <Q̂₂, K₂> + (c₂ J₂, K₂) + c₂(P B E, K₂) = 0,  ∀ K₂ ∈ (H¹(Ω))²
//  (Q₂, L₂) - (J₂, b ⋅ ∇L₂) + <Ĵ₂, L₂> = 0,  ∀ L₂ ∈ (H¹(Ω))²

// with the adjoint graph test norm defined by:
// ‖v‖²_V = ‖A^⋆ v‖² + ‖v‖²
//        = ‖∇×F + i ω ϵ₀ ϵ⋆ᵣ W - c₁ P̄ B K₁ + c₂ P̄ B K₂‖²
//        + ‖-i ω μ₀ F + ∇ × W‖²
//        + ‖-ω ϵ₀ W + c₁ K₁ - b ⋅ ∇ L₁‖²
//        + ‖- b ⋅ ∇ K₁ + L₁‖²
//        + ‖-ω ϵ₀ W + c₂ K₂ - b ⋅ ∇ L₂‖²
//        + ‖- b ⋅ ∇ K₂ + L₂‖²
//        + ‖F‖² + ‖W‖² + ‖K₁‖² + ‖L₁‖² + ‖K₂‖² + ‖L₂‖²
//

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

// Define spaces
enum TrialSpace
{
   E_space     = 0,
   H_space     = 1,
   J1_space    = 2,
   Q1_space    = 3,
   J2_space    = 4,
   Q2_space    = 5,
   hatE_space  = 6,
   hatH_space  = 7,
   hatJ1_space = 8,
   hatQ1_space = 9,
   hatJ2_space = 10,
   hatQ2_space = 11
};

enum TestSpace
{
   F_space = 0,
   W_space = 1,
   K1_space = 2,
   L1_space = 3,
   K2_space = 4,
   L2_space = 5
};

inline const char * ToString(TrialSpace ts)
{
   switch (ts)
   {
      case E_space:     return " L²(Ω)²  for E  ";
      case H_space:     return " L²(Ω)   for H  ";
      case J1_space:    return " L²(Ω)²  for J₁ ";
      case Q1_space:    return " L²(Ω)²  for Q₁ ";
      case J2_space:    return " L²(Ω)²  for J₂ ";
      case Q2_space:    return " L²(Ω)²  for Q₂ ";
      case hatE_space:  return " tr(RT)  for Ê  ";
      case hatH_space:  return " tr(H¹)  for Ĥ  ";
      case hatJ1_space: return " tr(RT)² for Ĵ₁ ";
      case hatQ1_space: return " tr(RT)² for Q̂₁ ";
      case hatJ2_space: return " tr(RT)² for Ĵ₂ ";
      case hatQ2_space: return " tr(RT)² for Q̂₂ ";
      default:          return "Unknown TrialSpace";
   }
}

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

   if (debug) { delta = 0.01; } 

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

   real_t omega = 2.*M_PI*rnum;
   real_t neg_omega_eps0 = -omega * eps0;
   real_t omega_eps0 = omega * eps0;

   Vector cvals(ndiffusionequations);
   Vector csigns(ndiffusionequations);
   real_t cfactor = 1.0;
   cvals(0)  = 25e6;  cvals(1)  = 1e6;
   csigns(0) = -1.0;  csigns(1) = 1.0;
   cvals *= cfactor; // scale the coefficients

   // List of coefficients
   // Trial integrators coefficients
   DenseMatrix R(dim), Rt(dim);
   R(0,0) =  0.0;  R(0,1) = 1.0;
   R(1,0) = -1.0;  R(1,1) = 0.0;
   R.Transpose(Rt);

   MatrixFunctionCoefficient B_cf(dim, bcrossb);
   FunctionCoefficient p_r_cf(pfunc_r);
   FunctionCoefficient p_i_cf(pfunc_i);

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

   Array<Vector *> cf_arrays(ndiffusionequations);
   Array<PWConstCoefficient *> c_cf(ndiffusionequations);
   Array<ProductCoefficient *> signed_c_cf(ndiffusionequations);
   Array<MatrixCoefficient*> signed_PB_r_cf(ndiffusionequations);
   Array<MatrixCoefficient*> signed_PB_i_cf(ndiffusionequations);
   Vector zerovec(nattr); zerovec=0.0;

   // ϵᵣ in plasma (real)
   MatrixFunctionCoefficient eps_r_temp_cf(dim, epsilon_func_r);
   // ϵᵣ in plasma(imag)
   MatrixFunctionCoefficient eps_i_temp_cf(dim, epsilon_func_i);

   coefs_r[nattr-1] = &eps_r_temp_cf;
   coefs_i[nattr-1] = &eps_i_temp_cf;

   // ωμ₀ 
   ConstantCoefficient omegamu_cf(mu * omega);
   // R 
   MatrixConstantCoefficient R_cf(R);
   // Rᵀ
   MatrixConstantCoefficient Rt_cf(Rt);
   // ϵᵣ (real)
   PWMatrixCoefficient eps_r_cf(dim, attr, coefs_r);
   // ϵᵣ (imag)
   PWMatrixCoefficient eps_i_cf(dim, attr, coefs_i);
   // -ω ϵ₀ ϵᵣ (real)
   ScalarMatrixProductCoefficient neg_omega_eps0_eps_r_cf(neg_omega_eps0, eps_r_cf);
   // ω ϵ₀ ϵᵣ (imag)
   ScalarMatrixProductCoefficient omega_eps0_eps_i_cf(omega_eps0, eps_i_cf);
   // 1
   ConstantCoefficient one_cf(1.0);
   // -1
   ConstantCoefficient neg_one_cf(-1.0);
   // - ω ϵ₀
   ConstantCoefficient neg_omega_eps0_cf(neg_omega_eps0);

   // P B
   ScalarMatrixProductCoefficient PB_r_cf(p_r_cf, B_cf);
   ScalarMatrixProductCoefficient PB_i_cf(p_i_cf, B_cf);

   // cᵢ  
   for (int i = 0; i<ndiffusionequations; i++)
   {
      zerovec[nattr-1] = cvals(i);
      c_cf[i] = new PWConstCoefficient(zerovec);
      signed_c_cf[i] = new ProductCoefficient(csigns(i), *c_cf[i]);
      signed_PB_r_cf[i] = new ScalarMatrixProductCoefficient(*signed_c_cf[i], PB_r_cf);
      signed_PB_i_cf[i] = new ScalarMatrixProductCoefficient(*signed_c_cf[i], PB_i_cf);
   }   

   Array<FiniteElementCollection *> trial_fecols;
   Array<ParFiniteElementSpace *> trial_pfes;
   Array<FiniteElementCollection *> test_fecols;

   // L²(Ω)² space for E
   trial_fecols.Append(new L2_FECollection(order-1, dim));
   trial_pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last(), dim));
   // L²(Ω) space for H
   trial_fecols.Append(new L2_FECollection(order-1, dim));
   trial_pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last()));
   // L²(Ω)² space for J₁, Q₁, J₂, Q₂
   for (int i = 0; i < 2*ndiffusionequations; ++i)
   {
      trial_fecols.Append(new L2_FECollection(order-1, dim));
      trial_pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last(), dim));
   }
   // H⁻¹/²(Γₕ) space for Ê
   trial_fecols.Append(new RT_Trace_FECollection(order-1, dim));
   trial_pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last()));
   // H¹/²(Γₕ) space for Ĥ
   trial_fecols.Append(new H1_Trace_FECollection(order, dim));
   trial_pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last()));
   // H⁻¹/²(Γₕ)² space for Ĵ₁, Q̂₁, Ĵ₂, Q̂₂
   for (int i = 0; i < 2*ndiffusionequations; ++i)
   {
      trial_fecols.Append(new RT_Trace_FECollection(order-1, dim));
      trial_pfes.Append(new ParFiniteElementSpace(&pmesh, trial_fecols.Last(), dim));
   }

   Array<HYPRE_BigInt> tdofs(trial_pfes.Size());
   for (int i = 0; i < trial_pfes.Size(); ++i)
   {
      tdofs[i] = trial_pfes[i]->GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << "ParFiniteElementSpace " << ToString(TrialSpace(i))  << " has " << tdofs[i]
              << " true dofs." << endl;
      }
   }
   if (Mpi::Root())
   {
      cout << "Total number of true dofs: " << tdofs.Sum() << endl;
   }

   // Test spaces   
   // H¹(Ω) space for F
   test_fecols.Append(new H1_FECollection(order, dim));
   // H(curl,Ω) space for W
   test_fecols.Append(new ND_FECollection(order, dim));
   // Test spaces for K₁, L₁, K₂, L₂
   for (int i = 0; i < 2*ndiffusionequations; ++i)
   {
      test_fecols.Append(new H1_FECollection(order, dim));
   }

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_pfes,test_fecols);
   for (int i = 0; i < 2*ndiffusionequations; ++i)
   {
      a->SetTestFECollVdim(i+2,dim);
   }

   // Trial integrators
   // i(ω μ₀ H, F)
   a->AddTrialIntegrator(nullptr,
                         new MixedScalarMassIntegrator(omegamu_cf),
                         TrialSpace::H_space,TestSpace::F_space);

   // (E, ∇×F)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one_cf)),
                         nullptr,
                         TrialSpace::E_space,TestSpace::F_space);
   
   // -i ω ϵ₀ ϵᵣ (E, W) = -i(ω ϵ₀ (ϵᵣᵣ+i ϵᵣᵢ)  E, W)   
   //                   = (ω ϵ₀ ϵᵣᵢ + i (-ωϵ₀ϵᵣᵣ) )  E, W)
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(omega_eps0_eps_i_cf)),
                         new TransposeIntegrator(new VectorFEMassIntegrator(neg_omega_eps0_eps_r_cf)),
                         TrialSpace::E_space,TestSpace::W_space);
   
   // (H, ∇ ×  W)   
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one_cf)), 
                         nullptr,
                         TrialSpace::H_space, TestSpace::W_space);
   
   // - ω ϵ₀ (J₁,W)   
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(neg_omega_eps0_cf)),
                         nullptr,
                         TrialSpace::J1_space,TestSpace::W_space);

   // - ω ϵ₀ (J₂,W)   
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(neg_omega_eps0_cf)),
                         nullptr,
                         TrialSpace::J2_space,TestSpace::W_space);    
                         
   // (Q₁, - b ⋅ ∇ K₁)   
   VectorFunctionCoefficient b_cf(dim, bfunc);
   ScalarVectorProductCoefficient neg_b_cf(-1.0,b_cf);
   a->AddTrialIntegrator(new TransposeIntegrator(new DirectionalVectorGradientIntegrator(neg_b_cf)),
                         nullptr,
                         TrialSpace::Q1_space,TestSpace::K1_space);

   // (c₁ J₁, K₁)   
   a->AddTrialIntegrator(new VectorMassIntegrator(*c_cf[0]),
                         nullptr,
                         TrialSpace::J1_space,TestSpace::K1_space);

   // - c₁(P B E, K₁) = (- c₁ Pᵣ B E, K₁) + i (-c₁ Pᵢ B E, K₁) 
   ScalarMatrixProductCoefficient signed_c1_PB_r_cf((*signed_c_cf[0]), PB_r_cf);
   ScalarMatrixProductCoefficient signed_c1_PB_i_cf((*signed_c_cf[0]), PB_i_cf);  
   a->AddTrialIntegrator(new VectorMassIntegrator(signed_c1_PB_r_cf),
                         new VectorMassIntegrator(signed_c1_PB_i_cf),
                         TrialSpace::E_space,TestSpace::K1_space);

   // (Q₁, L₁)   
   a->AddTrialIntegrator(new VectorMassIntegrator(one_cf),
                         nullptr,
                         TrialSpace::Q1_space,TestSpace::L1_space);

   // (J₁, - b ⋅ ∇L₁)
   a->AddTrialIntegrator(new TransposeIntegrator(new DirectionalVectorGradientIntegrator(neg_b_cf)),
                         nullptr,
                         TrialSpace::J1_space,TestSpace::L1_space);

   // (Q₂, - b⋅ ∇ K₂)
   a->AddTrialIntegrator(new TransposeIntegrator(new DirectionalVectorGradientIntegrator(neg_b_cf)),
                         nullptr,
                         TrialSpace::Q2_space,TestSpace::K2_space);           

   // (c₂ J₂, K₂)       
   a->AddTrialIntegrator(new VectorMassIntegrator(*c_cf[1]),
                         nullptr,
                         TrialSpace::J2_space,TestSpace::K2_space);       

   // c₂(P B E, K₂) = ( c₂ Pᵣ B E, K₂) + i (c₂ Pᵢ B E, K₂)      
   ScalarMatrixProductCoefficient signed_c2_PB_r_cf((*signed_c_cf[1]), PB_r_cf);
   ScalarMatrixProductCoefficient signed_c2_PB_i_cf((*signed_c_cf[1]), PB_i_cf);      
   a->AddTrialIntegrator(new VectorMassIntegrator(signed_c2_PB_r_cf),
                         new VectorMassIntegrator(signed_c2_PB_i_cf),
                         TrialSpace::E_space,TestSpace::K2_space); 

   // (Q₂, L₂)
   a->AddTrialIntegrator(new VectorMassIntegrator(one_cf),
                         nullptr,
                         TrialSpace::Q2_space,TestSpace::L2_space); 
   
   // - (J₂, b ⋅ ∇L₂)
   a->AddTrialIntegrator(new TransposeIntegrator(new DirectionalVectorGradientIntegrator(neg_b_cf)),
                         nullptr,
                         TrialSpace::J2_space,TestSpace::L2_space);

   // Trace integrators
   // < R Ê, F > (we can include R in the variable)
   a->AddTrialIntegrator(new TraceIntegrator,
                         nullptr,
                         TrialSpace::hatE_space,TestSpace::F_space);
   //< Ĥ, W × n >                         
   a->AddTrialIntegrator(new TangentTraceIntegrator,
                         nullptr,
                         TrialSpace::hatH_space,TestSpace::W_space);
   // <Q̂₁, K₁>
   a->AddTrialIntegrator(new VectorTraceIntegrator,nullptr,
                         TrialSpace::hatQ1_space,TestSpace::K1_space);
   // <Ĵ₁, L₁>
   a->AddTrialIntegrator(new VectorTraceIntegrator,nullptr,
                         TrialSpace::hatJ1_space,TestSpace::L1_space);
   // <Q̂₂, K₂>
   a->AddTrialIntegrator(new VectorTraceIntegrator,nullptr,
                         TrialSpace::hatQ2_space,TestSpace::K2_space);
   // <Ĵ₂, L₂>
   a->AddTrialIntegrator(new VectorTraceIntegrator,nullptr,
                         TrialSpace::hatJ2_space,TestSpace::L2_space);

   // Test integrators 
   // (∇×F,∇×δF) = (R ∇ F, R ∇δF) = (Rᵀ R ∇F, ∇δF) = (∇F, ∇δF)

   // i(-ω ϵ₀ ϵᵣ ∇ × F, δW)
   // (-c₁ P B ∇ × F, δK₁) 
   // ( c₂ P B ∇ × F, δK₂)
   // i(ω ϵ₀ ϵᵣ⋆ W, ∇×δF)  
   // (ω² ϵ²₀ ϵᵣϵᵣ⋆ W, δ W)
   // i(-ω ϵ₀c₁ P B ϵᵣ⋆ W, δK₁) 
   // i( ω ϵ₀c₂ P B ϵᵣ⋆ W, δK₂)  
   // (-c₁ P̄ B K₁, ∇×δF)
   // i(ω ϵ₀c₁ P̄ B K₁, δW)
   // (c²₁ P P̄ B B K₁, δK₁)
   // (-c₁ c₂ P P̄ B B K₁, δK₂)
   // (c₂ P̄ B K₂, ∇×δF)
   // i(-ω ϵ₀c₂ P̄ B K₂, δW)
   // (- c₁c₂ P P̄ B B K₂, δK₁)
   // (c²₂ P P̄ B B K₂, δK₂)
   // (∇ × W, ∇ × δW)
   // i(ω μ₀ ∇ × W, δF)
   // i(-ω μ₀ F, ∇ ×δW)
   // (ω² μ₀² F, δF)
   // (c₁²K₁, δK₁) 
   // (-c₁ ω ϵ₀ K₁, δW) 
   // (-K₁,c₁ b ⋅ ∇δL₁)
   // (-c₁ ωϵ₀W, δK₁) 
   // (ω² ϵ₀² W, δW) 
   // (W, ω ϵ₀ b ⋅ ∇δL₁)
   // (-c₁ b⋅∇L₁, δK₁) 
   // (ω ϵ₀ b⋅ ∇L₁, δW) 
   // (b⋅ ∇L₁, b ⋅ ∇δL₁)   
   // (L₁, δL₁) 
   //-(L₁, b ⋅ ∇δK₁) 
   //-(b ⋅ ∇K₁, δL₁) 
   // (b ⋅ ∇K₁, b ⋅ ∇δK₁)
   // (c₂² K₂, δK₂) 
   //-(c₂ ω ϵ₀ K₂, δW) 
   //-(K₂, c₂ b ⋅ ∇δL₂)
   //-(c₂ ω ϵ₀ W, δK₂) 
   // (ω² ϵ₀² W, δW) 
   // (W, ω ϵ₀ b ⋅ ∇δL₂)
   //-(c₂ b⋅ ∇L₂, δK₂) 
   // (ω ϵ₀ b⋅ ∇L₂, δW) 
   // (b⋅ ∇L₂, b ⋅ ∇δL₂)
   // (L₂, δL₂) 
   //-(L₂, b ⋅ ∇δK₂) 
   //-(b ⋅ ∇K₂, δL₂) 
   // (b ⋅ ∇K₂, b ⋅ ∇δK₂)

   // (F, δF) 
   a->AddTestIntegrator(new MassIntegrator(one_cf),
                        nullptr,
                        TestSpace::F_space,TestSpace::F_space);
   // (W, δW) 
   a->AddTestIntegrator(new VectorFEMassIntegrator(one_cf),
                        nullptr,
                        TestSpace::W_space,TestSpace::W_space);
   // (K₁, δK₁) 
   a->AddTestIntegrator(new VectorMassIntegrator(one_cf),
                        nullptr,
                        TestSpace::K1_space,TestSpace::K1_space);                  
   // (L₁, δL₁)
   a->AddTestIntegrator(new VectorMassIntegrator(one_cf),
                        nullptr,
                        TestSpace::L1_space,TestSpace::L1_space); 
   // (K₂, δK₂) 
   a->AddTestIntegrator(new VectorMassIntegrator(one_cf),
                        nullptr,
                        TestSpace::K2_space,TestSpace::K2_space);
   // (L₂, δL₂)
   a->AddTestIntegrator(new VectorMassIntegrator(one_cf),
                        nullptr,
                        TestSpace::L2_space,TestSpace::L2_space);

   a->Assemble();

for (int i = 0; i<ndiffusionequations; i++)
   {
      delete c_cf[i];
      delete signed_c_cf[i];
      delete signed_PB_r_cf[i];
      delete signed_PB_i_cf[i];
   }   

   return 0;
}   