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
//     i ω μ₀ H + ∇ × E                    = 0, in Ω
// -i ω ϵ₀ ϵᵣ E + ∇ × H - α ω ϵ₀ (J₁ + J₂) = 0, in Ω 
//            b ⋅ ∇Q₁ + c₁ J₁ - c₁ P B E = 0, in Ω
//                          Q₁ + b ⋅ ∇J₁ = 0, in Ω 
//            b ⋅ ∇Q₂ + c₂ J₂ + c₂ P B E = 0, in Ω  
//                          Q₂ + b ⋅ ∇J₂ = 0, in Ω
// where B = b ⊗ b. Here α serves as a scaling factor. For α = 0 the system becomes 
// lower trianular (Maxwell decouples from the diffusion equations).

// Note that in 2D we have the following 2 definitions of the curl.
// a) for a 2D vector E, ∇ × E:= ∇ ⋅ (R E) where R = [0 1; -1 0]
// b) for a scalar H, ∇ × H:= R ∇ H

// Define the group variables 
// then the strong formulation reads:
// (A u, v) = (i ω μ₀ H, F) + (∇ × E, F) 
//          - (i ω ϵ₀ ϵᵣ E, W) + (∇ × H, W) - (α ω ϵ₀ (J₁ + J₂), W)
//          + (b ⋅ ∇Q₁, K₁) + (c₁ J₁, K₁) - (c₁ P B E, K₁) + (Q₁, L₁) + (b ⋅ ∇J₁, L₁)
//          + (b ⋅ ∇Q₂, K₂) + (c₂ J₂, K₂) + (c₂ P B E, K₂) + (Q₂, L₂) + (b ⋅ ∇J₂, L₂) 
//
// and the adjoint operator is defined by:
// (u, A^⋆ v) = (E, ∇×F + i ω ϵ₀ ϵ⋆ᵣ W - c₁ P̄ B K₁ + c₂ P̄ B K₂)
//            + (H, -i ω μ₀ F + ∇ × W)
//            + (J₁, -α ω ϵ₀ W + c₁ K₁ - b ⋅ ∇ L₁)
//            + (Q₁, - b ⋅ ∇ K₁ + L₁)
//            + (J₂, -α ω ϵ₀ W + c₂ K₂ - b ⋅ ∇ L₂)
//            + (Q₂, - b ⋅ ∇ K₂ + L₂)
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
// -i ω ϵ₀ ϵᵣ (E, W) + (H, ∇ ×  W) + < Ĥ, W × n > - α ω ϵ₀ (J₁ + J₂, W) = 0,  ∀ W ∈ H(curl,Ω)
// -(Q, b ⋅ ∇ K₁) + <Q̂₁, K₁> + (c₁ J₁, K₁) - c₁(P B E, K₁) = 0,  ∀ K₁ ∈ (H¹(Ω))²
//  (Q₁, L₁) - (J₁, b ⋅ ∇L₁) + <Ĵ₁, L₁> = 0,  ∀ L₁ ∈ (H¹(Ω))²
// -(Q, b⋅ ∇ K₂) + <Q̂₂, K₂> + (c₂ J₂, K₂) + c₂(P B E, K₂) = 0,  ∀ K₂ ∈ (H¹(Ω))²
//  (Q₂, L₂) - (J₂, b ⋅ ∇L₂) + <Ĵ₂, L₂> = 0,  ∀ L₂ ∈ (H¹(Ω))²

// with the adjoint graph test norm defined by:
// ‖v‖²_V = ‖A^⋆ v‖² + ‖v‖²
//        = ‖∇×F + i ω ϵ₀ ϵ⋆ᵣ W - c₁ P̄ B K₁ + c₂ P̄ B K₂‖²
//        + ‖-i ω μ₀ F + ∇ × W‖²
//        + ‖-α ω ϵ₀ W + c₁ K₁ - b ⋅ ∇ L₁‖²
//        + ‖- b ⋅ ∇ K₁ + L₁‖²
//        + ‖-α ω ϵ₀ W + c₂ K₂ - b ⋅ ∇ L₂‖²
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
   Hypre::Init();

   const char *mesh_file = "data/LH_hot.msh";
   int order = 1;
   int delta_order = 1;
   int par_ref_levels = 0;
   int ser_ref_levels = 0;

   real_t rnum=1.5;
   real_t mu = 1.257;
   real_t eps0 = 8.8541878128;
   real_t alpha = 1.0; // scaling factor for E, Js coupling
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
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Scaling factor for E, Js coupling.");
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

   if (!debug) 
   {
      delta = 0.0; // disable delta if electron Landau damping is enabled
      if (Mpi::Root())
      {
         cout << "Electron Landau damping enabled, delta set to 0.0." << endl;
      }
   }    
   else
   {
      if (Mpi::Root())
      {
         alpha = 0.0; 
         cout << "Setting alpha = 0.0 to disable coupling between E and Js." << endl;
         cout << "Debug mode enabled, delta = 0.01 and no coupling." << endl;
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

   real_t omega = 2.*M_PI*rnum;
   real_t neg_omega_eps0 = -omega * eps0;
   real_t omega_eps0 = omega * eps0;

   Vector cvals(ndiffusionequations);
   Vector csigns(ndiffusionequations);
   real_t cfactor = 1e-6;
   cvals(0)  = 25e6;  cvals(1)  = 1e6;
   csigns(0) = -1.0;  csigns(1) = 1.0;
   cvals *= cfactor; // scale the coefficients

   // List of coefficients
   // Trial integrators coefficients
   DenseMatrix R(dim), Rt(dim);
   R(0,0) =  0.0;  R(0,1) = 1.0;
   R(1,0) = -1.0;  R(1,1) = 0.0;
   R.Transpose(Rt);

   VectorFunctionCoefficient no_scale_bvec_cf(dim,bfunc);// b
   real_t scaled_cfactor = std::sqrt(cfactor);
   ScalarVectorProductCoefficient bvec_cf(scaled_cfactor,no_scale_bvec_cf); // scaled b
   // VectorFunctionCoefficient bvec_cf(dim,bfunc);// b
   ScalarVectorProductCoefficient neg_bvec_cf(-1.0,bvec_cf); // -b
   ScalarVectorProductCoefficient alpha_omega_eps0_bvec_cf(eps0*omega*alpha,bvec_cf); // α ω ϵ₀ b


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
   Array<ProductCoefficient *> neg_c_alpha_omega_eps0_cf(ndiffusionequations);
   Array<ProductCoefficient *> c2_cf(ndiffusionequations);
   Array<ProductCoefficient *> signed_c_cf(ndiffusionequations);
   Array<MatrixCoefficient*> signed_PB_r_cf(ndiffusionequations);
   Array<MatrixCoefficient*> signed_PB_i_cf(ndiffusionequations);
   Array<ProductCoefficient*> neg_signed_c_omega_eps_cf(ndiffusionequations);
   Array<ProductCoefficient*> signed_c_omega_eps_cf(ndiffusionequations);
   Array<MatrixCoefficient*> signed_PBR_r_cf(ndiffusionequations);
   Array<MatrixCoefficient*> signed_PBR_i_cf(ndiffusionequations);
   Array<MatrixCoefficient *> c2_absP2BB_cf(ndiffusionequations);
   Array<VectorCoefficient*> neg_c_bvec_cf(ndiffusionequations);

   Vector zerovec(nattr); zerovec=0.0;

   // ϵᵣ in plasma (real)
   MatrixFunctionCoefficient eps_r_temp_cf(dim, epsilon_func_r);
   // ϵᵣ in plasma(imag)
   MatrixFunctionCoefficient eps_i_temp_cf(dim, epsilon_func_i);

   coefs_r[nattr-1] = &eps_r_temp_cf;
   coefs_i[nattr-1] = &eps_i_temp_cf;

   // ωμ₀ 
   ConstantCoefficient omegamu_cf(mu * omega);
   // -ωμ₀
   ConstantCoefficient neg_omegamu_cf(-mu * omega);
   // ω² μ₀²
   ConstantCoefficient omega2mu2_cf(mu * mu * omega * omega);  
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
   // ω ϵ₀ ϵᵣᵢ R 
   MatrixProductCoefficient omega_eps0_eps_i_R_cf(omega_eps0_eps_i_cf, R_cf);
   // -ω ϵ₀ ϵᵣᵣ R
   MatrixProductCoefficient neg_omega_eps0_eps_r_R_cf(neg_omega_eps0_eps_r_cf, R_cf); 
   // 
   // ω ϵ₀ Rᵀ ϵᵣᵢ
   MatrixProductCoefficient omega_eps0_Rt_eps_i_cf(Rt_cf, omega_eps0_eps_i_cf);
   // ω ϵ₀ ϵᵣ 
   ScalarMatrixProductCoefficient omega_eps0_eps_r_cf(omega_eps0, eps_r_cf);
   // ω ϵ₀ Rᵀ ϵᵣᵣ 
   MatrixProductCoefficient omega_eps0_Rt_eps_r_cf(Rt_cf, omega_eps0_eps_r_cf);
   // |ϵᵣ|² = ϵᵣᵣϵᵣᵣ + ϵᵣᵢϵᵣᵢ   
   MatrixProductCoefficient eps_r_eps_r_cf(eps_r_cf, eps_r_cf);
   MatrixProductCoefficient eps_i_eps_i_cf(eps_i_cf, eps_i_cf);

   MatrixSumCoefficient abseps2_cf(eps_r_eps_r_cf, eps_i_eps_i_cf);
   // ω² ϵ₀² 
   ConstantCoefficient alpha2_omega2_eps02(alpha*alpha*omega * omega * eps0 * eps0);
   // ω² ϵ₀² |ϵᵣ|²   
   ScalarMatrixProductCoefficient omega2_eps02_abseps2_cf(omega*omega*eps0*eps0, abseps2_cf);

   // 1
   ConstantCoefficient one_cf(1.0);
   // -1
   ConstantCoefficient neg_one_cf(-1.0);
   // - ω ϵ₀
   ConstantCoefficient neg_omega_eps0_cf(neg_omega_eps0);
   // - α ω ϵ₀
   ConstantCoefficient neg_alpha_omega_eps0_cf(alpha * neg_omega_eps0); 

   // P B
   ScalarMatrixProductCoefficient PB_r_cf(p_r_cf, B_cf);
   ScalarMatrixProductCoefficient PB_i_cf(p_i_cf, B_cf);

   // BB
   MatrixProductCoefficient BB_cf(B_cf, B_cf);
   
   // Pᵣ²
   ProductCoefficient P2_r_cf(p_r_cf, p_r_cf);
   // Pᵢ²
   ProductCoefficient P2_i_cf(p_i_cf, p_i_cf);
   // |P|²
   SumCoefficient absP2_cf(P2_r_cf, P2_i_cf);
   // |P|² B B
   ScalarMatrixProductCoefficient absP2_BB_cf(absP2_cf, BB_cf);
   // cᵢ  
   for (int i = 0; i<ndiffusionequations; i++)
   {
      zerovec[nattr-1] = cvals(i);
      c_cf[i] = new PWConstCoefficient(zerovec);
      neg_c_alpha_omega_eps0_cf[i] = new ProductCoefficient(-omega_eps0*alpha, *c_cf[i]);
      c2_cf[i] = new ProductCoefficient(*c_cf[i], *c_cf[i]);
      signed_c_cf[i] = new ProductCoefficient(csigns(i), *c_cf[i]);
      signed_PB_r_cf[i] = new ScalarMatrixProductCoefficient(*signed_c_cf[i], PB_r_cf);
      signed_PB_i_cf[i] = new ScalarMatrixProductCoefficient(*signed_c_cf[i], PB_i_cf);
      signed_c_omega_eps_cf[i] = new ProductCoefficient(omega_eps0, *signed_c_cf[i]);
      neg_signed_c_omega_eps_cf[i] = new ProductCoefficient(neg_omega_eps0, *signed_c_cf[i]);
      // cᵢ P B R
      signed_PBR_r_cf[i] = new MatrixProductCoefficient(*signed_PB_r_cf[i], R_cf);
      signed_PBR_i_cf[i] = new MatrixProductCoefficient(*signed_PB_i_cf[i], R_cf);
      // cᵢ² |P|² B B
      c2_absP2BB_cf[i] = new ScalarMatrixProductCoefficient(*c2_cf[i], absP2_BB_cf);
      neg_c_bvec_cf[i] = new ScalarVectorProductCoefficient(-1.0 * cvals(i), bvec_cf);
   }   

   ProductCoefficient neg_c1_c2_cf(*signed_c_cf[0], *signed_c_cf[1]);
   // -c₁ c₂ |P|² B B
   ScalarMatrixProductCoefficient neg_c1_c2_absP2BB_cf(neg_c1_c2_cf, absP2_BB_cf);


   MatrixProductCoefficient PB_r_eps_r_cf(PB_r_cf, eps_r_cf);
   MatrixProductCoefficient PB_r_eps_i_cf(PB_r_cf, eps_i_cf);
   MatrixProductCoefficient PB_i_eps_r_cf(PB_i_cf, eps_r_cf);
   MatrixProductCoefficient PB_i_eps_i_cf(PB_i_cf, eps_i_cf);


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
   


   VectorFunctionCoefficient b_cf(dim, bfunc);
   ScalarVectorProductCoefficient neg_b_cf(-1.0,b_cf);   

   for (int i = 0; i < ndiffusionequations; i++)
   {
      // - α ω ϵ₀ (Jᵢ,W)   
      a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(neg_alpha_omega_eps0_cf)),
                            nullptr,
                            TrialSpace::J1_space + 2*i,TestSpace::W_space);
      // (Qᵢ, - b ⋅ ∇ Kᵢ)   
      a->AddTrialIntegrator(new TransposeIntegrator(new DirectionalVectorGradientIntegrator(neg_b_cf)),
                         nullptr,
                         TrialSpace::Q1_space + 2*i,TestSpace::K1_space+ 2*i);

      // (cᵢ Jᵢ, Kᵢ)
      a->AddTrialIntegrator(new VectorMassIntegrator(*c_cf[i]),
                            nullptr,
                            TrialSpace::J1_space + 2*i,TestSpace::K1_space + 2*i);       
                            
      // (Qᵢ, Lᵢ)
      a->AddTrialIntegrator(new VectorMassIntegrator(one_cf),
                            nullptr,
                            TrialSpace::Q1_space + 2*i,TestSpace::L1_space + 2*i);      

      // (Jᵢ, - b ⋅ ∇Lᵢ)
      a->AddTrialIntegrator(new TransposeIntegrator(new DirectionalVectorGradientIntegrator(neg_b_cf)),
                            nullptr,
                            TrialSpace::J1_space + 2*i,TestSpace::L1_space + 2*i);

      // - c₁(P B E, K₁) = (- c₁ Pᵣ B E, K₁) + i (-c₁ Pᵢ B E, K₁) 
      //   c₂(P B E, K₂) = (  c₂ Pᵣ B E, K₂) + i ( c₂ Pᵢ B E, K₂) 
      a->AddTrialIntegrator(new VectorMassIntegrator(*signed_PB_r_cf[i]),
                            new VectorMassIntegrator(*signed_PB_i_cf[i]),
                            TrialSpace::E_space,TestSpace::K1_space + 2*i);

      // Trace integrators 
      // <Q̂ᵢ, Kᵢ>
      a->AddTrialIntegrator(new VectorTraceIntegrator,nullptr,
                            TrialSpace::hatQ1_space + 2*i,TestSpace::K1_space + 2*i);
      // <Ĵᵢ, Lᵢ>
      a->AddTrialIntegrator(new VectorTraceIntegrator,nullptr,
                            TrialSpace::hatJ1_space + 2*i,TestSpace::L1_space + 2*i);                      

   }

   // Trace integrators
   // < R Ê, F > (we include R in the variable)
   a->AddTrialIntegrator(new TraceIntegrator,
                         nullptr,
                         TrialSpace::hatE_space,TestSpace::F_space);
   //< Ĥ, W × n >                         
   a->AddTrialIntegrator(new TangentTraceIntegrator,
                         nullptr,
                         TrialSpace::hatH_space,TestSpace::W_space);

   // -------------------------------------------------------------------------------   
   //                         Test integrators 
   // -------------------------------------------------------------------------------   
   // (∇×F,∇×δF) = (R ∇ F, R ∇δF) = (Rᵀ R ∇F, ∇δF) = (∇F, ∇δF)
   a->AddTestIntegrator(new DiffusionIntegrator(one_cf),
                        nullptr,
                        TestSpace::F_space, TestSpace::F_space);
   // i(-ω ϵ₀ ϵᵣ ∇ × F, δW) = i(-ω ϵ₀ (ϵᵣᵣ + i ϵᵣᵢ) ∇ × F, δW)
   //                       = (ω ϵ₀ ϵᵣᵢ ∇ × F, δW) + i(-ω ϵ₀ ϵᵣᵣ ∇ × F, δW)
   //                       = (ω ϵ₀ ϵᵣᵢ R ∇ F, δW) + i(-ω ϵ₀ ϵᵣᵣ R ∇ F, δW)
   a->AddTestIntegrator(new MixedVectorGradientIntegrator(omega_eps0_eps_i_R_cf),
                        new MixedVectorGradientIntegrator(neg_omega_eps0_eps_r_R_cf),
                        TestSpace::F_space, TestSpace::W_space);

   for (int i = 0; i<ndiffusionequations; i++)
   {
      // (-c₁ P B ∇ × F, δK₁) = (-c₁ Pᵣ B ∇ × F, δK₁) + i (-c₁ Pᵢ B ∇ × F, δK₁)
      // ( c₂ P B ∇ × F, δK₂) = ( c₂ Pᵣ B ∇ × F, δK₂) + i ( c₂ Pᵢ B ∇ × F, δK₂)
      a->AddTestIntegrator(new MixedCurlIntegrator(*signed_PB_r_cf[i]),
                           new MixedCurlIntegrator(*signed_PB_i_cf[i]),
                           TestSpace::F_space, TestSpace::K1_space+ 2*i);




   }                     

   // i(ω ϵ₀ ϵᵣ⋆ W, ∇×δF) = i(ω ϵ₀ (ϵᵣᵣ - i ϵᵣᵢ) W, ∇×δF) (ϵᵣᵣ & ϵᵣᵢ are symmetric)
   //                     = (ω ϵ₀ ϵᵣᵢ W, ∇×δF) + i(ω ϵ₀ ϵᵣᵣ W, ∇×δF) 
   //                     = (ω ϵ₀ Rᵀ ϵᵣᵢ W, ∇δF) + i(ω ϵ₀ Rᵀ ϵᵣᵣ W, ∇δF) 
   a->AddTestIntegrator(new TransposeIntegrator(new MixedVectorGradientIntegrator(omega_eps0_Rt_eps_i_cf)),
                        new TransposeIntegrator(new MixedVectorGradientIntegrator(omega_eps0_Rt_eps_r_cf)),
                        TestSpace::W_space, TestSpace::F_space);
   // (ω² ϵ²₀ ϵᵣϵᵣ⋆ W, δ W) = (ω² ϵ²₀ |ϵᵣ|² W, δW) 
   a->AddTestIntegrator(new VectorFEMassIntegrator(omega2_eps02_abseps2_cf),
                        nullptr,
                        TestSpace::W_space, TestSpace::W_space); 
   
   // Pᵢ B ϵᵣᵣ - Pᵣ B ϵᵣᵢ
   MatrixSumCoefficient PB_i_eps_r_minus_PB_r_eps_i_cf(PB_i_eps_r_cf, PB_r_eps_i_cf, 1.0, -1.0);
   // Pᵣ B ϵᵣᵣ + Pᵢ B ϵᵣᵢ)
   MatrixSumCoefficient PB_r_eps_r_plus_PB_i_eps_i_cf(PB_r_eps_r_cf, PB_i_eps_i_cf, 1.0, 1.0);
   //   ω ϵ₀ c₁ (Pᵢ B ϵᵣᵣ - Pᵣ B ϵᵣᵢ) (negsigned)
   //  -ω ϵ₀ c₂ (Pᵢ B ϵᵣᵣ - Pᵣ B ϵᵣᵢ) (negsigned) 
   // - ω ϵ₀ c₁ (Pᵣ B ϵᵣᵣ + Pᵢ B ϵᵣᵢ) (signed)
   //   ω ϵ₀ c₂ (Pᵣ B ϵᵣᵣ + Pᵢ B ϵᵣᵢ) (signed)
   Array<MatrixCoefficient*> neg_signed_c_PB_i_eps_r_minus_PB_r_eps_i_cf(ndiffusionequations);
   Array<MatrixCoefficient*> signed_c_PB_i_eps_r_minus_PB_r_eps_i_cf(ndiffusionequations);
   Array<MatrixCoefficient*> signed_c_PB_r_eps_r_plus_PB_i_eps_i_cf(ndiffusionequations);
   Array<MatrixCoefficient*> neg_signed_c_PB_r_eps_r_plus_PB_i_eps_i_cf(ndiffusionequations);
   for (int i = 0; i<ndiffusionequations; i++)
   {
      neg_signed_c_PB_i_eps_r_minus_PB_r_eps_i_cf[i] = new ScalarMatrixProductCoefficient(*neg_signed_c_omega_eps_cf[i], PB_i_eps_r_minus_PB_r_eps_i_cf);
      signed_c_PB_r_eps_r_plus_PB_i_eps_i_cf[i] = new ScalarMatrixProductCoefficient(*signed_c_omega_eps_cf[i], PB_r_eps_r_plus_PB_i_eps_i_cf);
   
      // i(-ω ϵ₀c₁ P B ϵᵣ⋆ W, δK₁) = ω ϵ₀ c₁ (Pᵢ B ϵᵣᵣ - Pᵣ B ϵᵣᵢ) + i (-ωϵ₀ c₁ (Pᵣ B ϵᵣᵣ + Pᵢ B ϵᵣᵢ))
      // i( ω ϵ₀c₂ P B ϵᵣ⋆ W, δK₂) = -ω ϵ₀ c₂ (Pᵢ B ϵᵣᵣ - Pᵣ B ϵᵣᵢ) + i (ωϵ₀ c₂ (Pᵣ B ϵᵣᵣ + Pᵢ B ϵᵣᵢ))
      // TestSpace K1_space and K2_space (2,4)
      TestSpace tspace = static_cast<TestSpace>(2*i+2);
      a->AddTestIntegrator(new VectorFEMassIntegrator(*neg_signed_c_PB_i_eps_r_minus_PB_r_eps_i_cf[i]),
                           new VectorFEMassIntegrator(*signed_c_PB_r_eps_r_plus_PB_i_eps_i_cf[i]),
                           TestSpace::W_space, tspace);
   
      signed_c_PB_i_eps_r_minus_PB_r_eps_i_cf[i] = new ScalarMatrixProductCoefficient(*signed_c_omega_eps_cf[i], PB_i_eps_r_minus_PB_r_eps_i_cf);
      neg_signed_c_PB_r_eps_r_plus_PB_i_eps_i_cf[i] = new ScalarMatrixProductCoefficient(*neg_signed_c_omega_eps_cf[i], PB_r_eps_r_plus_PB_i_eps_i_cf);
      // Note that P is scalar and also ϵᵣ B = B ϵᵣ 
      // i( ω ϵ₀c₁ ϵᵣ P̄ B K₁, δW) = i (K₁,  ω ϵ₀ c₁ P B ϵᵣ⋆ δW) 
      // i(-ω ϵ₀c₂ ϵᵣ P̄ B K₂, δW) = i (K₂, -ω ϵ₀ c₂ P B ϵᵣ⋆ δW)
      a->AddTestIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(*signed_c_PB_i_eps_r_minus_PB_r_eps_i_cf[i])),
                           new TransposeIntegrator(new VectorFEMassIntegrator(*neg_signed_c_PB_r_eps_r_plus_PB_i_eps_i_cf[i])),
                           tspace, TestSpace::W_space);
      // (c²₁ P P̄ B B K₁, δK₁)
      // (c²₂ P P̄ B B K₂, δK₂)
      a->AddTestIntegrator(new VectorMassIntegrator(*c2_absP2BB_cf[i]),
                           nullptr,
                           tspace, tspace);

      // (c₁²K₁, δK₁) 
      // (c₂²K₂, δK₂)
      a->AddTestIntegrator(new VectorMassIntegrator(*c2_cf[i]),
                           nullptr,
                           tspace, tspace);     
                           
      // (-c₁ α ω ϵ₀ W, δK₁) 
      // (-c₂ α ω ϵ₀ W, δK₂)
      a->AddTestIntegrator(new VectorFEMassIntegrator(*neg_c_alpha_omega_eps0_cf[i]),
                           nullptr,
                           TestSpace::W_space, tspace);
      // (-c₁ α  ω ϵ₀ K₁, δW) 
      // (-c₂ α  ω ϵ₀ K₂, δW)                            
      a->AddTestIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(*neg_c_alpha_omega_eps0_cf[i])),
                           nullptr,
                           tspace, TestSpace::W_space);

      // (-c₁ b⋅∇L₁, δK₁) 
      // (-c₂ b⋅∇L₂, δK₂) 
      TestSpace Kspace = static_cast<TestSpace>(2*i+2);
      TestSpace Lspace = static_cast<TestSpace>(2*i+3);

      a->AddTestIntegrator(new DirectionalVectorGradientIntegrator(*neg_c_bvec_cf[i]),
                           nullptr,
                           Lspace, Kspace);
      // (K₁,-c₁ b ⋅ ∇δL₁)
      // (K₂,-c₂ b ⋅ ∇δL₂)
      a->AddTestIntegrator(new TransposeIntegrator(new DirectionalVectorGradientIntegrator(*neg_c_bvec_cf[i])),
                           nullptr,
                           Kspace, Lspace);
      // (L₁, δL₁) 
      // (L₂, δL₂)
      a->AddTestIntegrator(new VectorMassIntegrator(one_cf),
                           nullptr,
                           Lspace, Lspace);   
      // (b⋅ ∇L₁, b ⋅ ∇δL₁)   
      // (b⋅ ∇L₂, b ⋅ ∇δL₂)
      a->AddTestIntegrator(new DirectionalVectorDiffusionIntegrator(bvec_cf),
                           nullptr, Lspace, Lspace);
      // (b ⋅ ∇K₁, b ⋅ ∇δK₁)
      // (b ⋅ ∇K₂, b ⋅ ∇δK₂)       
      a->AddTestIntegrator(new DirectionalVectorDiffusionIntegrator(bvec_cf),
                           nullptr, Kspace, Kspace);                    

      //-(b ⋅ ∇K₁, δL₁) 
      //-(b ⋅ ∇K₂, δL₂) 
      a->AddTestIntegrator(new DirectionalVectorGradientIntegrator(neg_b_cf),
                           nullptr, Kspace, Lspace);
      //(L₁, - b ⋅ ∇δK₁) 
      //(L₂, - b ⋅ ∇δK₂) 
      a->AddTestIntegrator(new TransposeIntegrator(new DirectionalVectorGradientIntegrator(neg_b_cf)),
                           nullptr, Lspace, Kspace);

      // (α ω ϵ₀ b⋅ ∇L₁, δW) 
      // (α ω ϵ₀ b⋅ ∇L₂, δW)                            
      a->AddTestIntegrator(new MixedDirectionalVectorGradientIntegrator(alpha_omega_eps0_bvec_cf),
                           nullptr,
                           Lspace, TestSpace::W_space);
      // (W, α ω ϵ₀ b ⋅ ∇δL₁)
      // (W, α ω ϵ₀ b ⋅ ∇δL₂)
      a->AddTestIntegrator(new TransposeIntegrator(new MixedDirectionalVectorGradientIntegrator(alpha_omega_eps0_bvec_cf)),
                           nullptr,
                           TestSpace::W_space, Lspace);

   }

   // (-c₁ P̄ B K₁, ∇×δF) = (K₁, -c₁ P B ∇×δF) = (K₁, -c₁ Pᵣ B ∇×δF) + i (K₁, -c₁ Pᵢ B ∇×δF)
   a->AddTestIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(*signed_PB_r_cf[0])),
                        new TransposeIntegrator(new MixedCurlIntegrator(*signed_PB_i_cf[0])),
                        TestSpace::K1_space, TestSpace::F_space); 
   // (c₂ P̄ B K₂, ∇×δF) = (K₂, c₂ P B ∇×δF) = (K₂, c₂ Pᵣ B ∇×δF) + i (K₂, c₂ Pᵢ B ∇×δF)
   a->AddTestIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(*signed_PB_r_cf[1])),
                        new TransposeIntegrator(new MixedCurlIntegrator(*signed_PB_i_cf[1])),
                        TestSpace::K2_space, TestSpace::F_space); 


   // (- c₁c₂ P P̄ B B K₁, δK₂)
   // (- c₁c₂ P P̄ B B K₂, δK₁)
   a->AddTestIntegrator(new VectorMassIntegrator(neg_c1_c2_absP2BB_cf),
                        nullptr,
                        TestSpace::K1_space, TestSpace::K2_space);
   a->AddTestIntegrator(new VectorMassIntegrator(neg_c1_c2_absP2BB_cf),
                        nullptr,
                        TestSpace::K2_space, TestSpace::K1_space);

   // (∇ × W, ∇ × δW)
   a->AddTestIntegrator(new CurlCurlIntegrator(one_cf),
                        nullptr,
                        TestSpace::W_space, TestSpace::W_space);
   // i(ω μ₀ ∇ × W, δF)
   a->AddTestIntegrator(nullptr,
                        new MixedCurlIntegrator(omegamu_cf),
                        TestSpace::W_space, TestSpace::F_space);
   // i(-ω μ₀ F, ∇ ×δW)
   a->AddTestIntegrator(nullptr, 
                        new TransposeIntegrator(new MixedCurlIntegrator(neg_omegamu_cf)),
                        TestSpace::F_space, TestSpace::W_space);
   // (ω² μ₀² F, δF)
   a->AddTestIntegrator(new MassIntegrator(omega2mu2_cf),
                        nullptr,
                        TestSpace::F_space, TestSpace::F_space);
   // (α² ω² ϵ₀² W, δW) 
   a->AddTestIntegrator(new VectorFEMassIntegrator(alpha2_omega2_eps02),
                        nullptr,
                        TestSpace::W_space,TestSpace::W_space);   

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
      delete signed_PBR_r_cf[i];
      delete signed_PBR_i_cf[i];
   }   

   socketstream E_out_r;

   int npfes = trial_pfes.Size();
   Array<int> offsets(npfes+1);  offsets[0] = 0;
   Array<int> toffsets(npfes+1); toffsets[0] = 0;
   for (int i = 0; i<npfes; i++)
   {
      offsets[i+1] = trial_pfes[i]->GetVSize();
      toffsets[i+1] = trial_pfes[i]->TrueVSize();
   }
   offsets.PartialSum();
   toffsets.PartialSum();

   Vector x(2*offsets.Last());
   x = 0.;
   
   Array<ParGridFunction *> pgf_r(npfes);
   Array<ParGridFunction *> pgf_i(npfes);

   for (int i = 0; i < npfes; ++i)
   {
      pgf_r[i] = new ParGridFunction(trial_pfes[i], x, offsets[i]);
      pgf_i[i] = new ParGridFunction(trial_pfes[i], x, offsets.Last() + offsets[i]);
   }

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);
   ParGridFunction E_par_r(&L2_fes);
   ParGridFunction E_par_i(&L2_fes);

   ParaViewDataCollection * paraview_dc = nullptr;

   std::string output_dir = "ParaView/UW/" + GetTimestamp();

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
      paraview_dc->RegisterField("E_par_r",&E_par_r);
      paraview_dc->RegisterField("E_par_i",&E_par_i);
      paraview_dc->RegisterField("H_r",pgf_r[1]);
      paraview_dc->RegisterField("H_i",pgf_i[1]);      
      paraview_dc->RegisterField("J_1_r",pgf_r[2]);
      paraview_dc->RegisterField("J_1_i",pgf_i[2]);
      paraview_dc->RegisterField("Q_1_r",pgf_r[3]);
      paraview_dc->RegisterField("Q_1_i",pgf_i[3]);
      paraview_dc->RegisterField("J_2_r",pgf_r[4]);
      paraview_dc->RegisterField("J_2_i",pgf_i[4]);
      paraview_dc->RegisterField("Q_2_r",pgf_r[5]);
      paraview_dc->RegisterField("Q_2_i",pgf_i[5]);
   }

   Array<int> ess_tdof_list;
   Array<int> ess_tdof_listJhat;
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

      trial_pfes[2+2*ndiffusionequations]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += toffsets[2+2*ndiffusionequations];
      }
      // ess_bdr=1;
      for (int i = 0; i<ndiffusionequations;i++)
      {
         ess_tdof_listJhat.SetSize(0);
         trial_pfes[2*i+8]->GetEssentialTrueDofs(ess_bdr, ess_tdof_listJhat);
         for (int j = 0; j < ess_tdof_listJhat.Size(); j++)
         {
            ess_tdof_listJhat[j] += toffsets[2*i+8];
         }

         ess_tdof_list.Append(ess_tdof_listJhat);
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

   pgf_r[2+2*ndiffusionequations]->ProjectBdrCoefficientNormal(rot_one_x_cf, one_r_bdr);
   pgf_r[2+2*ndiffusionequations]->ProjectBdrCoefficientNormal(rot_negone_x_cf, negone_r_bdr);
   pgf_i[2+2*ndiffusionequations]->ProjectBdrCoefficientNormal(rot_one_x_cf, one_i_bdr);
   pgf_i[2+2*ndiffusionequations]->ProjectBdrCoefficientNormal(rot_negone_x_cf, negone_i_bdr);

   OperatorPtr Ah;
   Vector X,B;
   a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);
   ComplexOperator * Ahc = Ah.As<ComplexOperator>();


   BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
   int nblocks = BlockA_r->NumRowBlocks();

   {
      ComplexBlockOperator Ac(*Ahc);
      Vector Xc(X.Size()); Xc = 0.0;
      Vector Bc(B.Size());
      Ac.BlockComplexToComplexBlock(B, Bc);
   
      BlockDiagonalPreconditioner Mc(Ac.RowOffsets());
      for (int i = 0; i < nblocks; ++i)
      {
         auto solver = new ComplexMUMPSSolver(MPI_COMM_WORLD);
         solver->SetPrintLevel(0);
         solver->SetOperator(Ac.GetBlock(i,i));
         Mc.SetDiagonalBlock(i, solver);
      }

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-5);
      cg.SetMaxIter(500);
      cg.SetPrintLevel(1);
      cg.SetPreconditioner(Mc);
      cg.SetOperator(Ac);
      cg.Mult(Bc, Xc);
      Ac.ComplexBlockToBlockComplex(Xc, X);
   }

   a->RecoverFEMSolution(X, x);

   for (int i = 0; i < npfes; ++i)
   {
      pgf_r[i]->MakeRef(trial_pfes[i], x, offsets[i]);
      pgf_i[i]->MakeRef(trial_pfes[i], x, offsets.Last() + offsets[i]);
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
   for (int i = 0; i < trial_fecols.Size(); ++i)
   {
      delete trial_fecols[i];
      delete trial_pfes[i];
   }
   for (int i = 0; i< test_fecols.Size(); ++i)
   {
      delete test_fecols[i];
   }


   return 0;
}   

