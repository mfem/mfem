//                   MFEM Ultraweak DPG Maxwell parallel example
//
// Compile with: make pmaxwell
//
// sample run
// mpirun -np 4 pmaxwell -m ../../data/star.mesh -o 2 -sref 0 -pref 3 -rnum 0.5 -prob 0
// mpirun -np 4 pmaxwell -m ../../data/inline-quad.mesh -o 3 -sref 0 -pref 3 -rnum 4.8 -sc -prob 0
// mpirun -np 4 pmaxwell -m ../../data/inline-hex.mesh -o 2 -sref 0 -pref 1 -rnum 0.8 -sc -prob 0
// mpirun -np 4 pmaxwell -m ../../data/inline-quad.mesh -o 3 -sref 1 -pref 3 -rnum 4.8 -sc -prob 2
// mpirun -np 4 pmaxwell -o 3 -sref 1 -pref 2 -rnum 11.8 -sc -prob 3
// mpirun -np 4 pmaxwell -o 3 -sref 1 -pref 2 -rnum 9.8 -sc -prob 4

// AMR run. Note that this is a computationally intensive sample run.
// We recommend trying it on a large machine with more mpi ranks
// mpirun -np 4 pmaxwell -o 3 -sref 0 -pref 15 -prob 1 -theta 0.7 -sc

// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Maxwell problem

//      ∇×(1/μ ∇×E) - ω² ϵ E = Ĵ ,   in Ω
//                       E×n = E₀ , on ∂Ω

// It solves the following kinds of problems
// 1) Known exact solutions with error convergence rates
//    a) A manufactured solution problem where E is a plane beam
// 2) Fichera "microwave" problem
// 3) PML problems
//    a) Generic PML problem with point source given by the load
//    b) Plane wave scattering from a square
//    c) PML problem with a point source prescribed on the boundary

// The DPG UW deals with the First Order System
//  i ω μ H + ∇ × E = 0,   in Ω
// -i ω ϵ E + ∇ × H = J,   in Ω
//            E × n = E_0, on ∂Ω
// Note: Ĵ = -iωJ

// The ultraweak-DPG formulation is obtained by integration by parts of both
// equations and the introduction of trace unknowns on the mesh skeleton

// in 2D
// E is vector valued and H is scalar.
//    (∇ × E, F) = (E, ∇ × F) + < n × E , F>
// or (∇ ⋅ AE , F) = (AE, ∇ F) + < AE ⋅ n, F>
// where A = [0 1; -1 0];

// E ∈ (L²(Ω))² , H ∈ L²(Ω)
// Ê ∈ H^-1/2(Ω)(Γₕ), Ĥ ∈ H^1/2(Γₕ)
//  i ω μ (H,F) + (E, ∇ × F) + < AÊ, F > = 0,      ∀ F ∈ H¹
// -i ω ϵ (E,G) + (H,∇ × G)  + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)
//                                        Ê = E₀      on ∂Ω
// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) |   < Ê, F >   |              |         |
// |   |              |             |              |              |         |
// | G | -i ω ϵ (E,G) |  (H,∇ × G)  |              | < Ĥ, G × n > |  (J,G)  |
// where (F,G) ∈  H¹ × H(curl,Ω)

// in 3D
// E,H ∈ (L^2(Ω))³
// Ê ∈ H_0^1/2(Ω)(curl, Γₕ), Ĥ ∈ H^-1/2(curl, Γₕ)
//  i ω μ (H,F) + (E,∇ × F) + < Ê, F × n > = 0,      ∀ F ∈ H(curl,Ω)
// -i ω ϵ (E,G) + (H,∇ × G) + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)
//                                   Ê × n = E₀      on ∂Ω
// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) | < n × Ê, F > |              |         |
// |   |              |             |              |              |         |
// | G | -i ω ϵ (E,G) |  (H,∇ × G)  |              | < n × Ĥ, G > |  (J,G)  |
// where (F,G) ∈  H(curl,Ω) × H(curl,Ω)

// Here we use the "Adjoint Graph" norm on the test space i.e.,
// ||(F,G)||²ᵥ  = ||A^*(F,G)||² + ||(F,G)||² where A is the
// maxwell operator defined by (1)

// The PML formulation is

//      ∇×(1/μ α ∇×E) - ω² ϵ β E = Ĵ ,   in Ω
//                E×n = E₀ , on ∂Ω

// where α = |J|⁻¹ Jᵀ J (in 2D it's the scalar |J|⁻¹),
// β = |J| J⁻¹ J⁻ᵀ, J is the Jacobian of the stretching map
// and |J| its determinant.

// The first order system reads
//  i ω μ α⁻¹ H + ∇ × E = 0,   in Ω
//    -i ω ϵ β E + ∇ × H = J,   in Ω
//                 E × n = E₀,  on ∂Ω

// and the ultraweak formulation is

// in 2D
// E ∈ (L²(Ω))² , H ∈ L²(Ω)
// Ê ∈ H^-1/2(Ω)(Γₕ), Ĥ ∈ H^1/2(Γₕ)
//  i ω μ (α⁻¹ H,F) + (E, ∇ × F) + < AÊ, F > = 0,          ∀ F ∈ H¹
// -i ω ϵ (β E,G)   + (H,∇ × G)  + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)
//                                            Ê = E₀     on ∂Ω
// ---------------------------------------------------------------------------------
// |   |       E        |        H         |      Ê       |       Ĥ      |  RHS    |
// ---------------------------------------------------------------------------------
// | F |  (E,∇ × F)     | i ω μ (α⁻¹ H,F)  |   < Ê, F >   |              |         |
// |   |                |                  |              |              |         |
// | G | -i ω ϵ (β E,G) |    (H,∇ × G)     |              | < Ĥ, G × n > |  (J,G)  |

// where (F,G) ∈  H¹ × H(curl,Ω)

//
// in 3D
// E,H ∈ (L^2(Ω))³
// Ê ∈ H_0^1/2(Ω)(curl, Γ_h), Ĥ ∈ H^-1/2(curl, Γₕ)
//  i ω μ (α⁻¹ H,F) + (E,∇ × F) + < Ê, F × n > = 0,      ∀ F ∈ H(curl,Ω)
// -i ω ϵ (β E,G)    + (H,∇ × G) + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)
//                                        Ê × n = E_0     on ∂Ω
// -------------------------------------------------------------------------------
// |   |       E      |      H           |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------------
// | F |  ( E,∇ × F)  | i ω μ (α⁻¹ H,F)  | < n × Ê, F > |              |         |
// |   |              |                  |              |              |         |
// | G | -iωϵ (β E,G) |   (H,∇ × G)      |              | < n × Ĥ, G > |  (J,G)  |
// where (F,G) ∈  H(curl,Ω) × H(curl,Ω)

// For more information see https://doi.org/10.1016/j.camwa.2021.01.017

#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "util/pml.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

void E_exact_r(const Vector &x, Vector & E_r);
void E_exact_i(const Vector &x, Vector & E_i);

void H_exact_r(const Vector &x, Vector & H_r);
void H_exact_i(const Vector &x, Vector & H_i);


void  rhs_func_r(const Vector &x, Vector & J_r);
void  rhs_func_i(const Vector &x, Vector & J_i);

void curlE_exact_r(const Vector &x, Vector &curlE_r);
void curlE_exact_i(const Vector &x, Vector &curlE_i);
void curlH_exact_r(const Vector &x,Vector &curlH_r);
void curlH_exact_i(const Vector &x,Vector &curlH_i);

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r);
void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i);

void hatE_exact_r(const Vector & X, Vector & hatE_r);
void hatE_exact_i(const Vector & X, Vector & hatE_i);

void hatH_exact_r(const Vector & X, Vector & hatH_r);
void hatH_exact_i(const Vector & X, Vector & hatH_i);

double hatH_exact_scalar_r(const Vector & X);
double hatH_exact_scalar_i(const Vector & X);

void maxwell_solution(const Vector & X,
                      std::vector<complex<double>> &E);

void maxwell_solution_curl(const Vector & X,
                           std::vector<complex<double>> &curlE);

void maxwell_solution_curlcurl(const Vector & X,
                               std::vector<complex<double>> &curlcurlE);

void source_function(const Vector &x, Vector & f);

int dim;
int dimc;
double omega;
double mu = 1.0;
double epsilon = 1.0;

enum prob_type
{
   plane_wave,
   fichera_oven,
   pml_general,
   pml_plane_wave_scatter,
   pml_pointsource
};

static const char *enum_str[] =
{
   "plane_wave",
   "fichera_oven",
   "pml_general",
   "pml_plane_wave_scatter",
   "pml_pointsource"
};

prob_type prob;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   double rnum=1.0;
   double theta = 0.0;
   bool static_cond = false;
   int iprob = 0;
   int sr = 0;
   int pr = 1;
   bool exact_known = false;
   bool with_pml = false;
   bool visualization = true;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&rnum, "-rnum", "--number-of-wavelengths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: plane wave, 1: Fichera 'oven', "
                  " 2: Generic PML problem with point source given as a load "
                  " 3: Scattering of a plane wave, "
                  " 4: Point source given on the boundary");
   args.AddOption(&delta_order, "-do", "--delta-order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&theta, "-theta", "--theta",
                  "Theta parameter for AMR");
   args.AddOption(&sr, "-sref", "--serial-ref",
                  "Number of parallel refinements.");
   args.AddOption(&pr, "-pref", "--parallel-ref",
                  "Number of parallel refinements.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }

   if (iprob > 4) { iprob = 0; }
   prob = (prob_type)iprob;
   omega = 2.*M_PI*rnum;

   if (prob == 0)
   {
      exact_known = true;
   }
   else if (prob == 1)
   {
      mesh_file = "meshes/fichera-waveguide.mesh";
      omega = 5.0;
      rnum = omega/(2.*M_PI);
   }
   else if (prob == 2)
   {
      with_pml = true;
   }
   else
   {
      with_pml = true;
      mesh_file = "meshes/scatter.mesh";
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   MFEM_VERIFY(dim > 1, "Dimension = 1 is not supported in this example");

   dimc = (dim == 3) ? 3 : 1;

   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }
   mesh.EnsureNCMesh(false);

   CartesianPML * pml = nullptr;
   if (with_pml)
   {
      Array2D<double> length(dim, 2); length = 0.25;
      pml = new CartesianPML(&mesh,length);
      pml->SetOmega(omega);
      pml->SetEpsilonAndMu(epsilon,mu);
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // PML element attribute marker
   Array<int> attr;
   Array<int> attrPML;
   if (pml) { pml->SetAttributes(&pmesh, &attr, &attrPML); }

   // Define spaces
   enum TrialSpace
   {
      E_space     = 0,
      H_space     = 1,
      hatE_space  = 2,
      hatH_space  = 3
   };
   enum TestSpace
   {
      F_space = 0,
      G_space = 1
   };
   // L2 space for E
   FiniteElementCollection *E_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh,E_fec,dim);

   // Vector L2 space for H
   FiniteElementCollection *H_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *H_fes = new ParFiniteElementSpace(&pmesh,H_fec, dimc);

   // H^-1/2 (curl) space for Ê
   FiniteElementCollection * hatE_fec = nullptr;
   FiniteElementCollection * hatH_fec = nullptr;
   FiniteElementCollection * F_fec = nullptr;
   int test_order = order+delta_order;
   if (dim == 3)
   {
      hatE_fec = new ND_Trace_FECollection(order,dim);
      hatH_fec = new ND_Trace_FECollection(order,dim);
      F_fec = new ND_FECollection(test_order, dim);
   }
   else
   {
      hatE_fec = new RT_Trace_FECollection(order-1,dim);
      hatH_fec = new H1_Trace_FECollection(order,dim);
      F_fec = new H1_FECollection(test_order, dim);
   }
   ParFiniteElementSpace *hatE_fes = new ParFiniteElementSpace(&pmesh,hatE_fec);
   ParFiniteElementSpace *hatH_fes = new ParFiniteElementSpace(&pmesh,hatH_fec);
   FiniteElementCollection * G_fec = new ND_FECollection(test_order, dim);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;
   trial_fes.Append(E_fes);
   trial_fes.Append(H_fes);
   trial_fes.Append(hatE_fes);
   trial_fes.Append(hatH_fes);
   test_fec.Append(F_fec);
   test_fec.Append(G_fec);

   // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient eps2omeg2(epsilon*epsilon*omega*omega);
   ConstantCoefficient mu2omeg2(mu*mu*omega*omega);
   ConstantCoefficient muomeg(mu*omega);
   ConstantCoefficient negepsomeg(-epsilon*omega);
   ConstantCoefficient epsomeg(epsilon*omega);
   ConstantCoefficient negmuomeg(-mu*omega);
   // for the 2D case
   DenseMatrix rot_mat(2);
   rot_mat(0,0) = 0.; rot_mat(0,1) = 1.;
   rot_mat(1,0) = -1.; rot_mat(1,1) = 0.;
   MatrixConstantCoefficient rot(rot_mat);
   ScalarMatrixProductCoefficient epsrot(epsomeg,rot);
   ScalarMatrixProductCoefficient negepsrot(negepsomeg,rot);

   Coefficient * epsomeg_cf = nullptr;
   Coefficient * negepsomeg_cf = nullptr;
   Coefficient * eps2omeg2_cf = nullptr;
   Coefficient * muomeg_cf = nullptr;
   Coefficient * negmuomeg_cf = nullptr;
   Coefficient * mu2omeg2_cf = nullptr;
   MatrixCoefficient *epsrot_cf = nullptr;
   MatrixCoefficient *negepsrot_cf = nullptr;

   if (pml)
   {
      epsomeg_cf = new RestrictedCoefficient(epsomeg,attr);
      negepsomeg_cf = new RestrictedCoefficient(negepsomeg,attr);
      eps2omeg2_cf = new RestrictedCoefficient(eps2omeg2,attr);
      muomeg_cf = new RestrictedCoefficient(muomeg,attr);
      negmuomeg_cf = new RestrictedCoefficient(negmuomeg,attr);
      mu2omeg2_cf = new RestrictedCoefficient(mu2omeg2,attr);
      epsrot_cf = new MatrixRestrictedCoefficient(epsrot,attr);
      negepsrot_cf = new MatrixRestrictedCoefficient(negepsrot,attr);
   }
   else
   {
      epsomeg_cf = &epsomeg;
      negepsomeg_cf = &negepsomeg;
      eps2omeg2_cf = &eps2omeg2;
      muomeg_cf = &muomeg;
      negmuomeg_cf = &negmuomeg;
      mu2omeg2_cf = &mu2omeg2;
      epsrot_cf = &epsrot;
      negepsrot_cf = &negepsrot;
   }

   // PML coefficients;
   PmlCoefficient detJ_r(detJ_r_function,pml);
   PmlCoefficient detJ_i(detJ_i_function,pml);
   PmlCoefficient abs_detJ_2(abs_detJ_2_function,pml);
   PmlMatrixCoefficient detJ_Jt_J_inv_r(dim,detJ_Jt_J_inv_r_function,pml);
   PmlMatrixCoefficient detJ_Jt_J_inv_i(dim,detJ_Jt_J_inv_i_function,pml);
   PmlMatrixCoefficient abs_detJ_Jt_J_inv_2(dim,abs_detJ_Jt_J_inv_2_function,pml);

   ProductCoefficient negmuomeg_detJ_r(negmuomeg,detJ_r);
   ProductCoefficient negmuomeg_detJ_i(negmuomeg,detJ_i);
   ProductCoefficient muomeg_detJ_r(muomeg,detJ_r);
   ProductCoefficient mu2omeg2_detJ_2(mu2omeg2,abs_detJ_2);
   ScalarMatrixProductCoefficient epsomeg_detJ_Jt_J_inv_i(epsomeg,
                                                          detJ_Jt_J_inv_i);
   ScalarMatrixProductCoefficient epsomeg_detJ_Jt_J_inv_r(epsomeg,
                                                          detJ_Jt_J_inv_r);
   ScalarMatrixProductCoefficient negepsomeg_detJ_Jt_J_inv_r(negepsomeg,
                                                             detJ_Jt_J_inv_r);
   ScalarMatrixProductCoefficient muomeg_detJ_Jt_J_inv_r(muomeg,detJ_Jt_J_inv_r);
   ScalarMatrixProductCoefficient negmuomeg_detJ_Jt_J_inv_i(negmuomeg,
                                                            detJ_Jt_J_inv_i);
   ScalarMatrixProductCoefficient negmuomeg_detJ_Jt_J_inv_r(negmuomeg,
                                                            detJ_Jt_J_inv_r);
   ScalarMatrixProductCoefficient mu2omeg2_detJ_Jt_J_inv_2(mu2omeg2,
                                                           abs_detJ_Jt_J_inv_2);
   ScalarMatrixProductCoefficient eps2omeg2_detJ_Jt_J_inv_2(eps2omeg2,
                                                            abs_detJ_Jt_J_inv_2);

   RestrictedCoefficient negmuomeg_detJ_r_restr(negmuomeg_detJ_r,attrPML);
   RestrictedCoefficient negmuomeg_detJ_i_restr(negmuomeg_detJ_i,attrPML);
   RestrictedCoefficient muomeg_detJ_r_restr(muomeg_detJ_r,attrPML);
   RestrictedCoefficient mu2omeg2_detJ_2_restr(mu2omeg2_detJ_2,attrPML);
   MatrixRestrictedCoefficient epsomeg_detJ_Jt_J_inv_i_restr(
      epsomeg_detJ_Jt_J_inv_i,attrPML);
   MatrixRestrictedCoefficient epsomeg_detJ_Jt_J_inv_r_restr(
      epsomeg_detJ_Jt_J_inv_r,attrPML);
   MatrixRestrictedCoefficient negepsomeg_detJ_Jt_J_inv_r_restr(
      negepsomeg_detJ_Jt_J_inv_r,attrPML);
   MatrixRestrictedCoefficient muomeg_detJ_Jt_J_inv_r_restr(muomeg_detJ_Jt_J_inv_r,
                                                            attrPML);
   MatrixRestrictedCoefficient negmuomeg_detJ_Jt_J_inv_i_restr(
      negmuomeg_detJ_Jt_J_inv_i,attrPML);
   MatrixRestrictedCoefficient negmuomeg_detJ_Jt_J_inv_r_restr(
      negmuomeg_detJ_Jt_J_inv_r,attrPML);
   MatrixRestrictedCoefficient mu2omeg2_detJ_Jt_J_inv_2_restr(
      mu2omeg2_detJ_Jt_J_inv_2,attrPML);
   MatrixRestrictedCoefficient eps2omeg2_detJ_Jt_J_inv_2_restr(
      eps2omeg2_detJ_Jt_J_inv_2,attrPML);

   MatrixProductCoefficient * epsomeg_detJ_Jt_J_inv_i_rot = nullptr;
   MatrixProductCoefficient * epsomeg_detJ_Jt_J_inv_r_rot = nullptr;
   MatrixProductCoefficient * negepsomeg_detJ_Jt_J_inv_r_rot = nullptr;
   MatrixRestrictedCoefficient * epsomeg_detJ_Jt_J_inv_i_rot_restr = nullptr;
   MatrixRestrictedCoefficient * epsomeg_detJ_Jt_J_inv_r_rot_restr = nullptr;
   MatrixRestrictedCoefficient * negepsomeg_detJ_Jt_J_inv_r_rot_restr = nullptr;

   if (pml && dim == 2)
   {
      epsomeg_detJ_Jt_J_inv_i_rot = new MatrixProductCoefficient(
         epsomeg_detJ_Jt_J_inv_i, rot);
      epsomeg_detJ_Jt_J_inv_r_rot = new MatrixProductCoefficient(
         epsomeg_detJ_Jt_J_inv_r, rot);
      negepsomeg_detJ_Jt_J_inv_r_rot = new MatrixProductCoefficient(
         negepsomeg_detJ_Jt_J_inv_r, rot);
      epsomeg_detJ_Jt_J_inv_i_rot_restr = new MatrixRestrictedCoefficient(
         *epsomeg_detJ_Jt_J_inv_i_rot, attrPML);
      epsomeg_detJ_Jt_J_inv_r_rot_restr = new MatrixRestrictedCoefficient(
         *epsomeg_detJ_Jt_J_inv_r_rot, attrPML);
      negepsomeg_detJ_Jt_J_inv_r_rot_restr = new MatrixRestrictedCoefficient(
         *negepsomeg_detJ_Jt_J_inv_r_rot, attrPML);
   }

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices(); // needed for AMR

   // (E,∇ × F)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,TrialSpace::E_space, TestSpace::F_space);
   // -i ω ϵ (E , G) = i (- ω ϵ E, G)
   a->AddTrialIntegrator(nullptr,
                         new TransposeIntegrator(new VectorFEMassIntegrator(*negepsomeg_cf)),
                         TrialSpace::E_space,TestSpace::G_space);
   //  (H,∇ × G)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,TrialSpace::H_space, TestSpace::G_space);
   // < n×Ĥ ,G>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                         TrialSpace::hatH_space, TestSpace::G_space);
   // test integrators
   // (∇×G ,∇× δG)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,
                        TestSpace::G_space,TestSpace::G_space);
   // (G,δG)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                        TestSpace::G_space,TestSpace::G_space);

   if (dim == 3)
   {
      // i ω μ (H, F)
      a->AddTrialIntegrator(nullptr, new TransposeIntegrator(
                               new VectorFEMassIntegrator(*muomeg_cf)),
                            TrialSpace::H_space,TestSpace::F_space);
      // < n×Ê,F>
      a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                            TrialSpace::hatE_space, TestSpace::F_space);

      // test integrators
      // (∇×F,∇×δF)
      a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,
                           TestSpace::F_space, TestSpace::F_space);
      // (F,δF)
      a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                           TestSpace::F_space,TestSpace::F_space);
      // μ^2 ω^2 (F,δF)
      a->AddTestIntegrator(new VectorFEMassIntegrator(*mu2omeg2_cf),nullptr,
                           TestSpace::F_space, TestSpace::F_space);
      // -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
      a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(*negmuomeg_cf),
                           TestSpace::F_space, TestSpace::G_space);
      // -i ω ϵ (∇ × F, δG)
      a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(*negepsomeg_cf),
                           TestSpace::F_space, TestSpace::G_space);
      // i ω μ (∇ × G,δF)
      a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(*muomeg_cf),
                           TestSpace::G_space, TestSpace::F_space);
      // i ω ϵ (G, ∇ × δF )
      a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(*epsomeg_cf),
                           TestSpace::G_space, TestSpace::F_space);
      // ϵ^2 ω^2 (G,δG)
      a->AddTestIntegrator(new VectorFEMassIntegrator(*eps2omeg2_cf),nullptr,
                           TestSpace::G_space, TestSpace::G_space);
   }
   else
   {
      // i ω μ (H, F)
      a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(*muomeg_cf),
                            TrialSpace::H_space, TestSpace::F_space);
      // < n×Ê,F>
      a->AddTrialIntegrator(new TraceIntegrator,nullptr,
                            TrialSpace::hatE_space, TestSpace::F_space);
      // test integrators
      // (∇F,∇δF)
      a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,
                           TestSpace::F_space, TestSpace::F_space);
      // (F,δF)
      a->AddTestIntegrator(new MassIntegrator(one),nullptr,
                           TestSpace::F_space, TestSpace::F_space);
      // μ^2 ω^2 (F,δF)
      a->AddTestIntegrator(new MassIntegrator(*mu2omeg2_cf),nullptr,
                           TestSpace::F_space, TestSpace::F_space);
      // -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
      a->AddTestIntegrator(nullptr,
                           new TransposeIntegrator(new MixedCurlIntegrator(*negmuomeg_cf)),
                           TestSpace::F_space, TestSpace::G_space);
      // -i ω ϵ (∇ × F, δG) = i (- ω ϵ A ∇ F,δG), A = [0 1; -1; 0]
      a->AddTestIntegrator(nullptr,new MixedVectorGradientIntegrator(*negepsrot_cf),
                           TestSpace::F_space, TestSpace::G_space);
      // i ω μ (∇ × G,δF) = i (ω μ ∇ × G, δF )
      a->AddTestIntegrator(nullptr,new MixedCurlIntegrator(*muomeg_cf),
                           TestSpace::G_space, TestSpace::F_space);
      // i ω ϵ (G, ∇ × δF ) =  i (ω ϵ G, A ∇ δF) = i ( G , ω ϵ A ∇ δF)
      a->AddTestIntegrator(nullptr,
                           new TransposeIntegrator(
                              new MixedVectorGradientIntegrator(*epsrot_cf)),
                           TestSpace::G_space, TestSpace::F_space);
      // ϵ^2 ω^2 (G, δG)
      a->AddTestIntegrator(new VectorFEMassIntegrator(*eps2omeg2_cf),nullptr,
                           TestSpace::G_space, TestSpace::G_space);
   }
   if (pml)
   {
      //trial integrators
      // -i ω ϵ (β E , G) = -i ω ϵ ((β_re + i β_im) E, G)
      //                  = (ω ϵ β_im E, G) + i (- ω ϵ β_re E, G)
      a->AddTrialIntegrator(
         new TransposeIntegrator(new VectorFEMassIntegrator(
                                    epsomeg_detJ_Jt_J_inv_i_restr)),
         new TransposeIntegrator(new VectorFEMassIntegrator(
                                    negepsomeg_detJ_Jt_J_inv_r_restr)),
         TrialSpace::E_space,TestSpace::G_space);
      if (dim == 3)
      {
         //trial integrators
         // i ω μ (α^-1 H, F) = i ω μ ( (α^-1_re + i α^-1_im) H, F)
         //                   = (- ω μ α^-1_im, H,F) + i *(ω μ α^-1_re H, F)
         a->AddTrialIntegrator(
            new TransposeIntegrator(new VectorFEMassIntegrator(
                                       negmuomeg_detJ_Jt_J_inv_i_restr)),
            new TransposeIntegrator(new VectorFEMassIntegrator(
                                       muomeg_detJ_Jt_J_inv_r_restr)),
            TrialSpace::H_space, TestSpace::F_space);
         // test integrators
         // μ^2 ω^2 (|α|^-2 F,δF)
         a->AddTestIntegrator(
            new VectorFEMassIntegrator(mu2omeg2_detJ_Jt_J_inv_2_restr),nullptr,
            TestSpace::F_space, TestSpace::F_space);
         // -i ω μ (α^-* F,∇ × δG) = i (F, - ω μ α^-1 ∇ × δ G)
         //                        = i (F, - ω μ (α^-1_re + i α^-1_im) ∇ × δ G)
         //                        = (F, - ω μ α^-1_im ∇ × δ G) + i (F, - ω μ α^-1_re ∇×δG)
         a->AddTestIntegrator(new MixedVectorWeakCurlIntegrator(
                                 negmuomeg_detJ_Jt_J_inv_i_restr),
                              new MixedVectorWeakCurlIntegrator(negmuomeg_detJ_Jt_J_inv_r_restr),
                              TestSpace::F_space,TestSpace::G_space);
         // -i ω ϵ (β ∇ × F, δG) = -i ω ϵ ((β_re + i β_im) ∇ × F, δG)
         //                      = (ω ϵ β_im  ∇ × F, δG) + i (- ω ϵ β_re ∇ × F, δG)
         a->AddTestIntegrator(new MixedVectorCurlIntegrator(
                                 epsomeg_detJ_Jt_J_inv_i_restr),
                              new MixedVectorCurlIntegrator(negepsomeg_detJ_Jt_J_inv_r_restr),
                              TestSpace::F_space,TestSpace::G_space);
         // i ω μ (α^-1 ∇ × G,δF) = i ω μ ((α^-1_re + i α^-1_im) ∇ × G,δF)
         //                       = (- ω μ α^-1_im ∇ × G,δF) + i (ω μ α^-1_re ∇ × G,δF)
         a->AddTestIntegrator(new MixedVectorCurlIntegrator(
                                 negmuomeg_detJ_Jt_J_inv_i_restr),
                              new MixedVectorCurlIntegrator(muomeg_detJ_Jt_J_inv_r_restr),
                              TestSpace::G_space, TestSpace::F_space);
         // i ω ϵ (β^* G, ∇×δF) = i ω ϵ ( (β_re - i β_im) G, ∇×δF)
         //                     = (ω ϵ β_im G, ∇×δF) + i ( ω ϵ β_re G, ∇×δF)
         a->AddTestIntegrator(new MixedVectorWeakCurlIntegrator(
                                 epsomeg_detJ_Jt_J_inv_i_restr),
                              new MixedVectorWeakCurlIntegrator(epsomeg_detJ_Jt_J_inv_r_restr),
                              TestSpace::G_space, TestSpace::F_space);
         // ϵ^2 ω^2 (|β|^2 G,δG)
         a->AddTestIntegrator(new VectorFEMassIntegrator(
                                 eps2omeg2_detJ_Jt_J_inv_2_restr),nullptr,
                              TestSpace::G_space, TestSpace::G_space);
      }
      else
      {
         //trial integrators
         // i ω μ (α^-1 H, F) = i ω μ ( (α^-1_re + i α^-1_im) H, F)
         //                   = (- ω μ α^-1_im, H,F) + i *(ω μ α^-1_re H, F)
         a->AddTrialIntegrator(
            new MixedScalarMassIntegrator(negmuomeg_detJ_i_restr),
            new MixedScalarMassIntegrator(muomeg_detJ_r_restr),
            TrialSpace::H_space, TestSpace::F_space);
         // test integrators
         // μ^2 ω^2 (|α|^-2 F,δF)
         a->AddTestIntegrator(new MassIntegrator(mu2omeg2_detJ_2_restr),nullptr,
                              TestSpace::F_space, TestSpace::F_space);
         // -i ω μ (α^-* F,∇ × δG) = (F, ω μ α^-1 ∇ × δ G)
         //                        =(F, - ω μ α^-1_im ∇ × δ G) + i (F, - ω μ α^-1_re ∇×δG)
         a->AddTestIntegrator(
            new TransposeIntegrator(new MixedCurlIntegrator(negmuomeg_detJ_i_restr)),
            new TransposeIntegrator(new MixedCurlIntegrator(negmuomeg_detJ_r_restr)),
            TestSpace::F_space, TestSpace::G_space);
         // -i ω ϵ (β ∇ × F, δG) = i (- ω ϵ β A ∇ F,δG), A = [0 1; -1; 0]
         //                      = (ω ϵ β_im A ∇ F, δG) + i (- ω ϵ β_re A ∇ F, δG)
         a->AddTestIntegrator(new MixedVectorGradientIntegrator(
                                 *epsomeg_detJ_Jt_J_inv_i_rot_restr),
                              new MixedVectorGradientIntegrator(*negepsomeg_detJ_Jt_J_inv_r_rot_restr),
                              TestSpace::F_space, TestSpace::G_space);
         // i ω μ (α^-1 ∇ × G,δF) = i (ω μ α^-1 ∇ × G, δF )
         //                       = (- ω μ α^-1_im ∇ × G,δF) + i (ω μ α^-1_re ∇ × G,δF)
         a->AddTestIntegrator(new MixedCurlIntegrator(negmuomeg_detJ_i_restr),
                              new MixedCurlIntegrator(muomeg_detJ_r_restr),
                              TestSpace::G_space, TestSpace::F_space);
         // i ω ϵ (β^* G, ∇ × δF ) = i ( G , ω ϵ β A ∇ δF)
         //                        =  ( G , ω ϵ β_im A ∇ δF) + i ( G , ω ϵ β_re A ∇ δF)
         a->AddTestIntegrator(
            new TransposeIntegrator(new MixedVectorGradientIntegrator(
                                       *epsomeg_detJ_Jt_J_inv_i_rot_restr)),
            new TransposeIntegrator(new MixedVectorGradientIntegrator(
                                       *epsomeg_detJ_Jt_J_inv_r_rot_restr)),
            TestSpace::G_space, TestSpace::F_space);
         // ϵ^2 ω^2 (|β|^2 G,δG)
         a->AddTestIntegrator(new VectorFEMassIntegrator(
                                 eps2omeg2_detJ_Jt_J_inv_2_restr),nullptr,
                              TestSpace::G_space, TestSpace::G_space);
      }
   }
   // RHS
   VectorFunctionCoefficient f_rhs_r(dim,rhs_func_r);
   VectorFunctionCoefficient f_rhs_i(dim,rhs_func_i);
   VectorFunctionCoefficient f_source(dim,source_function);
   if (prob == 0)
   {
      a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f_rhs_r),
                               new VectorFEDomainLFIntegrator(f_rhs_i),
                               TestSpace::G_space);
   }
   else if (prob == 2)
   {
      a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f_source),nullptr,
                               TestSpace::G_space);
   }

   VectorFunctionCoefficient hatEex_r(dim,hatE_exact_r);
   VectorFunctionCoefficient hatEex_i(dim,hatE_exact_i);

   socketstream E_out_r;
   socketstream H_out_r;
   if (myid == 0)
   {
      std::cout << "\n  Ref |"
                << "    Dofs    |"
                << "    ω    |" ;
      if (exact_known)
      {
         std::cout  << "  L2 Error  |"
                    << "  Rate  |" ;
      }
      std::cout << "  Residual  |"
                << "  Rate  |"
                << " PCG it |" << endl;
      std::cout << std::string((exact_known) ? 82 : 60,'-')
                << endl;
   }

   double res0 = 0.;
   double err0 = 0.;
   int dof0;

   Array<int> elements_to_refine;

   ParGridFunction E_r, E_i, H_r, H_i;

   ParaViewDataCollection * paraview_dc = nullptr;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection(enum_str[prob], &pmesh);
      paraview_dc->SetPrefixPath("ParaView/Maxwell");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",&E_r);
      paraview_dc->RegisterField("E_i",&E_i);
      paraview_dc->RegisterField("H_r",&H_r);
      paraview_dc->RegisterField("H_i",&H_i);
   }

   if (static_cond) { a->EnableStaticCondensation(); }
   for (int it = 0; it<=pr; it++)
   {
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         hatE_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
         if (pml)
         {
            ess_bdr = 0;
            ess_bdr[1] = 1;
         }
      }

      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += E_fes->GetTrueVSize() + H_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = E_fes->GetVSize();
      offsets[2] = H_fes->GetVSize();
      offsets[3] = hatE_fes->GetVSize();
      offsets[4] = hatH_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;

      if (prob != 2)
      {
         ParGridFunction hatE_gf_r(hatE_fes, x, offsets[2]);
         ParGridFunction hatE_gf_i(hatE_fes, x, offsets.Last() + offsets[2]);
         if (dim == 3)
         {
            hatE_gf_r.ProjectBdrCoefficientTangent(hatEex_r, ess_bdr);
            hatE_gf_i.ProjectBdrCoefficientTangent(hatEex_i, ess_bdr);
         }
         else
         {
            hatE_gf_r.ProjectBdrCoefficientNormal(hatEex_r, ess_bdr);
            hatE_gf_i.ProjectBdrCoefficientNormal(hatEex_i, ess_bdr);
         }
      }

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

      ComplexOperator * Ahc = Ah.As<ComplexOperator>();

      BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
      BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

      int num_blocks = BlockA_r->NumRowBlocks();
      Array<int> tdof_offsets(2*num_blocks+1);

      tdof_offsets[0] = 0;
      int skip = (static_cond) ? 0 : 2;
      int k = (static_cond) ? 2 : 0;
      for (int i=0; i<num_blocks; i++)
      {
         tdof_offsets[i+1] = trial_fes[i+k]->GetTrueVSize();
         tdof_offsets[num_blocks+i+1] = trial_fes[i+k]->GetTrueVSize();
      }
      tdof_offsets.PartialSum();

      BlockOperator blockA(tdof_offsets);
      for (int i = 0; i<num_blocks; i++)
      {
         for (int j = 0; j<num_blocks; j++)
         {
            blockA.SetBlock(i,j,&BlockA_r->GetBlock(i,j));
            blockA.SetBlock(i,j+num_blocks,&BlockA_i->GetBlock(i,j), -1.0);
            blockA.SetBlock(i+num_blocks,j+num_blocks,&BlockA_r->GetBlock(i,j));
            blockA.SetBlock(i+num_blocks,j,&BlockA_i->GetBlock(i,j));
         }
      }

      X = 0.;
      BlockDiagonalPreconditioner M(tdof_offsets);

      if (!static_cond)
      {
         HypreBoomerAMG * solver_E = new HypreBoomerAMG((HypreParMatrix &)
                                                        BlockA_r->GetBlock(0,0));
         solver_E->SetPrintLevel(0);
         solver_E->SetSystemsOptions(dim);
         HypreBoomerAMG * solver_H = new HypreBoomerAMG((HypreParMatrix &)
                                                        BlockA_r->GetBlock(1,1));
         solver_H->SetPrintLevel(0);
         solver_H->SetSystemsOptions(dim);
         M.SetDiagonalBlock(0,solver_E);
         M.SetDiagonalBlock(1,solver_H);
         M.SetDiagonalBlock(num_blocks,solver_E);
         M.SetDiagonalBlock(num_blocks+1,solver_H);
      }

      HypreSolver * solver_hatH = nullptr;
      HypreAMS * solver_hatE = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(skip,
                                                                                 skip),
                                            hatE_fes);
      solver_hatE->SetPrintLevel(0);
      if (dim == 2)
      {
         solver_hatH = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(skip+1,
                                                                               skip+1));
         dynamic_cast<HypreBoomerAMG*>(solver_hatH)->SetPrintLevel(0);
      }
      else
      {
         solver_hatH = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(skip+1,skip+1),
                                    hatH_fes);
         dynamic_cast<HypreAMS*>(solver_hatH)->SetPrintLevel(0);
      }

      M.SetDiagonalBlock(skip,solver_hatE);
      M.SetDiagonalBlock(skip+1,solver_hatH);
      M.SetDiagonalBlock(skip+num_blocks,solver_hatE);
      M.SetDiagonalBlock(skip+num_blocks+1,solver_hatH);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-6);
      cg.SetMaxIter(10000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(M);
      cg.SetOperator(blockA);
      cg.Mult(B, X);

      for (int i = 0; i<num_blocks; i++)
      {
         delete &M.GetDiagonalBlock(i);
      }

      int num_iter = cg.GetNumIterations();

      a->RecoverFEMSolution(X,x);

      Vector & residuals = a->ComputeResidual(x);

      double residual = residuals.Norml2();
      double maxresidual = residuals.Max();
      double globalresidual = residual * residual;
      MPI_Allreduce(MPI_IN_PLACE,&maxresidual,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&globalresidual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      globalresidual = sqrt(globalresidual);

      E_r.MakeRef(E_fes,x, 0);
      E_i.MakeRef(E_fes,x, offsets.Last());

      H_r.MakeRef(H_fes,x, offsets[1]);
      H_i.MakeRef(H_fes,x, offsets.Last()+offsets[1]);

      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GlobalTrueVSize();
      }

      double L2Error = 0.0;
      double rate_err = 0.0;

      if (exact_known)
      {
         VectorFunctionCoefficient E_ex_r(dim,E_exact_r);
         VectorFunctionCoefficient E_ex_i(dim,E_exact_i);
         VectorFunctionCoefficient H_ex_r(dim,H_exact_r);
         VectorFunctionCoefficient H_ex_i(dim,H_exact_i);
         double E_err_r = E_r.ComputeL2Error(E_ex_r);
         double E_err_i = E_i.ComputeL2Error(E_ex_i);
         double H_err_r = H_r.ComputeL2Error(H_ex_r);
         double H_err_i = H_i.ComputeL2Error(H_ex_i);
         L2Error = sqrt(  E_err_r*E_err_r + E_err_i*E_err_i
                          + H_err_r*H_err_r + H_err_i*H_err_i );
         rate_err = (it) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;
         err0 = L2Error;
      }

      double rate_res = (it) ? dim*log(res0/globalresidual)/log((
                                                                   double)dof0/dofs) : 0.0;

      res0 = globalresidual;
      dof0 = dofs;

      if (myid == 0)
      {
         std::ios oldState(nullptr);
         oldState.copyfmt(std::cout);
         std::cout << std::right << std::setw(5) << it << " | "
                   << std::setw(10) <<  dof0 << " | "
                   << std::setprecision(1) << std::fixed
                   << std::setw(4) <<  2.0*rnum << " π  | "
                   << std::setprecision(3);
         if (exact_known)
         {
            std::cout << std::setw(10) << std::scientific <<  err0 << " | "
                      << std::setprecision(2)
                      << std::setw(6) << std::fixed << rate_err << " | " ;
         }
         std::cout << std::setprecision(3)
                   << std::setw(10) << std::scientific <<  res0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_res << " | "
                   << std::setw(6) << std::fixed << num_iter << " | "
                   << std::endl;
         std::cout.copyfmt(oldState);
      }

      if (visualization)
      {
         const char * keys = (it == 0 && dim == 2) ? "jRcml\n" : nullptr;
         char vishost[] = "localhost";
         int  visport   = 19916;
         VisualizeField(E_out_r,vishost, visport, E_r,
                        "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
         VisualizeField(H_out_r,vishost, visport, H_r,
                        "Numerical Magnetic field (real part)", 501, 0, 500, 500, keys);
      }

      if (paraview)
      {
         paraview_dc->SetCycle(it);
         paraview_dc->SetTime((double)it);
         paraview_dc->Save();
      }

      if (it == pr)
      {
         break;
      }

      if (theta > 0.0)
      {
         elements_to_refine.SetSize(0);
         for (int iel = 0; iel<pmesh.GetNE(); iel++)
         {
            if (residuals[iel] > theta * maxresidual)
            {
               elements_to_refine.Append(iel);
            }
         }
         pmesh.GeneralRefinement(elements_to_refine,1,1);
      }
      else
      {
         pmesh.UniformRefinement();
      }
      if (pml) { pml->SetAttributes(&pmesh); }
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   if (pml && dim == 2)
   {
      delete epsomeg_detJ_Jt_J_inv_i_rot;
      delete epsomeg_detJ_Jt_J_inv_r_rot;
      delete negepsomeg_detJ_Jt_J_inv_r_rot;
      delete epsomeg_detJ_Jt_J_inv_i_rot_restr;
      delete epsomeg_detJ_Jt_J_inv_r_rot_restr;
      delete negepsomeg_detJ_Jt_J_inv_r_rot_restr;
   }

   if (paraview)
   {
      delete paraview_dc;
   }

   delete a;
   delete F_fec;
   delete G_fec;
   delete hatH_fes;
   delete hatH_fec;
   delete hatE_fes;
   delete hatE_fec;
   delete H_fec;
   delete E_fec;
   delete H_fes;
   delete E_fes;

   return 0;
}

void E_exact_r(const Vector &x, Vector & E_r)
{
   std::vector<std::complex<double>> E;
   maxwell_solution(x,E);
   E_r.SetSize(E.size());
   for (unsigned i = 0; i < E.size(); i++)
   {
      E_r[i]= E[i].real();
   }
}

void E_exact_i(const Vector &x, Vector & E_i)
{
   std::vector<std::complex<double>> E;
   maxwell_solution(x, E);
   E_i.SetSize(E.size());
   for (unsigned i = 0; i < E.size(); i++)
   {
      E_i[i]= E[i].imag();
   }
}

void curlE_exact_r(const Vector &x, Vector &curlE_r)
{
   std::vector<std::complex<double>> curlE;
   maxwell_solution_curl(x, curlE);
   curlE_r.SetSize(curlE.size());
   for (unsigned i = 0; i < curlE.size(); i++)
   {
      curlE_r[i]= curlE[i].real();
   }
}

void curlE_exact_i(const Vector &x, Vector &curlE_i)
{
   std::vector<std::complex<double>> curlE;
   maxwell_solution_curl(x, curlE);
   curlE_i.SetSize(curlE.size());
   for (unsigned i = 0; i < curlE.size(); i++)
   {
      curlE_i[i]= curlE[i].imag();
   }
}

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r)
{
   std::vector<std::complex<double>> curlcurlE;
   maxwell_solution_curlcurl(x, curlcurlE);
   curlcurlE_r.SetSize(curlcurlE.size());
   for (unsigned i = 0; i < curlcurlE.size(); i++)
   {
      curlcurlE_r[i]= curlcurlE[i].real();
   }
}

void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i)
{
   std::vector<std::complex<double>> curlcurlE;
   maxwell_solution_curlcurl(x, curlcurlE);
   curlcurlE_i.SetSize(curlcurlE.size());
   for (unsigned i = 0; i < curlcurlE.size(); i++)
   {
      curlcurlE_i[i]= curlcurlE[i].imag();
   }
}


void H_exact_r(const Vector &x, Vector & H_r)
{
   // H = i ∇ × E / ω μ
   // H_r = - ∇ × E_i / ω μ
   Vector curlE_i;
   curlE_exact_i(x,curlE_i);
   H_r.SetSize(dimc);
   for (int i = 0; i<dimc; i++)
   {
      H_r(i) = - curlE_i(i) / (omega * mu);
   }
}

void H_exact_i(const Vector &x, Vector & H_i)
{
   // H = i ∇ × E / ω μ
   // H_i =  ∇ × E_r / ω μ
   Vector curlE_r;
   curlE_exact_r(x,curlE_r);
   H_i.SetSize(dimc);
   for (int i = 0; i<dimc; i++)
   {
      H_i(i) = curlE_r(i) / (omega * mu);
   }
}

void curlH_exact_r(const Vector &x,Vector &curlH_r)
{
   // ∇ × H_r = - ∇ × ∇ × E_i / ω μ
   Vector curlcurlE_i;
   curlcurlE_exact_i(x,curlcurlE_i);
   curlH_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      curlH_r(i) = -curlcurlE_i(i) / (omega * mu);
   }
}

void curlH_exact_i(const Vector &x,Vector &curlH_i)
{
   // ∇ × H_i = ∇ × ∇ × E_r / ω μ
   Vector curlcurlE_r;
   curlcurlE_exact_r(x,curlcurlE_r);
   curlH_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      curlH_i(i) = curlcurlE_r(i) / (omega * mu);
   }
}

void hatE_exact_r(const Vector & x, Vector & hatE_r)
{
   if (dim == 3)
   {
      E_exact_r(x,hatE_r);
   }
   else
   {
      Vector E_r;
      E_exact_r(x,E_r);
      hatE_r.SetSize(hatE_r.Size());
      // rotate E_hat
      hatE_r[0] = E_r[1];
      hatE_r[1] = -E_r[0];
   }
}

void hatE_exact_i(const Vector & x, Vector & hatE_i)
{
   if (dim == 3)
   {
      E_exact_i(x,hatE_i);
   }
   else
   {
      Vector E_i;
      E_exact_i(x,E_i);
      hatE_i.SetSize(hatE_i.Size());
      // rotate E_hat
      hatE_i[0] = E_i[1];
      hatE_i[1] = -E_i[0];
   }
}

void hatH_exact_r(const Vector & x, Vector & hatH_r)
{
   H_exact_r(x,hatH_r);
}

void hatH_exact_i(const Vector & x, Vector & hatH_i)
{
   H_exact_i(x,hatH_i);
}

double hatH_exact_scalar_r(const Vector & x)
{
   Vector hatH_r;
   H_exact_r(x,hatH_r);
   return hatH_r[0];
}

double hatH_exact_scalar_i(const Vector & x)
{
   Vector hatH_i;
   H_exact_i(x,hatH_i);
   return hatH_i[0];
}

// J = -i ω ϵ E + ∇ × H
// J_r + iJ_i = -i ω ϵ (E_r + i E_i) + ∇ × (H_r + i H_i)
void  rhs_func_r(const Vector &x, Vector & J_r)
{
   // J_r = ω ϵ E_i + ∇ × H_r
   Vector E_i, curlH_r;
   E_exact_i(x,E_i);
   curlH_exact_r(x,curlH_r);
   J_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      J_r(i) = omega * epsilon * E_i(i) + curlH_r(i);
   }
}

void  rhs_func_i(const Vector &x, Vector & J_i)
{
   // J_i = - ω ϵ E_r + ∇ × H_i
   Vector E_r, curlH_i;
   E_exact_r(x,E_r);
   curlH_exact_i(x,curlH_i);
   J_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      J_i(i) = -omega * epsilon * E_r(i) + curlH_i(i);
   }
}

void maxwell_solution(const Vector & X, std::vector<complex<double>> &E)
{
   complex<double> zi = complex<double>(0., 1.);
   E.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = 0.0;
   }
   switch (prob)
   {
      case plane_wave:
      {
         E[0] = exp(zi * omega * (X.Sum()));
      }
      break;
      case pml_plane_wave_scatter:
      {
         E[1] = exp(zi * omega * (X(0)));
      }
      break;
      case fichera_oven:
      {
         if (abs(X(2) - 3.0) < 1e-10)
         {
            E[0] = sin(M_PI*X(1));
         }
      }
      break;
      case pml_pointsource:
      {
         Vector shift(dim);
         double k = omega * sqrt(epsilon * mu);
         shift = -0.5;

         if (dim == 2)
         {
            double x0 = X(0) + shift(0);
            double x1 = X(1) + shift(1);
            double r = sqrt(x0 * x0 + x1 * x1);
            double beta = k * r;

            // Bessel functions
            complex<double> Ho, Ho_r, Ho_rr;
            Ho = jn(0, beta) + zi * yn(0, beta);
            Ho_r = -k * (jn(1, beta) + zi * yn(1, beta));
            Ho_rr = -k * k * (1.0 / beta *
                              (jn(1, beta) + zi * yn(1, beta)) -
                              (jn(2, beta) + zi * yn(2, beta)));

            // First derivatives
            double r_x = x0 / r;
            double r_y = x1 / r;
            double r_xy = -(r_x / r) * r_y;
            double r_xx = (1.0 / r) * (1.0 - r_x * r_x);

            complex<double> val, val_xx, val_xy;
            val = 0.25 * zi * Ho;
            val_xx = 0.25 * zi * (r_xx * Ho_r + r_x * r_x * Ho_rr);
            val_xy = 0.25 * zi * (r_xy * Ho_r + r_x * r_y * Ho_rr);
            E[0] = zi / k * (k * k * val + val_xx);
            E[1] = zi / k * val_xy;
         }
         else
         {
            double x0 = X(0) + shift(0);
            double x1 = X(1) + shift(1);
            double x2 = X(2) + shift(2);
            double r = sqrt(x0 * x0 + x1 * x1 + x2 * x2);

            double r_x = x0 / r;
            double r_y = x1 / r;
            double r_z = x2 / r;
            double r_xx = (1.0 / r) * (1.0 - r_x * r_x);
            double r_yx = -(r_y / r) * r_x;
            double r_zx = -(r_z / r) * r_x;

            complex<double> val, val_r, val_rr;
            val = exp(zi * k * r) / r;
            val_r = val / r * (zi * k * r - 1.0);
            val_rr = val / (r * r) * (-k * k * r * r
                                      - 2.0 * zi * k * r + 2.0);

            complex<double> val_xx, val_yx, val_zx;
            val_xx = val_rr * r_x * r_x + val_r * r_xx;
            val_yx = val_rr * r_x * r_y + val_r * r_yx;
            val_zx = val_rr * r_x * r_z + val_r * r_zx;
            complex<double> alpha = zi * k / 4.0 / M_PI / k / k;
            E[0] = alpha * (k * k * val + val_xx);
            E[1] = alpha * val_yx;
            E[2] = alpha * val_zx;
         }
      }
      break;

      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }
}

void maxwell_solution_curl(const Vector & X,
                           std::vector<complex<double>> &curlE)
{
   complex<double> zi = complex<double>(0., 1.);
   curlE.resize(dimc);
   for (int i = 0; i < dimc; ++i)
   {
      curlE[i] = 0.0;
   }
   switch (prob)
   {
      case plane_wave:
      {
         std::complex<double> pw = exp(zi * omega * (X.Sum()));
         if (dim == 3)
         {
            curlE[0] = 0.0;
            curlE[1] = zi * omega * pw;
            curlE[2] = -zi * omega * pw;
         }
         else
         {
            curlE[0] = -zi * omega * pw;
         }
      }
      break;
      case pml_plane_wave_scatter:
      {
         std::complex<double> pw = exp(zi * omega * (X(0)));
         curlE[0] = zi * omega * pw;
      }
      break;
      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }
}

void maxwell_solution_curlcurl(const Vector & X,
                               std::vector<complex<double>> &curlcurlE)
{
   complex<double> zi = complex<double>(0., 1.);
   curlcurlE.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      curlcurlE[i] = 0.0;;
   }
   switch (prob)
   {
      case plane_wave:
      {
         std::complex<double> pw = exp(zi * omega * (X.Sum()));
         if (dim == 3)
         {
            curlcurlE[0] = 2.0 * omega * omega * pw;
            curlcurlE[1] = - omega * omega * pw;
            curlcurlE[2] = - omega * omega * pw;
         }
         else
         {
            curlcurlE[0] = omega * omega * pw;
            curlcurlE[1] = -omega * omega * pw;
         }
      }
      break;
      case pml_plane_wave_scatter:
      {
         std::complex<double> pw = exp(zi * omega * (X(0)));
         curlcurlE[1] = omega * omega * pw;
      }
      break;
      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }
}

void source_function(const Vector &x, Vector &f)
{
   Vector center(dim);
   center = 0.5;
   double r = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      r += pow(x[i] - center[i], 2.);
   }
   double n = 5.0 * omega * sqrt(epsilon * mu) / M_PI;
   double coeff = pow(n, 2) / M_PI;
   double alpha = -pow(n, 2) * r;
   f = 0.0;
   f[0] = -omega * coeff * exp(alpha)/omega;
}
