//                   MFEM Ultraweak DPG acoustics example
//
// Compile with: make pacoustics
//
// sample runs

// mpirun -np 4 pacoustics -o 3 -m ../../data/star.mesh -sref 1 -pref 2 -rnum 1.9 -sc -prob 0
// mpirun -np 4 pacoustics -o 3 -m ../../data/inline-quad.mesh -sref 1 -pref 2  -rnum 5.2 -sc -prob 1
// mpirun -np 4 pacoustics -o 4 -m ../../data/inline-tri.mesh -sref 1 -pref 2  -rnum 7.1 -sc -prob 1
// mpirun -np 4 pacoustics -o 2 -m ../../data/inline-hex.mesh -sref 0 -pref 1 -rnum 1.9 -sc -prob 0
// mpirun -np 4 pacoustics -o 3 -m ../../data/inline-quad.mesh -sref 2 -pref 1 -rnum 7.1 -sc -prob 2
// mpirun -np 4 pacoustics -o 2 -m ../../data/inline-hex.mesh -sref 0 -pref 1  -rnum 4.1 -sc -prob 2
// mpirun -np 4 pacoustics -o 3 -m meshes/scatter.mesh -sref 1 -pref 1  -rnum 7.1 -sc -prob 3
// mpirun -np 4 pacoustics -o 4 -m meshes/scatter.mesh -sref 1 -pref 1  -rnum 10.1 -sc -prob 4
// mpirun -np 4 pacoustics -o 4 -m meshes/scatter.mesh -sref 1 -pref 1  -rnum 12.1 -sc -prob 5

// AMR runs
// mpirun -np 4 pacoustics -o 3 -m meshes/scatter.mesh -sref 0 -pref 10 -theta 0.75 -rnum 10.1 -sc -prob 3
// mpirun -np 4 pacoustics -o 3 -m meshes/scatter.mesh -sref 0 -pref 12 -theta 0.75 -rnum 20.1 -sc -prob 3

// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Helmholtz problem

//     - Δ p - ω² p = f̃ ,   in Ω
//                p = p₀, on ∂Ω

// It solves the following kinds of problems
// 1) Known exact solutions with error convergence rates
//    a) f̃ = 0 and p₀ is a plane wave
//    b) A manufactured solution problem where p_exact is a gaussian beam
// 2) PML problems
//    a) Gausian beam scattering from a square
//    b) Plane wave scattering from a square
//    c) Point Source

// The DPG UW deals with the First Order System
//  ∇ p + i ω u = 0, in Ω
//  ∇⋅u + i ω p = f, in Ω        (1)
//            p = p₀, in ∂Ω
// where f:=f̃/(i ω)

// The ultraweak-DPG formulation is obtained by integration by parts of both
// equations and the introduction of trace unknowns on the mesh skeleton

// p ∈ L²(Ω), u ∈ (L²(Ω))ᵈⁱᵐ
// p̂ ∈ H^1/2(Ω), û ∈ H^-1/2(Ω)
// -(p,∇⋅v) + i ω (u,v) + <p̂,v⋅n> = 0,      ∀ v ∈ H(div,Ω)
// -(u,∇ q) + i ω (p,q) + <û,q >  = (f,q)   ∀ q ∈ H^1(Ω)
//                             p̂  = p₀      on ∂Ω

// Note:
// p̂ := p, û := u on the mesh skeleton

// -------------------------------------------------------------
// |   |     p     |     u     |    p̂      |    û    |  RHS    |
// -------------------------------------------------------------
// | v | -(p, ∇⋅v) | i ω (u,v) | < p̂, v⋅n> |         |         |
// |   |           |           |           |         |         |
// | q | i ω (p,q) |-(u , ∇ q) |           | < û,q > |  (f,q)  |

// where (q,v) ∈  H¹(Ω) × H(div,Ω)

// Here we use the "Adjoint Graph" norm on the test space i.e.,
// ||(q,v)||²ᵥ  = ||A^*(q,v)||² + ||(q,v)||² where A is the
// acoustics operator defined by (1)

// The PML formulation is

//    - ∇⋅(|J| J⁻¹ J⁻ᵀ ∇ p) - ω²  |J| p = f

// where J is the Jacobian of the stretching map and |J| its determinant.

// The first order system reads

//  ∇ p + i ω α u = 0, in Ω
//  ∇⋅u + i ω β p = f, in Ω         (2)
//              p = p₀, in ∂Ω
// where f:=f̃/(i ω), α:= Jᵀ J / |J|, β:= |J|

// and the ultraweak DPG formulation
//
// p ∈ L²(Ω), u ∈ (L²(Ω))ᵈⁱᵐ
// p̂ ∈ H^1/2(Ω), û ∈ H^-1/2(Ω)
// -(p,  ∇⋅v) + i ω (α u , v) + < p̂, v⋅n> = 0,      ∀ v ∈ H(div,Ω)
// -(u , ∇ q) + i ω (β p , q) + < û, q >  = (f,q)   ∀ q ∈ H¹(Ω)
//                                      p̂ = p₀      on ∂Ω

// Note:
// p̂ := p on Γₕ  (skeleton)
// û := u on Γₕ

// ----------------------------------------------------------------
// |   |     p       |     u       |    p̂      |    û    |  RHS    |
// ----------------------------------------------------------------
// | v | -(p, ∇⋅v)   | i ω (α u,v) | < p̂, v⋅n> |         |         |
// |   |             |             |           |         |         |
// | q | i ω (β p,q) |-(u , ∇ q)   |           | < û,q > |  (f,q)  |

// where (q,v) ∈  H¹(Ω) × H(div,Ω)

// Finally the test norm is defined by the adjoint operator of (2) i.e.,

// ||(q,v)||²ᵥ   = ||A^*(q,v)||² + ||(q,v)||²

// where A is the operator defined by (2)

// For more information see https://doi.org/10.1016/j.camwa.2017.06.044

#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "util/pml.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

complex<double> acoustics_solution(const Vector & X);
void acoustics_solution_grad(const Vector & X,vector<complex<double>> &dp);
complex<double> acoustics_solution_laplacian(const Vector & X);

double p_exact_r(const Vector &x);
double p_exact_i(const Vector &x);
void u_exact_r(const Vector &x, Vector & u);
void u_exact_i(const Vector &x, Vector & u);
double rhs_func_r(const Vector &x);
double rhs_func_i(const Vector &x);
void gradp_exact_r(const Vector &x, Vector &gradu);
void gradp_exact_i(const Vector &x, Vector &gradu);
double divu_exact_r(const Vector &x);
double divu_exact_i(const Vector &x);
double d2_exact_r(const Vector &x);
double d2_exact_i(const Vector &x);
double hatp_exact_r(const Vector & X);
double hatp_exact_i(const Vector & X);
void hatu_exact_r(const Vector & X, Vector & hatu);
void hatu_exact_i(const Vector & X, Vector & hatu);
double source_function(const Vector &x);

int dim;
double omega;

enum prob_type
{
   plane_wave,
   gaussian_beam,
   pml_general,
   pml_beam_scatter,
   pml_plane_wave_scatter,
   pml_pointsource
};

static const char *enum_str[] =
{
   "plane_wave",
   "gaussian_beam",
   "pml_general",
   "pml_beam_scatter",
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
   bool visualization = true;
   double rnum=1.0;
   double theta = 0.0;
   bool static_cond = false;
   int iprob = 0;
   int sr = 0;
   int pr = 0;
   bool exact_known = false;
   bool with_pml = false;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&rnum, "-rnum", "--number-of-wavelengths",
                  "Number of wavelengths");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: plane wave, 1: Gaussian beam, 2: Generic PML,"
                  " 3: Scattering of a Gaussian beam"
                  " 4: Scattering of a plane wave, 5: Point source");
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

   if (iprob > 5) { iprob = 0; }
   prob = (prob_type)iprob;
   omega = 2.*M_PI*rnum;

   if (prob > 1)
   {
      with_pml = true;
      if (prob > 2) { mesh_file = "meshes/scatter.mesh"; }
   }
   else
   {
      exact_known = true;
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh mesh(mesh_file, 1, 1);

   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }
   dim = mesh.Dimension();
   MFEM_VERIFY(dim > 1, "Dimension = 1 is not supported in this example");

   CartesianPML * pml = nullptr;
   if (with_pml)
   {
      Array2D<double> length(dim, 2); length = 0.125;
      pml = new CartesianPML(&mesh,length);
      pml->SetOmega(omega);
   }

   mesh.EnsureNCMesh(true);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   Array<int> attr;
   Array<int> attrPML;
   // PML element attribute marker
   if (pml) { pml->SetAttributes(&pmesh, &attr, &attrPML); }

   // Define spaces
   enum TrialSpace
   {
      p_space     = 0,
      u_space     = 1,
      hatp_space  = 2,
      hatu_space  = 3
   };
   enum TestSpace
   {
      q_space = 0,
      v_space = 1
   };

   // L2 space for p
   FiniteElementCollection *p_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *p_fes = new ParFiniteElementSpace(&pmesh,p_fec);

   // Vector L2 space for u
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *u_fes = new ParFiniteElementSpace(&pmesh,u_fec, dim);

   // H^1/2 space for p̂
   FiniteElementCollection * hatp_fec = new H1_Trace_FECollection(order,dim);
   ParFiniteElementSpace *hatp_fes = new ParFiniteElementSpace(&pmesh,hatp_fec);

   // H^-1/2 space for û
   FiniteElementCollection * hatu_fec = new RT_Trace_FECollection(order-1,dim);
   ParFiniteElementSpace *hatu_fes = new ParFiniteElementSpace(&pmesh,hatu_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * q_fec = new H1_FECollection(test_order, dim);
   FiniteElementCollection * v_fec = new RT_FECollection(test_order-1, dim);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;
   trial_fes.Append(p_fes);
   trial_fes.Append(u_fes);
   trial_fes.Append(hatp_fes);
   trial_fes.Append(hatu_fes);
   test_fec.Append(q_fec);
   test_fec.Append(v_fec);

   // Bilinear form Coefficients
   Coefficient * omeg_cf = nullptr;
   Coefficient * negomeg_cf = nullptr;
   Coefficient * omeg2_cf = nullptr;

   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient omeg2(omega*omega);
   ConstantCoefficient negomeg(-omega);

   if (pml)
   {
      omeg_cf = new RestrictedCoefficient(omeg,attr);
      negomeg_cf = new RestrictedCoefficient(negomeg,attr);
      omeg2_cf = new RestrictedCoefficient(omeg2,attr);
   }
   else
   {
      omeg_cf = &omeg;
      negomeg_cf = &negomeg;
      omeg2_cf = &omeg2;
   }

   // PML coefficients
   PmlCoefficient detJ_r(detJ_r_function,pml);
   PmlCoefficient detJ_i(detJ_i_function,pml);
   PmlCoefficient abs_detJ_2(abs_detJ_2_function,pml);
   ProductCoefficient omeg_detJ_r(omeg,detJ_r);
   ProductCoefficient omeg_detJ_i(omeg,detJ_i);
   ProductCoefficient negomeg_detJ_r(negomeg,detJ_r);
   ProductCoefficient negomeg_detJ_i(negomeg,detJ_i);
   ProductCoefficient omeg2_abs_detJ_2(omeg2,abs_detJ_2);
   RestrictedCoefficient omeg_detJ_r_restr(omeg_detJ_r,attrPML);
   RestrictedCoefficient omeg_detJ_i_restr(omeg_detJ_i,attrPML);
   RestrictedCoefficient negomeg_detJ_r_restr(negomeg_detJ_r,attrPML);
   RestrictedCoefficient negomeg_detJ_i_restr(negomeg_detJ_i,attrPML);
   RestrictedCoefficient omeg2_abs_detJ_2_restr(omeg2_abs_detJ_2,attrPML);
   PmlMatrixCoefficient Jt_J_detJinv_r(dim, Jt_J_detJinv_r_function,pml);
   PmlMatrixCoefficient Jt_J_detJinv_i(dim, Jt_J_detJinv_i_function,pml);
   PmlMatrixCoefficient abs_Jt_J_detJinv_2(dim, abs_Jt_J_detJinv_2_function,pml);
   ScalarMatrixProductCoefficient omeg_Jt_J_detJinv_r(omeg,Jt_J_detJinv_r);
   ScalarMatrixProductCoefficient omeg_Jt_J_detJinv_i(omeg,Jt_J_detJinv_i);
   ScalarMatrixProductCoefficient negomeg_Jt_J_detJinv_r(negomeg,Jt_J_detJinv_r);
   ScalarMatrixProductCoefficient negomeg_Jt_J_detJinv_i(negomeg,Jt_J_detJinv_i);
   ScalarMatrixProductCoefficient omeg2_abs_Jt_J_detJinv_2(omeg2,
                                                           abs_Jt_J_detJinv_2);
   MatrixRestrictedCoefficient omeg_Jt_J_detJinv_r_restr(omeg_Jt_J_detJinv_r,
                                                         attrPML);
   MatrixRestrictedCoefficient omeg_Jt_J_detJinv_i_restr(omeg_Jt_J_detJinv_i,
                                                         attrPML);
   MatrixRestrictedCoefficient negomeg_Jt_J_detJinv_r_restr(negomeg_Jt_J_detJinv_r,
                                                            attrPML);
   MatrixRestrictedCoefficient negomeg_Jt_J_detJinv_i_restr(negomeg_Jt_J_detJinv_i,
                                                            attrPML);
   MatrixRestrictedCoefficient omeg2_abs_Jt_J_detJinv_2_restr(
      omeg2_abs_Jt_J_detJinv_2,attrPML);

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices(); // needed for AMR

   // Trial itegrators
   // Integrators not in PML
   // i ω (p,q)
   a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(*omeg_cf),
                         TrialSpace::p_space,TestSpace::q_space);
   // -(u , ∇ q)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(negone)),
                         nullptr,TrialSpace::u_space,TestSpace::q_space);
   // -(p, ∇⋅v)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),nullptr,
                         TrialSpace::p_space,TestSpace::v_space);
   //  i ω (u,v)
   a->AddTrialIntegrator(nullptr,
                         new TransposeIntegrator(new VectorFEMassIntegrator(*omeg_cf)),
                         TrialSpace::u_space,TestSpace::v_space);
   // < p̂, v⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,nullptr,
                         TrialSpace::hatp_space,TestSpace::v_space);
   // < û,q >
   a->AddTrialIntegrator(new TraceIntegrator,nullptr,
                         TrialSpace::hatu_space,TestSpace::q_space);

   // test integrators
   // (∇q,∇δq)
   a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,
                        TestSpace::q_space, TestSpace::q_space);
   // (q,δq)
   a->AddTestIntegrator(new MassIntegrator(one),nullptr,
                        TestSpace::q_space, TestSpace::q_space);
   // (∇⋅v,∇⋅δv)
   a->AddTestIntegrator(new DivDivIntegrator(one),nullptr,
                        TestSpace::v_space, TestSpace::v_space);
   // (v,δv)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                        TestSpace::v_space, TestSpace::v_space);
   // -i ω (∇q,δv)
   a->AddTestIntegrator(nullptr,new MixedVectorGradientIntegrator(*negomeg_cf),
                        TestSpace::q_space, TestSpace::v_space);
   // i ω (v,∇ δq)
   a->AddTestIntegrator(nullptr,
                        new MixedVectorWeakDivergenceIntegrator(*negomeg_cf),
                        TestSpace::v_space, TestSpace::q_space);
   // ω^2 (v,δv)
   a->AddTestIntegrator(new VectorFEMassIntegrator(*omeg2_cf),nullptr,
                        TestSpace::v_space, TestSpace::v_space);
   // - i ω (∇⋅v,δq)
   a->AddTestIntegrator(nullptr,new VectorFEDivergenceIntegrator(*negomeg_cf),
                        TestSpace::v_space, TestSpace::q_space);
   // i ω (q,∇⋅v)
   a->AddTestIntegrator(nullptr,new MixedScalarWeakGradientIntegrator(*negomeg_cf),
                        TestSpace::q_space, TestSpace::v_space);
   // ω^2 (q,δq)
   a->AddTestIntegrator(new MassIntegrator(*omeg2_cf),nullptr,
                        TestSpace::q_space, TestSpace::q_space);

   // integrators in the PML region
   // Custom integration rule for the test space in the PML region
   const IntegrationRule &ir = IntRules.Get(pmesh.GetElementGeometry(0),
                                            2*test_order + 1);
   if (pml)
   {
      // Trial integrators
      // i ω (p,q) = i ω ( (β_r p,q) + i (β_i p,q) )
      //           = (- ω b_i p ) + i (ω β_r p,q)
      a->AddTrialIntegrator(new MixedScalarMassIntegrator(negomeg_detJ_i_restr),
                            new MixedScalarMassIntegrator(omeg_detJ_r_restr),
                            TrialSpace::p_space,TestSpace::q_space);

      // i ω (α u,v) =  i ω ( (α_re u,v) + i (α_im u,v) )
      //             = (-ω a_im u,v) + i (ω a_re u, v)
      a->AddTrialIntegrator(new TransposeIntegrator(
                               new VectorFEMassIntegrator(negomeg_Jt_J_detJinv_i_restr)),
                            new TransposeIntegrator(
                               new VectorFEMassIntegrator(omeg_Jt_J_detJinv_r_restr)),
                            TrialSpace::u_space,TestSpace::v_space);
      // Test integrators
      // -i ω (α ∇q,δv) = -i ω ( (α_r ∇q,δv) + i (α_i ∇q,δv) )
      //                = (ω α_i ∇q,δv) + i (-ω α_r ∇q,δv)
      MixedVectorGradientIntegrator * integ0_r = new MixedVectorGradientIntegrator(
         omeg_Jt_J_detJinv_i_restr);
      integ0_r->SetIntegrationRule(ir);
      MixedVectorGradientIntegrator * integ0_i = new MixedVectorGradientIntegrator(
         negomeg_Jt_J_detJinv_r_restr);
      integ0_i->SetIntegrationRule(ir);
      a->AddTestIntegrator(integ0_r, integ0_i,
                           TestSpace::q_space,TestSpace::v_space);

      // i ω (α^* v,∇ δq)  = i ω (ᾱ v,∇ δq) (since α is diagonal)
      //                   = i ω ( (α_r v,∇ δq) - i (α_i v,∇ δq)
      //                   = (ω α_i v, ∇ δq) + i (ω α_r v,∇ δq )
      a->AddTestIntegrator(new MixedVectorWeakDivergenceIntegrator(
                              negomeg_Jt_J_detJinv_i_restr),
                           new MixedVectorWeakDivergenceIntegrator(negomeg_Jt_J_detJinv_r_restr),
                           TestSpace::v_space,TestSpace::q_space);

      // ω^2 (|α|^2 v,δv) α α^* = |α|^2 since α is diagonal
      VectorFEMassIntegrator * integ1 = new VectorFEMassIntegrator(
         omeg2_abs_Jt_J_detJinv_2_restr);
      integ1->SetIntegrationRule(ir);
      a->AddTestIntegrator(integ1, nullptr,TestSpace::v_space,TestSpace::v_space);

      // - i ω (β ∇⋅v,δq) = - i ω ( (β_re ∇⋅v,δq) + i (β_im ∇⋅v,δq) )
      //                  = (ω β_im ∇⋅v,δq) + i (-ω β_re ∇⋅v,δq )
      a->AddTestIntegrator(new VectorFEDivergenceIntegrator(omeg_detJ_i_restr),
                           new VectorFEDivergenceIntegrator(negomeg_detJ_r_restr),
                           TestSpace::v_space,TestSpace::q_space);

      // i ω (β̄ q,∇⋅v) =  i ω ( (β_re ∇⋅v,δq) - i (β_im ∇⋅v,δq) )
      //               =  (ω β_im ∇⋅v,δq) + i (ω β_re ∇⋅v,δq )
      a->AddTestIntegrator(new MixedScalarWeakGradientIntegrator(
                              negomeg_detJ_i_restr),
                           new MixedScalarWeakGradientIntegrator(negomeg_detJ_r_restr),
                           TestSpace::q_space,TestSpace::v_space);

      // ω^2 (β̄ β q,δq) = (ω^2 |β|^2 )
      MassIntegrator * integ = new MassIntegrator(omeg2_abs_detJ_2_restr);
      integ->SetIntegrationRule(ir);
      a->AddTestIntegrator(integ,nullptr,
                           TestSpace::q_space,TestSpace::q_space);
   }

   // RHS
   FunctionCoefficient f_rhs_r(rhs_func_r);
   FunctionCoefficient f_rhs_i(rhs_func_i);
   FunctionCoefficient f_source(source_function);
   if (prob == prob_type::gaussian_beam)
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(f_rhs_r),
                               new DomainLFIntegrator(f_rhs_i),
                               TestSpace::q_space);
   }
   if (prob == prob_type::pml_general)
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(f_source),nullptr,
                               TestSpace::q_space);
   }

   FunctionCoefficient hatpex_r(hatp_exact_r);
   FunctionCoefficient hatpex_i(hatp_exact_i);

   Array<int> elements_to_refine;

   socketstream p_out_r;
   socketstream p_out_i;
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
   int dof0 = 0;

   ParGridFunction p_r, p_i, u_r, u_i;

   ParaViewDataCollection * paraview_dc = nullptr;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection(enum_str[prob], &pmesh);
      paraview_dc->SetPrefixPath("ParaView/Acoustics");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("p_r",&p_r);
      paraview_dc->RegisterField("p_i",&p_i);
      paraview_dc->RegisterField("u_r",&u_r);
      paraview_dc->RegisterField("u_i",&u_i);
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
         hatp_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
         if (pml && prob>2)
         {
            ess_bdr = 0;
            ess_bdr[1] = 1;
         }
      }

      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += p_fes->GetTrueVSize() + u_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = p_fes->GetVSize();
      offsets[2] = u_fes->GetVSize();
      offsets[3] = hatp_fes->GetVSize();
      offsets[4] = hatu_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;

      if (prob!=2)
      {
         ParGridFunction hatp_gf_r(hatp_fes, x, offsets[2]);
         ParGridFunction hatp_gf_i(hatp_fes, x, offsets.Last()+ offsets[2]);
         hatp_gf_r.ProjectBdrCoefficient(hatpex_r, ess_bdr);
         hatp_gf_i.ProjectBdrCoefficient(hatpex_i, ess_bdr);
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
      M.owns_blocks=0;

      if (!static_cond)
      {
         HypreBoomerAMG * solver_p = new HypreBoomerAMG((HypreParMatrix &)
                                                        BlockA_r->GetBlock(0,0));
         solver_p->SetPrintLevel(0);
         solver_p->SetSystemsOptions(dim);
         HypreBoomerAMG * solver_u = new HypreBoomerAMG((HypreParMatrix &)
                                                        BlockA_r->GetBlock(1,1));
         solver_u->SetPrintLevel(0);
         solver_u->SetSystemsOptions(dim);
         M.SetDiagonalBlock(0,solver_p);
         M.SetDiagonalBlock(1,solver_u);
         M.SetDiagonalBlock(num_blocks,solver_p);
         M.SetDiagonalBlock(num_blocks+1,solver_u);
      }

      HypreBoomerAMG * solver_hatp = new HypreBoomerAMG((HypreParMatrix &)
                                                        BlockA_r->GetBlock(skip,skip));
      solver_hatp->SetPrintLevel(0);

      HypreSolver * solver_hatu = nullptr;
      if (dim == 2)
      {
         // AMS preconditioner for 2D H(div) (trace) space
         solver_hatu = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(skip+1,skip+1),
                                    hatu_fes);
         dynamic_cast<HypreAMS*>(solver_hatu)->SetPrintLevel(0);
      }
      else
      {
         // ADS preconditioner for 3D H(div) (trace) space
         solver_hatu = new HypreADS((HypreParMatrix &)BlockA_r->GetBlock(skip+1,skip+1),
                                    hatu_fes);
         dynamic_cast<HypreADS*>(solver_hatu)->SetPrintLevel(0);
      }

      M.SetDiagonalBlock(skip,solver_hatp);
      M.SetDiagonalBlock(skip+1,solver_hatu);
      M.SetDiagonalBlock(skip+num_blocks,solver_hatp);
      M.SetDiagonalBlock(skip+num_blocks+1,solver_hatu);

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

      p_r.MakeRef(p_fes, x, 0);
      p_i.MakeRef(p_fes, x, offsets.Last());

      u_r.MakeRef(u_fes,x, offsets[1]);
      u_i.MakeRef(u_fes,x, offsets.Last()+offsets[1]);

      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GlobalTrueVSize();
      }

      double L2Error = 0.0;
      double rate_err = 0.0;
      if (exact_known)
      {
         FunctionCoefficient p_ex_r(p_exact_r);
         FunctionCoefficient p_ex_i(p_exact_i);
         double p_err_r = p_r.ComputeL2Error(p_ex_r);
         double p_err_i = p_i.ComputeL2Error(p_ex_i);

         // Error in velocity
         VectorFunctionCoefficient u_ex_r(dim,u_exact_r);
         VectorFunctionCoefficient u_ex_i(dim,u_exact_i);

         double u_err_r = u_r.ComputeL2Error(u_ex_r);
         double u_err_i = u_i.ComputeL2Error(u_ex_i);

         L2Error = sqrt(p_err_r*p_err_r + p_err_i*p_err_i
                        +u_err_r*u_err_r + u_err_i*u_err_i);

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
                   << std::setw(4) <<  2*rnum << " π  | ";
         if (exact_known)
         {
            std::cout << std::setprecision(3) << std::setw(10)
                      << std::scientific <<  err0 << " | "
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
         VisualizeField(p_out_r,vishost, visport, p_r,
                        "Numerical presure (real part)", 0, 0, 500, 500, keys);
         VisualizeField(p_out_i,vishost, visport, p_i,
                        "Numerical presure (imaginary part)", 501, 0, 500, 500, keys);
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

   if (paraview)
   {
      delete paraview_dc;
   }

   if (pml)
   {
      delete omeg_cf;
      delete omeg2_cf;
      delete negomeg_cf;
      delete pml;
   }
   delete a;
   delete q_fec;
   delete v_fec;
   delete hatp_fes;
   delete hatp_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete u_fec;
   delete p_fec;
   delete u_fes;
   delete p_fes;

   return 0;
}

double p_exact_r(const Vector &x)
{
   return acoustics_solution(x).real();
}

double p_exact_i(const Vector &x)
{
   return acoustics_solution(x).imag();
}

double hatp_exact_r(const Vector & X)
{
   return p_exact_r(X);
}

double hatp_exact_i(const Vector & X)
{
   return p_exact_i(X);
}

void gradp_exact_r(const Vector &x, Vector &grad_r)
{
   grad_r.SetSize(x.Size());
   vector<complex<double>> grad;
   acoustics_solution_grad(x,grad);
   for (unsigned i = 0; i < grad.size(); i++)
   {
      grad_r[i] = grad[i].real();
   }
}

void gradp_exact_i(const Vector &x, Vector &grad_i)
{
   grad_i.SetSize(x.Size());
   vector<complex<double>> grad;
   acoustics_solution_grad(x,grad);
   for (unsigned i = 0; i < grad.size(); i++)
   {
      grad_i[i] = grad[i].imag();
   }
}

double d2_exact_r(const Vector &x)
{
   return acoustics_solution_laplacian(x).real();
}

double d2_exact_i(const Vector &x)
{
   return acoustics_solution_laplacian(x).imag();
}

//  u = - ∇ p / (i ω )
//    = i (∇ p_r + i * ∇ p_i)  / ω
//    = - ∇ p_i / ω + i ∇ p_r / ω
void u_exact_r(const Vector &x, Vector & u)
{
   gradp_exact_i(x,u);
   u *= -1./omega;
}

void u_exact_i(const Vector &x, Vector & u)
{
   gradp_exact_r(x,u);
   u *= 1./omega;
}

void hatu_exact_r(const Vector & X, Vector & hatu)
{
   u_exact_r(X,hatu);
}
void hatu_exact_i(const Vector & X, Vector & hatu)
{
   u_exact_i(X,hatu);
}

//  ∇⋅u = i Δ p / ω
//      = i (Δ p_r + i * Δ p_i)  / ω
//      = - Δ p_i / ω + i Δ p_r / ω

double divu_exact_r(const Vector &x)
{
   return -d2_exact_i(x)/omega;
}

double divu_exact_i(const Vector &x)
{
   return d2_exact_r(x)/omega;
}

// f = ∇⋅u + i ω p
// f_r = ∇⋅u_r - ω p_i
double rhs_func_r(const Vector &x)
{
   double p = p_exact_i(x);
   double divu = divu_exact_r(x);
   return divu - omega * p;
}

// f_i = ∇⋅u_i + ω p_r
double rhs_func_i(const Vector &x)
{
   double p = p_exact_r(x);
   double divu = divu_exact_i(x);
   return divu + omega * p;
}

complex<double> acoustics_solution(const Vector & X)
{
   complex<double> zi = complex<double>(0., 1.);
   switch (prob)
   {
      case pml_plane_wave_scatter:
      case plane_wave:
      {
         double beta = omega/std::sqrt((double)X.Size());
         complex<double> alpha = beta * zi * X.Sum();
         return exp(alpha);
      }
      break;
      case gaussian_beam:
      case pml_beam_scatter:
      {
         double rk = omega;
         double degrees = 45;
         double alpha = (180+degrees) * M_PI/180.;
         double sina = sin(alpha);
         double cosa = cos(alpha);
         // shift the origin
         double shift = 0.1;
         double xprim=X(0) + shift;
         double yprim=X(1) + shift;

         double x = xprim*sina - yprim*cosa;
         double y = xprim*cosa + yprim*sina;
         //wavelength
         double rl = 2.*M_PI/rk;

         // beam waist radius
         double w0 = 0.05;

         // function w
         double fact = rl/M_PI/(w0*w0);
         double aux = 1. + (fact*y)*(fact*y);

         double w = w0*sqrt(aux);

         double phi0 = atan(fact*y);

         double r = y + 1./y/(fact*fact);

         // pressure
         complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r +
                              zi*phi0/2.;
         double pf = pow(2.0/M_PI/(w*w),0.25);

         return pf*exp(ze);
      }
      break;
      case pml_pointsource:
      {
         double x = X(0)-0.5;
         double y = X(1)-0.5;
         double r = sqrt(x*x + y*y);
         double beta = omega * r;
         complex<double> Ho = jn(0, beta) + zi * yn(0, beta);
         return 0.25*zi*Ho;
      }
      break;
      default:
         MFEM_ABORT("Should be unreachable");
         return 1;
         break;
   }
}

void acoustics_solution_grad(const Vector & X, vector<complex<double>> & dp)
{
   dp.resize(X.Size());
   complex<double> zi = complex<double>(0., 1.);
   // initialize
   for (int i = 0; i<X.Size(); i++) { dp[i] = 0.0; }
   switch (prob)
   {
      case pml_plane_wave_scatter:
      case plane_wave:
      {
         double beta = omega/std::sqrt((double)X.Size());
         complex<double> alpha = beta * zi * X.Sum();
         complex<double> p = exp(alpha);
         for (int i = 0; i<X.Size(); i++)
         {
            dp[i] = zi * beta * p;
         }
      }
      break;
      case gaussian_beam:
      case pml_beam_scatter:
      {
         double rk = omega;
         double degrees = 45;
         double alpha = (180+degrees) * M_PI/180.;
         double sina = sin(alpha);
         double cosa = cos(alpha);
         // shift the origin
         double shift = 0.1;
         double xprim=X(0) + shift;
         double yprim=X(1) + shift;

         double x = xprim*sina - yprim*cosa;
         double y = xprim*cosa + yprim*sina;
         double dxdxprim = sina, dxdyprim = -cosa;
         double dydxprim = cosa, dydyprim =  sina;
         //wavelength
         double rl = 2.*M_PI/rk;

         // beam waist radius
         double w0 = 0.05;

         // function w
         double fact = rl/M_PI/(w0*w0);
         double aux = 1. + (fact*y)*(fact*y);

         double w = w0*sqrt(aux);
         double dwdy = w0*fact*fact*y/sqrt(aux);

         double phi0 = atan(fact*y);
         double dphi0dy = cos(phi0)*cos(phi0)*fact;

         double r = y + 1./y/(fact*fact);
         double drdy = 1. - 1./(y*y)/(fact*fact);

         // pressure
         complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r +
                              zi*phi0/2.;

         complex<double> zdedx = -2.*x/(w*w) - 2.*zi*M_PI*x/rl/r;
         complex<double> zdedy = 2.*x*x/(w*w*w)*dwdy - zi*rk + zi*M_PI*x*x/rl/
                                 (r*r)*drdy + zi*dphi0dy/2.;

         double pf = pow(2.0/M_PI/(w*w),0.25);
         double dpfdy = -pow(2./M_PI/(w*w),-0.75)/M_PI/(w*w*w)*dwdy;

         complex<double> zp = pf*exp(ze);
         complex<double> zdpdx = zp*zdedx;
         complex<double> zdpdy = dpfdy*exp(ze)+zp*zdedy;

         dp[0] = (zdpdx*dxdxprim + zdpdy*dydxprim);
         dp[1] = (zdpdx*dxdyprim + zdpdy*dydyprim);
      }
      break;
      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }
}

complex<double> acoustics_solution_laplacian(const Vector & X)
{
   complex<double> zi = complex<double>(0., 1.);
   switch (prob)
   {
      case pml_plane_wave_scatter:
      case plane_wave:
      {
         double beta = omega/std::sqrt((double)X.Size());
         complex<double> alpha = beta * zi * X.Sum();
         return dim * beta * beta * exp(alpha);
      }
      break;
      case gaussian_beam:
      case pml_beam_scatter:
      {
         double rk = omega;
         double degrees = 45;
         double alpha = (180+degrees) * M_PI/180.;
         double sina = sin(alpha);
         double cosa = cos(alpha);
         // shift the origin
         double shift = 0.1;
         double xprim=X(0) + shift;
         double yprim=X(1) + shift;

         double x = xprim*sina - yprim*cosa;
         double y = xprim*cosa + yprim*sina;
         double dxdxprim = sina, dxdyprim = -cosa;
         double dydxprim = cosa, dydyprim =  sina;
         //wavelength
         double rl = 2.*M_PI/rk;

         // beam waist radius
         double w0 = 0.05;

         // function w
         double fact = rl/M_PI/(w0*w0);
         double aux = 1. + (fact*y)*(fact*y);

         double w = w0*sqrt(aux);
         double dwdy = w0*fact*fact*y/sqrt(aux);
         double d2wdydy = w0*fact*fact*(1. - (fact*y)*(fact*y)/aux)/sqrt(aux);

         double phi0 = atan(fact*y);
         double dphi0dy = cos(phi0)*cos(phi0)*fact;
         double d2phi0dydy = -2.*cos(phi0)*sin(phi0)*fact*dphi0dy;

         double r = y + 1./y/(fact*fact);
         double drdy = 1. - 1./(y*y)/(fact*fact);
         double d2rdydy = 2./(y*y*y)/(fact*fact);

         // pressure
         complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r +
                              zi*phi0/2.;

         complex<double> zdedx = -2.*x/(w*w) - 2.*zi*M_PI*x/rl/r;
         complex<double> zdedy = 2.*x*x/(w*w*w)*dwdy - zi*rk + zi*M_PI*x*x/rl/
                                 (r*r)*drdy + zi*dphi0dy/2.;
         complex<double> zd2edxdx = -2./(w*w) - 2.*zi*M_PI/rl/r;
         complex<double> zd2edxdy = 4.*x/(w*w*w)*dwdy + 2.*zi*M_PI*x/rl/(r*r)*drdy;
         complex<double> zd2edydx = zd2edxdy;
         complex<double> zd2edydy = -6.*x*x/(w*w*w*w)*dwdy*dwdy + 2.*x*x/
                                    (w*w*w)*d2wdydy - 2.*zi*M_PI*x*x/rl/(r*r*r)*drdy*drdy
                                    + zi*M_PI*x*x/rl/(r*r)*d2rdydy + zi/2.*d2phi0dydy;

         double pf = pow(2.0/M_PI/(w*w),0.25);
         double dpfdy = -pow(2./M_PI/(w*w),-0.75)/M_PI/(w*w*w)*dwdy;
         double d2pfdydy = -1./M_PI*pow(2./M_PI,-0.75)*(-1.5*pow(w,-2.5)
                                                        *dwdy*dwdy + pow(w,-1.5)*d2wdydy);


         complex<double> zp = pf*exp(ze);
         complex<double> zdpdx = zp*zdedx;
         complex<double> zdpdy = dpfdy*exp(ze)+zp*zdedy;
         complex<double> zd2pdxdx = zdpdx*zdedx + zp*zd2edxdx;
         complex<double> zd2pdxdy = zdpdy*zdedx + zp*zd2edxdy;
         complex<double> zd2pdydx = dpfdy*exp(ze)*zdedx + zdpdx*zdedy + zp*zd2edydx;
         complex<double> zd2pdydy = d2pfdydy*exp(ze) + dpfdy*exp(
                                       ze)*zdedy + zdpdy*zdedy + zp*zd2edydy;


         return (zd2pdxdx*dxdxprim + zd2pdydx*dydxprim)*dxdxprim
                + (zd2pdxdy*dxdxprim + zd2pdydy*dydxprim)*dydxprim
                + (zd2pdxdx*dxdyprim + zd2pdydx*dydyprim)*dxdyprim
                + (zd2pdxdy*dxdyprim + zd2pdydy*dydyprim)*dydyprim;
      }
      break;
      default:
         MFEM_ABORT("Should be unreachable");
         return 1;
         break;
   }
}

double source_function(const Vector &x)
{
   Vector center(dim);
   center = 0.5;
   double r = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      r += pow(x[i] - center[i], 2.);
   }
   double n = 5.0 * omega / M_PI;
   double coeff = pow(n, 2) / M_PI;
   double alpha = -pow(n, 2) * r;
   return -omega * coeff * exp(alpha)/omega;
}
