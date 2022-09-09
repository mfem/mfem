//                   MFEM Ultraweak DPG Maxwell parallel example
//
// Compile with: make pmaxwell
//
// sample run 
// mpirun -np 4 ./pmaxwell -o 3 -m ../../../data/inline-quad.mesh -sref 2 -pref 3 -rnum 4.1 -prob 0 -sc 
// mpirun -np 6 ./pmaxwell -o 3 -sref 0 -pref 15 -prob 2 -theta 0.5 


// Description:  
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Maxwell problem

//      ∇×(1/μ ∇×E) - ω^2 ϵ E = Ĵ ,   in Ω
//                E×n = E_0, on ∂Ω

// It solves the following kinds of problems 
// 1) Known exact solutions with error convergence rates
//    a) A manufactured solution problem where E is a plane beam
// 2) Fichera "microwave" problem
// 3) PML problems
//    a) Gausian beam scattering from a square
//    b) Plane wave scattering from a square
//    c) Point Source

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
// where A = A = [0 1; -1 0];

// E ∈ L^2(Ω)^2, H ∈ L^2(Ω)
// Ê ∈ H^-1/2(Ω)(Γ_h), Ĥ ∈ H^1/2(Γ_h)  
//  i ω μ (H,F) + (E, ∇ × F) + < AÊ, F > = 0,      ∀ F ∈ H^1      
// -i ω ϵ (E,G) + (H,∇ × G)  + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)      
//                                    Ê = E_0     on ∂Ω 
// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) |   < Ê, F >   |              |         |
// |   |              |             |              |              |         |
// | G | -i ω ϵ (E,G) |  (H,∇ × G)  |              | < Ĥ, G × n > |  (J,G)  |  
// where (F,G) ∈  H^1 × H(curl,Ω)

// in 3D 
// E,H ∈ (L^2(Ω))^3 
// Ê ∈ H_0^1/2(Ω)(curl, Γ_h), Ĥ ∈ H^-1/2(curl, Γ_h)  
//  i ω μ (H,F) + (E,∇ × F) + < Ê, F × n > = 0,      ∀ F ∈ H(curl,Ω)      
// -i ω ϵ (E,G) + (H,∇ × G) + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)      
//                                   Ê × n = E_0     on ∂Ω 
// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) | < n × Ê, F > |              |         |
// |   |              |             |              |              |         |
// | G | -i ω ϵ (E,G) |  (H,∇ × G)  |              | < n × Ĥ, G > |  (J,G)  |  
// where (F,G) ∈  H(curl,Ω) × H(curl,Ω)

// Here we use the "Adjoint Graph" norm on the test space i.e.,
// ||(F,G)||^2_V = ||A^*(F,G)||^2 + ||(F,G)||^2 where A is the
// maxwell operator defined by (1)

// The PML formulation is

//      ∇×(1/μ α ∇×E) - ω^2 ϵ β E = Ĵ ,   in Ω
//                E×n = E_0, on ∂Ω

// where α = |J|^-1 J^T J (in 2D it's the scalar |J|^-1),
// β = |J| J^-1 J^-T, J is the Jacobian of the stretching map
// and |J| its determinant.

// The first order system reads
//  i ω μ α^-1 H + ∇ × E = 0,   in Ω
//    -i ω ϵ β E + ∇ × H = J,   in Ω
//                 E × n = E_0, on ∂Ω

// and the ultraweak formulation is

// in 2D 
// E ∈ L^2(Ω)^2, H ∈ L^2(Ω)
// Ê ∈ H^-1/2(Ω)(Γ_h), Ĥ ∈ H^1/2(Γ_h)  
//  i ω μ (α^-1 H,F) + (E, ∇ × F) + < AÊ, F > = 0,         ∀ F ∈ H^1      
// -i ω ϵ (α E,G)    + (H,∇ × G)  + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)      
//                                             Ê = E_0     on ∂Ω 
// ---------------------------------------------------------------------------------
// |   |       E        |        H         |      Ê       |       Ĥ      |  RHS    |
// ---------------------------------------------------------------------------------
// | F |  (E,∇ × F)     | i ω μ (α^-1 H,F) |   < Ê, F >   |              |         |
// |   |                |                  |              |              |         |
// | G | -i ω ϵ (β E,G) |    (H,∇ × G)     |              | < Ĥ, G × n > |  (J,G)  |  

// where (F,G) ∈  H^1 × H(curl,Ω)

// 
// in 3D 
// E,H ∈ (L^2(Ω))^3 
// Ê ∈ H_0^1/2(Ω)(curl, Γ_h), Ĥ ∈ H^-1/2(curl, Γ_h)  
//  i ω μ (α^-1 H,F) + (E,∇ × F) + < Ê, F × n > = 0,      ∀ F ∈ H(curl,Ω)      
// -i ω ϵ (β E,G)    + (H,∇ × G) + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)      
//                                        Ê × n = E_0     on ∂Ω 
// -------------------------------------------------------------------------------
// |   |       E      |      H           |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------------
// | F |  ( E,∇ × F)  | i ω μ (α^-1 H,F) | < n × Ê, F > |              |         |
// |   |              |                  |              |              |         |
// | G | -iωϵ (β E,G) |   (H,∇ × G)      |              | < n × Ĥ, G > |  (J,G)  |  
// where (F,G) ∈  H(curl,Ω) × H(curl,Ω)


#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "util/pml.hpp"
#include "../../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

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
                      std::vector<complex<double>> &E, 
                      std::vector<complex<double>> &curlE, 
                      std::vector<complex<double>> &curlcurlE);

void maxwell_solution_r(const Vector & X, Vector &E_r, 
                      Vector &curlE_r, 
                      Vector &curlcurlE_r);

void maxwell_solution_i(const Vector & X, Vector &E_i, 
                      Vector &curlE_i, 
                      Vector &curlcurlE_i);     

int dim;
int dimc;
double omega;
double mu = 1.0;
double epsilon = 1.0;

enum prob_type
{
   plane_wave,
   fichera_oven,
   pml_beam_scatter,
   pml_plane_wave_scatter,
   pml_pointsource     
};

prob_type prob;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   double theta = 0.0;
   bool static_cond = false;
   int iprob = 0;
   int sr = 0;
   int pr = 1;
   bool exact_known = false;
   bool with_pml = false;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");    
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");                  
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: plane wave, 1: Fichera 'oven', "
                  " 2: Scattering of a Gaussian beam, 3: Scattering of a plane wave, " 
                  " 4: Point source");                       
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");     
   args.AddOption(&theta, "-theta", "--theta",
                  "Theta parameter for AMR");                    
   args.AddOption(&sr, "-sref", "--serial_ref",
                  "Number of parallel refinements.");                                              
   args.AddOption(&pr, "-pref", "--parallel_ref",
                  "Number of parallel refinements.");        
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");                                            
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

   if (iprob > 4) { iprob = 0; }
   prob = (prob_type)iprob;
   omega = 2.*M_PI*rnum;

   if (prob == 0)
   {
      exact_known = true;
   }
   else if (prob == 1)
   {
      mesh_file = "fichera-waveguide.mesh";
      omega = 5.0;
   }
   else
   {
      with_pml = true;
      mesh_file = "scatter.mesh";
   }

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   dimc = (dim == 3) ? 3 : 1;

   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }
   mesh.EnsureNCMesh(false);

   CartesianPML * pml = nullptr;
   if (with_pml)
   {
      Array2D<double> length(dim, 2); length = 0.125;
      pml = new CartesianPML(&mesh,length);
      pml->SetOmega(omega);
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   
   // PML element attribute marker
   Array<int> attr;
   Array<int> attrPML;
   if (pml) pml->SetAttributes(&pmesh, &attr, &attrPML);

   // Define spaces
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
   PmlMatrixCoefficient Jt_J_detJinv_r(dim,Jt_J_detJinv_r_function,pml);
   PmlMatrixCoefficient Jt_J_detJinv_i(dim,Jt_J_detJinv_i_function,pml);
   PmlMatrixCoefficient abs_Jt_J_detJinv_2(dim,abs_Jt_J_detJinv_2_function,pml);

   ProductCoefficient negmuomeg_detJ_r(negmuomeg,detJ_r);
   ProductCoefficient negmuomeg_detJ_i(negmuomeg,detJ_i);
   ProductCoefficient muomeg_detJ_r(muomeg,detJ_r);
   ProductCoefficient mu2omeg2_detJ_2(mu2omeg2,abs_detJ_2);
   ScalarMatrixProductCoefficient epsomeg_Jt_J_detJinv_i(epsomeg, Jt_J_detJinv_i);
   ScalarMatrixProductCoefficient epsomeg_Jt_J_detJinv_r(epsomeg, Jt_J_detJinv_r);
   ScalarMatrixProductCoefficient negepsomeg_Jt_J_detJinv_r(negepsomeg, Jt_J_detJinv_r);
   ScalarMatrixProductCoefficient muomeg_Jt_J_detJinv_r(muomeg,Jt_J_detJinv_r);
   ScalarMatrixProductCoefficient negmuomeg_Jt_J_detJinv_i(negmuomeg,Jt_J_detJinv_i);
   ScalarMatrixProductCoefficient negmuomeg_Jt_J_detJinv_r(negmuomeg,Jt_J_detJinv_r);
   ScalarMatrixProductCoefficient mu2omeg2_Jt_J_detJ_inv_2(mu2omeg2,abs_Jt_J_detJinv_2);
   ScalarMatrixProductCoefficient eps2omeg2_Jt_J_detJ_inv_2(eps2omeg2,abs_Jt_J_detJinv_2);
   MatrixProductCoefficient epsomeg_Jt_J_detJinv_i_rot(epsomeg_Jt_J_detJinv_i, rot); 
   MatrixProductCoefficient epsomeg_Jt_J_detJinv_r_rot(epsomeg_Jt_J_detJinv_r, rot); 
   MatrixProductCoefficient negepsomeg_Jt_J_detJinv_r_rot(negepsomeg_Jt_J_detJinv_r, rot); 

   RestrictedCoefficient negmuomeg_detJ_r_restr(negmuomeg_detJ_r,attrPML);
   RestrictedCoefficient negmuomeg_detJ_i_restr(negmuomeg_detJ_i,attrPML);
   RestrictedCoefficient muomeg_detJ_r_restr(muomeg_detJ_r,attrPML);
   RestrictedCoefficient mu2omeg2_detJ_2_restr(mu2omeg2_detJ_2,attrPML);
   MatrixRestrictedCoefficient epsomeg_Jt_J_detJinv_i_restr(epsomeg_Jt_J_detJinv_i,attrPML);
   MatrixRestrictedCoefficient epsomeg_Jt_J_detJinv_r_restr(epsomeg_Jt_J_detJinv_r,attrPML);
   MatrixRestrictedCoefficient negepsomeg_Jt_J_detJinv_r_restr(negepsomeg_Jt_J_detJinv_r,attrPML);
   MatrixRestrictedCoefficient muomeg_Jt_J_detJinv_r_restr(muomeg_Jt_J_detJinv_r,attrPML);
   MatrixRestrictedCoefficient negmuomeg_Jt_J_detJinv_i_restr(negmuomeg_Jt_J_detJinv_i,attrPML);
   MatrixRestrictedCoefficient negmuomeg_Jt_J_detJinv_r_restr(negmuomeg_Jt_J_detJinv_r,attrPML);
   MatrixRestrictedCoefficient mu2omeg2_Jt_J_detJ_inv_2_restr(mu2omeg2_Jt_J_detJ_inv_2,attrPML);
   MatrixRestrictedCoefficient eps2omeg2_Jt_J_detJ_inv_2_restr(eps2omeg2_Jt_J_detJ_inv_2,attrPML);
   MatrixRestrictedCoefficient epsomeg_Jt_J_detJinv_i_rot_restr(epsomeg_Jt_J_detJinv_i_rot, attrPML); 
   MatrixRestrictedCoefficient epsomeg_Jt_J_detJinv_r_rot_restr(epsomeg_Jt_J_detJinv_r_rot, attrPML); 
   MatrixRestrictedCoefficient negepsomeg_Jt_J_detJinv_r_rot_restr(negepsomeg_Jt_J_detJinv_r_rot, attrPML); 


   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices();

   // (E,∇ × F)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),nullptr,0,0);
   // -i ω ϵ (E , G) = i (- ω ϵ E, G)
   a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(*negepsomeg_cf)),0,1);
   //  (H,∇ × G) 
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),nullptr,1,1);
   // < n×Ĥ ,G>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,3,1);
   // test integrators
   // (∇×G ,∇× δG)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,1,1);
   // (G,δG)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,1,1);
   //
   
   if (dim == 3)
   {
      // i ω μ (H, F)
      a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(*muomeg_cf)),1,0);
      // < n×Ê,F>
      a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,2,0);

      // test integrators 
      // (∇×F,∇×δF)
      a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,0,0);
      // (F,δF)
      a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,0,0);
      // μ^2 ω^2 (F,δF)
      a->AddTestIntegrator(new VectorFEMassIntegrator(*mu2omeg2_cf),nullptr,0,0);
      // -i ω μ (F,∇ × δG) = i (F, ω μ ∇ × δ G)
      a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(*negmuomeg_cf),0,1);
      // -i ω ϵ (∇ × F, δG)
      a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(*negepsomeg_cf),0,1);
      // i ω μ (∇ × G,δF)
      a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(*muomeg_cf),1,0);
      // i ω ϵ (G, ∇ × δF )
      a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(*epsomeg_cf),1,0);
      // ϵ^2 ω^2 (G,δG)
      a->AddTestIntegrator(new VectorFEMassIntegrator(*eps2omeg2_cf),nullptr,1,1);
   }
   else
   {
      // i ω μ (H, F)
      a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(*muomeg_cf),1,0);
      // < n×Ê,F>
      a->AddTrialIntegrator(new TraceIntegrator,nullptr,2,0);
      // test integrators 
      // (∇F,∇δF)
      a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,0,0);
      // (F,δF)
      a->AddTestIntegrator(new MassIntegrator(one),nullptr,0,0);
      // μ^2 ω^2 (F,δF)
      a->AddTestIntegrator(new MassIntegrator(*mu2omeg2_cf),nullptr,0,0);
      // -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
      a->AddTestIntegrator(nullptr,
         new TransposeIntegrator(new MixedCurlIntegrator(*negmuomeg_cf)),0,1);

      // -i ω ϵ (∇ × F, δG) = i (- ω ϵ A ∇ F,δG), A = [0 1; -1; 0]
      a->AddTestIntegrator(nullptr,new MixedVectorGradientIntegrator(*negepsrot_cf),0,1);   

      // i ω μ (∇ × G,δF) = i (ω μ ∇ × G, δF )
         a->AddTestIntegrator(nullptr,new MixedCurlIntegrator(*muomeg_cf),1,0);

         // i ω ϵ (G, ∇ × δF ) =  i (ω ϵ G, A ∇ δF) = i ( G , ω ϵ A ∇ δF) 
      a->AddTestIntegrator(nullptr,
         new TransposeIntegrator(new MixedVectorGradientIntegrator(*epsrot_cf)),1,0);

      a->AddTestIntegrator(new VectorFEMassIntegrator(*eps2omeg2_cf),nullptr,1,1);    
   }
   if (pml)
   {
      //trial integrators
      // -i ω ϵ (β E , G) = -i ω ϵ ((β_re + i β_im) E, G) 
      //                  = (ω ϵ β_im E, G) + i (- ω ϵ β_re E, G)     
      a->AddTrialIntegrator(
         new TransposeIntegrator(new VectorFEMassIntegrator(epsomeg_Jt_J_detJinv_i_restr)),
         new TransposeIntegrator(new VectorFEMassIntegrator(negepsomeg_Jt_J_detJinv_r_restr)),0,1);
      if (dim == 3)
      {
         //trial integrators
         // i ω μ (α^-1 H, F) = i ω μ ( (α^-1_re + i α^-1_im) H, F) 
         //                   = (- ω μ α^-1_im, H,F) + i *(ω μ α^-1_re H, F)
         a->AddTrialIntegrator(
            new TransposeIntegrator(new VectorFEMassIntegrator(negmuomeg_Jt_J_detJinv_i_restr)),
            new TransposeIntegrator(new VectorFEMassIntegrator(muomeg_Jt_J_detJinv_r_restr)),1,0);
         // test integrators

         // μ^2 ω^2 (|α|^-2 F,δF)
         a->AddTestIntegrator(new VectorFEMassIntegrator
         (mu2omeg2_Jt_J_detJ_inv_2_restr),nullptr,0,0);
         // -i ω μ (α^-* F,∇ × δG) = i (F, - ω μ α^-1 ∇ × δ G)
         //                        = i (F, - ω μ (α^-1_re + i α^-1_im) ∇ × δ G)
         //                        = (F, - ω μ α^-1_im ∇ × δ G) + i (F, - ω μ α^-1_re ∇×δG)
         a->AddTestIntegrator(new MixedVectorWeakCurlIntegrator(negmuomeg_Jt_J_detJinv_i_restr),
                              new MixedVectorWeakCurlIntegrator(negmuomeg_Jt_J_detJinv_r_restr),0,1);
         // -i ω ϵ (β ∇ × F, δG) = -i ω ϵ ((β_re + i β_im) ∇ × F, δG) 
         //                      = (ω ϵ β_im  ∇ × F, δG) + i (- ω ϵ β_re ∇ × F, δG)
         a->AddTestIntegrator(new MixedVectorCurlIntegrator(epsomeg_Jt_J_detJinv_i_restr),
                              new MixedVectorCurlIntegrator(negepsomeg_Jt_J_detJinv_r_restr),0,1);
         // i ω μ (α^-1 ∇ × G,δF) = i ω μ ((α^-1_re + i α^-1_im) ∇ × G,δF) 
         //                       = (- ω μ α^-1_im ∇ × G,δF) + i (ω μ α^-1_re ∇ × G,δF)
         a->AddTestIntegrator(new MixedVectorCurlIntegrator(negmuomeg_Jt_J_detJinv_i_restr),
                              new MixedVectorCurlIntegrator(muomeg_Jt_J_detJinv_r_restr),1,0);
         // i ω ϵ (β^* G, ∇×δF) = i ω ϵ ( (β_re - i β_im) G, ∇×δF)
         //                     = (ω ϵ β_im G, ∇×δF) + i ( ω ϵ β_re G, ∇×δF)
         a->AddTestIntegrator(new MixedVectorWeakCurlIntegrator(epsomeg_Jt_J_detJinv_i_restr),
                              new MixedVectorWeakCurlIntegrator(epsomeg_Jt_J_detJinv_r_restr),1,0);
         // ϵ^2 ω^2 (|β|^2 G,δG)
         a->AddTestIntegrator(new VectorFEMassIntegrator(eps2omeg2_Jt_J_detJ_inv_2_restr),nullptr,1,1);
      }
      else
      {
         //trial integrators
         // i ω μ (α^-1 H, F) = i ω μ ( (α^-1_re + i α^-1_im) H, F) 
         //                   = (- ω μ α^-1_im, H,F) + i *(ω μ α^-1_re H, F)
         a->AddTrialIntegrator(
            new MixedScalarMassIntegrator(negmuomeg_detJ_i_restr),
            new MixedScalarMassIntegrator(muomeg_detJ_r_restr),1,0);
         // test integrators
         // μ^2 ω^2 (|α|^-2 F,δF)
         a->AddTestIntegrator(new MassIntegrator(mu2omeg2_detJ_2_restr),nullptr,0,0);
         // -i ω μ (α^-* F,∇ × δG) = (F, ω μ α^-1 ∇ × δ G)
         //                        =(F, - ω μ α^-1_im ∇ × δ G) + i (F, - ω μ α^-1_re ∇×δG)
         a->AddTestIntegrator(
            new TransposeIntegrator(new MixedCurlIntegrator(negmuomeg_detJ_i_restr)),
            new TransposeIntegrator(new MixedCurlIntegrator(negmuomeg_detJ_r_restr)),0,1);
         // -i ω ϵ (β ∇ × F, δG) = i (- ω ϵ β A ∇ F,δG), A = [0 1; -1; 0] 
         //                      = (ω ϵ β_im A ∇ F, δG) + i (- ω ϵ β_re A ∇ F, δG)
         a->AddTestIntegrator(new MixedVectorGradientIntegrator(epsomeg_Jt_J_detJinv_i_rot_restr),
                              new MixedVectorGradientIntegrator(negepsomeg_Jt_J_detJinv_r_rot_restr),0,1);            
         // i ω μ (α^-1 ∇ × G,δF) = i (ω μ α^-1 ∇ × G, δF )
         //                       = (- ω μ α^-1_im ∇ × G,δF) + i (ω μ α^-1_re ∇ × G,δF)
         a->AddTestIntegrator(new MixedCurlIntegrator(negmuomeg_detJ_i_restr),
                              new MixedCurlIntegrator(muomeg_detJ_r_restr),1,0);
         // i ω ϵ (β^* G, ∇ × δF ) = i ( G , ω ϵ β A ∇ δF) 
         //                        =  ( G , ω ϵ β_im A ∇ δF) + i ( G , ω ϵ β_re A ∇ δF)
         a->AddTestIntegrator(
            new TransposeIntegrator(new MixedVectorGradientIntegrator(epsomeg_Jt_J_detJinv_i_rot_restr)),
            new TransposeIntegrator(new MixedVectorGradientIntegrator(epsomeg_Jt_J_detJinv_r_rot_restr)),1,0);
         // ϵ^2 ω^2 (|β|^2 G,δG)
         a->AddTestIntegrator(new VectorFEMassIntegrator(eps2omeg2_Jt_J_detJ_inv_2_restr),nullptr,1,1);            
      }
   }
   // RHS

   VectorFunctionCoefficient f_rhs_r(dim,rhs_func_r);
   VectorFunctionCoefficient f_rhs_i(dim,rhs_func_i);
   if (prob < 2)
   {
      a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f_rhs_r),
                               new VectorFEDomainLFIntegrator(f_rhs_i),1);
   }

   VectorFunctionCoefficient hatEex_r(dim,hatE_exact_r);
   VectorFunctionCoefficient hatEex_i(dim,hatE_exact_i);

   Array<int> elements_to_refine;

   socketstream E_out_r;
   socketstream E_out_i;

   if (myid == 0)
   {
      std::cout << "\n  Ref |" 
                << "    Dofs    |" 
                << "   ω   |" ;
      if (exact_known)
      {
         std::cout  << "  L2 Error  |" 
                    << "  Rate  |" ;
      }
      std::cout << "  Residual  |" 
                << "  Rate  |" 
                << " PCG it |" << endl;
      std::cout << std::string((exact_known) ? 80 : 58,'-')      
                << endl;      
   }

   double res0 = 0.;
   double err0 = 0.;
   int dof0;

   for (int it = 0; it<pr; it++)
   {
      if (static_cond) { a->EnableStaticCondensation(); }
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
      double * xdata = x.GetData();

      ParComplexGridFunction hatE_gf(hatE_fes);
      hatE_gf.real().MakeRef(hatE_fes,&xdata[offsets[2]]);
      hatE_gf.imag().MakeRef(hatE_fes,&xdata[offsets.Last()+ offsets[2]]);

      if (dim == 3)
      {
         hatE_gf.ProjectBdrCoefficientTangent(hatEex_r,hatEex_i, ess_bdr);
      }
      else
      {
         hatE_gf.ProjectBdrCoefficientNormal(hatEex_r,hatEex_i, ess_bdr);
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
      for (int i=0; i<num_blocks;i++)
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
         HypreBoomerAMG * solver_E = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(0,0));
         solver_E->SetPrintLevel(0);
         solver_E->SetSystemsOptions(dim);
         HypreBoomerAMG * solver_H = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(1,1));
         solver_H->SetPrintLevel(0);
         solver_H->SetSystemsOptions(dim);
         M.SetDiagonalBlock(0,solver_E);
         M.SetDiagonalBlock(1,solver_H);
         M.SetDiagonalBlock(num_blocks,solver_E);
         M.SetDiagonalBlock(num_blocks+1,solver_H);
      }

      HypreSolver * solver_hatH = nullptr;
      HypreAMS * solver_hatE = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(skip,skip), 
                               hatE_fes);
      solver_hatE->SetPrintLevel(0);  
      if (dim == 2)
      {
         solver_hatH = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(skip+1,skip+1));
         dynamic_cast<HypreBoomerAMG*>(solver_hatH)->SetPrintLevel(0);
      }
      else
      {
         solver_hatH = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(skip+1,skip+1), hatH_fes);
         dynamic_cast<HypreAMS*>(solver_hatH)->SetPrintLevel(0);
      }

      M.SetDiagonalBlock(skip,solver_hatE);
      M.SetDiagonalBlock(skip+1,solver_hatH);
      M.SetDiagonalBlock(skip+num_blocks,solver_hatE);
      M.SetDiagonalBlock(skip+num_blocks+1,solver_hatH);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-6);
      cg.SetAbsTol(1e-6);
      cg.SetMaxIter(100000);
      cg.SetPrintLevel(1);
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

      ParComplexGridFunction E(E_fes);
      E.real().MakeRef(E_fes,x.GetData());
      E.imag().MakeRef(E_fes,&x.GetData()[offsets.Last()]);

      ParComplexGridFunction H(H_fes);
      H.real().MakeRef(H_fes,&x.GetData()[offsets[1]]);
      H.imag().MakeRef(H_fes,&x.GetData()[offsets.Last()+offsets[1]]);

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
         double E_err_r = E.real().ComputeL2Error(E_ex_r);
         double E_err_i = E.imag().ComputeL2Error(E_ex_i);
         double H_err_r = H.real().ComputeL2Error(H_ex_r);
         double H_err_i = H.imag().ComputeL2Error(H_ex_i);
         L2Error = sqrt(  E_err_r*E_err_r + E_err_i*E_err_i 
                        + H_err_r*H_err_r + H_err_i*H_err_i );
         rate_err = (it) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;
         err0 = L2Error;
      }

      double rate_res = (it) ? dim*log(res0/globalresidual)/log((double)dof0/dofs) : 0.0;

      res0 = globalresidual;
      dof0 = dofs;

      if (myid == 0)
      {
         std::ios oldState(nullptr);
         oldState.copyfmt(std::cout);
         std::cout << std::right << std::setw(5) << it << " | " 
                  << std::setw(10) <<  dof0 << " | " 
                  << std::setprecision(0) << std::fixed
                  << std::setw(2) <<  2*rnum << " π  | " 
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
         common::VisualizeField(E_out_r,vishost, visport, E.real(), 
                               "Numerical presure (real part)", 0, 0, 500, 500, keys);
         common::VisualizeField(E_out_i,vishost, visport, E.imag(), 
                        "Numerical presure (imaginary part)", 501, 0, 500, 500, keys);   
      }

      if (it == pr-1)
         break;

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
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
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
   Vector curlE_r;
   Vector curlcurlE_r;

   maxwell_solution_r(x,E_r,curlE_r,curlcurlE_r);
}

void E_exact_i(const Vector &x, Vector & E_i)
{
   Vector curlE_i;
   Vector curlcurlE_i;

   maxwell_solution_i(x,E_i,curlE_i,curlcurlE_i);
}

void curlE_exact_r(const Vector &x, Vector &curlE_r)
{
   Vector E_r;
   Vector curlcurlE_r;

   maxwell_solution_r(x,E_r,curlE_r,curlcurlE_r);
}

void curlE_exact_i(const Vector &x, Vector &curlE_i)
{
   Vector E_i;
   Vector curlcurlE_i;

   maxwell_solution_i(x,E_i,curlE_i,curlcurlE_i);
}

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r)
{
   Vector E_r;
   Vector curlE_r;
   maxwell_solution_r(x,E_r,curlE_r,curlcurlE_r);
}

void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i)
{
   Vector E_i;
   Vector curlE_i;
   maxwell_solution_i(x,E_i,curlE_i,curlcurlE_i);
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

void maxwell_solution(const Vector & X, std::vector<complex<double>> &E, 
                      std::vector<complex<double>> &curlE, 
                      std::vector<complex<double>> &curlcurlE)
{
   complex<double> zi = complex<double>(0., 1.);
   E.resize(dim);
   curlE.resize(dimc);
   curlcurlE.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = 0.0;
      curlcurlE[i] = 0.0;;
   }
   for (int i = 0; i < dimc; ++i)
   {
      curlE[i] = 0.0;
   }

   switch (prob)
   {
   case plane_wave:
   case pml_plane_wave_scatter:
   {
      std::complex<double> pw = exp(-zi * omega * (X.Sum()));
      E[0] = pw;
      E[1] = 0.0;
      if (dim == 3)
      {
         E[2] = 0.0;
         curlE[0] = 0.0;
         curlE[1] = -zi * omega * pw;
         curlE[2] =  zi * omega * pw;

         curlcurlE[0] = 2.0 * omega * omega * pw;
         curlcurlE[1] = - omega * omega * pw;
         curlcurlE[2] = - omega * omega * pw;
      }
      else
      {
         curlE[0] = zi * omega * pw;
         curlcurlE[0] =   omega * omega * pw;
         curlcurlE[1] = - omega * omega * pw ;
      }
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
   case pml_beam_scatter:
   {
      double rk = omega;
      double alpha = 45 * M_PI/180.;
      double sina = sin(alpha); 
      double cosa = cos(alpha);
      // shift the origin
      double xprim=X(0) - 0.5;
      double yprim=X(1) - 0.5;

      double  x = xprim*sina - yprim*cosa;
      double  y = xprim*cosa + yprim*sina;
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

      complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r + zi*phi0/2.;
      double pf = pow(2.0/M_PI/(w*w),0.25);
      complex<double> zp = pf*exp(ze);

      E[0] = zp;   
      E[1] = 0.0;
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

void maxwell_solution_r(const Vector & X, Vector &E_r, 
                        Vector &curlE_r, 
                        Vector &curlcurlE_r)
{
   E_r.SetSize(dim);
   curlE_r.SetSize(dimc);
   curlcurlE_r.SetSize(dim);

   std::vector<complex<double>> E;
   std::vector<complex<double>> curlE;
   std::vector<complex<double>> curlcurlE;

   maxwell_solution(X,E,curlE,curlcurlE);
   for (int i = 0; i<dim ; i++)
   {
      E_r(i) = E[i].real();
      curlcurlE_r(i) = curlcurlE[i].real();
   }
   for (int i = 0; i<dimc; i++)
   {
      curlE_r(i) = curlE[i].real();
   }
}

void maxwell_solution_i(const Vector & X, Vector &E_i, 
                      Vector &curlE_i, 
                      Vector &curlcurlE_i)
{
   E_i.SetSize(dim);
   curlE_i.SetSize(dimc);
   curlcurlE_i.SetSize(dim);

   std::vector<complex<double>> E;
   std::vector<complex<double>> curlE;
   std::vector<complex<double>> curlcurlE;

   maxwell_solution(X,E,curlE,curlcurlE);
   for (int i = 0; i<dim; i++)
   {
      E_i(i) = E[i].imag();
      curlcurlE_i(i) = curlcurlE[i].imag();
   }
   for (int i = 0; i<dimc; i++)
   {
      curlE_i(i) = curlE[i].imag();
   }
}  