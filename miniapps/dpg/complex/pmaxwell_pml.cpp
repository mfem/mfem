//                   MFEM Ultraweak DPG Maxwell parallel example
//
// Compile with: make pcomplex_uw_dpg
//
// sample run 
// mpirun -np 2 ./pcomplex_uw_dpg -o 3 -m ../../../data/inline-quad.mesh -sref 2 -pref 3 -rnum 4.1 -prob 1 -sc -graph-norm

//      ∇×(1/μ α ∇×E) - ω^2 ϵ β E = Ĵ ,   in Ω
//                E×n = E_0, on ∂Ω

// where α = |J|^-1 J^T J (in 2D it's the scalar |J|^-1)
// and   β = |J| J^-1 J^-T

// First Order System

//  i ω μ α^-1 H + ∇ × E = 0,   in Ω
// -i ω ϵ β    E + ∇ × H = J,   in Ω
//            E × n = E_0, on ∂Ω

// note: Ĵ = -iωJ
// in 2D 
// E is vector valued and H is scalar. 
//    (∇ × E, F) = (E, ∇ × F) + < n × E , F>
// or (∇ ⋅ AE , F) = (AE, ∇ F) + < AE ⋅ n, F>
// where A = A = [0 1; -1 0];

// UW-DPG:
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


#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Class for setting up a simple Cartesian PML region
class CartesianPML
{
private:
   Mesh *mesh;

   int dim;

   // Length of the PML Region in each direction
   Array2D<double> length;

   // Computational Domain Boundary
   Array2D<double> comp_dom_bdr;

   // Domain Boundary
   Array2D<double> dom_bdr;

   // Integer Array identifying elements in the PML
   // 0: in the PML, 1: not in the PML
   Array<int> elems;

   // Compute Domain and Computational Domain Boundaries
   void SetBoundaries();

public:
   // Constructor
   CartesianPML(Mesh *mesh_,Array2D<double> length_);

   // Return Computational Domain Boundary
   Array2D<double> GetCompDomainBdr() {return comp_dom_bdr;}

   // Return Domain Boundary
   Array2D<double> GetDomainBdr() {return dom_bdr;}

   // Return Markers list for elements
   Array<int> * GetMarkedPMLElements() {return &elems;}

   // Mark elements in the PML region
   void SetAttributes(ParMesh *pmesh);

   // PML complex stretching function
   void StretchFunction(const Vector &x, vector<complex<double>> &dxs);
};

// Class for returning the PML coefficients of the bilinear form
class PMLMatrixCoefficient : public MatrixCoefficient
{
private:
   CartesianPML * pml = nullptr;
   void (*Function)(const Vector &, CartesianPML *, DenseMatrix &);
public:
   PMLMatrixCoefficient(int dim, void(*F)(const Vector &, CartesianPML *,
                                              DenseMatrix &),
                            CartesianPML * pml_)
      : MatrixCoefficient(dim), pml(pml_), Function(F)
   {}

   using MatrixCoefficient::Eval;

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(height,width);
      (*Function)(transip, pml, K);
   }
};

class PMLScalarCoefficient : public Coefficient
{
private:
   CartesianPML * pml = nullptr;
   double (*Function)(const Vector &, CartesianPML *);
public:
   PMLScalarCoefficient(double(*F)(const Vector &, CartesianPML *),
                            CartesianPML * pml_)
      : Coefficient(), pml(pml_), Function(F)
   {}

   using Coefficient::Eval;

   virtual double Eval(ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      return (*Function)(transip, pml);
   }
};

void maxwell_solution(const Vector &x, vector<complex<double>> &Eval);
void source(const Vector &x, Vector & f);

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void inv_alpha_function_re(const Vector &x, CartesianPML * pml, DenseMatrix & K);
void inv_alpha_function_im(const Vector &x, CartesianPML * pml, DenseMatrix & K);
void inv_abs2_alpha_function(const Vector &x, CartesianPML * pml, DenseMatrix & K);

double inv_scalar_alpha_function_re(const Vector &x, CartesianPML * pml);
double inv_scalar_alpha_function_im(const Vector &x, CartesianPML * pml);
double inv_scalar_abs2_alpha_function(const Vector &x, CartesianPML * pml);

void beta_function_re(const Vector &x, CartesianPML * pml, DenseMatrix & K);
void beta_function_im(const Vector &x, CartesianPML * pml, DenseMatrix & K);
void abs2_beta_function(const Vector &x, CartesianPML * pml, DenseMatrix & K);

Array2D<double> comp_domain_bdr;
Array2D<double> domain_bdr;

int dim;
int dimc;
double omega;
double mu = 1.0;
double epsilon = 1.0;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   bool adjoint_graph_norm = false;
   bool static_cond = false;
   int iprob = 0;
   int sr = 0;
   int pr = 1;

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
                  " 0: polynomial, 1: plane wave, 2: Gaussian beam");                     
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");     
   args.AddOption(&adjoint_graph_norm, "-graph-norm", "--adjoint-graph-norm",
                  "-no-graph-norm", "--no-adjoint-graph-norm",
                  "Enable or disable Adjoint Graph Norm on the test space");                                
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

   omega = 2.*M_PI*rnum;

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   dimc = (dim == 3) ? 3 : 1;

   // Setup PML length
   Array2D<double> length(dim, 2); length = 0.0;
   length = 0.25;

   CartesianPML * pml = new CartesianPML(&mesh,length);
   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr = pml->GetDomainBdr();

   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 8. Set element attributes in order to distinguish elements in the PML
   pml->SetAttributes(&pmesh);

   int test_order = order+delta_order;

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

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   Array<int> attr;
   Array<int> attrPML;
   if (pmesh.attributes.Size())
   {
      attr.SetSize(pmesh.attributes.Max());
      attrPML.SetSize(pmesh.attributes.Max());
      attr = 0;   attr[0] = 1;
      attrPML = 0;
      if (pmesh.attributes.Max() > 1)
      {
         attrPML[1] = 1;
      }
   }

   // PML coefficients
   // α^-1 = |J| J^-1 J-T (in case of d=2  α the scalar |J|)
   PMLMatrixCoefficient * inv_alpha_re=nullptr; 
   PMLMatrixCoefficient * inv_alpha_im=nullptr; 

   PMLScalarCoefficient * inv_scalar_alpha_re=nullptr; 
   PMLScalarCoefficient * inv_scalar_alpha_im=nullptr; 

   if (dim == 2)
   {
      inv_scalar_alpha_re = new PMLScalarCoefficient(inv_scalar_alpha_function_re, pml);
      inv_scalar_alpha_im = new PMLScalarCoefficient(inv_scalar_alpha_function_im, pml);
   }
   else
   {
      inv_alpha_re = new PMLMatrixCoefficient(dim,inv_alpha_function_re, pml);
      inv_alpha_im = new PMLMatrixCoefficient(dim,inv_alpha_function_im, pml);
   }


   PMLMatrixCoefficient beta_re(dim, beta_function_re,pml);
   PMLMatrixCoefficient beta_im(dim, beta_function_im,pml);

   VectorFunctionCoefficient f(dim, source);


   // Coefficients
   Vector dim_zero(dim); dim_zero = 0.0;
   Vector dimc_zero(dimc); dimc_zero = 0.0;
   VectorConstantCoefficient E_zero(dim_zero);
   VectorConstantCoefficient H_zero(dimc_zero);


   ConstantCoefficient one(1.0);
   ConstantCoefficient eps2omeg2(epsilon*epsilon*omega*omega);
   ConstantCoefficient mu2omeg2(mu*mu*omega*omega);
   ConstantCoefficient muomeg(mu*omega);
   ConstantCoefficient negepsomeg(-epsilon*omega);
   ConstantCoefficient epsomeg(epsilon*omega);
   ConstantCoefficient negmuomeg(-mu*omega);

   DenseMatrix rot_mat(2);
   rot_mat(0,0) = 0.; rot_mat(0,1) = 1.;
   rot_mat(1,0) = -1.; rot_mat(1,1) = 0.;
   MatrixConstantCoefficient rot(rot_mat);
   ScalarMatrixProductCoefficient epsrot(epsomeg,rot);
   ScalarMatrixProductCoefficient negepsrot(negepsomeg,rot);

   Array<ParFiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(E_fes);
   trial_fes.Append(H_fes);
   trial_fes.Append(hatE_fes);
   trial_fes.Append(hatH_fes);

   test_fec.Append(F_fec);
   test_fec.Append(G_fec);

   if (myid == 0)
   {
      mfem::out << "\n  Ref |" 
                << "       Mesh       |"
                << "    Dofs    |" 
                << "   ω   |" 
                << " PCG it |"
                << " PCG time |"  << endl;
      mfem::out << " --------------------"      
                <<  "---------------------"    
                <<  "------"    
                <<  "-------------------" << endl;      
   }


   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);
   // a->StoreMatrices();


   // (E,∇ × F)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),nullptr,0,0);

   // Not in PML region
   // -i ω ϵ (E , G)
   RestrictedCoefficient negepsomeg_restr(negepsomeg,attr);
   a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(negepsomeg_restr)),0,1);

   // In PML region
   // -i ω ϵ (β E , G) = -i ω ϵ ((β_re + i β_im) E, G) 
   //                  = (ω ϵ β_im E, G) + i (- ω ϵ β_re E, G)  
   ScalarMatrixProductCoefficient c1_re(epsomeg, beta_im);
   ScalarMatrixProductCoefficient c1_im(negepsomeg, beta_re);
   MatrixRestrictedCoefficient restr_c1_re(c1_re,attrPML);
   MatrixRestrictedCoefficient restr_c1_im(c1_im,attrPML);

   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(restr_c1_re)),new TransposeIntegrator(new VectorFEMassIntegrator(restr_c1_im)),0,1);


   // Not in PML
   // i ω μ (H, F)
   RestrictedCoefficient muomeg_restr(muomeg,attr);

   if (dim == 3)
   {
      a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(muomeg_restr)),1,0);
   }
   else
   {
      a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(muomeg_restr),1,0);
   }

   // In PML 
   // i ω μ (α^-1 H, F) = i ω μ ( (α^-1_re + i α^-1_im) H, F) 
   //                   = (- ω μ α^-1_im, H,F) + i *(ω μ α^-1_re H, F)

   ScalarMatrixProductCoefficient * c2_re = nullptr;
   ScalarMatrixProductCoefficient * c2_im = nullptr;
   MatrixRestrictedCoefficient * restr_c2_re = nullptr;
   MatrixRestrictedCoefficient * restr_c2_im = nullptr;
   ProductCoefficient * scalar_c2_re = nullptr;
   ProductCoefficient * scalar_c2_im = nullptr;
   RestrictedCoefficient * restr_scalar_c2_re = nullptr;
   RestrictedCoefficient * restr_scalar_c2_im = nullptr;
   if (dim == 2)
   {
      scalar_c2_re = new ProductCoefficient(negmuomeg,*inv_scalar_alpha_im);
      scalar_c2_im = new ProductCoefficient(muomeg,*inv_scalar_alpha_re);
      restr_scalar_c2_re = new RestrictedCoefficient(*scalar_c2_re,attrPML);
      restr_scalar_c2_im = new RestrictedCoefficient(*scalar_c2_im,attrPML);
   }
   else
   {
      c2_re = new ScalarMatrixProductCoefficient(negmuomeg,*inv_alpha_im);
      c2_im = new ScalarMatrixProductCoefficient(muomeg,*inv_alpha_re);
      restr_c2_re = new MatrixRestrictedCoefficient(*c2_re,attrPML);
      restr_c2_im = new MatrixRestrictedCoefficient(*c2_im,attrPML);
   }


  if (dim == 3)
   {
      a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(*restr_c2_re)),
      new TransposeIntegrator(new VectorFEMassIntegrator(*restr_c2_im)),1,0);
   }
   else
   {
      a->AddTrialIntegrator(new MixedScalarMassIntegrator(*restr_scalar_c2_re),
      new MixedScalarMassIntegrator(*restr_scalar_c2_im),1,0);
   }

   //  (H,∇ × G) 
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),nullptr,1,1);

   // < n×Ê,F>
   if (dim == 3)
   {
      a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,2,0);
   }
   else
   {
      a->AddTrialIntegrator(new TraceIntegrator,nullptr,2,0);
   }

   // < n×Ĥ ,G>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,3,1);


   // test integrators 
   //space-induced norm for H(curl) × H(curl)
   if (dim == 3)
   {
      // (∇×F,∇×δF)
      a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,0,0);
      // (F,δF)
      a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,0,0);
   }
   else
   {
      // (∇F,∇δF)
      a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,0,0);
      // (F,δF)
      a->AddTestIntegrator(new MassIntegrator(one),nullptr,0,0);
   }

   // (∇×G ,∇× δG)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,1,1);
   // (G,δG)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,1,1);

   // additional integrators for the adjoint graph norm
   RestrictedCoefficient * restr_mu2omeg2 = nullptr;
   RestrictedCoefficient * restr_negmuomeg = nullptr;
   RestrictedCoefficient * restr_negepsomeg = nullptr;
   RestrictedCoefficient * restr_epsomeg = nullptr;
   RestrictedCoefficient * restr_muomeg = nullptr;
   RestrictedCoefficient * restr_eps2omeg2 = nullptr;
   MatrixRestrictedCoefficient *restr_negepsrot = nullptr;
   MatrixRestrictedCoefficient *restr_epsrot = nullptr;

   // PML Coefficients
   // μ^2 ω^2 |α|^-2
   // 3D
   PMLMatrixCoefficient * inv_abs2alpha = nullptr;
   ScalarMatrixProductCoefficient * mu2omeg2_inv_abs2alpha = nullptr;
   MatrixRestrictedCoefficient * restr_mu2omeg2_inv_abs2alpha = nullptr;
   // 2D
   PMLScalarCoefficient * inv_abs2_scalar_alpha = nullptr;
   ProductCoefficient * mu2omeg2_inv_abs2_scalar_alpha = nullptr;
   RestrictedCoefficient * restr_mu2omeg2_inv_abs2_scalar_alpha = nullptr;

   // i ω μ α^-1 ---> restr_c2_re, restr_c2_im & restr_scalar_c2_re, restr_scalar_c2_im
   // -i ω ϵ β = - i ω ϵ (β_re + i β_im) = ω ϵ β_im - i ω ϵ β_re   ---> restr_c1_re & restr_c1_im


   // i ω ϵ β^* =  i ω ϵ (β_re^T - i β_im^T) = i ω ϵ (β_re - i β_im) = ω ϵ β_im + i ω ϵ β_re ---> restr_c1_re, -restr_c1_im


   // ϵ^2 ω^2 |β|^2
   PMLMatrixCoefficient * absbeta2 = nullptr;
   ScalarMatrixProductCoefficient * eps2omeg2_abs2beta = nullptr;
   MatrixRestrictedCoefficient * restr_eps2omeg2_abs2beta = nullptr;

   if (adjoint_graph_norm)
   {   

      // not in PML
      // Restricted coefficients in computational domain
      restr_mu2omeg2 = new RestrictedCoefficient(mu2omeg2, attr);
      restr_negmuomeg = new RestrictedCoefficient(negmuomeg,attr);
      restr_negepsomeg = new RestrictedCoefficient(negepsomeg,attr);
      restr_epsomeg = new RestrictedCoefficient(epsomeg,attr);
      restr_muomeg = new RestrictedCoefficient(muomeg,attr);
      restr_eps2omeg2 = new RestrictedCoefficient(eps2omeg2,attr);

      if(dim == 3)
      {
         // μ^2 ω^2 (F,δF)
         a->AddTestIntegrator(new VectorFEMassIntegrator(*restr_mu2omeg2),nullptr,0,0);
         // -i ω μ (F,∇ × δG) = (F, ω μ ∇ × δ G)
         a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(*restr_negmuomeg),0,1);
         // -i ω ϵ (∇ × F, δG)
         a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(*restr_negepsomeg),0,1);
         // i ω μ (∇ × G,δF)
         a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(*restr_muomeg),1,0);
         // i ω ϵ (G, ∇ × δF )
         a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(*restr_epsomeg),1,0);
         // ϵ^2 ω^2 (G,δG)
         a->AddTestIntegrator(new VectorFEMassIntegrator(*restr_eps2omeg2),nullptr,1,1);
      }
      else
      {
         // μ^2 ω^2 (F,δF)
         a->AddTestIntegrator(new MassIntegrator(*restr_mu2omeg2),nullptr,0,0);

         // -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
         a->AddTestIntegrator(nullptr,
            new TransposeIntegrator(new MixedCurlIntegrator(*restr_negmuomeg)),0,1);

         // -i ω ϵ (∇ × F, δG) = i (- ω ϵ A ∇ F,δG), A = [0 1; -1; 0]
         restr_negepsrot = new MatrixRestrictedCoefficient(negepsrot, attr);
         a->AddTestIntegrator(nullptr,new MixedVectorGradientIntegrator(*restr_negepsrot),0,1);   

         // i ω μ (∇ × G,δF) = i (ω μ ∇ × G, δF )
         a->AddTestIntegrator(nullptr,new MixedCurlIntegrator(muomeg),1,0);

         // i ω ϵ (G, ∇ × δF ) =  i (ω ϵ G, A ∇ δF) = i ( G , ω ϵ A ∇ δF) 
         restr_epsrot = new MatrixRestrictedCoefficient(epsrot,attr);
         a->AddTestIntegrator(nullptr,
                   new TransposeIntegrator(new MixedVectorGradientIntegrator(*restr_epsrot)),1,0);

         // or    i ( ω ϵ A^t G, ∇ δF) = i (- ω ϵ A G, ∇ δF)
         // a->AddTestIntegrator(nullptr,
                  //  new MixedVectorWeakDivergenceIntegrator(epsrot),1,0);
         // ϵ^2 ω^2 (G,δG)
         a->AddTestIntegrator(new VectorFEMassIntegrator(*restr_eps2omeg2),nullptr,1,1);            
      }

      // In PML
      // Restricted Coefficient in the PML region
      if(dim == 3)
      {
         // μ^2 ω^2 (|α|^-2 F,δF)
         inv_abs2alpha = new PMLMatrixCoefficient(dim,inv_abs2_alpha_function,pml);
         mu2omeg2_inv_abs2alpha = new ScalarMatrixProductCoefficient(mu2omeg2,*inv_abs2alpha);
         restr_mu2omeg2_inv_abs2alpha = new MatrixRestrictedCoefficient(*mu2omeg2_inv_abs2alpha,attrPML);
         a->AddTestIntegrator(new VectorFEMassIntegrator(*restr_mu2omeg2_inv_abs2alpha),nullptr,0,0);
         // -i ω μ (α^-* F,∇ × δG) = i (F, - ω μ α^-1 ∇ × δ G)
         //                        = i (F, - ω μ (α^-1_re + i α^-1_im) ∇ × δ G)
         //                        = (F, - ω μ α^-1_im ∇ × δ G) + i (F, - ω μ α^-1_re ∇×δG)
         ScalarMatrixProductCoefficient * neg_restr_c2_im 
         = new ScalarMatrixProductCoefficient(-1.0,*restr_c2_im);
         a->AddTestIntegrator(new MixedVectorWeakCurlIntegrator(*restr_c2_re),
                              new MixedVectorWeakCurlIntegrator(*neg_restr_c2_im),0,1);

         // -i ω ϵ (β ∇ × F, δG) = -i ω ϵ ((β_re + i β_im) ∇ × F, δG) 
         //                      = (ω ϵ β_im  ∇ × F, δG) + i (- ω ϵ β_re ∇ × F, δG)

         a->AddTestIntegrator(new MixedVectorCurlIntegrator(restr_c1_re),
                              new MixedVectorCurlIntegrator(restr_c1_im),0,1);
         // i ω μ (α^-1 ∇ × G,δF) = i ω μ ((α^-1_re + i α^-1_im) ∇ × G,δF) 
         //                       = (- ω μ α^-1_im ∇ × G,δF) + i (ω μ α^-1_re ∇ × G,δF)

         a->AddTestIntegrator(new MixedVectorCurlIntegrator(*restr_c2_re),
                              new MixedVectorCurlIntegrator(*restr_c2_im),1,0);

         // i ω ϵ (β^* G, ∇×δF) = i ω ϵ ( (β_re - i β_im) G, ∇×δF)
         //                     = (ω ϵ β_im G, ∇×δF) + i ( ω ϵ β_re G, ∇×δF)
         ScalarMatrixProductCoefficient *neg_restr_c1_im 
         = new ScalarMatrixProductCoefficient(-1.0, restr_c1_im);

         a->AddTestIntegrator(new MixedVectorWeakCurlIntegrator(restr_c1_re),
                              new MixedVectorWeakCurlIntegrator(*neg_restr_c1_im),1,0);
         // ϵ^2 ω^2 (|β|^2 G,δG)
         absbeta2 = new PMLMatrixCoefficient(dim, abs2_beta_function,pml);
         eps2omeg2_abs2beta = new ScalarMatrixProductCoefficient(eps2omeg2, *absbeta2);
         restr_eps2omeg2_abs2beta = new MatrixRestrictedCoefficient(*eps2omeg2_abs2beta, attrPML);
         a->AddTestIntegrator(new VectorFEMassIntegrator(*restr_eps2omeg2_abs2beta),nullptr,1,1);
      }
      else
      {
         // μ^2 ω^2 (|α|^-2 F,δF)
         inv_abs2_scalar_alpha = new PMLScalarCoefficient(inv_scalar_abs2_alpha_function,pml);
         mu2omeg2_inv_abs2_scalar_alpha = new ProductCoefficient(mu2omeg2, *inv_abs2_scalar_alpha);
         restr_mu2omeg2_inv_abs2_scalar_alpha = 
            new RestrictedCoefficient(*mu2omeg2_inv_abs2_scalar_alpha,attrPML);
         a->AddTestIntegrator(new MassIntegrator(*restr_mu2omeg2_inv_abs2_scalar_alpha),nullptr,0,0);

         // -i ω μ (α^-* F,∇ × δG) = (F, ω μ α^-1 ∇ × δ G)
         ProductCoefficient *neg_restr_scalar_c2_im = new ProductCoefficient(-1.0, *restr_scalar_c2_im);

         a->AddTestIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(*restr_scalar_c2_re)),
                              new TransposeIntegrator(new MixedCurlIntegrator(*neg_restr_scalar_c2_im)),0,1);

         // -i ω ϵ (β ∇ × F, δG) = i (- ω ϵ β A ∇ F,δG), A = [0 1; -1; 0] 

         MatrixProductCoefficient * restr_betaA_re = new MatrixProductCoefficient(restr_c1_re, rot); 
         MatrixProductCoefficient * restr_betaA_im = new MatrixProductCoefficient(restr_c1_im, rot); 
         a->AddTestIntegrator(new MixedVectorGradientIntegrator(*restr_betaA_re),
                              new MixedVectorGradientIntegrator(*restr_betaA_im),0,1);   

         // i ω μ (α^-1 ∇ × G,δF) = i (ω μ α^-1 ∇ × G, δF )
         a->AddTestIntegrator(new MixedCurlIntegrator(*restr_scalar_c2_re),
                              new MixedCurlIntegrator(*restr_scalar_c2_im),1,0);

         // i ω ϵ (β^* G, ∇ × δF ) =  i (ω ϵ β^* G, A ∇ δF) = i ( G , ω ϵ β A ∇ δF) 
         // = i ( G , ω ϵ (β_re + i β_im) A ∇ δF) 
         // =  ( G , ω ϵ β_im A ∇ δF) + i ( G , ω ϵ β_re A ∇ δF)

         MatrixProductCoefficient * d1_re = new MatrixProductCoefficient(beta_im, epsrot);
         MatrixProductCoefficient * d1_im = new MatrixProductCoefficient(beta_re, epsrot);
         a->AddTestIntegrator(new TransposeIntegrator(new MixedVectorGradientIntegrator(*d1_re)),
                              new TransposeIntegrator(new MixedVectorGradientIntegrator(*d1_im)),1,0);

         // ϵ^2 ω^2 (|β|^2 G,δG)
         absbeta2 = new PMLMatrixCoefficient(dim, abs2_beta_function,pml);
         eps2omeg2_abs2beta = new ScalarMatrixProductCoefficient(eps2omeg2, *absbeta2);
         restr_eps2omeg2_abs2beta = new MatrixRestrictedCoefficient(*eps2omeg2_abs2beta, attrPML);
         a->AddTestIntegrator(new VectorFEMassIntegrator(*restr_eps2omeg2_abs2beta),nullptr,1,1);            
      }


   }

   // RHS

   a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f),nullptr,1);

   int dofs;

   for (int it = 0; it<pr; it++)
   {

      dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GlobalTrueVSize();
      }

      if (static_cond) { a->EnableStaticCondensation(); }
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         hatE_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
         // hatH_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += E_fes->GetTrueVSize() + H_fes->GetTrueVSize();
                           // + hatE_fes->GetTrueVSize();
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
      StopWatch chrono1;
      chrono1.Clear();
      chrono1.Start();
      BlockDiagonalPreconditioner * M = new BlockDiagonalPreconditioner(tdof_offsets);

      if (!static_cond)
      {
         HypreBoomerAMG * solver_E = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(0,0));
         solver_E->SetPrintLevel(0);
         solver_E->SetSystemsOptions(dim);
         HypreBoomerAMG * solver_H = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(1,1));
         solver_H->SetPrintLevel(0);
         solver_H->SetSystemsOptions(dim);
         M->SetDiagonalBlock(0,solver_E);
         M->SetDiagonalBlock(1,solver_H);
         M->SetDiagonalBlock(num_blocks,solver_E);
         M->SetDiagonalBlock(num_blocks+1,solver_H);
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

      M->SetDiagonalBlock(skip,solver_hatE);
      M->SetDiagonalBlock(skip+1,solver_hatH);
      M->SetDiagonalBlock(skip+num_blocks,solver_hatE);
      M->SetDiagonalBlock(skip+num_blocks+1,solver_hatH);



      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-5);
      cg.SetAbsTol(1e-5);
      cg.SetMaxIter(10000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(*M); 
      cg.SetOperator(blockA);
      chrono1.Stop();
      if (myid == 0)
      {
         mfem::out << "T setup = " << chrono1.RealTime() << endl;
      }
      StopWatch chrono;
      chrono.Clear();
      chrono.Start();
      cg.Mult(B, X);
      chrono.Stop();
      delete M;

      int ne = pmesh.GetNE();
      MPI_Allreduce(MPI_IN_PLACE,&ne,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      int ne_x = (dim == 2) ? (int)sqrt(ne) : (int)cbrt(ne);
      ostringstream oss;
      double pcg_time = chrono.RealTime();
      if (myid == 0)
      {
         if (dim == 2)
         {
            oss << ne_x << " x " << ne_x ;
         }
         else
         {
            oss << ne_x << " x " << ne_x << " x " << ne_x ;
         }
         // mfem::out << "Mesh: " << oss.str() << std::endl;
         // mfem::out << "PCG time = " << pcg_time << std::endl;
      }

      int num_iter = cg.GetNumIterations();

      a->RecoverFEMSolution(X,x);


      ParComplexGridFunction E(E_fes);
      E.real().MakeRef(E_fes,x.GetData());
      E.imag().MakeRef(E_fes,&x.GetData()[offsets.Last()]);

      if (myid == 0)
      {
         mfem::out << std::right << std::setw(5) << it << " | " 
                   << std::setw(16) << oss.str() << " | " 
                   << std::setw(10) <<  dofs << " | " 
                   << std::setprecision(0) << std::fixed
                   << std::setw(2) <<  2*rnum << " π  | " 
                   << std::setw(6) << std::fixed << num_iter << " | " 
                   << std::setprecision(5) 
                   << std::setw(8) << std::fixed << pcg_time << " | " 
                   << std::scientific 
                  << std::endl;
      }

      if (visualization)
      {
         char vishost[] = "localhost";
         int visport = 19916;
         socketstream E_out_r(vishost, visport);

         E_out_r << "parallel " << num_procs << " " << myid << "\n";
         E_out_r.precision(8);
         E_out_r << "solution\n" << pmesh << E.real() <<
                  "window_title 'Real Numerical Electric field' "
                  << flush;
      }

      if (it == pr-1)
         break;

      pmesh.UniformRefinement();
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


void source(const Vector &x, Vector &f)
{
   Vector center(dim);
   double r = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      center(i) = 0.5 * (comp_domain_bdr(i, 0) + comp_domain_bdr(i, 1));
      r += pow(x[i] - center[i], 2.);
   }
   double n = 5.0 * omega * sqrt(epsilon * mu) / M_PI;
   double coeff = pow(n, 2) / M_PI;
   double alpha = -pow(n, 2) * r;
   f = 0.0;
   f[0] = -coeff * exp(alpha)/omega;
}

void maxwell_solution(const Vector &x, vector<complex<double>> &E)
{
   // Initialize
   for (int i = 0; i < dim; ++i)
   {
      E[i] = 0.0;
   }
}

void E_exact_Re(const Vector &x, Vector &E)
{
   vector<complex<double>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].real();
   }
}

void E_exact_Im(const Vector &x, Vector &E)
{
   vector<complex<double>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].imag();
   }
}

void E_bdr_data_Re(const Vector &x, Vector &E)
{
   E = 0.0;
}

// Define bdr_data solution
void E_bdr_data_Im(const Vector &x, Vector &E)
{
   E = 0.0;
}

// α = |J|^-1 J^T J (in 2D it's the scalar |J|^-1)
// α^-1 = |J| J^-1 J^-T (in 2D it's the scalar |J|)
void inv_alpha_function_re(const Vector &x, CartesianPML * pml, DenseMatrix & K)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   K = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      K(i,i) = (det/pow(dxs[i], 2)).real();
   }
}

void inv_alpha_function_im(const Vector &x, CartesianPML * pml, DenseMatrix & K)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }
   K = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      K(i,i) = (det/pow(dxs[i], 2)).imag();
   }
}

void inv_abs2_alpha_function(const Vector &x, CartesianPML * pml, DenseMatrix & K)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }
   K=0.0;
   for (int i = 0; i < dim; ++i)
   {
      complex<double> a = det/pow(dxs[i], 2);
      K(i,i) = a.imag() * a.imag() + a.real() * a.real();
   }
}

double inv_scalar_alpha_function_re(const Vector &x, CartesianPML * pml)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   return det.real();
}

double inv_scalar_alpha_function_im(const Vector &x, CartesianPML * pml)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   return det.imag();
}

double inv_scalar_abs2_alpha_function(const Vector &x, CartesianPML * pml)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   return det.imag()*det.imag() + det.real()*det.real();
}

// β = |J| J^-1 J^-T
void beta_function_re(const Vector &x, CartesianPML * pml, DenseMatrix & K)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }
   K=0.0;
   for (int i = 0; i < dim; ++i)
   {
      K(i,i) = (det / pow(dxs[i], 2)).real();
   }
}

void beta_function_im(const Vector &x, CartesianPML * pml, DenseMatrix & K)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }
   K = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      K(i,i) = (det / pow(dxs[i], 2)).imag();
   }
}

void abs2_beta_function(const Vector &x, CartesianPML * pml, DenseMatrix & K)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }
   K = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      complex<double> a = det/pow(dxs[i], 2);

      K(i,i) = a.imag()*a.imag() + a.real()*a.real();
   }
}


CartesianPML::CartesianPML(Mesh *mesh_, Array2D<double> length_)
   : mesh(mesh_), length(length_)
{
   dim = mesh->Dimension();
   SetBoundaries();
}

void CartesianPML::SetBoundaries()
{
   comp_dom_bdr.SetSize(dim, 2);
   dom_bdr.SetSize(dim, 2);
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   for (int i = 0; i < dim; i++)
   {
      dom_bdr(i, 0) = pmin(i);
      dom_bdr(i, 1) = pmax(i);
      comp_dom_bdr(i, 0) = dom_bdr(i, 0) + length(i, 0);
      comp_dom_bdr(i, 1) = dom_bdr(i, 1) - length(i, 1);
   }
}

void CartesianPML::SetAttributes(ParMesh *pmesh)
{
   // Initialize bdr attributes
   for (int i = 0; i < pmesh->GetNBE(); ++i)
   {
      pmesh->GetBdrElement(i)->SetAttribute(i+1);
   }

   int nrelem = pmesh->GetNE();

   // Initialize list with 1
   elems.SetSize(nrelem);

   // Loop through the elements and identify which of them are in the PML
   for (int i = 0; i < nrelem; ++i)
   {
      elems[i] = 1;
      bool in_pml = false;
      Element *el = pmesh->GetElement(i);
      Array<int> vertices;

      // Initialize attribute
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();

      // Check if any vertex is in the PML
      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         double *coords = pmesh->GetVertex(vert_idx);
         for (int comp = 0; comp < dim; ++comp)
         {
            if (coords[comp] > comp_dom_bdr(comp, 1) ||
                coords[comp] < comp_dom_bdr(comp, 0))
            {
               in_pml = true;
               break;
            }
         }
      }
      if (in_pml)
      {
         elems[i] = 0;
         el->SetAttribute(2);
      }
   }
   pmesh->SetAttributes();
}

void CartesianPML::StretchFunction(const Vector &x,
                                   vector<complex<double>> &dxs)
{
   complex<double> zi = complex<double>(0., 1.);

   double n = 2.0;
   double c = 5.0;
   double coeff;
   double k = omega * sqrt(epsilon * mu);

   // Stretch in each direction independently
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_domain_bdr(i, 1))
      {
         coeff = n * c / k / pow(length(i, 1), n);
         dxs[i] = 1.0 + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 1), n - 1.0));
      }
      if (x(i) <= comp_domain_bdr(i, 0))
      {
         coeff = n * c / k / pow(length(i, 0), n);
         dxs[i] = 1.0 + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 0), n - 1.0));
      }
   }
}
