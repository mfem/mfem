//                                MFEM Example 3 -- modified for NURBS FE
//
// Compile with: make igaopt_wg
//
// Sample runs:  mpirun -np 1 igaopt_wg
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Static condensation is
//               also illustrated.
//
//               NURBS-based H(curl) spaces only implemented for meshes
//               consisting of a single patch.
//
//               We recommend viewing examples 1-2 before viewing this example.


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <unistd.h>
#include "IGAopt.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <nlopt.hpp>

using namespace std;
using namespace mfem;



/// @brief Inverse sigmoid function
real_t inv_sigmoid(real_t x)
{
   real_t tol = 1e-12;
   x = std::min(std::max(tol,x), real_t(1.0)-tol);
   return std::log(x/(1.0-x));
}

/// @brief Sigmoid function
real_t sigmoid(real_t x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+std::exp(-x));
   }
   else
   {
      return std::exp(x)/(1.0+std::exp(x));
   }
}

/// @brief Derivative of sigmoid function
real_t der_sigmoid(real_t x)
{
   real_t tmp = sigmoid(-x);
   return tmp - std::pow(tmp,2);
}

class EM_Grad_Coefficient : public Coefficient
{
protected:
   ParComplexGridFunction *e = nullptr;
   ParComplexGridFunction *e_adj = nullptr;
   GridFunction *rho_filter = nullptr; // filter density
   real_t exponent;
   real_t rho_min;
   real_t omegaem;

public:
   EM_Grad_Coefficient(ParComplexGridFunction *e_, ParComplexGridFunction *e_adj_, real_t omega_)
      : e(e_), e_adj(e_adj_),omegaem(omega_)
   {}

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      int dim = T.GetSpaceDim();
      real_t density = 0.0;
      Vector e_real(dim);
      Vector e_imag(dim);
      Vector e_adj_real(dim);
      Vector e_adj_imag(dim);
      e_real = 0.0;
      e_imag = 0.0;
      e_adj_real = 0.0;
      e_adj_imag = 0.0;
      e->real().GetVectorValue(T,ip,e_real);
      e->imag().GetVectorValue(T,ip,e_imag);
      e_adj->real().GetVectorValue(T,ip,e_adj_real);
      e_adj->imag().GetVectorValue(T,ip,e_adj_imag);
      for(int i=0; i<dim; i++)
      {
        density += (e_real[i]*e_adj_real[i] - e_imag[i]*e_adj_imag[i]);
      }
      density *= (2*omegaem*omegaem);

      return density;
   }
};

void Integrate3Dto2D(FiniteElementSpace * fes_3D, Coefficient &cf_3D, Array2D<real_t> &array_2D)
{
    Mesh * mesh_3D = fes_3D->GetMesh();
    array_2D = 0;
    // Integrate the scalar function
    for (int i = 0; i < mesh_3D->GetNE(); i++)  // Loop over elements
    {
      ElementTransformation *trans = fes_3D->GetMesh()->GetElementTransformation(i);
      const FiniteElement &fe = *(fes_3D->GetFE(i));
      Array<int> IJK;
      IJK.SetSize(3);
      IJK = 0;
      (mesh_3D->NURBSext)->GetElementIJK(i, IJK);
      const IntegrationRule &ir = IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder());
      double integral_value = 0.0;
      for (int j = 0; j < ir.GetNPoints(); j++)  // Loop over quadrature points
      {
           const IntegrationPoint &ip = ir.IntPoint(j);
           trans->SetIntPoint(&ip);
           // Evaluate scalar function at the quadrature point in physical coordinates
           double scalar_value = cf_3D.Eval(*trans, ip);
           // Accumulate the integral (scalar value * weight * Jacobian determinant)
           integral_value += scalar_value * ip.weight * trans->Weight();
      }
      array_2D(IJK[0],IJK[1]) += integral_value;
    }
}


class Z2Dto3DCoefficient : public Coefficient
{
protected:
    Array2D<real_t> * array_2D;
    NURBSExtension * NURBSext;
public:
    Z2Dto3DCoefficient(NURBSExtension * NURBSext_, Array2D<real_t> * array_2D_)
    :NURBSext(NURBSext_),array_2D(array_2D_)
    {}
    virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
      int el_num = T.ElementNo;
      Array<int> IJK;
      IJK.SetSize(3);
      IJK = 0;
      NURBSext->GetElementIJK(el_num,IJK);
      real_t arr = (*array_2D)(IJK[0],IJK[1]);
      //mfem::out<<arr<<" ";
      return arr;
    }
};

class Grad_Z2Dto3DCoefficient : public Coefficient
{
protected:
    Array2D<real_t> * array_2D;
    NURBSExtension * NURBSext;
public:
    Grad_Z2Dto3DCoefficient(NURBSExtension * NURBSext_, Array2D<real_t> * array_2D_)
    :NURBSext(NURBSext_),array_2D(array_2D_)
    {}
    virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
      int el_num = T.ElementNo;
      Array<int> IJK;
      IJK.SetSize(3);
      IJK = 0;
      NURBSext->GetElementIJK(el_num,IJK);
      real_t arr = (*array_2D)(IJK[0],IJK[1]);
      return arr;
    }
};
/// @brief Solid isotropic material penalization (SIMP) coefficient
class SIMPInterpolationCoefficient : public Coefficient
{
protected:
   Coefficient *rho_filter;
   real_t min_val;
   real_t max_val;
   real_t exponent;

public:
   SIMPInterpolationCoefficient(Coefficient *rho_filter_, real_t min_val_= 1e-6,
                                real_t max_val_ = 1.0, real_t exponent_ = 1)
      : rho_filter(rho_filter_), min_val(min_val_), max_val(max_val_),
        exponent(exponent_) { }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t val = rho_filter->Eval(T, ip);
      if(val>1)
      {
         val = 1;
      }
      else if(val < 0)
      {
         val = 0;
      }
      real_t coeff = min_val + val*(max_val-min_val);
      //mfem::out<<coeff<<" ";
      return coeff;
   }
};

// Class for setting up a simple Cartesian PML region
class PML
{
private:
   Mesh *mesh;

   int dim;

   real_t omega;

   // Length of the PML Region in each direction
   Array2D<real_t> length;

   // Computational Domain Boundary
   Array2D<real_t> comp_dom_bdr;

   // Domain Boundary
   Array2D<real_t> dom_bdr;

   // Integer Array identifying elements in the PML
   // 0: in the PML, 1: not in the PML
   Array<int> elems;

   // Compute Domain and Computational Domain Boundaries
   void SetBoundaries();

public:
   // Constructor
   PML(Mesh *mesh_,Array2D<real_t> length_, real_t omega_);

   // Return Computational Domain Boundary
   Array2D<real_t> GetCompDomainBdr() {return comp_dom_bdr;}

   // Return Domain Boundary
   Array2D<real_t> GetDomainBdr() {return dom_bdr;}

   // Return Markers list for elements
   Array<int> * GetMarkedPMLElements() {return &elems;}

   // Mark elements in the PML region
   void SetAttributes(ParMesh *mesh_);

   // PML complex stretching function
   void StretchFunction(const Vector &x, vector<complex<real_t>> &dxs, real_t eps);
};

// Class for returning the PML coefficients of the bilinear form
class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   PML * pml = nullptr;
   PWConstCoefficient epsc;
   void (*Function)(const Vector &, PML *, Vector &, real_t);
public:
   PMLDiagMatrixCoefficient(int dim, void(*F)(const Vector &, PML *,
                                              Vector &, real_t),
                            PML * pml_,
                            PWConstCoefficient epsc)
      : VectorCoefficient(dim), pml(pml_), Function(F), epsc(epsc)
   {}

   using VectorCoefficient::Eval;

   virtual void Eval(Vector &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   { 
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(vdim);
      real_t eps = epsc.Eval(T,ip);
      //mfem::out << "MAIN.cpp row 108: " << eps <<std::endl;
      (*Function)(transip, pml, K, eps);
   }
};

template <typename T> T pow2(const T &x) { return x*x; }

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector & D, real_t eps);
void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector & D, real_t eps);
void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector & D, real_t eps);

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector & D, real_t eps);
void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector & D, real_t eps);
void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector & D, real_t eps);

Array2D<double> comp_domain_bdr;
Array2D<double> domain_bdr;

class DiffusionSolver
{
private:
   Mesh * mesh = nullptr;
   int order = 1;
   // diffusion coefficient
   Coefficient * diffcf = nullptr;
   // mass coefficient
   Coefficient * masscf = nullptr;
   Coefficient * rhscf = nullptr;
   Coefficient * essbdr_cf = nullptr;
   Coefficient * neumann_cf = nullptr;
   VectorCoefficient * gradient_cf = nullptr;

   // FEM solver
   int dim;
   FiniteElementCollection * fec = nullptr;
   FiniteElementSpace * fes = nullptr;
   Array<int> ess_bdr;
   Array<int> neumann_bdr;
   GridFunction * u = nullptr;
   LinearForm * b = nullptr;
   bool parallel;
#ifdef MFEM_USE_MPI
   ParMesh * pmesh = nullptr;
   ParFiniteElementSpace * pfes = nullptr;
#endif

public:
   DiffusionSolver() { }
   DiffusionSolver(Mesh * mesh_, int order_, Coefficient * diffcf_,
                   Coefficient * cf_);

   void SetMesh(Mesh * mesh_)
   {
      mesh = mesh_;
      parallel = false;
#ifdef MFEM_USE_MPI
      pmesh = dynamic_cast<ParMesh *>(mesh);
      if (pmesh) { parallel = true; }
#endif
   }
   void SetOrder(int order_) { order = order_ ; }
   void SetDiffusionCoefficient(Coefficient * diffcf_) { diffcf = diffcf_; }
   void SetMassCoefficient(Coefficient * masscf_) { masscf = masscf_; }
   void SetRHSCoefficient(Coefficient * rhscf_) { rhscf = rhscf_; }
   void SetEssentialBoundary(const Array<int> & ess_bdr_) { ess_bdr = ess_bdr_;};
   void SetNeumannBoundary(const Array<int> & neumann_bdr_) { neumann_bdr = neumann_bdr_;};
   void SetNeumannData(Coefficient * neumann_cf_) {neumann_cf = neumann_cf_;}
   void SetEssBdrData(Coefficient * essbdr_cf_) {essbdr_cf = essbdr_cf_;}
   void SetGradientData(VectorCoefficient * gradient_cf_) {gradient_cf = gradient_cf_;}

   void ResetFEM();
   void SetupFEM();

   void Solve();
   GridFunction * GetFEMSolution();
   LinearForm * GetLinearForm() {return b;}
#ifdef MFEM_USE_MPI
   ParGridFunction * GetParFEMSolution();
   ParLinearForm * GetParLinearForm()
   {
      if (parallel)
      {
         return dynamic_cast<ParLinearForm *>(b);
      }
      else
      {
         MFEM_ABORT("Wrong code path. Call GetLinearForm");
         return nullptr;
      }
   }
#endif

   ~DiffusionSolver();

};

// Class for solving maxwell equations in Hcurl NURBS space:
class NURBSEMSolver
{
private:
   Mesh * mesh = nullptr;
   PML * pml = nullptr;
   Coefficient * design_epsilon = nullptr;
   int order;
   real_t freq;
   int dim;
   int cdim;
   real_t omega;
   FiniteElementCollection * fec = nullptr;
   NURBSExtension *NURBSext = nullptr;
   ParComplexGridFunction * J = nullptr;
   ParComplexGridFunction * x = nullptr;
   VectorCoefficient * b_r = nullptr;
   VectorCoefficient * b_i = nullptr;
   bool parallel;
   ParMesh * pmesh = nullptr;
   ParFiniteElementSpace * pfes = nullptr;
   //ParGridFunction structure_pm;
   Array<int> attr;
   Array<int> attrPML;
   Array<int> attrDesign;
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   ComplexOperator::Convention conv;
   bool adjoint = false;

public:
   NURBSEMSolver() { }
   void SetMesh(Mesh * mesh_)
   {
      mesh = mesh_;
      parallel = false;
      pmesh = dynamic_cast<ParMesh *>(mesh);
      if (pmesh) { parallel = true; }
      if(!parallel)
      {
        MFEM_ABORT("MUST PARALLEL MESH");
      }
   }
   void SetAdjoint(bool adjoint_){adjoint = adjoint_; }
   void SetOrder(int order_) { order = order_ ; }
   void SetFrequency(real_t freq_) {freq = freq_; }
   void SetPML(PML * pml_) {pml = pml_; }
   void SetepsilonCoefficients(Coefficient * design_epsilon_) { design_epsilon = design_epsilon_;}
   void SetRHSCoefficient(VectorCoefficient * b_r_, VectorCoefficient * b_i_) {b_r = b_r_; b_i = b_i_;}
   void SetupFEM();
   void Solve();
   ParComplexGridFunction * GetParFEMSolution();
   ParComplexLinearForm * GetParLinearForm()
   {
      if (parallel)
      {
         return dynamic_cast<ParComplexLinearForm *>(J);
      }
      else
      {
         MFEM_ABORT("Wrong code path. Call GetLinearForm");
         return nullptr;
      }
   }
   ~NURBSEMSolver();
};

int opt_num;
double mu = 1.0;
int el_num_x;
int el_num_y;
//double epsilon = 1.0;
double omega_1;
double omega_2;
int dim;
bool exact_known = false;
Mesh * mesh = nullptr;
ParMesh * pmesh = nullptr;
FiniteElementCollection * fec = nullptr;
NURBSExtension * NURBSext = nullptr;
NURBSEMSolver * EMsolver_1 = nullptr;
NURBSEMSolver * adjoint_EMsolver_1 = nullptr;
NURBSEMSolver * EMsolver_2 = nullptr;
NURBSEMSolver * adjoint_EMsolver_2 = nullptr;
//DiffusionSolver * FilterSolver = nullptr;
VectorGridFunctionCoefficient * adju_b_r = nullptr;
VectorGridFunctionCoefficient * adju_b_i = nullptr;
VectorGridFunctionCoefficient * adjd_b_r = nullptr;
VectorGridFunctionCoefficient * adjd_b_i = nullptr;
VectorGridFunctionCoefficient * b1_r_cf = nullptr;
VectorGridFunctionCoefficient * b1_i_cf = nullptr;
VectorGridFunctionCoefficient * b2_r_cf = nullptr;
VectorGridFunctionCoefficient * b2_i_cf = nullptr;
ParComplexGridFunction * ue_1 = nullptr;
ParComplexGridFunction * ue_2 = nullptr;
ParComplexGridFunction * u_add = nullptr;
ParComplexGridFunction * u_adu = nullptr;
ParFiniteElementSpace *fespace = nullptr;
ConstantCoefficient one(1.0);
double objfunc(unsigned n,  const double *x, double *grad, void *data);
int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   opt_num = 0;
   // Parse command-line options.
   const char *mesh_file = "./meshes/cubes-nurbs.mesh";
   int order = 1;
   const char *device_config = "cuda";
   double freq_1 = 1.0/2000;
   omega_1 = real_t(2.0 * M_PI) * freq_1;
   double freq_2 = 1.0/1600;
   omega_2 = real_t(2.0 * M_PI) * freq_2;

   Device device(device_config);
   device.Print();

   mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // Setup PML length
   Array2D<real_t> length(dim, 2); length(0,0) = 500; length(0,1) = 500;
   length(1,0) = 400; length(1,1) = 400;
   length(2,0) = 400; length(2,1) = 400;
   PML * pml_1 = new PML(mesh,length,omega_1);
   comp_domain_bdr = pml_1->GetCompDomainBdr();
   domain_bdr = pml_1->GetDomainBdr();
   PML * pml_2 = new PML(mesh,length,omega_2);

   // Refine the mesh to increase the resolution.
   for (int l = 0; l < 4; l++)
   {
      mesh->UniformRefinement();
   }
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   pml_1->SetAttributes(pmesh);
   pml_2->SetAttributes(pmesh);
   int nelem = pmesh->GetNE();
   for (int i = 0; i < nelem; ++i)
   {
      bool is_waveguide = false;
      Element *el = pmesh->GetElement(i);
      Array<int> vertices;
      el->GetVertices(vertices);
      int nrvert = vertices.Size();

      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         real_t *coords = pmesh->GetVertex(vert_idx);    
         if (((coords[1]<3400)&&(coords[1]>2800)&&(coords[2]<1100)&&(coords[2]>900)&&(coords[0]>6500))
            ||((coords[1]<(3400+1000))&&(coords[1]>(2800+1000))&&(coords[2]<1200)&&(coords[2]>800)&&(coords[0]<1500))
            ||((coords[1]<(3400-1000))&&(coords[1]>(2800-1000))&&(coords[2]<1200)&&(coords[2]>800)&&(coords[0]<1500)))
         {
            is_waveguide = true;
            break;
         }
      }
      if (is_waveguide && (el->GetAttribute() == 1))
      {
         el->SetAttribute(3);
      }
      else if (is_waveguide && (el->GetAttribute() == 2))
      {
         el->SetAttribute(4);
      }
   }
   pmesh->SetAttributes();
   int num_design_el = 0;
   for (int i = 0; i < nelem; ++i)
   {
      bool is_design_domain = false;
      Element *el = pmesh->GetElement(i);
      Array<int> vertices;
      el->GetVertices(vertices);
      int nrvert = vertices.Size();

      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         real_t *coords = pmesh->GetVertex(vert_idx);    
         if ((coords[0]<6500)&&(coords[0]>1500)&&(coords[2]<1200)&&(coords[2]>800)&&(coords[1]>800)&&(coords[1]<5200))
         {
            is_design_domain = true;
            break;
         }
      }
      if (is_design_domain)
      {
         el->SetAttribute(5);
         num_design_el++;
      }
   }
   pmesh->SetAttributes();
   std::ifstream file_J_r("./J_r.gf");
   std::ifstream file_J_i("./J_i.gf");
   std::ifstream file_AUJ_r("./aduJ_r.gf");
   std::ifstream file_AUJ_i("./aduJ_i.gf");
   std::ifstream file_ADJ_r("./addJ_r.gf");
   std::ifstream file_ADJ_i("./addJ_i.gf");
   GridFunction J_r(pmesh,file_J_r);
   GridFunction J_i(pmesh,file_J_i);
   GridFunction AUJ_r(pmesh,file_AUJ_r);
   GridFunction AUJ_i(pmesh,file_AUJ_i);
   GridFunction ADJ_r(pmesh,file_ADJ_r);
   GridFunction ADJ_i(pmesh,file_ADJ_i);

   b1_r_cf = new VectorGridFunctionCoefficient(&J_r);
   b1_i_cf = new VectorGridFunctionCoefficient(&J_i);
   b2_r_cf = new VectorGridFunctionCoefficient(&J_r);
   b2_i_cf = new VectorGridFunctionCoefficient(&J_i);

   fec = new NURBS_HCurlFECollection(order,dim);
   NURBSext  = new NURBSExtension(pmesh->NURBSext, order);

   fespace = new ParFiniteElementSpace(pmesh, NURBSext, fec);
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ue_1 = new ParComplexGridFunction(fespace);
   u_add = new ParComplexGridFunction(fespace);
   u_adu = new ParComplexGridFunction(fespace);
   *ue_1 = 0.0;
   *u_add = 0.0;
   *u_adu = 0.0;   

   Array<const KnotVector*> kv(dim);
   int p = 0;
   NURBSext->GetPatchKnotVectors(p, kv);
   el_num_x = kv[0]->GetNE();
   el_num_y = kv[1]->GetNE();

   EMsolver_1 = new NURBSEMSolver();
   EMsolver_1->SetMesh(pmesh);
   EMsolver_1->SetFrequency(freq_1);
   EMsolver_1->SetOrder(order);
   EMsolver_1->SetPML(pml_1);

   adjoint_EMsolver_1 = new NURBSEMSolver();
   adjoint_EMsolver_1->SetMesh(pmesh);
   adjoint_EMsolver_1->SetFrequency(freq_1);
   adjoint_EMsolver_1->SetOrder(order);
   adjoint_EMsolver_1->SetPML(pml_1);

   EMsolver_2 = new NURBSEMSolver();
   EMsolver_2->SetMesh(pmesh);
   EMsolver_2->SetFrequency(freq_2);
   EMsolver_2->SetOrder(order);
   EMsolver_2->SetPML(pml_2);

   adjoint_EMsolver_2 = new NURBSEMSolver();
   adjoint_EMsolver_2->SetMesh(pmesh);
   adjoint_EMsolver_2->SetFrequency(freq_2);
   adjoint_EMsolver_2->SetOrder(order);
   adjoint_EMsolver_2->SetPML(pml_2);

   H1_FECollection filter_fec(2, dim); // space for ρ̃
   ParFiniteElementSpace filter_fes(pmesh, &filter_fec);

   Array<int> ess_bdr_filter;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr_filter.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr_filter = 0;
   }

   adju_b_r = new VectorGridFunctionCoefficient(&AUJ_r);
   adju_b_i = new VectorGridFunctionCoefficient(&AUJ_i);
   adjd_b_r = new VectorGridFunctionCoefficient(&ADJ_r);
   adjd_b_i = new VectorGridFunctionCoefficient(&ADJ_i);
   int num_var = el_num_x*el_num_y;
   mfem::out<<" "<<el_num_x<<" "<<el_num_y<<std::endl;
   //nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, num_var);
   double lb[num_var];
   double ub[num_var];
   for(int i=0; i<num_var; i++)
   {
      lb[i]=0;
   }
   for(int i=0; i<num_var; i++)
   {
      ub[i]=1;
   }
   // nlopt_set_lower_bounds(opt,lb);
   // nlopt_set_upper_bounds(opt,ub);
   // nlopt_set_min_objective(opt,objfunc, NULL);

   // nlopt_set_xtol_rel(opt,1e-6);

   double varx[num_var];
   for(int i=0; i<num_var; i++)
   {
      varx[i] = 0.8;
   }

   double grad[num_var];
   for(int i=0; i<num_var; i++)
   {
      grad[i] = 0;
   }
   double minf = 100000000.0;
   // nlopt_result result = nlopt_optimize(opt, varx, &minf);

   real_t alpha = 1.0;
   for(int iter = 0; iter < 10000; iter++)
   {  
      if (iter > 1) { alpha *= ((real_t) iter) / ((real_t) iter-1); }
      minf = objfunc(num_var, varx, grad, NULL);
      for(int i = 0; i<num_var; i++)
      {
         varx[i] = varx[i] - alpha*grad[i];
         if(varx[i]>ub[i])
         {
            varx[i] = ub[i];
         }
         if(varx[i]<lb[i])
         {
            varx[i] = lb[i];
         }
      }
      mfem::out<<minf<<std::endl;
   }

   for(int i=0; i<num_var; i++)
   {
      mfem::out<<varx[i]<<" ";
   }
   mfem::out<<std::endl;
   mfem::out<<"ID "<<myid<<" "<<getpid()<<std::endl;
   int k = 1;
   while(k)
   {
        sleep(5);
   }

   delete mesh;
   delete pml_1;
   delete pml_2;
   delete pmesh;
   delete fec;
   delete fespace;
   delete EMsolver_1;
   delete adjoint_EMsolver_1;
   delete EMsolver_2;
   delete adjoint_EMsolver_2;
   //delete FilterSolver;
   if(myid == 0)
   {
      mfem::out<<"****** FINISH ******"<<std::endl;
   }
   return 0;
}

void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector & D, real_t eps)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs, eps);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(dxs[i], 2)).real();
   }
   
}

void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector & D, real_t eps)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, eps);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(dxs[i], 2)).imag();
   }
}

void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector & D, real_t eps)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, eps);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = abs(det / pow(dxs[i], 2));
   }
}

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector & D, real_t eps)
{  
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs, eps);
   
   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }
   // in the 2D case the coefficient is scalar 1/det(J)
   if (dim == 2)
   {  
      D = (1.0 / det).real();
   }
   else
   {  
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow(dxs[i], 2) / det).real();
      }
   }
}

void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector & D, real_t eps)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, eps);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = (1.0 / det).imag();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow(dxs[i], 2) / det).imag();
      }
   }
}

void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector & D, real_t eps)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, eps);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = abs(1.0 / det);
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = abs(pow(dxs[i], 2) / det);
      }
   }
}

PML::PML(Mesh *mesh_, Array2D<real_t> length_, real_t omega_)
   : mesh(mesh_), length(length_), omega(omega_)
{
   dim = mesh->Dimension();
   SetBoundaries();
}

void PML::SetBoundaries()
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

void PML::SetAttributes(ParMesh *mesh_)
{
   // Initialize bdr attributes
   for (int i = 0; i < mesh_->GetNBE(); ++i)
   {
      mesh_->GetBdrElement(i)->SetAttribute(i+1);
   }

   int nrelem = mesh_->GetNE();

   elems.SetSize(nrelem);

   // Loop through the elements and identify which of them are in the PML
   for (int i = 0; i < nrelem; ++i)
   {
      elems[i] = 1;
      bool in_pml = false;
      Element *el = mesh_->GetElement(i);
      Array<int> vertices;

      // Initialize attribute
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();

      // Check if any vertex is in the PML
      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         real_t *coords = mesh_->GetVertex(vert_idx);
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
         el->SetAttribute(1);
      }
      else if(!in_pml)
      {
         elems[i] = 1;
         el->SetAttribute(2);
      }
   }
   mesh_->SetAttributes();
}

void PML::StretchFunction(const Vector &x,
                          vector<complex<real_t>> &dxs, real_t eps)
{
   constexpr complex<real_t> zi = complex<real_t>(0., 1.);

   real_t n = 2.0;
   real_t c = 5.0;
   real_t coeff;
   real_t k = omega * sqrt(eps * mu);

   // Stretch in each direction independently
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_domain_bdr(i, 1))
      {
         coeff = n * c / k / pow(length(i, 1), n);
         dxs[i] = 1_r + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 1), n - 1_r));
      }
      if (x(i) <= comp_domain_bdr(i, 0))
      {
         coeff = n * c / k / pow(length(i, 0), n);
         dxs[i] = 1_r + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 0), n - 1_r));
      }
   }
}

void NURBSEMSolver::SetupFEM()
{
   if (!parallel)
   {
    MFEM_ABORT("must parallel");
   }
   dim = mesh->Dimension();
   cdim = (dim == 2) ? 1 : dim;
   omega = real_t(2.0 * M_PI) * freq;
   conv = ComplexOperator::HERMITIAN;
   fec = new NURBS_HCurlFECollection(order,dim);
   NURBSext  = new NURBSExtension(pmesh->NURBSext, order);                                                                           
   pfes = new ParFiniteElementSpace(pmesh, NURBSext, fec);
   ess_bdr.SetSize(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   pfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list); 
   delete x;
   x = new ParComplexGridFunction(pfes);
   *x=0.0;
   if(pmesh->attributes.Max() != 5)
   {
    MFEM_ABORT("number of pmesh attributes must be 5");
   }
   attr.SetSize(pmesh->attributes.Max());
   attrPML.SetSize(pmesh->attributes.Max());
   attrDesign.SetSize(pmesh->attributes.Max());
   attr = 0;
   attr[1] = 1;  attr[3] = 1;  attr[4] = 1;
   attrPML = 0;
   attrPML[0] = 1; attrPML[2] = 1;
   attrDesign = 0;
   attrDesign[4] = 1;
}

void NURBSEMSolver::Solve()
{
   real_t wg_eps = 4.0; 
   Vector k2epsilon(pmesh->attributes.Max());
   k2epsilon = -pow2(omega);
   k2epsilon(2) = k2epsilon(0)*wg_eps;
   k2epsilon(3) = k2epsilon(0)*wg_eps;
   ConstantCoefficient muinv(1_r / mu);
   Array<Coefficient*> k2_eps;
   k2_eps.SetSize(pmesh->attributes.Max());
   k2_eps[0] = new ConstantCoefficient(k2epsilon(0));
   k2_eps[1] = new ConstantCoefficient(k2epsilon(1));
   k2_eps[2] = new ConstantCoefficient(k2epsilon(2));
   k2_eps[3] = new ConstantCoefficient(k2epsilon(3));
   k2_eps[4] = new ProductCoefficient(k2epsilon(4),*design_epsilon);
   Array<int> k2_eps_attr;
   k2_eps_attr.SetSize(pmesh->attributes.Max());
   k2_eps_attr[0] = 1; 
   k2_eps_attr[1] = 2; 
   k2_eps_attr[2] = 3; 
   k2_eps_attr[3] = 4; 
   k2_eps_attr[4] = 5; 
   PWCoefficient k2eps(k2_eps_attr,k2_eps);
   RestrictedCoefficient restr_muinv(muinv,attr);
   RestrictedCoefficient restr_omeg(k2eps,attr);

   // Integrators inside the computational domain (excluding the PML region)
   ParSesquilinearForm a(pfes, conv);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   NURBSCurlCurlIntegrator *di_NURBSCurlCurlIntegrator = new NURBSCurlCurlIntegrator(restr_muinv);
   NURBSHCurl_VectorMassIntegrator *di_NURBSVectorMassIntegrator = new NURBSHCurl_VectorMassIntegrator(restr_omeg);

   a.AddDomainIntegrator(di_NURBSCurlCurlIntegrator,NULL);
   a.AddDomainIntegrator(di_NURBSVectorMassIntegrator,NULL);

   //Integrators inside the pml
   Vector pml_eps(pmesh->attributes.Max());
   pml_eps = 1;
   pml_eps(2) = pml_eps(0)*wg_eps;
   PWConstCoefficient pmleps(pml_eps);

   Vector pml_k2_eps(pmesh->attributes.Max());
   pml_k2_eps = -pow2(omega);
   pml_k2_eps(2) = pml_k2_eps(0)*wg_eps;
   PWConstCoefficient pml_k2eps(pml_k2_eps);

   PMLDiagMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, pml, pmleps);
   PMLDiagMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, pml, pmleps);
   ScalarVectorProductCoefficient c1_Re(muinv,pml_c1_Re);
   ScalarVectorProductCoefficient c1_Im(muinv,pml_c1_Im);
   VectorRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
   VectorRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

   PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re, pml, pmleps);
   PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im, pml, pmleps);
   ScalarVectorProductCoefficient c2_Re(pml_k2eps,pml_c2_Re);
   ScalarVectorProductCoefficient c2_Im(pml_k2eps,pml_c2_Im);
   VectorRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
   VectorRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

   NURBSCurlCurlIntegrator *di_NURBSCurlCurlIntegrator_Re = new NURBSCurlCurlIntegrator(restr_c1_Re);
   NURBSCurlCurlIntegrator *di_NURBSCurlCurlIntegrator_Im = new NURBSCurlCurlIntegrator(restr_c1_Im);
   NURBSHCurl_VectorMassIntegrator *di_NURBSVectorMassIntegrator_Re = new NURBSHCurl_VectorMassIntegrator(restr_c2_Re);
   NURBSHCurl_VectorMassIntegrator *di_NURBSVectorMassIntegrator_Im = new NURBSHCurl_VectorMassIntegrator(restr_c2_Im);

   // Integrators inside the PML region
   a.AddDomainIntegrator(di_NURBSCurlCurlIntegrator_Re,
                       di_NURBSCurlCurlIntegrator_Im);
   a.AddDomainIntegrator(di_NURBSVectorMassIntegrator_Re,
                       di_NURBSVectorMassIntegrator_Im);

   OperatorPtr A;
   Vector B, X;
   a.Assemble(0);
   ParComplexLinearForm b(pfes, conv);
   b.Vector::operator=(0.0);

   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*b_r),new VectorFEDomainLFIntegrator(*b_i));

   b.Assemble();
   a.FormLinearSystem(ess_tdof_list, *x, b, A, X, B);

   Vector  prec_k2eps(pmesh->attributes.Max());
   prec_k2eps = pow2(omega);
   prec_k2eps(3) = prec_k2eps(0)*wg_eps;
   prec_k2eps(4) = prec_k2eps(0)*wg_eps;

   Array<Coefficient*> prec_k2_eps;
   prec_k2_eps.SetSize(pmesh->attributes.Max());
   prec_k2_eps[0] = new ConstantCoefficient(prec_k2eps(0));
   prec_k2_eps[1] = new ConstantCoefficient(prec_k2eps(1));
   prec_k2_eps[2] = new ConstantCoefficient(prec_k2eps(2));
   prec_k2_eps[3] = new ConstantCoefficient(prec_k2eps(3));
   prec_k2_eps[4] = new ProductCoefficient(prec_k2eps(4),*design_epsilon);
   Array<int> prec_k2_eps_attr;
   prec_k2_eps_attr.SetSize(pmesh->attributes.Max());
   prec_k2_eps_attr[0] = 1; 
   prec_k2_eps_attr[1] = 2; 
   prec_k2_eps_attr[2] = 3; 
   prec_k2_eps_attr[3] = 4; 
   prec_k2_eps_attr[4] = 5; 
   PWCoefficient prec_k2epsilon(prec_k2_eps_attr,prec_k2_eps);
   RestrictedCoefficient restr_absomeg(prec_k2epsilon,attr);

   ParBilinearForm prec(pfes);
   prec.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   NURBSCurlCurlIntegrator *di_NURBSCurlCurlIntegrator_restr_muinv = new NURBSCurlCurlIntegrator(restr_muinv);
   NURBSHCurl_VectorMassIntegrator *di_NURBSVectorMassIntegrator_restr_absomeg = new NURBSHCurl_VectorMassIntegrator(restr_absomeg);

   PMLDiagMatrixCoefficient pml_c1_abs(cdim,detJ_inv_JT_J_abs, pml, pmleps);
   ScalarVectorProductCoefficient c1_abs(muinv,pml_c1_abs);
   VectorRestrictedCoefficient restr_c1_abs(c1_abs,attrPML);
   
   PMLDiagMatrixCoefficient pml_c2_abs(dim, detJ_JT_J_inv_abs, pml, pmleps);
   ScalarVectorProductCoefficient c2_abs(prec_k2epsilon,pml_c2_abs);
   VectorRestrictedCoefficient restr_c2_abs(c2_abs,attrPML);

   NURBSCurlCurlIntegrator *di_NURBSCurlCurlIntegrator_restr_c1_abs = new NURBSCurlCurlIntegrator(restr_c1_abs);
   NURBSHCurl_VectorMassIntegrator *di_NURBSVectorMassIntegrator_restr_c2_abs = new NURBSHCurl_VectorMassIntegrator(restr_c2_abs);
   prec.AddDomainIntegrator(di_NURBSVectorMassIntegrator_restr_absomeg);
   prec.AddDomainIntegrator(di_NURBSVectorMassIntegrator_restr_c2_abs);
   prec.AddDomainIntegrator(di_NURBSCurlCurlIntegrator_restr_c1_abs);
   prec.AddDomainIntegrator(di_NURBSCurlCurlIntegrator_restr_muinv);
   prec.Assemble();

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = pfes->GetTrueVSize();
   offsets[2] = pfes->GetTrueVSize();
   offsets.PartialSum();
   std::unique_ptr<Operator> pc_r;
   std::unique_ptr<Operator> pc_i;
   real_t s = (conv == ComplexOperator::HERMITIAN) ? -1_r : 1_r;
   
   pc_r.reset(new OperatorJacobiSmoother(prec, ess_tdof_list));
   pc_i.reset(new ScaledOperator(pc_r.get(), s));
   
   BlockDiagonalPreconditioner BlockDP(offsets);
   BlockDP.SetDiagonalBlock(0, pc_r.get());
   BlockDP.SetDiagonalBlock(1, pc_i.get());

   GMRESSolver gmres(MPI_COMM_WORLD);

   gmres.SetPrintLevel(3);

   gmres.SetKDim(200);
   gmres.SetMaxIter(200000);
   if(adjoint)
   {
    gmres.SetRelTol(1e-3);
   }
   else{
    gmres.SetRelTol(1e-3);
   }
   gmres.SetAbsTol(0.0);
   gmres.SetOperator(*A);
   gmres.SetPreconditioner(BlockDP);
   gmres.Mult(B, X);
   a.RecoverFEMSolution(X, b, *x);
}

ParComplexGridFunction * NURBSEMSolver::GetParFEMSolution()
{
   if (parallel)
   {
      return x;
   }
   else
   {
      MFEM_ABORT("Wrong code path. Call GetFEMSolution");
      return nullptr;
   }
}

NURBSEMSolver::~NURBSEMSolver()
{
   delete x; x = nullptr;
   delete pfes; pfes = nullptr;
   delete fec; fec = nullptr;
   delete J; J = nullptr;
}

DiffusionSolver::DiffusionSolver(Mesh * mesh_, int order_,
                                 Coefficient * diffcf_, Coefficient * rhscf_)
   : mesh(mesh_), order(order_), diffcf(diffcf_), rhscf(rhscf_)
{

#ifdef MFEM_USE_MPI
   pmesh = dynamic_cast<ParMesh *>(mesh);
   if (pmesh) { parallel = true; }
#endif

   SetupFEM();
}

void DiffusionSolver::SetupFEM()
{
   dim = mesh->Dimension();
   fec = new H1_FECollection(order, dim);

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      pfes = new ParFiniteElementSpace(pmesh, fec);
      u = new ParGridFunction(pfes);
      b = new ParLinearForm(pfes);
   }
   else
   {
      fes = new FiniteElementSpace(mesh, fec);
      u = new GridFunction(fes);
      b = new LinearForm(fes);
   }
#else
   fes = new FiniteElementSpace(mesh, fec);
   u = new GridFunction(fes);
   b = new LinearForm(fes);
#endif
   *u=0.0;

   if (!ess_bdr.Size())
   {
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
      }
   }
}

void DiffusionSolver::Solve()
{
   OperatorPtr A;
   Vector B, X;
   Array<int> ess_tdof_list;

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      pfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
   else
   {
      fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
   }
#else
   fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
#endif
   *u=0.0;
   if (b)
   {
      delete b;
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         b = new ParLinearForm(pfes);
      }
      else
      {
         b = new LinearForm(fes);
      }
#else
      b = new LinearForm(fes);
#endif
   }
   if (rhscf)
   {
      b->AddDomainIntegrator(new DomainLFIntegrator(*rhscf));
   }
   if (neumann_cf)
   {
      MFEM_VERIFY(neumann_bdr.Size(), "neumann_bdr attributes not provided");
      b->AddBoundaryIntegrator(new BoundaryLFIntegrator(*neumann_cf),neumann_bdr);
   }
   else if (gradient_cf)
   {
      MFEM_VERIFY(neumann_bdr.Size(), "neumann_bdr attributes not provided");
      b->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*gradient_cf),
                               neumann_bdr);
   }

   b->Assemble();

   BilinearForm * a = nullptr;

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      a = new ParBilinearForm(pfes);
   }
   else
   {
      a = new BilinearForm(fes);
   }
#else
   a = new BilinearForm(fes);
#endif
   a->AddDomainIntegrator(new DiffusionIntegrator(*diffcf));
   if (masscf)
   {
      a->AddDomainIntegrator(new MassIntegrator(*masscf));
   }
   a->Assemble();
   if (essbdr_cf)
   {
      u->ProjectBdrCoefficient(*essbdr_cf,ess_bdr);
   }
   a->FormLinearSystem(ess_tdof_list, *u, *b, A, X, B);

   CGSolver * cg = nullptr;
   Solver * M = nullptr;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      M = new HypreBoomerAMG;
      dynamic_cast<HypreBoomerAMG*>(M)->SetPrintLevel(0);
      cg = new CGSolver(pmesh->GetComm());
   }
   else
   {
      M = new GSSmoother((SparseMatrix&)(*A));
      cg = new CGSolver;
   }
#else
   M = new GSSmoother((SparseMatrix&)(*A));
   cg = new CGSolver;
#endif
   cg->SetRelTol(1e-12);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->Mult(B, X);
   delete M;
   delete cg;
   a->RecoverFEMSolution(X, *b, *u);
   delete a;
}

GridFunction * DiffusionSolver::GetFEMSolution()
{
   return u;
}

#ifdef MFEM_USE_MPI
ParGridFunction * DiffusionSolver::GetParFEMSolution()
{
   if (parallel)
   {
      return dynamic_cast<ParGridFunction*>(u);
   }
   else
   {
      MFEM_ABORT("Wrong code path. Call GetFEMSolution");
      return nullptr;
   }
}
#endif

DiffusionSolver::~DiffusionSolver()
{
   delete u; u = nullptr;
   delete fes; fes = nullptr;
#ifdef MFEM_USE_MPI
   delete pfes; pfes=nullptr;
#endif
   delete fec; fec = nullptr;
   delete b;
}

double objfunc(unsigned n,  const double *x, double *grad, void *data)
{
   mfem::out<<"omega_1 = real_t(2.0 * M_PI) * freq_1: 1/"<<2*M_PI/omega_1<<" : "<<omega_1<<std::endl;
   mfem::out<<"omega_2 = real_t(2.0 * M_PI) * freq_2: 1/"<<2*M_PI/omega_2<<" : "<<omega_2<<std::endl;
   Array2D<real_t> psi_array_2D(el_num_x,el_num_y);
   Array2D<real_t> grad_array_2D_1(el_num_x,el_num_y);
   Array2D<real_t> grad_array_2D_2(el_num_x,el_num_y);
   real_t a1 = 1.0;
   real_t b1 = 1.0;
   real_t a2 = 1.0;
   real_t b2 = 1.0;
   mfem::out<<"set a1:"<<std::endl;
   cin>>a1;
   mfem::out<<"set b1:"<<std::endl;
   cin>>b1;
   mfem::out<<"set a2:"<<std::endl;
   cin>>a2;
   mfem::out<<"set b2:"<<std::endl;
   cin>>b2;
   mfem::out<<"computing"<<std::endl;


   real_t compoment = 2.0;
   grad_array_2D_1 = 0.0;
   grad_array_2D_2 = 0.0;
   for(int i=0; i<el_num_x; i++)
   {
      for(int j=0; j<el_num_y; j++)
      {
         psi_array_2D(i,j) = pow(x[i*el_num_y+j],compoment);
      }
   }
   Z2Dto3DCoefficient eps_cf(NURBSext,&psi_array_2D);
   H1_FECollection filter_fec(2, dim); // space for ρ̃
   L2_FECollection control_fec(1, dim, BasisType::GaussLobatto); // space for ψ
   ParFiniteElementSpace filter_fes(pmesh, &filter_fec);
   ParFiniteElementSpace control_fes(pmesh, &control_fec);
   GridFunction one_control(&control_fes);
   one_control = 1.0;

   Array<int> is_design_arr;
   is_design_arr.SetSize(pmesh->attributes.Max());
   is_design_arr = 0;
   is_design_arr[4] = 1;
   RestrictedCoefficient restr_PW(one,is_design_arr);

   real_t eps_materials = 1.0;
   ConstantCoefficient eps_initial(eps_materials);
   real_t epsilon_min = 1;
   real_t epsilon_max = 4;
   SIMPInterpolationCoefficient SIMP_cf(&eps_cf,epsilon_min, epsilon_max, compoment);
   ProductCoefficient eps_design(eps_initial,SIMP_cf);

   const IntegrationRule *ir3d = &IntRules.Get(Geometry::CUBE, 5);
   QuadratureSpace qs_eps(*mesh, *ir3d);
   QuadratureFunction eps_design_qf(qs_eps);
   eps_design.Project(eps_design_qf);

   ScalarVectorProductCoefficient J_adju_b_r(1,*adju_b_r);
   ScalarVectorProductCoefficient J_adju_b_i(1,*adju_b_i);
   ScalarVectorProductCoefficient J_adjd_b_r(1,*adjd_b_r);
   ScalarVectorProductCoefficient J_adjd_b_i(1,*adjd_b_i);
   InnerProductCoefficient adju_b_r_dot_adju_b_r_cf(J_adju_b_r,J_adju_b_r);
   InnerProductCoefficient adju_b_i_dot_adju_b_i_cf(J_adju_b_i,J_adju_b_i);
   InnerProductCoefficient adjd_b_r_dot_adjd_b_r_cf(J_adjd_b_r,J_adjd_b_r);
   InnerProductCoefficient adjd_b_i_dot_adjd_b_i_cf(J_adjd_b_i,J_adjd_b_i);
   SumCoefficient adjuc_dot_adju_sum(adju_b_r_dot_adju_b_r_cf,adju_b_i_dot_adju_b_i_cf);
   ProductCoefficient adjuc_dot_adju(10000000,adjuc_dot_adju_sum);
   SumCoefficient adjdc_dot_adjd_sum(adjd_b_r_dot_adjd_b_r_cf,adjd_b_i_dot_adjd_b_i_cf);
   ProductCoefficient adjdc_dot_adjd(10000000,adjdc_dot_adjd_sum);

   //another
   EMsolver_2->SetepsilonCoefficients(&eps_design);
   EMsolver_2->SetupFEM();
   EMsolver_2->SetRHSCoefficient(b2_r_cf,b2_i_cf);
   EMsolver_2->Solve();
   ue_2 = EMsolver_2->GetParFEMSolution();

   VectorGridFunctionCoefficient u_2_r(&(ue_2->real()));
   VectorGridFunctionCoefficient u_2_i(&(ue_2->imag()));
   InnerProductCoefficient u2r_dot_u2r_cf(u_2_r,u_2_r);
   InnerProductCoefficient u2i_dot_u2i_cf(u_2_i,u_2_i);
   SumCoefficient u2c_dot_u2(u2r_dot_u2r_cf,u2i_dot_u2i_cf);

   ScalarVectorProductCoefficient minus_u_2_i(-1.0,u_2_i);
   SumCoefficient u2c_dot_u2_minus_b(adjdc_dot_adjd,u2c_dot_u2,-b2,1);

   ProductCoefficient u2c_dot_u2_minus_b_mult_u2c_dot_u2_minus_b(u2c_dot_u2_minus_b,u2c_dot_u2_minus_b);
   ProductCoefficient overlap_d_2(u2c_dot_u2_minus_b_mult_u2c_dot_u2_minus_b,adjdc_dot_adjd);
   ProductCoefficient overlap_d_2_mult_b(b1,overlap_d_2);
   ProductCoefficient u2c_dot_u2_minus_b_mult_2(2*b1,u2c_dot_u2_minus_b);
   ProductCoefficient u2c_dot_u2_minus_b_mult_2_mult_adjdc_dot_adjd(u2c_dot_u2_minus_b_mult_2,adjdc_dot_adjd);

   // ProductCoefficient u2c_dot_u2_mult_u2c_dot_u2(u2c_dot_u2,u2c_dot_u2);
   // ProductCoefficient overlap_u_2(u2c_dot_u2_mult_u2c_dot_u2,adjuc_dot_adju);
   // ProductCoefficient overlap_u_2_mult_b(b1,overlap_u_2);
   // ProductCoefficient u2c_dot_u2_mult_2(2*b1,u2c_dot_u2);
   // ProductCoefficient u2c_dot_u2_mult_2_mult_adjuc_dot_adju(u2c_dot_u2_mult_2,adjuc_dot_adju);

   ScalarVectorProductCoefficient Jd2_r(u2c_dot_u2_minus_b_mult_2_mult_adjdc_dot_adjd,u_2_r);
   ScalarVectorProductCoefficient Jd2_i(u2c_dot_u2_minus_b_mult_2_mult_adjdc_dot_adjd,minus_u_2_i);
   // ScalarVectorProductCoefficient Ju2_r(u2c_dot_u2_mult_2_mult_adjuc_dot_adju,u_2_r);
   // ScalarVectorProductCoefficient Ju2_i(u2c_dot_u2_mult_2_mult_adjuc_dot_adju,minus_u_2_i);
   // VectorSumCoefficient Jd_r(Ju2_r,Jd2_r);
   // VectorSumCoefficient Jd_i(Ju2_i,Jd2_i);

   LinearForm distance_2(&control_fes);
   distance_2.AddDomainIntegrator(new DomainLFIntegrator(overlap_d_2_mult_b));
   //distance_2.AddDomainIntegrator(new DomainLFIntegrator(overlap_u_2_mult_b));
   distance_2.Assemble();
   real_t obj_2 = distance_2(one_control);


   adjoint_EMsolver_2->SetAdjoint(true);
   adjoint_EMsolver_2->SetepsilonCoefficients(&eps_design);
   adjoint_EMsolver_2->SetupFEM();
   adjoint_EMsolver_2->SetRHSCoefficient(&Jd2_r,&Jd2_i);
   adjoint_EMsolver_2->Solve();
   u_add = adjoint_EMsolver_2->GetParFEMSolution();

   EM_Grad_Coefficient rhs_cf_2(ue_2,u_add,omega_2);
   ProductCoefficient prhs_cf_2(restr_PW,rhs_cf_2);
   Integrate3Dto2D(fespace,prhs_cf_2,grad_array_2D_2);


   //one
   EMsolver_1->SetepsilonCoefficients(&eps_design);
   EMsolver_1->SetupFEM();
   EMsolver_1->SetRHSCoefficient(b1_r_cf,b1_i_cf);
   EMsolver_1->Solve();
   ue_1 = EMsolver_1->GetParFEMSolution();

   VectorGridFunctionCoefficient u_1_r(&(ue_1->real()));
   VectorGridFunctionCoefficient u_1_i(&(ue_1->imag()));
   InnerProductCoefficient u1r_dot_u1r_cf(u_1_r,u_1_r);
   InnerProductCoefficient u1i_dot_u1i_cf(u_1_i,u_1_i);
   SumCoefficient u1c_dot_u1(u1r_dot_u1r_cf,u1i_dot_u1i_cf);

   ScalarVectorProductCoefficient minus_u_1_i(-1.0,u_1_i);
   SumCoefficient u1c_dot_u1_minus_a(adjuc_dot_adju,u1c_dot_u1,-a2,1);

   ProductCoefficient u1c_dot_u1_minus_a_mult_u1c_dot_u1_minus_a(u1c_dot_u1_minus_a,u1c_dot_u1_minus_a);
   ProductCoefficient overlap_u_1(u1c_dot_u1_minus_a_mult_u1c_dot_u1_minus_a,adjuc_dot_adju);
   ProductCoefficient overlap_u_1_mult_a(a1,overlap_u_1);
   ProductCoefficient u1c_dot_u1_minus_a_mult_2(2.0*a1,u1c_dot_u1_minus_a);
   ProductCoefficient u1c_dot_u1_minus_a_mult_2_mult_adjuc_dot_adju(u1c_dot_u1_minus_a_mult_2,adjuc_dot_adju);

   // ProductCoefficient u1c_dot_u1_mult_u1c_dot_u1(u1c_dot_u1,u1c_dot_u1);
   // ProductCoefficient overlap_d_1(u1c_dot_u1_mult_u1c_dot_u1,adjdc_dot_adjd);
   // ProductCoefficient overlap_d_1_mult_a(a1,overlap_d_1);
   // ProductCoefficient u1c_dot_u1_mult_2(2.0*a1,u1c_dot_u1);
   // ProductCoefficient u1c_dot_u1_mult_2_mult_adjdc_dot_adjd(u1c_dot_u1_mult_2,adjdc_dot_adjd);

   ScalarVectorProductCoefficient Ju1_r(u1c_dot_u1_minus_a_mult_2_mult_adjuc_dot_adju,u_1_r);
   ScalarVectorProductCoefficient Ju1_i(u1c_dot_u1_minus_a_mult_2_mult_adjuc_dot_adju,minus_u_1_i);
   // ScalarVectorProductCoefficient Jd1_r(u1c_dot_u1_mult_2_mult_adjdc_dot_adjd,u_1_r);
   // ScalarVectorProductCoefficient Jd1_i(u1c_dot_u1_mult_2_mult_adjdc_dot_adjd,minus_u_1_i);
   // VectorSumCoefficient Ju_r(Ju1_r,Jd1_r);
   // VectorSumCoefficient Ju_i(Ju1_i,Jd1_i);

   LinearForm distance(&control_fes);
   distance.AddDomainIntegrator(new DomainLFIntegrator(overlap_u_1_mult_a));
   // distance.AddDomainIntegrator(new DomainLFIntegrator(overlap_d_1_mult_a));
   distance.Assemble();
   real_t obj_1 = distance(one_control);

   adjoint_EMsolver_1->SetAdjoint(true);
   adjoint_EMsolver_1->SetepsilonCoefficients(&eps_design);
   adjoint_EMsolver_1->SetupFEM();
   adjoint_EMsolver_1->SetRHSCoefficient(&Ju1_r,&Ju1_i);
   adjoint_EMsolver_1->Solve();
   u_adu = adjoint_EMsolver_1->GetParFEMSolution();

   EM_Grad_Coefficient rhs_cf_1(ue_1,u_adu,omega_1);
   ProductCoefficient prhs_cf_1(restr_PW,rhs_cf_1);
   Integrate3Dto2D(fespace,prhs_cf_1,grad_array_2D_1);



   (void)data;
   if (grad)
   {
      for(int i=0; i<el_num_x; i++)
      {
         for(int j=0; j<el_num_y; j++)
         {
            grad[i*el_num_y+j] = 1.0 * grad_array_2D_1(i,j) * compoment * pow(x[i*el_num_y+j],compoment-1)
                       + 1.0 * grad_array_2D_2(i,j) * compoment * pow(x[i*el_num_y+j],compoment-1);
         }
      }
   }

   real_t obj = obj_1*a1+obj_2*b1;
   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("nd_nurbs", pmesh);
   pd->SetPrefixPath("./ue");
   pd->RegisterField("u1r", &(ue_1->real()));
   pd->RegisterField("u1i", &(ue_1->imag()));
   pd->RegisterField("u2r", &(ue_2->real()));
   pd->RegisterField("u2i", &(ue_2->imag()));
   pd->RegisterField("u_adur", &(u_adu->real()));
   pd->RegisterField("u_adui", &(u_adu->imag()));
   pd->RegisterField("u_addr", &(u_add->real()));
   pd->RegisterField("u_addi", &(u_add->imag()));
   pd->RegisterQField("eps_design_now", &eps_design_qf);
   pd->SetLevelsOfDetail(1);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
   delete pd;
   mfem::out<<"opt_num: "<<opt_num++<<" the obj_1 obj_2 obj are: "<<obj_1<<" "<<obj_2<<" "<<obj<<std::endl;
   return obj;
}