// Compile with: make igaopt_wg_mf
// nvcc  -g -Xcompiler=-Wall -std=c++11 -x=cu --expt-extended-lambda -arch=sm_86 -ccbin mpicxx -I../.. -I../../../hypre/src/hypre/include -I../../../metis/include igaopt_wg_mf.cpp -o igaopt_wg_mf -L../.. -lmfem -L../../../hypre/src/hypre/lib -lHYPRE -lcusparse -lcurand -lcublas -L../../../metis/lib -lmetis -lcusparse -lrt -lnlopt -lm
// Sample runs:  mpirun -np 1 igaopt_wg_mf
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
   std::vector<ParComplexGridFunction*> e;
   std::vector<ParComplexGridFunction*> e_adj;
   GridFunction *rho_filter = nullptr; // filter density
   real_t exponent;
   real_t omegaem;
   int omega_num;
   real_t d_omega;
   real_t delta;
   real_t diff_eps = 3;
   Array<real_t> omegas;

public:
   EM_Grad_Coefficient(std::vector<ParComplexGridFunction*> e_, std::vector<ParComplexGridFunction*> e_adj_, real_t omega_, int omega_num_, real_t d_omega_, real_t delta_)
      : e(e_), e_adj(e_adj_),omegaem(omega_),omega_num(omega_num_),d_omega(d_omega_),delta(delta_)
   {
      omegas.SetSize(omega_num);
      for(int i=0;i<omega_num;i++)
      {
         omegas[i] = omegaem - (omega_num+1)/2 * d_omega + i*d_omega;
      }
   }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      int dim = T.GetSpaceDim();

      real_t density = 0.0;
      Vector e_real(dim);
      Vector e_imag(dim);
      Vector e_adj_real(dim);
      Vector e_adj_imag(dim);
      Vector e_adj_real_minus(dim);
      Vector e_adj_imag_minus(dim);
      Vector e_adj_real_plus(dim);
      Vector e_adj_imag_plus(dim);
      for(int j=0; j<omega_num; j++)
      {
         e_real = 0.0;
         e_imag = 0.0;
         e_adj_real = 0.0;
         e_adj_imag = 0.0;
         e_adj_real_minus = 0.0;
         e_adj_imag_minus = 0.0;
         e_adj_real_plus = 0.0;
         e_adj_imag_plus = 0.0;
         e[j]->real().GetVectorValue(T,ip,e_real);
         e[j]->imag().GetVectorValue(T,ip,e_imag);
         e_adj[j]->real().GetVectorValue(T,ip,e_adj_real);
         e_adj[j]->imag().GetVectorValue(T,ip,e_adj_imag);
         if(j > 0)
         { 
            e_adj[j-1]->real().GetVectorValue(T,ip,e_adj_real_minus);
            e_adj[j-1]->imag().GetVectorValue(T,ip,e_adj_imag_minus);
         }
         if(j < omega_num-1)
         {
            e_adj[j+1]->real().GetVectorValue(T,ip,e_adj_real_plus);
            e_adj[j+1]->imag().GetVectorValue(T,ip,e_adj_imag_plus);
         }
         for(int i=0; i<dim; i++)
         {
          density += (2*omegas[j]*omegas[j])*(e_real[i]*e_adj_real[i] - e_imag[i]*e_adj_imag[i]);
          if(j > 0)
          { 
            density += delta/diff_eps*omegas[j-1]*omegas[j-1]*(e_real[i]*e_adj_real_minus[i] - e_imag[i]*e_adj_imag_minus[i]);
          }
          if(j < omega_num-1)
          {
            density += delta/diff_eps*omegas[j+1]*omegas[j+1]*(e_real[i]*e_adj_real_plus[i] - e_imag[i]*e_adj_imag_plus[i]);
          }
         }
      }

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

// Class for solving maxwell equations in Hcurl NURBS space:
class MF_NURBSEMSolver
{
private:
   Mesh * mesh;
   Array<PML*> pmls;
   Coefficient * design_epsilon;
   int order;
   real_t freq;
   int dim;
   int cdim;
   real_t omega;
   real_t d_omega;
   int omega_num;
   FiniteElementCollection * fec;
   NURBSExtension * NURBSext;
   std::vector<ParComplexGridFunction *> x;
   VectorCoefficient * b_r;
   VectorCoefficient * b_i;
   bool parallel;
   ParMesh * pmesh;
   ParFiniteElementSpace * pfes;
   //ParGridFunction structure_pm;
   Array<int> attr;
   Array<int> attrPML;
   Array<int> attrDesign;
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   ComplexOperator::Convention conv;
   bool adjoint = false;
   MemoryType mt;
   real_t delta;
   int source_w_i;

public:
   MF_NURBSEMSolver() { }
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
   void SetMT(MemoryType mt_) { mt = mt_ ; }
   void SetFrequency(real_t freq_) {freq = freq_; }
   void SetDFrequency(real_t d_omega_, int omega_num_, real_t delta_) {d_omega = d_omega_; omega_num = omega_num_; delta = delta_;}
   void SetPML(Array<PML*> pmls_) {pmls = pmls_; }
   void SetepsilonCoefficients(Coefficient * design_epsilon_) { design_epsilon = design_epsilon_;}
   void SetRHSCoefficient(VectorCoefficient * b_r_, VectorCoefficient * b_i_, int source_w_i_) {b_r = b_r_; b_i = b_i_; source_w_i = source_w_i_;}
   void SetupFEM();
   void Solve();
   std::vector<ParComplexGridFunction *> GetParFEMSolution();
   ~MF_NURBSEMSolver();
};

int opt_num;
double mu = 1.0;
int el_num_x;
int el_num_y;
double omega;
int dim;
bool exact_known = false;
Mesh * mesh = nullptr;
ParMesh * pmesh = nullptr;
FiniteElementCollection * fec = nullptr;
NURBSExtension * NURBSext = nullptr;
VectorGridFunctionCoefficient * b1_r_cf = nullptr;
VectorGridFunctionCoefficient * b1_i_cf = nullptr;
VectorGridFunctionCoefficient * b2_r_cf = nullptr;
VectorGridFunctionCoefficient * b2_i_cf = nullptr;
VectorGridFunctionCoefficient * b3_r_cf = nullptr;
VectorGridFunctionCoefficient * b3_i_cf = nullptr;
ParFiniteElementSpace *fespace = nullptr;
ConstantCoefficient one(1.0);
int omegas_num;
real_t d_omega;
real_t delta;
Array<PML *> pmls;
double freq;
MemoryType mt;
int order;
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
   const char *mesh_file = "./meshes/cubes-nurbs_bend.mesh";
   order = 1;
   const char *device_config = "cuda";
   freq = 1.0/2000;
   omega = real_t(2.0 * M_PI) * freq;
   d_omega = omega*0.01;


   Device device(device_config);
   device.Print();
   mt = device.GetMemoryType();

   mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   omegas_num = 5;
   // Setup PML length
   Array2D<real_t> length(dim, 2); length(0,0) = 500; length(0,1) = 500;
   length(1,0) = 500; length(1,1) = 500;
   length(2,0) = 400; length(2,1) = 400;
   pmls.SetSize(omegas_num);
   for(int i = 0; i < omegas_num; i++)
   {
      pmls[i] = new PML(mesh,length,omega - (((omegas_num+1)/2)+i)*d_omega);
   }
   comp_domain_bdr = pmls[0]->GetCompDomainBdr();
   domain_bdr = pmls[0]->GetDomainBdr();
   // Refine the mesh to increase the resolution.
   for (int l = 0; l < 3; l++)
   {
      mesh->UniformRefinement();
   }
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   pmls[0]->SetAttributes(pmesh);
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
         if (((coords[1]<4400)&&(coords[1]>3600)&&(coords[2]<1200)&&(coords[2]>800)&&(coords[0]>6000))
            ||((coords[1]<(4400+1000))&&(coords[1]>(3600+1000))&&(coords[2]<1200)&&(coords[2]>800)&&(coords[0]<2000))
            ||((coords[1]<(4400-1000))&&(coords[1]>(3600-1000))&&(coords[2]<1200)&&(coords[2]>800)&&(coords[0]<2000)))
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
         if ((coords[0]<6000)&&(coords[0]>2000)&&(coords[2]<1200)&&(coords[2]>800)&&(coords[1]>2000)&&(coords[1]<6000))
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
   mfem::out<<"row 532"<<std::endl;
   std::ifstream file_J_in_1_r("./mf_in_J_r.gf");
   std::ifstream file_J_in_1_i("./mf_in_J_i.gf");
   std::ifstream file_J_in_2_r("./mf_in_2_J_r.gf");
   std::ifstream file_J_in_2_i("./mf_in_2_J_i.gf");
   std::ifstream file_J_in_3_r("./mf_in_3_J_r.gf");
   std::ifstream file_J_in_3_i("./mf_in_3_J_i.gf");
   GridFunction J_in_1_r(pmesh,file_J_in_1_r);
   GridFunction J_in_1_i(pmesh,file_J_in_1_i);
   GridFunction J_in_2_r(pmesh,file_J_in_2_r);
   GridFunction J_in_2_i(pmesh,file_J_in_2_i);
   GridFunction J_in_3_r(pmesh,file_J_in_3_r);
   GridFunction J_in_3_i(pmesh,file_J_in_3_i);

   b1_r_cf = new VectorGridFunctionCoefficient(&J_in_1_r);
   b1_i_cf = new VectorGridFunctionCoefficient(&J_in_1_i);
   b2_r_cf = new VectorGridFunctionCoefficient(&J_in_2_r);
   b2_i_cf = new VectorGridFunctionCoefficient(&J_in_2_i);
   b3_r_cf = new VectorGridFunctionCoefficient(&J_in_3_r);
   b3_i_cf = new VectorGridFunctionCoefficient(&J_in_3_i);

   fec = new NURBS_HCurlFECollection(order,dim);
   NURBSext  = new NURBSExtension(pmesh->NURBSext, order);

   fespace = new ParFiniteElementSpace(pmesh, NURBSext, fec);
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   Array<const KnotVector*> kv(dim);
   int p = 0;
   NURBSext->GetPatchKnotVectors(p, kv);
   el_num_x = kv[0]->GetNE();
   el_num_y = kv[1]->GetNE();


   H1_FECollection filter_fec(2, dim); // space for ρ̃
   ParFiniteElementSpace filter_fes(pmesh, &filter_fec);

   Array<int> ess_bdr_filter;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr_filter.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr_filter = 0;
   }

   int num_var = el_num_x*el_num_y;
   mfem::out<<" "<<el_num_x<<" "<<el_num_y<<std::endl;
   //nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, num_var);
   // double lb[num_var];
   // double ub[num_var];
   // for(int i=0; i<num_var; i++)
   // {
   //    lb[i]=0;
   // }
   // for(int i=0; i<num_var; i++)
   // {
   //    ub[i]=1;
   // }
   // nlopt_set_lower_bounds(opt,lb);
   // nlopt_set_upper_bounds(opt,ub);
   // nlopt_set_max_objective(opt, objfunc, NULL);

   // nlopt_set_xtol_rel(opt,1e-6);

   real_t varx[num_var];
   real_t temp_varx[num_var];
   for(int i=0; i<num_var; i++)
   {
      varx[i] = 0.8;
      temp_varx[i] = 0.8;
   }


   //double maxf = 0;
   // nlopt_result result = nlopt_optimize(opt, varx, &maxf);
   real_t grad[num_var];
   real_t temp_grad[num_var];
   for(int i=0; i<num_var; i++)
   {
      grad[i] = 0;
      temp_grad[i] = 0;
   }
   real_t maxf = 0;
   real_t maxf_old = 0; 
   // nlopt_result result = nlopt_optimize(opt, varx, &minf);

   ifstream in_var_file;
   ifstream in_grad_file;
   in_var_file.open("accmf_var109.dat");
   in_grad_file.open("accmf_grad109.dat");
   string buf_var;
   string buf_grad;
   int ii = 0;
   while (getline(in_var_file, buf_var))
   {  
      temp_varx[ii] = stod(buf_var);
      varx[ii] = stod(buf_var);
      ii ++;
   }
   ii = 0;
   while (getline(in_grad_file, buf_var))
   {  
      temp_grad[ii] = stod(buf_var)/2;
      grad[ii] = stod(buf_var)/2;
      ii ++;
   }
   in_var_file.close();
   in_grad_file.close();


   real_t alpha = 2.32613e-06;

   // for(int i = 0; i<num_var; i++)
   // {
   //    varx[i] = temp_varx[i];
   //    grad[i] = temp_grad[i];
   // }
   // for(int i = 0; i<num_var; i++)
   // {
   //    temp_varx[i] = temp_varx[i] + alpha*temp_grad[i];
   //    if(temp_varx[i]>ub[i])
   //    {
   //       temp_varx[i] = ub[i];
   //    }
   //    if(temp_varx[i]<lb[i])
   //    {
   //       temp_varx[i] = lb[i];
   //    }
   // }


   real_t psi[num_var];
   real_t temp_psi[num_var];
   real_t scale_old = 1024*4;
   real_t scale = 1024*4;
   for(int i=0; i<num_var; i++)
   {
      psi[i] = inv_sigmoid(varx[i])/scale_old;
      temp_psi[i] = inv_sigmoid(temp_varx[i])/scale_old;
   }

   for(int iter = 69; iter < 10000; iter++)
   {  
      mfem::out<<"alpha: "<<alpha<<" num: "<<iter<<std::endl;

      for(int i=0; i<num_var; i++)
      {
         varx[i] = sigmoid(scale*psi[i]);
         temp_varx[i] = sigmoid(scale*temp_psi[i]);
      }
      ofstream varfile;
      varfile.open(string("accmf_var")+to_string(iter+41)+string(".dat"));
      ofstream gradfile;
      gradfile.open(string("accmf_grad")+to_string(iter+41)+string(".dat"));
      for(int i = 0; i<num_var;i++)
      {
         varfile<<varx[i]<<std::endl;
         gradfile<<grad[i]<<std::endl;
      }
      varfile.close();
      gradfile.close();
      maxf = objfunc(num_var, temp_varx, temp_grad, NULL);
      if(abs(maxf - maxf_old) < 100)
      {
         break;
      }

      if(maxf > maxf_old)
      {  
         if (iter > 1) { alpha *= ((real_t) iter) / ((real_t) iter-1); }
         maxf_old =maxf;
         for(int i = 0; i<num_var; i++)
         {
            // varx[i] = temp_varx[i];
            grad[i] = temp_grad[i];
            psi[i] = temp_psi[i];
         }
         for(int i = 0; i<num_var; i++)
         {
            // temp_varx[i] = temp_varx[i] + alpha*temp_grad[i];
            // if(temp_varx[i]>ub[i])
            // {
            //    temp_varx[i] = ub[i];
            // }
            // if(temp_varx[i]<lb[i])
            // {
            //    temp_varx[i] = lb[i];
            // }
            temp_psi[i] = temp_psi[i] + alpha*temp_grad[i]*der_sigmoid(scale*temp_psi[i]);
         }
      }
      else{
         if (iter > 1) { alpha = alpha/2; }
         for(int i = 0; i<num_var; i++)
         {
            // temp_varx[i] = varx[i] + alpha*grad[i];
            // if(temp_varx[i]>ub[i])
            // {
            //    temp_varx[i] = ub[i];
            // }
            // if(temp_varx[i]<lb[i])
            // {
            //    temp_varx[i] = lb[i];
            // }
            temp_psi[i] = temp_psi[i] + alpha*temp_grad[i]*der_sigmoid(scale*temp_psi[i]);
         }
      }
   }


   // for(int i=0; i<num_var; i++)
   // {
   //    mfem::out<<varx[i]<<" ";
   // }
   // mfem::out<<std::endl;

   mfem::out<<"ID "<<myid<<" "<<getpid()<<std::endl;
   int k = 1;
   while(k)
   {
      sleep(5);
   }

   delete mesh;
   delete pmesh;
   delete fec;
   delete fespace;
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

void MF_NURBSEMSolver::SetupFEM()
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
   x.clear();
   std::vector <ParComplexGridFunction *>().swap(x); 
   for(int i=0; i<omega_num;i++)
   {
      x.push_back(new ParComplexGridFunction(pfes));
      *(x[i])=0.0;
   }
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

void MF_NURBSEMSolver::Solve()
{  
   Array<real_t> omegas;
   omegas.SetSize(omega_num);
   for(int i=0; i<omega_num; i++)
   {
      omegas[i] = omega - (omega_num-1)/2*d_omega + i*d_omega;
   }
   real_t wg_eps = 4.0; 
   
   //    Define the two BlockStructure of the problem.  block_offsets is used
   //    for Vector based on dof (like ParGridFunction or ParLinearForm),
   //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
   //    for the rhs and solution of the linear system).  The offsets computed
   //    here are local to the processor.
   Array<int> block_offsets(omega_num+1); // number of variables + 1
   block_offsets[0] = 0;
   for(int i = 1; i<= omega_num;i++)
   {
      block_offsets[i] = 2*(pfes->GetVSize());
   }
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(omega_num+1); // number of variables + 1
   block_trueOffsets[0] = 0;
   for(int i = 1; i<= omega_num;i++)
   {
      block_trueOffsets[i] = 2*(pfes->GetVSize());
   }
   block_trueOffsets.PartialSum();
   //mfem::out<<"row 994 "<<pfes->GetVSize()<<std::endl;

   BlockOperator *MF_Op = new BlockOperator(block_trueOffsets);

   BlockVector trueX(block_trueOffsets, mt), trueRhs(block_trueOffsets, mt);

   trueRhs = 0.0;

   Array<int> pre_offsets(2*omega_num+1);
   pre_offsets[0] = 0;
   for(int i = 1; i<= 2*omega_num;i++)
   {
      pre_offsets[i] = pfes->GetVSize();
   }
   pre_offsets.PartialSum();

   BlockDiagonalPreconditioner BlockDP(pre_offsets);
   //BlockDP.owns_blocks = 1;
   std::vector<OperatorPtr> A(omega_num);
   std::vector<OperatorPtr> C1(omega_num-1);
   std::vector<OperatorPtr> C2(omega_num-1);
   std::vector<ParSesquilinearForm *> a(omega_num);
   std::vector<ParSesquilinearForm *> cp1(omega_num-1);
   std::vector<ParSesquilinearForm *> cp2(omega_num-1);
   std::vector<Array<Coefficient*>> k2_eps(omega_num);
   std::vector<Array<Coefficient*>> prec_k2_eps(omega_num);
   std::vector<ParBilinearForm *> prec(omega_num);
   trueRhs = 0.0;
   std::vector<OperatorJacobiSmoother *> ojs(omega_num);
   std::vector<ScaledOperator *> SO(omega_num);
   for(int i = 0;i<omega_num;i++)
   {
      //mfem::out<<"row 1012"<<std::endl;
      Vector k2epsilon(pmesh->attributes.Max());
      k2epsilon = -pow2(omegas[i]);
      k2epsilon(2) = k2epsilon(0)*wg_eps;
      k2epsilon(3) = k2epsilon(0)*wg_eps;
      ConstantCoefficient muinv(1_r / mu);
      k2_eps[i].SetSize(pmesh->attributes.Max());
      (k2_eps[i])[0] = new ConstantCoefficient(k2epsilon(0));
      (k2_eps[i])[1] = new ConstantCoefficient(k2epsilon(1));
      (k2_eps[i])[2] = new ConstantCoefficient(k2epsilon(2));
      (k2_eps[i])[3] = new ConstantCoefficient(k2epsilon(3));
      (k2_eps[i])[4] = new ProductCoefficient(k2epsilon(4),*design_epsilon);

      Array<int> k2_eps_attr;
      k2_eps_attr.SetSize(pmesh->attributes.Max());
      k2_eps_attr[0] = 1; 
      k2_eps_attr[1] = 2; 
      k2_eps_attr[2] = 3; 
      k2_eps_attr[3] = 4; 
      k2_eps_attr[4] = 5; 
      PWCoefficient k2eps(k2_eps_attr,k2_eps[i]);
      RestrictedCoefficient restr_muinv(muinv,attr);
      RestrictedCoefficient restr_omeg(k2eps,attr);

      // Integrators inside the computational domain (excluding the PML region)
      a[i] = new ParSesquilinearForm(pfes, conv);
      a[i]->SetAssemblyLevel(AssemblyLevel::PARTIAL);

      a[i]->AddDomainIntegrator(new NURBSCurlCurlIntegrator(restr_muinv),NULL);
      a[i]->AddDomainIntegrator(new NURBSHCurl_VectorMassIntegrator(restr_omeg),NULL);

      //Integrators inside the pml
      Vector pml_eps(pmesh->attributes.Max());
      pml_eps = 1;
      pml_eps(2) = pml_eps(0)*wg_eps;
      PWConstCoefficient pmleps(pml_eps);

      Vector pml_k2_eps(pmesh->attributes.Max());
      pml_k2_eps = -pow2(omega);
      pml_k2_eps(2) = pml_k2_eps(0)*wg_eps;
      PWConstCoefficient pml_k2eps(pml_k2_eps);

      PMLDiagMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, pmls[i], pmleps);
      PMLDiagMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, pmls[i], pmleps);
      ScalarVectorProductCoefficient c1_Re(muinv,pml_c1_Re);
      ScalarVectorProductCoefficient c1_Im(muinv,pml_c1_Im);
      VectorRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
      VectorRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

      PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re, pmls[i], pmleps);
      PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im, pmls[i], pmleps);
      ScalarVectorProductCoefficient c2_Re(pml_k2eps,pml_c2_Re);
      ScalarVectorProductCoefficient c2_Im(pml_k2eps,pml_c2_Im);
      VectorRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
      VectorRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

      // Integrators inside the PML region
      a[i]->AddDomainIntegrator(new NURBSCurlCurlIntegrator(restr_c1_Re),new NURBSCurlCurlIntegrator(restr_c1_Im));
      a[i]->AddDomainIntegrator(new NURBSHCurl_VectorMassIntegrator(restr_c2_Re),new NURBSHCurl_VectorMassIntegrator(restr_c2_Im));

      a[i]->Assemble(0);
      a[i]->FormSystemMatrix(ess_tdof_list,A[i]);

      MF_Op->SetBlock(i,i, A[i].Ptr());

      //mfem::out<<"row 1084 "<< (A[i].Ptr())->NumRows()<<std::endl;
      if(i>=1)
      {
         SumCoefficient d_design(-1,*design_epsilon);
         cp1[i-1] = new ParSesquilinearForm(pfes, conv);
         cp1[i-1]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         cp2[i-1] = new ParSesquilinearForm(pfes, conv);
         cp2[i-1]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         ProductCoefficient k2epsilon_c1(-pow2(omegas[i])*delta*0.5/(wg_eps-1),d_design);
         Array<int> k2_eps_c1_attr;
         k2_eps_c1_attr.SetSize(pmesh->attributes.Max());
         k2_eps_c1_attr = 0; 
         k2_eps_c1_attr[4] = 1;

         RestrictedCoefficient k2epsilon_c1_rcf(k2epsilon_c1,k2_eps_c1_attr);
         cp1[i-1]->AddDomainIntegrator(new NURBSHCurl_VectorMassIntegrator(k2epsilon_c1_rcf),NULL);
         cp1[i-1]->Assemble(0);
         cp1[i-1]->FormSystemMatrix(ess_tdof_list,C1[i-1]);
         if(adjoint)
         {
            MF_Op->SetBlock(i-1,i, C1[i-1].Ptr());
         }
         else{
            MF_Op->SetBlock(i,i-1, C1[i-1].Ptr());
         }

         ProductCoefficient k2epsilon_c2(-pow2(omegas[i-1])*delta*0.5/(wg_eps-1),d_design);
         Array<int> k2_eps_c2_attr;
         k2_eps_c2_attr.SetSize(pmesh->attributes.Max());
         k2_eps_c2_attr = 0; 
         k2_eps_c2_attr[4] = 1;
         RestrictedCoefficient k2epsilon_c2_rcf(k2epsilon_c2,k2_eps_c2_attr);
         cp2[i-1]->AddDomainIntegrator(new NURBSHCurl_VectorMassIntegrator(k2epsilon_c2_rcf),NULL);
         cp2[i-1]->Assemble(0);
         cp2[i-1]->FormSystemMatrix(ess_tdof_list,C2[i-1]);

         if(adjoint)
         {
            MF_Op->SetBlock(i,i-1, C2[i-1].Ptr());
         }
         else{
            MF_Op->SetBlock(i-1,i, C2[i-1].Ptr());
         }
      }

      Vector  prec_k2eps(pmesh->attributes.Max());
      prec_k2eps = pow2(omegas[i]);
      prec_k2eps(3) = prec_k2eps(0)*wg_eps;
      prec_k2eps(4) = prec_k2eps(0)*wg_eps;

      prec_k2_eps[i].SetSize(pmesh->attributes.Max());
      (prec_k2_eps[i])[0] = new ConstantCoefficient(prec_k2eps(0));
      (prec_k2_eps[i])[1] = new ConstantCoefficient(prec_k2eps(1));
      (prec_k2_eps[i])[2] = new ConstantCoefficient(prec_k2eps(2));
      (prec_k2_eps[i])[3] = new ConstantCoefficient(prec_k2eps(3));
      (prec_k2_eps[i])[4] = new ProductCoefficient(prec_k2eps(4),*design_epsilon);
      Array<int> prec_k2_eps_attr;
      prec_k2_eps_attr.SetSize(pmesh->attributes.Max());
      prec_k2_eps_attr[0] = 1; 
      prec_k2_eps_attr[1] = 2; 
      prec_k2_eps_attr[2] = 3; 
      prec_k2_eps_attr[3] = 4; 
      prec_k2_eps_attr[4] = 5; 
      PWCoefficient prec_k2epsilon(prec_k2_eps_attr,prec_k2_eps[i]);
      RestrictedCoefficient restr_absomeg(prec_k2epsilon,attr);

      prec[i] = new ParBilinearForm(pfes);
      prec[i]->SetAssemblyLevel(AssemblyLevel::PARTIAL);

      PMLDiagMatrixCoefficient pml_c1_abs(cdim,detJ_inv_JT_J_abs, pmls[i], pmleps);
      ScalarVectorProductCoefficient c1_abs(muinv,pml_c1_abs);
      VectorRestrictedCoefficient restr_c1_abs(c1_abs,attrPML);
      
      PMLDiagMatrixCoefficient pml_c2_abs(dim, detJ_JT_J_inv_abs, pmls[i], pmleps);
      ScalarVectorProductCoefficient c2_abs(prec_k2epsilon,pml_c2_abs);
      VectorRestrictedCoefficient restr_c2_abs(c2_abs,attrPML);

      prec[i]->AddDomainIntegrator(new NURBSHCurl_VectorMassIntegrator(restr_absomeg));
      prec[i]->AddDomainIntegrator(new NURBSHCurl_VectorMassIntegrator(restr_c2_abs));
      prec[i]->AddDomainIntegrator(new NURBSCurlCurlIntegrator(restr_c1_abs));
      prec[i]->AddDomainIntegrator(new NURBSCurlCurlIntegrator(restr_muinv));
      prec[i]->Assemble();


      real_t s = (conv == ComplexOperator::HERMITIAN) ? -1_r : 1_r;
      // mfem::out<<"in solve break 1149"<<std::endl;
      // sleep(10);
      ojs[i] = new OperatorJacobiSmoother(*prec[i], ess_tdof_list);
      // mfem::out<<"in solve break 1152"<<std::endl;
      // sleep(10);
      SO[i] = new ScaledOperator(ojs[i], s);
      // mfem::out<<"in solve break 1158"<<std::endl;
      // sleep(10);
      BlockDP.SetDiagonalBlock(2*i, ojs[i]);
      BlockDP.SetDiagonalBlock(2*i+1, SO[i]);
      mfem::out<<"in solve break 1162"<<std::endl;
   }

   ParComplexLinearForm b(pfes, conv);
   b.Update(pfes);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*b_r),new VectorFEDomainLFIntegrator(*b_i));
   b.Assemble();
   b.ParallelAssemble(trueRhs.GetBlock(source_w_i));
   trueRhs.GetBlock((int)(source_w_i)).SyncAliasMemory(trueRhs);

   mfem::out<<"in solve after assemble"<<std::endl;
   //mfem::out<<"row 1184"<<std::endl;
   trueX = 0.0;
   
   GMRESSolver gmres(MPI_COMM_WORLD);

   gmres.SetPrintLevel(3);

   gmres.SetKDim(500);
   gmres.SetMaxIter(200000);
   gmres.SetRelTol(1e-2);
   gmres.SetAbsTol(0.0);
   gmres.SetPreconditioner(BlockDP);
   gmres.SetOperator(*MF_Op);
   mfem::out<<"row 1203"<<std::endl;

   // mfem::out<<"ID "<<getpid()<<std::endl;
   // int k = 1;
   // while(k)
   // {
   //      sleep(5);
   // }
   gmres.Mult(trueRhs, trueX);
   //mfem::out<<"row 1204"<<std::endl;
   for(int i=0; i<omega_num; i++)
   {
      x[i]->Distribute(&(trueX.GetBlock(i)));
   }
   //mfem::out<<"row 1209"<<std::endl;
   delete MF_Op;
   for(int i=0; i<omega_num;i++)
   {
      delete a[i];
      delete prec[i];
      delete ojs[i];
      delete SO[i];
      if(i<omega_num-1)
      {
         delete cp1[i];
         delete cp2[i];
      }
      for(int num_k2 = 0; num_k2<k2_eps[i].Size(); num_k2++)
      {
         delete k2_eps[i][num_k2];
         delete prec_k2_eps[i][num_k2];
      }
   }
   mfem::out<<"finish solve"<<std::endl;
}

std::vector<ParComplexGridFunction *> MF_NURBSEMSolver::GetParFEMSolution()
{
   if (parallel)
   {
      return x;
   }
   else
   {
      MFEM_ABORT("Wrong code path. Call GetFEMSolution");
   }
}

MF_NURBSEMSolver::~MF_NURBSEMSolver()
{
   // delete design_epsilon;
   // delete b_r;
   // delete b_i;
   delete fec;
   //delete NURBSext;
   delete pfes;
   for(int i=0; i<x.size();i++)
   {
      delete x[i];
   }
}




double objfunc(unsigned n,  const double *x, double *grad, void *data)
{  
   mfem::out<<"omega = real_t(2.0 * M_PI) * freq: 1/"<<2*M_PI/omega<<" : "<<omega<<std::endl;
   delta = 1;

   std::vector<ParComplexGridFunction*> ue_1;
   std::vector<ParComplexGridFunction*> ue_2;
   std::vector<ParComplexGridFunction*> ua_1;
   std::vector<ParComplexGridFunction*> ua_2;
   mfem::out<<"begin simulation"<<std::endl;
   MF_NURBSEMSolver EMsolver_1;
   EMsolver_1.SetMesh(pmesh);
   EMsolver_1.SetFrequency(freq);
   EMsolver_1.SetOrder(order);
   EMsolver_1.SetPML(pmls);
   EMsolver_1.SetMT(mt);
   EMsolver_1.SetDFrequency(d_omega,omegas_num,delta);
   EMsolver_1.SetAdjoint(false);

   MF_NURBSEMSolver adjoint_EMsolver_1;
   adjoint_EMsolver_1.SetMesh(pmesh);
   adjoint_EMsolver_1.SetFrequency(freq);
   adjoint_EMsolver_1.SetOrder(order);
   adjoint_EMsolver_1.SetPML(pmls);
   adjoint_EMsolver_1.SetMT(mt);
   adjoint_EMsolver_1.SetDFrequency(d_omega,omegas_num,delta);
   adjoint_EMsolver_1.SetAdjoint(true);

   MF_NURBSEMSolver EMsolver_2;
   EMsolver_2.SetMesh(pmesh);
   EMsolver_2.SetFrequency(freq);
   EMsolver_2.SetOrder(order);
   EMsolver_2.SetPML(pmls);
   EMsolver_2.SetMT(mt);
   EMsolver_2.SetDFrequency(d_omega,omegas_num,delta);
   EMsolver_2.SetAdjoint(false);


   MF_NURBSEMSolver adjoint_EMsolver_2;
   adjoint_EMsolver_2.SetMesh(pmesh);
   adjoint_EMsolver_2.SetFrequency(freq);
   adjoint_EMsolver_2.SetOrder(order);
   adjoint_EMsolver_2.SetPML(pmls);
   adjoint_EMsolver_2.SetMT(mt);
   adjoint_EMsolver_2.SetDFrequency(d_omega,omegas_num,delta);
   adjoint_EMsolver_2.SetAdjoint(true);

   Array2D<real_t> psi_array_2D(el_num_x,el_num_y);
   Array2D<real_t> grad_array_2D_1(el_num_x,el_num_y);
   Array2D<real_t> grad_array_2D_2(el_num_x,el_num_y);
   real_t compoment = 1.0;
   grad_array_2D_1 = 0.0;
   grad_array_2D_2 = 0.0;
   for(int i=0; i<el_num_x; i++)
   {
      for(int j=0; j<el_num_y; j++)
      {
         psi_array_2D(i,j) = pow(x[i*el_num_y+j],compoment);
      }
   }
   real_t a = 100;
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

   ScalarVectorProductCoefficient minus_J_in_i_cf(-1,*b1_i_cf);

   ScalarVectorProductCoefficient J_in_1_b_r(1,*b1_r_cf);
   ScalarVectorProductCoefficient J_in_1_b_i(1,*b1_i_cf);
   InnerProductCoefficient J_in_1_b_r_dot_J_in_1_b_r_cf(J_in_1_b_r,J_in_1_b_r);
   InnerProductCoefficient J_in_1_b_i_dot_J_in_1_b_i_cf(J_in_1_b_i,J_in_1_b_i);
   SumCoefficient J1_dot_J1_sum(J_in_1_b_r_dot_J_in_1_b_r_cf,J_in_1_b_r_dot_J_in_1_b_r_cf);
   ProductCoefficient J1_dot_J1(1000000,J1_dot_J1_sum);

   ScalarVectorProductCoefficient J_in_3_b_r(1,*b3_r_cf);
   ScalarVectorProductCoefficient J_in_3_b_i(1,*b3_i_cf);
   InnerProductCoefficient J_in_3_b_r_dot_J_in_3_b_r_cf(J_in_3_b_r,J_in_3_b_r);
   InnerProductCoefficient J_in_3_b_i_dot_J_in_3_b_i_cf(J_in_3_b_i,J_in_3_b_i);
   SumCoefficient J3_dot_J3_sum(J_in_3_b_r_dot_J_in_3_b_r_cf,J_in_3_b_r_dot_J_in_3_b_r_cf);
   ProductCoefficient J3_dot_J3(1000000,J3_dot_J3_sum);
   mfem::out<<"in objfun"<<std::endl;
   //one
   EMsolver_1.SetepsilonCoefficients(&eps_design);
   EMsolver_1.SetupFEM();
   EMsolver_1.SetRHSCoefficient(b1_r_cf,b1_i_cf,2);
   EMsolver_1.Solve();
   ue_1 = EMsolver_1.GetParFEMSolution();

   VectorGridFunctionCoefficient u_1_r(&(ue_1[(int)(omegas_num/2+1)]->real()));
   VectorGridFunctionCoefficient u_1_i(&(ue_1[(int)(omegas_num/2+1)]->imag()));
   InnerProductCoefficient u1r_dot_u1r_cf(u_1_r,u_1_r);
   InnerProductCoefficient u1i_dot_u1i_cf(u_1_i,u_1_i);
   SumCoefficient u1c_dot_u1(u1r_dot_u1r_cf,u1i_dot_u1i_cf);

   ScalarVectorProductCoefficient minus_u_1_i(-1.0,u_1_i);

   ProductCoefficient overlap_1(u1c_dot_u1,J3_dot_J3);
   ProductCoefficient overlap_1_mult_a(a,overlap_1);
   ProductCoefficient u1c_dot_u1_mult_2(2.0*a,u1c_dot_u1);
   ProductCoefficient u1c_dot_mult_2_mult_adjc_dot_adj(u1c_dot_u1_mult_2,J3_dot_J3);


   ScalarVectorProductCoefficient J_in_2_b_r(1,*b2_r_cf);
   ScalarVectorProductCoefficient J_in_2_b_i(1,*b2_i_cf);
   InnerProductCoefficient J_in_2_b_r_dot_J_in_2_b_r_cf(J_in_2_b_r,J_in_2_b_r);
   InnerProductCoefficient J_in_2_b_i_dot_J_in_3_b_i_cf(J_in_2_b_i,J_in_2_b_i);
   SumCoefficient J2_dot_J2_sum(J_in_2_b_r_dot_J_in_2_b_r_cf,J_in_2_b_r_dot_J_in_2_b_r_cf);
   ProductCoefficient J2_dot_J2(1000000,J2_dot_J2_sum);
   ProductCoefficient overlap_minus_1(u1c_dot_u1,J2_dot_J2);
   ProductCoefficient overlap_minus_1_mult_a(-a,overlap_minus_1);
   ProductCoefficient u1c_dot_mult_2_mult_adjc_dot_adj_2(u1c_dot_u1_mult_2,J2_dot_J2);

   SumCoefficient u1c_dot_mult_2_mult_adjc_dot_adj_minus_u1c_dot_mult_2_mult_adjc_dot_adj_2(u1c_dot_mult_2_mult_adjc_dot_adj, u1c_dot_mult_2_mult_adjc_dot_adj_2, 1, -3);

   ScalarVectorProductCoefficient J_adj_1_r(u1c_dot_mult_2_mult_adjc_dot_adj_minus_u1c_dot_mult_2_mult_adjc_dot_adj_2,u_1_r);
   ScalarVectorProductCoefficient J_adj_1_i(u1c_dot_mult_2_mult_adjc_dot_adj_minus_u1c_dot_mult_2_mult_adjc_dot_adj_2,minus_u_1_i);

   LinearForm distance1(&control_fes);
   distance1.AddDomainIntegrator(new DomainLFIntegrator(overlap_1_mult_a));
   distance1.Assemble();
   real_t obj_1 = 0;
   obj_1 = distance1(one_control);

   LinearForm distance1_minus(&control_fes);
   distance1_minus.AddDomainIntegrator(new DomainLFIntegrator(overlap_minus_1_mult_a));
   distance1_minus.Assemble();
   real_t obj_1_minus = 0;
   obj_1_minus = distance1_minus(one_control);

   adjoint_EMsolver_1.SetAdjoint(true);
   adjoint_EMsolver_1.SetepsilonCoefficients(&eps_design);
   adjoint_EMsolver_1.SetupFEM();
   adjoint_EMsolver_1.SetRHSCoefficient(&J_adj_1_r,&J_adj_1_i,2+1);
   adjoint_EMsolver_1.Solve();
   ua_1 = adjoint_EMsolver_1.GetParFEMSolution();

   EM_Grad_Coefficient rhs_cf_1(ue_1,ua_1,omega,omegas_num,d_omega,delta);//important
   ProductCoefficient prhs_cf_1(restr_PW,rhs_cf_1);
   Integrate3Dto2D(fespace,prhs_cf_1,grad_array_2D_1);

   //another
   ScalarVectorProductCoefficient minus_b2_i_cf(-1.0,*b2_i_cf);
   EMsolver_2.SetepsilonCoefficients(&eps_design);
   EMsolver_2.SetupFEM();
   EMsolver_2.SetRHSCoefficient(b2_r_cf,&minus_b2_i_cf,2);
   EMsolver_2.Solve();
   ue_2 = EMsolver_2.GetParFEMSolution();

   VectorGridFunctionCoefficient u_2_r(&(ue_2[(int)(omegas_num/2+1)]->real()));
   VectorGridFunctionCoefficient u_2_i(&(ue_2[(int)(omegas_num/2+1)]->imag()));
   InnerProductCoefficient u2r_dot_u2r_cf(u_2_r,u_2_r);
   InnerProductCoefficient u2i_dot_u2i_cf(u_2_i,u_2_i);
   SumCoefficient u2c_dot_u2(u2r_dot_u2r_cf,u2i_dot_u2i_cf);

   ScalarVectorProductCoefficient minus_u_2_i(-1.0,u_2_i);
   ProductCoefficient overlap_2(u2c_dot_u2,J1_dot_J1);
   ProductCoefficient overlap_2_mult_a(a,overlap_2);
   ProductCoefficient u2c_dot_u2_mult_2(2.0*a,u2c_dot_u2);
   ProductCoefficient u2c_dot_mult_2_mult_adjc_dot_adj(u2c_dot_u2_mult_2,J1_dot_J1);

   ScalarVectorProductCoefficient J_adj_2_r(u2c_dot_mult_2_mult_adjc_dot_adj,u_2_r);
   ScalarVectorProductCoefficient J_adj_2_i(u2c_dot_mult_2_mult_adjc_dot_adj,minus_u_2_i);

   LinearForm distance2(&control_fes);
   distance2.AddDomainIntegrator(new DomainLFIntegrator(overlap_2_mult_a));
   distance2.Assemble();
   real_t obj_2 = 0;
   obj_2 = distance2(one_control);

   adjoint_EMsolver_2.SetAdjoint(true);
   adjoint_EMsolver_2.SetepsilonCoefficients(&eps_design);
   adjoint_EMsolver_2.SetupFEM();
   adjoint_EMsolver_2.SetRHSCoefficient(&J_adj_2_r,&J_adj_2_i,2+1);
   adjoint_EMsolver_2.Solve();
   ua_2 = adjoint_EMsolver_2.GetParFEMSolution();

   EM_Grad_Coefficient rhs_cf_2(ue_2,ua_2,omega,omegas_num,d_omega,delta);
   ProductCoefficient prhs_cf_2(restr_PW,rhs_cf_2);
   Integrate3Dto2D(fespace,prhs_cf_2,grad_array_2D_2);
   real_t a1 = 25;
   real_t a2 = 15;
   (void)data;
   if (grad)
   {
      for(int i=0; i<el_num_x; i++)
      {
         for(int j=0; j<el_num_y; j++)
         {
            grad[i*el_num_y+j] = a1 * grad_array_2D_1(i,j) * compoment * pow(x[i*el_num_y+j],compoment-1)
                       + a2 * grad_array_2D_2(i,j) * compoment * pow(x[i*el_num_y+j],compoment-1);
            //mfem::out<<grad[i*el_num_y+j]<<" ";
         }
      }
      //mfem::out<<std::endl;
   }

   real_t obj = a1*(obj_1+3*obj_1_minus)+a2*obj_2;
   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("nd_nurbs", pmesh);
   pd->SetPrefixPath("./mfue");
   for(int num=0;num<omegas_num;num++)
   {
      pd->RegisterField("u1r"+to_string(num), &(ue_1[(int)(num)]->real()));
      pd->RegisterField("u1i"+to_string(num), &(ue_1[(int)(num)]->imag()));
      pd->RegisterField("u2r"+to_string(num), &(ue_2[(int)(num)]->real()));
      pd->RegisterField("u2i"+to_string(num), &(ue_2[(int)(num)]->imag()));
      pd->RegisterField("ua_1r"+to_string(num), &(ua_1[(int)(num)]->real()));
      pd->RegisterField("ua_1i"+to_string(num), &(ua_1[(int)(num)]->imag()));
      pd->RegisterField("ua_2r"+to_string(num), &(ua_2[(int)(num)]->real()));
      pd->RegisterField("ua_2i"+to_string(num), &(ua_2[(int)(num)]->imag()));
   }
   pd->RegisterQField("eps_design_now", &eps_design_qf);
   pd->SetLevelsOfDetail(1);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
   delete pd;
   mfem::out<<"opt_num: "<<opt_num++<<" the obj_1 obj_2 obj are: "<<obj_1<<" "<<obj_2<<" "<<obj_1_minus<<" "<<obj<<std::endl;
   return obj;
}