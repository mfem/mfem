// Compile with: nvcc  -g -Xcompiler=-Wall -std=c++11 -x=cu --expt-extended-lambda -arch=sm_86 -ccbin mpicxx -I../.. -I../../../hypre/src/hypre/include -I../../../metis/include igaopt_wg.cpp -o igaopt_wg -L../.. -lmfem -L../../../hypre/src/hypre/lib -lHYPRE -lcusparse -lcurand -lcublas -L../../../metis/lib -lmetis -lcusparse -lrt -lnlopt -lm
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
double omega;
int dim;
bool exact_known = false;
Mesh * mesh = nullptr;
ParMesh * pmesh = nullptr;
FiniteElementCollection * fec = nullptr;
NURBSExtension * NURBSext = nullptr;
NURBSEMSolver * EMsolver = nullptr;
NURBSEMSolver * adjoint_EMsolver = nullptr;
//DiffusionSolver * FilterSolver = nullptr;
VectorGridFunctionCoefficient * adju_b_r = nullptr;
VectorGridFunctionCoefficient * adju_b_i = nullptr;
VectorGridFunctionCoefficient * adjd_b_r = nullptr;
VectorGridFunctionCoefficient * adjd_b_i = nullptr;
VectorGridFunctionCoefficient * J_out_r_cf = nullptr;
VectorGridFunctionCoefficient * J_out_i_cf = nullptr;
VectorGridFunctionCoefficient * J_in_r_cf = nullptr;
VectorGridFunctionCoefficient * J_in_i_cf = nullptr;
ParComplexGridFunction * ue = nullptr;
ParComplexGridFunction * uadj = nullptr;
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
   const char *mesh_file = "./meshes/cubes-nurbs_bend.mesh";
   int order = 1;
   const char *device_config = "cuda";
   double freq = 1.0/2000;
   omega = real_t(2.0 * M_PI) * freq;

   Device device(device_config);
   device.Print();

   mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // Setup PML length
   Array2D<real_t> length(dim, 2); length(0,0) = 500; length(0,1) = 500;
   length(1,0) = 400; length(1,1) = 400;
   length(2,0) = 400; length(2,1) = 400;
   PML * pml = new PML(mesh,length,omega);
   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr = pml->GetDomainBdr();

   // Refine the mesh to increase the resolution.
   for (int l = 0; l < 3; l++)
   {
      mesh->UniformRefinement();
   }
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   pml->SetAttributes(pmesh);
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
         if (((coords[1]<4400)&&(coords[1]>3600)&&(coords[2]<1200)&&(coords[2]>800)&&(coords[0]<2500))
            ||((coords[0]<4400)&&(coords[0]>3600)&&(coords[2]<1200)&&(coords[2]>800)&&(coords[1]<2500)))
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
         if ((coords[0]<5500)&&(coords[0]>2500)&&(coords[2]<1200)&&(coords[2]>800)&&(coords[1]>2500)&&(coords[1]<5500))
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
   std::ifstream file_J_in_r("./bend_in_J_r.gf");
   std::ifstream file_J_in_i("./bend_in_J_i.gf");
   std::ifstream file_J_out_r("./bend_out_J_r.gf");
   std::ifstream file_J_out_i("./bend_out_J_i.gf");

   GridFunction J_in_r(pmesh,file_J_in_r);
   GridFunction J_in_i(pmesh,file_J_in_i);
   GridFunction J_out_r(pmesh,file_J_out_r);
   GridFunction J_out_i(pmesh,file_J_out_i);

   J_in_r_cf = new VectorGridFunctionCoefficient(&J_in_r);
   J_in_i_cf = new VectorGridFunctionCoefficient(&J_in_i);
   J_out_r_cf = new VectorGridFunctionCoefficient(&J_out_r);
   J_out_i_cf = new VectorGridFunctionCoefficient(&J_out_i);

   fec = new NURBS_HCurlFECollection(order,dim);
   NURBSext  = new NURBSExtension(pmesh->NURBSext, order);

   fespace = new ParFiniteElementSpace(pmesh, NURBSext, fec);
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ue = new ParComplexGridFunction(fespace);
   uadj = new ParComplexGridFunction(fespace);
   *ue = 0.0;
   *uadj = 0.0;

   Array<const KnotVector*> kv(dim);
   int p = 0;
   NURBSext->GetPatchKnotVectors(p, kv);
   el_num_x = kv[0]->GetNE();
   el_num_y = kv[1]->GetNE();

   EMsolver = new NURBSEMSolver();
   EMsolver->SetMesh(pmesh);
   EMsolver->SetFrequency(freq);
   EMsolver->SetOrder(order);
   EMsolver->SetPML(pml);

   adjoint_EMsolver = new NURBSEMSolver();
   adjoint_EMsolver->SetMesh(pmesh);
   adjoint_EMsolver->SetFrequency(freq);
   adjoint_EMsolver->SetOrder(order);
   adjoint_EMsolver->SetPML(pml);

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
   nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, num_var);
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
   // nlopt_set_max_objective(opt, objfunc, NULL);

   // nlopt_set_xtol_rel(opt,1e-6);

   real_t varx[num_var];
   real_t temp_varx[num_var];
   for(int i=0; i<num_var; i++)
   {
      varx[i] = 0.8;
      temp_varx[i] = 0.8;
   }


   real_t grad[num_var];
   real_t temp_grad[num_var];
   for(int i=0; i<num_var; i++)
   {
      grad[i] = 0;
      temp_grad[i] = 0;
   }
   real_t maxf = 0;
   real_t maxf_old =maxf; 
   // nlopt_result result = nlopt_optimize(opt, varx, &minf);

   real_t alpha = 0.1;
   for(int iter = 0; iter < 10000; iter++)
   {  
      maxf = objfunc(num_var, temp_varx, temp_grad, NULL);
      if(abs(maxf - maxf_old) < 100)
      {
         break;
      }
      mfem::out<<"iter num: "<<iter<<"maxf: "<<maxf<<endl;
      ofstream varfile;
      varfile.open(string("awgccmf_var")+to_string(iter+41)+string(".dat"));
      ofstream gradfile;
      gradfile.open(string("awgccmf_grad")+to_string(iter+41)+string(".dat"));
      for(int i = 0; i<num_var;i++)
      {
         varfile<<varx[i]<<std::endl;
         gradfile<<grad[i]<<std::endl;
      }
      varfile.close();
      gradfile.close();

      if(maxf > maxf_old)
      {  
         if (iter > 1) { alpha *= ((real_t) iter) / ((real_t) iter-1); }
         maxf_old =maxf;
         for(int i = 0; i<num_var; i++)
         {
            varx[i] = temp_varx[i];
            grad[i] = temp_grad[i];
         }
         for(int i = 0; i<num_var; i++)
         {
            temp_varx[i] = temp_varx[i] + alpha*temp_grad[i];
            if(temp_varx[i]>ub[i])
            {
               temp_varx[i] = ub[i];
            }
            if(temp_varx[i]<lb[i])
            {
               temp_varx[i] = lb[i];
            }
         }
      }
      else{
         if (iter > 1) { alpha = alpha/2; }
         for(int i = 0; i<num_var; i++)
         {
            temp_varx[i] = varx[i] + alpha*grad[i];
            if(temp_varx[i]>ub[i])
            {
               temp_varx[i] = ub[i];
            }
            if(temp_varx[i]<lb[i])
            {
               temp_varx[i] = lb[i];
            }
         }
      }
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
   delete pml;
   delete pmesh;
   delete fec;
   delete fespace;
   delete EMsolver;
   delete adjoint_EMsolver;
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
   a.AddDomainIntegrator(new NURBSCurlCurlIntegrator(restr_muinv),NULL);
   a.AddDomainIntegrator(new NURBSHCurl_VectorMassIntegrator(restr_omeg),NULL);

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

   // Integrators inside the PML region
   a.AddDomainIntegrator(new NURBSCurlCurlIntegrator(restr_c1_Re),
                       new NURBSCurlCurlIntegrator(restr_c1_Im));
   a.AddDomainIntegrator(new NURBSHCurl_VectorMassIntegrator(restr_c2_Re),
                       new NURBSHCurl_VectorMassIntegrator(restr_c2_Im));

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


   PMLDiagMatrixCoefficient pml_c1_abs(cdim,detJ_inv_JT_J_abs, pml, pmleps);
   ScalarVectorProductCoefficient c1_abs(muinv,pml_c1_abs);
   VectorRestrictedCoefficient restr_c1_abs(c1_abs,attrPML);
   
   PMLDiagMatrixCoefficient pml_c2_abs(dim, detJ_JT_J_inv_abs, pml, pmleps);
   ScalarVectorProductCoefficient c2_abs(prec_k2epsilon,pml_c2_abs);
   VectorRestrictedCoefficient restr_c2_abs(c2_abs,attrPML);

   prec.AddDomainIntegrator(new NURBSHCurl_VectorMassIntegrator(restr_absomeg));
   prec.AddDomainIntegrator(new NURBSHCurl_VectorMassIntegrator(restr_c2_abs));
   prec.AddDomainIntegrator(new NURBSCurlCurlIntegrator(restr_c1_abs));
   prec.AddDomainIntegrator(new NURBSCurlCurlIntegrator(restr_muinv));
   prec.Assemble();

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = pfes->GetTrueVSize();
   offsets[2] = pfes->GetTrueVSize();
   offsets.PartialSum();
   std::unique_ptr<Operator> pc_r;
   std::unique_ptr<Operator> pc_i;
   real_t s = (conv == ComplexOperator::HERMITIAN) ? -1_r : 1_r;
   
   OperatorJacobiSmoother * ojs = new OperatorJacobiSmoother(prec, ess_tdof_list);

   ScaledOperator * SO = new ScaledOperator(ojs, s);
   
   BlockDiagonalPreconditioner BlockDP(offsets);
   BlockDP.SetDiagonalBlock(0, ojs);
   BlockDP.SetDiagonalBlock(1, SO);

   GMRESSolver gmres(MPI_COMM_WORLD);

   gmres.SetPrintLevel(3);

   gmres.SetKDim(200);
   gmres.SetMaxIter(200000);
   gmres.SetRelTol(1e-2);
   gmres.SetAbsTol(0.0);
   gmres.SetOperator(*A);
   gmres.SetPreconditioner(BlockDP);
   gmres.Mult(B, X);
   a.RecoverFEMSolution(X, b, *x);

   mfem::out<<"main row 1090"<<std::endl;
   for(int num_k2 = 0; num_k2<k2_eps.Size(); num_k2++)
   {
      delete k2_eps[num_k2];
      delete prec_k2_eps[num_k2];
   }
   delete ojs;
   delete SO;
   mfem::out<<"main row 1098"<<std::endl;
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




double objfunc(unsigned n,  const double *x, double *grad, void *data)
{
   mfem::out<<"omega_1 = real_t(2.0 * M_PI) * freq: 1/"<<2*M_PI/omega<<" : "<<omega<<std::endl;
   Array2D<real_t> psi_array_2D(el_num_x,el_num_y);
   Array2D<real_t> grad_array_2D(el_num_x,el_num_y);
   real_t compoment = 2.0;
   grad_array_2D = 0.0;
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

   ScalarVectorProductCoefficient minus_J_in_i_cf(-1,*J_in_i_cf);

   ScalarVectorProductCoefficient J_adj_b_r(1,*J_out_r_cf);
   ScalarVectorProductCoefficient J_adj_b_i(1,*J_out_i_cf);
   InnerProductCoefficient J_adj_b_r_dot_J_adj_b_r_cf(J_adj_b_r,J_adj_b_r);
   InnerProductCoefficient J_adj_b_i_dot_J_adj_b_i_cf(J_adj_b_i,J_adj_b_i);
   SumCoefficient adjc_dot_adj_sum(J_adj_b_r_dot_J_adj_b_r_cf,J_adj_b_i_dot_J_adj_b_i_cf);
   ProductCoefficient adjc_dot_adj(1000000,adjc_dot_adj_sum);

   EMsolver->SetepsilonCoefficients(&eps_design);
   EMsolver->SetupFEM();
   EMsolver->SetRHSCoefficient(J_in_r_cf,&minus_J_in_i_cf);
   EMsolver->Solve();
   ue = EMsolver->GetParFEMSolution();

   VectorGridFunctionCoefficient u_r(&(ue->real()));
   VectorGridFunctionCoefficient u_i(&(ue->imag()));
   InnerProductCoefficient ur_dot_ur_cf(u_r,u_r);
   InnerProductCoefficient ui_dot_ui_cf(u_i,u_i);
   SumCoefficient uc_dot_u(ur_dot_ur_cf,ui_dot_ui_cf);

   ScalarVectorProductCoefficient minus_u_i(-1.0,u_i);
   ProductCoefficient overlap_u(uc_dot_u,adjc_dot_adj);
   ProductCoefficient overlap_u_mult_a(a,overlap_u);
   ProductCoefficient uc_dot_u_mult_2(2.0*a,uc_dot_u);
   ProductCoefficient uc_dot_mult_2_mult_adjc_dot_adj(uc_dot_u_mult_2,adjc_dot_adj);

   ScalarVectorProductCoefficient Ju_r(uc_dot_mult_2_mult_adjc_dot_adj,u_r);
   ScalarVectorProductCoefficient Ju_i(uc_dot_mult_2_mult_adjc_dot_adj,minus_u_i);


   LinearForm distance(&control_fes);
   distance.AddDomainIntegrator(new DomainLFIntegrator(overlap_u_mult_a));
   distance.Assemble();
   real_t obj = distance(one_control);

   //ScalarVectorProductCoefficient minus_J_out_i_cf(-1,*J_out_i_cf);

   adjoint_EMsolver->SetAdjoint(true);
   adjoint_EMsolver->SetepsilonCoefficients(&eps_design);
   adjoint_EMsolver->SetupFEM();
   adjoint_EMsolver->SetRHSCoefficient(&Ju_r, &Ju_i);
   adjoint_EMsolver->Solve();
   uadj = adjoint_EMsolver->GetParFEMSolution();

   EM_Grad_Coefficient rhs_cf(ue,uadj,omega);
   ProductCoefficient prhs_cf(restr_PW,rhs_cf);
   Integrate3Dto2D(fespace,prhs_cf,grad_array_2D);

   (void)data;
   if (grad)
   {
      for(int i=0; i<el_num_x; i++)
      {
         for(int j=0; j<el_num_y; j++)
         {
            grad[i*el_num_y+j] = 1.0 * grad_array_2D(i,j) * compoment * pow(x[i*el_num_y+j],compoment-1);
         }
      }
   }

   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("nd_nurbs", pmesh);
   pd->SetPrefixPath("./wgue");
   pd->RegisterField("ue_r", &(ue->real()));
   pd->RegisterField("ue_i", &(ue->imag()));
   pd->RegisterField("uadj_r", &(uadj->real()));
   pd->RegisterField("uadj_i", &(uadj->imag()));

   pd->RegisterQField("eps_design_now", &eps_design_qf);
   pd->SetLevelsOfDetail(1);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
   delete pd;
   mfem::out<<"opt_num: "<<opt_num++<<" the obj are: "<<obj<<std::endl;
   return obj;
}