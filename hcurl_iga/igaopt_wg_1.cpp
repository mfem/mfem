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

      return -density;
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
      real_t arr = sigmoid((*array_2D)(IJK[0],IJK[1]));
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
   GridFunction *rho_filter;
   real_t min_val;
   real_t max_val;
   real_t exponent;

public:
   SIMPInterpolationCoefficient(GridFunction *rho_filter_, real_t min_val_= 1e-6,
                                real_t max_val_ = 1.0, real_t exponent_ = 1)
      : rho_filter(rho_filter_), min_val(min_val_), max_val(max_val_),
        exponent(exponent_) { }

   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t val = rho_filter->GetValue(T, ip);
      real_t coeff = min_val + pow(val,exponent)*(max_val-min_val);
      return coeff;
   }
};

// Class for setting up a simple Cartesian PML region
class PML
{
private:
   Mesh *mesh;

   int dim;

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
   PML(Mesh *mesh_,Array2D<real_t> length_);

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

double mu = 1.0;
//double epsilon = 1.0;
double omega;
int dim;
bool exact_known = false;
real_t delta(const Vector &x);
double objfunc(const std::vector<double> &x, std::vector<double> &grad, void *data);
int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   // Parse command-line options.
   const char *mesh_file = "./meshes/cubes-nurbs.mesh";
   int order = 1;
   const char *device_config = "cuda";
   double freq = 1.0/1550;
   omega = real_t(2.0 * M_PI) * freq;
   real_t epsilon_min = 1.0;
   real_t epsilon_max = 4.0;
   real_t eps_fraction = 0.5;

   Device device(device_config);
   device.Print();

   // Read the mesh from the given mesh file. We can handle triangular,
   // quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   // the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // Setup PML length
   Array2D<real_t> length(dim, 2); length(0,0) = 500; length(0,1) = 500;
   length(1,0) = 400; length(1,1) = 400;
   length(2,0) = 400; length(2,1) = 400;

   //Array2D<real_t> design_domain(dim, 2);
   PML * pml = new PML(mesh,length);

   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr = pml->GetDomainBdr();

   // Refine the mesh to increase the resolution.
   for (int l = 0; l < 4; l++)
   {
      mesh->UniformRefinement();
   }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
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
         if (((coords[1]<3400)&&(coords[1]>2600)&&(coords[2]<1200)&&(coords[2]>800)&&(coords[0]>6500))
            ||((coords[1]<(3400+1000))&&(coords[1]>(2600+1000))&&(coords[2]<1200)&&(coords[2]>800)&&(coords[0]<1500))
            ||((coords[1]<(3400-1000))&&(coords[1]>(2600-1000))&&(coords[2]<1200)&&(coords[2]>800)&&(coords[0]<1500)))
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


   mfem::out<<"main.cpp: test row 789"<<std::endl;

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

   mfem::out<<"main.cpp: test row 804"<<std::endl;

   VectorGridFunctionCoefficient b_r_cf(&J_r);
   VectorGridFunctionCoefficient b_i_cf(&J_i);

   FiniteElementCollection *fec = nullptr;
   NURBSExtension *NURBSext = nullptr;
   fec = new NURBS_HCurlFECollection(order,dim);
   NURBSext  = new NURBSExtension(pmesh->NURBSext, order);

   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, NURBSext, fec);
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ParComplexGridFunction u(fespace);
   ParComplexGridFunction u_add(fespace);
   ParComplexGridFunction u_adu(fespace);
   u = 0.0;
   u_add = 0.0;
   u_adu = 0.0;   

   Array<const KnotVector*> kv(dim);
   int p = 0;
   NURBSext->GetPatchKnotVectors(p, kv);
   int el_num_x = kv[0]->GetNE();
   int el_num_y = kv[1]->GetNE();
   Array2D<real_t> psi_array_2D(el_num_x,el_num_y);
   Array2D<real_t> psi_array_2D_old(el_num_x,el_num_y);
   Array2D<real_t> grad_array_2D(el_num_x,el_num_y);
   psi_array_2D = inv_sigmoid(eps_fraction);
   psi_array_2D_old = inv_sigmoid(eps_fraction);
   grad_array_2D = 0.0;
   H1_FECollection filter_fec(2, dim); // space for ρ̃
   L2_FECollection control_fec(1, dim, BasisType::GaussLobatto); // space for ψ
   ParFiniteElementSpace filter_fes(pmesh, &filter_fec);
   ParFiniteElementSpace control_fes(pmesh, &control_fec);
   ParGridFunction eps_filter(&filter_fes);
   eps_filter = eps_fraction;

   Z2Dto3DCoefficient eps_cf(NURBSext,&psi_array_2D);
   real_t eps_materials = 1.0;
   ConstantCoefficient eps_initial(eps_materials);

   NURBSEMSolver * EMsolver = new NURBSEMSolver();
   EMsolver->SetMesh(pmesh);
   EMsolver->SetFrequency(freq);
   EMsolver->SetOrder(order);
   EMsolver->SetPML(pml);

   NURBSEMSolver * adjoint_EMsolver = new NURBSEMSolver();
   adjoint_EMsolver->SetMesh(pmesh);
   adjoint_EMsolver->SetFrequency(freq);
   adjoint_EMsolver->SetOrder(order);
   adjoint_EMsolver->SetPML(pml);

   //Set-up the filter solver.
   real_t epsilon_for_filter = 0.01;
   ConstantCoefficient one(1.0);
   ConstantCoefficient eps2_cf(epsilon_for_filter*epsilon_for_filter);
   DiffusionSolver * FilterSolver = new DiffusionSolver();
   FilterSolver->SetMesh(pmesh);
   FilterSolver->SetOrder(filter_fec.GetOrder());
   FilterSolver->SetDiffusionCoefficient(&eps2_cf);
   FilterSolver->SetMassCoefficient(&one);
   Array<int> ess_bdr_filter;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr_filter.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr_filter = 0;
   }
   FilterSolver->SetEssentialBoundary(ess_bdr_filter);
   FilterSolver->SetupFEM();

   real_t alpha_opt = 0.005;
   ParGridFunction grad(&control_fes);
   ParGridFunction w_filter(&filter_fes);

   Array<int> is_design_arr;
   is_design_arr.SetSize(pmesh->attributes.Max());
   is_design_arr = 0;
   is_design_arr[4] = 1;
   RestrictedCoefficient restr_PW(one,is_design_arr);

   BilinearForm m_mask(&control_fes);
   m_mask.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
   m_mask.Assemble(0);
   OperatorPtr Mass;
   m_mask.FormSystemMatrix(ess_tdof_list, Mass);

   ConstantCoefficient zero(0.0);
   ParGridFunction onegf(&control_fes);
   onegf = 1.0;
   ParGridFunction zerogf(&control_fes);
   zerogf = 0.0;
   ParLinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   real_t domain_volume = vol_form(onegf);
   const real_t target_volume = domain_volume * eps_fraction;
   GridFunction ONE(&filter_fes);
   ONE = 1.0;

   GridFunction one_control(&control_fes);
   one_control = 1.0;
   VectorGridFunctionCoefficient adju_b_r(&AUJ_r);
   VectorGridFunctionCoefficient adju_b_i(&AUJ_i);

   int max_it = 20;
   omega = real_t(2.0 * M_PI) * freq;
   EM_Grad_Coefficient rhs_cf(&u,&u_adu,omega);
   ProductCoefficient prhs_cf(restr_PW,rhs_cf);
   Vector overlaps(max_it);
   overlaps = 0.0;
   for(int iter_num = 0; iter_num < max_it; iter_num++)
   {
        if (iter_num > 1) { alpha_opt *= ((real_t) iter_num) / ((real_t) iter_num-1); }
        mfem::out << "\nStep = " << iter_num << std::endl;
        // Filter solve
        // Solve (ϵ^2 ∇ ρ̃, ∇ v ) + (ρ̃,v) = (ρ,v)
        FilterSolver->SetRHSCoefficient(&eps_cf);
        FilterSolver->Solve();
        eps_filter = *FilterSolver->GetFEMSolution();

        SIMPInterpolationCoefficient SIMP_cf(&eps_filter,epsilon_min, epsilon_max, 1.0);
        ProductCoefficient eps_design(eps_initial,SIMP_cf);

        EMsolver->SetepsilonCoefficients(&eps_design);
        EMsolver->SetupFEM();
        EMsolver->SetRHSCoefficient(&b_r_cf,&b_i_cf);
        EMsolver->Solve();
        u = * EMsolver->GetParFEMSolution();

        VectorGridFunctionCoefficient u_r(&(u.real()));
        VectorGridFunctionCoefficient u_i(&(u.imag()));

        InnerProductCoefficient adju_b_r_dot_ur_cf(adju_b_r,u_r);
        InnerProductCoefficient adju_b_i_dot_ui_cf(adju_b_i,u_i);

        InnerProductCoefficient adju_b_r_dot_ui_cf(adju_b_r,u_i);
        InnerProductCoefficient adju_b_i_dot_ur_cf(adju_b_i,u_r);

        ProductCoefficient minus_adju_b_i_dot_ur_cf(-1.0,adju_b_i_dot_ur_cf);

        LinearForm adju_b_r_dot_ur_and_adju_b_i_dot_ui_lf(&control_fes);
        adju_b_r_dot_ur_and_adju_b_i_dot_ui_lf.AddDomainIntegrator(new DomainLFIntegrator(adju_b_r_dot_ur_cf));
        adju_b_r_dot_ur_and_adju_b_i_dot_ui_lf.AddDomainIntegrator(new DomainLFIntegrator(adju_b_i_dot_ui_cf));
        adju_b_r_dot_ur_and_adju_b_i_dot_ui_lf.Assemble();
        real_t sum_adju_b_r_dot_ur_and_adju_b_i_dot_ui = adju_b_r_dot_ur_and_adju_b_i_dot_ui_lf(one_control);

        LinearForm adju_b_r_dot_ui_minus_adju_b_i_dot_ur_lf(&control_fes);
        adju_b_r_dot_ui_minus_adju_b_i_dot_ur_lf.AddDomainIntegrator(new DomainLFIntegrator(adju_b_r_dot_ui_cf));
        adju_b_r_dot_ui_minus_adju_b_i_dot_ur_lf.AddDomainIntegrator(new DomainLFIntegrator(minus_adju_b_i_dot_ur_cf));
        adju_b_r_dot_ui_minus_adju_b_i_dot_ur_lf.Assemble();
        real_t sum_adju_b_r_dot_ui_minus_adju_b_i_dot_ur = adju_b_r_dot_ui_minus_adju_b_i_dot_ur_lf(one_control);

        real_t overlap = pow(sum_adju_b_r_dot_ur_and_adju_b_i_dot_ui,2)+pow(sum_adju_b_r_dot_ui_minus_adju_b_i_dot_ur,2);

        real_t sum_J_dot_uc_r = sum_adju_b_r_dot_ur_and_adju_b_i_dot_ui;
        real_t sum_J_dot_uc_i = sum_adju_b_r_dot_ui_minus_adju_b_i_dot_ur;
        ScalarVectorProductCoefficient adj_u_r(sum_J_dot_uc_r,adju_b_r);
        ScalarVectorProductCoefficient adj_u_i(sum_J_dot_uc_i,adju_b_i);

        adjoint_EMsolver->SetAdjoint(true);
        adjoint_EMsolver->SetepsilonCoefficients(&eps_design);
        adjoint_EMsolver->SetupFEM();
        adjoint_EMsolver->SetRHSCoefficient(&adj_u_r,&adj_u_i);
        adjoint_EMsolver->Solve();
        u_adu = * adjoint_EMsolver->GetParFEMSolution();
        mfem::out<<"main.cpp: test row 957"<<std::endl;

        Integrate3Dto2D(fespace,prhs_cf,grad_array_2D);
        mfem::out<<std::endl;
        for(int M=0;M<el_num_x;M++)
        {
          for(int N=0;N<el_num_y;N++)
          {
            real_t S = sigmoid(psi_array_2D(M,N));
            psi_array_2D(M,N) += alpha_opt*(grad_array_2D(M,N))*S*(1-S);
            mfem::out<<" grad: "<<grad_array_2D(M,N)<<"... ";
          }
        }
        mfem::out<<std::endl;
        mfem::out<<"finish step "<<iter_num<<std::endl;
        overlaps[iter_num] = overlap; 
        for(int overlap_num = 0; overlap_num <= iter_num; overlap_num++)
        {
            mfem::out<<"overlap "<<overlap_num<<": "<<overlaps[overlap_num]<<std::endl;
        }
   }
   Grad_Z2Dto3DCoefficient show_grad(NURBSext,&grad_array_2D);
   ProductCoefficient prhs_cf2(restr_PW,show_grad);
   LinearForm w_rhs(&control_fes);
   w_rhs.AddDomainIntegrator(new DomainLFIntegrator(prhs_cf2));
   w_rhs.Assemble();
   Mass->Mult(w_rhs,grad);
   GridFunction grad_gf2(&control_fes);
   LinearForm w_rhs2(&control_fes);
   w_rhs2.AddDomainIntegrator(new DomainLFIntegrator(prhs_cf));
   w_rhs2.Assemble();
   Mass->Mult(w_rhs2,grad_gf2);
   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("nd_nurbs", pmesh);
   pd->SetPrefixPath("./ue");
   pd->RegisterField("ur", &(u.real()));
   pd->RegisterField("ui", &(u.imag()));
   pd->RegisterField("u_adur", &(u_adu.real()));
   pd->RegisterField("u_adui", &(u_adu.imag()));
   pd->RegisterField("eps_filter", &(eps_filter));
   pd->RegisterField("grad", &(grad));
   pd->RegisterField("grad_gf2", &(grad_gf2));
   pd->SetLevelsOfDetail(order);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
   delete pd;
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
   //delete FilterSolver;
   if(myid == 0)
   {
      mfem::out<<"****** FINISH ******"<<std::endl;
   }
   return 0;
}

real_t delta(const Vector &x)
{
   Vector center(3);
   center(0) = 2000;
   center(1) = 3000;
   center(2) = 1000;
   real_t r = 0.0;
   for (int i = 0; i < 3; ++i)
   {
      r += pow2(x[i] - center[i]);
   }
   real_t n = 5_r * omega * 0.5/ real_t(M_PI);
   real_t alpha = -pow2(n) * r;
   return exp(alpha);
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

PML::PML(Mesh *mesh_, Array2D<real_t> length_)
   : mesh(mesh_), length(length_)
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
   mfem::out<<"main.cpp: test row 856: "<<pmesh->attributes.Max()<<std::endl;
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
   mfem::out<<"main.cpp: test row 868 dim: "<<dim<<std::endl;
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
   mfem::out<<"main.cpp: EMsolver row 930"<<std::endl;
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

   gmres.SetPrintLevel(1);
   gmres.SetKDim(200);
   gmres.SetMaxIter(200000);
   if(adjoint)
   {
    gmres.SetRelTol(1e-2);
   }
   else{
    gmres.SetRelTol(1e-2);
   }
   gmres.SetAbsTol(0.001);
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

double objfunc(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  (void)data;
  if (!grad.empty()) {
    grad[0] = 0.0;
    grad[1] = 0.5 / sqrt(x[1]);
  }
  return sqrt(x[1]);
}