//                                MFEM Example 3 -- modified for NURBS FE
//
// Compile with: make QAAQmode_mf_in_2
//
// Sample runs:  mpirun -np 1 QAAQmode_mf_in_2
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

using namespace std;
using namespace mfem;

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

class GridFunction_VectorFunctionCoefficient : public VectorCoefficient
{
private:
   GridFunction mode_gf;
   std::function<void(const Vector &, Vector &, GridFunction &, real_t)> Function;
   Coefficient *Q;
   PWConstCoefficient pweps;
public:
   GridFunction_VectorFunctionCoefficient(int dim,
                             std::function<void(const Vector &, Vector &, GridFunction &, real_t )> F,
                             GridFunction gf, PWConstCoefficient pweps,
                             Coefficient *q = nullptr)
                             : VectorCoefficient(dim),mode_gf(gf),Function(std::move(F)),pweps(pweps),Q(q)
                             {};
   using VectorCoefficient::Eval;
   /// Evaluate the vector coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
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

double mu = 1.0;
//double epsilon = 1.0;
double omega;
int dim;
bool exact_known = false;


int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   // Parse command-line options.
   const char *mesh_file = "./meshes/cubes-nurbs_bend.mesh";
   int order = 1;
   const char *device_config = "cpu";
   double freq = 1.0/2000;

   Device device(device_config);
   device.Print();

   // Read the mesh from the given mesh file. We can handle triangular,
   // quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   // the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // Setup PML length
   Array2D<real_t> length(dim, 2); length(0,0) = 500; length(0,1) = 500; length(1,0) = 500; length(1,1) = 500;
   length(2,0) = 500;length(2,1) = 500;
   PML * pml = new PML(mesh,length);
   // Angular frequency
   omega = real_t(2.0 * M_PI) * freq;

   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr = pml->GetDomainBdr();

   // Refine the mesh to increase the resolution.
   for (int l = 0; l < 3; l++)
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
         if ((coords[1] < 4800) && (coords[1] > 3200) && (coords[2] < 2000) && (coords[2] > 1200))
         {
            is_waveguide = true;
            break;
         }
      }
      if (is_waveguide && (el->GetAttribute() == 1))
      {
         el->SetAttribute(5);
      }
      else if (is_waveguide && (el->GetAttribute() == 2))
      {
         el->SetAttribute(6);
      }
      else if (is_waveguide && (el->GetAttribute() == 3))
      {
         el->SetAttribute(7);
      }
      else if (is_waveguide && (el->GetAttribute() == 4))
      {
         el->SetAttribute(8);
      }
   }
   pmesh->SetAttributes();

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *fec = nullptr;
   NURBSExtension *NURBSext = nullptr;
   fec = new NURBS_HCurlFECollection(order,dim);
   NURBSext  = new NURBSExtension(pmesh->NURBSext, order);                                                                              
   mfem::out<<"ID "<<myid<<" "<<getpid()<<" Create NURBS fec and ext"<<std::endl;

   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, NURBSext, fec);
   mfem::out << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   mfem::out << "Number of knowns in essential BCs: "
        << ess_tdof_list.Size() << endl;

   // 7. Set up the linear form b(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f is
   //    given by the function f_exact and phi_i are the basis functions in the
   //    finite element fespace.
   Vector epsilon(pmesh->attributes.Max());
   epsilon = 1;
   epsilon(4) = epsilon(0)*2;
   epsilon(5) = epsilon(0)*2;
   epsilon(6) = epsilon(0)*2;
   epsilon(7) = epsilon(0)*2;
   PWConstCoefficient pweps(epsilon);

   Vector mepsilon(pmesh->attributes.Max());
   mepsilon = -pow2(omega);
   mepsilon(4) = mepsilon(0)*2;
   mepsilon(5) = mepsilon(0)*2;
   mepsilon(6) = mepsilon(0)*2;
   mepsilon(7) = mepsilon(0)*2;
   ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;

   std::ifstream file_x_r("./mf_in_2_nurbs-sol_r.gf");
   std::ifstream file_x_i("./mf_in_2_nurbs-sol_i.gf");

   GridFunction x_r(pmesh,file_x_r);
   GridFunction x_i(pmesh,file_x_i);

   ParComplexGridFunction x(fespace);
   x = 0.0;

   x.real() = x_r;
   x.imag() = x_i;

   Array<int> attr;
   Array<int> attrPML;

   attr.SetSize(pmesh->attributes.Max());
   attrPML.SetSize(pmesh->attributes.Max());
   attr = 0;
   attr[2] = 1;  attr[3] = 1;  attr[6] = 1; attr[7] = 1;
   attrPML = 0;
   attrPML[0] = 1; attrPML[1] = 1; attrPML[4] = 1; attrPML[5] = 1;

   ConstantCoefficient muinv(1_r / mu);
   PWConstCoefficient omeg(mepsilon);
   RestrictedCoefficient restr_muinv(muinv,attr);
   RestrictedCoefficient restr_omeg(omeg,attr);

   // Integrators inside the computational domain (excluding the PML region)
   ParSesquilinearForm a(fespace, conv);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   NURBSCurlCurlIntegrator *di_NURBSCurlCurlIntegrator = new NURBSCurlCurlIntegrator(restr_muinv);
   NURBSHCurl_VectorMassIntegrator *di_NURBSVectorMassIntegrator = new NURBSHCurl_VectorMassIntegrator(restr_omeg);

   a.AddDomainIntegrator(di_NURBSCurlCurlIntegrator,NULL);
   a.AddDomainIntegrator(di_NURBSVectorMassIntegrator,NULL);
   
   int cdim = (dim == 2) ? 1 : dim;
   PMLDiagMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, pml, pweps);
   PMLDiagMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, pml, pweps);
   ScalarVectorProductCoefficient c1_Re(muinv,pml_c1_Re);
   ScalarVectorProductCoefficient c1_Im(muinv,pml_c1_Im);
   VectorRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
   VectorRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

   PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re, pml, pweps);
   PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im, pml, pweps);
   ScalarVectorProductCoefficient c2_Re(omeg,pml_c2_Re);
   ScalarVectorProductCoefficient c2_Im(omeg,pml_c2_Im);
   VectorRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
   VectorRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

   NURBSCurlCurlIntegrator *di_NURBSCurlCurlIntegrator_Re = new NURBSCurlCurlIntegrator(restr_c1_Re);
   NURBSCurlCurlIntegrator *di_NURBSCurlCurlIntegrator_Im = new NURBSCurlCurlIntegrator(restr_c1_Im);
   NURBSHCurl_VectorMassIntegrator *di_NURBSVectorMassIntegrator_Re = new NURBSHCurl_VectorMassIntegrator(restr_c2_Re);
   NURBSHCurl_VectorMassIntegrator *di_NURBSVectorMassIntegrator_Im = new NURBSHCurl_VectorMassIntegrator(restr_c2_Im);

   // Integrators inside the PML region
   // a.AddDomainIntegrator(di_NURBSCurlCurlIntegrator_Re,
   //                     di_NURBSCurlCurlIntegrator_Im);
   // a.AddDomainIntegrator(di_NURBSVectorMassIntegrator_Re,
   //                     di_NURBSVectorMassIntegrator_Im);

   mfem::out<<"main.cpp: test row 511"<<std::endl;
   a.Assemble(0);
   mfem::out<<"main.cpp: test row 518"<<std::endl;
   OperatorPtr A;
   Vector B, X;

   a.FormSystemMatrix(ess_tdof_list, A);


   Array<int> Q_mask;
   Q_mask.SetSize(pmesh->attributes.Max());
   Q_mask = 1;
   Q_mask[0] = 0;
   Q_mask[1] = 1;
   Q_mask[2] = 0;
   Q_mask[3] = 1;
   Q_mask[4] = 0;
   Q_mask[5] = 1;
   Q_mask[6] = 0;
   Q_mask[7] = 1;
   ConstantCoefficient one(100.0);
   RestrictedCoefficient restr_Q_PW(one,Q_mask);

   NURBSHCurl_VectorMassIntegrator *Q_NURBSVectorMassIntegrator = new NURBSHCurl_VectorMassIntegrator(restr_Q_PW);
   ParSesquilinearForm q_mask(fespace, conv);
   q_mask.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   q_mask.AddDomainIntegrator(Q_NURBSVectorMassIntegrator,NULL);
   q_mask.Assemble(0);
   OperatorPtr Q;
   q_mask.FormSystemMatrix(ess_tdof_list, Q);

   ParComplexGridFunction JQA(fespace);
   JQA = 0.0;
   ParComplexGridFunction JAQ(fespace);
   JAQ = 0.0;

   ParComplexGridFunction J_temp(fespace);
   J_temp = 0.0;
   ParComplexGridFunction J(fespace);
   J = 0.0;

   ParComplexGridFunction MS(fespace);
   ParComplexGridFunction MSQ(fespace);
   MS = 1.0;
   MSQ = 0.0;
   Q->Mult(MS,MSQ);
   

   for(int i = 0; i < MSQ.real().Size(); i++ )
   {
      if(MSQ.real()[i] > 0)
      {
         J_temp.real()[i] = x.real()[i];
         J_temp.imag()[i] = x.imag()[i];
      }
   }
   A->Mult(J_temp,JAQ);

   J_temp = 0.0;
   A->Mult(x,J_temp);
   for(int i = 0; i < MSQ.real().Size(); i++ )
   {
      if(MSQ.real()[i] > 0)
      {
         JQA.real()[i] = J_temp.real()[i];
         JQA.imag()[i] = J_temp.imag()[i];
      }
   }

   subtract(JQA, JAQ, J);

   ParComplexLinearForm b(fespace, conv);

   ofstream J_r_ofs("mf_in_2_J_r.gf");
   ofstream J_i_ofs("mf_in_2_J_i.gf");

   J_r_ofs.precision(8);
   J_i_ofs.precision(8);
   J.real().Save(J_r_ofs);
   J.imag().Save(J_i_ofs);

   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("nd_nurbs", pmesh);
   pd->SetPrefixPath("./MFin2Paraview");
   pd->RegisterField("solution_real", &(x.real()));
   pd->RegisterField("solution_imag", &(x.imag()));
   pd->RegisterField("source_real", &(J.real()));
   pd->RegisterField("source_imag", &(J.imag()));
   pd->RegisterField("JAQ_real", &(JAQ.real()));
   pd->RegisterField("JAQ_imag", &(JAQ.imag()));
   pd->RegisterField("JQA_real", &(JQA.real()));
   pd->RegisterField("JQA_imag", &(JQA.imag()));
   pd->SetLevelsOfDetail(order);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
   delete pd;
   delete mesh;


   // Free the used memory.
   delete pml;
   delete fespace;
   delete fec;
   delete pmesh;

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
      bool in_SF = false;
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
         if (coords[0]<1000)
         {
            in_SF = true;
         }
      }
      if (in_pml && !in_SF)
      {
         elems[i] = 0;
         el->SetAttribute(2);
      }
      else if(in_pml && in_SF)
      {
         elems[i] = 0;
         el->SetAttribute(1);
      }
      else if(!in_pml && in_SF)
      {
         elems[i] = 1;
         el->SetAttribute(3);
      }
      else if(!in_pml && !in_SF)
      {
         elems[i] = 1;
         el->SetAttribute(4); 
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

