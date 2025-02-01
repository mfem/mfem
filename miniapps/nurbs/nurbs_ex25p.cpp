//                                MFEM Example nurbs_25p -- modified for NURBS FE
//
// Compile with: make nurbs_ex25p
//
// Sample runs:  mpirun -np 4 nurbs_ex25p -m ../../data/square-nurbs.mesh
//               mpirun -np 4 nurbs_ex25p -m ../../data/square-nurbs.mesh -o 2
//               mpirun -np 4 nurbs_ex25p -m ../../data/cube-nurbs.mesh

// Description:  This example code solves a simple electromagnetic wave parallel IGA(NURBS)
//               propagation problem corresponding to the second order
//               indefinite Maxwell equation
//
//                  (1/mu) * curl curl E - \omega^2 * epsilon E = f
//
//               with a Perfectly Matched Layer (PML).
//
//               The example demonstrates discretization with Nedelec finite
//               elements in 2D or 3D, as well as the use of complex-valued
//               bilinear and linear forms.
//
//               We recommend viewing Example 25p before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

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
   void StretchFunction(const Vector &x, vector<complex<real_t>> &dxs);
};

// Class for returning the PML coefficients of the bilinear form
class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   PML * pml = nullptr;
   void (*Function)(const Vector &, PML *, Vector &);
public:
   PMLDiagMatrixCoefficient(int dim, void(*F)(const Vector &, PML *,
                                              Vector &),
                            PML * pml_)
      : VectorCoefficient(dim), pml(pml_), Function(F)
   {}

   using VectorCoefficient::Eval;

   virtual void Eval(Vector &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(vdim);
      (*Function)(transip, pml, K);
   }
};
template <typename T> T pow2(const T &x) { return x*x; }
void maxwell_solution(const Vector &x, vector<complex<double>> &Eval);

void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);

void E_exact_Re(const Vector &x, Vector &E);
void E_exact_Im(const Vector &x, Vector &E);

void source(const Vector &x, Vector & f);

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector & D);
void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector & D);
void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector & D);

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector & D);
void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector & D);
void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector & D);

Array2D<double> comp_domain_bdr;
Array2D<double> domain_bdr;

double mu = 1.0;
double epsilon = 1.0;
double omega;
int dim;
bool exact_known = false;


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();
   // Parse command-line options.
   const char *mesh_file = "../../data/square-nurbs.mesh";
   int ref_levels = -1;
   bool NURBS = true;
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;
   double freq = 5.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&NURBS, "-n", "--nurbs", "-nn","--no-nurbs",
                  "NURBS.");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0 )
   {
      args.PrintOptions(cout);
   }


   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // 3. Setup PML length
   Array2D<real_t> length(dim, 2); length = 0.25;
   PML * pml = new PML(mesh,length);
   // Angular frequency
   omega = real_t(2.0 * M_PI) * freq;

   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr = pml->GetDomainBdr();

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < 5; l++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   pml->SetAttributes(pmesh);


   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *fec = nullptr;
   NURBSExtension *NURBSext = nullptr;

   fec = new NURBS_HCurlFECollection(order,dim);
   NURBSext  = new NURBSExtension(pmesh->NURBSext, order);
   mfem::out<<" Create NURBS fec and ext"<<std::endl;

   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, NURBSext,
                                                              fec);
   cout <<"Number of finite element unknowns in rank "<<myid<<": "
        << fespace->GetTrueVSize() << endl;


   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   cout << "Number of knowns in essential BCs: "
        << ess_tdof_list.Size() << endl;

   // 7. Set up the linear form b(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f is
   //    given by the function f_exact and phi_i are the basis functions in the
   //    finite element fespace.

   ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;
   VectorFunctionCoefficient f(dim, source);
   ParComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f));
   b.Vector::operator=(0.0);
   b.Assemble();
   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);

   // 9. Set up the bilinear form corresponding to the EM diffusion operator
   //    curl muinv curl + sigma I, by adding the curl-curl and the mass domain
   //    integrators.
   Array<int> attr;
   Array<int> attrPML;
   if (pmesh->attributes.Size())
   {
      attr.SetSize(pmesh->attributes.Max());
      attrPML.SetSize(pmesh->attributes.Max());
      attr = 0;   attr[0] = 1;
      attrPML = 0;
      if (pmesh->attributes.Max() > 1)
      {
         attrPML[1] = 1;
      }
   }

   ConstantCoefficient muinv(1_r / mu);
   ConstantCoefficient omeg(-pow2(omega) * epsilon);
   RestrictedCoefficient restr_muinv(muinv,attr);
   RestrictedCoefficient restr_omeg(omeg,attr);

   // Integrators inside the computational domain (excluding the PML region)
   ParSesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv),NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg),NULL);

   int cdim = (dim == 2) ? 1 : dim;
   PMLDiagMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, pml);
   PMLDiagMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, pml);
   ScalarVectorProductCoefficient c1_Re(muinv,pml_c1_Re);
   ScalarVectorProductCoefficient c1_Im(muinv,pml_c1_Im);
   VectorRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
   VectorRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

   PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,pml);
   PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,pml);
   ScalarVectorProductCoefficient c2_Re(omeg,pml_c2_Re);
   ScalarVectorProductCoefficient c2_Im(omeg,pml_c2_Im);
   VectorRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
   VectorRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

   // Integrators inside the PML region
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                         new CurlCurlIntegrator(restr_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                         new VectorFEMassIntegrator(restr_c2_Im));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.

   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.Assemble(0);

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   ConstantCoefficient absomeg(pow2(omega) * epsilon);
   RestrictedCoefficient restr_absomeg(absomeg,attr);

   ParBilinearForm prec(fespace);
   prec.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv));
   prec.AddDomainIntegrator(new VectorFEMassIntegrator(restr_absomeg));

   PMLDiagMatrixCoefficient pml_c1_abs(cdim,detJ_inv_JT_J_abs, pml);
   ScalarVectorProductCoefficient c1_abs(muinv,pml_c1_abs);
   VectorRestrictedCoefficient restr_c1_abs(c1_abs,attrPML);

   PMLDiagMatrixCoefficient pml_c2_abs(dim, detJ_JT_J_inv_abs,pml);
   ScalarVectorProductCoefficient c2_abs(absomeg,pml_c2_abs);
   VectorRestrictedCoefficient restr_c2_abs(c2_abs,attrPML);

   prec.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_abs));
   prec.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_abs));

   prec.Assemble();

   // 11. Define and apply a GMRES solver for AU=B with a block diagonal
   //     preconditioner based on the Gauss-Seidel or Jacobi sparse smoother.
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = fespace->GetTrueVSize();
   offsets[2] = fespace->GetTrueVSize();
   offsets.PartialSum();

   std::unique_ptr<Operator> pc_r;
   std::unique_ptr<Operator> pc_i;
   real_t s = (conv == ComplexOperator::HERMITIAN) ? -1_r : 1_r;

   OperatorPtr PCOpAh;
   prec.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
   prec.FormSystemMatrix(ess_tdof_list, PCOpAh);


   // diagonal AMG preconditioner
   pc_r.reset(new HypreBoomerAMG(*PCOpAh.As<HypreParMatrix>()));
   pc_i.reset(new ScaledOperator(pc_r.get(), s));

   BlockDiagonalPreconditioner BlockDP(offsets);
   BlockDP.SetDiagonalBlock(0, pc_r.get());
   BlockDP.SetDiagonalBlock(1, pc_i.get());

   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetPrintLevel(1);
   gmres.SetKDim(200);
   gmres.SetMaxIter(10000);
   gmres.SetRelTol(1e-5);
   gmres.SetAbsTol(0.0);
   gmres.SetOperator(*A);
   gmres.SetPreconditioner(BlockDP);
   gmres.Mult(B, X);

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   ofstream mesh_ofs("nurbs_ex25p.mesh");
   mesh_ofs.precision(8);

   pmesh->Print(mesh_ofs);

   ofstream sol_r_ofs("nurbs_ex25p-sol_r.gf");
   ofstream sol_i_ofs("nurbs_ex25p-sol_i.gf");
   sol_r_ofs.precision(8);
   sol_i_ofs.precision(8);
   x.real().Save(sol_r_ofs);
   x.imag().Save(sol_i_ofs);

   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("nurbs_ex25p", pmesh);
   pd->SetPrefixPath("./nurbs_ex25p_ParaView");
   pd->RegisterField("solution_real", &(x.real()));
   pd->RegisterField("solution_imag", &(x.imag()));
   pd->SetLevelsOfDetail(order);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
   delete pd;


   // Free the used memory.
   delete pml;
   delete fespace;
   delete fec;
   delete pmesh;
   // fespace owns and destroys NURBSext
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
   f[0] = 1000* coeff * exp(alpha);
}

void maxwell_solution(const Vector &x, vector<complex<double>> &E)
{
   // Initialize
   for (int i = 0; i < dim; ++i)
   {
      E[i] = 0.0;
   }

   complex<double> zi = complex<double>(0., 1.);
   double k = omega * sqrt(epsilon * mu);
   // T_10 mode
   if (dim == 3)
   {
      double k10 = sqrt(k * k - M_PI * M_PI);
      E[1] = -zi * k / M_PI * sin(M_PI*x(2))*exp(zi * k10 * x(0));
   }
   else if (dim == 2)
   {
      E[1] = -zi * k / M_PI * exp(zi * k * x(0));
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
   bool in_pml = false;


   for (int i = 0; i < dim; ++i)
   {
      // check if in PML
      if (x(i) - comp_domain_bdr(i, 0) < 0.0 ||
          x(i) - comp_domain_bdr(i, 1) > 0.0)
      {
         in_pml = true;
         break;
      }
   }
   if (!in_pml)
   {
      vector<complex<double>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].real();
      }
   }
}

// Define bdr_data solution
void E_bdr_data_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   bool in_pml = false;

   for (int i = 0; i < dim; ++i)
   {
      // check if in PML
      if (x(i) - comp_domain_bdr(i, 0) < 0.0 ||
          x(i) - comp_domain_bdr(i, 1) > 0.0)
      {
         in_pml = true;
         break;
      }
   }
   if (!in_pml)
   {
      vector<complex<double>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].imag();
      }
   }
}

void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(dxs[i], 2)).real();
   }

}

void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(dxs[i], 2)).imag();
   }
}

void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = abs(det / pow(dxs[i], 2));
   }
}

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

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

void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

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

void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector & D)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

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

PML::PML(Mesh *mesh_, Array2D<real_t> length_) : mesh(mesh_), length(length_)
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
         el->SetAttribute(2);
      }
   }
   mesh_->SetAttributes();
}

void PML::StretchFunction(const Vector &x,
                          vector<complex<real_t>> &dxs)
{
   constexpr complex<real_t> zi = complex<real_t>(0., 1.);

   real_t n = 2.0;
   real_t c = 5.0;
   real_t coeff;
   real_t k = omega * sqrt(epsilon * mu);

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
