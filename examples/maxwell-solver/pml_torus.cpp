
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifdef _WIN32
#define jn(n, x) _jn(n, x)
#define yn(n, x) _yn(n, x)
#endif

using namespace std;
using namespace mfem;

// Class for setting up a simple Cartesian PML region
class Cartesian_PML
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
   Cartesian_PML(Mesh *mesh_,Array2D<double> length_);

   // Return Computational Domain Boundary
   Array2D<double> GetCompDomainBdr() {return comp_dom_bdr;}

   // Return Domain Boundary
   Array2D<double> GetDomainBdr() {return dom_bdr;}

   // Return Markers list for elements
   Array<int> * GetMarkedPMLElements() {return &elems;}

   // Mark elements in the PML region
   void SetAttributes(Mesh *mesh_);

   // PML complex stretching function
   void StretchFunction(const Vector &x, vector<complex<double>> &dxs);
};

// Class for returning the PML coefficients of the bilinear form
class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   Cartesian_PML * pml = nullptr;
   void (*Function)(const Vector &, Cartesian_PML * , Vector &);
public:
   PMLDiagMatrixCoefficient(int dim, void(*F)(const Vector &, Cartesian_PML *,
                                              Vector &),
                            Cartesian_PML * pml_)
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

void maxwell_solution(const Vector &x, vector<complex<double>> &Eval);

void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);

void E_exact_Re(const Vector &x, Vector &E);
void E_exact_Im(const Vector &x, Vector &E);

void source(const Vector &x, Vector & f);

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void detJ_JT_J_inv_Re(const Vector &x, Cartesian_PML * pml, Vector &D);
void detJ_JT_J_inv_Im(const Vector &x, Cartesian_PML * pml, Vector &D);
void detJ_JT_J_inv_abs(const Vector &x, Cartesian_PML * pml, Vector &D);

void detJ_inv_JT_J_Re(const Vector &x, Cartesian_PML * pml, Vector &D);
void detJ_inv_JT_J_Im(const Vector &x, Cartesian_PML * pml, Vector &D);
void detJ_inv_JT_J_abs(const Vector &x, Cartesian_PML * pml, Vector &D);

Array2D<double> comp_domain_bdr;
Array2D<double> domain_bdr;

double mu = 1.0;
double epsilon = 1.0;
double omega;
int dim;
bool exact_known = false;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_SELF, &num_procs);
   MPI_Comm_rank(MPI_COMM_SELF, &myid);
   // 1. Parse command-line options.
   const char *mesh_file = nullptr;
   int order = 1;
   int ref_levels = 3;
   double freq = 5.0;
   bool herm_conv = true;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-ref", "--refinements",
                  "Number of refinements");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency (in Hz).");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // 2. Setup the mesh
   if (!mesh_file)
   {
      exact_known = true;
      mesh_file = "../../data/beam-hex.mesh";
   }
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh * mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // Angular frequency
   omega = 2.0 * M_PI * freq;

   // Setup PML length
   Array2D<double> length(dim, 2); length = 0.0;

   // 3. Setup the Cartesian PML region.
   length(0, 1) = 2.0;

   Cartesian_PML * pml = new Cartesian_PML(mesh,length);
   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr = pml->GetDomainBdr();

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }
   // Set element attributes in order to distinguish elements in the PML region
   pml->SetAttributes(mesh);

   // 6. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   int size = fespace->GetTrueVSize();

   cout << "Number of finite element unknowns: " << size << endl;

   // 7. Determine the list of true essential boundary dofs. In this example,
   //    the boundary conditions are defined based on the specific mesh and the
   //    problem type.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 1;
   }
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 8. Setup Complex Operator convention
   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // 9. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   VectorFunctionCoefficient f(dim, source);
   ComplexLinearForm b(fespace, conv);
   b.Vector::operator=(0.0);
   b.Assemble();

   // 10. Define the solution vector x as a complex finite element grid function
   //     corresponding to fespace.
   ComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);
   x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);

   // 11. Set up the sesquilinear form a(.,.)
   //
   //     In Comp
   //     Domain:   1/mu (Curl E, Curl F) - omega^2 * epsilon (E,F)
   //
   //     In PML:   1/mu (1/det(J) J^T J Curl E, Curl F)
   //               - omega^2 * epsilon (det(J) * (J^T J)^-1 * E, F)
   //
   //     where J denotes the Jacobian Matrix of the PML Stretching function
   Array<int> attr;
   Array<int> attrPML;
   if (mesh->attributes.Size())
   {
      attr.SetSize(mesh->attributes.Max());
      attrPML.SetSize(mesh->attributes.Max());
      attr = 0;   attr[0] = 1;
      attrPML = 0;
      if (mesh->attributes.Max() > 1)
      {
         attrPML[1] = 1;
      }
   }

   ConstantCoefficient muinv(1.0/mu);
   ConstantCoefficient omeg(-pow(omega, 2) * epsilon);
   RestrictedCoefficient restr_muinv(muinv,attr);
   RestrictedCoefficient restr_omeg(omeg,attr);

   // Integrators inside the computational domain (excluding the PML region)
   SesquilinearForm a(fespace, conv);
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

   // 12. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: assembly, eliminating
   //     boundary conditions, applying conforming constraints for
   //     non-conforming AMR, etc.
   a.Assemble(0);

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);


   SparseMatrix * SpMat = (*A.As<ComplexSparseMatrix>()).GetSystemMatrix();
   StopWatch chrono;
   chrono.Clear();
   chrono.Start();
   HYPRE_Int global_size = SpMat->Height();
   HYPRE_Int row_starts[2]; row_starts[0] = 0; row_starts[1] = global_size;
   HypreParMatrix * HypreMat = new HypreParMatrix(MPI_COMM_SELF,global_size,row_starts,SpMat);
   {
      MUMPSSolver mumps;
      mumps.SetOperator(*HypreMat);
      mumps.Mult(B,X);
   }
   chrono.Stop();

   cout << "mumps time = " << chrono.RealTime() << endl;

   // 13. Solve using a direct or an iterative solver
   chrono.Clear();
   chrono.Start();
   // ComplexUMFPackSolver csolver(*A.As<ComplexSparseMatrix>());
   UMFPackSolver csolver(*SpMat);
   csolver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   csolver.SetPrintLevel(1);
   csolver.Mult(B, X);
   chrono.Stop();
   cout << "UMFPack time = " << chrono.RealTime() << endl;

   // 14. Recover the solution as a finite element grid function and compute the
   //     errors if the exact solution is known.
   a.RecoverFEMSolution(X, b, x);

   // If exact is known compute the error
   if (exact_known)
   {
      VectorFunctionCoefficient E_ex_Re(dim, E_exact_Re);
      VectorFunctionCoefficient E_ex_Im(dim, E_exact_Im);
      int order_quad = max(2, 2 * order + 1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i = 0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      double L2Error_Re = x.real().ComputeL2Error(E_ex_Re, irs,
                                                  pml->GetMarkedPMLElements());
      double L2Error_Im = x.imag().ComputeL2Error(E_ex_Im, irs,
                                                  pml->GetMarkedPMLElements());

      ComplexGridFunction x_gf0(fespace);
      x_gf0 = 0.0;
      double norm_E_Re, norm_E_Im;
      norm_E_Re = x_gf0.real().ComputeL2Error(E_ex_Re, irs,
                                              pml->GetMarkedPMLElements());
      norm_E_Im = x_gf0.imag().ComputeL2Error(E_ex_Im, irs,
                                              pml->GetMarkedPMLElements());

      cout << "\n Relative Error (Re part): || E_h - E || / ||E|| = "
           << L2Error_Re / norm_E_Re
           << "\n Relative Error (Im part): || E_h - E || / ||E|| = "
           << L2Error_Im / norm_E_Im
           << "\n Total Error: "
           << sqrt(L2Error_Re*L2Error_Re + L2Error_Im*L2Error_Im) << "\n\n";
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      // Define visualization keys for GLVis (see GLVis documentation)
      string keys;
      keys = (dim == 3) ? "keys macF\n" : keys = "keys amrRljcUUuu\n";
      if (dim == 3) {keys = "keys macFFiYYYYYYYYYYYYYYYYYY\n";}
      if (dim == 2) {keys = "keys amrRljcUUuuu\n"; }

      char vishost[] = "localhost";
      int visport = 19916;

      socketstream sol_sock_re(vishost, visport);
      sol_sock_re.precision(8);
      sol_sock_re << "solution\n"
                  << *mesh << x.real() << keys
                  << "window_title 'Solution real part'" << flush;

      socketstream sol_sock_im(vishost, visport);
      sol_sock_im.precision(8);
      sol_sock_im << "solution\n"
                  << *mesh << x.imag() << keys
                  << "window_title 'Solution imag part'" << flush;

      GridFunction x_t(fespace);
      x_t = x.real();
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *mesh << x_t << keys << "autoscale off\n"
               << "window_title 'Harmonic Solution (t = 0.0 T)'"
               << "pause\n" << flush;
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
      int num_frames = 32;
      int i = 0;
      while (sol_sock)
      {
         double t = (double)(i % num_frames) / num_frames;
         ostringstream oss;
         oss << "Harmonic Solution (t = " << t << " T)";

         add(cos(2.0 * M_PI * t), x.real(),
             sin(2.0 * M_PI * t), x.imag(), x_t);
         sol_sock << "solution\n"
                  << *mesh << x_t
                  << "window_title '" << oss.str() << "'" << flush;
         i++;
      }
   }

   // 17. Free the used memory.
   delete pml;
   delete fespace;
   delete fec;
   delete mesh;


// typedef struct {double r,i;} double_complex;
//    double_complex dc[2];
//    dc[0].r = 1.0;
//    dc[0].i = 2.0;
//    dc[1].r = -1.0;
//    dc[1].i = -2.0;

   MPI_Finalize();

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
   f[0] = coeff * exp(alpha);
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

void detJ_JT_J_inv_Re(const Vector &x, Cartesian_PML * pml, Vector &D)
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

void detJ_JT_J_inv_Im(const Vector &x, Cartesian_PML * pml, Vector &D)
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

void detJ_JT_J_inv_abs(const Vector &x, Cartesian_PML * pml, Vector &D)
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

void detJ_inv_JT_J_Re(const Vector &x, Cartesian_PML * pml, Vector &D)
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

void detJ_inv_JT_J_Im(const Vector &x, Cartesian_PML * pml, Vector &D)
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

void detJ_inv_JT_J_abs(const Vector &x, Cartesian_PML * pml, Vector &D)
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

Cartesian_PML::Cartesian_PML(Mesh *mesh_, Array2D<double> length_)
   : mesh(mesh_), length(length_)
{
   dim = mesh->Dimension();
   SetBoundaries();
}

void Cartesian_PML::SetBoundaries()
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

void Cartesian_PML::SetAttributes(Mesh *mesh_)
{
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
         double *coords = mesh_->GetVertex(vert_idx);
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

void Cartesian_PML::StretchFunction(const Vector &x,
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
