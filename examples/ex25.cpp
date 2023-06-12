//                                MFEM Example 25
//
// Compile with: make ex25
//
// Sample runs:  ex25 -o 2 -f 1.0 -ref 2 -prob 0
//               ex25 -o 3 -f 10.0 -ref 2 -prob 1
//               ex25 -o 2 -f 5.0 -ref 4 -prob 2
//               ex25 -o 2 -f 1.0 -ref 2 -prob 3
//               ex25 -o 2 -f 1.0 -ref 2 -prob 0 -m ../data/beam-quad.mesh
//               ex25 -o 2 -f 8.0 -ref 3 -prob 4 -m ../data/inline-quad.mesh
//               ex25 -o 2 -f 2.0 -ref 1 -prob 4 -m ../data/inline-hex.mesh
//
// Device sample runs:
//               ex25 -o 2 -f 8.0 -ref 3 -prob 4 -m ../data/inline-quad.mesh -pa -d cuda
//               ex25 -o 2 -f 2.0 -ref 1 -prob 4 -m ../data/inline-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple electromagnetic wave
//               propagation problem corresponding to the second order
//               indefinite Maxwell equation
//
//                  (1/mu) * curl curl E - \omega^2 * epsilon E = f
//
//               with a Perfectly Matched Layer (PML).
//
//               The example demonstrates discretization with Nedelec finite
//               elements in 2D or 3D, as well as the use of complex-valued
//               bilinear and linear forms. Several test problems are included,
//               with prob = 0-3 having known exact solutions, see "On perfectly
//               matched layers for discontinuous Petrov-Galerkin methods" by
//               Vaziri Astaneh, Keith, Demkowicz, Comput Mech 63, 2019.
//
//               We recommend viewing Example 22 before viewing this example.

#include "mfem.hpp"
#include <memory>
#include <fstream>
#include <iostream>

#ifdef _WIN32
#define jn(n, x) _jn(n, x)
#define yn(n, x) _yn(n, x)
#endif

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
   void SetAttributes(Mesh *mesh_);

   // PML complex stretching function
   void StretchFunction(const Vector &x, vector<complex<double>> &dxs);
};

// Class for returning the PML coefficients of the bilinear form
class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   CartesianPML * pml = nullptr;
   void (*Function)(const Vector &, CartesianPML *, Vector &);
public:
   PMLDiagMatrixCoefficient(int dim, void(*F)(const Vector &, CartesianPML *,
                                              Vector &),
                            CartesianPML * pml_)
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
void detJ_JT_J_inv_Re(const Vector &x, CartesianPML * pml, Vector &D);
void detJ_JT_J_inv_Im(const Vector &x, CartesianPML * pml, Vector &D);
void detJ_JT_J_inv_abs(const Vector &x, CartesianPML * pml, Vector &D);

void detJ_inv_JT_J_Re(const Vector &x, CartesianPML * pml, Vector &D);
void detJ_inv_JT_J_Im(const Vector &x, CartesianPML * pml, Vector &D);
void detJ_inv_JT_J_abs(const Vector &x, CartesianPML * pml, Vector &D);

Array2D<double> comp_domain_bdr;
Array2D<double> domain_bdr;

double mu = 1.0;
double epsilon = 1.0;
double omega;
int dim;
bool exact_known = false;

enum prob_type
{
   beam,     // Wave propagating in a beam-like domain
   disc,     // Point source propagating in the square-disc domain
   lshape,   // Point source propagating in the L-shape domain
   fichera,  // Point source propagating in the fichera domain
   load_src  // Approximated point source with PML all around
};
prob_type prob;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = nullptr;
   int order = 1;
   int ref_levels = 3;
   int iprob = 4;
   double freq = 5.0;
   bool herm_conv = true;
   bool umf_solver = false;
   bool visualization = 1;
   bool pa = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: beam, 1: disc, 2: lshape, 3: fichera, 4: General");
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
#ifdef MFEM_USE_SUITESPARSE
   args.AddOption(&umf_solver, "-umf", "--umfpack", "-no-umf",
                  "--no-umfpack", "Use the UMFPack Solver.");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();

   if (iprob > 4) { iprob = 4; }
   prob = (prob_type)iprob;

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Setup the mesh
   if (!mesh_file)
   {
      exact_known = true;
      switch (prob)
      {
         case beam:
            mesh_file = "../data/beam-hex.mesh";
            break;
         case disc:
            mesh_file = "../data/square-disc.mesh";
            break;
         case lshape:
            mesh_file = "../data/l-shape.mesh";
            break;
         case fichera:
            mesh_file = "../data/fichera.mesh";
            break;
         default:
            exact_known = false;
            mesh_file = "../data/inline-quad.mesh";
            break;
      }
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

   // 4. Setup the Cartesian PML region.
   switch (prob)
   {
      case disc:
         length = 0.2;
         break;
      case lshape:
         length(0, 0) = 0.1;
         length(1, 0) = 0.1;
         break;
      case fichera:
         length(0, 1) = 0.5;
         length(1, 1) = 0.5;
         length(2, 1) = 0.5;
         break;
      case beam:
         length(0, 1) = 2.0;
         break;
      default:
         length = 0.25;
         break;
   }
   CartesianPML * pml = new CartesianPML(mesh,length);
   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr = pml->GetDomainBdr();

   // 5. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 6. Set element attributes in order to distinguish elements in the
   //    PML region
   pml->SetAttributes(mesh);

   // 7. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   int size = fespace->GetTrueVSize();

   cout << "Number of finite element unknowns: " << size << endl;

   // 8. Determine the list of true essential boundary dofs. In this example,
   //    the boundary conditions are defined based on the specific mesh and the
   //    problem type.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      if (prob == lshape || prob == fichera)
      {
         ess_bdr = 0;
         for (int j = 0; j < mesh->GetNBE(); j++)
         {
            Vector center(dim);
            int bdrgeom = mesh->GetBdrElementBaseGeometry(j);
            ElementTransformation * tr = mesh->GetBdrElementTransformation(j);
            tr->Transform(Geometries.GetCenter(bdrgeom),center);
            int k = mesh->GetBdrAttribute(j);
            switch (prob)
            {
               case lshape:
                  if (center[0] == 1.0 || center[0] == 0.5 || center[1] == 0.5)
                  {
                     ess_bdr[k - 1] = 1;
                  }
                  break;
               case fichera:
                  if (center[0] == -1.0 || center[0] == 0.0 ||
                      center[1] ==  0.0 || center[2] == 0.0)
                  {
                     ess_bdr[k - 1] = 1;
                  }
                  break;
               default:
                  break;
            }
         }
      }
   }
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 9. Setup Complex Operator convention
   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // 10. Set up the linear form b(.) which corresponds to the right-hand side of
   //     the FEM linear system.
   VectorFunctionCoefficient f(dim, source);
   ComplexLinearForm b(fespace, conv);
   if (prob == load_src)
   {
      b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f));
   }
   b.Vector::operator=(0.0);
   b.Assemble();

   // 11. Define the solution vector x as a complex finite element grid function
   //     corresponding to fespace.
   ComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);
   x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);

   // 12. Set up the sesquilinear form a(.,.)
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

   // 13. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: assembly, eliminating
   //     boundary conditions, applying conforming constraints for
   //     non-conforming AMR, etc.
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.Assemble(0);

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // 14. Solve using a direct or an iterative solver
#ifdef MFEM_USE_SUITESPARSE
   if (!pa && umf_solver)
   {
      ComplexUMFPackSolver csolver(*A.As<ComplexSparseMatrix>());
      csolver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      csolver.SetPrintLevel(1);
      csolver.Mult(B, X);
   }
#endif
   // 14a. Set up the Bilinear form a(.,.) for the preconditioner
   //
   //    In Comp
   //    Domain:   1/mu (Curl E, Curl F) + omega^2 * epsilon (E,F)
   //
   //    In PML:   1/mu (abs(1/det(J) J^T J) Curl E, Curl F)
   //              + omega^2 * epsilon (abs(det(J) * (J^T J)^-1) * E, F)
   if (pa || !umf_solver)
   {
      ConstantCoefficient absomeg(pow(omega, 2) * epsilon);
      RestrictedCoefficient restr_absomeg(absomeg,attr);

      BilinearForm prec(fespace);
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

      if (pa) { prec.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      prec.Assemble();

      // 14b. Define and apply a GMRES solver for AU=B with a block diagonal
      //      preconditioner based on the Gauss-Seidel or Jacobi sparse smoother.
      Array<int> offsets(3);
      offsets[0] = 0;
      offsets[1] = fespace->GetTrueVSize();
      offsets[2] = fespace->GetTrueVSize();
      offsets.PartialSum();

      std::unique_ptr<Operator> pc_r;
      std::unique_ptr<Operator> pc_i;
      double s = (conv == ComplexOperator::HERMITIAN) ? -1.0 : 1.0;
      if (pa)
      {
         // Jacobi Smoother
         pc_r.reset(new OperatorJacobiSmoother(prec, ess_tdof_list));
         pc_i.reset(new ScaledOperator(pc_r.get(), s));
      }
      else
      {
         OperatorPtr PCOpAh;
         prec.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
         prec.FormSystemMatrix(ess_tdof_list, PCOpAh);

         // Gauss-Seidel Smoother
         pc_r.reset(new GSSmoother(*PCOpAh.As<SparseMatrix>()));
         pc_i.reset(new ScaledOperator(pc_r.get(), s));
      }

      BlockDiagonalPreconditioner BlockDP(offsets);
      BlockDP.SetDiagonalBlock(0, pc_r.get());
      BlockDP.SetDiagonalBlock(1, pc_i.get());

      GMRESSolver gmres;
      gmres.SetPrintLevel(1);
      gmres.SetKDim(200);
      gmres.SetMaxIter(pa ? 5000 : 2000);
      gmres.SetRelTol(1e-5);
      gmres.SetAbsTol(0.0);
      gmres.SetOperator(*A);
      gmres.SetPreconditioner(BlockDP);
      gmres.Mult(B, X);
   }

   // 15. Recover the solution as a finite element grid function and compute the
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

   // 16. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m mesh -g sol".
   {
      ofstream mesh_ofs("ex25.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream sol_r_ofs("ex25-sol_r.gf");
      ofstream sol_i_ofs("ex25-sol_i.gf");
      sol_r_ofs.precision(8);
      sol_i_ofs.precision(8);
      x.real().Save(sol_r_ofs);
      x.imag().Save(sol_i_ofs);
   }

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      // Define visualization keys for GLVis (see GLVis documentation)
      string keys;
      keys = (dim == 3) ? "keys macF\n" : keys = "keys amrRljcUUuu\n";
      if (prob == beam && dim == 3) {keys = "keys macFFiYYYYYYYYYYYYYYYYYY\n";}
      if (prob == beam && dim == 2) {keys = "keys amrRljcUUuuu\n"; }

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

   // 18. Free the used memory.
   delete pml;
   delete fespace;
   delete fec;
   delete mesh;
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
   switch (prob)
   {
      case disc:
      case lshape:
      case fichera:
      {
         Vector shift(dim);
         shift = 0.0;
         if (prob == fichera) { shift =  1.0; }
         if (prob == disc)    { shift = -0.5; }
         if (prob == lshape)  { shift = -1.0; }

         if (dim == 2)
         {
            double x0 = x(0) + shift(0);
            double x1 = x(1) + shift(1);
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
         else if (dim == 3)
         {
            double x0 = x(0) + shift(0);
            double x1 = x(1) + shift(1);
            double x2 = x(2) + shift(2);
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
         break;
      }
      case beam:
      {
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
         break;
      }
      default:
         break;
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

void detJ_JT_J_inv_Re(const Vector &x, CartesianPML * pml, Vector &D)
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

void detJ_JT_J_inv_Im(const Vector &x, CartesianPML * pml, Vector &D)
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

void detJ_JT_J_inv_abs(const Vector &x, CartesianPML * pml, Vector &D)
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

void detJ_inv_JT_J_Re(const Vector &x, CartesianPML * pml, Vector &D)
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

void detJ_inv_JT_J_Im(const Vector &x, CartesianPML * pml, Vector &D)
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

void detJ_inv_JT_J_abs(const Vector &x, CartesianPML * pml, Vector &D)
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

void CartesianPML::SetAttributes(Mesh *mesh_)
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
