//                       MFEM Example 23 - Serial Version
//
// Compile with: make ex23
//
// Sample runs:  ex23 -o 2 -f 1.3 -rs 1 -rp 1 -prob 0
//               ex23 -o 2 -f 1.3 -rs 1 -rp 1 -prob 1
//               ex23 -o 2 -f 3.3 -rs 3 -rp 1 -prob 2
//               ex23 -o 2 -f 1.3 -rs 2 -rp 1 -prob 3
//               ex23 -o 2 -f 10.3 -rs 2 -rp 2 -m ../data/inline-quad.mesh
//               ex23 -o 2 -f 2.3 -rs 1 -rp 1 -m ../data/inline-hex.mesh

// Description:  This example code solves a simple electromagnetic wave
//               propagation problem corresponding to the second order indefinite
//               Maxwell equation (1/mu) * curl curl E - \omega^2 * epsilon E = f
//               with a Perfectly Matched Layer (PML).
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example also demonstrates the use of complex valued bilinear and linear forms.
//               We recommend viewing example 22 before viewing this example.
//               Four examples are probided with exact solutions (iprob = 0-4).

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifdef _WIN32
#define jn(n, x) _jn(n, x)
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

   // Integer Array identifying elements in the pml
   // 0: in the pml, 1: not in the pml
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

   // Return Marker list for elements
   Array<int> * GetMarkedPMLElements() {return &elems;}

   // Mark element in the PML region
   void SetAttributes(Mesh *mesh_);

   // PML complex streching function
   void StretchFunction(const Vector &x, std::vector<std::complex<double>> &dxs);
};

class PmlMatrixCoefficient : public MatrixCoefficient
{
private:

   CartesianPML * pml = nullptr;
   void (*Function)(const Vector &, CartesianPML * , DenseMatrix &);

public:
   PmlMatrixCoefficient(int dim, void(*F)(const Vector &, CartesianPML *,
                                          DenseMatrix &),
                        CartesianPML * pml_)
      : MatrixCoefficient(dim), pml(pml_), Function(F)
   {}
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(height, width);
      (*Function)(transip, pml, K);
   }
};

void maxwell_solution(const Vector &x, std::vector<std::complex<double>> &Eval);
void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);
void E_exact_Re(const Vector &x, Vector &E);
void E_exact_Im(const Vector &x, Vector &E);
void source(const Vector &x, Vector & f);

// PML Coefficients Functions
void detJ_JT_J_inv_Re(const Vector &x, CartesianPML * pml, DenseMatrix &M);
void detJ_JT_J_inv_Im(const Vector &x, CartesianPML * pml, DenseMatrix &M);
void detJ_inv_JT_J_Re(const Vector &x, CartesianPML * pml, DenseMatrix &M);
void detJ_inv_JT_J_Im(const Vector &x, CartesianPML * pml, DenseMatrix &M);

Array2D<double> comp_domain_bdr;
Array2D<double> domain_bdr;

double mu = 1.0;
double epsilon = 1.0;
double omega;
int dim;
bool exact_known = false;

enum prob_type
{
   beam,     // PML on one end of the domain
   scatter,  // Scattering in a square or a cube
   lshape,   // Scattering in 1/4 of a square
   fichera,   // Scattering in 1/8 of a cube
   load_src // point source with PML all around
};
prob_type prob;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = nullptr;
   int order = 1;
   int ref_levels = 1;
   int iprob = 4;
   double freq = 2.0;
   bool herm_conv = true;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: General, 1: beam, 2: scatter, 3: lshape, 4: fichera");
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

   // 2. Setup the (serial) mesh on all processors.
   if (iprob > 4) { iprob = 4; }
   prob = (prob_type)iprob;

   if (!mesh_file)
   {
      exact_known = true;
      switch (prob)
      {
         case beam:
            mesh_file = "../data/beam-hex.mesh";
            break;
         case scatter:
            mesh_file = "../data/square_w_hole.mesh";
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

   //Angular frequency
   omega = 2.0 * M_PI * freq;

   // Setup PML length
   Array2D<double> lngth(dim, 2);  lngth = 0.0;

   // 3. Setup the Cartesian PML region.
   switch (prob)
   {
      case scatter:
         lngth = 1.0;
         break;
      case lshape:
         lngth(0, 1) = 0.5;
         lngth(1, 1) = 0.5;
         break;
      case fichera:
         lngth(0, 1) = 0.5;
         lngth(1, 1) = 0.5;
         lngth(2, 1) = 0.5;
         break;
      case beam:
         lngth(0, 1) = 2.0;
         break;
      default:
         lngth = 0.25;
         break;
   }
   CartesianPML * pml = new CartesianPML(mesh,lngth);
   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr = pml->GetDomainBdr();

   // 4. Refine the serial mesh on all processors to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh.
   mesh->ReorientTetMesh();

   // Set element attributes in order to destiguish elements in the PML region
   pml->SetAttributes(mesh);

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   int size = fespace->GetTrueVSize();

   cout << "Number of finite element unknowns: " << size << endl;

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    based on the specific mesh and the problem type.
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
            ElementTransformation * trans = mesh->GetBdrElementTransformation(j);
            trans->Transform(Geometries.GetCenter(bdrgeom),center);
            int k = mesh->GetBdrAttribute(j);
            switch (prob)
            {
               case lshape:
                  if (center[0] == 1.0 || center[0] == 0.0 || center[1] == 1.0)
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

   // Setup Complex Operator convention
   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system.
   VectorFunctionCoefficient f(dim, source);
   ComplexLinearForm b(fespace, conv);
   if (prob == load_src)
   {
      b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f));
   }
   b.Vector::operator=(0.0);
   b.Assemble();

   // 9. Define the solution vector x as a parallel complex finite element grid
   //    function corresponding to fespace.
   ComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);
   x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);

   // 10. Set up the parallel sesquilinear form a(.,.) on the finite element
   //     space corresponding to the problem of the appropriate type:

   Array<int> attr;
   if (mesh->attributes.Size())
   {
      attr.SetSize(mesh->attributes.Max());
      attr = 0;
      attr[0] = 1;
   }
   ConstantCoefficient muinv(1.0/mu);
   ConstantCoefficient sigma(-pow(omega, 2) * epsilon);
   RestrictedCoefficient restr_muinv(muinv,attr);
   RestrictedCoefficient restr_sigma(sigma,attr);

   SesquilinearForm a(fespace, conv);
   // Restricted coefficient to the computational domain
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv),NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_sigma),NULL);

   // pml region attributes
   if (mesh->attributes.Size())
   {
      attr = 0;
      if (mesh->attributes.Max() > 1) { attr[1] = 1; }
   }

   // PML coefficients
   int cdim = (dim == 2) ? 1 : dim;
   PmlMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, pml);
   PmlMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, pml);
   ScalarMatrixProductCoefficient c1_Re(muinv,pml_c1_Re);
   ScalarMatrixProductCoefficient c1_Im(muinv,pml_c1_Im);
   MatrixRestrictedCoefficient restr_c1_Re(c1_Re,attr);
   MatrixRestrictedCoefficient restr_c1_Im(c1_Im,attr);


   PmlMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,pml);
   PmlMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,pml);
   ScalarMatrixProductCoefficient c2_Re(sigma,pml_c2_Re);
   ScalarMatrixProductCoefficient c2_Im(sigma,pml_c2_Im);
   MatrixRestrictedCoefficient restr_c2_Re(c2_Re,attr);
   MatrixRestrictedCoefficient restr_c2_Im(c2_Im,attr);

   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                         new CurlCurlIntegrator(restr_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                         new VectorFEMassIntegrator(restr_c2_Im));

   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, etc.
   a.Assemble();

   OperatorHandle Ah;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, Ah, X, B);

   // Transform to monolithic HypreParMatrix
   SparseMatrix *A = Ah.As<ComplexSparseMatrix>()->GetSystemMatrix();

   cout << "Size of linear system: " << A->Height() << endl;

   //12.  Solve using a direct or an iterative solver
#ifdef MFEM_USE_SUITESPARSE
   // Direct Solver
   UMFPackSolver  solver(*A);
   solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   solver.Mult(B, X);
#else
   GMRESSolver * gmres = new GMRESSolver;
   gmres->SetPrintLevel(1);
   gmres->SetMaxIter(1000);
   gmres->SetRelTol(1e-6);
   gmres->SetAbsTol(0.0);
   gmres->SetOperator(*A);
   gmres->Mult(B, X);
   delete gmres;
#endif

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   // If exact is known compute the error
   if (exact_known)
   {
      ComplexGridFunction x_gf(fespace);
      VectorFunctionCoefficient E_ex_Re(dim, E_exact_Re);
      VectorFunctionCoefficient E_ex_Im(dim, E_exact_Im);
      x_gf.ProjectCoefficient(E_ex_Re, E_ex_Im);
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
      double norm_E_Re = x_gf0.real().ComputeL2Error(E_ex_Re, irs,
                                                     pml->GetMarkedPMLElements());
      double norm_E_Im = x_gf0.imag().ComputeL2Error(E_ex_Im, irs,
                                                     pml->GetMarkedPMLElements());

      cout << " Rel Error - Real Part: || E_h - E || / ||E|| = " << L2Error_Re /
           norm_E_Re << '\n' << endl;
      cout << " Rel Error - Imag Part: || E_h - E || / ||E|| = " << L2Error_Im /
           norm_E_Im << '\n' << endl;
      cout << " Total Error: " << sqrt(L2Error_Re * L2Error_Re + L2Error_Im *
                                       L2Error_Im)  << endl;
   }

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      string keys;
      keys = (dim == 3) ? "keys macFF\n" : keys = "keys amrRljcUUuuu\n";
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

   // 15. Free the used memory.
   delete A;
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

void maxwell_solution(const Vector &x, std::vector<std::complex<double>> &E)
{
   // Initialize
   for (int i = 0; i < dim; ++i)
   {
      E[i] = 0.0;
   }

   std::complex<double> zi = std::complex<double>(0., 1.);
   double k = omega * sqrt(epsilon * mu);
   switch (prob)
   {
      case scatter:
      case lshape:
      case fichera:
      {
         Vector shift(dim);
         shift = 0.0;
         if (prob == fichera) { shift = 1.0; }

         if (dim == 2)
         {
            double x0 = x(0) + shift(0);
            double x1 = x(1) + shift(1);
            double r = sqrt(x0 * x0 + x1 * x1);
            double beta = k * r;

            // Bessel functions
            complex<double> Ho = jn(0, beta) + zi * yn(0, beta);
            complex<double> Ho_r = -k * (jn(1, beta) + zi * yn(1, beta));
            complex<double> Ho_rr = -k * k * (1.0 / beta * (jn(1, beta)
                                                            + zi * yn(1, beta)) - (jn(2, beta) + zi * yn(2, beta)));

            // First derivatives
            double r_x = x0 / r;
            double r_y = x1 / r;
            double r_xy = -(r_x / r) * r_y;
            double r_xx = (1.0 / r) * (1.0 - r_x * r_x);

            std::complex<double> val = 0.25 * zi * Ho;
            std::complex<double> val_xx = 0.25 * zi * (r_xx * Ho_r + r_x * r_x * Ho_rr);
            std::complex<double> val_xy = 0.25 * zi * (r_xy * Ho_r + r_x * r_y * Ho_rr);
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
   std::vector<std::complex<double>> Eval(E.Size());
   maxwell_solution(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].real();
   }
}

void E_exact_Im(const Vector &x, Vector &E)
{
   std::vector<std::complex<double>> Eval(E.Size());
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
      std::vector<std::complex<double>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].real();
      }
   }
}

//define bdr_data solution
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
      std::vector<std::complex<double>> Eval(E.Size());
      maxwell_solution(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].imag();
      }
   }
}

// PML
void CartesianPML::StretchFunction(const Vector &x,
                                   std::vector<std::complex<double>> &dxs)
{
   std::complex<double> zi = std::complex<double>(0., 1.);

   double n = 2.0;
   double c = 5.0;
   double coeff;
   double k = omega * sqrt(epsilon * mu);
   // Stretch in each direction independenly
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_domain_bdr(i, 1))
      {
         coeff = n * c / k / pow(length(i, 1), n);
         dxs[i] = 1.0 + zi * coeff * abs(pow(x(i) - comp_domain_bdr(i, 1), n - 1.0));
      }
      if (x(i) <= comp_domain_bdr(i, 0))
      {
         coeff = n * c / k / pow(length(i, 0), n);
         dxs[i] = 1.0 + zi * coeff * abs(pow(x(i) - comp_domain_bdr(i, 0), n - 1.0));
      }
   }
}

void detJ_JT_J_inv_Re(const Vector &x, CartesianPML * pml ,  DenseMatrix &M)
{
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   M = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      M(i, i) = (det / pow(dxs[i], 2)).real();
   }
}


void detJ_JT_J_inv_Im(const Vector &x, CartesianPML * pml,  DenseMatrix &M)
{
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   M = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      M(i, i) = (det / pow(dxs[i], 2)).imag();
   }
}


void detJ_inv_JT_J_Re(const Vector &x, CartesianPML * pml , DenseMatrix &M)
{
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   // in the 2D case the coefficient is scalar (1/det(J))
   if (dim == 2)
   {
      M = (1.0 / det).real();
   }
   else
   {
      M = 0.0;
      for (int i = 0; i < dim; ++i)
      {
         M(i, i) = (pow(dxs[i], 2) / det).real();
      }
   }
}

void detJ_inv_JT_J_Im(const Vector &x, CartesianPML * pml, DenseMatrix &M)
{
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      M = (1.0 / det).imag();
   }
   else
   {
      M = 0.0;
      for (int i = 0; i < dim; ++i)
      {
         M(i, i) = (pow(dxs[i], 2) / det).imag();
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
   // initialize with any vertex
   for (int i = 0; i < dim; i++)
   {
      dom_bdr(i, 0) = mesh->GetVertex(0)[i];
      dom_bdr(i, 1) = mesh->GetVertex(0)[i];
   }
   // loop through boundary vertices
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      Array<int> bdr_vertices;
      mesh->GetBdrElementVertices(i, bdr_vertices);
      // loop through vertices
      for (int j = 0; j < bdr_vertices.Size(); j++)
      {
         for (int k = 0; k < dim; k++)
         {
            dom_bdr(k, 0) = min(dom_bdr(k, 0), mesh->GetVertex(bdr_vertices[j])[k]);
            dom_bdr(k, 1) = max(dom_bdr(k, 1), mesh->GetVertex(bdr_vertices[j])[k]);
         }
      }
   }
   for (int i = 0; i < dim; i++)
   {
      comp_dom_bdr(i, 0) = dom_bdr(i, 0) + length(i, 0);
      comp_dom_bdr(i, 1) = dom_bdr(i, 1) - length(i, 1);
   }
}

void CartesianPML::SetAttributes(Mesh *mesh_)
{
   int nrelem = mesh_->GetNE();
   // initialize list with 1
   elems.SetSize(nrelem);
   // loop through the elements and identify which of them are in the pml
   for (int i = 0; i < nrelem; ++i)
   {
      elems[i] = 1;
      bool in_pml = false;
      Element *el = mesh_->GetElement(i);
      Array<int> vertices;
      // Initialize Attribute
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();
      // Check if any vertex is in the pml
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
