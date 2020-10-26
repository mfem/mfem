
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifdef _WIN32
#define jn(n, x) _jn(n, x)
#define yn(n, x) _yn(n, x)
#endif

using namespace std;
using namespace mfem;

void maxwell_solution(const Vector &x, vector<complex<double>> &E);

int prob_kind=0;
double L;
double ylim;
// Class for setting up a simple Cartesian PML region
class TorusPML
{
private:
   Mesh *mesh;

   int dim;

   // Length of the PML Region given in radians
   double PmlThicknessAngle;

   double theta0;

   double a,b;// defining the cross-section plane (vertical) /line for the pml

   // Integer Array identifying elements in the PML
   // 0: in the PML, 1: not in the PML
   Array<int> elems;

public:
   // Constructor
   TorusPML(Mesh *mesh_, double length_rad_);

   // Return Markers list for elements
   Array<int> * GetMarkedPMLElements() {return &elems;}

   // Mark elements in the PML region
   void SetAttributes(Mesh *mesh_);

   double PmlBdrPlaneEquation(double x, double theta)
   {
      return tan(theta) * x;
   }   

   // PML complex stretching function
   void StretchFunction(const Vector &x, vector<complex<double>> &dxs);
};

// Class for returning the PML coefficients of the bilinear form
class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   TorusPML * pml = nullptr;
   void (*Function)(const Vector &, TorusPML * , Vector &);
public:
   PMLDiagMatrixCoefficient(int dim, void(*F)(const Vector &, TorusPML *,
                                              Vector &),
                            TorusPML * pml_)
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

void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);

void source(const Vector &x, Vector & f);

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void detJ_JT_J_inv_Re(const Vector &x, TorusPML * pml, Vector &D);
void detJ_JT_J_inv_Im(const Vector &x, TorusPML * pml, Vector &D);
void detJ_JT_J_inv_abs(const Vector &x, TorusPML * pml, Vector &D);

void detJ_inv_JT_J_Re(const Vector &x, TorusPML * pml, Vector &D);
void detJ_inv_JT_J_Im(const Vector &x, TorusPML * pml, Vector &D);
void detJ_inv_JT_J_abs(const Vector &x, TorusPML * pml, Vector &D);

Array2D<double> comp_domain_bdr;
Array2D<double> domain_bdr;

double mu = 1.0;
double epsilon = 1.0;
double omega;
int dim;

int main(int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_SELF, &num_procs);
   MPI_Comm_rank(MPI_COMM_SELF, &myid);
   // 1. Parse command-line options.
   // const char *mesh_file = "torus1_4.mesh";
   // const char *mesh_file = "waveguide-bend2.mesh";
   const char *mesh_file = "waveguide-bend.mesh";

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
   args.AddOption(&prob_kind, "-prob", "--problem-kind",
                  "Problem/mesh choice");                  
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
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);


   switch (prob_kind)
   {
   case 0: 
   {
      mesh_file = "waveguide-bend.mesh"; 
      L = -2.;
      ylim = -3;
   }
   break;
   case 1: 
   {
      mesh_file = "waveguide-bend2.mesh";
      L = -5.;
      ylim = 0.0;
   }
   break;
   case 2: mesh_file = "toroid3_4.mesh"; break;
   default:
      MFEM_ABORT("Not a valid problem choice ");
      break;
   }

   Mesh * mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   mesh->RemoveInternalBoundaries();

   // Angular frequency
   omega = 2.0 * M_PI * freq;

   // 4. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }


   // Setup PMLThickness angle in rad
   double length_rad = 4.*M_PI/2./20.;

   TorusPML * pml = new TorusPML(mesh,length_rad);

   // // Set element attributes in order to distinguish elements in the PML region
   pml->SetAttributes(mesh);


    if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;

      socketstream mesh_sock(vishost, visport);
      mesh_sock.precision(8);
      mesh_sock << "mesh\n"
                  << *mesh << "window_title 'Mesh'" << flush;
      ofstream mesh_ofs("pml_torus.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);                  
   }
   // 6. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   int size = fespace->GetTrueVSize();

   cout << "Number of finite element unknowns: " << size << endl;

   // 7. Essential boundary dofs on the whole boundary.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 1;
   }
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // ess_tdof_list.Print();

   // 8. Setup Complex Operator convention
   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // 9. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   VectorFunctionCoefficient f(dim, source);
   ComplexLinearForm b(fespace, conv);
   // b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f));
   b.Vector::operator=(0.0);
   b.Assemble();

   // 10. Define the solution vector x as a complex finite element grid function
   //     corresponding to fespace.
   ComplexGridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);
   x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);
   // x.ProjectCoefficient(E_Re, E_Im);

   cout << "x.norm = " << x.Norml2() << endl;

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

   // // 13. Solve using a direct or an iterative solver
   // chrono.Clear();
   // chrono.Start();
   // // ComplexUMFPackSolver csolver(*A.As<ComplexSparseMatrix>());
   // UMFPackSolver csolver(*SpMat);
   // csolver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   // csolver.SetPrintLevel(1);
   // csolver.Mult(B, X);
   // chrono.Stop();
   // cout << "UMFPack time = " << chrono.RealTime() << endl;

   // 14. Recover the solution as a finite element grid function and compute the
   //     errors if the exact solution is known.
   a.RecoverFEMSolution(X, b, x);

   // If exact is known compute the error

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      // Define visualization keys for GLVis (see GLVis documentation)
      string keys;
      keys = (dim == 3) ? "keys macF\n" : keys = "keys amrRljcUUuu\n";

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
      int num_frames = 16;
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


   MPI_Finalize();

   return 0;
}

void source(const Vector &x, Vector &f)
{
   Vector center(dim);
   double r = 0.0;
   center = 0.5;
   center(2) = 0.15;

   for (int i = 0; i < dim; ++i)
   {
      r += pow(x[i] - center[i], 2.);
   }
   double n = 5.0 * omega * sqrt(epsilon * mu) / M_PI;
   double coeff = pow(n, 2) / M_PI;
   double alpha = -pow(n, 2) * r;
   f = 0.0;
   f[0] = coeff * exp(alpha);
}

void E_bdr_data_Re(const Vector &x, Vector &E)
{
   E = 0.0;
   if (prob_kind != 2)
   {
      if (x(1) == ylim) 
      {
         vector<complex<double>> Eval(E.Size());
         maxwell_solution(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].real();
         }
      }
   }
   else
   {
      // if (abs(x(1))<1e-12 && abs(x(0) - 7.0) >= 1e-12) 
      if (abs(x(1))<1e-12 && x(0)>0) 
      {
         vector<complex<double>> Eval(E.Size());
         maxwell_solution(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].real();
         }
      }
   }
}

// Define bdr_data solution
void E_bdr_data_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   if (prob_kind != 2)
   {
      if (x(1) == ylim) 
      {
         vector<complex<double>> Eval(E.Size());
         maxwell_solution(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].imag();
         }
      }
   }
   else
   {
      // if (x(1) == 0.0 && x(0)>= 7.0) 
      if (abs(x(1))<1e-12 && x(0)>0) 
      {
         vector<complex<double>> Eval(E.Size());
         maxwell_solution(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].imag();
         }
      }
   }
}

void detJ_JT_J_inv_Re(const Vector &x, TorusPML * pml, Vector &D)
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

void detJ_JT_J_inv_Im(const Vector &x, TorusPML * pml, Vector &D)
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

void detJ_JT_J_inv_abs(const Vector &x, TorusPML * pml, Vector &D)
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

void detJ_inv_JT_J_Re(const Vector &x, TorusPML * pml, Vector &D)
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

void detJ_inv_JT_J_Im(const Vector &x, TorusPML * pml, Vector &D)
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

void detJ_inv_JT_J_abs(const Vector &x, TorusPML * pml, Vector &D)
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

TorusPML::TorusPML(Mesh *mesh_, double length_rad_)
   : mesh(mesh_), PmlThicknessAngle(length_rad_)
{
   dim = mesh->Dimension();
   theta0 = M_PI/2.0 - PmlThicknessAngle;
   a = tan(theta0);
   b = -1.0;
}

void TorusPML::SetAttributes(Mesh *mesh_)
{
   // set pml attribute according to the angle of element center
   // int nrelem = mesh_->GetNE();
   // elems.SetSize(nrelem);

   // // get min and max angle of the mesh (up to centers)
   // double dmin = 2.*M_PI;
   // double dmax = 0.;
   // for (int i = 0; i < nrelem; ++i)
   // {
   //    Vector center;
   //    mesh_->GetElementCenter(i,center);
   //    double x = center[0];
   //    double y = center[1];
   //    double theta = atan(y/x);
   //    int k = 0;
   //    if (x<0)
   //    {
   //       k = 1;
   //    }
   //    else if (y<0)
   //    {
   //       k = 2;
   //    }
   //    theta += k*M_PI;
   //    double thetad = theta * 180.0/M_PI;
   //    dmin = min(dmin,theta);
   //    dmax = max(dmax,theta);
   // }

   // cout << "min angle in degrees = " << dmin * 180. / M_PI << endl;
   // cout << "max angle in degrees = " << dmax * 180. / M_PI << endl;


   // // Loop through the elements and identify which of them are in the PML
   // for (int i = 0; i < nrelem; ++i)
   // {
   //    // initialize with 1
   //    elems[i] = 1;
   //    Element *el = mesh_->GetElement(i);
   //    // Initialize attribute
   //    el->SetAttribute(1);
   //    Vector center;
   //    mesh_->GetElementCenter(i,center);
   //    double x = center[0];
   //    double y = center[1];
   //    double theta = atan(y/x);
   //    int k = 0;
   //    if (x<0)
   //    {
   //       k = 1;
   //    }
   //    else if (y<0)
   //    {
   //       k = 2;
   //    }
   //    theta += k*M_PI;

   //    // Check if the center is in the PML
   //    if (theta > M_PI/2.0 - PmlThicknessAngle)
   //    {
   //       elems[i] = 0;
   //       el->SetAttribute(2);
   //    }
   // }
   // mesh_->SetAttributes();
   // 
   int nrelem = mesh_->GetNE();
   elems.SetSize(nrelem);
   // Loop through the elements and identify which of them are in the PML

   for (int i = 0; i < nrelem; ++i)
   {
      // initialize with 1
      elems[i] = 1;
      Element *el = mesh_->GetElement(i);
      // Initialize attribute
      el->SetAttribute(1);
      Vector center;
      mesh_->GetElementCenter(i,center);
      double x = center[0];
      double y = center[1];
      if (prob_kind !=2)
      {
         if (x < L)
         {
            elems[i] = 0;
            el->SetAttribute(2);
         }
      }
      else
      {
         if (x > -3.0 && y<-7.0)
         {
            elems[i] = 0;
            el->SetAttribute(2);
         }
      }
   }
   mesh_->SetAttributes();
}

void TorusPML::StretchFunction(const Vector &x,
                               vector<complex<double>> &dxs)
{
   complex<double> zi = complex<double>(0., 1.);

   double n = 2.0;
   double c = 5.0;
   double coeff;
   double k = omega * sqrt(epsilon * mu);


   double x0 = x[0];
   double y0 = x[1];
   double th = atan(y0/x0);
   int m = 0;
   if (x0<0)
   {
      m = 1;
   }
   else if (y0<0)
   {
      m = 2;
   }
   th += m*M_PI;

   double thetad = th * 180.0/M_PI;
   // find the distance from the plane defined by a * x + b * y = 0;
   // Point on the plane coords
   Vector xp(2);
   xp(0) = b * (b * x(0) - a * x(1))/(a * a + b * b);
   xp(1) = a * (-b * x(0) + a * x(1))/(a * a + b * b);

   double th0 = atan(xp(1)/xp(0));
   m = 0;
   if (xp(0)<0)
   {
      m = 1;
   }
   else if (xp(1)<0)
   {
      m = 2;
   }
   th0 += m*M_PI;

   double thetad0 = th0 * 180.0/M_PI;
   // if (thetad < thetad0)
   // {
   //    cout << "thetad  = " << thetad << endl;
   //    cout << "thetad0 = " << thetad0 << endl;
   //    cin.get();
   // }

   // Stretch in each direction independently
   coeff = n * c / k / pow(1, n);
   if (prob_kind != 2)
   {
      dxs[0] = 1.0 + zi * coeff *
                  abs(pow(x(0) - L, n - 1.0));
   // dxs[0] = 1.0;
      dxs[1] = 1.0;
      dxs[2] = 1.0;
   }
   else
   {
      dxs[0] = 1.0 + zi * coeff *
                  abs(pow(x(0) + 3.0, n - 1.0));
      dxs[1] = 1.0;
      dxs[2] = 1.0;
   }
}


void maxwell_solution(const Vector &x, vector<complex<double>> &E)
{
   complex<double> zi = complex<double>(0., 1.);
   double k = omega * sqrt(epsilon * mu);
   // T_10 mode
   double k10 = sqrt(k * k - M_PI * M_PI);
   // E[2] = -zi * k / M_PI * sin(M_PI*(x(0)-7.0))*exp(zi * k10 * x(1));
   E[2] = 1.0 + zi;
   // E[1] = -zi * k / M_PI *exp(zi * k10 * x(2));
}