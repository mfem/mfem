//                                MFEM Example multigrid-grid Cycle
//
// Compile with: make mg_maxwellp
//
// Sample runs:  mg_maxwellp -m ../data/one-hex.mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "Schwarz.hpp"

using namespace std;
using namespace mfem;

// #define DEFINITE

// Define exact solution
void E_exact(const Vector & x, Vector & E);
void f_exact(const Vector & x, Vector & f);
void get_maxwell_solution(const Vector & x, double E[], double curl2E[]);

int dim;
double omega;
int isol = 1;


int main(int argc, char *argv[])
{
    // 1. Parse command-line options.
   const char *mesh_file = "../data/one-hex.mesh";
   int order = 1;
   int sdim = 2;
   bool static_cond = false;
   const char *device_config = "cpu";
   bool visualization = true;
   int ref_levels = 1;
   int initref    = 1;
   // number of wavelengths
   double k = 0.5;
   StopWatch chrono;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sdim, "-d", "--dimension", "Dimension");
   args.AddOption(&ref_levels, "-sr", "--serial-refinements", 
                  "Number of mesh refinements");
   args.AddOption(&initref, "-iref", "--init-refinements", 
                  "Number of initial mesh refinements");
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&isol, "-sol", "--solution", 
                  "Exact Solution: 0) Polynomial, 1) Sinusoidal.");               
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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
   args.PrintOptions(cout);

   // Angular frequency
   omega = 2.0 * M_PI * k;

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. 
   // Mesh *mesh = new Mesh(mesh_file, 1, 1);

   Mesh * mesh; 
   // Define a simple square or cubic mesh
   if (sdim == 2)
   {
      mesh = new Mesh(1, 1, Element::QUADRILATERAL, true,1.0, 1.0,false);
      // mesh = new Mesh(1, 1, Element::TRIANGLE, true,1.0, 1.0,false);
   }
   else
   {
      mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true,1.0, 1.0,1.0, false);
   }
   dim = mesh->Dimension();
   for (int i=0; i<initref; i++) {mesh->UniformRefinement();}


   Mesh * cmesh = new Mesh(*mesh);

   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }


   // 4. Define a finite element space on the mesh.
   FiniteElementCollection *fec   = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }


   ConstantCoefficient muinv(1.0);
#ifdef DEFINITE
   ConstantCoefficient sigma(pow(omega, 2));
#else
   ConstantCoefficient sigma(-pow(omega, 2));
#endif
   // 6. Linear form (i.e RHS b = (f,v) = (1,v))
   LinearForm *b = new LinearForm(fespace);
   VectorFunctionCoefficient f(sdim, f_exact);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 7. Bilinear form a(.,.) on the finite element space
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv)); // one is the coeff
   a->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   GridFunction x(fespace);
   x = 0.0;
   VectorFunctionCoefficient E_ex(sdim, E_exact);
   x.ProjectCoefficient(E_ex);

   SparseMatrix A;
   Vector B, X;

   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "Size of linear system: " << A.Height() << endl;

   FiniteElementSpace *prec_fespace = (a->StaticCondensationIsEnabled() ? a->SCFESpace() : fespace);

   chrono.Clear();
   chrono.Start();
   SchwarzSmoother * prec = new SchwarzSmoother(cmesh,ref_levels, prec_fespace, &A, ess_tdof_list);
   prec->SetType(Schwarz::SmootherType::ADDITIVE);
   chrono.Stop();
   // Need to invastigate the time scalings. TODO
   cout << "Smoother construction time " << chrono.RealTime() << "s. \n";
   

   // DSmoother M(A);
   GSSmoother M(A);

   int maxit(1000);
   double rtol(0.0);
   double atol(1.e-8);
   // CGSolver solver;
   GMRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxit);
   solver.SetOperator(A);
   solver.SetPreconditioner(*prec);
   // solver.SetPreconditioner(M);
   solver.SetPrintLevel(1);
   solver.Mult(B,X);

   a->RecoverFEMSolution(X, *b, x);


   GridFunction Egf(fespace);
   Egf.ProjectCoefficient(E_ex);


   int order_quad = max(2, 2 * order + 1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
   double L2Error = x.ComputeL2Error(E_ex, irs);

   double norm_E = ComputeLpNorm(2, E_ex, *mesh, irs);

   cout << "\n || E_h - E || / ||E|| = " << L2Error / norm_E << '\n' << endl;

   // if (visualization)
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock.precision(8);
   //    sol_sock <<  "mesh\n" << *cmesh  << "keys n\n" << flush;
   // }

   //  if (visualization)
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock.precision(8);
   //    sol_sock <<  "mesh\n" << *mesh  << "keys n\n" << flush;
   // }


   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      if (dim == 2) 
      {
         sol_sock <<  "solution\n" << *mesh << x  << "keys rRljc\n" << flush;
      }
      else
      {
         sol_sock <<  "solution\n" << *mesh << x  << "keys lc\n" << flush;
      }
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      if (dim == 2) 
      {
         sol_sock <<  "solution\n" << *mesh << Egf  << "keys rRljc\n" << flush;
      }
      else
      {
         sol_sock <<  "solution\n" << *mesh << Egf  << "keys lc\n" << flush;
      }
   }
   delete a;
   delete b;
   delete fec;
   delete fespace;
   delete mesh;
   return 0;
}


//define exact solution
void E_exact(const Vector &x, Vector &E)
{
   double curl2E[3];
   get_maxwell_solution(x, E, curl2E);
}

//calculate RHS from exact solution
// f = curl (mu curl E ) + omega^2*E
void f_exact(const Vector &x, Vector &f)
{
   double coeff;
#ifdef DEFINITE
   coeff = omega * omega;
#else
   coeff = -omega * omega;
#endif
   double E[3], curl2E[3];
   get_maxwell_solution(x, E, curl2E);
   // curl ( curl E) +/- omega^2 E = f
   f(0) = curl2E[0] + coeff * E[0];
   f(1) = curl2E[1] + coeff * E[1];
   if (dim == 2)
   {
      if (x.Size() == 3) {f(2)=0.0;}
   }
   else
   {
      f(2) = curl2E[2] + coeff * E[2];
   }
}

void get_maxwell_solution(const Vector & x, double E[], double curl2E[])
{
   if (isol == 0) // polynomial
   {
      if (dim == 2)
      {
         E[0] = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]);
         E[1] = 0.0;
         //
         curl2E[0] =  - 2.0 * x[0] * (x[0] - 1.0);
         curl2E[1] = (2.0*x[0]-1.0)*(2.0*x[1]-1);
         curl2E[2] = 0.0;
      }
      else
      {
         // Polynomial vanishing on the boundary
         E[0] = x[1] * x[2]      * (1.0 - x[1]) * (1.0 - x[2]);
         E[1] = x[0] * x[1] * x[2] * (1.0 - x[0]) * (1.0 - x[2]);
         E[2] = x[0] * x[1]      * (1.0 - x[0]) * (1.0 - x[1]);
         //
         curl2E[0] = 2.0 * x[1] * (1.0 - x[1]) - (2.0 * x[0] - 3.0) * x[2] * (1 - x[2]);
         curl2E[1] = 2.0 * x[1] * (x[0] * (1.0 - x[0]) + (1.0 - x[2]) * x[2]);
         curl2E[2] = 2.0 * x[1] * (1.0 - x[1]) + x[0] * (3.0 - 2.0 * x[2]) * (1.0 - x[0]);
      }
   }

   else if (isol == 1) // sinusoidal
   {
      if (dim == 2)
      {
         E[0] = sin(omega * x[1]);
         E[1] = sin(omega * x[0]);
         curl2E[0] = omega * omega * E[0];
         curl2E[1] = omega * omega * E[1];
         curl2E[2] = 0.0;
      }
      else
      {
         E[0] = sin(omega * x[1]);
         E[1] = sin(omega * x[2]);
         E[2] = sin(omega * x[0]);

         curl2E[0] = omega * omega * E[0];
         curl2E[1] = omega * omega * E[1];
         curl2E[2] = omega * omega * E[2];
      }
   }
   else if (isol == 2) //point source
   {
      if (dim == 2)
      {
         // shift to avoid singularity
         double x0 = x(0) + 0.1;
         double x1 = x(1) + 0.1;
         //
         double r = sqrt(x0 * x0 + x1 * x1);

         E[0] = cos(omega * r);
         E[1] = 0.0;

         double r_x = x0 / r;
         double r_y = x1 / r;
         double r_xy = -(r_x / r) * r_y;
         double r_yx = r_xy;
         double r_yy = (1.0 / r) * (1.0 - r_y * r_y);

         curl2E[0] = omega * ((r_yy ) * sin(omega * r) + (omega * r_y * r_y) * cos(omega * r));
         curl2E[1] = -omega * (r_yx * sin(omega * r) + omega * r_y * r_x * cos(omega * r));
         curl2E[2] = 0.0;
      }
      else
      {
      // shift to avoid singularity
         double x0 = x(0) + 0.1;
         double x1 = x(1) + 0.1;
         double x2 = x(2) + 0.1;
         //
         double r = sqrt(x0 * x0 + x1 * x1 + x2 * x2);

         E[0] = cos(omega * r);
         E[1] = 0.0;
         E[2] = 0.0;

         double r_x = x0 / r;
         double r_y = x1 / r;
         double r_z = x2 / r;
         double r_xy = -(r_x / r) * r_y;
         double r_xz = -(r_x / r) * r_z;
         double r_yx = r_xy;
         double r_yy = (1.0 / r) * (1.0 - r_y * r_y);
         double r_zx = r_xz;
         double r_zz = (1.0 / r) * (1.0 - r_z * r_z);

         curl2E[0] = omega * ((r_yy + r_zz) * sin(omega * r) +
                              (omega * r_y * r_y + omega * r_z * r_z) * cos(omega * r));
         curl2E[1] = -omega * (r_yx * sin(omega * r) + omega * r_y * r_x * cos(omega * r));
         curl2E[2] = -omega * (r_zx * sin(omega * r) + omega * r_z * r_x * cos(omega * r));
      }
   }
   else if (isol == 3) // plane wave
   {
      if (dim == 2)
      {
         E[0] = cos(omega * (x(0) + x(1)) / sqrt(2.0));
         E[1] = 0.0;

         curl2E[0] = omega * omega * E[0] / 2.0;
         curl2E[1] = -omega * omega * E[0] / 2.0;
      }
      else
      {
         E[0] = cos(omega * (x(0) + x(1) + x(2)) / sqrt(3.0));
         E[1] = 0.0;
         E[2] = 0.0;

         curl2E[0] = 2.0 * omega * omega * E[0] / 3.0;
         curl2E[1] = -omega * omega * E[0] / 3.0;
         curl2E[2] = -omega * omega * E[0] / 3.0;
      }
   }

}