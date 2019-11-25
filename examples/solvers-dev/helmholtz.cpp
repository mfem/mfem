//                                MFEM Example 1
//
// Compile with: make helmholtz
//

#include "mfem.hpp"
#include "as/schwarz.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void get_solution(const Vector &x, double & u, double & d2u);
double u_exact(const Vector &x);
double f_exact(const Vector &x);

int isol=0;
int dim;
double omega;

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
   double theta = 0.5;
   double smth_maxit = 1;
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
   args.AddOption(&smth_maxit, "-sm", "--smoother-maxit",
                  "Number of smoothing steps.");               
   args.AddOption(&theta, "-th", "--theta",
                  "Dumping parameter for the smoother.");               
   args.AddOption(&isol, "-sol", "--solution", 
                  "Exact Solution: 0) Polynomial, 1) Sinusoidal.");               
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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


   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }


   // 7. Set up the linear form b(.) 
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   FunctionCoefficient f(f_exact);
   b->AddDomainIntegrator(new DomainLFIntegrator(f));
   // b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   GridFunction x(fespace);
   x = 0.0;
   FunctionCoefficient u_ex(u_exact);
   x.ProjectCoefficient(u_ex);

   // 9. Set up the bilinear form a(.,.) 
   ConstantCoefficient sigma(-pow(omega, 2));

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddDomainIntegrator(new MassIntegrator(sigma));
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "Size of linear system: " << A.Height() << endl;

   FiniteElementSpace *prec_fespace = (a->StaticCondensationIsEnabled() ? a->SCFESpace() : fespace);
   
   chrono.Clear();
   chrono.Start();
   SchwarzSmoother * prec = new SchwarzSmoother(cmesh,ref_levels, prec_fespace, &A, ess_bdr);
   prec->SetType(Schwarz::SmootherType::ADDITIVE);
   prec->SetNumSmoothSteps(smth_maxit);
   prec->SetDumpingParam(theta);
   chrono.Stop();

   // Need to invasticate the time scalings. TODO
   cout << "Smoother construction time " << chrono.RealTime() << "s. \n";
   

   // DSmoother M(A);
   // GSSmoother M(A);

   int maxit(1000);
   double rtol(0.0);
   double atol(1.e-6);
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


   GridFunction ugf(fespace);
   ugf.ProjectCoefficient(u_ex);


   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double L2error = x.ComputeL2Error(u_ex);
      
   cout << " || u_h - u ||_{L^2} = " << L2error <<  endl;



   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      socketstream ex_sock(vishost, visport);
      ex_sock.precision(8);
      if (dim == 2) 
      {
         sol_sock <<  "solution\n" << *mesh << x  << "keys rRljc\n" << flush;
         ex_sock <<  "solution\n" << *mesh << ugf  << "keys rRljc\n" << flush;
      }
      else
      {
         sol_sock <<  "solution\n" << *mesh << x  << "keys lc\n" << flush;
         ex_sock <<  "solution\n" << *mesh << ugf  << "keys lc\n" << flush;
      }
   }




   // if (visualization)
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock.precision(8);
   //    sol_sock << "mesh\n" << *cmesh << flush;
   // }

   // if (visualization)
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock.precision(8);
   //    sol_sock << "mesh\n" << *mesh << flush;
   // }


   // 15. Free the used memory.
   delete a;
   delete b;
   delete fec;
   delete fespace;
   delete mesh;
   return 0;
}


void get_solution(const Vector &x, double & u, double & d2u)
{

   if (dim == 2)
   {
      if (isol == 0)
      {
         u = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]);
         d2u = -2.0* ( x[1]*(1.0 - x[1]) + x[0]*(1.0 - x[0])); 
      }
      else if (isol == 1)
      {  // Point source
         //shift to avoid singularity
         double x0 = x(0) + 0.1;
         double x1 = x(1) + 0.1;
         //
         double r = sqrt(x0 * x0 + x1 * x1);

         u = cos(omega * r);

         double r_x = x0 / r;
         double r_y = x1 / r;
         double r_xx = (1.0 / r) * (1.0 - r_x * r_x);
         double r_yy = (1.0 / r) * (1.0 - r_y * r_y);

         double u_xx = - omega * omega * u * r_x * r_x - omega * sin(omega * r) * r_xx;
         double u_yy = - omega * omega * u * r_y * r_y - omega * sin(omega * r) * r_yy;

         d2u = u_xx + u_yy;
      }
      else
      {
         double alpha = omega / sqrt(2.0);
         u = cos(alpha * (x[0] + x[1]));
         d2u = -2.0* alpha * alpha * u;
      }
   }
   else
   {
      if (isol == 0)
      {
         u = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]);
         d2u = -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[1]) * x[1] 
            -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[2]) * x[2] 
            -2.0*(-1.0 + x[1]) * x[1] * (-1.0 + x[2]) * x[2];
      }
      else if (isol == 1)
      {  // Point source
         // shift to avoid singularity
         double x0 = x(0) + 0.1;
         double x1 = x(1) + 0.1;
         double x2 = x(2) + 0.1;
         //
         double r = sqrt(x0 * x0 + x1 * x1 + x2 * x2);

         u = cos(omega * r);

         double r_x = x0 / r;
         double r_y = x1 / r;
         double r_z = x2 / r;
         double r_xx = (1.0 / r) * (1.0 - r_x * r_x);
         double r_yy = (1.0 / r) * (1.0 - r_y * r_y);
         double r_zz = (1.0 / r) * (1.0 - r_z * r_z);

         double u_xx = - omega * omega * u * r_x * r_x - omega * sin(omega * r) * r_xx;
         double u_yy = - omega * omega * u * r_y * r_y - omega * sin(omega * r) * r_yy;
         double u_zz = - omega * omega * u * r_z * r_z - omega * sin(omega * r) * r_zz;

         d2u = u_xx + u_yy + u_zz;
      }
      else
      {
         double alpha = omega / sqrt(3.0);
         u = cos(alpha * (x[0] + x[1] + x[2]));
         d2u = -3.0* alpha * alpha * u;
      }
   }
}

double u_exact(const Vector &x)
{
   double u, d2u;
   get_solution(x, u, d2u);
   return u;
}

double f_exact(const Vector &x)
{
   double u, d2u;
   get_solution(x, u, d2u);
   // return -d2u;
   return -d2u - omega*omega * u;
}