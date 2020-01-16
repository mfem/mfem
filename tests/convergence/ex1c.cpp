//                                MFEM Example 1 - complex
//
// Compile with: make ex1c
//
//  ./ex1c -m ../../data/inline-hex.mesh -o 1 -ref 4 -sol 1 -iprob 1 -no-vis

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

enum prob {comp,rl,im};
prob prob_kind;
int dim;
int sol=1;

double u_exact_re(const Vector &x);
double u_exact_im(const Vector &x);
void u_grad_exact_re(const Vector &x, Vector &du);
void u_grad_exact_im(const Vector &x, Vector &du);
double f_exact_re(const Vector &x);
double f_exact_im(const Vector &x);
void get_solution_re(const Vector &x, double & u, double du[], double & d2u);
void get_solution_im(const Vector &x, double & u, double du[], double & d2u);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = true;
   int ref_levels = 0;
   int iprob = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sol, "-sol", "--exact",
                  "Exact solution flag - 0:polynomial, 1: plane wave");
   args.AddOption(&iprob, "-iprob", "--iproblem",
                  "Problem kind - 0:complex, 1: purely real, 2: purely imaginary");
   args.AddOption(&ref_levels, "-ref", "--ref_levels",
                  "Number of Refinement Levels.");
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

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // Define a finite element space on the mesh.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   prob_kind = (prob)iprob;
   mfem::ComplexOperator::Convention conv = mfem::ComplexOperator::HERMITIAN;


   // Set up the linear form b(.)
   ComplexLinearForm *b = new ComplexLinearForm(fespace,conv);
   FunctionCoefficient f_re(f_exact_re);
   FunctionCoefficient f_im(f_exact_im);
   b->AddDomainIntegrator(new DomainLFIntegrator(f_re),
                          new DomainLFIntegrator(f_im));
   b->real().Vector::operator=(0.0);
   b->imag().Vector::operator=(0.0);

   // Define the solution vector x as a finite element grid function
   // corresponding to fespace.
   ComplexGridFunction x(fespace);
   ComplexGridFunction x_ex(fespace);
   FunctionCoefficient u_ex_re(u_exact_re);
   FunctionCoefficient u_ex_im(u_exact_im);
   VectorFunctionCoefficient u_grad_coeff_re(dim, u_grad_exact_re);
   VectorFunctionCoefficient u_grad_coeff_im(dim, u_grad_exact_im);
   // Set up the sesquilinear form a(.,.)
   ConstantCoefficient one(1.0);
   SesquilinearForm *a = new SesquilinearForm(fespace,conv);
   switch (prob_kind)
   {
      case comp :
         a->AddDomainIntegrator(new DiffusionIntegrator(one),
                                new DiffusionIntegrator(one));
         break;
      case rl :
         a->AddDomainIntegrator(new DiffusionIntegrator(one), NULL);
         break;
      case im :
         a->AddDomainIntegrator(NULL,new DiffusionIntegrator(one));
         break;
   }

   double L2err_re0, L2err_im0;
   double H1err_re0, H1err_im0;
   for (int l=0; l<ref_levels; l++)
   {
      a->Assemble();
      b->Assemble();
      OperatorHandle Ah;
      Vector B, X;
      // Determine the list of true (i.e. conforming) essential boundary dofs.
      Array<int> ess_bdr, ess_tdof_list;
      if (mesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 1;
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      x.ProjectBdrCoefficient(u_ex_re,u_ex_im,ess_bdr);
      x_ex.ProjectCoefficient(u_ex_re,u_ex_im);

      a->FormLinearSystem(ess_tdof_list, x, *b, Ah, X, B);

      ComplexSparseMatrix * AZ = Ah.As<ComplexSparseMatrix>();
      SparseMatrix * A = AZ->GetSystemMatrix();

      cout << "Size of linear system: " << A->Height() << endl;


#ifndef MFEM_USE_SUITESPARSE
      {
         GMRESSolver gmres;
         gmres.SetPrintLevel(1);
         gmres.SetMaxIter(200);
         gmres.SetRelTol(1e-12);
         gmres.SetAbsTol(0.0);
         gmres.SetOperator(*A);
         gmres.Mult(B, X);
      }
#else
      {
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(*A);
         umf_solver.Mult(B, X);
      }
#endif
      // Recover the solution as a finite element grid function.
      a->RecoverFEMSolution(X, *b, x);

      // Compute relative error
      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      double L2err_re  = x.real().ComputeL2Error(u_ex_re, irs);
      double L2err_im  = x.imag().ComputeL2Error(u_ex_im, irs);

      double H1err_re = x.real().ComputeH1Error(&u_ex_re, &u_grad_coeff_re,
                                                &one, 1.0, 1);
      double H1err_im = x.imag().ComputeH1Error(&u_ex_im, &u_grad_coeff_im,
                                                &one, 1.0, 1);

      double alpha0, beta0;
      double alpha1, beta1;
      if (l==0)
      {
         alpha0 = 0.0; beta0 = 0.0;
         alpha1 = 0.0; beta1 = 0.0;
      }
      else
      {
         alpha0 = log(L2err_re0/L2err_re)/log(2.0);
         beta0  = log(L2err_im0/L2err_im)/log(2.0);
         alpha1 = log(H1err_re0/H1err_re)/log(2.0);
         beta1  = log(H1err_im0/H1err_im)/log(2.0);
      }

      cout << setprecision(3);


      cout << "real: || u_h - u ||_{H^1} = " << scientific
           << H1err_re << ",  rate: " << fixed  << alpha1 << endl;
      cout << "imag: || u_h - u ||_{H^1} = " << scientific
           << H1err_im << ",  rate: " << fixed  << beta1 << endl;

      cout << "real: || u_h - u ||_{L^2} = " << scientific
           << L2err_re <<
           ",  rate: " << fixed << alpha0 << endl;
      cout << "imag: || u_h - u ||_{L^2} = " << scientific
           << L2err_im <<
           ",  rate: " << fixed << beta0 << endl;
      cout << endl;

      L2err_re0 = L2err_re;
      L2err_im0 = L2err_im;

      H1err_re0 = H1err_re;
      H1err_im0 = H1err_im;

      if (l == ref_levels-1) { break; }

      mesh->UniformRefinement();
      fespace->Update();
      a->Update();
      b->Update();
      x.Update();
      x_ex.Update();
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock_re(vishost, visport);
      sol_sock_re.precision(8);
      sol_sock_re << "solution\n" << *mesh << x.real() << "keys cm\n"
                  << "window_title 'Real Part' " << flush;
      socketstream sol_sock_im(vishost, visport);
      sol_sock_im.precision(8);
      sol_sock_im << "solution\n" << *mesh << x.imag() << "keys cm\n"
                  << "window_title 'Imag Part' " << flush;
   }

   // 15. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;
   return 0;
}

double u_exact_re(const Vector &x)
{
   double u_r, d2u_r;
   double du_r[dim];
   get_solution_re(x, u_r, du_r, d2u_r);
   return u_r;
}

void u_grad_exact_re(const Vector &x, Vector &du)
{
   du.SetSize(x.Size());
   double u_r, d2u_r;
   double *du_r = du.GetData();
   get_solution_re(x, u_r, du_r, d2u_r);
}


double f_exact_re(const Vector &x)
{
   double u_r, d2u_r, u_i, d2u_i, f;
   double du_r[dim];
   double du_i[dim];
   get_solution_re(x, u_r, du_r, d2u_r);
   get_solution_im(x, u_i, du_i, d2u_i);
   switch (prob_kind)
   {
      case comp : f = -d2u_r + d2u_i; break;
      case rl   : f = -d2u_r; break;
      case im   : f =  d2u_i; break;
   }
   return f;
}

void get_solution_re(const Vector &x, double & u, double du[], double & d2u)
{
   if (sol==0)
   {
      if (dim == 2)
      {
         u = x[1] * (1.0 - x[1])* x[0] * (1.0 - x[0])+3.0;
         du[0] = (1.0 - 2.0 *x[0]) * x[1]*(1.0 - x[1]);
         du[1] = (1.0 - 2.0 *x[1]) * x[0]*(1.0 - x[0]);
         d2u = - 2.0 * x[1] * (1.0 - x[1]) - 2.0 * x[0] * (1.0 - x[0]);
      }
      else
      {
         u = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]) + 3.0;
         du[0] = (1.0 - 2.0 *x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]);
         du[1] = (1.0 - 2.0 *x[1]) * x[0]*(1.0 - x[0]) * x[2]*(1.0 - x[2]);
         du[2] = (1.0 - 2.0 *x[2]) * x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]);
         d2u = -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[1]) * x[1]
               -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[2]) * x[2]
               -2.0*(-1.0 + x[1]) * x[1] * (-1.0 + x[2]) * x[2];
      }
   }
   else if (sol == 1)
   {
      double alpha;
      if (dim == 2)
      {
         alpha = 10.0/sqrt(2);
         u = cos(alpha * ( x(0) + x(1) ) );
         du[0] = -alpha * sin(alpha * ( x(0) + x(1) ) );
         du[1] = du[0];
         d2u = -2.0 * alpha * alpha * u;
      }
      else
      {
         alpha = 10.0/sqrt(3);
         u = cos(alpha * ( x(0) + x(1) + x(2) ) );
         du[0] = -alpha * sin(alpha * ( x(0) + x(1) + x(2) ) );
         du[1] = du[0];
         du[2] = du[0];
         d2u = -3.0 * alpha * alpha * u;
      }
   }
}

double u_exact_im(const Vector &x)
{
   double u, d2u;
   double du[dim];
   get_solution_im(x, u, du, d2u);
   return u;
}

void u_grad_exact_im(const Vector &x, Vector &du)
{
   du.SetSize(x.Size());
   double u_i, d2u_i;
   double *du_i = du.GetData();
   get_solution_im(x, u_i, du_i, d2u_i);
}

double f_exact_im(const Vector &x)
{
   double u_r, d2u_r, u_i, d2u_i, f;
   double du_r[dim];
   double du_i[dim];
   get_solution_re(x, u_r, du_r, d2u_r);
   get_solution_im(x, u_i, du_i, d2u_i);
   switch (prob_kind)
   {
      case comp : f = -d2u_r - d2u_i; break;
      case rl   : f = -d2u_i; break;
      case im   : f = -d2u_r; break;
   }
   return f;
}

void get_solution_im(const Vector &x, double & u, double du[], double & d2u)
{
   if (sol==0)
   {
      if (dim == 2)
      {
         u = x[1] * (1.0 - x[1])* x[0] * (1.0 - x[0])+2.0;
         du[0] = (1.0 - 2.0 *x[0]) * x[1]*(1.0 - x[1]);
         du[1] = (1.0 - 2.0 *x[1]) * x[0]*(1.0 - x[0]);
         d2u = - 2.0 * x[1] * (1.0 - x[1]) - 2.0 * x[0] * (1.0 - x[0]);
      }
      else
      {
         u = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]) + 2.0;
         du[0] = (1.0 - 2.0 *x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]);
         du[1] = (1.0 - 2.0 *x[1]) * x[0]*(1.0 - x[0]) * x[2]*(1.0 - x[2]);
         du[2] = (1.0 - 2.0 *x[2]) * x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]);
         d2u = -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[1]) * x[1]
               -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[2]) * x[2]
               -2.0*(-1.0 + x[1]) * x[1] * (-1.0 + x[2]) * x[2];
      }
   }
   else if (sol == 1)
   {
      double alpha;
      if (dim == 2)
      {
         alpha = 10.0/sqrt(2);
         u = -sin(alpha * ( x(0) + x(1) ) );
         du[0] = -alpha * cos(alpha * ( x(0) + x(1) ) );
         du[1] = du[0];
         d2u = -2.0 * alpha * alpha * u;
      }
      else
      {
         alpha = 10.0/sqrt(3);
         u = -sin(alpha * ( x(0) + x(1) + x(2) ) );
         du[0] = -alpha * cos(alpha * ( x(0) + x(1) + x(2) ) );
         du[1] = du[0];
         du[2] = du[0];
         d2u = -3.0 * alpha * alpha * u;
      }
   }
}

