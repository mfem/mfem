//                       Complex Convergence Test
//
// Compile with: make complex
//
//  ./complex -m ../../data/inline-hex.mesh -o 1 -ref 2 -sol 1 -iprob 1 -no-vis

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "conv_rates.hpp"

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
   bool visualization = true;
   int ref_levels = 1;
   int iprob = 0;
   bool herm_conv = true;

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
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
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

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

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

   Convergence rates_re;
   Convergence rates_im;
   for (int l=0; l<=ref_levels; l++)
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


// #ifndef MFEM_USE_SUITESPARSE
      {
         GSSmoother * S = nullptr;
         Array<int>offsets(3);
         offsets[0] = 0;
         offsets[1] = A->Height()/2;
         offsets[2] = A->Height()/2;
         offsets.PartialSum();
         BlockDiagonalPreconditioner prec(offsets);
         switch (prob_kind)
         {
         case comp:
         case rl:
            S = new GSSmoother(AZ->real());
            break;
         case im:
            S = new GSSmoother(AZ->imag());
            break;
         }
         prec.SetDiagonalBlock(0, S);
         prec.SetDiagonalBlock(1, S);

         GMRESSolver gmres;
         gmres.SetPrintLevel(2);
         gmres.SetMaxIter(2000);
         gmres.SetRelTol(1e-8);
         gmres.SetAbsTol(1e-8);
         gmres.SetPreconditioner(prec);
         gmres.SetOperator(*A);
         gmres.Mult(B, X);
         delete S;
      }
// #else
//       {
//          UMFPackSolver umf_solver;
//          umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
//          umf_solver.SetOperator(*A);
//          umf_solver.Mult(B, X);
//       }
// #endif
      // Recover the solution as a finite element grid function.
      a->RecoverFEMSolution(X, *b, x);

      // Compute relative error
      rates_re.AddGridFunction(&x.real(),&u_ex_re);
      rates_im.AddGridFunction(&x.imag(),&u_ex_im);

      if (l==ref_levels) break;


      mesh->UniformRefinement();
      fespace->Update();
      a->Update();
      b->Update();
      x.Update();
      x_ex.Update();
   }

   rates_re.Print();
   rates_im.Print();

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
   else
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
   else
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
