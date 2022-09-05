//                                MFEM Obstacle Problem
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double spherical_obstacle(const Vector &pt);
void spherical_obstacle_gradient(const Vector &pt, Vector &grad);
double exact_solution_obstacle(const Vector &pt);
double exact_solution_biactivity(const Vector &pt);
double load_biactivity(const Vector &pt);

class LogarithmGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   Coefficient *obstacle;
   double min_val;

public:
   LogarithmGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_, double min_val_=-1e6)
      : u(&u_), obstacle(&obst_), min_val(min_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ExponentialGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   Coefficient *obstacle;
   double min_val;
   double max_val;

public:
   ExponentialGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_, double min_val_=1e-6, double max_val_=1e6)
      : u(&u_), obstacle(&obst_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "./disk.mesh";
   int order = 1;
   bool visualization = true;
   int max_it = 1;
   int ref_levels = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  "isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   mesh.SetCurvature(2);
   mesh.EnsureNCMesh();
   mesh.RandomRefinement(0.5);

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Need help for Socratis to use AMR

   H1_FECollection H1fec(order, dim);
   ParFiniteElementSpace H1fes(&pmesh, &H1fec);

   L2_FECollection L2fec(order-1, dim);
   ParFiniteElementSpace L2fes(&pmesh, &L2fec);

   cout << "Number of finite element unknowns: "
        << H1fes.GetTrueVSize() 
        << " "
        << L2fes.GetTrueVSize() << endl;

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = H1fes.GetVSize();
   offsets[2] = L2fes.GetVSize();
   offsets.PartialSum();

   Array<int> toffsets(3);
   toffsets[0] = 0;
   toffsets[1] = H1fes.GetTrueVSize();
   toffsets[2] = L2fes.GetTrueVSize();
   toffsets.PartialSum();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   if (pmesh.bdr_attributes.Size())
   {
      // ess_bdr = 0;
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   auto sol_func = [](const Vector &x)
   {
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      return val + 2.0;
   };

   auto rhs_func = [](const Vector &x)
   {
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      return (x.Size()*pow(M_PI,2)) * val  + log(val + 2.0);
   };

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   BlockVector tx(toffsets), trhs(toffsets);
   tx = 0.0; trhs = 0.0;

   ParGridFunction u_gf, delta_psi_gf;
   u_gf.MakeRef(&H1fes,x.GetBlock(0));
   delta_psi_gf.MakeRef(&L2fes,x.GetBlock(1));
   delta_psi_gf = 0.0;

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.

   auto IC_func = [](const Vector &x)
   {
      double r0 = 1.0;
      double rr = 0.0;
      for (int i=0; i<x.Size(); i++)
      {
         rr += x(i)*x(i);
      }
      return r0*r0 - rr;
   };
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction u_old_gf(&H1fes);
   ParGridFunction psi_old_gf(&L2fes);
   ParGridFunction psi_gf(&L2fes);
   u_old_gf = 0.0;
   psi_old_gf = 0.0;

   /////////// Example 1   
   // FunctionCoefficient f(rhs_func);
   // FunctionCoefficient IC_coef(IC_func);
   // ConstantCoefficient bdry_coef(0.1);
   // ConstantCoefficient obstacle(0.0);
   // SumCoefficient bdry_funcoef(bdry_coef, IC_coef);
   // u_gf.ProjectCoefficient(bdry_funcoef);
   // double alpha0 = 0.1;

   /////////// Example 2
   // FunctionCoefficient IC_coef(IC_func);
   // ConstantCoefficient f(0.0);
   // FunctionCoefficient obstacle(spherical_obstacle);
   // u_gf.ProjectCoefficient(IC_coef);
   // u_old_gf = u_gf;
   // double alpha0 = 1.0;

   /////////// Example 3
   u_gf = 0.5;
   FunctionCoefficient f(load_biactivity);
   FunctionCoefficient bdry_coef(exact_solution_biactivity);
   ConstantCoefficient obstacle(0.0);
   u_gf.ProjectBdrCoefficient(bdry_coef, ess_bdr);
   double alpha0 = 1.0;

   /////////// Newton TEST
   // FunctionCoefficient f(rhs_func);
   // ConstantCoefficient obstacle(0.0);
   // FunctionCoefficient sol(sol_func);
   // u_gf.ProjectCoefficient(sol);
   // u_old_gf = 0.0;
   // double alpha0 = 1.0;

   LogarithmGridFunctionCoefficient ln_u(u_gf, obstacle);
   psi_gf.ProjectCoefficient(ln_u);
   psi_old_gf = psi_gf;
   // psi_old_gf = 0.0;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   // socketstream psi_sock(vishost, visport);
   // psi_sock.precision(8);

   // 12. Iterate
   int k;
   int total_iterations = 0;
   double tol = 1e-3;
   double increment_u = 0.1;
   double comp;
   double entropy;

   for (k = 0; k < max_it; k++)
   {
      // double alpha = alpha0 / sqrt(k+1);
      double alpha = alpha0 * sqrt(k+1);
      // double alpha = alpha0;
      // alpha *= 2;

      ParGridFunction u_tmp(&H1fes);
      u_tmp = u_old_gf;

      mfem::out << "\nOUTER ITERATION " << k+1 << endl;

      int j;
      for ( j = 0; j < 15; j++)
      {
         total_iterations++;
         // A. Assembly
         
         // // MD
         // double c1 = 1.0;
         // double c2 = 1.0 - alpha;

         // // IMD
         double c1 = 1.0 + alpha;
         double c2 = 1.0;

         ConstantCoefficient c1_cf(c1);

         ParLinearForm b0,b1;
         b0.Update(&H1fes,rhs.GetBlock(0),0);
         b1.Update(&L2fes,rhs.GetBlock(1),0);

         ExponentialGridFunctionCoefficient exp_psi(psi_gf, zero);
         ProductCoefficient neg_exp_psi(-1.0,exp_psi);
         GradientGridFunctionCoefficient grad_u_old(&u_old_gf);
         ScalarVectorProductCoefficient c2_grad_u_old(c2, grad_u_old);
         ProductCoefficient alpha_f(alpha, f);
         GridFunctionCoefficient psi_cf(&psi_gf);
         GridFunctionCoefficient psi_old_cf(&psi_old_gf);
         SumCoefficient psi_old_minus_psi(psi_old_cf, psi_cf, 1.0, -1.0);

         b0.AddDomainIntegrator(new DomainLFIntegrator(alpha_f));
         b0.AddDomainIntegrator(new DomainLFGradIntegrator(c2_grad_u_old));
         b0.AddDomainIntegrator(new DomainLFIntegrator(psi_old_minus_psi));
         b0.Assemble();

         b1.AddDomainIntegrator(new DomainLFIntegrator(exp_psi));
         b1.AddDomainIntegrator(new DomainLFIntegrator(obstacle));
         b1.Assemble();

         ParBilinearForm a00(&H1fes);
         a00.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
         a00.AddDomainIntegrator(new DiffusionIntegrator(c1_cf));
         a00.Assemble();
         HypreParMatrix A00;
         a00.FormLinearSystem(ess_tdof_list, x.GetBlock(0), rhs.GetBlock(0), 
                              A00, tx.GetBlock(0), trhs.GetBlock(0));


         ParMixedBilinearForm a10(&H1fes,&L2fes);
         a10.AddDomainIntegrator(new MixedScalarMassIntegrator());
         a10.Assemble();
         HypreParMatrix A10;
         Array<int> empty;
         a10.FormRectangularLinearSystem(ess_tdof_list, empty, x.GetBlock(0), rhs.GetBlock(1), 
                                         A10, tx.GetBlock(0), trhs.GetBlock(1));

         HypreParMatrix &A01 = *A10.Transpose();

         ParBilinearForm a11(&L2fes);
         a11.AddDomainIntegrator(new MassIntegrator(neg_exp_psi));
         a11.Assemble();
         a11.Finalize();
         HypreParMatrix &A11 = *a11.ParallelAssemble();

         BlockOperator A(toffsets);
         A.SetBlock(0,0,&A00);
         A.SetBlock(1,0,&A10);
         A.SetBlock(0,1,&A01);
         A.SetBlock(1,1,&A11);


         Array2D<HypreParMatrix *> BlockA(2,2);
         BlockA(0,0) = &A00;
         BlockA(0,1) = &A01;
         BlockA(1,0) = &A10;
         BlockA(1,1) = &A11;
         HypreParMatrix * Ah = HypreParMatrixFromBlocks(BlockA);
         
         MUMPSSolver mumps;
         mumps.SetPrintLevel(0);
         mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
         mumps.SetOperator(*Ah);
         mumps.Mult(trhs,tx);
         delete Ah;

         // iterative solver
         // BlockDiagonalPreconditioner prec(toffsets);
         // prec.SetDiagonalBlock(0,new HypreBoomerAMG(A00));
         // prec.SetDiagonalBlock(1,new HypreSmoother(A11));

         // GMRESSolver gmres(MPI_COMM_WORLD);
         // gmres.SetRelTol(1e-12);
         // gmres.SetPrintLevel(1);
         // gmres.SetMaxIter(2000);
         // gmres.SetKDim(50);
         // gmres.SetOperator(A);
         // // gmres.SetPreconditioner(prec);
         // gmres.Mult(trhs,tx);


         u_gf.SetFromTrueDofs(tx.GetBlock(0));
         delta_psi_gf.SetFromTrueDofs(tx.GetBlock(1));

         u_tmp -= u_gf;
         double Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << pmesh << u_gf << "window_title 'Discrete solution'" << flush;
         mfem::out << endl;
         // psi_sock << "solution\n" << mesh << delta_psi_gf << "window_title 'delta psi'" << flush;

         double gamma = 1.0;
         // double gamma = 1.0;
         delta_psi_gf *= gamma;
         psi_gf += delta_psi_gf;
         
         mfem::out << "Newton_update_size = " << Newton_update_size << endl;

         // double update_tol = 1e-10;
         if (Newton_update_size < increment_u)
         // if (Newton_update_size < increment_u/10.0)
         {
            break;
         }
      }
      mfem::out << "Number of Newton iterations = " << j+1 << endl;
      
      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero);

      mfem::out << "|| u_h - u_h_prvs || = " << increment_u << endl;

      delta_psi_gf = psi_gf;
      delta_psi_gf -= psi_old_gf;
      delta_psi_gf = 0.0;

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;

      {
         // CHECK COMPLIMENTARITY | a(u_h, \phi_h - u_h) - (f, \phi_h - u_h) | < tol.
         // TODO: Need to check this with Socratis

         ParLinearForm b(&H1fes);
         b.AddDomainIntegrator(new DomainLFIntegrator(f));
         b.Assemble();

         ParBilinearForm a(&H1fes);
         a.AddDomainIntegrator(new DiffusionIntegrator());
         a.Assemble();
         a.EliminateEssentialBC(ess_bdr, u_gf, b, mfem::Operator::DIAG_ONE);
         a.Finalize();

         ParGridFunction obstacle_gf(&H1fes);
         obstacle_gf.ProjectCoefficient(obstacle);
         obstacle_gf -= u_gf;
         
         comp = a.InnerProduct(u_gf, obstacle_gf);
         comp -= b(obstacle_gf);
         comp = abs(comp);
         mfem::out << "|< phi - u_h, A u_h - f >| = " << comp << endl;


         // Compute entropy: (u - phi ,1-ln(u - phi)) - || \nabla u ||^2
         // TODO: Talk to Socratis about a better way to do this

         // LogarithmGridFunctionCoefficient ln_u(u_gf, obstacle);
         // GridFunction ln_u_gf(&H1fes);
         // ln_u_gf.ProjectCoefficient(ln_u);
         // ln_u_gf -= 1.0;

         // entropy = -( a.InnerProduct(u_gf, u_gf) );
         // entropy = -( m.InnerProduct(u_gf, ln_u_gf) );
         // mfem::out << "entropy = " << entropy << endl;

         // Compute entropy: D(u_h, u_h_prvs) = (u_h ln(u_h) - u_h ln(u_h_prvs), 1) + (u_)
          
      }

      // sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'" << flush;


      // 14. Send the solution by socket to a GLVis server.
      // u_old = u;
      // ExponentialGridFunctionCoefficient exp_psi(psi, obstacle);
      // u.ProjectCoefficient(exp_psi);
      // sol_sock << "solution\n" << mesh << exp_psi << "window_title 'Discrete solution'" << flush;
      // sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'" << flush;
      
      if (comp < tol)
      {
         break;
      }
   }

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << endl;

   // 14. Exact solution.
   if (visualization)
   {
      socketstream err_sock(vishost, visport);
      err_sock.precision(8);
      FunctionCoefficient exact_coef(exact_solution_biactivity);
      // FunctionCoefficient exact_coef(exact_solution_obstacle);

      ParGridFunction error(&H1fes);
      error = 0.0;
      error.ProjectCoefficient(exact_coef);
      error -= u_gf;

      mfem::out << "\n Final L2-error (|| u - u_h||) = " << error.ComputeL2Error(zero) << endl;

      err_sock << "parallel " << num_procs << " " << myid << "\n";
      err_sock << "solution\n" << pmesh << error << "window_title 'Error'"  << flush;
   }
   
   return 0;
}

double LogarithmGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip) - obstacle->Eval(T, ip);
   return max(min_val, log(val));
}

double ExponentialGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, exp(val) + obstacle->Eval(T, ip)));
}

double spherical_obstacle(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double beta = 0.9;

   double b = r0*beta;
   double tmp = sqrt(r0*r0 - b*b);
   double B = tmp + b*b/tmp;
   double C = -b/tmp;

   if (r > b)
   {
      return B + r * C;
   }
   else
   {
      return sqrt(r0*r0 - r*r);
   }
}

void spherical_obstacle_gradient(const Vector &pt, Vector &grad)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double beta = 0.9;

   double b = r0*beta;
   double tmp = sqrt(r0*r0-b*b);
   double C = -b/tmp;

   if (r > b)
   {
      grad(0) = C * x / r;
      grad(1) = C * y / r;
   }
   else
   {
      grad(0) = - x / sqrt( r0*r0 - r*r );
      grad(1) = - y / sqrt( r0*r0 - r*r );
   }
}

double exact_solution_obstacle(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double a =  0.348982574111686;
   double A = -0.340129705945858;

   if (r > a)
   {
      return A * log(r);
   }
   else
   {
      return sqrt(r0*r0-r*r);
   }
}

double exact_solution_biactivity(const Vector &pt)
{
   double x = pt(0);

   if (x > 0.0)
   {
      return x*x;
   }
   else
   {
      return 0.0;
   }
}

double load_biactivity(const Vector &pt)
{
   double x = pt(0);

   if (x > 0.0)
   {
      return -2.0;
   }
   else
   {
      return 0.0;
   }
}

// double IC_biactivity(const Vector &pt)
// {
//    double x = pt(0);
//    return x*x;
// }