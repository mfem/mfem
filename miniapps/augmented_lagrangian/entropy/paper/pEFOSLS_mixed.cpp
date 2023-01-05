//                                MFEM EFOSLS
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

Vector beta;
double eps = 1e-2;
double Ramp_BC(const Vector &pt);
double EJ_exact_solution(const Vector &pt);

double lnit(double x)
{
   double tol = 1e-12;
   x = min(max(tol,x),1.0-tol);
   // MFEM_ASSERT(x>0.0, "Argument must be > 0");
   // MFEM_ASSERT(x<1.0, "Argument must be < 1");
   return log(x/(1.0-x));
}

double expit(double x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+exp(-x));
   }
   else
   {
      return exp(x)/(1.0+exp(x));
   }
}

double dexpitdx(double x)
{
   double tmp = expit(-x);
   return tmp - pow(tmp,2);
}

class LnitGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double alpha;
   double min_val;
   double max_val;

public:
   LnitGridFunctionCoefficient(GridFunction &u_, double alpha_=1.0, double min_val_=-1e1, double max_val_=1e1)
      : u(&u_), alpha(alpha_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ExpitGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double alpha;
   double min_val;
   double max_val;

public:
   ExpitGridFunctionCoefficient(GridFunction &u_, double alpha_=1.0, double min_val_=0.0, double max_val_=1.0)
      : u(&u_), alpha(alpha_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class dExpitdxGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double alpha;
   double min_val;
   double max_val;

public:
   dExpitdxGridFunctionCoefficient(GridFunction &u_, double alpha_=1.0, double min_val_=0.0, double max_val_=1.0)
      : u(&u_), alpha(alpha_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();
   // 1. Parse command-line options.
   const char *mesh_file = "../../../../data/inline-quad.mesh";
   int order = 1;
   bool visualization = true;
   int max_it = 5;
   int ref_levels = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
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

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   H1_FECollection H1fec(order, dim);
   ParFiniteElementSpace H1fes(&pmesh, &H1fec);

   RT_FECollection RTfec(order, dim);
   ParFiniteElementSpace RTfes(&pmesh, &RTfec);

   H1_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2fes(&pmesh, &L2fec);

   cout << "Number of finite element unknowns: "
        << H1fes.GetTrueVSize()
        << " "
        << RTfes.GetTrueVSize()
        << " "
        << L2fes.GetTrueVSize() << endl;

   Array<int> offsets(4);
   offsets[0] = 0;
   offsets[1] = H1fes.GetVSize();
   offsets[2] = RTfes.GetVSize();
   offsets[3] = L2fes.GetVSize();
   offsets.PartialSum();

   Array<int> toffsets(4);
   toffsets[0] = 0;
   toffsets[1] = H1fes.GetTrueVSize();
   toffsets[2] = RTfes.GetTrueVSize();
   toffsets[3] = L2fes.GetTrueVSize();
   toffsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   BlockVector tx(toffsets), trhs(toffsets);
   tx = 0.0; trhs = 0.0;

   Array<int> empty;
   Array<int> ess_tdof_list;
   Array<int> ess_tdof_list_L2;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   if (pmesh.bdr_attributes.Size())
   {
      // ess_bdr = 0;
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      if (L2fes.Conforming()) { L2fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_L2); }
   }

   auto exact_func = [](const Vector &x)
   {
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(2.0*M_PI*x(i));
      }
      return val / 2.0 + 0.5;
   };
   auto rhs_func = [](const Vector &x)
   {
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(2.0*M_PI*x(i));
      }
      Vector gradient(2);
      gradient(0) = 2.0*M_PI * cos(2.0*M_PI*x(0))*sin(2.0*M_PI*x(1));
      gradient(1) = 2.0*M_PI * sin(2.0*M_PI*x(0))*cos(2.0*M_PI*x(1));
      return (eps * x.Size() * pow(2.0*M_PI,2) * val + beta*gradient) / 2.0;
   };
   auto perturbation_func = [](const Vector &x)
   {
      double scale = 5e-1;
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      return 1.0 + scale * pow(val, 3.0);
   };
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   Vector zero_vec(dim);
   zero_vec = 0.0;

   ParGridFunction u_gf, q_gf, delta_psi_gf;
   u_gf.MakeRef(&H1fes,x.GetBlock(0));
   q_gf.MakeRef(&RTfes,x.GetBlock(1));
   delta_psi_gf.MakeRef(&L2fes,x.GetBlock(2));
   delta_psi_gf = 0.0;

   ParGridFunction u_old_gf(&H1fes);
   ParGridFunction psi_old_gf(&L2fes);
   ParGridFunction psi_gf(&L2fes);
   u_old_gf = 0.0;
   psi_old_gf = 0.0;

   /////////// Example 1   
   // u_gf = 0.5;
   // beta.SetSize(dim);
   // beta(0) = 1.0;
   // beta(1) = 1.0;
   // FunctionCoefficient f_cf(rhs_func);
   // ConstantCoefficient bdry_coef(0.5);
   // u_gf.ProjectBdrCoefficient(bdry_coef, ess_bdr);
   // double alpha0 = 1.0;

   // /////////// Example 2
   u_gf = 0.5;
   beta.SetSize(dim);
   beta(0) = 1.0;
   beta(1) = 0.5; 
   beta /= sqrt(1.25);
   // double eps = 1.0;
   ConstantCoefficient f_cf(0.0);
   FunctionCoefficient bdry_coef(Ramp_BC);
   double alpha0 = 0.1;
   u_gf.ProjectBdrCoefficient(bdry_coef, ess_bdr);

   // /////////// Example 3
   // u_gf = 0.5;
   // beta.SetSize(dim);
   // beta(0) = 1.0;
   // beta(1) = 0.0;
   // ConstantCoefficient f_cf(0.0);
   // FunctionCoefficient bdry_coef(EJ_exact_solution);
   // double alpha0 = 0.1;
   // // double alpha0 = 0.01;
   // // FunctionCoefficient perturbation(perturbation_func);
   // // ProductCoefficient IC_coeff(bdry_coef, perturbation);
   // // u.ProjectCoefficient(IC_coeff);
   // u_gf.ProjectCoefficient(bdry_coef);

   // u.ProjectBdrCoefficient(bdry_coef, ess_bdr);

   LnitGridFunctionCoefficient lnit_u(u_gf);
   psi_gf.ProjectCoefficient(lnit_u);
   psi_old_gf = psi_gf;

   VectorConstantCoefficient beta_cf(beta);
   DenseMatrix betabeta(2,2);
   betabeta(0,0) = beta(0)*beta(0);
   betabeta(1,0) = beta(1)*beta(0);
   betabeta(0,1) = betabeta(1,0);
   betabeta(1,1) = beta(1)*beta(1);
   MatrixConstantCoefficient betabeta_cf(betabeta);

   ConstantCoefficient eps_cf(eps);
   ConstantCoefficient eps_sqr_cf(eps*eps);
   ConstantCoefficient sqrteps_cf(sqrt(eps));
   ConstantCoefficient beta_dot_beta_cf(beta(0)*beta(0) + beta(1)*beta(1));

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   // 12. Iterate
   int k;
   int total_iterations = 0;
   double tol = 1e-8;
   double increment_u = 0.1;
   for (k = 0; k < max_it; k++)
   {
      // double alpha = alpha0 / log(k+2);
      // double alpha = alpha0 / sqrt(k+1);
      double alpha = alpha0 * sqrt(k+1);
      // double alpha = alpha0 * sqrt(k+1);
      // double alpha = alpha0;
      // alpha *= 2;

      ParGridFunction u_tmp(&H1fes);
      u_tmp = u_old_gf;

      mfem::out << "\nOUTER ITERATION " << k+1 << endl;

      int j;
      for (j = 0; j < 3; j++)
      {
         // A. Assembly
         GridFunctionCoefficient psi_cf(&psi_gf);
         GridFunctionCoefficient psi_old_cf(&psi_old_gf);
         ExpitGridFunctionCoefficient expit_psi(psi_gf, alpha);
         dExpitdxGridFunctionCoefficient dexpitdx_psi(psi_gf, alpha);
         SumCoefficient psi_old_minus_psi(psi_old_cf, psi_cf, 1.0, -1.0);
         
         ParLinearForm b0,b1,b2;
         b0.Update(&H1fes,rhs.GetBlock(0),0);
         b1.Update(&RTfes,rhs.GetBlock(1),0);
         b2.Update(&L2fes,rhs.GetBlock(2),0);

         // b0 and b1 are zero
         b0.AddDomainIntegrator(new DomainLFIntegrator(psi_old_minus_psi));
         b0.Assemble();
         b1.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(f_cf));
         b1.Assemble();
         b2.AddDomainIntegrator(new DomainLFIntegrator(expit_psi));
         b2.Assemble();

         // // a00: ϵ(∇u,∇v) + (β⋅∇u,β⋅∇v)
         // ParBilinearForm a00(&H1fes);
         // a00.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
         // a00.AddDomainIntegrator(new DiffusionIntegrator(eps_cf));
         // a00.AddDomainIntegrator(new DiffusionIntegrator(betabeta_cf));
         // a00.Assemble();
         // HypreParMatrix A00;
         // a00.FormLinearSystem(ess_tdof_list, x.GetBlock(0), rhs.GetBlock(0), 
         //                      A00, tx.GetBlock(0), trhs.GetBlock(0));

         // // a10: √ϵ(∇u,q) + √ϵ(β⋅∇u,∇⋅q)
         // Vector tmp(2);
         // tmp(0) = beta(0);
         // tmp(1) = beta(1);
         // tmp *= -sqrt(eps);
         // VectorConstantCoefficient tmp_cf(tmp);
         // ParMixedBilinearForm a10(&H1fes,&RTfes);
         // a10.AddDomainIntegrator(new MixedVectorGradientIntegrator(sqrteps_cf));
         // a10.AddDomainIntegrator(new MixedGradDivIntegrator(tmp_cf));
         // a10.Assemble();
         // HypreParMatrix A10;
         // a10.FormRectangularLinearSystem(ess_tdof_list, empty, x.GetBlock(0), rhs.GetBlock(1), 
         //                                 A10, tx.GetBlock(0), trhs.GetBlock(1));

         // HypreParMatrix &A01 = *A10.Transpose();

         // // a20: (u,w)
         // ParMixedBilinearForm a20(&H1fes,&L2fes);
         // a20.AddDomainIntegrator(new MixedScalarMassIntegrator());
         // a20.Assemble();
         // HypreParMatrix A20;
         // a20.FormRectangularLinearSystem(ess_tdof_list, ess_tdof_list_L2, x.GetBlock(0), rhs.GetBlock(2), 
         //                                 A20, tx.GetBlock(0), trhs.GetBlock(2));

         // HypreParMatrix &A02 = *A20.Transpose();

         // // a11: ϵ(∇⋅p,∇⋅q) + (p,q)
         // ParBilinearForm a11(&RTfes);
         // a11.AddDomainIntegrator(new DivDivIntegrator(eps_cf));
         // a11.AddDomainIntegrator(new VectorFEMassIntegrator());
         // a11.Assemble();
         // a11.Finalize();
         // HypreParMatrix &A11 = *a11.ParallelAssemble();

         // // a22: -α(dexpit(αψ)δψ,w)
         // ProductCoefficient neg_dexpitdx_psi(-1.0, dexpitdx_psi);
         // ParBilinearForm a22(&L2fes);
         // a22.AddDomainIntegrator(new MassIntegrator(neg_dexpitdx_psi));
         // a22.Assemble();
         // HypreParMatrix A22;
         // a22.FormLinearSystem(ess_tdof_list_L2, x.GetBlock(2), rhs.GetBlock(2), 
         //                      A22, tx.GetBlock(2), trhs.GetBlock(2));

         // a00: ϵ²(∇u,∇v) + (β⋅β u,v) - ϵ(β⋅∇u,v) - ϵ(u,β⋅∇v)
         Vector tmp(2);
         tmp(0) = beta(0);
         tmp(1) = beta(1);
         tmp *= eps;
         VectorConstantCoefficient tmp_cf(tmp);
         ParBilinearForm a00(&H1fes);
         a00.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
         a00.AddDomainIntegrator(new DiffusionIntegrator(eps_sqr_cf));
         a00.AddDomainIntegrator(new MassIntegrator(beta_dot_beta_cf));
         a00.AddDomainIntegrator(new ConvectionIntegrator(tmp_cf,-1.0));
         a00.AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(tmp_cf,-1.0)));
         a00.Assemble();
         HypreParMatrix A00;
         a00.FormLinearSystem(ess_tdof_list, x.GetBlock(0), rhs.GetBlock(0), 
                              A00, tx.GetBlock(0), trhs.GetBlock(0));

         // a10: ϵ(∇u,q) - (u,β⋅q)
         ScalarVectorProductCoefficient minus_beta_cf(-1.0, beta_cf);
         ParMixedBilinearForm a10(&H1fes,&RTfes);
         a10.AddDomainIntegrator(new MixedVectorGradientIntegrator(eps_cf));
         a10.AddDomainIntegrator(new TransposeIntegrator(new MixedDotProductIntegrator(minus_beta_cf)));
         a10.Assemble();
         HypreParMatrix A10;
         a10.FormRectangularLinearSystem(ess_tdof_list, empty, x.GetBlock(0), rhs.GetBlock(1), 
                                         A10, tx.GetBlock(0), trhs.GetBlock(1));

         HypreParMatrix &A01 = *A10.Transpose();

         // a20: (u,w)
         ParMixedBilinearForm a20(&H1fes,&L2fes);
         a20.Assemble();
         HypreParMatrix A20;
         a20.FormRectangularLinearSystem(ess_tdof_list, ess_tdof_list_L2, x.GetBlock(0), rhs.GetBlock(2), 
                                         A20, tx.GetBlock(0), trhs.GetBlock(2));

         HypreParMatrix &A02 = *A20.Transpose();

         // a11: (∇⋅p,∇⋅q) + (p,q)
         ParBilinearForm a11(&RTfes);
         a11.AddDomainIntegrator(new DivDivIntegrator());
         a11.AddDomainIntegrator(new VectorFEMassIntegrator());
         a11.Assemble();
         a11.Finalize();
         HypreParMatrix &A11 = *a11.ParallelAssemble();

         // a22: -α(dexpit(αψ)δψ,w)
         ProductCoefficient neg_dexpitdx_psi(-1.0, dexpitdx_psi);
         ParBilinearForm a22(&L2fes);
         // a22.AddDomainIntegrator(new MassIntegrator(neg_dexpitdx_psi));
         a22.AddDomainIntegrator(new MassIntegrator());
         a22.Assemble();
         HypreParMatrix A22;
         a22.FormLinearSystem(ess_tdof_list_L2, x.GetBlock(2), rhs.GetBlock(2), 
                              A22, tx.GetBlock(2), trhs.GetBlock(2));

         // DIRECT solver
         Array2D<HypreParMatrix *> BlockA(3,3);
         BlockA(0,0) = &A00;
         BlockA(1,0) = &A10;
         BlockA(2,0) = &A20;
         BlockA(0,1) = &A01;
         BlockA(1,1) = &A11;
         BlockA(0,2) = &A02;
         BlockA(2,2) = &A22;
         HypreParMatrix * Ah = HypreParMatrixFromBlocks(BlockA);
         
         MUMPSSolver mumps;
         mumps.SetPrintLevel(0);
         mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
         mumps.SetOperator(*Ah);
         mumps.Mult(trhs,tx);
         delete Ah;

         u_gf.SetFromTrueDofs(tx.GetBlock(0));
         delta_psi_gf.SetFromTrueDofs(tx.GetBlock(2));

         u_tmp -= u_gf;
         double Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << pmesh << u_gf << "window_title 'Discrete solution'" << flush;

         double gamma = 1.0;
         delta_psi_gf *= gamma;
         psi_gf += delta_psi_gf;

         mfem::out << "Newton_update_size = " << Newton_update_size << endl;

         if (Newton_update_size < increment_u)
         {
            break;
         }
      }
      mfem::out << "Number of Newton iterations = " << j+1 << endl;

      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero);

      mfem::out << "|| u_h - u_h_prvs || = " << increment_u << endl;

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;

      if (increment_u < tol || k == max_it-1)
      {
         break;
      }
   }

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << "\n dofs:             " << H1fes.GetTrueVSize() + RTfes.GetTrueVSize() + L2fes.GetTrueVSize()
             << endl;

   // if (visualization)
   // {
   //    socketstream err_sock(vishost, visport);
   //    err_sock.precision(8);

   //    FunctionCoefficient exact_cf(exact_func);
   //    ParGridFunction error(&H1fes);
   //    error = 0.0;
   //    error.ProjectCoefficient(exact_cf);
   //    error -= u_gf;

   //    mfem::out << "\n Final L2-error (|| u - u_h||) = " << u_gf.ComputeL2Error(exact_cf) << endl;

   //    err_sock << "parallel " << num_procs << " " << myid << "\n";
   //    err_sock << "solution\n" << pmesh << error << "window_title 'Error'"  << flush;
   // }

   return 0;
}

double LnitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, lnit(val)/alpha));
}

double ExpitGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   val *= alpha;
   return min(max_val, max(min_val, expit(val)));
}

double dExpitdxGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   val *= alpha;
   return min(max_val, max(min_val, alpha * dexpitdx(val)));
}

double Ramp_BC(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double tol = 1e-10;
   double eps = 0.05;

   if (  (abs(y) < tol && x >= 0.2)
      || (abs(x-1.0) < tol)
      || (abs(y-1.0) < tol) )
   {
      return 0.0;
   }
   else if (  (abs(x) < tol && y <= 1.0 - eps)
           || (abs(y) < tol && x <= 0.2 - eps) )
   {
      return 1.0;
   }
   else if (x >= (0.2 - eps) && abs(y) < tol)
   {
      return (0.2 - x)/eps;
   }
   else if (y >= (1.0 - eps) && abs(x) < tol)
   {
      return  (1.0 - y)/eps;
   }
   else
   {
      return 0.5;
   }
}

double EJ_exact_solution(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double lambda = M_PI*M_PI*eps;
   double r1 = (1.0 + sqrt(1.0 + 4.0 * eps * lambda))/(2*eps);
   double r2 = (1.0 - sqrt(1.0 + 4.0 * eps * lambda))/(2*eps);

   double num = exp(r2 * (x - 1.0)) - exp(r1 * (x-1.0));
   double denom = exp(-r2) - exp(-r1);
   // double denom = r1 * exp(-r2) - r2 * exp(-r1);

   double scale = 0.5;
   // double scale = (r1 * exp(-r2) - r2 * exp(-r1)) / (exp(-r2) - exp(-r1));

   return scale * num / denom * cos(M_PI * y) + 0.5;
   
}