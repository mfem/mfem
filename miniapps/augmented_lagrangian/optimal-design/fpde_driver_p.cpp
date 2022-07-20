
// ./fpde_driver -beta 1.0 -m ../../mfem/data/inline-quad.mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "common/fpde.hpp"
#include "common/rational_approximation.hpp"


using namespace std;
using namespace mfem;

double alpha = 1.0;
double beta = 0.0;


double u_function(const Vector & x)
{
   double u = sin(M_PI*x(0))*sin(M_PI*x(1));
   return u;
}


double rhs_function(const Vector & x)
{
   double c = pow(2*M_PI*M_PI,alpha)+beta;
   double u = sin(M_PI*x(0))*sin(M_PI*x(1));
   return c*u;
}


int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   
   // 1. Parse command-line options.
   const char *mesh_file = "../../mfem/data/star.mesh";
   int order = 1;
   int num_refs = 3;
   bool visualization = true;
   double diff_coeff = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&num_refs, "-r", "--refs",
                  "Number of uniform refinements");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Fractional exponent");
   args.AddOption(&beta, "-beta", "--beta",
                  "Shift in f(z) = z^-α ");        
   args.AddOption(&diff_coeff, "-diff-coeff", "--diff-coeff",
                  "Diffusion coefficient ");                              
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


   Vector center;
   for (int i = 0; i<mesh.GetNE(); i++)
   {
      mesh.GetElementCenter(i,center);
      if (center[0]>0.5)
      {
         mesh.SetAttribute(i,2);
      }
      else
      {
         mesh.SetAttribute(i,1);
      }
   }
   mesh.SetAttributes();

   Vector constants(mesh.attributes.Max());
   constants(0) = -1.; 
   constants(1) = 1;

   PWConstCoefficient pw_cf(constants); 

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   for (int i = 0; i < num_refs; i++)
   {
      pmesh.UniformRefinement();
   }

   FunctionCoefficient rhs(rhs_function);
   ConstantCoefficient cf(pow(diff_coeff,1./alpha));
   FPDESolver fpde;
   fpde.SetMesh(&pmesh);
   fpde.SetOrder(order);
   fpde.SetAlpha(alpha);
   fpde.SetBeta(beta);


   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 0;
   fpde.SetEssentialBoundary(ess_bdr);

   
   fpde.Init();
   fpde.SetDiffusionCoefficient(&cf);
   fpde.SetRHSCoefficient(&pw_cf);
   fpde.SetupFEM();
   fpde.Solve();


   ParGridFunction * u = fpde.GetParFEMSolution();

   L2_FECollection L2fec(order, pmesh.Dimension());
   ParFiniteElementSpace L2fes(&pmesh, &L2fec);

   ParGridFunction rhs_gf(&L2fes);
   rhs_gf.ProjectCoefficient(pw_cf);


   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << *u << flush;

      socketstream rhs_sock(vishost, visport);
      rhs_sock << "parallel " << num_procs << " " << myid << "\n";
      rhs_sock.precision(8);
      rhs_sock << "solution\n" << pmesh << rhs_gf << flush;
   }



   // Array<double> coeffs, poles; 
   // ComputePartialFractionApproximation(alpha,beta,coeffs,  poles);

   // mfem::out << "coeffs = " << endl;
   // coeffs.Print();

   // mfem::out << "poles = " << endl;
   // poles.Print();




   return 0;
}