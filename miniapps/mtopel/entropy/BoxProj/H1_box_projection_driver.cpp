#include "mfem.hpp"
#include "H1_box_projection.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double sin_func(const Vector & X)
{
   Vector c(X.Size()); 
   c = 1.0;
   double alpha = c*X;
   return 0.5+0.5*sin(M_PI*alpha);
}

void grad_sin_func(const Vector & X, Vector &grad)
{
   Vector c(X.Size()); 
   c = 1.0;
   double alpha = c*X;
   grad = 0.5*M_PI*cos(M_PI*alpha);
}

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int ref = 1;
   bool visualization = true;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&ref, "-ref", "--refinements",
                  "Number of refinements");                  
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

   for (int i = 0; i<ref; i++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   ConstantCoefficient half(0.5);
   Vector vzero(dim); vzero = 0.0;
   VectorConstantCoefficient zero(vzero);

   FunctionCoefficient sin_cf(sin_func);
   VectorFunctionCoefficient grad_sin_cf(dim,grad_sin_func);

   BoxProjection Pu(&pmesh,order,&sin_cf,&grad_sin_cf);
   Pu.SetNewtonStepSize(0.25);
   Pu.SetBregmanStepSize(0.25);
   Pu.SetNormWeight(0.0);
   Pu.Solve();
   ParGridFunction u = Pu.GetH1Solution();
   ParGridFunction psi = Pu.GetL2Solution();



   ParGridFunction sigmoid_psi(psi.ParFESpace());
   ExpitGridFunctionCoefficient expit_psi(psi);
   sigmoid_psi.ProjectCoefficient(expit_psi);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << pmesh << u  
                  << "window_title 'Solution u'" << flush;

      socketstream psi_sock(vishost, visport);
      psi_sock.precision(8);
      psi_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << pmesh << psi  
                  << "window_title 'Solution psi'" << flush;   

      socketstream psi2_sock(vishost, visport);
      psi2_sock.precision(8);
      psi2_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << pmesh << sigmoid_psi  
                  << "window_title 'Solution sigmoid_psi'" << flush;                                       
   }

}