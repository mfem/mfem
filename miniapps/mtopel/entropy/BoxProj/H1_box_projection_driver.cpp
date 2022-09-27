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
   return 0.5+2.0*sin(M_PI*alpha);
}

void grad_sin_func(const Vector & X, Vector &grad)
{
   Vector c(X.Size()); 
   c = 1.0;
   double alpha = c*X;
   grad = 2.0*M_PI*cos(M_PI*alpha);
}

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();
   const char *mesh_file = "../disk.mesh";
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

   BoxProjection Pu(&pmesh,order,&sin_cf,&grad_sin_cf,false);
   Pu.SetNewtonStepSize(0.25);
   Pu.SetBregmanStepSize(0.25);
   Pu.SetNormWeight(0.0);
   Pu.SetPrintLevel(1);
   Pu.Solve();
   ParGridFunction u = Pu.Getu();
   ParGridFunction p = Pu.Getp();

   ParGridFunction sigmoid_p(p.ParFESpace());
   ExpitGridFunctionCoefficient expit_p(p);
   sigmoid_p.ProjectCoefficient(expit_p);

   // ParGridFunction H1_sigmoid_p(u.ParFESpace());
   // H1_sigmoid_p.ProjectGridFunction(sigmoid_p);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << pmesh << u  
                  << "window_title 'Solution u'" << flush;

      socketstream p_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << pmesh << p  
                  << "window_title 'Solution p'" << flush;   

      socketstream p2_sock(vishost, visport);
      p2_sock.precision(8);
      p2_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << pmesh << sigmoid_p  
                  << "window_title 'Solution sigmoid_p'" << flush;                                       
   }

}