#include "mfem.hpp"
#include "mtop_coefficients.hpp"


#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int nprocs, myrank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 2;
   bool visualization = true;
   int rs_levels = 2;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myrank == 0) { args.PrintOptions(cout); }

   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   mesh.EnsureNCMesh(true);
   const int dim = mesh.Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define random shooting coefficient
   RandShootingCoefficient rsc(&pmesh, 0.1); rsc.Sample();
   ConstantCoefficient one(1.0);

   // Refine the mesh
   CoefficientRefiner refiner(rsc,order);
   refiner.SetThreshold(1e-4);
   refiner.SetMaxElements(100000);
   refiner.SetNCLimit(1);
   refiner.PreprocessMesh(pmesh,4);

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParGridFunction pg(&pfes);

   pg.ProjectCoefficient(one);

   //compute the volume of the mesh
   double tvol=0.0;
   {
       ParLinearForm lf(&pfes);
       lf.AddDomainIntegrator(new DomainLFIntegrator(one));
       lf.Assemble();
       tvol=lf(pg);
   }

   double rvol=0.0;
   {
       ParLinearForm lf(&pfes);
       lf.AddDomainIntegrator(new DomainLFIntegrator(rsc));
       lf.Assemble();
       rvol=lf(pg);
   }

   if(myrank==0)
   {
       cout<<"Total mesh volume: "<<tvol<<endl;
       cout<<"Reduced mesh volume: "<<rvol<<endl;
   }

   // ParaView output.
   pg.ProjectCoefficient(rsc);
   ParaViewDataCollection dacol("ParaView", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("density", &pg);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();



   MPI_Finalize();
   return 0;
}
