

#include "mfem.hpp"
#include "kdtree.hpp"
#include <random>

using namespace mfem;


int main(int argc, char *argv[])
{
   //   Parse command-line options
   const char *mesh_file = "../../data/star-q3.mesh";
   int ser_ref_levels = 1;
   int order = 2;
   int iorder = 2; //MFEM integration integration points
   int aorder = 2; //Algoim integration integration points
   bool visualization = true;
   int print_level = 0;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption((&print_level), "-prt", "--print-level", "Print level.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   //   Read the (serial) mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   //    Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }


   // Define the finite element spaces for the solution
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(mesh, &fec, 1, Ordering::byVDIM);
   int glob_size = fespace.GetTrueVSize();
   std::cout << "Number of finite element unknowns: " << glob_size << std::endl;

   // Define the level set grid function
   GridFunction x(&fespace); x=0.0;

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0,1.0);
   double dist1,dist2;
   int  ind1,ind2;

   // build kdtree from the mesh
   if(dim==3){
       KDTree3D* kdt=BuildKDTree3D(mesh);
       KDTree3D::PointND pt;
       for(int i=0;i<10;i++){
           pt.xx[0]= distribution(generator);
           pt.xx[1]= distribution(generator);
           pt.xx[2]= distribution(generator);

           kdt->NNS(pt,ind1,dist1);
           kdt->rNNS(pt,ind2,dist2);
           std::cout<<"ind1="<<ind1<<" ind2="<<ind2<<" d1="<<dist1<<" d2="<<dist2<<std::endl;
       }

       delete kdt;
   }else
   if(dim==2){
       KDTree2D* kdt=BuildKDTree2D(mesh);
       KDTree2D::PointND pt;
       for(int i=0;i<10;i++){
           pt.xx[0]= distribution(generator);
           pt.xx[1]= distribution(generator);

           kdt->NNS(pt,ind1,dist1);
           kdt->rNNS(pt,ind2,dist2);
           std::cout<<"ind1="<<ind1<<" ind2="<<ind2<<" d1="<<dist1<<" d2="<<dist2<<std::endl;
       }

       delete kdt;

   }

   ParaViewDataCollection dacol("ParaViewDistance", mesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("dist",&x);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();


   delete mesh;

   return 0;
}
