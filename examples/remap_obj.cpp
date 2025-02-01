#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "remap_obj.hpp"

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   mfem::Mpi::Init();
   int num_procs = mfem::Mpi::WorldSize();
   int myid = mfem::Mpi::WorldRank();
   mfem::Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 2;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(std::cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(std::cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   mfem::Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   mfem::Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(100./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // Define the discontinuous DG finite element space of the given
   // polynomial order on the refined mesh.
   const int btype = mfem::BasisType::Positive;
   mfem::DG_FECollection fec(order, dim, btype);
   mfem::ParFiniteElementSpace pfes(&pmesh, &fec);

   mfem::ParGridFunction tgf(&pfes);
   SphCoefficient sph;
   tgf.ProjectCoefficient(sph);

   {
      L2Objective* obj=new L2Objective(pfes,tgf);

      mfem::Vector

      std::cout<<obj->Eval(tgf.GetTrueVector())<<" "<<std::endl;
      obj->Test();
      delete obj;
   }

   double tvol=0.0;

   {
      VolConstr* g=new VolConstr(pfes,0.0);
      tvol=g->Eval(tgf.GetTrueVector());
      std::cout<<tvol<<" "<<std::endl;
      g->Test();
      delete g;
   }

   return 0;
}
