#include "vector_advectors.hpp"

using namespace mfem;

void BFieldFunc(const Vector &, Vector&);

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   // Parse command-line options.
   const char *mesh_file = "../../data/toroid-hex.mesh";
   int sOrder = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   bool visualization = true;
   bool visit = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&sOrder, "-so", "--spatial-order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&serial_ref_levels, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&parallel_ref_levels, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visualization",
                  "Enable or disable VisIt visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (mpi.Root())
      {
         cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      }
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();   

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine this
   // mesh further in parallel to increase the resolution. Once the parallel
   // mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Refine this mesh in parallel to increase the resolution.
   for (int l = 0; l < parallel_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   //Create a copy of the refined paralle mesh and peturb some inner nodes
   ParMesh pmesh_new(pmesh);

   //Set up the pre and post advection fields on the relevant meshes/spaces
   RT_ParFESpace *HDivFESpaceOld  = new RT_ParFESpace(pmesh,order,pmesh.Dimension());
   RT_ParFESpace *HDivFESpaceNew  = new RT_ParFESpace(pmesh_new,order,pmesh_new.Dimension());
   ParGridFunction *b = new ParGridFunction(HDivFESpaceOld);
   ParGridFunction *b_new = new ParGridFunction(HDivFESpaceNew);

   //|Set the initial B value
   VectorFunctionCoefficient BFieldCoef(3,BFieldFunc);
   b->ProjectCoefficient(BFieldCoef);
   //b->ParallelProject(*B);

   BFieldAdvector advector(&pmesh, &pmesh_new);
   advector.advect(b, b_new);
   ParGridFunction *b_recon = advector.GetReconstructedB();

   // Handle the visit viusalization
   if (visit)
   {
      VisItDataCollection visit_dc("bfield-advect", &pmesh);
      visit_dc.RegisterField("B", b);
      visit_dc.RegisterField("B_recon", b_recon);
      visit_dc_->SetCycle(0);
      visit_dc_->SetTime(0);
      visit_dc_->Save();
   }

}


void BFieldFunc(const Vector &x, Vector &B)
{
   B.SetSize(3);
   B[0] =  x[1];
   B[1] = -x[0];
   B[2] =  0.0;
}