#include "bfieldadvect_solver.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::electromagnetics;

void BFieldFunc(const Vector &, Vector&);

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   // Parse command-line options.
   const char *mesh_file = "../../data/toroid-hex.mesh";
   int order = 1;
   int serial_ref_levels = 0;
   int parallel_ref_levels = 0;
   bool visualization = false;
   bool visit = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
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

   Mesh *mesh_old = new Mesh(20, 20, 5, Element::HEXAHEDRON, false, 4.0, 4.0, 1.0);
   Mesh *mesh_new = new Mesh(30, 30, 8, Element::HEXAHEDRON, false, 4.0, 4.0, 1.0);   


   /*Mesh *mesh_test = new Mesh("../../data/ref-cube.mesh");

   Array<int> verts({0,1,2,3,4,5,6});
   Array<int> faces;
   mesh_test->FacesWithAllVerts(faces, verts);

   std::cout << "Faces size" << faces.Size() << std::endl;
   for (int i = 0; i < faces.Size(); i ++)
   {
      std::cout << faces[i] << std::endl;
   }*/


   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh_old->UniformRefinement();
      mesh_new->UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine this
   // mesh further in parallel to increase the resolution. Once the parallel
   // mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh_old(MPI_COMM_WORLD, *mesh_old);
   ParMesh pmesh_new(MPI_COMM_WORLD, *mesh_new);
   delete mesh_old;
   delete mesh_new;

   // Refine this mesh in parallel to increase the resolution.
   for (int l = 0; l < parallel_ref_levels; l++)
   {
      pmesh_old.UniformRefinement();
      pmesh_new.UniformRefinement();
   }

   //Set up the pre and post advection fields on the relevant meshes/spaces
   RT_ParFESpace *HDivFESpaceOld  = new RT_ParFESpace(&pmesh_old,order,pmesh_old.Dimension());
   RT_ParFESpace *HDivFESpaceNew  = new RT_ParFESpace(&pmesh_new,order,pmesh_new.Dimension());
   ParGridFunction *b = new ParGridFunction(HDivFESpaceOld);
   ParGridFunction *b_new = new ParGridFunction(HDivFESpaceNew);

   //Set the initial B value
   *b = 0.0;
   *b_new = 0.0;
   VectorFunctionCoefficient BFieldCoef(3,BFieldFunc);
   b->ProjectCoefficient(BFieldCoef);


   BFieldAdvector advector(&pmesh_old, &pmesh_new, 1);
   advector.Advect(b, b_new);

   ParGridFunction *b_recon = advector.GetReconstructedB();
   ParGridFunction *curl_b = advector.GetCurlB();
   ParGridFunction *a = advector.GetA();
   ParGridFunction *a_new = advector.GetANew();

   // Handle the visit visualization
   if (visit)
   {
      VisItDataCollection visit_dc_old("bfa-old", &pmesh_old);
      visit_dc_old.RegisterField("B", b);
      visit_dc_old.RegisterField("Curl_B", curl_b);
      visit_dc_old.RegisterField("A", a);
      visit_dc_old.RegisterField("B_recon", b_recon);
      visit_dc_old.SetCycle(0);
      visit_dc_old.SetTime(0);
      visit_dc_old.Save();
      
      VisItDataCollection visit_dc_new("bfa-new", &pmesh_new);
      visit_dc_new.RegisterField("A_new", a_new);
      visit_dc_new.RegisterField("B_new", b_new);
      visit_dc_new.SetCycle(0);
      visit_dc_new.SetTime(0);
      visit_dc_new.Save();
   }

}


void BFieldFunc(const Vector &x, Vector &B)
{
   B.SetSize(3);
   B[0] =  x[1] - 2.0;
   B[1] = -(x[0] - 2.0);
   B[2] =  0.0;
}