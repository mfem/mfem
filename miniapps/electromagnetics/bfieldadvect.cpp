#include "bfieldadvect_solver.hpp"
#include <random>

using namespace std;
using namespace mfem;
using namespace mfem::electromagnetics;

void BFieldFunc(const Vector &, Vector&);
void PeturbBoxMesh(Mesh *mesh, double xlen, double ylen, double zlen, double window);

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


   Mesh *test_mesh = new Mesh("../../data/ref-pyramid.mesh", 1, 1);
   std::cout << test_mesh->GetElementVolume(0) << std::endl;
   Array<Refinement> ref(1);
   ref[0].ref_type = Refinement::XYZ;
   ref[0].index = 0;
   test_mesh->GeneralRefinement(ref, 1);
   double sum = 0.0;
   for (int i = 0; i < 10; ++i)
      sum += test_mesh->GetElementVolume(i);
   std::cout << sum << std::endl;
   Array<double> elem_error(10);
   for (int i = 0; i < 10; ++i)
      elem_error[i] = 0.0;
   test_mesh->DerefineByError(elem_error, 1.0);
   std::cout << test_mesh->GetElementVolume(0) << std::endl;

   VisItDataCollection visit_dc_test("test", test_mesh);
   visit_dc_test.SetCycle(0);
   visit_dc_test.SetTime(0);
   visit_dc_test.Save();



   Mesh *mesh_old = new Mesh(20, 20, 5, Element::HEXAHEDRON, false, 4.0, 4.0, 1.0);
   Mesh *mesh_new = new Mesh(*mesh_old, true);
   double dl = 1.0/5.0;
   PeturbBoxMesh(mesh_new, 4.0, 4.0, 1.0, 0.05*dl);


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
   ParGridFunction *b_new_exact = new ParGridFunction(HDivFESpaceNew);

   //Set the initial B value
   *b = 0.0;
   *b_new = 0.0;
   *b_new_exact = 0.0;
   VectorFunctionCoefficient BFieldCoef(3,BFieldFunc);
   b->ProjectCoefficient(BFieldCoef);
   b_new_exact->ProjectCoefficient(BFieldCoef);

   BFieldAdvector advector(&pmesh_old, &pmesh_new, 1);
   advector.Advect(b, b_new);
   ParGridFunction *b_recon = advector.GetReconstructedB();
   ParGridFunction *curl_b = advector.GetCurlB();
   ParGridFunction *a = advector.GetA();
   ParGridFunction *a_new = advector.GetANew();

   Vector diff_b(*b_new_exact);
   diff_b -= *b_new;    //diff = b_new_exact - b_new
   std::cout << "Vector diff in B field on the new mesh:  " << diff_b.Normlinf() << std::endl;

   Vector diff_a(*a);
   diff_a -= *a_new;    //diff = b_new_exact - b_new
   std::cout << "Vector diff in A field on the new mesh:  " << diff_a.Normlinf() << ", " << a->Normlinf() << std::endl;

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


void PeturbBoxMesh(Mesh *mesh, double xlen, double ylen, double zlen, double window)
{
   Vector displacements(3*mesh->GetNV());
   displacements = 0.0;
   std::random_device rd;  // Will be used to obtain a seed for the random number engine
   std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
   std::uniform_real_distribution<> r(-0.5*window, 0.5*window);   
   for (int vi = 0; vi < mesh->GetNV(); ++vi)
   {
      double *v = mesh->GetVertex(vi);
      if (fabs(v[0]) > 1e-6 && fabs(v[0] - xlen) > 1e-6)
      {
         displacements[3*vi+0] = r(gen);
      }

      if (fabs(v[1]) > 1e-6 && fabs(v[1] - ylen) > 1e-6)
      {
         displacements[3*vi+1] = r(gen);
      }

      if (fabs(v[2]) > 1e-6 && fabs(v[2] - zlen) > 1e-6)
      {
         displacements[3*vi+2] = r(gen);
      }      
   }
   std::cout << "Displacement norm:  " << displacements.Norml2() << std::endl;
   mesh->MoveVertices(displacements);
}