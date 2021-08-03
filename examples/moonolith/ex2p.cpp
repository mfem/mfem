//                                MFEM Example 2 - Parallel Version
//                              Moonolith Modification
//
// Compile with: make ex2p
//
// Moonolith sample runs:
//               mpirun -np 4 ex2p
//               mpirun -np 4 ex2p --source_refinements 1 --dest_refinements 2
//               mpirun -np 4 ex2p -s ../../data/inline-hex.mesh -d
//               ../../data/inline-tet.mesh
//
// Description:  This example code demonstrates the use of MFEM for transferring
//               discrete fields from one finite element mesh to another. The
//               meshes can be of arbitrary shape and completely unrelated with
//               each other. This feature can be used for implementing immersed
//               domain methods for fluid-structure interaction or general
//               multi-physics applications.
//
//               This particular example concerns discontinuos Galrkin FEM and
//               it is for parallel runtimes.

#include "example_utils.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace std;

void destination_transform(const Vector &x, Vector &x_new)
{
   x_new = x;
   // x_new *= .5;
}

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int num_procs, rank;

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   // Init transfer library context, with MPI handled outside the library
   InitTransfer(argc, argv, MPI_COMM_WORLD);

   const char *source_mesh_file = "../../data/inline-tri.mesh";
   const char *destination_mesh_file = "../../data/inline-quad.mesh";

   int src_n_refinements = 0;
   int dest_n_refinements = 0;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&source_mesh_file, "-s", "--source_mesh",
                  "Mesh file to use for src.");
   args.AddOption(&destination_mesh_file, "-d", "--destination_mesh",
                  "Mesh file to use for dest.");
   args.AddOption(&src_n_refinements, "-sr", "--source_refinements",
                  "Number of src refinements");
   args.AddOption(&dest_n_refinements, "-dr", "--dest_refinements",
                  "Number of dest refinements");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   check_options(args);

   ///////////////////////////////////////////////////

   ifstream imesh(source_mesh_file);
   shared_ptr<Mesh> src_mesh, dest_mesh;
   if (imesh)
   {
      src_mesh = make_shared<Mesh>(imesh, 1, 1);
      imesh.close();
   }
   else
   {
      if (rank == 0)
         mfem::err << "WARNING: Source mesh file not found: " << source_mesh_file
                   << "\n"
                   << "Using default 2D triangle mesh.";
      src_mesh = make_shared<Mesh>(4, 4, Element::TRIANGLE);
   }

   imesh.open(destination_mesh_file);
   if (imesh)
   {
      dest_mesh = make_shared<Mesh>(imesh, 1, 1);
      imesh.close();
   }
   else
   {
      if (rank == 0)
         mfem::err << "WARNING: Destination mesh file not found: "
                   << destination_mesh_file << "\n"
                   << "Using default 2D quad mesh.";
      dest_mesh = make_shared<Mesh>(4, 4, Element::QUADRILATERAL);
   }

   dest_mesh->Transform(&destination_transform);

   for (int i = 0; i < src_n_refinements; ++i)
   {
      src_mesh->UniformRefinement();
   }

   for (int i = 0; i < dest_n_refinements; ++i)
   {
      dest_mesh->UniformRefinement();
   }

   src_mesh->EnsureNCMesh();
   dest_mesh->EnsureNCMesh();

   {
      for (int l = 0; l < 4; l++)
      {
         src_mesh->RandomRefinement(0.1); // 10% probability
      }
   }

   {
      for (int l = 0; l < 4; l++)
      {
         dest_mesh->RandomRefinement(0.1); // 10% probability
      }
   }

   auto p_src_mesh = make_shared<ParMesh>(MPI_COMM_WORLD, *src_mesh);
   auto p_dest_mesh = make_shared<ParMesh>(MPI_COMM_WORLD, *dest_mesh);

   ///////////////////////////////////////////////////

   auto src_fe_coll = make_shared<DG_FECollection>(0, p_src_mesh->Dimension());
   auto src_fe =
      make_shared<ParFiniteElementSpace>(p_src_mesh.get(), src_fe_coll.get());

   auto dest_fe_coll = make_shared<DG_FECollection>(1, p_dest_mesh->Dimension());
   auto dest_fe =
      make_shared<ParFiniteElementSpace>(p_dest_mesh.get(), dest_fe_coll.get());

   ///////////////////////////////////////////////////

   ParGridFunction src_fun(src_fe.get());
   FunctionCoefficient coeff(example_fun);
   make_fun(*src_fe, coeff, src_fun);

   ParGridFunction dest_fun(dest_fe.get());
   dest_fun = 0.0;
   dest_fun.Update();

   ParMortarAssembler assembler(src_fe, dest_fe);
   assembler.AddMortarIntegrator(make_shared<L2MortarIntegrator>());
   if (assembler.Transfer(src_fun, dest_fun))
   {
      if (visualization)
      {

         const double src_err = src_fun.ComputeL2Error(coeff);
         const double dest_err = dest_fun.ComputeL2Error(coeff);

         if (rank == 0)
         {
            mfem::out << "l2 error: src: " << src_err << ", dest: " << dest_err
                      << std::endl;
         }

         plot(*p_src_mesh, src_fun, "source");
         plot(*p_dest_mesh, dest_fun, "destination");
      }
   }
   else
   {
      mfem::out << "No intersection no transfer!" << std::endl;
   }

   // Finalize transfer library context
   FinalizeTransfer();
   return MPI_Finalize();
}
