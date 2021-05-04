#include <memory>

#include "mfem.hpp"
#include "example_utils.hpp"

using namespace mfem;
using namespace std;

void vector_fun(const Vector &x, Vector &f)
{
   double n = x.Norml2();
   f.SetSize(x.Size());
   f = n;
}

void dest_transform(const Vector &x, Vector &x_new)
{
   x_new = x;
   // x_new *= .7;
}

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);


   int num_procs, rank;

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   const char *source_mesh_file      = "../data/inline-tri.mesh";
   const char *destination_mesh_file = "../data/inline-quad.mesh";

   int src_n_refinements  = 0;
   int dest_n_refinements = 0;
   bool debug = false;
   bool wait = false;
   const char * logger_type = "file";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&source_mesh_file,     "-s", "--source_mesh",
                  "Mesh file to use for src.");
   args.AddOption(&destination_mesh_file, "-d", "--destination_mesh",
                  "Mesh file to use for dest.");
   args.AddOption(&src_n_refinements, "-sr", "--source_refinements",
                  "Number of src refinements");
   args.AddOption(&dest_n_refinements, "-dr", "--dest_refinements",
                  "Number of dest refinements");
   args.AddOption(&debug, "-debug", "--debug", "-ndebug", "--no-debug", "debug");
   args.AddOption(&logger_type, "-logger", "--logger", "logger");
   args.AddOption(&wait, "-wait", "--wait", "-nwait", "-no-wait", "wait");

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
         mfem::err << "WARNING: Source mesh file not found: "
                   << source_mesh_file << "\n"
                   << "Using default 2D triangle mesh.";
      src_mesh = make_shared<Mesh>(4,4,Element::QUADRILATERAL);
      //        src_mesh = make_shared<Mesh>(4,4,Element::TRIANGLE,1);
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
         std::cerr << "WARNING: Destination mesh file not found: "
                   << destination_mesh_file << "\n"
                   << "Using default 2D quad mesh.";
      dest_mesh = make_shared<Mesh>(4, 4,Element::QUADRILATERAL);
      // dest_mesh = make_shared<Mesh>(4,7,Element::TRIANGLE,1);
   }

   dest_mesh->Transform(&dest_transform);

   for (int i = 0; i < src_n_refinements;  ++i)
   {
      src_mesh->UniformRefinement();
   }

   for (int i = 0; i < dest_n_refinements; ++i)
   {
      dest_mesh->UniformRefinement();
   }

   src_mesh->ReorientTetMesh();
   dest_mesh->ReorientTetMesh();

   auto p_src_mesh   = make_shared<ParMesh>(MPI_COMM_WORLD, *src_mesh);
   auto p_dest_mesh  = make_shared<ParMesh>(MPI_COMM_WORLD, *dest_mesh);

   ///////////////////////////////////////////////////
   auto src_fe_coll  = make_shared<RT_FECollection>(1, p_src_mesh->Dimension());
   auto src_fe        = make_shared<ParFiniteElementSpace>(p_src_mesh.get(),
                                                           src_fe_coll.get());

   auto dest_fe_coll = make_shared<ND_FECollection>(1, p_dest_mesh->Dimension());
   auto dest_fe       = make_shared<ParFiniteElementSpace>(p_dest_mesh.get(),
                                                           dest_fe_coll.get());

   bool is_vector_fe = true;
   ///////////////////////////////////////////////////

   const int dim = p_src_mesh->Dimension();

   ParGridFunction src_fun(src_fe.get());

   Vector local_coeff(dim);
   local_coeff = 1.0/sqrt(dim);

   // VectorConstantCoefficient coeff(local_coeff);
   VectorFunctionCoefficient coeff(dim, &vector_fun);
   src_fun.ProjectCoefficient(coeff);
   src_fun.Update();

   ParGridFunction dest_fun(dest_fe.get());
   dest_fun = 0.0;
   dest_fun.Update();

   ParMortarAssembler assembler(MPI_COMM_WORLD, src_fe, dest_fe);

   //Use Vector integrator
   assembler.AddMortarIntegrator(make_shared<VectorL2MortarIntegrator>());
   if (assembler.Transfer(src_fun, dest_fun, is_vector_fe) && visualization)
   {
      const double err = dest_fun.ComputeL2Error(coeff);
      if (rank == 0) { mfem::out << "l2 error: " << err << std::endl; }

      plot(*p_src_mesh,  src_fun);
      plot(*p_dest_mesh, dest_fun);
   }
   else
   {
      mfem::out << "No intersection no transfer!" << std::endl;
   }

   return MPI_Finalize();
}
