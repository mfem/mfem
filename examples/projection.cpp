
#include "mfem.hpp"
#include "example_utils.hpp"

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   const char *source_mesh_file    = "../data/inline-tri.mesh";
   const char *destination_mesh_file = "../data/inline-quad.mesh";

   int src_n_refinements  = 0;
   int dest_n_refinements = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&source_mesh_file,      "-s", "--source_mesh",
                  "Mesh file to use for src.");
   args.AddOption(&destination_mesh_file, "-d", "--destination_mesh",
                  "Mesh file to use for dest.");
   args.AddOption(&src_n_refinements, "-sr", "--source_refinements",
                  "Number of src refinements");
   args.AddOption(&dest_n_refinements, "-dr", "--dest_refinements",
                  "Number of dest refinements");

   args.Parse();
   check_options(args);

   ///////////////////////////////////////////////////


  ///////////////////////////////////////////////////
  shared_ptr<Mesh> src_mesh, dest_mesh;

  ifstream imesh;

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

     dest_mesh = make_shared<Mesh>(4, 4, Element::QUADRILATERAL, 1);
  }

  const int dim = dest_mesh->Dimension();
  
  Vector box_min(dim), box_max(dim), range(dim);
  dest_mesh->GetBoundingBox(box_min, box_max);
  range = box_max;
  range -= box_min;

  imesh.open(source_mesh_file);
  
  if (imesh)
  {
     src_mesh = make_shared<Mesh>(imesh, 1, 1);
     imesh.close();
  }
  else
  {
     if (rank == 0)
        std::cerr << "WARNING: Source mesh file not found: "
                  << source_mesh_file << "\n"
                  << "Using default box mesh.\n";

     if(dim == 2) {
        src_mesh = make_shared<Mesh>(4, 4, Element::TRIANGLE, 1, range[0], range[1]);
     } else if(dim == 3) {
        src_mesh = make_shared<Mesh>(4, 4, 4, Element::TETRAHEDRON, 1, range[0], range[1], range[2]);
     }

     for(int i = 0; i < src_mesh->GetNV(); ++i) {
        double * v = src_mesh->GetVertex(i);

        for(int d = 0; d < dim; ++d) {
           v[d] += box_min[d];
        }
     }
  }

   for (int i = 0; i < src_n_refinements;  ++i)
   {
      src_mesh->UniformRefinement();
   }

   for (int i = 0; i < dest_n_refinements; ++i)
   {
      dest_mesh->UniformRefinement();
   }

   ///////////////////////////////////////////////////

   auto src_fe_coll  = make_shared<L2_FECollection>(1, src_mesh->Dimension());
   auto src_fe      = make_shared<FiniteElementSpace>(src_mesh.get(),
                                                      src_fe_coll.get());

   auto dest_fe_coll = make_shared<L2_FECollection>(1, dest_mesh->Dimension());
   auto dest_fe        = make_shared<FiniteElementSpace>(dest_mesh.get(),
                                                         dest_fe_coll.get());

   ///////////////////////////////////////////////////

   GridFunction src_fun(src_fe.get());
   GridFunction dest_fun(dest_fe.get());
   src_fun = 1.0;

   FunctionCoefficient coeff(example_fun);
   // ConstantCoefficient coeff(1);
   make_fun(*src_fe, coeff, src_fun);

   dest_fun = 0.0;
   dest_fun.Update();

   MortarAssembler assembler(src_fe, dest_fe);
   assembler.AddMortarIntegrator(make_shared<L2MortarIntegrator>());

   if (assembler.Transfer(src_fun, dest_fun))
   {
      dest_fun.Update();

      const double src_err  = src_fun.ComputeL2Error(coeff);
      const double dest_err = dest_fun.ComputeL2Error(coeff);
      std::cout << "l2 error: src: " << src_err << ", dest: " << dest_err <<
                std::endl;

      plot(*src_mesh, src_fun);
      plot(*dest_mesh, dest_fun);
   }
   else
   {
      std::cout << "No intersection no transfer!" << std::endl;
   }

   return MPI_Finalize();
}
