//                         MFEM + Moonolith Example 1
//
// Compile with: make ex1
//
// Moonolith sample runs:
//   ex1
//   ex1 --source_refinements 1 --dest_refinements 2
//   ex1 --source_refinements 1 --dest_refinements 2 --use_vector_fe
//   ex1 -s ../../data/inline-hex.mesh -d ../../data/inline-tet.mesh
//
// Description:  This example code demonstrates the use of MFEM for transferring
//               discrete fields from one finite element mesh to another. The
//               meshes can be of arbitrary shape and completely unrelated with
//               each other. This feature can be used for implementing immersed
//               domain methods for fluid-structure interaction or general
//               multi-physics applications.
//
//               This particular example is only for serial runtimes.

#include "example_utils.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   // Init transfer library context
   InitTransfer(argc, argv);

   const char *source_mesh_file = "../../data/inline-tri.mesh";
   const char *destination_mesh_file = "../../data/inline-quad.mesh";

   int src_n_refinements = 0;
   int dest_n_refinements = 0;
   int source_fe_order = 1;
   int dest_fe_order = 1;
   bool visualization = true;
   bool use_vector_fe = false;
   bool verbose = false;

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
   args.AddOption(&source_fe_order, "-so", "--source_fe_order",
                  "Order of the src finite elements");
   args.AddOption(&dest_fe_order, "-do", "--dest_fe_order",
                  "Order of the dest finite elements");
   args.AddOption(&verbose, "-verb", "--verbose", "--no-verb", "--no-verbose",
                  "Enable/Disable verbose output");
   args.AddOption(&use_vector_fe, "-vfe", "--use_vector_fe", "-no-vfe",
                  "--no-vector_fe", "Use vector finite elements");
   args.Parse();
   check_options(args);

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
      mfem::err << "WARNING: Destination mesh file not found: "
                << destination_mesh_file << "\n"
                << "Using default 2D quad mesh.";

      dest_mesh = make_shared<Mesh>(4, 4, Element::QUADRILATERAL);
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
      mfem::err << "WARNING: Source mesh file not found: " << source_mesh_file
                << "\n"
                << "Using default box mesh.\n";

      if (dim == 2)
      {
         src_mesh =
            make_shared<Mesh>(4, 4, Element::TRIANGLE, 1, range[0], range[1]);
      }
      else if (dim == 3)
      {
         src_mesh = make_shared<Mesh>(4, 4, 4, Element::TETRAHEDRON, 1, range[0],
                                      range[1], range[2]);
      }

      for (int i = 0; i < src_mesh->GetNV(); ++i)
      {
         double *v = src_mesh->GetVertex(i);

         for (int d = 0; d < dim; ++d)
         {
            v[d] += box_min[d];
         }
      }
   }

   for (int i = 0; i < src_n_refinements; ++i)
   {
      src_mesh->UniformRefinement();
   }

   for (int i = 0; i < dest_n_refinements; ++i)
   {
      dest_mesh->UniformRefinement();
   }

   shared_ptr<FiniteElementCollection> src_fe_coll, dest_fe_coll;

   if (use_vector_fe)
   {
      src_fe_coll =
         make_shared<RT_FECollection>(source_fe_order, src_mesh->Dimension());
      dest_fe_coll =
         make_shared<RT_FECollection>(dest_fe_order, dest_mesh->Dimension());
   }
   else
   {
      src_fe_coll =
         make_shared<L2_FECollection>(source_fe_order, src_mesh->Dimension());
      dest_fe_coll =
         make_shared<L2_FECollection>(dest_fe_order, dest_mesh->Dimension());
   }

   auto src_fe =
      make_shared<FiniteElementSpace>(src_mesh.get(), src_fe_coll.get());

   auto dest_fe =
      make_shared<FiniteElementSpace>(dest_mesh.get(), dest_fe_coll.get());

   GridFunction src_fun(src_fe.get());
   GridFunction dest_fun(dest_fe.get());
   src_fun = 1.0;

   // To be used with standard fe
   FunctionCoefficient coeff(example_fun);

   // To be used with vector fe
   VectorFunctionCoefficient vector_coeff(dim, &vector_fun);

   if (use_vector_fe)
   {
      src_fun.ProjectCoefficient(vector_coeff);
      src_fun.Update();
   }
   else
   {
      src_fun.ProjectCoefficient(coeff);
      src_fun.Update();
   }

   dest_fun = 0.0;
   dest_fun.Update();

   MortarAssembler assembler(src_fe, dest_fe);
   assembler.SetVerbose(verbose);

   if (use_vector_fe)
   {
      assembler.AddMortarIntegrator(make_shared<VectorL2MortarIntegrator>());
   }
   else
   {
      assembler.AddMortarIntegrator(make_shared<L2MortarIntegrator>());
   }

   if (assembler.Transfer(src_fun, dest_fun))
   {
      if (visualization)
      {
         dest_fun.Update();

         double src_err = 0;
         double dest_err = 0;

         if (use_vector_fe)
         {
            src_err = src_fun.ComputeL2Error(vector_coeff);
            dest_err = dest_fun.ComputeL2Error(vector_coeff);
         }
         else
         {
            src_err = src_fun.ComputeL2Error(coeff);
            dest_err = dest_fun.ComputeL2Error(coeff);
         }

         mfem::out << "l2 error: src: " << src_err << ", dest: " << dest_err
                   << std::endl;

         plot(*src_mesh, src_fun, "source");
         plot(*dest_mesh, dest_fun, "destination");
      }
   }
   else
   {
      mfem::out << "No intersection -> no transfer!" << std::endl;
   }

   // Finalize transfer library context
   return FinalizeTransfer();
}
