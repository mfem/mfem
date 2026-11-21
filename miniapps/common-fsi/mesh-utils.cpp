//            -------------------------------
//                 Mesh Utils executable
//            -------------------------------
//
// This executable is used to export mesh domains, boundaries and partitioning for visualization.
// A mesh can be loaded from a file or generated uniformly.
//
// Sample runs:
// 1. Load mesh and patition it into 4 parts
//          ./mesh-utils -m ../../data/<MESH-FILE> -p 4 -of ./Output
// 2. Generate a uniform quad mesh and partition it into 4 parts
//          ./mesh-utils -d 2 -e 0 -n 10 -p 4 -of ./Output

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <limits>
#include <cstdlib>
#include "FilesystemHelper.hpp"  
#include "fsi_utils.hpp"

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   //
   /// 1. Parse command-line options.
   //
   const char *mesh_file = "";  // mesh
   int n = 10; 
   int dim = 2;
   int elem = 0;
   int order = 1; // FE order of mesh nodes

   int ser_ref_levels = 0; // mesh refinement

   int partitions = 0; // partitioning
   int partitioning_type = 1;

   const char *folderPath = "./"; // output

   // TODO: check parsing and assign variables
   OptionsParser args(argc, argv);
   // Mesh file or Mesh generation
   args.AddOption(&mesh_file,
                  "-m",
                  "--mesh",
                  "Mesh file to use.");
   args.AddOption(&dim,
                  "-d",
                  "--dimension",
                  "Dimension of the problem (2 = 2d, 3 = 3d)");
   args.AddOption(&elem,
                  "-e",
                  "--element-type",
                  "Type of elements used (0: Quad/Hex, 1: Tri/Tet)");
   args.AddOption(&n,
                  "-n",
                  "--num-elements",
                  "Number of elements in uniform mesh.");
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Order of the finite elements.");
   // Mesh refinement
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   // Partitioning
   args.AddOption(&partitions,
                  "-p",
                  "--partitions",
                  "Number of partitions.");
   args.AddOption(&partitioning_type,
                  "-pt",
                  "--partitioning-type",
                  "Type of partitioning (0: METIS_PartGraphRecursive, 1: METIS_PartGraphKway, 2: METIS_PartGraphVKway)");
   // Output
   args.AddOption(&folderPath,
                  "-of",
                  "--output-folder",
                  "Output folder.");
   args.Parse();

   args.ParseCheck();


   //
   /// 2. Read the (serial) mesh from the given mesh file on all processors.
   //

   Mesh mesh;

   if (mesh_file[0] != '\0')
   {
      mfem::out << "Reading serial mesh file: " << mesh_file << std::endl;
      mesh = Mesh::LoadFromFile(mesh_file);
      dim = mesh.Dimension();
   }
   else // Create square mesh
   {
      mfem::out << "Generating a uniform mesh..." << std::endl;
      Element::Type type;
      switch (elem)
      {
      case 0: // quad
         type = (dim == 2) ? Element::QUADRILATERAL : Element::HEXAHEDRON;
         break;
      case 1: // tri
         type = (dim == 2) ? Element::TRIANGLE : Element::TETRAHEDRON;
         break;
      }

      switch (dim)
      {
      case 2: // 2d
         mesh = Mesh::MakeCartesian2D(n, n, type, true);
         break;
      case 3: // 3d
         mesh = Mesh::MakeCartesian3D(n, n, n, type, true);
         break;
      }
   }


   // Serial Refinement
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   //
   /// 3. Generate the partitioning of the mesh
   //
   // Partition type:
   // 0) METIS_PartGraphRecursive (sorted neighbor lists)
   // 1) METIS_PartGraphKway      (sorted neighbor lists) (default)
   // 2) METIS_PartGraphVKway     (sorted neighbor lists)
   // 3) METIS_PartGraphRecursive
   // 4) METIS_PartGraphKway
   // 5) METIS_PartGraphVKway
   int *partitioning;
   if (partitions > 0)
   {
      mfem::out << "Generating partitioning..." << std::endl;
      partitioning = mesh.GeneratePartitioning(partitions, partitioning_type);}

   // Create gridfunction for visualization of quadrature points
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction gf(&fespace);
   gf = 0.0;

   ParaViewDataCollection paraview_dc("QP", &mesh);
   paraview_dc.SetPrefixPath(folderPath);
   paraview_dc.RegisterField("zero", &gf);
   if (order > 1)
   {
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetLevelsOfDetail(order);
   }
   paraview_dc.Save();

   
   //
   /// 4. Save the refined mesh 
   //
   
   mfem::out << "Saving the mesh..." << std::endl;

   if (!fs::is_directory(folderPath) || !fs::exists(folderPath)) { // Check if folder exists
      fs::create_directories(folderPath); // create folder
   }

   // Save mesh
   std::ostringstream mesh_name;
   std::ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);

   // Saver mesh boundaries
   std::ostringstream omesh_file, omesh_file_bdr;
   omesh_file << folderPath << "/mesh.vtk";
   omesh_file_bdr << folderPath << "/mesh_bdr";
   std::fstream omesh(omesh_file.str().c_str(), std::ios::out);
   omesh.precision(14);
   mesh.PrintVTK(omesh);
   mesh.PrintBdrVTU(omesh_file_bdr.str());

   // Save partitioning
   if (partitions > 0)
      common_fsi_utils::ExportMeshwithPartitioning(folderPath, mesh, partitioning);

   // Free used memory.

   return 0;
}
