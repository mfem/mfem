//                                MFEM Example Print Mesh
//
// Compile with: make exPrintMesh
//
// Sample runs:  exPrintMesh
//               exPrintMesh -m ../data/fichera.mesh
//
// Description: This example code prints out information about the mesh and
//              optionally visualizes and/or saves it in MFEM and vtu formats
//              It is useful for tests that are meant to compare different
//              mesh readers.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   string mesh_file = "../data/star.mesh";
   bool visualization = 0;
   bool dump = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable/disable GLVis visualization.");
   args.AddOption(&dump, "-dump", "--dump_mesh", "-no-dump",
                  "--no-dump_mesh",
                  "Enable/disable dumping the mesh in MFEM and vtu formats.");
   args.ParseCheck();

   // 1. Read the mesh from the given mesh file, and print info.
   Mesh mesh(mesh_file);
   mesh.PrintInfo();

   // 2. Dump the mesh.
   if (dump)
   {
      size_t lastindex = mesh_file.find_last_of(".");
      string meshname = mesh_file.substr(0, lastindex);
      mesh.PrintVTU("exPrintMesh.vtu");
      ofstream mesh_out("exPrintMesh.mesh");
      mesh.Print(mesh_out);
      mesh_out.close();
   }

   // 3. Send the mesh by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << mesh << flush;
   }

   return 0;
}
