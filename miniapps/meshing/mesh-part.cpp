#include "mfem.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int npart = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&npart, "-n", "--num-partitions",
                  "Number of partitions in output mesh.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);


   int part_method = 1;

   Mesh mesh(mesh_file, 1, 1);
   int *partitioning = mesh.GeneratePartitioning(npart, part_method);


   mesh.ParPrint("pmesh", npart, partitioning);

   // ostringstream oss; oss << "mesh-part.pmesh" << "." << setw(5) << setfill('0') << myid;
   // ofstream ofs(oss.str().c_str());

   /*
   for (int p=0; p<num_procs; p++)
   {
   Mesh mesh(mesh_file, 1, 1);
   int *partitioning = mesh.GeneratePartitioning(num_procs, part_method);

   ParMesh pmesh(MPI_COMM_WORLD, mesh, partitioning);

   ostringstream oss; oss << "mesh-part.pmesh" << "." << setw(5) << setfill('0') << p;
   ofstream ofs(oss.str().c_str());
   pmesh.ParPrint(ofs);
   }
   */
}
