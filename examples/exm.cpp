#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   //Mesh mesh = Mesh::MakeCartesian3D(2, 2 ,2, Element::Type::HEXAHEDRON);
   Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, Element::Type::TETRAHEDRON);

   // Build faces and boundary
   mesh.FinalizeTopology();
   mesh.Finalize();

   // Changing element and boundary attributes
   for (int i=0; i<mesh.GetNE(); ++i)
   {
      mesh.SetAttribute(i, myid + 1);
   }

   for (int i=0; i<mesh.GetNBE(); ++i)
   {
      mesh.SetBdrAttribute(i, 100);
   }

   mesh.SetAttributes();

   // Add internal boundary facets used for integrators
   // TODO: what should be added here?

   // Finalize connectivity and topology (is this even needed?)
   mesh.FinalizeTopology();
   mesh.Finalize(true);

   // Make sure mesh is non-conforming
   mesh.EnsureNCMesh(true);

   // Make parallel mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   pmesh.EnsureNCMesh(true);

   // Refinement
   Array<Refinement> refinements;
   refinements.Append(Refinement(0));  // Local element 0 on this rank

   pmesh.GeneralRefinement(refinements);
   pmesh.SetAttributes();

   {
      ostringstream mesh_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);
   }

   return 0;
}
