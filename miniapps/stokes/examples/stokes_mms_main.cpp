#include "examples/support/mesh_manufactured.hpp"

int main(int argc, char **argv)
{
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();
   mfem::Device device("cpu");
   return mfem::flow::RunMeshMMS(argc, argv);
}
