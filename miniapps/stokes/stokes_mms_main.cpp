#include "stokes_mesh_mms.hpp"

int main(int argc, char **argv)
{
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();
   mfem::Device device("cpu");
   return mfem::stokes::RunMeshMMS(argc, argv);
}
