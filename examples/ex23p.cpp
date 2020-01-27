//                                MFEM Example 23
//
// Compile with: make ex23
//
// Sample runs:  ex23 -m ../data/square-disc.mesh
//
// Device sample runs:
//               ex23 -pa -d cuda
//
// Description:  This example code
//               s=0: Catenoid
//               s=1: Helicoid
//               s=2: Enneper
//               s=3: Scherk
//               s=4: Shell
//               s=5: Hold
//               s=6: QPeach
//               s=7: FPeach
//               s=8: SlottedSphere

#define XMesh ParMesh
#define XGridFunction ParGridFunction
#define XBilinearForm ParBilinearForm
#define XFiniteElementSpace ParFiniteElementSpace
#define XMeshConstructor(this) ParMesh(MPI_COMM_WORLD, *this)
#define XInit(num_procs, myid){\
  MPI_Init(&argc, &argv);\
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);\
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);}
#define XCGArguments (MPI_COMM_WORLD)
#define XPreconditioner new HypreBoomerAMG
#define XFinalize MPI_Finalize

#include "ex23.cpp"
