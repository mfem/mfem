
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void get_solution(const Vector &x, double &u, double &d2u);
double u_exact(const Vector &x);
double f_exact(const Vector &x);

int isol = 0;
int dim;
double omega;

int main(int argc, char *argv[])
{
   // 1. Initialise MPI
   MPI_Session mpi(argc, argv);

   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   const char *mesh_file = "../data/one-hex.mesh";
   int order = 1;
   int sdim = 2;
   bool static_cond = false;
   const char *device_config = "cpu";
   bool visualization = true;
   int ref_levels = 1;
   int par_ref_levels = 1;
   int initref = 1;
   // number of wavelengths
   double k = 0.5;
   double theta = 0.5;
   double smth_maxit = 1;
   StopWatch chrono;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sdim, "-d", "--dimension", "Dimension");
   args.AddOption(&ref_levels, "-sr", "--serial-refinements",
                  "Number of mesh refinements");
   args.AddOption(&par_ref_levels, "-pr", "--parallel-refinements",
                  "Number of parallel mesh refinements");
   args.AddOption(&initref, "-iref", "--init-refinements",
                  "Number of initial mesh refinements");
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&smth_maxit, "-sm", "--smoother-maxit",
                  "Number of smoothing steps.");
   args.AddOption(&theta, "-th", "--theta",
                  "Dumping parameter for the smoother.");
   args.AddOption(&isol, "-sol", "--solution",
                  "Exact Solution: 0) Polynomial, 1) Sinusoidal.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   omega = 2.0 * M_PI * k;

   // 3. Read the mesh from the given mesh file.
   // Mesh *mesh = new Mesh(mesh_file, 1, 1);

   Mesh *mesh;
   // Define a simple square or cubic mesh
   if (sdim == 2)
   {
      mesh = new Mesh(1, 1, Element::QUADRILATERAL, true, 1.0, 1.0, false);
      // mesh = new Mesh(1, 1, Element::TRIANGLE, true,1.0, 1.0,false);
   }
   else
   {
      mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, 1.0, 1.0, 1.0, false);
   }
   dim = mesh->Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   // for (int l = 0; l < par_ref_levels; l++) {pmesh->UniformRefinement();}

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();

   // Constract an array of dof offsets on all processors
   // cout << "my rank, offset, true offset = " << fespace->GetMyRank() << ", " 
   //                                           << fespace->GetMyDofOffset() << ", " 
   //                                           << fespace->GetMyTDofOffset() << endl;

   int num_ranks = fespace->GetNRanks();
   Array<int> dof_offsets(num_ranks);
   Array<int> tdof_offsets(num_ranks);

   dof_offsets[myid] = fespace->GetMyDofOffset();
   tdof_offsets[myid] = fespace->GetMyTDofOffset();
   MPI_Status status;
   MPI_Request reqs;
   MPI_Request reqr;

   //--------------------------------------------------------------------------
   // Implementation 0:
   //--------------------------------------------------------------------------
   // MPI_Allgather(&dof_offsets[myid],1,MPI_INT,dof_offsets,1,MPI_INT,MPI_COMM_WORLD);
   // MPI_Allgather(&tdof_offsets[myid],1,MPI_INT,tdof_offsets,1,MPI_INT,MPI_COMM_WORLD);

   //--------------------------------------------------------------------------
   // Implementation 1:
   //--------------------------------------------------------------------------
   // for (int i = 0; i<num_ranks; i++)
   // {
   //    MPI_Send(&dof_offsets[myid],1,MPI_INT,i,myid,MPI_COMM_WORLD);
   //    MPI_Send(&tdof_offsets[myid],1,MPI_INT,i,myid,MPI_COMM_WORLD);
   //    MPI_Recv(&dof_offsets[i],1,MPI_INT,i,i,MPI_COMM_WORLD, &status);
   //    MPI_Recv(&tdof_offsets[i],1,MPI_INT,i,i,MPI_COMM_WORLD, &status);
   // }

   //--------------------------------------------------------------------------
   // Implementation 2:
   //--------------------------------------------------------------------------
   for (int i = 0; i<num_ranks; i++)
   {
      if( myid != i)
      {
         MPI_Isend(&dof_offsets[myid],1,MPI_INT,i,myid,MPI_COMM_WORLD,&reqs);
         MPI_Isend(&tdof_offsets[myid],1,MPI_INT,i,myid,MPI_COMM_WORLD,&reqs);
      }
   }
   for (int i = 0; i<num_ranks; i++)
   {
      if( myid != i)
      {
         MPI_Recv(&dof_offsets[i],1,MPI_INT,i,i,MPI_COMM_WORLD, &status);
         MPI_Recv(&tdof_offsets[i],1,MPI_INT,i,i,MPI_COMM_WORLD, &status);
      }
   }

   //--------------------------------------------------------------------------
   // Implementation 3:
   //--------------------------------------------------------------------------
   // for (int i = 0; i<num_ranks; i++)
   // {
   //    if( myid != i)
   //    {
   //       MPI_Isend(&dof_offsets[myid],1,MPI_INT,i,myid,MPI_COMM_WORLD,&reqs);
   //       MPI_Isend(&tdof_offsets[myid],1,MPI_INT,i,myid,MPI_COMM_WORLD,&reqs);
   //    }
   // }
   // for (int i = 0; i<num_ranks; i++)
   // {
   //    if( myid != i)
   //    {
   //       MPI_Irecv(&dof_offsets[i],1,MPI_INT,i,i,MPI_COMM_WORLD, &reqr);
   //       MPI_Irecv(&tdof_offsets[i],1,MPI_INT,i,i,MPI_COMM_WORLD, &reqr);
   //    }
   // }
   // MPI_Wait(&reqr, &status);

   //--------------------------------------------------------------------------

   cout << "myid, dofoffsets = " << myid << ": " ; dof_offsets.Print() ;
   cout << "myid, tdofoffsets = " << myid << ": " ; tdof_offsets.Print() ;

   // 17. Free the used memory.
   delete fespace;
   delete fec;
   delete pmesh;
   return 0;
}
