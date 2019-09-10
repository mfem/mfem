
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

   // 7. Define a parallel finite element space on the parallel mesh. 
   FiniteElementCollection *fec;
   fec = new H1_FECollection(1, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   int mycdofoffset = fespace->GetMyDofOffset();

   // Constract an array of dof offsets on all processors
   // cout << "my rank, offset, true offset = " << fespace->GetMyRank() << ", " 
   //                                           << fespace->GetMyDofOffset() << ", " 
   //                                           << fespace->GetMyTDofOffset() << endl;

   int num_ranks = fespace->GetNRanks();
   Array<int> dof_offsets(num_ranks);
   Array<int> tdof_offsets(num_ranks);

   dof_offsets[myid] = fespace->GetMyDofOffset();
   tdof_offsets[myid] = fespace->GetMyTDofOffset();

   MPI_Allgather(&dof_offsets[myid],1,MPI_INT,dof_offsets,1,MPI_INT,MPI_COMM_WORLD);
   MPI_Allgather(&tdof_offsets[myid],1,MPI_INT,tdof_offsets,1,MPI_INT,MPI_COMM_WORLD);

   // cout << "myid, dofoffsets = " << myid << ": " ; dof_offsets.Print() ;
   // cout << "myid, tdofoffsets = " << myid << ": " ; tdof_offsets.Print() ;

   HypreParMatrix *Pr = nullptr;
   ParMesh *cpmesh = new ParMesh(*pmesh);
   HypreParMatrix *cDofTrueDof;
   for (int i = 0; i < par_ref_levels; i++)
   {
      const ParFiniteElementSpace cfespace(*fespace);
      cDofTrueDof = new HypreParMatrix(*fespace->Dof_TrueDof_Matrix());
      pmesh->UniformRefinement();
      // Update fespace
      fespace->Update();
      OperatorHandle Tr(Operator::Hypre_ParCSR);
      fespace->GetTrueTransferOperator(cfespace, Tr);
      Tr.SetOperatorOwner(false);
      HypreParMatrix *P;
      Tr.Get(P);
      if (!Pr)
      {
         Pr = P;
      }
      else
      {
         Pr = ParMult(P, Pr);
      }
   }
   Pr->Threshold(0.0);
   HypreParMatrix *DofTrueDof;
   DofTrueDof = fespace->Dof_TrueDof_Matrix();

   // cout << "myid, DtD size, cDtD size, Pr Size: " << myid <<  ", " 
   //                                     << DofTrueDof->Height() << " x " << DofTrueDof->Width() <<  ", " 
   //                                     << Pr->Height() << " x " << Pr->Width() << ", " 
   //                                     << cDofTrueDof->Height() << " x " << cDofTrueDof->Width() << endl; 


   HypreParMatrix * A = ParMult(DofTrueDof,Pr);
   HypreParMatrix * B = ParMult(A,cDofTrueDof->Transpose());


   // Pr->Transpose()->Print("H1prTdof_Tdof.mat");
   // A->Transpose()->Print("H1prTDof_Dof.mat");
   // B->Transpose()->Print("H1prDof_Dof.mat");

   // Construct the information for the patches. The construction will always use tdofs unless otherwise stated (required)

   // --------------------------------------------------
   // Step 0: Each rank identifies the number of patches and their unique identifier
   int mynrpatch = Pr->Width();
   int mynrvertices = cpmesh->GetNV();

   // The unique identifier should be given in terms of global dofs numbering (not true dofs)
   // Collect vertex identifiers for each patch and convert them to local numbering. 
   Array<int> patch_true_ids(mynrpatch);
   for (int i=0; i<mynrpatch; i++)
   {
      int k = tdof_offsets[myid]+i;
      patch_true_ids[i] = k; 
   }

   // cout << "rank id: " << myid << ", " << "patch_true_ids: " ; patch_true_ids.Print();

   // Now convert this to global dof numbering
   Array<int> patch_global_dof_ids(mynrpatch);
   SparseMatrix diag;
   cDofTrueDof->Transpose()->GetDiag(diag);
   for (int i=0; i<mynrpatch; i++)
   {
      int *col = diag.GetRowColumns(i);
      int k = dof_offsets[myid]+*col;
      patch_global_dof_ids[i] = k; 
   }
   cout << "rank id: " << myid << ", " << "patch_global_dof_ids: " ; patch_global_dof_ids.Print();

   // Now we build the list of geometric entities 
   // For this we use the H1 low order prolongation combined with Dof_TDof matrix (B matrix built above)
   SparseMatrix H1pr_diag, H1pr_offd;
   B->Transpose()->GetDiag(H1pr_diag);
   HYPRE_Int *cmap;
   B->Transpose()->GetOffd(H1pr_offd,cmap);

   vector<Array<int>> myvertex_contr(mynrpatch);
   // The non zero columns of the diag matrices are the vertex contributions from this processor
   int mydofoffset = fespace->GetMyDofOffset();
   for (int i=0; i<mynrpatch; i++)
   {
      // Pickup the row index
      int row = patch_global_dof_ids[i] - mycdofoffset;
      int row_size = H1pr_diag.RowSize(row);
      if (row_size != 0)
      {
         myvertex_contr[i].SetSize(row_size);
         int *col = H1pr_diag.GetRowColumns(row);
         for (int j = 0; j < row_size; j++)
         {
            myvertex_contr[i][j] = col[j] + mydofoffset ;
         }
      }
   }

   vector<Array<int>> vertex_contr(mynrpatch);
   // The non zero columns of the diag matrices are the vertex contributions from this processor
   for (int i=0; i<mynrpatch; i++)
   {
      // Pickup the row index
      int row = patch_global_dof_ids[i] - mycdofoffset;
      int row_size = H1pr_offd.RowSize(row);
      if (row_size != 0)
      {
         vertex_contr[i].SetSize(row_size);
         int *col = H1pr_offd.GetRowColumns(row);
         for (int j = 0; j < row_size; j++)
         {
            vertex_contr[i][j] = cmap[col[j]] ;
         }
      }
   }



   // // Finally print the vertices corresponding to each patch
   // if (myid == 0) 
   // {
   //    cout << "my id = " << myid << endl;
   //    for (int i=0; i<mynrpatch; i++)
   //    {
   //    int size = myvertex_contr[i].Size() + vertex_contr[i].Size();
   //    if (size != 0) 
   //        cout << "patch no, vertices = " << patch_global_dof_ids[i] << "," ; myvertex_contr[i].Print(); cout << " and " ; vertex_contr[i].Print();
   //    }
   // }
   // MPI_Barrier(MPI_COMM_WORLD);
   // if (myid == 1) 
   // {
   //    cout << "my id = " << myid << endl;
   //    for (int i=0; i<mynrpatch; i++)
   //    {
   //    int size = myvertex_contr[i].Size() + vertex_contr[i].Size();
   //    if (size != 0) 
   //        cout << "patch no, vertices = " << patch_global_dof_ids[i] << "," ; myvertex_contr[i].Print(); cout << " and " ; vertex_contr[i].Print();
   //    }
   // }
   // MPI_Barrier(MPI_COMM_WORLD);
   // if (myid == 2) 
   // {
   //    cout << "my id = " << myid << endl;
   //    for (int i=0; i<mynrpatch; i++)
   //    {
   //    int size = myvertex_contr[i].Size() + vertex_contr[i].Size();
   //    if (size != 0) 
   //        cout << "patch no, vertices = " << patch_global_dof_ids[i] << "," ; myvertex_contr[i].Print(); cout << " and " ; vertex_contr[i].Print();
   //    }
   // }

   // MPI_Barrier(MPI_COMM_WORLD);


   // if we want to remove the dublicated (common vertices from the patches) we can use the diag Dof_Tdof matrix
   DofTrueDof->GetDiag(diag);
   Array<int> own_vertices;
   int nv=0;
   for (int k=0; k<diag.Height(); k++)
   {
      int nz = diag.RowSize(k);
      int i = mydofoffset + k;
      if (nz != 0)
      {
         nv++;
         own_vertices.SetSize(nv);
         own_vertices[nv-1] = i;
      }
   }
   cout << "myid = " << myid << " my vertices : "; own_vertices.Print();

   // Array<int> vertex_dofs;
   // if(myid == 1)
   // {
   //    cout << fespace->GetGlobalTDofNumber(0) << endl;
   // }

   // SparseMatrix temp;
   // B->GetDiag(temp);
   // Array<int> vertexdofs;
   // int nv = pmesh->GetNV();
   // for (int i=0; i<nv; i++)
   // {
   //    fespace->GetVertexVDofs(i,vertexdofs);
   //    if (myid == 0)
   //    {
   //       cout << myid  << ": " ; vertexdofs.Print() ;
   //    }
   // }

   // MPI_Barrier(MPI_COMM_WORLD);

   // if (myid == 2)
   // {
   //    cout << fespace->GetMyDofOffset() << endl;
   //    cout << fespace->GetMyTDofOffset() << endl;

   //    for (int i = 0; i < nv; i++)
   //    {
   //       fespace->GetVertexDofs(i, vertexdofs);

   //       // cout << myid << ": ";
   //       vertexdofs.Print();
   //    }
   // }

   ParGridFunction x(fespace);
   x = 0.0;

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      int num_procs, myid;
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);
      char vishost[] = "localhost";
      int visport = 19916;
      // socketstream cmesh_sock(vishost, visport);
      // cmesh_sock << "parallel " << num_procs << " " << myid << "\n";
      // cmesh_sock.precision(8);
      socketstream mesh_sock(vishost, visport);
      mesh_sock << "parallel " << num_procs << " " << myid << "\n";
      mesh_sock.precision(8);
      // socketstream sol_sock(vishost, visport);
      // sol_sock << "parallel " << num_procs << " " << myid << "\n";
      // sol_sock.precision(8);

      if (dim == 2)
      {
         mesh_sock << "mesh\n"
                   << *pmesh << "keys nn\n"
                   << flush;
         // cmesh_sock << "mesh\n"
         //            << *cpmesh << "keys nn\n"
         //            << flush;
         // sol_sock << "solution\n" << *pmesh << x << "keys mrRljc\n" << flush;
      }
      else
      {
         mesh_sock << "mesh\n"
                   << *pmesh << flush;
         // sol_sock << "solution\n"
         //          << *pmesh << x << flush;
      }
   }

   if (visualization)
   {
      int num_procs, myid;
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
      MPI_Comm_rank(MPI_COMM_WORLD, &myid);
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream cmesh_sock(vishost, visport);
      cmesh_sock << "parallel " << num_procs << " " << myid << "\n";
      cmesh_sock.precision(8);

      cmesh_sock << "mesh\n" << *cpmesh << "keys nn\n"  << flush;
   }

   // 17. Free the used memory.
   delete fespace;
   delete fec;
   delete pmesh;

   return 0;
}
