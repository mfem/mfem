
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

bool its_a_patch(int iv, Array<int> patch_ids);

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

   // 6. Define a parallel mesh 
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 7. Define a parallel finite element space on the parallel mesh. 
   FiniteElementCollection *fec = new H1_FECollection(1, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   int mycdofoffset = fespace->GetMyDofOffset();

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
   HypreParMatrix * A = ParMult(DofTrueDof,Pr);
   HypreParMatrix * B = ParMult(A,cDofTrueDof->Transpose());



   SparseMatrix cdiag, coffd;
   cDofTrueDof->GetDiag(cdiag);
   Array<int> cown_vertices;
   int cnv=0;
   for (int k=0; k<cdiag.Height(); k++)
   {
      int nz = cdiag.RowSize(k);
      int i = mycdofoffset + k;
      if (nz != 0)
      {
         cnv++;
         cown_vertices.SetSize(cnv);
         cown_vertices[cnv-1] = i;
      }
   }

   int mynrpatch = cown_vertices.Size();
   int nrpatch;
   // Compute total number of patches.
   MPI_Allreduce(&mynrpatch,&nrpatch,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
   Array <int> patch_global_dofs_ids(nrpatch);
   // Create a list of patches identifiers to all procs
   MPI_Gather(&cown_vertices[0],mynrpatch,MPI_INT,&patch_global_dofs_ids[0],mynrpatch,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&patch_global_dofs_ids[0],nrpatch,MPI_INT,0,MPI_COMM_WORLD);

   
   // Now on each processor we identify the vertices that it owns (fine grid)
   SparseMatrix diag;
   DofTrueDof->GetDiag(diag);
   Array<int> own_vertices;
   int nv=0;
   for (int k=0; k<diag.Height(); k++)
   {
      int nz = diag.RowSize(k);
      int i = fespace->GetMyDofOffset() + k;
      if (nz != 0)
      {
         nv++;
         own_vertices.SetSize(nv);
         own_vertices[nv-1] = i;
      }
   }

   int mynrvertices = own_vertices.Size();
   vector<Array<int>> vertex_contr(mynrvertices);
   SparseMatrix H1pr_diag;
   SparseMatrix H1pr_offd;
   int *cmap;
   B->GetDiag(H1pr_diag);
   B->GetOffd(H1pr_offd,cmap);
   /* each process will go through its owned vertices and create a list of the patches 
   that contributes to. First the for the patches that are owned ny the processor */
   for (int i = 0; i<mynrvertices; i++)
   {
      int kv = 0;
      int iv = own_vertices[i];
      int row = iv - fespace->GetMyDofOffset();
      int row_size = H1pr_diag.RowSize(row);
      int *col = H1pr_diag.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         int jv = col[j] + mycdofoffset;
         if (its_a_patch(jv,patch_global_dofs_ids))
         {
            kv++;
            vertex_contr[i].SetSize(kv);
            vertex_contr[i][kv-1] = jv; 
         }
      }
   }
   // Next for the patches which are not owned by the processor.
   for (int i = 0; i<mynrvertices; i++)
   {
      int kv = vertex_contr[i].Size();
      int iv = own_vertices[i];
      int row = iv - fespace->GetMyDofOffset();
      int row_size = H1pr_offd.RowSize(row);
      int *col = H1pr_offd.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         int jv = cmap[col[j]];
         if (its_a_patch(jv,patch_global_dofs_ids))
         {
            kv++;
            vertex_contr[i].SetSize(kv);
            vertex_contr[i][kv-1] = jv; 
         }
      }
   }
   

   if (myid == 2)
   {
      for (int i = 0; i<mynrvertices; i++)
      {
         cout << "vertex number, vertex id: " << i << ", " << own_vertices[i] << endl; 
         cout << "contributes to: " ; vertex_contr[i].Print(); 
         Array<int> vertex_contr_index(vertex_contr[i].Size());
         for (int j = 0; j<vertex_contr[i].Size(); j++)
         {
            vertex_contr_index[j] = patch_global_dofs_ids.FindSorted(vertex_contr[i][j]);
         }
         cout << "contributes to: " ; vertex_contr_index.Print(); 
      }
      // cout << "vertex patches no = "; patch_global_dofs_ids.Print();
   }

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


bool its_a_patch(int iv, Array<int> patch_ids)
{
   if (patch_ids.FindSorted(iv)== -1)
   {
      return false;
   }
   else
   {
      return true;
   }
}
