//compare operators from loaded mesh vs generated mesh
#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

bool yregion(const Vector &x, const double y0)
{
   return std::max(-x(1)-y0, x(1) - y0)<1e-8;
}

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   //++++Parse command-line options.
   const char *mesh_file = "./Meshes/xperiodic-square.mesh";
   int ser_ref_levels = 1;
   int order = 2;
   bool local_refine = true;
   int local_refine_levels = 2;

   bool readmesh=false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&readmesh, "-read", "--read", "-no-read",
                  "--no-read", "Read or generate mesh.");
   args.AddOption(&local_refine, "-local", "--local-refine", "-no-local", "--no-local-refine",
                  "Enable or disable local refinement before unifrom refinement.");
   args.AddOption(&local_refine_levels, "-lr", "--local-refine",
                  "Number of levels to refine locally.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   
   if (myid == 0) args.PrintOptions(cout);

   int dim = 2;
   ParMesh *pmesh;

   if(!readmesh)
   {
      Mesh *mesh = new Mesh(mesh_file, 1, 1);

      //++++++Refine the mesh to increase the resolution.    
      for (int lev = 0; lev < ser_ref_levels; lev++)
      {
         mesh->UniformRefinement();
      }

      //++++++Refine locally first    
      if (local_refine)
      {
         for(int lev=0; lev<local_refine_levels; lev++)
         {

           Vector pt;
           Array<int> marked_elements;
           for (int i = 0; i < mesh->GetNE(); i++)
           {
              // check all nodes of the element
              IsoparametricTransformation T;
              mesh->GetElementTransformation(i, &T);
              for (int j = 0; j < T.GetPointMat().Width(); j++)
              {
                 T.GetPointMat().GetColumnReference(j, pt);

                   double x0, y0;
                   switch (lev)
                   {
                       case 0: y0=0.5; break;
                       case 1: y0=0.3; break;
                       case 2: y0=0.2; break;
                       default:
                           if (myid == 0) cout << "Unknown level: " << lev << '\n';
                           delete mesh;
                           MPI_Finalize();
                           return 3;
                   }
                   if (yregion(pt, y0))
                   {
                          marked_elements.Append(i);
                          break;
                   }
                 
              }
           }
           mesh->GeneralRefinement(marked_elements);
         }
      }

      //+++++++here we need to generate a partitioning because the default one is wrong for ncmesh when local_refine is truned on
      int *partitioning = NULL;
      partitioning=mesh->GeneratePartitioning(num_procs, 1);
 
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);
      delete mesh;
      delete partitioning;
   }
   else
   {
      ifstream ifs(MakeParFilename("mesh_test.", myid));
      MFEM_VERIFY(ifs.good(), "Mesh file not found.");
      pmesh = new ParMesh(MPI_COMM_WORLD, ifs);
   }

   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll); 

   HYPRE_Int global_size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "dim: " << dim << endl;
      cout << "Number of total scalar unknowns: " << global_size << endl;
   }
   int fe_size = fespace.TrueVSize();
   cout << "TrueVSize is: " << fe_size<<" in id = "<<myid << endl;

   //++++++this is a periodic boundary condition in x and Direchlet in y 
   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;  

   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   //output ess_tdof_list
   {
       ostringstream dof_save;
       dof_save << "dof." << setfill('0') << setw(6) << myid;
       ofstream myfile(dof_save.str().c_str());
       ess_tdof_list.Print(myfile, 1000);
   }

   HypreParMatrix Mmat;

   //mass matrix
   ParBilinearForm *M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator);
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   //output mass matrix
   hypre_CSRMatrix* A_serial = hypre_ParCSRMatrixToCSRMatrixAll(Mmat);
   mfem::SparseMatrix A_sparse(
   hypre_CSRMatrixI(A_serial), hypre_CSRMatrixJ(A_serial), hypre_CSRMatrixData(A_serial),
   hypre_CSRMatrixNumRows(A_serial), hypre_CSRMatrixNumCols(A_serial),
   false, false, true);

   if (myid == 0)
   {
      ofstream outfile("Mmat.m");
      A_sparse.PrintMatlab(outfile);
      outfile.close();
   }

   if(!readmesh)
   {
      ostringstream mesh_save, mesh_save2, mat_save;
      mesh_save << "mesh_test." << setfill('0') << setw(6) << myid;
      mesh_save2 << "mesh_vis." << setfill('0') << setw(6) << myid;

      ofstream ncmesh(mesh_save.str().c_str());
      ncmesh.precision(16);
      pmesh->ParPrint(ncmesh);

      ofstream ncmesh2(mesh_save2.str().c_str());
      ncmesh2.precision(8);
      pmesh->Print(ncmesh2);
   }
   
   delete M;
   delete pmesh;

   MPI_Finalize();

   return 0;
}



