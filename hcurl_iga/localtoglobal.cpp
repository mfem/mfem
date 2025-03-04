#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#include <mpi.h>

int main(int argc, char** argv)
{
   int n_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   const char *mesh_file = "../data/star.mesh";
   Mesh* mesh = new Mesh(mesh_file, 1, 1);
   for (int i=0; i<5; i++) mesh->UniformRefinement();
   ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   int dim = mesh->Dimension();
   int order = 1;

   int* partitioning = mesh->GeneratePartitioning(n_procs);
   FiniteElementCollection* fec = new H1_FECollection(order, dim);
   FiniteElementSpace* glob_fes = new FiniteElementSpace(mesh, fec);
   ParFiniteElementSpace* loc_fes = new ParFiniteElementSpace(pmesh, glob_fes, partitioning, fec);
   HYPRE_Int size = loc_fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   ParGridFunction* loc_gf = new ParGridFunction(loc_fes);
   (*loc_gf) = 0.0;

   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      loc_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ParLinearForm* b = new ParLinearForm(loc_fes);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   ParBilinearForm* a = new ParBilinearForm(loc_fes);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();
   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, *loc_gf, *b, A, X, B);

   Solver *prec = new HypreBoomerAMG;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);
   delete prec;
   a->RecoverFEMSolution(X, *b, *loc_gf);

   MPI_Barrier(MPI_COMM_WORLD);

   GridFunction* glob_gf = NULL;   // owned only by root process
   if (myid == 0) {
      glob_gf = new GridFunction(glob_fes);
   }
   Array<int> glob_vdofs, loc_vdofs;
   Vector loc_values;
   int element_counter = 0;
   const int glob_ne = glob_fes->GetNE();

   MPI_Request request;
   MPI_Status status;
   double* message_loc_values;
   int count;

   for (int i = 0; i < glob_ne; i++) {
      if (partitioning[i] == myid) {
         loc_fes->GetElementVDofs(element_counter, loc_vdofs);
         loc_gf->GetSubVector(loc_vdofs, loc_values);

         if (myid == 0) {  // root process, set values
            glob_fes->GetElementVDofs(i, glob_vdofs);
            glob_gf->SetSubVector(glob_vdofs, loc_values);
         }
         else {   // other process, send values to root process
            count = loc_values.Size();
            message_loc_values = new double[count];
            message_loc_values = loc_values.GetData();
            MPI_Send(&count, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
            // MPI_Isend(&count, 1, MPI_INT, 0, 99, MPI_COMM_WORLD, &request);
            // MPI_Wait(&request, &status);

            MPI_Send(message_loc_values, count, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
            // MPI_Isend(message_loc_values, count, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &request);
            // MPI_Wait(&request, &status);
         }

         element_counter++;
      }

      if (myid==0 && partitioning[i] != 0) {
         MPI_Recv(&count, 1, MPI_INT, partitioning[i], 99, MPI_COMM_WORLD, &status);
         // MPI_Irecv(&count, 1, MPI_INT, partitioning[i], 99, MPI_COMM_WORLD, &request);
         // MPI_Wait(&request, &status);

         message_loc_values = new double[count];
         MPI_Recv(message_loc_values, count, MPI_DOUBLE, partitioning[i], 100, MPI_COMM_WORLD, &status);
         // MPI_Irecv(message_loc_values, count, MPI_DOUBLE, partitioning[i], 100, MPI_COMM_WORLD, &request);
         // MPI_Wait(&request, &status);

         glob_fes->GetElementVDofs(i, glob_vdofs);
         loc_values = Vector(message_loc_values, count);
         glob_gf->SetSubVector(glob_vdofs, loc_values);
      }

   }

   MPI_Barrier(MPI_COMM_WORLD);

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex_par2seq_mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex_par2seq_sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      loc_gf->SaveAsOne(sol_ofs);
   }

   if (myid == 0) {
      std::string out_str("ex_par2seq_solution.dat");
      std::ofstream out(out_str);
      glob_gf->Save(out);
   }


   delete a;
   delete b;
   delete loc_fes;
   delete fec;
   delete pmesh;
   MPI_Finalize();

   return 0;
}