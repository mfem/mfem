#include "parproblems_util.hpp"

int get_rank(int tdof, std::vector<int> & tdof_offsets)
{
   int size = tdof_offsets.size();
   if (size == 1) { return 0; }
   std::vector<int>::iterator up;
   up=std::upper_bound(tdof_offsets.begin(), tdof_offsets.end(),tdof); //
   return std::distance(tdof_offsets.begin(),up)-1;
}

void ComputeTdofOffsets(const ParFiniteElementSpace * pfes,
                        std::vector<int> & tdof_offsets)
{
   MPI_Comm comm = pfes->GetComm();
   int num_procs;
   MPI_Comm_size(comm, &num_procs);
   tdof_offsets.resize(num_procs);
   int mytoffset = pfes->GetMyTDofOffset();
   MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
}

void ComputeTdofOffsets(MPI_Comm comm, int mytoffset, std::vector<int> & tdof_offsets)
{
   int num_procs;
   MPI_Comm_size(comm,&num_procs);
   tdof_offsets.resize(num_procs);
   MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
}

void ComputeTdofs(MPI_Comm comm, int mytoffs, std::vector<int> & tdofs)
{
   int num_procs;
   MPI_Comm_size(comm,&num_procs);
   tdofs.resize(num_procs);
   MPI_Allgather(&mytoffs,1,MPI_INT,&tdofs,1,MPI_INT,comm);
}         


// Performs Pᵀ * A * P for BlockOperator  P (with blocks as HypreParMatrices)
// and A a HypreParMatrix, i.e., this handles the special case 
// where P = [P₁ P₂ ⋅⋅⋅ Pₙ] 
// C = Pᵀ * A * P 
void RAP(const HypreParMatrix & A, const BlockOperator & P, 
         BlockOperator & C)
{
   int nblocks = P.NumColBlocks();

   const HypreParMatrix * Pi = nullptr;
   const HypreParMatrix * Pj = nullptr;
   HypreParMatrix * PitAPj = nullptr; 

   for (int i = 0; i< nblocks; i++)
   {
      if (P.IsZeroBlock(0,i)) continue;
      Pi = dynamic_cast<const HypreParMatrix*>(&P.GetBlock(0,i));
      for (int j = 0; j<nblocks; j++)
      {
         if (P.IsZeroBlock(0,j)) continue;
         Pj = dynamic_cast<const HypreParMatrix*>(&P.GetBlock(0,j));
         if (i == j) 
         {
            PitAPj = RAP(&A, Pj); 
         }
         else
         {
            PitAPj = RAP(Pi, &A, Pj); 
         }
         C.SetBlock(i,j,PitAPj);
      }
   }
}

void ParAdd(const BlockOperator & A, const BlockOperator & B, BlockOperator & C)
{
   int n = A.NumRowBlocks();
   int m = A.NumColBlocks();
   MFEM_VERIFY(B.NumRowBlocks() == n, "Inconsistent number of row blocks");
   MFEM_VERIFY(B.NumColBlocks() == m, "Inconsistent number of column blocks");

   const HypreParMatrix * a;
   const HypreParMatrix * b;
   for (int i = 0; i<n; i++)
   {
      for (int j = 0; j<m; j++)
      {
         a = nullptr;
         b = nullptr;
         if (!A.IsZeroBlock(i,j))
         {
            a = dynamic_cast<const HypreParMatrix*>(&A.GetBlock(i,j));
         }
         if (!B.IsZeroBlock(i,j))
         {
            b = dynamic_cast<const HypreParMatrix*>(&B.GetBlock(i,j));
         }
         if (a && b)
         {
            C.SetBlock(i,j,ParAdd(a,b));
         }
         else if (a)
         {
            C.SetBlock(i,j,new HypreParMatrix(*a));
         }
         else if (b)
         {
            C.SetBlock(i,j,new HypreParMatrix(*b));
         }
         else
         {
            // do nothing
         }
      }
   }
}


GeneralSolutionMonitor::GeneralSolutionMonitor(ParFiniteElementSpace * fes_, HypreParMatrix * A, Vector & B, int output_rate_) : fes(fes_), output_rate(output_rate_)
{
   MUMPSSolver mumps(MPI_COMM_WORLD);
   mumps.SetPrintLevel(0);
   mumps.SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
   mumps.SetOperator(*A);
   Vector X(B.Size());
   mumps.Mult(B,X);
   true_gf = new ParGridFunction(fes);
   error_gf = new ParGridFunction(fes);
   true_gf->SetFromTrueDofs(X);
   std::ostringstream paraview_file_name;
   paraview_file_name << "QPContact-Monitor";
   ParMesh * pmesh = fes->GetParMesh();
   pgf = new ParGridFunction(fes);
   paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), pmesh);
   paraview_dc->SetPrefixPath("ParaView");
   paraview_dc->SetLevelsOfDetail(1);
   paraview_dc->SetDataFormat(VTKFormat::BINARY);
   paraview_dc->SetHighOrderOutput(true);
   paraview_dc->RegisterField("u", pgf);
   paraview_dc->RegisterField("true_u", true_gf);
   paraview_dc->RegisterField("error", error_gf);
   mfem::out << "GeneralSolutionMonitor" << endl;
}

void GeneralSolutionMonitor::MonitorResidual(int it, real_t norm,
                                             const Vector &r, bool final)
{

}

void GeneralSolutionMonitor::MonitorSolution(int it, real_t norm, const Vector &x, bool final)
{
   if (it%output_rate == 0 || final)
   {
      mfem::out << "GeneralSolutionMonitor::Saving iteration: " << it << endl;
      pgf->SetFromTrueDofs(x);
      add(1.0, *true_gf, -1.0, *pgf, *error_gf);
      paraview_dc->SetCycle(counter++);
      paraview_dc->SetTime(double(it));
      paraview_dc->Save();
   }
}