//
#include "utils.hpp"

std::string GetTimestamp()
{
   std::time_t now = std::time(nullptr);
   std::tm* local_time = std::localtime(&now);
   std::ostringstream oss;
   oss << (local_time->tm_mon + 1) << "-"
       << local_time->tm_mday << "-"
       << (local_time->tm_year + 1900) << "_"
       << local_time->tm_hour << ":"
       << local_time->tm_min << ":"
       << local_time->tm_sec;
   return oss.str();
}

// Write parsed options to a file in the ParaView directory
void WriteParametersToFile(const mfem::OptionsParser& args,
                           const std::string& output_dir)
{
   // Ensure directory exists
   std::string mkdir_command = "mkdir -p " + output_dir;
   int ret = system(mkdir_command.c_str());
   if (ret != 0)
   {
      std::cerr << "Warning: Failed to create ParaView output directory.\n";
   }

   std::string filename = output_dir + "/run_parameters.txt";

   std::ofstream param_file(filename);
   if (param_file.is_open())
   {
      param_file << "Simulation Parameters \n";
      param_file << "------------------------------------\n";

      // Use OptionsParser's Print method to output parameters to the file
      args.PrintOptions(param_file);

      param_file.close();
      std::cout << "Parameters saved to " << filename << "\n";
   }
   else
   {
      std::cerr << "Error: Unable to open file to save parameters.\n";
   }
}

void CreateParaViewPath(const char* mesh_file, std::string& output_dir)
{
   std::string timestamp = GetTimestamp();
   std::string paraview_file = timestamp;

   output_dir = output_dir + paraview_file;
}

std::string GetFilename(const std::string& filePath)
{
   // Step 1: Find the last '/' or '\' to isolate the filename
   size_t lastSlash = filePath.find_last_of("/\\");
   std::string filename = (lastSlash == std::string::npos) ? filePath :
                          filePath.substr(lastSlash + 1);

   // Step 2: Find the last '.' to remove the extension
   size_t lastDot = filename.find_last_of('.');
   return (lastDot == std::string::npos) ? filename : filename.substr(0, lastDot);
}



// Compute offsets for True DOFs across all processors
void ElementTdofs::ComputeTdofOffsets()
{
   tdof_offsets.resize(num_procs);
   int mytoffset = pfes->GetMyTDofOffset();
   MPI_Allgather(&mytoffset, 1, MPI_INT, tdof_offsets.data(), 1, MPI_INT, comm);
}

// Determine which rank owns a given True DOF (tdof)
int ElementTdofs::GetRank(int tdof)
{
   auto up = std::upper_bound(tdof_offsets.begin(), tdof_offsets.end(), tdof);
   return std::distance(tdof_offsets.begin(), up) - 1;
}

// Distribute indices using MPI_Alltoallv
void ElementTdofs::DistributeIndices(Array<int>& indices,
                                     Array<int>& processors,
                                     Array<int>& recv_idx)
{
   // Step 1: Prepare data to send
   std::vector<int> send_counts(num_procs, 0);
   std::map<int, std::vector<int>> send_buffers;

   for (int i = 0; i < indices.Size(); ++i)
   {
      send_buffers[processors[i]].push_back(indices[i]);
   }

   std::vector<int> send_displs(num_procs, 0);
   std::vector<int> send_data;

   for (int i = 0; i < num_procs; ++i)
   {
      send_counts[i] = send_buffers[i].size();
      send_data.insert(send_data.end(), send_buffers[i].begin(),
                       send_buffers[i].end());
   }

   for (int i = 1; i < num_procs; ++i)
   {
      send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
   }

   // Step 2: Gather receive counts
   std::vector<int> recv_counts(num_procs, 0);
   MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                comm);

   // Step 3: Compute receive displacements
   std::vector<int> recv_displs(num_procs, 0);
   int total_recv_size = 0;

   for (int i = 0; i < num_procs; ++i)
   {
      recv_displs[i] = total_recv_size;
      total_recv_size += recv_counts[i];
   }

   // Step 4: Allocate buffer and perform MPI_Alltoallv
   recv_idx.SetSize(total_recv_size);
   MPI_Alltoallv(send_data.data(), send_counts.data(), send_displs.data(), MPI_INT,
                 recv_idx.GetData(), recv_counts.data(), recv_displs.data(), MPI_INT, comm);
}


ElementTdofs::ElementTdofs(const ParFiniteElementSpace* pfes_)
   : pfes(pfes_), comm(pfes_->GetComm())
{
   MPI_Comm_size(comm, &num_procs);
   ComputeTdofOffsets();
}

// Extract and distribute True DOFs based on the element attribute
Array<int> ElementTdofs::GetTrueDOFs(int element_attribute)
{
   Array<int> mydof_list, other_dof_list;
   Array<int> myltdof, other_ltdof;

   if (boundary)
   {
      for (int i = 0; i < pfes->GetNBE(); i++)
      {
         if (pfes->GetParMesh()->GetBdrAttribute(i) == element_attribute)
         {
            int el,info;
            pfes->GetParMesh()->GetBdrElementAdjacentElement(i,el,info);
            Array<int> dofs;
            pfes->GetElementVDofs(el, dofs);

            for (int j = 0; j < dofs.Size(); j++)
            {
               int decoded_dof = ParFiniteElementSpace::DecodeDof(dofs[j]);
               int ltdof = pfes->GetLocalTDofNumber(decoded_dof);
               if (ltdof < 0)
               {
                  continue; // Skip if this is not a true DOF
               }
               int tdof = pfes->GetGlobalTDofNumber(decoded_dof);
               if (Mpi::WorldRank() == GetRank(tdof))
               {
                  mydof_list.Append(tdof);
                  myltdof.Append(ltdof);
               }
               else
               {
                  other_dof_list.Append(tdof);
                  other_ltdof.Append(ltdof);
               }
            }
         }
      }
   }
   else
   {
      for (int i = 0; i < pfes->GetNE(); i++)
      {
         if (pfes->GetParMesh()->GetAttribute(i) == element_attribute)
         {
            Array<int> dofs;
            pfes->GetElementVDofs(i, dofs);

            for (int j = 0; j < dofs.Size(); j++)
            {
               int decoded_dof = ParFiniteElementSpace::DecodeDof(dofs[j]);
               int ltdof = pfes->GetLocalTDofNumber(decoded_dof);
               if (ltdof < 0)
               {
                  continue; // Skip if this is not a true DOF
               }
               int tdof = pfes->GetGlobalTDofNumber(decoded_dof);
               if (Mpi::WorldRank() == GetRank(tdof))
               {
                  mydof_list.Append(tdof);
                  myltdof.Append(ltdof);
               }
               else
               {
                  other_dof_list.Append(tdof);
                  other_ltdof.Append(ltdof);
               }
            }
         }
      }
   }

   mydof_list.Sort();
   mydof_list.Unique();
   other_dof_list.Sort();
   other_dof_list.Unique();

   myltdof.Sort();
   myltdof.Unique();
   other_ltdof.Sort();
   other_ltdof.Unique();

   // Assign processors for other DOFs
   Array<int> other_dof_proc_list(other_dof_list.Size());
   for (int i = 0; i < other_dof_list.Size(); i++)
   {
      other_dof_proc_list[i] = GetRank(other_dof_list[i]);
   }

   // Communicate DOFs to appropriate processors
   Array<int> recv_idx;
   DistributeIndices(other_dof_list, other_dof_proc_list, recv_idx);

   mydof_list.Append(recv_idx);
   mydof_list.Sort();
   mydof_list.Unique();

   return mydof_list;
}

HypreParMatrix * ElementTdofs::GetProlongationMatrix(int element_attribute)
{
   Array<int> tdofs = GetTrueDOFs(element_attribute);
   int h = tdofs.Size();
   SparseMatrix St(h,pfes->GlobalTrueVSize());

   for (int i = 0; i<h; i++)
   {
      int col = tdofs[i];
      St.Set(i,col,1.0);
   }
   St.Finalize();
   int rows[2];
   int cols[2];
   int nrows = St.Height();

   int row_offset;
   MPI_Scan(&nrows,&row_offset,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

   row_offset-=nrows;
   rows[0] = row_offset;
   rows[1] = row_offset+nrows;
   for (int i = 0; i < 2; i++)
   {
      cols[i] = pfes->GetTrueDofOffsets()[i];
   }
   int glob_nrows;
   int glob_ncols = pfes->GlobalTrueVSize();
   MPI_Allreduce(&nrows, &glob_nrows,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

   HypreParMatrix * Pt = new HypreParMatrix(MPI_COMM_WORLD, nrows, glob_nrows,
                                            glob_ncols, St.GetI(), St.GetJ(),
                                            St.GetData(), rows,cols);
   HypreParMatrix * P = Pt->Transpose();
   return P;
}


SolverWithFiltering::SolverWithFiltering(MPI_Comm comm_) : Solver()
{
   Init(comm_);
}

void SolverWithFiltering::Init(MPI_Comm comm_)
{
   comm=comm_;
   MPI_Comm_size(comm, &numProcs);
   MPI_Comm_rank(comm, &myid);
}

void SolverWithFiltering::SetOperator(const Operator & Op_)
{
   Op = &Op_;
   height = Op->Height();
   width = Op->Width();
}

void SolverWithFiltering::SetSubspaceProlongationMap(const Operator & P_)
{
   P = &P_;
   MFEM_VERIFY(P->Height() == Op->Height(),
               "Prolongation operator height does not match the operator.");
}

void SolverWithFiltering::SetSolver(const Solver * solver_)
{
   solver = solver_;
   MFEM_VERIFY(solver->Height() == Op->Height(),
               "Solver height does not match the operator.");
}

void SolverWithFiltering::SetFilterSolver(const Solver * filter_solver_)
{
   filter_solver = filter_solver_;
   MFEM_VERIFY(filter_solver->Height() == P->Width(),
               "Filter solver height does not match the subspace dimension.");
}

void SolverWithFiltering::Mult(const Vector & b, Vector & x) const
{
   MFEM_VERIFY(b.Size() == x.Size(), "Inconsistent x and y size");
   x = 0.0;
   Vector z(x);
   solver->Mult(b, z);
   // 1. Full space correction
   x+=z;
   Vector rf(P->Width());
   Vector xf(P->Width());
   Vector r(b.Size());
   // 2. Compute Residual r = b - A x
   Op->Mult(x,r);
   r.Neg(); r+=b;
   // 3. Restrict to subspace
   P->MultTranspose(r,rf);
   // 4. Solve on the subspace
   filter_solver->Mult(rf,xf);
   // 5. Transfer to fine space
   P->Mult(xf,z);
   // 6. Update Correction
   x+=z;
   // 7. Compute Residual r = b - A x
   Op->Mult(x,r);
   r.Neg(); r+=b;
   solver->Mult(r, z);
   // 8. Full space correction
   x+= z;
}