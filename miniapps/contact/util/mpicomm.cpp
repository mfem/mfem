#include "mpicomm.hpp"
#include "util.hpp"


MPICommunicator::MPICommunicator(MPI_Comm comm_, int offset_, int gsize)
: comm(comm_), offset(offset_)
{
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&myid);
   offsets.resize(num_procs);
   MPI_Allgather(&offset,1,MPI_INT,&offsets[0],1,MPI_INT,comm);
   lsize = (myid == num_procs-1) ? gsize - offsets[myid]
                                : offsets[myid+1]-offsets[myid];

   send_count.SetSize(num_procs); send_count = 0;
   send_displ.SetSize(num_procs); send_displ = 0;
   recv_count.SetSize(num_procs); recv_count = 0;
   recv_displ.SetSize(num_procs); recv_displ = 0;
} 

MPICommunicator::MPICommunicator(MPI_Comm comm_, Array<unsigned int> & destination_procs_)
: comm(comm_), destination_procs(destination_procs_)
{
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&myid);   
   send_count.SetSize(num_procs); 
   send_displ.SetSize(num_procs); 
   recv_count.SetSize(num_procs); 
   recv_displ.SetSize(num_procs); 
   resetcounts();
}


int MPICommunicator::get_rank(int dof)
{
   if (num_procs == 1) { return 0; }
   std::vector<int>::iterator up;
   up=std::upper_bound(offsets.begin(), offsets.end(),dof);
   return std::distance(offsets.begin(),up)-1;
}


void MPICommunicator::Communicate(const Vector & x_s, Vector & x_r, int vdim, int ordering)
{
   int npts = x_s.Size()/vdim;
   MFEM_VERIFY(npts == destination_procs.Size(), "Inconsistent number of points to be send");

   // construct send count
   for (int i = 0; i<npts; i++)
   {
      int rank = destination_procs[i];
      send_count[rank] += vdim + 1; // including the sending processor id
   }

   // 2. Compute recv_count
   MPI_Alltoall(&send_count[0],1,MPI_INT,&recv_count[0],1,MPI_INT,comm);

   // 3. Compute displacements
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();

   // 4. Allocate memory and fill in send buffers
   Array<double> sendvals(sbuff_size);  sendvals = 0.0;
   Array<int> sendoffs(num_procs); sendoffs = 0;
   for (int i = 0; i<npts; i++)
   {
      int rank = destination_procs[i];
      int j = send_displ[rank] + sendoffs[rank];
      sendoffs[rank] += vdim+1;
      sendvals[j] = (double)myid;
      for (int k = 0; k<vdim; k++)
      {
         int kk = (ordering == mfem::Ordering::byNODES) ? k*npts+i : i*vdim + k;
         sendvals[j+k+1] = x_s(kk);
      }
   }

   // 5. Communication
   Array<double> recvvals(rbuff_size);

   double * sendvals_ptr = nullptr;
   double * recvvals_ptr = nullptr;
   if (sbuff_size !=0 ) { sendvals_ptr = &sendvals[0]; }   
   if (rbuff_size !=0 ) { recvvals_ptr = &recvvals[0]; }

   MPI_Alltoallv(sendvals_ptr, send_count, send_displ, MPI_DOUBLE, recvvals_ptr,
                 recv_count, recv_displ, MPI_DOUBLE, comm);

   // 6. Unpack
   int n = rbuff_size/(vdim+1);
   origin_procs.SetSize(n);
   x_r.SetSize(vdim*n);
   for (int i = 0; i<n; i++)
   {
      origin_procs[i] = (unsigned int)recvvals[(vdim+1)*i];
      for (int j=0; j<vdim; j++)
      {
         int kk = (ordering == mfem::Ordering::byNODES) ? j*n+i : i*vdim + j;
         x_r(kk) = recvvals[(vdim+1)*i + j+1];
      }
   }
   resetcounts();
}

void MPICommunicator::Communicate(const Array<unsigned int> & x_s, Array<unsigned int> & x_r, int vdim, int ordering)
{
   int npts = x_s.Size()/vdim;
   MFEM_VERIFY(npts == destination_procs.Size(), "Inconsistent number of points to be send");

   // construct send count
   for (int i = 0; i<npts; i++)
   {
      int rank = destination_procs[i];
      send_count[rank] += vdim + 1; // including the sending processor id
   }

   // 2. Compute recv_count
   MPI_Alltoall(&send_count[0],1,MPI_INT,&recv_count[0],1,MPI_INT,comm);

   // 3. Compute displacements
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();

   // 4. Allocate memory and fill in send buffers
   Array<unsigned int> sendvals(sbuff_size);  sendvals = 0.0;
   Array<int> sendoffs(num_procs); sendoffs = 0;
   for (int i = 0; i<npts; i++)
   {
      int rank = destination_procs[i];
      int j = send_displ[rank] + sendoffs[rank];
      sendoffs[rank] += vdim+1;
      sendvals[j] = myid;
      for (int k = 0; k<vdim; k++)
      {
         int kk = (ordering == mfem::Ordering::byNODES) ? k*npts+i : i*vdim + k;
         sendvals[j+k+1] = x_s[kk];
      }
   }

   // 5. Communication
   Array<unsigned int> recvvals(rbuff_size);

   unsigned int * sendvals_ptr = nullptr;
   unsigned int * recvvals_ptr = nullptr;
   if (sbuff_size !=0 ) { sendvals_ptr = &sendvals[0]; }   
   if (rbuff_size !=0 ) { recvvals_ptr = &recvvals[0]; }

   MPI_Alltoallv(sendvals_ptr, send_count, send_displ, MPI_UNSIGNED, recvvals_ptr,
                 recv_count, recv_displ, MPI_UNSIGNED, comm);

   // 6. Unpack
   int n = rbuff_size/(vdim+1);
   origin_procs.SetSize(n);
   x_r.SetSize(vdim*n);
   for (int i = 0; i<n; i++)
   {
      origin_procs[i] = recvvals[(vdim+1)*i];
      for (int j=0; j<vdim; j++)
      {
         int kk = (ordering == mfem::Ordering::byNODES) ? j*n+i : i*vdim + j;
         x_r[kk] = recvvals[(vdim+1)*i + j+1];
      }
   }
   resetcounts();
}

void MPICommunicator::Communicate(const Array<int> & x_s, Array<int> & x_r, int vdim, int ordering)
{
   int npts = x_s.Size()/vdim;
   MFEM_VERIFY(npts == destination_procs.Size(), "Inconsistent number of points to be send");

   // construct send count
   for (int i = 0; i<npts; i++)
   {
      int rank = destination_procs[i];
      send_count[rank] += vdim + 1; // including the sending processor id
   }

   // 2. Compute recv_count
   MPI_Alltoall(&send_count[0],1,MPI_INT,&recv_count[0],1,MPI_INT,comm);

   // 3. Compute displacements
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();

   // 4. Allocate memory and fill in send buffers
   Array<int> sendvals(sbuff_size);  sendvals = 0.0;
   Array<int> sendoffs(num_procs); sendoffs = 0;
   for (int i = 0; i<npts; i++)
   {
      int rank = destination_procs[i];
      int j = send_displ[rank] + sendoffs[rank];
      sendoffs[rank] += vdim+1;
      sendvals[j] = myid;
      for (int k = 0; k<vdim; k++)
      {
         int kk = (ordering == mfem::Ordering::byNODES) ? k*npts+i : i*vdim + k;
         sendvals[j+k+1] = x_s[kk];
      }
   }

   // 5. Communication
   Array<int> recvvals(rbuff_size);

   int * sendvals_ptr = nullptr;
   int * recvvals_ptr = nullptr;
   if (sbuff_size !=0 ) { sendvals_ptr = &sendvals[0]; }   
   if (rbuff_size !=0 ) { recvvals_ptr = &recvvals[0]; }

   MPI_Alltoallv(sendvals_ptr, send_count, send_displ, MPI_INT, recvvals_ptr,
                 recv_count, recv_displ, MPI_INT, comm);

   // 6. Unpack
   int n = rbuff_size/(vdim+1);
   origin_procs.SetSize(n);
   x_r.SetSize(vdim*n);
   for (int i = 0; i<n; i++)
   {
      origin_procs[i] = (unsigned int)recvvals[(vdim+1)*i];
      for (int j=0; j<vdim; j++)
      {
         int kk = (ordering == mfem::Ordering::byNODES) ? j*n+i : i*vdim + j;
         x_r[kk] = recvvals[(vdim+1)*i + j+1];
      }
   }
   resetcounts();
}

void MPICommunicator::Communicate(const DenseMatrix & A_s, DenseMatrix & A_r, int vdim, int ordering)
{
   // matrix width corresponds to dim coordinates
   // matrix rows might include vdim copies
   int npts = A_s.Height()/vdim;
   int dim = A_s.Width();
   MFEM_VERIFY(npts == destination_procs.Size(), "Inconsistent number of points to be send");

   // construct send count
   for (int i = 0; i<npts; i++)
   {
      int rank = destination_procs[i];
      send_count[rank] += dim*vdim + 1; // including the sending processor id
   }

   // 2. Compute recv_count
   MPI_Alltoall(&send_count[0],1,MPI_INT,&recv_count[0],1,MPI_INT,comm);

   // 3. Compute displacements
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();

   // 4. Allocate memory and fill in send buffers
   Array<double> sendvals(sbuff_size);  sendvals = 0.0;
   Array<int> sendoffs(num_procs); sendoffs = 0;
   for (int i = 0; i<npts; i++)
   {
      int rank = destination_procs[i];
      int j = send_displ[rank] + sendoffs[rank];
      sendoffs[rank] += dim*vdim+1;
      sendvals[j] = myid;
      for (int k = 0; k<vdim; k++)
      {
         int kk = (ordering == mfem::Ordering::byNODES) ? k*npts+i : i*vdim + k;
         for (int d=0; d<dim; d++)
         {
            sendvals[j+k*dim+d+1] = A_s(kk,d);
         }
      }
   }
   // 5. Communication
   Array<double> recvvals(rbuff_size);

   double * sendvals_ptr = nullptr;
   double * recvvals_ptr = nullptr;
   if (sbuff_size !=0 ) { sendvals_ptr = &sendvals[0]; }   
   if (rbuff_size !=0 ) { recvvals_ptr = &recvvals[0]; }

   MPI_Alltoallv(sendvals_ptr, send_count, send_displ, MPI_DOUBLE, recvvals_ptr,
                 recv_count, recv_displ, MPI_DOUBLE, comm);

   // 6. Unpack
   int n = rbuff_size/(dim*vdim+1);
   origin_procs.SetSize(n);
   A_r.SetSize(vdim*n,dim);

   for (int i = 0; i<n; i++)
   {
      origin_procs[i] = (unsigned int)recvvals[(dim*vdim+1)*i];
      for (int j=0; j<vdim; j++)
      {
         int kk = (ordering == mfem::Ordering::byNODES) ? j*n+i : i*vdim + j;
         for (int d=0; d<dim; d++)
         {
            A_r(kk,d) = recvvals[(dim*vdim+1)*i + j*dim + d+1];
         }
      }
   }
   resetcounts();

}


void MPICommunicator::Communicate(const SparseMatrix & mat_s , SparseMatrix & mat_r)
{
   // 1. Compute send_count 
   int n = mat_s.NumRows();
   for (int i = 0; i<n; i++)
   {
      int rsize = mat_s.RowSize(i);
      if (rsize == 0) continue;
      int rank = get_rank(i);
      send_count[rank] += rsize+2;
   }
   // 2. Compute recv_count
   MPI_Alltoall(&send_count[0],1,MPI_INT,&recv_count[0],1,MPI_INT,comm);

   // 3. Compute displacements
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();

   // 4. Allocate memory and fill in send buffers
   Array<double> sendvals(sbuff_size);  sendvals = 0.0;
   Array<int> sendcols(sbuff_size);  sendcols = 0;
   Array<int> sendoffs(num_procs); sendoffs = 0;
   Array<int> cols;
   Vector vals;
   for (int i = 0; i<n; i++)
   {
      int rsize = mat_s.RowSize(i);
      if (rsize == 0) continue;
      int rank = get_rank(i);
      int j = send_displ[rank] + sendoffs[rank];
      mat_s.GetRow(i,cols,vals);
      sendoffs[rank] += rsize+2;
      sendvals[j] = (double)i;
      sendvals[j+1] = (double)rsize;
      sendcols[j] = i;
      sendcols[j+1] = rsize;
      for (int l=0; l<rsize ; l++)
      {
         sendvals[j+l+2] = vals[l];
         sendcols[j+l+2] = cols[l];
      }
   }

   // 5. Communication
   Array<double> recvvals(rbuff_size);
   Array<int> recvcols(rbuff_size);

   double * sendvals_ptr = nullptr;
   double * recvvals_ptr = nullptr;
   int * sendcols_ptr = nullptr;
   int * recvcols_ptr = nullptr;
   if (sbuff_size !=0 ) 
   {
      sendvals_ptr = &sendvals[0]; 
      sendcols_ptr = &sendcols[0]; 
   }   
   if (rbuff_size !=0 ) 
   {
      recvvals_ptr = &recvvals[0]; 
      recvcols_ptr = &recvcols[0]; 
   }

   MPI_Alltoallv(sendvals_ptr, send_count, send_displ, MPI_DOUBLE, recvvals_ptr,
                 recv_count, recv_displ, MPI_DOUBLE, comm);

   MPI_Alltoallv(sendcols_ptr, send_count, send_displ, MPI_INT, recvcols_ptr,
                 recv_count, recv_displ, MPI_INT, comm);

   // 6. Unpack and store to the output SparseMatrix
   MFEM_VERIFY(mat_r.Height() == lsize, "Inconsistent row size of output SparseMatrix");
   MFEM_VERIFY(mat_r.Width() == mat_s.Width(), "Inconsistent column size of output SparseMatrix");

   int counter = 0;
   while (counter < rbuff_size)
   {
      int row = recvcols[counter] - offset;
      int size = recvcols[counter+1];
      vals.SetSize(size);
      cols.SetSize(size);
      for (int i = 0; i<size; i++)
      {
         vals[i] = recvvals[counter+2 + i];
         cols[i] = recvcols[counter+2 + i];
      }
      mat_r.AddRow(row,cols,vals);
      counter += size+2; 
   }
   MFEM_VERIFY(counter == rbuff_size, "inconsistent rbuff size");
   mat_r.Finalize();
   mat_r.SortColumnIndices();
   resetcounts();
}

void MPICommunicator::Communicate(const Array<SparseMatrix*> & vmat_s, Array<SparseMatrix*> & vmat_r)
{
   // 1. Compute send_count 
   for (int k = 0; k<vmat_s.Size(); k++)
   {
      if (!vmat_s[k]) continue;
      if (vmat_s[k]->NumNonZeroElems() == 0) continue;
      int nrows = vmat_s[k]->NumRows();
      for (int i = 0; i<nrows; i++)
      {
         int rsize = vmat_s[k]->RowSize(i);
         if (rsize == 0) continue;
         int rank = get_rank(i);
         send_count[rank] += rsize+3;
      }
   } 

   // 2. Compute recv_count
   MPI_Alltoall(&send_count[0],1,MPI_INT,&recv_count[0],1,MPI_INT,comm);
   
   // 3. Compute displacements
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();

   // 4. Allocate memory and fill in send buffers
   Array<double> sendvals(sbuff_size);  sendvals = 0.0;
   Array<int> sendcols(sbuff_size);  sendcols = 0;
   Array<int> sendoffs(num_procs); sendoffs = 0;
   for (int k = 0; k<vmat_s.Size(); k++)
   {
      if (!vmat_s[k]) continue;
      if (vmat_s[k]->NumNonZeroElems() == 0) continue;
      int nrows = vmat_s[k]->NumRows();
      for (int i = 0; i<nrows; i++)
      {
         int rsize = vmat_s[k]->RowSize(i);
         if (rsize == 0) continue;
         int rank = get_rank(i);
         int j = send_displ[rank] + sendoffs[rank];
         Array<int> cols;
         Vector vals;
         vmat_s[k]->GetRow(i,cols,vals);
         sendoffs[rank] += rsize+3;
         sendvals[j] = (double)k;
         sendvals[j+1] = (double)i;
         sendvals[j+2] = (double)rsize;
         sendcols[j] = k;
         sendcols[j+1] = i;
         sendcols[j+2] = rsize;
         for (int l=0; l<rsize ; l++)
         {
            sendvals[j+l+3] = vals[l];
            sendcols[j+l+3] = cols[l];
         }
      }
   }
   // 5. Communication
   Array<double> recvvals(rbuff_size);
   Array<int> recvcols(rbuff_size);
   double * sendvals_ptr = nullptr;
   double * recvvals_ptr = nullptr;
   int * sendcols_ptr = nullptr;
   int * recvcols_ptr = nullptr;
   if (sbuff_size !=0 ) 
   {
      sendvals_ptr = &sendvals[0]; 
      sendcols_ptr = &sendcols[0]; 
   }   
   if (rbuff_size !=0 ) 
   {
      recvvals_ptr = &recvvals[0]; 
      recvcols_ptr = &recvcols[0]; 
   }

   MPI_Alltoallv(sendvals_ptr, send_count, send_displ, MPI_DOUBLE, recvvals_ptr,
                 recv_count, recv_displ, MPI_DOUBLE,comm);

   MPI_Alltoallv(sendcols_ptr, send_count, send_displ, MPI_INT, recvcols_ptr,
                 recv_count, recv_displ, MPI_INT,comm);

   // 6. Unpack and store to the output SparseMatrix
   int counter = 0;
   while (counter < rbuff_size)
   {
      int npt = recvcols[counter];
      int row = recvcols[counter+1] - offset;
      int size = recvcols[counter+2];
      Vector vals(size);
      Array<int> cols(size);
      for (int i = 0; i<size; i++)
      {
         vals[i] = recvvals[counter+3 + i];
         cols[i] = recvcols[counter+3 + i];
      }
      vmat_r[npt]->AddRow(row,cols,vals);
      counter += size+3; 
   }
   MFEM_VERIFY(counter == rbuff_size, "inconsistent size");

   for (int i = 0; i<vmat_r.Size(); i++)
   {
      vmat_r[i]->Finalize();
      vmat_r[i]->SortColumnIndices();
   }
   resetcounts();
}