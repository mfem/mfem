// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>
#ifdef __bgq__
#include <mpix.h>
#endif

#include "array.hpp"
#include "table.hpp"
#include "sets.hpp"
#include "communication.hpp"
#include <iostream>
#include <map>

using namespace std;

namespace mfem
{

GroupTopology::GroupTopology(const GroupTopology &gt)
   : MyComm(gt.MyComm),
     group_lproc(gt.group_lproc)
{
   gt.groupmaster_lproc.Copy(groupmaster_lproc);
   gt.lproc_proc.Copy(lproc_proc);
   gt.group_mgroup.Copy(group_mgroup);
}

void GroupTopology::ProcToLProc()
{
   int NRanks;
   MPI_Comm_size(MyComm, &NRanks);

   map<int, int> proc_lproc;

   int lproc_counter = 0;
   for (int i = 0; i < group_lproc.Size_of_connections(); i++)
   {
      const pair<const int, int> p(group_lproc.GetJ()[i], lproc_counter);
      if (proc_lproc.insert(p).second)
      {
         lproc_counter++;
      }
   }
   // Note: group_lproc.GetJ()[0] == MyRank --> proc_lproc[MyRank] == 0

   lproc_proc.SetSize(lproc_counter);
   for (map<int, int>::iterator it = proc_lproc.begin();
        it != proc_lproc.end(); ++it)
   {
      lproc_proc[it->second] = it->first;
   }

   for (int i = 0; i < group_lproc.Size_of_connections(); i++)
   {
      group_lproc.GetJ()[i] = proc_lproc[group_lproc.GetJ()[i]];
   }

   for (int i = 0; i < NGroups(); i++)
   {
      groupmaster_lproc[i] = proc_lproc[groupmaster_lproc[i]];
   }
}

void GroupTopology::Create(ListOfIntegerSets &groups, int mpitag)
{
   groups.AsTable(group_lproc); // group_lproc = group_proc

   Table group_mgroupandproc;
   group_mgroupandproc.SetDims(NGroups(),
                               group_lproc.Size_of_connections() + NGroups());
   for (int i = 0; i < NGroups(); i++)
   {
      int j = group_mgroupandproc.GetI()[i];
      group_mgroupandproc.GetI()[i+1] = j + group_lproc.RowSize(i) + 1;
      group_mgroupandproc.GetJ()[j] = i;
      j++;
      for (int k = group_lproc.GetI()[i];
           j < group_mgroupandproc.GetI()[i+1]; j++, k++)
      {
         group_mgroupandproc.GetJ()[j] = group_lproc.GetJ()[k];
      }
   }

   // build groupmaster_lproc with lproc = proc
   groupmaster_lproc.SetSize(NGroups());

   // simplest choice of the group owner
   for (int i = 0; i < NGroups(); i++)
   {
      groupmaster_lproc[i] = groups.PickElementInSet(i);
   }

   // load-balanced choice of the group owner, which however can lead to
   // isolated dofs
   // for (i = 0; i < NGroups(); i++)
   //    groupmaster_lproc[i] = groups.PickRandomElementInSet(i);

   ProcToLProc();

   // build group_mgroup
   group_mgroup.SetSize(NGroups());

   int send_counter = 0;
   int recv_counter = 0;
   for (int i = 1; i < NGroups(); i++)
      if (groupmaster_lproc[i] != 0) // we are not the master
      {
         recv_counter++;
      }
      else
      {
         send_counter += group_lproc.RowSize(i)-1;
      }

   MPI_Request *requests = new MPI_Request[send_counter];
   MPI_Status  *statuses = new MPI_Status[send_counter];

   int max_recv_size = 0;
   send_counter = 0;
   for (int i = 1; i < NGroups(); i++)
   {
      if (groupmaster_lproc[i] == 0) // we are the master
      {
         group_mgroup[i] = i;

         for (int j = group_lproc.GetI()[i];
              j < group_lproc.GetI()[i+1]; j++)
         {
            if (group_lproc.GetJ()[j] != 0)
            {
               MPI_Isend(group_mgroupandproc.GetRow(i),
                         group_mgroupandproc.RowSize(i),
                         MPI_INT,
                         lproc_proc[group_lproc.GetJ()[j]],
                         mpitag,
                         MyComm,
                         &requests[send_counter]);
               send_counter++;
            }
         }
      }
      else // we are not the master
         if (max_recv_size < group_lproc.RowSize(i))
         {
            max_recv_size = group_lproc.RowSize(i);
         }
   }
   max_recv_size++;

   IntegerSet group;
   if (recv_counter > 0)
   {
      int count;
      MPI_Status status;
      int *recv_buf = new int[max_recv_size];
      for ( ; recv_counter > 0; recv_counter--)
      {
         MPI_Recv(recv_buf, max_recv_size, MPI_INT,
                  MPI_ANY_SOURCE, mpitag, MyComm, &status);

         MPI_Get_count(&status, MPI_INT, &count);

         group.Recreate(count-1, recv_buf+1);
         int g = groups.Lookup(group);
         group_mgroup[g] = recv_buf[0];

         if (lproc_proc[groupmaster_lproc[g]] != status.MPI_SOURCE)
         {
            cerr << "\n\n\nGroupTopology::GroupTopology: "
                 << MyRank() << ": ERROR\n\n\n" << endl;
            mfem_error();
         }
      }
      delete [] recv_buf;
   }

   MPI_Waitall(send_counter, requests, statuses);

   delete [] statuses;
   delete [] requests;
}


// specializations of the MPITypeMap static member
template<> const MPI_Datatype MPITypeMap<int>::mpi_type = MPI_INT;
template<> const MPI_Datatype MPITypeMap<double>::mpi_type = MPI_DOUBLE;


GroupCommunicator::GroupCommunicator(GroupTopology &gt)
   : gtopo(gt)
{
   group_buf_size = 0;
   requests = NULL;
   statuses = NULL;
}

void GroupCommunicator::Create(Array<int> &ldof_group)
{
   group_ldof.MakeI(gtopo.NGroups());
   for (int i = 0; i < ldof_group.Size(); i++)
   {
      int group = ldof_group[i];
      if (group != 0)
      {
         group_ldof.AddAColumnInRow(group);
      }
   }
   group_ldof.MakeJ();

   for (int i = 0; i < ldof_group.Size(); i++)
   {
      int group = ldof_group[i];
      if (group != 0)
      {
         group_ldof.AddConnection(group, i);
      }
   }
   group_ldof.ShiftUpI();

   Finalize();
}

void GroupCommunicator::Finalize()
{
   int request_counter = 0;

   for (int gr = 1; gr < group_ldof.Size(); gr++)
      if (group_ldof.RowSize(gr) != 0)
      {
         int gr_requests;
         if (!gtopo.IAmMaster(gr)) // we are not the master
         {
            gr_requests = 1;
         }
         else
         {
            gr_requests = gtopo.GetGroupSize(gr)-1;
         }

         request_counter += gr_requests;
         group_buf_size += gr_requests * group_ldof.RowSize(gr);
      }

   requests = new MPI_Request[request_counter];
   statuses = new MPI_Status[request_counter];
}

template <class T>
void GroupCommunicator::Bcast(T *ldata)
{
   if (group_buf_size == 0)
   {
      return;
   }

   group_buf.SetSize(group_buf_size*sizeof(T));
   T *buf = (T *)group_buf.GetData();

   int i, gr, request_counter = 0;

   for (gr = 1; gr < group_ldof.Size(); gr++)
   {
      const int nldofs = group_ldof.RowSize(gr);

      // ignore groups without dofs
      if (nldofs == 0)
      {
         continue;
      }

      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         MPI_Irecv(buf,
                   nldofs,
                   MPITypeMap<T>::mpi_type,
                   gtopo.GetGroupMasterRank(gr),
                   40822 + gtopo.GetGroupMasterGroup(gr),
                   gtopo.GetComm(),
                   &requests[request_counter]);
         request_counter++;
      }
      else // we are the master
      {
         // fill send buffer
         const int *ldofs = group_ldof.GetRow(gr);
         for (i = 0; i < nldofs; i++)
         {
            buf[i] = ldata[ldofs[i]];
         }

         const int  gs  = gtopo.GetGroupSize(gr);
         const int *nbs = gtopo.GetGroup(gr);
         for (i = 0; i < gs; i++)
         {
            if (nbs[i] != 0)
            {
               MPI_Isend(buf,
                         nldofs,
                         MPITypeMap<T>::mpi_type,
                         gtopo.GetNeighborRank(nbs[i]),
                         40822 + gtopo.GetGroupMasterGroup(gr),
                         gtopo.GetComm(),
                         &requests[request_counter]);
               request_counter++;
            }
         }
      }
      buf += nldofs;
   }

   MPI_Waitall(request_counter, requests, statuses);

   // copy the received data from the buffer to ldata
   buf = (T *)group_buf.GetData();
   for (gr = 1; gr < group_ldof.Size(); gr++)
   {
      const int nldofs = group_ldof.RowSize(gr);

      // ignore groups without dofs
      if (nldofs == 0)
      {
         continue;
      }

      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         const int *ldofs = group_ldof.GetRow(gr);
         for (i = 0; i < nldofs; i++)
         {
            ldata[ldofs[i]] = buf[i];
         }
      }
      buf += nldofs;
   }
}

template <class T>
void GroupCommunicator::Reduce(T *ldata, void (*Op)(OpData<T>))
{
   if (group_buf_size == 0)
   {
      return;
   }

   int i, gr, request_counter = 0;
   OpData<T> opd;

   group_buf.SetSize(group_buf_size*sizeof(T));
   opd.ldata = ldata;
   opd.buf = (T *)group_buf.GetData();
   for (gr = 1; gr < group_ldof.Size(); gr++)
   {
      opd.nldofs = group_ldof.RowSize(gr);

      // ignore groups without dofs
      if (opd.nldofs == 0)
      {
         continue;
      }

      opd.ldofs = group_ldof.GetRow(gr);

      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         for (i = 0; i < opd.nldofs; i++)
         {
            opd.buf[i] = ldata[opd.ldofs[i]];
         }

         MPI_Isend(opd.buf,
                   opd.nldofs,
                   MPITypeMap<T>::mpi_type,
                   gtopo.GetGroupMasterRank(gr),
                   43822 + gtopo.GetGroupMasterGroup(gr),
                   gtopo.GetComm(),
                   &requests[request_counter]);
         request_counter++;
         opd.buf += opd.nldofs;
      }
      else // we are the master
      {
         const int  gs  = gtopo.GetGroupSize(gr);
         const int *nbs = gtopo.GetGroup(gr);
         for (i = 0; i < gs; i++)
         {
            if (nbs[i] != 0)
            {
               MPI_Irecv(opd.buf,
                         opd.nldofs,
                         MPITypeMap<T>::mpi_type,
                         gtopo.GetNeighborRank(nbs[i]),
                         43822 + gtopo.GetGroupMasterGroup(gr),
                         gtopo.GetComm(),
                         &requests[request_counter]);
               request_counter++;
               opd.buf += opd.nldofs;
            }
         }
      }
   }

   MPI_Waitall(request_counter, requests, statuses);

   // perform the reduce operation
   opd.buf = (T *)group_buf.GetData();
   for (gr = 1; gr < group_ldof.Size(); gr++)
   {
      opd.nldofs = group_ldof.RowSize(gr);

      // ignore groups without dofs
      if (opd.nldofs == 0)
      {
         continue;
      }

      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         opd.buf += opd.nldofs;
      }
      else // we are the master
      {
         opd.ldofs = group_ldof.GetRow(gr);
         opd.nb = gtopo.GetGroupSize(gr)-1;
         Op(opd);
         opd.buf += opd.nb * opd.nldofs;
      }
   }
}

template <class T>
void GroupCommunicator::Sum(OpData<T> opd)
{
   for (int i = 0; i < opd.nldofs; i++)
   {
      T data = opd.ldata[opd.ldofs[i]];
      for (int j = 0; j < opd.nb; j++)
      {
         data += opd.buf[j*opd.nldofs+i];
      }
      opd.ldata[opd.ldofs[i]] = data;
   }
}

template <class T>
void GroupCommunicator::Min(OpData<T> opd)
{
   for (int i = 0; i < opd.nldofs; i++)
   {
      T data = opd.ldata[opd.ldofs[i]];
      for (int j = 0; j < opd.nb; j++)
      {
         T b = opd.buf[j*opd.nldofs+i];
         if (data > b)
         {
            data = b;
         }
      }
      opd.ldata[opd.ldofs[i]] = data;
   }
}

template <class T>
void GroupCommunicator::Max(OpData<T> opd)
{
   for (int i = 0; i < opd.nldofs; i++)
   {
      T data = opd.ldata[opd.ldofs[i]];
      for (int j = 0; j < opd.nb; j++)
      {
         T b = opd.buf[j*opd.nldofs+i];
         if (data < b)
         {
            data = b;
         }
      }
      opd.ldata[opd.ldofs[i]] = data;
   }
}

template <class T>
void GroupCommunicator::BitOR(OpData<T> opd)
{
   for (int i = 0; i < opd.nldofs; i++)
   {
      T data = opd.ldata[opd.ldofs[i]];
      for (int j = 0; j < opd.nb; j++)
      {
         data |= opd.buf[j*opd.nldofs+i];
      }
      opd.ldata[opd.ldofs[i]] = data;
   }
}

GroupCommunicator::~GroupCommunicator()
{
   delete [] statuses;
   delete [] requests;
}

// @cond DOXYGEN_SKIP

// instantiate GroupCommunicator::Bcast and Reduce for int and double
template void GroupCommunicator::Bcast<int>(int *);
template void GroupCommunicator::Reduce<int>(int *, void (*)(OpData<int>));

template void GroupCommunicator::Bcast<double>(double *);
template void GroupCommunicator::Reduce<double>(
   double *, void (*)(OpData<double>));

// @endcond

// instantiate reduce operators for int and double
template void GroupCommunicator::Sum<int>(OpData<int>);
template void GroupCommunicator::Min<int>(OpData<int>);
template void GroupCommunicator::Max<int>(OpData<int>);
template void GroupCommunicator::BitOR<int>(OpData<int>);

template void GroupCommunicator::Sum<double>(OpData<double>);
template void GroupCommunicator::Min<double>(OpData<double>);
template void GroupCommunicator::Max<double>(OpData<double>);


#ifdef __bgq__
static void DebugRankCoords(int** coords, int dim, int size)
{
   for (int i = 0; i < size; i++)
   {
      std::cout << "Rank " << i << " coords: ";
      for (int j = 0; j < dim; j++)
      {
         std::cout << coords[i][j] << " ";
      }
      std::cout << endl;
   }
}

struct CompareCoords
{
   CompareCoords(int coord) : coord(coord) {}
   int coord;

   bool operator()(int* const &a, int* const &b) const
   { return a[coord] < b[coord]; }
};

void KdTreeSort(int** coords, int d, int dim, int size)
{
   if (size > 1)
   {
      bool all_same = true;
      for (int i = 1; i < size && all_same; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            if (coords[i][j] != coords[0][j]) { all_same = false; break; }
         }
      }
      if (all_same) { return; }

      // sort by coordinate 'd'
      std::sort(coords, coords + size, CompareCoords(d));
      int next = (d + 1) % dim;

      if (coords[0][d] < coords[size-1][d])
      {
         KdTreeSort(coords, next, dim, size/2);
         KdTreeSort(coords + size/2, next, dim, size - size/2);
      }
      else
      {
         // skip constant dimension
         KdTreeSort(coords, next, dim, size);
      }
   }
}

MPI_Comm ReorderRanksZCurve(MPI_Comm comm)
{
   MPI_Status status;

   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   int dim;
   MPIX_Torus_ndims(&dim);

   int* mycoords = new int[dim + 1];
   MPIX_Rank2torus(rank, mycoords);

   MPI_Send(mycoords, dim, MPI_INT, 0, 111, comm);
   delete [] mycoords;

   if (rank == 0)
   {
      int** coords = new int*[size];
      for (int i = 0; i < size; i++)
      {
         coords[i] = new int[dim + 1];
         coords[i][dim] = i;
         MPI_Recv(coords[i], dim, MPI_INT, i, 111, comm, &status);
      }

      KdTreeSort(coords, 0, dim, size);

      //DebugRankCoords(coords, dim, size);

      for (int i = 0; i < size; i++)
      {
         MPI_Send(&coords[i][dim], 1, MPI_INT, i, 112, comm);
         delete [] coords[i];
      }
      delete [] coords;
   }

   int new_rank;
   MPI_Recv(&new_rank, 1, MPI_INT, 0, 112, comm, &status);

   MPI_Comm new_comm;
   MPI_Comm_split(comm, 0, new_rank, &new_comm);
   return new_comm;
}

#else // __bgq__

MPI_Comm ReorderRanksZCurve(MPI_Comm comm)
{
   // pass
   return comm;
}
#endif // __bgq__


} // namespace mfem

#endif
