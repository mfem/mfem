// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifdef MFEM_USE_MPI

#include <mpi.h>

#include "array.hpp"
#include "table.hpp"
#include "sets.hpp"
#include "communication.hpp"

void GroupTopology::ProcToLProc()
{
   int NRanks;
   MPI_Comm_size(MyComm, &NRanks);

   Array<int> proc_lproc(NRanks); // array of size number of processors!
   proc_lproc = -1;

   int lproc_counter = 0;
   for (int i = 0; i < group_lproc.Size_of_connections(); i++)
      if (proc_lproc[group_lproc.GetJ()[i]] < 0)
         proc_lproc[group_lproc.GetJ()[i]] = lproc_counter++;
   // Note: group_lproc.GetJ()[0] == MyRank --> proc_lproc[MyRank] == 0

   lproc_proc.SetSize(lproc_counter);
   for (int i = 0; i < NRanks; i++)
      if (proc_lproc[i] >= 0)
         lproc_proc[proc_lproc[i]] = i;

   for (int i = 0; i < group_lproc.Size_of_connections(); i++)
      group_lproc.GetJ()[i] = proc_lproc[group_lproc.GetJ()[i]];

   for (int i = 0; i < NGroups(); i++)
      groupmaster_lproc[i] = proc_lproc[groupmaster_lproc[i]];
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
         group_mgroupandproc.GetJ()[j] = group_lproc.GetJ()[k];
   }

   // build groupmaster_lproc with lproc = proc
   groupmaster_lproc.SetSize(NGroups());

   // simplest choice of the group owner
   for (int i = 0; i < NGroups(); i++)
      groupmaster_lproc[i] = groups.PickElementInSet(i);

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
         recv_counter++;
      else
         send_counter += group_lproc.RowSize(i)-1;

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
            max_recv_size = group_lproc.RowSize(i);
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


GroupCommunicator::GroupCommunicator(GroupTopology &gt)
   : gtopo(gt)
{
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
         group_ldof.AddAColumnInRow(group);
   }
   group_ldof.MakeJ();

   for (int i = 0; i < ldof_group.Size(); i++)
   {
      int group = ldof_group[i];
      if (group != 0)
         group_ldof.AddConnection(group, i);
   }
   group_ldof.ShiftUpI();

   Finalize();
}

void GroupCommunicator::Finalize()
{
   int request_counter = 0;
   int reduce_buf_size = 0;

   for (int gr = 1; gr < group_ldof.Size(); gr++)
      if (group_ldof.RowSize(gr) != 0)
      {
         int gr_requests;
         if (!gtopo.IAmMaster(gr)) // we are not the master
            gr_requests = 1;
         else
            gr_requests = gtopo.GetGroupSize(gr)-1;

         request_counter += gr_requests;
         reduce_buf_size += gr_requests * group_ldof.RowSize(gr);
      }

   requests = new MPI_Request[request_counter];
   statuses = new MPI_Status[request_counter];

   group_buf.SetSize(reduce_buf_size);
}

void GroupCommunicator::Bcast(Array<int> &ldata)
{
   if (group_buf.Size() == 0)
      return;

   int i, gr, request_counter = 0;

   for (gr = 1; gr < group_ldof.Size(); gr++)
   {
      // ignore groups without dofs
      if (group_ldof.RowSize(gr) == 0)
         continue;

      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         MPI_Irecv(&group_buf[group_ldof.GetI()[gr]],
                   group_ldof.RowSize(gr),
                   MPI_INT,
                   gtopo.GetGroupMasterRank(gr),
                   40822 + gtopo.GetGroupMasterGroup(gr),
                   gtopo.GetComm(),
                   &requests[request_counter]);
         request_counter++;
      }
      else // we are the master
      {
         // fill send buffer
         for (i = group_ldof.GetI()[gr]; i < group_ldof.GetI()[gr+1]; i++)
            group_buf[i] = ldata[group_ldof.GetJ()[i]];

         const int  gs  = gtopo.GetGroupSize(gr);
         const int *nbs = gtopo.GetGroup(gr);
         for (i = 0; i < gs; i++)
         {
            if (nbs[i] != 0)
            {
               MPI_Isend(&group_buf[group_ldof.GetI()[gr]],
                         group_ldof.RowSize(gr),
                         MPI_INT,
                         gtopo.GetNeighborRank(nbs[i]),
                         40822 + gtopo.GetGroupMasterGroup(gr),
                         gtopo.GetComm(),
                         &requests[request_counter]);
               request_counter++;
            }
         }
      }
   }

   MPI_Waitall(request_counter, requests, statuses);

   // copy the received data from the buffer to ldata
   for (gr = 1; gr < group_ldof.Size(); gr++)
      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         for (i = group_ldof.GetI()[gr]; i < group_ldof.GetI()[gr+1]; i++)
            ldata[group_ldof.GetJ()[i]] = group_buf[i];
      }
}

void GroupCommunicator::Reduce(Array<int> &ldata)
{
   if (group_buf.Size() == 0)
      return;

   int i, gr, request_counter = 0, buf_offset = 0;

   for (gr = 1; gr < group_ldof.Size(); gr++)
   {
      // ignore groups without dofs
      if (group_ldof.RowSize(gr) == 0)
         continue;

      const int *ldofs = group_ldof.GetRow(gr);
      const int nldofs = group_ldof.RowSize(gr);

      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         for (i = 0; i < nldofs; i++)
            group_buf[buf_offset+i] = ldata[ldofs[i]];

         MPI_Isend(&group_buf[buf_offset],
                   nldofs,
                   MPI_INT,
                   gtopo.GetGroupMasterRank(gr),
                   43822 + gtopo.GetGroupMasterGroup(gr),
                   gtopo.GetComm(),
                   &requests[request_counter]);
         request_counter++;
         buf_offset += nldofs;
      }
      else // we are the master
      {
         const int  gs  = gtopo.GetGroupSize(gr);
         const int *nbs = gtopo.GetGroup(gr);
         for (i = 0; i < gs; i++)
         {
            if (nbs[i] != 0)
            {
               MPI_Irecv(&group_buf[buf_offset],
                         nldofs,
                         MPI_INT,
                         gtopo.GetNeighborRank(nbs[i]),
                         43822 + gtopo.GetGroupMasterGroup(gr),
                         gtopo.GetComm(),
                         &requests[request_counter]);
               request_counter++;
               buf_offset += nldofs;
            }
         }
      }
   }

   MPI_Waitall(request_counter, requests, statuses);

   // perform the reduce operation
   buf_offset = 0;
   for (gr = 1; gr < group_ldof.Size(); gr++)
   {
      // ignore groups without dofs
      if (group_ldof.RowSize(gr) == 0)
         continue;

      const int nldofs = group_ldof.RowSize(gr);

      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         buf_offset += nldofs;
      }
      else // we are the master
      {
         const int *ldofs = group_ldof.GetRow(gr);
         const int nb = gtopo.GetGroupSize(gr)-1;
         for (i = 0; i < nldofs; i++)
         {
            int data = ldata[ldofs[i]];
            for (int j = 0; j < nb; j++)
               data |= group_buf[buf_offset+j*nldofs+i];
            ldata[ldofs[i]] = data;
         }
         buf_offset += nb * nldofs;
      }
   }
}

GroupCommunicator::~GroupCommunicator()
{
   delete [] statuses;
   delete [] requests;
}

#endif
