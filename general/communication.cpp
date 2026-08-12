// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

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
#include "text.hpp"
#include "sort_pairs.hpp"
#include "globals.hpp"

#ifdef MFEM_USE_STRUMPACK
#include <StrumpackConfig.hpp> // STRUMPACK_USE_PTSCOTCH, etc.
#endif

#include <iostream>
#include <map>

using namespace std;

namespace mfem
{

#if defined(MFEM_USE_STRUMPACK) && \
    (defined(STRUMPACK_USE_PTSCOTCH) || defined(STRUMPACK_USE_SLATE_SCALAPACK))
int Mpi::default_thread_required = MPI_THREAD_MULTIPLE;
#else
int Mpi::default_thread_required = MPI_THREAD_SINGLE;
#endif


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

   // The local processor ids are assigned following the group order and within
   // a group following their ordering in the group. In other words, the ids are
   // assigned based on their order in the J array of group_lproc.
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

   // Build 'group_mgroup':

   // Use aggregated neighbor communication: at most one send to and/or one
   // receive from each neighbor.

   group_mgroup.SetSize(NGroups());
   MFEM_DEBUG_DO(group_mgroup = -1);
   for (int g = 0; g < NGroups(); g++)
   {
      if (IAmMaster(g)) { group_mgroup[g] = g; }
   }

   // The Table 'lproc_cgroup': for each lproc, list the groups that are owned
   // by this rank or by that lproc.
   Table lproc_cgroup;
   {
      Array<Connection> lproc_cgroup_list;
      for (int g = 1; g < NGroups(); g++)
      {
         if (IAmMaster(g))
         {
            const int gs = GetGroupSize(g);
            const int *lprocs = GetGroup(g);
            for (int i = 0; i < gs; i++)
            {
               if (lprocs[i])
               {
                  lproc_cgroup_list.Append(Connection(lprocs[i],g));
               }
            }
         }
         else
         {
            lproc_cgroup_list.Append(Connection(GetGroupMaster(g),g));
         }
      }
      lproc_cgroup_list.Sort();
      lproc_cgroup_list.Unique();
      lproc_cgroup.MakeFromList(GetNumNeighbors(), lproc_cgroup_list);
   }

   // Determine size of the send-receive buffer. For each neighbor the buffer
   // contains: <send-part><receive-part> with each part consisting of a list of
   // groups. Each group, g, has group_lproc.RowSize(g)+2 integers: the first
   // entry is group_lproc.RowSize(g) - the number of processors in the group,
   // followed by the group-id in the master processor, followed by the ranks of
   // the processors in the group.
   Table buffer;
   buffer.MakeI(2*lproc_cgroup.Size()-2); // excluding the "local" lproc, 0
   for (int nbr = 1; nbr < lproc_cgroup.Size(); nbr++)
   {
      const int send_row = 2*(nbr-1);
      const int recv_row = send_row+1;
      const int ng = lproc_cgroup.RowSize(nbr);
      const int *g = lproc_cgroup.GetRow(nbr);
      for (int j = 0; j < ng; j++)
      {
         const int gs = group_lproc.RowSize(g[j]);
         if (IAmMaster(g[j]))
         {
            buffer.AddColumnsInRow(send_row, gs+2);
         }
         else
         {
            MFEM_ASSERT(GetGroupMaster(g[j]) == nbr, "internal error");
            buffer.AddColumnsInRow(recv_row, gs+2);
         }
      }
   }
   buffer.MakeJ();
   for (int nbr = 1; nbr < lproc_cgroup.Size(); nbr++)
   {
      const int send_row = 2*(nbr-1);
      const int recv_row = send_row+1;
      const int ng = lproc_cgroup.RowSize(nbr);
      const int *g = lproc_cgroup.GetRow(nbr);
      for (int j = 0; j < ng; j++)
      {
         const int gs = group_lproc.RowSize(g[j]);
         if (IAmMaster(g[j]))
         {
            buffer.AddConnection(send_row, gs);
            buffer.AddConnections(
               send_row, group_mgroupandproc.GetRow(g[j]), gs+1);
         }
         else
         {
            buffer.AddColumnsInRow(recv_row, gs+2);
         }
      }
   }
   buffer.ShiftUpI();
   Array<MPI_Request> send_requests(lproc_cgroup.Size()-1);
   Array<MPI_Request> recv_requests(lproc_cgroup.Size()-1);
   send_requests = MPI_REQUEST_NULL;
   recv_requests = MPI_REQUEST_NULL;
   for (int nbr = 1; nbr < lproc_cgroup.Size(); nbr++)
   {
      const int send_row = 2*(nbr-1);
      const int recv_row = send_row+1;
      const int send_size = buffer.RowSize(send_row);
      const int recv_size = buffer.RowSize(recv_row);
      if (send_size > 0)
      {
         MPI_Isend(buffer.GetRow(send_row), send_size, MPI_INT, lproc_proc[nbr],
                   mpitag, MyComm, &send_requests[nbr-1]);
      }
      if (recv_size > 0)
      {
         MPI_Irecv(buffer.GetRow(recv_row), recv_size, MPI_INT, lproc_proc[nbr],
                   mpitag, MyComm, &recv_requests[nbr-1]);
      }
   }

   if (recv_requests.Size() > 0)
   {
      int idx;
      IntegerSet group;
      while (MPI_Waitany(recv_requests.Size(), recv_requests.GetData(), &idx,
                         MPI_STATUS_IGNORE),
             idx != MPI_UNDEFINED)
      {
         const int recv_size = buffer.RowSize(2*idx+1);
         const int *recv_buf = buffer.GetRow(2*idx+1);
         for (int s = 0;  s < recv_size; s += recv_buf[s]+2)
         {
            group.Recreate(recv_buf[s], recv_buf+s+2);
            const int g = groups.Lookup(group);
            MFEM_ASSERT(group_mgroup[g] == -1, "communication error");
            group_mgroup[g] = recv_buf[s+1];
         }
      }
   }

   MPI_Waitall(send_requests.Size(), send_requests.GetData(),
               MPI_STATUSES_IGNORE);

   // debug barrier: MPI_Barrier(MyComm);
}

void GroupTopology::Save(ostream &os) const
{
   os << "\ncommunication_groups\n";
   os << "number_of_groups " << NGroups() << "\n\n";

   os << "# number of entities in each group, followed by ranks in group\n";
   for (int group_id = 0; group_id < NGroups(); ++group_id)
   {
      int group_size = GetGroupSize(group_id);
      const int * group_ptr = GetGroup(group_id);
      os << group_size;
      for ( int group_member_index = 0; group_member_index < group_size;
            ++group_member_index)
      {
         os << " " << GetNeighborRank( group_ptr[group_member_index] );
      }
      os << "\n";
   }

   // For future use, optional ownership strategy.
   // os << "# ownership";
}

void GroupTopology::Load(istream &in)
{
   // Load in group topology and create list of integer sets.  Use constructor
   // that uses list of integer sets.
   std::string ident;

   // Read in number of groups
   int number_of_groups = -1;
   in >> ident;
   MFEM_VERIFY(ident == "number_of_groups",
               "GroupTopology::Load - expected 'number_of_groups' entry.");
   in >> number_of_groups;

   // Skip number of entries in each group comment.
   skip_comment_lines(in, '#');

   ListOfIntegerSets integer_sets;
   for (int group_id = 0; group_id < number_of_groups; ++group_id)
   {
      IntegerSet integer_set;
      Array<int>& array = integer_set;
      int group_size;
      in >> group_size;
      array.Reserve(group_size);
      for ( int index = 0; index < group_size; ++index )
      {
         int value;
         in >> value;
         array.Append(value);
      }
      integer_sets.Insert(integer_set);
   }

   Create(integer_sets, 823);
}

void GroupTopology::Copy(GroupTopology& copy) const
{
   copy.SetComm(MyComm);
   group_lproc.Copy(copy.group_lproc);
   groupmaster_lproc.Copy(copy.groupmaster_lproc);
   lproc_proc.Copy(copy.lproc_proc);
   group_mgroup.Copy(copy.group_mgroup);
}

void GroupTopology::Swap(GroupTopology &other)
{
   mfem::Swap(MyComm, other.MyComm);
   mfem::Swap(group_lproc, other.group_lproc);
   mfem::Swap(groupmaster_lproc, other.groupmaster_lproc);
   mfem::Swap(lproc_proc, other.lproc_proc);
   mfem::Swap(group_mgroup, other.group_mgroup);
}

/// \cond DO_NOT_DOCUMENT
// Initialize the static mpi_type for the specializations of MPITypeMap:
const MPI_Datatype MPITypeMap<bool>::mpi_type = MFEM_MPI_CXX_BOOL;
const MPI_Datatype MPITypeMap<char>::mpi_type = MPI_CHAR;
const MPI_Datatype MPITypeMap<unsigned char>::mpi_type = MPI_UNSIGNED_CHAR;
const MPI_Datatype MPITypeMap<short>::mpi_type = MPI_SHORT;
const MPI_Datatype MPITypeMap<unsigned short>::mpi_type = MPI_UNSIGNED_SHORT;
const MPI_Datatype MPITypeMap<int>::mpi_type = MPI_INT;
const MPI_Datatype MPITypeMap<unsigned int>::mpi_type = MPI_UNSIGNED;
const MPI_Datatype MPITypeMap<long>::mpi_type = MPI_LONG;
const MPI_Datatype MPITypeMap<unsigned long>::mpi_type = MPI_UNSIGNED_LONG;
const MPI_Datatype MPITypeMap<long long>::mpi_type = MPI_LONG_LONG;
const MPI_Datatype MPITypeMap<unsigned long long>::mpi_type =
   MPI_UNSIGNED_LONG_LONG;
const MPI_Datatype MPITypeMap<float>::mpi_type = MPI_FLOAT;
const MPI_Datatype MPITypeMap<double>::mpi_type = MPI_DOUBLE;
/// \endcond DO_NOT_DOCUMENT

GroupCommunicator::GroupCommunicator(const GroupTopology &gt, Mode m)
   : gtopo(gt), mode(m)
{
   group_buf_size = 0;
   requests = NULL;
   // statuses = NULL;
   comm_lock = 0;
   num_requests = 0;
   request_marker = NULL;
   buf_offsets = NULL;
}

void GroupCommunicator::Create(const Array<int> &ldof_group)
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

   // size buf_offsets = max(number of groups, number of neighbors)
   buf_offsets = new int[max(group_ldof.Size(), gtopo.GetNumNeighbors())];
   buf_offsets[0] = 0;
   for (int gr = 1; gr < group_ldof.Size(); gr++)
   {
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
   }

   requests = new MPI_Request[request_counter];
   // statuses = new MPI_Status[request_counter];
   request_marker = new int[request_counter];

   // Construct nbr_send_groups and nbr_recv_groups: (nbr 0 = me)
   nbr_send_groups.MakeI(gtopo.GetNumNeighbors());
   nbr_recv_groups.MakeI(gtopo.GetNumNeighbors());
   for (int gr = 1; gr < group_ldof.Size(); gr++)
   {
      const int nldofs = group_ldof.RowSize(gr);
      if (nldofs == 0) { continue; }

      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         nbr_recv_groups.AddAColumnInRow(gtopo.GetGroupMaster(gr));
      }
      else // we are the master
      {
         const int grp_size = gtopo.GetGroupSize(gr);
         const int *grp_nbr_list = gtopo.GetGroup(gr);
         for (int i = 0; i < grp_size; i++)
         {
            if (grp_nbr_list[i] != 0)
            {
               nbr_send_groups.AddAColumnInRow(grp_nbr_list[i]);
            }
         }
      }
   }
   nbr_send_groups.MakeJ();
   nbr_recv_groups.MakeJ();
   for (int gr = 1; gr < group_ldof.Size(); gr++)
   {
      const int nldofs = group_ldof.RowSize(gr);
      if (nldofs == 0) { continue; }

      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         nbr_recv_groups.AddConnection(gtopo.GetGroupMaster(gr), gr);
      }
      else // we are the master
      {
         const int grp_size = gtopo.GetGroupSize(gr);
         const int *grp_nbr_list = gtopo.GetGroup(gr);
         for (int i = 0; i < grp_size; i++)
         {
            if (grp_nbr_list[i] != 0)
            {
               nbr_send_groups.AddConnection(grp_nbr_list[i], gr);
            }
         }
      }
   }
   nbr_send_groups.ShiftUpI();
   nbr_recv_groups.ShiftUpI();
   // The above construction creates the Tables with the column indices
   // sorted, i.e. the group lists are sorted. To coordinate this order between
   // processors, we will sort the group lists in the nbr_recv_groups Table
   // according to their indices in the master. This does not require any
   // communication because we have access to the group indices in the master
   // by calling: master_group_id = gtopo.GetGroupMasterGroup(my_group_id).
   Array<Pair<int,int> > group_ids;
   for (int nbr = 1; nbr < nbr_recv_groups.Size(); nbr++)
   {
      const int num_recv_groups = nbr_recv_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
         int *grp_list = nbr_recv_groups.GetRow(nbr);
         group_ids.SetSize(num_recv_groups);
         for (int i = 0; i < num_recv_groups; i++)
         {
            group_ids[i].one = gtopo.GetGroupMasterGroup(grp_list[i]);
            group_ids[i].two = grp_list[i]; // my_group_id
         }
         group_ids.Sort();
         for (int i = 0; i < num_recv_groups; i++)
         {
            grp_list[i] = group_ids[i].two;
         }
      }
   }
}

void GroupCommunicator::SetLTDofTable(const Array<int> &ldof_ltdof)
{
   if (group_ltdof.Size() == group_ldof.Size()) { return; }

   group_ltdof.MakeI(group_ldof.Size());
   for (int gr = 1; gr < group_ldof.Size(); gr++)
   {
      if (gtopo.IAmMaster(gr))
      {
         group_ltdof.AddColumnsInRow(gr, group_ldof.RowSize(gr));
      }
   }
   group_ltdof.MakeJ();
   for (int gr = 1; gr < group_ldof.Size(); gr++)
   {
      if (gtopo.IAmMaster(gr))
      {
         const int *ldofs = group_ldof.GetRow(gr);
         const int nldofs = group_ldof.RowSize(gr);
         for (int i = 0; i < nldofs; i++)
         {
            group_ltdof.AddConnection(gr, ldof_ltdof[ldofs[i]]);
         }
      }
   }
   group_ltdof.ShiftUpI();
}

void GroupCommunicator::GetNeighborLTDofTable(Table &nbr_ltdof) const
{
   nbr_ltdof.MakeI(nbr_send_groups.Size());
   for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
   {
      const int num_send_groups = nbr_send_groups.RowSize(nbr);
      if (num_send_groups > 0)
      {
         const int *grp_list = nbr_send_groups.GetRow(nbr);
         for (int i = 0; i < num_send_groups; i++)
         {
            const int group = grp_list[i];
            const int nltdofs = group_ltdof.RowSize(group);
            nbr_ltdof.AddColumnsInRow(nbr, nltdofs);
         }
      }
   }
   nbr_ltdof.MakeJ();
   for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
   {
      const int num_send_groups = nbr_send_groups.RowSize(nbr);
      if (num_send_groups > 0)
      {
         const int *grp_list = nbr_send_groups.GetRow(nbr);
         for (int i = 0; i < num_send_groups; i++)
         {
            const int group = grp_list[i];
            const int nltdofs = group_ltdof.RowSize(group);
            const int *ltdofs = group_ltdof.GetRow(group);
            nbr_ltdof.AddConnections(nbr, ltdofs, nltdofs);
         }
      }
   }
   nbr_ltdof.ShiftUpI();
}

void GroupCommunicator::GetNeighborLDofTable(Table &nbr_ldof) const
{
   nbr_ldof.MakeI(nbr_recv_groups.Size());
   for (int nbr = 1; nbr < nbr_recv_groups.Size(); nbr++)
   {
      const int num_recv_groups = nbr_recv_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
         const int *grp_list = nbr_recv_groups.GetRow(nbr);
         for (int i = 0; i < num_recv_groups; i++)
         {
            const int group = grp_list[i];
            const int nldofs = group_ldof.RowSize(group);
            nbr_ldof.AddColumnsInRow(nbr, nldofs);
         }
      }
   }
   nbr_ldof.MakeJ();
   for (int nbr = 1; nbr < nbr_recv_groups.Size(); nbr++)
   {
      const int num_recv_groups = nbr_recv_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
         const int *grp_list = nbr_recv_groups.GetRow(nbr);
         for (int i = 0; i < num_recv_groups; i++)
         {
            const int group = grp_list[i];
            const int nldofs = group_ldof.RowSize(group);
            const int *ldofs = group_ldof.GetRow(group);
            nbr_ldof.AddConnections(nbr, ldofs, nldofs);
         }
      }
   }
   nbr_ldof.ShiftUpI();
}

template <class T>
T *GroupCommunicator::CopyGroupToBuffer(const T *ldata, T *buf, int group,
                                        int layout) const
{
   switch (layout)
   {
      case 1:
      {
         return std::copy(ldata + group_ldof.GetI()[group],
                          ldata + group_ldof.GetI()[group+1],
                          buf);
      }
      case 2:
      {
         const int nltdofs = group_ltdof.RowSize(group);
         const int *ltdofs = group_ltdof.GetRow(group);
         for (int j = 0; j < nltdofs; j++)
         {
            buf[j] = ldata[ltdofs[j]];
         }
         return buf + nltdofs;
      }
      default:
      {
         const int nldofs = group_ldof.RowSize(group);
         const int *ldofs = group_ldof.GetRow(group);
         for (int j = 0; j < nldofs; j++)
         {
            buf[j] = ldata[ldofs[j]];
         }
         return buf + nldofs;
      }
   }
}

template <class T>
const T *GroupCommunicator::CopyGroupFromBuffer(const T *buf, T *ldata,
                                                int group, int layout) const
{
   const int nldofs = group_ldof.RowSize(group);
   switch (layout)
   {
      case 1:
      {
         std::copy(buf, buf + nldofs, ldata + group_ldof.GetI()[group]);
         break;
      }
      case 2:
      {
         const int *ltdofs = group_ltdof.GetRow(group);
         for (int j = 0; j < nldofs; j++)
         {
            ldata[ltdofs[j]] = buf[j];
         }
         break;
      }
      default:
      {
         const int *ldofs = group_ldof.GetRow(group);
         for (int j = 0; j < nldofs; j++)
         {
            ldata[ldofs[j]] = buf[j];
         }
         break;
      }
   }
   return buf + nldofs;
}

template <class T>
const T *GroupCommunicator::ReduceGroupFromBuffer(const T *buf, T *ldata,
                                                  int group, int layout,
                                                  void (*Op)(OpData<T>)) const
{
   OpData<T> opd;
   opd.ldata = ldata;
   opd.nldofs = group_ldof.RowSize(group);
   opd.nb = 1;
   opd.buf = const_cast<T*>(buf);

   switch (layout)
   {
      case 1:
      {
         MFEM_ABORT("layout 1 is not supported");
         T *dest = ldata + group_ldof.GetI()[group];
         for (int j = 0; j < opd.nldofs; j++)
         {
            dest[j] += buf[j];
         }
         break;
      }
      case 2:
      {
         opd.ldofs = const_cast<int*>(group_ltdof.GetRow(group));
         Op(opd);
         break;
      }
      default:
      {
         opd.ldofs = const_cast<int*>(group_ldof.GetRow(group));
         Op(opd);
         break;
      }
   }
   return buf + opd.nldofs;
}

template <class T>
void GroupCommunicator::BcastBegin(T *ldata, int layout) const
{
   MFEM_VERIFY(comm_lock == 0, "object is already in use");

   if (group_buf_size == 0) { return; }

   int request_counter = 0;
   switch (mode)
   {
      case byGroup: // ***** Communication by groups *****
      {
         T *buf;
         if (layout != 1)
         {
            group_buf.SetSize(group_buf_size*sizeof(T));
            buf = (T *)group_buf.GetData();
            MFEM_VERIFY(layout != 2 || group_ltdof.Size() == group_ldof.Size(),
                        "'group_ltdof' is not set, use SetLTDofTable()");
         }
         else
         {
            buf = ldata;
         }

         for (int gr = 1; gr < group_ldof.Size(); gr++)
         {
            const int nldofs = group_ldof.RowSize(gr);

            // ignore groups without dofs
            if (nldofs == 0) { continue; }

            if (!gtopo.IAmMaster(gr)) // we are not the master
            {
               MPI_Irecv(buf,
                         nldofs,
                         MPITypeMap<T>::mpi_type,
                         gtopo.GetGroupMasterRank(gr),
                         40822 + gtopo.GetGroupMasterGroup(gr),
                         gtopo.GetComm(),
                         &requests[request_counter]);
               request_marker[request_counter] = gr;
               request_counter++;
            }
            else // we are the master
            {
               if (layout != 1)
               {
                  CopyGroupToBuffer(ldata, buf, gr, layout);
               }
               const int  gs  = gtopo.GetGroupSize(gr);
               const int *nbs = gtopo.GetGroup(gr);
               for (int i = 0; i < gs; i++)
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
                     request_marker[request_counter] = -1; // mark as send req.
                     request_counter++;
                  }
               }
            }
            buf += nldofs;
         }
         break;
      }

      case byNeighbor: // ***** Communication by neighbors *****
      {
         group_buf.SetSize(group_buf_size*sizeof(T));
         T *buf = (T *)group_buf.GetData();
         for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
         {
            const int num_send_groups = nbr_send_groups.RowSize(nbr);
            if (num_send_groups > 0)
            {
               // Possible optimization:
               //    if (num_send_groups == 1) and (layout == 1) then we do not
               //    need to copy the data in order to send it.
               T *buf_start = buf;
               const int *grp_list = nbr_send_groups.GetRow(nbr);
               for (int i = 0; i < num_send_groups; i++)
               {
                  buf = CopyGroupToBuffer(ldata, buf, grp_list[i], layout);
               }
               MPI_Isend(buf_start,
                         buf - buf_start,
                         MPITypeMap<T>::mpi_type,
                         gtopo.GetNeighborRank(nbr),
                         40822,
                         gtopo.GetComm(),
                         &requests[request_counter]);
               request_marker[request_counter] = -1; // mark as send request
               request_counter++;
            }

            const int num_recv_groups = nbr_recv_groups.RowSize(nbr);
            if (num_recv_groups > 0)
            {
               // Possible optimization (requires interface change):
               //    if (num_recv_groups == 1) and the (output layout == 1) then
               //    we can receive directly in the output buffer; however, at
               //    this point we do not have that information.
               const int *grp_list = nbr_recv_groups.GetRow(nbr);
               int recv_size = 0;
               for (int i = 0; i < num_recv_groups; i++)
               {
                  recv_size += group_ldof.RowSize(grp_list[i]);
               }
               MPI_Irecv(buf,
                         recv_size,
                         MPITypeMap<T>::mpi_type,
                         gtopo.GetNeighborRank(nbr),
                         40822,
                         gtopo.GetComm(),
                         &requests[request_counter]);
               request_marker[request_counter] = nbr;
               request_counter++;
               buf_offsets[nbr] = buf - (T*)group_buf.GetData();
               buf += recv_size;
            }
         }
         MFEM_ASSERT(buf - (T*)group_buf.GetData() == group_buf_size, "");
         break;
      }
   }

   comm_lock = 1; // 1 - locked for Bcast
   num_requests = request_counter;
}

template <class T>
void GroupCommunicator::BcastEnd(T *ldata, int layout) const
{
   if (comm_lock == 0) { return; }
   // The above also handles the case (group_buf_size == 0).
   MFEM_VERIFY(comm_lock == 1, "object is NOT locked for Bcast");

   switch (mode)
   {
      case byGroup: // ***** Communication by groups *****
      {
         if (layout == 1)
         {
            MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
         }
         else if (layout == 0)
         {
            // copy the received data from the buffer to ldata, as it arrives
            int idx;
            while (MPI_Waitany(num_requests, requests, &idx, MPI_STATUS_IGNORE),
                   idx != MPI_UNDEFINED)
            {
               int gr = request_marker[idx];
               if (gr == -1) { continue; } // skip send requests

               // groups without dofs are skipped, so here nldofs > 0.
               T *buf = (T *)group_buf.GetData() + group_ldof.GetI()[gr];
               CopyGroupFromBuffer(buf, ldata, gr, layout);
            }
         }
         break;
      }

      case byNeighbor: // ***** Communication by neighbors *****
      {
         // copy the received data from the buffer to ldata, as it arrives
         int idx;
         while (MPI_Waitany(num_requests, requests, &idx, MPI_STATUS_IGNORE),
                idx != MPI_UNDEFINED)
         {
            int nbr = request_marker[idx];
            if (nbr == -1) { continue; } // skip send requests

            const int num_recv_groups = nbr_recv_groups.RowSize(nbr);
            if (num_recv_groups > 0)
            {
               const int *grp_list = nbr_recv_groups.GetRow(nbr);
               const T *buf = (T*)group_buf.GetData() + buf_offsets[nbr];
               for (int i = 0; i < num_recv_groups; i++)
               {
                  buf = CopyGroupFromBuffer(buf, ldata, grp_list[i], layout);
               }
            }
         }
         break;
      }
   }

   comm_lock = 0; // 0 - no lock
   num_requests = 0;
}

template <class T>
void GroupCommunicator::ReduceBegin(const T *ldata) const
{
   MFEM_VERIFY(comm_lock == 0, "object is already in use");

   if (group_buf_size == 0) { return; }

   int request_counter = 0;
   group_buf.SetSize(group_buf_size*sizeof(T));
   T *buf = (T *)group_buf.GetData();
   switch (mode)
   {
      case byGroup: // ***** Communication by groups *****
      {
         for (int gr = 1; gr < group_ldof.Size(); gr++)
         {
            const int nldofs = group_ldof.RowSize(gr);
            // ignore groups without dofs
            if (nldofs == 0) { continue; }

            if (!gtopo.IAmMaster(gr)) // we are not the master
            {
               const int layout = 0;
               CopyGroupToBuffer(ldata, buf, gr, layout);
               MPI_Isend(buf,
                         nldofs,
                         MPITypeMap<T>::mpi_type,
                         gtopo.GetGroupMasterRank(gr),
                         43822 + gtopo.GetGroupMasterGroup(gr),
                         gtopo.GetComm(),
                         &requests[request_counter]);
               request_marker[request_counter] = -1; // mark as send request
               request_counter++;
               buf += nldofs;
            }
            else // we are the master
            {
               const int  gs  = gtopo.GetGroupSize(gr);
               const int *nbs = gtopo.GetGroup(gr);
               buf_offsets[gr] = buf - (T *)group_buf.GetData();
               for (int i = 0; i < gs; i++)
               {
                  if (nbs[i] != 0)
                  {
                     MPI_Irecv(buf,
                               nldofs,
                               MPITypeMap<T>::mpi_type,
                               gtopo.GetNeighborRank(nbs[i]),
                               43822 + gtopo.GetGroupMasterGroup(gr),
                               gtopo.GetComm(),
                               &requests[request_counter]);
                     request_marker[request_counter] = gr;
                     request_counter++;
                     buf += nldofs;
                  }
               }
            }
         }
         break;
      }

      case byNeighbor: // ***** Communication by neighbors *****
      {
         for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
         {
            // In Reduce operation: send_groups <--> recv_groups
            const int num_send_groups = nbr_recv_groups.RowSize(nbr);
            if (num_send_groups > 0)
            {
               T *buf_start = buf;
               const int *grp_list = nbr_recv_groups.GetRow(nbr);
               for (int i = 0; i < num_send_groups; i++)
               {
                  const int layout = 0; // ldata is an array on all ldofs
                  buf = CopyGroupToBuffer(ldata, buf, grp_list[i], layout);
               }
               MPI_Isend(buf_start,
                         buf - buf_start,
                         MPITypeMap<T>::mpi_type,
                         gtopo.GetNeighborRank(nbr),
                         43822,
                         gtopo.GetComm(),
                         &requests[request_counter]);
               request_marker[request_counter] = -1; // mark as send request
               request_counter++;
            }

            // In Reduce operation: send_groups <--> recv_groups
            const int num_recv_groups = nbr_send_groups.RowSize(nbr);
            if (num_recv_groups > 0)
            {
               const int *grp_list = nbr_send_groups.GetRow(nbr);
               int recv_size = 0;
               for (int i = 0; i < num_recv_groups; i++)
               {
                  recv_size += group_ldof.RowSize(grp_list[i]);
               }
               MPI_Irecv(buf,
                         recv_size,
                         MPITypeMap<T>::mpi_type,
                         gtopo.GetNeighborRank(nbr),
                         43822,
                         gtopo.GetComm(),
                         &requests[request_counter]);
               request_marker[request_counter] = nbr;
               request_counter++;
               buf_offsets[nbr] = buf - (T*)group_buf.GetData();
               buf += recv_size;
            }
         }
         MFEM_ASSERT(buf - (T*)group_buf.GetData() == group_buf_size, "");
         break;
      }
   }

   comm_lock = 2;
   num_requests = request_counter;
}

template <class T>
void GroupCommunicator::ReduceEnd(T *ldata, int layout,
                                  void (*Op)(OpData<T>)) const
{
   if (comm_lock == 0) { return; }
   // The above also handles the case (group_buf_size == 0).
   MFEM_VERIFY(comm_lock == 2, "object is NOT locked for Reduce");

   switch (mode)
   {
      case byGroup: // ***** Communication by groups *****
      {
         OpData<T> opd;
         opd.ldata = ldata;
         Array<int> group_num_req(group_ldof.Size());
         for (int gr = 1; gr < group_ldof.Size(); gr++)
         {
            group_num_req[gr] =
               gtopo.IAmMaster(gr) ? gtopo.GetGroupSize(gr)-1 : 0;
         }
         int idx;
         while (MPI_Waitany(num_requests, requests, &idx, MPI_STATUS_IGNORE),
                idx != MPI_UNDEFINED)
         {
            int gr = request_marker[idx];
            if (gr == -1) { continue; } // skip send requests

            // Delay the processing of a group until all receive requests, for
            // that group, are done:
            if ((--group_num_req[gr]) != 0) { continue; }

            opd.nldofs = group_ldof.RowSize(gr);
            // groups without dofs are skipped, so here nldofs > 0.

            opd.buf = (T *)group_buf.GetData() + buf_offsets[gr];
            opd.ldofs = (layout == 0) ?
                        group_ldof.GetRow(gr) : group_ltdof.GetRow(gr);
            opd.nb = gtopo.GetGroupSize(gr)-1;
            Op(opd);
         }
         break;
      }

      case byNeighbor: // ***** Communication by neighbors *****
      {
         MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);

         for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
         {
            // In Reduce operation: send_groups <--> recv_groups
            const int num_recv_groups = nbr_send_groups.RowSize(nbr);
            if (num_recv_groups > 0)
            {
               const int *grp_list = nbr_send_groups.GetRow(nbr);
               const T *buf = (T*)group_buf.GetData() + buf_offsets[nbr];
               for (int i = 0; i < num_recv_groups; i++)
               {
                  buf = ReduceGroupFromBuffer(buf, ldata, grp_list[i],
                                              layout, Op);
               }
            }
         }
         break;
      }
   }

   comm_lock = 0; // 0 - no lock
   num_requests = 0;
}

template <class T>
void GroupCommunicator::Sum(OpData<T> opd)
{
   if (opd.nb == 1)
   {
      for (int i = 0; i < opd.nldofs; i++)
      {
         opd.ldata[opd.ldofs[i]] += opd.buf[i];
      }
   }
   else
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

void GroupCommunicator::PrintInfo(std::ostream &os) const
{
   char c = '\0';
   const int tag = 46800;
   const int myid = gtopo.MyRank();

   int num_sends = 0, num_recvs = 0;
   size_t mem_sends = 0, mem_recvs = 0;
   int num_master_groups = 0, num_empty_groups = 0;
   int num_active_neighbors = 0; // for mode == byNeighbor
   switch (mode)
   {
      case byGroup:
         for (int gr = 1; gr < group_ldof.Size(); gr++)
         {
            const int nldofs = group_ldof.RowSize(gr);
            if (nldofs == 0)
            {
               num_empty_groups++;
               continue;
            }
            if (gtopo.IAmMaster(gr))
            {
               num_sends += (gtopo.GetGroupSize(gr)-1);
               mem_sends += sizeof(double)*nldofs*(gtopo.GetGroupSize(gr)-1);
               num_master_groups++;
            }
            else
            {
               num_recvs++;
               mem_recvs += sizeof(double)*nldofs;
            }
         }
         break;

      case byNeighbor:
         for (int gr = 1; gr < group_ldof.Size(); gr++)
         {
            const int nldofs = group_ldof.RowSize(gr);
            if (nldofs == 0)
            {
               num_empty_groups++;
               continue;
            }
            if (gtopo.IAmMaster(gr))
            {
               num_master_groups++;
            }
         }
         for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
         {
            const int num_send_groups = nbr_send_groups.RowSize(nbr);
            if (num_send_groups > 0)
            {
               const int *grp_list = nbr_send_groups.GetRow(nbr);
               for (int i = 0; i < num_send_groups; i++)
               {
                  mem_sends += sizeof(double)*group_ldof.RowSize(grp_list[i]);
               }
               num_sends++;
            }

            const int num_recv_groups = nbr_recv_groups.RowSize(nbr);
            if (num_recv_groups > 0)
            {
               const int *grp_list = nbr_recv_groups.GetRow(nbr);
               for (int i = 0; i < num_recv_groups; i++)
               {
                  mem_recvs += sizeof(double)*group_ldof.RowSize(grp_list[i]);
               }
               num_recvs++;
            }
            if (num_send_groups > 0 || num_recv_groups > 0)
            {
               num_active_neighbors++;
            }
         }
         break;
   }
   if (myid != 0)
   {
      MPI_Recv(&c, 1, MPI_CHAR, myid-1, tag, gtopo.GetComm(),
               MPI_STATUS_IGNORE);
   }
   else
   {
      os << "\nGroupCommunicator:\n";
   }
   os << "Rank " << myid << ":\n"
      "   mode             = " <<
      (mode == byGroup ? "byGroup" : "byNeighbor") << "\n"
      "   number of sends  = " << num_sends <<
      " (" << mem_sends << " bytes)\n"
      "   number of recvs  = " << num_recvs <<
      " (" << mem_recvs << " bytes)\n";
   os <<
      "   num groups       = " << group_ldof.Size() << " = " <<
      num_master_groups << " + " <<
      group_ldof.Size()-num_master_groups-num_empty_groups << " + " <<
      num_empty_groups << " (master + slave + empty)\n";
   if (mode == byNeighbor)
   {
      os <<
         "   num neighbors    = " << nbr_send_groups.Size() << " = " <<
         num_active_neighbors << " + " <<
         nbr_send_groups.Size()-num_active_neighbors <<
         " (active + inactive)\n";
   }
   if (myid != gtopo.NRanks()-1)
   {
      os << std::flush;
      MPI_Send(&c, 1, MPI_CHAR, myid+1, tag, gtopo.GetComm());
   }
   else
   {
      os << std::endl;
   }
   MPI_Barrier(gtopo.GetComm());
}

GroupCommunicator::~GroupCommunicator()
{
   delete [] buf_offsets;
   delete [] request_marker;
   // delete [] statuses;
   delete [] requests;
}

// @cond DOXYGEN_SKIP

// instantiate GroupCommunicator::Bcast and Reduce for int and double
template void GroupCommunicator::BcastBegin<int>(int *, int) const;
template void GroupCommunicator::BcastEnd<int>(int *, int) const;
template void GroupCommunicator::ReduceBegin<int>(const int *) const;
template void GroupCommunicator::ReduceEnd<int>(
   int *, int, void (*)(OpData<int>)) const;

template void GroupCommunicator::BcastBegin<double>(double *, int) const;
template void GroupCommunicator::BcastEnd<double>(double *, int) const;
template void GroupCommunicator::ReduceBegin<double>(const double *) const;
template void GroupCommunicator::ReduceEnd<double>(
   double *, int, void (*)(OpData<double>)) const;

template void GroupCommunicator::BcastBegin<float>(float *, int) const;
template void GroupCommunicator::BcastEnd<float>(float *, int) const;
template void GroupCommunicator::ReduceBegin<float>(const float *) const;
template void GroupCommunicator::ReduceEnd<float>(
   float *, int, void (*)(OpData<float>)) const;

// @endcond

// instantiate reduce operators for int and double
template void GroupCommunicator::Sum<int>(OpData<int>);
template void GroupCommunicator::Min<int>(OpData<int>);
template void GroupCommunicator::Max<int>(OpData<int>);
template void GroupCommunicator::BitOR<int>(OpData<int>);

template void GroupCommunicator::Sum<double>(OpData<double>);
template void GroupCommunicator::Min<double>(OpData<double>);
template void GroupCommunicator::Max<double>(OpData<double>);

template void GroupCommunicator::Sum<float>(OpData<float>);
template void GroupCommunicator::Min<float>(OpData<float>);
template void GroupCommunicator::Max<float>(OpData<float>);


#ifdef __bgq__
static void DebugRankCoords(int** coords, int dim, int size)
{
   for (int i = 0; i < size; i++)
   {
      mfem::out << "Rank " << i << " coords: ";
      for (int j = 0; j < dim; j++)
      {
         mfem::out << coords[i][j] << " ";
      }
      mfem::out << endl;
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

      // DebugRankCoords(coords, dim, size);

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
