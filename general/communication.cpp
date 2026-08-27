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
#include "device.hpp"
#include "forall.hpp"
#include "text.hpp"
#include "sort_pairs.hpp"
#include "globals.hpp"

#ifdef MFEM_USE_STRUMPACK
#include <StrumpackConfig.hpp> // STRUMPACK_USE_PTSCOTCH, etc.
#endif

#include <iostream>
#include <map>
#include <utility> // std::as_const

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
   have_ltdof_ldof = false;
   ldof_size = -1; // unknown ldof_size
   device_gc = NULL;
}

void GroupCommunicator::Create(const Array<int> &ldof_group)
{
   MFEM_VERIFY(buf_offsets == nullptr,
               "the GroupCommunicator is already Finalized!");

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
   if (buf_offsets) { return; } // Finalize() was already called.

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
   MFEM_VERIFY(!have_ltdof_ldof,
               "SetLTDofTable() should be called at most once!");

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

   ldof_size = ldof_ltdof.Size();
   const int ltdof_size = (ldof_size == 0) ? 0 :
                          std::max(ldof_ltdof.Max() + 1, 0);
   ltdof_ldof.SetSize(ltdof_size);
#ifdef MFEM_DEBUG
   int ltdof_counter = 0;
   ltdof_ldof = -1;
#endif
   for (int ldof = 0; ldof < ldof_ltdof.Size(); ldof++)
   {
      const int ltdof = ldof_ltdof[ldof];
      if (ltdof >= 0)
      {
#ifdef MFEM_DEBUG
         ltdof_counter++;
#endif
         MFEM_ASSERT(ltdof_ldof[ltdof] == -1, "repeated ltdof indices found!");
         ltdof_ldof[ltdof] = ldof;
      }
   }
   MFEM_ASSERT(ltdof_counter == ltdof_size, "unassigned ltdof indices found!");
   have_ltdof_ldof = true;
}

namespace internal
{

static void BuildNeighborDofTable(const Table &group_dof,
                                  const Table &nbr_groups,
                                  Table &nbr_dof)
{
   nbr_dof.MakeI(nbr_groups.Size());
   for (int nbr = 1; nbr < nbr_groups.Size(); nbr++)
   {
      const int num_groups = nbr_groups.RowSize(nbr);
      if (num_groups == 0) { continue; }
      const int *grp_list = nbr_groups.GetRow(nbr);
      for (int i = 0; i < num_groups; i++)
      {
         const int group = grp_list[i];
         const int ndofs = group_dof.RowSize(group);
         nbr_dof.AddColumnsInRow(nbr, ndofs);
      }
   }
   nbr_dof.MakeJ();
   for (int nbr = 1; nbr < nbr_groups.Size(); nbr++)
   {
      const int num_groups = nbr_groups.RowSize(nbr);
      if (num_groups == 0) { continue; }
      const int *grp_list = nbr_groups.GetRow(nbr);
      for (int i = 0; i < num_groups; i++)
      {
         const int group = grp_list[i];
         const int ndofs = group_dof.RowSize(group);
         const int *dofs = group_dof.GetRow(group);
         nbr_dof.AddConnections(nbr, dofs, ndofs);
      }
   }
   nbr_dof.ShiftUpI();
}

} // namespace internal

void GroupCommunicator::GetNeighborLTDofTable(Table &nbr_ltdof) const
{
   internal::BuildNeighborDofTable(group_ltdof, nbr_send_groups, nbr_ltdof);
}

void GroupCommunicator::GetNeighborLDofTable(Table &nbr_ldof) const
{
   internal::BuildNeighborDofTable(group_ldof, nbr_recv_groups, nbr_ldof);
}

const DeviceGroupCommunicator &GroupCommunicator::GetDeviceComm() const
{
   if (!device_gc)
   {
      // The ctor of DeviceGroupCommunicator verifies that the GroupCommunicator
      // meets all requirements: mode == byNeighbor and have_ltdof_ldof == true.
      device_gc = new DeviceGroupCommunicator(*this);
   }
   return *device_gc;
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
   MFEM_ASSERT(0 <= layout && layout <= 2, "invalid layout: " << layout);

   if (group_buf_size == 0)
   {
      comm_lock = 1; // 1 - locked for Bcast
      return;
   }

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
void GroupCommunicator::BcastBegin(Array<T> &ldata, int layout) const
{
   MFEM_VERIFY(comm_lock == 0, "object is already in use");
   MFEM_ASSERT(0 <= layout && layout <= 2, "invalid layout: " << layout);
#ifdef MFEM_DEBUG
   // for layouts 0 and 2, ldata_size is known only when have_ltdof_ldof is true
   if (layout == 1 || have_ltdof_ldof)
   {
      // FIXME: Currently, this check causes a failure in the unit test
      //          "Parallel Variable Order FiniteElementSpace" "Quad mesh"
      //        Re-enable this check when the issue is fixed.

      // const int ldata_size = layout == 0 ? ldof_size :
      //                        layout == 1 ? group_ldof.Size_of_connections() :
      //                        ltdof_ldof.Size();
      // MFEM_ASSERT(ldata.Size() == ldata_size, "invalid 'ldata' size");
   }
#endif

   if (group_buf_size == 0)
   {
      comm_lock = 1; // 1 - locked for Bcast
      return;
   }

   // Use 'while' instead of 'if' so that we can break out without using 'goto'.
   while (ldata.UseDevice() &&
          Device::Allows(Backend::DEVICE_MASK) &&
          have_ltdof_ldof &&
          mode == byNeighbor)
   {
      if (layout == 0)  // input is ldofs array
      {
         GetDeviceComm().BcastBeginLDofs(ldata);
      }
      else if (layout == 2)  // input is ltdofs array
      {
         GetDeviceComm().BcastBeginTDofs(ldata);
      }
      else
      {
         break;
      }
      comm_lock = 1; // 1 - locked for Bcast
      return;
   }

   // Call the host version of this method with the data moved to host
   BcastBegin(ldata.HostReadWrite(), layout);
   // comm_lock is set by the above call
}

template <class T>
void GroupCommunicator::BcastEnd(T *ldata, int layout) const
{
   // Is there a real case where we want to allow BcastEnd without corresponding
   // BcastBegin?
   // if (comm_lock == 0) { return; }

   MFEM_VERIFY(comm_lock == 1, "object is NOT locked for Bcast");
   MFEM_ASSERT(layout == 0 || layout == 1, "invalid layout: " << layout);

   if (group_buf_size == 0)
   {
      comm_lock = 0; // 0 - no lock
      return;
   }

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
void GroupCommunicator::BcastEnd(Array<T> &ldata, int layout) const
{
   MFEM_VERIFY(comm_lock == 1, "object is NOT locked for Bcast");
   MFEM_ASSERT(layout == 0 || layout == 1, "invalid layout: " << layout);
#ifdef MFEM_DEBUG
   // for layouts 0 and 2, ldata_size is known only when have_ltdof_ldof is true
   if (layout == 1 || have_ltdof_ldof)
   {
      // FIXME: Currently, this check causes a failure in the unit test
      //          "Parallel Variable Order FiniteElementSpace" "Quad mesh"
      //        Re-enable this check when the issue is fixed.

      // const int ldata_size = layout == 0 ? ldof_size :
      //                        group_ldof.Size_of_connections();
      // MFEM_ASSERT(ldata.Size() == ldata_size, "invalid 'ldata' size");
   }
#endif

   if (group_buf_size == 0)
   {
      comm_lock = 0; // 0 - no lock
      return;
   }

   if (ldata.UseDevice() &&
       Device::Allows(Backend::DEVICE_MASK) &&
       have_ltdof_ldof &&
       mode == byNeighbor &&
       layout == 0)  // output is ldofs array
   {
      GetDeviceComm().BcastEndLDofs(ldata);
      comm_lock = 0; // 0 - no lock
      return;
   }

   // call the host version of this method with the data moved to host
   BcastEnd(ldata.HostReadWrite(), layout);
   // comm_lock is set by the above call
}

template <class T>
void GroupCommunicator::ReduceBegin(const T *ldata) const
{
   MFEM_VERIFY(comm_lock == 0, "object is already in use");

   if (group_buf_size == 0)
   {
      comm_lock = 2; // 2 - locked for Reduce
      return;
   }

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

   comm_lock = 2; // 2 - locked for Reduce
   num_requests = request_counter;
}

template <class T>
void GroupCommunicator::ReduceBegin(const Array<T> &ldata) const
{
   MFEM_VERIFY(comm_lock == 0, "object is already in use");
   // layout is 0
#ifdef MFEM_DEBUG
   if (ldof_size >= 0) // ldof_size is -1 when it is unknown
   {
      // FIXME: Currently, this check causes a failure in the unit test
      //          "Parallel Variable Order FiniteElementSpace" "Quad mesh"
      //        Re-enable this check when the issue is fixed.

      // MFEM_ASSERT(ldata.Size() == ldof_size, "invalid 'ldata' size");
   }
#endif

   if (group_buf_size == 0)
   {
      comm_lock = 2; // 2 - locked for Reduce
      return;
   }

   if (ldata.UseDevice() &&
       Device::Allows(Backend::DEVICE_MASK) &&
       have_ltdof_ldof &&
       mode == byNeighbor)
   {
      GetDeviceComm().ReduceBeginLDofs(ldata);
      comm_lock = 2; // 2 - locked for Reduce
      return;
   }

   // call the host version of this method with the data copied to host
   ReduceBegin(ldata.HostRead());
   // comm_lock is set by the above call
}

template <class T>
void GroupCommunicator::ReduceEnd(T *ldata, int layout,
                                  void (*Op)(OpData<T>)) const
{
   // Is there a real case where we want to allow ReduceEnd without
   // corresponding ReduceBegin?
   // if (comm_lock == 0) { return; }

   MFEM_VERIFY(comm_lock == 2, "object is NOT locked for Reduce");
   MFEM_ASSERT(layout == 0 || layout == 2, "invalid layout: " << layout);

   if (group_buf_size == 0)
   {
      comm_lock = 0; // 0 - no lock
      return;
   }

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
void GroupCommunicator::ReduceEnd(Array<T> &ldata, int layout,
                                  void (*Op)(OpData<T>)) const
{
   MFEM_VERIFY(comm_lock == 2, "object is NOT locked for Reduce");
   MFEM_ASSERT(layout == 0 || layout == 2, "invalid layout: " << layout);
#ifdef MFEM_DEBUG
   // for layouts 0 and 2, ldata_size is known only when have_ltdof_ldof is true
   if (have_ltdof_ldof)
   {
      // FIXME: Currently, this check causes a failure in the unit test
      //          "Parallel Variable Order FiniteElementSpace" "Quad mesh"
      //        Re-enable this check when the issue is fixed.

      // const int ldata_size = layout == 0 ? ldof_size : ltdof_ldof.Size();
      // MFEM_ASSERT(ldata.Size() == ldata_size, "invalid 'ldata' size");
   }
#endif

   if (group_buf_size == 0)
   {
      comm_lock = 0; // 0 - no lock
      return;
   }

   if (ldata.UseDevice() &&
       Device::Allows(Backend::DEVICE_MASK) &&
       have_ltdof_ldof &&
       mode == byNeighbor)
   {
      DeviceGroupCommunicator::Op device_op{};
      auto op_is_supported_device_op = [&]() -> bool
      {
         // Materialize typed function pointers before comparison so stricter
         // GPU toolchains do not have to resolve overloaded template names
         // here.
         using OpFunc = void (*)(OpData<T>);
         const OpFunc sum_op = &GroupCommunicator::template Sum<T>;
         const OpFunc min_op = &GroupCommunicator::template Min<T>;
         const OpFunc max_op = &GroupCommunicator::template Max<T>;
         if (Op == sum_op)
         {
            device_op = DeviceGroupCommunicator::Op::Sum;
         }
         else if (Op == min_op)
         {
            device_op = DeviceGroupCommunicator::Op::Min;
         }
         else if (Op == max_op)
         {
            device_op = DeviceGroupCommunicator::Op::Max;
         }
         else
         {
            return false; // Op is not supported on device
         }
         return true; // Op is supported on device
      };
      if (op_is_supported_device_op())
      {
         if (layout == 0)  // output is ldofs array
         {
            GetDeviceComm().ReduceEndLDofs(ldata, device_op);
         }
         else // layout == 2 -- output is ltdofs array
         {
            GetDeviceComm().ReduceEndTDofs(ldata, device_op);
         }
         comm_lock = 0; // 0 - no lock
         return;
      }
   }

   // call the host version of this method with the data moved to host
   ReduceEnd(ldata.HostReadWrite(), layout, Op);
   // comm_lock is set by the above call
}

template <class T>
void GroupCommunicator::ReduceMarked(T *ldata, const Array<int> &marker,
                                     int layout,
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

            // Apply operation only to marked DOFs. The receive buffer is
            // neighbor-major with stride opd.nldofs, i.e. the contributions to
            // DOF i are buf[j*opd.nldofs + i] for j = 0 ... opd.nb-1. Setting
            // nldofs = 1 for a single DOF changes that stride to 1, so the
            // strided values must first be gathered into a contiguous buffer.
            Array<T> single_buf(opd.nb);
            for (int i = 0; i < opd.nldofs; i++)
            {
               if (marker[opd.ldofs[i]])
               {
                  for (int j = 0; j < opd.nb; j++)
                  {
                     single_buf[j] = opd.buf[j*opd.nldofs + i];
                  }

                  // Create a temporary OpData with just this one DOF
                  OpData<T> single_opd;
                  single_opd.ldata = ldata;
                  single_opd.buf = single_buf.GetData();
                  single_opd.ldofs = opd.ldofs + i;
                  single_opd.nldofs = 1;
                  single_opd.nb = opd.nb;

                  // Apply the operation
                  Op(single_opd);
               }
            }
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
                  // Custom version of ReduceGroupFromBuffer that checks marker
                  int gr = grp_list[i];
                  const int *ldofs = (layout == 0) ?
                                     group_ldof.GetRow(gr) : group_ltdof.GetRow(gr);
                  const int nldofs = group_ldof.RowSize(gr);

                  for (int j = 0; j < nldofs; j++)
                  {
                     if (marker[ldofs[j]])
                     {
                        // Create a temporary OpData with just this one DOF
                        OpData<T> opd;
                        opd.ldata = ldata;
                        opd.buf = const_cast<T*>(buf) + j;
                        opd.ldofs = ldofs + j;
                        opd.nldofs = 1;
                        opd.nb = 1;

                        // Apply the operation
                        Op(opd);
                     }
                  }

                  buf += nldofs;
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
   static_assert(std::is_integral<T>::value,
                 "BitOR reduction requires an integral type.");
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

template <class T>
void GroupCommunicator::MaxAbs(OpData<T> opd)
{
   for (int i = 0; i < opd.nldofs; i++)
   {
      T data = opd.ldata[opd.ldofs[i]];
      T abs_data = std::abs(data);

      for (int j = 0; j < opd.nb; j++)
      {
         T b = opd.buf[j*opd.nldofs+i];
         T abs_b = std::abs(b);

         // On an equal-magnitude tie keep the more positive value, so
         // opposite-sign ties resolve deterministically to the positive one.
         if (abs_data < abs_b || (abs_data == abs_b && data < b))
         {
            data = b;
            abs_data = abs_b;
         }
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
   delete device_gc;
   delete [] buf_offsets;
   delete [] request_marker;
   // delete [] statuses;
   delete [] requests;
}


namespace internal
{

/** @brief Extract a sub-array: xout[i] = xin[indices[i]].
    Note that the 'indices' can contain repeated integers. */
template <typename T>
static void ExtractSubArray(const Array<int> &indices,
                            const Array<T> &xin,
                            Array<T> &xout)
{
   MFEM_ASSERT(indices.Size() == xout.Size(), "incompatible sizes!");
   auto y = xout.Write();
   const auto x = xin.Read();
   const auto I = indices.Read();
   mfem::forall(indices.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      y[i] = x[I[i]];
   });
}

/** @brief Set a sub-array: xout[indices[i]] = xin[i].
    Note that the 'indices' can NOT contain repeated integers because that will
    create a race condition during parallel execution. */
template <typename T>
static void SetSubArray(const Array<int> &indices,
                        const Array<T> &xin,
                        Array<T> &xout)
{
   MFEM_ASSERT(indices.Size() == xin.Size(), "incompatible sizes!");
   // Use ReadWrite() since we modify only a subset of the indices:
   auto y = xout.ReadWrite();
   const auto x = xin.Read();
   const auto I = indices.Read();
   mfem::forall(indices.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      y[I[i]] = x[i];
   });
}

/** @brief Set a sub-array: xout[indices[i]] = val.
    Note that the 'indices' can contain repeated integers. Since the same value
    is assigned to all given entries, there no real race condition during
    parallel execution. */
template <typename T>
static void SetSubArray(const Array<int> &indices, Array<T> &xout, T val)
{
   // Use ReadWrite() since we modify only a subset of the indices:
   auto y = xout.ReadWrite();
   const auto I = indices.Read();
   mfem::forall(indices.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      y[I[i]] = val;
   });
}

/** @brief Perform the operation: dst += A src, where:
    - A is a Boolean matrix
    - unique_dst_indices are the nonzeros rows of A
    - unique_to_src_offsets and unique_to_src_indices are the I and J arrays of
      the csr format of A restricted to its nonzero rows. */
template <typename T>
static void BooleanAddMult(const Array<int> &unique_dst_indices,
                           const Array<int> &unique_to_src_offsets,
                           const Array<int> &unique_to_src_indices,
                           const Array<T> &src,
                           Array<T> &dst)
{
   auto y = dst.ReadWrite();
   const auto x = src.Read();
   const auto DST_I = unique_dst_indices.Read();
   const auto SRC_O = unique_to_src_offsets.Read();
   const auto SRC_I = unique_to_src_indices.Read();
   mfem::forall(unique_dst_indices.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      const int dst_idx = DST_I[i];
      T sum = y[dst_idx];
      const int end = SRC_O[i+1];
      for (int j = SRC_O[i]; j != end; ++j) { sum += x[SRC_I[j]]; }
      y[dst_idx] = sum;
   });
}

/** @brief Operation similar to BooleanAddMult(): dst += A src, where:
    - the addition operations are replaced by the reduction operation op
    - only nonzero entries of A participate in the reduction operation
    - A is a Boolean matrix
    - unique_dst_indices are the nonzeros rows of A
    - unique_to_src_offsets and unique_to_src_indices are the I and J arrays of
      the csr format of A restricted to its nonzero rows. */
template <typename T>
static void BooleanReduceApply(const Array<int> &unique_dst_indices,
                               const Array<int> &unique_to_src_offsets,
                               const Array<int> &unique_to_src_indices,
                               const Array<T> &src,
                               Array<T> &dst,
                               DeviceGroupCommunicator::Op op)
{
   auto y = dst.ReadWrite();
   const auto x = src.Read();
   const auto DST_I = unique_dst_indices.Read();
   const auto SRC_O = unique_to_src_offsets.Read();
   const auto SRC_I = unique_to_src_indices.Read();
   mfem::forall(unique_dst_indices.Size(), [=] MFEM_HOST_DEVICE (int i)
   {
      const int dst_idx = DST_I[i];
      T val = y[dst_idx];
      const int end = SRC_O[i+1];
      switch (op)
      {
         case DeviceGroupCommunicator::Op::Sum:
            for (int j = SRC_O[i]; j != end; ++j) { val += x[SRC_I[j]]; }
            break;
         case DeviceGroupCommunicator::Op::Min:
            for (int j = SRC_O[i]; j != end; ++j)
            {
               const T xj = x[SRC_I[j]];
               val = (xj < val) ? xj : val;
            }
            break;
         case DeviceGroupCommunicator::Op::Max:
            for (int j = SRC_O[i]; j != end; ++j)
            {
               const T xj = x[SRC_I[j]];
               val = (xj > val) ? xj : val;
            }
            break;
      }
      y[dst_idx] = val;
   });
}

} // namespace internal


DeviceGroupCommunicator::DeviceGroupCommunicator(const GroupCommunicator &gc_)
   : gc(gc_)
{
   MFEM_VERIFY(gc.mode == gc.byNeighbor,
               "Device group-communicator requires neighbor mode.");
   MFEM_VERIFY(gc.have_ltdof_ldof,
               "The GroupCommunicator method SetLTDofTable() must be called "
               "before constructing the DeviceGroupCommunicator!");
   {
      Table nbr_ltdof;
      gc.GetNeighborLTDofTable(nbr_ltdof);
      // Transfer the I and J arrays of nbr_ltdof to shr_buf_offsets and
      // shr_ltdof, respectively:
      shr_buf_offsets.NewMemoryAndSize(nbr_ltdof.GetIMemory(),
                                       nbr_ltdof.Size()+1, true);
      shr_ltdof.NewMemoryAndSize(nbr_ltdof.GetJMemory(),
                                 nbr_ltdof.Size_of_connections(), true);
      nbr_ltdof.LoseData();
   }
   shr_ldof.SetSize(shr_ltdof.Size());
   // shr_ldof[i] = gc.ltdof_ldof[shr_ltdof[i]]:
   internal::ExtractSubArray(shr_ltdof, gc.ltdof_ldof, shr_ldof);
   {
      // Sort() is a host method, so initialize 'unique_ltdof' on host:
      Array<int> unique_ltdof(shr_ltdof.Size());
      unique_ltdof.CopyFrom(shr_ltdof.HostRead());
      unique_ltdof.Sort();
      unique_ltdof.Unique();
      unq_ltdof = unique_ltdof;
   }
   {
      Array<int> shr_unique(shr_ltdof.Size());
      for (int i = 0; i < shr_unique.Size(); i++)
      {
         shr_unique[i] = unq_ltdof.FindSorted(std::as_const(shr_ltdof)[i]);
         MFEM_ASSERT(shr_unique[i] != -1, "internal error");
      }
      Table unique_shr;
      Transpose(shr_unique, unique_shr, unq_ltdof.Size());
      // Transfer the I and J arrays of unique_shr to unq_shr_i and unq_shr_j,
      // respectively:
      unq_shr_i.NewMemoryAndSize(unique_shr.GetIMemory(),
                                 unique_shr.Size()+1, true);
      unq_shr_j.NewMemoryAndSize(unique_shr.GetJMemory(),
                                 unique_shr.Size_of_connections(), true);
      unique_shr.LoseData();
   }
   unq_ldof.SetSize(unq_ltdof.Size());
   // unq_ldof[i] = gc.ltdof_ldof[unq_ltdof[i]]:
   internal::ExtractSubArray(unq_ltdof, gc.ltdof_ldof, unq_ldof);
   {
      Table nbr_ldof;
      gc.GetNeighborLDofTable(nbr_ldof);
      // Transfer the I and J arrays of nbr_ldof to ext_buf_offsets and
      // ext_ldof, respectively:
      ext_buf_offsets.NewMemoryAndSize(nbr_ldof.GetIMemory(),
                                       nbr_ldof.Size()+1, true);
      ext_ldof.NewMemoryAndSize(nbr_ldof.GetJMemory(),
                                nbr_ldof.Size_of_connections(), true);
      ext_ldof.GetMemory().UseDevice(true);
      nbr_ldof.LoseData();
   }

   shr_buf.SetSize(shr_ltdof.Size());
   ext_buf.SetSize(ext_ldof.Size());
   // Allocate the buffers on device to make sure the reinterpred_cast versions
   // used by MakeTypedBufferView do not need to allocate on device -- they will
   // allocate less memory if the type has smaller size.
   shr_buf.Write();
   ext_buf.Write();

   const GroupTopology &gtopo = gc.GetGroupTopology();
   int req_counter = 0;
   for (int nbr = 1; nbr < gtopo.GetNumNeighbors(); nbr++)
   {
      const int send_offset = shr_buf_offsets[nbr];
      const int send_size = shr_buf_offsets[nbr+1] - send_offset;
      if (send_size > 0) { req_counter++; }

      const int recv_offset = ext_buf_offsets[nbr];
      const int recv_size = ext_buf_offsets[nbr+1] - recv_offset;
      if (recv_size > 0) { req_counter++; }
   }
   requests.SetSize(req_counter);
   num_requests = 0;
}

// Returns 'storage' reinterpret_cast as Array<T> &.
// The returned array should not be resized in a way where new bigger
// allocations are needed, unless the type T has the same size as BT.
template <typename T, typename BT>
static inline Array<T> &MakeTypedBufferView(Array<BT> &storage)
{
   static_assert(sizeof(BT) >= sizeof(T),
                 "internal buffer type is too small for this view!");
   return *reinterpret_cast<Array<T>*>(&storage);
}

template <typename T>
void DeviceGroupCommunicator::BcastBeginTDofs(Array<T> &x_tdof) const
{
   Array<T> &shr_buf_t = MakeTypedBufferView<T>(shr_buf);
   Array<T> &ext_buf_t = MakeTypedBufferView<T>(ext_buf);
   BcastBeginCopyTDofs(x_tdof, shr_buf_t);
   ExchangeSharedToExternal(shr_buf_t, ext_buf_t);
}

template <typename T>
void DeviceGroupCommunicator::BcastBeginLDofs(Array<T> &x_ldof) const
{
   Array<T> &shr_buf_t = MakeTypedBufferView<T>(shr_buf);
   Array<T> &ext_buf_t = MakeTypedBufferView<T>(ext_buf);
   BcastBeginCopyLDofs(x_ldof, shr_buf_t);
   ExchangeSharedToExternal(shr_buf_t, ext_buf_t);
}

template <typename T>
void DeviceGroupCommunicator::BcastEndLDofs(Array<T> &x_ldof) const
{
   Array<T> &ext_buf_t = MakeTypedBufferView<T>(ext_buf);
   WaitAll();
   BcastEndCopy(ext_buf_t, x_ldof);
}

template <typename T>
void DeviceGroupCommunicator::ReduceBeginLDofs(const Array<T> &x_ldof) const
{
   Array<T> &ext_buf_t = MakeTypedBufferView<T>(ext_buf);
   Array<T> &shr_buf_t = MakeTypedBufferView<T>(shr_buf);
   ReduceBeginCopy(x_ldof, ext_buf_t);
   ExchangeExternalToShared(ext_buf_t, shr_buf_t);
}

template <typename T>
void DeviceGroupCommunicator::ReduceEndTDofs(Array<T> &x_tdof, Op op) const
{
   Array<T> &shr_buf_t = MakeTypedBufferView<T>(shr_buf);
   WaitAll();
   ReduceEndAssembleTDofs(shr_buf_t, x_tdof, op);
}

template <typename T>
void DeviceGroupCommunicator::ReduceEndLDofs(Array<T> &x_ldof, Op op) const
{
   Array<T> &shr_buf_t = MakeTypedBufferView<T>(shr_buf);
   WaitAll();
   if (unq_ldof.Size() == 0) { return; }
   if (op == Op::Sum)
   {
      internal::BooleanAddMult(unq_ldof, unq_shr_i, unq_shr_j,
                               shr_buf_t, x_ldof);
   }
   else
   {
      internal::BooleanReduceApply(unq_ldof, unq_shr_i, unq_shr_j,
                                   shr_buf_t, x_ldof, op);
   }
}

template <typename T>
void DeviceGroupCommunicator::CopyTDofsToLDofs(const Array<T> &x_tdof,
                                               Array<T> &x_ldof) const
{
   if (gc.ltdof_ldof.Size() == 0) { return; }
   internal::SetSubArray(gc.ltdof_ldof, x_tdof, x_ldof);
}

template <typename T>
void DeviceGroupCommunicator::Prolongate(const Array<T> &x_tdof,
                                         Array<T> &x_ldof) const
{
   MFEM_ASSERT(x_tdof.Size() == gc.ltdof_ldof.Size(), "incompatible sizes!");
   MFEM_ASSERT(x_ldof.Size() == gc.ldof_size, "incompatible sizes!");
   Array<T> &ext_buf_t = MakeTypedBufferView<T>(ext_buf);
   Array<T> &shr_buf_t = MakeTypedBufferView<T>(shr_buf);
   BcastBeginCopyTDofs(x_tdof, shr_buf_t);
   ExchangeSharedToExternal(shr_buf_t, ext_buf_t);
   CopyTDofsToLDofs(x_tdof, x_ldof);
   WaitAll();
   BcastEndCopy(ext_buf_t, x_ldof);
}

template <typename T>
void DeviceGroupCommunicator::ProlongateTranspose(const Array<T> &x_ldof,
                                                  Array<T> &x_tdof,
                                                  Op op) const
{
   MFEM_ASSERT(x_ldof.Size() == gc.ldof_size, "incompatible sizes!");
   MFEM_ASSERT(x_tdof.Size() == gc.ltdof_ldof.Size(), "incompatible sizes!");
   Array<T> &ext_buf_t = MakeTypedBufferView<T>(ext_buf);
   Array<T> &shr_buf_t = MakeTypedBufferView<T>(shr_buf);
   ReduceBeginCopy(x_ldof, ext_buf_t);
   ExchangeExternalToShared(ext_buf_t, shr_buf_t);
   Restrict(x_ldof, x_tdof);
   WaitAll();
   ReduceEndAssembleTDofs(shr_buf_t, x_tdof, op);
}

template <typename T>
void DeviceGroupCommunicator::Restrict(const Array<T> &x_ldof,
                                       Array<T> &x_tdof) const
{
   if (gc.ltdof_ldof.Size() == 0) { return; }
   internal::ExtractSubArray(gc.ltdof_ldof, x_ldof, x_tdof);
}

template <typename T>
void DeviceGroupCommunicator::RestrictTranspose(const Array<T> &x_tdof,
                                                Array<T> &x_ldof) const
{
   CopyTDofsToLDofs(x_tdof, x_ldof);
   internal::SetSubArray(ext_ldof, x_ldof, T(0));
}

template <typename T>
void DeviceGroupCommunicator::Exchange(const Array<T> &send_buf,
                                       const Array<int> &send_offsets,
                                       Array<T> &recv_buf,
                                       const Array<int> &recv_offsets,
                                       int tag) const
{
   const GroupTopology &gtopo = gc.GetGroupTopology();
   const bool mpi_gpu_aware = Device::GetGPUAwareMPI();
   auto send_ptr = mpi_gpu_aware ? send_buf.Read() : send_buf.HostRead();
   auto recv_ptr = mpi_gpu_aware ? recv_buf.Write() : recv_buf.HostWrite();
   num_requests = 0;
   for (int nbr = 1; nbr < gtopo.GetNumNeighbors(); nbr++)
   {
      const int send_offset = send_offsets[nbr];
      const int send_size = send_offsets[nbr+1] - send_offset;
      if (send_size > 0)
      {
         MPI_Isend(send_ptr + send_offset, send_size, MPITypeMap<T>::mpi_type,
                   gtopo.GetNeighborRank(nbr), tag, gtopo.GetComm(),
                   &requests[num_requests++]);
      }
      const int recv_offset = recv_offsets[nbr];
      const int recv_size = recv_offsets[nbr+1] - recv_offset;
      if (recv_size > 0)
      {
         MPI_Irecv(recv_ptr + recv_offset, recv_size, MPITypeMap<T>::mpi_type,
                   gtopo.GetNeighborRank(nbr), tag, gtopo.GetComm(),
                   &requests[num_requests++]);
      }
   }
}

template <typename T>
void DeviceGroupCommunicator::ExchangeSharedToExternal(
   const Array<T> &shr_buf_t, Array<T> &ext_buf_t) const
{
   const int tag = 41822;
   Exchange(shr_buf_t, shr_buf_offsets, ext_buf_t, ext_buf_offsets, tag);
}

template <typename T>
void DeviceGroupCommunicator::ExchangeExternalToShared(
   const Array<T> &ext_buf_t, Array<T> &shr_buf_t) const
{
   const int tag = 41823;
   Exchange(ext_buf_t, ext_buf_offsets, shr_buf_t, shr_buf_offsets, tag);
}

template <typename T>
void DeviceGroupCommunicator::ReduceBeginCopy(const Array<T> &x_ldof,
                                              Array<T> &ext_buf_t) const
{
   if (ext_ldof.Size() == 0) { return; }
   internal::ExtractSubArray(ext_ldof, x_ldof, ext_buf_t);
   if (Device::GetGPUAwareMPI()) { MFEM_STREAM_SYNC; }
}

template <typename T>
void DeviceGroupCommunicator::ReduceEndAssembleTDofs(const Array<T> &shr_buf_t,
                                                     Array<T> &x_tdof,
                                                     Op op) const
{
   if (unq_ltdof.Size() == 0) { return; }
   if (op == Op::Sum)
   {
      internal::BooleanAddMult(unq_ltdof, unq_shr_i, unq_shr_j,
                               shr_buf_t, x_tdof);
   }
   else
   {
      internal::BooleanReduceApply(unq_ltdof, unq_shr_i, unq_shr_j,
                                   shr_buf_t, x_tdof, op);
   }
}

template <typename T>
void DeviceGroupCommunicator::BcastBeginCopyTDofs(const Array<T> &x_tdof,
                                                  Array<T> &shr_buf_t) const
{
   if (shr_ltdof.Size() == 0) { return; }
   internal::ExtractSubArray(shr_ltdof, x_tdof, shr_buf_t);
   if (Device::GetGPUAwareMPI()) { MFEM_STREAM_SYNC; }
}

template <typename T>
void DeviceGroupCommunicator::BcastBeginCopyLDofs(const Array<T> &x_ldof,
                                                  Array<T> &shr_buf_t) const
{
   if (shr_ldof.Size() == 0) { return; }
   internal::ExtractSubArray(shr_ldof, x_ldof, shr_buf_t);
   if (Device::GetGPUAwareMPI()) { MFEM_STREAM_SYNC; }
}

template <typename T>
void DeviceGroupCommunicator::BcastEndCopy(const Array<T> &ext_buf_t,
                                           Array<T> &x_ldof) const
{
   if (ext_ldof.Size() == 0) { return; }
   internal::SetSubArray(ext_ldof, ext_buf_t, x_ldof);
}

void DeviceGroupCommunicator::WaitAll() const
{
   MPI_Waitall(num_requests, requests.GetData(), MPI_STATUSES_IGNORE);
}

/// @cond DOXYGEN_SKIP

// instantiate GroupCommunicator::Bcast and Reduce for int, double, and float
template void GroupCommunicator::BcastBegin<int>(int *, int) const;
template void GroupCommunicator::BcastBegin<int>(Array<int> &, int) const;
template void GroupCommunicator::BcastEnd<int>(int *, int) const;
template void GroupCommunicator::BcastEnd<int>(Array<int> &, int) const;
template void GroupCommunicator::ReduceBegin<int>(const int *) const;
template void GroupCommunicator::ReduceBegin<int>(const Array<int> &) const;
template void GroupCommunicator::ReduceEnd<int>(
   int *, int, void (*)(OpData<int>)) const;
template void GroupCommunicator::ReduceEnd<int>(
   Array<int> &, int, void (*)(OpData<int>)) const;
template void GroupCommunicator::ReduceMarked<int>(
   int*, const Array<int>&, int, void (*)(OpData<int>)) const;

template void GroupCommunicator::BcastBegin<double>(double *, int) const;
template void GroupCommunicator::BcastBegin<double>(Array<double> &, int) const;
template void GroupCommunicator::BcastEnd<double>(double *, int) const;
template void GroupCommunicator::BcastEnd<double>(Array<double> &, int) const;
template void GroupCommunicator::ReduceBegin<double>(const double *) const;
template void GroupCommunicator::ReduceBegin<double>(
   const Array<double> &) const;
template void GroupCommunicator::ReduceEnd<double>(
   double *, int, void (*)(OpData<double>)) const;
template void GroupCommunicator::ReduceEnd<double>(
   Array<double> &, int, void (*)(OpData<double>)) const;
template void GroupCommunicator::ReduceMarked<double>(
   double*, const Array<int>&, int, void (*)(OpData<double>)) const;

template void GroupCommunicator::BcastBegin<float>(float *, int) const;
template void GroupCommunicator::BcastBegin<float>(Array<float> &, int) const;
template void GroupCommunicator::BcastEnd<float>(float *, int) const;
template void GroupCommunicator::BcastEnd<float>(Array<float> &, int) const;
template void GroupCommunicator::ReduceBegin<float>(const float *) const;
template void GroupCommunicator::ReduceBegin<float>(const Array<float> &) const;
template void GroupCommunicator::ReduceEnd<float>(
   float *, int, void (*)(OpData<float>)) const;
template void GroupCommunicator::ReduceEnd<float>(
   Array<float> &, int, void (*)(OpData<float>)) const;
template void GroupCommunicator::ReduceMarked<float>(
   float*, const Array<int>&, int, void (*)(OpData<float>)) const;

/// @endcond

// instantiate reduce operators for int, double, and float
template void GroupCommunicator::Sum<int>(OpData<int>);
template void GroupCommunicator::Min<int>(OpData<int>);
template void GroupCommunicator::Max<int>(OpData<int>);
template void GroupCommunicator::BitOR<int>(OpData<int>);
template void GroupCommunicator::MaxAbs<int>(OpData<int>);

template void GroupCommunicator::Sum<double>(OpData<double>);
template void GroupCommunicator::Min<double>(OpData<double>);
template void GroupCommunicator::Max<double>(OpData<double>);
template void GroupCommunicator::MaxAbs<double>(OpData<double>);

template void GroupCommunicator::Sum<float>(OpData<float>);
template void GroupCommunicator::Min<float>(OpData<float>);
template void GroupCommunicator::Max<float>(OpData<float>);
template void GroupCommunicator::MaxAbs<float>(OpData<float>);


/// @cond DOXYGEN_SKIP

template void DeviceGroupCommunicator::BcastBeginTDofs<int>(Array<int> &) const;
template void DeviceGroupCommunicator::BcastBeginLDofs<int>(Array<int> &) const;
template void DeviceGroupCommunicator::BcastEndLDofs<int>(Array<int> &) const;
template void DeviceGroupCommunicator::ReduceBeginLDofs<int>(
   const Array<int> &) const;
template void DeviceGroupCommunicator::ReduceEndTDofs<int>(
   Array<int> &, Op) const;
template void DeviceGroupCommunicator::ReduceEndLDofs<int>(
   Array<int> &, Op) const;
template void DeviceGroupCommunicator::CopyTDofsToLDofs<int>(
   const Array<int> &, Array<int> &) const;
template void DeviceGroupCommunicator::Prolongate<int>(
   const Array<int> &, Array<int> &) const;
template void DeviceGroupCommunicator::ProlongateTranspose<int>(
   const Array<int> &, Array<int> &, Op) const;
template void DeviceGroupCommunicator::Restrict<int>(
   const Array<int> &, Array<int> &) const;
template void DeviceGroupCommunicator::RestrictTranspose<int>(
   const Array<int> &, Array<int> &) const;

template void DeviceGroupCommunicator::BcastBeginTDofs<real_t>(
   Array<real_t> &) const;
template void DeviceGroupCommunicator::BcastBeginLDofs<real_t>(
   Array<real_t> &) const;
template void DeviceGroupCommunicator::BcastEndLDofs<real_t>(
   Array<real_t> &) const;
template void DeviceGroupCommunicator::ReduceBeginLDofs<real_t>(
   const Array<real_t> &) const;
template void DeviceGroupCommunicator::ReduceEndTDofs<real_t>(
   Array<real_t> &, Op) const;
template void DeviceGroupCommunicator::ReduceEndLDofs<real_t>(
   Array<real_t> &, Op) const;
template void DeviceGroupCommunicator::CopyTDofsToLDofs<real_t>(
   const Array<real_t> &, Array<real_t> &) const;
template void DeviceGroupCommunicator::Prolongate<real_t>(
   const Array<real_t> &, Array<real_t> &) const;
template void DeviceGroupCommunicator::ProlongateTranspose<real_t>(
   const Array<real_t> &, Array<real_t> &, Op) const;
template void DeviceGroupCommunicator::Restrict<real_t>(
   const Array<real_t> &, Array<real_t> &) const;
template void DeviceGroupCommunicator::RestrictTranspose<real_t>(
   const Array<real_t> &, Array<real_t> &) const;

/// @endcond


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
