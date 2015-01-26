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

#ifndef MFEM_COMMUNICATION
#define MFEM_COMMUNICATION

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <mpi.h>
#include "array.hpp"
#include "table.hpp"
#include "sets.hpp"

namespace mfem
{

class GroupTopology
{
private:
   MPI_Comm   MyComm;

   /* The shared entities (e.g. vertices, faces and edges) are split into
      groups, each group determined by the set of participating processors.
      They are numbered locally in lproc. Assumptions:
      - group 0 is the 'local' group
      - groupmaster_lproc[0] = 0
      - lproc_proc[0] = MyRank */
   Table      group_lproc;
   Array<int> groupmaster_lproc;
   Array<int> lproc_proc;
   Array<int> group_mgroup; // group --> group number in the master

   void ProcToLProc();

public:
   GroupTopology(MPI_Comm comm) { MyComm = comm; }

   MPI_Comm GetComm() { return MyComm; }
   int MyRank() { int r; MPI_Comm_rank(MyComm, &r); return r; }
   int NRanks() { int s; MPI_Comm_size(MyComm, &s); return s; }

   void Create(ListOfIntegerSets &groups, int mpitag);

   int NGroups() const { return group_lproc.Size(); }
   // return the number of neighbors including the local processor
   int GetNumNeighbors() const { return lproc_proc.Size(); }
   int GetNeighborRank(int i) const { return lproc_proc[i]; }
   // am I master for group 'g'?
   bool IAmMaster(int g) const { return (groupmaster_lproc[g] == 0); }
   // return the neighbor index of the group master for a given group.
   // neighbor 0 is the local processor
   int GetGroupMaster(int g) const { return groupmaster_lproc[g]; }
   // return the rank of the group master for a given group
   int GetGroupMasterRank(int g) const
   { return lproc_proc[groupmaster_lproc[g]]; }
   // for a given group return the group number in the master
   int GetGroupMasterGroup(int g) const { return group_mgroup[g]; }
   // get the number of processors in a group
   int GetGroupSize(int g) const { return group_lproc.RowSize(g); }
   // return a pointer to a list of neighbors for a given group.
   // neighbor 0 is the local processor
   const int *GetGroup(int g) const { return group_lproc.GetRow(g); }
};

class GroupCommunicator
{
private:
   GroupTopology &gtopo;
   Table group_ldof;
   int group_buf_size;
   Array<char> group_buf;
   MPI_Request *requests;
   MPI_Status  *statuses;

   /** Function template that returns the MPI_Datatype for a given C++ type.
       We explicitly define this function for int and double. */
   template <class T> static inline MPI_Datatype Get_MPI_Datatype();

public:
   GroupCommunicator(GroupTopology &gt);
   /** Initialize the communicator from a local-dof to group map.
       Finalize is called internally. */
   void Create(Array<int> &ldof_group);
   /** Fill-in the returned Table reference to initialize the communicator
       then call Finalize. */
   Table &GroupLDofTable() { return group_ldof; }
   /// Allocate internal buffers after the GroupLDofTable is defined
   void Finalize();

   /// Get a reference to the group topology object
   GroupTopology & GetGroupTopology() { return gtopo; }

   /** Broadcast within each group where the master is the root.
       This method is instantiated for int and double. */
   template <class T> void Bcast(T *ldata);
   template <class T> void Bcast(Array<T> &ldata) { Bcast<T>((T *)ldata); }

   /** Data structure on which we define reduce operations. The data is
       associated with (and the operation is performed on) one group at a
       time. */
   template <class T> struct OpData
   {
      int nldofs, nb, *ldofs;
      T *ldata, *buf;
   };

   /** Reduce within each group where the master is the root. The reduce
       operation is given by the second argument (see below for list of the
       supported operations.) This method is instantiated for int and double. */
   template <class T> void Reduce(T *ldata, void (*Op)(OpData<T>));
   template <class T> void Reduce(Array<T> &ldata, void (*Op)(OpData<T>))
   { Reduce<T>((T *)ldata, Op); }

   /// Reduce operation Sum, instantiated for int and double
   template <class T> static void Sum(OpData<T>);
   /// Reduce operation Min, instantiated for int and double
   template <class T> static void Min(OpData<T>);
   /// Reduce operation Max, instantiated for int and double
   template <class T> static void Max(OpData<T>);
   /// Reduce operation bitwise OR, instantiated for int only
   template <class T> static void BitOR(OpData<T>);

   ~GroupCommunicator();
};

}

#endif

#endif
