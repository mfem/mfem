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

#ifndef MFEM_COMMUNICATION
#define MFEM_COMMUNICATION

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "array.hpp"
#include "table.hpp"
#include "sets.hpp"
#include "globals.hpp"
#include <mpi.h>
#include <cstdint>

// can't directly use MPI_CXX_BOOL because Microsoft's MPI implementation
// doesn't include MPI_CXX_BOOL. Fallback to MPI_C_BOOL if unavailable.
#ifdef MPI_CXX_BOOL
#define MFEM_MPI_CXX_BOOL MPI_CXX_BOOL
#else
#define MFEM_MPI_CXX_BOOL MPI_C_BOOL
#endif

namespace mfem
{

/** @brief A simple singleton class that calls MPI_Init() at construction and
    MPI_Finalize() at destruction. It also provides easy access to
    MPI_COMM_WORLD's rank and size. */
class Mpi
{
public:
   /// Singleton creation with Mpi::Init(argc, argv).
   static void Init(int &argc, char **&argv,
                    int required = default_thread_required,
                    int *provided = nullptr)
   { Init(&argc, &argv, required, provided); }
   /// Singleton creation with Mpi::Init().
   static void Init(int *argc = nullptr, char ***argv = nullptr,
                    int required = default_thread_required,
                    int *provided = nullptr)
   {
      MFEM_VERIFY(!IsInitialized(), "MPI already initialized!");
      if (required == MPI_THREAD_SINGLE)
      {
         int mpi_err = MPI_Init(argc, argv);
         MFEM_VERIFY(!mpi_err, "error in MPI_Init()!");
         if (provided) { *provided = MPI_THREAD_SINGLE; }
      }
      else
      {
         int mpi_provided;
         int mpi_err = MPI_Init_thread(argc, argv, required, &mpi_provided);
         MFEM_VERIFY(!mpi_err, "error in MPI_Init()!");
         if (provided) { *provided = mpi_provided; }
      }
      // The Mpi singleton object below needs to be created after MPI_Init() for
      // some MPI implementations.
      Singleton();
   }
   /// Finalize MPI (if it has been initialized and not yet already finalized).
   static void Finalize()
   {
      if (IsInitialized() && !IsFinalized()) { MPI_Finalize(); }
   }
   /// Return true if MPI has been initialized.
   static bool IsInitialized()
   {
      int mpi_is_initialized;
      int mpi_err = MPI_Initialized(&mpi_is_initialized);
      return (mpi_err == MPI_SUCCESS) && mpi_is_initialized;
   }
   /// Return true if MPI has been finalized.
   static bool IsFinalized()
   {
      int mpi_is_finalized;
      int mpi_err = MPI_Finalized(&mpi_is_finalized);
      return (mpi_err == MPI_SUCCESS) && mpi_is_finalized;
   }
   /// Return the MPI rank in MPI_COMM_WORLD.
   static int WorldRank()
   {
      int world_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      return world_rank;
   }
   /// Return the size of MPI_COMM_WORLD.
   static int WorldSize()
   {
      int world_size;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      return world_size;
   }
   /// Return true if the rank in MPI_COMM_WORLD is zero.
   static bool Root() { return WorldRank() == 0; }
   /// Default level of thread support for MPI_Init_thread.
   static MFEM_EXPORT int default_thread_required;
private:
   /// Initialize the Mpi singleton.
   static Mpi &Singleton()
   {
      static Mpi mpi;
      return mpi;
   }
   /// Finalize MPI.
   ~Mpi() { Finalize(); }
   /// Prevent direct construction of objects of this class.
   Mpi() {}
};

/** @brief A simple convenience class based on the Mpi singleton class above.
    Preserved for backward compatibility. New code should use Mpi::Init() and
    other Mpi methods instead. */
class MPI_Session
{
public:
   MPI_Session() { Mpi::Init(); }
   MPI_Session(int &argc, char **&argv) { Mpi::Init(argc, argv); }
   /// Return MPI_COMM_WORLD's rank.
   int WorldRank() const { return Mpi::WorldRank(); }
   /// Return MPI_COMM_WORLD's size.
   int WorldSize() const { return Mpi::WorldSize(); }
   /// Return true if WorldRank() == 0.
   bool Root() const { return Mpi::Root(); }
};


/** The shared entities (e.g. vertices, faces and edges) are split into groups,
    each group determined by the set of participating processors. They are
    numbered locally in lproc. Assumptions:
    - group 0 is the 'local' group
    - groupmaster_lproc[0] = 0
    - lproc_proc[0] = MyRank */
class GroupTopology
{
private:
   MPI_Comm   MyComm;

   /// Neighbor ids (lproc) in each group.
   Table      group_lproc;
   /// Master neighbor id for each group.
   Array<int> groupmaster_lproc;
   /// MPI rank of each neighbor.
   Array<int> lproc_proc;
   /// Group --> Group number in the master.
   Array<int> group_mgroup;

   void ProcToLProc();

public:
   /// Constructor with the MPI communicator = 0.
   GroupTopology() : MyComm(0) {}

   /// Constructor given the MPI communicator 'comm'.
   GroupTopology(MPI_Comm comm) { MyComm = comm; }

   /// Copy constructor
   GroupTopology(const GroupTopology &gt);

   /// Set the MPI communicator to 'comm'.
   void SetComm(MPI_Comm comm) { MyComm = comm; }

   /// Return the MPI communicator.
   MPI_Comm GetComm() const { return MyComm; }

   /// Return the MPI rank within this object's communicator.
   int MyRank() const { int r; MPI_Comm_rank(MyComm, &r); return r; }

   /// Return the number of MPI ranks within this object's communicator.
   int NRanks() const { int s; MPI_Comm_size(MyComm, &s); return s; }

   /// Set up the group topology given the list of sets of shared entities.
   void Create(ListOfIntegerSets &groups, int mpitag);

   /// Return the number of groups.
   int NGroups() const { return group_lproc.Size(); }

   /// Return the number of neighbors including the local processor.
   int GetNumNeighbors() const { return lproc_proc.Size(); }

   /// Return the MPI rank of neighbor 'i'.
   int GetNeighborRank(int i) const { return lproc_proc[i]; }

   /// Return true if I am master for group 'g'.
   bool IAmMaster(int g) const { return (groupmaster_lproc[g] == 0); }

   /** @brief Return the neighbor index of the group master for a given group.
       Neighbor 0 is the local processor. */
   int GetGroupMaster(int g) const { return groupmaster_lproc[g]; }

   /// Return the rank of the group master for group 'g'.
   int GetGroupMasterRank(int g) const
   { return lproc_proc[groupmaster_lproc[g]]; }

   /// Return the group number in the master for group 'g'.
   int GetGroupMasterGroup(int g) const { return group_mgroup[g]; }

   /// Get the number of processors in a group
   int GetGroupSize(int g) const { return group_lproc.RowSize(g); }

   /** @brief Return a pointer to a list of neighbors for a given group.
       Neighbor 0 is the local processor */
   const int *GetGroup(int g) const { return group_lproc.GetRow(g); }

   /// Save the data in a stream.
   void Save(std::ostream &out) const;

   /// Load the data from a stream.
   void Load(std::istream &in);

   /// Copy the internal data to the external 'copy'.
   void Copy(GroupTopology & copy) const;

   /// Swap the internal data with another @a GroupTopology object.
   void Swap(GroupTopology &other);

   virtual ~GroupTopology() {}
};

/** @brief Communicator performing operations within groups defined by a
    GroupTopology with arbitrary-size data associated with each group. */
class GroupCommunicator
{
public:
   /// Communication mode.
   enum Mode
   {
      byGroup,    ///< Communications are performed one group at a time.
      byNeighbor  /**< Communications are performed one neighbor at a time,
                       aggregating over groups. */
   };

protected:
   const GroupTopology &gtopo;
   Mode mode;
   Table group_ldof;
   Table group_ltdof; // only for groups for which this processor is master.
   int group_buf_size;
   mutable Array<char> group_buf;
   MPI_Request *requests;
   // MPI_Status  *statuses;
   // comm_lock: 0 - no lock, 1 - locked for Bcast, 2 - locked for Reduce
   mutable int comm_lock;
   mutable int num_requests;
   int *request_marker;
   int *buf_offsets; // size = max(number of groups, number of neighbors)
   Table nbr_send_groups, nbr_recv_groups; // nbr 0 = me

public:
   /// Construct a GroupCommunicator object.
   /** The object must be initialized before it can be used to perform any
       operations. To initialize the object, either
       - call Create() or
       - initialize the Table reference returned by GroupLDofTable() and then
         call Finalize().
   */
   GroupCommunicator(const GroupTopology &gt, Mode m = byNeighbor);

   /** @brief Initialize the communicator from a local-dof to group map.
       Finalize() is called internally. */
   void Create(const Array<int> &ldof_group);

   /** @brief Fill-in the returned Table reference to initialize the
       GroupCommunicator then call Finalize(). */
   Table &GroupLDofTable() { return group_ldof; }

   /// Read-only access to group-ldof Table.
   const Table &GroupLDofTable() const { return group_ldof; }

   /// Allocate internal buffers after the GroupLDofTable is defined
   void Finalize();

   /// Initialize the internal group_ltdof Table.
   /** This method must be called before performing operations that use local
       data layout 2, see CopyGroupToBuffer() for layout descriptions. */
   void SetLTDofTable(const Array<int> &ldof_ltdof);

   /// Get a reference to the associated GroupTopology object
   const GroupTopology &GetGroupTopology() { return gtopo; }

   /// Get a const reference to the associated GroupTopology object
   const GroupTopology &GetGroupTopology() const { return gtopo; }

   /// Dofs to be sent to communication neighbors
   void GetNeighborLTDofTable(Table &nbr_ltdof) const;

   /// Dofs to be received from communication neighbors
   void GetNeighborLDofTable(Table &nbr_ldof) const;

   /** @brief Data structure on which we define reduce operations.
       The data is associated with (and the operation is performed on) one
       group at a time. */
   template <class T> struct OpData
   {
      int nldofs, nb;
      const int *ldofs;
      T *ldata, *buf;
   };

   /** @brief Copy the entries corresponding to the group @a group from the
       local array @a ldata to the buffer @a buf. */
   /** The @a layout of the local array can be:
       - 0 - @a ldata is an array on all ldofs: copied indices:
         `{ J[j] : I[group] <= j < I[group+1] }` where `I,J=group_ldof.{I,J}`
       - 1 - @a ldata is an array on the shared ldofs: copied indices:
         `{ j : I[group] <= j < I[group+1] }` where `I,J=group_ldof.{I,J}`
       - 2 - @a ldata is an array on the true ldofs, ltdofs: copied indices:
         `{ J[j] : I[group] <= j < I[group+1] }` where `I,J=group_ltdof.{I,J}`.
       @returns The pointer @a buf plus the number of elements in the group. */
   template <class T>
   T *CopyGroupToBuffer(const T *ldata, T *buf, int group, int layout) const;

   /** @brief Copy the entries corresponding to the group @a group from the
       buffer @a buf to the local array @a ldata. */
   /** For a description of @a layout, see CopyGroupToBuffer().
       @returns The pointer @a buf plus the number of elements in the group. */
   template <class T>
   const T *CopyGroupFromBuffer(const T *buf, T *ldata, int group,
                                int layout) const;

   /** @brief Perform the reduction operation @a Op to the entries of group
       @a group using the values from the buffer @a buf and the values from the
       local array @a ldata, saving the result in the latter. */
   /** For a description of @a layout, see CopyGroupToBuffer().
       @returns The pointer @a buf plus the number of elements in the group. */
   template <class T>
   const T *ReduceGroupFromBuffer(const T *buf, T *ldata, int group,
                                  int layout, void (*Op)(OpData<T>)) const;

   /// Begin a broadcast within each group where the master is the root.
   /** For a description of @a layout, see CopyGroupToBuffer(). */
   template <class T> void BcastBegin(T *ldata, int layout) const;

   /** @brief Finalize a broadcast started with BcastBegin().

       The output data @a layout can be:
       - 0 - @a ldata is an array on all ldofs; the input layout should be
             either 0 or 2
       - 1 - @a ldata is the same array as given to BcastBegin(); the input
             layout should be 1.

       For more details about @a layout, see CopyGroupToBuffer(). */
   template <class T> void BcastEnd(T *ldata, int layout) const;

   /** @brief Broadcast within each group where the master is the root.

       The data @a layout can be either 0 or 1.

       For a description of @a layout, see CopyGroupToBuffer(). */
   template <class T> void Bcast(T *ldata, int layout) const
   {
      BcastBegin(ldata, layout);
      BcastEnd(ldata, layout);
   }

   /// Broadcast within each group where the master is the root.
   template <class T> void Bcast(T *ldata) const { Bcast<T>(ldata, 0); }
   /// Broadcast within each group where the master is the root.
   template <class T> void Bcast(Array<T> &ldata) const
   { Bcast<T>((T *)ldata); }

   /** @brief Begin reduction operation within each group where the master is
       the root. */
   /** The input data layout is an array on all ldofs, i.e. layout 0, see
       CopyGroupToBuffer().

       The reduce operation will be specified when calling ReduceEnd(). This
       method is instantiated for int and double. */
   template <class T> void ReduceBegin(const T *ldata) const;

   /** @brief Finalize reduction operation started with ReduceBegin().

       The output data @a layout can be either 0 or 2, see CopyGroupToBuffer().

       The reduce operation is given by the third argument (see below for list
       of the supported operations.) This method is instantiated for int and
       double.

       @note If the output data layout is 2, then the data from the @a ldata
       array passed to this call is used in the reduction operation, instead of
       the data from the @a ldata array passed to ReduceBegin(). Therefore, the
       data for master-groups has to be identical in both arrays.
   */
   template <class T> void ReduceEnd(T *ldata, int layout,
                                     void (*Op)(OpData<T>)) const;

   /** @brief Reduce within each group where the master is the root.

       The reduce operation is given by the second argument (see below for list
       of the supported operations.) */
   template <class T> void Reduce(T *ldata, void (*Op)(OpData<T>)) const
   {
      ReduceBegin(ldata);
      ReduceEnd(ldata, 0, Op);
   }

   /// Reduce within each group where the master is the root.
   template <class T> void Reduce(Array<T> &ldata, void (*Op)(OpData<T>)) const
   { Reduce<T>((T *)ldata, Op); }

   /// Reduce operation Sum, instantiated for int and double
   template <class T> static void Sum(OpData<T>);
   /// Reduce operation Min, instantiated for int and double
   template <class T> static void Min(OpData<T>);
   /// Reduce operation Max, instantiated for int and double
   template <class T> static void Max(OpData<T>);
   /// Reduce operation bitwise OR, instantiated for int only
   template <class T> static void BitOR(OpData<T>);

   /// Print information about the GroupCommunicator from all MPI ranks.
   void PrintInfo(std::ostream &out = mfem::out) const;

   /** @brief Destroy a GroupCommunicator object, deallocating internal data
       structures and buffers. */
   ~GroupCommunicator();
};

/// General MPI message tags used by MFEM
enum MessageTag
{
   DEREFINEMENT_MATRIX_CONSTRUCTION_DATA =
      291, /// ParFiniteElementSpace ParallelDerefinementMatrix and
   /// ParDerefineMatrixOp
};

enum VarMessageTag
{
   NEIGHBOR_ELEMENT_RANK_VM, ///< NeighborElementRankMessage
   NEIGHBOR_ORDER_VM,        ///< NeighborOrderMessage
   NEIGHBOR_DEREFINEMENT_VM, ///< NeighborDerefinementMessage
   NEIGHBOR_REFINEMENT_VM,   ///< NeighborRefinementMessage
   NEIGHBOR_PREFINEMENT_VM,  ///< NeighborPRefinementMessage
   NEIGHBOR_ROW_VM,          ///< NeighborRowMessage
   REBALANCE_VM,             ///< RebalanceMessage
   REBALANCE_DOF_VM,         ///< RebalanceDofMessage
};

/// \brief Variable-length MPI message containing unspecific binary data.
template<int Tag>
struct VarMessage
{
   std::string data;
   MPI_Request send_request;

   /** @brief Non-blocking send to processor 'rank'.
       Returns immediately. Completion (as tested by MPI_Wait/Test) does not
       mean the message was received -- it may be on its way or just buffered
       locally. */
   void Isend(int rank, MPI_Comm comm)
   {
      Encode(rank);
      MPI_Isend((void*) data.data(), static_cast<int>(data.length()), MPI_BYTE, rank,
                Tag, comm, &send_request);
   }

   /** @brief Non-blocking synchronous send to processor 'rank'.
       Returns immediately. Completion (MPI_Wait/Test) means that the message
       was received. */
   void Issend(int rank, MPI_Comm comm)
   {
      Encode(rank);
      MPI_Issend((void*) data.data(), static_cast<int>(data.length()), MPI_BYTE, rank,
                 Tag, comm, &send_request);
   }

   /// Helper to send all messages in a rank-to-message map container.
   template<typename MapT>
   static void IsendAll(MapT& rank_msg, MPI_Comm comm)
   {
      for (auto it = rank_msg.begin(); it != rank_msg.end(); ++it)
      {
         it->second.Isend(it->first, comm);
      }
   }

   /// Helper to wait for all messages in a map container to be sent.
   template<typename MapT>
   static void WaitAllSent(MapT& rank_msg)
   {
      for (auto it = rank_msg.begin(); it != rank_msg.end(); ++it)
      {
         MPI_Wait(&it->second.send_request, MPI_STATUS_IGNORE);
         it->second.Clear();
      }
   }

   /** @brief Return true if all messages in the map container were sent,
       otherwise return false, without waiting. */
   template<typename MapT>
   static bool TestAllSent(MapT& rank_msg)
   {
      for (auto it = rank_msg.begin(); it != rank_msg.end(); ++it)
      {
         VarMessage &msg = it->second;
         if (msg.send_request != MPI_REQUEST_NULL)
         {
            int sent;
            MPI_Test(&msg.send_request, &sent, MPI_STATUS_IGNORE);
            if (!sent) { return false; }
            msg.Clear();
         }
      }
      return true;
   }

   /** @brief Blocking probe for incoming message of this type from any rank.
       Returns the rank and message size. */
   static void Probe(int &rank, int &size, MPI_Comm comm)
   {
      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, Tag, comm, &status);
      rank = status.MPI_SOURCE;
      MPI_Get_count(&status, MPI_BYTE, &size);
   }

   /** @brief Non-blocking probe for incoming message of this type from any
       rank. If there is an incoming message, returns true and sets 'rank' and
       'size'. Otherwise returns false. */
   static bool IProbe(int &rank, int &size, MPI_Comm comm)
   {
      int flag;
      MPI_Status status;
      MPI_Iprobe(MPI_ANY_SOURCE, Tag, comm, &flag, &status);
      if (!flag) { return false; }

      rank = status.MPI_SOURCE;
      MPI_Get_count(&status, MPI_BYTE, &size);
      return true;
   }

   /// Post-probe receive from processor 'rank' of message size 'size'.
   void Recv(int rank, int size, MPI_Comm comm)
   {
      MFEM_ASSERT(size >= 0, "");
      data.resize(size);
      MPI_Status status;
      MPI_Recv((void*) data.data(), size, MPI_BYTE, rank, Tag, comm, &status);
#ifdef MFEM_DEBUG
      int count;
      MPI_Get_count(&status, MPI_BYTE, &count);
      MFEM_VERIFY(count == size, "");
#endif
      Decode(rank);
   }

   /// Like Recv(), but throw away the message.
   void RecvDrop(int rank, int size, MPI_Comm comm)
   {
      data.resize(size);
      MPI_Status status;
      MPI_Recv((void*) data.data(), size, MPI_BYTE, rank, Tag, comm, &status);
      data.resize(0); // don't decode
   }

   /// Helper to receive all messages in a rank-to-message map container.
   template<typename MapT>
   static void RecvAll(MapT& rank_msg, MPI_Comm comm)
   {
      int recv_left = static_cast<int>(rank_msg.size());
      while (recv_left > 0)
      {
         int rank, size;
         Probe(rank, size, comm);
         MFEM_ASSERT(rank_msg.find(rank) != rank_msg.end(), "Unexpected message"
                     " (tag " << Tag << ") from rank " << rank);
         // NOTE: no guard against receiving two messages from the same rank
         rank_msg[rank].Recv(rank, size, comm);
         --recv_left;
      }
   }

   VarMessage() : send_request(MPI_REQUEST_NULL) {}

   /// Clear the message and associated request.
   void Clear() { data.clear(); send_request = MPI_REQUEST_NULL; }

   virtual ~VarMessage()
   {
      MFEM_ASSERT(send_request == MPI_REQUEST_NULL,
                  "WaitAllSent was not called after Isend");
   }

   VarMessage(const VarMessage &other)
      : data(other.data), send_request(other.send_request)
   {
      MFEM_ASSERT(send_request == MPI_REQUEST_NULL,
                  "Cannot copy message with a pending send.");
   }

protected:
   virtual void Encode(int rank) = 0;
   virtual void Decode(int rank) = 0;
};


/// Helper struct to convert a C++ type to an MPI type
template <typename Type> struct MPITypeMap;

// Specializations of MPITypeMap; mpi_type initialized in communication.cpp:
template<> struct MPITypeMap<bool>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<char>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<unsigned char>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<short>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<unsigned short>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<int>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<unsigned int>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<long>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<unsigned long>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<long long>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<unsigned long long>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<double>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};
template<> struct MPITypeMap<float>
{
   static MFEM_EXPORT const MPI_Datatype mpi_type;
};

/** Reorder MPI ranks to follow the Z-curve within the physical machine topology
    (provided that functions to query physical node coordinates are available).
    Returns a new communicator with reordered ranks. */
MPI_Comm ReorderRanksZCurve(MPI_Comm comm);


} // namespace mfem

#endif

#endif
