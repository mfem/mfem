// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#ifndef LAGHOS_RAJA_COMM_D
#define LAGHOS_RAJA_COMM_D

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

// ***************************************************************************
// * First communicator, buf goes on the device
// ***************************************************************************
class RajaCommD : public GroupCommunicator, public rmemcpy
{
private:
   RajaTable d_group_ldof;
   RajaTable d_group_ltdof;
   void *d_group_buf;
   int comm_lock; // 0 - no lock, 1 - locked for Bcast, 2 - locked for Reduce
   int num_requests;
public:
   RajaCommD(ParFiniteElementSpace&);
   ~RajaCommD();

   template <class T> T *d_CopyGroupToBuffer(const T*,T*,int,int) const;
   template <class T> const T *d_CopyGroupFromBuffer(const T*, T*,int, int) const;
   template <class T> const T *d_ReduceGroupFromBuffer(const T*,T*,int,int,
                                                       void (*)(OpData<T>)) const;

   template <class T> void d_BcastBegin(T*,int);
   template <class T> void d_BcastEnd(T*, int);

   template <class T> void d_ReduceBegin(const T*);
   template <class T> void d_ReduceEnd(T*,int,void (*)(OpData<T>));
};


} // mfem

#endif // LAGHOS_RAJA_COMM_D
