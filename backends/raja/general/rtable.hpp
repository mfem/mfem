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
#ifndef LAGHOS_RAJA_TABLE
#define LAGHOS_RAJA_TABLE

namespace mfem
{

class RajaTable : public rmalloc<int>
{
private:
   int size = 0;
   int *I = NULL;
   int *J = NULL;
public:
   RajaTable(const Table&);
   inline int Size() {return size;}
   int RowSize(int i) const { return I[i+1]-I[i]; }
   const int *GetRow(int i) const { return J+I[i]; }
   int *GetRow(int i) { return J+I[i]; }
};

} // mfem

#endif // LAGHOS_RAJA_TABLE
