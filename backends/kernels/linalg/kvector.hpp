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

#ifndef MFEM_BACKENDS_KERNELS_RVECTOR_HPP
#define MFEM_BACKENDS_KERNELS_RVECTOR_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

class KernelsVector : public rmalloc<double>
{
private:
   size_t size = 0;
   double* data = NULL;
   bool own = true;
public:
   KernelsVector(): size(0),data(NULL),own(true) {}
   KernelsVector(const KernelsVector&);
   KernelsVector(const KernelsVector*);
   KernelsVector(const size_t);
   KernelsVector(const size_t,double);
   KernelsVector(const mfem::Vector& v);
   KernelsVector(array<double>& v);
   operator mfem::Vector();
   operator mfem::Vector() const;
   double* alloc(const size_t);
   inline double* ptr() const { return data;}
   inline double* GetData() const { return data;}
   inline operator double* () { return data; }
   inline operator const double* () const { return data; }
   void Print(std::ostream& = std::cout, int = 8) const;
   void SetSize(const size_t,const void* =NULL);
   inline size_t Size() const { return size; }
   inline size_t bytes() const { return size*sizeof(double); }
   double operator* (const KernelsVector& v) const;
   KernelsVector& operator = (const KernelsVector& v);
   KernelsVector& operator = (const mfem::Vector& v);
   KernelsVector& operator = (double value);
   KernelsVector& operator -= (const KernelsVector& v);
   KernelsVector& operator += (const KernelsVector& v);
   KernelsVector& operator += (const mfem::Vector& v);
   KernelsVector& operator *=(const double d);
   KernelsVector& Add(const double a, const KernelsVector& Va);
   void Neg();
   KernelsVector* GetRange(const size_t, const size_t) const;
   void SetSubVector(const array<int> &, const double, const int);
   double Min() const;
   ~KernelsVector();
};

// ***************************************************************************
void add(const KernelsVector&,const double,const KernelsVector&,KernelsVector&);
void add(const KernelsVector&,const KernelsVector&,KernelsVector&);
void add(const double,const KernelsVector&,const double,const KernelsVector&,
         KernelsVector&);
void subtract(const KernelsVector&,const KernelsVector&,KernelsVector&);

} // kernels

} // mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_RVECTOR_HPP
