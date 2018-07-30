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

#ifndef MFEM_BACKENDS_KERNELS_KVECTOR_HPP
#define MFEM_BACKENDS_KERNELS_KVECTOR_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

class kvector : public kmalloc<double>
{
private:
   size_t size = 0;
   double* data = NULL;
   bool own = true;
public:
   kvector(): size(0),data(NULL),own(true) {}
   kvector(const kvector&);
   kvector(const kvector*);
   kvector(const size_t);
   kvector(const size_t,double);
   kvector(const mfem::Vector& v);
   kvector(array<double>& v);
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
   double operator* (const kvector& v) const;
   kvector& operator = (const kvector& v);
   kvector& operator = (const mfem::Vector& v);
   kvector& operator = (double value);
   kvector& operator -= (const kvector& v);
   kvector& operator += (const kvector& v);
   kvector& operator += (const mfem::Vector& v);
   kvector& operator *=(const double d);
   kvector& Add(const double a, const kvector& Va);
   void Neg();
   kvector* GetRange(const size_t, const size_t) const;
   void SetSubVector(const array<int> &, const double, const int);
   double Min() const;
   ~kvector();
};

// ***************************************************************************
void add(const kvector&,const double,const kvector&,kvector&);
void add(const kvector&,const kvector&,kvector&);
void add(const double,const kvector&,const double,const kvector&,
         kvector&);
void subtract(const kvector&,const kvector&,kvector&);

} // kernels

} // mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_KVECTOR_HPP
