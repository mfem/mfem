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
#ifndef LAGHOS_RAJA_GRIDFUNC
#define LAGHOS_RAJA_GRIDFUNC

namespace mfem
{

class RajaGridFunction : public RajaVector
{
public:
   const RajaFiniteElementSpace& fes;
public:

   RajaGridFunction(const RajaFiniteElementSpace& f):
      RajaVector(f.GetVSize()),fes(f) {}

   RajaGridFunction(const RajaFiniteElementSpace& f,const RajaVector* v):
      RajaVector(v), fes(f) {}

   void ToQuad(const IntegrationRule&,RajaVector&);

   RajaGridFunction& operator=(const RajaVector& v)
   {
      RajaVector::operator=(v);
      return *this;
   }
   RajaGridFunction& operator=(const Vector& v)
   {
      RajaVector::operator=(v);
      return *this;
   }
};

} // mfem

#endif // LAGHOS_RAJA_GRIDFUNC
