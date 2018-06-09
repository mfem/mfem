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

namespace raja
{

// *****************************************************************************
class RajaParGridFunction : public raja::RajaVector
{
public:
   const RajaParFiniteElementSpace& pfes;
public:

   RajaParGridFunction(const raja::RajaParFiniteElementSpace& f):
      RajaVector(f.GetFESpace()->GetVSize()),pfes(f) {}

   RajaParGridFunction(const RajaParFiniteElementSpace& f,const raja::RajaVector* v):
      RajaVector(v), pfes(f) {}

   void ToQuad(const IntegrationRule&,raja::RajaVector&);

   RajaParGridFunction& operator=(const raja::RajaVector& v)
   {
      RajaVector::operator=(v);
      return *this;
   }
   RajaParGridFunction& operator=(const mfem::Vector& v)
   {
      RajaVector::operator=(v);
      return *this;
   }
};
   
} // raja
   
} // mfem

#endif // LAGHOS_RAJA_GRIDFUNC
