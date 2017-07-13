// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_OCCA
#  ifndef MFEM_OCCA_GRIDFUNCTION
#  define MFEM_OCCA_GRIDFUNCTION

#include <map>

#include "occa.hpp"

#include "../linalg/ovector.hpp"
#include "ofespace.hpp"

namespace mfem {
  class OccaIntegrator;

  extern std::map<std::string, occa::kernel> gridFunctionKernels;

  occa::kernel GetGridFunctionKernel(OccaIntegrator &integ);

  class OccaGridFunction : public OccaVector {
  protected:
    OccaFiniteElementSpace *ofespace;
    long sequence;

    occa::kernel gridFuncToQuad[3];

  public:
    OccaGridFunction();

    OccaGridFunction(OccaFiniteElementSpace *ofespace_);

    OccaGridFunction(occa::device device_,
                     OccaFiniteElementSpace *ofespace_);

    OccaGridFunction(OccaFiniteElementSpace *ofespace_,
                     OccaVectorRef ref);

    OccaGridFunction(const OccaGridFunction &gf);

    OccaGridFunction& operator = (double value);
    OccaGridFunction& operator = (const OccaVector &v);
    OccaGridFunction& operator = (const OccaVectorRef &v);
    OccaGridFunction& operator = (const OccaGridFunction &gf);

    void SetGridFunction(GridFunction &gf);

    void GetTrueDofs(OccaVector &v) const;

    void SetFromTrueDofs(const OccaVector &v);

    FiniteElementSpace* GetFESpace();
    const FiniteElementSpace* GetFESpace() const;

    void ToQuad(OccaIntegrator &integ,
                OccaVector &quadValues);
  };
};

#  endif
#endif
