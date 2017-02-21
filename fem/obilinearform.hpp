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
#  ifndef MFEM_OCCABILINEARFORM
#  define MFEM_OCCABILINEARFORM

#include <vector>
#include <map>
#include "stdint.h"

#include "../linalg/operator.hpp"
#include "bilinearform.hpp"

#include "occa.hpp"

namespace mfem {
  /** Class for bilinear form - "Matrix" with associated FE space and
      BLFIntegrators. */
  class OccaBilinearForm : public Operator
  {
  public:
    enum IntegratorType {
      DomainIntegrator       = 0,
      BoundaryIntegrator     = 1,
      InteriorFaceIntegrator = 2,
      BoundaryFaceIntegrator = 3,
    };

  protected:
    typedef std::vector<occa::kernel> IntegratorVector;
    typedef occa::kernel (OccaBilinearForm::*KernelBuilder)(
                                                            BilinearFormIntegrator &bfi,
                                                            const occa::properties &props,
                                                            const IntegratorType itype);
    typedef std::map<std::string,KernelBuilder> KernelBuilderCollection;

    /// State information
    FiniteElementSpace *fes;
    Mesh *mesh;

    /// Group of integrators used to build kernels
    static KernelBuilderCollection kernelBuilderCollection;
    IntegratorVector integrators;

    // Device data
    occa::device device;
    occa::properties baseKernelProps;

    // Partially assembled data
    occa::memory geometricFactors;

  public:
    OccaBilinearForm(FiniteElementSpace *f);
    OccaBilinearForm(occa::device dev, FiniteElementSpace *f);

    // Setup the kernel builder collection
    void SetupKernelBuilderCollection();

    /// Setup kernel properties
    void SetupBaseKernelProps();

    /// Useful FE information
    int GetDim();
    int64_t GetNE();
    int64_t GetNDofs();
    int64_t GetVSize();

    /// Adds new Domain Integrator.
    void AddDomainIntegrator(BilinearFormIntegrator *bfi,
                             const occa::properties &props = occa::properties());

    /// Adds new Boundary Integrator.
    void AddBoundaryIntegrator(BilinearFormIntegrator *bfi,
                               const occa::properties &props = occa::properties());

    /// Adds new interior Face Integrator.
    void AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi,
                                   const occa::properties &props = occa::properties());

    /// Adds new boundary Face Integrator.
    void AddBoundaryFaceIntegrator(BilinearFormIntegrator *bfi,
                                   const occa::properties &props = occa::properties());

    /// Adds Integrator based on IntegratorType
    void AddIntegrator(BilinearFormIntegrator &bfi,
                       const occa::properties &props,
                       const IntegratorType itype);

    /// Builds the DiffusionIntegrator
    occa::kernel BuildDiffusionIntegrator(BilinearFormIntegrator &bfi,
                                          const occa::properties &props,
                                          const IntegratorType itype);

    /// Assembles the form i.e. sums over all domain/bdr integrators.
    virtual void Assemble();

    /// Matrix vector multiplication.
    virtual void Mult(const OccaVector &x, OccaVector &y) const;

    /// Matrix vector multiplication.
    virtual void MultTranspose(const OccaVector &x, OccaVector &y) const;

    void FormLinearSystem(const Array<int> &ess_tdof_list,
                                  OccaVector &x, OccaVector &b,
                                  Operator* &Aout, OccaVector &X, OccaVector &B,
                                  int copy_interior = 0);

    virtual void ImposeBoundaryConditions(const Array<int> &ess_tdof_list,
                                          Operator *rap,
                                          Operator* &Aout, OccaVector &X, OccaVector &B);

    /** @brief Eliminate "essential boundary condition" values specified in @a x
        from the given right-hand side @a b.

        Performs the following steps:

        z = A((0,x_b));  b_i -= z_i;  b_b = x_b;

        where the "_b" subscripts denote the essential (boundary) indices/dofs of
        the vectors, and "_i" -- the rest of the entries. */
    virtual void EliminateRHS(const OccaVector &x, OccaVector &b) const;

    /// Destroys bilinear form.
    ~OccaBilinearForm();
  };
}

#  endif
#endif
