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
#  ifndef MFEM_OCCA_BILINEARFORM
#  define MFEM_OCCA_BILINEARFORM

#include <vector>
#include <map>
#include "stdint.h"

#include "../linalg/operator.hpp"
#include "bilinearform.hpp"
#include "ofespace.hpp"

#include "occa.hpp"

namespace mfem {
  enum OccaIntegratorType {
    DomainIntegrator       = 0,
    BoundaryIntegrator     = 1,
    InteriorFaceIntegrator = 2,
    BoundaryFaceIntegrator = 3,
  };

  //---[ Bilinear Form ]----------------
  class OccaIntegrator;

  /** Class for bilinear form - "Matrix" with associated FE space and
      BLFIntegrators. */
  class OccaBilinearForm : public Operator {
    friend class OccaIntegrator;

  protected:
    typedef std::vector<OccaIntegrator*> IntegratorVector;

    occa::device device;

    // State information
    mutable Mesh *mesh;

    mutable OccaFiniteElementSpace *otrialFESpace;
    mutable FiniteElementSpace *trialFESpace;

    mutable OccaFiniteElementSpace *otestFESpace;
    mutable FiniteElementSpace *testFESpace;

    IntegratorVector integrators;

    // Device data
    occa::properties baseKernelProps;

    // The input and output vectors are mapped to local nodes for efficient operations
    // The size is: (number of elements) * (nodes in element)
    mutable OccaVector localX, localY;

  public:
    OccaBilinearForm(OccaFiniteElementSpace *ofespace_);
    OccaBilinearForm(occa::device device_,
                     OccaFiniteElementSpace *ofespace_);

    OccaBilinearForm(OccaFiniteElementSpace *otrialFESpace_,
                     OccaFiniteElementSpace *otestFESpace_);
    OccaBilinearForm(occa::device device_,
                     OccaFiniteElementSpace *otrialFESpace_,
                     OccaFiniteElementSpace *otestFESpace_);

    void Init(occa::device device_,
              OccaFiniteElementSpace *otrialFESpace_,
              OccaFiniteElementSpace *otestFESpace_);

    occa::device GetDevice();

    // Useful mesh Information
    int BaseGeom() const;
    int GetDim() const;
    int64_t GetNE() const;

    Mesh& GetMesh() const;

    OccaFiniteElementSpace& GetTrialOccaFESpace() const;
    OccaFiniteElementSpace& GetTestOccaFESpace() const;

    FiniteElementSpace& GetTrialFESpace() const;
    FiniteElementSpace& GetTestFESpace() const;

    // Useful FE information
    int64_t GetTrialNDofs() const;
    int64_t GetTestNDofs() const;

    int64_t GetTrialVDim() const;
    int64_t GetTestVDim() const;

    const FiniteElement& GetTrialFE(const int i) const;
    const FiniteElement& GetTestFE(const int i) const;

    // Adds new Domain Integrator.
    void AddDomainIntegrator(OccaIntegrator *integrator,
                             const occa::properties &props = occa::properties());

    // Adds new Boundary Integrator.
    void AddBoundaryIntegrator(OccaIntegrator *integrator,
                               const occa::properties &props = occa::properties());

    // Adds new interior Face Integrator.
    void AddInteriorFaceIntegrator(OccaIntegrator *integrator,
                                   const occa::properties &props = occa::properties());

    // Adds new boundary Face Integrator.
    void AddBoundaryFaceIntegrator(OccaIntegrator *integrator,
                                   const occa::properties &props = occa::properties());

    // Adds Integrator based on OccaIntegratorType
    void AddIntegrator(OccaIntegrator *integrator,
                       const occa::properties &props,
                       const OccaIntegratorType itype);

    virtual const Operator *GetTrialProlongation() const;
    virtual const Operator *GetTestProlongation() const;

    virtual const Operator *GetTrialRestriction() const;
    virtual const Operator *GetTestRestriction() const;

    // Assembles the form i.e. sums over all domain/bdr integrators.
    virtual void Assemble();

    void FormLinearSystem(const Array<int> &constraintList,
                          OccaVector &x, OccaVector &b,
                          Operator *&Aout,
                          OccaVector &X, OccaVector &B,
                          int copy_interior = 0);

    void FormOperator(const Array<int> &constraintList,
                      Operator *&Aout);

    void InitRHS(const Array<int> &constraintList,
                 OccaVector &x, OccaVector &b,
                 Operator *Aout,
                 OccaVector &X, OccaVector &B,
                 int copy_interior = 0);

    // Matrix vector multiplication.
    virtual void Mult(const OccaVector &x, OccaVector &y) const;

    // Matrix vector multiplication.
    virtual void MultTranspose(const OccaVector &x, OccaVector &y) const;

    void RecoverFEMSolution(const OccaVector &X, const OccaVector &b, OccaVector &x);

    // Destroys bilinear form.
    ~OccaBilinearForm();
  };
  //====================================

  //---[ Constrained Operator ]---------
  class OccaConstrainedOperator : public Operator {
  protected:
    occa::device device;

    Operator *A;                   //< The unconstrained Operator.
    bool own_A;                    //< Ownership flag for A.
    occa::array<int> constraintList;  //< List of constrained indices/dofs.
    int constraintIndices;
    mutable OccaVector z, w;       //< Auxiliary vectors.

    static occa::kernelBuilder mapDofBuilder, clearDofBuilder;

  public:
    /** @brief Constructor from a general Operator and a list of essential
        indices/dofs.

        Specify the unconstrained operator @a *A and a @a list of indices to
        constrain, i.e. each entry @a list[i] represents an essential-dof. If the
        ownership flag @a own_A is true, the operator @a *A will be destroyed
        when this object is destroyed. */
    OccaConstrainedOperator(Operator *A_,
                            const Array<int> &constraintList_,
                            bool own_A_ = false);

    OccaConstrainedOperator(occa::device device_,
                            Operator *A_,
                            const Array<int> &constraintList_,
                            bool own_A_ = false);

    void Setup(occa::device device_,
               Operator *A_,
               const Array<int> &constraintList_,
               bool own_A_ = false);

    /** @brief Eliminate "essential boundary condition" values specified in @a x
        from the given right-hand side @a b.

        Performs the following steps:

        z = A((0,x_b));  b_i -= z_i;  b_b = x_b;

        where the "_b" subscripts denote the essential (boundary) indices/dofs of
        the vectors, and "_i" -- the rest of the entries. */
    void EliminateRHS(const OccaVector &x, OccaVector &b) const;

    /** @brief Constrained operator action.

        Performs the following steps:

        z = A((x_i,0));  y_i = z_i;  y_b = x_b;

        where the "_b" subscripts denote the essential (boundary) indices/dofs of
        the vectors, and "_i" -- the rest of the entries. */
    virtual void Mult(const OccaVector &x, OccaVector &y) const;

    // Destructor: destroys the unconstrained Operator @a A if @a own_A is true.
    virtual ~OccaConstrainedOperator();
  };
  //====================================
}

#  endif
#endif
