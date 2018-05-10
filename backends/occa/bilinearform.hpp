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

#ifndef MFEM_BACKENDS_OCCA_BILINEAR_FORM_HPP
#define MFEM_BACKENDS_OCCA_BILINEAR_FORM_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "fespace.hpp"

namespace mfem
{

namespace occa
{

enum OccaIntegratorType
{
   DomainIntegrator       = 0,
   BoundaryIntegrator     = 1,
   InteriorFaceIntegrator = 2,
   BoundaryFaceIntegrator = 3
};

class OccaIntegrator;


/** Class for bilinear form - "Matrix" with associated FE space and
    BLFIntegrators. */
class OccaBilinearForm : public Operator
{
   friend class OccaIntegrator;

protected:
   typedef std::vector<OccaIntegrator*> IntegratorVector;

   SharedPtr<const Engine> engine;

   // State information
   mutable mfem::Mesh *mesh;

   mutable FiniteElementSpace *otrialFESpace;
   mutable mfem::FiniteElementSpace *trialFESpace;

   mutable FiniteElementSpace *otestFESpace;
   mutable mfem::FiniteElementSpace *testFESpace;

   IntegratorVector integrators;

   // Device data
   ::occa::properties baseKernelProps;

   // The input and output vectors are mapped to local nodes for efficient
   // operations. In other words, they are E-vectors.
   // The size is: (number of elements) * (nodes in element) * (vector dim)
   mutable Vector localX, localY;

public:
   OccaBilinearForm(FiniteElementSpace *ofespace_);

   OccaBilinearForm(FiniteElementSpace *otrialFESpace_,
                    FiniteElementSpace *otestFESpace_);

   void Init(const Engine &e,
             FiniteElementSpace *otrialFESpace_,
             FiniteElementSpace *otestFESpace_);

   const Engine &OccaEngine() const { return *engine; }

   ::occa::device GetDevice(int idx = 0) const
   { return engine->GetDevice(idx); }

   // Useful mesh Information
   int BaseGeom() const;
   int GetDim() const;
   int64_t GetNE() const;

   mfem::Mesh& GetMesh() const;

   FiniteElementSpace& GetTrialOccaFESpace() const;
   FiniteElementSpace& GetTestOccaFESpace() const;

   mfem::FiniteElementSpace& GetTrialFESpace() const;
   mfem::FiniteElementSpace& GetTestFESpace() const;

   // Useful FE information
   int64_t GetTrialNDofs() const;
   int64_t GetTestNDofs() const;

   int64_t GetTrialVDim() const;
   int64_t GetTestVDim() const;

   const mfem::FiniteElement& GetTrialFE(const int i) const;
   const mfem::FiniteElement& GetTestFE(const int i) const;

   // Adds new Domain Integrator.
   void AddDomainIntegrator(OccaIntegrator *integrator,
                            const ::occa::properties &props =
                               ::occa::properties());

   // Adds new Boundary Integrator.
   void AddBoundaryIntegrator(OccaIntegrator *integrator,
                              const ::occa::properties &props =
                                 ::occa::properties());

   // Adds new interior Face Integrator.
   void AddInteriorFaceIntegrator(OccaIntegrator *integrator,
                                  const ::occa::properties &props =
                                     ::occa::properties());

   // Adds new boundary Face Integrator.
   void AddBoundaryFaceIntegrator(OccaIntegrator *integrator,
                                  const ::occa::properties &props =
                                     ::occa::properties());

   // Adds Integrator based on OccaIntegratorType
   void AddIntegrator(OccaIntegrator *integrator,
                      const ::occa::properties &props,
                      const OccaIntegratorType itype);

   virtual const mfem::Operator *GetTrialProlongation() const;
   virtual const mfem::Operator *GetTestProlongation() const;

   virtual const mfem::Operator *GetTrialRestriction() const;
   virtual const mfem::Operator *GetTestRestriction() const;

   // Assembles the form i.e. sums over all domain/bdr integrators.
   virtual void Assemble();

   void FormLinearSystem(const mfem::Array<int> &constraintList,
                         mfem::Vector &x, mfem::Vector &b,
                         mfem::Operator *&Aout,
                         mfem::Vector &X, mfem::Vector &B,
                         int copy_interior = 0);

   void FormOperator(const mfem::Array<int> &constraintList,
                     mfem::Operator *&Aout);

   void InitRHS(const mfem::Array<int> &constraintList,
                mfem::Vector &x, mfem::Vector &b,
                mfem::Operator *Aout,
                mfem::Vector &X, mfem::Vector &B,
                int copy_interior = 0);

   // overrides
   virtual void Mult_(const Vector &x, Vector &y) const;
   virtual void MultTranspose_(const Vector &x, Vector &y) const;

   void OccaRecoverFEMSolution(const mfem::Vector &X, const mfem::Vector &b,
                               mfem::Vector &x);

   // Destroys bilinear form.
   ~OccaBilinearForm();
};


/// TODO: doxygen
class BilinearForm : public mfem::PBilinearForm
{
protected:
   //
   // Inherited fields
   //
   // SharedPtr<const mfem::Engine> engine;
   // mfem::BilinearForm *bform;
   OccaBilinearForm *obform;

   // Called from Assemble() if obform is NULL to initialize obform.
   void InitOccaBilinearForm();

public:
   /// TODO: doxygen
   BilinearForm(const Engine &e, mfem::BilinearForm &bf)
      : mfem::PBilinearForm(e, bf), obform(NULL) { }

   /// Virtual destructor
   virtual ~BilinearForm() { }

   /// Assemble the PBilinearForm.
   /** This method is called from the method mfem::BilinearForm::Assemble() of
       the associated mfem::BilinearForm, #bform.
       @returns True, if the host assembly should NOT be performed. */
   virtual bool Assemble();

   virtual void FormSystemMatrix(const mfem::Array<int> &ess_tdof_list,
                                 mfem::OperatorHandle &A);

   virtual void FormLinearSystem(const mfem::Array<int> &ess_tdof_list,
                                 mfem::Vector &x, mfem::Vector &b,
                                 mfem::OperatorHandle &A,
                                 mfem::Vector &X, mfem::Vector &B,
                                 int copy_interior);

   virtual void RecoverFEMSolution(const mfem::Vector &X, const mfem::Vector &b,
                                   mfem::Vector &x);
};

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#endif // MFEM_BACKENDS_OCCA_BILINEAR_FORM_HPP
