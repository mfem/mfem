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
#ifndef LAGHOS_RAJA_BILINEARFORM
#define LAGHOS_RAJA_BILINEARFORM

namespace mfem
{

// ***************************************************************************
// * RajaIntegratorType
// ***************************************************************************
enum RajaIntegratorType
{
   DomainIntegrator       = 0,
   BoundaryIntegrator     = 1,
   InteriorFaceIntegrator = 2,
   BoundaryFaceIntegrator = 3,
};

class RajaIntegrator;

// ***************************************************************************
// * RajaBilinearForm
// ***************************************************************************
class RajaBilinearForm : public RajaOperator
{
   friend class RajaIntegrator;
protected:
   typedef std::vector<RajaIntegrator*> IntegratorVector;
   mutable Mesh* mesh;
   mutable RajaFiniteElementSpace* trialFes;
   mutable RajaFiniteElementSpace* testFes;
   IntegratorVector integrators;
   mutable RajaVector localX, localY;
public:
   RajaBilinearForm(RajaFiniteElementSpace*);
   ~RajaBilinearForm();
   Mesh& GetMesh() const { return *mesh; }
   RajaFiniteElementSpace& GetTrialFESpace() const { return *trialFes;}
   RajaFiniteElementSpace& GetTestFESpace() const { return *testFes;}
   // *************************************************************************
   void AddDomainIntegrator(RajaIntegrator*);
   void AddBoundaryIntegrator(RajaIntegrator*);
   void AddInteriorFaceIntegrator(RajaIntegrator*);
   void AddBoundaryFaceIntegrator(RajaIntegrator*);
   void AddIntegrator(RajaIntegrator*, const RajaIntegratorType);
   // *************************************************************************
   virtual void Assemble();
   void FormLinearSystem(const Array<int>& constraintList,
                         RajaVector& x, RajaVector& b,
                         RajaOperator*& Aout,
                         RajaVector& X, RajaVector& B,
                         int copy_interior = 0);
   void FormOperator(const Array<int>& constraintList, RajaOperator*& Aout);
   void InitRHS(const Array<int>& constraintList,
                const RajaVector& x, const RajaVector& b,
                RajaOperator* Aout,
                RajaVector& X, RajaVector& B,
                int copy_interior = 0);
   virtual void Mult(const RajaVector& x, RajaVector& y) const;
   virtual void MultTranspose(const RajaVector& x, RajaVector& y) const;
   void RecoverFEMSolution(const RajaVector&, const RajaVector&, RajaVector&);
};


// ***************************************************************************
// * Constrained Operator
// ***************************************************************************
class RajaConstrainedOperator : public RajaOperator
{
protected:
   RajaOperator *A;
   bool own_A;
   RajaArray<int> constraintList;
   int constraintIndices;
   mutable RajaVector z, w;
public:
   RajaConstrainedOperator(RajaOperator*, const Array<int>&, bool = false);
   void Setup(RajaOperator*, const Array<int>&, bool = false);
   void EliminateRHS(const RajaVector&, RajaVector&) const;
   virtual void Mult(const RajaVector&, RajaVector&) const;
   virtual ~RajaConstrainedOperator() {}
};

} // mfem

#endif // LAGHOS_RAJA_BILINEARFORM
