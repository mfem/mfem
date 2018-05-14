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
   
namespace raja
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
   SharedPtr<const Engine> engine;
   mutable Mesh* mesh;
   mutable RajaFiniteElementSpace* trialFes;
   mutable RajaFiniteElementSpace* testFes;
   IntegratorVector integrators;
   mutable RajaVector localX, localY;
public:
   // **************************************************************************
   RajaBilinearForm(FiniteElementSpace*);
   RajaBilinearForm(RajaFiniteElementSpace*);
   ~RajaBilinearForm();
   // **************************************************************************
   const Engine &OccaEngine() const { return *engine; }
   mfem::Mesh& GetMesh() const { return *mesh; }
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
   void FormLinearSystem(const mfem::Array<int>& constraintList,
                         RajaVector& x, RajaVector& b,
                         RajaOperator*& Aout,
                         RajaVector& X, RajaVector& B,
                         int copy_interior = 0);
   void FormOperator(const mfem::Array<int>& constraintList, RajaOperator*& Aout);
   void InitRHS(const mfem::Array<int>& constraintList,
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
   RajaConstrainedOperator(RajaOperator*, const mfem::Array<int>&, bool = false);
   void Setup(RajaOperator*, const mfem::Array<int>&, bool = false);
   void EliminateRHS(const RajaVector&, RajaVector&) const;
   virtual void Mult(const RajaVector&, RajaVector&) const;
   virtual ~RajaConstrainedOperator() {}
};

// *****************************************************************************
// *****************************************************************************
class BilinearForm : public mfem::PBilinearForm
{
protected:
   //
   // Inherited fields
   //
   RajaBilinearForm *rbform;

   // Called from Assemble() if rbform is NULL to initialize rbform.
   void InitRajaBilinearForm();

public:
   /// TODO: doxygen
   BilinearForm(const Engine &e, mfem::BilinearForm &bf)
      : mfem::PBilinearForm(e, bf), rbform(NULL) {push();pop(); }

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

   
} // raja
   
} // mfem

#endif // LAGHOS_RAJA_BILINEARFORM

