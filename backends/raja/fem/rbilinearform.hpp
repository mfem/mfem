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
class RajaBilinearForm : public Operator
{
   friend class RajaIntegrator;
protected:
   typedef std::vector<RajaIntegrator*> IntegratorVector;
   SharedPtr<const Engine> engine;
   mutable mfem::Mesh* mesh;
   mutable FiniteElementSpace *rtrialFESpace;
   mutable mfem::FiniteElementSpace *trialFESpace;
   mutable FiniteElementSpace *rtestFESpace;
   mutable mfem::FiniteElementSpace *testFESpace;
   IntegratorVector integrators;
   mutable Vector localX, localY;
public:
   // **************************************************************************
   RajaBilinearForm(FiniteElementSpace*);
   ~RajaBilinearForm();
   // **************************************************************************
   const Engine &OccaEngine() const { return *engine; }
   mfem::Mesh& GetMesh() const { return *mesh; }
   mfem::FiniteElementSpace& GetTrialFESpace() const { return *trialFESpace;}
   mfem::FiniteElementSpace& GetTestFESpace() const { return *testFESpace;}
   // *************************************************************************
   void AddDomainIntegrator(RajaIntegrator*);
   void AddBoundaryIntegrator(RajaIntegrator*);
   void AddInteriorFaceIntegrator(RajaIntegrator*);
   void AddBoundaryFaceIntegrator(RajaIntegrator*);
   void AddIntegrator(RajaIntegrator*, const RajaIntegratorType);
   // **************************************************************************
   virtual const mfem::Operator *GetTrialProlongation() const;
   virtual const mfem::Operator *GetTestProlongation() const;
   virtual const mfem::Operator *GetTrialRestriction() const;
   virtual const mfem::Operator *GetTestRestriction() const;
   // *************************************************************************
   virtual void Assemble();
   void FormLinearSystem(const mfem::Array<int>& constraintList,
                         mfem::Vector& x, mfem::Vector& b,
                         mfem::Operator*& Aout,
                         mfem::Vector& X, mfem::Vector& B,
                         int copy_interior = 0);
   void FormOperator(const mfem::Array<int>& constraintList, mfem::Operator*& Aout);
   void InitRHS(const mfem::Array<int>& constraintList,
                const mfem::Vector& x, const mfem::Vector& b,
                mfem::Operator* Aout,
                mfem::Vector& X, mfem::Vector& B,
                int copy_interior = 0);
   virtual void Mult_(const Vector& x, Vector& y) const;
   virtual void MultTranspose_(const Vector& x, Vector& y) const;
   void RecoverFEMSolution(const mfem::Vector&, const mfem::Vector&, mfem::Vector&);
};


// ***************************************************************************
// * Constrained Operator
// ***************************************************************************
class RajaConstrainedOperator : public Operator
{
protected:
   mfem::Operator *A;
   bool own_A;
   RajaArray<int> constraintList;
   int constraintIndices;
   mutable mfem::Vector z, w;
public:
   RajaConstrainedOperator(mfem::Operator*, const mfem::Array<int>&, bool = false);
   void Setup(mfem::Operator*, const mfem::Array<int>&, bool = false);
   void EliminateRHS(const Vector&, Vector&) const;
   virtual void Mult_(const Vector&, Vector&) const;
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

