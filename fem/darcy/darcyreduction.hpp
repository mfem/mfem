// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DARCYREDUCTION
#define MFEM_DARCYREDUCTION

#include "../../config/config.hpp"
#include "../bilinearform.hpp"
#include "../nonlinearform.hpp"

#define MFEM_DARCY_REDUCTION_ELIM_BCS

namespace mfem
{

class DarcyReduction : public Operator
{
protected:
   FiniteElementSpace &fes_u, &fes_p;
   NonlinearFormIntegrator *m_nlfi_u{}, *m_nlfi_p{};
   bool own_m_nlfi_u{}, own_m_nlfi_p{};

   Array<int> Af_offsets, Af_f_offsets;
   Array<real_t> Af_data;

   Array<int> Bf_offsets, Bf_face_offsets;
   Array<real_t> Bf_data, Bf_face_data;

   Array<int> D_offsets, D_f_offsets, D_face_offsets;
   Array<real_t> D_data, D_face_data;

   bool bsym{};

   std::unique_ptr<SparseMatrix> S;

#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes_u, *pfes_p;
   OperatorHandle pS;
#endif

   void InitA();
   void InitBD();
   void InitBFaces();
   void InitDFaces();
   virtual void ComputeS() = 0;
#ifdef MFEM_USE_MPI
   virtual void CountBSharedFaces(Array<int> &face_offs) const = 0;
   virtual void CountDSharedFaces(Array<int> &face_offs) const = 0;

   static HypreParMatrix *ConstructParMatrix(SparseMatrix *spmat,
                                             ParFiniteElementSpace *pfes_tr, ParFiniteElementSpace *pfes_te = NULL);
#endif

public:
   /// Constructor
   DarcyReduction(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
                  bool bsymmetrize = true);

   /// Destructor
   virtual ~DarcyReduction();

   void SetFluxMassNonlinearIntegrator(NonlinearFormIntegrator *flux_integ,
                                       bool own = true);

   void SetPotMassNonlinearIntegrator(NonlinearFormIntegrator *pot_integ,
                                      bool own = true);

   NonlinearFormIntegrator* GetFluxMassNonlinearIntegrator() const { return m_nlfi_p; }
   NonlinearFormIntegrator* GetPotMassNonlinearIntegrator() const { return m_nlfi_p; }

   /// Prepare the DarcyReduction object for assembly.
   virtual void Init(const Array<int> &ess_flux_tdof_list);

   virtual void AssembleFluxMassMatrix(int el, const DenseMatrix &A);

   virtual void AssemblePotMassMatrix(int el, const DenseMatrix &D);

   virtual void AssembleDivMatrix(int el, const DenseMatrix &B);

   virtual void AssembleDivFaceMatrix(int face, const DenseMatrix &);

   virtual void AssemblePotFaceMatrix(int face, const DenseMatrix &);

#ifdef MFEM_USE_MPI
   virtual void AssembleDivSharedFaceMatrix(int sface, const DenseMatrix &) = 0;

   virtual void AssemblePotSharedFaceMatrix(int sface, const DenseMatrix &) = 0;
#endif

   /** @brief Use the stored eliminated part of the matrix to modify the r.h.s.
       @a b; @a vdofs_flux is a list of DOFs (non-directional, i.e. >= 0). */
   virtual void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                    const BlockVector &x, BlockVector &b)
   { MFEM_ABORT("Not implemented"); }

   /// Operator application: `y=A(x)`.
   void Mult(const Vector &x, Vector &y) const override;

   /// Finalize the construction of the reduced matrix.
   virtual void Finalize();

   /// Return the serial reduced matrix.
   SparseMatrix &GetMatrix() { return *S; }

#ifdef MFEM_USE_MPI
   /// Return the parallel reduced matrix.
   HypreParMatrix &GetParallelMatrix() { return *pS.Is<HypreParMatrix>(); }

   /** @brief Return the parallel reduced matrix in the format specified by
       SetOperatorType(). */
   void GetParallelMatrix(OperatorHandle &H_h) const { H_h = pS; }

   /// Set the operator type id for the parallel reduced matrix/operator.
   void SetOperatorType(Operator::Type tid) { pS.SetType(tid); }

   /** @brief Use the stored eliminated part of the matrix to modify the r.h.s.
       @a b; @a tdofs_flux is a list of true DOFs (non-directional, i.e. >= 0).
       */
   virtual void ParallelEliminateTDofsInRHS(const Array<int> &tdofs_flux,
                                            const BlockVector &X, BlockVector &B)
   { MFEM_ABORT("Not implemented"); }
#endif

   virtual void ReduceRHS(const BlockVector &b, Vector &b_r) const = 0;

   virtual void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                                BlockVector &sol) const = 0;

   virtual void Reset();
};

class DarcyFluxReduction : public DarcyReduction
{
   Array<int> Af_ipiv;

#ifdef MFEM_USE_MPI
   std::unique_ptr<HypreParMatrix> hB;

   void InitDNbr();
   void CountBSharedFaces(Array<int> &face_offs) const override;
   void CountDSharedFaces(Array<int> &face_offs) const override;
   int GetFaceNbrVDofs(int el, Array<int> &vdofs, bool adjust_vdofs = true) const;
   bool Parallel() const { return (pfes_p != NULL); }
#else
   bool Parallel() const { return false; }
#endif
   int GetInteriorFaceNbr(int f, int el, int &ori, Array<int> &vdofs,
                          bool adjust_vdofs = true) const;
   void ComputeS() override;

public:
   DarcyFluxReduction(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
                      bool bsymmetrize = true);

   ~DarcyFluxReduction();

   void Init(const Array<int> &ess_flux_tdof_list) override;

#ifdef MFEM_USE_MPI
   void AssembleDivSharedFaceMatrix(int sface, const DenseMatrix &) override;

   void AssemblePotSharedFaceMatrix(int sface, const DenseMatrix &) override;
#endif

   void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                            const BlockVector &x, BlockVector &b) override
   { MFEM_ASSERT(vdofs_flux.Size() == 0, "Essential VDOFs are not supported"); }

#ifdef MFEM_USE_MPI
   void ParallelEliminateTDofsInRHS(const Array<int> &tdofs_flux,
                                    const BlockVector &x, BlockVector &b) override
   { MFEM_ASSERT(tdofs_flux.Size() == 0, "Essential TDOFs are not supported"); }
#endif

   void ReduceRHS(const BlockVector &b, Vector &b_r) const override;

   void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                        BlockVector &sol) const override;
};

class DarcyPotentialReduction : public DarcyReduction
{
   Array<int> hat_offsets, hat_dofs_marker;

   Array<int> Ae_offsets;
   Array<real_t> Ae_data;

   Array<int> Be_offsets;
   Array<real_t> Be_data;

   Array<int> D_ipiv;

   void GetFDofs(int el, Array<int> &fdofs) const;
   void GetEDofs(int el, Array<int> &edofs) const;

#ifdef MFEM_USE_MPI
   void CountBSharedFaces(Array<int> &face_offs) const override { }
   void CountDSharedFaces(Array<int> &face_offs) const override { }
   bool Parallel() const { return (pfes_u != NULL); }
#else
   bool Parallel() const { return false; }
#endif
   void ComputeS() override;

public:
   DarcyPotentialReduction(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
                           bool bsymmetrize = true);

   ~DarcyPotentialReduction();

   void Init(const Array<int> &ess_flux_tdof_list) override;

   void AssembleFluxMassMatrix(int el, const DenseMatrix &A) override;

   void AssembleDivMatrix(int el, const DenseMatrix &B) override;

   void AssemblePotFaceMatrix(int face, const DenseMatrix &) override
   { MFEM_ABORT("Cannot eliminate potential with face contributions!"); }

#ifdef MFEM_USE_MPI
   void AssembleDivSharedFaceMatrix(int sface, const DenseMatrix &) override
   { MFEM_ABORT("Face contributions are not supported!"); }

   void AssemblePotSharedFaceMatrix(int sface, const DenseMatrix &) override
   { MFEM_ABORT("Cannot eliminate potential with face contributions!"); }
#endif

   void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                            const BlockVector &x, BlockVector &b) override;

#ifdef MFEM_USE_MPI
   void ParallelEliminateTDofsInRHS(const Array<int> &tdofs_flux,
                                    const BlockVector &x, BlockVector &b) override;
#endif

   void ReduceRHS(const BlockVector &b, Vector &b_r) const override;

   void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                        BlockVector &sol) const override;

   void Reset() override;
};

}

#endif
