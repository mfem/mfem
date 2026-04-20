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

/// Class for algebraic reduction of Darcy-like mixed systems
/** DarcyReduction is the base class for objects performing algebraic reduction
    by elimination of one of the equations of mixed systems with
    (anti)symmetric weak form common for parabolic and elliptic problems. They
    can be written as:
    \verbatim
        ┌        ┐┌   ┐   ┌    ┐
        | Mu ±Bᵀ || u | _ | bu |
        | B  Mp  || p | ̅  | bp |
        └        ┘└   ┘   └    ┘
    \endverbatim
    where @a u is the flux (continuous or discontinuous) and @a p is the
    potential (assumed always discontinuous). The bilinear forms @a Mu
    and @a Mp are the mass terms of the flux and potential respectively. The
    mixed bilinear form @a B is the divergence of flux (in a generalized sense)
    and @a bu and @a bp are the right-hand-side terms of the flux and potential
    respectively.

    The sign convention of the system is chosen in the constructor
    DarcyReduction(). Given the set of the forms (Mu, B, Mp), either a
    symmetric system without a sign change (#bsym == false) or with a flipped
    sign (#bsym == true) is formed respectively:
    \verbatim
        ┌       ┐        ┌        ┐
        | Mu Bᵀ |        | Mu -Bᵀ |
        | B  Mp |   or   | -B -Mp |
        └       ┘        └        ┘
    \endverbatim

    The first step of the reduction process is assembly at the element/face
    level. It is initiated by a call of Init() followed by subsequent calls of
    Assemble*Matrix() methods. The assembly process is finished by Finalize(),
    enabling to use Mult() or access the reduced system matrix through
    GetMatrix() (or GetParallelMatrix() in parallel). The right hand side of
    the mixed system can be reduced through ReduceRHS(). After solution of the
    reduced system, the original quantities of the system can be recovered
    through ComputeSolution().
 */
class DarcyReduction : public Operator
{
protected:
   FiniteElementSpace &fes_u;   ///< flux FE space
   FiniteElementSpace &fes_p;   ///< potential FE space
   NonlinearFormIntegrator *m_nlfi_u{};
   NonlinearFormIntegrator *m_nlfi_p{};
   bool own_m_nlfi_u{};
   bool own_m_nlfi_p{};

   Array<int> Af_offsets;        ///< @a Mu element matrix offsets
   Array<int> Af_f_offsets;      ///< @a Mu element vector offsets
   Array<real_t> Af_data;        ///< @a Mu element matrix data

   Array<int> Bf_offsets;        ///< @a B element matrix offsets
   Array<int> Bf_face_offsets;   ///< @a B face element matrix offsets
   Array<real_t> Bf_data;        ///< @a B element matrix data
   Array<real_t> Bf_face_data;   ///< @a B face element matrix data

   Array<int> D_offsets;         ///< @a Mp element matrix offsets
   Array<int> D_f_offsets;       ///< @a Mp element vector offsets
   Array<int> D_face_offsets;    ///< @a Mp face element matrix offsets
   Array<real_t> D_data;         ///< @a Mp element matrix data
   Array<real_t> D_face_data;    ///< @a Mp face element matrix data

   bool bsym{};      ///< sign convention, see DarcyReduction()

   std::unique_ptr<SparseMatrix> S;    ///< reduced system matrix

#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfes_u;      ///< parallel flux FE space
   ParFiniteElementSpace *pfes_p;      ///< parallel potential FE space
   OperatorHandle pS;                  ///< parallel reduced system matrix
#endif

   /// Initializes @a Mu for element assembly
   void InitA();

   /// Initializes @a B and @a Mp for element assembly
   void InitBD();

   /// Initializes @a B for face assembly
   void InitBFaces();

   /// Initializes @a Mp for face assembly
   void InitDFaces();

   /// Computes the reduced system matrix
   virtual void ComputeS() = 0;
#ifdef MFEM_USE_MPI
   /// Helper method for calculating offsets of shared faces of @a B
   virtual void CountBSharedFaces(Array<int> &face_offs) const = 0;

   /// Helper method for calculating offsets of shared faces of @a Mp
   virtual void CountDSharedFaces(Array<int> &face_offs) const = 0;

   /// Helper function for construction of the parallel system matrix
   static HypreParMatrix *ConstructParMatrix(SparseMatrix *spmat,
                                             ParFiniteElementSpace *pfes_tr, ParFiniteElementSpace *pfes_te = NULL);
#endif

public:
   /// Constructor
   /** @param fes_u         flux space
       @param fes_p         potential space
       @param bsymmetrize   sign convention of the mixed formulation, where
                            false keeps all terms without a change, while true
                            flips the sign of B and Mp to obtain a symmetric
                            system with -Bᵀ in the flux equation
    */
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

   /// Prepare the DarcyReduction object for assembly
   /** @param ess_flux_tdof_list    essential true DOFs of the flux */
   virtual void Init(const Array<int> &ess_flux_tdof_list);

   /// Assemble element matrix of @a Mu
   virtual void AssembleFluxMassMatrix(int el, const DenseMatrix &A);

   /// Assemble element matrix of @a Mp
   virtual void AssemblePotMassMatrix(int el, const DenseMatrix &D);

   /// Assemble element matrix of @a B
   virtual void AssembleDivMatrix(int el, const DenseMatrix &B);

   /// Assemble face element matrix of @a B
   virtual void AssembleDivFaceMatrix(int face, const DenseMatrix &);

   /// Assemble face element matrix of @a Mp
   virtual void AssemblePotFaceMatrix(int face, const DenseMatrix &);

#ifdef MFEM_USE_MPI
   /// Assemble shared face element matrix of @a B
   virtual void AssembleDivSharedFaceMatrix(int sface, const DenseMatrix &) = 0;

   /// Assemble shared face element matrix of @a Mp
   virtual void AssemblePotSharedFaceMatrix(int sface, const DenseMatrix &) = 0;
#endif

   /// Use the stored eliminated part of the sytem to modify the r.h.s.
   /** @param vdofs_flux   list of flux VDOFs (non-directional, i.e. >= 0)
       @param x            solution vector providing the VDOF values
       @param b            right hand side vector
   */
   virtual void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                    const BlockVector &x, BlockVector &b)
   { MFEM_ABORT("Not implemented"); }

   /// Use the stored eliminated part of the sytem to modify the r.h.s.
   /** @param tdofs_flux   list of flux true DOFs
       @param X            solution vector providing the true DOF values
       @param B            (true) right hand side vector
   */
   virtual void EliminateTrueDofsInRHS(const Array<int> &tdofs_flux,
                                       const BlockVector &X, BlockVector &B)
   { MFEM_ABORT("Not implemented"); }

   /// Apply the reduced operator.
   /** @note The DarcyReduction object must be finalized by Finalize(). */
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
#endif

   /// Reduce r.h.s. of the mixed system.
   /** @param b      r.h.s. of the mixed system
       @param b_r    r.h.s of the reduced system
    */
   virtual void ReduceRHS(const BlockVector &b, Vector &b_r) const = 0;

   /// Compute solution of the mixed system.
   /** @param b      r.h.s. of the mixed system
       @param sol_r  solution of the reduced system
       @param sol    solution of the mixed system
    */
   virtual void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                                BlockVector &sol) const = 0;

   /// Resets the assembled data
   /** @note Assumes topology of the mesh does not change, otherwise recreate
       the object. */
   virtual void Reset();
};

/// Class for flux elimination from Darcy-like mixed systems
/** Assuming the notation of DarcyReduction, the algebraic reduction by Schur
    complement to eliminate the flux is performed as follows:
    \verbatim
        S = Mp ± B Mu⁻¹ Bᵀ
        br = bp - B Mu⁻¹ bu
    \endverbatim
    This yields the reduced system S p = br for the discontinuous potential
    @a p only. The flux @a u must be discontinuous for the local inversion.
    However, the forms @a B and @a Mp can have face integrators. This enables
    construction of Local Discontinuous Galerkin (LDG) methods, for example.
 */
class DarcyFluxReduction : public DarcyReduction
{
   Array<int> Af_ipiv;
   std::unique_ptr<SparseMatrix> sBt, sAiBt;
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
   void ComputeS() override;

public:
   /// @copydoc DarcyReduction::DarcyReduction()
   DarcyFluxReduction(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
                      bool bsymmetrize = true);

   /// Destructor
   ~DarcyFluxReduction();

   void Init(const Array<int> &ess_flux_tdof_list) override;

#ifdef MFEM_USE_MPI
   void AssembleDivSharedFaceMatrix(int sface, const DenseMatrix &) override;

   void AssemblePotSharedFaceMatrix(int sface, const DenseMatrix &) override;
#endif

   void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                            const BlockVector &x, BlockVector &b) override
   { MFEM_ASSERT(vdofs_flux.Size() == 0, "Essential VDOFs are not supported"); }

   void EliminateTrueDofsInRHS(const Array<int> &tdofs_flux,
                               const BlockVector &x, BlockVector &b) override
   { MFEM_ASSERT(tdofs_flux.Size() == 0, "Essential TDOFs are not supported"); }

   void ReduceRHS(const BlockVector &b, Vector &b_r) const override;

   void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                        BlockVector &sol) const override;
};

/// Class for potential elimination from Darcy-like mixed systems
/** Assuming the notation of DarcyReduction, the algebraic reduction by Schur
    complement to eliminate the potential is performed as follows:
    \verbatim
        S = Mu ± Bᵀ Mp⁻¹ B
        br = bu ± Bᵀ Mp⁻¹ bp
    \endverbatim
    This yields the reduced system S u = br for the flux @a u only. The flux
    can be discontinuous or continuous. However, the forms @a B and @a Mp
    cannot have face integrators.
 */
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
   /// @copydoc DarcyReduction::DarcyReduction()
   DarcyPotentialReduction(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
                           bool bsymmetrize = true);

   /// Destructor
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

   void EliminateTrueDofsInRHS(const Array<int> &tdofs_flux,
                               const BlockVector &x, BlockVector &b) override;

   void ReduceRHS(const BlockVector &b, Vector &b_r) const override;

   void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                        BlockVector &sol) const override;

   void Reset() override;
};

}

#endif
