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

#ifndef MFEM_PDARCYFORM
#define MFEM_PDARCYFORM

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "darcyform.hpp"
#include "../pbilinearform.hpp"
#include "../pnonlinearform.hpp"
#include "../plinearform.hpp"

namespace mfem
{

/// Parallel block bilinear form for Darcy-like mixed systems
/** Class ParDarcyForm is a parallel version of DarcyForm, wrapping the block
    system operator as follows:
    \verbatim
        ┌             ┐┌   ┐   ┌      ┐
        | PᵀMuP ±PᵀBᵀ || u | _ | Pᵀbu |
        |   BP     Mp || p | ̅  |   bp |
        └             ┘└   ┘   └      ┘
    \endverbatim
    where @a P is the prolongation operator of the flux FE space. This
    transformatiom to true DOFs is performed in FormSystemMatrix() or
    FormLinearSystem() automatically. Alternatively, ParallelAssembleInternal()
    can be called after assembling and finalizing the serial forms to use
    Mult() method directly.
 */
class ParDarcyForm : public DarcyForm
{
protected:
   ParFiniteElementSpace &pfes_u;   ///< flux FE space
   ParFiniteElementSpace &pfes_p;   ///< potential FE space

   ParBilinearForm *pM_u{};      ///< flux mass form
   ParBilinearForm *pM_p{};      ///< potential mass form
   ParNonlinearForm *pMnl_u{};
   ParNonlinearForm *pMnl_p{};
   ParMixedBilinearForm *pB{};   ///< flux divergence form
   ParBlockNonlinearForm *pMnl{};
   ParLinearForm *pb_u{};        ///< flux r.h.s.
   ParLinearForm *pb_p{};        ///< potential r.h.s.

   void UpdateTOffsets();
   void AllocBlockOp();

   void AssembleDivLDGSharedFaces(int skip_zeros);
   void AssemblePotLDGSharedFaces(int skip_zeros);
   void AssemblePotHDGSharedFaces(int skip_zeros);

public:
   class ParGradient;
   friend class ParOperator;
   class ParOperator : public Operator
   {
   protected:
      const ParDarcyForm &darcy;
      mutable std::unique_ptr<BlockOperator> block_grad;
      mutable OperatorHandle opG;
      friend class ParGradient;

   public:
      ParOperator(const ParDarcyForm &darcy)
         : Operator(darcy.toffsets.Last()), darcy(darcy) { }

      void Mult(const Vector &x, Vector &y) const override;
      Operator& GetGradient(const Vector &x) const override;
   };

   friend class ParGradient;
   class ParGradient : public Operator
   {
      const ParOperator &p;
      const Operator &G;
      mutable std::unique_ptr<BlockOperator> block_grad;
      mutable std::array<std::array<std::unique_ptr<HypreParMatrix>,2>,2> hmats;

   public:
      ParGradient(const ParOperator &p, const Vector &x)
         : Operator(p.Width()), p(p), G(p.darcy.Mnl->GetGradient(x)) { }

      void Mult(const Vector &x, Vector &y) const override;
      const BlockOperator& BlockMatrices() const;
   };

   /// Constructor
   /** @param fes_u         flux space
       @param fes_p         potential space
       @param bsymmetrize   sign convention of the mixed formulation, where
                            false keeps all terms without a change, while true
                            flips the sign of B and Mp to obtain a symmetric
                            system with -Bᵀ in the flux equation
    */
   ParDarcyForm(ParFiniteElementSpace *fes_u, ParFiniteElementSpace *fes_p,
                bool bsymmetrize = true);

   /// @name Flux mass
   ///@{

   using DarcyForm::GetFluxMassForm;

   /// Get the flux mass form (non-const)
   /** @note The form is constructed if it has not been already. */
   BilinearForm *GetFluxMassForm();

   /// Get the parallel flux mass form (non-const)
   /** @note The form is constructed if it has not been already. */
   ParBilinearForm *GetParFluxMassForm()
   { return static_cast<ParBilinearForm*>(GetFluxMassForm()); }

   /// Get the parallel flux mass form (const)
   const ParBilinearForm *GetParFluxMassForm() const { return pM_u; }

   using DarcyForm::GetFluxMassNonlinearForm;
   NonlinearForm *GetFluxMassNonlinearForm();
   ParNonlinearForm *GetParFluxMassNonlinearForm()
   { return static_cast<ParNonlinearForm*>(GetFluxMassNonlinearForm()); }
   const ParNonlinearForm *GetParFluxMassNonlinearForm() const { return pMnl_u; }

   ///@}

   /// @name Potential mass
   ///@{

   using DarcyForm::GetPotentialMassForm;

   /// Get the potential mass form (non-const)
   /** @note The form is constructed if it has not been already. */
   BilinearForm *GetPotentialMassForm();

   /// Get the parallel potential mass form (non-const)
   /** @note The form is constructed if it has not been already. */
   ParBilinearForm *GetParPotentialMassForm()
   { return static_cast<ParBilinearForm*>(GetPotentialMassForm()); }

   /// Get the parallel potential mass form (const)
   const ParBilinearForm *GetParPotentialMassForm() const { return pM_p; }

   using DarcyForm::GetPotentialMassNonlinearForm;
   NonlinearForm *GetPotentialMassNonlinearForm();
   ParNonlinearForm *GetParPotentialMassNonlinearForm()
   { return static_cast<ParNonlinearForm*>(GetPotentialMassNonlinearForm()); }
   const ParNonlinearForm *GetParPotentialMassNonlinearForm() const { return pMnl_p; }

   ///@}

   /// @name Flux divergence
   ///@{

   using DarcyForm::GetFluxDivForm;

   /// Get the flux divergence form (non-const)
   /** @note The form is constructed if it has not been already. */
   MixedBilinearForm *GetFluxDivForm();

   /// Get the parallel flux divergence form (non-const)
   /** @note The form is constructed if it has not been already. */
   ParMixedBilinearForm *GetParFluxDivForm()
   { return static_cast<ParMixedBilinearForm*>(GetFluxDivForm()); }

   /// Get the parallel flux divergence form (const)
   const ParMixedBilinearForm *GetParFluxDivForm() const { return pB; }

   ///@}

   ParBlockNonlinearForm *GetParBlockNonlinearForm();
   const ParBlockNonlinearForm *GetParBlockNonlinearForm() const { return pMnl; }

   /// @name Flux r.h.s.
   ///@{

   using DarcyForm::GetFluxRHS;

   /// Get the flux right-hand-side form (non-const)
   /** @note The form is constructed if it has not been already. */
   LinearForm *GetFluxRHS();

   /// Get the parallel flux right-hand-side form (non-const)
   /** @note The form is constructed if it has not been already. */
   ParLinearForm *GetParFluxRHS()
   { return static_cast<ParLinearForm*>(GetFluxRHS()); }

   /// Get the parallel flux right-hand-side form (const)
   const ParLinearForm *GetParFluxRHS() const { return pb_u; }

   ///@}

   /// @name Potential r.h.s.
   ///@{

   using DarcyForm::GetPotentialRHS;

   /// Get the potential right-hand-side form (non-const)
   /** @note The form is constructed if it has not been already. */
   LinearForm *GetPotentialRHS();

   /// Get the parallel potential right-hand-side form (non-const)
   /** @note The form is constructed if it has not been already. */
   ParLinearForm *GetParPotentialRHS()
   { return static_cast<ParLinearForm*>(GetPotentialRHS()); }

   /// Get the parallel potential right-hand-side form (const)
   const ParLinearForm *GetParPotentialRHS() const { return pb_p; }

   ///@}

   /// Assembles the form i.e. sums over all integrators
   /** All bilinear forms are assembled internally, including the right-hand-
       side linear forms (if they are used). However, ParDarcyForm must be
       finalized (see Finalize()) and assembled on true DOFs (see
       ParallelAssembleInternal()) before Mult() can be used. */
   void Assemble(int skip_zeros = 1);

   /// Finalizes the form
   /** All bilinear forms are finalized, enabling to assemble them on true DOFs
       through ParallelAssembleInternal(). */
   void Finalize(int skip_zeros = 1);

   /// Assembles the forms on the true DOFs, i.e. P^t A P.
   /** This enables to perform Mult() for true DOFs vectors. */
   void ParallelAssembleInternal();

   using DarcyForm::FormLinearSystem;

   /** @brief Form the linear system A X = B, corresponding to this bilinear
       form and the linear form @a b(.). */
   /** This method applies any necessary transformations to the linear system
       such as: eliminating boundary conditions; applying conforming
       constraints for non-conforming meshes; parallel assembly; reduction or
       hybridization.

       The GridFunction-size vector @a x must contain the essential VDOF
       values. The right-hand-side vector @a b must be initialized.

       The vector @a X is initialized with a suitable initial guess, the
       essential entries of @a X are set to the corresponding VDOF values of
       @a x and all other entries are set to zero (@a copy_interior == 0) or
       copied from @a x (@a copy_interior != 0). For hybridization or
       reduction, the values of @a x are not used, but the initial guess can
       be provided in @a X directly (with @a copy_interior == 0).

       This method can be called multiple times (with the same @a ess_tdof_list
       array) to initialize different right-hand sides and boundary condition
       values.

       After solving the linear system, the finite element solution @a x can be
       recovered by calling RecoverFEMSolution() (with the same vectors @a X,
       @a b, and @a x).

       @note If there are no transformations, @a X simply reuses the data of
             @a x. */
   void FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                         BlockVector &x, BlockVector &b, OperatorHandle &A, Vector &X,
                         Vector &B, int copy_interior = 0) override;

   /** @brief Form the linear system A X = B, corresponding to this bilinear
       form and its internal right-hand-side linear form. */
   /** @see FormLinearSystem(const Array<int> &, BlockVector &, BlockVector &,
                             OperatorHandle &, Vector &, Vector &, int) */
   void FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                         BlockVector &x, OperatorHandle &A, Vector &X,
                         Vector &B, int copy_interior = 0) override;

   /// Form the linear system matrix @a A, see FormLinearSystem() for details.
   void FormSystemMatrix(const Array<int> &ess_flux_tdof_list,
                         OperatorHandle &A) override;

   /** @brief Not available, use RecoverFEMSolution(const Vector &, const
       BlockVector &, BlockVector &) instead. */
   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x) override
   { MFEM_ABORT("This class uses BlockVectors instead of Vectors."); }

   /// Recover the solution of a linear system formed with FormLinearSystem().
   /** Call this method after solving a linear system constructed using the
       FormLinearSystem() method to recover the solution as a GridFunction-size
       vector in @a x. Use the same arguments as in the FormLinearSystem() call.
   */
   void RecoverFEMSolution(const Vector &X, const BlockVector &b,
                           BlockVector &x) override;

   /// Recover the solution of a linear system formed with FormLinearSystem().
   /** Call this method after solving a linear system constructed using the
       FormLinearSystem() method to recover the solution as a GridFunction-size
       vector in @a x. Use the same arguments as in the FormLinearSystem() call.
   */
   void RecoverFEMSolution(const Vector &X, BlockVector &x) override;

   /** @brief Use the stored eliminated part of the sytem to modify the r.h.s.
       @a b; @a tdofs_flux is a list of true DOFs. */
   void ParallelEliminateTDofsInRHS(const Array<int> &tdofs_flux,
                                    const BlockVector &x, BlockVector &b);

   /// Operator application
   void Mult (const Vector & x, Vector & y) const override;

   /// Return the associated parallel flux FE space.
   ParFiniteElementSpace *ParFluxFESpace() { return &pfes_u; }
   /// Read-only access to the associated parallel flux FE space.
   const ParFiniteElementSpace *ParFluxFESpace() const { return &pfes_u; }

   /// Return the associated parallel potential FE space.
   ParFiniteElementSpace *ParPotentialFESpace() { return &pfes_p; }
   /// Read-only access to the associated parallel potential FE space.
   const ParFiniteElementSpace *ParPotentialFESpace() const { return &pfes_p; }

   /** @brief Update the ParFiniteElementSpace%s and delete all data associated
       with the old ones. */
   void Update() override;

   /// Destroys the form.
   virtual ~ParDarcyForm();
};

}

#endif // MFEM_USE_MPI

#endif
