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

#ifndef MFEM_DARCYFORM
#define MFEM_DARCYFORM

#include "../../config/config.hpp"
#include "../bilinearform.hpp"
#include "../nonlinearform.hpp"
#include "darcyreduction.hpp"
#include "darcyhybridization.hpp"

namespace mfem
{

/// Block bilinear form for Darcy-like mixed systems
/** Class DarcyForm represents mixed systems with (anti)symmetric weak form
    common for parabolic and elliptic problems. They can be written as:
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
    respectively. The r.h.s terms can be constructed internally or
    provided externally (see FormLinearSystem()).

    The sign convention of the system is chosen in the constructor DarcyForm().
    Given the set of the forms (Mu, B, Mp), either a symmetric system without
    a sign change (#bsym == false) or with a flipped sign (#bsym == true) is
    formed respectively:
    \verbatim
        ┌       ┐        ┌        ┐
        | Mu Bᵀ |        | Mu -Bᵀ |
        | B  Mp |   or   | -B -Mp |
        └       ┘        └        ┘
    \endverbatim

    The individual terms of the system are accessed through Get*Form() or
    Get*RHS() methods, where the non-const versions construct the corresponding
    forms at the first occurance. After setting up the (bi)linear forms, the
    system is assembled through Assemble() and finilized by Finalize(),
    similarly to BilinearForm. Furthermore, the elimination of the essential
    TDOFs and construction of the discrete linear system is done through
    FormLinearSystem() followed by RecoverFEMSolution(). Note the primary
    variables and the r.h.s terms are packed together in BlockVector%s
    (with offsets obtained through GetOffsets()).

    A notable feature of DarcyForm is the capability to perform algebraic
    reduction or hybridization of the system. To reduce the system by
    elimination of discontinuous fluxes, use EnableFluxReduction(). Similarly,
    elimination of potentials (without face integrators) is triggered by
    EnablePotentialReduction() before the assembling process. Alternatively,
    the system can be hybridized in terms of the total flux by
    EnableHybridization(), reducing the system to the trace unknowns through
    DarcyHybridization. In any case, the reductions are performed in
    FormSystemMatrix() or FormLinearSystem(), where the original set of the
    quantities (flux and potential) is recovered though RecoverFEMSolution().

    Additionally, reconstruction of the superconvergent quantities can be
    performed after solution of the hybridized system has been obtained.
    It combines the original solution (flux and potential) with the trace
    solution to obtain the normally continuous total flux through
    ReconstructTotalFlux(). The second step is reconstruction of the fluxes
    and potentials with the increased polynomial order through
    ReconstructFluxAndPot(). Both steps are combined in Reconstruct(). Refer
    to DarcyHybridization for more detailed explanation of the process.
 */
class DarcyForm : public Operator
{
protected:
   Array<int> offsets;      ///< block offsets (VDOFs)
   Array<int> toffsets;     ///< block offsets (TDOFs)

   FiniteElementSpace *fes_u;   ///< flux FE space
   FiniteElementSpace *fes_p;   ///< potential FE space

   bool bsym;   ///< sign convention, see DarcyForm()

   std::unique_ptr<BilinearForm> M_u;       ///< flux mass form
   std::unique_ptr<BilinearForm> M_p;       ///< potential mass form
   std::unique_ptr<NonlinearForm> Mnl_u;
   std::unique_ptr<NonlinearForm> Mnl_p;
   std::unique_ptr<MixedBilinearForm> B;    ///< flux divergence form
   std::unique_ptr<BlockNonlinearForm> Mnl;
   std::unique_ptr<LinearForm> b_u;         ///< flux r.h.s
   std::unique_ptr<LinearForm> b_p;         ///< potential r.h.s.

   mutable OperatorHandle opM_u;    ///< flux mass operator
   mutable OperatorHandle opM_p;    ///< potential mass operator
   mutable OperatorHandle opB;      ///< flux divergence operator
   mutable OperatorHandle opBt;     ///< transposed flux divergence operator
   mutable OperatorHandle opG;

   /// The assembly level of the form (full, partial, etc.)
   AssemblyLevel assembly{AssemblyLevel::LEGACY};

   std::unique_ptr<BlockOperator> block_op; ///< block operator
   std::unique_ptr<BlockVector> block_b;    ///< block r.h.s.
   mutable std::unique_ptr<BlockOperator> block_grad;

   std::unique_ptr<DarcyReduction> reduction;           ///< reduction
   std::unique_ptr<DarcyHybridization> hybridization;   ///< hybridization

   /// The DarcyForm of the reconstructed super-convergent system
   mutable std::unique_ptr<DarcyForm> reconstruction;
   mutable std::unique_ptr<MixedBilinearForm> M_p_src;

   void UpdateOffsetsAndSize();
   void UpdateTOffsetsAndSize();
   void EnableReduction(const Array<int> &ess_flux_tdof_list,
                        DarcyReduction *reduction);

   void AssembleDivLDGFaces(int skip_zeros);
   void AssemblePotLDGFaces(int skip_zeros);
   void AssemblePotHDGFaces(int skip_zeros);

   void AllocBlockOp(bool nonconforming = false);
   void AllocRHS();
   const Operator* ConstructBT(const MixedBilinearForm *B) const;
   const Operator* ConstructBT(const OperatorHandle &B) const;

   void ReconstructFluxAndPot(const DarcyHybridization &h, const GridFunction &pc,
                              const GridFunction &ut, GridFunction &u, GridFunction &p,
                              GridFunction &tr, MixedBilinearForm *D = NULL) const;

public:
   friend class Gradient;
   class Gradient : public Operator
   {
      const DarcyForm &p;
      const Operator &G;
      mutable std::unique_ptr<BlockOperator> block_grad;
      mutable std::array<std::array<std::unique_ptr<SparseMatrix>,2>,2> smats;

   public:
      Gradient(const DarcyForm &p, const Vector &x)
         : Operator(p.Width()), p(p), G(p.Mnl->GetGradient(x)) { }

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
   DarcyForm(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
             bool bsymmetrize = true);

   /// Get VDOF block offsets
   inline const Array<int>& GetOffsets() const { return offsets; }

   /// Get TDOF block offsets
   inline const Array<int>& GetTrueOffsets() const { return toffsets; }

   /// @name Flux mass
   ///@{

   /// Get the flux mass form (non-const)
   /** @note The form is constructed if it has not been already. */
   BilinearForm *GetFluxMassForm();

   /// Get the flux mass form (const)
   const BilinearForm *GetFluxMassForm() const { return M_u.get(); }

   NonlinearForm *GetFluxMassNonlinearForm();
   const NonlinearForm *GetFluxMassNonlinearForm() const { return Mnl_u.get(); }

   ///@}

   /// @name Potential mass
   ///@{

   /// Get the potential mass form (non-const)
   /** @note The form is constructed if it has not been already. */
   BilinearForm *GetPotentialMassForm();

   /// Get the potential mass form (const)
   const BilinearForm *GetPotentialMassForm() const { return M_p.get(); }

   NonlinearForm *GetPotentialMassNonlinearForm();
   const NonlinearForm *GetPotentialMassNonlinearForm() const { return Mnl_p.get(); }

   ///@}

   /// @name Flux divergence
   ///@{

   /// Get the flux divergence form (non-const)
   /** @note The form is constructed if it has not been already. */
   MixedBilinearForm *GetFluxDivForm();

   /// Get the flux divergence form (const)
   const MixedBilinearForm *GetFluxDivForm() const { return B.get(); }

   ///@}

   BlockNonlinearForm *GetBlockNonlinearForm();
   const BlockNonlinearForm *GetBlockNonlinearForm() const { return Mnl.get(); }

   /// @name Flux r.h.s.
   ///@{

   /// Get the flux right-hand-side form (non-const)
   /** @note The form is constructed if it has not been already. */
   LinearForm *GetFluxRHS();

   /// Get the flux right-hand-side form (const)
   const LinearForm *GetFluxRHS() const { return b_u.get(); }

   ///@}

   /// @name Potential r.h.s.
   ///@{

   /// Get the potential right-hand-side form (non-const)
   /** @note The form is constructed if it has not been already. */
   LinearForm *GetPotentialRHS();

   /// Get the potential right-hand-side form (non-const)
   const LinearForm *GetPotentialRHS() const { return b_p.get(); }

   ///@}

   /// Set the desired assembly level.
   /** Valid choices are:

       - AssemblyLevel::LEGACY (default)
       - AssemblyLevel::FULL
       - AssemblyLevel::PARTIAL
       - AssemblyLevel::ELEMENT
       - AssemblyLevel::NONE

       If used, this method must be called before assembly. */
   void SetAssemblyLevel(AssemblyLevel assembly_level);

   /// Returns the assembly level
   AssemblyLevel GetAssemblyLevel() const { return assembly; }

   /// Enable flux reduction
   /** For details see the description for class DarcyFluxReduction. This
       method should be called before assembly. */
   void EnableFluxReduction();

   /// Enable potential reduction
   /** For details see the description for class DarcyPotentialReduction. This
       method should be called before assembly. */
   void EnablePotentialReduction(const Array<int> &ess_flux_tdof_list);

   /// Get the applied reduction
   /** @see EnableFluxReduction() and EnablePotentialReduction() */
   DarcyReduction *GetReduction() const { return reduction.get(); }

   /// Enable hybridization
   /** For details see the description for class DarcyHybridization. This
       method should be called before assembly. */
   void EnableHybridization(FiniteElementSpace *constr_space,
                            BilinearFormIntegrator *constr_flux_integ,
                            const Array<int> &ess_flux_tdof_list);

   /// Get the applied hybridization
   DarcyHybridization *GetHybridization() const { return hybridization.get(); }

   /// Assembles the form i.e. sums over all integrators
   /** All bilinear forms are assembled internally, including the right-hand-
       side linear forms (if they are used). However, DarcyForm must be
       finalized (see Finalize()) before Mult() can be used. */
   void Assemble(int skip_zeros = 1);

   /// Finalizes the form
   /** All bilinear forms are finalized, enabling to perform Mult(). */
   void Finalize(int skip_zeros = 1);

   /** @brief Form the linear system A X = B, corresponding to this bilinear
       form and the linear form @a b(.). */
   /** This method applies any necessary transformations to the linear system
       such as: eliminating boundary conditions; applying conforming
       constraints for non-conforming meshes; reduction or hybridization.

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
   virtual void FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                                 BlockVector &x, BlockVector &b, OperatorHandle &A, Vector &X,
                                 Vector &B, int copy_interior = 0);

   /** @brief Form the linear system A X = B, corresponding to this bilinear
       form and its internal right-hand-side linear form. */
   /** @see FormLinearSystem(const Array<int> &, BlockVector &, BlockVector &,
                             OperatorHandle &, Vector &, Vector &, int) */
   virtual void FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                                 BlockVector &x, OperatorHandle &A, Vector &X, Vector &B,
                                 int copy_interior = 0);

   /** @brief Form the linear system A X = B, corresponding to this bilinear
       form and the linear form @a b(.). */
   /** Version of the method FormLinearSystem() where the system matrix is
       returned in the variable @a A, of type OpType, holding a *reference* to
       the system matrix (created with the method OpType::MakeRef()). The
       reference will be invalidated when SetOperatorType(), Update(), or the
       destructor is called. */
   template <typename OpType>
   void FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                         BlockVector &x, BlockVector &b,
                         OpType &A, Vector &X, Vector &B,
                         int copy_interior = 0)
   {
      OperatorHandle Ah;
      FormLinearSystem(ess_flux_tdof_list, x, b, Ah, X, B, copy_interior);
      OpType *A_ptr = Ah.Is<OpType>();
      MFEM_VERIFY(A_ptr, "invalid OpType used");
      A.MakeRef(*A_ptr);
   }

   /** @brief Form the linear system A X = B, corresponding to this bilinear
       form and its internal right-hand-side linear form. */
   /** Version of the method FormLinearSystem() where the system matrix is
       returned in the variable @a A, of type OpType, holding a *reference* to
       the system matrix (created with the method OpType::MakeRef()). The
       reference will be invalidated when SetOperatorType(), Update(), or the
       destructor is called. */
   template <typename OpType>
   void FormLinearSystem(const Array<int> &ess_flux_tdof_list,
                         BlockVector &x, OpType &A, Vector &X, Vector &B,
                         int copy_interior = 0)
   {
      OperatorHandle Ah;
      FormLinearSystem(ess_flux_tdof_list, x, Ah, X, B, copy_interior);
      OpType *A_ptr = Ah.Is<OpType>();
      MFEM_VERIFY(A_ptr, "invalid OpType used");
      A.MakeRef(*A_ptr);
   }

   /// Form the linear system matrix @a A, see FormLinearSystem() for details.
   virtual void FormSystemMatrix(const Array<int> &ess_flux_tdof_list,
                                 OperatorHandle &A);

   /// Form the linear system matrix A, see FormLinearSystem() for details.
   /** Version of the method FormSystemMatrix() where the system matrix is
       returned in the variable @a A, of type OpType, holding a *reference* to
       the system matrix (created with the method OpType::MakeRef()). The
       reference will be invalidated when SetOperatorType(), Update(), or the
       destructor is called. */
   template <typename OpType>
   void FormSystemMatrix(const Array<int> &ess_flux_tdof_list, OpType &A)
   {
      OperatorHandle Ah;
      FormSystemMatrix(ess_flux_tdof_list, Ah);
      OpType *A_ptr = Ah.Is<OpType>();
      MFEM_VERIFY(A_ptr, "invalid OpType used");
      A.MakeRef(*A_ptr);
   }

   /** @brief Not available, use RecoverFEMSolution(const Vector &, const
       BlockVector &, BlockVector &) instead. */
   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x) override
   { MFEM_ABORT("This class uses BlockVectors instead of Vectors."); }

   /// Recover the solution of a linear system formed with FormLinearSystem().
   /** Call this method after solving a linear system constructed using the
       FormLinearSystem() method to recover the solution as a GridFunction-size
       vector in @a x. Use the same arguments as in the FormLinearSystem() call.
   */
   virtual void RecoverFEMSolution(const Vector &X, const BlockVector &b,
                                   BlockVector &x);

   /// Recover the solution of a linear system formed with FormLinearSystem().
   /** Call this method after solving a linear system constructed using the
       FormLinearSystem() method to recover the solution as a GridFunction-size
       vector in @a x. Use the same arguments as in the FormLinearSystem() call.
   */
   virtual void RecoverFEMSolution(const Vector &X, BlockVector &x);

   /// Reconstruct the total flux from the provided hybridized solution.
   /** The total flux function is continuous and its finite element space is
       assumed to have equal number of DOFs at faces as the trace variable.
       The definition of the total flux inside the elements is deduced from
       the potential integrators used. If no finite element space is assigned
       to the grid function, Raviart-Thomas space is constructed and owned by
       the function.
   */
   void ReconstructTotalFlux(const BlockVector &sol, const Vector &sol_r,
                             GridFunction &ut) const;

   /// Reconstruct the flux, potential and traces from solution and total flux.
   /** The reconstructed quantities are in finite element spaces of one order
       higher than the original spaces. If no are assigned to the functions,
       they are automatically constructed from the primary ones and owned by
       the functions.
    */
   void ReconstructFluxAndPot(const BlockVector &sol, const GridFunction &ut,
                              GridFunction &u, GridFunction &p, GridFunction &tr) const;

   /// Reconstruct all quantities from hybridized solution.
   /** The reconstruction combines the reconstruction of the total flux through
       ReconstructTotalFlux() and the primary quantities through
       ReconstructFluxAndPot(). See them for details.
    */
   void Reconstruct(const BlockVector &sol, const Vector &sol_r, GridFunction &ut,
                    GridFunction &u, GridFunction &p, GridFunction &tr) const
   {
      ReconstructTotalFlux(sol, sol_r, ut);
      ReconstructFluxAndPot(sol, ut, u, p, tr);
   }

   /// Use the stored eliminated part of the sytem to modify the r.h.s.
   /** @param tdofs_flux   list of flux true DOFs
       @param x            solution vector providing the true DOF values
       @param b            (true) right hand side vector
   */
   void EliminateTrueDofsInRHS(const Array<int> &tdofs_flux,
                               const BlockVector &x, BlockVector &b);

   /// Use the stored eliminated part of the sytem to modify the r.h.s.
   /** @param vdofs_flux   list of flux VDOFs (non-directional, i.e. >= 0)
       @param x            solution vector providing the VDOF values
       @param b            right hand side vector
   */
   void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                            const BlockVector &x, BlockVector &b);

   /// Operator application
   void Mult (const Vector & x, Vector & y) const override;

   /// Evaluate the gradient operator at the point @a x.
   Operator &GetGradient(const Vector &x) const override;

   /// Return the associated flux FE space.
   FiniteElementSpace *FluxFESpace() { return fes_u; }
   /// Read-only access to the associated flux FiniteElementSpace.
   const FiniteElementSpace *FluxFESpace() const { return fes_u; }

   /// Return the associated flux FE space.
   FiniteElementSpace *PotentialFESpace() { return fes_p; }
   /// Read-only access to the associated flux FiniteElementSpace.
   const FiniteElementSpace *PotentialFESpace() const { return fes_p; }

   /** @brief Update the FiniteElementSpace%s and delete all data associated
       with the old ones. */
   virtual void Update();

   /// Destroys the form.
   virtual ~DarcyForm();
};

}

#endif
