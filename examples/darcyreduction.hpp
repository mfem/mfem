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

#include "../config/config.hpp"
#include "../fem/bilinearform.hpp"
#include "../fem/nonlinearform.hpp"

#define MFEM_DARCY_REDUCTION_ELIM_BCS

namespace mfem
{

class DarcyReduction : public Operator
{
protected:
   FiniteElementSpace *fes_u, *fes_p;
   NonlinearFormIntegrator *m_nlfi_u, *m_nlfi_p;
   bool own_m_nlfi_u, own_m_nlfi_p;

   Array<int> hat_offsets, hat_dofs_marker;

   Array<int> Af_offsets, Af_f_offsets, Ae_offsets;
   real_t *Af_data, *Ae_data;

   Array<int> Bf_offsets, Be_offsets;
   real_t *Bf_data, *Be_data;

   Array<int> Df_offsets, Df_f_offsets;
   real_t *Df_data;

   bool bsym;

   SparseMatrix *S;

   void GetFDofs(int el, Array<int> &fdofs) const;
   void GetEDofs(int el, Array<int> &edofs) const;

   virtual void ComputeS() = 0;

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

   void AssembleFluxMassMatrix(int el, const DenseMatrix &A);

   void AssemblePotMassMatrix(int el, const DenseMatrix &D);

   void AssembleDivMatrix(int el, const DenseMatrix &B);

   /** @brief Use the stored eliminated part of the matrix to modify the r.h.s.
       @a b; @a vdofs_flux is a list of DOFs (non-directional, i.e. >= 0). */
   virtual void EliminateVDofsInRHS(const Array<int> &vdofs_flux,
                                    const BlockVector &x, BlockVector &b);

   /// Operator application: `y=A(x)`.
   void Mult(const Vector &x, Vector &y) const override;

   /// Finalize the construction of the reduced matrix.
   virtual void Finalize();

   /// Return the serial reduced matrix.
   SparseMatrix &GetMatrix() { return *S; }

   virtual void ReduceRHS(const BlockVector &b, Vector &b_r) const = 0;

   virtual void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                                BlockVector &sol) const = 0;

   virtual void Reset();
};

class DarcyPotentialReduction : public DarcyReduction
{
   int *Df_ipiv;

   void ComputeS() override;

public:
   DarcyPotentialReduction(FiniteElementSpace *fes_u, FiniteElementSpace *fes_p,
                           bool bsymmetrize = true);

   ~DarcyPotentialReduction();

   void Init(const Array<int> &ess_flux_tdof_list) override;

   void ReduceRHS(const BlockVector &b, Vector &b_r) const override;

   void ComputeSolution(const BlockVector &b, const Vector &sol_r,
                        BlockVector &sol) const override;
};

}

#endif
