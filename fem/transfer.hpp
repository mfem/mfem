// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TRANSFER_HPP
#define MFEM_TRANSFER_HPP

#include "../linalg/linalg.hpp"
#include "fespace.hpp"

#ifdef MFEM_USE_MPI
#include "pfespace.hpp"
#endif

namespace mfem
{

/** @brief Base class for transfer algorithms that construct transfer Operator%s
    between two finite element (FE) spaces. */
/** Generally, the two FE spaces (domain and range) can be defined on different
    meshes. */
class GridTransfer
{
protected:
   FiniteElementSpace &dom_fes; ///< Domain FE space
   FiniteElementSpace &ran_fes; ///< Range FE space

   /** @brief Desired Operator::Type for the construction of all operators
       defined by the underlying transfer algorithm. It can be ignored by
       derived classes. */
   Operator::Type oper_type;

   OperatorHandle fw_t_oper; ///< Forward true-dof operator
   OperatorHandle bw_t_oper; ///< Backward true-dof operator

#ifdef MFEM_USE_MPI
   bool parallel;
#endif
   bool Parallel() const
   {
#ifndef MFEM_USE_MPI
      return false;
#else
      return parallel;
#endif
   }

   const Operator &MakeTrueOperator(FiniteElementSpace &fes_in,
                                    FiniteElementSpace &fes_out,
                                    const Operator &oper,
                                    OperatorHandle &t_oper);

public:
   /** Construct a transfer algorithm between the domain, @a dom_fes_, and
       range, @a ran_fes_, FE spaces. */
   GridTransfer(FiniteElementSpace &dom_fes_, FiniteElementSpace &ran_fes_);

   /// Virtual destructor
   virtual ~GridTransfer() { }

   /** @brief Set the desired Operator::Type for the construction of all
       operators defined by the underlying transfer algorithm. */
   /** The default value is Operator::ANY_TYPE which typically corresponds to a
       matrix-free operator representation. Note that derived classes are not
       required to support this setting and can ignore it. */
   void SetOperatorType(Operator::Type type) { oper_type = type; }

   /** @brief Return an Operator that transfers GridFunction%s from the domain
       FE space to GridFunction%s in the range FE space. */
   virtual const Operator &ForwardOperator() = 0;

   /** @brief Return an Operator that transfers GridFunction%s from the range FE
       space back to GridFunction%s in the domain FE space. */
   virtual const Operator &BackwardOperator() = 0;

   /** @brief Return an Operator that transfers true-dof Vector%s from the
       domain FE space to true-dof Vector%s in the range FE space. */
   /** This method is implemented in the base class, based on ForwardOperator(),
       however, derived classes can overload the construction, if necessary. */
   virtual const Operator &TrueForwardOperator()
   {
      return MakeTrueOperator(dom_fes, ran_fes, ForwardOperator(), fw_t_oper);
   }

   /** @brief Return an Operator that transfers true-dof Vector%s from the range
       FE space back to true-dof Vector%s in the domain FE space. */
   /** This method is implemented in the base class, based on
       BackwardOperator(), however, derived classes can overload the
       construction, if necessary. */
   virtual const Operator &TrueBackwardOperator()
   {
      return MakeTrueOperator(ran_fes, dom_fes, BackwardOperator(), bw_t_oper);
   }
};


/** @brief Transfer data between a coarse mesh and an embedded refined mesh
    using interpolation. */
/** The forward, coarse-to-fine, transfer uses nodal interpolation. The
    backward, fine-to-coarse, transfer is defined locally (on a coarse element)
    as B = (F^t M_f F)^{-1} F^t M_f, where F is the forward transfer matrix, and
    M_f is a mass matrix on the union of all fine elements comprising the coarse
    element. Note that the backward transfer operator, B, is a left inverse of
    the forward transfer operator, F, i.e. B F = I. Both F and B are defined in
    reference space and do not depend on the actual physical shape of the mesh
    elements.

    It is assumed that both the coarse and the fine FiniteElementSpace%s use
    compatible types of elements, e.g. finite elements with the same map-type
    (VALUE, INTEGRAL, H_DIV, H_CURL - see class FiniteElement). Generally, the
    FE spaces can have different orders, however, in order for the backward
    operator to be well-defined, the (local) number of the fine dofs should not
    be smaller than the number of coarse dofs. */
class InterpolationGridTransfer : public GridTransfer
{
protected:
   BilinearFormIntegrator *mass_integ; ///< Ownership depends on #own_mass_integ
   bool own_mass_integ; ///< Ownership flag for #mass_integ

   OperatorHandle F; ///< Forward, coarse-to-fine, operator
   OperatorHandle B; ///< Backward, fine-to-coarse, operator

public:
   InterpolationGridTransfer(FiniteElementSpace &coarse_fes,
                             FiniteElementSpace &fine_fes)
      : GridTransfer(coarse_fes, fine_fes),
        mass_integ(NULL), own_mass_integ(false)
   { }

   virtual ~InterpolationGridTransfer();

   /** @brief Assign a mass integrator to be used in the construction of the
       backward, fine-to-coarse, transfer operator. */
   void SetMassIntegrator(BilinearFormIntegrator *mass_integ_,
                          bool own_mass_integ_ = true);

   virtual const Operator &ForwardOperator();

   virtual const Operator &BackwardOperator();
};


/** @brief Transfer data in L2 finite element spaces between a coarse mesh and
    an embedded refined mesh using L2 projection. */
/** The forward, coarse-to-fine, transfer uses L2 projection. The backward,
    fine-to-coarse, transfer is defined locally (on a coarse element) as
    B = (F^t M_f F)^{-1} F^t M_f, where F is the forward transfer matrix, and
    M_f is the mass matrix on the union of all fine elements comprising the
    coarse element. Note that the backward transfer operator, B, is a left
    inverse of the forward transfer operator, F, i.e. B F = I. Both F and B are
    defined in physical space and, generally, vary between different mesh
    elements.
    */
class L2ProjectionGridTransfer : public GridTransfer
{
protected:
   /** Class representing projection operator between a high-order L2 finite
       element space on a coarse mesh, and a low-order L2 finite element space
       on a refined mesh (LOR). We assume that the low-order space, fes_lor,
       lives on a mesh obtained by refining the mesh of the high-order space,
       fes_ho. */
   class L2Projection : public Operator
   {
   public:
      virtual void Prolongate(const Vector& x, Vector& y) const = 0;
      virtual void ProlongateTranspose(const Vector& x, Vector& y) const = 0;
      /// Sets relative tolerance and absolute tolerance in preconditioned
      /// conjugate gradient solver.  Only used for H1 spaces.
      virtual void SetTolerance(double p_rtol_, double p_atol_) = 0;
   protected:
      const FiniteElementSpace& fes_ho;
      const FiniteElementSpace& fes_lor;

      Table ho2lor;

      L2Projection(const FiniteElementSpace& fes_ho_,
                   const FiniteElementSpace& fes_lor_);

      void BuildHo2Lor(int nel_ho, int nel_lor,
                       const CoarseFineTransformations& cf_tr);
   };

   class L2ProjectionL2Space : public L2Projection
   {
      // The restriction and prolongation operators are represented as dense
      // elementwise matrices (of potentially different sizes, because of mixed
      // meshes or p-refinement). The matrix entries are stored in the R and P
      // arrays. The entries of the i'th high-order element are stored at the
      // index given by offsets[i].
      mutable Array<double> R, P;
      Array<int> offsets;

   public:
      L2ProjectionL2Space(const FiniteElementSpace& fes_ho_,
                          const FiniteElementSpace& fes_lor_);
      /// Maps <tt>x</tt>, primal field coefficients defined on a coarse mesh with
      /// a higher order L2 finite element space, to <tt>y</tt>, primal field
      /// coefficients defined on a refined mesh with a low order L2 finite element
      /// space.  Refined mesh should be a uniform refinement of the coarse mesh.
      /// Coefficients are computed through minimization of L2 error between the
      /// fields.
      virtual void Mult(const Vector& x, Vector& y) const;
      /// Maps <tt>x</tt>, dual field coefficients defined on a refined mesh with
      /// a low order L2 finite element space, to <tt>y</tt>, dual field
      /// coefficients defined on a coarse mesh with a higher order L2 finite
      /// element space.  Refined mesh should be a uniform refinement of the coarse
      /// mesh. Coefficients are computed through minimization of L2 error between
      /// the primal fields.  Note, if the <tt>x</tt>-coefficients come from
      /// ProlongateTranspose, then mass is conserved.
      virtual void MultTranspose(const Vector& x, Vector& y) const;
      /// Maps <tt>x</tt>, primal field coefficients defined on a refined mesh with
      /// a low order L2 finite element space, to <tt>y</tt>, primal field
      /// coefficients defined on a coarse mesh with a higher order L2 finite
      /// element space.  Refined mesh should be a uniform refinement of the coarse
      /// mesh. Coefficients are computed from the mass conservative left-inverse
      /// prolongation operation.  This functionality is also provided as an
      /// Operator by L2Prolongation.
      virtual void Prolongate(const Vector& x, Vector& y) const;
      /// Maps <tt>x</tt>, dual field coefficients defined on a coarse mesh with
      /// a higher order L2 finite element space, to <tt>y</tt>, dual field
      /// coefficients defined on a refined mesh with a low order L2 finite
      /// element space.  Refined mesh should be a uniform refinement of the coarse
      /// mesh. Coefficients are computed from the transpose of the mass
      /// conservative left-inverse prolongation operation.  This functionality is
      /// also provided as an Operator by L2Prolongation.
      virtual void ProlongateTranspose(const Vector& x, Vector& y) const;
      virtual void SetTolerance(double p_rtol_, double p_atol_) {}
   };

   class L2ProjectionH1Space : public L2Projection
   {
      SparseMatrix R;
      // Used to compute P = (RTxM_LH)^(-1) M_LH^T
      SparseMatrix M_LH;
      SparseMatrix RTxM_LH;
      CGSolver pcg;
      DSmoother Ds;

   public:
      L2ProjectionH1Space(const FiniteElementSpace& fes_ho_,
                          const FiniteElementSpace& fes_lor_);
      /// Maps <tt>x</tt>, primal field coefficients defined on a coarse mesh with
      /// a higher order H1 finite element space, to <tt>y</tt>, primal field
      /// coefficients defined on a refined mesh with a low order H1 finite element
      /// space.  Refined mesh should be a uniform refinement of the coarse mesh.
      /// Coefficients are computed through minimization of L2 error between the
      /// fields.
      virtual void Mult(const Vector& x, Vector& y) const;
      /// Maps <tt>x</tt>, dual field coefficients defined on a refined mesh with
      /// a low order H1 finite element space, to <tt>y</tt>, dual field
      /// coefficients defined on a coarse mesh with a higher order H1 finite
      /// element space.  Refined mesh should be a uniform refinement of the coarse
      /// mesh. Coefficients are computed through minimization of L2 error between
      /// the primal fields.  Note, if the <tt>x</tt>-coefficients come from
      /// ProlongateTranspose, then mass is conserved.
      virtual void MultTranspose(const Vector& x, Vector& y) const;
      /// Maps <tt>x</tt>, primal field coefficients defined on a refined mesh with
      /// a low order H1 finite element space, to <tt>y</tt>, primal field
      /// coefficients defined on a coarse mesh with a higher order H1 finite
      /// element space.  Refined mesh should be a uniform refinement of the coarse
      /// mesh. Coefficients are computed from the mass conservative left-inverse
      /// prolongation operation.  This functionality is also provided as an
      /// Operator by L2Prolongation.
      virtual void Prolongate(const Vector& x, Vector& y) const;
      /// Maps <tt>x</tt>, dual field coefficients defined on a coarse mesh with
      /// a higher order H1 finite element space, to <tt>y</tt>, dual field
      /// coefficients defined on a refined mesh with a low order H1 finite
      /// element space.  Refined mesh should be a uniform refinement of the coarse
      /// mesh. Coefficients are computed from the transpose of the mass
      /// conservative left-inverse prolongation operation.  This functionality is
      /// also provided as an Operator by L2Prolongation.
      virtual void ProlongateTranspose(const Vector& x, Vector& y) const;
      virtual void SetTolerance(double p_rtol_, double p_atol_);
   private:
      /// Computes sparsity pattern and initializes R matrix.  Based on
      /// BilinearForm::AllocMat() except maps between HO elements and LOR
      /// elements.
      void AllocR();
   };

   /** Mass-conservative prolongation operator going in the opposite direction
       as L2Projection. This operator is a left inverse to the L2Projection. */
   class L2Prolongation : public Operator
   {
      const L2Projection &l2proj;

   public:
      L2Prolongation(const L2Projection &l2proj_)
         : Operator(l2proj_.Width(), l2proj_.Height()), l2proj(l2proj_) { }
      void Mult(const Vector &x, Vector &y) const
      {
         l2proj.Prolongate(x, y);
      }
      void MultTranspose(const Vector &x, Vector &y) const
      {
         l2proj.ProlongateTranspose(x, y);
      }
      virtual ~L2Prolongation() { }
   };

   L2Projection   *F; ///< Forward, coarse-to-fine, operator
   L2Prolongation *B; ///< Backward, fine-to-coarse, operator
   bool force_l2_space;

public:
   L2ProjectionGridTransfer(FiniteElementSpace &coarse_fes_,
                            FiniteElementSpace &fine_fes_,
                            bool force_l2_space_ = false)
      : GridTransfer(coarse_fes_, fine_fes_),
        F(NULL), B(NULL), force_l2_space(force_l2_space_)
   { }
   virtual ~L2ProjectionGridTransfer();

   virtual const Operator &ForwardOperator();

   virtual const Operator &BackwardOperator();
private:
   void BuildF();
};

/// Matrix-free transfer operator between finite element spaces
class TransferOperator : public Operator
{
private:
   Operator* opr;

public:
   /// Constructs a transfer operator from \p lFESpace to \p hFESpace.
   /** No matrices are assembled, only the action to a vector is being computed.
       If both spaces' FE collection pointers are pointing to the same collection
       we assume that the grid was refined while keeping the order constant. If
       the FE collections are different, it is assumed that both spaces have are
       using the same mesh. If the first element of the high-order space is a
       `TensorBasisElement`, the optimized tensor-product transfers are used. If
       not, the general transfers used. */
   TransferOperator(const FiniteElementSpace& lFESpace,
                    const FiniteElementSpace& hFESpace);

   /// Destructor
   virtual ~TransferOperator();

   /// @brief Interpolation or prolongation of a vector \p x corresponding to the
   /// coarse space to the vector \p y corresponding to the fine space.
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The vector \p x corresponding to the fine space is restricted to the vector
       \p y corresponding to the coarse space. */
   virtual void MultTranspose(const Vector& x, Vector& y) const override;
};

/// Matrix-free transfer operator between finite element spaces on the same mesh
class PRefinementTransferOperator : public Operator
{
private:
   const FiniteElementSpace& lFESpace;
   const FiniteElementSpace& hFESpace;

public:
   /// @brief Constructs a transfer operator from \p lFESpace to \p hFESpace
   /// which have different FE collections.
   /** No matrices are assembled, only the action to a vector is being computed.
       The underlying finite elements need to implement the GetTransferMatrix
       methods. */
   PRefinementTransferOperator(const FiniteElementSpace& lFESpace_,
                               const FiniteElementSpace& hFESpace_);

   /// Destructor
   virtual ~PRefinementTransferOperator();

   /// @brief Interpolation or prolongation of a vector \p x corresponding to the
   /// coarse space to the vector \p y corresponding to the fine space.
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The vector \p x corresponding to the fine space is restricted to the vector
   \p y corresponding to the coarse space. */
   virtual void MultTranspose(const Vector& x, Vector& y) const override;
};

/// @brief Matrix-free transfer operator between finite element spaces on the same
/// mesh exploiting the tensor product structure of the finite elements
class TensorProductPRefinementTransferOperator : public Operator
{
private:
   const FiniteElementSpace& lFESpace;
   const FiniteElementSpace& hFESpace;
   int dim;
   int NE;
   int D1D;
   int Q1D;
   Array<double> B;
   Array<double> Bt;
   const Operator* elem_restrict_lex_l;
   const Operator* elem_restrict_lex_h;
   Vector mask;
   mutable Vector localL;
   mutable Vector localH;

public:
   /// @brief Constructs a transfer operator from \p lFESpace to \p hFESpace which
   /// have different FE collections.
   /** No matrices are assembled, only the action to a vector is being computed.
   The underlying finite elements need to be of the type `TensorBasisElement`. It
   is also assumed that all the elements in the spaces are of the same type. */
   TensorProductPRefinementTransferOperator(
      const FiniteElementSpace& lFESpace_,
      const FiniteElementSpace& hFESpace_);

   /// Destructor
   virtual ~TensorProductPRefinementTransferOperator();

   /// @brief Interpolation or prolongation of a vector \p x corresponding to the
   /// coarse space to the vector \p y corresponding to the fine space.
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The vector \p x corresponding to the fine space is restricted to the vector
   \p y corresponding to the coarse space. */
   virtual void MultTranspose(const Vector& x, Vector& y) const override;
};

#ifdef MFEM_USE_MPI
/// @brief Matrix-free transfer operator between finite element spaces working on
/// true degrees of freedom
class TrueTransferOperator : public Operator
{
private:
   const ParFiniteElementSpace& lFESpace;
   const ParFiniteElementSpace& hFESpace;
   TransferOperator* localTransferOperator;
   mutable Vector tmpL;
   mutable Vector tmpH;

public:
   /// @brief Constructs a transfer operator working on true degrees of freedom from
   /// from \p lFESpace to \p hFESpace
   TrueTransferOperator(const ParFiniteElementSpace& lFESpace_,
                        const ParFiniteElementSpace& hFESpace_);

   /// Destructor
   ~TrueTransferOperator();

   /// @brief Interpolation or prolongation of a true dof vector \p x to a true dof
   /// vector \p y.
   /** The true dof vector \p x corresponding to the coarse space is restricted to
       the true dof vector \p y corresponding to the fine space. */
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The true dof vector \p x corresponding to the fine space is restricted to
       the true dof vector \p y corresponding to the coarse space. */
   virtual void MultTranspose(const Vector& x, Vector& y) const override;
};
#endif

} // namespace mfem

#endif
