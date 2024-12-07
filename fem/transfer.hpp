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

   bool use_ea;

   MemoryType d_mt;

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
       range, @a ran_fes_, FE spaces, d_mt_ will specify memory space for
       large data structures */
   GridTransfer(FiniteElementSpace &dom_fes_,
                FiniteElementSpace &ran_fes_);

   /// Virtual destructor
   virtual ~GridTransfer() { }

   /** Uses device friendly element assembly versions for L2Projection
       transfers, L2, H1 FEM spaces currently supported */
   void UseEA(bool use_ea_) { use_ea = use_ea_;}

   /** Set memory type for large data structures */
   void SetMemType(MemoryType d_mt_) {d_mt = d_mt_;}

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

   virtual bool SupportsBackwardsOperator() const { return true; }
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

   const Operator &ForwardOperator() override;

   const Operator &BackwardOperator() override;
};


/** @brief Transfer data in L2 and H1 finite element spaces between a coarse
    mesh and an embedded refined mesh using L2 projection. */
/** The forward, coarse-to-fine, transfer uses L2 projection. The backward,
    fine-to-coarse, transfer is defined as B = (F^t M_f F)^{-1} F^t M_f, where F
    is the forward transfer matrix, and M_f is the mass matrix on the coarse
    element. For L2 spaces, M_f is the mass matrix on the union of all fine
    elements comprising the coarse element. For H1 spaces, M_f is a diagonal
    (lumped) mass matrix computed through row-summation. Note that the backward
    transfer operator, B, is a left inverse of the forward transfer operator, F,
    i.e. B F = I. Both F and B are defined in physical space and, generally for
    L2 spaces, vary between different mesh elements.

    This class supports H1 and L2 finite element spaces. Fine meshes are a
    uniform refinement of the coarse mesh, usually created through
    Mesh::MakeRefined. Generally, the coarse and fine FE spaces can have
    different orders, however, in order for the backward operator to be
    well-defined, the number of fine dofs (in a coarse element) should not be
    smaller than the number of coarse dofs. */
class L2ProjectionGridTransfer : public GridTransfer
{
   // Must be public due to host device lambdas
public:
   /** Abstract class representing projection operator between a high-order
       finite element space on a coarse mesh, and a low-order finite element
       space on a refined mesh (LOR). We assume that the low-order space,
       fes_lor, lives on a mesh obtained by refining the mesh of the high-order
       space, fes_ho. */
   class L2Projection : public Operator
   {
   public:
      virtual void Prolongate(const Vector& x, Vector& y) const = 0;
      virtual void ProlongateTranspose(const Vector& x, Vector& y) const = 0;
      /// @brief Sets relative tolerance in preconditioned conjugate gradient
      /// solver.
      ///
      /// Only used for H1 spaces.
      virtual void SetRelTol(real_t p_rtol_) = 0;
      /// @brief Sets absolute tolerance in preconditioned conjugate gradient
      /// solver.
      ///
      /// Only used for H1 spaces.
      virtual void SetAbsTol(real_t p_atol_) = 0;
   protected:
      const FiniteElementSpace& fes_ho;
      const FiniteElementSpace& fes_lor;

      MemoryType d_mt;
      Array<int> offsets;
      Table ho2lor;

      L2Projection(const FiniteElementSpace& fes_ho_,
                   const FiniteElementSpace& fes_lor_,
                   MemoryType d_mt_ = Device::GetHostMemoryType());

      void BuildHo2Lor(int nel_ho, int nel_lor,
                       const CoarseFineTransformations& cf_tr);

      void ElemMixedMass(Geometry::Type geom, const FiniteElement& fe_ho,
                         const FiniteElement& fe_lor, ElementTransformation* tr_ho,
                         ElementTransformation* tr_lor,
                         IntegrationPointTransformation& ip_tr,
                         DenseMatrix& M_mixed_el) const;

      void ElemMixedMass(Geometry::Type geom, const FiniteElement& fe_ho,
                         const FiniteElement& fe_lor,
                         ElementTransformation* el_tr,
                         IntegrationPointTransformation& ip_tr,
                         DenseMatrix& B_L, DenseMatrix& B_H) const;
   public:
      /* Returns the Mixed Mass M_LH via device element assembly by building the
      basis functions and data at the quadrature points. */
      void MixedMassEA(const FiniteElementSpace& fes_ho_,
                       const FiniteElementSpace& fes_lor_,
                       Vector &M_LH,
                       MemoryType d_mt_ = Device::GetHostMemoryType());
   };

   // Class below must be public as we now have device code
public:
   class H1SpaceMixedMassOperator : public Operator
   {
   protected:
      const FiniteElementSpace* fes_ho;
      const FiniteElementSpace* fes_lor;
      Table* ho2lor;
      Vector* M_LH_ea;
   public:
      H1SpaceMixedMassOperator(const FiniteElementSpace* fes_ho_,
                               const FiniteElementSpace* fes_lor_,
                               Table* ho2lor_, Vector* M_LH_ea_);
      void Mult(const Vector& x, Vector& y) const;
      void MultTranspose(const Vector& x, Vector& y) const;
   };

   class H1SpaceLumpedMassOperator : public Operator
   {
   protected:
      const FiniteElementSpace* fes_ho;
      const FiniteElementSpace* fes_lor;
      Vector* ML_inv; // inverse of lumped M_L
   public:
      H1SpaceLumpedMassOperator(const FiniteElementSpace* fes_ho_,
                                const FiniteElementSpace* fes_lor_,
                                Vector& ML_inv_);
      void Mult(const Vector& x, Vector& y) const;
      void MultTranspose(const Vector& x, Vector& y) const;
   };

   /** Class for projection operator between a L2 high-order finite element
       space on a coarse mesh, and a L2 low-order finite element space on a
       refined mesh (LOR). */
   class L2ProjectionL2Space : public L2Projection
   {
      /// The restriction and prolongation operators are represented as dense
      /// elementwise matrices (of potentially different sizes, because of mixed
      /// meshes or p-refinement). The matrix entries are stored in the R and P
      /// arrays. The entries of the i'th high-order element are stored at the
      /// index given by offsets[i].
      mutable Array<real_t> R, P;

      const bool use_ea;

   public:
      L2ProjectionL2Space(const FiniteElementSpace& fes_ho_,
                          const FiniteElementSpace& fes_lor_,
                          const bool use_ea_,
                          MemoryType d_mt_ = Device::GetHostMemoryType());

      /*Same as above but assembles and stores R_ea, P_ea */
      void EAL2ProjectionL2Space();

      /// Maps <tt>x</tt>, primal field coefficients defined on a coarse mesh
      /// with a higher order L2 finite element space, to <tt>y</tt>, primal
      /// field coefficients defined on a refined mesh with a low order L2
      /// finite element space. Refined mesh should be a uniform refinement of
      /// the coarse mesh. Coefficients are computed through minimization of L2
      /// error between the fields.
      void Mult(const Vector& x, Vector& y) const override;

      /// Perform mult on the device (same as above)
      void EAMult(const Vector& x, Vector& y) const;

      /// Maps <tt>x</tt>, dual field coefficients defined on a refined mesh
      /// with a low order L2 finite element space, to <tt>y</tt>, dual field
      /// coefficients defined on a coarse mesh with a higher order L2 finite
      /// element space. Refined mesh should be a uniform refinement of the
      /// coarse mesh. Coefficients are computed through minimization of L2
      /// error between the primal fields. Note, if the <tt>x</tt>-coefficients
      /// come from ProlongateTranspose, then mass is conserved.
      void MultTranspose(const Vector& x, Vector& y) const override;

      void EAMultTranspose(const Vector& x, Vector& y) const;

      /// Maps <tt>x</tt>, primal field coefficients defined on a refined mesh
      /// with a low order L2 finite element space, to <tt>y</tt>, primal field
      /// coefficients defined on a coarse mesh with a higher order L2 finite
      /// element space. Refined mesh should be a uniform refinement of the
      /// coarse mesh. Coefficients are computed from the mass conservative
      /// left-inverse prolongation operation. This functionality is also
      /// provided as an Operator by L2Prolongation.
      void Prolongate(const Vector& x, Vector& y) const override;

      void EAProlongate(const Vector& x, Vector& y) const;

      /// Maps <tt>x</tt>, dual field coefficients defined on a coarse mesh with
      /// a higher order L2 finite element space, to <tt>y</tt>, dual field
      /// coefficients defined on a refined mesh with a low order L2 finite
      /// element space. Refined mesh should be a uniform refinement of the
      /// coarse mesh. Coefficients are computed from the transpose of the mass
      /// conservative left-inverse prolongation operation. This functionality
      /// is also provided as an Operator by L2Prolongation.
      void ProlongateTranspose(const Vector& x, Vector& y) const override;

      void EAProlongateTranspose(const Vector& x, Vector& y) const;

      void SetRelTol(real_t p_rtol_) override { } ///< No-op.
      void SetAbsTol(real_t p_atol_) override { } ///< No-op.
   };

protected:

   /// Class below must be public as we now have device code
public:

   /** Projection operator between a H1 high-order finite element space on a
       coarse mesh, and a H1 low-order finite element space on a refined mesh
       (LOR). */
   class L2ProjectionH1Space : public L2Projection
   {
      const bool use_ea;

   public:
      L2ProjectionH1Space(const FiniteElementSpace &fes_ho_,
                          const FiniteElementSpace &fes_lor_,
                          const bool use_ea_,
                          MemoryType d_mt_ = Device::GetHostMemoryType());
#ifdef MFEM_USE_MPI
      L2ProjectionH1Space(const ParFiniteElementSpace &pfes_ho_,
                          const ParFiniteElementSpace &pfes_lor_,
                          const bool use_ea_,
                          MemoryType d_mt_ = Device::GetHostMemoryType());
#endif
      /// Same as above but assembles action of R through 4 parts:
      ///   ( )  inv( lumped(M_L) ), which is a diagonal matrix (essentially a vector)
      ///   ( )  ElementRestrictionOperator for LOR space
      ///   ( )  mixed mass matrix M_{LH}
      ///   ( )  ElementRestrictionOperator for HO space
      void EAL2ProjectionH1Space();

#ifdef MFEM_USE_MPI
      void EAL2ProjectionH1Space(const ParFiniteElementSpace &pfes_ho_,
                                 const ParFiniteElementSpace &pfes_lor_);
#endif
      /// Maps <tt>x</tt>, primal field coefficients defined on a coarse mesh
      /// with a higher order H1 finite element space, to <tt>y</tt>, primal
      /// field coefficients defined on a refined mesh with a low order H1
      /// finite element space. Refined mesh should be a uniform refinement of
      /// the coarse mesh. Coefficients are computed through minimization of L2
      /// error between the fields.
      void Mult(const Vector& x, Vector& y) const override;

      /// Maps <tt>x</tt>, dual field coefficients defined on a refined mesh
      /// with a low order H1 finite element space, to <tt>y</tt>, dual field
      /// coefficients defined on a coarse mesh with a higher order H1 finite
      /// element space. Refined mesh should be a uniform refinement of the
      /// coarse mesh. Coefficients are computed through minimization of L2
      /// error between the primal fields. Note, if the <tt>x</tt>-coefficients
      /// come from ProlongateTranspose, then mass is conserved.
      void MultTranspose(const Vector& x, Vector& y) const override;

      /// Maps <tt>x</tt>, primal field coefficients defined on a refined mesh
      /// with a low order H1 finite element space, to <tt>y</tt>, primal field
      /// coefficients defined on a coarse mesh with a higher order H1 finite
      /// element space. Refined mesh should be a uniform refinement of the
      /// coarse mesh. Coefficients are computed from the mass conservative
      /// left-inverse prolongation operation. This functionality is also
      /// provided as an Operator by L2Prolongation.
      void Prolongate(const Vector& x, Vector& y) const override;

      /// Maps <tt>x</tt>, dual field coefficients defined on a coarse mesh with
      /// a higher order H1 finite element space, to <tt>y</tt>, dual field
      /// coefficients defined on a refined mesh with a low order H1 finite
      /// element space. Refined mesh should be a uniform refinement of the
      /// coarse mesh. Coefficients are computed from the transpose of the mass
      /// conservative left-inverse prolongation operation. This functionality
      /// is also provided as an Operator by L2Prolongation.
      void ProlongateTranspose(const Vector& x, Vector& y) const override;

      /// Returns the inverse of an on-rank lumped mass matrix
      void LumpedMassInverse(Vector& ML_inv) const;

      void SetRelTol(real_t p_rtol_) override;
      void SetAbsTol(real_t p_atol_) override;

   protected:
      /// Sets up the PCG solver (sets parameters, operator, and preconditioner)
      void SetupPCG();

      /// @brief Computes on-rank R and M_LH matrices. If true, computes mixed mass and/or
      /// inverse lumped mass matrix error when compared to device implementation.
      std::pair<std::unique_ptr<SparseMatrix>,
          std::unique_ptr<SparseMatrix>> ComputeSparseRAndM_LH();

      /// @brief Recovers vector of tdofs given a vector of dofs and a finite
      /// element space
      void GetTDofs(const FiniteElementSpace& fes, const Vector& x, Vector& X) const;
      /// Sets dof values given a vector of tdofs and a finite element space
      void SetFromTDofs(const FiniteElementSpace& fes,
                        const Vector& X,
                        Vector& x) const;
      /// @brief Recovers a vector of dual field coefficients on the tdofs given
      /// a vector of dual coefficients and a finite element space
      void GetTDofsTranspose(const FiniteElementSpace& fes,
                             const Vector& x,
                             Vector& X) const;
      /// @brief Sets dual field coefficients given a vector of dual field
      /// coefficients on the tdofs and a finite element space
      void SetFromTDofsTranspose(const FiniteElementSpace& fes,
                                 const Vector& X,
                                 Vector& x) const;
      /// @brief Fills the vdofs_list array with a list of vdofs for a given
      /// vdim and a given finite element space
      void TDofsListByVDim(const FiniteElementSpace& fes,
                           int vdim,
                           Array<int>& vdofs_list) const;

      /// @brief Computes sparsity pattern and initializes R matrix.
      /// Based on BilinearForm::AllocMat(), except maps between coarse HO
      /// elements and refined LOR elements.
      std::unique_ptr<SparseMatrix> AllocR();

      CGSolver pcg;
      std::unique_ptr<Solver> precon;
      // The restriction operator is represented as an Operator R. The
      // prolongation operator is a dense matrix computed as the inverse of (R^T
      // M_L R), and hence, is not stored.
      // If element assembly is enabled
      std::unique_ptr<Operator> R;
      // Used to compute P = (RT*M_LH)^(-1) M_LH^T
      std::unique_ptr<Operator> M_LH;
      // Inverted operator in P = (RT*M_LH)^(-1) M_LH^T. Used to compute P via PCG.
      std::unique_ptr<Operator> RTxM_LH;
      // Lumped M_L inverse operator built via EA. Wrapped with restriction maps
      // to multiply with scalar TDof LOR vectors.
      std::unique_ptr<Operator> ML_inv_vea;
      // LDof Mixed mass operator built via EA. Wrapped with restrition maps to send
      // scalar LDof HO vectors to LDof LOR vectors.
      Operator *M_LH_local_op;

      // Scalar finite element spaces for stored Tdof-to-and-from-LDof maps.
      std::unique_ptr<FiniteElementSpace> fes_ho_scalar;
      std::unique_ptr<FiniteElementSpace> fes_lor_scalar;
      // Element Assembled mixed mass
      Vector M_LH_ea;
      // Element Assembled lumped M_L inverse built via EA. Stores diagonal as a Ldof vector.
      Vector ML_inv_ea;

#ifdef MFEM_USE_MPI
      std::unique_ptr<ParFiniteElementSpace> pfes_ho_scalar;
      std::unique_ptr<ParFiniteElementSpace> pfes_lor_scalar;
      Vector RML_inv;
#endif

      friend class L2ProjectionL2Space;
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
                            bool force_l2_space_ = false,
                            MemoryType d_mt_ = Device::GetHostMemoryType()) //move to method
      : GridTransfer(coarse_fes_, fine_fes_),
        F(NULL), B(NULL), force_l2_space(force_l2_space_)
   { }
   virtual ~L2ProjectionGridTransfer();

   const Operator &ForwardOperator() override;

   const Operator &BackwardOperator() override;

   bool SupportsBackwardsOperator() const override;
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
       If both spaces' FE collection pointers are pointing to the same
       collection we assume that the grid was refined while keeping the order
       constant. If the FE collections are different, it is assumed that both
       spaces have are using the same mesh. If the first element of the
       high-order space is a `TensorBasisElement`, the optimized tensor-product
       transfers are used. If not, the general transfers used. */
   TransferOperator(const FiniteElementSpace& lFESpace,
                    const FiniteElementSpace& hFESpace);

   /// Destructor
   virtual ~TransferOperator();

   /// @brief Interpolation or prolongation of a vector \p x corresponding to
   /// the coarse space to the vector \p y corresponding to the fine space.
   void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The vector \p x corresponding to the fine space is restricted to the
       vector \p y corresponding to the coarse space. */
   void MultTranspose(const Vector& x, Vector& y) const override;
};

/// Matrix-free transfer operator between finite element spaces on the same mesh
class PRefinementTransferOperator : public Operator
{
private:
   const FiniteElementSpace& lFESpace;
   const FiniteElementSpace& hFESpace;
   bool isvar_order;

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

   /// @brief Interpolation or prolongation of a vector \p x corresponding to
   /// the coarse space to the vector \p y corresponding to the fine space.
   void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The vector \p x corresponding to the fine space is restricted to the
   vector \p y corresponding to the coarse space. */
   void MultTranspose(const Vector& x, Vector& y) const override;
};

/// @brief Matrix-free transfer operator between finite element spaces on the
/// same mesh exploiting the tensor product structure of the finite elements
class TensorProductPRefinementTransferOperator : public Operator
{
private:
   const FiniteElementSpace& lFESpace;
   const FiniteElementSpace& hFESpace;
   int dim;
   int NE;
   int D1D;
   int Q1D;
   Array<real_t> B;
   Array<real_t> Bt;
   const Operator* elem_restrict_lex_l;
   const Operator* elem_restrict_lex_h;
   Vector mask;
   mutable Vector localL;
   mutable Vector localH;

public:
   /// @brief Constructs a transfer operator from \p lFESpace to \p hFESpace
   /// which have different FE collections.
   /** No matrices are assembled, only the action to a vector is being computed.
   The underlying finite elements need to be of type `TensorBasisElement`. It is
   also assumed that all the elements in the spaces are of the same type. */
   TensorProductPRefinementTransferOperator(
      const FiniteElementSpace& lFESpace_,
      const FiniteElementSpace& hFESpace_);

   /// Destructor
   virtual ~TensorProductPRefinementTransferOperator();

   /// @brief Interpolation or prolongation of a vector \p x corresponding to
   /// the coarse space to the vector \p y corresponding to the fine space.
   void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The vector \p x corresponding to the fine space is restricted to the
   vector \p y corresponding to the coarse space. */
   void MultTranspose(const Vector& x, Vector& y) const override;
};

/// @brief Matrix-free transfer operator between finite element spaces working
/// on true degrees of freedom
class TrueTransferOperator : public Operator
{
private:
   const FiniteElementSpace& lFESpace;
   const FiniteElementSpace& hFESpace;
   const Operator * P = nullptr;
   const SparseMatrix * R = nullptr;
   TransferOperator* localTransferOperator;
   mutable Vector tmpL;
   mutable Vector tmpH;

public:
   /// @brief Constructs a transfer operator working on true degrees of freedom
   /// from \p lFESpace to \p hFESpace
   TrueTransferOperator(const FiniteElementSpace& lFESpace_,
                        const FiniteElementSpace& hFESpace_);

   /// Destructor
   ~TrueTransferOperator();

   /// @brief Interpolation or prolongation of a true dof vector \p x to a true
   /// dof vector \p y.
   /** The true dof vector \p x corresponding to the coarse space is restricted
       to the true dof vector \p y corresponding to the fine space. */
   void Mult(const Vector& x, Vector& y) const override;

   /// Restriction by applying the transpose of the Mult method.
   /** The true dof vector \p x corresponding to the fine space is restricted to
       the true dof vector \p y corresponding to the coarse space. */
   void MultTranspose(const Vector& x, Vector& y) const override;
};

} // namespace mfem

#endif
