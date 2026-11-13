// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FUNCTIONAL_HPP
#define MFEM_FUNCTIONAL_HPP

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../general/communication.hpp"
#include "operator.hpp"
#include "blockoperator.hpp"
#include "multivector.hpp"
#include <memory>
#include <vector>

namespace mfem
{

namespace optimization
{

/** @brief Operator that computes a weighted sum and concatenates the results of
 * multiple operators of the same size, y = sum_i w_i A_i(x) */
class WeightedSumOperator : public Operator
{
   Array<const Operator *> ops;
   Vector w;
public:
   WeightedSumOperator(Array<const Operator *> operators, Vector weight);

   void SetWeight(Vector weight) { w = weight; }
   const Vector &GetWeight() const { return w; }

   Array<const Operator *> &GetOperators() { return ops; }
   const Array<const Operator *> &GetOperators() const { return ops; }

   void Mult(const Vector &x, Vector &y) const override;
};

/// @brief A BlockOperator that can also be applied to MultiVector%s with one
/// block per column block (input) and per row block (output),
/// y[i] = sum_j c_ij A_ij x[j].
class BlockMVOperator : public BlockOperator
{
   mutable Vector mv_tmp;
public:
   using BlockOperator::BlockOperator;

   /// The blocks A_ij are applied with their Mult() method; the blocks of
   /// @a y must have the sizes of the row blocks.
   void MultMV(const MultiVector &x, MultiVector &y) const override;

   /// The operator is linear: its gradient is the operator itself.
   Operator &GetGradientMV(const MultiVector &x) const override
   { return const_cast<BlockMVOperator&>(*this); }
};

/** @brief Abstract base class of a vector-valued functional f : X -> R^m with
    a few outputs, m << dim X.

    The input x is a MultiVector with one block per field; its sizes are
    GetFieldSizes(). In parallel, the blocks are distributed (e.g., true dofs)
    and the outputs are replicated on all ranks. Typical examples are the
    objective and the global constraints of an optimization problem.

    - Eval(x, fx): fx_i = f_i(x), i = 0, ..., m-1.
    - ComponentDerivative(x, i, df_i): df_i = f_i'(x), with one block per
      field; Derivative(x, df) evaluates all components.
    - ComponentHessian(x, k, i, j): the block d^2 f_k / dx_i dx_j, an Operator
      from field j to field i, or nullptr when the block is zero.
    - ComponentHessian(x, k): f_k''(x) on all fields, or nullptr when f_k is
      linear. The returned Operator supports Mult() on the concatenated input
      and MultMV() with one block per field.
    - WeightedSumHessian(x, lambda, i, j) and WeightedSumHessian(x, lambda):
      the same for sum_k lambda_k f_k.

    As in Operator::GetGradient(), the Hessians are returned non-const: they
    are owned by the functional and stay valid until the next call with the
    same indices, unless a derived class documents otherwise. */
class GlobalFunctional
{
#ifdef MFEM_USE_MPI
   MPI_Comm comm = MPI_COMM_NULL;
#endif
   // Hessians built by the default implementations: one per output and one
   // for the weighted sum on all fields, and the weighted sum blocks.
   mutable Array<int> hess_offsets;
   mutable std::vector<std::unique_ptr<BlockMVOperator>> full_hess;
   mutable std::unique_ptr<BlockMVOperator> full_sum_hess;
   mutable std::vector<std::unique_ptr<Operator>> block_sum_hess;
protected:
   Array<int> field_sizes;
   int height;
public:
   /// @brief Construct a functional with given field sizes and output size.
   GlobalFunctional(Array<int> field_sizes_, int output_size)
      : field_sizes(std::move(field_sizes_)), height(output_size) {}
   GlobalFunctional(int field_size, int output_size)
      : GlobalFunctional(Array<int>({field_size}), output_size) {}
#ifdef MFEM_USE_MPI
   GlobalFunctional(MPI_Comm comm_, Array<int> field_sizes_, int output_size);
   GlobalFunctional(MPI_Comm comm_, int field_size, int output_size)
      : GlobalFunctional(comm_, Array<int>({field_size}), output_size) {}
   void SetComm(MPI_Comm comm_) { comm = comm_; }
   MPI_Comm GetComm() const { return comm; }
#endif
   virtual ~GlobalFunctional() = default;

   /// @brief Evaluate functional, fx[i] = f_i(x)
   virtual void Eval(const MultiVector &x, Vector &fx) const = 0;
   /// @brief Evaluate i-th component derivative, df_i = df_i/dx
   virtual void ComponentDerivative(const MultiVector &x, int i,
                                    MultiVector &df_i) const;
   /// @brief Evaluate derivative, df[i] = df_i/dx
   virtual void Derivative(const MultiVector &x, Array<MultiVector*> &df) const;
   /// @brief Return the block (i, j) of the Hessian of f_k(x), an Operator
   /// from field j to field i, or nullptr (zero block)
   virtual Operator *ComponentHessian(const MultiVector &x, int k,
                                      int i, int j) const;
   /// @brief Return the Hessian of f_k(x) or nullptr (linear)
   ///
   /// Default path is to collect the blocks ComponentHessian(x, k, i, j) in a
   /// BlockMVOperator. A derived class that overrides this function must return
   /// an Operator that supports Mult() on the concatenated input and MultMV()
   /// with one block per field.
   virtual Operator *ComponentHessian(const MultiVector &x, int k) const;

   /// @brief Return the block (i, j) of the Hessian of
   /// $ \sum_k \lambda_k f_k(x) $ or nullptr (zero block)
   ///
   /// Default path is to call ComponentHessian() for each component and return
   /// a WeightedSumOperator. Derived classes can override this function to provide
   /// a more efficient implementation (weighted sum integration).
   ///
   /// @note Since this relies on pointers returned by ComponentHessian(),
   /// the returned Operator is not guaranteed to be valid or correct after
   /// the next call to WeightedSumHessian() or ComponentHessian().
   ///
   /// @note Currently, there is no way to return an assembled sparse matrix
   /// for the weighted sum Hessian, because there is no unified interface for
   /// sparse matrix types. Later, a helper function can be added to assemble
   /// the weighted sum Hessian into a sparse matrix if all component Hessians
   /// are of the same type.
   virtual Operator *WeightedSumHessian(const MultiVector &x,
                                        const Vector &lambda,
                                        int i, int j) const;
   /// @brief Return the Hessian of $ \sum_k \lambda_k f_k(x) $ or nullptr (linear)
   ///
   /// Default path is to collect the blocks WeightedSumHessian(x, lambda, i, j)
   /// in a BlockMVOperator; see ComponentHessian(x, k).
   virtual Operator *WeightedSumHessian(const MultiVector &x,
                                        const Vector &lambda) const;

   /// @brief Return the output size of the functional, i.e., number of components.
   int Height() const { return height; }
   /// @brief Return the total input size of the functional, i.e., sum of field sizes.
   int Width() const { return field_sizes.Sum(); }

   /// @name Getter and setter for field sizes
   ///@{
   Array<int> &GetFieldSizes() { return field_sizes; }
   const Array<int> &GetFieldSizes() const { return field_sizes; }
   void SetFieldSizes(const Array<int> &sizes) { field_sizes = sizes; }
   ///@}

   /// @brief Return true if the functional has a non-null MPI communicator
   bool IsParallel() const;
   /// @brief Helper function to verify the input size of the functional.
   void VerifyInputSize(const Array<int> &sizes) const;

   /// @brief In-place reduction of a vector across all MPI ranks, if applicable.
   void Reduce(Vector &vec) const;

   real_t Reduce(real_t value) const;
   real_t Dot(const Vector &x, const Vector &y) const;

private:
   /// The block operator of @a blocks (row-major, one per field pair), built in
   /// @a hess unless it already holds these blocks; nullptr if all are null.
   Operator *CollectBlocks(const Array<Operator *> &blocks,
                           std::unique_ptr<BlockMVOperator> &hess) const;
};

/** @brief A global functional representing [w1*f1(x) + w2*f2(x); ...]
 * summation can be weighted, and the output can be concatenated.
 * A subset of the fields can be used for each functional
 *
 * @note Output size of the functional is determined in Finalize()
 * and any modification to the functional after Finalize() will abort.
 * Usage of an unfinalized WeightedSumFunctional will also abort.
 */
class WeightedSumFunctional : public GlobalFunctional
{
   bool finalized = false;
   Array<int> out_offsets;

   Array<int> block_row_idx;
   std::vector<Array<int>> field_ids;
   std::vector<Vector> weights;
   Array<const GlobalFunctional *> functionals;
   Array<const GlobalFunctional *> owned_functionals;

   mutable MultiVector work;

   // Hessian blocks that combine several terms: one per (output, field pair)
   // and one per field pair for the weighted sum.
   mutable Vector unit;
   mutable std::vector<std::unique_ptr<Operator>> component_hess;
   mutable std::vector<std::unique_ptr<Operator>> sum_hess;
public:
   WeightedSumFunctional(Array<int> field_sizes)
      : GlobalFunctional(std::move(field_sizes), 0)
      , out_offsets(1), work(this->field_sizes)
   { out_offsets[0] = 0; }
#ifdef MFEM_USE_MPI
   WeightedSumFunctional(MPI_Comm comm, Array<int> field_sizes)
      : GlobalFunctional(comm, std::move(field_sizes), 0)
      , out_offsets(1), work(this->field_sizes)
   { out_offsets[0] = 0; }
#endif
   ~WeightedSumFunctional() override;

   /// @brief Add a functional to the weighted sum
   /// If row index is greater than the current number of rows,
   /// then it will be appended.
   ///
   /// Empty weight vector means that the functional is added directly to the output vector.
   ///
   /// @a f must have its final sizes: adding a functional that still has to
   /// be finalized (e.g., another WeightedSumFunctional) is incorrect usage.
   void Add(int blk_row, Array<int> subfield_ids, Vector weight,
            const GlobalFunctional *f, bool own=false);

   void Finalize();

   void Eval(const MultiVector &x, Vector &fx) const override;

   void ComponentDerivative(const MultiVector &x, int i,
                            MultiVector &df_i) const override;

   using GlobalFunctional::ComponentHessian;
   using GlobalFunctional::WeightedSumHessian;

   /// @brief Return the block (i, j) of the Hessian of the @a k-th output.
   ///
   /// @note A functional added on different fields must return Hessians that
   /// stay valid over calls of its ComponentHessian().
   Operator *ComponentHessian(const MultiVector &x, int k,
                              int i, int j) const override;

   Operator *WeightedSumHessian(const MultiVector &x, const Vector &lambda,
                                int i, int j) const override;

private:
   /// Block (i, j) of the Hessian of sum_k lambda_k f_k, built in @a hess
   /// when it combines several blocks.
   Operator *HessianBlock(const MultiVector &x, const Vector &lambda,
                          int i, int j, std::unique_ptr<Operator> &hess) const;
};

/** @brief A functional @a f with essential dofs held fixed, similar to
    @ref ConstrainedOperator.

    The values are those of @a f. The derivatives are zero on the essential
    dofs. Every returned Hessian is the Hessian of @a f with the rows and
    columns of the essential dofs replaced by the identity, so that a Newton
    system H dx = -df gives dx = 0 there. This holds for each ComponentHessian()
    and each WeightedSumHessian() separately, for any weights, while a linear
    functional returns nullptr. A solver that adds the Hessians of k such
    functionals (e.g., an objective and its constraints) gets k on that
    diagonal; it is nonzero, but not one.

    The identity is set on the diagonal blocks (i, i) that @a f returns; a
    field whose diagonal block is zero gets no identity. */
class ConstrainedFunctional : public GlobalFunctional
{
   const GlobalFunctional *f;
   bool own = false;
   // Not owned; one list per field, nullptr for none. The lists must not
   // change during the lifetime of this functional.
   std::vector<const Array<int> *> ess_dofs;
   const Array<int> no_dofs; // for the fields without essential dofs
   // The constrained Hessian blocks and the blocks of f that they wrap: one
   // per (output, field pair), then one per field pair for the weighted sum.
   mutable std::vector<std::unique_ptr<Operator>> constrained_hessians;
   mutable std::vector<const Operator *> wrapped_hessians;
public:
   ConstrainedFunctional(std::vector<const Array<int>*> ess_dofs_,
                         const GlobalFunctional *f_, bool own_ = false);

   ~ConstrainedFunctional() override { if (own) { delete f; } }

   void Eval(const MultiVector &x, Vector &fx) const override
   { f->Eval(x, fx); }

   void ComponentDerivative(const MultiVector &x, int i,
                            MultiVector &df_i) const override;

   using GlobalFunctional::ComponentHessian;
   using GlobalFunctional::WeightedSumHessian;

   Operator *ComponentHessian(const MultiVector &x, int k,
                              int i, int j) const override;

   Operator *WeightedSumHessian(const MultiVector &x, const Vector &lambda,
                                int i, int j) const override;
   const GlobalFunctional *GetFunctional() const { return f; }

private:
   /// The block (i, j) @a hess with the essential dofs constrained, rebuilt
   /// in @a slot only when @a hess differs from the block wrapped there.
   Operator *Constrain(Operator *hess, int i, int j, int slot) const;
};

class LinearFunctional : public GlobalFunctional
{
   const Vector *b;
   bool own = false;
public:
   LinearFunctional(const Vector *b_, bool own_ = false);

#ifdef MFEM_USE_MPI
   LinearFunctional(MPI_Comm comm, const Vector *b_, bool own_ = false);
#endif
   ~LinearFunctional() override { if (own) { delete b; } }
   void Eval(const MultiVector &x, Vector &fx) const override;
   void ComponentDerivative(const MultiVector &x, int i,
                            MultiVector &df_i) const override;
   using GlobalFunctional::ComponentHessian;
   Operator *ComponentHessian(const MultiVector &x, int k,
                              int i, int j) const override;
};

/// @brief Quadratic functional f(x) = 0.5 * x^T A x - b^T x for a symmetric A
///
/// @note We do not check if A is symmetric.
///
/// @note The Hessian is A itself, not a copy: it must not be modified.
class QuadraticFunctional : public GlobalFunctional
{
protected:
   const Operator *A;
   bool own = false;
   const Vector *b = nullptr; // optional linear term
   bool own_b = false;
   mutable Vector work; // temporary vector for Mult
public:
   /// @brief Construct a quadratic functional with given symmetric matrix A
   QuadraticFunctional(const Operator *A_, const Vector *b_ = nullptr,
                       bool own_ = false, bool own_b_ = false);
#ifdef MFEM_USE_MPI
   QuadraticFunctional(MPI_Comm comm, const Operator *A_,
                       const Vector *b_ = nullptr,
                       bool own_ = false, bool own_b_ = false);
#endif

   ~QuadraticFunctional() override;

   void Eval(const MultiVector &x, Vector &fx) const override;

   void ComponentDerivative(const MultiVector &x, int i,
                            MultiVector &df_i) const override;

   using GlobalFunctional::ComponentHessian;
   Operator *ComponentHessian(const MultiVector &x, int k,
                              int i, int j) const override;
};

/** @brief Compare the derivative of @a f at @a x in the direction @a d with
    the central difference (f(x + h d) - f(x - h d)) / (2h).

    Returns max_i |f_i'(x) d - (f_i(x + h d) - f_i(x - h d)) / (2h)|. */
real_t DiffDerivativeAgainstCentralDiff(const GlobalFunctional &f,
                                        const MultiVector &x,
                                        const MultiVector &d, real_t h);

} // namespace optimization

} // namespace mfem

#endif // MFEM_FUNCTIONAL_HPP
