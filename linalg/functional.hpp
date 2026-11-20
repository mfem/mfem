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

/// @brief y = sum_j P_j^T A_j P_j x, where P_j extracts the entries
/// @a indices[j] of x, or all entries when @a indices[j] is nullptr.
class SubBlockSumOperator : public Operator
{
   Array<const Operator *> ops;
   std::vector<const Array<int> *> indices;
   mutable Vector xs, ys;
public:
   SubBlockSumOperator(int n, Array<const Operator *> operators,
                       std::vector<const Array<int> *> indices_);

   void Mult(const Vector &x, Vector &y) const override;
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
    - ComponentHessian(x, i): f_i''(x) as an Operator on the concatenated
      input, or nullptr when f_i is linear.
    - WeightedSumHessian(x, lambda): (sum_i lambda_i f_i)''(x), or nullptr
      when all components are linear.

    The returned Hessians are owned by the functional. See the documentation
    of each function for how long they stay valid. */
class GlobalFunctional
{
#ifdef MFEM_USE_MPI
   MPI_Comm comm = MPI_COMM_NULL;
#endif
   mutable std::unique_ptr<Operator> contracted_hess;
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
   /// @brief Return the Hessian of f_i(x) or nullptr (linear)
   virtual const Operator *ComponentHessian(const MultiVector &x, int i) const;

   /// @brief Return the Hessian of $ \sum_i \lambda_i f_i(x) $ or nullptr (linear)
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
   virtual const Operator *WeightedSumHessian(const MultiVector &x,
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

   // Hessians: terms with the same functional and fields form one group.
   struct HessianGroup
   {
      const GlobalFunctional *f;
      Array<int> subfield_ids;
      Array<int> indices; // entries of the concatenated input
      bool all_fields;    // subfield_ids == (0, 1, ..., NumFields-1)
      mutable Vector mu;  // weights of the components of f
   };
   std::vector<HessianGroup> groups;
   Array<int> term_group;
   mutable Vector unit;
   mutable std::vector<std::unique_ptr<Operator>> component_hess;
   mutable std::unique_ptr<Operator> sum_hess;
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

   /// @brief Return the Hessian of the @a i-th output.
   ///
   /// @note A functional added on different fields must return Hessians that
   /// stay valid over calls of its ComponentHessian() and
   /// WeightedSumHessian().
   const Operator* ComponentHessian(const MultiVector &x, int i) const override;

   const Operator *WeightedSumHessian(const MultiVector &x,
                                      const Vector &lambda) const override;

private:
   /// Hessian of sum_i lambda_i f_i, built in @a hess when it combines
   /// several Hessians.
   const Operator *Hessian(const MultiVector &x, const Vector &lambda,
                           std::unique_ptr<Operator> &hess) const;
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
    diagonal; it is nonzero, but not one. */
class ConstrainedFunctional : public GlobalFunctional
{
   const GlobalFunctional *f;
   bool own = false;
   // Not owned; one list per field, nullptr for none. The lists must not
   // change during the lifetime of this functional.
   std::vector<const Array<int> *> ess_dofs;
   Array<int> ess_tdofs; // ess_dofs in the concatenated input numbering
   // The constrained Hessians and the Hessians of f that they wrap: one per
   // component, and one for the weighted sum.
   mutable std::vector<std::unique_ptr<ConstrainedOperator>> constrained_hessians;
   mutable std::vector<const Operator *> wrapped_hessians;
   mutable std::unique_ptr<ConstrainedOperator> constrained_sum_hessian;
   mutable const Operator *wrapped_sum_hessian = nullptr;
public:
   ConstrainedFunctional(std::vector<const Array<int>*> ess_dofs_,
                         const GlobalFunctional *f_, bool own_ = false);

   ~ConstrainedFunctional() override { if (own) { delete f; } }

   void Eval(const MultiVector &x, Vector &fx) const override
   { f->Eval(x, fx); }

   void ComponentDerivative(const MultiVector &x, int i,
                            MultiVector &df_i) const override;

   const Operator* ComponentHessian(const MultiVector &x, int i) const override;

   const Operator *WeightedSumHessian(const MultiVector &x,
                                      const Vector &lambda) const override;
   const GlobalFunctional *GetFunctional() const { return f; }

private:
   /// @a hess with the identity on the essential dofs, rebuilt in
   /// @a constrained only when @a hess differs from @a wrapped.
   const Operator *Constrain(const Operator *hess, const Operator *&wrapped,
                             std::unique_ptr<ConstrainedOperator> &constrained) const;
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
   const Operator *ComponentHessian(const MultiVector &x, int i) const override;
};

/// @brief Quadratic functional f(x) = 0.5 * x^T A x - b^T x for a symmetric A
///
/// @note We do not check if A is symmetric.
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

   const Operator *ComponentHessian(const MultiVector &x, int i) const override;
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
