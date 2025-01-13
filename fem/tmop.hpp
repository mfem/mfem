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

#ifndef MFEM_TMOP_HPP
#define MFEM_TMOP_HPP

#include "../linalg/invariants.hpp"
#include "nonlininteg.hpp"

namespace mfem
{

/** @brief Abstract class for local mesh quality metrics in the target-matrix
    optimization paradigm (TMOP) by P. Knupp et al. */
class TMOP_QualityMetric : public HyperelasticModel
{
protected:
   const DenseMatrix *Jtr; /**< Jacobian of the reference-element to
                                target-element transformation. */

   /** @brief The method HyperelasticModel::SetTransformation() is hidden
       for TMOP_QualityMetric%s, because it is not used. */
   void SetTransformation(ElementTransformation &) { }

public:
   TMOP_QualityMetric() : Jtr(NULL) { }
   virtual ~TMOP_QualityMetric() { }

   /** @brief Specify the reference-element -> target-element Jacobian matrix
       for the point of interest.

       The specified Jacobian matrix, #Jtr, can be used by metrics that cannot
       be written just as a function of the target->physical Jacobian matrix,
       Jpt. */
   virtual void SetTargetJacobian(const DenseMatrix &Jtr_) { Jtr = &Jtr_; }

   /** @brief Evaluates the metric in matrix form (opposed to invariant form).
       Used for validating the invariant evaluations. */
   virtual real_t EvalWMatrixForm(const DenseMatrix &Jpt) const
   { return -1.0; /* not implemented -> checks would fail. */ }

   /** @brief Evaluate the strain energy density function, W = W(Jpt), by using
       the 2D or 3D matrix invariants, see linalg/invariants.hpp.
       @param[in] Jpt  Represents the target->physical transformation
                       Jacobian matrix. */
   virtual real_t EvalW(const DenseMatrix &Jpt) const = 0;

   /** @brief Evaluate the 1st Piola-Kirchhoff stress tensor, P = P(Jpt).
       @param[in] Jpt  Represents the target->physical transformation
                       Jacobian matrix.
       @param[out]  P  The evaluated 1st Piola-Kirchhoff stress tensor. */
   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const = 0;

   /** @brief Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor
       and assemble its contribution to the local gradient matrix 'A'.
       @param[in] Jpt     Represents the target->physical transformation
                          Jacobian matrix.
       @param[in] DS      Gradient of the basis matrix (dof x dim).
       @param[in] weight  Quadrature weight coefficient for the point.
       @param[in,out]  A  Local gradient matrix where the contribution from this
                          point will be added.

       Computes weight * d(dW_dxi)_d(xj) at the current point, for all i and j,
       where x1 ... xn are the FE dofs. This function is usually defined using
       the matrix invariants and their derivatives. */
   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const real_t weight, DenseMatrix &A) const = 0;

   /** @brief Return the metric ID. */
   virtual int Id() const { return 0; }
};

class TargetConstructor;

/// Abstract class used to define explicit combination of metrics with constant
/// coefficients.
class TMOP_Combo_QualityMetric : public TMOP_QualityMetric
{
protected:
   Array<TMOP_QualityMetric *> tmop_q_arr; //the metrics are not owned
   Array<real_t> wt_arr;

public:
   virtual void AddQualityMetric(TMOP_QualityMetric *tq, real_t wt = 1.0)
   {
      tmop_q_arr.Append(tq);
      wt_arr.Append(wt);
   }

   void SetTargetJacobian(const DenseMatrix &Jtr_) override
   {
      for (int i = 0; i < tmop_q_arr.Size(); i++)
      {
         tmop_q_arr[i]->SetTargetJacobian(Jtr_);
      }
   }

   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   /// Computes the averages of all metrics (integral of metric / volume).
   /// Works in parallel when called with a ParGridFunction.
   void ComputeAvgMetrics(const GridFunction &nodes,
                          const TargetConstructor &tc,
                          Vector &averages) const;

   /// Computes weights so that the averages of all metrics are equal, and the
   /// weights sum to one. Works in parallel when called with a ParGridFunction.
   void ComputeBalancedWeights(const GridFunction &nodes,
                               const TargetConstructor &tc,
                               Vector &weights) const;

   void GetWeights(Array<real_t> &weights) const { weights = wt_arr; }

   /// Changes the weights of the metrics in the combination.
   void SetWeights(const Vector &weights)
   {
      MFEM_VERIFY(tmop_q_arr.Size() == weights.Size(), "Incorrect #weights");
      for (int i = 0; i < tmop_q_arr.Size(); i++) { wt_arr[i] = weights(i); }
   }
};

/// Simultaneous Untangler + Worst Case Improvement Metric
/// Uses a base metric mu and is defined as:
/// mu_tilde = mu_hat,                 when WorstCaseType = None,
///          = mu_hat/(beta - mu_hat), when WorstCaseType = Beta,
///          = mu_hat^p,               when WorstCaseType = PMean,
/// where beta = max(mu_hat) + muT_ep,
/// and mu_hat = (mu/2phi(tau,ep)) where
/// 2phi(tau,ep) = 1, when                                 when BarrierType = None,
///             = 2*(tau - min(alpha*min(tau)-detT_ep,0)), when BarrierType = Shifted
///             = tau^2 + sqrt(tau^2 + ep^2),              when BarrierType = Pseudo
/// where tau = det(T), and max(mu_hat) and min(tau) are computed over the
/// entire mesh.
/// Ultimately, this metric can be used for mesh untangling with the BarrierType
/// option and for worst case quality improvement with the WorstCaseType option.
class TMOP_WorstCaseUntangleOptimizer_Metric : public TMOP_QualityMetric
{
public:
   enum class BarrierType
   {
      None,
      Shifted,
      Pseudo
   };
   enum class WorstCaseType
   {
      None,
      Beta,
      PMean
   };

protected:
   TMOP_QualityMetric &tmop_metric; // non-barrier metric to use
   real_t min_detT;                 // minimum Jacobian in the mesh
   real_t max_muT;                  // max mu_k/phi(tau,ep) in the mesh
   int exponent;                    // used for p-mean metrics
   real_t alpha;                    // scaling factor for min(det(T))
   real_t detT_ep;                  // small constant subtracted from min(detT)
   real_t muT_ep;                   // small constant added to muT term
   BarrierType btype;
   WorstCaseType wctype;

public:
   TMOP_WorstCaseUntangleOptimizer_Metric(TMOP_QualityMetric &tmop_metric_,
                                          int exponent_ = 1,
                                          real_t alpha_ = 1.5,
                                          real_t detT_ep_ = 0.0001,
                                          real_t muT_ep_ = 0.0001,
                                          BarrierType btype_ = BarrierType::None,
                                          WorstCaseType wctype_ = WorstCaseType::None) :
      tmop_metric(tmop_metric_), exponent(exponent_), alpha(alpha_),
      detT_ep(detT_ep_), muT_ep(muT_ep_), btype(btype_), wctype(wctype_)
   {
      MFEM_VERIFY(wctype == WorstCaseType::None,
                  "Worst-case optimization has not been fully developed!");
      if (btype != BarrierType::None)
      {
         const int m_id = tmop_metric.Id();
         MFEM_VERIFY(m_id == 4 || m_id == 14 || m_id == 66,
                     "Incorrect input barrier metric --  must be 4 / 14 / 66");
      }
   }

   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override
   { MFEM_ABORT("Not implemented"); }

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override
   { MFEM_ABORT("Not implemented"); }

   // Compute mu_hat.
   virtual real_t EvalWBarrier(const DenseMatrix &Jpt) const;

   virtual void SetMinDetT(real_t min_detT_) { min_detT = min_detT_; }

   virtual void SetMaxMuT(real_t max_muT_) { max_muT = max_muT_; }

   virtual BarrierType GetBarrierType() { return btype; }

   virtual WorstCaseType GetWorstCaseType() { return wctype; }
};

/// 0 metric
class TMOP_Metric_000 : public TMOP_QualityMetric
{
public:
   // W = 0.
   virtual real_t EvalW(const DenseMatrix &Jpt) const {return 0.0;}

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const {P = 0.0;}

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const real_t weight, DenseMatrix &A) const {A = 0.0;}

   virtual int Id() const { return 0; }
};

/// 2D non-barrier metric without a type.
class TMOP_Metric_001 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   // W = |J|^2.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 1; }
};

/// 2D non-barrier Skew metric.
class TMOP_Metric_skew2D : public TMOP_QualityMetric
{
public:
   // W = 0.5 (1 - cos(angle_Jpr - angle_Jtr)).
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override
   { MFEM_ABORT("Not implemented"); }

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override
   { MFEM_ABORT("Not implemented"); }
};

/// 3D non-barrier Skew metric.
class TMOP_Metric_skew3D : public TMOP_QualityMetric
{
public:
   // W = 1/6 (3 - sum_i cos(angle_Jpr_i - angle_Jtr_i)), i = 1..3.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override
   { MFEM_ABORT("Not implemented"); }

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override
   { MFEM_ABORT("Not implemented"); }
};

/// 2D non-barrier Aspect ratio metric.
class TMOP_Metric_aspratio2D : public TMOP_QualityMetric
{
public:
   // W = 0.5 (ar_Jpr/ar_Jtr + ar_Jtr/ar_Jpr) - 1.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override
   { MFEM_ABORT("Not implemented"); }

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override
   { MFEM_ABORT("Not implemented"); }
};

/// 3D non-barrier Aspect ratio metric.
class TMOP_Metric_aspratio3D : public TMOP_QualityMetric
{
public:
   // W = 1/3 sum [0.5 (ar_Jpr_i/ar_Jtr_i + ar_Jtr_i/ar_Jpr_i) - 1], i = 1..3.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override
   { MFEM_ABORT("Not implemented"); }

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override
   { MFEM_ABORT("Not implemented"); }
};

/// 2D barrier shape (S) metric (polyconvex).
/// Grade - A.
class TMOP_Metric_002 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   // W = 0.5 |J|^2 / det(J) - 1.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = 0.5 I1b - 1.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 2; }
};

/// 2D non-barrier shape (S) metric.
/// Grade - F.
class TMOP_Metric_004 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   // W = |J|^2 - 2*det(J)
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 4; }
};

/// 2D barrier Shape+Size (VS) metric (not polyconvex).
class TMOP_Metric_007 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   // W = |J - J^-t|^2.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 7; }
};

/// 2D barrier Shape+Size (VS) metric (not polyconvex).
class TMOP_Metric_009 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   // W = det(J) * |J - J^-t|^2.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// 2D non-barrier Shape+Size+Orientation (VOS) metric (polyconvex).
class TMOP_Metric_014 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   // W = |J - I|^2.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = I1[J-I].
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// 2D Shifted barrier form of shape metric (mu_2).
class TMOP_Metric_022 : public TMOP_QualityMetric
{
protected:
   real_t &min_detT;
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   TMOP_Metric_022(real_t &t0): min_detT(t0) {}

   // W = 0.5(|J|^2 - 2det(J)) / (det(J) - tau0).
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// 2D barrier shape metric (polyconvex).
/// Grade - A.
class TMOP_Metric_050 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   // W = 0.5 |J^t J|^2 / det(J)^2 - 1.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = 0.5 I1b^2 - 2.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// 2D non-barrier size (V) metric (not polyconvex).
/// Grade - F.
class TMOP_Metric_055 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   // W = (det(J) - 1)^2.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

};

/// 2D barrier size (V) metric (polyconvex).
/// Grade - C.
class TMOP_Metric_056 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   // W = 0.5 (det(J) + 1 / det(J)) - 1.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = 0.5 (I2b + 1/I2b) - 1.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// 2D barrier shape (S) metric (not polyconvex).
class TMOP_Metric_058 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   // W = |J^t J|^2 / det(J)^2 - 2|J|^2 / det(J) + 2
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = I1b (I1b - 2).
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// 2D non-barrier Shape+Size (VS) metric.
/// Grade - F.
class TMOP_Metric_066 : public TMOP_Combo_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;
   TMOP_QualityMetric *sh_metric, *sz_metric;

public:
   TMOP_Metric_066(real_t gamma)
      : sh_metric(new TMOP_Metric_004), sz_metric(new TMOP_Metric_055)
   {
      // (1-gamma) mu_4 + gamma mu_55
      AddQualityMetric(sh_metric, 1.-gamma);
      AddQualityMetric(sz_metric, gamma);
   }
   int Id() const override { return 66; }
   real_t GetGamma() const { return wt_arr[1]; }

   virtual ~TMOP_Metric_066() { delete sh_metric; delete sz_metric; }
};

/// 2D barrier size (V) metric (polyconvex).
/// Grade - C.
class TMOP_Metric_077 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   // W = 0.5 (det(J) - 1 / det(J))^2.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = 0.5 (I2 + 1 / I2) - 1.0.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 77; }
};

/// 2D barrier Shape+Size (VS) metric (polyconvex).
/// Grade - A.
class TMOP_Metric_080 : public TMOP_Combo_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;
   TMOP_QualityMetric *sh_metric, *sz_metric;

public:
   TMOP_Metric_080(real_t gamma)
      : sh_metric(new TMOP_Metric_002), sz_metric(new TMOP_Metric_077)
   {
      // (1-gamma) mu_2 + gamma mu_77
      AddQualityMetric(sh_metric, 1.0 - gamma);
      AddQualityMetric(sz_metric, gamma);
   }

   int Id() const override { return 80; }
   real_t GetGamma() const { return wt_arr[1]; }

   virtual ~TMOP_Metric_080() { delete sh_metric; delete sz_metric; }
};

/// 2D barrier Shape+Orientation (OS) metric (polyconvex).
class TMOP_Metric_085 : public TMOP_QualityMetric
{
public:
   // W = |T-T'|^2, where T'= |T|*I/sqrt(2).
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override
   { MFEM_ABORT("Not implemented"); }

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override
   { MFEM_ABORT("Not implemented"); }
};

/// 2D compound barrier Shape+Size (VS) metric (balanced).
class TMOP_Metric_090 : public TMOP_Combo_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;
   TMOP_QualityMetric *sh_metric, *sz_metric;

public:
   TMOP_Metric_090()
      : sh_metric(new TMOP_Metric_050), sz_metric(new TMOP_Metric_077)
   {
      // mu_50 + lambda mu_77.
      // 1 <= lambda <= 4 should produce best asymptotic balance.
      AddQualityMetric(sh_metric, 1.0);
      AddQualityMetric(sz_metric, 2.5);
   }

   int Id() const override { return 90; }
   virtual ~TMOP_Metric_090() { delete sh_metric; delete sz_metric; }
};

/// 2D compound barrier Shape+Size (VS) metric (balanced).
class TMOP_Metric_094 : public TMOP_Combo_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;
   TMOP_QualityMetric *sh_metric, *sz_metric;

public:
   TMOP_Metric_094()
      : sh_metric(new TMOP_Metric_002), sz_metric(new TMOP_Metric_056)
   {
      // mu_2 + lambda mu_56.
      // 1 <= lambda <= 2 should produce best asymptotic balance.
      AddQualityMetric(sh_metric, 1.0);
      AddQualityMetric(sz_metric, 1.5);
   }

   int Id() const override { return 94; }
   virtual ~TMOP_Metric_094() { delete sh_metric; delete sz_metric; }
};

/// 2D barrier Shape+Size+Orientation (VOS) metric (polyconvex).
class TMOP_Metric_098 : public TMOP_QualityMetric
{
public:
   // W = 1/tau |T-I|^2.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override
   { MFEM_ABORT("Not implemented"); }

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override
   { MFEM_ABORT("Not implemented"); }
};

/// 2D untangling metric.
class TMOP_Metric_211 : public TMOP_QualityMetric
{
protected:
   const real_t eps;
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   TMOP_Metric_211(real_t epsilon = 1e-4) : eps(epsilon) { }

   // W = (det(J) - 1)^2 - det(J) + sqrt(det(J)^2 + eps).
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// Shifted barrier form of metric 56 (area, ideal barrier metric), 2D
class TMOP_Metric_252 : public TMOP_QualityMetric
{
protected:
   real_t &tau0;
   mutable InvariantsEvaluator2D<real_t> ie;

public:
   /// Note that @a t0 is stored by reference
   TMOP_Metric_252(real_t &t0): tau0(t0) {}

   // W = 0.5(det(J) - 1)^2 / (det(J) - tau0).
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// 3D barrier Shape (S) metric, well-posed (polyconvex & invex).
class TMOP_Metric_301 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // W = 1/3 |J| |J^-1| - 1.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = 1/3 sqrt(I1b * I2b) - 1
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// 3D barrier Shape (S) metric, well-posed (polyconvex & invex).
class TMOP_Metric_302 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // W = |J|^2 |J^{-1}|^2 / 9 - 1.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = I1b * I2b / 9 - 1.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 302; }
};

/// 3D barrier Shape (S) metric, well-posed (polyconvex & invex).
class TMOP_Metric_303 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // W = |J|^2 / 3 / det(J)^(2/3) - 1.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = I1b / 3 - 1.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 303; }
};

/// 3D barrier Shape (S) metric, well-posed (polyconvex & invex).
class TMOP_Metric_304 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // W = |J|^3 / 3^(3/2) / det(J) - 1.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = (I1b/3)^3/2 - 1.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 304; }
};

/// 3D Size (V) untangling metric.
class TMOP_Metric_311 : public TMOP_QualityMetric
{
protected:
   const real_t eps;
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   TMOP_Metric_311(real_t epsilon = 1e-4) : eps(epsilon) { }

   // W = (det(J) - 1)^2 - det(J)  + (det(J)^2 + eps)^(1/2).
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// 3D Shape (S) metric, untangling version of 303.
class TMOP_Metric_313 : public TMOP_QualityMetric
{
protected:
   real_t &min_detT;
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   TMOP_Metric_313(real_t &mindet) : min_detT(mindet) { }

   // W = 1/3 |J|^2 / [det(J)-tau0]^(-2/3).
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 313; }
};

/// 3D Size (V) metric.
class TMOP_Metric_315 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // W = (det(J) - 1)^2.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 315; }
};

/// 3D Size (V) metric.
class TMOP_Metric_316 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // W = 0.5 (det(J) + 1/det(J)) - 1.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = 0.5 (I3b + 1/I3b) - 1.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// 3D Size (V) metric.
class TMOP_Metric_318 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // W = 0.5 (det(J)^2 + 1/det(J)^2) - 1.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = 0.5 (I3 + 1/I3) - 1.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 318; }
};

/// 3D barrier Shape+Size (VS) metric, well-posed (invex).
class TMOP_Metric_321 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // W = |J - J^-t|^2.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = I1 + I2/I3 - 6.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 321; }
};

/// 3D barrier Shape+Size (VS) metric, well-posed (invex).
class TMOP_Metric_322 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // W = |J - adjJ^-t|^2.
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = I1b / (I3b^-1/3) / 6 + I2b (I3b^1/3) / 6 - 1
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 322; }
};

/// 3D barrier Shape+Size (VS) metric, well-posed (invex).
class TMOP_Metric_323 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // W = |J|^3 - 3 sqrt(3) ln(det(J)) - 3 sqrt(3).
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = I1^3/2 - 3 sqrt(3) ln(I3b) - 3 sqrt(3).
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 323; }
};

/// 3D compound barrier Shape+Size (VS) metric (polyconvex, balanced).
class TMOP_Metric_328 : public TMOP_Combo_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;
   TMOP_QualityMetric *sh_metric, *sz_metric;

public:
   TMOP_Metric_328()
      : sh_metric(new TMOP_Metric_301), sz_metric(new TMOP_Metric_316)
   {
      // lambda mu_301 + mu_316.
      // 3/8 <= lambda <= 9/8 should produce best asymptotic balance.
      AddQualityMetric(sh_metric, 0.75);
      AddQualityMetric(sz_metric, 1.0);
   }

   int Id() const override { return 328; }
   virtual ~TMOP_Metric_328() { delete sh_metric; delete sz_metric; }
};

/// 3D compound barrier Shape+Size (VS) metric (polyconvex).
class TMOP_Metric_332 : public TMOP_Combo_QualityMetric
{
protected:
   TMOP_QualityMetric *sh_metric, *sz_metric;

public:
   TMOP_Metric_332(real_t gamma)
      : sh_metric(new TMOP_Metric_302), sz_metric(new TMOP_Metric_315)
   {
      // (1-gamma) mu_302 + gamma mu_315
      AddQualityMetric(sh_metric, 1.-gamma);
      AddQualityMetric(sz_metric, gamma);
   }

   int Id() const override { return 332; }
   real_t GetGamma() const { return wt_arr[1]; }

   virtual ~TMOP_Metric_332() { delete sh_metric; delete sz_metric; }
};

/// 3D barrier Shape+Size (VS) metric, well-posed (polyconvex).
class TMOP_Metric_333 : public TMOP_Combo_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;
   TMOP_QualityMetric *sh_metric, *sz_metric;

public:
   TMOP_Metric_333(real_t gamma)
      : sh_metric(new TMOP_Metric_302), sz_metric(new TMOP_Metric_316)
   {
      // (1-gamma) mu_302 + gamma mu_316
      AddQualityMetric(sh_metric, 1.-gamma);
      AddQualityMetric(sz_metric, gamma);
   }

   virtual ~TMOP_Metric_333() { delete sh_metric; delete sz_metric; }
};

/// 3D barrier Shape+Size (VS) metric, well-posed (polyconvex).
class TMOP_Metric_334 : public TMOP_Combo_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;
   TMOP_QualityMetric *sh_metric, *sz_metric;

public:
   TMOP_Metric_334(real_t gamma)
      : sh_metric(new TMOP_Metric_303), sz_metric(new TMOP_Metric_316)
   {
      // (1-gamma) mu_303 + gamma mu_316
      AddQualityMetric(sh_metric, 1.-gamma);
      AddQualityMetric(sz_metric, gamma);
   }

   int Id() const override { return 334; }
   real_t GetGamma() const { return wt_arr[1]; }

   virtual ~TMOP_Metric_334() { delete sh_metric; delete sz_metric; }
};

/// 3D compound barrier Shape+Size (VS) metric (polyconvex, balanced).
class TMOP_Metric_338 : public TMOP_Combo_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;
   TMOP_QualityMetric *sh_metric, *sz_metric;

public:
   TMOP_Metric_338()
      : sh_metric(new TMOP_Metric_302), sz_metric(new TMOP_Metric_318)
   {
      // mu_302 + lambda mu_318.
      // 4/9 <= lambda <= 3 should produce best asymptotic balance.
      AddQualityMetric(sh_metric, 1.0);
      AddQualityMetric(sz_metric, 0.5 * (4.0/9.0 + 3.0));
   }

   int Id() const override { return 338; }
   virtual ~TMOP_Metric_338() { delete sh_metric; delete sz_metric; }
};

/// 3D barrier Shape+Size (VS) metric, well-posed (polyconvex).
class TMOP_Metric_347 : public TMOP_Combo_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;
   TMOP_QualityMetric *sh_metric, *sz_metric;

public:
   TMOP_Metric_347(real_t gamma)
      : sh_metric(new TMOP_Metric_304), sz_metric(new TMOP_Metric_316)
   {
      // (1-gamma) mu_304 + gamma mu_316
      AddQualityMetric(sh_metric, 1.-gamma);
      AddQualityMetric(sz_metric, gamma);
   }

   int Id() const override { return 347; }
   real_t GetGamma() const { return wt_arr[1]; }

   virtual ~TMOP_Metric_347() { delete sh_metric; delete sz_metric; }
};

/// 3D shifted barrier form of metric 316 (not typed).
class TMOP_Metric_352 : public TMOP_QualityMetric
{
protected:
   real_t &tau0;
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   TMOP_Metric_352(real_t &t0): tau0(t0) {}

   // W = 0.5(det(J) - 1)^2 / (det(J) - tau0).
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;
};

/// 3D non-barrier Shape (S) metric.
class TMOP_Metric_360 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // W = |J|^3 / 3^(3/2) - det(J).
   real_t EvalWMatrixForm(const DenseMatrix &Jpt) const override;

   // W = (I1b/3)^3/2 - 1.
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override;

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override;

   int Id() const override { return 360; }
};

/// A-metrics
/// 2D barrier Shape (S) metric (polyconvex).
class TMOP_AMetric_011 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // (1/4 alpha) | A - (adj A)^t W^t W / omega |^2
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override
   { MFEM_ABORT("Not implemented"); }

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override
   { MFEM_ABORT("Not implemented"); }
};

/// 2D barrier Size (V) metric (polyconvex).
class TMOP_AMetric_014a : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // 0.5 * ( sqrt(alpha/omega) - sqrt(omega/alpha) )^2
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override
   { MFEM_ABORT("Not implemented"); }

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override
   { MFEM_ABORT("Not implemented"); }
};

/// 2D barrier Shape+Size+Orientation (VOS) metric (polyconvex).
class TMOP_AMetric_036 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // (1/alpha) | A - W |^2
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override
   { MFEM_ABORT("Not implemented"); }

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override
   { MFEM_ABORT("Not implemented"); }
};

/// 2D barrier Shape+Orientation (OS) metric (polyconvex).
class TMOP_AMetric_107a : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<real_t> ie;

public:
   // (1/2 alpha) | A - (|A|/|W|) W |^2
   real_t EvalW(const DenseMatrix &Jpt) const override;

   void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const override
   { MFEM_ABORT("Not implemented"); }

   void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                  const real_t weight, DenseMatrix &A) const override
   { MFEM_ABORT("Not implemented"); }
};

/// 2D barrier Shape+Size (VS) metric (polyconvex).
class TMOP_AMetric_126 : public TMOP_Combo_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<real_t> ie;
   TMOP_QualityMetric *sh_metric, *sz_metric;

public:
   TMOP_AMetric_126(real_t gamma)
      : sh_metric(new TMOP_AMetric_011), sz_metric(new TMOP_AMetric_014a)
   {
      // (1-gamma) nu_11 + gamma nu_14
      AddQualityMetric(sh_metric, 1.-gamma);
      AddQualityMetric(sz_metric, gamma);
   }

   virtual ~TMOP_AMetric_126() { delete sh_metric; delete sz_metric; }
};

/// Base class for limiting functions to be used in class TMOP_Integrator.
/** This class represents a scalar function f(x, x0, d), where x and x0 are
    positions in physical space, and d is a reference physical distance
    associated with the point x0. */
class TMOP_LimiterFunction
{
public:
   /// Returns the limiting function, f(x, x0, d).
   virtual real_t Eval(const Vector &x, const Vector &x0, real_t d) const = 0;

   /** @brief Returns the gradient of the limiting function f(x, x0, d) with
       respect to x. */
   virtual void Eval_d1(const Vector &x, const Vector &x0, real_t dist,
                        Vector &d1) const = 0;

   /** @brief Returns the Hessian of the limiting function f(x, x0, d) with
       respect to x. */
   virtual void Eval_d2(const Vector &x, const Vector &x0, real_t dist,
                        DenseMatrix &d2) const = 0;

   /// Virtual destructor.
   virtual ~TMOP_LimiterFunction() { }
};

/// Default limiter function in TMOP_Integrator.
class TMOP_QuadraticLimiter : public TMOP_LimiterFunction
{
public:
   real_t Eval(const Vector &x, const Vector &x0, real_t dist) const override
   {
      MFEM_ASSERT(x.Size() == x0.Size(), "Bad input.");

      return 0.5 * x.DistanceSquaredTo(x0) / (dist * dist);
   }

   void Eval_d1(const Vector &x, const Vector &x0, real_t dist,
                Vector &d1) const override
   {
      MFEM_ASSERT(x.Size() == x0.Size(), "Bad input.");

      d1.SetSize(x.Size());
      subtract(1.0 / (dist * dist), x, x0, d1);
   }

   void Eval_d2(const Vector &x, const Vector &x0, real_t dist,
                DenseMatrix &d2) const override
   {
      MFEM_ASSERT(x.Size() == x0.Size(), "Bad input.");

      d2.Diag(1.0 / (dist * dist), x.Size());
   }

   virtual ~TMOP_QuadraticLimiter() { }
};

/// Exponential limiter function in TMOP_Integrator.
class TMOP_ExponentialLimiter : public TMOP_LimiterFunction
{
public:
   real_t Eval(const Vector &x, const Vector &x0, real_t dist) const override
   {
      MFEM_ASSERT(x.Size() == x0.Size(), "Bad input.");

      return  exp(10.0*((x.DistanceSquaredTo(x0) / (dist * dist))-1.0));
   }

   void Eval_d1(const Vector &x, const Vector &x0, real_t dist,
                Vector &d1) const override
   {
      MFEM_ASSERT(x.Size() == x0.Size(), "Bad input.");

      d1.SetSize(x.Size());
      real_t dist_squared = dist*dist;
      subtract(20.0*exp(10.0*((x.DistanceSquaredTo(x0) / dist_squared) - 1.0)) /
               dist_squared, x, x0, d1);
   }

   void Eval_d2(const Vector &x, const Vector &x0, real_t dist,
                DenseMatrix &d2) const override
   {
      MFEM_ASSERT(x.Size() == x0.Size(), "Bad input.");
      Vector tmp;
      tmp.SetSize(x.Size());
      real_t dist_squared = dist*dist;
      real_t dist_squared_squared = dist_squared*dist_squared;
      real_t f = exp(10.0*((x.DistanceSquaredTo(x0) / dist_squared)-1.0));

      subtract(x,x0,tmp);
      d2.SetSize(x.Size());
      d2(0,0) = ((400.0*tmp(0)*tmp(0)*f)/dist_squared_squared)+(20.0*f/dist_squared);
      d2(1,1) = ((400.0*tmp(1)*tmp(1)*f)/dist_squared_squared)+(20.0*f/dist_squared);
      d2(0,1) = (400.0*tmp(0)*tmp(1)*f)/dist_squared_squared;
      d2(1,0) = d2(0,1);

      if (x.Size() == 3)
      {
         d2(0,2) = (400.0*tmp(0)*tmp(2)*f)/dist_squared_squared;
         d2(1,2) = (400.0*tmp(1)*tmp(2)*f)/dist_squared_squared;
         d2(2,0) = d2(0,2);
         d2(2,1) = d2(1,2);
         d2(2,2) = ((400.0*tmp(2)*tmp(2)*f)/dist_squared_squared)+(20.0*f/dist_squared);
      }

   }

   virtual ~TMOP_ExponentialLimiter() { }
};

class FiniteElementCollection;
class FiniteElementSpace;
class ParFiniteElementSpace;

class AdaptivityEvaluator
{
protected:
   // Owned.
   Mesh *mesh;
   FiniteElementSpace *fes;

#ifdef MFEM_USE_MPI
   // Owned.
   ParMesh *pmesh;
   ParFiniteElementSpace *pfes;
#endif

public:
   AdaptivityEvaluator() : mesh(NULL), fes(NULL)
   {
#ifdef MFEM_USE_MPI
      pmesh = NULL;
      pfes = NULL;
#endif
   }
   virtual ~AdaptivityEvaluator();

   /// Specifies the Mesh and FiniteElementSpace of the solution that will
   /// be evaluated. The given mesh will be copied into the internal object.
   void SetSerialMetaInfo(const Mesh &m,
                          const FiniteElementSpace &f);

#ifdef MFEM_USE_MPI
   /// Parallel version of SetSerialMetaInfo.
   void SetParMetaInfo(const ParMesh &m,
                       const ParFiniteElementSpace &f);
#endif

   // TODO use GridFunctions to make clear it's on the ldofs? Then do we
   // need the SetMetaInfo at all -- the space and mesh can be extracted?
   virtual void SetInitialField(const Vector &init_nodes,
                                const Vector &init_field) = 0;

   /// Called when the FE space of the final field is different than
   /// the FE space of the initial field.
   virtual void SetNewFieldFESpace(const FiniteElementSpace &fes) = 0;

   /** @brief Perform field transfer between the original and a new mesh. The
              source mesh and field are given by SetInitialField().

       @param[in]  new_mesh_nodes  Mesh node positions of the new mesh (ldofs).
                                   It is assumed that this is the field's mesh.
       @param[out] new_field       Result of the transfer (ldofs).
       @param[in]  nodes_ordering  Ordering of new_mesh_nodes.      */
   virtual void ComputeAtNewPosition(const Vector &new_mesh_nodes,
                                     Vector &new_field,
                                     int nodes_ordering = Ordering::byNODES) = 0;

   /** @brief Using the source mesh and field given by SetInitialField(),
              compute corresponding values at specified physical positions.

       @param[in]  positions   Physical positions to compute values.
       @param[out] values      Computed field values.
       @param[in]  p_ordering  Ordering of the positions Vector.     */
   virtual void ComputeAtGivenPositions(const Vector &positions,
                                        Vector &values,
                                        int p_ordering = Ordering::byNODES) = 0;

   void ClearGeometricFactors();
};

/** @brief Base class representing target-matrix construction algorithms for
    mesh optimization via the target-matrix optimization paradigm (TMOP). */
/** This class is used by class TMOP_Integrator to construct the target Jacobian
    matrices (reference-element to target-element) at quadrature points. It
    supports a set of algorithms chosen by the #TargetType enumeration.

    New target-matrix construction algorithms can be defined by deriving new
    classes and overriding the methods ComputeElementTargets() and
    ContainsVolumeInfo(). */
class TargetConstructor
{
public:
   /// Target-matrix construction algorithms supported by this class.
   enum TargetType
   {
      IDEAL_SHAPE_UNIT_SIZE, /**<
         Ideal shape, unit size; the nodes are not used. */
      IDEAL_SHAPE_EQUAL_SIZE, /**<
         Ideal shape, equal size/volume; the given nodes define the total target
         volume; for each mesh element, the target volume is the average volume
         multiplied by the volume scale, set with SetVolumeScale(). */
      IDEAL_SHAPE_GIVEN_SIZE, /**<
         Ideal shape, given size/volume; the given nodes define the target
         volume at all quadrature points. */
      GIVEN_SHAPE_AND_SIZE, /**<
         Given shape, given size/volume; the given nodes define the exact target
         Jacobian matrix at all quadrature points. */
      GIVEN_FULL /**<
         Full target tensor is specified at every quadrature point. */
   };

protected:
   // Nodes that are used in ComputeElementTargets(), depending on target_type.
   const GridFunction *nodes; // not owned
   mutable real_t avg_volume;
   real_t volume_scale;
   const TargetType target_type;
   bool uses_phys_coords; // see UsesPhysicalCoordinates()

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif

   // should be called only if avg_volume == 0.0, i.e. avg_volume is not
   // computed yet
   void ComputeAvgVolume() const;

   template<int DIM>
   bool ComputeAllElementTargets(const FiniteElementSpace &fes,
                                 const IntegrationRule &ir,
                                 const Vector &xe,
                                 DenseTensor &Jtr) const;

   // CPU fallback that uses ComputeElementTargets()
   void ComputeAllElementTargets_Fallback(const FiniteElementSpace &fes,
                                          const IntegrationRule &ir,
                                          const Vector &xe,
                                          DenseTensor &Jtr) const;

public:
   /// Constructor for use in serial
   TargetConstructor(TargetType ttype)
      : nodes(NULL), avg_volume(), volume_scale(1.0), target_type(ttype),
        uses_phys_coords(false)
   {
#ifdef MFEM_USE_MPI
      comm = MPI_COMM_NULL;
#endif
   }
#ifdef MFEM_USE_MPI
   /// Constructor for use in parallel
   TargetConstructor(TargetType ttype, MPI_Comm mpicomm)
      : nodes(NULL), avg_volume(), volume_scale(1.0), target_type(ttype),
        uses_phys_coords(false), comm(mpicomm) { }
#endif
   virtual ~TargetConstructor() { }

#ifdef MFEM_USE_MPI
   bool Parallel() const { return (comm != MPI_COMM_NULL); }
   MPI_Comm GetComm() const { return comm; }
#else
   bool Parallel() const { return false; }
#endif

   /** @brief Set the nodes to be used in the target-matrix construction.

       This method should be called every time the target nodes are updated
       externally and recomputation of the target average volume is needed. The
       nodes are used by all target types except IDEAL_SHAPE_UNIT_SIZE. */
   void SetNodes(const GridFunction &n) { nodes = &n; avg_volume = 0.0; }

   /** @brief Get the nodes to be used in the target-matrix construction. */
   const GridFunction *GetNodes() const { return nodes; }

   /// Used by target type IDEAL_SHAPE_EQUAL_SIZE. The default volume scale is 1.
   void SetVolumeScale(real_t vol_scale) { volume_scale = vol_scale; }

   TargetType GetTargetType() const { return target_type; }

   /** @brief Return true if the methods ComputeElementTargets(),
       ComputeAllElementTargets(), and ComputeElementTargetsGradient() use the
       physical node coordinates provided by the parameters 'elfun', or 'xe'. */
   bool UsesPhysicalCoordinates() const { return uses_phys_coords; }

   /// Checks if the target matrices contain non-trivial size specification.
   virtual bool ContainsVolumeInfo() const;

   /** @brief Given an element and quadrature rule, computes ref->target
       transformation Jacobians for each quadrature point in the element.
       The physical positions of the element's nodes are given by @a elfun. */
   virtual void ComputeElementTargets(int e_id, const FiniteElement &fe,
                                      const IntegrationRule &ir,
                                      const Vector &elfun,
                                      DenseTensor &Jtr) const;

   /** @brief Computes reference-to-target transformation Jacobians for all
       quadrature points in all elements.

       @param[in] fes  The nodal FE space
       @param[in] ir   The quadrature rule to use for all elements
       @param[in] xe   E-vector with the current physical coordinates/positions;
                       this parameter is used only when needed by the target
                       constructor, see UsesPhysicalCoordinates()
       @param[out] Jtr The computed ref->target Jacobian matrices. */
   virtual void ComputeAllElementTargets(const FiniteElementSpace &fes,
                                         const IntegrationRule &ir,
                                         const Vector &xe,
                                         DenseTensor &Jtr) const;

   virtual void ComputeElementTargetsGradient(const IntegrationRule &ir,
                                              const Vector &elfun,
                                              IsoparametricTransformation &Tpr,
                                              DenseTensor &dJtr) const;
};

class TMOPMatrixCoefficient : public MatrixCoefficient
{
public:
   explicit TMOPMatrixCoefficient(int dim) : MatrixCoefficient(dim, dim) { }

   /** @brief Evaluate the derivative of the matrix coefficient with respect to
       @a comp in the element described by @a T at the point @a ip, storing the
       result in @a K. */
   virtual void EvalGrad(DenseMatrix &K, ElementTransformation &T,
                         const IntegrationPoint &ip, int comp) = 0;

   virtual ~TMOPMatrixCoefficient() { }
};

class AnalyticAdaptTC : public TargetConstructor
{
protected:
   // Analytic target specification.
   Coefficient *scalar_tspec;
   VectorCoefficient *vector_tspec;
   TMOPMatrixCoefficient *matrix_tspec;

public:
   AnalyticAdaptTC(TargetType ttype)
      : TargetConstructor(ttype),
        scalar_tspec(NULL), vector_tspec(NULL), matrix_tspec(NULL)
   { uses_phys_coords = true; }

   virtual void SetAnalyticTargetSpec(Coefficient *sspec,
                                      VectorCoefficient *vspec,
                                      TMOPMatrixCoefficient *mspec);

   /** @brief Given an element and quadrature rule, computes ref->target
       transformation Jacobians for each quadrature point in the element.
       The physical positions of the element's nodes are given by @a elfun. */
   void ComputeElementTargets(int e_id, const FiniteElement &fe,
                              const IntegrationRule &ir,
                              const Vector &elfun,
                              DenseTensor &Jtr) const override;

   void ComputeAllElementTargets(const FiniteElementSpace &fes,
                                 const IntegrationRule &ir,
                                 const Vector &xe,
                                 DenseTensor &Jtr) const override;

   void ComputeElementTargetsGradient(const IntegrationRule &ir,
                                      const Vector &elfun,
                                      IsoparametricTransformation &Tpr,
                                      DenseTensor &dJtr) const override;
};

#ifdef MFEM_USE_MPI
class ParGridFunction;
#endif

class DiscreteAdaptTC : public TargetConstructor
{
protected:
   // Discrete target specification.
   // Data is owned, updated by UpdateTargetSpecification.
   int ncomp, sizeidx, skewidx, aspectratioidx, orientationidx;
   Vector tspec;             //eta(x) - we enforce Ordering::byNODES
   Vector tspec_sav;
   Vector tspec_pert1h;      //eta(x+h)
   Vector tspec_pert2h;      //eta(x+2*h)
   Vector tspec_pertmix;     //eta(x+h,y+h)
   // The order inside these perturbation vectors (e.g. in 2D) is
   // eta1(x+h,y), eta2(x+h,y) ... etan(x+h,y), eta1(x,y+h), eta2(x,y+h) ...
   // same for tspec_pert2h and tspec_pertmix.

   // DenseMatrix to hold target_spec values for the (children of the)
   // element being refined to consider for h-refinement.
   DenseMatrix tspec_refine;
   // Vector to hold the target_spec values for the coarse version of the
   // current mesh. Used for derefinement decision with hr-adaptivity.
   Vector tspec_derefine;

   // Components of Target Jacobian at each quadrature point of an element. This
   // is required for computation of the derivative using chain rule.
   mutable DenseTensor Jtrcomp;

   // Note: do not use the Nodes of this space as they may not be on the
   // positions corresponding to the values of tspec.
   FiniteElementSpace *tspec_fesv;         //owned
   FiniteElementSpace *coarse_tspec_fesv;  //not owned, derefinement FESpace
   GridFunction *tspec_gf;                 //owned, uses tspec and tspec_fes
   // discrete adaptivity
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *ptspec_fesv;     //owned, needed for derefinement to
   // get update operator.
   ParGridFunction *tspec_pgf;             // similar to tspec_gf
#endif

   int amr_el;
   real_t lim_min_size;

   // These flags can be used by outside functions to avoid recomputing the
   // tspec and tspec_perth fields again on the same mesh.
   bool good_tspec, good_tspec_grad, good_tspec_hess;

   // Evaluation of the discrete target specification on different meshes.
   // Owned.
   AdaptivityEvaluator *adapt_eval;

   void SetDiscreteTargetBase(const GridFunction &tspec_);
   void SetTspecAtIndex(int idx, const GridFunction &tspec_);
   void FinalizeSerialDiscreteTargetSpec(const GridFunction &tspec_);
#ifdef MFEM_USE_MPI
   void SetTspecAtIndex(int idx, const ParGridFunction &tspec_);
   void FinalizeParDiscreteTargetSpec(const ParGridFunction &tspec_);
#endif

public:
   DiscreteAdaptTC(TargetType ttype)
      : TargetConstructor(ttype),
        ncomp(0),
        sizeidx(-1), skewidx(-1), aspectratioidx(-1), orientationidx(-1),
        tspec(), tspec_sav(), tspec_pert1h(), tspec_pert2h(), tspec_pertmix(),
        tspec_refine(), tspec_derefine(),
        tspec_fesv(NULL), coarse_tspec_fesv(NULL), tspec_gf(NULL),
#ifdef MFEM_USE_MPI
        ptspec_fesv(NULL), tspec_pgf(NULL),
#endif
        amr_el(-1), lim_min_size(-0.1),
        good_tspec(false), good_tspec_grad(false), good_tspec_hess(false),
        adapt_eval(NULL) { }

   virtual ~DiscreteAdaptTC();

   /** @name Target specification methods.
       The following methods are used to specify geometric parameters of the
       targets when these parameters are given by discrete FE functions.
       Note that every GridFunction given to the Set methods must use a
       H1_FECollection of the same order. The number of components must
       correspond to the type of geometric parameter and dimension.

       @param[in] tspec_  Input values of a geometric parameter. Note that
                          the methods in this class support only functions that
                          use H1_FECollection collection of the same order. */
   ///@{
   virtual void SetSerialDiscreteTargetSpec(const GridFunction &tspec_);
   virtual void SetSerialDiscreteTargetSize(const GridFunction &tspec_);
   virtual void SetSerialDiscreteTargetSkew(const GridFunction &tspec_);
   virtual void SetSerialDiscreteTargetAspectRatio(const GridFunction &tspec_);
   virtual void SetSerialDiscreteTargetOrientation(const GridFunction &tspec_);
#ifdef MFEM_USE_MPI
   virtual void SetParDiscreteTargetSpec(const ParGridFunction &tspec_);
   virtual void SetParDiscreteTargetSize(const ParGridFunction &tspec_);
   virtual void SetParDiscreteTargetSkew(const ParGridFunction &tspec_);
   virtual void SetParDiscreteTargetAspectRatio(const ParGridFunction &tspec_);
   virtual void SetParDiscreteTargetOrientation(const ParGridFunction &tspec_);
#endif
   ///@}

   /// Used in combination with the Update methods to avoid extra computations.
   void ResetUpdateFlags()
   { good_tspec = good_tspec_grad = good_tspec_hess = false; }

   /// Get one of the discrete fields from tspec.
   void GetDiscreteTargetSpec(GridFunction &tspec_, int idx);
   /// Get the FESpace associated with tspec.
   FiniteElementSpace *GetTSpecFESpace() { return tspec_fesv; }
   /// Get the entire tspec.
   GridFunction *GetTSpecData() { return tspec_gf; }
   /// Update all discrete fields based on tspec and update for AMR
   void UpdateAfterMeshTopologyChange();

#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *GetTSpecParFESpace() { return ptspec_fesv; }
   void ParUpdateAfterMeshTopologyChange();
#endif

   /** Used to update the target specification after the mesh has changed. The
       new mesh positions are given by new_x. If @a reuse_flag is true,
       repeated calls won't do anything until ResetUpdateFlags() is called. */
   void UpdateTargetSpecification(const Vector &new_x, bool reuse_flag = false,
                                  int new_x_ordering=Ordering::byNODES);

   void UpdateTargetSpecification(Vector &new_x, Vector &IntData,
                                  int new_x_ordering=Ordering::byNODES);

   void UpdateTargetSpecificationAtNode(const FiniteElement &el,
                                        ElementTransformation &T,
                                        int nodenum, int idir,
                                        const Vector &IntData);

   void RestoreTargetSpecificationAtNode(ElementTransformation &T, int nodenum);

   /** Used for finite-difference based computations. Computes the target
       specifications after a mesh perturbation in x or y direction.
       If @a reuse_flag is true, repeated calls won't do anything until
       ResetUpdateFlags() is called. */
   void UpdateGradientTargetSpecification(const Vector &x, real_t dx,
                                          bool reuse_flag = false,
                                          int x_ordering = Ordering::byNODES);
   /** Used for finite-difference based computations. Computes the target
       specifications after two mesh perturbations in x and/or y direction.
       If @a reuse_flag is true, repeated calls won't do anything until
       ResetUpdateFlags() is called. */
   void UpdateHessianTargetSpecification(const Vector &x, real_t dx,
                                         bool reuse_flag = false,
                                         int x_ordering = Ordering::byNODES);

   void SetAdaptivityEvaluator(AdaptivityEvaluator *ae)
   {
      if (adapt_eval) { delete adapt_eval; }
      adapt_eval = ae;
   }

   const AdaptivityEvaluator *GetAdaptivityEvaluator() const
   {
      return adapt_eval;
   }

   const Vector &GetTspecPert1H()   { return tspec_pert1h; }
   const Vector &GetTspecPert2H()   { return tspec_pert2h; }
   const Vector &GetTspecPertMixH() { return tspec_pertmix; }

   /** @brief Given an element and quadrature rule, computes ref->target
       transformation Jacobians for each quadrature point in the element.
       The physical positions of the element's nodes are given by @a elfun.
       Note that this function assumes that UpdateTargetSpecification() has
       been called with the position vector corresponding to @a elfun. */
   void ComputeElementTargets(int e_id, const FiniteElement &fe,
                              const IntegrationRule &ir,
                              const Vector &elfun,
                              DenseTensor &Jtr) const override;

   void ComputeAllElementTargets(const FiniteElementSpace &fes,
                                 const IntegrationRule &ir,
                                 const Vector &xe,
                                 DenseTensor &Jtr) const override;

   void ComputeElementTargetsGradient(const IntegrationRule &ir,
                                      const Vector &elfun,
                                      IsoparametricTransformation &Tpr,
                                      DenseTensor &dJtr) const override;

   // Generates tspec_vals for target construction using intrule
   // Used for the refinement component in hr-adaptivity.
   void SetTspecFromIntRule(int e_id, const IntegrationRule &intrule);

   // Targets based on discrete functions can result in invalid (negative)
   // size at the quadrature points. This method can be used to set a
   // minimum target size.
   void SetMinSizeForTargets(real_t min_size_) { lim_min_size = min_size_; }

   /// Computes target specification data with respect to the coarse FE space.
   void SetTspecDataForDerefinement(FiniteElementSpace *fes);

   // Reset refinement data associated with h-adaptivity component.
   void ResetRefinementTspecData()
   {
      tspec_refine.Clear();
      amr_el = -1;
   }

   // Reset derefinement data associated with h-adaptivity component.
   void ResetDerefinementTspecData()
   {
      tspec_derefine.Destroy();
      coarse_tspec_fesv = NULL;
   }

   // Used to specify the fine element for determining energy of children of a
   // parent element.
   void SetRefinementSubElement(int amr_el_) { amr_el = amr_el_; }
};

class TMOPNewtonSolver;

/** @brief A TMOP integrator class based on any given TMOP_QualityMetric and
    TargetConstructor.

    Represents $ \int W(Jpt) dx $ over a target zone, where W is the
    metric's strain energy density function, and Jpt is the Jacobian of the
    target->physical coordinates transformation. The virtual target zone is
    defined by the TargetConstructor. */
class TMOP_Integrator : public NonlinearFormIntegrator
{
protected:
   friend class TMOPNewtonSolver;
   friend class TMOPComboIntegrator;

   TMOP_QualityMetric *h_metric;
   TMOP_QualityMetric *metric;        // not owned
   const TargetConstructor *targetC;  // not owned

   // Custom integration rules.
   IntegrationRules *IntegRules;
   int integ_order;
   bool integ_over_target = true;

   // Weight Coefficient multiplying the quality metric term.
   Coefficient *metric_coeff; // not owned, if NULL -> metric_coeff is 1.
   // Normalization factor for the metric term.
   real_t metric_normal;

   // Nodes and weight Coefficient used for "limiting" the TMOP_Integrator.
   // These are both NULL when there is no limiting.
   // The class doesn't own lim_nodes0 and lim_coeff.
   const GridFunction *lim_nodes0;
   Coefficient *lim_coeff;
   // Limiting reference distance. Not owned.
   const GridFunction *lim_dist;
   // Limiting function. Owned.
   TMOP_LimiterFunction *lim_func;
   // Normalization factor for the limiting term.
   real_t lim_normal;

   // Adaptive limiting.
   const GridFunction *adapt_lim_gf0;    // Not owned.
#ifdef MFEM_USE_MPI
   const ParGridFunction *adapt_lim_pgf0;
#endif
   GridFunction *adapt_lim_gf;           // Owned. Updated by adapt_lim_eval.
   Coefficient *adapt_lim_coeff;         // Not owned.
   AdaptivityEvaluator *adapt_lim_eval;  // Not owned.

   // Surface fitting.
   const Array<bool> *surf_fit_marker;      // Not owned. Nodes to fit.
   Coefficient *surf_fit_coeff;             // Not owned. Fitting term scaling.
   // Fitting to a discrete level set.
   GridFunction *surf_fit_gf;               // Owned. Updated by surf_fit_eval.
   AdaptivityEvaluator *surf_fit_eval;      // Not owned.
   // Fitting to given physical positions.
   TMOP_QuadraticLimiter *surf_fit_limiter; // Owned. Created internally.
   const GridFunction *surf_fit_pos;        // Not owned. Positions to fit.
   real_t surf_fit_normal;                  // Normalization factor.
   GridFunction *surf_fit_grad, *surf_fit_hess; // Owned. Created internally.
   AdaptivityEvaluator *surf_fit_eval_grad, *surf_fit_eval_hess; // Not owned.
   Array<int> surf_fit_dof_count;            // Number of dofs per node.
   Array<int> surf_fit_marker_dof_index;     // Indices of nodes to fit.

   DiscreteAdaptTC *discr_tc;

   // Parameters for FD-based Gradient & Hessian calculation.
   bool fdflag;
   real_t dx;
   real_t dxscale;
   // Specifies that ComputeElementTargets is being called by a FD function.
   // It's used to skip terms that have exact derivative calculations.
   bool fd_call_flag;
   // Compute the exact action of the Integrator (includes derivative of the
   // target with respect to spatial position)
   bool exact_action;

   Array <Vector *> ElemDer;        //f'(x)
   Array <Vector *> ElemPertEnergy; //f(x+h)

   //   Jrt: the inverse of the ref->target Jacobian, Jrt = Jtr^{-1}.
   //   Jpr: the ref->physical transformation Jacobian, Jpr = PMatI^t DS.
   //   Jpt: the target->physical transformation Jacobian, Jpt = Jpr Jrt.
   //     P: represents dW_d(Jtp) (dim x dim).
   //   DSh: gradients of reference shape functions (dof x dim).
   //    DS: gradients of the shape functions in the target configuration,
   //        DS = DSh Jrt (dof x dim).
   // PMatI: current coordinates of the nodes (dof x dim).
   // PMat0: reshaped view into the local element contribution to the operator
   //        output - the result of AssembleElementVector() (dof x dim).
   DenseMatrix DSh, DS, Jrt, Jpr, Jpt, P, PMatI, PMatO;

   // PA extension
   // ------------
   // Jtr: all ref->target Jacobians, (dim x dim) Q-Vector as DenseTensor.
   //      updated when needed, based on Jtr_needs_update.
   //
   //  E: Q-vector for TMOP-energy
   //     Used as temporary storage when the total energy is computed.
   //  O: Q-Vector of 1.0, used to compute sums using the dot product kernel.
   // X0: E-vector for initial nodal coordinates used for limiting.
   //     Does not change during the TMOP iteration.
   //  H: Q-Vector for Hessian associated with the metric term.
   //     Updated by every call to PANonlinearFormExtension::GetGradient().
   // C0: Q-Vector for spatial weight used for the limiting term.
   //     Updated when the mesh nodes change.
   // LD: E-Vector constructed using limiting distance grid function (delta).
   //     Does not change during the TMOP iteration.
   // H0: Q-Vector for Hessian associated with the limiting term.
   //     Updated by every call to PANonlinearFormExtension::GetGradient().
   // MC: Q-Vector for the metric Coefficient.
   //     Updated when the mesh nodes change.
   //
   // maps:     Dof2Quad map for fes associated with the nodal coordinates.
   // maps_lim: Dof2Quad map for fes associated with the limiting dist GridFunc.
   //
   // Jtr_debug_grad
   //     We keep track if Jtr was set by AssembleGradPA() in Jtr_debug_grad: it
   //     is set to true by AssembleGradPA(); any other call to
   //     ComputeAllElementTargets() will set the flag to false. This flag will
   //     be used to check that Jtr is the one set by AssembleGradPA() when
   //     performing operations with the gradient like AddMultGradPA() and
   //     AssembleGradDiagonalPA().
   //
   // TODO:
   //   * Merge LD, C0, H0 into one scalar Q-vector
   struct
   {
      bool enabled;
      int dim, ne, nq;
      mutable DenseTensor Jtr;
      mutable bool Jtr_needs_update;
      mutable bool Jtr_debug_grad;
      mutable Vector E, O, X0, H, C0, LD, H0, MC;
      const DofToQuad *maps;
      const DofToQuad *maps_lim = nullptr;
      const GeometricFactors *geom;
      const FiniteElementSpace *fes;
      const IntegrationRule *ir;
   } PA;

   void ComputeNormalizationEnergies(const GridFunction &x,
                                     real_t &metric_energy, real_t &lim_energy,
                                     real_t &surf_fit_gf_energy);

   void AssembleElementVectorExact(const FiniteElement &el,
                                   ElementTransformation &T,
                                   const Vector &elfun, Vector &elvect);

   void AssembleElementGradExact(const FiniteElement &el,
                                 ElementTransformation &T,
                                 const Vector &elfun, DenseMatrix &elmat);

   void AssembleElementVectorFD(const FiniteElement &el,
                                ElementTransformation &T,
                                const Vector &elfun, Vector &elvect);

   // Assumes that AssembleElementVectorFD has been called.
   void AssembleElementGradFD(const FiniteElement &el,
                              ElementTransformation &T,
                              const Vector &elfun, DenseMatrix &elmat);

   void AssembleElemVecAdaptLim(const FiniteElement &el,
                                IsoparametricTransformation &Tpr,
                                const IntegrationRule &ir,
                                const Vector &weights, DenseMatrix &mat);
   void AssembleElemGradAdaptLim(const FiniteElement &el,
                                 IsoparametricTransformation &Tpr,
                                 const IntegrationRule &ir,
                                 const Vector &weights, DenseMatrix &m);

   // First derivative of the surface fitting term.
   void AssembleElemVecSurfFit(const FiniteElement &el_x,
                               IsoparametricTransformation &Tpr,
                               DenseMatrix &mat);

   // Second derivative of the surface fitting term.
   void AssembleElemGradSurfFit(const FiniteElement &el_x,
                                IsoparametricTransformation &Tpr,
                                DenseMatrix &mat);

   real_t GetFDDerivative(const FiniteElement &el,
                          ElementTransformation &T,
                          Vector &elfun, const int nodenum,const int idir,
                          const real_t baseenergy, bool update_stored);

   /** @brief Determines the perturbation, h, for FD-based approximation. */
   void ComputeFDh(const Vector &x, const FiniteElementSpace &fes);
   void ComputeMinJac(const Vector &x, const FiniteElementSpace &fes);

   void UpdateAfterMeshPositionChange(const Vector &x_new,
                                      const FiniteElementSpace &x_fes);

   void DisableLimiting()
   {
      lim_nodes0 = NULL; lim_coeff = NULL; lim_dist = NULL;
      delete lim_func; lim_func = NULL;
   }

   const IntegrationRule &EnergyIntegrationRule(const FiniteElement &el) const
   {
      if (IntegRules)
      {
         return IntegRules->Get(el.GetGeomType(), integ_order);
      }
      return (IntRule) ? *IntRule
             /*     */ : IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3);
   }
   const IntegrationRule &ActionIntegrationRule(const FiniteElement &el) const
   {
      // TODO the energy most likely needs less integration points.
      return EnergyIntegrationRule(el);
   }
   const IntegrationRule &GradientIntegrationRule(const FiniteElement &el) const
   {
      // TODO the action and energy most likely need less integration points.
      return EnergyIntegrationRule(el);
   }

   // Auxiliary PA methods
   void AssembleGradPA_2D(const Vector&) const;
   void AssembleGradPA_3D(const Vector&) const;
   void AssembleGradPA_C0_2D(const Vector&) const;
   void AssembleGradPA_C0_3D(const Vector&) const;

   real_t GetLocalStateEnergyPA_2D(const Vector&) const;
   real_t GetLocalStateEnergyPA_C0_2D(const Vector&) const;
   real_t GetLocalStateEnergyPA_3D(const Vector&) const;
   real_t GetLocalStateEnergyPA_C0_3D(const Vector&) const;

   void AddMultPA_2D(const Vector&, Vector&) const;
   void AddMultPA_3D(const Vector&, Vector&) const;
   void AddMultPA_C0_2D(const Vector&, Vector&) const;
   void AddMultPA_C0_3D(const Vector&, Vector&) const;

   void AddMultGradPA_2D(const Vector&, Vector&) const;
   void AddMultGradPA_3D(const Vector&, Vector&) const;
   void AddMultGradPA_C0_2D(const Vector&, Vector&) const;
   void AddMultGradPA_C0_3D(const Vector&, Vector&) const;

   void AssembleDiagonalPA_2D(Vector&) const;
   void AssembleDiagonalPA_3D(Vector&) const;
   void AssembleDiagonalPA_C0_2D(Vector&) const;
   void AssembleDiagonalPA_C0_3D(Vector&) const;

   void AssemblePA_Limiting();
   void ComputeAllElementTargets(const Vector &xe = Vector()) const;
   // Updates the Q-vectors for the metric_coeff and lim_coeff, based on the
   // new physical positions of the quadrature points.
   void UpdateCoefficientsPA(const Vector &x_loc);

   // Compute Min(Det(Jpt)) in the mesh, does not reduce over MPI.
   real_t ComputeMinDetT(const Vector &x, const FiniteElementSpace &fes);
   // Compute Max(mu_hat) for the TMOP_WorstCaseUntangleOptimizer_Metric,
   // does not reduce over MPI.
   real_t ComputeUntanglerMaxMuBarrier(const Vector &x,
                                       const FiniteElementSpace &fes);

   // Remaps the internal surface fitting gridfunction object at provided
   // locations.
   void RemapSurfaceFittingLevelSetAtNodes(const Vector &new_x,
                                           int new_x_ordering);
public:
   /** @param[in] m    TMOP_QualityMetric for r-adaptivity (not owned).
       @param[in] tc   Target-matrix construction algorithm to use (not owned).
       @param[in] hm   TMOP_QualityMetric for h-adaptivity (not owned). */
   TMOP_Integrator(TMOP_QualityMetric *m, TargetConstructor *tc,
                   TMOP_QualityMetric *hm)
      : h_metric(hm), metric(m), targetC(tc), IntegRules(NULL),
        integ_order(-1), metric_coeff(NULL), metric_normal(1.0),
        lim_nodes0(NULL), lim_coeff(NULL),
        lim_dist(NULL), lim_func(NULL), lim_normal(1.0),
        adapt_lim_gf0(NULL), adapt_lim_gf(NULL), adapt_lim_coeff(NULL),
        adapt_lim_eval(NULL),
        surf_fit_marker(NULL), surf_fit_coeff(NULL),
        surf_fit_gf(NULL), surf_fit_eval(NULL),
        surf_fit_limiter(NULL), surf_fit_pos(NULL),
        surf_fit_normal(1.0), surf_fit_grad(NULL), surf_fit_hess(NULL),
        surf_fit_eval_grad(NULL), surf_fit_eval_hess(NULL),
        discr_tc(dynamic_cast<DiscreteAdaptTC *>(tc)),
        fdflag(false), dxscale(1.0e3), fd_call_flag(false), exact_action(false)
   { PA.enabled = false; }

   TMOP_Integrator(TMOP_QualityMetric *m, TargetConstructor *tc)
      : TMOP_Integrator(m, tc, m) { }

   ~TMOP_Integrator();

   /// Release the device memory of large PA allocations. This will copy device
   /// memory back to the host before releasing.
   void ReleasePADeviceMemory(bool copy_to_host = true);

   /// Prescribe a set of integration rules; relevant for mixed meshes.
   /** This function has priority over SetIntRule(), if both are called. */
   void SetIntegrationRules(IntegrationRules &irules, int order)
   {
      IntegRules = &irules;
      integ_order = order;
   }

   /// The TMOP integrals can be computed over the reference element or the
   /// target elements. This function is used to switch between the two options.
   /// By default integration is performed over the target elements.
   void IntegrateOverTarget(bool integ_over_target_)
   {
      MFEM_VERIFY(metric_normal == 1.0 && lim_normal == 1.0,
                  "This function must be called before EnableNormalization, as "
                  "the normalization computations must know how to integrate.");

      integ_over_target = integ_over_target_;
   }

   /// Sets a scaling Coefficient for the quality metric term of the integrator.
   /** With this addition, the integrator becomes
          $ \int w1 W(Jpt) dx $.

       Note that the Coefficient is evaluated in the physical configuration and
       not in the target configuration which may be undefined. */
   void SetCoefficient(Coefficient &w1) { metric_coeff = &w1; }

   /** @brief Limiting of the mesh displacements (general version).

       Adds the term $ \int w_0 f(x, x_0, d) dx $, where f is a measure of
       the displacement between x and x_0, given the max allowed displacement d.

       @param[in] n0     Original mesh node coordinates (x0 above).
       @param[in] dist   Allowed displacement in physical space (d above).
       @param[in] w0     Coefficient scaling the limiting integral.
       @param[in] lfunc  TMOP_LimiterFunction defining the function f. If
                         NULL, a TMOP_QuadraticLimiter will be used. The
                         TMOP_Integrator assumes ownership of this pointer. */
   void EnableLimiting(const GridFunction &n0, const GridFunction &dist,
                       Coefficient &w0, TMOP_LimiterFunction *lfunc = NULL);

   /** @brief Adds a limiting term to the integrator with limiting distance
       function (@a dist in the general version of the method) equal to 1. */
   void EnableLimiting(const GridFunction &n0, Coefficient &w0,
                       TMOP_LimiterFunction *lfunc = NULL);

   /** @brief Restriction of the node positions to certain regions.

       Adds the term $ \int c (z(x) - z_0(x_0))^2 $, where z0(x0) is a given
       function on the starting mesh, and z(x) is its image on the new mesh.
       Minimizing this term means that a node at x0 is allowed to move to a
       position x(x0) only if z(x) ~ z0(x0).
       Such term can be used for tangential mesh relaxation.

       @param[in] z0     Function z0 that controls the adaptive limiting.
       @param[in] coeff  Coefficient c for the above integral.
       @param[in] ae     AdaptivityEvaluator to compute z(x) from z0(x0). */
   void EnableAdaptiveLimiting(const GridFunction &z0, Coefficient &coeff,
                               AdaptivityEvaluator &ae);
#ifdef MFEM_USE_MPI
   /// Parallel support for adaptive limiting.
   void EnableAdaptiveLimiting(const ParGridFunction &z0, Coefficient &coeff,
                               AdaptivityEvaluator &ae);
#endif

   /** @brief Fitting of certain DOFs to the zero level set of a function.

       Having a level set function s0(x0) on the starting mesh, and a set of
       marked nodes (or DOFs), we move these nodes to the zero level set of s0.
       If s(x) is the image of s0(x0) on the current mesh, this function adds to
       the TMOP functional the term $ \int c \bar{s}(x))^2 $, where
       $\bar{s}(x)$ is the restriction of s(x) on the aligned DOFs.
       Minimizing this term means that a marked node at x0 is allowed to move to
       a position x(x0) only if s(x) ~ 0.
       Such term can be used for surface fitting and tangential relaxation.

       @param[in] s0      The level set function on the initial mesh.
       @param[in] smarker Indicates which DOFs will be aligned.
       @param[in] coeff   Coefficient c for the above integral.
       @param[in] ae      AdaptivityEvaluator to compute s(x) from s0(x0). */
   void EnableSurfaceFitting(const GridFunction &s0,
                             const Array<bool> &smarker, Coefficient &coeff,
                             AdaptivityEvaluator &ae);

#ifdef MFEM_USE_MPI
   /// Parallel support for surface fitting to the zero level set of a function.
   /// Here, we add two optional inputs: @a aegrad and @a aehess. When provided,
   /// the first and second derivative of the input level set are computed on
   /// the initial mesh, and @a aegrad and @a aehess are used to remap grad_s(x)
   /// from grad_s0(x0) and hess_s(x) from hess_s0(x0), respectively.
   void EnableSurfaceFitting(const ParGridFunction &s0,
                             const Array<bool> &smarker, Coefficient &coeff,
                             AdaptivityEvaluator &ae,
                             AdaptivityEvaluator *aegrad = NULL,
                             AdaptivityEvaluator *aehess = NULL);

   /** @brief Fitting of certain DOFs in the current mesh to the zero level set
       of a function defined on another (finer) source mesh.

       Having a level set function s_bg(x_bg) on a source/background mesh,
       a set of marked nodes (or DOFs) in the current mesh, we move the marked
       nodes to the zero level set of s_bg. This functionality is used for
       surface fitting and tangential relaxation.

       @param[in] s_bg       The level set function on the background mesh.
       @param[in] s0         The level set function (automatically) interpolated
                             on the initial mesh.
       @param[in] smarker    Marker for aligned DOFs in the current mesh.
       @param[in] coeff      Coefficient c for the fitting penalty term.
       @param[in] ae         Interpolates s(x) from s_bg(x_bg).
       @param[in] s_bg_grad  Gradient of s_bg on the background mesh.
       @param[in] s0_grad    Gradient of s0 on the initial mesh.
       @param[in] age        Interpolates s_grad(x) from s_bg_grad(x_bg).
       @param[in] s_bg_hess  Hessian of s(x) on the background mesh.
       @param[in] s0_hess    Hessian of s0 on the initial mesh.
       @param[in] ahe        Interpolates s_hess(x) from s_bg_hess(x_bg).
       See the pmesh-fitting miniapp for details on usage. */
   void EnableSurfaceFittingFromSource(const ParGridFunction &s_bg,
                                       ParGridFunction &s0,
                                       const Array<bool> &smarker,
                                       Coefficient &coeff,
                                       AdaptivityEvaluator &ae,
                                       const ParGridFunction &s_bg_grad,
                                       ParGridFunction &s0_grad,
                                       AdaptivityEvaluator &age,
                                       const ParGridFunction &s_bg_hess,
                                       ParGridFunction &s0_hess,
                                       AdaptivityEvaluator &ahe);
#endif
   /** @brief Fitting of certain DOFs to given positions in physical space.

       Having a set S of marked nodes (or DOFs) and their target positions in
       physical space x_t, we move these nodes to the target positions during
       the optimization process.
       This function adds to the TMOP functional the term
       $ \sum_{i \in S} c \frac{1}{2} (x_i - x_{t,i})^2 $,
       where $c$ corresponds to @a coeff below and is evaluated at the
       DOF locations.

       @param[in] pos     The desired positions for the mesh nodes.
       @param[in] smarker Indicates which DOFs will be aligned.
       @param[in] coeff   Coefficient c for the above integral. */
   void EnableSurfaceFitting(const GridFunction &pos,
                             const Array<bool> &smarker, Coefficient &coeff);
   void GetSurfaceFittingErrors(const Vector &pos,
                                real_t &err_avg, real_t &err_max);
   bool IsSurfaceFittingEnabled()
   {
      return surf_fit_gf != NULL || surf_fit_pos != NULL;
   }

   /// Update the original/reference nodes used for limiting.
   void SetLimitingNodes(const GridFunction &n0) { lim_nodes0 = &n0; }

   /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone.
       @param[in] el     Type of FiniteElement.
       @param[in] T      Mesh element transformation.
       @param[in] elfun  Physical coordinates of the zone. */
   real_t GetElementEnergy(const FiniteElement &el,
                           ElementTransformation &T,
                           const Vector &elfun) override;

   /** @brief Computes the mean of the energies of the given element's children.

       In addition to the inputs for GetElementEnergy, this function requires an
       IntegrationRule to be specified that will give the decomposition of the
       given element based on the refinement type being considered. */
   virtual real_t GetRefinementElementEnergy(const FiniteElement &el,
                                             ElementTransformation &T,
                                             const Vector &elfun,
                                             const IntegrationRule &irule);

   /// This function is similar to GetElementEnergy, but ignores components
   /// such as limiting etc. to compute the element energy.
   virtual real_t GetDerefinementElementEnergy(const FiniteElement &el,
                                               ElementTransformation &T,
                                               const Vector &elfun);

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &T,
                              const Vector &elfun, Vector &elvect) override;

   void AssembleElementGrad(const FiniteElement &el,
                            ElementTransformation &T,
                            const Vector &elfun, DenseMatrix &elmat) override;

   TMOP_QualityMetric &GetAMRQualityMetric() { return *h_metric; }

   void UpdateAfterMeshTopologyChange();
#ifdef MFEM_USE_MPI
   void ParUpdateAfterMeshTopologyChange();
#endif

   // PA extension
   using NonlinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace&) override;

   void AssembleGradPA(const Vector&, const FiniteElementSpace&) override;

   real_t GetLocalStateEnergyPA(const Vector&) const override;

   void AddMultPA(const Vector&, Vector&) const override;

   void AddMultGradPA(const Vector&, Vector&) const override;

   void AssembleGradDiagonalPA(Vector&) const override;

   DiscreteAdaptTC *GetDiscreteAdaptTC() const { return discr_tc; }

   /** @brief Computes the normalization factors of the metric and limiting
       integrals using the mesh position given by @a x. */
   void EnableNormalization(const GridFunction &x);
#ifdef MFEM_USE_MPI
   void ParEnableNormalization(const ParGridFunction &x);
#endif

   /** @brief Enables FD-based approximation and computes dx. */
   void EnableFiniteDifferences(const GridFunction &x);
#ifdef MFEM_USE_MPI
   void EnableFiniteDifferences(const ParGridFunction &x);
#endif

   void   SetFDhScale(real_t dxscale_) { dxscale = dxscale_; }
   bool   GetFDFlag() const { return fdflag; }
   real_t GetFDh()    const { return dx; }

   /** @brief Flag to control if exact action of Integration is effected. */
   void SetExactActionFlag(bool flag_) { exact_action = flag_; }

   /// Update the surface fitting weight as surf_fit_coeff *= factor;
   void UpdateSurfaceFittingWeight(real_t factor);

   /// Get the surface fitting weight.
   real_t GetSurfaceFittingWeight();

   /// Computes quantiles needed for UntangleMetrics. Note that in parallel,
   /// the ParFiniteElementSpace must be passed as argument for consistency
   /// across MPI ranks.
   void ComputeUntangleMetricQuantiles(const Vector &x,
                                       const FiniteElementSpace &fes);
};

class TMOPComboIntegrator : public NonlinearFormIntegrator
{
protected:
   // Integrators in the combination. Owned.
   Array<TMOP_Integrator *> tmopi;

public:
   TMOPComboIntegrator() : tmopi(0) { }

   ~TMOPComboIntegrator()
   {
      for (int i = 0; i < tmopi.Size(); i++) { delete tmopi[i]; }
   }

   /// Adds a new TMOP_Integrator to the combination.
   void AddTMOPIntegrator(TMOP_Integrator *ti) { tmopi.Append(ti); }

   const Array<TMOP_Integrator *> &GetTMOPIntegrators() const { return tmopi; }

   /// Adds the limiting term to the first integrator. Disables it for the rest.
   void EnableLimiting(const GridFunction &n0, const GridFunction &dist,
                       Coefficient &w0, TMOP_LimiterFunction *lfunc = NULL);

   /** @brief Adds the limiting term to the first integrator. Disables it for
       the rest (@a dist in the general version of the method) equal to 1. */
   void EnableLimiting(const GridFunction &n0, Coefficient &w0,
                       TMOP_LimiterFunction *lfunc = NULL);

   /// Update the original/reference nodes used for limiting.
   void SetLimitingNodes(const GridFunction &n0);

   real_t GetElementEnergy(const FiniteElement &el,
                           ElementTransformation &T,
                           const Vector &elfun) override;
   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &T,
                              const Vector &elfun, Vector &elvect) override;
   void AssembleElementGrad(const FiniteElement &el,
                            ElementTransformation &T,
                            const Vector &elfun, DenseMatrix &elmat) override;

   virtual real_t GetRefinementElementEnergy(const FiniteElement &el,
                                             ElementTransformation &T,
                                             const Vector &elfun,
                                             const IntegrationRule &irule);

   virtual real_t GetDerefinementElementEnergy(const FiniteElement &el,
                                               ElementTransformation &T,
                                               const Vector &elfun);

   /// Normalization factor that considers all integrators in the combination.
   void EnableNormalization(const GridFunction &x);
#ifdef MFEM_USE_MPI
   void ParEnableNormalization(const ParGridFunction &x);
#endif

   // PA extension
   using NonlinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace&) override;
   void AssembleGradPA(const Vector&, const FiniteElementSpace&) override;
   real_t GetLocalStateEnergyPA(const Vector&) const override;
   void AddMultPA(const Vector&, Vector&) const override;
   void AddMultGradPA(const Vector&, Vector&) const override;
   void AssembleGradDiagonalPA(Vector&) const override;
};

/// Interpolates the @a metric's values at the nodes of @a metric_gf.
/** Assumes that @a metric_gf's FiniteElementSpace is initialized. */
void InterpolateTMOP_QualityMetric(TMOP_QualityMetric &metric,
                                   const TargetConstructor &tc,
                                   const Mesh &mesh, GridFunction &metric_gf);
}

#endif
