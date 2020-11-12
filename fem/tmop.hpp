// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

   /** @brief The method SetTransformation() is hidden for TMOP_QualityMetric%s,
       because it is not used. */
   void SetTransformation(ElementTransformation &) { }

public:
   TMOP_QualityMetric() : Jtr(NULL) { }
   virtual ~TMOP_QualityMetric() { }

   /** @brief Specify the reference-element -> target-element Jacobian matrix
       for the point of interest.

       The specified Jacobian matrix, #Jtr, can be used by metrics that cannot
       be written just as a function of the target->physical Jacobian matrix,
       Jpt. */
   void SetTargetJacobian(const DenseMatrix &_Jtr) { Jtr = &_Jtr; }

   /** @brief Evaluate the strain energy density function, W = W(Jpt).
       @param[in] Jpt  Represents the target->physical transformation
                       Jacobian matrix. */
   virtual double EvalW(const DenseMatrix &Jpt) const = 0;

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
       the matrix invariants and their derivatives.
   */
   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const = 0;
};


/// Metric without a type, 2D
class TMOP_Metric_001 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<double> ie;

public:
   // W = |J|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Skew metric, 2D.
class TMOP_Metric_skew2D : public TMOP_QualityMetric
{
public:
   // W = 0.5 (1 - cos(angle_Jpr - angle_Jtr)).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
   { MFEM_ABORT("Not implemented"); }

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const
   { MFEM_ABORT("Not implemented"); }
};

/// Skew metric, 3D.
class TMOP_Metric_skew3D : public TMOP_QualityMetric
{
public:
   // W = 1/6 (3 - sum_i cos(angle_Jpr_i - angle_Jtr_i)), i = 1..3.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
   { MFEM_ABORT("Not implemented"); }

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const
   { MFEM_ABORT("Not implemented"); }
};

/// Aspect ratio metric, 2D.
class TMOP_Metric_aspratio2D : public TMOP_QualityMetric
{
public:
   // W = 0.5 (ar_Jpr/ar_Jtr + ar_Jtr/ar_Jpr) - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
   { MFEM_ABORT("Not implemented"); }

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const
   { MFEM_ABORT("Not implemented"); }
};

/// Aspect ratio metric, 3D.
class TMOP_Metric_aspratio3D : public TMOP_QualityMetric
{
public:
   // W = 1/3 sum [0.5 (ar_Jpr_i/ar_Jtr_i + ar_Jtr_i/ar_Jpr_i) - 1], i = 1..3.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
   { MFEM_ABORT("Not implemented"); }

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const
   { MFEM_ABORT("Not implemented"); }
};

/// Shape+Size+Orientation metric, 2D.
class TMOP_Metric_SSA2D : public TMOP_QualityMetric
{
public:
   // W = 0.5 (1 - cos(theta_Jpr - theta_Jtr)).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
   { MFEM_ABORT("Not implemented"); }

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const
   { MFEM_ABORT("Not implemented"); }
};

/// Shape, ideal barrier metric, 2D
class TMOP_Metric_002 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<double> ie;

public:
   // W = 0.5|J|^2 / det(J) - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Shape & area, ideal barrier metric, 2D
class TMOP_Metric_007 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<double> ie;

public:
   // W = |J - J^-t|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Shape & area metric, 2D
class TMOP_Metric_009 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<double> ie;

public:
   // W = det(J) * |J - J^-t|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Shifted barrier form of metric 2 (shape, ideal barrier metric), 2D
class TMOP_Metric_022 : public TMOP_QualityMetric
{
protected:
   double &tau0;
   mutable InvariantsEvaluator2D<double> ie;

public:
   TMOP_Metric_022(double &t0): tau0(t0) {}

   // W = 0.5(|J|^2 - 2det(J)) / (det(J) - tau0).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Shape, ideal barrier metric, 2D
class TMOP_Metric_050 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<double> ie;

public:
   // W = 0.5|J^t J|^2 / det(J)^2 - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Area metric, 2D
class TMOP_Metric_055 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<double> ie;

public:
   // W = (det(J) - 1)^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;

};

/// Area, ideal barrier metric, 2D
class TMOP_Metric_056 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<double> ie;

public:
   // W = 0.5( sqrt(det(J)) - 1 / sqrt(det(J)) )^2
   //   = 0.5( det(J) - 1 )^2 / det(J)
   //   = 0.5( det(J) + 1/det(J) ) - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;

};

/// Shape, ideal barrier metric, 2D
class TMOP_Metric_058 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<double> ie;

public:
   // W = |J^t J|^2 / det(J)^2 - 2|J|^2 / det(J) + 2
   //   = I1b (I1b - 2).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;

};

/// Area, ideal barrier metric, 2D
class TMOP_Metric_077 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator2D<double> ie;

public:
   // W = 0.5(det(J) - 1 / det(J))^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;

};

/// Shape & orientation metric, 2D.
class TMOP_Metric_085 : public TMOP_QualityMetric
{
public:
   // W = |T-T'|^2, where T'= |T|*I/sqrt(2).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const
   { MFEM_ABORT("Not implemented"); }

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const
   { MFEM_ABORT("Not implemented"); }
};

/// Untangling metric, 2D
class TMOP_Metric_211 : public TMOP_QualityMetric
{
protected:
   const double eps;
   mutable InvariantsEvaluator2D<double> ie;

public:
   TMOP_Metric_211(double epsilon = 1e-4) : eps(epsilon) { }

   // W = (det(J) - 1)^2 - det(J) + sqrt(det(J)^2 + eps).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Shifted barrier form of metric 56 (area, ideal barrier metric), 2D
class TMOP_Metric_252 : public TMOP_QualityMetric
{
protected:
   double &tau0;
   mutable InvariantsEvaluator2D<double> ie;

public:
   /// Note that @a t0 is stored by reference
   TMOP_Metric_252(double &t0): tau0(t0) {}

   // W = 0.5(det(J) - 1)^2 / (det(J) - tau0).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Shape, ideal barrier metric, 3D
class TMOP_Metric_301 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<double> ie;

public:
   // W = |J| |J^-1| / 3 - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Shape, ideal barrier metric, 3D
class TMOP_Metric_302 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<double> ie;

public:
   // W = |J|^2 |J^-1|^2 / 9 - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Shape, ideal barrier metric, 3D
class TMOP_Metric_303 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<double> ie;

public:
   // W = |J|^2 / 3 * det(J)^(2/3) - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Volume metric, 3D
class TMOP_Metric_315 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<double> ie;

public:
   // W = (det(J) - 1)^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Volume, ideal barrier metric, 3D
class TMOP_Metric_316 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<double> ie;

public:
   // W = 0.5( sqrt(det(J)) - 1 / sqrt(det(J)) )^2
   //   = 0.5( det(J) - 1 )^2 / det(J)
   //   = 0.5( det(J) + 1/det(J) ) - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Shape & volume, ideal barrier metric, 3D
class TMOP_Metric_321 : public TMOP_QualityMetric
{
protected:
   mutable InvariantsEvaluator3D<double> ie;

public:
   // W = |J - J^-t|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

/// Shifted barrier form of 3D metric 16 (volume, ideal barrier metric), 3D
class TMOP_Metric_352 : public TMOP_QualityMetric
{
protected:
   double &tau0;
   mutable InvariantsEvaluator3D<double> ie;

public:
   TMOP_Metric_352(double &t0): tau0(t0) {}

   // W = 0.5(det(J) - 1)^2 / (det(J) - tau0).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};


/// Base class for limiting functions to be used in class TMOP_Integrator.
/** This class represents a scalar function f(x, x0, d), where x and x0 are
    positions in physical space, and d is a reference physical distance
    associated with the point x0. */
class TMOP_LimiterFunction
{
public:
   /// Returns the limiting function, f(x, x0, d).
   virtual double Eval(const Vector &x, const Vector &x0, double d) const = 0;

   /** @brief Returns the gradient of the limiting function f(x, x0, d) with
       respect to x. */
   virtual void Eval_d1(const Vector &x, const Vector &x0, double dist,
                        Vector &d1) const = 0;

   /** @brief Returns the Hessian of the limiting function f(x, x0, d) with
       respect to x. */
   virtual void Eval_d2(const Vector &x, const Vector &x0, double dist,
                        DenseMatrix &d2) const = 0;

   /// Virtual destructor.
   virtual ~TMOP_LimiterFunction() { }
};

/// Default limiter function in TMOP_Integrator.
class TMOP_QuadraticLimiter : public TMOP_LimiterFunction
{
public:
   virtual double Eval(const Vector &x, const Vector &x0, double dist) const
   {
      MFEM_ASSERT(x.Size() == x0.Size(), "Bad input.");

      return 0.5 * x.DistanceSquaredTo(x0) / (dist * dist);
   }

   virtual void Eval_d1(const Vector &x, const Vector &x0, double dist,
                        Vector &d1) const
   {
      MFEM_ASSERT(x.Size() == x0.Size(), "Bad input.");

      d1.SetSize(x.Size());
      subtract(1.0 / (dist * dist), x, x0, d1);
   }

   virtual void Eval_d2(const Vector &x, const Vector &x0, double dist,
                        DenseMatrix &d2) const
   {
      MFEM_ASSERT(x.Size() == x0.Size(), "Bad input.");

      d2.Diag(1.0 / (dist * dist), x.Size());
   }

   virtual ~TMOP_QuadraticLimiter() { }
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

   int dim, ncomp;

public:
   AdaptivityEvaluator() : mesh(NULL), fes(NULL)
   {
#ifdef MFEM_USE_MPI
      pmesh = NULL;
      pfes = NULL;
#endif
   }
   virtual ~AdaptivityEvaluator();

   /** Specifies the Mesh and FiniteElementCollection of the solution that will
       be evaluated. The given mesh will be copied into the internal object. */
   void SetSerialMetaInfo(const Mesh &m,
                          const FiniteElementCollection &fec, int num_comp);

#ifdef MFEM_USE_MPI
   /// Parallel version of SetSerialMetaInfo.
   void SetParMetaInfo(const ParMesh &m,
                       const FiniteElementCollection &fec, int num_comp);
#endif

   // TODO use GridFunctions to make clear it's on the ldofs?
   virtual void SetInitialField(const Vector &init_nodes,
                                const Vector &init_field) = 0;

   virtual void ComputeAtNewPosition(const Vector &new_nodes,
                                     Vector &new_field) = 0;
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
   mutable double avg_volume;
   double volume_scale;
   const TargetType target_type;

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
   bool Parallel() const { return (comm != MPI_COMM_NULL); }
#else
   bool Parallel() const { return false; }
#endif

   // should be called only if avg_volume == 0.0, i.e. avg_volume is not
   // computed yet
   void ComputeAvgVolume() const;

public:
   /// Constructor for use in serial
   TargetConstructor(TargetType ttype)
      : nodes(NULL), avg_volume(), volume_scale(1.0), target_type(ttype)
   {
#ifdef MFEM_USE_MPI
      comm = MPI_COMM_NULL;
#endif
   }
#ifdef MFEM_USE_MPI
   /// Constructor for use in parallel
   TargetConstructor(TargetType ttype, MPI_Comm mpicomm)
      : nodes(NULL), avg_volume(), volume_scale(1.0), target_type(ttype),
        comm(mpicomm) { }
#endif
   virtual ~TargetConstructor() { }

   /** @brief Set the nodes to be used in the target-matrix construction.

       This method should be called every time the target nodes are updated
       externally and recomputation of the target average volume is needed. The
       nodes are used by all target types except IDEAL_SHAPE_UNIT_SIZE. */
   void SetNodes(const GridFunction &n) { nodes = &n; avg_volume = 0.0; }

   /// Used by target type IDEAL_SHAPE_EQUAL_SIZE. The default volume scale is 1.
   void SetVolumeScale(double vol_scale) { volume_scale = vol_scale; }

   /// Checks if the target matrices contain non-trivial size specification.
   virtual bool ContainsVolumeInfo() const;

   /** @brief Given an element and quadrature rule, computes ref->target
       transformation Jacobians for each quadrature point in the element.
       The physical positions of the element's nodes are given by @a elfun. */
   virtual void ComputeElementTargets(int e_id, const FiniteElement &fe,
                                      const IntegrationRule &ir,
                                      const Vector &elfun,
                                      DenseTensor &Jtr) const;

   virtual void ComputeElementTargetsGradient(const IntegrationRule &ir,
                                              const Vector &elfun,
                                              IsoparametricTransformation &Tpr,
                                              DenseTensor &dJtr) const;
};

class TMOPMatrixCoefficient  : public MatrixCoefficient
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
        scalar_tspec(NULL), vector_tspec(NULL), matrix_tspec(NULL) { }

   virtual void SetAnalyticTargetSpec(Coefficient *sspec,
                                      VectorCoefficient *vspec,
                                      TMOPMatrixCoefficient *mspec);

   /** @brief Given an element and quadrature rule, computes ref->target
       transformation Jacobians for each quadrature point in the element.
       The physical positions of the element's nodes are given by @a elfun. */
   virtual void ComputeElementTargets(int e_id, const FiniteElement &fe,
                                      const IntegrationRule &ir,
                                      const Vector &elfun,
                                      DenseTensor &Jtr) const;

   virtual void ComputeElementTargetsGradient(const IntegrationRule &ir,
                                              const Vector &elfun,
                                              IsoparametricTransformation &Tpr,
                                              DenseTensor &dJtr) const;
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
   Vector tspec;             //eta(x)
   Vector tspec_sav;
   Vector tspec_pert1h;      //eta(x+h)
   Vector tspec_pert2h;      //eta(x+2*h)
   Vector tspec_pertmix;     //eta(x+h,y+h)
   // The order inside these perturbation vectors (e.g. in 2D) is
   // eta1(x+h,y), eta2(x+h,y) ... etan(x+h,y), eta1(x,y+h), eta2(x,y+h) ...
   // same for tspec_pert2h and tspec_pertmix.

   // Components of Target Jacobian at each quadrature point of an element. This
   // is required for computation of the derivative using chain rule.
   mutable DenseTensor Jtrcomp;

   // Note: do not use the Nodes of this space as they may not be on the
   // positions corresponding to the values of tspec.
   const FiniteElementSpace *tspec_fes;
   const FiniteElementSpace *tspec_fesv;

   // These flags can be used by outside functions to avoid recomputing the
   // tspec and tspec_perth fields again on the same mesh.
   bool good_tspec, good_tspec_grad, good_tspec_hess;

   // Evaluation of the discrete target specification on different meshes.
   // Owned.
   AdaptivityEvaluator *adapt_eval;

   void SetDiscreteTargetBase(const GridFunction &tspec_);
   void SetTspecAtIndex(int idx, const GridFunction &tspec_);
   void FinalizeSerialDiscreteTargetSpec();
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
        tspec_fes(NULL), tspec_fesv(NULL),
        good_tspec(false), good_tspec_grad(false), good_tspec_hess(false),
        adapt_eval(NULL) { }

   virtual ~DiscreteAdaptTC()
   {
      delete adapt_eval;
      delete tspec_fes;
      delete tspec_fesv;
   }

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

   /** Used to update the target specification after the mesh has changed. The
       new mesh positions are given by new_x. If @a use_flags is true, repeated
       calls won't do anything until ResetUpdateFlags() is called. */
   void UpdateTargetSpecification(const Vector &new_x, bool use_flag = false);

   void UpdateTargetSpecification(Vector &new_x, Vector &IntData);

   void UpdateTargetSpecificationAtNode(const FiniteElement &el,
                                        ElementTransformation &T,
                                        int nodenum, int idir,
                                        const Vector &IntData);

   void RestoreTargetSpecificationAtNode(ElementTransformation &T, int nodenum);

   /** Used for finite-difference based computations. Computes the target
       specifications after a mesh perturbation in x or y direction.
       If @a use_flags is true, repeated calls won't do anything until
       ResetUpdateFlags() is called. */
   void UpdateGradientTargetSpecification(const Vector &x, double dx,
                                          bool use_flag = false);
   /** Used for finite-difference based computations. Computes the target
       specifications after two mesh perturbations in x and/or y direction.
       If @a use_flags is true, repeated calls won't do anything until
       ResetUpdateFlags() is called. */
   void UpdateHessianTargetSpecification(const Vector &x, double dx,
                                         bool use_flag = false);

   void SetAdaptivityEvaluator(AdaptivityEvaluator *ae)
   {
      if (adapt_eval) { delete adapt_eval; }
      adapt_eval = ae;
   }

   const Vector &GetTspecPert1H()   { return tspec_pert1h; }
   const Vector &GetTspecPert2H()   { return tspec_pert2h; }
   const Vector &GetTspecPertMixH() { return tspec_pertmix; }

   /** @brief Given an element and quadrature rule, computes ref->target
       transformation Jacobians for each quadrature point in the element.
       The physical positions of the element's nodes are given by @a elfun.
       Note that this function assumes that UpdateTargetSpecification() has
       been called with the position vector corresponding to @a elfun. */
   virtual void ComputeElementTargets(int e_id, const FiniteElement &fe,
                                      const IntegrationRule &ir,
                                      const Vector &elfun,
                                      DenseTensor &Jtr) const;

   virtual void ComputeElementTargetsGradient(const IntegrationRule &ir,
                                              const Vector &elfun,
                                              IsoparametricTransformation &Tpr,
                                              DenseTensor &dJtr) const;
};

class TMOPNewtonSolver;

/** @brief A TMOP integrator class based on any given TMOP_QualityMetric and
    TargetConstructor.

    Represents @f$ \int W(Jpt) dx @f$ over a target zone, where W is the
    metric's strain energy density function, and Jpt is the Jacobian of the
    target->physical coordinates transformation. The virtual target zone is
    defined by the TargetConstructor. */
class TMOP_Integrator : public NonlinearFormIntegrator
{
protected:
   friend class TMOPNewtonSolver;
   friend class TMOPComboIntegrator;

   TMOP_QualityMetric *metric;        // not owned
   const TargetConstructor *targetC;  // not owned

   // Custom integration rules.
   IntegrationRules *IntegRules;
   int integ_order;

   // Weight Coefficient multiplying the quality metric term.
   Coefficient *coeff1; // not owned, if NULL -> coeff1 is 1.
   // Normalization factor for the metric term.
   double metric_normal;

   // Nodes and weight Coefficient used for "limiting" the TMOP_Integrator.
   // These are both NULL when there is no limiting.
   // The class doesn't own nodes0 and coeff0.
   const GridFunction *nodes0;
   Coefficient *coeff0;
   // Limiting reference distance. Not owned.
   const GridFunction *lim_dist;
   // Limiting function. Owned.
   TMOP_LimiterFunction *lim_func;
   // Normalization factor for the limiting term.
   double lim_normal;

   // Adaptive limiting.
   const GridFunction *zeta_0;       // Not owned.
   GridFunction *zeta;               // Owned. Updated by adapt_eval.
   Coefficient *coeff_zeta;          // Not owned.
   AdaptivityEvaluator *adapt_eval;  // Not owned.

   DiscreteAdaptTC *discr_tc;

   // Parameters for FD-based Gradient & Hessian calculation.
   bool fdflag;
   double dx;
   double dxscale;
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

   void ComputeNormalizationEnergies(const GridFunction &x,
                                     double &metric_energy, double &lim_energy);


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

   void AssembleElemVecAdaptLim(const FiniteElement &el, const Vector &weights,
                                IsoparametricTransformation &Tpr,
                                const IntegrationRule &ir, DenseMatrix &m);
   void AssembleElemGradAdaptLim(const FiniteElement &el, const Vector &weights,
                                 IsoparametricTransformation &Tpr,
                                 const IntegrationRule &ir, DenseMatrix &m);

   double GetFDDerivative(const FiniteElement &el,
                          ElementTransformation &T,
                          Vector &elfun, const int nodenum,const int idir,
                          const double baseenergy, bool update_stored);

   /** @brief Determines the perturbation, h, for FD-based approximation. */
   void ComputeFDh(const Vector &x, const FiniteElementSpace &fes);
#ifdef MFEM_USE_MPI
   void ComputeFDh(const Vector &x, const ParFiniteElementSpace &pfes);
#endif
   void ComputeMinJac(const Vector &x, const FiniteElementSpace &fes);

   void UpdateAfterMeshChange(const Vector &new_x);

   void DisableLimiting()
   {
      nodes0 = NULL; coeff0 = NULL; lim_dist = NULL; lim_func = NULL;
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

public:
   /** @param[in] m  TMOP_QualityMetric that will be integrated (not owned).
       @param[in] tc Target-matrix construction algorithm to use (not owned). */
   TMOP_Integrator(TMOP_QualityMetric *m, TargetConstructor *tc)
      : metric(m), targetC(tc), IntegRules(NULL), integ_order(-1),
        coeff1(NULL), metric_normal(1.0),
        nodes0(NULL), coeff0(NULL),
        lim_dist(NULL), lim_func(NULL), lim_normal(1.0),
        zeta_0(NULL), zeta(NULL), coeff_zeta(NULL), adapt_eval(NULL),
        discr_tc(dynamic_cast<DiscreteAdaptTC *>(tc)),
        fdflag(false), dxscale(1.0e3), fd_call_flag(false), exact_action(false)
   { }

   ~TMOP_Integrator();

   /// Prescribe a set of integration rules; relevant for mixed meshes.
   /** This function has priority over SetIntRule(), if both are called. */
   void SetIntegrationRules(IntegrationRules &irules, int order)
   {
      IntegRules = &irules;
      integ_order = order;
   }

   /// Sets a scaling Coefficient for the quality metric term of the integrator.
   /** With this addition, the integrator becomes
          @f$ \int w1 W(Jpt) dx @f$.

       Note that the Coefficient is evaluated in the physical configuration and
       not in the target configuration which may be undefined. */
   void SetCoefficient(Coefficient &w1) { coeff1 = &w1; }

   /** @brief Limiting of the mesh displacements (general version).

       Adds the term @f$ \int w_0 f(x, x_0, d) dx @f$, where f is a measure of
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

       Adds the term @f$ \int c (z(x) - z_0(x_0))^2 @f$, where z0(x0) is a given
       function on the starting mesh, and z(x) is its image on the new mesh.
       Minimizing this, means that a node at x0 is allowed to move to a
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

   /// Update the original/reference nodes used for limiting.
   void SetLimitingNodes(const GridFunction &n0) { nodes0 = &n0; }

   /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone.
       @param[in] el     Type of FiniteElement.
       @param[in] T      Mesh element transformation.
       @param[in] elfun  Physical coordinates of the zone. */
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &T,
                                   const Vector &elfun);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &T,
                                      const Vector &elfun, Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &T,
                                    const Vector &elfun, DenseMatrix &elmat);

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

   void   SetFDhScale(double _dxscale) { dxscale = _dxscale; }
   bool   GetFDFlag() const { return fdflag; }
   double GetFDh()    const { return dx; }

   /** @brief Flag to control if exact action of Integration is effected. */
   void SetExactActionFlag(bool flag_) { exact_action = flag_; }
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

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &T,
                                   const Vector &elfun);
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &T,
                                      const Vector &elfun, Vector &elvect);
   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &T,
                                    const Vector &elfun, DenseMatrix &elmat);

   /// Normalization factor that considers all integrators in the combination.
   void EnableNormalization(const GridFunction &x);
#ifdef MFEM_USE_MPI
   void ParEnableNormalization(const ParGridFunction &x);
#endif
};

/// Interpolates the @a metric's values at the nodes of @a metric_gf.
/** Assumes that @a metric_gf's FiniteElementSpace is initialized. */
void InterpolateTMOP_QualityMetric(TMOP_QualityMetric &metric,
                                   const TargetConstructor &tc,
                                   const Mesh &mesh, GridFunction &metric_gf);
}

#endif
