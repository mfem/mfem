// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TMOP_HPP
#define MFEM_TMOP_HPP

#include "../config/config.hpp"
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


/** @brief Base class representing target-matrix construction algorithms for
    mesh optimization via the target-matrix optimization paradigm (TMOP). */
/** This class is used by class TMOP_Integrator to construct the target Jacobian
    matrices (reference-element to target-element) at quadrature points. It
    supports a set of algorithms chosen by the #TargetType enumeration.

    New target-matrix construction algorithms can be defined by deriving new
    classes and overriding the method ComputeElementTargets(). */
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
      GIVEN_SHAPE_AND_SIZE /**<
         Given shape, given size/volume; the given nodes define the exact target
         Jacobian matrix at all quadrature points. */
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

   /** @brief Given an element and quadrature rule, computes ref->target
       transformation Jacobians for each quadrature point in the element. */
   virtual void ComputeElementTargets(int e_id, const FiniteElement &fe,
                                      const IntegrationRule &ir,
                                      DenseTensor &Jtr) const;
};

class ParGridFunction;

/** @brief A TMOP integrator class based on any given TMOP_QualityMetric and
    TargetConstructor.

    Represents @f$ \int W(Jpt) dx @f$ over a target zone, where W is the
    metric's strain energy density function, and Jpt is the Jacobian of the
    target->physical coordinates transformation. The virtual target zone is
    defined by the TargetConstructor. */
class TMOP_Integrator : public NonlinearFormIntegrator
{
protected:
   TMOP_QualityMetric *metric;        // not owned
   const TargetConstructor *targetC;  // not owned

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

public:
   /** @param[in] m  TMOP_QualityMetric that will be integrated (not owned).
       @param[in] tc Target-matrix construction algorithm to use (not owned). */
   TMOP_Integrator(TMOP_QualityMetric *m, TargetConstructor *tc)
      : metric(m), targetC(tc),
        coeff1(NULL), metric_normal(1.0),
        nodes0(NULL), coeff0(NULL),
        lim_dist(NULL), lim_func(NULL), lim_normal(1.0)
   { }

   ~TMOP_Integrator() { delete lim_func; }

   /// Sets a scaling Coefficient for the quality metric term of the integrator.
   /** With this addition, the integrator becomes
          @f$ \int w1 W(Jpt) dx @f$.

       Note that the Coefficient is evaluated in the physical configuration and
       not in the target configuration which may be undefined. */
   void SetCoefficient(Coefficient &w1) { coeff1 = &w1; }

   /// Adds a limiting term to the integrator (general version).
   /** With this addition, the integrator becomes
          @f$ \int w1 W(Jpt) + w0 f(x, x_0, d) dx @f$,
       where the second term measures the change with respect to the original
       physical positions, @a n0.
       @param[in] n0     Original mesh node coordinates.
       @param[in] dist   Limiting physical distances.
       @param[in] w0     Coefficient scaling the limiting term.
       @param[in] lfunc  TMOP_LimiterFunction defining the limiting term f. If
                         NULL, a TMOP_QuadraticLimiter will be used. The
                         TMOP_Integrator assumes ownership of this pointer. */
   void EnableLimiting(const GridFunction &n0, const GridFunction &dist,
                       Coefficient &w0, TMOP_LimiterFunction *lfunc = NULL);

   /** @brief Adds a limiting term to the integrator with limiting distance
       function (@a dist in the general version of the method) equal to 1. */
   void EnableLimiting(const GridFunction &n0,
                       Coefficient &w0, TMOP_LimiterFunction *lfunc = NULL);

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

   /** @brief Computes the normalization factors of the metric and limiting
       integrals using the mesh position given by @a x. */
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
