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

   /// First invariant of the given 2x2 matrix @a M.
   static double Dim2Invariant1(const DenseMatrix &M);
   /// Second invariant of the given 2x2 matrix @a M.
   static double Dim2Invariant2(const DenseMatrix &M);

   /// 1st derivative of the first invariant for the given 2x2 matrix @a M.
   static void Dim2Invariant1_dM(const DenseMatrix &M, DenseMatrix &dM);
   /// 1st derivative of the second invariant for the given 2x2 matrix @a M.
   static void Dim2Invariant2_dM(const DenseMatrix &M, DenseMatrix &dM);

   /// 2nd derivative of the first invariant for the given 2x2 matrix @a M.
   static void Dim2Invariant1_dMdM(const DenseMatrix &M, int i, int j,
                                   DenseMatrix &dMdM);
   /// 2nd derivative of the second invariant for the given 2x2 matrix @a M.
   static void Dim2Invariant2_dMdM(const DenseMatrix &M, int i, int j,
                                   DenseMatrix &dMdM);

   /// First invariant of the given 3x3 matrix @a M.
   static double Dim3Invariant1(const DenseMatrix &M);
   /// Second invariant of the given 3x3 matrix @a M.
   static double Dim3Invariant2(const DenseMatrix &M);
   /// Third invariant of the given 3x3 matrix @a M.
   static double Dim3Invariant3(const DenseMatrix &M);

   /// 1st derivative of the first invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant1_dM(const DenseMatrix &M, DenseMatrix &dM);
   /// 1st derivative of the second invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant2_dM(const DenseMatrix &M, DenseMatrix &dM);
   /// 1st derivative of the third invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant3_dM(const DenseMatrix &M, DenseMatrix &dM);

   /// 2nd derivative of the first invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant1_dMdM(const DenseMatrix &M, int i, int j,
                                   DenseMatrix &dMdM);
   /// 2nd derivative of the second invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant2_dMdM(const DenseMatrix &M, int i, int j,
                                   DenseMatrix &dMdM);
   /// 2nd derivative of the third invariant for the given 3x3 matrix @a M.
   static void Dim3Invariant3_dMdM(const DenseMatrix &M, int i, int j,
                                   DenseMatrix &dMdM);

public:
   TMOP_QualityMetric() : Jtr(NULL) { }
   virtual ~TMOP_QualityMetric() { }

   /** @brief Specify the reference-element -> target-element Jacobian matrix
       for the point of interest.

       Using #Jtr is an alternative to using #Ttr, when one cannot define
       the target Jacobians by a single ElementTransformation for the whole
       zone, e.g., in the TMOP paradigm. */
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


class TMOP_Metric_001 : public TMOP_QualityMetric
{
public:
   // W = |J|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

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

class TMOP_Metric_007 : public TMOP_QualityMetric
{
public:
   // W = |J - J^-t|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOP_Metric_009 : public TMOP_QualityMetric
{
public:
   // W = det(J) * |J - J^-t|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOP_Metric_022 : public TMOP_QualityMetric
{
private:
   double &tau0;

public:
   TMOP_Metric_022(double &t0): tau0(t0) {}

   // W = 0.5(|J|^2 - 2det(J)) / (det(J) - tau0).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOP_Metric_050 : public TMOP_QualityMetric
{
public:
   // W = 0.5|J^t J|^2 / det(J)^2 - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOP_Metric_052 : public TMOP_QualityMetric
{
private:
   double &tau0;

public:
   TMOP_Metric_052(double &t0): tau0(t0) {}

   // W = 0.5(det(J) - 1)^2 / (det(J) - tau0).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOP_Metric_055 : public TMOP_QualityMetric
{
public:
   // W = (det(J) - 1)^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;

};

class TMOP_Metric_056 : public TMOP_QualityMetric
{
public:
   // W = 0.5( sqrt(det(J)) - 1 / sqrt(det(J)) )^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;

};

class TMOP_Metric_058 : public TMOP_QualityMetric
{
public:
   // W = |J^t J|^2 / det(J)^2 - 2|J|^2 / det(J) + 2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;

};

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

class TMOP_Metric_211 : public TMOP_QualityMetric
{
private:
   const double eps;

public:
   TMOP_Metric_211() : eps(1e-4) { }

   // W = (det(J) - 1)^2 - det(J) + sqrt(det(J)^2 + eps).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

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

class TMOP_Metric_302 : public TMOP_QualityMetric
{
public:
   // W = |J|^2 |J^-1|^2 / 9 - 1.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

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

class TMOP_Metric_315 : public TMOP_QualityMetric
{
public:
   // W = (det(J) - 1)^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOP_Metric_316 : public TMOP_QualityMetric
{
public:
   // W = 0.5( sqrt(det(J)) - 1 / sqrt(det(J)) )^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOP_Metric_321 : public TMOP_QualityMetric
{
public:
   // W = |J - J^-t|^2.
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};

class TMOP_Metric_352 : public TMOP_QualityMetric
{
private:
   double &tau0;

public:
   TMOP_Metric_352(double &t0): tau0(t0) {}

   // W = 0.5(det(J) - 1)^2 / (det(J) - tau0).
   virtual double EvalW(const DenseMatrix &Jpt) const;

   virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const;

   virtual void AssembleH(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const;
};


/** This class is used to compute reference-element -> target-element Jacobian
    for different target options; used in class TMOP_Integrator. */
class TargetJacobian
{
private:
   // Current nodes, initial nodes, target nodes that are
   // used in ComputeElementTargets(int), depending on target_type.
   const GridFunction *nodes, *nodes0, *tnodes;
   double avg_volume0;
   const bool serial_use;

   static void ConstructIdealJ(int geom, DenseMatrix &J);

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif

public:
   enum target {CURRENT, IDEAL, IDEAL_EQ_SIZE, IDEAL_INIT_SIZE, TARGET_MESH};
   const target target_type;

   // Additional scaling applied for IDEAL_EQ_SIZE. When the target is active
   // only in some part of the domain, this can be used to set relative sizes.
   double size_scale;

   TargetJacobian(target ttype)
      : nodes(NULL), nodes0(NULL), tnodes(NULL), serial_use(true),
        target_type(ttype), size_scale(1.0) { }
#ifdef MFEM_USE_MPI
   TargetJacobian(target ttype, MPI_Comm mpicomm)
      : nodes(NULL), nodes0(NULL), tnodes(NULL), serial_use(false),
        comm(mpicomm), target_type(ttype), size_scale(1.0) { }
#endif
   ~TargetJacobian() { }

   void SetNodes(const GridFunction &n)         { nodes  = &n;  }
   // Sets initial nodes and computes the average volume of the initial mesh.
   void SetInitialNodes(const GridFunction &n0);
   void SetTargetNodes(const GridFunction &tn)  { tnodes = &tn; }

   /** @brief Given an element and quadrature rule, computes ref->target
       transformation Jacobians for each quadrature point in the element. */
   void ComputeElementTargets(int e_id, const FiniteElement &fe,
                              const IntegrationRule &ir,
                              DenseTensor &Jtr) const;
};


/// A TMOP integrator class based on any given TMOP_QualityMetric.
/** Represents @f$ \int W(Jpt) dx @f$ over a target zone, where W is the
    metric's strain energy density function, and Jpt is the Jacobian of the
    target->physical coordinates transformation. */
class TMOP_Integrator : public NonlinearFormIntegrator
{
private:
   TMOP_QualityMetric *metric;
   const TargetJacobian *targetJ;

   // Data used for "limiting" the HyperelasticNLFIntegrator.
   bool limited;
   double eps;
   const GridFunction *nodes0;

   // Can be used to create "composite" integrators for the TMOP purposes.
   Coefficient *coeff;

   //   Jrt: the inverse of the ref->target transformation Jacobian.
   //   Jpr: the ref->physical transformation Jacobian.
   //   Jpt: the target->physical transformation Jacobians.
   //     P: represents dW_d(Jtp) (dim x dim).
   //   DSh: gradients of reference shape functions (dof x dim).
   //    DS: represents d(Jtp)_dx (dof x dim).
   // PMatI: current coordinates of the nodes (dof x dim).
   // PMat0: represents dW_dx (dof x dim).
   DenseMatrix DSh, DS, Jrt, Jpr, Jpt, P, PMatI, PMatO;

public:
   /** @param[in] m  TMOP_QualityMetric that will be integrated.
       @param[in] tJ See TMOP_QualityMetric::SetTargetJacobian(). */
   TMOP_Integrator(TMOP_QualityMetric *m, TargetJacobian *tJ = NULL)
      : metric(m), targetJ(tJ),
        limited(false), eps(0.0), nodes0(NULL), coeff(NULL) { }

   const TargetJacobian *GetTargetJacobian() { return targetJ; } const

   /// Adds an extra term to the integral.
   /** The integral of interest becomes
       @f$ \int \epsilon F(T) + 0.5 (x - x_0)^2 dx@f$,
       where the second term measures the change with respect to the
       original physical positions.
       @param[in] eps_  Scaling of the @a model's contribution.
       @param[in] n0    Original mesh coordinates. */
   void SetLimited(double eps_, const GridFunction &n0)
   {
      limited = true;
      eps = eps_;
      nodes0 = &n0;
   }

   /// Sets a scaling Coefficient for the integral.
   void SetCoefficient(Coefficient &c) { coeff = &c; }

   /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
       @param[in] el     Type of FiniteElement.
       @param[in] Ttr    Represents ref->target coordinates transformation.
       @param[in] elfun  Physical coordinates of the zone. */
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Ttr,
                                   const Vector &elfun);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Ttr,
                                      const Vector &elfun, Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Ttr,
                                    const Vector &elfun, DenseMatrix &elmat);

   virtual ~TMOP_Integrator();
};


/// Interpolates the @a metric's values at the nodes of @a gf.
/** Assumes that @a gf's FiniteElementSpace is initialized. */
void InterpolateTMOP_QualityMetric(TMOP_QualityMetric &metric,
                                   const TargetJacobian &tj,
                                   const Mesh &mesh, GridFunction &gf);

}

#endif
