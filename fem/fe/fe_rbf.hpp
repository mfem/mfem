// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_FE_RBF
#define MFEM_FE_RBF

#include "fe_base.hpp"

namespace mfem
{

// Pure virtual class for dimensionless radial basis functions
class RBFFunction
{
public:
   static const double GlobalRadius; // functions with r>=GR are considered global
   RBFFunction() { };
   virtual ~RBFFunction() { }

   // The r is a normalized distance
   virtual double BaseFunction(double r) const = 0;
   virtual double BaseDerivative(double r) const = 0;
   virtual double BaseDerivative2(double r) const = 0;

   // The support radius, outside of which the function is zero
   virtual double Radius() const { return GlobalRadius; }

   // Does function have compact support?
   virtual bool CompactSupport() const { return false; }

   /* This normalizes the smoothing parameter h such that h doesn't need to
      be changed based on the choice of basis function */
   virtual double HNorm() const = 0;
};

// Gaussian RBF, exp(-r^2)
class GaussianRBF : public RBFFunction
{
   // hNorm minimizes integral of Gaussian minus Wendland kernel over r=0,1
   static const double hNorm;
public:
   GaussianRBF() { };
   virtual ~GaussianRBF() { }

   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double HNorm() const { return hNorm; }
};

// Multiquadric RBF, sqrt(1+r^2)
class MultiquadricRBF : public RBFFunction
{
   // Same as inverse multiquadric
   static const double hNorm;
public:
   MultiquadricRBF() { };
   virtual ~MultiquadricRBF() { }

   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double HNorm() const { return hNorm; }
};

// Inverse multiquadric RBF, 1/sqrt(1+r^2)
class InvMultiquadricRBF : public RBFFunction
{
   // hNorm minimizes integral of Gaussian minus InvMQ kernel over r=0,0.5
   static const double hNorm;
public:
   InvMultiquadricRBF() { };
   virtual ~InvMultiquadricRBF() { }

   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double HNorm() const { return hNorm; }
};

/* Same as the Gaussian, but subtracted by a factor such that the
   function is exactly zero at the chosen radius */
class CompactGaussianRBF : public RBFFunction
{
   static const double hNorm;
   const double radius;
   double multK, shiftK;

public:
   CompactGaussianRBF(const double rad = 5.0);
   virtual ~CompactGaussianRBF() { }

   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double Radius() const { return radius; }

   virtual double HNorm() const { return hNorm; }
   virtual bool CompactSupport() const { return true; }
};

// Same as the Gaussian, but truncated (set to zero) at the chosen radius
class TruncatedGaussianRBF : public RBFFunction
{
   static const double hNorm;
   const double radius;
public:
   TruncatedGaussianRBF(const double rad = 5.0)
      : radius(rad) { }
   virtual ~TruncatedGaussianRBF() { }

   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double Radius() const { return radius; }

   virtual double HNorm() const { return hNorm; }
   virtual bool CompactSupport() const { return true; }
};

// Wendland 11 RBF, (1-r)^3 * (1+3r) if r < 1
class Wendland11RBF : public RBFFunction
{
   static const double radius;

public:
   Wendland11RBF() { }
   virtual ~Wendland11RBF() { }

   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double Radius() const { return radius; }

   virtual double HNorm() const { return 1.0 / radius; }
   virtual bool CompactSupport() const { return true; }
};

// Wendland 31 RBF, (1-r)^4 * (1+4r) if r < 1
class Wendland31RBF : public RBFFunction
{
   static const double radius;

public:
   Wendland31RBF() { };
   virtual ~Wendland31RBF() { }

   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double Radius() const { return radius; }

   virtual double HNorm() const { return 1.0 / radius; }
   virtual bool CompactSupport() const { return true; }
};

// Wendland 33 RBF, (1-r)^8 * (1+8r+25r^2+32r^3) if r < 1
class Wendland33RBF : public RBFFunction
{
   static const double radius;

public:
   Wendland33RBF() { };
   virtual ~Wendland33RBF() { }

   virtual double BaseFunction(double r) const;
   virtual double BaseDerivative(double r) const;
   virtual double BaseDerivative2(double r) const;

   virtual double Radius() const { return radius; }

   virtual double HNorm() const { return 1.0 / radius; }
   virtual bool CompactSupport() const { return true; }
};

// Class for storing and creating the various RBFs
class RBFType
{
public:
   // Assign integers to each type of function
   enum
   {
      Gaussian = 0,
      Multiquadric = 1,
      InvMultiquadric = 2,
      TruncatedGaussian = 3,
      CompactGaussian = 4,
      Wendland11 = 5,
      Wendland31 = 6,
      Wendland33 = 7,
      NumRBFTypes = 8
   };

   // Return the requested RBF
   static RBFFunction *GetRBF(const int rbfType)
   {
      switch (rbfType)
      {
         case RBFType::Gaussian:
            return new GaussianRBF();
         case RBFType::Multiquadric:
            return new MultiquadricRBF();
         case RBFType::InvMultiquadric:
            return new InvMultiquadricRBF();
         case RBFType::TruncatedGaussian:
            return new TruncatedGaussianRBF();
         case RBFType::CompactGaussian:
            return new CompactGaussianRBF();
         case RBFType::Wendland11:
            return new Wendland11RBF();
         case RBFType::Wendland31:
            return new Wendland31RBF();
         case RBFType::Wendland33:
            return new Wendland33RBF();
      }
      MFEM_ABORT("unknown RBF type");
      return NULL;
   }

   // Abort if rbfType is invalid
   static int Check(const int rbfType)
   {
      MFEM_VERIFY(0 <= rbfType && rbfType < NumRBFTypes,
                  "unknown RBF type: " << rbfType);
      return rbfType;
   }

   // Convert rbf int to identifier for storage
   static char GetChar(const int rbfType)
   {
      static const char ident[] = { 'G', 'M', 'I',
                                    'T', 'C',
                                    '1', '3', '6'
                                  };
      return ident[Check(rbfType)];
   }

   // Convert identifier to rbf int
   static int GetType(const char rbfIdent)
   {
      switch (rbfIdent)
      {
         case 'G': return Gaussian;
         case 'M': return Multiquadric;
         case 'I': return InvMultiquadric;
         case 'T': return TruncatedGaussian;
         case 'C': return CompactGaussian;
         case '1': return Wendland11;
         case '3': return Wendland31;
         case '6': return Wendland33;
      }
      MFEM_ABORT("unknown RBF identifier: " << rbfIdent);
      return -1;
   }
};

// Dimensionless distance metrics, whose output is the input for RBFs
class DistanceMetric
{
protected:
   int dim;
public:
   DistanceMetric(int D) { dim = D; }
   virtual ~DistanceMetric() { }

   virtual void SetDim(int D) { dim = D; }

   virtual void Distance(const Vector &x,
                         double &r) const = 0;
   virtual void DDistance(const Vector &x,
                          Vector &dr) const = 0;
   virtual void DDDistance(const Vector &x,
                           DenseMatrix &ddr) const = 0;

   static DistanceMetric *GetDistance(int dim, int pnorm);
};

// Dimensionless distance with r = |x| + |y| + ...
class L1Distance : public DistanceMetric
{
public:
   L1Distance(int D) : DistanceMetric(D) { };
   virtual ~L1Distance() { }

   virtual void Distance(const Vector &x,
                         double &r) const;
   virtual void DDistance(const Vector &x,
                          Vector &dr) const;
   virtual void DDDistance(const Vector &x,
                           DenseMatrix &ddr) const;
};

// Dimensionless distance with r = (x^2 + y^2 + ...)^(1/2)
class L2Distance : public DistanceMetric
{
public:
   L2Distance(int D) : DistanceMetric(D) { };
   virtual ~L2Distance() { }

   virtual void Distance(const Vector &x,
                         double &r) const;
   virtual void DDistance(const Vector &x,
                          Vector &dr) const;
   virtual void DDDistance(const Vector &x,
                           DenseMatrix &ddr) const;
};

// Dimensionless distance with r = (x^p + y^p + ...)^(1/p)
class LpDistance : public DistanceMetric
{
   const int p;
   const double pinv;
public:
   LpDistance(int D, int pnorm)
      : DistanceMetric(D),
        p(pnorm),
        pinv(1. / static_cast<double>(p))
   { };
   virtual ~LpDistance() { }

   virtual void Distance(const Vector &x,
                         double &r) const;
   virtual void DDistance(const Vector &x,
                          Vector &dr) const;
   virtual void DDDistance(const Vector &x,
                           DenseMatrix &ddr) const;
};


/* Pure virtual class for a finite element with radial basis functions
   instead of polynomials inside each element */
class KernelFiniteElement : public ScalarFiniteElement
{
private:
   // Choose whether to interpolate or project when Project is called
   bool interpolate = false;
public:
   KernelFiniteElement(int D, Geometry::Type G, int Do, int O, int F)
      : ScalarFiniteElement(D, G, Do, O, F) { }
   virtual ~KernelFiniteElement() { }

   // Converts integration rule to vector
   virtual void IntRuleToVec(const IntegrationPoint &ip,
                             Vector &vec) const;

   // Is base RBF compact?
   virtual bool IsCompact() const = 0;

   // Return base kernel
   virtual const RBFFunction *Kernel() const = 0;

   // Get range of i,j,k indices that are nonzero for compact support
   virtual bool TensorIndexed() const { return false; }
   virtual void GetTensorIndices(const Vector &ip,
                                 int (&indices)[3][2]) const
   { MFEM_ABORT("GetTensorIndices(...)"); }
   virtual void GetTensorNumPoints(int (&tNumPoints)[3]) const
   { MFEM_ABORT("GetTensorNumPoints(...)"); }

   using FiniteElement::Project;

   virtual void Project(Coefficient &coeff, ElementTransformation &Trans,
                        Vector &dofs) const;

   virtual void Project(VectorCoefficient &vc, ElementTransformation &Trans,
                        Vector &dofs) const;

   virtual void Project(const FiniteElement &fe, ElementTransformation &Trans,
                        DenseMatrix &I) const;

   virtual void GetLocalInterpolation(ElementTransformation &Trans,
                                      DenseMatrix &I) const
   { ScalarLocalInterpolation(Trans, I, *this); }

   virtual void GetTransferMatrix(const FiniteElement &fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &I) const
   { CheckScalarFE(fe).ScalarLocalInterpolation(Trans, I, *this); }
};

// Finite element using base radial basis functions without polynomial corrections
class RBFFiniteElement : public KernelFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable double r_scr, f_scr, df_scr, ddf_scr;
   mutable Vector x_scr, y_scr, dy_scr, dr_scr;
   mutable DenseMatrix ddr_scr;
   mutable int cInd[3][2];
#endif
   bool isCompact; // Is the RBF with the given h compact?
   int dimPoints[3];
   int numPointsD; // Number of points across the element in each D
   double delta; // Distance between points
   double h; // Shape parameter, approx number of points in 1d support radius
   double hPhys; // Shape parameter times distance between points times HNorm
   double hPhysInv; // Inverse hPhys
   double radPhys; // Radius adjusted by h
   const RBFFunction *rbf;
   const DistanceMetric *distance;
   void InitializeGeometry();

   // Get the dimensionless distance from x to the center of the RBF indexed i
   virtual void DistanceVec(const int i,
                            const Vector &x,
                            Vector &y) const;

public:
   RBFFiniteElement(const int D,
                    const int numPointsD,
                    const double h,
                    const int rbfType,
                    const int distNorm,
                    const int intOrder);
   virtual ~RBFFiniteElement() { delete rbf; delete distance; }

   virtual bool TensorIndexed() const { return true; }
   virtual void GetCompactIndices(const Vector &ip,
                                  int (&indices)[3][2]) const;
   virtual void GetGlobalIndices(const Vector &ip,
                                 int (&indices)[3][2]) const;
   virtual void GetTensorIndices(const Vector &ip,
                                 int (&indices)[3][2]) const;
   virtual void GetTensorNumPoints(int (&tNumPoints)[3]) const
   {
      tNumPoints[0] = dimPoints[0];
      tNumPoints[1] = dimPoints[1];
      tNumPoints[2] = dimPoints[2];
   }

   virtual bool IsCompact() const { return isCompact; }
   virtual const RBFFunction *Kernel() const { return rbf; }

   virtual void CalcShape(const IntegrationPoint &ip,
                          Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   virtual void CalcHessian(const IntegrationPoint &ip,
                            DenseMatrix &h) const;
};

/* Reproducing kernel finite element, which includes polynomial corrections
   to the standard radial basis function finite element to guarantee a
   chosen order of accuracy */
class RKFiniteElement : public KernelFiniteElement
{
private:
#ifndef MFEM_THREAD_SAFE
   mutable double f_scr;
   mutable Vector x_scr, y_scr, g_scr, c_scr, s_scr, p_scr, df_scr;
   mutable DenseMatrix q_scr, dq_scr, M_scr;
   mutable Vector dc_scr[3], dp_scr[3];
   mutable DenseMatrix dM_scr[3];
   mutable DenseMatrixInverse Minv_scr;
   mutable int cInd[3][2];
   mutable int dimPoints[3];
#endif
   int polyOrd, numPoly, numPoly1d;
   KernelFiniteElement *baseFE;

   // Get the vector of polynomials for the corrections, evaluated at x
   virtual void GetPoly(const Vector &x,
                        Vector &p) const;
   virtual void GetDPoly(const Vector &x,
                         Vector &p,
                         Vector (&dp)[3]) const;

   /* Helper functions that return pieces of the RK evaluation
      W_{RK,i} = P_i^T C_i W_{RBF,i}, where
      P_i = [1, x, y, z, ...] is the polynomial vector evaluated at the point i,
      M_i = P_i P_i^T W_{RBF,i} is a matrix used in calculating the corrections,
      C_i = M_i^{-1} G are the RK corrections,
      G = [1, 0, 0, ...] is a convencience vector */
   virtual void GetG(Vector &g) const;
   virtual void GetM(const Vector &baseShape,
                     const IntegrationPoint &ip,
                     DenseMatrix &M) const;
   virtual void GetDM(const Vector &baseShape,
                      const DenseMatrix &baseDeriv,
                      const IntegrationPoint &ip,
                      DenseMatrix &M,
                      DenseMatrix (&dM)[3]) const;
   virtual void AddToM(const Vector &p,
                       const double &f,
                       DenseMatrix &M) const;
   virtual void AddToDM(const Vector &p,
                        const Vector (&dp)[3],
                        const double &f,
                        const Vector &df,
                        DenseMatrix (&dM)[3]) const;

   // Given the corrections and base values, calculate the RK value at the ip
   virtual void CalculateValues(const Vector &c,
                                const Vector &baseShape,
                                const IntegrationPoint &ip,
                                Vector &shape) const;
   virtual void CalculateDValues(const Vector &c,
                                 const Vector (&dc)[3],
                                 const Vector &baseShape,
                                 const DenseMatrix &baseDShape,
                                 const IntegrationPoint &ip,
                                 DenseMatrix &dshape) const;

   // The corrections make the distance dimensionless, so no smoothing parameter
   virtual void DistanceVec(const int i,
                            const Vector &x,
                            Vector &y) const;

public:
   RKFiniteElement(const int D,
                   const int numPointsD,
                   const double h,
                   const int rbfType,
                   const int distNorm,
                   const int order,
                   const int intOrder);
   virtual ~RKFiniteElement() { delete baseFE; }

   virtual bool IsCompact() const { return baseFE->IsCompact(); }
   virtual const RBFFunction *Kernel() const { return baseFE->Kernel(); }

   static int GetNumPoly(int polyOrd, int dim);

   virtual void CalcShape(const IntegrationPoint &ip,
                          Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
   // Could put in a method to calculate shape and dshape simultaneously
};

} // end namespace mfem

#endif
