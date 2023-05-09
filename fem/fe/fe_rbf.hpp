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

   // This makes the shape parameter consistent across kernels
   virtual double HNorm() const = 0;
};

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

// Shifted down to be exactly zero at radius
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

// Truncated at radius
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

// Choose the type of RBF to use
class RBFType
{
public:
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

   // Convert rbf int to identifier
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

class KernelFiniteElement : public ScalarFiniteElement
{
public:
   KernelFiniteElement(int D, Geometry::Type G, int Do, int O, int F)
      : ScalarFiniteElement(D, G, Do, O, F) { }
   virtual ~KernelFiniteElement() { }

   // Converts integration rule to vector
   virtual void IntRuleToVec(const IntegrationPoint &ip,
                             Vector &vec) const;

   virtual bool IsCompact() const = 0;
   virtual const RBFFunction *Kernel() const = 0;

   // Get range of i,j,k indices that are nonzero for compact support
   virtual bool TensorIndexed() const { return false; }
   virtual void GetTensorIndices(const Vector &ip,
                                 int (&indices)[3][2]) const
   { MFEM_ABORT("GetTensorIndices(...)"); }
   virtual void GetTensorNumPoints(int (&tNumPoints)[3]) const
   { MFEM_ABORT("GetTensorNumPoints(...)"); }

   using FiniteElement::Project;

   virtual void Project(Coefficient &coeff,
                        ElementTransformation &Trans, Vector &dofs) const;

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

   virtual void GetPoly(const Vector &x,
                        Vector &p) const;
   virtual void GetDPoly(const Vector &x,
                         Vector &p,
                         Vector (&dp)[3]) const;
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
   // Should put in a method to calculate shape and dshape simultaneously
};

} // end namespace mfem

#endif
