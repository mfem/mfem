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

#ifndef MFEM_META_MATERIAL_SOLVER
#define MFEM_META_MATERIAL_SOLVER

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../common/pfem_extras.hpp"

namespace mfem
{

using miniapps::H1_ParFESpace;
using miniapps::ND_ParFESpace;
using miniapps::RT_ParFESpace;
using miniapps::L2_ParFESpace;
using miniapps::ParDiscreteGradOperator;
using miniapps::VisData;

namespace meta_material
{

class LinearCoefficient : public GridFunctionCoefficient
{
public:
   LinearCoefficient(GridFunction * gf, double a0, double a1);

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);

   double GetSensitivity(ElementTransformation &T,
                         const IntegrationPoint &ip)
   { return (c1_ - c0_); }

private:
   double c0_;
   double c1_;
};
/*
class PenaltyCoefficient : public GridFunctionCoefficient
{
public:
   PenaltyCoefficient(GridFunction * gf, int penalty, double a0, double a1);

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);

   double GetSensitivity(ElementTransformation &T,
                         const IntegrationPoint &ip);

private:
   int penalty_;
   double c0_;
   double c1_;
};
*/
class Homogenization
{
public:
   Homogenization(MPI_Comm comm);
   virtual ~Homogenization() {}

   void SetVolumeFraction(ParGridFunction & vf) { vf_ = &vf; newVF_ = true; }

   virtual void GetHomogenizedProperties(std::vector<double> & p) = 0;
   virtual void GetPropertySensitivities(std::vector<ParGridFunction> & dp)
      = 0;

   void InitializeGLVis(VisData & vd) {}

   void DisplayToGLVis() {}

   void WriteVisItFields(const std::string & prefix,
                         const std::string & label) {}

protected:
   MPI_Comm comm_;
   int      myid_;
   int      numProcs_;
   bool     newVF_;

   ParGridFunction * vf_;
};

class Density : public Homogenization
{
public:
   // Density(ParMesh & pmesh, double vol,
   //       double rho0, double rho1);
   Density(ParMesh & pmesh, double refDensity, double vol,
           Coefficient &rhoCoef, double tol = 0.05);
   ~Density();

   // void SetVolumeFraction(ParGridFunction & vf);

   void GetHomogenizedProperties(std::vector<double> & p);
   void GetPropertySensitivities(std::vector<ParGridFunction> & dp) {}

   void InitializeGLVis(VisData & vd);

   void DisplayToGLVis();

   void WriteVisItFields(const std::string & prefix,
                         const std::string & label);

private:
   void updateRho();

   ParMesh         * pmesh_;
   L2_ParFESpace   * L2FESpace_;
   ParGridFunction * rho_;
   ParGridFunction * divGradRho_;
   ParLinearForm   * cellVol_;

   // LinearCoefficient   rhoCoef_;
   Coefficient * rhoCoef_;
   ConstantCoefficient one_;

   double refDensity_;
   double vol_;
   double tol_;

   VisData      * vd_;
   socketstream * sock_;
   socketstream * sock2_;
};

class StiffnessTensor : public Homogenization
{
public:
   /*
   StiffnessTensor(ParMesh & pmesh, double vol,
                    double lambda0, double mu0,
                    double lambda1, double mu1);
   */
   StiffnessTensor(ParMesh & pmesh, double vol,
                   Coefficient &lambdaCoef, Coefficient &muCoef,
		   double tol = 0.05);
   ~StiffnessTensor();

   // void SetVolumeFraction(ParGridFunction & vf);

   void GetHomogenizedProperties(std::vector<double> & p);
   void GetPropertySensitivities(std::vector<ParGridFunction> & dp) {}

   void InitializeGLVis(VisData & vd);

   void DisplayToGLVis();

   void WriteVisItFields(const std::string & prefix,
                         const std::string & label);

private:

   void TensorGradient(const Vector & x, Vector & y);
   void TensorGradientTranspose(const Vector & x, Vector & y);
   void TensorMassMatrix(const Vector & x, Vector & y);
   void RestrictedTensorMassMatrix(int r, const Vector & x, Vector & y);
   void RestrictedVectorAdd(int r, const Vector & x, Vector & y);

   // Produces a 2-tensor coefficient for the diagonal portion of an
   // elasticity operator.  The coefficient is defined by:
   //     / lambda + 2 * mu, i == j == axis
   // K = |              mu, i == j != axis
   //     \               0, i != j
   class DiagElasticityCoef : public MatrixCoefficient
   {
   public:
      DiagElasticityCoef(Coefficient & lambda,
                         Coefficient & mu,
                         int axis);

      virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                        const IntegrationPoint &ip);

   private:
      int axis_;
      Coefficient * lambda_;
      Coefficient * mu_;
   };

   // Produces a 2-tensor coefficient for the off-diagonal portion of an
   // elasticity operator.  The coefficient is defined by:
   //     / lambda, i == axis0, j == axis1
   // K = |     mu, i == axis1, j == axis0
   //     \      0, otherwise
   class OffDiagElasticityCoef : public MatrixCoefficient
   {
   public:
      OffDiagElasticityCoef(Coefficient & lambda,
                            Coefficient & mu,
                            int axis0, int axis1);

      virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                        const IntegrationPoint &ip);

   private:
      int axis0_;
      int axis1_;
      Coefficient * lambda_;
      Coefficient * mu_;
   };

   void solve(const Vector & E, Vector & Chi);

   int dim_;
   int irOrder_;
   int geom_;
   bool amg_elast_;

   ParMesh * pmesh_;

   L2_ParFESpace * L2FESpace_;
   L2_ParFESpace * L2VFESpace_;
   H1_ParFESpace * H1FESpace_;
   H1_ParFESpace * H1VFESpace_;
   ND_ParFESpace * HCurlFESpace_;
   ND_ParFESpace * HCurlVFESpace_;
   RT_ParFESpace * HDivFESpace_;

   // LinearCoefficient lambdaCoef_;
   // LinearCoefficient muCoef_;
   Coefficient * lambdaCoef_;
   Coefficient * muCoef_;

   DiagElasticityCoef    xxCoef_;
   DiagElasticityCoef    yyCoef_;
   DiagElasticityCoef    zzCoef_;
   OffDiagElasticityCoef yzCoef_;
   OffDiagElasticityCoef xzCoef_;
   OffDiagElasticityCoef xyCoef_;

   ParGridFunction * lambda_;
   ParGridFunction * mu_;

   ParBilinearForm * a_;
   ParBilinearForm * m_[6];
   ParGridFunction * Chi_[6];
   ParGridFunction * E_[3];
   ParGridFunction * F_[6];
   ParLinearForm   * MF_[6];

   ParDiscreteGradOperator * grad_;

   BilinearFormIntegrator * diffInteg_[3];
   ParGridFunction * errSol_;
   ErrorEstimator  * errEst_[3];
   ParGridFunction * errors_[4];

   ParGridFunction * b_;
   // HypreParVector  * tmp1_;

   double vol_;
   double tol_;

   VisData      * vd_;
   socketstream   socks_[8];
   socketstream   err_socks_[4];

   int seqVF_;
};

} // namespace meta_material
} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_META_MATERIAL_SOLVER
