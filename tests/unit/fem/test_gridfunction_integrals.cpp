// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace gridfunction_integrals
{

double func_1D_lin(const Vector &x)
{
   return x[0] - 0.5;
}

double wgt_1D_lin(const Vector &x)
{
   return 7.0 - 3.0 * x[0];
}

double func_1D_gaussian(const Vector &x)
{
   return sqrt(50.0/M_PI) * exp(-50.0 * pow(x[0] - 0.5, 2.0));
}

class FourierCosine1D : public Coefficient
{
private:
   int n_;
   double l_;
   mutable Vector x_;

public:
   FourierCosine1D(double l) : n_(0), l_(l), x_(1) {}

   void SetMode(int n) { n_ = n; }

   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);

      return cos(2.0 * M_PI * (double)n_ * x_[0] / l_);
   }
};

TEST_CASE("1D GridFunction::ComputeIntegral",
          "[GridFunction]"
          "[GridFunction::ComputeIntegral]")
{
   int n = 10;
   int dim = 1;
   int order = 1;

   FunctionCoefficient funcCoef(func_1D_lin);
   ConstantCoefficient wgt0Coef(1.0);
   FunctionCoefficient wgt1Coef(wgt_1D_lin);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, 2.0);

      H1_FECollection h1_fec(order, dim);
      DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                              FiniteElement::VALUE);
      DG_FECollection dgi_fec(order, dim, BasisType::GaussLegendre,
                              FiniteElement::INTEGRAL);

      FiniteElementSpace h1_fespace(&mesh, &h1_fec);
      FiniteElementSpace dgv_fespace(&mesh, &dgv_fec);
      FiniteElementSpace dgi_fespace(&mesh, &dgi_fec);

      GridFunction h1_x(&h1_fespace);
      GridFunction dgv_x(&dgv_fespace);
      GridFunction dgi_x(&dgi_fespace);

      h1_x.ProjectCoefficient(funcCoef);
      dgv_x.ProjectCoefficient(funcCoef);
      dgi_x.ProjectCoefficient(funcCoef);

      // First integrate with a weight of 1
      double w0 = 1.0;

      double w0_h1 = h1_x.ComputeIntegral(wgt0Coef, 0);
      double w0_dgv = dgv_x.ComputeIntegral(wgt0Coef, 0);
      double w0_dgi = dgi_x.ComputeIntegral(wgt0Coef, 0);

      REQUIRE(w0_h1  == MFEM_Approx(w0));
      REQUIRE(w0_dgv == MFEM_Approx(w0));
      REQUIRE(w0_dgi == MFEM_Approx(w0));

      // Integrate with a linear weight function
      double w1 = 2.0;

      double w1_h1  = h1_x.ComputeIntegral(wgt1Coef, 2);
      double w1_dgv = dgv_x.ComputeIntegral(wgt1Coef, 2);
      double w1_dgi = dgi_x.ComputeIntegral(wgt1Coef, 2);

      REQUIRE(w1_h1  == MFEM_Approx(w1));
      REQUIRE(w1_dgv == MFEM_Approx(w1));
      REQUIRE(w1_dgi == MFEM_Approx(w1));
   }
}

TEST_CASE("1D GridFunction::ComputeIntegral (Fourier)",
          "[GridFunction]"
          "[GridFunction::ComputeIntegral]")
{
   int n = 20;
   int dim = 1;
   int order = 3;
   double l = 1.0;

   FunctionCoefficient funcCoef(func_1D_gaussian);
   FourierCosine1D wgtCoef(l);

   for (int type = (int)Element::SEGMENT;
        type <= (int)Element::SEGMENT; type++)
   {
      Mesh mesh = Mesh::MakeCartesian1D(n, l);

      H1_FECollection h1_fec(order, dim);
      DG_FECollection dgv_fec(order, dim, BasisType::GaussLegendre,
                              FiniteElement::VALUE);
      DG_FECollection dgi_fec(order, dim, BasisType::GaussLegendre,
                              FiniteElement::INTEGRAL);

      FiniteElementSpace h1_fespace(&mesh, &h1_fec);
      FiniteElementSpace dgv_fespace(&mesh, &dgv_fec);
      FiniteElementSpace dgi_fespace(&mesh, &dgi_fec);

      GridFunction h1_x(&h1_fespace);
      GridFunction dgv_x(&dgv_fespace);
      GridFunction dgi_x(&dgi_fespace);

      h1_x.ProjectCoefficient(funcCoef);
      dgv_x.ProjectCoefficient(funcCoef);
      dgi_x.ProjectCoefficient(funcCoef);

      LinearForm h1_lf(&h1_fespace);
      LinearForm dgv_lf(&dgv_fespace);
      LinearForm dgi_lf(&dgi_fespace);

      h1_lf.AddDomainIntegrator(new DomainLFIntegrator(wgtCoef));
      dgv_lf.AddDomainIntegrator(new DomainLFIntegrator(wgtCoef));
      dgi_lf.AddDomainIntegrator(new DomainLFIntegrator(wgtCoef));

      // First integrate with a weight of 1
      double w0 = 1.0;

      h1_lf.Assemble();
      dgi_lf.Assemble();
      dgv_lf.Assemble();

      double i1_w0_h1  = h1_lf(h1_x);
      double i1_w0_dgv = dgv_lf(dgv_x);
      double i1_w0_dgi = dgi_lf(dgi_x);

      REQUIRE(i1_w0_h1  == MFEM_Approx(w0, 1e-6));
      REQUIRE(i1_w0_dgv == MFEM_Approx(w0, 1e-6));
      REQUIRE(i1_w0_dgi == MFEM_Approx(w0, 1e-6));

      double i2_w0_h1  = h1_x.ComputeIntegral(wgtCoef, 2 * order + 1);
      double i2_w0_dgv = dgv_x.ComputeIntegral(wgtCoef, 2 * order + 1);
      double i2_w0_dgi = dgi_x.ComputeIntegral(wgtCoef, 2 * order + 1);

      REQUIRE(i2_w0_h1  == MFEM_Approx(i1_w0_h1));
      REQUIRE(i2_w0_dgv == MFEM_Approx(i1_w0_dgv));
      REQUIRE(i2_w0_dgi == MFEM_Approx(i1_w0_dgi));

      // Integrate with a weight of cos(2 pi x)
      wgtCoef.SetMode(1);

      double w1 = -exp(-M_PI * M_PI / 50.0);

      h1_lf.Assemble();
      dgi_lf.Assemble();
      dgv_lf.Assemble();

      double i1_w1_h1  = h1_lf(h1_x);
      double i1_w1_dgv = dgv_lf(dgv_x);
      double i1_w1_dgi = dgi_lf(dgi_x);

      REQUIRE(i1_w1_h1  == MFEM_Approx(w1, 1e-6));
      REQUIRE(i1_w1_dgv == MFEM_Approx(w1, 1e-6));
      REQUIRE(i1_w1_dgi == MFEM_Approx(w1, 1e-6));

      double i2_w1_h1  = h1_x.ComputeIntegral(wgtCoef, 2 * order + 1);
      double i2_w1_dgv = dgv_x.ComputeIntegral(wgtCoef, 2 * order + 1);
      double i2_w1_dgi = dgi_x.ComputeIntegral(wgtCoef, 2 * order + 1);

      REQUIRE(i2_w1_h1  == MFEM_Approx(i1_w1_h1));
      REQUIRE(i2_w1_dgv == MFEM_Approx(i1_w1_dgv));
      REQUIRE(i2_w1_dgi == MFEM_Approx(i1_w1_dgi));

      // Integrate with a weight of cos(4 pi x)
      wgtCoef.SetMode(2);

      double w2 = exp(-4.0 * M_PI * M_PI / 50.0);

      h1_lf.Assemble();
      dgi_lf.Assemble();
      dgv_lf.Assemble();

      double i1_w2_h1  = h1_lf(h1_x);
      double i1_w2_dgv = dgv_lf(dgv_x);
      double i1_w2_dgi = dgi_lf(dgi_x);

      REQUIRE(i1_w2_h1  == MFEM_Approx(w2, 1e-6));
      REQUIRE(i1_w2_dgv == MFEM_Approx(w2, 1e-6));
      REQUIRE(i1_w2_dgi == MFEM_Approx(w2, 1e-6));

      double i2_w2_h1  = h1_x.ComputeIntegral(wgtCoef, 2 * order + 1);
      double i2_w2_dgv = dgv_x.ComputeIntegral(wgtCoef, 2 * order + 1);
      double i2_w2_dgi = dgi_x.ComputeIntegral(wgtCoef, 2 * order + 1);

      REQUIRE(i2_w2_h1  == MFEM_Approx(i1_w2_h1));
      REQUIRE(i2_w2_dgv == MFEM_Approx(i1_w2_dgv));
      REQUIRE(i2_w2_dgi == MFEM_Approx(i1_w2_dgi));
   }
}

} // gridfunction_integrals
