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

// MFEM Mesh Optimizer Miniapp - Serial/Parallel Shared Code

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

double size_indicator(const Vector &x)
{
   // semi-circle
   const double xc = x(0) - 0.0, yc = x(1) - 0.5,
                zc = (x.Size() == 3) ? x(2) - 0.5 : 0.0;
   const double r = sqrt(xc*xc + yc*yc + zc*zc);
   double r1 = 0.45; double r2 = 0.55; double sf=30.0;
   double val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));

   val = fmax(0.,val);
   val = fmin(1.,val);
   return val;
}

void calc_mass_volume(const GridFunction &g, double &mass, double &vol)
{
   Mesh &mesh = *g.FESpace()->GetMesh();
   const int NE = mesh.GetNE();
   Vector g_vals;
   mass = 0.0, vol = 0.0;
   for (int e = 0; e < NE; e++)
   {
      ElementTransformation &Tr = *mesh.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(mesh.GetElementBaseGeometry(e),
                                               Tr.OrderJ());
      g.GetValues(Tr, ir, g_vals);
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         Tr.SetIntPoint(&ip);
         mass   += g_vals(j) * ip.weight * Tr.Weight();
         vol    += ip.weight * Tr.Weight();
      }
   }

#ifdef MFEM_USE_MPI
   auto gp = dynamic_cast<const ParGridFunction *>(&g);
   if (gp)
   {
      MPI_Comm comm = gp->ParFESpace()->GetComm();
      MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(MPI_IN_PLACE, &vol,  1, MPI_DOUBLE, MPI_SUM, comm);
   }
#endif
}

void ConstructSizeGF(GridFunction &size)
{
   // Indicator for small (value -> 1) or big (value -> 0) elements.
   FunctionCoefficient size_ind_coeff(size_indicator);
   size.ProjectCoefficient(size_ind_coeff);

   // Determine small/big target sizes based on the total number of
   // elements and the volume occupied by small elements.
   double volume_ind, volume;
   calc_mass_volume(size, volume_ind, volume);
   Mesh &mesh = *size.FESpace()->GetMesh();
   int NE = mesh.GetNE();
#ifdef MFEM_USE_MPI
   auto size_p = dynamic_cast<const ParGridFunction *>(&size);
   if (size_p) { NE = size_p->ParFESpace()->GetParMesh()->GetGlobalNE(); }
#endif
   NCMesh *ncmesh = mesh.ncmesh;
   // For parallel NC meshes, all tasks have all root elements.
   NE = (ncmesh) ? ncmesh->GetNumRootElements() : NE;
   const double size_ratio = (mesh.Dimension() == 2) ? 9 : 27;
   const double small_el_size = volume_ind / NE +
                                (volume - volume_ind) / (size_ratio * NE);
   const double big_el_size   = size_ratio * small_el_size;
   for (int i = 0; i < size.Size(); i++)
   {
      size(i) = size(i) * small_el_size + (1.0 - size(i)) * big_el_size;
   }
}

double material_indicator_2d(const Vector &x)
{
   double xc = x(0)-0.5, yc = x(1)-0.5;
   double th = 22.5*M_PI/180.;
   double xn =  cos(th)*xc + sin(th)*yc;
   double yn = -sin(th)*xc + cos(th)*yc;
   double th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
   double stretch = 1/cos(th2);
   xc = xn/stretch; yc = yn/stretch;
   double tfac = 20;
   double s1 = 3;
   double s2 = 3;
   double wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1);
   if (wgt > 1) { wgt = 1; }
   if (wgt < 0) { wgt = 0; }
   return wgt;
}

double discrete_ori_2d(const Vector &x)
{
   return M_PI * x(1) * (1.0 - x(1)) * cos(2 * M_PI * x(0));
}

void discrete_aspr_3d(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v.SetSize(dim);
   double l1, l2, l3;
   l1 = 1.;
   l2 = 1. + 5*x(1);
   l3 = 1. + 10*x(2);
   v[0] = l1/pow(l2*l3,0.5);
   v[1] = l2/pow(l1*l3,0.5);
   v[2] = l3/pow(l2*l1,0.5);
}

class HessianCoefficient : public TMOPMatrixCoefficient
{
private:
   int metric;

public:
   HessianCoefficient(int dim, int metric_id)
      : TMOPMatrixCoefficient(dim), metric(metric_id) { }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector pos(3);
      T.Transform(ip, pos);
      if (metric != 14 && metric != 36 && metric != 85)
      {
         const double xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         const double r = sqrt(xc*xc + yc*yc);
         double r1 = 0.15; double r2 = 0.35; double sf=30.0;
         const double eps = 0.5;

         const double tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));

         K(0, 0) = eps + 1.0 * (tan1 - tan2);
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
      }
      else if (metric == 14 || metric == 36) // Size + Alignment
      {
         const double xc = pos(0), yc = pos(1);
         double theta = M_PI * yc * (1.0 - yc) * cos(2 * M_PI * xc);
         double alpha_bar = 0.1;

         K(0, 0) =  cos(theta);
         K(1, 0) =  sin(theta);
         K(0, 1) = -sin(theta);
         K(1, 1) =  cos(theta);

         K *= alpha_bar;
      }
      else if (metric == 85) // Shape + Alignment
      {
         Vector x = pos;
         double xc = x(0)-0.5, yc = x(1)-0.5;
         double th = 22.5*M_PI/180.;
         double xn =  cos(th)*xc + sin(th)*yc;
         double yn = -sin(th)*xc + cos(th)*yc;
         xc = xn; yc=yn;

         double tfac = 20;
         double s1 = 3;
         double s2 = 2;
         double wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1)
                      - std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) - 1);
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         xc = pos(0), yc = pos(1);
         double theta = M_PI * (yc) * (1.0 - yc) * cos(2 * M_PI * xc);

         K(0, 0) =  cos(theta);
         K(1, 0) =  sin(theta);
         K(0, 1) = -sin(theta);
         K(1, 1) =  cos(theta);

         double asp_ratio_tar = 0.1 + 1*(1-wgt)*(1-wgt);

         K(0, 0) *=  1/pow(asp_ratio_tar,0.5);
         K(1, 0) *=  1/pow(asp_ratio_tar,0.5);
         K(0, 1) *=  pow(asp_ratio_tar,0.5);
         K(1, 1) *=  pow(asp_ratio_tar,0.5);
      }
   }

   virtual void EvalGrad(DenseMatrix &K, ElementTransformation &T,
                         const IntegrationPoint &ip, int comp)
   {
      Vector pos(3);
      T.Transform(ip, pos);
      K = 0.;
      if (metric != 14 && metric != 85)
      {
         const double xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         const double r = sqrt(xc*xc + yc*yc);
         double r1 = 0.15; double r2 = 0.35; double sf=30.0;

         const double tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));
         double tan1d = 0., tan2d = 0.;
         if (r > 0.001)
         {
            tan1d = (1.-tan1*tan1)*(sf)/r,
            tan2d = (1.-tan2*tan2)*(sf)/r;
         }

         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
         if (comp == 0) { K(0, 0) = tan1d*xc - tan2d*xc; }
         else if (comp == 1) { K(0, 0) = tan1d*yc - tan2d*yc; }
      }
   }
};

class HRHessianCoefficient : public TMOPMatrixCoefficient
{
private:
   int dim;
   // 0 - size target in an annular region,
   // 1 - size+aspect-ratio in an annular region,
   // 2 - size+aspect-ratio target for a rotate sine wave.
   int hr_target_type;

public:
   HRHessianCoefficient(int dim_, int hr_target_type_ = 0)
      : TMOPMatrixCoefficient(dim_), dim(dim_),
        hr_target_type(hr_target_type_) { }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector pos(3);
      T.Transform(ip, pos);
      if (hr_target_type == 0) // size only circle
      {
         double small = 0.001, big = 0.01;
         if (dim == 3) { small = 0.005, big = 0.1; }
         const double xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         double zc;
         if (dim == 3) { zc = pos(2) - 0.5; }
         double r = sqrt(xc*xc + yc*yc);
         if (dim == 3) { r = sqrt(xc*xc + yc*yc + zc*zc); }
         double r1 = 0.15; double r2 = 0.35; double sf=30.0;

         const double tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));

         double ind = (tan1 - tan2);
         if (ind > 1.0) {ind = 1.;}
         if (ind < 0.0) {ind = 0.;}
         double val = ind * small + (1.0 - ind) * big;
         K = 0.0;
         K(0, 0) = 1.0;
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
         K(0, 0) *= pow(val,0.5);
         K(1, 1) *= pow(val,0.5);
         if (dim == 3) { K(2, 2) = pow(val,0.5); }
      }
      else if (hr_target_type == 1) // circle with size and AR
      {
         const double small = 0.001, big = 0.01;
         const double xc = pos(0)-0.5, yc = pos(1)-0.5;
         const double rv = xc*xc + yc*yc;
         double r = 0;
         if (rv>0.) {r = sqrt(rv);}

         double r1 = 0.2; double r2 = 0.3; double sf=30.0;
         const double szfac = 1;
         const double asfac = 4;
         const double eps2 = szfac/asfac;
         const double eps1 = szfac;

         double tan1 = std::tanh(sf*(r-r1)+1),
                tan2 = std::tanh(sf*(r-r2)-1);
         double wgt = 0.5*(tan1-tan2);

         tan1 = std::tanh(sf*(r-r1)),
         tan2 = std::tanh(sf*(r-r2));

         double ind = (tan1 - tan2);
         if (ind > 1.0) {ind = 1.;}
         if (ind < 0.0) {ind = 0.;}
         double szval = ind * small + (1.0 - ind) * big;

         double th = std::atan2(yc,xc)*180./M_PI;
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         double maxval = eps2 + eps1*(1-wgt)*(1-wgt);
         double minval = eps1;
         double avgval = 0.5*(maxval+minval);
         double ampval = 0.5*(maxval-minval);
         double val1 = avgval + ampval*sin(2.*th*M_PI/180.+90*M_PI/180.);
         double val2 = avgval + ampval*sin(2.*th*M_PI/180.-90*M_PI/180.);

         K(0,1) = 0.0;
         K(1,0) = 0.0;
         K(0,0) = val1;
         K(1,1) = val2;

         K(0,0) *= pow(szval,0.5);
         K(1,1) *= pow(szval,0.5);
      }
      else if (hr_target_type == 2) // sharp rotated sine wave
      {
         double xc = pos(0)-0.5, yc = pos(1)-0.5;
         double th = 15.5*M_PI/180.;
         double xn =  cos(th)*xc + sin(th)*yc;
         double yn = -sin(th)*xc + cos(th)*yc;
         double th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
         double stretch = 1/cos(th2);
         xc = xn/stretch;
         yc = yn;
         double tfac = 20;
         double s1 = 3;
         double s2 = 2;
         double yl1 = -0.025;
         double yl2 =  0.025;
         double wgt = std::tanh((tfac*(yc-yl1) + s2*std::sin(s1*M_PI*xc)) + 1) -
                      std::tanh((tfac*(yc-yl2) + s2*std::sin(s1*M_PI*xc)) - 1);
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         const double eps2 = 25;
         const double eps1 = 1;
         K(1,1) = eps1/eps2 + eps1*(1-wgt)*(1-wgt);
         K(0,0) = eps1;
         K(0,1) = 0.0;
         K(1,0) = 0.0;
      }
      else { MFEM_ABORT("Unsupported option / wrong input."); }
   }

   virtual void EvalGrad(DenseMatrix &K, ElementTransformation &T,
                         const IntegrationPoint &ip, int comp)
   {
      K = 0.;
   }
};

// Additional IntegrationRules that can be used with the --quad-type option.
IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);

// Defined with respect to the icf mesh.
double weight_fun(const Vector &x)
{
   const double r = sqrt(x(0)*x(0) + x(1)*x(1) + 1e-12);
   const double den = 0.002;
   double l2 = 0.2 + 0.5*std::tanh((r-0.16)/den) - 0.5*std::tanh((r-0.17)/den)
               + 0.5*std::tanh((r-0.23)/den) - 0.5*std::tanh((r-0.24)/den);
   return l2;
}

// Used for the adaptive limiting examples.
double adapt_lim_fun(const Vector &x)
{
   const double xc = x(0) - 0.1, yc = x(1) - 0.2;
   const double r = sqrt(xc*xc + yc*yc);
   double r1 = 0.45; double r2 = 0.55; double sf=30.0;
   double val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));

   val = std::max(0.,val);
   val = std::min(1.,val);
   return val;
}

// Used for exact surface alignment
double surface_level_set(const Vector &x)
{
   const int type = 1;

   const int dim = x.Size();
   if (type == 0)
   {
      const double sine = 0.25 * std::sin(4 * M_PI * x(0));
      return (x(1) >= sine + 0.5) ? 1.0 : -1.0;
   }
   else
   {
      if (dim == 2)
      {
         const double xc = x(0) - 0.5, yc = x(1) - 0.5;
         const double r = sqrt(xc*xc + yc*yc);
         return r-0.3;
      }
      else
      {
         const double xc = x(0) - 0.5, yc = x(1) - 0.5, zc = x(2) - 0.5;
         const double r = sqrt(xc*xc + yc*yc + zc*zc);
         return r-0.3;
      }
   }
}

int material_id(int el_id, const GridFunction &g)
{
   const FiniteElementSpace *fes = g.FESpace();
   const FiniteElement *fe = fes->GetFE(el_id);
   Vector g_vals;
   const IntegrationRule &ir =
      IntRules.Get(fe->GetGeomType(), fes->GetOrder(el_id) + 2);

   double integral = 0.0;
   g.GetValues(el_id, ir, g_vals);
   ElementTransformation *Tr = fes->GetMesh()->GetElementTransformation(el_id);
   int approach = 1;
   if (approach == 0)   // integral based
   {
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);
         integral += ip.weight * g_vals(q) * Tr->Weight();
      }
      return (integral > 0.0) ? 1.0 : 0.0;
   }
   else if (approach == 1)   // minimum value based
   {
      double minval = g_vals.Min();
      return minval > 0.0 ? 1.0 : 0.0;
   }
   return 0.0;
}

void DiffuseField(GridFunction &field, int smooth_steps)
{
   // Setup the Laplacian operator
   BilinearForm *Lap = new BilinearForm(field.FESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();

   // Setup the smoothing operator
   DSmoother *S = new DSmoother(0,1.0,smooth_steps);
   S->iterative_mode = true;
   S->SetOperator(Lap->SpMat());

   Vector tmp(field.Size());
   tmp = 0.0;
   S->Mult(tmp, field);

   delete S;
   delete Lap;
}

#ifdef MFEM_USE_MPI
void DiffuseField(ParGridFunction &field, int smooth_steps)
{
   // Setup the Laplacian operator
   ParBilinearForm *Lap = new ParBilinearForm(field.ParFESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();
   HypreParMatrix *A = Lap->ParallelAssemble();

   HypreSmoother *S = new HypreSmoother(*A,0,smooth_steps);
   S->iterative_mode = true;

   Vector tmp(A->Width());
   field.SetTrueVector();
   Vector fieldtrue = field.GetTrueVector();
   tmp = 0.0;
   S->Mult(tmp, fieldtrue);

   field.SetFromTrueDofs(fieldtrue);

   delete S;
   delete A;
   delete Lap;
}
#endif
