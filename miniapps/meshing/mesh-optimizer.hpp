//           MFEM Mesh Optimizer Miniapp - Serial/Parallel Shared Code

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

double discrete_size_2d(const Vector &x)
{
   int opt = 2;
   const double small = 0.001, big = 0.01;
   double val = 0.;

   if (opt == 1) // sine wave.
   {
      const double X = x(0), Y = x(1);
      val = std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) + 1) -
            std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) - 1);
   }
   else if (opt == 2) // semi-circle
   {
      const double xc = x(0) - 0.0, yc = x(1) - 0.5;
      const double r = sqrt(xc*xc + yc*yc);
      double r1 = 0.45; double r2 = 0.55; double sf=30.0;
      val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
   }

   val = std::max(0.,val);
   val = std::min(1.,val);

   return val * small + (1.0 - val) * big;
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

double discrete_aspr_2d(const Vector &x)
{
   double xc = x(0)-0.5, yc = x(1)-0.5;
   double th = 22.5*M_PI/180.;
   double xn =  cos(th)*xc + sin(th)*yc;
   double yn = -sin(th)*xc + cos(th)*yc;
   //double th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
   //double stretch = 1/cos(th2);
   xc = xn; yc = yn;

   double tfac = 20;
   double s1 = 3;
   double s2 = 2;
   double wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1)
                - std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) - 1);
   if (wgt > 1) { wgt = 1; }
   if (wgt < 0) { wgt = 0; }
   return 0.1 + 1*(1-wgt)*(1-wgt);
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
      if (metric != 14 && metric != 85)
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
      else if (metric == 14) // Size + Alignment
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

void DiffuseField(GridFunction &field, int smooth_steps)
{
   //Setup the Laplacian operator
   BilinearForm *Lap = new BilinearForm(field.FESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();

   //Setup the smoothing operator
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
   //Setup the Laplacian operator
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
   delete Lap;
}
#endif
