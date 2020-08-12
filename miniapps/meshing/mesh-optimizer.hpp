//           MFEM Mesh Optimizer Miniapp - Serial/Parallel Shared Code
#include "../../mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

class HessianCoefficientAMR : public TMOPMatrixCoefficient
{
private:
   int typemod = 5;
   int dim;

public:
   HessianCoefficientAMR(int dim_, int type_)
      : TMOPMatrixCoefficient(dim_), typemod(type_), dim(dim_) { }

   virtual void SetType(int typemod_) { typemod = typemod_; }
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector pos(3);
      T.Transform(ip, pos);
      (this)->Eval(K,pos);
   }

   virtual void Eval(DenseMatrix &K)
   {
      Vector pos(3);
      for (int i=0; i<K.Size(); i++) {pos(i)=K(i,i);}
      (this)->Eval(K,pos);
   }

   virtual void Eval(DenseMatrix &K, Vector pos)
   {
      if (typemod == 0)
      {
         K(0, 0) = 1.0 + 3.0 * std::sin(M_PI*pos(0));
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
      }
      else if (typemod==1) //size only circle
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
         //K(0, 0) = eps + 1.0 * (tan1 - tan2);
         K = 0.0;
         K(0, 0) = 1.0;
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
         K(0, 0) *= pow(val,0.5);
         K(1, 1) *= pow(val,0.5);
         if (dim == 3) { K(2, 2) = pow(val,0.5); }
      }
      else if (typemod==2) // size only sine wave
      {
         const double small = 0.001, big = 0.01;
         const double X = pos(0), Y = pos(1);
         double ind = std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) + 1) -
                      std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) - 1);
         if (ind > 1.0) {ind = 1.;}
         if (ind < 0.0) {ind = 0.;}
         double val = ind * small + (1.0 - ind) * big;
         K(0, 0) = pow(val,0.5);
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = pow(val,0.5);
      }
      else if (typemod==3) //circle with size and AR
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
      else if (typemod == 4) //sharp sine wave
      {
         //const double small = 0.001, big = 0.01;
         const double xc = pos(0), yc = pos(1);

         double tfac = 40;
         double yl1 = 0.45;
         double yl2 = 0.55;
         double wgt = std::tanh((tfac*(yc-yl1) + 2*std::sin(4.0*M_PI*xc)) + 1) -
                      std::tanh((tfac*(yc-yl2) + 2*std::sin(4.0*M_PI*xc)) - 1);
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         const double eps2 = 10;
         const double eps1 = 1;
         K(1,1) = eps1/eps2 + eps1*(1-wgt)*(1-wgt);
         K(0,0) = eps1;
         K(0,1) = 0.0;
         K(1,0) = 0.0;

         //K(0,0) *= pow(szval,0.5);
         //K(1,1) *= pow(szval,0.5);
      }
      else if (typemod == 5) //sharp rotated sine wave
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
      else if (typemod == 6) //BOUNDARY LAYER REFINEMENT
      {
         const double szfac = 1;
         const double asfac = 500;
         const double eps = szfac;
         const double eps2 = szfac/asfac;
         double yscale = 1.5;
         yscale = 2 - 2/asfac;
         K(0, 0) = eps;
         K(1, 1) = eps2 + szfac*yscale*pos(1);
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
      }
   }

   virtual void EvalGrad(DenseMatrix &K, ElementTransformation &T,
                         const IntegrationPoint &ip, int comp)
   {
      Vector pos(3);
      T.Transform(ip, pos);
      K = 0.;
   }
};

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


class TMOPAMR
{
protected:
   NonlinearForm *nlf;
   Array<GridFunction *> meshnodarr;
   Mesh *mesh;
#ifdef MFEM_USE_MPI
   ParNonlinearForm *pnlf;
   Array<ParGridFunction *> pmeshnodarr;
   ParMesh *pmesh;
#endif

   bool move_bnd;


public:
   TMOPAMR(NonlinearForm &nlf_, Mesh &mesh_, bool move_bnd_) :
      nlf(&nlf_), meshnodarr(), mesh(&mesh_), move_bnd(move_bnd_) { }
#ifdef MFEM_USE_MPI
   TMOPAMR(ParNonlinearForm &pnlf_, ParMesh &pmesh_, bool move_bnd_) :
      nlf(&pnlf_), pnlf(&pnlf_), pmeshnodarr(), pmesh(&pmesh_), move_bnd(move_bnd_) { }
#endif

   void AddMeshNodeAr(GridFunction *gf_) { meshnodarr.Append(gf_); }
#ifdef MFEM_USE_MPI
   void AddMeshNodeAr(ParGridFunction *pgf_) { pmeshnodarr.Append(pgf_); }
#endif

   void Update();
#ifdef MFEM_USE_MPI
   void ParUpdate();
#endif

   void RebalanceParNCMesh();
};

void TMOPAMR::RebalanceParNCMesh()
{
   ParNCMesh *pncmesh = pmesh->pncmesh;
   if (pncmesh)
   {
      const Table &dreftable = pncmesh->GetDerefinementTable();
      Array<int> drefs, new_ranks;
      for (int i = 0; i < dreftable.Size(); i++)
      {
         drefs.Append(i);
      }
      pncmesh->GetFineToCoarsePartitioning(drefs, new_ranks);
      new_ranks.SetSize(pmesh->GetNE());
      pmesh->Rebalance(new_ranks);
   }
}


void TMOPAMR::Update()
{
   // Update nodal GF
   for (int i = 0; i < meshnodarr.Size(); i++)
   {
      meshnodarr[i]->Update();
      meshnodarr[i]->SetTrueVector();
      meshnodarr[i]->SetFromTrueVector();
   }

   const FiniteElementSpace *fespace = mesh->GetNodalFESpace();

   // Update Discrete Indicator for all the TMOP_Integrators in NonLinearForm
   Array<NonlinearFormIntegrator*> &integs = *(nlf->GetDNFI());
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   DiscreteAdaptTC *dtc = NULL;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         dtc = ti->GetDiscreteAdaptTC();
         if (dtc) { dtc->Update(); }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            dtc = ati[j]->GetDiscreteAdaptTC();
            if (dtc) { dtc->Update(); }
         }
      }
   }
   //if (tmopi->GetDiscreteAdaptTC()) { tmopi->GetDiscreteAdaptTC()->Update(); }

   // Update Nonlinear form and Set Essential BC
   nlf->Update();
   int dim = fespace->GetFE(0)->GetDim();
   if (move_bnd == false)
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      nlf->SetEssentialBC(ess_bdr);
   }
   else
   {
      const int nd  = fespace->GetBE(0)->GetDof();
      int n = 0;
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int attr = mesh->GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int attr = mesh->GetBdrElement(i)->GetAttribute();
         fespace->GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr == 4) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      nlf->SetEssentialVDofs(ess_vdofs);
   }
};

#ifdef MFEM_USE_MPI
void TMOPAMR::ParUpdate()
{
   // Update nodal GF
   for (int i = 0; i < pmeshnodarr.Size(); i++)
   {
      pmeshnodarr[i]->Update();
      pmeshnodarr[i]->SetTrueVector();
      pmeshnodarr[i]->SetFromTrueVector();
   }

   // Update Discrete Indicator
   Array<NonlinearFormIntegrator*> &integs = *(nlf->GetDNFI());
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   DiscreteAdaptTC *dtc = NULL;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         dtc = ti->GetDiscreteAdaptTC();
         if (dtc) { dtc->ParUpdate(); }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            dtc = ati[j]->GetDiscreteAdaptTC();
            if (dtc) { dtc->ParUpdate(); }
         }
      }
   }
   //if (tmopi->GetDiscreteAdaptTC()) { tmopi->GetDiscreteAdaptTC()->ParUpdate(); }

   const FiniteElementSpace *pfespace = pmesh->GetNodalFESpace();

   // Update Nonlinear form and Set Essential BC
   pnlf->Update();
   int dim = pfespace->GetFE(0)->GetDim();
   if (move_bnd == false)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      pnlf->SetEssentialBC(ess_bdr);
   }
   else
   {
      const int nd  = pfespace->GetBE(0)->GetDof();
      int n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < pmesh->GetNBE(); i++)
      {
         const int attr = pmesh->GetBdrElement(i)->GetAttribute();
         pfespace->GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr == 4) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      pnlf->SetEssentialVDofs(ess_vdofs);
   }
};
#endif

class TMOPRefinerEstimator : public AnisotropicErrorEstimator
{
protected:
   Mesh *mesh;
   TMOP_Integrator *tmopi;
   int order;
   int amrmetric;
   Array<IntegrationRule *> TriIntRule;
   Array<IntegrationRule *> QuadIntRule;
   long current_sequence;
   Vector error_estimates;
   Array<int> aniso_flags;

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = mesh->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();


   // Construct the integration rules for each element type - only 2D right now
   void SetQuadIntRules();
   void SetTriIntRules();
   void GetTMOPRefinementEnergy(int reftype, Vector &el_energy_vec);

   IntegrationRule* SetIntRulesFromMesh(Mesh &meshsplit);
public:
   TMOPRefinerEstimator(Mesh &mesh_,
                        TMOP_Integrator &tmopi_, int order_, int amrmetric_) :
      mesh(&mesh_), tmopi(&tmopi_), order(order_), amrmetric(amrmetric_),
      TriIntRule(0), QuadIntRule(0),
      current_sequence(-1), error_estimates(), aniso_flags()
   {
      MFEM_VERIFY(mesh->Dimension()==2," 3D not implemented yet.");
      SetQuadIntRules();
      SetTriIntRules();
   }
   // destructor
   ~TMOPRefinerEstimator()
   {
      for (int i = 0; i < QuadIntRule.Size(); i++)
      {
         delete QuadIntRule[i];
      }
      for (int i = 0; i < TriIntRule.Size(); i++)
      {
         delete TriIntRule[i];
      }
   }

   virtual const Vector &GetLocalErrors()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }
   virtual const Array<int> &GetAnisotropicFlags()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return aniso_flags;
   }

   /// Reset the error estimator.
   virtual void Reset() { current_sequence = -1; }
};

void TMOPRefinerEstimator::ComputeEstimates()
{
   bool iso = false;
   bool aniso = false;
   if (amrmetric == 1 || amrmetric == 2 || amrmetric == 58
       || amrmetric == 301 || amrmetric == 302 || amrmetric == 303)
   {
      aniso = true;
   }
   if (amrmetric == 55 || amrmetric == 56 || amrmetric == 77
       || amrmetric == 315 || amrmetric == 316)
   {
      iso = true;
   }
   if (amrmetric == 7 || amrmetric == 9 || amrmetric == 321)
   {
      iso = true; aniso = true;
   }

   if (iso == false && aniso == false)
   {
      MFEM_ABORT("Metric type not supported in hr-adaptivity.");
   }

   const int dim = mesh->Dimension();
   const int num_ref_types = 3 + 4*(dim-2);
   const int NEorig = mesh->GetNE();

   aniso_flags.SetSize(NEorig);
   error_estimates.SetSize(NEorig);
   Vector amr_base_energy(NEorig), amr_temp_energy(NEorig);
   error_estimates = 1.*std::numeric_limits<float>::max();
   aniso_flags = -1;
   GetTMOPRefinementEnergy(0, amr_base_energy);

   for (int i = 1; i < num_ref_types+1; i++)
   {
      if ( dim == 2 && i < 3 && aniso != true ) { continue; }
      if ( dim == 2 && i == 3 && iso != true ) { continue; }
      if ( dim == 3 && i < 7 && aniso != true ) { continue; }
      if ( dim == 3 && i == 7 && iso != true ) { continue; }

      GetTMOPRefinementEnergy(i, amr_temp_energy);

      for (int e = 0; e < NEorig; e++)
      {
         if ( amr_temp_energy(e) < error_estimates(e) )
         {
            error_estimates(e) = amr_temp_energy(e);
            aniso_flags[e] = i;
         }
      }
   }

   error_estimates -= amr_base_energy;
   error_estimates *= -1;
   current_sequence = mesh->GetSequence();
}

void TMOPRefinerEstimator::GetTMOPRefinementEnergy(int reftype,
                                                   Vector &el_energy_vec)
{
   const FiniteElementSpace *fes = mesh->GetNodalFESpace();
   const int NE = fes->GetNE();
   GridFunction *xdof = mesh->GetNodes();
   xdof->SetTrueVector();
   xdof->SetFromTrueVector();

   el_energy_vec.SetSize(NE);

   for (int e = 0; e < NE; e++)
   {
      Geometry::Type gtype = fes->GetFE(e)->GetGeomType();
      DenseMatrix tr, vals;
      int NEsplit = 1;
      IntegrationRule *irule = NULL;
      switch (gtype)
      {
         case Geometry::TRIANGLE:
         {
            if (reftype != 0) { NEsplit = 4; }
            int ref_access = reftype == 0 ? 0 : 1;
            xdof->GetVectorValues(e, *TriIntRule[ref_access], vals, tr);
            irule = TriIntRule[ref_access];
            break;
         }
         case Geometry::SQUARE:
         {
            MFEM_VERIFY(QuadIntRule[reftype], " Integration rule does not exist.");
            xdof->GetVectorValues(e, *QuadIntRule[reftype], vals, tr);
            if (reftype == 0) { NEsplit = 1; }
            else if (reftype == 1 || reftype == 2) { NEsplit = 2; }
            else { NEsplit = 4; }
            irule = QuadIntRule[reftype];
            break;
         }
         default:
            MFEM_ABORT("Incompatible geometry type!");
      }
      vals.Transpose();

      // The data format is xe1,xe2,..xen,ye1,ye2..yen.
      // We will reformat it inside GetAMRElementENergy
      Vector elfun(vals.GetData(), vals.NumCols()*vals.NumRows());

      el_energy_vec(e) = tmopi->GetAMRElementEnergy(*fes->GetFE(e),
                                                    *mesh->GetElementTransformation(e),
                                                    elfun,
                                                    *irule);
   }
}

void TMOPRefinerEstimator::SetQuadIntRules()
{
   QuadIntRule.SetSize(3+1);

   // Reftype = 0 // original element
   Mesh *meshsplit = NULL;
   int Nvert = 9;
   int NEsplit = 1;
   meshsplit = new Mesh(2, Nvert, NEsplit, 0, 2);

   double quad_v[9][2] =
   {
      {0, 0}, {1, 0}, {0, 1}, {1, 1},
      {0., 0.5}, {1.0, 0.5}, {0.5, 0.}, {0.5, 1.0},
      {0.5, 0.5}
   };
   int quad_e0[1][4] =
   {
      {0, 1, 3, 2}
   };

   for (int j = 0; j < Nvert; j++)
   {
      meshsplit->AddVertex(quad_v[j]);
   }
   for (int j = 0; j < NEsplit; j++)
   {
      int attribute = j + 1;
      meshsplit->AddQuad(quad_e0[j], attribute);
   }
   meshsplit->FinalizeQuadMesh(1, 1, true);

   QuadIntRule[0] = SetIntRulesFromMesh(*meshsplit);
   delete meshsplit;

   // Reftype = 1
   NEsplit = 2;
   meshsplit = new Mesh(2, Nvert, NEsplit, 0, 2);

   int quad_e1[2][4] =
   {
      {0, 6, 7, 2}, {6, 1, 3, 7}
   };

   for (int j = 0; j < Nvert; j++)
   {
      meshsplit->AddVertex(quad_v[j]);
   }
   for (int j = 0; j < NEsplit; j++)
   {
      int attribute = j + 1;
      meshsplit->AddQuad(quad_e1[j], attribute);
   }
   meshsplit->FinalizeQuadMesh(1, 1, true);

   QuadIntRule[1] = SetIntRulesFromMesh(*meshsplit);
   delete meshsplit;

   // Reftype = 2
   meshsplit = new Mesh(2, Nvert, NEsplit, 0, 2);
   int quad_e2[2][4] =
   {
      {0, 1, 5, 4}, {4, 5, 3, 2}
   };

   for (int j = 0; j < Nvert; j++)
   {
      meshsplit->AddVertex(quad_v[j]);
   }
   for (int j = 0; j < NEsplit; j++)
   {
      int attribute = j + 1;
      meshsplit->AddQuad(quad_e2[j], attribute);
   }
   meshsplit->FinalizeQuadMesh(1, 1, true);

   QuadIntRule[2] = SetIntRulesFromMesh(*meshsplit);
   delete meshsplit;

   // Reftype = 3
   NEsplit = 4;
   meshsplit = new Mesh(2, Nvert, NEsplit, 0, 2);
   int quad_e3[4][4] =
   {
      {0, 6, 8, 4}, {6, 1, 5, 8}, {8, 5, 3, 7}, {4, 8, 7, 2}
   };

   for (int j = 0; j < Nvert; j++)
   {
      meshsplit->AddVertex(quad_v[j]);
   }
   for (int j = 0; j < NEsplit; j++)
   {
      int attribute = j + 1;
      meshsplit->AddQuad(quad_e3[j], attribute);
   }
   meshsplit->FinalizeQuadMesh(1, 1, true);

   QuadIntRule[3] = SetIntRulesFromMesh(*meshsplit);
   delete meshsplit;
}

void TMOPRefinerEstimator::SetTriIntRules()
{
   TriIntRule.SetSize(1+1);

   // Reftype = 0
   Mesh *meshsplit = NULL;
   int Nvert = 6;
   int NEsplit = 1;
   meshsplit = new Mesh(2, Nvert, NEsplit, 0, 2);

   double tri_v[6][2] =
   {
      {0, 0}, {1, 0}, {0, 1}, {0.5, 0}, {0.5, 0.5}, {0., 0.5}
   };
   int tri_e0[4][3] =
   {
      {0, 1, 2}
   };

   for (int j = 0; j < Nvert; j++)
   {
      meshsplit->AddVertex(tri_v[j]);
   }
   for (int j = 0; j < NEsplit; j++)
   {
      int attribute = j + 1;
      meshsplit->AddTri(tri_e0[j], attribute);
   }
   meshsplit->FinalizeTriMesh(1, 1, true);

   TriIntRule[0] = SetIntRulesFromMesh(*meshsplit);
   delete meshsplit;

   // Reftype = 1
   NEsplit = 4;
   meshsplit = new Mesh(2, Nvert, NEsplit, 0, 2);

   int tri_e1[4][3] =
   {
      {0, 3, 5}, {5, 3, 4}, {3, 1, 4}, {5, 4, 2}
   };

   for (int j = 0; j < Nvert; j++)
   {
      meshsplit->AddVertex(tri_v[j]);
   }
   for (int j = 0; j < NEsplit; j++)
   {
      int attribute = j + 1;
      meshsplit->AddTri(tri_e1[j], attribute);
   }
   meshsplit->FinalizeTriMesh(1, 1, true);

   TriIntRule[1] = SetIntRulesFromMesh(*meshsplit);
   delete meshsplit;
}

IntegrationRule* TMOPRefinerEstimator::SetIntRulesFromMesh(Mesh &meshsplit)
{
   const int dim = meshsplit.Dimension();
   H1_FECollection fec(order, dim);
   FiniteElementSpace nodal_fes(&meshsplit, &fec, dim);
   meshsplit.SetNodalFESpace(&nodal_fes);

   const int NEsplit = meshsplit.GetNE();
   const int dof_cnt = nodal_fes.GetFE(0)->GetDof(),
             pts_cnt = NEsplit * dof_cnt;

   DenseMatrix pos(dof_cnt, dim);
   Vector posV(pos.Data(), dof_cnt * dim);
   Array<int> xdofs(dof_cnt * dim);

   // Create an IntegrationRule on the nodes of the reference submesh.
   IntegrationRule *irule = new IntegrationRule(pts_cnt);
   GridFunction *nodesplit = meshsplit.GetNodes();

   int pt_id = 0;
   for (int i = 0; i < NEsplit; i++)
   {
      nodal_fes.GetElementVDofs(i, xdofs);
      nodesplit->GetSubVector(xdofs, posV);
      for (int j = 0; j < dof_cnt; j++)
      {
         if (dim == 2)
         {
            irule->IntPoint(pt_id).Set2(pos(j, 0), pos(j, 1));
         }
         else if (dim == 3)
         {
            irule->IntPoint(pt_id).Set3(pos(j, 0), pos(j, 1), pos(j, 2));
         }
         pt_id++;
      }
   }
   return irule;
}

// TMOPRefiner is ThresholdRefiner with total_error_fraction = 0.;
class TMOPRefiner : public ThresholdRefiner
{
public:
   /// Construct a ThresholdRefiner using the given ErrorEstimator.
   TMOPRefiner(TMOPRefinerEstimator &est) : ThresholdRefiner(est)
   {
      SetTotalErrorFraction(0.);
   }
};

class TMOPDeRefinerEstimator : public ErrorEstimator
{
protected:
   Mesh *mesh;
#ifdef MFEM_USE_MPI
   ParMesh *pmesh;
#endif
   TMOP_Integrator *tmopi;
   int order;
   int amrmetric;
   long current_sequence;
   Vector error_estimates;
   bool serial;

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = mesh->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();

   void GetTMOPDerefinementEnergy(Mesh &cmesh,
                                  Vector &el_energy_vec,
                                  FiniteElementSpace *tcdfes = NULL);
public:
   TMOPDeRefinerEstimator(Mesh &mesh_, TMOP_Integrator &tmopi_) :
      mesh(&mesh_), tmopi(&tmopi_),
      current_sequence(-1), error_estimates(), serial(true)
   {
      MFEM_VERIFY(mesh->Dimension()==2," 3D not implemented yet.");
   }
#ifdef MFEM_USE_MPI
   TMOPDeRefinerEstimator(ParMesh &pmesh_, TMOP_Integrator &tmopi_) :
      mesh(&pmesh_), pmesh(&pmesh_), tmopi(&tmopi_),
      current_sequence(-1), error_estimates(), serial(false)
   {
      MFEM_VERIFY(pmesh->Dimension()==2," 3D not implemented yet.");
   }
#endif


   // destructor
   ~TMOPDeRefinerEstimator()
   {
   }

   virtual const Vector &GetLocalErrors()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /// Reset the error estimator.
   virtual void Reset() { current_sequence = -1; }
};

void TMOPDeRefinerEstimator::ComputeEstimates()
{
   DiscreteAdaptTC *tcd = tmopi->GetDiscreteAdaptTC();
   const Operator *c_op = NULL;

   if (serial)
   {
      Mesh meshcopy(*mesh);

      FiniteElementSpace *tcdfes = NULL;
      if (tcd)
      {
         tcdfes = new FiniteElementSpace(*tcd->GetTSpecFESpace(), &meshcopy);
      }

      Vector local_err(meshcopy.GetNE());
      local_err = 0.;
      double threshold = std::numeric_limits<float>::max();
      meshcopy.DerefineByError(local_err, threshold, 0, 1);

      if (meshcopy.GetGlobalNE() == mesh->GetGlobalNE())
      {
         error_estimates = 1;
         delete tcdfes;
         return;
      }

      if (tcd)
      {
         tcdfes->Update();
         c_op = tcdfes->GetUpdateOperator();
         Vector tcd_data = *(tcd->GetTSpecVec());
         Vector amr_tspec_vals(c_op->Height());
         c_op->Mult(tcd_data, amr_tspec_vals);
         tcd->SetTspecFromVec(amr_tspec_vals);
      }

      Vector coarse_el_energy(meshcopy.GetNE());
      GetTMOPDerefinementEnergy(meshcopy, coarse_el_energy, tcdfes);
      if (tcd) { tcd->ResetAMRTspecVec(); }
      GetTMOPDerefinementEnergy(*mesh, error_estimates);

      const CoarseFineTransformations &dtrans =
         meshcopy.ncmesh->GetDerefinementTransforms();
      Table coarse_to_fine;
      dtrans.GetCoarseToFineMapFast(meshcopy, coarse_to_fine);

      for (int pe = 0; pe < coarse_to_fine.Size(); pe++)
      {
         Array<int> tabrow;
         coarse_to_fine.GetRow(pe, tabrow);
         int nchild = tabrow.Size();
         double parent_energy = coarse_el_energy(pe);
         for (int fe = 0; fe < nchild; fe++)
         {
            int child = tabrow[fe];
            MFEM_VERIFY(child < mesh->GetNE(), " invalid coarse to fine mapping");
            error_estimates(child) -= parent_energy/nchild;
         }
      }
      delete tcdfes;
   }
   else
   {
#ifdef MFEM_USE_MPI
      ParMesh meshcopy(*pmesh);
      ParFiniteElementSpace *tcdfes = NULL;
      if (tcd)
      {
         tcdfes = new ParFiniteElementSpace(*tcd->GetTSpecParFESpace(), meshcopy);
      }

      Vector local_err(meshcopy.GetNE());
      local_err = 0.;
      double threshold = std::numeric_limits<float>::max();
      meshcopy.DerefineByError(local_err, threshold, 0, 1);

      if (meshcopy.GetGlobalNE() == pmesh->GetGlobalNE())
      {
         error_estimates = 1;
         delete tcdfes;
         return;
      }

      if (tcd)
      {
         tcdfes->Update();
         c_op = tcdfes->GetUpdateOperator();
         Vector tcd_data = *(tcd->GetTSpecVec());
         Vector amr_tspec_vals(c_op->Height());
         c_op->Mult(tcd_data, amr_tspec_vals);
         tcd->SetTspecFromVec(amr_tspec_vals);
      }

      Vector coarse_el_energy(meshcopy.GetNE());
      GetTMOPDerefinementEnergy(meshcopy, coarse_el_energy, tcdfes);
      if (tcd) { tcd->ResetAMRTspecVec(); }
      MPI_Barrier(MPI_COMM_WORLD);

      GetTMOPDerefinementEnergy(*pmesh, error_estimates);
      MPI_Barrier(MPI_COMM_WORLD);

      const CoarseFineTransformations &dtrans =
         meshcopy.pncmesh->GetDerefinementTransforms();
      Table coarse_to_fine;
      MPI_Barrier(MPI_COMM_WORLD);
      dtrans.GetCoarseToFineMapFast(meshcopy, coarse_to_fine,
                                    pmesh->pncmesh->GetNGhostElements());
      MPI_Barrier(MPI_COMM_WORLD);

      for (int pe = 0; pe < meshcopy.GetNE(); pe++)
      {
         Array<int> tabrow;
         coarse_to_fine.GetRow(pe, tabrow);
         int nchild = tabrow.Size();
         double parent_energy = coarse_el_energy(pe);
         for (int fe = 0; fe < nchild; fe++)
         {
            int child = tabrow[fe];
            MFEM_VERIFY(child < pmesh->GetNE(), " invalid coarse to fine mapping");
            error_estimates(child) -= parent_energy/nchild;
         }
      }
      delete tcdfes;
#endif
   }
   error_estimates *= -1.; // error_estimate(e) = energy(parent_of_e)-energy(e)
   // -ve energy means derefinement is desirable
}

void TMOPDeRefinerEstimator::GetTMOPDerefinementEnergy(Mesh &cmesh,
                                                       Vector &el_energy_vec,
                                                       FiniteElementSpace *tcdfes)
{
   const int cNE = cmesh.GetNE();
   el_energy_vec.SetSize(cNE);
   const FiniteElementSpace *fespace = cmesh.GetNodalFESpace();

   GridFunction *cxdof = cmesh.GetNodes();

   Array<int> vdofs;
   Vector el_x;
   const FiniteElement *fe;
   ElementTransformation *T;

   for (int j = 0; j < cNE; j++)
   {
      fe = fespace->GetFE(j);
      fespace->GetElementVDofs(j, vdofs);
      T = cmesh.GetElementTransformation(j);
      cxdof->GetSubVector(vdofs, el_x);
      el_energy_vec(j) = tmopi->GetDeRefinementElementEnergy(*fe, *T, el_x, tcdfes);
   }
}
