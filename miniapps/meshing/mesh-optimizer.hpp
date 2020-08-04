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

class TMOPEstimator : public ErrorEstimator
{
protected:
   long current_sequence;
   double total_error;
   const int dim, amrmetric;
   Array<int> aniso_flags;
   Vector amr_ref_check;

   FiniteElementSpace *fespace;
   TMOP_Integrator *tmopi;
   GridFunction *dofgf;

   Vector SizeErr, AspErr;
   Array <Vector *> amr_refenergy;
   Vector amrreftypeenergy;

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = dofgf->FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();
   void ComputeRefEstimates(); //Refinement estimate
   void ComputeDeRefEstimates(); //Derefinement estimate

public:
   TMOPEstimator(FiniteElementSpace &fes_,
                 TMOP_Integrator &tmopi_,
                 GridFunction &x_,
                 int amrmetric_)
      : current_sequence(-1), total_error(0.),
        dim(fes_.GetFE(0)->GetDim()), amrmetric(amrmetric_),
        fespace(&fes_), tmopi(&tmopi_), dofgf(&x_)
   {
      //amr_refenergy.SetSize(7);
   }
   /// Return the total error from the last error estimate.
   double GetTotalError() const { return total_error; }

   virtual const Vector &GetLocalErrors() { return SizeErr; }

   virtual const Vector &GetAMREnergy(int type)
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return *amr_refenergy[type];
   }

   void SetEnergyVecPtr(Array<Vector *> vecin_) {
       amr_refenergy.DeleteAll();
       for (int i= 0; i < vecin_.Size(); i++) {
           amr_refenergy.Append(vecin_[i]);
       }
   }

   void GetAMRTypeEnergy(Vector &energyvec_);

   virtual void Reset() { current_sequence = -1; }

   virtual ~TMOPEstimator() {}
};

void TMOPEstimator::ComputeEstimates()
{

}

void TMOPEstimator::GetAMRTypeEnergy(Vector &energyvec_)
{
   MFEM_VERIFY(dim>1, " Use 2D or 3D mesh for hr-adaptivity");
   const int NE            = fespace->GetNE();
   dofgf->SetFromTrueVector();
   Vector tvecdofs = dofgf->GetTrueVector();

   const Operator *P = fespace->GetProlongationMatrix();
   Vector x_loc;
   if (P)
   {
      x_loc.SetSize(P->Height());
      P->Mult(tvecdofs,x_loc);
   }
   else
   {
      x_loc = tvecdofs;
   }

   amrreftypeenergy.SetSize(NE);

   Array<int> vdofs;
   Vector el_x;
   const FiniteElement *fe;
   ElementTransformation *T;

   for (int j = 0; j < NE; j++)
   {
       fe = fespace->GetFE(j);
       fespace->GetElementVDofs(j, vdofs);
       T = fespace->GetElementTransformation(j);
       x_loc.GetSubVector(vdofs, el_x);
       amrreftypeenergy(j) = tmopi->GetAMRElementEnergy(*fe, *T, el_x);
   }

   energyvec_ = amrreftypeenergy;
}

class TMOPTypeRefiner : public MeshOperator
{
protected:
   TMOPEstimator &estimator;

   long max_elements;
   long num_marked_elements;

   Array<Refinement> marked_elements;
   long current_sequence;

   int non_conforming;
   int nc_limit;
   int dim;
   int reftype;

   Vector parentenergy;
   Array<int> parentreftype;

   double GetNorm(const Vector &local_err, Mesh &mesh) const;

   /** @brief Apply the operator to theG mesh->
       @return STOP if a stopping criterion is satisfied or no elements were
       marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// Construct a ThresholdRefiner using the given ErrorEstimator.
   TMOPTypeRefiner(TMOPEstimator &est, int dim_);
   // default destructor (virtual)

   void SetRefType(int type_) { reftype = type_; }
   void SetParentEnergy(Vector &parentenergy_) { parentenergy = parentenergy_; }
   void SetParentEnergyRefType(Array<int> &parentenergyreftype_) {
                                   parentreftype = parentenergyreftype_; }

   void DetermineAMRTypeEnergy(Mesh &mesh, Vector &energyvecin_,
                               Vector &energyvecout_);

#ifdef MFEM_USE_MPI
   void DetermineAMRTypeEnergy(ParMesh &pmesh, Vector &energyvecin_,
                               Vector &energyvecout_);
#endif

   /// Use nonconforming refinement, if possible (triangles, quads, hexes).
   void PreferNonconformingRefinement() { non_conforming = 1; }

   /** @brief Use conforming refinement, if possible (triangles, tetrahedra)
       -- this is the default. */
   void PreferConformingRefinement() { non_conforming = -1; }

   /** @brief Set the maximum ratio of refinement levels of adjacent elements
       (0 = unlimited). */
   void SetNCLimit(int nc_limit)
   {
      MFEM_ASSERT(nc_limit >= 0, "Invalid NC limit");
      this->nc_limit = nc_limit;
   }

   /// Get the number of marked elements in the last Apply() call.
   long GetNumMarkedElements() const { return num_marked_elements; }

   /// Reset the associated estimator.
   virtual void Reset();
};

TMOPTypeRefiner::TMOPTypeRefiner(TMOPEstimator &est, int dim_)
   : estimator(est), dim(dim_)
{
   max_elements = std::numeric_limits<long>::max();

   num_marked_elements = 0L;
   current_sequence = -1;

   non_conforming = 1;
   nc_limit = 0;
}

int TMOPTypeRefiner::ApplyImpl(Mesh &mesh)
{
   num_marked_elements = 0;
   marked_elements.SetSize(0);
   current_sequence = mesh.GetSequence();

   const long num_elements = mesh.GetGlobalNE();
   if (num_elements >= max_elements) { return STOP; }

   const int NE = mesh.GetNE();

   if (reftype == 0) {
       for (int el = 0; el < NE; el++)
       {
          if (parentenergy(el) > 0 && parentreftype[el] > 0) {
              marked_elements.Append(Refinement(el, parentreftype[el]));
          }
       }
   }
   else {
       for (int el = 0; el < NE; el++)
       {
          marked_elements.Append(Refinement(el, reftype));
       }
   }

   num_marked_elements = mesh.ReduceInt(marked_elements.Size());
   if (num_marked_elements == 0) { return STOP; }
   mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
   return CONTINUE + REFINED;
}

void TMOPTypeRefiner::DetermineAMRTypeEnergy(Mesh &mesh,
                                             Vector &energyvecin_,
                                             Vector &energyvecout_)
{
    NCMesh *ncmesh = mesh.ncmesh;
    if (!ncmesh) { return; }

    const Table &dreftable = ncmesh->GetDerefinementTable();
    const int NE = mesh.GetNE();
    Array<int> tabrow;

    Table coarse_to_fine;
    const CoarseFineTransformations &rtrans= mesh.GetRefinementTransforms();
    rtrans.GetCoarseToFineMapFast(mesh, coarse_to_fine);

    Vector fine_to_coarse(mesh.GetNE());
    for (int i = 0; i < coarse_to_fine.Size(); i++) {
        coarse_to_fine.GetRow(i, tabrow);
        for (int j = 0; j < tabrow.Size(); j++) {
            fine_to_coarse(tabrow[j]) = i;
        }
    }

    int NEredfac = 1;
    if (reftype & 1) { NEredfac *= 2; }
    if (reftype & 2) { NEredfac *= 2; }
    if (reftype & 4) { NEredfac *= 2; }

    int NEmacro = NE/NEredfac;
    MFEM_VERIFY(dreftable.Size()==NE/NEredfac, " Not all elements can be derefined\n;")
    energyvecout_ = 0.0;
    Vector centerdum(dim), center(dim);
    for (int i = 0; i < NEmacro; i++)
    {
        center = 0.;
        dreftable.GetRow(i, tabrow);
        for (int j = 0; j < tabrow.Size(); j++) {
            int parent = fine_to_coarse(tabrow[j]); //i
            energyvecout_(parent) += energyvecin_(tabrow[j]);
        }
    }
    energyvecout_ *= 1./NEredfac;

//    // do max approach instead of sum
//    energyvecout_ = -1.*std::numeric_limits<float>::max();
//    for (int i = 0; i < NEmacro; i++)
//    {
//        dreftable.GetRow(i, tabrow);
//        for (int j = 0; j < tabrow.Size(); j++) {
//            energyvecout_(i) = std::max(energyvecin_(tabrow[j]), energyvecout_(i));
//        }
//    }
}

void TMOPTypeRefiner::Reset()
{
   estimator.Reset();
   current_sequence = -1;
   num_marked_elements = 0;
}

#ifdef MFEM_USE_MPI
void TMOPTypeRefiner::DetermineAMRTypeEnergy(ParMesh &pmesh,
                                             Vector &energyvecin_,
                                             Vector &energyvecout_)
{
    ParNCMesh *pncmesh = pmesh.pncmesh;
    if (!pncmesh) { return; }

    const Table &dreftable = pncmesh->GetDerefinementTable();
    const int NE = pmesh.GetGlobalNE();
    Array<int> tabrow;
    Table coarse_to_fine;
    const CoarseFineTransformations &rtrans=
            pncmesh->GetRefinementTransforms();
    rtrans.GetCoarseToFineMapFast(pmesh, coarse_to_fine,
                                  pncmesh->GetNGhostElements());

    Vector fine_to_coarse(pmesh.GetNE());
    for (int i = 0; i < coarse_to_fine.Size(); i++) {
        coarse_to_fine.GetRow(i, tabrow);
        for (int j = 0; j < tabrow.Size(); j++) {
            fine_to_coarse(tabrow[j]) = i;
        }
    }

    int NEredfac = 1;
    if (reftype & 1) { NEredfac *= 2; }
    if (reftype & 2) { NEredfac *= 2; }
    if (reftype & 4) { NEredfac *= 2; }

    int NEmacro = NE/NEredfac;
    int NE_current = dreftable.Size();
    int NE_current_total = NE_current;
    MPI_Allreduce(&NE_current, &NE_current_total, 1, MPI_INT, MPI_SUM, pmesh.GetComm());

    MFEM_VERIFY(NE_current_total==NEmacro, " Not all elements can be derefined\n;")
    energyvecout_ = 0.0;
    for (int i = 0; i < pmesh.GetNE()/NEredfac; i++)
    {
        dreftable.GetRow(i, tabrow);
        for (int j = 0; j < tabrow.Size(); j++) {
            int parent = fine_to_coarse(tabrow[j]); //i
            energyvecout_(parent) += energyvecin_(tabrow[j]);
        }
    }
    energyvecout_ *= 1./NEredfac;
}
#endif

class TMOPTypeDeRefiner : public MeshOperator
{
protected:
   TMOPEstimator &estimator;
   Mesh *mtemp;
   FiniteElementSpace *ftemp;
   GridFunction *xsav;
#ifdef MFEM_USE_MPI
   ParMesh *pmtemp;
   ParFiniteElementSpace *pftemp;
   ParGridFunction *pxsav;
#endif
   Table dtable;
   Array<int> dtable_par_ref_type;
   Table coarse_to_fine;

   long   max_elements;
   long num_marked_elements;

   Array<Refinement> marked_elements;
   long current_sequence;

   int non_conforming;
   int nc_limit;
   int dim;
   int reftype;

   /** @brief Apply the operator to theG mesh->
       @return STOP if a stopping criterion is satisfied or no elements were
       marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// Construct a ThresholdRefiner using the given ErrorEstimator.
   TMOPTypeDeRefiner(TMOPEstimator &est, int dim_);

   // default destructor (virtual)
   ~TMOPTypeDeRefiner()
   {
       delete mtemp;
       delete ftemp;
       delete xsav;
#ifdef MFEM_USE_MPI
       delete pmtemp;
       delete pftemp;
       delete pxsav;
#endif
   }

   void SetRefType(int type_) { reftype = type_; }

   void DetermineAMRTypeEnergy(Mesh &mesh,
                               Vector &amrevecin,
                               const Vector &baseenergychild,
                               Array<int> &new_parent_ref_type);

   void SetMetaInfo(Mesh &mesh, GridFunction &gf);

   void RestoreNodalPositions(Mesh &mesh, GridFunction &x);

#ifdef MFEM_USE_MPI
   void DetermineAMRTypeEnergy(ParMesh &mesh,
                               Vector &amrevecin,
                               const Vector &baseenergychild,
                               Array<int> &new_parent_ref_type,
                               int nghost = 0);
   void SetMetaInfo(ParMesh &pmesh, ParGridFunction &pgf);
   void RestoreNodalPositions(ParMesh &mesh, ParGridFunction &x);
#endif

   /// Use nonconforming refinement, if possible (triangles, quads, hexes).
   void PreferNonconformingRefinement() { non_conforming = 1; }

   /** @brief Use conforming refinement, if possible (triangles, tetrahedra)
       -- this is the default. */
   void PreferConformingRefinement() { non_conforming = -1; }

   /** @brief Set the maximum ratio of refinement levels of adjacent elements
       (0 = unlimited). */
   void SetNCLimit(int nc_limit)
   {
      MFEM_ASSERT(nc_limit >= 0, "Invalid NC limit");
      this->nc_limit = nc_limit;
   }

   /// Get the number of marked elements in the last Apply() call.
   long GetNumMarkedElements() const { return num_marked_elements; }

   /// Reset the associated estimator.
   virtual void Reset();
};

TMOPTypeDeRefiner::TMOPTypeDeRefiner(TMOPEstimator &est, int dim_)
   : estimator(est), mtemp(NULL), ftemp(NULL), xsav(NULL),
#ifdef MFEM_USE_MPI
     pmtemp(NULL), pftemp(NULL), pxsav(NULL),
#endif
     dim(dim_)
{
   max_elements = std::numeric_limits<long>::max();

   num_marked_elements = 0L;
   current_sequence = -1;

   non_conforming = -1;
   nc_limit = 0;

   reftype = 0;
}


void TMOPTypeDeRefiner::SetMetaInfo(Mesh &mesh, GridFunction &gf)
{
    if (mtemp) {
        delete mtemp;
        delete ftemp;
        delete xsav;
    }
    mtemp = new Mesh(mesh);
    ftemp = new FiniteElementSpace(mtemp, gf.FESpace()->FEColl(), dim);
    xsav = new GridFunction(ftemp);
    *xsav = gf;
    NCMesh *ncmesh = mesh.ncmesh;

    dtable = ncmesh->GetDerefinementTable();
    dtable_par_ref_type.SetSize(0);
    Array<int> tabrow;
    for (int i = 0; i < dtable.Size(); i++) {
        const int *child = dtable.GetRow(i);
        dtable_par_ref_type.Append(ncmesh->GetElementParentRefType(*child));
    }
}

void TMOPTypeDeRefiner::DetermineAMRTypeEnergy(Mesh &mesh,
                                               Vector &amrevecin,
                                               const Vector &baseenergychild,
                                               Array<int> &new_parent_ref_type)
{
    NCMesh *ncmesh = mesh.ncmesh;
    const CoarseFineTransformations &rtrans = ncmesh->GetDerefinementTransforms();
    rtrans.GetCoarseToFineMapFast(mesh, coarse_to_fine);

    new_parent_ref_type.SetSize(mesh.GetNE());
    new_parent_ref_type = 0;

    Array<int> tabrow;

    for (int i = 0; i < coarse_to_fine.Size(); i++) {
        coarse_to_fine.GetRow(i, tabrow);
        for (int j = 0; j < tabrow.Size(); j++) {
            int child = tabrow[j];
            int orig_parent_ref_type = -1;
            Array<int> tabroworig;
            for (int ii = 0; ii < dtable.Size() && orig_parent_ref_type == -1; ii++) {
                dtable.GetRow(ii, tabroworig);
                for (int jj = 0; jj < tabroworig.Size() && orig_parent_ref_type == -1; jj++) {
                    int childorig = tabroworig[jj];
                    if (childorig == child) {
                        orig_parent_ref_type = dtable_par_ref_type[ii];
                    }
                }
            }
            if (orig_parent_ref_type != -1) {
                new_parent_ref_type[i] = orig_parent_ref_type;
                double child_energy = baseenergychild(child);
                if (new_parent_ref_type[i] & 1) { child_energy /= 2.; }
                if (new_parent_ref_type[i] & 2) { child_energy /= 2.; }
                if (new_parent_ref_type[i] & 4) { child_energy /= 2.; }
                amrevecin(i) -= child_energy;
            }
        }
    }
}

void TMOPTypeDeRefiner::RestoreNodalPositions(Mesh &mesh, GridFunction &x)
{
    NCMesh *ncmesh = mesh.ncmesh;
    Table coarse_to_fine2, ref_type_to_matrix2;
    Array<int> coarse_to_ref_typ2;
    Array<mfem::Geometry::Type> ref_type_to_geom2;
    const CoarseFineTransformations &rtrans2 = ncmesh->GetRefinementTransforms();
    rtrans2.GetCoarseToFineMapFast(mesh, coarse_to_fine2);
    Array<int> vdofs, verdofs;
    for (int e = 0; e < mesh.GetNE(); e++) {
        x.FESpace()->GetElementVDofs(e, vdofs);
        x.FESpace()->GetElementVertices(e, verdofs);
        Vector loc_data;
        x.GetSubVector(verdofs, loc_data);

        // determine parent
        int parent = 0;
        for (int z = 0; z < coarse_to_fine2.Size(); z++) {
            Array<int> tabrow;
            coarse_to_fine2.GetRow(z, tabrow);
            for (int j = 0; j < tabrow.Size() && parent == 0; j++) {
                parent = tabrow[j] == e ? z : 0;
            }
        }

        // determine children of parents in original mesh and copy data if match
        Array<int> tabroworig;
        coarse_to_fine.GetRow(parent, tabroworig);
        int match = 0;
        for (int z = 0; z < tabroworig.Size() && match == 0; z++) {
            int child_potential = tabroworig[z];
            Array<int> verdofsm, vdofsm;
            xsav->FESpace()->GetElementVertices(child_potential, verdofsm);
            Vector loc_data_m;
            xsav->GetSubVector(verdofsm, loc_data_m);
            Vector dloc_data = loc_data_m;
            dloc_data -= loc_data;
            if (dloc_data.Norml2() == 0) {
                match = 1;
                xsav->FESpace()->GetElementVDofs(child_potential, vdofsm);
                xsav->GetSubVector(vdofsm, loc_data_m);
                x.SetSubVector(vdofs, loc_data_m);
            }
        }
    }

    x.SetTrueVector();
    x.SetFromTrueVector();
}

int TMOPTypeDeRefiner::ApplyImpl(Mesh &mesh)
{
   NCMesh *ncmesh = mesh.ncmesh;
   if (!ncmesh) { return NONE; }

   const Table &dreftable = ncmesh->GetDerefinementTable();

   const long num_elements = mesh.GetGlobalNE();
   if (num_elements >= max_elements) { return STOP; }

   int NEredfac = 1;
   if (reftype & 1) { NEredfac *= 2; }
   if (reftype & 2) { NEredfac *= 2; }
   if (reftype & 4) { NEredfac *= 2; }

   const int NE = mesh.GetNE();
   if (dreftable.Size()!=NE/NEredfac && reftype != 0) {
       MFEM_ABORT(" Not all elements can be derefined\n;")
   }

   Vector local_err(NE);
   local_err = 0.;
   double threshold = std::numeric_limits<float>::max();
   mesh.DerefineByError(local_err, threshold, 0, 1);


   return CONTINUE + DEREFINED;
}

void TMOPTypeDeRefiner::Reset()
{
   estimator.Reset();
   current_sequence = -1;
   num_marked_elements = 0;
}

#ifdef MFEM_USE_MPI
void TMOPTypeDeRefiner::SetMetaInfo(ParMesh &pmesh, ParGridFunction &pgf)
{
    if (pmtemp) {
        delete pmtemp;
        delete pftemp;
        delete pxsav;
    }
    pmtemp = new ParMesh(pmesh);
    pftemp = new ParFiniteElementSpace(pmtemp, pgf.ParFESpace()->FEColl(), dim);
    pxsav = new ParGridFunction(pftemp);
    *pxsav = pgf;
    ParNCMesh *pncmesh = pmesh.pncmesh;

    dtable = pncmesh->GetDerefinementTable();
    dtable_par_ref_type.SetSize(0);
    Array<int> tabrow;
    for (int i = 0; i < dtable.Size(); i++) {
        const int *child = dtable.GetRow(i);
        dtable_par_ref_type.Append(pncmesh->GetElementParentRefType(*child));
    }
}

void TMOPTypeDeRefiner::DetermineAMRTypeEnergy(ParMesh &pmesh,
                                               Vector &amrevecin,
                                               const Vector &baseenergychild,
                                               Array<int> &new_parent_ref_type,
                                               int nghost)
{
    ParNCMesh *pncmesh = pmesh.pncmesh;
    const CoarseFineTransformations &rtrans = pncmesh->GetDerefinementTransforms();
    rtrans.GetCoarseToFineMapFast(pmesh, coarse_to_fine, nghost);

    new_parent_ref_type.SetSize(pmesh.GetNE());
    new_parent_ref_type = 0;

    Array<int> tabrow;

    for (int i = 0; i < amrevecin.Size(); i++) {
        coarse_to_fine.GetRow(i, tabrow);
        for (int j = 0; j < tabrow.Size(); j++) {
            int child = tabrow[j];
            int orig_parent_ref_type = -1;
            Array<int> tabroworig;
            for (int ii = 0; ii < dtable.Size() && orig_parent_ref_type == -1; ii++) {
                dtable.GetRow(ii, tabroworig);
                for (int jj = 0; jj < tabroworig.Size() && orig_parent_ref_type == -1; jj++) {
                    int childorig = tabroworig[jj];
                    if (childorig == child) {
                        orig_parent_ref_type = dtable_par_ref_type[ii];
                    }
                }
            }
            if (orig_parent_ref_type != -1) {
                new_parent_ref_type[i] = orig_parent_ref_type;
                double child_energy = baseenergychild(child);
                if (new_parent_ref_type[i] & 1) { child_energy /= 2.; }
                if (new_parent_ref_type[i] & 2) { child_energy /= 2.; }
                if (new_parent_ref_type[i] & 4) { child_energy /= 2.; }
                amrevecin(i) -= child_energy;
            }
        }
    }
}

void TMOPTypeDeRefiner::RestoreNodalPositions(ParMesh &pmesh, ParGridFunction &x)
{
    ParNCMesh *pncmesh = pmesh.pncmesh;
    Table coarse_to_fine2, ref_type_to_matrix2;
    Array<int> coarse_to_ref_typ2;
    Array<mfem::Geometry::Type> ref_type_to_geom2;
    const CoarseFineTransformations &rtrans2 = pncmesh->GetRefinementTransforms();
    rtrans2.GetCoarseToFineMapFast(pmesh, coarse_to_fine2, pncmesh->GetNGhostElements());

    Array<int> vdofs, verdofs;
    for (int e = 0; e < pmesh.GetNE(); e++) {
        x.ParFESpace()->GetElementVDofs(e, vdofs);
        x.ParFESpace()->GetElementVertices(e, verdofs);
        Vector loc_data;
        x.GetSubVector(verdofs, loc_data);

        // determine parent
        int parent = 0;
        for (int z = 0; z < coarse_to_fine2.Size(); z++) {
            Array<int> tabrow;
            coarse_to_fine2.GetRow(z, tabrow);
            for (int j = 0; j < tabrow.Size() && parent == 0; j++) {
                parent = tabrow[j] == e ? z : 0;
            }
        }

        // determine children of parents in original mesh and copy data if match
        Array<int> tabroworig;
        coarse_to_fine.GetRow(parent, tabroworig);
        int match = 0;
        for (int z = 0; z < tabroworig.Size() && match == 0; z++) {
            int child_potential = tabroworig[z];
            Array<int> verdofsm, vdofsm;
            pxsav->ParFESpace()->GetElementVertices(child_potential, verdofsm);
            Vector loc_data_m;
            pxsav->GetSubVector(verdofsm, loc_data_m);
            Vector dloc_data = loc_data_m;
            dloc_data -= loc_data;
            if (dloc_data.Norml2() == 0) {
                match = 1;
                pxsav->ParFESpace()->GetElementVDofs(child_potential, vdofsm);
                pxsav->GetSubVector(vdofsm, loc_data_m);
                x.SetSubVector(vdofs, loc_data_m);
            }
        }
    }

    x.SetTrueVector();
    x.SetFromTrueVector();
}
#endif

class TMOPAMR
{
protected:
    Array<FiniteElementSpace *> fespacearr;
    Array<GridFunction *> gfarr;
    Array<GridFunction *> meshnodarr;
    Array<NonlinearForm *> nlfarr;
    DiscreteAdaptTC *tcd;
#ifdef MFEM_USE_MPI
    Array<ParFiniteElementSpace *> pfespacearr;
    Array<ParGridFunction *> pgfarr;
    Array<ParGridFunction *> pmeshnodarr;
    Array<ParNonlinearForm *> pnlfarr;
#endif

public:
    TMOPAMR() : tcd(NULL) { }
    TMOPAMR(DiscreteAdaptTC &tcd_) : tcd(&tcd_) { }

    void SetDiscreteTC(DiscreteAdaptTC *tcd_) {tcd = tcd_;}
    void AddFESpace(FiniteElementSpace *fespace_) { fespacearr.Append(fespace_); }
    void AddGF(GridFunction *gf_) { gfarr.Append(gf_); }
    void AddMeshNodeAr(GridFunction *gf_) { meshnodarr.Append(gf_); }
    void AddNonLinearFormAr(NonlinearForm *nlf_) { nlfarr.Append(nlf_); }
#ifdef MFEM_USE_MPI
    void AddFESpace(ParFiniteElementSpace *pfespace_) { pfespacearr.Append(pfespace_); }
    void AddGF(ParGridFunction *pgf_) { pgfarr.Append(pgf_); }
    void AddMeshNodeAr(ParGridFunction *pgf_) { pmeshnodarr.Append(pgf_); }
    void AddNonLinearFormAr(ParNonlinearForm *nlf_) { pnlfarr.Append(nlf_); }
#endif

    void Update(NonlinearForm &a, Mesh &mesh, FiniteElementSpace &fespace, bool move_bnd);
#ifdef MFEM_USE_MPI
    void Update(ParNonlinearForm &a, ParMesh &pmesh,
                ParFiniteElementSpace &pfespace, bool move_bnd);
#endif
};

void TMOPAMR::Update(NonlinearForm &a, Mesh &mesh, FiniteElementSpace &fespace,
                               bool move_bnd)
{
   // Update FE and GF
   for (int i = 0; i < fespacearr.Size(); i++) { fespacearr[i]->Update(); }
   for (int i = 0; i < gfarr.Size(); i++) { gfarr[i]->Update(); }
   for (int i = 0; i < meshnodarr.Size(); i++) {
       meshnodarr[i]->Update();
       meshnodarr[i]->SetTrueVector();
       meshnodarr[i]->SetFromTrueVector();
   }

   // Update Discrete Indicator
   if (tcd) { tcd->Update(); }

   // Update Nonlinear form and Set Essential BC
   a.Update();
   int dim = fespace.GetFE(0)->GetDim();
   if (move_bnd == false)
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      const int nd  = fespace.GetBE(0)->GetDof();
      int n = 0;
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         const int attr = mesh.GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         const int attr = mesh.GetBdrElement(i)->GetAttribute();
         fespace.GetBdrElementVDofs(i, vdofs);
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
      a.SetEssentialVDofs(ess_vdofs);
   }
};

#ifdef MFEM_USE_MPI
void TMOPAMR::Update(ParNonlinearForm &a, ParMesh &pmesh,
                     ParFiniteElementSpace &pfespace, bool move_bnd)
{
   // Update FE and GF
   for (int i = 0; i < pfespacearr.Size(); i++) { pfespacearr[i]->Update(); }
   for (int i = 0; i < pgfarr.Size(); i++) { pgfarr[i]->Update(); }
   for (int i = 0; i < pmeshnodarr.Size(); i++) {
       pmeshnodarr[i]->Update();
       pmeshnodarr[i]->SetTrueVector();
       pmeshnodarr[i]->SetFromTrueVector();
   }

   // Update Discrete Indicator
   if (tcd) { tcd->ParUpdate(); }

   // Update Nonlinear form and Set Essential BC
   a.Update();
   int dim = pfespace.GetFE(0)->GetDim();
   if (move_bnd == false)
   {
       Array<int> ess_bdr(pmesh.bdr_attributes.Max());
       ess_bdr = 1;
       a.SetEssentialBC(ess_bdr);
   }
   else
   {
       const int nd  = pfespace.GetBE(0)->GetDof();
       int n = 0;
       for (int i = 0; i < pmesh.GetNBE(); i++)
       {
          const int attr = pmesh.GetBdrElement(i)->GetAttribute();
          MFEM_VERIFY(!(dim == 2 && attr == 3),
                      "Boundary attribute 3 must be used only for 3D meshes. "
                      "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                      "components, rest for free nodes), or use -fix-bnd.");
          if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
          if (attr == 4) { n += nd * dim; }
       }
       Array<int> ess_vdofs(n), vdofs;
       n = 0;
       for (int i = 0; i < pmesh.GetNBE(); i++)
       {
          const int attr = pmesh.GetBdrElement(i)->GetAttribute();
          pfespace.GetBdrElementVDofs(i, vdofs);
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
       a.SetEssentialVDofs(ess_vdofs);
   }
};
#endif
