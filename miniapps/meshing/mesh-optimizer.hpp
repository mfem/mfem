#include "../../mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

class HessianCoefficient : public MatrixCoefficient
{
private:
   int type;
   int typemod = 5;

public:
   HessianCoefficient(int dim, int type_)
      : MatrixCoefficient(dim), typemod(type_) { }

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
         const double small = 0.001, big = 0.01;
         const double xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         const double r = sqrt(xc*xc + yc*yc);
         double r1 = 0.15; double r2 = 0.35; double sf=30.0;
         const double eps = 0.5;

         const double tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));

         double ind = (tan1 - tan2);
         if (ind > 1.0) {ind = 1.;}
         if (ind < 0.0) {ind = 0.;}
         double val = ind * small + (1.0 - ind) * big;
         //K(0, 0) = eps + 1.0 * (tan1 - tan2);
         K(0, 0) = 1.0;
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
         K(0, 0) *= pow(val,0.5);
         K(1, 1) *= pow(val,0.5);
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

         double r1 = 0.25; double r2 = 0.30; double sf=30.0;
         const double szfac = 1;
         const double asfac = 40;
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
         const double small = 0.001, big = 0.01;
         const double xc = pos(0), yc = pos(1);
         const double r = sqrt(xc*xc + yc*yc);

         double tfac = 40;
         double yl1 = 0.45;
         double yl2 = 0.55;
         double wgt = std::tanh((tfac*(yc-yl1) + 2*std::sin(4.0*M_PI*xc)) + 1) -
                      std::tanh((tfac*(yc-yl2) + 2*std::sin(4.0*M_PI*xc)) - 1);
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }
         double szval = wgt * small + (1.0 - wgt) * big;

         const double eps2 = 40;
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

         const double eps2 = 20;
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
         double yval = 0.25;
         K(0, 0) = eps;
         K(1, 1) = eps2 + szfac*yscale*pos(1);
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
      }
   }
};

double discrete_size_2d(const Vector &x)
{
   const int opt = 2;
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
   double th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
   double stretch = 1/cos(th2);
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

// Defined with respect to the icf mesh->
double weight_fun(const Vector &x)
{
   const double r = sqrt(x(0)*x(0) + x(1)*x(1) + 1e-12);
   const double den = 0.002;
   double l2 = 0.2 + 0.5*std::tanh((r-0.16)/den) - 0.5*std::tanh((r-0.17)/den)
               + 0.5*std::tanh((r-0.23)/den) - 0.5*std::tanh((r-0.24)/den);
   return l2;
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
        fespace(&fes_), tmopi(&tmopi_), dofgf(&x_),
        dim(fes_.GetFE(0)->GetDim()), amrmetric(amrmetric_)
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

   void Setevecptr(Array<Vector *> vecin_) {
       amr_refenergy.DeleteAll();
       for (int i= 0; i < vecin_.Size(); i++) {
           amr_refenergy.Append(vecin_[i]);
       }
   }

   void Setevecptr(int type, Vector &vec)
   {
      *amr_refenergy[type] = vec;
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
       amrreftypeenergy(j) = tmopi->GetElementEnergy(*fe, *T, el_x);
   }

   energyvec_ = amrreftypeenergy;
}

class TMOPRefiner : public MeshOperator
{
protected:
   TMOPEstimator &estimator;

   long   max_elements;
   long num_marked_elements;

   Array<Refinement> marked_elements;
   long current_sequence;

   int non_conforming;
   int nc_limit;
   int amrmetric; //0-Size, 1-AspectRatio, 2-Size+AspectRatio
   int dim;

   double GetNorm(const Vector &local_err, Mesh &mesh) const;

   /** @brief Apply the operator to theG mesh->
       @return STOP if a stopping criterion is satisfied or no elements were
       marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// Construct a ThresholdRefiner using the given ErrorEstimator.
   TMOPRefiner(TMOPEstimator &est, int dim_);

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

TMOPRefiner::TMOPRefiner(TMOPEstimator &est, int dim_)
   : estimator(est), dim(dim_)
{
   max_elements = std::numeric_limits<long>::max();

   num_marked_elements = 0L;
   current_sequence = -1;

   non_conforming = -1;
   nc_limit = 0;
}

int TMOPRefiner::ApplyImpl(Mesh &mesh)
{
   num_marked_elements = 0;
   marked_elements.SetSize(0);
   current_sequence = mesh.GetSequence();

   const long num_elements = mesh.GetGlobalNE();
   if (num_elements >= max_elements) { return STOP; }

   const int NE = mesh.GetNE();
   Vector Imp_Ref1 = estimator.GetAMREnergy(0);
   Vector Imp_Ref2 = estimator.GetAMREnergy(1);
   Vector Imp_Ref3 = estimator.GetAMREnergy(2);
   int num_ref_types = 3+4*(dim-2);

   int inum=0;
   for (int el = 0; el < NE; el++)
   {
      double maxval = 0.; //improvement should be atleast 0
      int reftype = 0;
      for (int rt = 0; rt < num_ref_types; rt++)
      {
         Vector Imp_Ref = estimator.GetAMREnergy(rt);
         double imp_ref_el = Imp_Ref(el);
         if (imp_ref_el > maxval) { reftype = rt+1; maxval = imp_ref_el; }
      }
      if ( reftype > 0)
      {
         marked_elements.Append(Refinement(el));
         marked_elements[inum].ref_type = reftype;
         inum += 1;
      }
   }

   std::cout << inum << " elements refined\n";

   num_marked_elements = mesh.ReduceInt(marked_elements.Size());
   if (num_marked_elements == 0) { return STOP; }
   mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
   return CONTINUE + REFINED;
}

void TMOPRefiner::Reset()
{
   estimator.Reset();
   current_sequence = -1;
   num_marked_elements = 0;
}

class TMOPTypeRefiner : public MeshOperator
{
protected:
   TMOPEstimator &estimator;

   long   max_elements;
   long num_marked_elements;

   Array<Refinement> marked_elements;
   long current_sequence;

   int non_conforming;
   int nc_limit;
   int dim;
   int reftype;

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

   non_conforming = -1;
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

   for (int el = 0; el < NE; el++)
   {
      marked_elements.Append(Refinement(el));
      marked_elements[el].ref_type = reftype;
   }

   num_marked_elements = mesh.ReduceInt(marked_elements.Size());
   if (num_marked_elements == 0) { return STOP; }
   mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
   return CONTINUE + REFINED;
}

void TMOPTypeRefiner::Reset()
{
   estimator.Reset();
   current_sequence = -1;
   num_marked_elements = 0;
}

class TMOPTypeDeRefiner : public MeshOperator
{
protected:
   TMOPEstimator &estimator;

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

   void SetRefType(int type_) { reftype = type_; }

   void DetermineAMRTypeEnergy(Mesh &mesh, Vector &energyvecin_,
                               Vector &energyvecout_);

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
   : estimator(est), dim(dim_)
{
   max_elements = std::numeric_limits<long>::max();

   num_marked_elements = 0L;
   current_sequence = -1;

   non_conforming = -1;
   nc_limit = 0;

   reftype = 0;
}

int TMOPTypeDeRefiner::ApplyImpl(Mesh &mesh)
{
   NCMesh *ncmesh = mesh.ncmesh;
   if (!ncmesh) { return NONE; }

   const Table &dreftable = ncmesh->GetDerefinementTable();
   Array<int> derefs;

   const long num_elements = mesh.GetGlobalNE();
   if (num_elements >= max_elements) { return STOP; }

   int NEredfac = 1;
   if (reftype & 1) { NEredfac *= 2; }
   if (reftype & 2) { NEredfac *= 2; }
   if (reftype & 4) { NEredfac *= 2; }

   const int NE = mesh.GetNE();
   MFEM_VERIFY(dreftable.Size()==NE/NEredfac, " Not all elements can be derefined\n;")

   Vector local_err(NE);
   local_err = 0.;
   double threshold = std::numeric_limits<float>::max();
   mesh.DerefineByError(local_err, threshold, 1000, 1);

   return CONTINUE + DEREFINED;
}

void TMOPTypeDeRefiner::DetermineAMRTypeEnergy(Mesh &mesh,
                                               Vector &energyvecin_,
                                               Vector &energyvecout_)
{
    NCMesh *ncmesh = mesh.ncmesh;
    if (!ncmesh) { return; }

    const Table &dreftable = ncmesh->GetDerefinementTable();
    const int NE = mesh.GetNE();
    Array<int> tabrow;

    Table ref_type_to_matrix;
    Table coarse_to_fine;
    Array<int> coarse_to_ref_type;
    Array<Geometry::Type> ref_type_to_geom;
    const CoarseFineTransformations &rtrans= mesh.GetRefinementTransforms();
    rtrans.GetCoarseToFineMap(mesh, coarse_to_fine, coarse_to_ref_type,
                              ref_type_to_matrix, ref_type_to_geom);

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
        //tabrow.Print();
        for (int j = 0; j < tabrow.Size(); j++) {
            int parent = fine_to_coarse(tabrow[j]); //i
            mesh.GetElementCenter(tabrow[j], centerdum);
            center += centerdum;
            energyvecout_(parent) += energyvecin_(tabrow[j]);
            //std::cout << i << " " << fine_to_coarse(tabrow[j]) << " k10parentcomp\n";
        }
        center /= NEredfac;
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

void TMOPTypeDeRefiner::Reset()
{
   estimator.Reset();
   current_sequence = -1;
   num_marked_elements = 0;
}

class TMOPAMRUpdate
{
protected:
    Array<FiniteElementSpace *> fespacearr;
    Array<GridFunction *> gfarr;
    Array<GridFunction *> meshnodarr;
    Array<NonlinearForm *> nlfarr;

public:
    TMOPAMRUpdate() { }

    void AddFESpace(FiniteElementSpace *fespace_) { fespacearr.Append(fespace_); }
    void AddGF(GridFunction *gf_) { gfarr.Append(gf_); }
    void AddMeshNodeAr(GridFunction *gf_) { meshnodarr.Append(gf_); }
    void AddNonLinearFormAr(NonlinearForm *nlf_) { nlfarr.Append(nlf_); }

    void hrupdateFEandGF( );
    void hrupdate(NonlinearForm &a, Mesh &mesh, FiniteElementSpace &fespace, bool move_bnd);
    void hrupdatediscfunction(DiscreteAdaptTC &tcd, GridFunction &gf1,
                              GridFunction &gf2, GridFunction &gf3,
                              int target_id); // specific for testing
};

void TMOPAMRUpdate::hrupdatediscfunction(DiscreteAdaptTC &tcd, GridFunction &gf1,
                                         GridFunction &gf2, GridFunction &gf3, int target_id)
{
    if (target_id == 5) {
        tcd.ResetDiscreteFields();
        tcd.SetAdaptivityEvaluator(new InterpolatorFP);
        tcd.SetSerialDiscreteTargetSize(gf1);
        tcd.FinalizeSerialDiscreteTargetSpec();
    }
    else if (target_id == 6) {
        tcd.ResetDiscreteFields();
        tcd.SetAdaptivityEvaluator(new InterpolatorFP);
        tcd.SetSerialDiscreteTargetSize(gf1);
        tcd.SetSerialDiscreteTargetAspectRatio(gf2);
        tcd.FinalizeSerialDiscreteTargetSpec();
    }
    else if (target_id == 7) {
        tcd.ResetDiscreteFields();
        tcd.SetAdaptivityEvaluator(new InterpolatorFP);
        tcd.SetSerialDiscreteTargetAspectRatio(gf3);
        tcd.FinalizeSerialDiscreteTargetSpec();
    }
    else {
        return;
    }

}

void TMOPAMRUpdate::hrupdateFEandGF()
{
   for (int i = 0; i < fespacearr.Size(); i++) { fespacearr[i]->Update(); }
   for (int i = 0; i < gfarr.Size(); i++) { gfarr[i]->Update(); }
   for (int i = 0; i < meshnodarr.Size(); i++) {
       meshnodarr[i]->Update();
       meshnodarr[i]->SetTrueVector();
   }
}

void TMOPAMRUpdate::hrupdate(NonlinearForm &a, Mesh &mesh, FiniteElementSpace &fespace,
                               bool move_bnd)
{
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
