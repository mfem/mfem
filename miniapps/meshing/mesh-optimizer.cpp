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
//
//            --------------------------------------------------
//            Mesh Optimizer Miniapp: Optimize high-order meshes
//            --------------------------------------------------
//
// This miniapp performs mesh optimization using the Target-Matrix Optimization
// Paradigm (TMOP) by P.Knupp et al., and a global variational minimization
// approach. It minimizes the quantity sum_T int_T mu(J(x)), where T are the
// target (ideal) elements, J is the Jacobian of the transformation from the
// target to the physical element, and mu is the mesh quality metric. This
// metric can measure shape, size or alignment of the region around each
// quadrature point. The combination of targets & quality metrics is used to
// optimize the physical node positions, i.e., they must be as close as possible
// to the shape / size / alignment of their targets. This code also demonstrates
// a possible use of nonlinear operators (the class TMOP_QualityMetric, defining
// mu(J), and the class TMOP_Integrator, defining int mu(J)), as well as their
// coupling to Newton methods for solving minimization problems. Note that the
// utilized Newton methods are oriented towards avoiding invalid meshes with
// negative Jacobian determinants. Each Newton step requires the inversion of a
// Jacobian matrix, which is done through an inner linear solver.
//
// Compile with: make mesh-optimizer
//
// Sample runs:
//   Adapted analytic Hessian:
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Adapted discrete size:
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 7 -tid 5 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//
//   Blade shape:
//     mesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Blade limited shape:
//     mesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8 -lc 5000
//   ICF shape and equal size:
//     mesh-optimizer -o 3 -rs 0 -mid 9 -tid 2 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF shape and initial size:
//     mesh-optimizer -o 3 -rs 0 -mid 9 -tid 3 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF shape:
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF limited shape:
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8 -lc 10
//   ICF combo shape + size (rings, slow convergence):
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 1000 -ls 2 -li 100 -bnd -qt 1 -qo 8 -cmb
//   3D pinched sphere shape (the mesh is in the mfem/data GitHub repository):
//   * mesh-optimizer -m ../../../mfem_data/ball-pert.mesh -o 4 -rs 0 -mid 303 -tid 1 -ni 20 -ls 2 -li 500 -fix-bnd
//   2D non-conforming shape and equal size:
//     mesh-optimizer -m ./amr-quad-q2.mesh -o 2 -rs 1 -mid 9 -tid 2 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8


#include "../../mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

double GetTargetSum(Mesh &mesh, GridFunction &size)
{
   L2_FECollection avg_fec(0, mesh.Dimension());
   FiniteElementSpace avg_fes(&mesh, &avg_fec);
   GridFunction elsize_avgs(&avg_fes);

   size.GetElementAverages(elsize_avgs);
   return elsize_avgs.Sum();
}

class TMOPEstimator : public ErrorEstimator
{
protected:
   long current_sequence;
   double total_error;
   Array<int> aniso_flags;

   FiniteElementSpace *fespace;
   GridFunction *size; ///< Not owned.
   MatrixCoefficient *target_spec;
   GridFunction tarsize;
   bool discrete_size_flag;
   GridFunction *aspr; ///< Not owned.
   MatrixCoefficient *aspr_spec;
   GridFunction taraspr;
   bool discrete_aspr_flag;

   Array<int> isorefs;
   Array<int> anisorefs;
   Array<int> anisorefst;

   int sizeflag; //USE AMR for Size
   int asprflag; //USE AMR for Aspect-Ratio


   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = size->FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();

public:
   TMOPEstimator(FiniteElementSpace &fes,
                 GridFunction &_size,
                 GridFunction &_aspr)
      : current_sequence(-1),
        total_error(0.),
        fespace(&fes),
        size(&_size),
        aspr(&_aspr),
        tarsize(),
        discrete_size_flag(true),
        sizeflag(0),
        asprflag(0) {}
   /// Return the total error from the last error estimate.
   double GetTotalError() const { return total_error; }

   void SetAnalyticTargetSpec(MatrixCoefficient *mspec)
   {target_spec = mspec; discrete_size_flag=false;}

   virtual const Vector &GetLocalErrors() {}

   virtual const GridFunction &GetLocalSolution() {}

   virtual const Array<int> &GetSizeRefinements()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return isorefs;
   }

   virtual const Array<int> &GetAsprRefinements()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return anisorefs;
   }

   virtual const Array<int> &GetAsprRefinementsType()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return anisorefst;
   }

   virtual void Reset() { current_sequence = -1; }

   virtual ~TMOPEstimator() {}

};

void TMOPEstimator::ComputeEstimates()
{
   // Compute error for each element
   Vector size_sol;
   Vector aspr_sol;
   if (!discrete_size_flag)
   {
      const int NE = fespace->GetNE();
      const int dim = fespace->GetMesh()->Dimension();
      GridFunction *nodes = fespace->GetMesh()->GetNodes();
      Vector nodesv(nodes->Size());
      nodesv.SetData(nodes->GetData());
      const int pnt_cnt = nodesv.Size()/dim;
      DenseMatrix K; K.SetSize(dim);
      size_sol.SetSize(pnt_cnt);
      aspr_sol.SetSize(pnt_cnt);
      Vector posv(dim);
      const IntegrationPoint *ip = NULL;
      IsoparametricTransformation *Tpr = NULL;

      for (int i=0; i<pnt_cnt; i++)
      {
         for (int j=0; j<dim; j++) {K(j,j) = nodesv(i+j*pnt_cnt);}
         target_spec->Eval(K,*Tpr,*ip);
         Vector col1, col2;
         K.GetColumn(0, col1);
         K.GetColumn(1, col2);

         size_sol(i) = K.Det(); //col1.Norml2()*col2.Norml2();
         aspr_sol(i) = col2.Norml2()/col1.Norml2();
      }
      size->SetDataAndSize(size_sol.GetData(),size_sol.Size());
      aspr->SetDataAndSize(aspr_sol.GetData(),aspr_sol.Size());
   }

   L2_FECollection avg_fec(0, fespace->GetMesh()->Dimension());
   FiniteElementSpace avg_fes(fespace->GetMesh(), &avg_fec);
   tarsize.SetSpace(&avg_fes);
   int dim = fespace->GetMesh()->Dimension();

   size->GetElementAverages(tarsize);
   const int NE = tarsize.Size();
   isorefs.SetSize(NE);
   for (int i=0; i<NE; i++)
   {
      double curr_size = fespace->GetMesh()->GetElementVolume(i);
      double tar_size  = tarsize(i);
      MFEM_ASSERT(tar_size>0,"Target element size should be greater than 0");
      double loc_err   = curr_size - tar_size;
      double tar_err   = curr_size/(pow(2.,dim)) - tar_size;

//      if (tar_err > 0 && loc_err > tar_err) {
//          isorefs[i] = 1;
//      }
//      else if (tar_err < 0 && loc_err > 0 && loc_err > std::fabs(tar_err))
//      {
//          isorefs[i] = 1;
//      }
      if (loc_err > 0 && loc_err > std::fabs(tar_err))
      {
          isorefs[i] = 1;
      }
      else
      {
          isorefs[i] = 0;
      }
   }

   taraspr.SetSpace(&avg_fes);
   anisorefs.SetSize(NE);
   anisorefst.SetSize(NE);

   //aspr->GetElementAverages(taraspr);

   Vector pos0V(fespace->GetFE(0)->GetDof());
   Array<int> pos_dofs;

   for (int i=0; i<NE; i++)
   {
       aspr->FESpace()->GetElementDofs(i, pos_dofs);
       aspr->GetSubVector(pos_dofs, pos0V);
       double prod = 1.;
       double sum = 1;
       for (int j=0;j<pos0V.Size();j++)
       {
           prod *= pos0V(j);
       }

       taraspr(i) = pow(prod,1./pos0V.Size());
   }


   for (int i=0; i<NE; i++)
   {
      double curr_aspr = fespace->GetMesh()->GetElementAspectRatio(i,0);
      double tar_aspr  = taraspr(i);
      double loc_err   = curr_aspr/tar_aspr;
      anisorefs[i] = 0;
      anisorefs[i] = 0;

      if (loc_err > 4./3.)
      {
          anisorefs[i] = 1;
          anisorefst[i] = 2;
      }
      else if (loc_err < 2./3.)
      {
          anisorefs[i] = 1;
          anisorefst[i] = 1;
      }
   }

   current_sequence = size->FESpace()->GetMesh()->GetSequence();
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

   double GetNorm(const Vector &local_err, Mesh &mesh) const;

   /** @brief Apply the operator to the mesh.
       @return STOP if a stopping criterion is satisfied or no elements were
       marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// Construct a ThresholdRefiner using the given ErrorEstimator.
   TMOPRefiner(TMOPEstimator &est);

   // default destructor (virtual)

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

TMOPRefiner::TMOPRefiner(TMOPEstimator &est)
   : estimator(est)
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
   Array<int> isorefs = estimator.GetSizeRefinements();
   Array<int> anisorefs = estimator.GetAsprRefinements();
   Array<int> anisorefst = estimator.GetAsprRefinementsType();
   MFEM_ASSERT(isorefs.Size() == NE, "invalid size of local_err");


   int inum=0;
   for (int el = 0; el < NE; el++)
   {
      if (isorefs[el] > 0 && anisorefs[el] == 0) //SIZE, NO AR
      {
          marked_elements.Append(Refinement(el));
          marked_elements[inum].ref_type = 3;
          inum += 1;
      }
      else if (isorefs[el] > 0 && anisorefs[el] > 0) //SIZE AND AR
      {
          marked_elements.Append(Refinement(el));
          marked_elements[inum].ref_type = anisorefst[el];
          inum += 1;
      }
      else if (isorefs[el] == 0 && anisorefs[el] > 0) //AR, NO SIZE
      {
          marked_elements.Append(Refinement(el));
          marked_elements[inum].ref_type = anisorefst[el];
          inum += 1;
      }
   }

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



double weight_fun(const Vector &x);

double ind_values(const Vector &x)
{
   const int opt = 2;
   const double small = 0.001, big = 0.01;

   // Sine wave.
   if (opt==1)
   {
      const double X = x(0), Y = x(1);
      double ind = std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) + 1) -
                   std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) - 1);

      if (ind > 1.0) {ind = 1.;}
      if (ind < 0.0) {ind = 0.;}
      return ind * small + (1.0 - ind) * big;
   }

   if (opt==2)
   {
      // Circle in the middle.
      double val = 0.;
      const double xc = x(0) - 0.5, yc = x(1) - 0.5;
      const double r = sqrt(xc*xc + yc*yc);
      double r1 = 0.15; double r2 = 0.35; double sf=30.0;
      val = 0.5*(std::tanh(sf*(r-r1)) - std::tanh(sf*(r-r2)));
      if (val > 1.) {val = 1;}

      return val * small + (1.0 - val) * big;
   }

   if (opt == 3)
   {
      // cross
      const double X = x(0), Y = x(1);
      const double r1 = 0.45, r2 = 0.55;
      const double sf = 40.0;

      double val = 0.5 * ( std::tanh(sf*(X-r1)) - std::tanh(sf*(X-r2)) +
                           std::tanh(sf*(Y-r1)) - std::tanh(sf*(Y-r2)) );
      if (val > 1.) { val = 1.0; }

      return val * small + (1.0 - val) * big;
   }

   if (opt==4)
   {
      // Multiple circles
      double r1,r2,val,rval;
      double sf = 10;
      val = 0.;
      // circle 1
      r1= 0.25; r2 = 0.25; rval = 0.1;
      double xc = x(0) - r1, yc = x(1) - r2;
      double r = sqrt(xc*xc+yc*yc);
      val =  0.5*(1+std::tanh(sf*(r+rval))) - 0.5*(1+std::tanh(sf*
                                                               (r-rval)));// std::exp(val1);
      // circle 2
      r1= 0.75; r2 = 0.75;
      xc = x(0) - r1, yc = x(1) - r2;
      r = sqrt(xc*xc+yc*yc);
      val +=  (0.5*(1+std::tanh(sf*(r+rval))) - 0.5*(1+std::tanh(sf*
                                                                 (r-rval))));// std::exp(val1);
      // circle 3
      r1= 0.75; r2 = 0.25;
      xc = x(0) - r1, yc = x(1) - r2;
      r = sqrt(xc*xc+yc*yc);
      val +=  0.5*(1+std::tanh(sf*(r+rval))) - 0.5*(1+std::tanh(sf*
                                                                (r-rval)));// std::exp(val1);
      // circle 4
      r1= 0.25; r2 = 0.75;
      xc = x(0) - r1, yc = x(1) - r2;
      r = sqrt(xc*xc+yc*yc);
      val +=  0.5*(1+std::tanh(sf*(r+rval))) - 0.5*(1+std::tanh(sf*(r-rval)));
      if (val > 1.0) {val = 1.;}
      if (val < 0.0) {val = 0.;}

      return val * small + (1.0 - val) * big;
   }

   if (opt==5)
   {
      // cross
      double val = 0.;
      double X = x(0)-0.5, Y = x(1)-0.5;
      double rval = std::sqrt(X*X + Y*Y);
      double thval = 60.*M_PI/180.;
      double Xmod,Ymod;
      Xmod = X*std::cos(thval) + Y*std::sin(thval);
      Ymod= -X*std::sin(thval) + Y*std::cos(thval);
      X = Xmod+0.5; Y = Ymod+0.5;
      double r1 = 0.45; double r2 = 0.55; double sf=30.0;
      val = ( 0.5*(1+std::tanh(sf*(X-r1))) - 0.5*(1+std::tanh(sf*(X-r2)))
              + 0.5*(1+std::tanh(sf*(Y-r1))) - 0.5*(1+std::tanh(sf*(Y-r2))) );
      if (rval > 0.4) {val = 0.;}
      if (val > 1.0) {val = 1.;}
      if (val < 0.0) {val = 0.;}

      return val * small + (1.0 - val) * big;
   }

   if (opt==6)
   {
      double val = 0.;
      const double xc = x(0) - 0.0, yc = x(1) - 0.5;
      const double r = sqrt(xc*xc + yc*yc);
      double r1 = 0.45; double r2 = 0.55; double sf=30.0;
      val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
      if (val > 1.) {val = 1;}
      if (val < 0.) {val = 0;}

      return val * small + (1.0 - val) * big;
   }

   return 0.0;
}

class HessianCoefficient : public MatrixCoefficient
{
private:
   int type;
   int typemod = 5;

public:
   HessianCoefficient(int dim, int type_)
      : MatrixCoefficient(dim), type(type_) { }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      if (&T==NULL)
      {
         Vector pos(3);
         for (int i=0; i<K.Size(); i++) {pos(i)=K(i,i);}
         (this)->Eval(K,pos);
      }
      else
      {
         Vector pos(3);
         T.Transform(ip, pos);
         (this)->Eval(K,pos);
      }
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
         const double small = 0.0001, big = 0.01;
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
         K(0,0) *= pow(val,0.5);
         K(1,1) *= pow(val,0.5);
      }
      else if (typemod==2)
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

          double r1 = 0.15; double r2 = 0.35; double sf=30.0;
          const double szfac = 1;
          const double asfac = 10;
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
          if (wgt > 1) wgt = 1;
          if (wgt < 0) wgt = 0;

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

          double tfac = 20;
          double yl1 = 0.45;
          double yl2 = 0.55;
          double wgt = std::tanh((tfac*(yc-yl1) + 2*std::sin(4.0*M_PI*xc)) + 1) -
             std::tanh((tfac*(yc-yl2) + 2*std::sin(4.0*M_PI*xc)) - 1);
          if (wgt > 1) wgt = 1;
          if (wgt < 0) wgt = 0;
          double szval = wgt * small + (1.0 - wgt) * big;

          const double eps2 = 20;
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
          double wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1) -
                       std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) - 1);
          if (wgt > 1) wgt = 1;
          if (wgt < 0) wgt = 0;

          const double eps2 = 40;
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
         K(1, 1) = eps2 + szfac*yscale*pos(1);
         K(0, 0) = eps;
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
      }
   }
};

void TMOPupdate(NonlinearForm &a, Mesh &mesh, FiniteElementSpace &fespace,
                bool move_bnd)
{
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

// Additional IntegrationRules that can be used with the --quad-type option.
IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);


int main (int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 1;
   int target_id         = 1;
   double lim_const      = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int newton_iter       = 10;
   double newton_rtol    = 1e-12;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   bool combomet         = 0;
   int amr_flag          = 1;
   bool normalization    = false;
   bool visualization    = true;
   int verbosity_level   = 0;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t"
                  "1  : |T|^2                          -- 2D shape\n\t"
                  "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                  "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                  "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                  "22 : 0.5(|T|^2-2*tau)/(tau-tau_0)   -- 2D untangling\n\t"
                  "50 : 0.5|T^tT|^2/tau^2-1            -- 2D shape\n\t"
                  "55 : (tau-1)^2                      -- 2D size\n\t"
                  "56 : 0.5(sqrt(tau)-1/sqrt(tau))^2   -- 2D size\n\t"
                  "58 : |T^tT|^2/(tau^2)-2*|T|^2/tau+2 -- 2D shape\n\t"
                  "77 : 0.5(tau-1/tau)^2               -- 2D size\n\t"
                  "211: (tau-1)^2-tau+sqrt(tau^2)      -- 2D untangling\n\t"
                  "252: 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
                  "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
                  "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
                  "303: (|T|^2)/3*tau^(2/3)-1        -- 3D shape\n\t"
                  "315: (tau-1)^2                    -- 3D size\n\t"
                  "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D size\n\t"
                  "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
                  "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "1: Ideal shape, unit size\n\t"
                  "2: Ideal shape, equal size\n\t"
                  "3: Ideal shape, initial size\n\t"
                  "4: Given full analytic Jacobian (in physical space)\n\t"
                  "5: Ideal shape, given size (in physical space)");
   args.AddOption(&lim_const, "-lc", "--limit-const", "Limiting constant.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&newton_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver: 0 - l1-Jacobi, 1 - CG, 2 - MINRES.");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&combomet, "-cmb", "--combo-met", "-no-cmb", "--no-combo-met",
                  "Combination of metrics.");
   args.AddOption(&amr_flag, "-amr", "--amr-flag",
                  "1 - AMR after TMOP");
   args.AddOption(&normalization, "-nor", "--normalization", "-no-nor",
                  "--no-normalization",
                  "Make all terms in the optimization functional unitless.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Initialize and refine the starting mesh.
   Mesh mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   const int dim = mesh.Dimension();
   cout << "Mesh curvature: ";
   if (mesh.GetNodes()) { cout << mesh.GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // 3. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   H1_FECollection fec(mesh_poly_deg, dim);
   FiniteElementSpace fespace(&mesh, &fec, dim);

   // 4. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh.SetNodalFESpace(&fespace);

   // 5. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 6. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   GridFunction x(&fespace);
   GridFunction xnew(&fespace);
   GridFunction x0new(&fespace);
   mesh.SetNodalGridFunction(&x);

   // 7. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in pfespace. Note: this is partition-dependent.
   //
   //    In addition, compute average mesh size and total volume.
   Vector h0(fespace.GetNDofs());
   h0 = infinity();
   double volume = 0.0;
   Array<int> dofs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      fespace.GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const double hi = mesh.GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      volume += mesh.GetElementVolume(i);
   }
   const double small_phys_size = pow(volume, 1.0 / dim) / 100.0;

   // 8. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   GridFunction rdm(&fespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < fespace.GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(fespace.DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < fespace.GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      fespace.GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   x -= rdm;
   x.SetTrueVector();
   x.SetFromTrueVector();

   // 9. Save the starting (prior to the optimization) mesh to a file. This
   //    output can be viewed later using GLVis: "glvis -m perturbed.mesh".
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh.Print(mesh_ofs);
   }

   // 10. Store the starting (prior to the optimization) positions.
   GridFunction x0(&fespace);
   x0 = x;

   // 11. Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 22: metric = new TMOP_Metric_022(tauval); break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 77: metric = new TMOP_Metric_077; break;
      case 211: metric = new TMOP_Metric_211; break;
      case 252: metric = new TMOP_Metric_252(tauval); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 352: metric = new TMOP_Metric_352(tauval); break;
      default: cout << "Unknown metric_id: " << metric_id << endl; return 3;
   }
   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   HessianCoefficient *adapt_coeff = NULL;
   H1_FECollection ind_fec(mesh_poly_deg, dim);
   FiniteElementSpace ind_fes(&mesh, &ind_fec);
   GridFunction size; size.SetSpace(&ind_fes);
   GridFunction aspr; aspr.SetSpace(&ind_fes);
   DiscreteAdaptTC *tcd = NULL;
   AnalyticAdaptTC *tca = NULL;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4:
      {
         target_t = TargetConstructor::GIVEN_FULL;
         tca = new AnalyticAdaptTC(target_t);
         adapt_coeff = new HessianCoefficient(dim, 1);
         tca->SetAnalyticTargetSpec(NULL, NULL, adapt_coeff);
         target_c = tca;
         break;
      }
      case 5:
      {
         target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
         tcd = new DiscreteAdaptTC(target_t);
         size.SetSpace(&ind_fes);
         FunctionCoefficient ind_coeff(ind_values);
         size.ProjectCoefficient(ind_coeff);
         tcd->SetSerialDiscreteTargetSpec(size);
         target_c = tcd;
         break;
      }
      default: cout << "Unknown target_id: " << target_id << endl; return 3;
   }

   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t);
   }
   target_c->SetNodes(x0);
   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, target_c);


   // 12. Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = NULL;
   const int geom_type = fespace.GetFE(0)->GetGeomType();
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default: cout << "Unknown quad_type: " << quad_type << endl;
         delete he_nlf_integ; return 3;
   }
   cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;
   he_nlf_integ->SetIntegrationRule(*ir);

   if (normalization) { he_nlf_integ->EnableNormalization(x0); }

   // 13. Limit the node movement.
   // The limiting distances can be given by a general function of space.
   GridFunction dist(&fespace);
   dist = 1.0;
   // The small_phys_size is relevant only with proper normalization.
   if (normalization) { dist = small_phys_size; }
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, lim_coeff); }

   // 14. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights. Note that there are no
   //     command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   NonlinearForm a(&fespace);
   ConstantCoefficient *coeff1 = NULL;
   TMOP_QualityMetric *metric2 = NULL;
   TargetConstructor *target_c2 = NULL;
   FunctionCoefficient coeff2(weight_fun);

   if (combomet == 1)
   {
      // TODO normalization of combinations.
      // We will probably drop this example and replace it with adaptivity.
      if (normalization) { MFEM_ABORT("Not implemented."); }

      // Weight of the original metric.
      coeff1 = new ConstantCoefficient(1.0);
      he_nlf_integ->SetCoefficient(*coeff1);
      a.AddDomainIntegrator(he_nlf_integ);

      metric2 = new TMOP_Metric_077;
      target_c2 = new TargetConstructor(
         TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE);
      target_c2->SetVolumeScale(0.01);
      target_c2->SetNodes(x0);
      TMOP_Integrator *he_nlf_integ2 = new TMOP_Integrator(metric2, target_c2);
      he_nlf_integ2->SetIntegrationRule(*ir);

      // Weight of metric2.
      he_nlf_integ2->SetCoefficient(coeff2);
      a.AddDomainIntegrator(he_nlf_integ2);
   }
   else { a.AddDomainIntegrator(he_nlf_integ); }

   const double init_energy = a.GetGridFunctionEnergy(x);

   // 15. Visualize the starting mesh and metric values.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, mesh, title, 0);
   }

   // 16. Fix all boundary nodes, or fix only a given component depending on the
   //     boundary attributes of the given mesh. Attributes 1/2/3 correspond to
   //     fixed x/y/z components of the node. Attribute 4 corresponds to an
   //     entirely fixed node. Other boundary attributes do not affect the node
   //     movement boundary conditions.
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

   // 17. As we use the Newton method to solve the resulting nonlinear system,
   //     here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL;
   const double linsol_rtol = 1e-12;
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver;
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver;
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = minres;
   }

   // 18. Compute the minimum det(J) of the starting mesh.
   tauval = infinity();
   const int NE = mesh.GetNE();
   for (int i = 0; i < NE; i++)
   {
      ElementTransformation *transf = mesh.GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir->IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   cout << "Minimum det(J) of the original mesh is " << tauval << endl;

   // 19. Finally, perform the nonlinear optimization.
   NewtonSolver *newton = NULL;
   if (tauval > 0.0)
   {
      tauval = 0.0;
      TMOPNewtonSolver *tns = new TMOPNewtonSolver(*ir);
      if (target_id == 5)
      {
         tns->SetDiscreteAdaptTC(dynamic_cast<DiscreteAdaptTC *>(target_c));
      }
      newton = tns;
      cout << "TMOPNewtonSolver is used (as all det(J) > 0).\n";
   }
   else
   {
      if ( (dim == 2 && metric_id != 22 && metric_id != 252) ||
           (dim == 3 && metric_id != 352) )
      {
         cout << "The mesh is inverted. Use an untangling metric." << endl;
         return 3;
      }
      tauval -= 0.01 * h0.Min(); // Slightly below minJ0 to avoid div by 0.
      newton = new TMOPDescentNewtonSolver(*ir);
      cout << "The TMOPDescentNewtonSolver is used (as some det(J) < 0).\n";
   }
   newton->SetPreconditioner(*S);
   newton->SetMaxIter(newton_iter);
   newton->SetRelTol(newton_rtol);
   newton->SetAbsTol(0.0);
   newton->SetPrintLevel(verbosity_level >= 1 ? 1 : -1);

   // 20. AMR based size refinemenet if a size metric is used
   TMOPEstimator tmope(ind_fes,size,aspr);
   if (target_id==4) {tmope.SetAnalyticTargetSpec(adapt_coeff);}
   TMOPRefiner tmopr(tmope);
   if (amr_flag==1)
   {
      int nc_limit = 1;
      int ni_limit = 10;
      int nic_limit = 4;
      int newtonstop = 0;
      int amrstop = 0;

      tmopr.PreferNonconformingRefinement();
      tmopr.SetNCLimit(nc_limit);

      for (int it = 0; it<ni_limit; it++)
      {

         std::cout << it << " Begin NEWTON+AMR Iteration\n";

         newton->SetOperator(a);
         newton->Mult(b, x.GetTrueVector());
         x.SetFromTrueVector();
         if (newton->GetConverged() == false)
         {
            cout << "NewtonIteration: rtol = " << newton_rtol << " not achieved."
                 << endl;
         }
         if (amrstop==1)
         {
            cout << it << " Newton and AMR have converged" << endl;
            break;
         }
         char title1[10];
         sprintf(title1, "%s %d","Newton", it);
         //vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, mesh, title1, 600);

         for (int amrit=0; amrit<nc_limit; amrit++)
         {
            tmopr.Reset();
            if (nc_limit!=0 && amrstop==0) {tmopr.Apply(mesh);}
            //Update stuff
            ind_fes.Update();
            size.Update();aspr.Update();
            fespace.Update();
            x.Update(); x.SetTrueVector();
            x0.Update(); x0.SetTrueVector();
            fespace.UpdatesFinished();
            if (target_id == 5)
            {
               tcd->SetSerialDiscreteTargetSpec(size);
               target_c = tcd;
               he_nlf_integ->UpdateTargetConstructor(target_c);
            }
            a.Update();
            TMOPupdate(a,mesh,fespace,move_bnd);
            if (amrstop==0)
            {
               if (tmopr.Stop())
               {
                  amrstop = 1;
                  cout << it << " " << amrit <<
                       " AMR stopping criterion satisfied. Stop." << endl;
               }
               else
               {std::cout << mesh.GetNE() << " Number of elements after AMR\n";}
            }
         }
         if (it==nic_limit-1) {amrstop=1;}
         //double newabstol = newton->GetNormGoal();
         //newton->SetAbsTol(newabstol);
         //newton->SetRelTol(0.);

         sprintf(title1, "%s %d","AMR", it);
         //qqvis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, mesh, title1, 600);
      } //ni_limit
   } //amr_flag==1
   delete newton;

   // 21. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized.mesh".
   {
      ofstream mesh_ofs("optimized.mesh");
      mesh_ofs.precision(14);
      mesh.Print(mesh_ofs);
   }

   // 22. Compute the amount of energy decrease.
   const double fin_energy = a.GetGridFunctionEnergy(x);
   double metric_part = fin_energy;
   if (lim_const != 0.0)
   {
      lim_coeff.constant = 0.0;
      metric_part = a.GetGridFunctionEnergy(x);
      lim_coeff.constant = lim_const;
   }
   cout << "Initial strain energy: " << init_energy
        << " = metrics: " << init_energy
        << " + limiting term: " << 0.0 << endl;
   cout << "  Final strain energy: " << fin_energy
        << " = metrics: " << metric_part
        << " + limiting term: " << fin_energy - metric_part << endl;
   cout << "The strain energy decreased by: " << setprecision(12)
        << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;

   // 22. Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, mesh, title, 600);
   }

   // 23. Visualize the mesh displacement.
   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh.Print(sock);
      x0 -= x;
      x0.Save(sock);
      sock.send();
      sock << "window_title 'Displacements'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   // 24. Free the used memory.
   delete S;
   delete target_c2;
   delete metric2;
   delete coeff1;
   delete target_c;
   delete metric;

   return 0;
}

// Defined with respect to the icf mesh.
double weight_fun(const Vector &x)
{
   const double r = sqrt(x(0)*x(0) + x(1)*x(1) + 1e-12);
   const double den = 0.002;
   double l2 = 0.2 + 0.5*std::tanh((r-0.16)/den) - 0.5*std::tanh((r-0.17)/den)
               + 0.5*std::tanh((r-0.23)/den) - 0.5*std::tanh((r-0.24)/den);
   return l2;
}
