//
// This code is designed to test MFEM's implementation of a simple system of
// advection diffusion equations. It is a modified version of linear_hyp_1d.
//
// Automatic time step computation is not currently selecting stable
// time steps.
//
// Periodic test:
//   ./linear_adv_diff_1d -m ../../data/periodic-segment.mesh -dt 1e-3
//
// Boundary condition test:
//   ./linear_adv_diff_1d -m ../../data/inline-segment.mesh -dt 1e-3
//
// NOTE: The boundary condition test shown above runs and seems to
// perform adequately. However, I am very skeptical about the current
// implementation. I'm curious to hear any feedback. There are two
// other locations in this file with comments starting with "NOTE"
// which indicate the boundary condition implementations.
//
#include "mfem.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Choice for the problem setup. See InitialCondition in ex18.hpp.
int problem;

// Equation constant parameters.
const int num_equation = 2;

// Maximum characteristic speed (updated by integrators)
double max_char_speed;

// Exact solution starts as a constant in `n` and a Gaussian in
// `q`. The time-dependent solution consists of Gaussians moving to
// the left and right in both `n` and `q`.
class ExactNCoef : public Coefficient
{
private:
   mutable Vector x_;

public:
   ExactNCoef() : x_(1) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);

      const double a = 50.0;
      double xp = x_[0] - 0.5 - time;
      double xm = x_[0] - 0.5 + time;
      return 1.0 + 0.5 * (exp(-a * xp * xp) - exp(-a * xm * xm));
   }
};

class ExactQCoef : public Coefficient
{
private:
   mutable Vector x_;

public:
   ExactQCoef() : x_(1) {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      T.Transform(ip, x_);

      const double a = 50.0;
      double xp = x_[0] - 0.5 - time;
      double xm = x_[0] - 0.5 + time;
      return 0.5 * (exp(-a * xp * xp) + exp(-a * xm * xm));
   }
};

class ExactNQCoef : public VectorCoefficient
{
private:
   mutable Vector x_;

public:
   ExactNQCoef() : VectorCoefficient(2), x_(1) {}

   void Eval(Vector &nq, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      nq.SetSize(2);

      T.Transform(ip, x_);

      const double a = 50.0;
      double xp = x_[0] - 0.5 - time;
      double xm = x_[0] - 0.5 + time;

      double ep = exp(-a * xp * xp);
      double em = exp(-a * xm * xm);

      nq[0] = 1.0 + 0.5 * (ep - em);
      nq[1] = 0.5 * (ep + em);
   }

   virtual void Project(QuadratureFunction &qf) {}
};


// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form.
class FE_Evolution : public TimeDependentOperator
{
private:
   const int dim;

   ParFiniteElementSpace &vfes;
   ParFiniteElementSpace &qfes;
   ParBlockNonlinearForm &A;
   DenseTensor Mn_inv;

   double sigma, kappa;
   Coefficient &dCoef;
   ProductCoefficient dtdCoef;

   ParBilinearForm mq;
   // HypreParMatrix Mq;
   OperatorHandle Mq;
   Solver * Mq_inv;

   mutable Vector state;
   mutable DenseMatrix f;
   mutable DenseTensor flux;
   mutable Vector z;

public:
   FE_Evolution(ParFiniteElementSpace &vfes_,
                ParFiniteElementSpace &qfes_,
                ParBlockNonlinearForm &A_,
                Coefficient &dCoef_, double sigma_, double kappa_, double dt_);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   virtual ~FE_Evolution() { }
};

// Implements a simple Rusanov flux
// (left-over from example 18 but not used here)
class RiemannSolver
{
private:
   Vector flux1;
   Vector flux2;

public:
   RiemannSolver();
   double Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux);
};

// A custom implementation of the MFEM class. This is a temporary
// work-around because the MFEM class does not support DG in
// parallel. This class has the same limitation but at least it will
// run in serial whereas the MFEM implentation will not.
class MyParBlockNonlinearForm : public ParBlockNonlinearForm
{
protected:
   mutable Array<ParGridFunction*> X, Y;

public:
   MyParBlockNonlinearForm(Array<ParFiniteElementSpace *> &pf);

   void Mult(const Vector &x, Vector &y) const;
};

// Custom integrators defining our linear hyperbolic test problem
class LinearAdvDiff1DIntegrator : public BlockNonlinearFormIntegrator
{
protected:
   Coefficient &dCoef;
   VectorCoefficient &nqCoef;

   double sigma, kappa;

   Vector nor, nq;
   Vector shape_n, shape_q;
   DenseMatrix dshape_n, dshape_q;
   DenseMatrix dshape1_q, dshape2_q;
   Vector shape1_n, shape2_n, shape1_q, shape2_q;

public:
   LinearAdvDiff1DIntegrator(Coefficient & dCoef_, double sigma_, double kappa_,
                             VectorCoefficient &nqCoef_)
      : dCoef(dCoef_), nqCoef(nqCoef_), sigma(sigma_), kappa(kappa_), nq(2) {}

   void AssembleElementVector(const Array<const FiniteElement *> &el,
                              ElementTransformation &Tr,
                              const Array<const Vector *> &elfun,
                              const Array<Vector *> &elvec);

   void AssembleFaceVector(const Array<const FiniteElement *> &el1,
                           const Array<const FiniteElement *> &el2,
                           FaceElementTransformations &Tr,
                           const Array<const Vector *> &elfun,
                           const Array<Vector *> &elvect);
};

// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(ParFiniteElementSpace &vfes_,
                           ParFiniteElementSpace &qfes_,
                           ParBlockNonlinearForm &A_,
                           Coefficient &dCoef_, double sigma_, double kappa_,
                           double dt_)
   : TimeDependentOperator(A_.Height()),
     dim(vfes_.GetFE(0)->GetDim()),
     vfes(vfes_),
     qfes(qfes_),
     A(A_),
     Mn_inv(vfes.GetFE(0)->GetDof(), vfes.GetFE(0)->GetDof(), vfes.GetNE()),
     sigma(sigma_),
     kappa(kappa_),
     dCoef(dCoef_),
     dtdCoef(dt_, dCoef),
     mq(&qfes),
     state(num_equation),
     z(A.Height())
{
   // Standard local assembly and inversion for energy mass matrices.
   const int dof = vfes.GetFE(0)->GetDof();

   DenseMatrix Mn(dof);
   DenseMatrixInverse inv(&Mn);
   MassIntegrator mi;
   for (int i = 0; i < vfes.GetNE(); i++)
   {
      mi.AssembleElementMatrix(*vfes.GetFE(i),
                               *vfes.GetElementTransformation(i), Mn);
      inv.Factor();
      inv.GetInverseMatrix(Mn_inv(i));
   }

   // NOTE: The integrator used in the call to
   // "mq.AddBdrFaceIntegrator" should probably be consistent with the
   // corresponding implementation in
   // LinearAdvDiff1DIntegrator::AssembleFaceVector. However, doing so
   // seems to lead to poorer results. Of course the purpose of this
   // miniaqpp is to experiment with this boundary condition so
   // perhaps neither choice is the best solution.
   mq.AddDomainIntegrator(new MassIntegrator);
   mq.AddDomainIntegrator(new DiffusionIntegrator(dtdCoef));
   mq.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dtdCoef, sigma, kappa));
   mq.AddBdrFaceIntegrator(new DGDiffusionIntegrator(dtdCoef, sigma, kappa));
   mq.Assemble();

   Array<int > ess_tdof_list;
   mq.FormSystemMatrix(ess_tdof_list, Mq);
   HyprePCG *pcg  = new HyprePCG(*Mq.As<HypreParMatrix>());
   pcg->SetTol(1e-8);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(0);
   Mq_inv = pcg;
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // Reset wavespeed computation before operator application.
   max_char_speed = 0.;

   // Create the vector z with the face terms -<F.n(u), [w]>.
   A.Mult(x, z);

   // Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> vdofs;
   const int dof = vfes.GetFE(0)->GetDof();
   DenseMatrix zmat, ymat(dof, num_equation);

   for (int i = 0; i < vfes.GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes.GetElementVDofs(i, vdofs);
      z.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), dof, num_equation);
      mfem::Mult(Mn_inv(i), zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
   }
}

void FE_Evolution::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
   // Create the vector z with the face terms -<F.n(u), [w]>.
   A.Mult(x, z);

   // Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> dofs;
   const int dof = vfes.GetFE(0)->GetDof();
   DenseMatrix zmat, ymat(dof, 1);

   for (int i = 0; i < vfes.GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes.GetElementDofs(i, dofs);
      z.GetSubVector(dofs, zval);
      zmat.UseExternalData(zval.GetData(), dof, 1);
      mfem::Mult(Mn_inv(i), zmat, ymat);
      k.SetSubVector(dofs, ymat.GetData());
   }

   Vector zq(z.GetData() + qfes.GetTrueVSize(), qfes.GetTrueVSize());
   Vector kq(k.GetData() + qfes.GetTrueVSize(), qfes.GetTrueVSize());

   kq = 0.0;
   Mq_inv->Mult(zq, kq);
}

// Compute the maximum characteristic speed.
inline double ComputeMaxCharSpeed(const Vector &state, const int dim)
{
   return 1.0;
}

MyParBlockNonlinearForm::MyParBlockNonlinearForm(Array<ParFiniteElementSpace *>
                                                 &pf)
   : ParBlockNonlinearForm(pf), X(pf.Size()), Y(pf.Size())
{
   for (int s=0; s<pf.Size(); s++)
   {
      X[s] = new ParGridFunction;
      Y[s] = new ParGridFunction;

      X[s]->MakeRef(pf[s], NULL);
      Y[s]->MakeRef(pf[s], NULL);
   }
}

void MyParBlockNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   // xs_true is not modified, so const_cast is okay
   xs_true.Update(const_cast<Vector &>(x), block_trueOffsets);
   ys_true.Update(y, block_trueOffsets);
   xs.Update(block_offsets);
   ys.Update(block_offsets);

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), xs.GetBlock(s));
   }

   BlockNonlinearForm::MultBlocked(xs, ys);

   const ParFiniteElementSpace *pfes = ParFESpace(0);
   ParMesh *pmesh = pfes->GetParMesh();
   const int n_shared_faces = pmesh->GetNSharedFaces();

   if (fnfi.Size() > 0 && n_shared_faces > 0)
   {
      // MFEM_ABORT("TODO: assemble contributions from shared face terms");

      // MFEM_VERIFY(!NonlinearForm::ext, "Not implemented (extensions + faces");
      // Terms over shared interior faces in parallel.

      FaceElementTransformations *tr;
      Array<const FiniteElement*> fe1(fes.Size());
      Array<const FiniteElement*> fe2(fes.Size());

      Array<Array<int> *> vdofs1(fes.Size());
      Array<Array<int> *> vdofs2(fes.Size());
      Array<Vector*> el_x(fes.Size()), el_y(fes.Size());

      for (int i=0; i<fes.Size(); ++i)
      {
         el_x[i] = new Vector();
         el_y[i] = new Vector();
         vdofs1[i] = new Array<int>;
         vdofs2[i] = new Array<int>;
      }

      aux1.HostReadWrite();

      for (int i=0; i<fes.Size(); ++i)
      {
         X[i]->MakeRef(aux1.GetBlock(i), 0); // aux1 contains P.x
         X[i]->ExchangeFaceNbrData();

         Y[i]->MakeRef(aux2.GetBlock(i), 0); // aux2 contains P.y
      }

      for (int i = 0; i < n_shared_faces; i++)
      {
         tr = pmesh->GetSharedFaceTransformations(i, true);
         int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

         for (int s=0; s<fes.Size(); s++)
         {
            fe1[s] = ParFESpace(s)->GetFE(tr->Elem1No);
            fe2[s] = ParFESpace(s)->GetFaceNbrFE(Elem2NbrNo);

            ParFESpace(s)->GetElementVDofs(tr->Elem1No, *vdofs1[s]);
            ParFESpace(s)->GetFaceNbrElementVDofs(Elem2NbrNo, *vdofs2[s]);

            el_x[s]->SetSize(vdofs1[s]->Size() + vdofs2[s]->Size());

            X[s]->GetSubVector(*vdofs1[s], el_x[s]->GetData());
            X[s]->FaceNbrData().GetSubVector(*vdofs2[s],
                                             el_x[s]->GetData() + vdofs1[s]->Size());
         }

         for (int k = 0; k < fnfi.Size(); k++)
         {
            fnfi[k]->AssembleFaceVector(fe1, fe2, *tr, el_x, el_y);

            for (int s=0; s<fes.Size(); s++)
            {
               Y[s]->AddElementVector(*vdofs1[s], el_y[s]->GetData());
            }
         }
      }
   }

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->MultTranspose(
         ys.GetBlock(s), ys_true.GetBlock(s));

      ys_true.GetBlock(s).SetSubVector(*ess_tdofs[s], 0.0);
   }

   ys_true.SyncFromBlocks();
   y.SyncMemory(ys_true);
}

void LinearAdvDiff1DIntegrator::AssembleElementVector(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvec)
{
   if (el.Size() != 2)
   {
      mfem_error("LinearAdvDiff1DIntegrator::AssembleElementVector"
                 " has finite element space of incorrect block number");
   }

   int dof_n = el[0]->GetDof();
   int dof_q = el[1]->GetDof();

   int dim = el[0]->GetDim();

   shape_n.SetSize(dof_n);
   shape_q.SetSize(dof_q);

   dshape_n.SetSize(dof_n, dim);
   dshape_q.SetSize(dof_q, dim);

   Vector dshape_n_v(dshape_n.GetData(), dof_n);
   Vector dshape_q_v(dshape_q.GetData(), dof_q);

   int intorder = 2*el[0]->GetOrder() + 0; // <---
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   elvec[0]->SetSize(dof_n);
   elvec[1]->SetSize(dof_q);

   *elvec[0] = 0.0;
   *elvec[1] = 0.0;

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);

      double detJ = Tr.Weight();

      el[0]->CalcPhysShape(Tr, shape_n);
      el[1]->CalcPhysShape(Tr, shape_q);

      el[0]->CalcPhysDShape(Tr, dshape_n);
      el[1]->CalcPhysDShape(Tr, dshape_q);

      double n = shape_n * *elfun[0];
      double q = shape_q * *elfun[1];

      elvec[0]->Add(ip.weight * detJ * q, dshape_n_v);
      elvec[1]->Add(ip.weight * detJ * n, dshape_q_v);

      double d = dCoef.Eval(Tr, ip);
      double diff_flux = -d * (dshape_q_v * *elfun[1]);

      elvec[1]->Add(ip.weight * detJ * diff_flux, dshape_q_v);
   }
}

void LinearAdvDiff1DIntegrator::AssembleFaceVector(
   const Array<const FiniteElement *> &el1,
   const Array<const FiniteElement *> &el2,
   FaceElementTransformations &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvec)
{
   if (el1.Size() != 2)
   {
      mfem_error("LinearAdvDiff1DIntegrator::AssembleFaceVector"
                 " has finite element space of incorrect block number");
   }

   bool bdr_face = Tr.Elem2 == NULL;

   double alpha = 1.0;

   if (!bdr_face)
   {
      int dof1_n = el1[0]->GetDof();
      int dof2_n = el2[0]->GetDof();
      int dof1_q = el1[1]->GetDof();
      int dof2_q = el2[1]->GetDof();

      int dim = el1[0]->GetDim();

      nor.SetSize(dim);

      shape1_n.SetSize(dof1_n);
      shape2_n.SetSize(dof2_n);
      shape1_q.SetSize(dof1_q);
      shape2_q.SetSize(dof2_q);

      dshape1_q.SetSize(dof1_q, dim);
      dshape2_q.SetSize(dof2_q, dim);

      Vector dshape1_q_v(dshape1_q.GetData(), dof1_q);
      Vector dshape2_q_v(dshape2_q.GetData(), dof2_q);

      int intorder = 2*el1[0]->GetOrder() + 1; // <---
      const IntegrationRule &ir = IntRules.Get(Tr.GetGeometryType(), intorder);

      Vector elfun1_n(elfun[0]->GetData(), dof1_n);
      Vector elfun2_n(elfun[0]->GetData() + dof1_n, dof2_n);
      Vector elfun1_q(elfun[1]->GetData(), dof1_q);
      Vector elfun2_q(elfun[1]->GetData() + dof1_q, dof2_q);

      elvec[0]->SetSize(dof1_n + dof2_n);
      elvec[1]->SetSize(dof1_q + dof2_q);

      *elvec[0] = 0.0;
      *elvec[1] = 0.0;

      Vector elvec1_n(elvec[0]->GetData(), dof1_n);
      Vector elvec2_n(elvec[0]->GetData() + dof1_n, dof2_n);
      Vector elvec1_q(elvec[1]->GetData(), dof1_q);
      Vector elvec2_q(elvec[1]->GetData() + dof1_q, dof2_q);

      for (int p = 0; p < ir.GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir.IntPoint(p);

         // Set the integration point in the face and the neighboring elements
         Tr.SetAllIntPoints(&ip);

         double detJ = Tr.Weight();

         // Access the neighboring elements' integration points
         const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
         const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();

         // Get the normal vector and the flux on the face
         if (dim == 1)
         {
            nor(0) = 2*eip1.x - 1.0;
         }
         else
         {
            CalcOrtho(Tr.Jacobian(), nor);
         }

         el1[0]->CalcPhysShape(*Tr.Elem1, shape1_n);
         el1[1]->CalcPhysShape(*Tr.Elem1, shape1_q);
         el2[0]->CalcPhysShape(*Tr.Elem2, shape2_n);
         el2[1]->CalcPhysShape(*Tr.Elem2, shape2_q);

         el1[1]->CalcPhysDShape(*Tr.Elem1, dshape1_q);
         el2[1]->CalcPhysDShape(*Tr.Elem2, dshape2_q);

         double n1 = elfun1_n * shape1_n;
         double n2 = elfun2_n * shape2_n;
         double q1 = elfun1_q * shape1_q;
         double q2 = elfun2_q * shape2_q;

         double nAvg = 0.5 * (n1 + n2);
         double qAvg = 0.5 * (q1 + q2);
         double nJmp = n1 - n2;
         double qJmp = q1 - q2;

         double Fn = qAvg * nor(0) + 0.5 * alpha * nJmp;
         double Fq = nAvg * nor(0) + 0.5 * alpha * qJmp;

         const double s = -1.0;

         elvec1_n.Add( ip.weight * detJ * Fn * s, shape1_n);
         elvec2_n.Add(-ip.weight * detJ * Fn * s, shape2_n);
         elvec1_q.Add( ip.weight * detJ * Fq * s, shape1_q);
         elvec2_q.Add(-ip.weight * detJ * Fq * s, shape2_q);

         double d1 = dCoef.Eval(*Tr.Elem1, eip1);
         double d2 = dCoef.Eval(*Tr.Elem2, eip2);
         double diff_flux1 = -d1 * (dshape1_q_v * elfun1_q) * nor(0);
         double diff_flux2 = -d2 * (dshape2_q_v * elfun2_q) * nor(0);

         double diff_flux_avg = 0.5 * (diff_flux1 + diff_flux2);

         elvec1_q.Add( ip.weight * detJ * diff_flux_avg * s, shape1_q);
         elvec2_q.Add(-ip.weight * detJ * diff_flux_avg * s, shape2_q);

         elvec1_q.Add(0.5 * ip.weight * detJ * sigma * d1 * qJmp * s,
                      dshape1_q_v);
         elvec2_q.Add(0.5 * ip.weight * detJ * sigma * d2 * qJmp * s,
                      dshape2_q_v);

         double h1 = fabs(Tr.Elem1->Weight());
         double h2 = fabs(Tr.Elem2->Weight());
         double KdAvg = 0.5 * kappa * (d1 / h1 + d2 / h2);

         elvec1_q.Add( ip.weight * detJ * KdAvg * qJmp * s, shape1_q);
         elvec2_q.Add(-ip.weight * detJ * KdAvg * qJmp * s, shape2_q);
      }
   }
   else
   {
      int dof1_n = el1[0]->GetDof();
      int dof1_q = el1[1]->GetDof();

      int dim = el1[0]->GetDim();

      nor.SetSize(dim);

      shape1_n.SetSize(dof1_n);
      shape1_q.SetSize(dof1_q);

      dshape1_q.SetSize(dof1_q, dim);

      Vector dshape1_q_v(dshape1_q.GetData(), dof1_q);

      int intorder = 2*el1[0]->GetOrder() + 1; // <---
      const IntegrationRule &ir = IntRules.Get(Tr.GetGeometryType(), intorder);

      Vector elfun1_n(elfun[0]->GetData(), dof1_n);
      Vector elfun1_q(elfun[1]->GetData(), dof1_q);

      elvec[0]->SetSize(dof1_n);
      elvec[1]->SetSize(dof1_q);

      *elvec[0] = 0.0;
      *elvec[1] = 0.0;

      Vector elvec1_n(elvec[0]->GetData(), dof1_n);
      Vector elvec1_q(elvec[1]->GetData(), dof1_q);

      for (int p = 0; p < ir.GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir.IntPoint(p);

         // Set the integration point in the face and the neighboring elements
         Tr.SetAllIntPoints(&ip);

         double detJ = Tr.Weight();

         // Access the neighboring elements' integration points
         const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
         // const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();

         // Get the normal vector and the flux on the face
         if (dim == 1)
         {
            nor(0) = 2*eip1.x - 1.0;
         }
         else
         {
            CalcOrtho(Tr.Jacobian(), nor);
         }

         el1[0]->CalcPhysShape(*Tr.Elem1, shape1_n);
         el1[1]->CalcPhysShape(*Tr.Elem1, shape1_q);

         el1[1]->CalcPhysDShape(*Tr.Elem1, dshape1_q);

         double n1 = elfun1_n * shape1_n;
         double q1 = elfun1_q * shape1_q;

         nqCoef.Eval(nq, *Tr.Elem1, eip1);

         double n2 = nq[0];
         double q2 = nq[1];

         double nAvg = 0.5 * (n1 + n2);
         double qAvg = 0.5 * (q1 + q2);
         double nJmp = n1 - n2;
         double qJmp = q1 - q2;

         double Fn = qAvg * nor(0) + 0.5 * alpha * nJmp;
         double Fq = nAvg * nor(0) + 0.5 * alpha * qJmp;

         const double s = -1.0;

         elvec1_n.Add( ip.weight * detJ * Fn * s, shape1_n);
         elvec1_q.Add( ip.weight * detJ * Fq * s, shape1_q);

         // NOTE: It's not clear to me what the appropriate boundary
         // conditions should be. The following would seem to be
         // consistent with the BCs applied in ex14p (which probably
         // differ from what we want here in any case). This code is
         // skipped because it produces poorer results. However, we
         // may want to implement something similar so we can use this
         // as a starting point.
         //
         if (false)
         {
            double d1 = dCoef.Eval(*Tr.Elem1, eip1);
            double diff_flux1 = -d1 * (dshape1_q_v * elfun1_q) * nor(0);

            elvec1_q.Add( ip.weight * detJ * diff_flux1 * s, shape1_q);

            elvec1_q.Add(0.5 * ip.weight * detJ * sigma * d1 * qJmp * s,
                         dshape1_q_v);

            double h1 = fabs(Tr.Elem1->Weight());
            double KdAvg = kappa * d1 / h1;

            elvec1_q.Add( ip.weight * detJ * KdAvg * qJmp * s, shape1_q);
         }
      }
   }
}

// Initial condition
void InitialCondition(const Vector &x, Vector &y)
{
   MFEM_ASSERT(x.Size() >= 1, "");
   MFEM_ASSERT(y.Size() >= 2, "");

   const double xc = 0.5;

   double px = x(0) - xc;

   y(0) = 1.0;
   y(1) = exp(-50.0 * px * px);
}


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command-line options.
   problem = 1;
   const char *mesh_file = "../../data/periodic-segment.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 3;
   int ode_solver_type = 1;
   double t_final = 2.0;
   double dt = -0.01;
   double cfl = 0.3;
   double diff_val = 1e-2;
   double sigma = -1.0;
   double kappa = -1.0;
   bool visualization = true;
   int vis_steps = 50;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly before parallel"
                  " partitioning, -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly after parallel"
                  " partitioning.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler,\n\t"
                  "            2 - SDIRK23 (L-stable), 3 - SDIRK33,\n\t"
                  "            4 - Implicit Midpoint Method,\n\t"
                  "            5 - SDIRK23 (A-stable), 6 - SDIRK34");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&diff_val, "-d", "--diff-coef",
                  "Diffusion coefficient for q.");
   args.AddOption(&sigma, "-dg-s", "--sigma",
                  "One of the three DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-dg-k", "--kappa",
                  "One of the three DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   // 3. Read the mesh from the given mesh file. This example requires a 2D
   //    periodic mesh, such as ../data/periodic-square.mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   // 4. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      // Implicit (L-stable) methods
      case 1: ode_solver = new BackwardEulerSolver; break;
      case 2: ode_solver = new SDIRK23Solver(2); break;
      case 3: ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 4: ode_solver = new ImplicitMidpointSolver; break;
      case 5: ode_solver = new SDIRK23Solver; break;
      case 6: ode_solver = new SDIRK34Solver; break;
      default:
         if (Mpi::Root())
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         return 3;
   }

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   cout << "Number of elements:       " << pmesh.GetNE() << endl;
   cout << "Number of faces:          " << pmesh.GetNumFaces() << endl;
   cout << "Number of boundary faces: " << pmesh.GetNBE() << endl;

   // Determine the minimum element size.
   double hmin;
   if (cfl < 0.0)
   {
      cfl = 0.1 / (2 * order + 1);
   }
   {
      double my_hmin = pmesh.GetElementSize(0, 1);
      for (int i = 1; i < pmesh.GetNE(); i++)
      {
         my_hmin = min(pmesh.GetElementSize(i, 1), my_hmin);
      }
      // Reduce to find the global minimum element size
      MPI_Allreduce(&my_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
   }
   if (dt < 0.0)
   {
      dt = hmin * cfl;
   }

   // 7. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim, BasisType::GaussLegendre);
   // DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   // Finite element space for a scalar (thermodynamic quantity)
   ParFiniteElementSpace fes(&pmesh, &fec);
   // Finite element space for all variables together (total thermodynamic state)
   ParFiniteElementSpace vfes(&pmesh, &fec, num_equation, Ordering::byNODES);

   Array<ParFiniteElementSpace*> afes(num_equation);
   afes = &fes;

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

   HYPRE_BigInt glob_size = vfes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << glob_size << endl;
   }

   // 8. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.

   // The solution u has components {density, x-momentum, y-momentum, energy}.
   // These are stored contiguously in the BlockVector u_block.
   Array<int> offsets(num_equation + 1);
   for (int k = 0; k <= num_equation; k++) { offsets[k] = k * fes.GetNDofs(); }
   BlockVector u_block(offsets);

   // Grid functions for visualization.
   ParGridFunction n(&fes, u_block.GetData() + offsets[0]);
   ParGridFunction q(&fes, u_block.GetData() + offsets[1]);

   // Initialize the state.
   VectorFunctionCoefficient u0(num_equation, InitialCondition);
   ParGridFunction sol(&vfes, u_block.GetData());
   sol.ProjectCoefficient(u0);

   // Output the initial solution.
   {
      ostringstream mesh_name;
      mesh_name << "lin-adv-diff-mesh." << setfill('0')
                << setw(6) << Mpi::WorldRank();
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << pmesh;

      for (int k = 0; k < num_equation; k++)
      {
         ParGridFunction uk(&fes, u_block.GetBlock(k));
         ostringstream sol_name;
         sol_name << "lin-adv-diff-" << k << "-init."
                  << setfill('0') << setw(6) << Mpi::WorldRank();
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 9. Set up the nonlinear form corresponding to the DG discretization of the
   //    flux divergence, and assemble the corresponding mass matrix.
   ConstantCoefficient diff_coef(diff_val);
   ExactNQCoef nq_exact;

   MyParBlockNonlinearForm A(afes);
   A.AddDomainIntegrator(new LinearAdvDiff1DIntegrator(diff_coef, sigma, kappa,
                                                       nq_exact));
   A.AddInteriorFaceIntegrator(new LinearAdvDiff1DIntegrator(diff_coef, sigma,
                                                             kappa, nq_exact));
   A.AddBdrFaceIntegrator(new LinearAdvDiff1DIntegrator(diff_coef, sigma,
                                                        kappa, nq_exact));

   // 10. Define the time-dependent evolution operator describing the ODE
   //     right-hand side, and perform time-integration (looping over the time
   //     iterations, ti, with a time-step dt).
   FE_Evolution lin_adv_diff(vfes, fes, A, diff_coef, sigma, kappa, dt);

   // Visualize the density
   socketstream nout, qout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      MPI_Barrier(pmesh.GetComm());
      nout.open(vishost, visport);
      if (!nout)
      {
         if (Mpi::Root())
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         }
         visualization = false;
         if (Mpi::Root())
         {
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         nout << "parallel " << Mpi::WorldSize()
              << " " << Mpi::WorldRank() << "\n";
         nout.precision(precision);
         nout << "solution\n" << pmesh << n;
         nout << "window_title 'n'\n";
         nout << "window_geometry 0 0 400 400\n";
         nout << "pause\n";
         nout << flush;
         if (Mpi::Root())
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
      MPI_Barrier(pmesh.GetComm());
      qout.open(vishost, visport);
      if (!qout)
      {
         if (Mpi::Root())
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         }
         visualization = false;
         if (Mpi::Root())
         {
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         qout << "parallel " << Mpi::WorldSize()
              << " " << Mpi::WorldRank() << "\n";
         qout.precision(precision);
         qout << "solution\n" << pmesh << q;
         qout << "window_title 'q'\n";
         qout << "window_geometry 400 0 400 400\n";
         qout << "pause\n";
         qout << flush;
         if (Mpi::Root())
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   double t = 0.0;
   lin_adv_diff.SetTime(t);
   nq_exact.SetTime(t);
   ode_solver->Init(lin_adv_diff);

   if (cfl > 0 && false)
   {
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces.
      max_char_speed = 0.;
      Vector z(sol.Size());
      A.Mult(sol, z);
      // Reduce to find the global maximum wave speed
      {
         double all_max_char_speed;
         MPI_Allreduce(&max_char_speed, &all_max_char_speed,
                       1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         max_char_speed = all_max_char_speed;
      }
      dt = cfl * hmin / max_char_speed / (2*order+1);
   }
   if (Mpi::Root())
   {
      cout << "Minimum edge length: " << hmin << endl;
      cout << "CFL factor:          " << cfl << endl;
      cout << "Time step:           " << dt << endl;
   }

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      nq_exact.SetTime(t + dt_real);

      ode_solver->Step(sol, t, dt_real);
      if (cfl > 0 && false)
      {
         // Reduce to find the global maximum wave speed
         {
            double all_max_char_speed;
            MPI_Allreduce(&max_char_speed, &all_max_char_speed,
                          1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
            max_char_speed = all_max_char_speed;
         }
         dt = cfl * hmin / max_char_speed / (2*order+1);
      }
      ti++;

      done = (t >= t_final - 1e-8*dt);
      if (done || ti % vis_steps == 0)
      {
         if (Mpi::Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }
         if (visualization)
         {
            MPI_Barrier(pmesh.GetComm());
            nout << "parallel " << Mpi::WorldSize()
                 << " " << Mpi::WorldRank() << "\n";
            nout << "solution\n" << pmesh << n << flush;

            MPI_Barrier(pmesh.GetComm());
            qout << "parallel " << Mpi::WorldSize()
                 << " " << Mpi::WorldRank() << "\n";
            qout << "solution\n" << pmesh << q << flush;
         }
      }
   }

   tic_toc.Stop();
   if (Mpi::Root())
   {
      cout << " done, " << tic_toc.RealTime() << "s." << endl;
   }

   // 11. Save the final solution. This output can be viewed later using GLVis:
   //     "glvis -np 4 -m vortex-mesh -g vortex-1-final".
   for (int k = 0; k < num_equation; k++)
   {
      ParGridFunction uk(&fes, u_block.GetBlock(k));
      ostringstream sol_name;
      sol_name << "lin-adv-diff-" << k << "-final."
               << setfill('0') << setw(6) << Mpi::WorldRank();
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << uk;
   }

   // 12. Compute the L2 solution error summed for all components.
   if ((t_final == 2.0 &&
        strcmp(mesh_file, "../data/periodic-square.mesh") == 0) ||
       (t_final == 3.0 &&
        strcmp(mesh_file, "../data/periodic-hexagon.mesh") == 0))
   {
      const double error = sol.ComputeLpError(2, u0);
      if (Mpi::Root())
      {
         cout << "Solution error: " << error << endl;
      }
   }

   // Free the used memory.
   delete ode_solver;

   return 0;
}
