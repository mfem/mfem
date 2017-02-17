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

#include "fem.hpp"

namespace mfem
{

void NonlinearFormIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   mfem_error("NonlinearFormIntegrator::AssembleElementGrad"
              " is not overloaded!");
}

double NonlinearFormIntegrator::GetElementEnergy(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
{
   mfem_error("NonlinearFormIntegrator::GetElementEnergy"
              " is not overloaded!");
   return 0.0;
}

// I1 = |M|^2 / det(M).
double HyperelasticModel::Invariant1(const DenseMatrix &M)
{
   if (M.Size() == 3) { MFEM_ABORT("3D is not implemented yet."); }

   double fnorm = M.FNorm();
   return fnorm * fnorm / M.Det();
}

// I2 = det(M).
double HyperelasticModel::Invariant2(const DenseMatrix &M)
{
   if (M.Size() == 3) { MFEM_ABORT("3D is not implemented yet."); }

   return M.Det();
}

// dI1_dM = [ 2 det(M) M - |M|^2 adj(M)^T ] / det(T)^2.
void HyperelasticModel::Invariant1_dM(const DenseMatrix &M, DenseMatrix &dM)
{
   if (M.Size() == 3) { MFEM_ABORT("3D is not implemented yet."); }

   double fnorm = M.FNorm();
   double det   = M.Det();

   Invariant2_dM(M, dM);
   dM *= - fnorm * fnorm;
   dM.Add(2.0 * det, M);
   dM *= 1.0 / (det * det);
}

// Assuming 2D.
// dI2_dM = d(det(M))_dM = adj(M)^T.
void HyperelasticModel::Invariant2_dM(const DenseMatrix &M, DenseMatrix &dM)
{
   if (M.Size() == 3) { MFEM_ABORT("3D is not implemented yet."); }

   dM(0, 0) =  M(1, 1); dM(0, 1) = -M(1, 0);
   dM(1, 0) = -M(0, 1); dM(1, 1) =  M(0, 0);
}

// (dI1_dM)_d(Mij) = d[(2 det(M) M - |M|^2 adj(M)^T) / det(T)^2]_d[Mij].
void HyperelasticModel::Invariant1_dMdM(const DenseMatrix &M, int i, int j,
                                        DenseMatrix &dMdM)
{
   if (M.Size() == 3) { MFEM_ABORT("3D is not implemented yet."); }

   // Compute d(det(M))_d(Mij), d(|M|^2)_d(Mij).
   DenseMatrix dI(2);
   Invariant2_dM(M, dI);
   const double ddet   = dI(i,j);
   const double dfnorm = 2.0 * M(i,j);

   const double det    = M.Det();
   const double det2   = det * det;
   const double fnorm  = M.FNorm();

   DenseMatrix dM(2); dM = 0.0; dM(i, j) = 1.0;
   for (int r = 0; r < 2; r++)
   {
      for (int c = 0; c < 2; c++)
      {
         dMdM(r,c) =
            (det2 *
             (2.0 * ddet * M(r,c) + 2.0 * det * dM(r,c) - dfnorm * dI(r,c))
             - 2.0 * det * ddet *
             (2.0 * det * M(r,c) - fnorm * fnorm * dI(r,c)) ) / (det2 * det2);
      }
   }
}

// Assuming 2D.
// (dI2_dM)_d(Mij) = 0.
void HyperelasticModel::Invariant2_dMdM(const DenseMatrix &M, int i, int j,
                                        DenseMatrix &dMdM)
{
   if (M.Size() == 3) { MFEM_ABORT("3D is not implemented yet."); }

   dMdM(i, j) = 0.0;
}

double InverseHarmonicModel::EvalW(const DenseMatrix &J) const
{
   Z.SetSize(J.Width());
   CalcAdjugateTranspose(J, Z);
   return 0.5*(Z*Z)/J.Det();
}

void InverseHarmonicModel::EvalP(const DenseMatrix &J, DenseMatrix &P) const
{
   int dim = J.Width();
   double t;

   Z.SetSize(dim);
   S.SetSize(dim);
   CalcAdjugateTranspose(J, Z);
   MultAAt(Z, S);
   t = 0.5*S.Trace();
   for (int i = 0; i < dim; i++)
   {
      S(i,i) -= t;
   }
   t = J.Det();
   S *= -1.0/(t*t);
   Mult(S, Z, P);
}

void InverseHarmonicModel::AssembleH(
   const DenseMatrix &J, const DenseMatrix &DS, const double weight,
   DenseMatrix &A) const
{
   int dof = DS.Height(), dim = DS.Width();
   double t;

   Z.SetSize(dim);
   S.SetSize(dim);
   G.SetSize(dof, dim);
   C.SetSize(dof, dim);

   CalcAdjugateTranspose(J, Z);
   MultAAt(Z, S);

   t = 1.0/J.Det();
   Z *= t;  // Z = J^{-t}
   S *= t;  // S = |J| (J.J^t)^{-1}
   t = 0.5*S.Trace();

   MultABt(DS, Z, G);  // G = DS.J^{-1}
   Mult(G, S, C);

   // 1.
   for (int i = 0; i < dof; i++)
      for (int j = 0; j <= i; j++)
      {
         double a = 0.0;
         for (int d = 0; d < dim; d++)
         {
            a += G(i,d)*G(j,d);
         }
         a *= weight;
         for (int k = 0; k < dim; k++)
            for (int l = 0; l <= k; l++)
            {
               double b = a*S(k,l);
               A(i+k*dof,j+l*dof) += b;
               if (i != j)
               {
                  A(j+k*dof,i+l*dof) += b;
               }
               if (k != l)
               {
                  A(i+l*dof,j+k*dof) += b;
                  if (i != j)
                  {
                     A(j+l*dof,i+k*dof) += b;
                  }
               }
            }
      }

   // 2.
   for (int i = 0; i < dof; i++)
      for (int j = 0; j < i; j++)
      {
         for (int k = 0; k < dim; k++)
            for (int l = 0; l < k; l++)
            {
               double a =
                  weight*(C(i,l)*G(j,k) - C(i,k)*G(j,l) +
                          C(j,k)*G(i,l) - C(j,l)*G(i,k) +
                          t*(G(i,k)*G(j,l) - G(i,l)*G(j,k)));

               A(i+k*dof,j+l*dof) += a;
               A(j+l*dof,i+k*dof) += a;

               A(i+l*dof,j+k*dof) -= a;
               A(j+k*dof,i+l*dof) -= a;
            }
      }
}


inline void NeoHookeanModel::EvalCoeffs() const
{
   mu = c_mu->Eval(*T, T->GetIntPoint());
   K = c_K->Eval(*T, T->GetIntPoint());
   if (c_g)
   {
      g = c_g->Eval(*T, T->GetIntPoint());
   }
}

double NeoHookeanModel::EvalW(const DenseMatrix &J) const
{
   int dim = J.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   double dJ = J.Det();
   double sJ = dJ/g;
   double bI1 = pow(dJ, -2.0/dim)*(J*J); // \bar{I}_1

   return 0.5*(mu*(bI1 - dim) + K*(sJ - 1.0)*(sJ - 1.0));
}

void NeoHookeanModel::EvalP(const DenseMatrix &J, DenseMatrix &P) const
{
   int dim = J.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   Z.SetSize(dim);
   CalcAdjugateTranspose(J, Z);

   double dJ = J.Det();
   double a  = mu*pow(dJ, -2.0/dim);
   double b  = K*(dJ/g - 1.0)/g - a*(J*J)/(dim*dJ);

   P = 0.0;
   P.Add(a, J);
   P.Add(b, Z);
}

void NeoHookeanModel::AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                                const double weight, DenseMatrix &A) const
{
   int dof = DS.Height(), dim = DS.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   Z.SetSize(dim);
   G.SetSize(dof, dim);
   C.SetSize(dof, dim);

   double dJ = J.Det();
   double sJ = dJ/g;
   double a  = mu*pow(dJ, -2.0/dim);
   double bc = a*(J*J)/dim;
   double b  = bc - K*sJ*(sJ - 1.0);
   double c  = 2.0*bc/dim + K*sJ*(2.0*sJ - 1.0);

   CalcAdjugateTranspose(J, Z);
   Z *= (1.0/dJ); // Z = J^{-t}

   MultABt(DS, J, C); // C = DS J^t
   MultABt(DS, Z, G); // G = DS J^{-1}

   a *= weight;
   b *= weight;
   c *= weight;

   // 1.
   for (int i = 0; i < dof; i++)
      for (int k = 0; k <= i; k++)
      {
         double s = 0.0;
         for (int d = 0; d < dim; d++)
         {
            s += DS(i,d)*DS(k,d);
         }
         s *= a;

         for (int d = 0; d < dim; d++)
         {
            A(i+d*dof,k+d*dof) += s;
         }

         if (k != i)
            for (int d = 0; d < dim; d++)
            {
               A(k+d*dof,i+d*dof) += s;
            }
      }

   a *= (-2.0/dim);

   // 2.
   for (int i = 0; i < dof; i++)
      for (int j = 0; j < dim; j++)
         for (int k = 0; k < dof; k++)
            for (int l = 0; l < dim; l++)
            {
               A(i+j*dof,k+l*dof) +=
                  a*(C(i,j)*G(k,l) + G(i,j)*C(k,l)) +
                  b*G(i,l)*G(k,j) + c*G(i,j)*G(k,l);
            }
}

double TMOPHyperelasticModel001::EvalW(const DenseMatrix &J) const
{
   return Invariant1(J) * Invariant2(J);
}

// Computation of dI1_dJ I1 + dI2_dJ I2.
void TMOPHyperelasticModel001::EvalP(const DenseMatrix &J, DenseMatrix &P) const
{
   Invariant1_dM(J, P);
   P *= Invariant2(J);

   DenseMatrix PP(P.Size());
   Invariant2_dM(J, PP);
   PP *= Invariant1(J);

   P += PP;
}

// Computation of d(dF_dxi)_d(xj).
void TMOPHyperelasticModel001::AssembleH(const DenseMatrix &J,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I1 = Invariant1(J), I2 = Invariant2(J);
   DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
   Invariant1_dM(J, dI1_dM);
   Invariant2_dM(J, dI2_dM);

   //   // Shorter version without using invariants.
   //   for (int i = 0; i < dof; i++)
   //   {
   //      for (int j = 0; j <= i; j++)
   //      {
   //         double a = 0.0;
   //         for (int d = 0; d < dim; d++)
   //         {
   //            a += DS(i,d)*DS(j,d);
   //         }
   //         a *= 0.5 * weight;
   //         for (int d = 0; d < dim; d++)
   //         {
   //            A(i+d*dof,j+d*dof) += a;
   //            if (i != j)
   //            {
   //               A(j+d*dof,i+d*dof) += a;
   //            }
   //         }
   //      }
   //   }

   // The first two go over the rows and cols of dG_dJ where G = dF_dxi
   for (int r = 0; r < dim; r++)
      for (int c = 0; c < dim; c++)
      {
         Invariant1_dMdM(J, r, c, dI1_dMdM);
         Invariant2_dMdM(J, r, c, dI2_dMdM);
         // Compute each entry of d(Grc)_dJ.
         for(int rr = 0; rr < dim; rr++)
         {
            for(int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                     dI1_dMdM(rr,cc) * I2 +
                     dI1_dM(r, c)    * dI2_dM(rr,cc) +
                     dI2_dMdM(rr,cc) * I1 +
                     dI2_dM(r, c)    * dI1_dM(rr,cc);

               for (int i = 0; i < dof; i++)
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
            }
         }
      }
}

double TMOPHyperelasticModel002::EvalW(const DenseMatrix &J) const
{
   double det = J.Det();
   if (det <= 0.0) { return 1e+100; }

   return 0.5 * Invariant1(J) - 1.0;
}

// TODO det(J) < 0.
void TMOPHyperelasticModel002::EvalP(const DenseMatrix &J, DenseMatrix &P) const
{
   Invariant1_dM(J, P);
   P *= 0.5;
}

// Computation of d(dF_dxi)_d(xj).
void TMOPHyperelasticModel002::AssembleH(const DenseMatrix &J,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   DenseMatrix dI1_dMdM(dim);

   // The first two go over the rows and cols of dG_dJ where G = dF_dxi
   for (int r = 0; r < dim; r++)
      for (int c = 0; c < dim; c++)
      {
         Invariant1_dMdM(J, r, c, dI1_dMdM);

         // Compute each entry of d(Grc)_dJ.
         for(int rr = 0; rr < dim; rr++)
         {
            for(int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc = 0.5 * dI1_dMdM(rr,cc);

               for (int i = 0; i < dof; i++)
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
            }
         }
      }
}

double TMOPHyperelasticModel007::EvalW(const DenseMatrix &J) const
{
   const double I2 = Invariant2(J);
   if (I2 <= 0.0) { return 1e+100; }

   return Invariant1(J) * (I2 + 1.0 / I2) - 4.0;
}

// TODO det(J) <= 0.
void TMOPHyperelasticModel007::EvalP(const DenseMatrix &J, DenseMatrix &P) const
{
   const double I1 = Invariant1(J), I2 = Invariant2(J);
   Invariant1_dM(J, P);
   P *= (I2 + 1.0 / I2);

   DenseMatrix PP(P.Size());
   Invariant2_dM(J, PP);
   PP *= I1 * (1.0 - 1.0 / (I2 * I2));

   P += PP;
}

// Computation of d(dF_dxi)_d(xj).
void TMOPHyperelasticModel007::AssembleH(const DenseMatrix &J,
                                         const DenseMatrix &DS,
                                         const double weight,
                                         DenseMatrix &A) const
{
   const int dof = DS.Height(), dim = DS.Width();
   const double I1 = Invariant1(J), I2 = Invariant2(J), iI2 = 1.0/I2;
   DenseMatrix dI1_dM(dim), dI1_dMdM(dim), dI2_dM(dim), dI2_dMdM(dim);
   Invariant1_dM(J, dI1_dM);
   Invariant2_dM(J, dI2_dM);

   // The first two go over the rows and cols of dG_dJ where G = dF_dxi.
   for (int r = 0; r < dim; r++)
      for (int c = 0; c < dim; c++)
      {
         Invariant1_dMdM(J, r, c, dI1_dMdM);
         Invariant2_dMdM(J, r, c, dI2_dMdM);
         // Compute each entry of d(Grc)_dJ.
         for(int rr = 0; rr < dim; rr++)
         {
            for(int cc = 0; cc < dim; cc++)
            {
               const double entry_rr_cc =
                     dI1_dMdM(rr,cc) * (I2 + iI2) +
                     dI1_dM(r,c) * dI2_dM(rr,cc) * (1.0 - iI2 * iI2) +
                     dI1_dM(rr,cc) * dI2_dM(r,c) * (1.0 - iI2 * iI2) +
                     I1 * ( dI2_dMdM(rr,cc) * (1.0 - iI2 * iI2) +
                            dI2_dM(r,c) * iI2 * iI2 * iI2 * dI2_dM(rr,cc));

               for (int i = 0; i < dof; i++)
                  for (int j = 0; j < dof; j++)
                  {
                     A(i+r*dof, j+rr*dof) +=
                        weight * DS(i, c) * DS(j, cc) * entry_rr_cc;
                  }
            }
         }
      }
}

void TargetJacobian::ComputeElementTargets(int e_id, const FiniteElement &fe,
                                           const IntegrationRule &ir,
                                           DenseTensor &W) const
{
   switch (target_type)
   {
   case CURRENT:
   case TARGET_MESH:
   case IDEAL_INIT_SIZE:
   {
      const GridFunction *nds;
      if (target_type == CURRENT)
      { MFEM_VERIFY(nodes, "Nodes are not set!");          nds = nodes; }
      else if (target_type == TARGET_MESH)
      { MFEM_VERIFY(tnodes, "Target nodes are not set!");  nds = tnodes; }
      else
      { MFEM_VERIFY(nodes0, "Initial nodes are not set!"); nds = nodes0; }

      const int dim = fe.GetDim(), dof = fe.GetDof();
      DenseMatrix dshape(dof, dim), pos(dof, dim);
      Array<int> xdofs(dof * dim);
      Vector posV(pos.Data(), dof * dim);

      DenseMatrix *Wideal = NULL;
      if (target_type == IDEAL_INIT_SIZE)
      {
         Wideal = new DenseMatrix(dim);
         ConstructIdealJ(fe.GetGeomType(), *Wideal);
      }

      nds->FESpace()->GetElementVDofs(e_id, xdofs);
      nds->GetSubVector(xdofs, posV);
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         fe.CalcDShape(ir.IntPoint(i), dshape);

         // W = Jac(ref->physical) for CURRENT and TARGET_MESH.
         MultAtB(pos, dshape, W(i));

         if (target_type == IDEAL_INIT_SIZE)
         {
            double det = W(i).Det();
            MFEM_VERIFY(det > 0.0, "Initial mesh is inverted!");
            W(i) = *Wideal;
            W(i) *= sqrt(det / Wideal->Det());
         }
      }
      delete Wideal;
      break;
   }
   case IDEAL:
   {
      DenseMatrix Wideal(fe.GetDim());
      ConstructIdealJ(fe.GetGeomType(), Wideal);
      for (int i = 0; i < ir.GetNPoints(); i++) { W(i) = Wideal; }
      break;
   }
   case IDEAL_EQ_SIZE:
   {
      // Average cell area.
      MFEM_VERIFY(nodes, "Nodes are not set!");
      L2_FECollection fec(0, fe.GetDim());
      FiniteElementSpace fes(nodes->FESpace()->GetMesh(), &fec);
      LinearForm lf(&fes);
      ConstantCoefficient one(1.0);
      lf.AddDomainIntegrator(new DomainLFIntegrator(one, &ir));
      lf.Assemble();
#ifdef MFEM_USE_MPI
      double area_NE[4];
      area_NE[0] = lf.Sum(); area_NE[1] = fes.GetNE();
      MPI_Allreduce(area_NE, area_NE + 2, 2, MPI_DOUBLE, MPI_SUM, comm);
      double avg_area = area_NE[2] / area_NE[3];
#else
      double avg_area = lf.Sum() / nodes->FESpace()->GetNE();
#endif

      DenseMatrix Wideal(fe.GetDim());
      ConstructIdealJ(fe.GetGeomType(), Wideal);
      Wideal *= sqrt(avg_area / Wideal.Det());
      for (int i = 0; i < ir.GetNPoints(); i++) { W(i) = Wideal; }
      break;
   }
   }
}

void TargetJacobian::ConstructIdealJ(int geom, DenseMatrix &J)
{
   switch(geom)
   {
   case Geometry::SQUARE:
   case Geometry::CUBE:
      J = 0.0;
      for (int i = 0; i < J.Size(); i++) { J(i, i) = 1.0; }
      break;
   case Geometry::TRIANGLE:
   {
      const double r3 = sqrt(3.0);
      J(0, 0) = 1.0; J(0, 1) = 0.5;
      J(1, 0) = 0.0; J(1, 1) = 0.5*r3;
      break;
   }
   case Geometry::TETRAHEDRON:
   {
      const double r3 = sqrt(3.0), r6 = sqrt(6.0);
      J(0, 0) = 1.0; J(0, 1) = 0.5;    J(0, 2) = 0.5;
      J(1, 0) = 0.0; J(1, 1) = 0.5*r3; J(1, 2) = 0.5*r3;
      J(2, 0) = 0.0; J(2, 1) = 0.0;    J(2, 2) = r6/3.0;
      break;
   }
   }
}

double HyperelasticNLFIntegrator::GetElementEnergy(const FiniteElement &el,
                                                   ElementTransformation &Tr,
                                                   const Vector &elfun)
{
   int dof = el.GetDof(), dim = el.GetDim();
   double energy;

   DSh.SetSize(dof, dim);
   J0i.SetSize(dim);
   J1.SetSize(dim);
   J.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);

   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   energy = 0.0;
   model->SetTransformation(Tr);
   DenseTensor *W = NULL;
   if (targetJ)
   {
      W = new DenseTensor(dim, dim, ir->GetNPoints());
      targetJ->ComputeElementTargets(Tr.ElementNo, el, *ir, *W);
   }

   // Limited case.
   DenseMatrix *pos0 = NULL;
   if (limited)
   {
      pos0 = new DenseMatrix(dof, dim);
      Vector pos0V(pos0->Data(), dof * dim);
      Array<int> pos_dofs;
      nodes0->FESpace()->GetElementVDofs(Tr.ElementNo, pos_dofs);
      nodes0->GetSubVector(pos_dofs, pos0V);
   }

   // Define ref->physical transformation.
   IsoparametricTransformation *rpT = NULL;
   if (coeff)
   {
      rpT = new IsoparametricTransformation;
      rpT->SetFE(&el);
      rpT->ElementNo = Tr.ElementNo;
      rpT->Attribute = Tr.Attribute;
      rpT->GetPointMat().SetSize(dim, dof);
      for (int i = 0; i < dof; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            rpT->GetPointMat()(d, i) = PMatI(i, d);
         }
      }
   }

   double weight;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      if (targetJ)
      {
         const DenseMatrix &Wi = (*W)(i);
         model->SetTargetJacobian(Wi);
         CalcInverse(Wi, J0i);
         weight = Wi.Det();
      }
      else
      {
         Tr.SetIntPoint(&ip);
         CalcInverse(Tr.Jacobian(), J0i);
         weight = Tr.Weight();
      }

      el.CalcDShape(ip, DSh);
      MultAtB(PMatI, DSh, J1);
      Mult(J1, J0i, J);

      double val = model->EvalW(J);
      if (limited)
      {
         val *= eps;
         Vector shape(dof), p(dim), p0(dim);
         el.CalcShape(ip, shape);
         PMatI.MultTranspose(shape, p);
         pos0->MultTranspose(shape, p0);
         for (int d = 0; d < dim; d++)
         {
            double diff = p(d) - p0(d);
            val += 0.5 * diff * diff;
         }
      }

      if (coeff) { weight *= coeff->Eval(*rpT, ip); }
      energy += ip.weight * weight * val;
   }
   delete rpT;
   delete pos0;
   delete W;
   return energy;
}

void HyperelasticNLFIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   Vector &elvect)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   J0i.SetSize(dim);
   J.SetSize(dim);
   P.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dof, dim);

   if(!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   elvect = 0.0;
   model->SetTransformation(Tr);
   DenseTensor *W = NULL;
   if (targetJ)
   {
      W = new DenseTensor(dim, dim, ir->GetNPoints());
      targetJ->ComputeElementTargets(Tr.ElementNo, el, *ir, *W);
   }

   // Limited case.
   DenseMatrix *pos0 = NULL;
   if (limited)
   {
      pos0 = new DenseMatrix(dof, dim);
      Vector pos0V(pos0->Data(), dof * dim);
      Array<int> pos_dofs;
      nodes0->FESpace()->GetElementVDofs(Tr.ElementNo, pos_dofs);
      nodes0->GetSubVector(pos_dofs, pos0V);
   }

   // Define ref->physical transformation.
   IsoparametricTransformation *rpT = NULL;
   if (coeff)
   {
      rpT = new IsoparametricTransformation;
      rpT->SetFE(&el);
      rpT->ElementNo = Tr.ElementNo;
      rpT->Attribute = Tr.Attribute;
      rpT->GetPointMat().SetSize(dim, dof);
      for (int i = 0; i < dof; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            rpT->GetPointMat()(d, i) = PMatI(i, d);
         }
      }
   }

   double weight;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      if (targetJ)
      {
         const DenseMatrix &Wi = (*W)(i);
         model->SetTargetJacobian(Wi);
         CalcInverse(Wi, J0i);
         weight = Wi.Det();
      }
      else
      {
         Tr.SetIntPoint(&ip);
         CalcInverse(Tr.Jacobian(), J0i);
         weight = Tr.Weight();
      }

      el.CalcDShape(ip, DSh);
      Mult(DSh, J0i, DS);
      MultAtB(PMatI, DS, J);

      model->EvalP(J, P);

      if (coeff) { weight *= coeff->Eval(*rpT, ip); }

      P *= ip.weight * weight;
      if (limited) { P *= eps; }
      AddMultABt(DS, P, PMatO);

      if (limited)
      {
         Vector shape(dof), p(dim), p0(dim);
         el.CalcShape(ip, shape);
         PMatI.MultTranspose(shape, p);
         pos0->MultTranspose(shape, p0);
         for (int d = 0; d < dim; d++)
         {
            Vector s(shape);
            s *= ip.weight * weight * (p(d) - p0(d));
            Vector tmp;
            PMatO.GetColumnReference(d, tmp);
            tmp += s;
         }
      }
   }
   delete rpT;
   delete pos0;
   delete W;
}

void HyperelasticNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   J0i.SetSize(dim);
   J.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elmat.SetSize(dof*dim);

   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   elmat = 0.0;
   DenseTensor *W = NULL;
   if (targetJ)
   {
      W = new DenseTensor(dim, dim, ir->GetNPoints());
      targetJ->ComputeElementTargets(Tr.ElementNo, el, *ir, *W);
   }

   // Define ref->physical transformation.
   IsoparametricTransformation *rpT = NULL;
   if (coeff)
   {
      rpT = new IsoparametricTransformation;
      rpT->SetFE(&el);
      rpT->ElementNo = Tr.ElementNo;
      rpT->Attribute = Tr.Attribute;
      rpT->GetPointMat().SetSize(dim, dof);
      for (int i = 0; i < dof; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            rpT->GetPointMat()(d, i) = PMatI(i, d);
         }
      }
   }

   double weight;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      if (targetJ)
      {
         const DenseMatrix &Wi = (*W)(i);
         model->SetTargetJacobian(Wi);
         CalcInverse(Wi, J0i);
         weight = Wi.Det();
      }
      else
      {
         Tr.SetIntPoint(&ip);
         CalcInverse(Tr.Jacobian(), J0i);
         weight = Tr.Weight();
      }

      el.CalcDShape(ip, DSh);
      Mult(DSh, J0i, DS);
      MultAtB(PMatI, DS, J);

      if (coeff) { weight *= coeff->Eval(*rpT, ip); }

      if (!limited)
      { model->AssembleH(J, DS, ip.weight * weight, elmat); }
      else
      {
         model->AssembleH(J, DS, eps * ip.weight * weight, elmat);
         Vector shape(dof);
         el.CalcShape(ip, shape);
         for (int i = 0; i < dof; i++)
         {
            for (int j = 0; j <= i; j++)
            {
               double a = shape(i) * shape(j) * ip.weight * weight;
               for (int d = 0; d < dim; d++)
               {
                  elmat(i+d*dof, j+d*dof) += a;
                  if (i != j) { elmat(j+d*dof, i+d*dof) += a; }
               }
            }
         }
      }
   }
   delete rpT;
   delete W;
}

HyperelasticNLFIntegrator::~HyperelasticNLFIntegrator()
{
   PMatI.ClearExternalData();
   PMatO.ClearExternalData();
   delete targetJ;
}

void InterpolateHyperElasticModel(HyperelasticModel &model,
                                  const TargetJacobian &tj,
                                  const Mesh &mesh, GridFunction &gf)
{
   const int NE = mesh.GetNE();
   const GridFunction &nodes = *mesh.GetNodes();

   for (int i = 0; i < NE; i++)
   {
      const FiniteElement &fe_pos = *nodes.FESpace()->GetFE(i);
      const IntegrationRule &ir = gf.FESpace()->GetFE(i)->GetNodes();
      const int dim = fe_pos.GetDim(), nsp = ir.GetNPoints(),
                dof = fe_pos.GetDof();

      DenseTensor W(dim, dim, nsp);
      tj.ComputeElementTargets(i, fe_pos, ir, W);

      DenseMatrix dshape(dof, dim), Winv(dim), T(dim), A(dim), pos(dof, dim);
      Array<int> pos_dofs(dof * dim), gf_dofs(nsp);
      Vector posV(pos.Data(), dof * dim);

      gf.FESpace()->GetElementDofs(i, gf_dofs);
      nodes.FESpace()->GetElementVDofs(i, pos_dofs);
      nodes.GetSubVector(pos_dofs, posV);

      for (int j = 0; j < nsp; j++)
      {
         const DenseMatrix &Wj = W(j);
         model.SetTargetJacobian(Wj);
         CalcInverse(Wj, Winv);

         const IntegrationPoint &ip = ir.IntPoint(j);
         fe_pos.CalcDShape(ip, dshape);
         MultAtB(pos, dshape, A);
         Mult(A, Winv, T);

         gf(gf_dofs[j]) = model.EvalW(T);
      }
   }
}

}
