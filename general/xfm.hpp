// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#include <string>
#include <vector>
#include <numeric>
#include <fstream>
#include <iostream>
#include "mfem.hpp"
#include "general/forall.hpp"
#include "fem/kernels.hpp"
#include "fem/intrules.hpp"
#include "linalg/dtensor.hpp"
#include "linalg/kernels.hpp"

using namespace std;
using namespace mfem;


#define DBG(...) { \
    printf("\033[33m"); \
    printf(__VA_ARGS__);\
    printf("\033[m\n");\
    fflush(0);\
}

namespace mfem
{

constexpr int MDQ = 8;

using Q = mfem::QuadratureInterpolator;

constexpr Q::EvalFlags VALUES = Q::VALUES;
constexpr Q::EvalFlags GRAD = Q::DERIVATIVES;

constexpr int quadrilateral = Element::Type::QUADRILATERAL;

// CPP addons //////////////////////////////////////////////////////////////////
namespace cpp
{

struct Range:public std::vector<int>
{
   Range(const int n):vector<int>(n)
   {
      // Fills the range with sequentially increasing values
      std::iota (std::begin(*this), std::end(*this), 0);
   }
};

}

// Kernels addons //////////////////////////////////////////////////////////////
namespace kernels
{

/// Multiply the transpose of a matrix A with a matrix B:   At*B
template<typename TA, typename TB, typename TC>
MFEM_HOST_DEVICE inline
void MultAtB(const int Aheight, const int Awidth, const int Bwidth,
             const TA *Adata, const TB *Bdata, TC *AtBdata)
{
   const int ah = Aheight;
   const int aw = Awidth;
   const int bw = Bwidth;
   const double *ad = Adata;
   const double *bd = Bdata;
   double *cd = AtBdata;

   for (int j = 0; j < bw; j++)
   {
      const double *ap = ad;
      for (int i = 0; i < aw; i++)
      {
         double d = 0.0;
         for (int k = 0; k < ah; k++)
         {
            d += ap[k] * bd[k];
         }
         *(cd++) = d;
         ap += ah;
      }
      bd += ah;
   }
}

/// 2D Scalar Transposed evaluation, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalXt(const int D1D, const int Q1D,
                                    const double sB[MQ1*MD1],
                                    const double sQQ[NBZ][MQ1*MQ1],
                                    double sDQ[NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceMatrix QQ(sQQ[tidz], MQ1, MQ1);
   DeviceMatrix DQ(sDQ[tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u = 0.0;
         for (int qx = 0; qx < Q1D; ++qx)
         {
            u += QQ(qx,qy) * Bt(qx,dx);
         }
         DQ(qy,dx) = u;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar Transposed evaluation, 2/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void EvalYt(const int D1D, const int Q1D,
                                    const double sB[MQ1*MD1],
                                    const double sDQ[NBZ][MD1*MQ1],
                                    DeviceTensor<3, double> Y, const int e)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sB, MQ1, MD1);
   ConstDeviceMatrix DQ(sDQ[tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            u += Bt(qy,dy) * DQ(qy,dx);
         }
         Y(dx,dy,e) += u;
      }
   }
   MFEM_SYNC_THREAD;
}

/// Push 2D Scalar Evaluation
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PushEval(const int qx, const int qy,
                                      const double &P,
                                      double sQQ[NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix QQ(sQQ[tidz], MQ1, MQ1);

   QQ(qx,qy) = P;
}

/// Push 2D Scalar Gradient
template<int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void PushGrad1(const int qx, const int qy,
                                       const double *A,
                                       double sQQ[2][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix X0(sQQ[0][tidz], MQ1, MQ1);
   DeviceMatrix X1(sQQ[1][tidz], MQ1, MQ1);

   X0(qx,qy) = A[0];
   X1(qx,qy) = A[1];
}

/// 2D Scalar Transposed gradient, 1/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Grad1Yt(const int D1D, const int Q1D,
                                     const double sBG[2][MQ1*MD1],
                                     const double GQ[2][NBZ][MQ1*MQ1],
                                     double GD[2][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);
   ConstDeviceMatrix QQ0(GQ[0][tidz], MQ1, MQ1);
   ConstDeviceMatrix QQ1(GQ[1][tidz], MQ1, MQ1);
   DeviceMatrix DQ0(GD[0][tidz], MQ1, MD1);
   DeviceMatrix DQ1(GD[1][tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u = 0.0;
         double v = 0.0;
         for (int qx = 0; qx < Q1D; ++qx)
         {
            u += Gt(qx,dx) * QQ0(qx,qy);
            v += Bt(qx,dx) * QQ1(qx,qy);
         }
         DQ0(qy,dx) = u;
         DQ1(qy,dx) = v;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar Transposed gradient, 2/2
template<int MD1, int MQ1, int NBZ>
MFEM_HOST_DEVICE inline void Grad1Xt(const int D1D, const int Q1D,
                                     const double sBG[2][MQ1*MD1],
                                     const double GD[2][NBZ][MD1*MQ1],
                                     mfem::DeviceTensor<3, double> Y,
                                     const int e)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix Bt(sBG[0], MQ1, MD1);
   ConstDeviceMatrix Gt(sBG[1], MQ1, MD1);
   ConstDeviceMatrix DQ0(GD[0][tidz], MQ1, MD1);
   ConstDeviceMatrix DQ1(GD[1][tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(dx,x,D1D)
      {
         double u = 0.0;
         double v = 0.0;
         for (int qy = 0; qy < Q1D; ++qy)
         {
            u += DQ0(qy,dx) * Bt(qy,dy);
            v += DQ1(qy,dx) * Gt(qy,dy);
         }
         Y(dx,dy,e) += u + v;
      }
   }
   MFEM_SYNC_THREAD;
}

}

/** ****************************************************************************
 * @brief The XFLOperator class
 **************************************************************************** */
template<int DIM> class XFLOperator;

template<> class XFLOperator<2> : public mfem::Operator
{
   static constexpr int DIM = 2;
   static constexpr int NBZ = 1;

   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const int flags = GeometricFactors::JACOBIANS |
                     GeometricFactors::COORDINATES |
                     GeometricFactors::DETERMINANTS;
   const ElementDofOrdering e_ordering = ElementDofOrdering::LEXICOGRAPHIC;

   const mfem::FiniteElementSpace *fes;
   mfem::Mesh *mesh;
   const GridFunction *nodes;
   const mfem::FiniteElementSpace *nfes;
   const int p, q;
   const Operator *R, *NR;
   const Geometry::Type type;
   const IntegrationRule &ir;
   const GeometricFactors *geom;
   const DofToQuad *maps;
   const QuadratureInterpolator *qi,*nqi;
   const int SDIM, VDIM, NE, ND, NQ, D1D, Q1D;
   const DeviceTensor<DIM, const double> W;
   const DeviceTensor<DIM+3, const double> J; // 2x2 | 3x3 + NE
   Vector J0;
   mutable Vector xe, ye, yg, val_xq, grad_xq;
   Vector dx;
   const Q::EvalFlags UF, VF;

public:
   XFLOperator(const mfem::FiniteElementSpace *fes,
               const Q::EvalFlags uf, const Q::EvalFlags vf):
      Operator(fes->GetNDofs()),
      fes(fes),
      mesh(fes->GetMesh()),
      nodes((mesh->EnsureNodes(), mesh->GetNodes())),
      nfes(nodes->FESpace()),
      p(fes->GetFE(0)->GetOrder()),
      q(2*p + mesh->GetElementTransformation(0)->OrderW()),
      R(fes->GetElementRestriction(e_ordering)),
      NR(nfes->GetElementRestriction(e_ordering)),
      type(mesh->GetElementBaseGeometry(0)),
      ir(IntRules.Get(type, q)),
      geom(mesh->GetGeometricFactors(ir, flags, mode)), // will populate cache
      maps(&fes->GetFE(0)->GetDofToQuad(ir, mode)),
      qi(fes->GetQuadratureInterpolator(ir, mode)),
      nqi(nfes->GetQuadratureInterpolator(ir, mode)),
      SDIM(mesh->SpaceDimension()),
      VDIM(fes->GetVDim()),
      NE(mesh->GetNE()),
      ND(fes->GetFE(0)->GetDof()),
      NQ(ir.GetNPoints()),
      D1D(fes->GetFE(0)->GetOrder() + 1),
      Q1D(IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints()),
      W(mfem::Reshape(ir.GetWeights().Read(), Q1D, Q1D)),
      J(mfem::Reshape(geom->J.Read(), Q1D,Q1D, DIM,DIM,NE)),
      J0(DIM*DIM*NQ*NE),
      xe(ND*VDIM*NE),
      ye(ND*VDIM*NE), yg(ND*VDIM*NE),
      val_xq(NQ*VDIM*NE),
      grad_xq(NQ*VDIM*DIM*NE),
      dx(NQ*NE),
      UF(uf), VF(vf)
   {
      //DBG("\033[32mDIM:%d VDIM:%d, NE:%d, ND:%d, NQ:%d", DIM,VDIM,NE,ND,NQ);
      MFEM_VERIFY(R,"");
      MFEM_VERIFY(DIM == 2, "");
      MFEM_VERIFY(VDIM == 1,"");
      MFEM_VERIFY(ND == D1D*D1D, "");
      MFEM_VERIFY(NQ == Q1D*Q1D, "");
      MFEM_VERIFY(ye.Size() == R->Height(),"");
      MFEM_VERIFY(DIM == mesh->Dimension(),"");
      /*DBG("[XFLOperator] DIM:<%d>, p:%d, q:%d, D1D:%d, Q1D:%d, R->Height():%d",
          DIM, p, q, D1D, Q1D, R->Height());*/

      qi->SetOutputLayout(QVectorLayout::byVDIM);
      nqi->SetOutputLayout(QVectorLayout::byVDIM);

      const FiniteElement *fe = nfes->GetFE(0);
      const int vdim = nfes->GetVDim();
      const int nd   = fe->GetDof();
      Vector Enodes(vdim*nd*NE);
      NR->Mult(*nodes, Enodes);
      nqi->Derivatives(Enodes, J0);

      Assemble();
   }

   /// 2D setup for partially assembld (DX) kernels: only W*detJ
   void Assemble()
   {
      //DBG("[XFLOperator] Assemble");
      auto DX = mfem::Reshape(dx.Write(), Q1D, Q1D, NE);
      MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               const double J11 = J(qx,qy,0,0,e);
               const double J12 = J(qx,qy,1,0,e);
               const double J21 = J(qx,qy,0,1,e);
               const double J22 = J(qx,qy,1,1,e);
               const double detJ = (J11*J22)-(J21*J12);
               DX(qx,qy,e) =  W(qx,qy) * detJ; // * coeff
            }
         }
      });
   }

   /// 2D Partially assembled (DX) mass kernel
   void MassMult(const mfem::Vector &x, mfem::Vector &y) const
   {
      ye = 0.0;
      R->Mult(x, xe);
      Vector q_der, q_det;
      qi->Mult(xe, Q::VALUES, val_xq, q_der, q_det);
      const auto X = mfem::Reshape(val_xq.Read(), Q1D, Q1D, NE);

      const auto b = Reshape(maps->B.Read(), Q1D, D1D);
      const auto DX = mfem::Reshape(dx.Read(), Q1D, Q1D, NE);
      auto YE = mfem::Reshape(ye.ReadWrite(), D1D, D1D, NE);

      constexpr const int MD1 = MDQ;
      constexpr const int MQ1 = MDQ;
      MFEM_VERIFY(MD1>=D1D,"");
      MFEM_VERIFY(MQ1>=Q1D,"");

      MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
      {
         MFEM_SHARED double B[MQ1*MD1];
         MFEM_SHARED double DQ[NBZ][MD1*MQ1];
         MFEM_SHARED double QQ[NBZ][MQ1*MQ1];

         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double D = X(qx,qy,e) * DX(qx,qy,e);
               kernels::PushEval<MQ1,NBZ>(qx,qy,D,QQ);
            }
         }
         MFEM_SYNC_THREAD;
         kernels::LoadBt<MD1,MQ1>(D1D,Q1D,b,B);
         kernels::EvalXt<MD1,MQ1,NBZ>(D1D,Q1D,B,QQ,DQ);
         kernels::EvalYt<MD1,MQ1,NBZ>(D1D,Q1D,B,DQ,YE,e);
      });
      R->MultTranspose(ye,y);
   }

   /// 2D Matrix free diffusion kernel
   void DiffusionMult(const mfem::Vector &x, mfem::Vector &y) const
   {
      ye = 0.0;
      R->Mult(x, xe);
      Vector q_val, q_det;
      qi->Mult(xe, Q::DERIVATIVES, q_val, grad_xq, q_det);
      const auto GradX = mfem::Reshape(grad_xq.Read(), DIM, Q1D, Q1D, NE);

      const auto b = Reshape(maps->B.Read(), Q1D, D1D);
      const auto g = Reshape(maps->G.Read(), Q1D, D1D);
      const auto DX = mfem::Reshape(dx.Read(), Q1D, Q1D, NE);
      auto YE = mfem::Reshape(ye.ReadWrite(), D1D, D1D, NE);

      constexpr const int MD1 = MDQ;
      constexpr const int MQ1 = MDQ;
      MFEM_VERIFY(MD1>=D1D,"");
      MFEM_VERIFY(MQ1>=Q1D,"");

      MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
      {
         MFEM_SHARED double BG[2][MQ1*MD1];
         MFEM_SHARED double DQ[2][NBZ][MD1*MQ1];
         MFEM_SHARED double QQ[2][NBZ][MQ1*MQ1];

         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double J11 = J(qx,qy,0,0,e);
               const double J21 = J(qx,qy,1,0,e);
               const double J12 = J(qx,qy,0,1,e);
               const double J22 = J(qx,qy,1,1,e);
               const double detJ = (J11*J22)-(J21*J12);
               const double dp = 1.0/(detJ*detJ);
               const double dx = dp*DX(qx,qy,e);

               const double O11 =  (J12*J12 + J22*J22);
               const double O12 = -(J12*J11 + J22*J21);
               const double O22 =  (J11*J11 + J21*J21);

               const double gX = GradX(0,qx,qy,e);
               const double gY = GradX(1,qx,qy,e);

               const double D0 = (O11 * gX) + (O12 * gY);
               const double D1 = (O12 * gX) + (O22 * gY);

               const double D[2] = { D0*dx, D1*dx };
               kernels::PushGrad1<MQ1,NBZ>(qx,qy,D,QQ);
            }
         }
         MFEM_SYNC_THREAD;
         kernels::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BG);
         kernels::Grad1Yt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);
         kernels::Grad1Xt<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,YE,e);
      });
      R->MultTranspose(ye,y);
   }

   /// 2D Matrix free generic kernel
   void GenericMult(const mfem::Vector &x, mfem::Vector &y) const
   {
      ye = 0.0;
      R->Mult(x, xe);
      Vector q_val, q_der, q_det;
      const bool ud = UF == Q::DERIVATIVES;
      Vector q(ud?grad_xq.Size():val_xq.Size());
      qi->Mult(xe, UF, ud?q_val:q, ud?q:q_der, q_det);

      const auto X = mfem::Reshape(q.Read(), ud?DIM:1, Q1D,Q1D, NE);
      const auto b = Reshape(maps->B.Read(), Q1D, D1D);
      const auto g = Reshape(maps->G.Read(), Q1D, D1D);
      const auto DX = mfem::Reshape(dx.Read(), Q1D, Q1D, NE);
      const auto Jac = mfem::Reshape(J0.Read(), DIM,DIM, Q1D,Q1D,NE);

      auto YE = mfem::Reshape(ye.ReadWrite(), D1D, D1D, NE);

      constexpr const int MD1 = MDQ;
      constexpr const int MQ1 = MDQ;
      MFEM_VERIFY(MD1>=D1D,"");
      MFEM_VERIFY(MQ1>=Q1D,"");

      MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
      {
         MFEM_SHARED double BG[2][MQ1*MD1];
         MFEM_SHARED double DQ[DIM][NBZ][MD1*MQ1];
         MFEM_SHARED double QQ[DIM][NBZ][MQ1*MQ1];

         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double D[DIM];

               const bool sc = UF == Q::VALUES && VF == Q::VALUES;
               const bool gg = UF == Q::DERIVATIVES && VF == Q::DERIVATIVES;

               if (sc) { D[0] = X(0,qx,qy,e); }

               if (gg)
               {
                  double A[DIM*DIM], C[DIM*DIM];
                  const double *Jtr = &Jac(0,0,qx,qy,e);
                  const double p[DIM]= {X(0,qx,qy,e), X(1,qx,qy,e)};
                  kernels::CalcInverse<DIM>(Jtr, A);
                  kernels::MultABt(DIM,DIM,DIM,A,A,C);
                  kernels::Mult(DIM,DIM,C,p,D);
               }

               const double dx = DX(qx,qy,e);
               const double VDX[DIM] = { D[0]*dx, D[1]*dx };
               if (VF == Q::VALUES) {kernels::PushEval<MQ1,NBZ>(qx,qy,VDX[0],QQ[0]);}
               if (VF == Q::DERIVATIVES) {kernels::PushGrad1<MQ1,NBZ>(qx,qy,VDX,QQ);}
            }
         }
         MFEM_SYNC_THREAD;
         kernels::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BG);
         if (VF == Q::VALUES)
         {
            kernels::EvalXt<MD1,MQ1,NBZ>(D1D,Q1D,BG[0],QQ[0],DQ[0]);
            kernels::EvalYt<MD1,MQ1,NBZ>(D1D,Q1D,BG[0],DQ[0],YE,e);
         }
         if (VF == Q::DERIVATIVES)
         {
            kernels::Grad1Yt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);
            kernels::Grad1Xt<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,YE,e);
         }
      });
      R->MultTranspose(ye,y);
   }

   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const
   {
      const bool mass = UF == Q::VALUES && VF == Q::VALUES;
      const bool diffusion = UF == Q::DERIVATIVES && VF == Q::DERIVATIVES;
      MFEM_VERIFY(mass || diffusion, "Unknown kernel!");

      if (mass) { MassMult(x,y); }

      if (diffusion) { DiffusionMult(x,y); }

      const double alpha = y*y;

      GenericMult(x,y);

      const double beta = y*y;

      MFEM_VERIFY(std::abs(alpha - beta)<1e-12, "");
   }
};


// XFL addons //////////////////////////////////////////////////////////////////
namespace xfl
{

class TestFunction;

/**
 * @brief The Problem struct
 */
struct Problem
{
   mfem::LinearForm *b;
   Operator *ufl;
   Problem(Operator *ufl, mfem::LinearForm *b): b(b), ufl(ufl) {}
};

/** ****************************************************************************
 * @brief The Integral classes
 ******************************************************************************/
struct Form { };

struct LinearForm
{
   mfem::LinearForm *b;

   LinearForm(FiniteElementSpace *fes): b(new mfem::LinearForm(fes)) {}

   LinearForm &operator *(Form dx) { return *this;}
};

struct ScalarForm
{
   FiniteElementSpace *fes;
   mfem::ConstantCoefficient cst;

   ScalarForm(double cst, FiniteElementSpace *fes): fes(fes), cst(cst) {}

   LinearForm operator*(Form dx)
   {
      assert(fes);
      LinearForm linear_form(fes);
      linear_form.b->AddDomainIntegrator(new DomainLFIntegrator(cst));
      return linear_form;
   }
};

struct XFLForm
{
   const int dim;
   Operator *ufl;

   XFLForm(FiniteElementSpace *fes, Q::EvalFlags u, Q::EvalFlags v):
      dim(fes->GetFE(0)->GetDim()),
      ufl(dim==2 ? static_cast<Operator*>(new XFLOperator<2>(fes,u,v)) :
          //dim==3 ? static_cast<Operator*>(new XFLOperator<3>(fes,u,v)) :
          nullptr) { MFEM_VERIFY(ufl,""); }

   Problem operator ==(LinearForm &li) { return Problem(ufl, li.b); }

   // XFLForm * dx
   XFLForm &operator *(Form dx) { return *this; }

   // XFLForm + XFLForm
   XFLForm &operator +(XFLForm rhs)
   {
      assert(false);
      return *this;
   }

   XFLForm &operator -(XFLForm rhs) { return *this + rhs; /*!*/ }
};

/** ****************************************************************************
 * @brief The Function class
 ******************************************************************************/
class Function : public GridFunction
{
public:
   FiniteElementSpace *fes;
   Function(FiniteElementSpace *f): GridFunction(f), fes(f) { }
   void operator =(double value) { GridFunction::operator =(value); }
   int geometric_dimension() { return fes->GetMesh()->SpaceDimension(); }
};

/** ****************************************************************************
 * @brief The GradFunction class
 **************************************************************************** */
class GradFunction: public Function
{
public:
   GradFunction(FiniteElementSpace *fes): Function(fes) { }

   // ∇u * ∇v
   XFLForm operator*(GradFunction &v)
   {
      //DBG("\033[32m[Diffusion] XFLForm");
      XFLForm xf(fes, GRAD, GRAD);
      return xf;
   }

};

/** ****************************************************************************
 * @brief The DivFunction class
 **************************************************************************** */
class DivFunction: public Function
{
public:
   DivFunction(FiniteElementSpace *fes): Function(fes) { }
};

/** ****************************************************************************
 * @brief The TrialFunction class
 ******************************************************************************/
struct TrialFunction: public Function
{
   TrialFunction &u;
public:
   TrialFunction(FiniteElementSpace *fes): Function(fes), u(*this) { }

   GradFunction Grad() { return GradFunction(fes); }
   DivFunction Div() { return DivFunction(fes); }

   // u * v
   XFLForm operator*(TestFunction&)
   {
      XFLForm xf(fes, VALUES, VALUES);
      return xf;
   }
};

/** ****************************************************************************
 * @brief The TestFunction class
 ******************************************************************************/
class TestFunction: public Function
{
   TestFunction &v;
public:
   TestFunction(FiniteElementSpace *fes): Function(fes), v(*this) { }

   GradFunction Grad() { return GradFunction(fes); }
   DivFunction Div() { return DivFunction(fes); }

   // v * u
   XFLForm operator*(TrialFunction &u) { return u * v;}

   // f * v
   ScalarForm operator*(double alpha)
   {
      ScalarForm scalar_form(alpha, fes);
      return scalar_form;
   }
};

/** ****************************************************************************
 * @brief Constant
 ******************************************************************************/
class Constant: public ConstantCoefficient
{
public:
   Constant(double constant): ConstantCoefficient(constant) { }
   // T can be a Trial or a Test function
   template <typename T> ScalarForm operator*(T &gf) { return gf * constant; }
   double operator *(Form dx) { return constant;}
};

/** ****************************************************************************
 * @brief FunctionSpace
 ******************************************************************************/
class FunctionSpace: public FiniteElementSpace {};

/** ****************************************************************************
 * @brief Mesh
 ******************************************************************************/
mfem::Mesh &Mesh(const char *mesh_file) // and & for mesh
{
   return * new mfem::Mesh(mesh_file, 1, 1);
}

mfem::Mesh *UnitSquareMesh(int nx, int ny)
{
   Element::Type quad = Element::Type::QUADRILATERAL;
   const bool generate_edges = false, sfc_ordering = true;
   const double sx = 1.0, sy = 1.0;
   return new mfem::Mesh(nx, ny, quad, generate_edges, sx, sy, sfc_ordering);
}

mfem::Mesh *UnitHexMesh(int nx, int ny, int nz)
{
   Element::Type hex = Element::Type::HEXAHEDRON;
   const bool generate_edges = false, sfc_ordering = true;
   const double sx = 1.0, sy = 1.0, sz = 1.0;
   return new mfem::Mesh(nx, ny, nz, hex, generate_edges,
                         sx, sy, sz, sfc_ordering);

}

/** ****************************************************************************
 * @brief Device
 ******************************************************************************/
mfem::Device Device(const char *device_config) { return { device_config }; }

/** ****************************************************************************
 * @brief FiniteElement
 ******************************************************************************/
mfem::FiniteElementCollection *FiniteElement(std::string family,
                                             int type,
                                             int order)
{
   MFEM_VERIFY(family == "Lagrange", "Unsupported family!");
   MFEM_VERIFY(type == quadrilateral, "Unsupported type!");
   const int dim = 2; // quadrilateral
   return new H1_FECollection(order, dim);
}

/** ****************************************************************************
 * @brief Scalar Function Space
 ******************************************************************************/
mfem::FiniteElementSpace *FunctionSpace(mfem::Mesh *mesh, std::string family,
                                        int order)
{
   const int dim = mesh->Dimension();
   MFEM_VERIFY(family == "P", "Unsupported FE!");
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   assert(fec);
   return new FiniteElementSpace(mesh, fec);
}

mfem::FiniteElementSpace *FunctionSpace(mfem::Mesh &mesh, std::string family,
                                        int order)
{
   return FunctionSpace(&mesh, family, order);
}

mfem::FiniteElementSpace *FunctionSpace(mfem::Mesh &mesh,
                                        mfem::FiniteElementCollection *fec)
{
   assert(fec);
   return new FiniteElementSpace(&mesh, fec);
}

/** ****************************************************************************
 * @brief Vector Function Space
 ******************************************************************************/
mfem::FiniteElementSpace *VectorFunctionSpace(mfem::Mesh *mesh,
                                              std::string family,
                                              int order)
{
   const int dim = mesh->Dimension();
   MFEM_VERIFY(family == "P", "Unsupported FE!");
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   assert(fec);
   return new FiniteElementSpace(mesh, fec, dim);
}

mfem::FiniteElementSpace *VectorFunctionSpace(mfem::Mesh &mesh,
                                              std::string family,
                                              int order)
{
   return VectorFunctionSpace(&mesh, family, order);
}

mfem::FiniteElementSpace *VectorFunctionSpace(mfem::Mesh &mesh,
                                              mfem::FiniteElementCollection *fec)
{
   assert(fec);
   return new FiniteElementSpace(&mesh, fec, mesh.Dimension());
}

/** ****************************************************************************
 * @brief Boundary Conditions
 ******************************************************************************/
//Constant* Expression(string, int degree) { return new Constant(1.0); }
//int DirichletBC(FiniteElementSpace&, Constant *u, bool on_boundary) { return 0; }
Array<int> DirichletBC(FiniteElementSpace *fes)
{
   assert(fes);
   Array<int> ess_tdof_list;
   mfem::Mesh &mesh = *fes->GetMesh();
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   return ess_tdof_list;
}

/**
 * @brief Grad, Div
 */
template<typename T> GradFunction grad(T w) { return w.Grad(); }
template<typename T> DivFunction div(T w) { return w.Div(); }

/**
 * @brief Dot/Inner product
 */
template<typename T, typename U> XFLForm dot(T u, U v) { return u * v; }
template<typename G> XFLForm dot(G u, G v) { return u * v; }

/**
 * @brief Math namespace
 */
namespace math
{

Constant Pow(Function &gf, double exp)
{
   return Constant(gf.Vector::Normlp(exp));
}

double Pow(double base, double exp) { return std::pow(base, exp); }

} // namespace math

/**
 * @brief solve with boundary conditions
 */
int solve(xfl::Problem pb, xfl::Function &x, Array<int> ess_tdof_list)
{
   //DBG("[solve] x:%d, ess_tdof_list:%d", x.Size(), ess_tdof_list.Size());
   FiniteElementSpace *fes = x.FESpace();
   assert(fes);
   MFEM_VERIFY(UsesTensorBasis(*fes), "FE Space must Use Tensor Basis!");

   Vector B, X;
   mfem::LinearForm &b = *pb.b;
   b.Assemble();
   Operator *A = nullptr;
   Operator *op = pb.ufl;
   op->FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   std::cout << "Size of linear system: " << A->Height() << std::endl;
   /*Vector Md(fes->GetNDofs());
   OperatorJacobiSmoother M(Md, ess_tdof_list);
   PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);*/
   CG(*A, B, X, 1, 200, 1e-12, 0.0);
   op->RecoverFEMSolution(X, b, x);
   x.HostReadWrite();
   std::cout << "L1norm:  " << x.Norml1() << std::endl;
   std::cout << "L2norm:  " << x.Norml2() << std::endl;
   return 0;
}

/**
 * @brief plot the x gridfunction
 */
int plot(xfl::Function &x)
{
   FiniteElementSpace *fes = x.FESpace(); assert(fes);
   mfem::Mesh *mesh = fes->GetMesh(); assert(mesh);
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << *mesh << x << std::flush;
   return 0;
}

/**
 * @brief plot the mesh
 */
int plot(mfem::Mesh *mesh)
{
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "mesh\n" << *mesh << std::flush;
   return 0;
}

/**
 * @brief save the x gridfunction
 */
int save(xfl::Function &x, const char *filename)
{
   ofstream sol_ofs(filename);
   sol_ofs.precision(8);
   x.Save(sol_ofs);
   return 0;
}

/**
 * @brief save the x gridfunction
 */
int save(mfem::Mesh &mesh, const char *filename)
{
   ofstream mesh_ofs(filename);
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   return 0;
}

} // namespace xfl

template<typename... Args>
void print(const char *fmt, Args... args) { printf(fmt, args...); }

inline bool UsesTensorBasis(const FiniteElementSpace *fes)
{
   return UsesTensorBasis(*fes);
}

} // namespace mfem
