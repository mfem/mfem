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
#include <utility>
#include <memory>
#include <numeric>
#include <fstream>
#include <iostream>
//#include <typeindex>
#include <functional>

/// ****************************************************************************
#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>

static inline std::string demangle(const char* name)
{
   int status = -1;
   std::unique_ptr<char, void(*)(void*)> res
   {
      abi::__cxa_demangle(name, NULL, NULL, &status),
      std::free
   };
   return (status==0) ? res.get() : name ;
}
#else
std::string demangle(const char* name) { return name; }
#endif // __GNUG__


#include "mfem.hpp"
#include "fem/kernels.hpp"
#define MFEM_DEBUG_COLOR 154
#include "general/debug.hpp"
#include "general/forall.hpp"
#include "linalg/kernels.hpp"

using namespace mfem;

namespace mfem
{

mfem::Mesh *CreateMeshEx7(int order);

using FE = mfem::FiniteElement;
using QI = mfem::QuadratureInterpolator;

// Kernels addons //////////////////////////////////////////////////////////////
namespace kernels
{

/// Multiply a vector with the transpose matrix.
template<typename TA, typename TX, typename TY> MFEM_HOST_DEVICE inline
void MultTranspose(const int H, const int W, TA *data, const TX *x, TY *y)
{
   double *d_col = data;
   for (int col = 0; col < W; col++)
   {
      double y_col = 0.0;
      for (int row = 0; row < H; row++)
      {
         y_col += x[row]*d_col[row];
      }
      y[col] = y_col;
      d_col += H;
   }
}

/// Multiply the transpose of a matrix A with a matrix B: At*B
template<typename TA, typename TB, typename TC> MFEM_HOST_DEVICE inline
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
template<int MD1, int MQ1, int NBZ> MFEM_HOST_DEVICE inline
void EvalXt(const int D1D, const int Q1D,
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
template<int MD1, int MQ1, int NBZ> MFEM_HOST_DEVICE inline
void EvalYt(const int D1D, const int Q1D,
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

/// Pull 2D Scalar Evaluation
template<int MQ1, int NBZ> MFEM_HOST_DEVICE inline
void PullEval1(const int qx, const int qy,
               const double sQQ[NBZ][MQ1*MQ1],
               double &P)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix QQ(sQQ[tidz], MQ1, MQ1);

   P = QQ(qx,qy);
}

/// Push 2D Scalar Evaluation
template<int MQ1, int NBZ> MFEM_HOST_DEVICE inline
void PushEval1(const int qx, const int qy,
               const double &P,
               double sQQ[NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix QQ(sQQ[tidz], MQ1, MQ1);

   QQ(qx,qy) = P;
}

/// Pull 2D Scalar Gradient
template<int MQ1, int NBZ> MFEM_HOST_DEVICE inline
void PullGrad1(const int qx, const int qy,
               const double sQQ[2][NBZ][MQ1*MQ1],
               double *A)
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix X0(sQQ[0][tidz], MQ1, MQ1);
   ConstDeviceMatrix X1(sQQ[1][tidz], MQ1, MQ1);

   A[0] = X0(qx,qy);
   A[1] = X1(qx,qy);
}

/// 2D Scalar gradient, 1/2
template<int MD1, int MQ1, int NBZ> MFEM_HOST_DEVICE inline
void Grad1X(const int D1D, const int Q1D,
            const double sBG[2][MQ1*MD1],
            const double XY[NBZ][MD1*MD1],
            double DQ[2][NBZ][MD1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sBG[0], MQ1, MD1);
   ConstDeviceMatrix G(sBG[1], MQ1, MD1);
   ConstDeviceMatrix X0(XY[tidz], MD1, MD1);
   DeviceMatrix QD0(DQ[0][tidz], MQ1, MD1);
   DeviceMatrix QD1(DQ[1][tidz], MQ1, MD1);

   MFEM_FOREACH_THREAD(dy,y,D1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u = 0.0;
         double v = 0.0;
         for (int dx = 0; dx < D1D; ++dx)
         {
            u += G(dx,qx) * X0(dx,dy);
            v += B(dx,qx) * X0(dx,dy);
         }
         QD0(qx,dy) = u;
         QD1(qx,dy) = v;
      }
   }
   MFEM_SYNC_THREAD;
}

/// 2D Scalar gradient, 2/2
template<int MD1, int MQ1, int NBZ> MFEM_HOST_DEVICE inline
void Grad1Y(const int D1D, const int Q1D,
            const double sBG[2][MQ1*MD1],
            const double QD[2][NBZ][MD1*MQ1],
            double sQQ[2][NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   ConstDeviceMatrix B(sBG[0], MQ1, MD1);
   ConstDeviceMatrix G(sBG[1], MQ1, MD1);
   ConstDeviceMatrix QD0(QD[0][tidz], MQ1, MD1);
   ConstDeviceMatrix QD1(QD[1][tidz], MQ1, MD1);
   DeviceMatrix QQ0(sQQ[0][tidz], MQ1, MQ1);
   DeviceMatrix QQ1(sQQ[1][tidz], MQ1, MQ1);

   MFEM_FOREACH_THREAD(qy,y,Q1D)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         double u = 0.0;
         double v = 0.0;
         for (int dy = 0; dy < D1D; ++dy)
         {
            u += QD0(qx,dy) * B(dy,qy);
            v += QD1(qx,dy) * G(dy,qy);
         }
         QQ0(qx,qy) = u;
         QQ1(qx,qy) = v;
      }
   }
   MFEM_SYNC_THREAD;
}

/// Push 2D Scalar Gradient
template<int MQ1, int NBZ> MFEM_HOST_DEVICE inline
void PushGrad1(const int qx, const int qy,
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
template<int MD1, int MQ1, int NBZ> MFEM_HOST_DEVICE inline
void Grad1Yt(const int D1D, const int Q1D,
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
template<int MD1, int MQ1, int NBZ> MFEM_HOST_DEVICE inline
void Grad1Xt(const int D1D, const int Q1D,
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

} // namespace kernels

// XFL addons //////////////////////////////////////////////////////////////////
namespace xfl
{

struct Func_q
{
   double *f;
   Func_q(double f[2]): f(f) { }
   operator double *() { return f; }
};

struct Test_q
{
   double *v;
   Test_q(double v[2]): v(v) { }
   operator double *() { return v; }
};

struct Trial_q
{
   double *u;
   Trial_q(double u[2]): u(u) { }
   operator double *() { return u; }
   void operator *(Test_q&) { /* nothing */ }
};

struct Cst_q
{
   void operator *(Test_q&) {}
};

template<typename T> T& grad(T &w)
{
   // 101 test segfault
   //static int loop = 0;
   //if (loop==0) { dbg("[#%d] %s", loop++, demangle(typeid(w).name())); }
   return w;
}
template<typename U, typename V> void dot(U &u, V &v) { u*v; }

FE::DerivType DerivType(const int t, const int i)
{
   const int shift = i<<2;
   const int mask = 0xF << shift;
   const int type = (t & mask) >> shift;
   if (type == 0) { return FE::NONE; }
   if (type == 1) { return FE::NONE; }
   if (type == 2) { return FE::GRAD; }
   assert(false);
}

/** ****************************************************************************
 * @brief The xfl::Operator class
 **************************************************************************** */
class Form
{
public:
   virtual int Types() const { assert(false); return 0; }
   virtual void operator ()(/*xfl::Trial_q&, xfl::Test_q&*/) const {assert(false);}
};

template<int DIM> class Operator;

template<> class Operator<2> : public mfem::Operator
{
   static constexpr int DIM = 2;
   static constexpr int NBZ = 1;

   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const int flags = GeometricFactors::JACOBIANS |
                     GeometricFactors::COORDINATES |
                     GeometricFactors::DETERMINANTS;
   const ElementDofOrdering e_ordering = ElementDofOrdering::LEXICOGRAPHIC;

   mfem::Mesh *mesh;
   const GridFunction *nodes;
   const mfem::FiniteElementSpace *nfes;
   const int p, q;
   const mfem::Operator *ER, *NR;
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
   //double u,v;
   const std::vector<Form*> &qall;
   //const QForm<R,Q,A...> &qf;
public:

   Operator(const FiniteElementSpace *fes,
            //const QForm<R,Q,A...> &qf,
            const std::vector<Form*> &qall):
      mfem::Operator(fes->GetNDofs()),
      mesh(fes->GetMesh()),
      nodes((mesh->EnsureNodes(), mesh->GetNodes())),
      nfes(nodes->FESpace()),
      p(fes->GetFE(0)->GetOrder()),
      q(2*p + mesh->GetElementTransformation(0)->OrderW()),
      ER(fes->GetElementRestriction(e_ordering)),
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
      J(mfem::Reshape(geom->J.Read(), Q1D,Q1D, SDIM,DIM,NE)),
      J0(SDIM*DIM*NQ*NE),
      xe(ND*VDIM*NE),
      ye(ND*VDIM*NE), yg(ND*VDIM*NE),
      val_xq(NQ*VDIM*NE),
      grad_xq(NQ*VDIM*DIM*NE),
      dx(VDIM*NQ*NE),
      //u(0.0), v(0.0),
      qall(qall)//,qf(qf)
   {
      MFEM_VERIFY(ER,"");
      MFEM_VERIFY(DIM == 2, "");
      MFEM_VERIFY(VDIM == 1,"");
      MFEM_VERIFY(SDIM == DIM,"");
      MFEM_VERIFY(ND == D1D*D1D, "");
      MFEM_VERIFY(NQ == Q1D*Q1D, "");
      MFEM_VERIFY(ye.Size() == ER->Height(),"");
      MFEM_VERIFY(DIM == mesh->Dimension(),"");
      qi->SetOutputLayout(QVectorLayout::byVDIM);
      nqi->SetOutputLayout(QVectorLayout::byVDIM);
      const FiniteElement *fe = nfes->GetFE(0);
      const int vdim = nfes->GetVDim();
      const int nd   = fe->GetDof();
      Vector Enodes(vdim*nd*NE);
      NR->Mult(*nodes, Enodes);
      dbg("VDIM:%d, SDIM:%d",VDIM,SDIM);
      nqi->Derivatives(Enodes, J0);
      if (SDIM==2) {ComputeDX2();}
      if (SDIM==3) {ComputeDX3();} // not supported yet
   }

   /// 2D setup to compute DX: W * detJ
   void ComputeDX2()
   {
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

   void ComputeDX3()
   {
      auto DX = mfem::Reshape(dx.Write(), Q1D, Q1D, SDIM, NE);
      MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               const double wq = W(qx,qy);
               const double J11 = J(qx,qy,0,0,e);
               const double J21 = J(qx,qy,1,0,e);
               const double J31 = J(qx,qy,2,0,e);
               const double J12 = J(qx,qy,0,1,e);
               const double J22 = J(qx,qy,1,1,e);
               const double J32 = J(qx,qy,2,1,e);
               const double E = J11*J11 + J21*J21 + J31*J31;
               const double G = J12*J12 + J22*J22 + J32*J32;
               const double F = J11*J12 + J21*J22 + J31*J32;
               const double iw = 1.0 / sqrt(E*G - F*F);
               const double alpha = wq * iw; // coeff
               DX(qx,qy,0,e) =  alpha * G; // 1,1
               DX(qx,qy,1,e) = -alpha * F; // 1,2
               DX(qx,qy,2,e) =  alpha * E; // 2,2
            }
         }
      });
   }

   /// 2D Generic Kernel
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const
   {
      ye = 0.0;
      ER->Mult(x, xe);
      for (const auto &qf : qall)
      {
         const int types = qf->Types();
         const FE::DerivType UF = DerivType(types,1);
         const FE::DerivType VF = DerivType(types,0);
         //dbg("0x%x",types);
         Mult1(UF,VF,x,y,qf);
      }
      ER->MultTranspose(ye,y);
   }

   template<typename QF>
   void Mult1(const FE::DerivType UF, const FE::DerivType VF,
              const mfem::Vector &x, mfem::Vector &y, QF *qf) const
   {
      const auto b = Reshape(maps->B.Read(), Q1D, D1D);
      const auto g = Reshape(maps->G.Read(), Q1D, D1D);
      const auto DX = Reshape(dx.Read(), Q1D, Q1D, NE);
      const auto J = Reshape(J0.Read(), DIM,DIM, Q1D,Q1D,NE);
      const auto XE = Reshape(xe.Read(), D1D, D1D, NE);

      auto YE = mfem::Reshape(ye.ReadWrite(), D1D, D1D, NE);

      constexpr const int MDQ = 8;
      constexpr const int MD1 = MDQ;
      constexpr const int MQ1 = MDQ;
      MFEM_VERIFY(MD1>=D1D,"");
      MFEM_VERIFY(MQ1>=Q1D,"");

      MFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,
      {
         MFEM_SHARED double BG[2][MQ1*MD1];
         MFEM_SHARED double XY[NBZ][MD1*MD1];
         MFEM_SHARED double DQ[2][NBZ][MD1*MQ1];
         MFEM_SHARED double QQ[2][NBZ][MQ1*MQ1];

         kernels::LoadX<MD1,NBZ>(e,D1D,XE,XY);
         kernels::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);

         if (UF == FE::NONE)
         {
            kernels::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,BG[0],XY,DQ[0]);
            kernels::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,BG[0],DQ[0],QQ[0]);
         }
         if (UF == FE::GRAD)
         {
            kernels::Grad1X<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
            kernels::Grad1Y<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);
         }

         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double dx = DX(qx,qy,e);
               const double Dx[DIM*DIM] = {dx,0,0,dx};

               double Jrt[DIM*DIM];
               double p[DIM], u[DIM], v[DIM], w[DIM];
               //xfl::Trial_q U(u); xfl::Test_q V(v);

               const double *Jtr = &J(0,0,qx,qy,e);
               kernels::CalcInverse<DIM>(Jtr, Jrt);

               if (UF == FE::NONE)
               {
                  kernels::PullEval1<MQ1,NBZ>(qx,qy,QQ[0],p[0]);
                  u[0] = p[0]; // u = p
               }
               if (UF == FE::GRAD)
               {
                  kernels::PullGrad1<MQ1,NBZ>(qx,qy,QQ,p);
                  kernels::MultTranspose(DIM,DIM,Jrt,p,u); // u = J^T.p
               }

               //qf->operator()(/*U,V*/); // apply the QFunction
               kernels::Mult(DIM,DIM,Dx,u,v); // v = Dx.u

               if (VF == FE::NONE)
               {
                  w[0] = v[0]; // w = v
                  kernels::PushEval1<MQ1,NBZ>(qx,qy,w[0],QQ[0]);
               }
               if (VF == FE::GRAD)
               {
                  kernels::Mult(DIM,DIM,Jrt,v,w); // w = J.v
                  kernels::PushGrad1<MQ1,NBZ>(qx,qy,w,QQ);
               }
            }
         }
         MFEM_SYNC_THREAD;
         kernels::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BG);
         if (VF == FE::NONE)
         {
            kernels::EvalXt<MD1,MQ1,NBZ>(D1D,Q1D,BG[0],QQ[0],DQ[0]);
            kernels::EvalYt<MD1,MQ1,NBZ>(D1D,Q1D,BG[0],DQ[0],YE,e);
         }
         if (VF == FE::GRAD)
         {
            kernels::Grad1Yt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);
            kernels::Grad1Xt<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,YE,e);
         }
      });
   }
};

/** ****************************************************************************
 * @brief The Problem struct
 ******************************************************************************/
struct Problem
{
   mfem::Operator &op;
   mfem::LinearForm &b;
   Problem(mfem::Operator &op, mfem::LinearForm &b): op(op), b(b) {}
   ~Problem() {dbg();}
};

/** ****************************************************************************
 * @brief The QForm classes
 ******************************************************************************/
class QForm: public Form
{
public:
   std::string qs;
   void (*qf)(void*);
   int dim, rank, types;
   mfem::Operator *op = nullptr;
   mfem::LinearForm *b = nullptr;
   mfem::ConstantCoefficient *constant_coeff = nullptr;
   mfem::FunctionCoefficient *function_coeff = nullptr;
   FiniteElementSpace *fes = nullptr;
   std::vector<Form*> qall = { this };

   // Capture information from arguments
   template <typename T>
   inline void CaptureFESpace(const T &arg)
   {
      FiniteElementSpace *arg_fes =
         const_cast<FiniteElementSpace*>(arg.FESpace());
      if (fes && arg_fes)
      { assert(fes->GetTrueVSize() == arg_fes->GetTrueVSize()); }
      if (!fes && arg_fes) { fes = arg_fes; }
   }

   template <typename T>
   inline void CaptureConstantCoeff(const T &arg)
   {
      if (!arg.ConstantCoeff()) { return; }
      assert(!arg.FESpace());
      dbg("ConstantCoeff!");
      assert(!constant_coeff);
      constant_coeff = arg.ConstantCoeff();
      assert(constant_coeff);
   }

   template <typename T>
   inline void CaptureFunctionCoeff(const T &arg)
   {
      if (!arg.FunctionCoeff()) { return; }
      assert(!arg.FESpace());
      dbg("FunctionCoeff!");
      assert(!function_coeff);
      function_coeff = arg.FunctionCoeff();
      assert(function_coeff);
   }

   template <typename T> // terminal case
   inline void UseArgs(const T &arg) noexcept
   {
      //dbg("%s", demangle(typeid(arg).name()));
      //std::cout << "\033[33m.\033[m ";
      if (arg.FESpace()) { rank++; }
      CaptureFESpace(arg);
      CaptureConstantCoeff(arg);
      CaptureFunctionCoeff(arg);
   }

   template<typename T, typename... Args> // capture case
   inline void UseArgs(const T &arg, Args& ...args) noexcept
   {
      //dbg("%s", demangle(typeid(arg).name()));
      //std::cout << "\033[31m.\033[m ";
      if (arg.FESpace()) { rank++; }
      CaptureFESpace(arg);
      CaptureConstantCoeff(arg);
      CaptureFunctionCoeff(arg);
      UseArgs(args...);
   }

public:
   // Constructor
   template<typename ... Args>
   QForm(const char *qs,
         void (*q)(void*), const int types, Args& ...args):
      Form(), qs(qs), qf(q), rank(0), types(types), fes(nullptr)
   {
      dbg("\033[33m%s",qs);
      UseArgs(args...);
      assert(fes);
      if (fes) { dim = fes->GetFE(0)->GetDim(); }
      dbg("rank:%d, types:0x%x", rank, types);
   }

   ~QForm() { dbg("\033[33m%s",qs); }

   // Create problem
   Problem &operator ==(QForm &rhs)
   {
      dbg();
      assert(fes);
      assert(dim == 2);

      dbg("Number of unknowns:%d",fes->GetTrueVSize());
      op = static_cast<mfem::Operator*>(new xfl::Operator<2>(fes, qall));
      assert(op);

      assert(!b);
      assert(rhs.Rank()==1);
      assert(rhs.FESpace()); // v
      mfem::LinearForm *b = new mfem::LinearForm(rhs.FESpace());
      assert(b);
      if (!rhs.ConstantCoeff() && !rhs.FunctionCoeff())
      {
         dbg("\033[31m!rhs Coeffs");
         ConstantCoefficient *cst = new ConstantCoefficient(1.0);
         b->AddDomainIntegrator(new DomainLFIntegrator(*cst));
      }
      else if (rhs.ConstantCoeff())
      {
         dbg("\033[31mrhs.ConstantCoeff()");
         ConstantCoefficient *cst = rhs.ConstantCoeff();
         b->AddDomainIntegrator(new DomainLFIntegrator(*cst));
      }
      else if (rhs.FunctionCoeff())
      {
         dbg("\033[31mrhs.FunctionCoeff()");
         FunctionCoefficient *func = rhs.FunctionCoeff();
         b->AddDomainIntegrator(new DomainLFIntegrator(*func));
      }
      else { assert(false); }

      dbg("\033[31mProblem");
      return *new Problem(*op, *b);
   }

   // + operator on QForms
   QForm &operator+(QForm &rhs)
   {
      qall.push_back(&rhs);
      return *this;
   }

   int Types() const { return types; }
   int Rank() { return rank; }
   //const int Rank() const { return rank; }

   // Call QFunction
   void operator ()() const
   {
      static int loop = 0;
      if (loop==0) { dbg("[#%d]",loop++); }
      qf(nullptr);
   }

   mfem::FiniteElementSpace *FESpace() const { return fes; }
   mfem::ConstantCoefficient *ConstantCoeff() const { return constant_coeff; }
   mfem::FunctionCoefficient *FunctionCoeff() const { return function_coeff; }
};


/** ****************************************************************************
 * @brief The Function class
 ******************************************************************************/
class Function : public GridFunction
{
public:
   Function(FiniteElementSpace *fes): GridFunction(fes) { dbg(); assert(fes); }
   void operator =(double value) { GridFunction::operator =(value); }
   int geometric_dimension() { return fes->GetMesh()->SpaceDimension(); }
   FiniteElementSpace *FESpace() { return GridFunction::FESpace(); }
   const FiniteElementSpace *FESpace() const { return GridFunction::FESpace(); }
   ConstantCoefficient *ConstantCoeff() const { return nullptr; }
   FunctionCoefficient *FunctionCoeff() const { return nullptr; }
};

/** ****************************************************************************
 * @brief The TrialFunction class
 ******************************************************************************/
class TrialFunction: public Function
{
public:
   TrialFunction(FiniteElementSpace *fes): Function(fes) { dbg(); }
   ~TrialFunction() {dbg();}
};

/** ****************************************************************************
 * @brief The TestFunction class
 ******************************************************************************/
class TestFunction: public Function
{
public:
   TestFunction(FiniteElementSpace *fes): Function(fes) { dbg(); }
   TestFunction(const TestFunction &) = default;
   ~TestFunction() {dbg();}
};

/** ****************************************************************************
 * @brief Constant
 ******************************************************************************/
class Constant
{
   const double value = 0.0;
   ConstantCoefficient *cst = nullptr;
public:
   Constant(double val): value(val), cst(new ConstantCoefficient(val))
   {
      dbg("%f", value);
   }
   ~Constant() { dbg(); /*delete cst;*/ }
   FiniteElementSpace *FESpace() const { return nullptr; }
   const double Value() const { return value; }
   double Value() { return value; }
   double operator *(TestFunction &v) { dbg(); return 0.0;}
   ConstantCoefficient *ConstantCoeff() const { return cst; }
   FunctionCoefficient *FunctionCoeff() const { return nullptr; }
};

/** ****************************************************************************
 * @brief Expressions
 ******************************************************************************/
class Expression
{
   FunctionCoefficient *fct = nullptr;
public:
   Expression(std::function<double(const Vector &)> F):
      fct(new FunctionCoefficient(F)) { dbg(); }
   ~Expression() { dbg(); /*delete fct;*/ }
   FiniteElementSpace *FESpace() const { return nullptr; } // qf args
   double operator *(TestFunction &v) { dbg(); return 0.0;} // qf eval
   ConstantCoefficient *ConstantCoeff() const { return nullptr; }
   FunctionCoefficient *FunctionCoeff() const { return fct; }
};

/** ****************************************************************************
 * @brief Mesh
 ******************************************************************************/
mfem::Mesh &Mesh(const char *mesh_file) // and & for mesh
{
   return * new mfem::Mesh(mesh_file, 1, 1);
}

mfem::Mesh &Mesh(mfem::Mesh *mesh) { return *mesh; }

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
FiniteElementCollection *FiniteElement(std::string family, int type, int p)
{
   MFEM_VERIFY(family == "Lagrange", "Unsupported family!");
   MFEM_VERIFY(type == Element::Type::QUADRILATERAL, "Unsupported type!");
   const int dim = (type == Element::Type::QUADRILATERAL) ? 2 : 0;
   return new H1_FECollection(p, dim);
}

/** ****************************************************************************
 * @brief Function Spaces
 ******************************************************************************/
class FunctionSpace: public FiniteElementSpace {};

FiniteElementSpace *FunctionSpace(mfem::Mesh *mesh, std::string family, int p)
{
   const int dim = mesh->Dimension();
   MFEM_VERIFY(family == "P", "Unsupported FE!");
   FiniteElementCollection *fec = new H1_FECollection(p, dim);
   return new FiniteElementSpace(mesh, fec);
}

FiniteElementSpace *FunctionSpace(mfem::Mesh &m, std::string f, int p)
{
   return FunctionSpace(&m, f, p);
}

FiniteElementSpace *FunctionSpace(mfem::Mesh &m, FiniteElementCollection *fec)
{
   return new FiniteElementSpace(&m, fec);
}

FiniteElementSpace *FunctionSpace(mfem::Mesh &m,
                                  FiniteElementCollection *fec,
                                  const int vdim)
{
   return new FiniteElementSpace(&m, fec, vdim);
}

/** ****************************************************************************
 * @brief Vector Function Space
 ******************************************************************************/
FiniteElementSpace *VectorFunctionSpace(mfem::Mesh *mesh,
                                        std::string family,
                                        const int p)
{
   const int dim = mesh->Dimension();
   MFEM_VERIFY(family == "P", "Unsupported FE!");
   FiniteElementCollection *fec = new H1_FECollection(p, dim);
   return new FiniteElementSpace(mesh, fec, dim);
}

FiniteElementSpace *VectorFunctionSpace(mfem::Mesh &mesh,
                                        std::string family,
                                        const int p)
{
   return VectorFunctionSpace(&mesh, family, p);
}

FiniteElementSpace *VectorFunctionSpace(mfem::Mesh &mesh,
                                        FiniteElementCollection *fec)
{
   return new FiniteElementSpace(&mesh, fec, mesh.Dimension());
}

/** ****************************************************************************
 * @brief Boundary Conditions
 ******************************************************************************/
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

/** ****************************************************************************
 * @brief Math namespace
 ******************************************************************************/
namespace math
{

Constant Pow(Function &gf, double exp)
{
   return Constant(gf.Vector::Normlp(exp));
}

double Pow(double base, double exp) { return std::pow(base, exp); }
double Pow(xfl::Func_q &base, double exp) { return std::pow(*(double*)base, exp); }

} // namespace math

/** ****************************************************************************
 * @brief solve with boundary conditions
 ******************************************************************************/
int solve(xfl::Problem &pb, xfl::Function &x, Array<int> ess_tdof_list)
{
   assert(x.FESpace());
   FiniteElementSpace *fes = x.FESpace();
   MFEM_VERIFY(UsesTensorBasis(*fes), "FE Space must Use Tensor Basis!");

   Vector B, X;
   pb.b.Assemble();
   mfem::Operator *A = nullptr;
   mfem::Operator &op = pb.op;
   op.FormLinearSystem(ess_tdof_list, x, pb.b, A, X, B);
   CG(*A, B, X, 1, 200, 1e-12, 0.0);
   op.RecoverFEMSolution(X, pb.b, x);
   x.HostReadWrite();
   return 0;
}

/// solve with empty boundary conditions
int solve(xfl::Problem &pb, xfl::Function &x)
{
   Array<int> empty_tdof_list;
   return solve(pb, x, empty_tdof_list);
}

/** ****************************************************************************
 * @brief plot the x gridfunction
 ******************************************************************************/
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

/** ****************************************************************************
 * @brief plot the mesh
 ******************************************************************************/
int plot(mfem::Mesh *mesh)
{
   char vishost[] = "localhost";
   int visport = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "mesh\n" << *mesh << std::flush;
   return 0;
}

/** ****************************************************************************
 * @brief save the x gridfunction
 ******************************************************************************/
int save(xfl::Function &x, const char *filename)
{
   std::ofstream sol_ofs(filename);
   sol_ofs.precision(8);
   x.Save(sol_ofs);
   return 0;
}

/** ****************************************************************************
 * @brief save the x gridfunction
 ******************************************************************************/
int save(mfem::Mesh &mesh, const char *filename)
{
   std::ofstream mesh_ofs(filename);
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   return 0;
}

} // namespace xfl

template<typename... Args>
void print(const char *fmt, Args... args)
{
   std::cout << std::flush;
   std::printf(fmt, args...);
   std::cout << std::endl;
}

inline bool UsesTensorBasis(const FiniteElementSpace *fes)
{
   return mfem::UsesTensorBasis(*fes);
}

int sym(int u) { return u; }
int dot(int u, int v) { return u*v; }
constexpr int quadrilateral = Element::Type::QUADRILATERAL;

// CPP addons //////////////////////////////////////////////////////////////////
namespace cpp
{

struct Range:public std::vector<int>
{
   Range(const int n):vector<int>(n)
   {
      // Fills the range with sequentially increasing values
      std::iota(std::begin(*this), std::end(*this), 0);
   }
};

struct QLambda
{
   template<typename R, typename T>
   static R lambda_ptr_exec(void *data) {return (R) (*(T*)qf<T>())(data);}

   template<typename R = void, typename S = R(*)(void*), typename T>
   static S ptr(T& t) { qf<T>(&t); return (S) lambda_ptr_exec<R,T>; }

   template<typename T> static void* qf(void* new_qf = nullptr)
   {
      static void* qf = nullptr;
      if (new_qf) { qf = new_qf; }
      return qf ;
   }
};

/// ****************************************************************************
template <typename F, int I, typename L, typename R, typename ...A>
inline F Cify(L&& l, R (*)(A...)
              noexcept(noexcept(std::declval<F>()(std::declval<A>()...))))
{
   static thread_local L l_(::std::forward<L>(l));
   static thread_local bool full;
   if (full)
   {
      l_.~L();
      new (static_cast<void*>(&l_)) L(::std::forward<L>(l));
   }
   else { full = true; }
   struct S
   {
      static R f(A... args)
      noexcept(noexcept(std::declval<F>()(std::forward<A>(args)...)))
      { return l_(::std::forward<A>(args)...); }
   };
   return &S::f;
}

template <typename F, int I = 0, typename L> F Cify(L&& l)
{ return Cify<F,I>(::std::forward<L>(l), F()); }

} // namespace cpp

void lambda_test()
{
   dbg();
   int a = 100;
   auto f = [&](void*) { return ++a; };

   {
      // QLambda
      void (*f1)(void*) = cpp::QLambda::ptr(f);
      f1(nullptr);
      printf("a:%d\n", a);  // 101

      auto f2 = cpp::QLambda::ptr(f);
      f2(nullptr);
      printf("a:%d\n", a); // 102

      //int (*f3)(void*) = cpp::QLambda::ptr<int>(f);
      auto f3 = cpp::QLambda::ptr<int>(f);
      printf("a:%d\n", f3(nullptr)); // 103

      auto g = [&](void* data) {return *(int*)(data) + a;};
      int (*f4)(void*) = cpp::QLambda::ptr<int>(g);
      int data = 5;
      printf("a+data:%d\n", f4(&data)); // 108
   }

   {
      // cify
      auto const f(cpp::Cify<void(*)()>([&a] {a++;}));
      f();
      printf("a:%d\n", a); // 102
   }
   //exit(0);
}

/*static inline void SnapNodes(Mesh &mesh)
{
   GridFunction &nodes = *mesh.GetNodes();
   Vector node(mesh.SpaceDimension());
   for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
   {
      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
         node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));
      }

      node /= node.Norml2();

      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
         nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
      }
   }
   if (mesh.Nonconforming())
   {
      // Snap hanging nodes to the master side.
      Vector tnodes;
      nodes.GetTrueDofs(tnodes);
      nodes.SetFromTrueDofs(tnodes);
   }
}

mfem::Mesh *CreateMeshEx7(int order)
{
   const int elem_type = 1;
   const int ref_levels = 2;
   const bool always_snap = false;

   int Nvert = 8, Nelem = 6;

   if (elem_type == 0)
   {
      Nvert = 6;
      Nelem = 8;
   }

   Mesh *mesh = new Mesh(2, Nvert, Nelem, 0, 3);

   // inscribed cube
   {
      const double quad_v[8][3] =
      {
         {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
         {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}
      };
      const int quad_e[6][4] =
      {
         {3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
         {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}
      };

      for (int j = 0; j < Nvert; j++)
      {
         mesh->AddVertex(quad_v[j]);
      }
      for (int j = 0; j < Nelem; j++)
      {
         int attribute = j + 1;
         mesh->AddQuad(quad_e[j], attribute);
      }
      mesh->FinalizeQuadMesh(1, 1, true);
   }

   // Set the space for the high-order mesh nodes.
   printf("\033[33m%d %d\033[m\n",mesh->Dimension(),mesh->SpaceDimension());
   fflush(0);

   H1_FECollection *fec = new H1_FECollection(order, mesh->Dimension());
   FiniteElementSpace *nodal_fes =
      new FiniteElementSpace(mesh, fec, mesh->SpaceDimension());
   mesh->SetNodalFESpace(nodal_fes);

   dbg("Refine the mesh while snapping nodes to the sphere.");
   for (int l = 0; l <= ref_levels; l++)
   {
      if (l > 0) // for l == 0 just perform snapping
      {
         mesh->UniformRefinement();
      }

      // Snap the nodes of the refined mesh back to sphere surface.
      if (always_snap || l == ref_levels)
      {
         SnapNodes(*mesh);
      }
   }
   dbg();
   assert(mesh);
   return mesh;
}*/

} // namespace mfem
