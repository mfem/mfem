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
#include <functional>

#include "mfem.hpp"
#include "fem/kernels.hpp"
#include "general/debug.hpp"
#include "general/forall.hpp"
#include "linalg/kernels.hpp"

using namespace mfem;

namespace mfem
{

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

/// Push 2D Scalar Evaluation
template<int MQ1, int NBZ> MFEM_HOST_DEVICE inline
void PushEval(const int qx, const int qy,
              const double &P,
              double sQQ[NBZ][MQ1*MQ1])
{
   const int tidz = MFEM_THREAD_ID(z);
   DeviceMatrix QQ(sQQ[tidz], MQ1, MQ1);

   QQ(qx,qy) = P;
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

using Function_q = double;
using Constant_q = double;
using TestFunction_q = double;
using TrialFunction_q = double;

FE::DerivType DerivType(const int type)
{
   if (type == 0) { return FE::NONE; }
   if (type == 1) { return FE::NONE; }
   if (type == 2) { return FE::GRAD; }
   assert(false);
}

/** ****************************************************************************
 * @brief The xfl::Operator class
 **************************************************************************** */
template<typename R, typename Q, typename ... A> class QForm;
template<int DIM, typename R, typename Q, typename ... A> class Operator;

template<typename R, typename Q, typename ... A>
class Operator<2,R,Q,A...> : public mfem::Operator
{
   static constexpr int DIM = 2;
   static constexpr int NBZ = 1;

   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const int flags = GeometricFactors::JACOBIANS |
                     GeometricFactors::COORDINATES |
                     GeometricFactors::DETERMINANTS;
   const ElementDofOrdering e_ordering = ElementDofOrdering::LEXICOGRAPHIC;

   //const mfem::FiniteElementSpace *fes;
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
   double u,v; const int types;
   const FE::DerivType UF, VF;
   const QForm<R,Q,A...> &qf;
public:

   Operator(const FiniteElementSpace *fes, const QForm<R,Q,A...> &qf):
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
      J(mfem::Reshape(geom->J.Read(), Q1D,Q1D, DIM,DIM,NE)),
      J0(DIM*DIM*NQ*NE),
      xe(ND*VDIM*NE),
      ye(ND*VDIM*NE), yg(ND*VDIM*NE),
      val_xq(NQ*VDIM*NE),
      grad_xq(NQ*VDIM*DIM*NE),
      dx(NQ*NE),
      u(0.0), v(0.0),
      types(qf(u, v, false)),
      UF(DerivType(types>>4)), VF(DerivType(types&0xF)),
      qf(qf)
   {
      MFEM_VERIFY(ER,"");
      MFEM_VERIFY(DIM == 2, "");
      MFEM_VERIFY(VDIM == 1,"");
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
      nqi->Derivatives(Enodes, J0);
      ComputeDX();
      dbg("0x%x",types);
   }

   /// 2D setup to compute DX: W * detJ
   void ComputeDX()
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

   /// 2D Generic Kernel
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const
   {
      ye = 0.0;
      ER->Mult(x, xe);
      Vector q_val, q_der, q_det;
      const bool ud = UF == FE::GRAD;
      Vector q(ud?grad_xq.Size():val_xq.Size());
      qi->Mult(xe, ud?QI::DERIVATIVES:QI::VALUES, ud?q_val:q, ud?q:q_der, q_det);

      const auto X = mfem::Reshape(q.Read(), ud?DIM:1, Q1D,Q1D, NE);
      const auto b = Reshape(maps->B.Read(), Q1D, D1D);
      const auto g = Reshape(maps->G.Read(), Q1D, D1D);
      const auto DX = mfem::Reshape(dx.Read(), Q1D, Q1D, NE);
      const auto Jac = mfem::Reshape(J0.Read(), DIM,DIM, Q1D,Q1D,NE);

      auto YE = mfem::Reshape(ye.ReadWrite(), D1D, D1D, NE);

      constexpr const int MDQ = 8;
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

               const bool sc = UF == FE::NONE && VF == FE::NONE;
               const bool gg = UF == FE::GRAD && VF == FE::GRAD;

               if (sc)
               {
                  D[0] = 1.0 * X(0,qx,qy,e) * 1.0;
               }

               if (gg)
               {
                  double Jrt[DIM*DIM], C[DIM*DIM];
                  const double *Jtr = &Jac(0,0,qx,qy,e);
                  const double p[DIM]= {X(0,qx,qy,e), X(1,qx,qy,e)};
                  kernels::CalcInverse<DIM>(Jtr, Jrt);
                  // D = J.(J^T.p)
                  kernels::MultTranspose(DIM,DIM,Jrt,p,C);
                  kernels::Mult(DIM,DIM,Jrt,C,D);
               }
               const double dx = DX(qx,qy,e);
               const double VDX[DIM] = { D[0]*dx, D[1]*dx };
               if (VF == FE::NONE) {kernels::PushEval<MQ1,NBZ>(qx,qy,VDX[0],QQ[0]);}
               if (VF == FE::GRAD) {kernels::PushGrad1<MQ1,NBZ>(qx,qy,VDX,QQ);}
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
      ER->MultTranspose(ye,y);
   }
};

/** ****************************************************************************
 * @brief The Problem struct
 ******************************************************************************/
struct Problem
{
   mfem::Operator *op;
   mfem::LinearForm *b;
   Problem(mfem::Operator *op, mfem::LinearForm *b): op(op), b(b) {}
};

/** ****************************************************************************
 * @brief The QForm classes
 ******************************************************************************/
template<typename R, typename Q, typename ... A>
class QForm
{
   int dim, rank;
   FiniteElementSpace *fes = nullptr;
   mfem::Operator *op;
   mfem::LinearForm *b;

   Q qf;
   R(Q::*Apply)(A...) const;

   // Arguments
   template <typename T>
   inline void UseArgs(const T &arg) noexcept
   {
      //std::cout << "\033[33m.\033[m ";
      if (arg.FESpace()) { rank++; }
   }

   template<typename T, typename... Args>
   inline void UseArgs(const T &arg, Args... args) noexcept
   {
      //std::cout << "\033[31m.\033[m ";
      if (arg.FESpace()) { rank++; }
      if (!fes && arg.FESpace())
      { fes = const_cast<FiniteElementSpace *>(arg.FESpace()); }
      UseArgs(args...);
   }

public:
   // Constructor
   template<typename ... Args> QForm(const Q &q, Args... args):
      rank(0), fes(nullptr),
      qf(q), Apply(&decltype(qf)::operator())
   {
      UseArgs(args...);
      if (fes) { dim = fes->GetFE(0)->GetDim(); }
      dbg("rank:%d",rank);
   }

   // Create problem
   template <typename r, typename q, typename ... a>
   Problem &operator ==(QForm<r,q,a...> &rhs)
   {
      assert(fes);
      assert(dim == 2);
      assert(rhs.Rank()==1);
      mfem::LinearForm *b = new mfem::LinearForm(fes);
      mfem::ConstantCoefficient *cst = new mfem::ConstantCoefficient(1.0);
      b->AddDomainIntegrator(new DomainLFIntegrator(*cst));
      op = static_cast<mfem::Operator*>(new xfl::Operator<2,R,Q,A...>(fes,*this));
      assert(op);
      return *new Problem(op,b);
   }

   int Rank() { return rank; }
   const int Rank() const { return rank; }

   // Call QFunction
   R operator() (A ... args) const { return (qf.*Apply)(args...); }
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
 * @brief The TrialFunction class
 ******************************************************************************/
class TrialFunction: public Function
{
public:
   TrialFunction(FiniteElementSpace *fes): Function(fes) { }
};

/** ****************************************************************************
 * @brief The TestFunction class
 ******************************************************************************/
class TestFunction: public Function
{
public:
   TestFunction(FiniteElementSpace *fes): Function(fes) { }
};

/** ****************************************************************************
 * @brief Constant
 ******************************************************************************/
class Constant
{
   const double value = 0.0;
   ConstantCoefficient *cst = nullptr;
public:
   Constant(double val): value(val), cst(new ConstantCoefficient(val)) { }
   FiniteElementSpace *FESpace() const { return nullptr; }
   const double Value() const { return value; }
   double Value() { return value; }
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
FiniteElementCollection *FiniteElement(std::string family, int type, int p)
{
   MFEM_VERIFY(family == "Lagrange", "Unsupported family!");
   MFEM_VERIFY(type == Element::Type::QUADRILATERAL, "Unsupported type!");
   const int dim = (type == Element::Type::QUADRILATERAL) ? 2 : 0;
   return new H1_FECollection(p, dim);
}

/** ****************************************************************************
 * @brief Scalar Function Space
 ******************************************************************************/
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

} // namespace math

/** ****************************************************************************
 * @brief solve with boundary conditions
 ******************************************************************************/
int solve(xfl::Problem &pb, xfl::Function &x, Array<int> ess_tdof_list)
{
   FiniteElementSpace *fes = x.FESpace();
   assert(fes);
   MFEM_VERIFY(UsesTensorBasis(*fes), "FE Space must Use Tensor Basis!");

   Vector B, X;
   assert(pb.b);
   mfem::LinearForm &b = *(pb.b);
   b.Assemble();
   mfem::Operator *A = nullptr;
   mfem::Operator *op = pb.op;
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

double sym(double w) { return w; }
double grad(double w) { return w; }
double dot(double u, double v) { return u * v; }

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

} // namespace cpp

} // namespace mfem
