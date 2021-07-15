// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <sstream>

#define MFEM_DEBUG_COLOR 155

using std::ostream;
using std::string;

#include "../mfem.hpp"
#include "xfl_ker.hpp"
#include "xfl.Y.hpp"

namespace mfem
{

namespace internal
{

// LibParanumal - like kernels

/** @cond */  // Doxygen warning: documented symbol was not declared or defined

// *****************************************************************************
#define QUOTE(...) #__VA_ARGS__
#define RAW(...) R"delimiter(#__VA_ARGS__)delimiter"
#define UTF8(...) u8"#__VA_ARGS__"

const char * KP_CUDA_FORALL = R"delimiter(

template <typename BODY> __global__ static
void KP_CuKernel3D(const int N, BODY body)
{
   const int k = blockIdx.x;
   if (k >= N) { return; }
   body(k);
}

template <typename DBODY>
void KP_CuWrap3D(const int N,
                 const int X, const int Y, const int Z,
                 DBODY &&d_body)
{
   if (N==0) { return; }
   const int GRID = N;
   const dim3 BLCK(X,Y,Z);
   KP_CuKernel3D<<<GRID,BLCK>>>(N,d_body);
})delimiter";

// *****************************************************************************
/// AST Kernel Ceed Code Generation
// *****************************************************************************
KerLibPMult::KerLibPMult(const int K, xfl &ufl, Node *root,
                         std::ostringstream &out, const xfl::fes &fes,
                         const xfl::fec &fec)
   : Middlend(ufl), root(root), out(out), fes(fes), fec(fec), dim(fec.dim)
{
   const int p = fec.order;
   const int node_order = 1;
   const int order_w = (node_order * fec.dim - 1);
   const int q = 2 * p + order_w;
   const int GeomType = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   const mfem::IntegrationRule &ir = mfem::IntRules.Get(GeomType, q);
   const int Q1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
   const int D1D = p + 1;
   assert(Q1D >= D1D);

   out << "//\033[33m\n";
   out << KP_CUDA_FORALL << std::endl;

   out << "#define D1D " << D1D << "\n";
   out << "#define Q1D " << Q1D << "\n";

   out << "\ntemplate<int DIM, int DX0, int DX1> inline static\n";
   out << "void KMult" << K << "(";
   out << "const int ndofs, const int vdim, const int NE,\n";
   out << "\tconst double * __restrict__ B_arg, ";
   out << "const double * __restrict__ G_arg, ";
   out << "const int * __restrict__ map_arg,\n";
   out << "\tconst double * __restrict__ dx_arg, ";
   out << "const double * __restrict__ xd_arg, ";
   out << "double * __restrict__ yd_arg) {\n\n";
   if (dim == 2)
   {
      out << "\tstatic constexpr int NBZ = 1;\n";
   }

   out << "\tconst auto b = Reshape(B_arg, Q1D, D1D);\n";
   out << "\tconst auto g = Reshape(G_arg, Q1D, D1D);\n";
   if (dim == 2)
   {
      out << "\tconst auto DX = Reshape(dx_arg, DX0,DX1, Q1D,Q1D, NE);\n";
      out << "\tconst auto MAP = Reshape(map_arg, D1D,D1D, NE);\n";
      out << "\tconst auto XD = Reshape(xd_arg, ndofs);\n";
      out << "\tauto YD = Reshape(yd_arg, ndofs);\n";
   }
   else if (dim == 3)
   {
      out << "\tconst auto DX = Reshape(dx_arg, DX0,DX1, Q1D,Q1D,Q1D, NE);\n";
      out << "\tconst auto MAP = Reshape(map_arg, D1D,D1D,D1D, NE);\n";
      out << "\tconst auto XD = Reshape(xd_arg, ndofs);\n";
      out << "\tauto YD = Reshape(yd_arg, ndofs);\n";
   }
   else
   {
      assert(false);
   }

   if (dim == 2)
   {
      assert(false);
      out << "\n\tMFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,{\n";
   }
   else
   {
      out << "\n\tKP_CuWrap3D(NE, Q1D, Q1D, 1, ";
      out << "[b,g,DX,XD,MAP,YD] __device__ (const int e) {\n";
   }

   out << "\t\tMFEM_SHARED double BG[2][Q1D*D1D];\n";
   out << "\t\tdouble (*B)[D1D] = (double (*)[D1D]) (BG+0);\n";
   out << "\t\tdouble (*G)[D1D] = (double (*)[D1D]) (BG+1);\n";
   out << "\t\tdouble (*Bt)[Q1D] = (double (*)[Q1D]) (BG+0);\n";
   out << "\t\tdouble (*Gt)[Q1D] = (double (*)[Q1D]) (BG+1);\n";
   out << "\t\tkernels::LoadBG<D1D,Q1D>(D1D,Q1D,b,g,BG);\n";

   if (dim == 2)
   {
      out << "\t\tMFEM_SHARED double XY[NBZ][D1D*D1D];\n";
      out << "\t\tkernels::LoadXD<D1D,NBZ>(e,D1D,MAP,XD,XY);\n";
   }
   else
   {
      out << "\t\tMFEM_SHARED double sm0[3][Q1D*Q1D*Q1D];\n";
      out << "\t\tMFEM_SHARED double sm1[3][Q1D*Q1D*Q1D];\n";
      out << "\t\tdouble (*DDD)[D1D][D1D]  = (double (*)[D1D][D1D]) (sm0+2);\n";
      out << "\t\tdouble (*DDQ0)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+0);\n";
      out << "\t\tdouble (*DDQ1)[D1D][Q1D] = (double (*)[D1D][Q1D]) (sm0+1);\n";
      out << "\t\tdouble (*DQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+0);\n";
      out << "\t\tdouble (*DQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+1);\n";
      out << "\t\tdouble (*DQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm1+2);\n";
      out << "\t\tdouble (*QQQ0)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+0);\n";
      out << "\t\tdouble (*QQQ1)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+1);\n";
      out << "\t\tdouble (*QQQ2)[Q1D][Q1D] = (double (*)[Q1D][Q1D]) (sm0+2);\n";
      out << "\t\tdouble (*QQD0)[Q1D][D1D] = (double (*)[Q1D][D1D]) (sm1+0);\n";
      out << "\t\tdouble (*QQD1)[Q1D][D1D] = (double (*)[Q1D][D1D]) (sm1+1);\n";
      out << "\t\tdouble (*QQD2)[Q1D][D1D] = (double (*)[Q1D][D1D]) (sm1+2);\n";
      out << "\t\tdouble (*QDD0)[D1D][D1D] = (double (*)[D1D][D1D]) (sm0+0);\n";
      out << "\t\tdouble (*QDD1)[D1D][D1D] = (double (*)[D1D][D1D]) (sm0+1);\n";
      out << "\t\tdouble (*QDD2)[D1D][D1D] = (double (*)[D1D][D1D]) (sm0+2);\n";

      // out << "\t\tMFEM_SHARED double DDD[D1D*D1D*D1D];\n";
      out << "\t\tkernels::LoadXD_ijkl<D1D>(e,MAP,XD,DDD);\n";
   }
   // dump kernel operations
   out << "\t\t// kernel operations: ";
   mfem::internal::KernelOperations po(ufl, out);
   ufl.DfsInOrder(root, po);
   out << "\n";
}

// *****************************************************************************
void KerLibPMult::token_IDENTIFIER(Token &t) const
{
   const std::string &var_name = t.Name();
   for (auto &p : ufl.ctx.var)
   {
      xfl::var &var = p.second;
      if (var.name != var_name)
      {
         continue;
      }
      out << "\t\t// [push] ";
      if (var.type == TOK::TEST_FUNCTION)
      {
         out << "test ";
         lhs = false;
      }
      if (var.type == TOK::TRIAL_FUNCTION)
      {
         out << "trial ";
      }
      out << var.name << ":" << var.mode << "\n";

      // Set the VDIM for the kernel
      const int vdim = var.mode == 2 ? dim : 1;
      if (lhs)
      {
         // var.mode == 2 for GRAD op, == 1 for Eval
         // out << "\t\t//static constexpr const int VDIM = " << vdim << ";\n";
         ufl.ctx.vdim = vdim;  // should pass it as argument
      }

      if (var.type == TOK::TRIAL_FUNCTION)
      {
         if (dim == 2)
         {
            const char *V = vdim > 1 ? "[2]" : "";
            out << "\t\t//MFEM_SHARED double DQ" << V << "[NBZ][D1D*Q1D];\n";
            out << "\t\tMFEM_SHARED double QQ" << V << "[NBZ][Q1D*Q1D];\n";
         }
         else
         {
         }
      }

      if (var.type == TOK::TEST_FUNCTION)
      {
         out << "\t\t}\n\t}\n";
         if (dim == 3)
         {
            out << "\t}\n";
         }
         out << "\t\tMFEM_SYNC_THREAD;\n";
         out << "\t\tkernels::LoadBGt<D1D,Q1D>(D1D,Q1D,b,g,BG);\n";
      }
      var_stack.push(&var);
   }
}

// *****************************************************************************
void KerLibPMult::rule_extra_status_rule_eval_xt(Rule &) const
{
   dbg();
   xfl::var &var = *(var_stack.top());
   out << "\t\t// Eval(" << var.name << ")\n";
   if (lhs)
   {
      if (dim == 2)
      {
         out << "\t\tkernels::EvalX<D1D,Q1D,NBZ>(D1D,Q1D,BG[0],XY,DQ);\n";
         out << "\t\tkernels::EvalY<D1D,Q1D,NBZ>(D1D,Q1D,BG[0],DQ,QQ);\n";
      }
      else
      {
         out << "\t\tkernels::EvalX<D1D,Q1D>(D1D,Q1D,BG[0],DDD,DDQ);\n";
         out << "\t\tkernels::EvalY<D1D,Q1D>(D1D,Q1D,BG[0],DDQ,DQQ);\n";
         out << "\t\tkernels::EvalZ<D1D,Q1D>(D1D,Q1D,BG[0],DQQ,QQQ);\n";
      }
   }
   else
   {
      if (dim == 2)
      {
         out << "\t\tkernels::EvalXt<D1D,Q1D,NBZ>(D1D,Q1D,BG[0],QQ,DQ);\n";
         out << "\t\tkernels::EvalYtD<D1D,Q1D,NBZ>(D1D,Q1D,BG[0],DQ,M,YD,e);\n";
      }
      else
      {
         out << "\t\tkernels::EvalXt<D1D,Q1D>(D1D,Q1D,BG[0],QQQ,DQQ);\n";
         out << "\t\tkernels::EvalYt<D1D,Q1D>(D1D,Q1D,BG[0],DQQ,DDQ);\n";
         out << "\t\tkernels::EvalZtD<D1D,Q1D>(D1D,Q1D,BG[0],DDQ,M,YD,e);\n";
      }
   }
   var_stack.pop();
   out << "\t\t// [ pop] " << var.name << "\n";
   ops_stack.push(TOK::EVAL_XT);
}

// *****************************************************************************
void KerLibPMult::rule_grad_expr_grad_op_form_args(Rule &) const
{
   dbg();
   assert(!var_stack.empty());
   xfl::var &var = *(var_stack.top());
   out << "\t\t// Grad(" << var.name << ")\n";
   if (lhs)
   {
      if (dim == 2)
      {
         // 32.799
         out << "\t\t//kernels::Grad1X<D1D,Q1D,NBZ>(D1D,Q1D,BG,XY,DQ);\n";
         out << "\t\t//kernels::Grad1Y<D1D,Q1D,NBZ>(D1D,Q1D,BG,DQ,QQ);\n";
         // 34.137
         out << "\t\tkernels::Grad2D<D1D,Q1D,NBZ>(D1D,Q1D,BG,XY,QQ);\n";
      }
      else
      {
         const char * Grad1X_ijkl = R"Grad1X_ijkl(
         MFEM_FOREACH_THREAD(dy, y, D1D)
         {
            MFEM_FOREACH_THREAD(qx, x, Q1D)
            {
               double u[D1D], v[D1D];
#pragma unroll 6
               for (int dz = 0; dz < D1D; dz++)
               {
                  u[dz] = v[dz] = 0.0;
               }
#pragma unroll 6
               for (int dx = 0; dx < D1D; ++dx)
               {
                  const double Bx = B[qx][dx];
                  const double Gx = G[qx][dx];
#pragma unroll 6
                  for (int dz = 0; dz < D1D; ++dz)
                  {
                     const double coords = DDD[dz][dy][dx];
                     u[dz] += coords * Bx;
                     v[dz] += coords * Gx;
                  }
               }
#pragma unroll 6
               for (int dz = 0; dz < D1D; ++dz)
               {
                  DDQ0[dz][dy][qx] = u[dz];
                  DDQ1[dz][dy][qx] = v[dz];
               }
            }
         }
         MFEM_SYNC_THREAD;)Grad1X_ijkl";
         out << Grad1X_ijkl << std::endl;

         //out << "\t\t//kernels::Grad1Y_ijkl<D1D,Q1D>(B,G,DDQ0,DDQ1,DQQ0,DQQ1,DQQ2);\n";
         const char * Grad1Y_ijkl = R"Grad1Y_ijkl(
   MFEM_FOREACH_THREAD(qy, y, Q1D)
   {
      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
         double u[D1D], v[D1D], w[D1D];
#pragma unroll 6
         for (int dz = 0; dz < D1D; dz++)
         {
            u[dz] = v[dz] = w[dz] = 0.0;
         }
#pragma unroll 6
         for (int dy = 0; dy < D1D; ++dy)
         {
            const double By = B[qy][dy];
            const double Gy = G[qy][dy];
#pragma unroll 6
            for (int dz = 0; dz < D1D; dz++)
            {
               u[dz] += DDQ1[dz][dy][qx] * By;
               v[dz] += DDQ0[dz][dy][qx] * Gy;
               w[dz] += DDQ0[dz][dy][qx] * By;
            }
         }
#pragma unroll 6
         for (int dz = 0; dz < D1D; dz++)
         {
            DQQ0[dz][qy][qx] = u[dz];
            DQQ1[dz][qy][qx] = v[dz];
            DQQ2[dz][qy][qx] = w[dz];
         }
      }
   }
   MFEM_SYNC_THREAD;)Grad1Y_ijkl";
         out << Grad1Y_ijkl;

         //out << "\t\t//kernels::Grad1Z_ijkl<D1D,Q1D>(B,G,DQQ0,DQQ1,DQQ2,QQQ0,QQQ1,QQQ2);\n";
         const char *Grad1Z_ijkl = R"Grad1Z_ijkl(
   MFEM_FOREACH_THREAD(qy, y, Q1D)
   {
      MFEM_FOREACH_THREAD(qx, x, Q1D)
      {
         double u[Q1D], v[Q1D], w[Q1D];
#pragma unroll 7
         for (int qz = 0; qz < Q1D; qz++)
         {
            u[qz] = v[qz] = w[qz] = 0.0;
         }
#pragma unroll 6
         for (int dz = 0; dz < D1D; ++dz)
         {
#pragma unroll 7
            for (int qz = 0; qz < Q1D; qz++)
            {
               u[qz] += DQQ0[dz][qy][qx] * B[qz][dz];
               v[qz] += DQQ1[dz][qy][qx] * B[qz][dz];
               w[qz] += DQQ2[dz][qy][qx] * G[qz][dz];
            }
         })Grad1Z_ijkl";
         out << Grad1Z_ijkl;
      }
   }
   else
   {
      if (dim == 2)
      {
         // 33.822
         out << "\t\t//kernels::Grad1Yt<D1D,Q1D,NBZ>(D1D,Q1D,BG,QQ,DQ);\n";
         out << "\t\t//kernels::Grad1XtD<D1D,Q1D,NBZ>(D1D,Q1D,BG,DQ,M,YD,e);\n";
         // 33.182
         out << "\t\tkernels::Grad2Dt<D1D,Q1D,NBZ>(D1D,Q1D,BG,QQ,M,YD,e);\n";
      }
      else
      {
         //out << "\t\t//kernels::Grad1Zt_ijkl<D1D,Q1D>(Bt,Gt,QQQ0,QQQ1,QQQ2,QQD0,QQD1,QQD2);\n";
         const char *Grad1Zt_ijkl = R"Grad1Zt_ijkl(
   MFEM_FOREACH_THREAD(qy, y, Q1D)
   {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
         double u[Q1D], v[Q1D], w[Q1D];
#pragma unroll 7
         for (int qz = 0; qz < Q1D; ++qz)
         {
            u[qz] = v[qz] = w[qz] = 0.0;
         }
#pragma unroll 7
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const double Btx = Bt[dx][qx];
            const double Gtx = Gt[dx][qx];
#pragma unroll 7
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQQ0[qz][qy][qx] * Gtx;
               v[qz] += QQQ1[qz][qy][qx] * Btx;
               w[qz] += QQQ2[qz][qy][qx] * Btx;
            }
         }
#pragma unroll 7
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QQD0[qz][qy][dx] = u[qz];
            QQD1[qz][qy][dx] = v[qz];
            QQD2[qz][qy][dx] = w[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;)Grad1Zt_ijkl";
         out << Grad1Zt_ijkl;

         //out << "\t\t//kernels::Grad1Yt_ijkl<D1D,Q1D>(Bt,Gt,QQD0,QQD1,QQD2,QDD0,QDD1,QDD2);\n";
         const char *Grad1Yt_ijkl = R"Grad1Yt_ijkl(
   MFEM_FOREACH_THREAD(dy, y, D1D)
   {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
         double u[Q1D], v[Q1D], w[Q1D];
#pragma unroll 7
         for (int qz = 0; qz < Q1D; ++qz)
         {
            u[qz] = v[qz] = w[qz] = 0.0;
         }
#pragma unroll 7
         for (int qy = 0; qy < Q1D; ++qy)
         {
            const double Bty = Bt[dy][qy];
            const double Gty = Gt[dy][qy];
#pragma unroll 7
            for (int qz = 0; qz < Q1D; ++qz)
            {
               u[qz] += QQD0[qz][qy][dx] * Bty;
               v[qz] += QQD1[qz][qy][dx] * Gty;
               w[qz] += QQD2[qz][qy][dx] * Bty;
            }
         }
#pragma unroll 7
         for (int qz = 0; qz < Q1D; ++qz)
         {
            QDD0[qz][dy][dx] = u[qz];
            QDD1[qz][dy][dx] = v[qz];
            QDD2[qz][dy][dx] = w[qz];
         }
      }
   }
   MFEM_SYNC_THREAD;)Grad1Yt_ijkl";
         out << Grad1Yt_ijkl;

         //out << "\t\t//kernels::Grad1XtD_ijkl<D1D,Q1D>(Bt,Gt,QDD0,QDD1,QDD2,M,YD,e);\n";
         const char *Grad1XtD_ijkl = R"Grad1XtD_ijkl(
   MFEM_FOREACH_THREAD(dy, y, D1D)
   {
      MFEM_FOREACH_THREAD(dx, x, D1D)
      {
         double u[D1D], v[D1D], w[D1D];
#pragma unroll 6
         for (int dz = 0; dz < D1D; ++dz)
         {
            u[dz] = v[dz] = w[dz] = 0.0;
         }
#pragma unroll 7
         for (int qz = 0; qz < Q1D; ++qz)
         {
#pragma unroll 6
            for (int dz = 0; dz < D1D; ++dz)
            {
               u[dz] += QDD0[qz][dy][dx] * Bt[dz][qz];
               v[dz] += QDD1[qz][dy][dx] * Bt[dz][qz];
               w[dz] += QDD2[qz][dy][dx] * Gt[dz][qz];
            }
         }
#pragma unroll 6
         for (int dz = 0; dz < D1D; ++dz)
         {
            const double value = (u[dz] + v[dz] + w[dz]);
            const int gid = MAP(dx, dy, dz, e);
            const int j = gid >= 0 ? gid : -1 - gid;
            atomicAdd(&YD(j), value);
         }
      }
   }
   MFEM_SYNC_THREAD;)Grad1XtD_ijkl";
         out << Grad1XtD_ijkl;
      }
   }
   var_stack.pop();
   out << "\t\t// [ pop] " << var.name << "\n";
   ops_stack.push(TOK::GRAD_OP);
}

// *****************************************************************************
void KerLibPMult::token_COMA(Token &) const
{
   // MFEM_FOREACH_THREAD

   out << "#pragma unroll 7\n";
   out << "\tfor (int qz = 0; qz < Q1D; qz++){\n";
   out << "\t\tdouble U[DX0], V[DX0];\n";

   // a = w[j]*Dx(u[i], j)*v[i]*dx is not supported yet
   if (ops_stack.empty())
   {
      assert(false);
   }  // something is wrong


   const int op = ops_stack.top();
   if (op == TOK::GRAD_OP)  // Grad operation
   {
      if (dim == 2)
      {
         out << "\t\tkernels::PullGrad1<Q1D,NBZ>(qx,qy,QQ,u);\n";
      }
      else
      {
         out << "\t\tU[0] = u[qz]; U[1] = v[qz]; U[2] = w[qz];\n";
      }
   }
   else    // Eval operation
   {
      if (dim == 2)
      {
         out << "\t\t\t\tkernels::PullEval1<Q1D,NBZ>(qx,qy,QQ,u[0]);\n";
      }
      else
      {
         out << "\t\t\t\tkernels::PullEval<Q1D>(qx,qy,qz,QQQ,u[0]);\n";
      }
   }

   ops_stack.pop();
   if (dim == 2)
   {
      out << "\t\tkernels::Mult(DX0,DX1,&DX(0,0,qx,qy,e),u,v);\n";
   }
   else
   {
      out << "\t\tconst double O[9] = {";
      out << "DX(0,0,qx,qy,qz,e),0,0,";
      out << "0,DX(1,1,qx,qy,qz,e),0,";
      out << "0,0,DX(2,2,qx,qy,qz,e)};";
      //&DX(0,0,qx,qy,qz,e)
      out << "\t\tkernels::Mult(DX0,DX1,&O[0],U,V);\n";
   }

   if (op == TOK::GRAD_OP)
   {
      if (dim == 2)
      {
         out << "\t\t\t\tkernels::PushGrad1<Q1D,NBZ>(qx,qy,v,QQ);\n";
      }
      else
      {
         out << "\t\tkernels::PushGrad1_ijkl<Q1D>(qx,qy,qz,V,QQQ0,QQQ1,QQQ2);\n";
      }
   }
   else
   {
      if (dim == 2)
      {
         out << "\t\t\t\tkernels::PushEval1<Q1D,NBZ>(qx,qy,v[0],QQ);\n";
      }
      else
      {
         out << "\t\t\t\tkernels::PushEval<Q1D>(qx,qy,qz,v[0],QQQ);\n";
      }
   }
}

// *****************************************************************************
KerLibPMult::~KerLibPMult()
{
   out << "\t});\n}" << std::endl;
   out << "//\033[m\n";
}

/** @endcond */

}  // namespace internal

}  // namespace mfem
