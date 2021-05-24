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
#include <array>
#include <cctype>
#include <sstream>
#include <iostream>
#include <algorithm>

#define MFEM_DEBUG_COLOR 155

using std::ostream;
using std::string;

#include "../mfem.hpp"
#include "xfl.Y.hpp"
#include "xfl_ker.hpp"

// *****************************************************************************
// MFEM_FORALL + vectorization kernels (AutoSIMD class)
// *****************************************************************************

namespace mfem
{

namespace internal
{

/** @cond */  // Doxygen warning: documented symbol was not declared or defined

// *****************************************************************************
/// AST Kernel Setup Code Generation
// *****************************************************************************
KerSimdSetup::KerSimdSetup(const int K, xfl &ufl, Node *root,
                           std::ostringstream &out,
                           const xfl::fes &fes,
                           const xfl::fec &fec)
   : Middlend(ufl), root(root), out(out), fes(fes), fec(fec), dim(fec.dim)
{
   const int p = fec.order;
   const int node_order = 1;
   const int order_w = (node_order * fec.dim - 1);  // FunctionSpace::Qk
   const int q = 2 * p + order_w;
   const int GeomType = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   const mfem::IntegrationRule &ir = mfem::IntRules.Get(GeomType, q);
   const int Q1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
   const int D1D = p + 1;
   MFEM_VERIFY(Q1D >= D1D, "");

   out << "\n";
   out << "#include <thread>\n";
   out << "#include \"linalg/simd.hpp\"\n";
   out << "using Real = AutoSIMDTraits<double,double>::vreal_t;\n";
   out << "#define SIMD_SIZE (MFEM_SIMD_BYTES/sizeof(double))\n";

   out << "\ntemplate<int DIM, int DX0, int DX1> inline static\n";
   out << "void KSetup" << K << "(";
   out << "const int ndofs,\n\t\tconst int vdim, const int NE,\n";
   out << "\t\tconst double * __restrict__ J0,\n";
   out << "\t\tconst double * __restrict__ w,\n";
   out << "\t\tdouble * __restrict__ dx) {\n";
   out << "\tassert(vdim == 1);\n";

   out << "\n";
   out << "\tstatic constexpr int Q1D = " << Q1D << ";\n";
   out << "\tstatic constexpr int SMS = SIMD_SIZE;\n";

   // Kernel operations
   out << "\n\t// kernel operations: ";
   mfem::internal::KernelOperations ko(ufl, out);
   ufl.DfsInOrder(root, ko);
   out << "\n";

   out << "\n";
   out << "\tconst auto J = Reshape(J0, DIM,DIM, Q1D,Q1D,Q1D, NE);\n";
   out << "\tconst auto W = Reshape(w, Q1D,Q1D,Q1D);\n";

   out << "\n\tMFEM_VERIFY((NE % SIMD_SIZE) == 0, \"NE vs SIMD_SIZE error!\");\n";
   out << "\tauto DX = Reshape((Real*)dx, DX0,DX1, Q1D,Q1D,Q1D, NE/SMS);\n";

   out << "\nfor(int e = 0; e < NE; e+=SMS){\n";
}

// *****************************************************************************
KerSimdSetup::~KerSimdSetup() { out << " }\n}" << std::endl; }

// *****************************************************************************
void KerSimdSetup::token_IDENTIFIER(Token &t) const
{
   const std::string &var_name = t.Name();
   for (auto &p : ufl.ctx.var)
   {
      xfl::var &var = p.second;
      if (var.name != var_name) { continue;  }

      if (var.type == TOK::TEST_FUNCTION)
      {
         out << "    }\n";
         out << "    for (int i = 0; i < DX0; i++)\n"
             << "     for (int j = 0; j < DX1; j++)\n"
             << "      DX(i,j,qx,qy,qz,e/SMS) = vdx[j+DX0*i];\n";
         out << "   }\n  }\n }\n";
         out << " MFEM_SYNC_THREAD;\n";
      }
      var_stack.push(&var);
   }
}

// *****************************************************************************
void KerSimdSetup::rule_extra_status_rule_eval_xt(Rule &) const
{
   if (!lhs) { return;}
   assert(!var_stack.empty());
   var_stack.pop();
   ops_stack.push(TOK::EVAL_XT);
}

// *****************************************************************************
void KerSimdSetup::rule_grad_expr_grad_op_form_args(Rule &) const
{
   if (!lhs) { return; }
   assert(!var_stack.empty());
   var_stack.pop();
   ops_stack.push(TOK::GRAD_OP);
}

// *****************************************************************************
void KerSimdSetup::token_COMA(Token &) const
{
   // MFEM_FOREACH_THREAD
   out << " MFEM_FOREACH_THREAD(qz,z,Q1D){\n"
       << "  MFEM_FOREACH_THREAD(qy,y,Q1D){\n"
       << "   MFEM_FOREACH_THREAD(qx,x,Q1D){\n"
       << "    Real vdx[DX0*DX1];\n"
       << "    for (int v = 0; v < SMS; v++){\n";

   if (ops_stack.empty()) { assert(false); }
   const int op = ops_stack.top();

   out << "     const double irw = W(qx,qy,qz);\n";
   out << "     const double *Jtr = &J(0,0,qx,qy,qz, e + v);\n";
   out << "     const double detJ = kernels::Det<DIM>(Jtr);\n";
   out << "     const double wd = irw * detJ;\n";

   if (op == TOK::GRAD_OP)
   {
      out << "     double Jrt[DIM*DIM];\n";
      out << "     kernels::CalcInverse<DIM>(Jtr, Jrt);\n";
      out << "     double A[DX0*DX1];\n";
      out << "     const double D[DX0*DX1] = {wd,0,0,0,wd,0,0,0,wd};\n";
      out << "     kernels::MultABt(DIM,DIM,DIM,D,Jrt,A);\n";
   }
   else if (op == TOK::EVAL_XT) { assert(false); }
   else { assert(false); }

   ops_stack.pop();

   if (op == TOK::GRAD_OP)
   {
      out << "     double B[DX0*DX1];\n";
      out << "     kernels::Mult(DIM,DIM,DIM,A,Jrt,B);\n";
      out << "     for (int i = 0; i < DX0; i++)\n"
          << "      for (int j = 0; j < DX1; j++)\n"
          << "       vdx[j+DX0*i][v] = B[j+DX0*i];\n";

   }
   else if (op == TOK::EVAL_XT) { assert(false); }
   else { assert(false); }
}

// *****************************************************************************
/// AST Kernel Code Generation
// *****************************************************************************
KerSimdMult::KerSimdMult(const int K, xfl &ufl, Node *root,
                         std::ostringstream &out,
                         const xfl::fes &fes,
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

   out << "\ntemplate<int DIM, int DX0, int DX1> inline static\n";
   out << "void KMult" << K << "(";
   out << "const int ndofs, const int vdim, const int NE,\n";
   out << "\t\tconst double * __restrict__ B,\n";
   out << "\t\tconst double * __restrict__ G,\n";
   out << "\t\tconst int * __restrict__ map,\n";
   out << "\t\tconst double * __restrict__ dx,\n";
   out << "\t\tconst double * __restrict__ xd,\n";
   out << "\t\tdouble * __restrict__ yd) {\n\n";
   out << "\tstatic constexpr int D1D = " << D1D << ";\n";
   out << "\tstatic constexpr int MD1 = " << D1D << ";\n";
   out << "\tstatic constexpr int Q1D = " << Q1D << ";\n";
   out << "\tstatic constexpr int MQ1 = " << Q1D << ";\n";
   out << "\tstatic constexpr int SMS = SIMD_SIZE;\n";

   // Kernel operations
   out << "\n\t// kernel operations: ";
   mfem::internal::KernelOperations ko(ufl, out);
   ufl.DfsInOrder(root, ko);
   out << "\n";

   if (dim == 2) { out << "\tstatic constexpr int NBZ = 1;\n"; }
   out << "\n";

   out << "\tassert(vdim == 1);\n\n";
   out << "\tconst auto b = Reshape(B, Q1D, D1D);\n";
   out << "\tconst auto g = Reshape(G, Q1D, D1D);\n";
   if (dim == 2)
   {
      out << "\tconst auto DX = Reshape(dx, DX0,DX1, Q1D,Q1D, NE);\n";
      out << "\tconst auto MAP = Reshape(map, D1D,D1D, NE);\n";
      out << "\tconst auto XD = Reshape(xd, ndofs/*, vdim*/);\n";
      out << "\tauto YD = Reshape(yd, ndofs/*, vdim*/);\n";
   }
   else if (dim == 3)
   {
      out << "\tconst auto DX = Reshape((Real*)dx, DX0,DX1, Q1D,Q1D,Q1D, NE/SMS);\n";
      out << "\tconst auto MAP = Reshape(map, D1D,D1D,D1D, NE);\n";
      out << "\tconst auto XD = Reshape(xd, ndofs);\n";
      out << "\tauto YD = Reshape(yd, ndofs);\n";
   }
   else { assert(false); }

   // Load BG and BGt for all elements
   out << "\n";
   out << "\tdouble BG[2][MQ1*MD1];\n";
   out << "\tkernels::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);\n";
   out << "\n";
   out << "\tdouble BGt[2][MQ1*MD1];\n";
   out << "\tkernels::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BGt);\n";

   // Standard Element for loop with OpenMP parallel for loop pragma
   //out << "#define MFEM_USE_THREADS\n";
   out << "\n#ifndef MFEM_USE_THREADS\n\t//#pragma omp parallel for\n";
   out << "\tMFEM_VERIFY((NE % SIMD_SIZE) == 0, \"NE vs SIMD_SIZE error!\")\n";
   out << "\tint BATCH_SIZE = 1;\n";
   out << "\twhile((NE % BATCH_SIZE)!=0){BATCH_SIZE>>=1;}\n";
   out << "\twhile(((NE/BATCH_SIZE)%SIMD_SIZE)!=0){BATCH_SIZE>>=1;}\n";
   out << "\t//printf(\"\\n\\033[33mBATCH_SIZE:%d\\033[m\", BATCH_SIZE);\n";
   out << "\tMFEM_VERIFY((NE % BATCH_SIZE) == 0, \"NE vs BATCH_SIZE error!\")\n";
   out << "\tMFEM_VERIFY(((NE/BATCH_SIZE) % SIMD_SIZE) == 0, \"NE/BATCH_SIZE vs SIMD_SIZE error!\")\n";
   out << "\tfor(size_t eb = 0; eb < (NE/(BATCH_SIZE*SIMD_SIZE)); eb+=1) {\n";
   out << "\tfor(size_t ek = eb*BATCH_SIZE*SIMD_SIZE; ek < (eb+1)*BATCH_SIZE*SIMD_SIZE; ek+=SIMD_SIZE) {\n";
   out << "\t\tconst size_t e = ek;\n";
   //out << "\tfor(int e = 0; e < NE; e+=SIMD_SIZE) {\n";
   out << "#else\n";
   // std::threads setup: num_threads
   out << "\tstd::vector<std::thread> threads;\n";
   out << "\tstatic const unsigned int num_threads = "
       << "std::thread::hardware_concurrency();\n";
   out << "\tdbg(\"NE:%d, num_threads:%d\",NE, num_threads);\n";
   out << "\tMFEM_VERIFY((NE % num_threads) == 0, \"NE vs #Threads error\")\n";
   // std::thread for-loop: batch range setup
   out << "\tint e0 = 0;\n";
   out << "\tconst int NEB = NE / num_threads;\n";
   out << "\tint NE0 = NEB;\n";
   // std::thread outer thread for loop
   out << "\tfor(unsigned int tid = 0; tid < num_threads; ++tid) {\n";
   out << "\t\tthreads.push_back(std::thread(\n\t[&](const int tid, const int e0, const int NE0){\n";
   // std::thread inner element for-loop
   out << "\t//printf(\"[#%d] e0:%d, NE0:%d\", tid, e0, NE0);\n";
   out << "\tfor(size_t e = e0; e < NE0; e+=SIMD_SIZE) {\n";
   out << "#endif\n";

   // Element loop body
   if (dim == 2)
   {
      out << "\t\tReal XY[NBZ][MD1*MD1];\n";
      out << "\t\tkernels::LoadXD<MD1,NBZ>(e,D1D,MAP,XD,XY);\n";
   }
   else
   {
      out << "\t\tReal DDD[MD1*MD1*MD1];\n";
      out << "\t\tkernels::LoadXDGather<MD1,Real,SMS>(e,D1D,MAP,XD,DDD);\n";
   }

   // dump kernel operations
   out << "\t\t// kernel operations: ";
   mfem::internal::KernelOperations po(ufl, out);
   ufl.DfsInOrder(root, po);
   out << "\n";
}

// *****************************************************************************
KerSimdMult::~KerSimdMult()
{
   out << "\t}} // Element for loop\n";
   out << "#ifdef MFEM_USE_THREADS\n";
   out << "\t}, tid, e0, NE0)); // lambda & thread vector push_back \n";
   out << "\te0 += NEB; NE0 += NEB;\n";
   out << "\t} // Thread for loop\n";
   out << "\tfor(auto &thr : threads) { thr.join(); }\n";
   out << "#endif\n";
   out << "} // KMult1" << std::endl;
}

// *****************************************************************************
void KerSimdMult::token_IDENTIFIER(Token &t) const
{
   const std::string &var_name = t.Name();
   for (auto &p : ufl.ctx.var)
   {
      xfl::var &var = p.second;
      if (var.name != var_name) { continue; }
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
         //out << "\t\t//static constexpr const int VDIM = " << vdim << ";\n";
         ufl.ctx.vdim = vdim;  // should pass it as argument
      }

      if (var.type == TOK::TRIAL_FUNCTION)
      {
         if (dim == 2)
         {
            const char *V = vdim > 1 ? "[2]" : "";
            out << "\t\tReal DQ" << V << "[NBZ][MD1*MQ1];\n";
            out << "\t\tReal QQ" << V << "[NBZ][MQ1*MQ1];\n";
         }
         else
         {
            const char *V = vdim > 1 ? "[3]" : "";
            out << "\t\tReal sm0" << V << "[MQ1*MQ1*MQ1];\n";
            out << "\t\tReal sm1" << V << "[MQ1*MQ1*MQ1];\n";
            out << "\t\tReal (*DDQ)[MD1*MD1*MQ1] = (Real (*)[MD1*MD1*MQ1]) "
                "(sm0);\n";
            out << "\t\tReal (*DQQ)[MD1*MQ1*MQ1] = (Real (*)[MD1*MQ1*MQ1]) "
                "(sm1);\n";
            out << "\t\tReal (*QQQ)[MQ1*MQ1*MQ1] = (Real (*)[MQ1*MQ1*MQ1]) "
                "(sm0);\n";
         }
      }

      if (var.type == TOK::TEST_FUNCTION)
      {
         out << "\t\t\t}\n\t\t}\n";
         if (dim == 3) { out << "\t}\n"; }
      }
      var_stack.push(&var);
   }
}

// *****************************************************************************
void KerSimdMult::rule_extra_status_rule_eval_xt(Rule &) const
{
   dbg();
   xfl::var &var = *(var_stack.top());
   out << "\t\t// Eval(" << var.name << ")\n";
   if (lhs)
   {
      if (dim == 2)
      {
         out << "\t\tkernels::EvalX<MD1,MQ1,NBZ>(D1D,Q1D,BG[0],XY,DQ);\n";
         out << "\t\tkernels::EvalY<MD1,MQ1,NBZ>(D1D,Q1D,BG[0],DQ,QQ);\n";
      }
      else
      {
         out << "\t\tkernels::EvalX<MD1,MQ1>(D1D,Q1D,BG[0],DDD,DDQ);\n";
         out << "\t\tkernels::EvalY<MD1,MQ1>(D1D,Q1D,BG[0],DDQ,DQQ);\n";
         out << "\t\tkernels::EvalZ<MD1,MQ1>(D1D,Q1D,BG[0],DQQ,QQQ);\n";
      }
   }
   else
   {
      if (dim == 2)
      {
         out << "\t\tkernels::EvalXt<MD1,MQ1,NBZ>(D1D,Q1D,BG[0],QQ,DQ);\n";
         out << "\t\tkernels::EvalYtD<MD1,MQ1,NBZ>(D1D,Q1D,BG[0],DQ,M,YD,e);\n";
      }
      else
      {
         out << "\t\tkernels::EvalXt<MD1,MQ1>(D1D,Q1D,BG[0],QQQ,DQQ);\n";
         out << "\t\tkernels::EvalYt<MD1,MQ1>(D1D,Q1D,BG[0],DQQ,DDQ);\n";
         out << "\t\tkernels::EvalZtD<MD1,MQ1>(D1D,Q1D,BG[0],DDQ,M,YD,e);\n";
      }
   }
   var_stack.pop();
   out << "\t\t// [ pop] " << var.name << "\n";
   ops_stack.push(TOK::EVAL_XT);
}

// *****************************************************************************
void KerSimdMult::rule_grad_expr_grad_op_form_args(Rule &) const
{
   dbg();
   assert(!var_stack.empty());
   xfl::var &var = *(var_stack.top());
   out << "\t\t// Grad(" << var.name << ")\n";
   if (lhs)
   {
      if (dim == 2)
      {
         out << "\t\tkernels::Grad1X<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);\n";
         out << "\t\tkernels::Grad1Y<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);\n";
      }
      else
      {
         out << "\t\tkernels::Grad1X<MD1,MQ1>(D1D,Q1D,BG,DDD,DDQ);\n";
         out << "\t\tkernels::Grad1Y<MD1,MQ1>(D1D,Q1D,BG,DDQ,DQQ);\n";
         out << "\t\tkernels::Grad1Z<MD1,MQ1>(D1D,Q1D,BG,DQQ,QQQ);\n";
      }
   }
   else // transpose
   {
      if (dim == 2)
      {
         // 33.822
         out << "\t\t//kernels::Grad1Yt<MD1,MQ1,NBZ>(D1D,Q1D,BGt,QQ,DQ);\n";
         out << "\t\t//kernels::Grad1XtD<MD1,MQ1,NBZ>(D1D,Q1D,BGt,DQ,M,YD,e);\n";
      }
      else
      {
         out << "\t\tkernels::Grad1Zt<MD1,MQ1>(D1D,Q1D,BGt,QQQ,DQQ);\n";
         out << "\t\tkernels::Grad1Yt<MD1,MQ1>(D1D,Q1D,BGt,DQQ,DDQ);\n";
         out << "\t\tkernels::Grad1XtDScatter<MD1,MQ1,Real,SMS>(D1D,Q1D,BGt,DDQ,MAP,YD,e);\n";
      }
   }
   var_stack.pop();
   out << "\t\t// [ pop] " << var.name << "\n";
   ops_stack.push(TOK::GRAD_OP);
}

// *****************************************************************************
void KerSimdMult::token_COMA(Token &) const
{
   // MFEM_FOREACH_THREAD
   if (dim == 3) { out << "\tfor(int qz = 0; qz < Q1D; qz++){\n"; }
   out << "\t\tfor(int qy = 0; qy < Q1D; qy++){\n";
   out << "\t\t\tfor(int qx = 0; qx < Q1D; qx++){\n";

   if (ops_stack.empty()) { assert(false); }  // something is wrong

   out << "\t\t\t\tReal u[DX0], v[DX0];\n";

   const int op = ops_stack.top();
   if (op == TOK::GRAD_OP)  // Grad operation
   {
      if (dim == 2)
      {
         out << "\t\t\t\tkernels::PullGrad1<MQ1,NBZ>(qx,qy,QQ,u);\n";
      }
      else
      {
         out << "\t\t\t\tkernels::PullGrad1<MQ1>(qx,qy,qz,QQQ,u);\n";
      }
   }
   else    // Eval operation
   {
      if (dim == 2)
      {
         out << "\t\t\t\tkernels::PullEval1<MQ1,NBZ>(qx,qy,QQ,u[0]);\n";
      }
      else
      {
         out << "\t\t\t\tkernels::PullEval<MQ1>(qx,qy,qz,QQQ,u[0]);\n";
      }
   }

   ops_stack.pop();
   if (dim == 2)
   {
      out << "\t\t\t\tkernels::Mult(DX0,DX1,&DX(0,0,qx,qy,e/SMS),u,v);\n";
   }
   else
   {
      out << "\t\t\t\tkernels::Mult(DX0,DX1,&DX(0,0,qx,qy,qz,e/SMS),u,v);\n";
   }

   if (op == TOK::GRAD_OP)
   {
      if (dim == 2)
      {
         out << "\t\t\t\tkernels::PushGrad1<MQ1,NBZ>(qx,qy,v,QQ);\n";
      }
      else
      {
         out << "\t\t\t\tkernels::PushGrad1<MQ1>(qx,qy,qz,v,QQQ);\n";
      }
   }
   else
   {
      if (dim == 2)
      {
         out << "\t\t\t\tkernels::PushEval1<MQ1,NBZ>(qx,qy,v[0],QQ);\n";
      }
      else
      {
         out << "\t\t\t\tkernels::PushEval<MQ1>(qx,qy,qz,v[0],QQQ);\n";
      }
   }
}

/** @endcond */

}  // namespace internal

}  // namespace mfem
