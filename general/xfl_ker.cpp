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


#define MFEM_DEBUG_COLOR 227

using std::ostream;
using std::string;

#include "../mfem.hpp"
#include "xfl_ker.hpp"
#include "xfl.Y.hpp"

namespace mfem
{

namespace internal
{

/** @cond */  // Doxygen warning: documented symbol was not declared or defined

// *****************************************************************************
/// AST Kernel Shapes
// *****************************************************************************
KernelShapes::KernelShapes(const int K, xfl &ufl, Node *root,
                           std::ostringstream &out, const xfl::fes &fes,
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
   assert(Q1D >= D1D);
}

// *****************************************************************************
void KernelShapes::token_IDENTIFIER(Token &t) const
{
   const std::string &var_name = t.Name();
   for (auto &p : ufl.ctx.var)
   {
      xfl::var &var = p.second;
      if (var.name != var_name)
      {
         continue;
      }
      var_stack.push(&var);
   }
}

// *****************************************************************************
void KernelShapes::rule_extra_status_rule_eval_xt(Rule &) const
{
   assert(!var_stack.empty());
   var_stack.pop();
   ops_stack.push(TOK::EVAL_XT);
   const int i = lhs ? 0 : 1;
   // out << "// eval"<<i<<"\n";
   shapes[i] = 1;
}

// *****************************************************************************
void KernelShapes::rule_grad_expr_grad_op_form_args(Rule &) const
{
   assert(!var_stack.empty());
   var_stack.pop();
   ops_stack.push(TOK::GRAD_OP);
   const int i = lhs ? 0 : 1;
   // out << "// grad "<<i<<"\n";
   shapes[i] = dim;
}

// *****************************************************************************
void KernelShapes::token_COMA(Token &) const
{
   if (ops_stack.empty())
   {
      assert(false);
   }
   ops_stack.pop();
   lhs = false;
}

// *****************************************************************************
/// AST Kernel Setup Code Generation
// *****************************************************************************
KernelSetup::KernelSetup(const int K, xfl &ufl, Node *root,
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
   assert(Q1D >= D1D);

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
   out << "\tstatic constexpr int Q1D = " << Q1D << ";\n";

   // Kernel operations
   out << "\n\t// kernel operations: ";
   mfem::internal::KernelOperations ko(ufl, out);
   ufl.DfsInOrder(root, ko);
   out << "\n";

   if (dim == 2)
   {
      out << "\tstatic constexpr int NBZ = 1;\n";
   }
   out << "\n";

   if (dim == 2)
   {
      out << "\tconst auto J = Reshape(J0, DIM,DIM, Q1D,Q1D, NE);\n";
      out << "\tconst auto W = Reshape(w, Q1D,Q1D);\n";
      out << "\tauto DX = Reshape(dx, DX0,DX1, Q1D,Q1D, NE);\n";
   }
   else if (dim == 3)
   {
      out << "\tconst auto J = Reshape(J0, DIM,DIM, Q1D,Q1D,Q1D, NE);\n";
      out << "\tconst auto W = Reshape(w, Q1D,Q1D,Q1D);\n";
      out << "\tauto DX = Reshape(dx, DX0,DX1, Q1D,Q1D,Q1D, NE);\n";
   }
   else
   {
      assert(false);
   }

   if (dim == 2)
   {
      out << "\n\tMFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,{\n";
   }
   else
   {
      out << "\n\tMFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,{\n";
   }
}

// *****************************************************************************
void KernelSetup::token_IDENTIFIER(Token &t) const
{
   const std::string &var_name = t.Name();
   for (auto &p : ufl.ctx.var)
   {
      xfl::var &var = p.second;
      if (var.name != var_name)
      {
         continue;
      }
      if (var.type == TOK::TEST_FUNCTION)
      {
         out << "\t\t\t}\n\t\t}\n";
         if (dim == 3)
         {
            out << "\t}\n";
         }
         out << "\t\tMFEM_SYNC_THREAD;\n";
      }
      var_stack.push(&var);
   }
}

// *****************************************************************************
void KernelSetup::rule_extra_status_rule_eval_xt(Rule &) const
{
   dbg();
   if (!lhs)
   {
      return;
   }
   assert(!var_stack.empty());
   var_stack.pop();
   ops_stack.push(TOK::EVAL_XT);
}

// *****************************************************************************
void KernelSetup::rule_grad_expr_grad_op_form_args(Rule &) const
{
   dbg();
   if (!lhs)
   {
      return;
   }
   assert(!var_stack.empty());
   var_stack.pop();
   ops_stack.push(TOK::GRAD_OP);
}

// *****************************************************************************
void KernelSetup::token_COMA(Token &) const
{
   // MFEM_FOREACH_THREAD
   if (dim == 3)
   {
      out << "\tMFEM_FOREACH_THREAD(qz,z,Q1D){\n";
   }
   out << "\t\tMFEM_FOREACH_THREAD(qy,y,Q1D){\n";
   out << "\t\t\tMFEM_FOREACH_THREAD(qx,x,Q1D){\n";

   if (ops_stack.empty())
   {
      assert(false);
   }
   const int op = ops_stack.top();

   out << "\t\t\t\tconst double irw = W(qx,qy" << (dim == 3 ? ",qz" : "")
       << ");\n";
   out << "\t\t\t\tconst double *Jtr = ";
   if (dim == 2)
   {
      out << "&J(0,0,qx,qy,e);\n";
   }
   else if (dim == 3)
   {
      out << "&J(0,0,qx,qy,qz,e);\n";
   }
   else
   {
      assert(false);
   }
   out << "\t\t\t\tconst double detJ = kernels::Det<DIM>(Jtr);\n";
   out << "\t\t\t\tconst double wd = irw * detJ;\n";

   if (op == TOK::GRAD_OP)
   {
      out << "\t\t\t\tdouble Jrt[DIM*DIM];\n";
      out << "\t\t\t\tkernels::CalcInverse<DIM>(Jtr, Jrt);\n";
      out << "\t\t\t\tdouble A[DX0*DX1];\n";
      out << "\t\t\t\tdouble D[DX0*DX1] = ";
      if (dim == 2)
      {
         out << "{wd,0,0,wd};\n";
      }
      if (dim == 3)
      {
         out << "{wd,0,0,0,wd,0,0,0,wd};\n";
      }
      out << "\t\t\t\tkernels::MultABt(DIM,DIM,DIM,D,Jrt,A);\n";
   }
   else if (op == TOK::EVAL_XT)
   {
      out << "\t\t\t\tdouble A[DX0*DX1] = { wd };\n";
   }
   else
   {
      assert(false);
   }

   ops_stack.pop();

   if (op == TOK::GRAD_OP)
   {
      out << "\t\t\t\tkernels::Mult(DIM,DIM,DIM,A,Jrt,";
      out << "&DX(0,0,qx,qy" << (dim == 3 ? ",qz" : "") << ",e));\n";
   }
   else if (op == TOK::EVAL_XT)
   {
      out << "\t\t\t\tDX(0,0,qx,qy" << (dim == 3 ? ",qz" : "") << ",e) = A[0];\n";
   }
   else
   {
      assert(false);
   }
}

// *****************************************************************************
KernelSetup::~KernelSetup() { out << "\t});\n}" << std::endl; }

// *****************************************************************************
/// AST kernel code generation
// *****************************************************************************
KernelMult::KernelMult(const int K, xfl &ufl, Node *root,
                       std::ostringstream &out, const xfl::fes &fes,
                       const xfl::fec &fec)
   : Middlend(ufl), root(root), out(out), fes(fes), fec(fec), dim(fec.dim)
{
   const int p = fec.order;
   // out << "\tMFEM_VERIFY("<< p <<" == p,\"\");\n";
   // q = 2*p + mesh->GetElementTransformation(0)->OrderW()
   const int node_order = 1;
   const int order_w = (node_order * fec.dim - 1);  // FunctionSpace::Qk
   // out << "\tMFEM_VERIFY("<< order_w <<" ==
   // mesh->GetElementTransformation(0)->OrderW(),\"\");\n";
   const int q = 2 * p + order_w;
   // out << "\tMFEM_VERIFY("<< q <<" == q,\"\");\n";
   const int GeomType = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   const mfem::IntegrationRule &ir = mfem::IntRules.Get(GeomType, q);
   const int Q1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
   // out << "\tMFEM_VERIFY("<< Q1D <<" == Q1D,\"\");\n";
   const int D1D = p + 1;
   // out << "\tMFEM_VERIFY("<< D1D <<" == D1D,\"\");\n";
   assert(Q1D >= D1D);

   out << "\ntemplate<int DIM, int DX0, int DX1> inline static\n";
   out << "void KMult" << K << "(";
   out << "const int ndofs, const int vdim, const int NE,\n";
   out << "\tconst double * __restrict__ B, const double * __restrict__ G, ";
   out << "const int * __restrict__ map,\n";
   out << "\tconst double * __restrict__ dx, ";
   out << "const double * __restrict__ xd, double * __restrict__ yd) {\n\n";
   out << "\tstatic constexpr int D1D = " << D1D << ";\n";
   out << "\tstatic constexpr int MD1 = " << D1D << ";\n";
   out << "\tstatic constexpr int Q1D = " << Q1D << ";\n";
   out << "\tstatic constexpr int MQ1 = " << Q1D << ";\n";

   if (dim == 2)
   {
      out << "\tstatic constexpr int NBZ = 1;\n";
   }

   out << "\tassert(vdim == 1);\n\n";
   out << "\tconst auto b = Reshape(B, Q1D, D1D);\n";
   out << "\tconst auto g = Reshape(G, Q1D, D1D);\n";
   if (dim == 2)
   {
      out << "\tconst auto DX = Reshape(dx, DX0,DX1, Q1D,Q1D, NE);\n";
      out << "\tconst auto M = Reshape(map, D1D,D1D, NE);\n";
      out << "\tconst auto XD = Reshape(xd, ndofs/*, vdim*/);\n";
      out << "\tauto YD = Reshape(yd, ndofs/*, vdim*/);\n";
   }
   else if (dim == 3)
   {
      out << "\tconst auto DX = Reshape(dx, DX0,DX1, Q1D,Q1D,Q1D, NE);\n";
      out << "\tconst auto M = Reshape(map, D1D,D1D,D1D, NE);\n";
      out << "\tconst auto XD = Reshape(xd, ndofs/*, vdim*/);\n";
      out << "\tauto YD = Reshape(yd, ndofs/*, vdim*/);\n";
   }
   else { assert(false); }

   // Element for loop
   if (dim == 2)
   {
      out << "\n\tMFEM_FORALL_2D(e, NE, Q1D, Q1D, NBZ,{\n";
   }
   else
   {
      out << "\n\tMFEM_FORALL_3D(e, NE, Q1D, Q1D, Q1D,{\n";
   }

   // B and G matrices
   out << "\t\tMFEM_SHARED double BG[2][MQ1*MD1];\n";
   out << "\t\t/*if (e==0)*/ kernels::LoadBG<MD1,MQ1>(D1D,Q1D,b,g,BG);\n";
   out << "\t\tif (e==0) for(int k=0; k<(MQ1*MD1); k++) { dbg(\"%f\",BG[0][k]);}\n";
   /*
   0.740029
   0.338259
   -0.114711
   0.0364234
   0.0864901
   0.978119
   -0.0905554
   0.0259464
   -0.125
   0.625
   0.625
   -0.125
   0.0259464
   -0.0905554
   0.978119
   0.0864901
   0.0364234
   -0.114711
   0.338259
   0.740029
   */
   out << "\t\tif (e==0) for(int k=0; k<(MQ1*MD1); k++) { dbg(\"%f\",BG[1][k]);}\n";

   // Load input X values
   if (dim == 2)
   {
      out << "\t\tMFEM_SHARED double XY[NBZ][MD1*MD1];\n";
      out << "\t\tkernels::LoadXD<MD1,NBZ>(e,D1D,M,XD,XY);\n";
   }
   else
   {
      out << "\t\tMFEM_SHARED double DDD[MD1*MD1*MD1];\n";
      out << "\t\tkernels::LoadXD<MD1>(e,D1D,M,XD,DDD);\n";
   }
   out << "\t\tif (e==0) for(int k=0; k<(MD1*MD1*MD1); k++) { dbg(\"%f\",DDD[k]);}\n";
   // 0.072338 x8

   // dump kernel operations
   out << "\t\t// kernel operations: ";
   mfem::internal::KernelOperations po(ufl, out);
   ufl.DfsInOrder(root, po);
   out << "\n";
}

// *****************************************************************************
void KernelMult::token_IDENTIFIER(Token &t) const
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
         out << "\t\t//static constexpr const int VDIM = " << vdim << ";\n";
         ufl.ctx.vdim = vdim;  // should pass it as argument
      }

      if (var.type == TOK::TRIAL_FUNCTION)
      {
         if (dim == 2)
         {
            const char *V = vdim > 1 ? "[2]" : "";
            out << "\t\tMFEM_SHARED double DQ" << V << "[NBZ][MD1*MQ1];\n";
            out << "\t\tMFEM_SHARED double QQ" << V << "[NBZ][MQ1*MQ1];\n";
         }
         else
         {
            const char *V = vdim > 1 ? "[3]" : "";
            out << "\t\tMFEM_SHARED double sm0" << V << "[MQ1*MQ1*MQ1];\n";
            out << "\t\tMFEM_SHARED double sm1" << V << "[MQ1*MQ1*MQ1];\n";
            out << "\t\tdouble (*DDQ)[MD1*MD1*MQ1] = (double (*)[MD1*MD1*MQ1]) "
                "(sm0);\n";
            out << "\t\tdouble (*DQQ)[MD1*MQ1*MQ1] = (double (*)[MD1*MQ1*MQ1]) "
                "(sm1);\n";
            out << "\t\tdouble (*QQQ)[MQ1*MQ1*MQ1] = (double (*)[MQ1*MQ1*MQ1]) "
                "(sm0);\n";
         }
      }

      if (var.type == TOK::TEST_FUNCTION)
      {
         out << "\t\t\t}\n\t\t}\n";
         if (dim == 3)
         {
            out << "\t}\n";
         }
         out << "\t\tMFEM_SYNC_THREAD;\n";
         out << "\t\t/*if (e==0)*/ kernels::LoadBGt<MD1,MQ1>(D1D,Q1D,b,g,BG);\n";
      }
      var_stack.push(&var);
   }
}

// *****************************************************************************
void KernelMult::rule_extra_status_rule_eval_xt(Rule &) const
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
void KernelMult::rule_grad_expr_grad_op_form_args(Rule &) const
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
         out << "\t\t//kernels::Grad1X<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);\n";
         out << "\t\t//kernels::Grad1Y<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);\n";
         // 34.137
         out << "\t\tkernels::Grad2D<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,QQ);\n";
      }
      else
      {
         out << "\t\tkernels::Grad1X<MD1,MQ1>(D1D,Q1D,BG,DDD,DDQ);\n";
         out << "\t\tkernels::Grad1Y<MD1,MQ1>(D1D,Q1D,BG,DDQ,DQQ);\n";
         out << "\t\tkernels::Grad1Z<MD1,MQ1>(D1D,Q1D,BG,DQQ,QQQ);\n";
      }
   }
   else
   {
      if (dim == 2)
      {
         // 33.822
         out << "\t\t//kernels::Grad1Yt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,DQ);\n";
         out << "\t\t//kernels::Grad1XtD<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,M,YD,e);\n";
         // 33.182
         out << "\t\tkernels::Grad2Dt<MD1,MQ1,NBZ>(D1D,Q1D,BG,QQ,M,YD,e);\n";
      }
      else
      {
         out << "\t\tkernels::Grad1Zt<MD1,MQ1>(D1D,Q1D,BG,QQQ,DQQ);\n";
         out << "\t\tkernels::Grad1Yt<MD1,MQ1>(D1D,Q1D,BG,DQQ,DDQ);\n";
         out << "\t\tkernels::Grad1XtD<MD1,MQ1>(D1D,Q1D,BG,DDQ,M,YD,e);\n";
      }
   }
   var_stack.pop();
   out << "\t\t// [ pop] " << var.name << "\n";
   ops_stack.push(TOK::GRAD_OP);
}

// *****************************************************************************
void KernelMult::token_COMA(Token &) const
{
   // MFEM_FOREACH_THREAD
   if (dim == 3)
   {
      out << "\tMFEM_FOREACH_THREAD(qz,z,Q1D){\n";
   }
   out << "\t\tMFEM_FOREACH_THREAD(qy,y,Q1D){\n";
   out << "\t\t\tMFEM_FOREACH_THREAD(qx,x,Q1D){\n";

   // a = w[j]*Dx(u[i], j)*v[i]*dx is not supported yet
   if (ops_stack.empty())
   {
      assert(false);
   }  // something is wrong

   out << "\t\t\t\tdouble u[DX0], v[DX0];\n";

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
      out << "\t\t\t\tkernels::Mult(DX0,DX1,&DX(0,0,qx,qy,e),u,v);\n";
   }
   else
   {
      out << "\t\t\t\tkernels::Mult(DX0,DX1,&DX(0,0,qx,qy,qz,e),u,v);\n";
   }
   out << "\t\tconst bool dump = e==0 && qx==0 && qy==0 && qz==0;\n";
   out << "\t\tif (dump) { dbg(\"u: %f %f %f\",u[0],u[1],u[2]);}\n";
   out << "\t\tif (dump) { dbg(\"v: %f %f %f\",v[0],v[1],v[2]);}\n";
   // u: 0.0163791 0.0163791 0.0163791
   // v: 2.72298e-05 2.72298e-05 2.72298e-05


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

// *****************************************************************************
KernelMult::~KernelMult() { out << "\t});\n}" << std::endl; }

// *****************************************************************************
void KernelMult::rule_extra_status_rule_dom_xt(Rule &) const { dbg(); }

// *****************************************************************************
void KernelMult::rule_extra_status_rule_transpose_xt(Rule &) const { dbg(); }

// *****************************************************************************
void KernelMult::token_LP(Token &) const   /* nothing */
{
}
void KernelMult::token_RP(Token &) const   /* nothing */
{
}
void KernelMult::token_GRAD_OP(Token &) const   /* nothing */
{
}

// *****************************************************************************
void KernelMult::rule_extra_status_rule_dot_xt(Rule &) const { dbg(); }

/** @endcond */

}  // namespace internal

}  // namespace mfem
