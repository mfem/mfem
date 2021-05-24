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
#include "xfl_ker.hpp"

// *****************************************************************************
// MFEM_FORALL, 2D+Column, Collocated + SIMD kernels
// *****************************************************************************

namespace mfem
{

namespace internal
{

/** @cond */  // Doxygen warning: documented symbol was not declared or defined

// *****************************************************************************
/// AST Kernel Setup Code Generation
// *****************************************************************************
KerLeanSetup::KerLeanSetup(const int K, xfl &ufl, Node *root,
                           std::ostringstream &out,
                           const xfl::fes &fes,
                           const xfl::fec &fec)
   : Middlend(ufl), root(root), out(out), fes(fes), fec(fec), dim(fec.dim)
{
   out << "\n// SIMD - Headers and Sizes\n";
   out << "#include \"linalg/simd.hpp\"\n";
   out << "using Real = AutoSIMDTraits<double,double>::vreal_t;\n";
   out << "#define SIMD_SIZE (MFEM_SIMD_BYTES/sizeof(double))\n";
   out << "#define SMS SIMD_SIZE\n";

   out << "\n// Pragma - UNROLL & ALIGN\n";
   out << "#define MFEM_ALIGN(T) alignas(alignof(T)) T\n";
   out << "#define PRAGMA(X) _Pragma(#X)\n";
   out << "#ifdef __clang__\n"
       << "  #define UNROLL(N) PRAGMA(unroll(N))\n"
       << "#elif defined(__FUJITSU)\n"
       << "  #define UNROLL(N) PRAGMA(loop unroll N)\n"
       << "#elif defined(__GNUC__) || defined(__GNUG__)\n"
       << "  #define UNROLL(N) PRAGMA(GCC unroll(N))\n"
       << "#else\n"
       << "  #define UNROLL(N)\n"
       << "#endif\n";

   out << "\n// Blocking\n";
   out << "#define BLK_SZ 1\n";
   out << "#define FOREACH_BLOCK(i,N) for(int i##i=0; i##i<N+BLK_SZ-1; i##i+=BLK_SZ)\n";
   out << "#define FOREACH_INNER(i) for(int i=i##i; i<i##i+BLK_SZ; i++)\n";

   out << "\n// Foreach Thread\n";
   out << "#define FOREACH_THREAD(i,k,N) for(int i=0; i<N; i+=1)\n";
   out << "#define SYNC_THREADS\n\n";

   out << "\ntemplate<int DIM, int DX0, int DX1, int Q1D> inline static\n";
   out << "void KSetup" << K << "(";
   out << "const int ndofs, const int vdim, const int NE,\n";
   out << "\t\tconst double * __restrict__ J0,\n";
   out << "\t\tconst double * __restrict__ w,\n";
   out << "\t\tdouble * __restrict__ dx) {\n";
   out << "\tassert(vdim == 1);\n";

   out << "\tconst auto J = Reshape(J0, DIM,DIM, Q1D,Q1D,Q1D, NE);\n";
   out << "\tconst auto W = Reshape(w, Q1D,Q1D,Q1D);\n";
   out << "\tauto DX = Reshape(dx, Q1D,Q1D,Q1D, 6);\n";

   out << "\tconst int e = 0;\n{\n"; // assuming same elements
}

// *****************************************************************************
KerLeanSetup::~KerLeanSetup() { out << " }\n}" << std::endl; }

// *****************************************************************************
void KerLeanSetup::token_IDENTIFIER(Token &t) const
{
   const std::string &var_name = t.Name();
   for (auto &p : ufl.ctx.var)
   {
      xfl::var &var = p.second;
      if (var.name != var_name) { continue;  }

      if (var.type == TOK::TEST_FUNCTION)
      {
         out << "   }\n  }\n }\n";
         out << " MFEM_SYNC_THREAD;\n";
      }
      var_stack.push(&var);
   }
}

// *****************************************************************************
void KerLeanSetup::rule_extra_status_rule_eval_xt(Rule &) const
{
   if (!lhs) { return;}
   assert(!var_stack.empty());
   var_stack.pop();
   ops_stack.push(TOK::EVAL_XT);
}

// *****************************************************************************
void KerLeanSetup::rule_grad_expr_grad_op_form_args(Rule &) const
{
   if (!lhs) { return; }
   assert(!var_stack.empty());
   var_stack.pop();
   ops_stack.push(TOK::GRAD_OP);
}

// *****************************************************************************
void KerLeanSetup::token_COMA(Token &) const
{
   // MFEM_FOREACH_THREAD
   out << " MFEM_FOREACH_THREAD(qz,z,Q1D){\n"
       << "  MFEM_FOREACH_THREAD(qy,y,Q1D){\n"
       << "   MFEM_FOREACH_THREAD(qx,x,Q1D){\n";

   if (ops_stack.empty()) { assert(false); }
   const int op = ops_stack.top();

   out << "     const double irw = W(qx,qy,qz);\n";
   out << "     const double *Jtr = &J(0,0,qx,qy,qz, e);\n";
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
      out << "     DX(qx,qy,qz,0) = B[0+DX0*0];\n"
          << "     DX(qx,qy,qz,1) = B[1+DX0*0];\n"
          << "     DX(qx,qy,qz,2) = B[2+DX0*0];\n"
          << "     DX(qx,qy,qz,3) = B[1+DX0*1];\n"
          << "     DX(qx,qy,qz,4) = B[2+DX0*1];\n"
          << "     DX(qx,qy,qz,5) = B[2+DX0*2];\n";

   }
   else if (op == TOK::EVAL_XT) { assert(false); }
   else { assert(false); }
}

// *****************************************************************************
/// AST Kernel Code Generation
// *****************************************************************************
KerLeanMult::KerLeanMult(const int K, xfl &ufl, Node *root,
                         std::ostringstream &out,
                         const xfl::fes &fes,
                         const xfl::fec &fec)
   : Middlend(ufl), root(root), out(out), fes(fes), fec(fec), dim(fec.dim)
{
   assert(dim == 3);
   const int p = fec.order;
   const int node_order = 1;
   const int order_w = (node_order * fec.dim - 1);
   const int q = 2 * p + order_w;
   const int GeomType = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   const mfem::IntegrationRule &ir = mfem::IntRules.Get(GeomType, q);
   Q1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
   D1D = p + 1;
   assert(Q1D >= D1D);

   out << "\ntemplate<int DIM, int DX0, int DX1, int D1D, int Q1D> "
       << "inline static\n";
   out << "void KMult" << K << "(";
   out << "const int ndofs, const int vdim, const int NE,\n";
   out << "\t\tconst double * __restrict__ B,\n";
   out << "\t\tconst double * __restrict__ G,\n";
   out << "\t\tconst int * __restrict__ map,\n";
   out << "\t\tconst double * __restrict__ dx,\n";
   out << "\t\tconst double * __restrict__ xd,\n";
   out << "\t\tdouble * __restrict__ yd) {\n";

   // Kernel operations
   out << "\t// kernel operations: ";
   mfem::internal::KernelOperations ko(ufl, out);
   ufl.DfsInOrder(root, ko);

   out << "\n";
   out << "\tconst auto b = Reshape(B, Q1D,D1D);\n";
   out << "\tconst auto g = Reshape(G, Q1D,Q1D);\n";
   out << "\tconst auto DX = Reshape(dx, Q1D,Q1D,Q1D, 6);\n";
   out << "\tconst auto MAP = Reshape(map, D1D,D1D,D1D, NE);\n";
   out << "\tconst auto XD = Reshape(xd, ndofs);\n";
   out << "\tauto YD = Reshape(yd, ndofs);\n";

   // Shared memory used by the threads
   out << "\tMFEM_ALIGN(Real) s_Iq[Q1D][Q1D][Q1D];\n";
   out << "\tdouble s_D[Q1D][Q1D];\n";
   out << "\tdouble s_I[Q1D][D1D];\n";

   // Main loop over the elements
   out << "\tMFEM_VERIFY((NE % SIMD_SIZE) == 0, \"NE vs SIMD_SIZE error!\");\n";
   out << "for(int e = 0; e < NE; e+=SMS){\n";

   // Registers used by the kernels (can be here, as long there are no threads)
   out << "\tMFEM_ALIGN(Real) r_qt[Q1D][Q1D];\n";
   out << "\tMFEM_ALIGN(Real) r_q[Q1D][Q1D][Q1D];\n";
   out << "\tMFEM_ALIGN(Real) r_Aq[Q1D][Q1D][Q1D];\n\n";
}

// *****************************************************************************
KerLeanMult::~KerLeanMult()
{
   out << "\t} // MFEM_FORALL_2D\n";
   out << "} // KMult1" << std::endl;
}

// *****************************************************************************
void KerLeanMult::token_IDENTIFIER(Token &t) const
{
   const std::string &var_name = t.Name();
   for (auto &p : ufl.ctx.var)
   {
      xfl::var &var = p.second;
      if (var.name != var_name) { continue; }
      if (var.type == TOK::TEST_FUNCTION) { lhs = false; }
      if (var.type == TOK::TRIAL_FUNCTION)
      {
         out << "// Scatter X\n"
             << "FOREACH_BLOCK(j,Q1D){\n"
             << " FOREACH_INNER(j){\n"
             << "  if (j>=Q1D) continue;\n"
             << "  FOREACH_THREAD(i,x,Q1D){\n"
             << "   s_D[j][i] = g(i,j);\n"
             << "   if (i<D1D) { s_I[j][i] = b(j,i); }\n"
             << "   if (i<D1D && j<D1D){\n"
             << "    UNROLL("<<D1D<<")\n"
             << "    for (int k = 0; k < D1D; k++){\n"
             << "     MFEM_ALIGN(Real) vXD;\n"
             << "     UNROLL(SMS)\n"
             << "     for (size_t v = 0; v < SMS; v++){\n"
             << "      const int gid = MAP(i, j, k, e + v);\n"
             << "      const int idx = gid >= 0 ? gid : -1 - gid;\n"
             << "      vXD[v] = XD(idx);\n"
             << "     }\n"
             << "     r_q[j][i][k] = vXD;\n"
             << "}}}}}SYNC_THREADS;\n\n";
      }
      var_stack.push(&var);
   }
}

// *****************************************************************************
void KerLeanMult::rule_grad_expr_grad_op_form_args(Rule &) const
{
   assert(!var_stack.empty());
   if (lhs)
   {
      out << "// Grad1X\n"
          << "FOREACH_THREAD(b,y,D1D){\n"
          << " FOREACH_THREAD(a,x,D1D){\n"
          << "  UNROLL("<<Q1D<<")\n"
          << "  for (int k=0; k<Q1D; ++k){\n"
          << "   MFEM_ALIGN(Real) res; res = 0.0;\n"
          << "   UNROLL("<<D1D<<")\n"
          << "   for (int c=0; c<D1D; ++c){\n"
          << "    res.fma(s_I[k][c], r_q[b][a][c]);\n"
          << "   }\n"
          << "   s_Iq[k][b][a] = res;\n"
          << "}}}SYNC_THREADS;\n\n";

      out << "// Grad1Y\n"
          << "FOREACH_THREAD(k,y,Q1D){\n"
          << " FOREACH_THREAD(a,x,D1D){\n"
          << "  for (int b=0; b<D1D; ++b)\n"
          << "   r_Aq[k][a][b] = s_Iq[k][b][a];\n"
          << "  UNROLL("<<Q1D<<")\n"
          << "  for (int j=0; j<Q1D; ++j){\n"
          << "   MFEM_ALIGN(Real) res; res = 0;\n"
          << "   UNROLL("<<D1D<<")\n"
          << "   for (int b=0; b<D1D; ++b)\n"
          << "    res.fma(s_I[j][b], r_Aq[k][a][b]);\n"
          << "   s_Iq[k][j][a] = res;\n"
          << "}}}SYNC_THREADS;\n\n";

      out << "// Grad1Z\n"
          << "FOREACH_THREAD(k,y,Q1D){\n"
          << " FOREACH_THREAD(j,x,Q1D){\n"
          << "  for (int a=0; a<D1D; ++a)\n"
          << "   r_Aq[k][j][a] = s_Iq[k][j][a];\n"
          << "  UNROLL("<<Q1D<<")\n"
          << "  for (int i=0; i<Q1D; ++i){\n"
          << "   MFEM_ALIGN(Real) res; res = 0;\n"
          << "   UNROLL("<<D1D<<")\n"
          << "   for (int a=0; a<D1D; ++a)\n"
          << "    res.fma(s_I[i][a], r_Aq[k][j][a]);\n"
          << "   s_Iq[k][j][i] = res;\n"
          << "}}}SYNC_THREADS;\n\n";
   }
   else // transpose
   {
      out << "// GradZT\n"
          << "FOREACH_THREAD(j,y,Q1D){\n"
          << " FOREACH_THREAD(i,x,Q1D){\n"
          << "  UNROLL("<<D1D<<")\n"
          << "  for (int c=0; c<D1D; ++c){\n"
          << "   MFEM_ALIGN(Real) res; res = 0;\n"
          << "   UNROLL("<<Q1D<<")\n"
          << "   for (int k=0; k<Q1D; ++k)\n"
          << "    res.fma(s_I[k][c], r_Aq[j][i][k]);\n"
          << "   s_Iq[c][j][i] = res;\n"
          << "}}}SYNC_THREADS;\n";

      out << "// GradYT\n"
          << "FOREACH_THREAD(c,y,D1D){\n"
          << " FOREACH_THREAD(i,x,Q1D){\n"
          << "  UNROLL("<<Q1D<<")\n"
          << "  for (int j=0; j<Q1D; ++j)\n"
          << "   r_Aq[c][i][j] = s_Iq[c][j][i];\n"
          << "  UNROLL("<<D1D<<")\n"
          << "  for (int b=0; b<D1D; ++b){\n"
          << "   MFEM_ALIGN(Real) res; res = 0;\n"
          << "   UNROLL("<<Q1D<<")\n"
          << "   for (int j=0; j<Q1D; ++j)\n"
          << "    res.fma(s_I[j][b], r_Aq[c][i][j]);\n"
          << "   s_Iq[c][b][i] = res;\n"
          << "}}}SYNC_THREADS;\n";

      out << "// GradXT\n"
          << "FOREACH_THREAD(c,y,D1D){\n"
          << " FOREACH_THREAD(b,x,D1D){\n"
          << "  UNROLL("<<Q1D<<")\n"
          << "  for (int i=0; i<Q1D; ++i)\n"
          << "   r_Aq[c][b][i] = s_Iq[c][b][i];\n"
          << "  UNROLL("<<D1D<<")\n"
          << "  for (int a=0; a<D1D; ++a){\n"
          << "   MFEM_ALIGN(Real) res; res = 0;\n"
          << "   UNROLL("<<Q1D<<")\n"
          << "   for (int i=0; i<Q1D; ++i)\n"
          << "    res.fma(s_I[i][a], r_Aq[c][b][i]);\n"
          << "   s_Iq[c][b][a] = res;\n"
          << "   s_Iq[c][b][a] = res;\n"
          << "}}}SYNC_THREADS;\n";

      out << "// Gather\n"
          << "FOREACH_THREAD(j,y,D1D){\n"
          << " FOREACH_THREAD(i,x,D1D){\n"
          << "  UNROLL("<<D1D<<")\n"
          << "  for (int k = 0; k < D1D; k++){\n"
          << "   UNROLL(SMS)\n"
          <<"    for (size_t v = 0; v < SMS; v++){\n"
          << "    const int gid = MAP(i, j, k, e + v);\n"
          << "    const int idx = gid >= 0 ? gid : -1 - gid;\n"
          << "    YD(idx) += (s_Iq[k][j][i])[v];\n"
          << "}}}}\n";
   }
   var_stack.pop();
   ops_stack.push(TOK::GRAD_OP);
}

// *****************************************************************************
void KerLeanMult::token_COMA(Token &) const
{
   out << "// Flush\n"
       << "FOREACH_THREAD(j,y,Q1D){\n"
       << " FOREACH_THREAD(i,x,Q1D){\n"
       << "  UNROLL("<<Q1D<<")\n"
       << "  for (int k = 0; k < Q1D; k++) { r_Aq[j][i][k] = 0.0; }\n"
       << "}}SYNC_THREADS;\n\n";

   // FOREACH_THREAD
   // Iteration :   0  (B r, r) = 2.95971e-06 ...
   // Iteration :  50  (B r, r) = 6.48197e-06
   out << "// Q-Function\n"
       << "UNROLL("<<Q1D<<")\n"
       << "for (int k = 0; k < Q1D; k++){\n"
       << " SYNC_THREADS;\n"
       << " MFEM_ALIGN(Real) r_Gqr[Q1D][Q1D];\n"
       << " MFEM_ALIGN(Real) r_Gqs[Q1D][Q1D];\n"
       << ""
       << " FOREACH_THREAD(j,y,Q1D){\n"
       << "  FOREACH_BLOCK(i,Q1D){\n"
       << "  FOREACH_INNER(i){\n"
       << "   if (i>=Q1D) continue;\n"
       << "   MFEM_ALIGN(Real) qr, qs, qt; qr = 0.0; qs = 0.0; qt = 0.0;\n"
       << "   UNROLL("<<Q1D<<")\n"
       << "   for (int m = 0; m < Q1D; m++){\n"
       << "    const double Dim = s_D[i][m];\n"
       << "    const double Djm = s_D[j][m];\n"
       << "    const double Dkm = s_D[k][m];\n"
       << "    qr.fma(Dim, s_Iq[k][j][m]);\n"
       << "    qs.fma(Djm, s_Iq[k][m][i]);\n"
       << "    qt.fma(Dkm, s_Iq[m][j][i]);\n"
       << "   }\n"
       << "   const double D00 = DX(i,j,k,0);\n"
       << "   const double D01 = DX(i,j,k,1);\n"
       << "   const double D02 = DX(i,j,k,2);\n"
       << "   const double D10 = D01;\n"
       << "   const double D11 = DX(i,j,k,3);\n"
       << "   const double D12 = DX(i,j,k,4);\n"
       << "   const double D20 = D02;\n"
       << "   const double D21 = D12;\n"
       << "   const double D22 = DX(i,j,k,5);\n"
       << "   r_Gqr[j][i] = 0.0;\n"
       << "   r_Gqr[j][i].fma(D00,qr);\n"
       << "   r_Gqr[j][i].fma(D10,qs);\n"
       << "   r_Gqr[j][i].fma(D20,qt);\n"
       << "   r_Gqs[j][i] = 0.0;\n"
       << "   r_Gqs[j][i].fma(D01,qr);\n"
       << "   r_Gqs[j][i].fma(D11,qs);\n"
       << "   r_Gqs[j][i].fma(D21,qt);\n"
       << "   r_qt[j][i] = 0.0;\n"
       << "   r_qt[j][i].fma(D02,qr);\n"
       << "   r_qt[j][i].fma(D12,qs);\n"
       << "   r_qt[j][i].fma(D22,qt);\n"
       << " }}\n"
       << " } // BLOCK\n"
       << " SYNC_THREADS;\n"
       << " FOREACH_THREAD(j,y,Q1D){\n"
       << "  FOREACH_THREAD(i,x,Q1D){\n"
       << "   MFEM_ALIGN(Real) Aqtmp; Aqtmp = 0.0;\n"
       << "   UNROLL("<<Q1D<<")\n"
       << "   for (int m = 0; m < Q1D; m++){\n"
       << "    const double Dmi = s_D[m][i];\n"
       << "    const double Dmj = s_D[m][j];\n"
       << "    const double Dkm = s_D[k][m];\n"
       << "    Aqtmp.fma(Dmi, r_Gqr[j][m]);\n"
       << "    Aqtmp.fma(Dmj, r_Gqs[m][i]);\n"
       << "    r_Aq[j][i][m].fma(Dkm, r_qt[j][i]);\n"
       << "   }\n"
       << "   r_Aq[j][i][k] += Aqtmp;\n"
       << " }}SYNC_THREADS;\n"
       << "}\n";
   if (ops_stack.empty()) { assert(false); }  // something is wrong
   const int op = ops_stack.top();
   assert(op == TOK::GRAD_OP) ; // Grad operation
}

/** @endcond */

}  // namespace internal

}  // namespace mfem
