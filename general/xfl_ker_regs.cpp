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

#define QUOTE(...) #__VA_ARGS__

// *****************************************************************************
// MFEM_FORALL + Registers kernels
// *****************************************************************************

namespace mfem
{

namespace internal
{

/** @cond */  // Doxygen warning: documented symbol was not declared or defined

// *****************************************************************************
/// AST Kernel Setup Code Generation
// *****************************************************************************
KerRegsSetup::KerRegsSetup(const int K, xfl &ufl, Node *root,
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

   out << "#define D1D "<<D1D<<"\n";
   out << "#define Q1D "<<Q1D<<"\n";

   out << "#include \"linalg/simd.hpp\"\n";
   out << "using Real = AutoSIMDTraits<double,double>::vreal_t;\n";
   out << "#define SIMD_SIZE static_cast<int>(MFEM_SIMD_BYTES/sizeof(double))\n";
   out << "#define SMS SIMD_SIZE\n";

   out << "\ntemplate<int DIM, int DX0, int DX1> inline static\n";
   out << "void KSetup" << K << "(";
   out << "const int ndofs,\n\t\tconst int vdim, const int NE,\n";
   out << "\t\tconst double * __restrict__ J0,\n";
   out << "\t\tconst double * __restrict__ w,\n";
   out << "\t\tdouble * __restrict__ dx) {\n";
   out << "\tassert(vdim == 1);\n";

   // Kernel operations
   out << "\t// kernel operations: ";
   mfem::internal::KernelOperations ko(ufl, out);
   ufl.DfsInOrder(root, ko);

   out << "\n\tconst auto J = Reshape(J0, DIM,DIM, Q1D,Q1D,Q1D, NE);\n";
   out << "\tconst auto W = Reshape(w, Q1D,Q1D,Q1D);\n";
   out << "\tauto DX = Reshape((Real*)dx, DX0,DX1, Q1D,Q1D,Q1D, NE/SMS);\n";

   out << "\tMFEM_VERIFY((NE % SIMD_SIZE) == 0, \"NE vs SIMD_SIZE error!\");\n";
   out << "for(int e = 0; e < NE; e+=SMS){\n";
}

// *****************************************************************************
KerRegsSetup::~KerRegsSetup() { out << " }\n}" << std::endl; }

// *****************************************************************************
void KerRegsSetup::token_IDENTIFIER(Token &t) const
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
void KerRegsSetup::rule_extra_status_rule_eval_xt(Rule &) const
{
   if (!lhs) { return;}
   assert(!var_stack.empty());
   var_stack.pop();
   ops_stack.push(TOK::EVAL_XT);
}

// *****************************************************************************
void KerRegsSetup::rule_grad_expr_grad_op_form_args(Rule &) const
{
   if (!lhs) { return; }
   assert(!var_stack.empty());
   var_stack.pop();
   ops_stack.push(TOK::GRAD_OP);
}

// *****************************************************************************
void KerRegsSetup::token_COMA(Token &) const
{
   // MFEM_FOREACH_THREAD
   out << " MFEM_FOREACH_THREAD(qz,z,Q1D){\n"
       << "  MFEM_FOREACH_THREAD(qy,y,Q1D){\n"
       << "   MFEM_FOREACH_THREAD(qx,x,Q1D){\n"
       << "    Real vdx[DX0*DX1];\n"
       << "    for (size_t v = 0; v < SMS; v++){\n";

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
KerRegsMult::KerRegsMult(const int K, xfl &ufl, Node *root,
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
   Q1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
   D1D = p + 1;
   assert(Q1D >= D1D);

   out << "\ntemplate<int DIM, int DX0, int DX1> inline static\n";
   out << "void KMult" << K << "(";
   out << "const int ndofs, const int vdim, const int NE,\n";
   out << "\t\tconst double * __restrict__ B,\n";
   out << "\t\tconst double * __restrict__ G,\n";
   out << "\t\tconst int * __restrict__ map,\n";
   out << "\t\tconst double * __restrict__ dx,\n";
   out << "\t\tconst double * __restrict__ xd,\n";
   out << "\t\tdouble * __restrict__ yd) {\n";

   // Kernel operations
   out << "\n\t// kernel operations: ";
   mfem::internal::KernelOperations ko(ufl, out);
   ufl.DfsInOrder(root, ko);
   out << "\n";

   assert(dim == 3);
   out << "\n";
   out << "\tassert(vdim == 1);\n\n";

   out << "\tconst auto b = Reshape(B, Q1D, D1D);\n";
   out << "\tconst auto g = Reshape(G, Q1D, Q1D);\n";

   out << "\tconst auto DX = Reshape((Real*)dx, DX0,DX1, Q1D,Q1D,Q1D, NE/SMS);\n";
   out << "\tconst auto MAP = Reshape(map, D1D,D1D,D1D, NE);\n";
   out << "\tconst auto XD = Reshape(xd, ndofs);\n";
   out << "\tauto YD = Reshape(yd, ndofs);\n";

   // dump kernel operations
   out << "\t\t// kernel operations: ";
   mfem::internal::KernelOperations po(ufl, out);
   ufl.DfsInOrder(root, po);
   out << "\n";

   // UNROLL
   out << "#define PRAGMA(X) _Pragma(#X)\n";
   out << "#ifdef __clang__\n"
       << "  #define UNROLL(N) PRAGMA(unroll(N))\n"
       << "#elif defined(__GNUC__) || defined(__GNUG__)\n"
       << "  #define UNROLL(N) PRAGMA(GCC unroll(N))\n"
       << "#else\n"
       << "  #define UNROLL(N)\n"
       << "#endif\n";

   out << "\n\tMFEM_VERIFY((NE % SIMD_SIZE) == 0, \"NE vs SIMD_SIZE error!\");\n";

   out << "\n\tfor(int ek = 0; ek < NE; ek+=SMS){\n";
   out << "\tconst int e = ek;\n";
   out << "\tconst int ve = e/SMS;\n";

   out << "\t\tMFEM_SHARED Real s_Iq[Q1D][Q1D][Q1D];\n";
   out << "\t\tMFEM_SHARED double s_D[Q1D][Q1D];\n";
   out << "\t\tMFEM_SHARED double s_I[Q1D][D1D];\n";
   out << "\t\tMFEM_SHARED Real s_Gqr[Q1D][Q1D];\n";
   out << "\t\tMFEM_SHARED Real s_Gqs[Q1D][Q1D];\n\n";

   out << "\t\tReal r_qt[Q1D][Q1D];\n";
   out << "\t\tReal r_q[Q1D][Q1D][Q1D];\n";
   out << "\t\tReal r_Aq[Q1D][Q1D][Q1D];\n\n";
}

// *****************************************************************************
KerRegsMult::~KerRegsMult()
{
   out << "\t} // MFEM_FORALL_2D\n";
   out << "} // KMult1" << std::endl;
}

// *****************************************************************************
void KerRegsMult::token_IDENTIFIER(Token &t) const
{
   const std::string &var_name = t.Name();
   for (auto &p : ufl.ctx.var)
   {
      xfl::var &var = p.second;
      if (var.name != var_name) { continue; }
      if (var.type == TOK::TEST_FUNCTION) { lhs = false; }
      if (var.type == TOK::TRIAL_FUNCTION)
      {
         out << "// Load X\n"
             << "MFEM_FOREACH_THREAD(j,y,Q1D){\n"
             << " MFEM_FOREACH_THREAD(i,x,Q1D){\n"
             << "  s_D[j][i] = g(i,j);\n"
             << "  if (i<D1D) { s_I[j][i] = b(j,i); }\n"
             << "  if (i<D1D && j<D1D){\n"
             << "   UNROLL("<<D1D<<")\n"
             << "   for (int k = 0; k < D1D; k++){\n"
             << "    Real vXD;\n"
             << "    for (int v = 0; v < SMS; v++){\n"
             << "     const int gid = MAP(i, j, k, e + v);\n"
             << "     const int idx = gid >= 0 ? gid : -1 - gid;\n"
             << "     vXD[v] = XD(idx);\n"
             << "    }\n"
             << "    r_q[j][i][k] = vXD;\n"
             << "}}}}MFEM_SYNC_THREAD;\n\n";
      }
      var_stack.push(&var);
   }
}

// *****************************************************************************
void KerRegsMult::rule_grad_expr_grad_op_form_args(Rule &) const
{
   dbg();
   assert(!var_stack.empty());
   if (lhs)
   {
      out << "// Grad1X\n"
          << "MFEM_FOREACH_THREAD(b,y,Q1D){\n"
          << " MFEM_FOREACH_THREAD(a,x,Q1D){\n"
          << "  if (a<D1D && b<D1D){\n"
          << "   UNROLL("<<Q1D<<")\n"
          << "   for (int k=0; k<Q1D; ++k){\n"
          << "    Real res; res = 0.0;\n"
          << "    UNROLL("<<D1D<<")\n"
          << "    for (int c=0; c<D1D; ++c){\n"
          << "     res += s_I[k][c]*r_q[b][a][c];\n"
          << "    }\n"
          << "    s_Iq[k][b][a] = res;\n"
          << "}}}}MFEM_SYNC_THREAD;\n\n";

      out << "// Grad1Y\n"
          << "MFEM_FOREACH_THREAD(k,y,Q1D){\n"
          << " MFEM_FOREACH_THREAD(a,x,Q1D){\n"
          << "  if (a<D1D){\n"
          << "   for (int b=0; b<D1D; ++b){\n"
          << "    r_Aq[k][a][b] = s_Iq[k][b][a];\n"
          << "   }\n"
          << "   UNROLL("<<Q1D<<")\n"
          << "   for (int j=0; j<Q1D; ++j){\n"
          << "    Real res; res = 0;\n"
          << "    UNROLL("<<D1D<<")\n"
          << "    for (int b=0; b<D1D; ++b){\n"
          << "     res += s_I[j][b]*r_Aq[k][a][b];\n"
          << "    }\n"
          << "    s_Iq[k][j][a] = res;\n"
          << "}}}}MFEM_SYNC_THREAD;\n\n";

      out << "// Grad1Z\n"
          << "MFEM_FOREACH_THREAD(k,y,Q1D){\n"
          << " MFEM_FOREACH_THREAD(j,x,Q1D){\n"
          << "  for (int a=0; a<D1D; ++a){\n"
          << "   r_Aq[k][j][a] = s_Iq[k][j][a];\n"
          << "  }\n"
          << "  UNROLL("<<Q1D<<")\n"
          << "  for (int i=0; i<Q1D; ++i){\n"
          << "   Real res; res = 0;\n"
          << "   UNROLL("<<D1D<<")\n"
          << "   for (int a=0; a<D1D; ++a){\n"
          << "    res += s_I[i][a]*r_Aq[k][j][a];\n"
          << "   }\n"
          << "   s_Iq[k][j][i] = res;\n"
          << "}}}MFEM_SYNC_THREAD;\n\n";
   }
   else // transpose
   {
      out << "// GradZT\n"
          << "MFEM_FOREACH_THREAD(j,y,Q1D){\n"
          << " MFEM_FOREACH_THREAD(i,x,Q1D){\n"
          << "  UNROLL("<<D1D<<")\n"
          << "  for (int c=0; c<D1D; ++c){\n"
          << "   Real res; res = 0;\n"
          << "   UNROLL("<<Q1D<<")\n"
          << "   for (int k=0; k<Q1D; ++k){\n"
          << "    res += s_I[k][c]*r_Aq[j][i][k];\n"
          << "   }\n"
          << "   s_Iq[c][j][i] = res;\n"
          << "}}}MFEM_SYNC_THREAD;\n";

      out << "// GradYT\n"
          << "MFEM_FOREACH_THREAD(c,y,Q1D){\n"
          << " MFEM_FOREACH_THREAD(i,x,Q1D){\n"
          << "  if (c<D1D){\n"
          << "   UNROLL("<<Q1D<<")\n"
          << "   for (int j=0; j<Q1D; ++j){\n"
          << "    r_Aq[c][i][j] = s_Iq[c][j][i];\n"
          << "   }\n"
          << "   UNROLL("<<D1D<<")\n"
          << "   for (int b=0; b<D1D; ++b){\n"
          << "    Real res; res = 0;\n"
          << "    UNROLL("<<Q1D<<")\n"
          << "    for (int j=0; j<Q1D; ++j){\n"
          << "     res += s_I[j][b]*r_Aq[c][i][j];\n"
          << "    }\n"
          << "    s_Iq[c][b][i] = res;\n"
          << "}}}}MFEM_SYNC_THREAD;\n";

      out << "// GradXT\n"
          << "MFEM_FOREACH_THREAD(c,y,Q1D){\n"
          << " MFEM_FOREACH_THREAD(b,x,Q1D){\n"
          << "  if (b<D1D && c<D1D){\n"
          << "   UNROLL("<<Q1D<<")\n"
          << "   for (int i=0; i<Q1D; ++i){\n"
          << "    r_Aq[c][b][i] = s_Iq[c][b][i];\n"
          << "   }\n"
          << "   UNROLL("<<D1D<<")\n"
          << "   for (int a=0; a<D1D; ++a){\n"
          << "    Real res; res = 0;\n"
          << "    UNROLL("<<Q1D<<")\n"
          << "    for (int i=0; i<Q1D; ++i){\n"
          << "     res += s_I[i][a]*r_Aq[c][b][i];\n"
          << "    }\n"
          << "    s_Iq[c][b][a] = res;\n"
          << "}}}}MFEM_SYNC_THREAD;\n";

      out << "// Scatter\n"
          << "MFEM_FOREACH_THREAD(j,y,Q1D){\n"
          << " MFEM_FOREACH_THREAD(i,x,Q1D){\n"
          << "  if (i<D1D && j<D1D){\n"
          << "   UNROLL("<<D1D<<")\n"
          << "   for (int k = 0; k < D1D; k++){\n"
          <<"     for (int v = 0; v < SMS; v++){\n"
          << "     const int gid = MAP(i, j, k, e + v);\n"
          << "     const int idx = gid >= 0 ? gid : -1 - gid;\n"
          << "     AtomicAdd(YD(idx), (s_Iq[k][j][i])[v]);\n"
          << "}}}}}\n";
   }
   var_stack.pop();
   ops_stack.push(TOK::GRAD_OP);
}

// *****************************************************************************
void KerRegsMult::token_COMA(Token &) const
{
   out << "// Flush\n"
       << "MFEM_FOREACH_THREAD(j,y,Q1D){\n"
       << " MFEM_FOREACH_THREAD(i,x,Q1D){\n"
       << "  UNROLL("<<Q1D<<")\n"
       << "  for (int k = 0; k < Q1D; k++) { r_Aq[j][i][k] = 0.0; }\n"
       << "}}MFEM_SYNC_THREAD;\n\n";

   // MFEM_FOREACH_THREAD
   out << "// Q-Function\n"
       << "UNROLL("<<Q1D<<")\n"
       << "for (int k = 0; k < Q1D; k++){\n"
       << " MFEM_SYNC_THREAD;\n"
       << " MFEM_FOREACH_THREAD(j,y,Q1D){\n"
       << "  MFEM_FOREACH_THREAD(i,x,Q1D){\n"
       << "   Real qr, qs; qr = 0.0; qs = 0.0;\n"
       << "   r_qt[j][i] = 0.0;\n"
       << "   UNROLL("<<Q1D<<")\n"
       << "   for (int m = 0; m < Q1D; m++){\n"
       << "    const double Dim = s_D[i][m];\n"
       << "    const double Djm = s_D[j][m];\n"
       << "    const double Dkm = s_D[k][m];\n"
       << "    qr += Dim*s_Iq[k][j][m];\n"
       << "    qs += Djm*s_Iq[k][m][i];\n"
       << "    r_qt[j][i] += Dkm*s_Iq[m][j][i];\n"
       << "   }\n"
       << "   const Real qt = r_qt[j][i];\n"
       << "   const Real u[DX0] = {qr, qs, qt}; Real v[DX0];\n"
       << "   kernels::Mult(DX0,DX1,&DX(0,0,i,j,k,ve),u,v);\n"
       << "   s_Gqr[j][i] = v[0];\n"
       << "   s_Gqs[j][i] = v[1];\n"
       << "   r_qt[j][i]  = v[2];\n"
       << " }}\n"
       << " MFEM_SYNC_THREAD;\n"
       << " MFEM_FOREACH_THREAD(j,y,Q1D){\n"
       << "  MFEM_FOREACH_THREAD(i,x,Q1D){\n"
       << "   Real Aqtmp; Aqtmp = 0.0;\n"
       << "   UNROLL("<<Q1D<<")\n"
       << "   for (int m = 0; m < Q1D; m++){\n"
       << "    const double Dmi = s_D[m][i];\n"
       << "    const double Dmj = s_D[m][j];\n"
       << "    const double Dkm = s_D[k][m];\n"
       << "    Aqtmp += Dmi*s_Gqr[j][m];\n"
       << "    Aqtmp += Dmj*s_Gqs[m][i];\n"
       << "    r_Aq[j][i][m] += Dkm*r_qt[j][i];\n"
       << "   }\n"
       << "   r_Aq[j][i][k] += Aqtmp;\n"
       << " }}MFEM_SYNC_THREAD;\n"
       << "}\n";

   if (ops_stack.empty()) { assert(false); }  // something is wrong

   //out << "\t\t\t\tReal u[DX0], v[DX0]; \n";

   const int op = ops_stack.top();
   MFEM_VERIFY(op == TOK::GRAD_OP, "Grad operation error!") ; // Grad operation
   //out << "\t\t\t\tkernels::PullGrad1<Q1D>(qx,qy,qz,QQQ,u); \n";
   //ops_stack.pop();
   //out << "\t\t\t\tkernels::Mult(DX0,DX1,&DX(0,0,qx,qy,qz,e),u,v); \n";
   //out << "\t\t\t\tkernels::PushGrad1<Q1D>(qx,qy,qz,v,QQQ); \n";
}

/** @endcond */

}  // namespace internal

}  // namespace mfem
