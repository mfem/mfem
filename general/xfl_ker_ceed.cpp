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

#define MFEM_DEBUG_COLOR 199

using std::string;
using std::ostream;

#include "../mfem.hpp"
#include "xfl.Y.hpp"
#include "xfl_ker.hpp"

#define QUOTE(...) #__VA_ARGS__
#define UTF8(...) u8"#__VA_ARGS__"
#define RAW(...) R"delimiter(#__VA_ARGS__)delimiter"


// *****************************************************************************
// LibCEED - like kernels
// *****************************************************************************

namespace mfem
{

namespace internal
{

/** @cond */ // Doxygen warning: documented symbol was not declared or defined

/* anonymous */ namespace // forward declaration
{
extern const char *deviceFunctions;
}

// ****************************************************************************
/// AST Kernel Code Generation
// *****************************************************************************
KerCeedMult::KerCeedMult(const int K, xfl &ufl, Node *root,
                         std::ostringstream &out,
                         const xfl::fes &fes,
                         const xfl::fec &fec):
   Middlend(ufl), root(root), out(out), fes(fes), fec(fec), dim(fec.dim)
{
   const int p = fec.order;
   const int node_order = 1;
   const int order_w = (node_order*fec.dim-1);
   const int q = 2*p + order_w;
   const int GeomType = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   const mfem::IntegrationRule &ir = mfem::IntRules.Get(GeomType, q);
   const int Q1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
   const int D1D = p + 1;
   assert(Q1D >= D1D);

   out << "\n#ifdef USE_CUDA\n";
   out << "\t#define DEVICE __device__\n";
   out << "\t#define GLOBAL __global__\n";
   out << "\t#define SHARED __shared__\n";
   out << "\t#define MFEM_HOST_DEVICE __host__ __device__\n";
   out << "\t#define SYNC_THREADS __syncthreads()\n";
   out << "#else // USE_CUDA\n";
   out << "\t#define DEVICE\n";
   out << "\t#define GLOBAL\n";
   out << "\t#define SHARED\n";
   out << "\t#define MFEM_HOST_DEVICE\n";
   out << "\t#define SYNC_THREADS\n";
   out << "\t#define BLOCK_ID(k) 0\n";
   out << "\t#define THREAD_ID(k) tid ## k\n";
   out << "\t#define BLOCK_DIM(k) BLOCK_DIM_ ## k\n";
   out << "\t#define GRID_DIM(k) (NE/NBZ)\n";
   out << "#endif // USE_CUDA\n";
   out << "\n";

   out << "#define D1D " << D1D << "\n";
   out << "#define Q1D " << Q1D << "\n";
   // T1D = 1 in serial, max(Q1D,D1D) in parallel mode
   out << "#define T1D " << 1 << "\n"; // Q1D >= D1D
   // MBZ tied to 1 for now
   out << "#define NBZ " << 1 << "\n";
   out << "#define BLOCK_DIM_x T1D\n";
   out << "#define BLOCK_DIM_y T1D\n";
   out << "#define BLOCK_DIM_z NBZ\n";
   out << "#define BLK T1D*T1D*NBZ\n";
   out << "\n";

   //out << AtomicAdd << "\n";
   out << "\n" << deviceFunctions << "\n";
   out << "\n\n\n\n\n\n\n\n\n\n\n\n";

   out << "template<int DIM, int DX0, int DX1,\n";
   out << "\tint NDOFS, int NCOMP, int SMEM> inline static GLOBAL\n";
   out << "void KMult"<<K<<"(";
   out << "const int ndofs, const int vdim,\n";
   out << "\tconst int NE,\n";
   out << "\tconst double * __restrict__ B_arg,\n";
   out << "\tconst double * __restrict__ G_arg,\n";
   out << "\tconst int * __restrict__ map_arg,\n";
   out << "\tconst double * __restrict__ dx_arg,\n";
   out << "\tconst double * __restrict__ xd_arg,\n";
   out << "\tdouble * __restrict__ yd_arg) {\n\n";

   out << "\tSHARED CeedScalar slice[SMEM];\n\n";
   out << "\tBackendData data;\n";
   //out << "\tdata.slice = slice;\n";

   out << "\tconst auto DX = Reshape(dx_arg, DX0,DX1, Q1D,Q1D,Q1D, NE);\n";

   out << "\n\t// Load BG, forward\n";
   out << "\tSHARED double s_B_in_0["<<Q1D*D1D<<"];\n";
   out << "\tloadMatrix<D1D,Q1D>(data, B_arg, s_B_in_0);\n";
   //out << "\tdbg(\"B:\");for(int i=0;i<D1D*Q1D;i++){dbg(\"%f \",s_B_in_0[i]);}";
   out << "\tSHARED double s_G_in_0["<<Q1D*D1D<<"];\n";
   out << "\tloadMatrix<D1D,Q1D>(data, G_arg, s_G_in_0);\n";
   //out << "\tdbg(\"G:\");for(int i=0;i<D1D*Q1D;i++){dbg(\"%f \",s_G_in_0[i]);}";

   out << "\n\t// Load BG, backward\n";
   out << "\tSHARED double s_B_out_0["<<Q1D*D1D<<"];\n";
   out << "\tloadMatrix<D1D,Q1D>(data, B_arg, s_B_out_0);\n"; // transpose
   //out << "\tfor(int i=0;i<D1D*Q1D;i++){dbg(\"%f \",s_B_out_0[i]);}";
   out << "\tSHARED double s_G_out_0["<<Q1D*D1D<<"];\n";
   out << "\tloadMatrix<D1D,Q1D>(data, G_arg, s_G_out_0);\n"; // transpose
   //out << "\tfor(int i=0;i<D1D*Q1D;i++){dbg(\"%f \",s_G_out_0[i]);}";

   /*
      // MFEM_FOREACH_THREAD: batch + only 2D if dim > 1
      out << "\n\n// Thread for loop\n";
      out << "for(int tidz = 0; tidz < NBZ; tidz++){\n";
      out << " for(int tidx = 0; tidx < Q1D; tidx++){\n";
      out << "  for(int tidy = 0; tidy < Q1D; tidy++){\n\n";
      out << "\t//dbg(\"\\033[33mNE:%d, xyz:%d%d%d/%d%d%d\",NE,tidx,tidy,tidz,Q1D,Q1D,NBZ);\n";

      // Per thread data
      out << "\tBackendData data;\n";
      out << "\tdata.tidx = THREAD_ID(x);\n";
      out << "\tdata.tidy = THREAD_ID(y);\n";
      out << "\tdata.tidz = THREAD_ID(z);\n";
      out << "\tdata.tid  = THREAD_ID(x)\n"
          "\t\t  + THREAD_ID(y)*BLOCK_DIM(x)\n"
          "\t\t  + THREAD_ID(z)*BLOCK_DIM(y)*BLOCK_DIM(x);\n";
      out << "\tdata.slice = slice + data.tidz*T1D"<<(dim > 1?"*T1D":"")<<";\n\n";
   */

   // Element for loop
   out << "\n";
   out << "\t// Element for loop\n";
   out << "\t//dbg(\"THREAD_ID(x):%d, BLOCK_DIM(z):%d\",THREAD_ID(x),BLOCK_DIM(z));\n";
   out << "\tfor (int elem = 0/*THREAD_ID(x)*BLOCK_DIM(z) + THREAD_ID(z)*/;\n";
   out << "\t         elem < NE;\n";
   out << "\t         elem += 1/*GRID_DIM(x)*BLOCK_DIM(z)*/) {\n";
   out << "\n";

}

// *****************************************************************************
KerCeedMult::~KerCeedMult()
{
   out << "\t} // Element loop\n";
   out << "  } // tidy\n } // tidx\n} // tidz\n";
   out << "} // KMult";
   out << std::endl;
}

// *****************************************************************************
void KerCeedMult::token_IDENTIFIER(Token &t) const
{
   const std::string &var_name = t.Name();
   for (auto &p : ufl.ctx.var)
   {
      xfl::var &var = p.second;
      if (var.name != var_name) { continue; }
      out << "\t// [push] ";
      if (var.type == TOK::TEST_FUNCTION)
      {
         out << "test ";
         lhs = false;
      }
      if (var.type == TOK::TRIAL_FUNCTION)
      {
         out << "trial ";
      }
      out << var.name << ":mode(" << var.mode << ")\n";

      // Set the VDIM for the kernel
      const int vdim = var.mode == 2 ? dim : 1;
      if (lhs)
      {
         // var.mode == 2 for GRAD op, == 1 for Eval
         ufl.ctx.vdim = vdim; // should pass it as argument
      }

      if (var.type == TOK::TRIAL_FUNCTION)
      {
         if (dim == 2)
         {
            assert(false);
         }
         else
         {
            // -- Input field restrictions and basis actions --
            out << "\tdouble r_u0[NCOMP*D1D]; // X\n";
            out << "\treadDofsOffset3d<NCOMP, NDOFS/NCOMP, D1D>";
            out << "(data, elem, map_arg, xd_arg, r_u0);\n";
            //out << "\tdbg(\"r_u0:%f %f %f %f\",r_u0[0],r_u0[1],r_u0[2],r_u0[3]);\n";
         }
      }

      if (var.type == TOK::TEST_FUNCTION)
      {
         out << "\t// switch to transpose\n";
      }
      var_stack.push(&var);
   }
}

// *****************************************************************************
void KerCeedMult::rule_extra_status_rule_eval_xt(Rule&) const
{
   dbg();
   assert(false); // no eval
}

// *****************************************************************************
void KerCeedMult::rule_grad_expr_grad_op_form_args(Rule&) const
{
   dbg();
   assert(dim == 3);
   assert(!var_stack.empty());
   xfl::var &var = *(var_stack.top());
   out << "\t// Grad(" << var.name << ")\n";
   if (lhs)
   {
      // EvalMode: gradient
      out << "\t// LHS grad\n";
      out << "\tdouble r_t0[NCOMP*DIM*Q1D]; // Grad(X) @ Q points\n";
      out << "\tgrad3d<NCOMP,D1D,Q1D>(data, r_u0, s_B_in_0, s_G_in_0, r_t0);\n";
      out << "\t//if (elem==0) for(int k=0; k<(DIM*Q1D); k++) { dbg(\"r_t0[%d]:%f\",k,r_t0[k]);}\n";
   }
   else // transpose
   {
      out << "\t// RHS grad\n";
      out << "\tCeedScalar r_v0[NCOMP*D1D];\n";
      out << "\tgradTranspose3d<NCOMP,D1D,Q1D>(data, r_tt0, s_B_out_0, s_G_out_0, r_v0);\n";

      out << "\twriteDofsOffset3d<NCOMP, NDOFS/NCOMP, D1D>";
      out << "(data, elem, map_arg, r_v0, yd_arg);\n";
   }
   var_stack.pop();
   out << "\t// [ pop] " << var.name << "\n";
   ops_stack.push(TOK::GRAD_OP);
}

// *****************************************************************************
void KerCeedMult::token_COMA(Token&) const
{
   const int op = ops_stack.top();
   if (op == TOK::GRAD_OP) // Grad operation
   {
      out << "\t// COMA pull grad\n";
      out << "\tdouble r_tt0[DIM*Q1D]; // Grad(X).DX\n";
   }

   ops_stack.pop();

   out << "\n\t// Q function\n";
   out << "\tfor (CeedInt q = 0; q < Q1D; q++) {\n";
   out << "\t\tconst CeedScalar u[3] =\n";
   //out << "\t\t\t{0.0163791, 0.0163791, 0.0163791};\n";
   out << "\t\t\t{r_t0[q+0*Q1D], r_t0[q+1*Q1D], r_t0[q+2*Q1D]};\n";
   //out << "\t\t\t{r_t0[3*q+0], r_t0[3*q+1], r_t0[3*q+2]};\n";
   out << "\t\tconst double *qdata = &DX(0,0,tidx,tidy,q,elem);\n";
   out << "\t\tCeedScalar v[3];\n";
   out << "\t\tkernels::Mult(DX0, DX1, qdata, u, v);\n";
   out << "\t\tr_tt0[q+0*Q1D] = v[0]; r_tt0[q+1*Q1D] = v[1]; r_tt0[q+2*Q1D] = v[2];\n";
   //out << "\t\tr_tt0[3*q+0] = v[0]; r_tt0[3*q+1] = v[1]; r_tt0[3*q+2] = v[2];\n";
   out << "\t\tconst bool dump = elem==0;// && tidx==0 && tidy==0 && q==0;\n";
   out << "\t\tif (dump) { dbg(\"u: %f %f %f\",u[0],u[1],u[2]);}\n";
   out << "\t\tif (dump) { dbg(\"v: %f %f %f\",v[0],v[1],v[2]);}\n";
   out << "\t} // Q function\n";
   out << "\n";
   //out << "assert(false);\n";

   if (op == TOK::GRAD_OP)
   {
      out << "\t//COMA push grad\n";
   }
}

// *****************************************************************************
/* anonymous */ namespace
{

const char *deviceFunctions =
   QUOTE(
      using CeedScalar = double;
      using CeedInt = int;

      typedef struct
{
   CeedInt tidx;
   CeedInt tidy;
   CeedInt tidz;
   CeedInt tid;
   CeedScalar* slice;
} BackendData;

//------------------------------------------------------------------------------
// Load matrices for basis actions
//------------------------------------------------------------------------------
template <int P, int Q> inline DEVICE
void loadMatrix(BackendData &data, const CeedScalar* d_B, CeedScalar* B)
{
   for (CeedInt i = 0;//data.tid;
        i < P*Q;
        i += 1/*BLOCK_DIM(x)*BLOCK_DIM(y)*BLOCK_DIM(z)*/)
   {
      B[i] = d_B[i];
   }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d> inline DEVICE
void readDofsOffset3d(BackendData &data,
                      const CeedInt elem,
                      const CeedInt* indices,
                      const CeedScalar* d_u,
                      CeedScalar* r_u) // NCOMP*D1D
{
   if (data.tidx < P1d && data.tidy < P1d)
   {
      for (CeedInt dz = 0; dz < P1d; ++dz)
      {
         const int dx = data.tidx;
         const int dy = data.tidy;
         const CeedInt node = dx + dy*P1d + dz*P1d*P1d;
         const CeedInt gid = indices[node + elem * P1d*P1d*P1d];
         assert(gid >= 0);
         for (CeedInt comp = 0; comp < NCOMP; ++comp)
         {
            assert(comp == 0);
            //r_u[dz + comp*P1d] = d_u[gid + COMPSTRIDE * comp];
            const int idx = dz + comp*P1d;
            const double value = d_u[gid + COMPSTRIDE * comp];
            r_u[idx] = value;
            dbg("r_u[%d]:%f (%d)", idx, value, gid);
         }
      }
   }
}

//------------------------------------------------------------------------------
// 3D tensor contract X
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d> inline DEVICE
void ContractX3d(BackendData &data,
                 const CeedScalar *U,
                 const CeedScalar *B, // B or G
                 CeedScalar *V)
{
   CeedScalar r_B[P1d];
   //dbg("data.tidx:%d P1d:%d", data.tidx, P1d);
   for (CeedInt i = 0; i < P1d; ++i) { r_B[i] = B[i + data.tidx * P1d]; }
   /*for (CeedInt i = 0; i < P1d; ++i)
   {
      dbg("r_B[%d(%d)]: %f(%f)", i, i + data.tidx * P1d,
          r_B[i], B[i + data.tidx * P1d]);
   }*/
   for (CeedInt k = 0; k < P1d; ++k)
   {
      data.slice[data.tidx + data.tidy * T1D] = U[k];
      //dbg("#%d%d U[%d]:%f",data.tidx, data.tidy, k, U[k]);
      SYNC_THREADS;
      V[k] = 0.0;
      if (data.tidx < Q1d && data.tidy < P1d)
         for (CeedInt i = 0; i < P1d; ++i)
         {
            //dbg("data:%f", data.slice[/*i*/data.tidx * T1D]);
            V[k] += r_B[i] * data.slice[i/*data.tidx*/ + data.tidy * T1D];
         }
      SYNC_THREADS;
   }
}

//------------------------------------------------------------------------------
// 3D tensor contract Y
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d> inline DEVICE
void ContractY3d(BackendData &data,
                 const CeedScalar *U,
                 const CeedScalar *B,
                 CeedScalar *V)
{
   CeedScalar r_B[P1d];
   for (CeedInt i = 0; i < P1d; ++i) { r_B[i] = B[i + data.tidy * P1d]; }
   for (CeedInt k = 0; k < P1d; ++k)
   {
      data.slice[data.tidx + data.tidy * T1D] = U[k];
      SYNC_THREADS;
      V[k] = 0.0;
      if (data.tidx < Q1d && data.tidy < Q1d)
         for (CeedInt i = 0; i < P1d; ++i)
         {
            //dbg("data:%f", data.slice[data.tidx + /*i*/data.tidy * T1D]);
            V[k] += r_B[i] * data.slice[data.tidx + i/*data.tidy*/ * T1D];
         }
      SYNC_THREADS;
   }
}

//------------------------------------------------------------------------------
// 3D tensor contract Z
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d> inline DEVICE
void ContractZ3d(BackendData& data,
                 const CeedScalar *U,
                 const CeedScalar *B,
                 CeedScalar *V)
{
   for (CeedInt k = 0; k < Q1d; ++k)
   {
      V[k] = 0.0;
      if (data.tidx < Q1d && data.tidy < Q1d)
         for (CeedInt i = 0; i < P1d; ++i)
         {
            V[k] += B[i + k*P1d] * U[i];   // Contract z direction
         }
   }
}

//------------------------------------------------------------------------------
// 3D derivatives at quadrature points
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d> inline DEVICE
void grad3d(BackendData &data,
            const CeedScalar *__restrict__ r_U, // NCOMP*D1D
            const CeedScalar *c_B,
            const CeedScalar *c_G,
            CeedScalar *__restrict__ r_V) // NCOMP*DIM*Q1D
{
   //dbg();
   CeedScalar r_t1[P1d];
   CeedScalar r_t2[P1d];
   assert(NCOMP==1);
   for (CeedInt comp = 0; comp < NCOMP; comp++)
   {
      assert(comp==0);
      // GBB
      //for (int d = 0; d < P1d; d++) { dbg("r_U:%f",r_U[d]); }
      ContractX3d<NCOMP, P1d, Q1d>(data, r_U + comp*P1d, c_G, r_t1);
      //for (int d = 0; d < P1d; d++) { dbg("r_t1:%f",r_t1[d]); }
      //assert(false);
      ContractY3d<NCOMP, P1d, Q1d>(data, r_t1, c_B, r_t2);
      //for (int d = 0; d < P1d; d++) { dbg("r_t2:%f",r_t2[d]); }
      ContractZ3d<NCOMP, P1d, Q1d>(data, r_t2, c_B, r_V + comp*Q1d + 0*NCOMP*Q1d);
      //for (int q = 0; q < Q1d; q++) { dbg("r_V:%f", r_V[comp*Q1d + 0*NCOMP*Q1d + q]); }
      //assert(false);

      // BGB
      ContractX3d<NCOMP, P1d, Q1d>(data, r_U + comp*P1d, c_B, r_t1);
      //for (int d = 0; d < P1d; d++) { dbg("r_t1:%f",r_t1[d]); }
      ContractY3d<NCOMP, P1d, Q1d>(data, r_t1, c_G, r_t2);
      //for (int d = 0; d < P1d; d++) { dbg("r_t2:%f",r_t2[d]); }
      ContractZ3d<NCOMP, P1d, Q1d>(data, r_t2, c_B,
                                   r_V + comp*Q1d + 1*NCOMP*Q1d);
      //for (int q = 0; q < Q1d; q++) { dbg("r_V:%f", r_V[comp*Q1d + 1*NCOMP*Q1d + q]); }
      //assert(false);

      // BBG
      ContractX3d<NCOMP, P1d, Q1d>(data, r_U + comp*P1d, c_B, r_t1);
      ContractY3d<NCOMP, P1d, Q1d>(data, r_t1, c_B, r_t2);
      ContractZ3d<NCOMP, P1d, Q1d>(data, r_t2, c_G,
                                   r_V + comp*Q1d + 2*NCOMP*Q1d);
      //for (int q = 0; q < Q1d; q++) { dbg("r_V:%f",r_V[comp*Q1d + 2*NCOMP*Q1d + q]); }
      //assert(false);
   }
   //assert(false);
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract Z
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d>
inline DEVICE void ContractTransposeZ3d(BackendData& data,
                                        const CeedScalar *U,
                                        const CeedScalar *B, CeedScalar *V)
{
   for (CeedInt k = 0; k < P1d; ++k)
   {
      V[k] = 0.0;
      if (data.tidx < Q1d && data.tidy < Q1d)
         for (CeedInt i = 0; i < Q1d; ++i)
         {
            V[k] += B[k + i*P1d] * U[i];   // Contract z direction
         }
   }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract Y
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d> inline DEVICE
void ContractTransposeY3d(BackendData& data,
                          const CeedScalar *U,
                          const CeedScalar *B,
                          CeedScalar *V)
{
   CeedScalar r_B[Q1d];
   //for (CeedInt i = 0; i < P1d; ++i) { r_B[i] = B[data.tidy + i*P1d]; }
   for (CeedInt i = 0; i < Q1d; ++i)
   {
      assert(data.tidy < P1d);
      r_B[i] = B[data.tidy + i*P1d];
   }
   for (CeedInt k = 0; k < P1d; ++k)
   {
      data.slice[data.tidx + data.tidy*T1D] = U[k];
      SYNC_THREADS;
      V[k] = 0.0;
      if (data.tidx < Q1d && data.tidy < P1d)
         for (CeedInt i = 0; i < Q1d; ++i)
         {
            V[k] += r_B[i] * data.slice[data.tidx + i*T1D];
         }
      SYNC_THREADS;
   }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract X
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d> inline DEVICE
void ContractTransposeX3d(BackendData &data,
                          const CeedScalar *U,
                          const CeedScalar *B,
                          CeedScalar *V)
{
   CeedScalar r_B[Q1d];
   for (CeedInt i = 0; i < P1d; ++i) { r_B[i] = B[data.tidx + i * P1d]; }
   //for (CeedInt i = 0; i < Q1d; ++i) { r_B[i] = B[data.tidx + i*P1d]; }
   for (CeedInt k = 0; k < P1d; ++k)
   {
      data.slice[data.tidx + data.tidy*T1D] = U[k];
      SYNC_THREADS;
      V[k] = 0.0;
      if (data.tidx < P1d && data.tidy < P1d)
         for (CeedInt i = 0; i < Q1d; ++i)
         {
            V[k] += r_B[i] * data.slice[i + data.tidy*T1D];
         }
      SYNC_THREADS;
   }
}

//------------------------------------------------------------------------------
// 3D transpose tensor contract X, no FTZ on output V
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d> inline DEVICE
void ContractTransposeAddX3d(BackendData &data,
                             const CeedScalar *U,
                             const CeedScalar *B,
                             CeedScalar *V)
{
   CeedScalar r_B[Q1d];
   for (CeedInt i = 0; i < P1d; ++i) { r_B[i] = B[data.tidx + i * P1d]; }
   //for (CeedInt i = 0; i < Q1d; ++i) { r_B[i] = B[data.tidx + i * P1d]; }
   for (CeedInt k = 0; k < P1d; ++k)
   {
      data.slice[data.tidx + data.tidy * T1D] = U[k];
      SYNC_THREADS;
      if (data.tidx < P1d && data.tidy < P1d)
         for (CeedInt i = 0; i < Q1d; ++i)
         {
            V[k] += r_B[i] * data.slice[i + data.tidy * T1D];
         }
      SYNC_THREADS;
   }
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
template <int NCOMP, int P1d, int Q1d> inline DEVICE
void gradTranspose3d(BackendData &data,
                     const CeedScalar *__restrict__ r_U, // NCOMPxDIMxQ1D
                     const CeedScalar *c_B,
                     const CeedScalar *c_G,
                     CeedScalar *__restrict__ r_V) // NCOMPxD1D
{
   //dbg("slice: %p", data.slice);
   CeedScalar r_t1[P1d];
   CeedScalar r_t2[P1d];
   for (CeedInt comp = 0; comp < NCOMP; comp++)
   {
      ContractTransposeZ3d<NCOMP,P1d,Q1d>(data,r_U+comp*Q1d+0*NCOMP*Q1d,c_B,r_t1);
      ContractTransposeY3d<NCOMP,P1d,Q1d>(data,r_t1,c_B,r_t2);
      ContractTransposeX3d<NCOMP,P1d,Q1d>(data,r_t2,c_G,r_V + comp*P1d);
      ContractTransposeZ3d<NCOMP,P1d,Q1d>(data,r_U+comp*Q1d+1*NCOMP*Q1d,c_B,r_t1);
      ContractTransposeY3d<NCOMP,P1d,Q1d>(data,r_t1,c_G,r_t2);
      ContractTransposeAddX3d<NCOMP,P1d,Q1d>(data,r_t2,c_B,r_V + comp*P1d);
      ContractTransposeZ3d<NCOMP,P1d,Q1d>(data,r_U+comp*Q1d+2*NCOMP*Q1d,c_G,r_t1);
      ContractTransposeY3d<NCOMP,P1d,Q1d>(data,r_t1,c_B,r_t2);
      ContractTransposeAddX3d<NCOMP,P1d,Q1d>(data,r_t2,c_B,r_V + comp*P1d);
   }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
template <int NCOMP, int COMPSTRIDE, int P1d>
inline DEVICE void writeDofsOffset3d(BackendData &data,
                                     const CeedInt elem,
                                     const CeedInt* indices,
                                     const CeedScalar* r_v, CeedScalar* d_v)
{
   if (data.tidx < P1d && data.tidy < P1d)
   {
      for (CeedInt dz = 0; dz < P1d; ++dz)
      {
         const int dx = data.tidx;
         const int dy = data.tidy;
         const CeedInt node = dx + dy*P1d + dz*P1d*P1d;
         const CeedInt gid = indices[node + elem * P1d*P1d*P1d];
         assert(gid >= 0);
         for (CeedInt comp = 0; comp < NCOMP; ++comp)
         {
            //AtomicAdd(/*&*/d_v[ind + COMPSTRIDE * comp], r_v[z+comp*P1d]);
            //d_v[ind + COMPSTRIDE * comp] += r_v[z+comp*P1d];
            const int idx = gid + COMPSTRIDE * comp;
            const double value = r_v[dz + comp*P1d];
            d_v[idx] += value;
            //dbg("d_v[%d] %f", idx, value);
         }
      }
   }
}
         ); // QUOTE
} // anonymous namespace

/** @endcond */

} // internal

} // mfem
