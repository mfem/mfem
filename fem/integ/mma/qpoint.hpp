// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

/** @file qpoint.hpp
    Generic device quadrature-point primitives for H1 simplex spaces on affine
    meshes: the building blocks of a partial assembly whose physics lives
    outside the library, e.g. a stabilized (SUPG/PSPG/LSIC/VMS) flow
    NonlinearFormIntegrator that owns its own quadrature-point state.

    ## Layouts
    All arrays are column-major (first index fastest), as GeometricFactors
    and QuadratureInterpolator with QVectorLayout::byNODES:
    - E-vectors x, y, diag:    (ND, VDIM, NE), in NATIVE dof order, i.e. the
      output of fes.GetElementRestriction(ElementDofOrdering::NATIVE), the
      E-vector ordering of simplex spaces (see GetEVectorOrdering).
    - values, Laplacians:      (NQ, VDIM, NE)
    - physical gradients:      (NQ, VDIM, DIM, NE), du(q,c,k,e) = du_c/dx_k
    - J, InvJ, Metric:         (NQ, DIM, DIM, NE), with
                               J(q,i,j,e) = dx_i/dxi_j and
                               InvJ(q,i,j,e) = dxi_i/dx_j
    - DetJ, WeightDetJ:        (NQ, NE)
    - reference tables:        B(q,i) (NQ, ND) and G(q,j,i) (NQ, DIM, ND)
      of DofToQuad::FULL, and the reference Hessian H(q,s,i) (NQ, NH, ND)
      with NH = DIM(DIM+1)/2 in symmetric storage, as FiniteElement::
      CalcHessian: s = (00, 01, 11) in 2D and (00, 01, 02, 11, 12, 22) in 3D.

    ## Affine elements
    On an affine simplex dxi/dx is constant in the element, so the chain rule
    has no term with derivatives of the Jacobian:
      grad u = InvJ^T grad_ref u,
      Hess u = InvJ^T H_ref InvJ,
      Lap u  = tr(InvJ^T H_ref InvJ) = sum_{a,b} H_ref(a,b) (InvJ InvJ^T)(a,b)
    per component, the off-diagonal entries of the symmetric storage counting
    twice. This matches FiniteElement::CalcPhysLinLaplacian. Meshes with
    curved nodes are rejected at construction; high-order nodes that
    describe straight elements are accepted (see IsAffine()).

    ## Engine
    Interpolation is one dense product U = P X with the stacked reference
    basis P = [B; G; H] (planes value, DIM gradients, NH Hessians) acting on
    every component of every element, through the simplex MMA batched GEMM
    (DMMA on CUDA, vendor GEMM with LAPACK on the host, else forall loops),
    followed by a pointwise map to physical quantities. The transposed
    contractions use the same P, the diagonals are pointwise loops.
*/

#include "mma.hpp"

namespace mfem
{

/** @brief Device quadrature-point primitives of an affine, fixed-order, nodal
    H1 triangle or tetrahedron space (H1_TriangleElement,
    H1_TetrahedronElement), any vdim.

    Typical use in a user NonlinearFormIntegrator, for an E-vector x:
    @code
       SimplexQPoint qp(fes, ir);              // at AssemblePA
       qp.Mult(x, SimplexQPoint::VALUES | SimplexQPoint::GRADIENTS |
               SimplexQPoint::LAPLACIANS, u, du, lap);
       // pointwise physics: dv, dg from u, du, lap, qp.Metric(), ...,
       // already multiplied by qp.WeightDetJ()
       qp.AddMultTranspose(dv, dg, y);         // y += B^T dv + G^T dg
    @endcode
    The quadrature data passed to AddMultTranspose() and AddDiagonal() must
    already include the quadrature weight and |J|, see WeightDetJ(). */
class SimplexQPoint
{
public:
   /// Quantities computed by Mult(), may be combined with '|'.
   enum EvalFlags
   {
      VALUES = 1 << 0,     ///< u at the quadrature points
      GRADIENTS = 1 << 1,  ///< physical gradients
      LAPLACIANS = 1 << 2  ///< physical Laplacians
   };

   /** @brief Set up the reference tables and the geometric factors of @a fes
       at @a ir. The mesh nodes are created if missing (Mesh::EnsureNodes);
       the elements must be affine, see IsAffine(). */
   SimplexQPoint(const FiniteElementSpace &fes, const IntegrationRule &ir);

   /** @brief True if the (local) elements of @a mesh are affine: no mesh
       nodes (the vertices only), mesh nodes of order 1, or higher-order
       nodes on a single simplex type that are, in every element, the P1
       interpolant of the element vertices, e.g. after SetCurvature(2) or
       SetNodalFESpace() on a straight mesh.

       The check is geometric, independent of any quadrature rule, and runs
       on the device: element e is affine when its nodes are at most
       1000 epsilon s_e away from their P1 interpolant, epsilon the machine
       epsilon of real_t and s_e the larger of the element size (largest
       distance between two vertices) and of the vertex norms, the scale of
       the roundoff in the nodes. Projected straight nodes, about epsilon
       s_e off, pass; curved elements do not. */
   static bool IsAffine(const Mesh &mesh);

   int Dim() const { return dim; }
   int VDim() const { return vdim; }
   int NumDofs() const { return nd; }
   int NumQuad() const { return nq; }
   int NumElements() const { return ne; }

   /// FULL reference tables B (NQ, ND) and G (NQ, DIM, ND).
   const DofToQuad &GetDofToQuad() const { return *maps; }

   /// Reference Hessian H(q,s,i), (NQ, NH, ND), symmetric storage.
   const Array<real_t> &GetRefHessian() const { return H; }

   /// Jacobians, (NQ, DIM, DIM, NE), from Mesh::GetGeometricFactors().
   const Vector &GetJ() const { return J; }
   /// Inverse Jacobians dxi/dx, (NQ, DIM, DIM, NE).
   const Vector &GetInvJ() const { return InvJ; }
   /// Jacobian determinants, (NQ, NE).
   const Vector &GetDetJ() const { return DetJ; }
   /// Quadrature weights times |J|, (NQ, NE).
   const Vector &GetWeightDetJ() const { return WDetJ; }

   /** @brief Element metric G(q,k,l,e), (NQ, DIM, DIM, NE), of the stabilized
       methods (tau_m, tau_c, tau_b):

         G = (dxi/dx)^T K (dxi/dx),  K = c (I + 1 1^T),

       c = 1/sqrt(3) on triangles and 2^(1/3)/2 on tetrahedra, i.e. K has 2c
       on the diagonal and c elsewhere (Pauli & Behr; Jansen et al.).

       Vertex ordering: with the barycentric coordinates lambda_0 = 1 - sum
       xi_j and lambda_j = xi_j, (dxi/dx)^T (I + 1 1^T) (dxi/dx) =
       sum_{a=0}^{DIM} grad lambda_a (grad lambda_a)^T. The sum runs over
       all the vertices, so G does not depend on how the element vertices
       are numbered. The tensor-product choice K = 4 I would single out
       vertex 0 and give a numbering-dependent G (the nodal permutation
       caveat of Liu et al., CMAME 2020); do not use it on simplices. */
   void Metric(Vector &G) const;

   /** @brief Interpolate the E-vector @a x (ND, VDIM, NE) at the quadrature
       points: @a u (NQ, VDIM, NE), @a du (NQ, VDIM, DIM, NE) and @a lap
       (NQ, VDIM, NE), each one only if requested in @a flags. The requested
       outputs are resized. */
   void Mult(const Vector &x, unsigned flags, Vector &u, Vector &du,
             Vector &lap) const;

   /** @brief y += B^T dv + G_phys^T dg, i.e. for each component c

         y(i,c,e) += sum_q [ phi_i dv(q,c,e) + sum_k dphi_i/dx_k dg(q,c,k,e) ]

       with @a dv (NQ, VDIM, NE) tested by values, @a dg (NQ, VDIM, DIM, NE)
       tested by physical gradients and @a y (ND, VDIM, NE). An empty @a dv
       or @a dg is skipped. */
   void AddMultTranspose(const Vector &dv, const Vector &dg, Vector &y) const;

   /** @brief diag += the diagonal of B^T Dvv B + B^T Dvg G + G^T Dgg G:

         diag(i,c,e) += sum_q [ phi_i^2 dvv(q,c,e)
                              + sum_k phi_i dphi_i/dx_k dvg(q,c,k,e)
                              + sum_{k,l} dphi_i/dx_k dgg(q,c,k,l,e)
                                          dphi_i/dx_l ]

       with @a dvv (NQ, VDIM, NE), @a dvg (NQ, VDIM, DIM, NE), @a dgg
       (NQ, VDIM, DIM, DIM, NE) and @a diag (ND, VDIM, NE): the (c,c) blocks
       of a vector operator. Both mixed terms B^T D1 G and G^T D2 B have the
       diagonal of B^T (D1 + D2) G: pass dvg = D1 + D2. An empty data vector
       is skipped. */
   void AddDiagonal(const Vector &dvv, const Vector &dvg, const Vector &dgg,
                    Vector &diag) const;

private:
   int dim, vdim, nd, nq, ne, nh;
   const DofToQuad *maps;
   Array<real_t> H; ///< (NQ, NH, ND)
   Array<real_t> P; ///< stacked [B; G; H], (NQ, 1 + DIM + NH, ND)
   Vector J, InvJ, DetJ, WDetJ;
   mutable Vector U, V; ///< quadrature scratch, (NQ * planes, VDIM NE)
};

} // namespace mfem
