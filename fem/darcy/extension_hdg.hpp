// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_EXTENSION_HDG
#define MFEM_EXTENSION_HDG

#include "../../config/config.hpp"
#include "../bilininteg.hpp"
#include "../coefficient.hpp"
#include "../eltrans.hpp"
#include "../gridfunc.hpp"
#include "../../mesh/mesh.hpp"

#include <functional>

namespace mfem
{

/** @brief Solving a boundary-value problem on a polyhedral subdomain of the
    true domain, by extension from that subdomain.

    The construction is Cockburn & Solano's. Given a domain @f$\Omega@f$ with a
    boundary @f$\Gamma@f$ that the mesh does not follow, one meshes a polyhedral
    subdomain @f$D_h \subset \Omega@f$ and solves there, with the Dirichlet
    datum given on @f$\Gamma@f$ transferred to the computational boundary
    @f$\Gamma_h = \partial D_h@f$ along a family of *transferring paths*. For
    the model problem

    @f[ \boldsymbol{u} + K \nabla p = 0, \qquad -\nabla\cdot\boldsymbol{u} = f
        \text{ in } \Omega, \qquad p = g \text{ on } \Gamma, @f]

    integrating the first equation along a path @f$\sigma@f$ running from
    @f$x \in \Gamma_h@f$ to @f$\bar{x} = a(x) \in \Gamma@f$ gives

    @f[ p(x) = g(\bar{x}) + \int_\sigma C\,\boldsymbol{u}\cdot\boldsymbol{m}\,ds,
        \qquad C = K^{-1}, @f]

    with @f$\boldsymbol{m}@f$ the unit tangent of @f$\sigma@f$ pointing from
    @f$x@f$ towards @f$\bar{x}@f$. The value is independent of the path because
    @f$C\boldsymbol{u} = -\nabla p@f$ is a gradient. The discrete method
    replaces @f$\boldsymbol{u}@f$ outside @f$D_h@f$ by the *extension*
    @f$E_h(\boldsymbol{u}_h)@f$ -- the polynomial of the element owning the
    face, evaluated outside it -- and takes the resulting

    @f[ \varphi_h(x) = g(a(x))
        + \int_\sigma C\,E_h(\boldsymbol{u}_h)\cdot\boldsymbol{m}\,ds @f]

    as the Dirichlet datum on @f$\Gamma_h@f$. The curved boundary is thereby
    reduced to the evaluation of line integrals; nothing depends on how
    @f$\Gamma@f$ is represented, on the space dimension, or on the method used
    inside @f$D_h@f$.

    The point of the construction is that the orders of convergence are those
    of the boundary-fitted method even when @f$\operatorname{dist}(\Gamma_h,
    \Gamma)@f$ is only @f$O(h)@f$, where earlier techniques needed
    @f$O(h^{k+1})@f$.

    References:
    - Cockburn, B., & Solano, M. (2012). Solving Dirichlet boundary-value
      problems on curved domains by extensions from subdomains. SIAM J. Sci.
      Comput. 34(1), A497-A519.
    - Cockburn, B., & Solano, M. (2014). Solving convection-diffusion problems
      on curved domains by extensions from subdomains. J. Sci. Comput. 59,
      512-543. The lifting is unchanged by the convection: it integrates the
      constitutive law, in which the convective field does not appear.
    - Cockburn, B., Qiu, W., & Solano, M. (2014). A priori error analysis for
      HDG methods using extensions from subdomains to achieve boundary
      conformity. Math. Comp. 83(286), 665-699. */

/// A scalar function of the physical position alone.
/** The level set and the Dirichlet datum are both evaluated at points that lie
    *outside* the mesh, where a Coefficient cannot be evaluated: a Coefficient
    reaches its position through an ElementTransformation, and no element
    contains these points. They are therefore taken as functions of position. */
using PositionFunction = std::function<real_t(const Vector &)>;

/// A vector-valued function of the physical position alone.
using VectorPositionFunction = std::function<void(const Vector &, Vector &)>;


/** @brief A family of transferring paths joining the computational boundary
    @f$\Gamma_h@f$ to the true boundary @f$\Gamma@f$.

    Each path is the straight segment from a point @f$x@f$ of @f$\Gamma_h@f$ to
    its image @f$a(x)@f$ on @f$\Gamma@f$, so a family is determined by the map
    @f$a@f$ alone. Two conditions are needed for the analysis to apply, and
    both should be checked when a new family is written:
    - @f$(a(x) - x)\cdot n_e > 0@f$ on every boundary face @f$e@f$, with
      @f$n_e@f$ the outward normal;
    - the paths do not cross before reaching @f$\Gamma@f$. */
class TransferPath
{
public:
   /** @brief The endpoint @f$a(x) \in \Gamma@f$ of the path issuing from the
       point @a x of @f$\Gamma_h@f$, whose outward unit normal is @a n. */
   virtual void Endpoint(const Vector &x, const Vector &n,
                         Vector &xbar) const = 0;

   /** @brief The endpoint at an integration point of a boundary face.

       The default implementation evaluates the physical point and the outward
       unit normal and defers to Endpoint(x, n). A family that depends on the
       face itself -- on its vertices, or on data precomputed per face --
       overrides this instead.

       @a FTr must have its integration points set to @a ip already. */
   virtual void Endpoint(FaceElementTransformations &FTr,
                         const IntegrationPoint &ip, Vector &xbar) const;

   virtual ~TransferPath() = default;
};


/** @brief The path family of a boundary with a closed-form closest-point map.

    When @f$\Gamma@f$ is an analytic surface whose closest point is known in
    closed form, the whole path construction collapses to that map: for a
    sphere of centre @f$c@f$ and radius @f$R@f$,
    @f$a(x) = c + R(x-c)/|x-c|@f$. This is worth having directly and not only
    as a fast path through the general construction. */
class ClosestPointPath : public TransferPath
{
   VectorPositionFunction cp;

public:
   /// @param cp_  the closest-point map onto @f$\Gamma@f$.
   ClosestPointPath(VectorPositionFunction cp_) : cp(std::move(cp_)) { }

   /// The closest-point map onto a sphere of centre @a c and radius @a R.
   static VectorPositionFunction Sphere(const Vector &c, real_t R);

   using TransferPath::Endpoint;

   void Endpoint(const Vector &x, const Vector &n,
                 Vector &xbar) const override { cp(x, xbar); }
};


/** @brief The path family issuing along the outward normal of @f$\Gamma_h@f$,
    terminated on the zero level set by bisection.

    This is the family the a priori error analysis is written for: its paths
    are parallel to the face normal, which satisfies the sign condition
    @f$(a(x)-x)\cdot n_e > 0@f$ by construction and, for a boundary that is
    resolved by the mesh, does not cross. It needs no representation of
    @f$\Gamma@f$ beyond the level set that selected the subdomain. */
class LevelSetPath : public TransferPath
{
   PositionFunction phi;
   real_t search_length;
   int search_steps;
   real_t tol;
   int max_iter;

public:
   /** @param phi_            the level set: negative inside @f$\Omega@f$, zero
                              on @f$\Gamma@f$, positive outside.
       @param search_length_  how far along the normal to look for the sign
                              change. It should exceed the largest expected
                              distance from @f$\Gamma_h@f$ to @f$\Gamma@f$ and
                              is normally a small multiple of the mesh size.
       @param search_steps_   how many equal steps the search interval is cut
                              into while bracketing the sign change. More than
                              one is needed only where @f$\Gamma@f$ can be
                              crossed twice within @a search_length_. */
   LevelSetPath(PositionFunction phi_, real_t search_length_,
                int search_steps_ = 1, real_t tol_ = 1e-13,
                int max_iter_ = 100)
      : phi(std::move(phi_)), search_length(search_length_),
        search_steps(search_steps_), tol(tol_), max_iter(max_iter_) { }

   using TransferPath::Endpoint;

   void Endpoint(const Vector &x, const Vector &n,
                 Vector &xbar) const override;
};


/** @brief The general path family: a direction chosen at each vertex of
    @f$\Gamma_h@f$ by a search, and interpolated along the faces.

    The two closed-form families above are enough where @f$\Gamma@f$ has a
    closest-point map, or where every face normal reaches it. Neither holds in
    general, and the Joukowsky airfoil of CS-Extensions §3.4 is where both give
    out. Measured there: **between four and fifteen per cent of the face
    normals never meet @f$\Gamma@f$ at all**, passing outside the thin tail or
    the nose, so LevelSetPath has no endpoint to return; and a family following
    each face's own normal does not tile the region beyond @f$\Gamma_h@f$,
    adjacent faces disagreeing on the path through the vertex they share.

    This is CS-Extensions' own construction, and it repairs both at once. A
    direction is chosen at each *vertex* of @f$\Gamma_h@f$ by searching a fan
    of rays for the one reaching @f$\Gamma@f$ nearest, restricted to the
    directions that leave @f$D_h@f$ through both faces meeting there; along a
    face, the unit tangents of its two vertices are interpolated. The search is
    what makes a path exist where a normal would miss, and the sharing of the
    vertex path is what makes the swept regions tile — which is the property
    the vertex-first construction is *for*, and which reading it as merely a
    way to find a nearby point on @f$\Gamma@f$ misses.

    Two dimensions only. The construction generalises, and the reference says
    so, but nothing here has been run in three.

    Must be evaluated through the FaceElementTransformations overload: a
    direction interpolated along a face is not a function of the point alone.
    The mesh must outlive the path. */
class VertexConePath : public TransferPath
{
   const Mesh *mesh;
   PositionFunction phi;
   real_t search_length;
   int n_rays, n_keep, search_steps, max_iter;
   real_t tol;

   /// Per face, the unit tangents of its two vertices, in the order the face's
   /// reference coordinate visits them; four entries per face.
   Array<real_t> tang;
   Array<int> has_tangent;
   int n_widened{};

   /// Choose the direction at one vertex, given the outward normals of the
   /// faces of @f$\Gamma_h@f$ meeting there. Returns false if no ray reached
   /// @f$\Gamma@f$ even after the admissible fan was widened.
   bool VertexDirection(const Vector &x, const Array<real_t> &normals,
                        Vector &t);

public:
   /** @param mesh_           the mesh of @f$D_h@f$.
       @param gamma_h_attr    the boundary attribute of @f$\Gamma_h@f$.
       @param phi_            the level set, negative inside @f$\Omega@f$.
       @param search_length_  how far along a ray to look for @f$\Gamma@f$.
       @param n_rays_         rays in the fan searched at each vertex.
       @param n_keep_         how many of the nearest endpoints found to
                              average before shooting once more along their
                              mean, which is what the reference does and what
                              keeps the direction from jumping between two
                              nearly equal rays. */
   VertexConePath(const Mesh &mesh_, int gamma_h_attr, PositionFunction phi_,
                  real_t search_length_, int n_rays_ = 16, int n_keep_ = 3,
                  int search_steps_ = 32, real_t tol_ = 1e-13,
                  int max_iter_ = 100);

   using TransferPath::Endpoint;

   /** @brief Not available: this family is defined face by face.

       The base class's point-and-normal form cannot express an interpolated
       direction, and silently falling back to the normal would reintroduce
       exactly the two failures this family exists to repair. */
   void Endpoint(const Vector &x, const Vector &n, Vector &xbar) const override;

   void Endpoint(FaceElementTransformations &FTr, const IntegrationPoint &ip,
                 Vector &xbar) const override;

   /** @brief How many vertices needed the admissible fan widened past the
       condition that the path leave @f$D_h@f$ through both adjacent faces.

       Non-zero means Assumption P.1 of the analysis is violated somewhere, so
       it is worth reporting rather than hiding: the method may still run, and
       the estimate no longer covers it. */
   int NumWidened() const { return n_widened; }
};


/** @brief The extension operator @f$E_h@f$: an element's own polynomial,
    evaluated outside the element.

    Mathematically trivial -- @f$E_h(q_h)|_{K^{ext}}(y) := q_h|_K(y)@f$ -- and
    the only thing to get right is that the reference coordinates of a point
    outside the element are not clamped to the reference element.
    ElementTransformation::TransformBack() does clamp: it uses
    InverseElementTransformation's default #NewtonElementProject solver, which
    projects every iterate back into the reference element, so a point outside
    silently comes back as the nearest point of the element boundary and the
    extension degenerates into a constant. This class configures the
    unrestricted #Newton solver instead.

    For an affine element the composition of the reference basis with the
    inverse map is a polynomial in the physical coordinates, so this is the
    polynomial extension the method is written against. For an element with a
    non-affine map -- a general quadrilateral, hexahedron or wedge -- it is the
    reference-space extension instead, which is the natural generalisation but
    not the same object. */
class ElementExtension
{
   mutable InverseElementTransformation inv_tr;

public:
   ElementExtension();

   /// Set the element whose polynomials are to be extended.
   void SetElement(ElementTransformation &Tr) { inv_tr.SetTransformation(Tr); }

   /** @brief Reference coordinates of the physical point @a y under the
       element map, not restricted to the reference element.

       Returns false if the Newton solve failed to converge, in which case
       @a ip is not meaningful. Note that the element transformation's own
       integration point is modified by the solve. */
   bool TransformBack(const Vector &y, IntegrationPoint &ip) const;
};


/** @brief The line integral @f$\int_\sigma C\,\boldsymbol{u}\cdot
    \boldsymbol{m}\,ds@f$ along the straight path from @a x to @a xbar.

    @param Cu       evaluates @f$C(y)\,\boldsymbol{u}(y)@f$ at a physical point
                    @a y, which lies outside the mesh. Its output is passed in
                    already sized, as MFEM's vector functions expect.
    @param line_ir  a rule on Geometry::SEGMENT. On a straight-sided element the
                    integrand is a polynomial of the degree of the flux space,
                    so a rule exact to that degree integrates it exactly; the
                    outer integral over the face is the one that is not exact,
                    because @f$a(x)@f$ is not polynomial in @f$x@f$.

    This is the whole of the lifting apart from the datum @f$g(a(x))@f$, and it
    is stated as a free function because that makes it checkable on its own:
    fed the exact flux, it must return @f$p(x) - p(a(x))@f$ to quadrature
    accuracy, whatever the path. */
real_t PathIntegral(const VectorPositionFunction &Cu, const Vector &x,
                    const Vector &xbar, const IntegrationRule &line_ir);


/** @brief The Dirichlet datum of @f$\Gamma@f$, seen from @f$\Gamma_h@f$.

    Evaluates @f$g(a(x))@f$ at a point @a x of a boundary face, which is the
    part of the lifting that does not depend on the solution. Paired with
    VectorBoundaryFluxLFIntegrator it supplies @f$\langle g\circ a, v\cdot
    n\rangle_e@f$, which is exactly how the miniapps already give a Dirichlet
    datum to the flux equation -- the only difference being where the datum is
    read from.

    Must be evaluated on a FaceElementTransformations, since the path family
    may need the face normal. */
class PathTraceCoefficient : public Coefficient
{
   const TransferPath &path;
   PositionFunction g;
   mutable Vector xbar;

public:
   PathTraceCoefficient(const TransferPath &path_, PositionFunction g_)
      : path(path_), g(std::move(g_)) { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};


/** @brief The solution-dependent part of the transferred Dirichlet datum, as a
    contribution to the flux mass form on a face of @f$\Gamma_h@f$.

    The transferred datum is @f$\varphi_h = g\circ a + L_e(\boldsymbol{u}_h)@f$
    and the flux equation reads it as @f$\langle \varphi_h, v\cdot n
    \rangle_e@f$. Only the first term is data; the second depends on the
    unknown flux, so it belongs on the left, and this integrator is it:

    @f[ \langle L_e(\boldsymbol\varphi_j), \boldsymbol\varphi_i \cdot n
        \rangle_e, \qquad
        L_e(\boldsymbol\varphi_j)(x) = \int_{\sigma(x)}
        C\,E_h(\boldsymbol\varphi_j)\cdot\boldsymbol{m}\,ds. @f]

    **The block is element-local.** A face of @f$\Gamma_h@f$ belongs to one
    element, and the extension @f$E_h@f$ on the region beyond it is that
    element's own polynomial, so nothing outside the element is read. The term
    is therefore an addition to the element's flux mass block and leaves the
    hybridization -- the constraint, its transpose, the static condensation --
    untouched. That is why the weak boundary route costs nothing structural
    here and the essential-trace route would.

    @a C is the same coefficient the flux mass form carries: the *inverse* of
    the diffusion tensor, as in @f$(C\boldsymbol{u}, v)@f$. It is evaluated at
    points outside the element, through that element's transformation, so a
    coefficient defined by a function of position extends as that function and
    one defined by a grid function extends as that grid function's own
    polynomial.

    The space is assumed fully discontinuous with @a vdim equal to the space
    dimension -- the arrangement `VectorMassIntegrator` also assumes, and the
    one the HDG configuration of `DarcyForm` uses. */
class HDGExtensionIntegrator : public BilinearFormIntegrator
{
   const TransferPath &path;
   Coefficient *C{};
   MatrixCoefficient *MC{};
   real_t sign;
   int line_order;

   Vector shape, shape_ext, nor, x, xbar, m, y, CTm;
   DenseMatrix Cmat, L;
   ElementExtension ext;

public:
   /** @param path_        the transferring paths.
       @param C_           the inverse diffusion tensor, as a scalar.
       @param sign_        **the negative of the sign with which the datum
                           enters the flux right-hand side**, since that is
                           where the term comes from. The default, `+1`, is
                           therefore the one to use with the branch's
                           convention of giving the *negated* potential to
                           VectorBoundaryFluxLFIntegrator, as `pNatural` does
                           in the harnesses. Measured rather than argued: with
                           the sign reversed the solve diverges outright, the
                           error growing by four orders between the two finest
                           meshes.
       @param line_order_  order of the rule along the path; negative takes
                           twice the element order plus two. */
   HDGExtensionIntegrator(const TransferPath &path_, Coefficient &C_,
                          real_t sign_ = +1., int line_order_ = -1)
      : path(path_), C(&C_), sign(sign_), line_order(line_order_) { }

   HDGExtensionIntegrator(const TransferPath &path_, MatrixCoefficient &C_,
                          real_t sign_ = +1., int line_order_ = -1)
      : path(path_), MC(&C_), sign(sign_), line_order(line_order_) { }

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   using BilinearFormIntegrator::AssembleFaceMatrix;
};


/** @brief A point of the region beyond a face of @f$\Gamma_h@f$, with the
    quadrature weight that goes with it. */
struct ExtensionPoint
{
   const Vector &y;                  ///< the physical point, outside @f$D_h@f$
   const Vector &xbar;               ///< the end @f$a(x)@f$ of its path
   const IntegrationPoint &face_ip;  ///< where that path leaves @f$\Gamma_h@f$
   real_t t;                         ///< along the path; 0 on @f$\Gamma_h@f$
   real_t weight;                    ///< quadrature weight, Jacobian included
};


/** @brief Quadrature over @f$K^{ext}_e@f$, the region swept by the paths
    issuing from one face of @f$\Gamma_h@f$.

    The region is the image of the face times the unit interval under

    @f[ y(\xi,t) = x(\xi) + t\,(a(x(\xi)) - x(\xi)), @f]

    and integrating in @f$(\xi,t)@f$ against the Jacobian of that map is
    integrating over @f$K^{ext}_e@f$. The Jacobian's columns are
    @f$(1-t)\,\partial x/\partial\xi_i + t\,\partial a/\partial\xi_i@f$ and
    @f$a(x)-x@f$; the derivative of @f$a@f$ along the face is taken by central
    differences, because a path family is a map and not required to supply one.

    Summed over the faces this is a quadrature over @f$D_h^c = \Omega\setminus
    D_h@f$ **provided the regions tile it**, which is a property of the path
    family and not of this routine: adjacent faces must agree on the path
    through a shared vertex. The closest-point map does agree, since it depends
    on the point and not on the face. A family whose paths follow each face's
    own normal does not, and leaves gaps and overlaps at the vertices -- which
    is why CS-Extensions builds its general family by interpolating the paths
    of the vertices rather than taking each face's normal.

    **The tiling is checkable and should be checked**: the weights sum to
    @f$|K^{ext}_e|@f$, so summing them over the faces must give
    @f$|\Omega| - |D_h|@f$.

    @param FTr       a boundary face of @f$\Gamma_h@f$.
    @param path      the transferring paths.
    @param face_ir   a rule on the face.
    @param line_ir   a rule on Geometry::SEGMENT, for the path direction.
    @param visit     called once per quadrature point. */
void ExtensionRegionQuadrature(
   FaceElementTransformations &FTr, const TransferPath &path,
   const IntegrationRule &face_ir, const IntegrationRule &line_ir,
   const std::function<void(const ExtensionPoint &)> &visit,
   real_t fd_step = 1e-6);


/** @brief Selection of the polyhedral subdomain @f$D_h@f$ of a background mesh.

    @f$D_h@f$ is the set of elements lying entirely inside @f$\Omega@f$. For a
    convex @f$\Omega@f$ this is decided by the vertices alone -- a convex
    element is inside if and only if all its vertices are -- and that is what
    @a extra_refine = 0 tests. For a domain that is not convex an element can
    have every vertex inside and still be cut by @f$\Gamma@f$, and additional
    points have to be tested; @a extra_refine >= 1 additionally samples the
    lattice of the reference element refined that many times.

    @param mesh         the background mesh.
    @param phi          the level set: negative inside @f$\Omega@f$.
    @param offset       an element is taken only if @f$\phi \le -@f$@a offset
                        at every point tested, which pushes @f$\Gamma_h@f$
                        further from @f$\Gamma@f$. Meaningful as a distance
                        only where @a phi is a signed distance. Used to set
                        @f$\operatorname{dist}(\Gamma_h,\Gamma)@f$ deliberately,
                        as the robustness studies of the papers do.
    @param marker       set to 1 for the elements of @f$D_h@f$ and 0 otherwise.
    @param extra_refine how many times to refine the sampling lattice; 0 tests
                        the vertices only, which is exact for a convex
                        @f$\Omega@f$.
    @returns the number of elements selected. */
int MarkLevelSetSubdomain(const Mesh &mesh, const PositionFunction &phi,
                          real_t offset, Array<int> &marker,
                          int extra_refine = 0);

} // namespace mfem

#endif
