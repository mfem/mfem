//                  MFEM Example 40 - Shared Code

#include "mfem.hpp"

using namespace std;
using namespace mfem;


inline real_t g1(real_t u);
inline real_t g2(real_t u);


class ModalBasis {
   private:
      DG_FECollection &fec;
      int dim;

      Array2D<int> ubdegs;
      Vector umc;
      DenseMatrix V;

      void ComputeUBDegs();
      void ComputeVDM();

   public:
      int order, npts;
      Geometry::Type &gtype;
      IntegrationRule solpts;

      ModalBasis(DG_FECollection &fec, Geometry::Type &gtype, int order, int dim);

      void SetSolution(Vector &u_elem);
      real_t Eval(Vector &x);
      Vector EvalGrad(Vector &x);

      ~ModalBasis() = default;
};


class ElementOptimizer {
   private:
      int dim, cfidx, nfaces;
      real_t dxi = 1E-4;
      real_t beta_0;
      real_t gbar;
      Geometry::Type &gtype;
      DenseMatrix n;


      void ComputeJacobian(Vector &x, Vector &J);
      
      void ComputeHessian(Vector &x, DenseMatrix &H);

      bool IsPD(DenseMatrix &M);

      void ComputeGeometryLevelset(Vector &x, Vector &s);

      void PrecomputeElementNormals();

      void ProjectAlongBoundary(Vector &x0, Vector &dx);

      void ProjectToElement(Vector &x0, Vector &dx);


   public:
      int ncon = 2;
      real_t eps = 1E-12;
      ModalBasis * MB;

      ElementOptimizer(ModalBasis * MB, int dim);

      void SetCostFunction(int cfidx);

      void SetGbar(real_t ubar);

      real_t g(real_t u);

      real_t h(real_t u);

      real_t Optimize(Vector &x0, bool extrapolate=true, int niters=3, int nbts=5);

      ~ElementOptimizer() = default;
};


ModalBasis::ModalBasis(DG_FECollection &fec_, Geometry::Type &gtype_, int order_, int dim_)
   : fec(fec_), gtype(gtype_), solpts(fec.FiniteElementForGeometry(gtype)->GetNodes()),
     order(order_), dim(dim_)
{
   // ASSERT INTERPOLATORY HERE??
   npts = solpts.GetNPoints();
   umc = Vector(npts);

   ComputeUBDegs();
   ComputeVDM();
}

// Computes matrix ubdegs of size (npts, dim) corresponding to the modal basis polynomial degrees.
void ModalBasis::ComputeUBDegs() {
   ubdegs = Array2D<int>(npts, dim);
   switch (gtype) {  
      // Segment: [0,1]
      case 1: {
         for (int i = 0; i < order + 1; i++) {
            ubdegs(i, 0) = i;
         }
         break;
      }
      // Triangle with vertices (0,0), (1,0), (0,1)
      case 2: {
         int n = 0;
         for (int i = 0; i < order + 1; i++) {
            for (int j = 0; j < order + 1 - i; j++) {
               ubdegs(n, 0) = i;
               ubdegs(n, 1) = j;
               n++;
            }
         }
         break;
      }
      // Quad: [0,1]^2
      case 3: {
         int n = 0;
         for (int i = 0; i < order + 1; i++) {
            for (int j = 0; j < order + 1; j++) {
               ubdegs(n, 0) = i;
               ubdegs(n, 1) = j;
               n++;
            }
         }
         break;
      }
      default:
         MFEM_ABORT("Element type not supported for continuously bounds-preserving limiting.")
   }
}

/*
Computes (and inverts) Vandermonde matrix for transformation from nodal basis to modal basis. Modal
basis uses Legendre polynomials, although not necessarily orthogonal for non-tensor product elements. 
*/
void ModalBasis::ComputeVDM() {
   V = DenseMatrix(npts);

   // Loop through solution nodes
   for (int i = 0; i < npts; i++) {
      // Compute nodal location in reference space
      Vector x(dim);
      solpts.IntPoint(i).Get(x, dim);

      // Compute L_k(x_i)*L_k(y_i)*L_k(z_i) for each modal basis function k corresponding to the 
      // polynomial degree in ubdegs.
      for (int j = 0; j < dim; j++) {
         Vector L(order+1);
         Poly_1D::CalcLegendre(order, x(j), L);
         for (int k = 0; k < npts; k++) {
            real_t v = L(ubdegs(k, j));
            V(i,k) = (j == 0) ? v : V(i,k)*v;
         }
      }
   }

   // Invert and store Vandermonde matrix
   V.Invert();
}

// Takes in vector of nodal DOFs, transforms to modal form, and stores in umc vector. 
void ModalBasis::SetSolution(Vector &u_elem) {
   MFEM_ASSERT(u_elem.Size() == npts, "Element-wise solution vector must be of same size as the modal basis.")

   V.Mult(u_elem, umc);
}

// Evalutes solution at arbitrary point x using modal basis
real_t ModalBasis::Eval(Vector &x) {
   MFEM_ASSERT(x.Size() == dim, "Modal basis can only be evaluated at one location at a time.")

   // Pre-compute L_i(x), L_i(y), L_i(z) for all degrees up to max polynomial order.  
   Array2D<real_t> L(order + 1, dim);
   for (int i = 0; i < dim; i++) {
      Vector Li(order+1);
      Poly_1D::CalcLegendre(order, x(i), Li);
      for (int j = 0; j < order + 1; j++) {
         L(j, i) = Li(j);
      }
   }

   // Compute u(x,y,z) as \sum L_i(x)*L_i(y)*L_i(z)
   real_t ux = 0;
   for (int i = 0; i < npts; i++) {
      real_t v = umc(i);
      for (int j = 0; j < dim; j++) {
         v *= L(ubdegs(i, j), j);
      }
      ux += v;
   }
   
   return ux;
}

// Evalutes solution gradient at arbitrary point x using modal basis
Vector ModalBasis::EvalGrad(Vector &x) {
   MFEM_ASSERT(x.Size() == dim, "Modal basis can only be evaluated at one location at a time.")

   // Pre-compute L_i(x), L_i(y), L_i(z) (and dL_i(x)/dx, etc.) for all degrees up to max polynomial order.  
   Array2D<real_t> L(order + 1, dim);
   Array2D<real_t> D(order + 1, dim);
   for (int i = 0; i < dim; i++) {
      Vector Li(order+1);
      Vector Di(order+1);
      Poly_1D::CalcLegendre(order, x(i), Li, Di);
      for (int j = 0; j < order + 1; j++) {
         L(j, i) = Li(j);
         D(j, i) = Di(j);
      }
   }

   // Compute du(x,y,z)/dx as \sum dL_i(x)/dx*L_i(y)*L_i(z)
   //         du(x,y,z)/dy as \sum L_i(x)*dL_i(y)/dy*L_i(z)
   //         du(x,y,z)/dz as \sum L_i(x)*L_i(y)*dL_i(z)/dz
   Vector gradu(dim);
   for (int d = 0; d < dim; d++) {
      real_t du = 0;
      for (int i = 0; i < npts; i++) {
         real_t v = umc(i);
         for (int j = 0; j < dim; j++) {
            if (j == d){ 
               v *= D(ubdegs(i, j), j);
            }
            else {
               v *= L(ubdegs(i, j), j);
            }
         }
         du += v;
      }
      gradu(d) = du;
   }
   
   return gradu;
}


ElementOptimizer::ElementOptimizer(ModalBasis * MB_, int dim_) 
   : MB(MB_), dim(dim_), gtype(MB_->gtype)
{
   // Generate and store matrix of face normals for the element
   PrecomputeElementNormals();
   nfaces = n.Height();

   // Set initial step size for GD + backtracking line search
   // as half the average distance between nodes
   beta_0 = 0.5/(MB->order + 1);
}


// Sets index of cost functional for g(real_t u)
void ElementOptimizer::SetCostFunction(int cfidx_) {
   cfidx = cfidx_;
}

// Computes and stores constraint functional of element-wise mean
void ElementOptimizer::SetGbar(real_t ubar) {
   gbar = g(ubar);
}

/*
Computes constraint functional based on index set by 
SetCostFunction(int cfidx_)
*/
real_t ElementOptimizer::g(real_t u) {
   switch (cfidx) {  
      case 0: 
         return g1(u);
      case 1: 
         return g2(u);
      default:
         MFEM_ABORT("Unknown constraint functional.")
   }
}

// Computes modified constraint functional
real_t ElementOptimizer::h(real_t u) {
   real_t gu = g(u);
   return gu > 0 ? gu/gbar : gu/(gbar - gu);
}

// Numerically computes Jacobian of h(u) with centered differences
void ElementOptimizer::ComputeJacobian(Vector &x, Vector &J) {
   for (int i = 0; i < dim; i++) {
      Vector x2 = Vector(x);
      
      x2(i) = x(i) + dxi; real_t hp = h(MB->Eval(x2));
      x2(i) = x(i) - dxi; real_t hm = h(MB->Eval(x2));
      J(i) = (hp - hm)/(2*dxi);
   }
}

// Numerically computes Hessian of h(u) with centered differences
void ElementOptimizer::ComputeHessian(Vector &x, DenseMatrix &H) {
   if (dim == 1) {
      Vector x2 = Vector(x);
      real_t h0 = h(MB->Eval(x2));

      x2(0) = x(0) + dxi; real_t hp = h(MB->Eval(x2));
      x2(0) = x(0) - dxi; real_t hm = h(MB->Eval(x2));
      H(0, 0) = (hp - 2*h0 + hm)/(dxi*dxi);
   }
   else if (dim == 2) {
      Vector x2 = Vector(x);
      real_t h0 = h(MB->Eval(x2));
      
      x2(0) = x(0) + dxi; real_t hp = h(MB->Eval(x2));
      x2(0) = x(0) - dxi; real_t hm = h(MB->Eval(x2));
      H(0, 0) = (hp - 2*h0 + hm)/(dxi*dxi);

      x2(0) = x(0);
      x2(1) = x(1) + dxi; hp = h(MB->Eval(x2));
      x2(1) = x(1) - dxi; hm = h(MB->Eval(x2));
      H(1,1) = (hp - 2*h0 + hm)/(dxi*dxi);

      x2(0) = x(0) + dxi;
      x2(1) = x(1) + dxi; real_t hpp = h(MB->Eval(x2));
      x2(0) = x(0) - dxi; real_t hmp = h(MB->Eval(x2));
      x2(1) = x(1) - dxi; real_t hmm = h(MB->Eval(x2));
      x2(0) = x(0) + dxi; real_t hpm = h(MB->Eval(x2));
      H(0,1) = H(1,0) = (hpp - hmp - hpm + hmm)/(4*dxi*dxi);
   }
}

/*
Finds minimum of h(u(x)) within element using Newton's method/backtracking gradient descent. 
Input: 
   x0:            Initial guess
   extrapolate:   Extrapolate lower bound for constraint functional
   niters:        Number of optimization iterations
   nbts:          Number of inner backtracking iterations
*/
real_t ElementOptimizer::Optimize(Vector &x0, bool extrapolate, int niters, int nbts) {
   Vector J(dim), dx(dim), x1(dim);
   DenseMatrix H(dim);

   // Compute h(u) at initial guess
   real_t h0 = h(MB->Eval(x0));
   real_t hstar = h0;

   // Perform optimization
   for (int i = 0; i < niters; i++) {
      // Numerically compute Jacobian and Hessian
      ComputeJacobian(x0, J);
      ComputeHessian(x0, H);

      // Use Newton's method if Hessian is positive definite
      if (IsPD(H)) {
         // Compute step dx = - H^-1 * J 
         H.Invert();
         H.Mult(J, dx);
         dx *= -1;
         
         // Ensure next point remains within element
         ProjectAlongBoundary(x0, dx);
         ProjectToElement(x0, dx);
      }
      // Else use gradient descent with backtracking line search
      else {
         real_t beta = beta_0;
         real_t Jnorm = J.Norml2();

         // Compute initial step
         for (int j = 0; j < dim; j++) {
            dx(j) = -beta*J(j)/max(Jnorm, real_t(eps));
         }
         
         // Ensure next point remains within element
         ProjectAlongBoundary(x0, dx);
         ProjectToElement(x0, dx);

         // Perform backtracking line search
         constexpr real_t c = 0.5; // Armijo-Goldstein coefficient
         for (int j = 0; j < nbts; j++) {
            // Compute next point
            for (int i = 0; i < dim; i++) {
               x1(i) = x0(i) + dx(i);
            }

            // Check cost function at next point
            real_t h1 = h(MB->Eval(x1));

            // Break if Armijo-Goldstein stopping condition is met
            if (h1 <= h0 - c*beta*Jnorm) {
               break;
            }
            // Else half the step size
            else {
               beta *= 0.5;
               dx *= 0.5;
            }
         }
      }
      
      // Set next point 
      for (int j = 0; j < dim; j++) {
         x0(j) = x0(j) + dx(j);
      }
   
      // Compute cost function at next point and track minimum
      h0 = h(MB->Eval(x0));
      hstar = min(hstar, h0);
   }

   // Extrapolate lower bound (see Sec. 3.2 in Dzanic, J. Comp. Phys., 508:113010, 2024)
   if (extrapolate && niters) {
      hstar -= J.Norml2()*dx.Norml2();
   }
   
   // Clip minima to -1.0 after extrapolation
   hstar = max(hstar, real_t(-1.0));
   return hstar;
}


// Tests if symmetric matrix M is positive definite using Sylvester’s criterion
bool ElementOptimizer::IsPD(DenseMatrix &M) {
   int dim = M.Height();
   switch (dim) {  
      case 1: {
         return (M(0,0) > eps);
      }
      case 2: {
         return (M(0,0) > eps && M.Det() > eps);
      }
      default:
         MFEM_ABORT("Continously bounds-preserving limiting not supported for dim > 2.")
   }
}


/*
Element geoemtry is represented using a level set for each face (s_i), with s_i = 0
if point is on the face and s_i > 0 if point is outside the element. This computes
the level sets for x and stores it in s. Currently only supports segments, triangles,
and quads.
*/
void ElementOptimizer::ComputeGeometryLevelset(Vector &x, Vector &s) { 
   switch (gtype) {  
      // Segment: [0,1]
      case 1: {
         s(0) =  x(0) - 1; // Right
         s(1) = -x(0); // Left
         break;
      }
      // Triangle with vertices (0,0), (1,0), (0,1)
      case 2: {
         s(0) = -x(0); // Left
         s(1) = -x(1); // Bottom
         s(2) = x(0) + x(1) - 1; // Diagonal
         break;
      }
      // Quad: [0,1]^2
      case 3: {
         s(0) =  x(0) - 1; // Right
         s(1) = -x(0); // Left
         s(2) =  x(1) - 1; // Top
         s(3) = -x(1); // Bottom
         break;
      }
      default:
         MFEM_ABORT("Element type not supported for continuously bounds-preserving limiting.")
   }
}

/*
Pre-computes the outward-facing face normals for the element (i.e., gradient of level sets)
and stores it in the DenseMatrix n of size (nfaces, dim). Currently only supports segments,
triangles, and quads.
*/
void ElementOptimizer::PrecomputeElementNormals() {
   switch (gtype) {  
      // Segment: [0,1]
      case 1: {
         n = DenseMatrix(2, 2);
         n(0,0) =  1; n(0,1) = 0; // Right
         n(1,0) = -1; n(1,1) = 0; // Left
         break;
      }
      // Triangle with vertices (0,0), (1,0), (0,1)
      case 2: {
         n = DenseMatrix(3, 2);
         n(0,0) = -1; n(0,1) =  0; // Left
         n(1,0) =  0; n(1,1) = -1; // Bottom
         n(2,0) =  sqrt(2.0)/2.0;; n(2,1) = sqrt(2.0)/2.0; // Diagonal
         break;
      }
      // Quad: [0,1]^2
      case 3: {
         n = DenseMatrix(4, 2);
         n(0,0) =  1; n(0,1) =  0; // Right
         n(1,0) = -1; n(1,1) =  0; // Left
         n(2,0) =  0; n(2,1) =  1; // Top
         n(3,0) =  0; n(3,1) = -1; // Bottom
         break;
      }
      default:
         MFEM_ABORT("Element type not supported for continuously bounds-preserving limiting.")
   }
}

/* 
Given an initial position (x0) and a search step (dx), zeros the perpendicular 
component of the dx if x0 is on an element face and the search direction 
points out of the element.
*/
void ElementOptimizer::ProjectAlongBoundary(Vector &x0, Vector &dx) {
   Vector s(nfaces), dxdn(nfaces);

   // Compute level sets s(x0)_i and n_i.dx for each element face i
   ComputeGeometryLevelset(x0, s);
   n.Mult(dx, dxdn);

   // Loop through faces
   for (int i = 0; i < s.Size(); i++) {
      // If x0 is on the element face and dx points outward (relative to the face)
      if (s(i) > -eps && dxdn(i) > 0) {
         // Zero perpendicular component
         for (int j = 0; j < dim; j++) {
            dx(j) -= dxdn(i)*dx(j);
         }
         n.Mult(dx, dxdn); // Re-calculate n.dx vector
      }
   }
}

/* 
Given an initial position (x0) and a search step (dx), reduces the search step
size if x0+dx is outside the element, such that x0+dx resides on the element 
face after limiting the step size.
*/
void ElementOptimizer::ProjectToElement(Vector &x0, Vector &dx) {
   Vector x1(dim), s0(nfaces), s1(nfaces);

   // Compute level sets s_i(x) (for each face i) at initial point and at candidate point x1
   ComputeGeometryLevelset(x0, s0);
   for (int i = 0; i < dim; i++) {
      x1(i) = x0(i) + dx(i);
   }
   ComputeGeometryLevelset(x1, s1);

   real_t zmin = 1.0;
   // Loop through faces
   for (int i = 0; i < s0.Size(); i++) {
      // If candidate point is outside the element  
      if (s1(i) > -eps) {
         // Compute search step reduction factor such at x1 is on the element face
         real_t z = -(s0(i) - eps)/max(s1(i) - s0(i), eps);
         zmin = min(zmin, z);
      }
   }

   // Reduce search step size
   dx *= zmin;
}

