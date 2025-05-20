#include "mfem.hpp"

using namespace std;
using namespace mfem;

/**
 * @brief Initializes the displacement vector u based on the input vector x.
 */
void InitDisplacement(const Vector &x, Vector &u)
{
   const real_t displacement = -0.1;
   const int dim = x.Size();

   u = 0.0;
   u(dim-1) = displacement*x(dim-1);
}

/**
 * @brief Computes the gap function φ₁ based on the input vector x; represents
 *        the distance between a point x and a plane.
 *
 * @param x Input vector
 * @return real_t Computed gap function value, φ₁(x)
 */
real_t GapFunction(const Vector &x)
{
   const int dim = x.Size();
   bool angled_plane = false;

   if (dim == 3 && angled_plane)
   {
      int a = 0;
      int b = 1;
      int c = 5;
      int d = -5;

      return abs((d - a*x(0) - b*x(1))/c);
   }
   else
   {
      float d;
      // d = -1.5; // Ball
      d = -0.5; // Tetrahedron/Cube

      return x(dim-1) - d;
   }
}

/**
 * @brief Computes the force function based on the input vector x.
 *
 * @param x Input vector
 * @param f Output vector representing the downward force
 */
void ForceFunction(const Vector &x, Vector &f)
{
   real_t force;
   // force = -0.25;   // Ball
   // force = -5;      // Tetrahedron
   force = -2;      // Cube

   const int dim = x.Size();
   f = 0.0;
   f(dim-1) = force;
}

/**
 * @brief Computes the stress tensor σ(u) based on the gradient of the
 *        displacement field u and the Lame parameters.
 *
 * @param grad_u Gradient of the displacement field
 * @param lambda First Lame parameter
 * @param mu     Second Lame parameter
 * @param sigma  Computed stress tensor
 */
void ComputeStress(const DenseMatrix &grad_u, const real_t lambda,
                   const real_t mu, DenseMatrix &sigma)
{
   const int dim = grad_u.Size();

   // Compute div(u): trace of Jacobian ∇u
   real_t div_u = grad_u.Trace();

   // Compute strain: ε(u) = (∇u + ∇uᵀ)/2
   DenseMatrix epsilon = grad_u;
   epsilon.Symmetrize();

   // Compute stress: σ(u) = λ div(u) I + 2μ ε(u)
   DenseMatrix I;
   I.Diag(1, dim);
   sigma = 0.0;
   Add(lambda * div_u, I, 2 * mu, epsilon, sigma);
}

/**
 * @brief Implements the boundary condition for the Signorini problem.
 *
 * @param dim Spatial dimension
 * @param u_prev Previous displacement vector
 * @param n_tilde Vector field
 * @param lambda First Lame parameter
 * @param mu Second Lame parameter
 * @param alpha Step-size parameter
 */
class TractionBoundary : public VectorCoefficient
{
private:
   GridFunction *u_prev;
   Vector n_tilde;
   real_t lambda, mu, alpha;

public:
   TractionBoundary(int _dim, GridFunction *_u_prev, Vector _n_tilde,
                    real_t _lambda, real_t _mu, real_t _alpha)
      : VectorCoefficient(_dim), u_prev(_u_prev), n_tilde(_n_tilde), lambda(_lambda),
        mu(_mu), alpha(_alpha) {}

   virtual void Eval(Vector &u, ElementTransformation &T,
                     const IntegrationPoint &ip) override
   {
#ifdef MFEM_USE_MPI
      ParGridFunction *par_u_prev = dynamic_cast<ParGridFunction*>(u_prev);
#endif
      const int dim = T.GetSpaceDim();

      // Get current point coordinates
      Vector x(dim);
      T.Transform(ip, x);

      // Get value and Jacobian of previous solution
      Vector u_prev_val(dim);
      DenseMatrix grad_u_prev(dim,dim);
#ifdef MFEM_USE_MPI
      if (par_u_prev)
      {
         par_u_prev->GetVectorValue(T, ip, u_prev_val);
         par_u_prev->GetVectorGradient(T, grad_u_prev);
      }
      else
      {
         u_prev->GetVectorValue(T, ip, u_prev_val);
         u_prev->GetVectorGradient(T, grad_u_prev);
      }
#else
      u_prev->GetVectorValue(T, ip, u_prev_val);
      u_prev->GetVectorGradient(T, grad_u_prev);
#endif
      // Evaluate the stress tensor σ(uᵏ⁻¹)
      DenseMatrix sigma(dim,dim);
      ComputeStress(grad_u_prev, lambda, mu, sigma);

      // Compute normal vector n
      Vector n(dim);
      CalcOrtho(T.Jacobian(), n);
      n /= n.Norml2();

      // Compute pressure σ(uᵏ⁻¹)n · ñ
      Vector sigma_n(dim);
      sigma.Mult(n, sigma_n);
      real_t pressure = sigma_n * n_tilde;

      // Evaluate the gap function φ₁
      real_t phi_1 = GapFunction(x);

      // Set the boundary condition
      // uᵏ · ñ = φ₁ - (φ₁ - uᵏ⁻¹ · ñ) exp((αₖ σ(uᵏ⁻¹) n) · ñ)
      u.SetSize(dim);
      u = u_prev_val;
      u(dim-1) = phi_1 - (phi_1 - u_prev_val * n_tilde) * exp(alpha * pressure);
      u(dim-1) /= n_tilde(dim-1);
   }
};
