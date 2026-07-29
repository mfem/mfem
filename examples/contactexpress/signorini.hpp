#include "mfem.hpp"

using namespace mfem;

// Problem data (initial displacement, forcing, gap function) is declared here
// and defined at the bottom of explicitp.cpp, driven by the file-scope
// parameters plane_g and force_g (settable via -p/--plane and -f/--force).

/**
 * @brief u(x,y,z) = (0,0,plane_g)
 */
void InitDisplacement(const Vector &x, Vector &u);

/**
 * @brief Computes the force function based on the input vector x.
 *
 * @param x Input vector
 * @param f Output vector representing the downward force
 */
void ForceFunction(const Vector &x, Vector &f);

/**
 * @brief Computes the gap function φ₁ based on the input vector x; represents
 *        the distance between a point x and the plane z = plane_g.
 *
 * @param  x Input vector
 * @return real_t Computed gap function value, φ₁(x)
 */
real_t GapFunction(const Vector &x);

/**
 * @brief Computes the stress tensor σ(u) based on the gradient of the
 *        displacement field u and the Lamé parameters.
 *
 * @param grad_u Gradient of the displacement field
 * @param lambda First Lamé parameter
 * @param mu     Second Lamé parameter
 * @param sigma  Computed stress tensor
 */
void ComputeStress(const DenseMatrix &grad_u, const real_t lambda,
                   const real_t mu, DenseMatrix &sigma)
{
   const int dim = grad_u.Size();

   // Compute div(u): trace of Jacobian ∇u
   const real_t div_u = grad_u.Trace();

   // Compute strain: ε(u) = (∇u + ∇uᵀ)/2
   DenseMatrix epsilon = grad_u;
   epsilon.Symmetrize();

   // Compute stress: σ(u) = λ div(u) I + 2μ ε(u)
   DenseMatrix I;
   I.Diag(1.0, dim);
   sigma = 0.0;
   Add(lambda * div_u, I, 2 * mu, epsilon, sigma);
}

/**
 * @brief Implements the contact express boundary condition for the Signorini
 *        problem. The vector n_tilde representing ñ is assumed to be equal to
 *        (0,...,0,-1).
 *
 *        Implicit variant: the gap term (u·ñ - φ₁) and the tangential
 *        components are taken from u_prev (uᵏ⁻¹), while the stress inside the
 *        exponential is evaluated from a separate grid function u_stress
 *        (the current estimate of uᵏ). Passing the same grid function for
 *        both arguments recovers the original explicit scheme.
 *
 * @param dim Spatial dimension
 * @param u_prev Displacement used for the gap term, uᵏ⁻¹
 * @param u_stress Displacement whose stress enters the exponential, uᵏ
 * @param n_tilde Vector field
 * @param lambda First Lamé parameter
 * @param mu Second Lamé parameter
 * @param alpha Step-size parameter
 */
class TractionBoundary : public VectorCoefficient
{
private:
   GridFunction *u_prev;
   GridFunction *u_stress;
   Vector n_tilde;
   real_t lambda, mu, alpha;

public:
   TractionBoundary(int _dim, GridFunction *_u_prev, GridFunction *_u_stress,
                    Vector _n_tilde, real_t _lambda, real_t _mu, real_t _alpha)
      : VectorCoefficient(_dim), u_prev(_u_prev), u_stress(_u_stress),
        n_tilde(_n_tilde), lambda(_lambda), mu(_mu), alpha(_alpha) {}

   virtual void Eval(Vector &u, ElementTransformation &T,
                     const IntegrationPoint &ip) override
   {
#ifdef MFEM_USE_MPI
      ParGridFunction *par_u_prev = dynamic_cast<ParGridFunction*>(u_prev);
      ParGridFunction *par_u_stress = dynamic_cast<ParGridFunction*>(u_stress);
#endif
      const int dim = T.GetSpaceDim();

      // Get current point coordinates. (This also sets the integration point
      // on T, which the subsequent gradient evaluation relies on.)
      Vector x(dim);
      T.Transform(ip, x);

      // Get the value of the gap solution uᵏ⁻¹ (used for the gap term and the
      // tangential components).
      Vector u_prev_val(dim);
#ifdef MFEM_USE_MPI
      if (par_u_prev)
      {
         par_u_prev->GetVectorValue(T, ip, u_prev_val);
      }
      else
      {
         u_prev->GetVectorValue(T, ip, u_prev_val);
      }
#else
      u_prev->GetVectorValue(T, ip, u_prev_val);
#endif

      // Get the Jacobian of the stress solution uᵏ (used for σ inside the
      // exponential).
      DenseMatrix grad_u_stress(dim,dim);
#ifdef MFEM_USE_MPI
      if (par_u_stress)
      {
         par_u_stress->GetVectorGradient(T, grad_u_stress);
      }
      else
      {
         u_stress->GetVectorGradient(T, grad_u_stress);
      }
#else
      u_stress->GetVectorGradient(T, grad_u_stress);
#endif
      // Evaluate the stress tensor σ(uᵏ)
      DenseMatrix sigma(dim,dim);
      ComputeStress(grad_u_stress, lambda, mu, sigma);

      // Compute normal vector n
      Vector n(dim);
      CalcOrtho(T.Jacobian(), n);
      n /= n.Norml2();

      // Compute pressure σ(uᵏ)n · ñ
      Vector sigma_n(dim);
      sigma.Mult(n, sigma_n);
      const real_t pressure = sigma_n * n_tilde;

      // Evaluate the gap function φ₁
      const real_t phi_1 = GapFunction(x);

      // Set the boundary condition
      // uᵏ · ñ = φ₁ + (uᵏ⁻¹ · ñ - φ₁) exp(αₖ (σ(uᵏ)n · ñ))
      u.SetSize(dim);
      u = u_prev_val;
      u(dim-1) = phi_1 + (u_prev_val * n_tilde - phi_1) * exp(alpha * pressure);
      u(dim-1) /= n_tilde(dim-1);
   }
};
