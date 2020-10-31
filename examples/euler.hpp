#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;

// extern variables
/// number of equations
extern const int num_states;
///specific heat ratio
extern const double gamma;
/// gas constant
extern const double R;

/// Function:computeFlux():
//  computes the flux at a given state
void computeFlux(const Vector &state, int dim, DenseMatrix &flux)
{
    const double rho = state(0);
    const Vector rvel(state.GetData() + 1, dim);
    const double energy = state(dim + 1);
    MFEM_ASSERT(StateIsPhysical(state, dim), "");
    const double press = computePressure(state, dim);
    for (int d = 0; d < dim; ++d)
    {
        flux(0, d) = rvel(d);
        for (int n = 0; n < dim; ++n)
        {
            flux(n + 1, d) = rvel(n) * rvel(d) / rho;
        }
        flux(1 + d, d) += press;
    }
    for (int d = 0; d < dim; ++d)
    {
        flux(dim + 1, d) = (e + press) * rvel(d);
    }
}

void calcEulerFlux(const Vector &dir, const Vector &q, const Vector &flux)
{
   double press = computePressure(state, dim);
   double U = (q + 1 *  dir);
   flux(0) = U;
   U /= q(0);
   for (int i = 0; i < dim; ++i)
   {
      flux(i + 1) = q(i + 1) * U + dir(i) * press;
   }
   flux(dim + 1) = (q(dim + 1) + press) * U;
}

/// computes the pressure for given state variables
inline double computePressure(const Vector &state, int dim)
{
    double press = 0;
    const double rho = state(0);
    const double e = state(dim + 1);

    for (int i = 1; i < dim + 1; ++i)
    {
        press += state(i) * state(i) / (rho);
    }
    return (gamma - 1.0) * (e - 0.5 * press);
}

class EulerSolver
{
private:
    Vector flux1;
    Vector flux2;

public:
    EulerSolver();
    double Eval(const Vector &state1, const Vector &state2,
                const Vector &nor, Vector &flux)
};

void calcBoundaryFlux(const mfem::Vector &dir, const mfem::Vector &qbnd, const mfem::Vector &q,
                      const mfem::Vector &work, const mfem::Vector &flux)
{
    using std::max;

    // Define some constants
    const double sat_Vn = 0.0; // 0.025
    const double sat_Vl = 0.0; // 0.025

    // Define some constants used to construct the "Jacobian"
    const double dA = sqrt(dir * dir);
    const double fac = 1.0 / qbnd(0);
    const double phi = 0.5 * (qbnd + 1) * (qbnd + 1) * fac * fac;
    const double H = euler::gamma * qbnd(dim + 1) * fac - euler::gami * phi;
    const double a = sqrt(euler::gami * (H - phi));
    const double Un = ((qbnd + 1) * dir) * fac;
    double lambda1 = Un + dA * a;
    double lambda2 = Un - dA * a;
    double lambda3 = Un;
    const double rhoA = fabs(Un) + dA * a;
    lambda1 = 0.5 * (max(fabs(lambda1), sat_Vn * rhoA) - lambda1);
    lambda2 = 0.5 * (max(fabs(lambda2), sat_Vn * rhoA) - lambda2);
    lambda3 = 0.5 * (max(fabs(lambda3), sat_Vl * rhoA) - lambda3);

    Vector dq = work;
    for (int i = 0; i < dim + 2; ++i)
    {
        dq(i) = q(i) - qbnd(i);
    }
    calcEulerFlux(dir, q, flux);

    // diagonal matrix multiply; note that flux was initialized by calcEulerFlux
    for (int i = 0; i < dim + 2; ++i)
    {
        flux(i) += lambda3 * dq(i);
    }

    // some scalars needed for E1*dq, E2*dq, E3*dq, and E4*dq
    double tmp1 = 0.5 * (lambda1 + lambda2) - lambda3;
    double E1dq_fac = tmp1 * euler::gami / (a * a);
    double E2dq_fac = tmp1 / (dA * dA);
    double E34dq_fac = 0.5 * (lambda1 - lambda2) / (dA * a);

    // get E1*dq + E4*dq and add to flux
    double Edq = phi * dq(0) + dq(dim + 1) - ((qbnd + 1) * (dq + 1)) * fac;
    flux(0) += E1dq_fac * Edq;
    for (int i = 0; i < dim; ++i)
    {
        flux(i + 1) += Edq * (E1dq_fac * qbnd(i + 1) * fac + euler::gami * E34dq_fac * dir(i));
    }
    flux(dim + 1) += Edq * (E1dq_fac * H + euler::gami * E34dq_fac * Un);

    // get E2*dq + E3*dq and add to flux
    Edq = -Un * dq(0) + (dir * (dq + 1));
    flux[0] += E34dq_fac * Edq;
    for (int i = 0; i < dim; ++i)
    {
        flux(i + 1) += Edq * (E2dq_fac * dir(i) + E34dq_fac * qbnd(i + 1) * fac);
    }
    flux(dim + 1) += Edq * (E2dq_fac * Un + E34dq_fac * H);
}

class InviscidBoundaryIntegrator : public mfem::NonlinearFormIntegrator
{
public:
    /// Constructs a boundary integrator based on a given boundary flux
    /// \param[in] diff_stack - for algorithmic differentiation
    /// \param[in] fe_coll - used to determine the face elements
    /// \param[in] num_state_vars - the number of state variables
    /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
    InviscidBoundaryIntegrator(const mfem::FiniteElementCollection *fe_coll,
                               int num_state_vars = 1, double a = 1.0)
        : num_states(num_state_vars), alpha(a),
          fec(fe_coll) {}

    /// Construct the contribution to a functional from the boundary element
    /// \param[in] el_bnd - boundary element that contribute to the functional
    /// \param[in] el_unused - dummy element that is not used for boundaries
    /// \param[in] trans - hold geometry and mapping information about the face
    /// \param[in] elfun - element local state function
    /// \return element local contribution to functional
    virtual double GetFaceEnergy(const mfem::FiniteElement &el_bnd,
                                 const mfem::FiniteElement &el_unused,
                                 mfem::FaceElementTransformations &trans,
                                 const mfem::Vector &elfun);

    /// Construct the contribution to the element local residual
    /// \param[in] el_bnd - the finite element whose residual we want to update
    /// \param[in] el_unused - dummy element that is not used for boundaries
    /// \param[in] trans - holds geometry and mapping information about the face
    /// \param[in] elfun - element local state function
    /// \param[out] elvect - element local residual
    virtual void AssembleFaceVector(const mfem::FiniteElement &el_bnd,
                                    const mfem::FiniteElement &el_unused,
                                    mfem::FaceElementTransformations &trans,
                                    const mfem::Vector &elfun,
                                    mfem::Vector &elvect);

protected:
    /// number of states
    int num_states;
    /// scales the terms; can be used to move to rhs/lhs
    double alpha;
    /// used to select the appropriate face element
    const mfem::FiniteElementCollection *fec;
#ifndef MFEM_THREAD_SAFE
    /// used to reference the state at face node
    mfem::Vector u_face;
    /// store the physical location of a node
    mfem::Vector x;
    /// the outward pointing (scaled) normal to the boundary at a node
    mfem::Vector nrm;
    /// stores the flux evaluated by `bnd_flux`
    mfem::Vector flux_face;
#endif

    /// Compute a scalar boundary function
    /// \param[in] x - coordinate location at which function is evaluated
    /// \param[in] dir - vector normal to the boundary at `x`
    /// \param[in] u - state at which to evaluate the function
    /// \returns fun - value of the function
    /// \note `x` can be ignored depending on the function
    /// \note This uses the CRTP, so it wraps a call to `calcFunction` in
    /// Derived.
    double bndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                    const mfem::Vector &u)
    {
        return calcBndryFun(x, dir, u);
    }

    /// Compute a boundary flux function
    /// \param[in] x - coordinate location at which flux is evaluated
    /// \param[in] dir - vector normal to the boundary at `x`
    /// \param[in] u - state at which to evaluate the flux
    /// \param[out] flux_vec - value of the flux
    /// \note `x` can be ignored depending on the flux
    /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
    void flux(const mfem::Vector &x, const mfem::Vector &dir,
              const mfem::Vector &u, mfem::Vector &flux_vec)
    {
        calcFlux(x, dir, u, flux_vec);
    }
};

/// Integrator for inviscid interface fluxes (fluxes that do not need gradient)
class InviscidFaceIntegrator : public mfem::NonlinearFormIntegrator
{
public:
    /// Constructs a face integrator based on a given interface flux
    /// \param[in] diff_stack - for algorithmic differentiation
    /// \param[in] fe_coll - used to determine the face elements
    /// \param[in] num_state_vars - the number of state variables
    /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
    InviscidFaceIntegrator(
        const mfem::FiniteElementCollection *fe_coll,
        int num_state_vars = 1, double a = 1.0)
        : num_states(num_state_vars), alpha(a),
          fec(fe_coll) {}

    /// Construct the contribution to the element local residuals
    /// \param[in] el_left - "left" element whose residual we want to update
    /// \param[in] el_right - "right" element whose residual we want to update
    /// \param[in] trans - holds geometry and mapping information about the face
    /// \param[in] elfun - element local state function
    /// \param[out] elvect - element local residual
    virtual void AssembleFaceVector(const mfem::FiniteElement &el_left,
                                    const mfem::FiniteElement &el_right,
                                    mfem::FaceElementTransformations &trans,
                                    const mfem::Vector &elfun,
                                    mfem::Vector &elvect);

protected:
    /// number of states
    int num_states;
    /// scales the terms; can be used to move to rhs/lhs
    double alpha;
    /// used to select the appropriate face element
    const mfem::FiniteElementCollection *fec;
#ifndef MFEM_THREAD_SAFE
    /// used to reference the left state at face node
    mfem::Vector u_face_left;
    /// used to reference the right state at face node
    mfem::Vector u_face_right;
    /// the outward pointing (scaled) normal to the boundary at a node
    mfem::Vector nrm;
    /// stores the flux evaluated by `bnd_flux`
    mfem::Vector flux_face;
    /// stores the jacobian of the flux with respect to the left state
    mfem::DenseMatrix flux_jac_left;
    /// stores the jacobian of the flux with respect to the right state
    mfem::DenseMatrix flux_jac_right;
#endif

    /// Compute an interface flux function
    /// \param[in] dir - vector normal to the face
    /// \param[in] u_left - "left" state at which to evaluate the flux
    /// \param[in] u_right - "right" state at which to evaluate the flux
    /// \param[out] flux_vec - value of the flux
    /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
    void flux(const mfem::Vector &dir, const mfem::Vector &u_left,
              const mfem::Vector &u_right, mfem::Vector &flux_vec)
    {
        calcFlux(dir, u_left, u_right, flux_vec);
    }
};

class SlipWallBC : public InviscidBoundaryIntegrator
{
public:
    /// Constructs an integrator for a slip-wall boundary flux
    /// \param[in] diff_stack - for algorithmic differentiation
    /// \param[in] fe_coll - used to determine the face elements
    /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
    SlipWallBC(const mfem::FiniteElementCollection *fe_coll,
               double a = 1.0)
        : InviscidBoundaryIntegrator(
              fe_coll, dim + 2, a) {}

    /// Compute an adjoint-consistent slip-wall boundary flux
    /// \param[in] x - coordinate location at which flux is evaluated (not used)
    /// \param[in] dir - vector normal to the boundary at `x`
    /// \param[in] q - conservative variables at which to evaluate the flux
    /// \param[out] flux_vec - value of the flux
    void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                  const mfem::Vector &q, mfem::Vector &flux_vec)
    {
        double press;
        press = computePressure(q, dim);
        flux(0) = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            flux(i + 1) = dir(i) * press;
        }
        flux(dim + 1) = 0.0;
    }
};

class IsentropicVortexBC : public InviscidBoundaryIntegrator
{
public:
    /// Constructs an integrator for isentropic vortex boundary flux
    /// \param[in] diff_stack - for algorithmic differentiation
    /// \param[in] fe_coll - used to determine the face elements
    /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
    IsentropicVortexBC(const mfem::FiniteElementCollection *fe_coll,
                       double a = 1.0)
        : InviscidBoundaryIntegrator(
              fe_coll, 4, a) {}

    void calcIsentropicVortexState(const mfem::Vector x, const mfem::Vector &qbnd)
    {
        double ri = 1.0;
        double Mai = 0.5; //0.95
        double rhoi = 2.0;
        double prsi = 1.0 / euler::gamma;
        double rinv = ri / sqrt(x(0) * x(0) + x(1) * x(1));
        double rho = rhoi * pow(1.0 + 0.5 * euler::gami * Mai * Mai * (1.0 - rinv * rinv),
                                1.0 / euler::gami);
        double Ma = sqrt((2.0 / euler::gami) * ((pow(rhoi / rho, euler::gami)) *
                                                    (1.0 + 0.5 * euler::gami * Mai * Mai) -
                                                1.0));
        double theta;
        if (x(0) > 1e-15)
        {
            theta = atan(x(1) / x(0));
        }
        else
        {
            theta = M_PI / 2.0;
        }
        double press = prsi * pow((1.0 + 0.5 * euler::gami * Mai * Mai) /
                                      (1.0 + 0.5 * euler::gami * Ma * Ma),
                                  euler::gamma / euler::gami);
        double a = sqrt(euler::gamma * press / rho);

        qbnd(0) = rho;
        qbnd(1) = rho * a * Ma * sin(theta);
        qbnd(2) = -rho * a * Ma * cos(theta);
        qbnd(3) = press / euler::gami + 0.5 * rho * a * a * Ma * Ma;
    }

    /// Compute a characteristic boundary flux for the isentropic vortex
    /// \param[in] x - coordinate location at which flux is evaluated
    /// \param[in] dir - vector normal to the boundary at `x`
    /// \param[in] q - conservative variables at which to evaluate the flux
    /// \param[out] flux_vec - value of the flux
    void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                  const mfem::Vector &q, mfem::Vector &flux_vec)
    {
        Vector qbnd(4);
        Vector work(4);
        calcIsentropicVortexState(x, qbnd);
        calcBoundaryFlux(dir, qbnd, q, work, flux);
    }
};

class FarFieldBC : public InviscidBoundaryIntegrator
{
public:
    /// Constructs an integrator for a far-field boundary flux
    /// \param[in] diff_stack - for algorithmic differentiation
    /// \param[in] fe_coll - used to determine the face elements
    /// \param[in] q_far - state at the far-field
    /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
    FarFieldBC(const mfem::FiniteElementCollection *fe_coll,
               const mfem::Vector q_far,
               double a = 1.0)
        : InviscidBoundaryIntegrator(
              fe_coll, dim + 2, a),
          qfs(q_far), work_vec(dim + 2) {}

    /// Compute an adjoint-consistent slip-wall boundary flux
    /// \param[in] x - coordinate location at which flux is evaluated (not used)
    /// \param[in] dir - vector normal to the boundary at `x`
    /// \param[in] q - conservative variables at which to evaluate the flux
    /// \param[out] flux_vec - value of the flux
    void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                  const mfem::Vector &q, mfem::Vector &flux_vec);

private:
    /// Stores the far-field state
    mfem::Vector qfs;
    /// Work vector for boundary flux computation
    mfem::Vector work_vec;
};

class InterfaceIntegrator : public InviscidFaceIntegrator
{
public:
    /// Construct an integrator for the Euler flux over elements
    /// \param[in] coeff - scales the dissipation (must be non-negative!)
    /// \param[in] fe_coll - pointer to a finite element collection
    /// \param[in] a - factor, usually used to move terms to rhs
    InterfaceIntegrator(double coeff,
                        const mfem::FiniteElementCollection *fe_coll,
                        double a = 1.0);

    /// Compute the interface function at a given (scaled) direction
    /// \param[in] dir - vector normal to the interface
    /// \param[in] qL - "left" state at which to evaluate the flux
    /// \param[in] qR - "right" state at which to evaluate the flu
    /// \param[out] flux - value of the flux
    /// \note wrapper for the relevant function in `euler_fluxes.hpp`
    void calcFlux(const mfem::Vector &dir, const mfem::Vector &qL,
                  const mfem::Vector &qR, mfem::Vector &flux);

protected:
    /// Scalar that controls the amount of dissipation
    double diss_coeff;
};

/// Domain Integrator
// (grad v, F(u))
class EulerDomainIntegrator : public NonlinearFormIntegrator
{
private:
    Vector shape;
    DenseMatrix flux;
    DenseMatrix dshapedx;

public:
    EulerDomainIntegrator(const int dim);

    virtual void AssembleElementVector(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       const Vector &elfun, Vector &elvect);
};
