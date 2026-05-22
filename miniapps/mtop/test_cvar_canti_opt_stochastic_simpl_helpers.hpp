#pragma once

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <functional>

namespace mfem
{

/// @brief Inverse sigmoid function
real_t inv_sigmoid(real_t x)
{
   real_t tol = 1e-12;
   x = std::min(std::max(tol,x), real_t(1.0)-tol);
   return std::log(x/(1.0-x));
}

/// @brief Sigmoid function
real_t sigmoid(real_t x)
{
   if (x >= 0)
   {
      return 1.0/(1.0+std::exp(-x));
   }
   else
   {
      return std::exp(x)/(1.0+std::exp(x));
   }
}

/// @brief Derivative of sigmoid function
real_t der_sigmoid(real_t x)
{
   real_t tmp = sigmoid(-x);
   return tmp - std::pow(tmp,2);
}

/// @brief Returns f(u(x)) where u is a scalar GridFunction and f:R → R
class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   std::function<real_t(const real_t)> fun; // f:R → R
public:
   MappedGridFunctionCoefficient()
      :GridFunctionCoefficient(),
       fun([](real_t x) {return x;}) {}
   MappedGridFunctionCoefficient(const GridFunction *gf,
                                 std::function<real_t(const real_t)> fun_,
                                 int comp=1)
      :GridFunctionCoefficient(gf, comp),
       fun(fun_) {}


   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      return fun(GridFunctionCoefficient::Eval(T, ip));
   }
   void SetFunction(std::function<real_t(const real_t)> fun_) { fun = fun_; }
};

/**
 * @brief Bregman projection of ρ = sigmoid(ψ) onto the subspace
 *        ∫_Ω ρ dx = θ vol(Ω) as follows:
 *
 *        1. Compute the root of the R → R function
 *            f(c) = ∫_Ω sigmoid(ψ + c) dx - θ vol(Ω)
 *        2. Set ψ ← ψ + c.
 *
 * @param psi a GridFunction to be updated
 * @param target_volume θ vol(Ω)
 * @param tol Newton iteration tolerance
 * @param max_its Newton maximum iteration number
 * @return real_t Final volume, ∫_Ω sigmoid(ψ)
 */
real_t proj(ParGridFunction &psi, real_t target_volume, real_t domain_volume, real_t tol = 1e-12,
            int max_its = 10, bool use_heaviside_projection = false,
            real_t heaviside_eta = 0.5, real_t heaviside_beta = 1.0)
{
    int myid = Mpi::WorldRank();

    ParGridFunction onegf(psi.ParFESpace());
    onegf = 1.0;

    class ProjectedDensityFromLatentCoeff : public Coefficient
    {
    public:
        ProjectedDensityFromLatentCoeff(ParGridFunction *psi_, real_t shift_,
                                        bool use_hproj_, real_t eta_, real_t beta_)
            : psi(psi_), shift(shift_), use_hproj(use_hproj_), eta(eta_), beta(beta_) {}

        virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
        {
            real_t rho = sigmoid(psi->GetValue(T, ip) + shift);
            if (use_hproj)
            {
                rho = PointwiseTrans::HProject(rho, eta, beta);
            }
            return rho;
        }

    private:
        ParGridFunction *psi;
        real_t shift;
        bool use_hproj;
        real_t eta;
        real_t beta;
    };

    real_t psimax;
    {
        real_t locmax = psi.Normlinf();
        // MPI_DOUBLE should be replaced with the MFEM data type
        MPI_Allreduce(&locmax, &psimax, 1, MPI_DOUBLE, MPI_MAX, psi.ParFESpace()->GetComm());

        if(myid == 0)
        {
            std::cout << "[proj] psi.Normlinf local=" << locmax
                      << ", global(psimax)=" << psimax << std::endl;
        }
    }

    const real_t volume_proportion = target_volume / domain_volume;

    if(myid == 0)
    {
        std::cout << "[proj] BEGIN" << std::endl;
        std::cout << "[proj] target_volume=" << target_volume
                  << ", domain_volume=" << domain_volume
                  << ", volume_proportion=" << volume_proportion
                  << ", use_heaviside_projection=" << use_heaviside_projection
                  << ", heaviside_eta=" << heaviside_eta
                  << ", heaviside_beta=" << heaviside_beta
                  << ", tol=" << tol
                  << ", max_its=" << max_its << std::endl;
    }

    if (!std::isfinite(volume_proportion))
    {
        if(myid == 0)
        {
            std::cout << "[proj] ERROR: volume_proportion is not finite." << std::endl;
        }
        throw std::runtime_error("proj: non-finite volume proportion");
    }

    real_t a = inv_sigmoid(volume_proportion) - psimax; // lower bound of 0
    real_t b = inv_sigmoid(volume_proportion) + psimax; // upper bound of 0

    if(myid == 0)
    {
        std::cout << "[proj] Initial bracket: a=" << a << ", b=" << b << std::endl;
    }

    ProjectedDensityFromLatentCoeff dens_a(&psi, a, use_heaviside_projection,
                                           heaviside_eta, heaviside_beta);
    ProjectedDensityFromLatentCoeff dens_b(&psi, b, use_heaviside_projection,
                                           heaviside_eta, heaviside_beta);

    ParLinearForm int_sigmoid_psi_a(psi.ParFESpace());
    int_sigmoid_psi_a.AddDomainIntegrator(new DomainLFIntegrator(dens_a));
    int_sigmoid_psi_a.Assemble();
    const real_t a_vol = int_sigmoid_psi_a(onegf);
    real_t a_vol_minus = a_vol - target_volume;

    if(myid == 0)
    {
        std::cout << "[proj] After assembling at a: int=" << a_vol
                  << ", int-target=" << a_vol_minus << std::endl;
    }

    if (!std::isfinite(a_vol) || !std::isfinite(a_vol_minus))
    {
        if(myid == 0)
        {
            std::cout << "[proj] ERROR: NaN/Inf detected at lower bracket evaluation." << std::endl;
        }
        throw std::runtime_error("proj: non-finite lower bracket linear form evaluation");
    }

    ParLinearForm int_sigmoid_psi_b(psi.ParFESpace());
    int_sigmoid_psi_b.AddDomainIntegrator(new DomainLFIntegrator(dens_b));
    int_sigmoid_psi_b.Assemble();
    const real_t b_vol = int_sigmoid_psi_b(onegf);
    real_t b_vol_minus = b_vol - target_volume;

    if(myid == 0)
    {
        std::cout << "[proj] After assembling at b: int=" << b_vol
                  << ", int-target=" << b_vol_minus << std::endl;
    }

    if (!std::isfinite(b_vol) || !std::isfinite(b_vol_minus))
    {
        if(myid == 0)
        {
            std::cout << "[proj] ERROR: NaN/Inf detected at upper bracket evaluation." << std::endl;
        }
        throw std::runtime_error("proj: non-finite upper bracket linear form evaluation");
    }

    if (a_vol_minus * b_vol_minus > 0)
    {
        if(myid == 0)
        {
            std::cout << "[proj] WARNING: Initial bracket may not contain root. "
                      << "a_vol_minus=" << a_vol_minus
                      << ", b_vol_minus=" << b_vol_minus << std::endl;
        }
    }

    bool done = false;
    real_t x;

    // ParLinearForm int_sigmoid_psi_x(psi.ParFESpace());
    real_t x_vol = std::numeric_limits<real_t>::quiet_NaN();

    for (int k = 0; k < max_its; k++) // Illinois iteration
    {
        const real_t denom = (b_vol_minus - a_vol_minus);

        if(myid == 0)
        {
            std::cout << "[proj] Iter " << k
                      << ": a=" << a << ", b=" << b
                      << ", a_vol_minus=" << a_vol_minus
                      << ", b_vol_minus=" << b_vol_minus
                      << ", denom=" << denom << std::endl;
        }

        if (!std::isfinite(denom) || std::abs(denom) <= std::numeric_limits<real_t>::epsilon())
        {
            if(myid == 0)
            {
                std::cout << "[proj] ERROR: invalid denominator in Illinois step." << std::endl;
            }
            throw std::runtime_error("proj: invalid denominator in root update");
        }

        x = b - b_vol_minus * (b - a) / denom;

        if (!std::isfinite(x))
        {
            if(myid == 0)
            {
                std::cout << "[proj] ERROR: x is NaN/Inf after false-position update." << std::endl;
            }
            throw std::runtime_error("proj: non-finite iterate x");
        }

        ProjectedDensityFromLatentCoeff dens_x(&psi, x, use_heaviside_projection,
                               heaviside_eta, heaviside_beta);

        ParLinearForm int_sigmoid_psi_x(psi.ParFESpace());
        int_sigmoid_psi_x.AddDomainIntegrator(new DomainLFIntegrator(dens_x));
        int_sigmoid_psi_x.Assemble();
        x_vol = int_sigmoid_psi_x(onegf);

        // int_sigmoid_psi_x.Update(psi.ParFESpace());
        // int_sigmoid_psi_x = 0.0;
        // int_sigmoid_psi_x.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi_x));
        // int_sigmoid_psi_x.Assemble();
        // x_vol = int_sigmoid_psi_x(onegf);

    //         ParLinearForm int_sigmoid_psi_b(psi.ParFESpace());
    // int_sigmoid_psi_b.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi_b));
    // int_sigmoid_psi_b.Assemble();
    // const real_t b_vol = int_sigmoid_psi_b(onegf);

        real_t x_vol_minus = x_vol - target_volume;

        if(myid == 0)
        {
            std::cout << "[proj] Iter " << k
                      << ": x=" << x
                      << ", int(sigmoid(psi+x))=" << x_vol
                      << ", x_vol_minus=" << x_vol_minus << std::endl;
        }

        if (!std::isfinite(x_vol) || !std::isfinite(x_vol_minus))
        {
            if(myid == 0)
            {
                std::cout << "[proj] ERROR: NaN/Inf from linear form at iteration "
                          << k << "." << std::endl;
            }
            throw std::runtime_error("proj: non-finite linear form during iteration");
        }

        if (b_vol_minus * x_vol_minus < 0)
        {
            a = b;
            a_vol_minus = b_vol_minus;

            if(myid == 0)
            {
                std::cout << "[proj] Iter " << k << ": root is bracketed in [b, x], setting a <- b." << std::endl;
            }
        }
        else
        {
            a_vol_minus = a_vol_minus / 2;

            if(myid == 0)
            {
                std::cout << "[proj] Iter " << k << ": applying Illinois damping to a_vol_minus -> "
                          << a_vol_minus << std::endl;
            }
        }
        b = x;
        b_vol_minus = x_vol_minus;

        if (abs(x_vol_minus) < tol)
        {
            done = true;

            if(myid == 0)
            {
                std::cout << "[proj] CONVERGED at iter " << k
                          << " with |x_vol_minus|=" << std::abs(x_vol_minus)
                          << " < tol=" << tol << std::endl;
            }
            break;
        }

        if(myid == 0)
        {
            std::cout << "[proj] Iter " << k
                      << ": not converged, |x_vol_minus|=" << std::abs(x_vol_minus)
                      << std::endl;
        }
    }

    psi += x;

    if(myid == 0)
    {
        std::cout << "[proj] Applied shift x=" << x
                  << ". Returning final volume=" << x_vol << std::endl;
    }

    if (!done)
    {
        mfem_warning("Projection reached maximum iteration without converging. "
                     "Result may not be accurate.");

        if(myid == 0)
        {
            std::cout << "[proj] WARNING: max iterations reached without convergence." << std::endl;
        }
    }
    return x_vol;
}


}; // namespace mfem