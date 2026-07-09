#ifndef FRACTIONAL_INVERSE_COEFFS_POLYLIB_HPP
#define FRACTIONAL_INVERSE_COEFFS_POLYLIB_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Nektar Polylib header from https://www.nektar.info/2nd_edition/Polylib.html
#include "polylib.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <limits>

#include "polylib.h"

class FractionalInverseCoeffsPolylib{
public:
    FractionalInverseCoeffsPolylib(double beta, std::size_t m)
        : beta_(beta), m_(m)
    {
        if (!(beta_ > 0.0 && beta_ < 1.0)) {
            throw std::invalid_argument("beta_frac must be in (0,1).");
        }
        if (m_ == 0) {
            throw std::invalid_argument("m must be >= 1.");
        }

        // Jacobi weight: (1-x)^alpha (1+x)^gamma
        alpha_ = -beta_;
        gamma_ = beta_ - 1.0;

        x_.assign(m_, 0.0);
        w_.assign(m_, 0.0);
        sigma_.assign(m_, 0.0);
        eta_.assign(m_, 0.0);

        compute_nodes();       // Polylib zeros
        compute_weights();     // recompute weights with std::lgamma
        compute_sigma_eta();   // your rational coefficients
    }

    std::size_t size() const { return m_; }

    const std::vector<double>& x()     const { return x_; }
    const std::vector<double>& w()     const { return w_; }
    const std::vector<double>& sigma() const { return sigma_; }
    const std::vector<double>& eta()   const { return eta_; }

private:
    double beta_{};
    std::size_t m_{};
    double alpha_{}, gamma_{};

    std::vector<double> x_, w_, sigma_, eta_;

    static double pow2(double p) { return std::exp(p * std::log(2.0)); }

    void compute_nodes() {
        // Use zwgj primarily to get Gauss-Jacobi nodes.
        // Note: zwgj also returns weights, but in Nektar Polylib those weights
        // rely on gammaF() (integer/half only). We'll overwrite w_ later.
#ifdef __cplusplus
        polylib::zwgj(x_.data(), w_.data(), static_cast<int>(m_), alpha_, gamma_);
#else
        zwgj(x_.data(), w_.data(), static_cast<int>(m_), alpha_, gamma_);
#endif
    }

    void compute_weights() {
        const int np = static_cast<int>(m_);

        // Compute P_np'(x_i) using Polylib jacobd:
        // jacobd(np_points, z_points, polyd, poly_order, alpha, beta)
        std::vector<double> dP(np, 0.0);
#ifdef __cplusplus
        polylib::jacobd(np, x_.data(), dP.data(), np, alpha_, gamma_);
#else
        jacobd(np, x_.data(), dP.data(), np, alpha_, gamma_);
#endif

        // fac = 2^{a+b+1} * Gamma(a+n+1)*Gamma(b+n+1) / (Gamma(n+1)*Gamma(a+b+n+1))
        const double a = alpha_;
        const double b = gamma_;
        const double n = static_cast<double>(np);

        const double ln_fac =
            (a + b + 1.0) * std::log(2.0)
            + std::lgamma(a + n + 1.0)
            + std::lgamma(b + n + 1.0)
            - std::lgamma(n + 1.0)
            - std::lgamma(a + b + n + 1.0);

        const double fac = std::exp(ln_fac);

        for (int i = 0; i < np; ++i) {
            const double zi = x_[static_cast<std::size_t>(i)];
            const double denom = dP[static_cast<std::size_t>(i)] * dP[static_cast<std::size_t>(i)]
                               * (1.0 - zi * zi);
            w_[static_cast<std::size_t>(i)] = fac / denom;
        }
    }

    void compute_sigma_eta() {
        const double pref = std::sin(M_PI * beta_) / M_PI;

        for (std::size_t j = 0; j < m_; ++j) {
            const double denom = 1.0 + x_[j];
            if (denom == 0.0) {
                throw std::runtime_error("Encountered x_j = -1 (unexpected for Gauss points).");
            }
            sigma_[j] = (1.0 - x_[j]) / denom;
            eta_[j]   = pref * (2.0 * w_[j]) / denom;
        }
    }
};

#endif // FRACTIONAL_INVERSE_COEFFS_POLYLIB_HPP
