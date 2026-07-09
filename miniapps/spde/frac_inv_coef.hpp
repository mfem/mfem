#ifndef FRACTIONAL_INVERSE_RATIONAL_COEFFICIENTS_HPP
#define FRACTIONAL_INVERSE_RATIONAL_COEFFICIENTS_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <limits>

class FractionalInverseRationalCoefficients {
public:
    // Constructs Gauss–Jacobi nodes/weights for
    //   w(x) = (1-x)^alpha (1+x)^gamma,  alpha=-beta, gamma=beta-1,
    // then computes
    //   sigma_j = (1-x_j)/(1+x_j)
    //   eta_j   = (sin(pi*beta)/pi) * (2*w_j)/(1+x_j)
    //
    // Valid for beta in (0,1) so that alpha,gamma > -1.
    FractionalInverseRationalCoefficients(double beta, std::size_t m)
        : beta_(beta), m_(m)
    {
        if (!(beta_ > 0.0 && beta_ < 1.0)) {
            throw std::invalid_argument("beta must be in (0,1) so alpha=-beta and gamma=beta-1 are > -1.");
        }
        if (m_ == 0) {
            throw std::invalid_argument("m must be >= 1.");
        }

        alpha_ = -beta_;
        gamma_ = beta_ - 1.0;

        compute_nodes_weights();
        compute_sigma_eta();
    }

    std::size_t size() const { return m_; }

    const std::vector<double>& x()  const { return x_; }
    const std::vector<double>& w() const { return w_; }
    const std::vector<double>& sigma()    const { return sigma_; }
    const std::vector<double>& eta()      const { return eta_; }

private:
    double beta_{};
    std::size_t m_{};

    // Jacobi parameters for w(x) = (1-x)^alpha (1+x)^gamma
    double alpha_{};
    double gamma_{};

    std::vector<double> x_;     // Gauss-Jacobi nodes on (-1,1)
    std::vector<double> w_;     // Gauss-Jacobi weights for that weight function
    std::vector<double> sigma_; // (1-x)/(1+x)
    std::vector<double> eta_;   // (sin(pi*beta)/pi) * 2*w/(1+x)

    // Evaluate Jacobi polynomial P_n^{(a,b)}(x) with the standard three-term recurrence.
    static double jacobiP(int n, double a, double b, double x) {
        if (n == 0) return 1.0;
        if (n == 1) {
            return 0.5 * ((a - b) + (a + b + 2.0) * x);
        }

        double Pnm2 = 1.0;
        double Pnm1 = 0.5 * ((a - b) + (a + b + 2.0) * x);
        double Pn = 0.0;

        for (int k = 2; k <= n; ++k) {
            const double kd = static_cast<double>(k);

            const double A1 = 2.0 * kd * (kd + a + b) * (2.0 * kd + a + b - 2.0);
            const double A2 = (2.0 * kd + a + b - 1.0)
                            * ((2.0 * kd + a + b) * (2.0 * kd + a + b - 2.0) * x + (a*a - b*b));
            const double A3 = 2.0 * (kd + a - 1.0) * (kd + b - 1.0) * (2.0 * kd + a + b);

            // P_k = (A2 * P_{k-1} - A3 * P_{k-2}) / A1
            Pn = (A2 * Pnm1 - A3 * Pnm2) / A1;

            Pnm2 = Pnm1;
            Pnm1 = Pn;
        }
        return Pn;
    }

    // Derivative using: d/dx P_n^{(a,b)}(x) = 0.5*(n+a+b+1)*P_{n-1}^{(a+1,b+1)}(x)
    static double jacobiP_derivative(int n, double a, double b, double x) {
        if (n == 0) return 0.0;
        const double c = 0.5 * (static_cast<double>(n) + a + b + 1.0);
        return c * jacobiP(n - 1, a + 1.0, b + 1.0, x);
    }

    // A decent initial guess for roots (works well for moderate m).
    // x_k ≈ cos( (2k + a - 1)π / (2n + a + b) ), k=1..n
    static double initial_root_guess(int k1_based, int n, double a, double b) {
        const double num = (2.0 * k1_based + a - 1.0) * M_PI;
        const double den = (2.0 * n + a + b);
        return std::cos(num / den);
    }

    void compute_nodes_weights() {
        const int n = static_cast<int>(m_);
        const double a = alpha_;
        const double b = gamma_;

        x_.assign(m_, 0.0);
        w_.assign(m_, 0.0);

        // Constant factor from Gauss–Jacobi weight formula (Wikipedia form):
        // λ_i = C / (P_n'(x_i) * P_{n+1}(x_i)),
        // C = - (2n+a+b+2)/(n+a+b+1) * Γ(n+a+1)Γ(n+b+1)/(Γ(n+a+b+1)(n+1)!) * 2^{a+b}
        const double lnC =
            std::log((2.0 * n + a + b + 2.0) / (n + a + b + 1.0))   // we'll add sign separately
            + std::lgamma(n + a + 1.0)
            + std::lgamma(n + b + 1.0)
            - std::lgamma(n + a + b + 1.0)
            - std::lgamma(n + 2.0)   // (n+1)! = Γ(n+2)
            + (a + b) * std::log(2.0);

        const double C = -std::exp(lnC); // includes the minus sign in the formula

        const double eps = 50.0 * std::numeric_limits<double>::epsilon();
        const int max_newton = 50;

        for (int k = 1; k <= n; ++k) {
            double x = initial_root_guess(k, n, a, b);

            // Newton iteration on P_n^{(a,b)}(x)=0
            for (int it = 0; it < max_newton; ++it) {
                const double Pn  = jacobiP(n, a, b, x);
                const double dPn = jacobiP_derivative(n, a, b, x);

                // Guard against pathological derivative (rare for reasonable n)
                if (dPn == 0.0) break;

                const double dx = -Pn / dPn;
                x += dx;

                if (std::abs(dx) <= eps * (1.0 + std::abs(x))) break;
            }

            // Store node
            x_[static_cast<std::size_t>(k - 1)] = x;

            // Compute weight
            const double dPn = jacobiP_derivative(n, a, b, x);
            const double Pn1 = jacobiP(n + 1, a, b, x);

            // This should be positive; if roundoff causes tiny negative, keep as computed.
            w_[static_cast<std::size_t>(k - 1)] = C / (dPn * Pn1);
        }

        // Optional: ensure ascending order in x (nice for reproducibility)
        // Simple insertion sort (m is usually small: 8–64).
        for (std::size_t i = 1; i < m_; ++i) {
            double xi = x_[i];
            double wi = w_[i];
            std::size_t j = i;
            while (j > 0 && x_[j - 1] > xi) {
                x_[j] = x_[j - 1];
                w_[j] = w_[j - 1];
                --j;
            }
            x_[j] = xi;
            w_[j] = wi;
        }
    }

    void compute_sigma_eta() {
        sigma_.assign(m_, 0.0);
        eta_.assign(m_, 0.0);

        const double pref = std::sin(M_PI * beta_) / M_PI;

        for (std::size_t j = 0; j < m_; ++j) {
            const double x = x_[j];

            // Avoid division by zero if x ~ -1 (should not happen for Gauss nodes, but be safe)
            const double denom = (1.0 + x);
            if (denom == 0.0) {
                throw std::runtime_error("Encountered x = -1 in nodes (unexpected).");
            }

            sigma_[j] = (1.0 - x) / denom;
            eta_[j]   = pref * (2.0 * w_[j]) / denom;
        }
    }
};


//Example usage:
/*
#include <iostream>

int main() {
    double beta = 0.7;
    std::size_t m = 10;

    FractionalInverseRationalCoefficients coeffs(beta, m);

    for (std::size_t j = 0; j < m; ++j) {
        std::cout << "j=" << j
                  << " x=" << coeffs.nodes_x()[j]
                  << " w=" << coeffs.weights_w()[j]
                  << " sigma=" << coeffs.sigma()[j]
                  << " eta=" << coeffs.eta()[j]
                  << "\n";
    }
}
*/



class ResolventSumApproxF {
public:
  // Approximates f(x)=1/(1+x^beta) on [lmin,lmax] by r(x)=Σ eta_j/(x+sigma_j).
  // m_terms >= 2 recommended.
  ResolventSumApproxF(double beta, double lmin, double lmax,
                      std::size_t m_terms,
                      double overspan_gamma = 4.0)
      : beta_(beta), lmin_(lmin), lmax_(lmax), m_(m_terms), gamma_(overspan_gamma) {
    validate_();
    build_sigma_();
    fit_eta_by_chebyshev_interpolation_();
  }

  const std::vector<double>& eta()   const { return eta_; }
  const std::vector<double>& sigma() const { return sigma_; }

  // Evaluate r(x)
  double eval(double x) const {
    double s = 0.0;
    for (std::size_t j = 0; j < m_; ++j) s += eta_[j] / (x + sigma_[j]);
    return s;
  }

  // Evaluate f(x)
  double f(double x) const {
    return 1.0 / (1.0 + std::pow(x, beta_));
  }

  // Certified sup-norm error bound using a grid of n_grid points.
  // Bound: max_grid_error + (Δ/2)*(sup|f'| + sup|r'|)
  double certified_sup_error(std::size_t n_grid = 2001) const {
    if (n_grid < 2) throw std::invalid_argument("n_grid must be >= 2");

    const double a = lmin_;
    const double b = lmax_;
    const double dx = (b - a) / double(n_grid - 1);

    // max error on grid
    double max_err = 0.0;
    for (std::size_t i = 0; i < n_grid; ++i) {
      const double x = a + dx * double(i);
      max_err = std::max(max_err, std::abs(f(x) - eval(x)));
    }

    // sup |f'| bound on [lmin,lmax]
    const double fp_sup = beta_ * std::pow(lmin_, beta_ - 1.0) /
                          std::pow(1.0 + std::pow(lmin_, beta_), 2.0);

    // sup |r'| bound on [lmin,lmax]
    double rp_sup = 0.0;
    for (std::size_t j = 0; j < m_; ++j) {
      const double denom = (lmin_ + sigma_[j]);
      rp_sup += std::abs(eta_[j]) / (denom * denom);
    }

    return max_err + 0.5 * dx * (fp_sup + rp_sup);
  }

private:
  double beta_{}, lmin_{}, lmax_{};
  std::size_t m_{};
  double gamma_{};

  std::vector<double> sigma_;
  std::vector<double> eta_;

  static constexpr double kPi() { return 3.141592653589793238462643383279502884; }

  void validate_() const {
    if (!std::isfinite(beta_) || beta_ < 0.0 || beta_ > 1.0)
      throw std::invalid_argument("beta must be in [0,1]");
    if (!std::isfinite(lmin_) || !std::isfinite(lmax_) || !(lmin_ > 0.0) || !(lmax_ > 0.0))
      throw std::invalid_argument("Require lmin>0, lmax>0 (for interval approximation)");
    if (lmin_ > lmax_) throw std::invalid_argument("Require lmin <= lmax");
    if (m_ < 2) throw std::invalid_argument("Require m_terms >= 2");
    if (!std::isfinite(gamma_) || gamma_ <= 0.0)
      throw std::invalid_argument("overspan_gamma must be finite and > 0");
  }

  void build_sigma_() {
    sigma_.assign(m_, 0.0);

    const double log_lmin = std::log(lmin_);
    const double log_lmax = std::log(lmax_);
    const double s0 = 0.5 * (log_lmin + log_lmax);
    const double log_kappa = log_lmax - log_lmin;

    // Overspan: cover [s0 - gamma*log_kappa, s0 + gamma*log_kappa] in m points
    const double span = 2.0 * gamma_ * log_kappa;
    const double h = span / double(m_ - 1);

    for (std::size_t j = 0; j < m_; ++j) {
      const double sj = (s0 - 0.5 * span) + h * double(j);
      sigma_[j] = std::exp(sj);
    }
  }

  // Chebyshev points of the first kind mapped to [lmin,lmax] (m points)
  std::vector<double> chebyshev_points_() const {
    std::vector<double> x(m_);
    const double c = 0.5 * (lmin_ + lmax_);
    const double r = 0.5 * (lmax_ - lmin_);
    for (std::size_t i = 0; i < m_; ++i) {
      const double theta = (2.0 * double(i) + 1.0) * kPi() / (2.0 * double(m_));
      x[i] = c + r * std::cos(theta);
    }
    return x;
  }

  // Solve linear system A eta = b with partial pivoting (A is m x m).
  static std::vector<double> solve_linear_system_(std::vector<double> A, std::vector<double> b, std::size_t m) {
    // A is row-major m*m
    auto idx = [m](std::size_t i, std::size_t j) { return i * m + j; };

    for (std::size_t k = 0; k < m; ++k) {
      // pivot
      std::size_t piv = k;
      double best = std::abs(A[idx(k, k)]);
      for (std::size_t i = k + 1; i < m; ++i) {
        const double v = std::abs(A[idx(i, k)]);
        if (v > best) { best = v; piv = i; }
      }
      if (best <= std::numeric_limits<double>::min())
        throw std::runtime_error("Singular/ill-conditioned system in fit");

      if (piv != k) {
        for (std::size_t j = k; j < m; ++j) std::swap(A[idx(k, j)], A[idx(piv, j)]);
        std::swap(b[k], b[piv]);
      }

      // eliminate
      const double Akk = A[idx(k, k)];
      for (std::size_t i = k + 1; i < m; ++i) {
        const double factor = A[idx(i, k)] / Akk;
        A[idx(i, k)] = 0.0;
        for (std::size_t j = k + 1; j < m; ++j) A[idx(i, j)] -= factor * A[idx(k, j)];
        b[i] -= factor * b[k];
      }
    }

    // back-substitute
    std::vector<double> x(m, 0.0);
    for (std::ptrdiff_t i = (std::ptrdiff_t)m - 1; i >= 0; --i) {
      double s = b[(std::size_t)i];
      for (std::size_t j = (std::size_t)i + 1; j < m; ++j) s -= A[idx((std::size_t)i, j)] * x[j];
      x[(std::size_t)i] = s / A[idx((std::size_t)i, (std::size_t)i)];
    }
    return x;
  }

  void fit_eta_by_chebyshev_interpolation_() {
    const auto x = chebyshev_points_();

    // Build A_ij = 1/(x_i + sigma_j), b_i = f(x_i)
    std::vector<double> A(m_ * m_, 0.0);
    std::vector<double> b(m_, 0.0);

    for (std::size_t i = 0; i < m_; ++i) {
      b[i] = f(x[i]);
      for (std::size_t j = 0; j < m_; ++j) {
        A[i * m_ + j] = 1.0 / (x[i] + sigma_[j]);
      }
    }

    eta_ = solve_linear_system_(A, b, m_);
  }
};


#endif // FRACTIONAL_INVERSE_RATIONAL_COEFFICIENTS_HPP