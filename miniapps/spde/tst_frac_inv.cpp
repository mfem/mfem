#include "mfem.hpp"
#include "frac_inv_coef.hpp"
#include "frac_inv_coef_polylib.hpp"
#include "aaacoeffs.hpp"
#include "aaa_rational.hpp"

using namespace std;
using namespace mfem;

constexpr auto MESH_TRI = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_tri.mesh";
constexpr auto MESH_QUAD = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_quad.mesh";


static double exact_g(double lambda, double beta)
{
    if (lambda == 0.0) return 1.0;
    return 1.0 / (std::pow(lambda, beta) + 1.0);
}

static std::complex<double>
eval_rational(double lambda,
              const AAACoefficients::Result& r)
{
    std::complex<double> val = r.c0;
    for (size_t j = 0; j < r.poles.size(); ++j) {
        val += r.eta[j] / (lambda - r.poles[j]);
        // equivalently: eta[j] / (lambda + sigma[j])
    }
    return val;
}


int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = MESH_QUAD;
   const char *device_config = "cpu";
   int order = 1;
   bool pa = false;
   bool dfem = false;
   bool mesh_tri = false;
   bool mesh_quad = false;
   int par_ref_levels = 4;
   int ser_ref_levels = 0;
   bool paraview = false;
   bool visualization = true;
   real_t s=0.0;
   real_t beta = 0.7;
   int m = 10;

   OptionsParser args(argc, argv);
   args.AddOption(&beta, "-b", "--beta", "Fractional inverse exponent beta in (0,1).");
   args.AddOption(&m, "-m", "--m", "Number of terms m in the rational approximation.");
   args.ParseCheck();

   FractionalInverseRationalCoefficients coeffs(beta, m);
   FractionalInverseCoeffsPolylib coeffp(beta, m);

    cout << "Rational approximation coefficients (RationalCoefficients class):\n";
    for (std::size_t j = 0; j < m; ++j) {
        cout << "j=" << j
             << " x=" << coeffs.x()[j]
             << " w=" << coeffs.w()[j]
             << " sigma=" << coeffs.sigma()[j]
             << " eta=" << coeffs.eta()[j]
             << "\n";
    }  

    cout << "\nRational approximation coefficients (Polylib-based class):\n";
    for (std::size_t j = 0; j < m; ++j) {
        cout << "j=" << j
             << " x=" << coeffp.x()[j]
             << " w=" << coeffp.w()[j]
             << " sigma=" << coeffp.sigma()[j]
             << " eta=" << coeffp.eta()[j]
             << "\n";
    }

   real_t val=0.5;

   real_t app1 = 0.0;
   real_t app2 = 0.0;
   real_t ref = 1.0/ std::pow(val, beta);

   for(int j=0; j<m; j++)
   {
      app1 += coeffs.eta()[j] / (val + coeffs.sigma()[j]);
      app2 += coeffp.eta()[j] / (val + coeffp.sigma()[j]);
   }

   std::cout << "\nApproximation at v=" << val << ":\n";
   std::cout << " Reference: " << ref << "\n";
   std::cout << " RationalCoefficients class: " << app1 << "\n";
   std::cout << " Polylib-based class:       " << app2 << "\n";
   std::cout << " Relative errors:\n";
   std::cout << "  RationalCoefficients class: " << std::abs(app1 - ref)/std::abs(ref) << "\n";
   std::cout << "  Polylib-based class:       " << std::abs(app2 - ref)/std::abs(ref) << "\n";



   ref= 1.0/ (std::pow(val, beta)+1);
   real_t app3=0.0;

   ResolventSumApproxF frac_inv_stieltjes(beta, val-0.1, val+0.1, m);

    cout << "\nSinc-Stieltjes quadrature coefficients:\n";
    for(int j=0; j<m; j++)
    {
        cout << "j=" << j
             << " sigma=" << frac_inv_stieltjes.sigma()[j]
             << " eta=" << frac_inv_stieltjes.eta()[j]
             << "\n";
    }

    for(int j=0; j<m; j++)
    {
        app3 += frac_inv_stieltjes.eta()[j] / (val + frac_inv_stieltjes.sigma()[j]);
    }   

    std::cout << "\nApproximation of 1/(v^beta +1) at v=" << val << ":\n";
    std::cout << " Reference: " << ref << "\n";
    std::cout << " SincStieltjesQuadrature class: " << app3 << "\n";
    std::cout << " Relative error:\n";
    std::cout << "  SincStieltjesQuadrature class: " << std::abs(app3 - ref)/std::abs(ref) << "\n";
    std::cout << " Certified sup-norm error bound: "
              << frac_inv_stieltjes.certified_sup_error() << "\n";


    const double lmin = val - 0.1;
    const double lmax = val + 0.1;
    
    {
    AAACoefficients aaa(beta, lmin, lmax,
                        /*max_terms=*/40,
                        /*tol=*/1e-16,
                        /*n_sample=*/500,
                        /*residue_rel_tol=*/1e-14,
                        /*froissart_rel_tol=*/1e-10);

    // --- build approximation ---
    auto res = aaa.fit();

    std::cout << "AAA scalar test\n";
    std::cout << "beta = " << beta << "\n";
    std::cout << "interval = [" << lmin << ", " << lmax << "]\n\n";

    std::cout << "Total poles found     : "
              << res.kept.size() + res.dropped.size() << "\n";
    std::cout << "Kept poles            : " << res.poles.size() << "\n";
    std::cout << "Dropped (fake) poles  : " << res.dropped.size() << "\n\n";

    std::cout << "c0 = " << res.c0 << "\n\n";

    for (size_t j = 0; j < res.poles.size(); ++j) {
        std::cout << "j=" << j
                  << "  pole=" << res.poles[j]
                  << "  sigma=" << res.sigma[j]
                  << "  eta=" << res.eta[j] << "\n";
    }

    std::cout << "\nApproximation of 1/(v^beta +1) at v=" << val << ":\n";
    std::cout << " Reference: " << ref << "\n";
    std::cout << " AAA approximation: " << std::real(eval_rational(val, res)) << "\n";
    std::cout << " Relative error:\n";
    std::cout << "  AAA approximation: "
              << std::abs(std::real(eval_rational(val, res)) - ref) / std::abs(ref) << "\n";

    // --- error test on dense grid ---
    const int ntest = 2000;
    double max_rel = 0.0;
    double rms_rel = 0.0;

    for (int i = 0; i < ntest; ++i) {
        double t = (double)i / (double)(ntest - 1);
        double lambda = std::exp(std::log(lmin) +
                                 t * (std::log(lmax) - std::log(lmin)));

        double g = exact_g(lambda, beta);
        double r = std::real(eval_rational(lambda, res));

        double rel = std::abs(r - g) / std::max(g, 1e-15);
        max_rel = std::max(max_rel, rel);
        rms_rel += rel * rel;
    }

    rms_rel = std::sqrt(rms_rel / ntest);

    std::cout << "\nError statistics:\n";
    std::cout << "  max relative error = " << std::scientific << max_rel << "\n";
    std::cout << "  rms relative error = " << std::scientific << rms_rel << "\n";
    
    } // end AAA coefficients test block

    {
    AAARational aaar(lmin, lmax,
                    /*max_terms=*/m,
                    /*tol=*/1e-16,
                    /*n_sample=*/800,
                    /*residue_rel_tol=*/1e-14,
                    /*froissart_rel_tol=*/1e-10,
                    /*log_sample=*/true,
                    /*enable_filtering=*/true);

    // -------------------------------
    // Fit rational approximation
    // -------------------------------

    auto f_exact = [&beta](double x) -> double {
        if (x == 0.0) return 1.0;                 // consistent with 1/(0^beta + 1)
        return 1.0 / (std::pow(x, beta) + 1.0);   // g(x) = (x^beta + 1)^{-1}
    };


    auto res = aaar.fit(f_exact);

    std::cout << "\nAAA Rational approximation test\n";
    std::cout << "Interval: [" << lmin << ", " << lmax << "]\n";
    std::cout << "Number of poles kept   : " << res.poles.size() << "\n";
    std::cout << "Number of poles dropped: " << res.dropped.size() << "\n\n";

    std::cout << "Constant term c0 = " << res.c0 << "\n\n";

    for (size_t k = 0; k < res.poles.size(); ++k) {
        std::cout << "k=" << std::setw(2) << k
                  << "  pole="  << std::setw(18) << res.poles[k]
                  << "  sigma=" << std::setw(18) << res.sigma[k]
                  << "  eta="   << std::setw(18) << res.eta[k] << "\n";
    }

    std::cout << "\nApproximation of 1/(v^beta +1) at v=" << val << ":\n";
    std::cout << " Reference: " << ref << "\n";
    std::cout << " AAA approximation: " << AAARational::eval_partial_fraction(val, res) << "\n";
    std::cout << " Relative error:\n";
    std::cout << "  AAA approximation: "
              << std::abs(std::real(AAARational::eval_partial_fraction(val, res)) - ref) / std::abs(ref) << "\n";


    // -------------------------------
    // Accuracy test on dense grid
    // -------------------------------
    const int ntest = 3000;
    double max_rel_bary = 0.0;
    double max_rel_pf   = 0.0;
    double rms_rel_pf   = 0.0;

    for (int i = 0; i < ntest; ++i) {
        double t = double(i) / double(ntest - 1);

        // log-distributed test points
        double x = std::exp(std::log(lmin) +
                            t * (std::log(lmax) - std::log(lmin)));

        double fex = f_exact(x);

        double rb = AAARational::eval_barycentric(x, res);
        double rp = std::real(AAARational::eval_partial_fraction(x, res));

        double rel_b = std::abs(rb - fex) / std::max(fex, 1e-15);
        double rel_p = std::abs(rp - fex) / std::max(fex, 1e-15);

        max_rel_bary = std::max(max_rel_bary, rel_b);
        max_rel_pf   = std::max(max_rel_pf,   rel_p);
        rms_rel_pf  += rel_p * rel_p;
    }

    rms_rel_pf = std::sqrt(rms_rel_pf / ntest);

    std::cout << "\nError statistics:\n";
    std::cout << "  max relative error (barycentric) = "
              << std::scientific << max_rel_bary << "\n";
    std::cout << "  max relative error (partial frac) = "
              << std::scientific << max_rel_pf << "\n";
    std::cout << "  rms relative error (partial frac) = "
              << std::scientific << rms_rel_pf << "\n";

    } // end AAA rational approximation test block

   return EXIT_SUCCESS;
   
}