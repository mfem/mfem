#pragma once

#include "mfem.hpp"
#include "mtop_solvers.hpp"

#include <algorithm>
#include <bitset>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "test_cvar_canti_opt_stochastic_initialization_helpers.hpp"
#include "test_cvar_canti_opt_stochastic_simpl_helpers.hpp"

template <std::size_t N, class URBG>
inline std::vector<std::size_t> sample_k_indices_without_replacement(
    const std::vector<std::tuple<std::bitset<N>, real_t, real_t>>& latent_probabilities,
    const real_t cvar_alpha,
    std::size_t k,
    real_t tau,
    URBG& rng)
{
    if (tau < real_t(0))
    {
        throw std::invalid_argument("tau must be nonnegative");
    }

    const std::size_t n = latent_probabilities.size();
    if (k > n)
    {
        throw std::invalid_argument("k cannot exceed latent_probabilities.size()");
    }

    std::vector<double> weights(n, 0.0);
    std::size_t positive_count = 0;

    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& [bits, original_probability, latent_probability] = latent_probabilities[i];
        const real_t w = sigmoid(latent_probability) * original_probability / (1 - cvar_alpha);

        if (w < real_t(0))
        {
            throw std::invalid_argument("distribution contains a negative weight");
        }

        if (w >= tau)
        {
            weights[i] = static_cast<double>(w);
            ++positive_count;
        }
    }

    if (k > positive_count)
    {
        throw std::invalid_argument(
            "Not enough entries with weight >= tau to sample k distinct indices");
    }

    std::vector<std::size_t> sampled_indices;
    sampled_indices.reserve(k);

    for (std::size_t draw = 0; draw < k; ++draw)
    {
        const double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);

        if (total_weight <= 0.0)
        {
            throw std::runtime_error("No positive weight left before completing k draws");
        }

        std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
        const std::size_t idx = dist(rng);

        sampled_indices.push_back(idx);
        weights[idx] = 0.0;
    }

    return sampled_indices;
}

inline real_t cvar_divergence_beween_distributions(
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_1,
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_2,
    real_t cvar_alpha)
{
    int myid = Mpi::WorldRank();
    if (latent_probabilities_1.size() != latent_probabilities_2.size())
    {
        if (myid == 0)
        {
            std::cout << "Different Distribution Sizes." << std::endl;
        }
        throw std::runtime_error(
            "Cannot calculate divergence between two distributions of different sizes.");
    }

    real_t sum = 0.0;
    real_t double_eps = std::numeric_limits<real_t>::epsilon() * 100.0;
    for (size_t i = 0; i < latent_probabilities_1.size(); i++)
    {
        auto [bits_1, original_probability_1, latent_probability_1] = latent_probabilities_1[i];
        auto [bits_2, original_probability_2, latent_probability_2] = latent_probabilities_2[i];

        if (bits_1 != bits_2)
        {
            if (myid == 0)
            {
                std::cout << "Key error." << std::endl;
            }
            throw std::runtime_error(
                "Cannot calculate divergence between two distributions with different supports.");
        }

        const real_t tol = double_eps;
        if (std::abs(original_probability_1 - original_probability_2) > tol)
        {
            throw std::runtime_error(
                "Cannot calculate divergence between two distributions with significantly different original probabilities.");
        }

        sum += (original_probability_1 / (1 - cvar_alpha)) * sigmoid(latent_probability_1) *
               log((sigmoid(latent_probability_1) + double_eps) /
                   (sigmoid(latent_probability_2) + double_eps));
        sum += (original_probability_1 / (1 - cvar_alpha)) * (1 - sigmoid(latent_probability_1)) *
               log((1 - sigmoid(latent_probability_1) + double_eps) /
                   (1 - sigmoid(latent_probability_2) + double_eps));
    }
    return sum;
}

inline real_t proj_latent_onto_probability_simplex(
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>> latent_probabilities_k_jp1_unnormalized,
    std::vector<std::tuple<std::bitset<N>, real_t, real_t>>& latent_probabilities_normalized,
    real_t alpha,
    real_t tol = 1e-12,
    int max_its = 10)
{
    int myid = Mpi::WorldRank();
    max_its = std::max(max_its, 10);

    if (myid == 0)
    {
        std::cout << "[simplex_proj] BEGIN"
                  << " size=" << latent_probabilities_k_jp1_unnormalized.size()
                  << " alpha=" << alpha
                  << " tol=" << tol
                  << " max_its=" << max_its << std::endl;
    }

    real_t negative_latent_max = (real_t)-(DBL_MAX - 1);
    real_t negative_latent_min = (real_t)DBL_MAX;
    int non_finite_latent_count = 0;

    for (auto& [bits, original_probability, latent_probability] : latent_probabilities_k_jp1_unnormalized)
    {
        if (!std::isfinite(latent_probability) || !std::isfinite(original_probability))
        {
            non_finite_latent_count++;
        }
        negative_latent_max = std::max(negative_latent_max, -latent_probability);
        negative_latent_min = std::min(negative_latent_min, -latent_probability);
    }

    if (myid == 0)
    {
        std::cout << "[simplex_proj] Latent summary:"
                  << " min(-latent)=" << negative_latent_min
                  << " max(-latent)=" << negative_latent_max
                  << " non_finite_count=" << non_finite_latent_count
                  << std::endl;
    }

    real_t a = inv_sigmoid(1 - alpha) + negative_latent_min - 1.0;
    real_t b = inv_sigmoid(1 - alpha) + negative_latent_max + 1.0;

    if (myid == 0)
    {
        std::cout << "[simplex_proj] Initial bracket: a=" << a << ", b=" << b
                  << ", inv_sigmoid(1-alpha)=" << inv_sigmoid(1 - alpha) << std::endl;
    }

    auto calculate_summation_for_t = [&](real_t t) -> real_t
    {
        real_t sum = 0.0;
        for (auto& [bits, original_probability, latent_probability] : latent_probabilities_k_jp1_unnormalized)
        {
            sum += (original_probability / (1 - alpha)) * sigmoid(latent_probability + t);
        }
        return sum;
    };

    real_t a_vol_minus = calculate_summation_for_t(a) - 1;
    real_t b_vol_minus = calculate_summation_for_t(b) - 1;

    if (myid == 0)
    {
        std::cout << "[simplex_proj] Endpoint evals:"
                  << " f(a)=" << a_vol_minus
                  << " f(b)=" << b_vol_minus
                  << " product=" << (a_vol_minus * b_vol_minus)
                  << std::endl;
    }

    if (a_vol_minus * b_vol_minus > 0)
    {
        if (myid == 0)
        {
            std::cout << "ERROR: a_vol_minus = " << a_vol_minus << ", b_vol_minus = "
                      << b_vol_minus << std::endl;
        }
        throw std::runtime_error("Invalid bounds for simplex projection.");
    }

    bool done = false;
    real_t x;
    real_t x_vol_minus;

    for (int k = 0; k < max_its; k++)
    {
        if (myid == 0)
        {
            std::cout << "[simplex_proj] Iter " << k
                      << ": a=" << a << ", b=" << b
                      << ", f(a)=" << a_vol_minus << ", f(b)=" << b_vol_minus
                      << ", denom=" << (b_vol_minus - a_vol_minus)
                      << std::endl;
        }

        x = b - b_vol_minus * (b - a) / (b_vol_minus - a_vol_minus);
        x_vol_minus = calculate_summation_for_t(x) - 1;

        if (myid == 0)
        {
            std::cout << "[simplex_proj] Iter " << k
                      << ": x=" << x << ", f(x)=" << x_vol_minus
                      << std::endl;
        }

        if (b_vol_minus * x_vol_minus < 0)
        {
            a = b;
            a_vol_minus = b_vol_minus;

            if (myid == 0)
            {
                std::cout << "[simplex_proj] Iter " << k
                          << ": sign change on [b, x], setting a <- b" << std::endl;
            }
        }
        else
        {
            a_vol_minus = a_vol_minus / 2;

            if (myid == 0)
            {
                std::cout << "[simplex_proj] Iter " << k
                          << ": Illinois damping, new f(a)=" << a_vol_minus << std::endl;
            }
        }
        b = x;
        b_vol_minus = x_vol_minus;

        if (std::abs(x_vol_minus) < tol)
        {
            done = true;
            if (myid == 0)
            {
                std::cout << "[simplex_proj] CONVERGED at iter " << k
                          << " with |f(x)|=" << std::abs(x_vol_minus)
                          << std::endl;
            }
            break;
        }
    }

    for (auto& [bits, original_probability, latent_probability] : latent_probabilities_k_jp1_unnormalized)
    {
        latent_probabilities_normalized.emplace_back(bits, original_probability, latent_probability + x);
    }

    if (!done)
    {
        mfem_warning("Simplex Projection reached maximum iteration without converging. Result may not be accurate.");

        if (myid == 0)
        {
            std::cout << "[simplex_proj] WARNING: reached max iterations without convergence." << std::endl;
        }
    }

    if (myid == 0)
    {
        std::cout << "[simplex_proj] END: t_shift=" << x << std::endl;
    }

    return x;
}

template <std::size_t N, class real_t, class Index>
inline void generate_distribution_from_latents(
    const std::vector<std::tuple<std::bitset<N>, real_t, real_t>>& latent_probabilities,
    real_t cvar_alpha,
    std::discrete_distribution<Index>& dist)
{
    if (latent_probabilities.empty())
    {
        throw std::runtime_error("Cannot build distribution from empty items.");
    }

    std::vector<double> weights;
    weights.reserve(latent_probabilities.size());

    for (auto const& [bits, original_p, latent_p] : latent_probabilities)
    {
        double w = static_cast<double>(sigmoid(latent_p) * original_p / (1 - cvar_alpha));
        if (!std::isfinite(w) || w < 0.0)
        {
            w = 0.0;
        }
        weights.push_back(w);
    }

    const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(sum > 0.0))
    {
        throw std::runtime_error("All weights are zero (or invalid).");
    }

    dist = std::discrete_distribution<Index>(weights.begin(), weights.end());
}

inline void generate_symmetric_index_vector(
    std::vector<std::pair<std::bitset<N>, real_t>> probability_space,
    std::vector<size_t>& symmetric_index_hash)
{
    for (int i = 0; i < static_cast<int>(probability_space.size()); i++)
    {
        std::bitset<N> current_bitset = probability_space[i].first;

        std::bitset<N> symmetrized_bitset = std::bitset<N>();
        for (int k = 0; k < static_cast<int>(N); k++)
        {
            symmetrized_bitset[k] = current_bitset[N - k - 1];
        }

        int symm_index = -1;
        for (int j = 0; j < static_cast<int>(probability_space.size()); j++)
        {
            if (symmetrized_bitset == probability_space[j].first)
            {
                symm_index = j;
                break;
            }
        }
        if (symm_index == -1)
        {
            throw std::runtime_error("No symmetrized bitset found");
        }
        symmetric_index_hash[i] = symm_index;
    }
}

