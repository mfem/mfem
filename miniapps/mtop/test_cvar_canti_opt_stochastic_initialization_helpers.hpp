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

#include "test_cvar_canti_opt_stochastic_simpl_helpers.hpp"

// inline constexpr std::size_t kFailureSpaceSize = 6;

/*
 *
 * Get the Failure probabilities or 0, 1, 2, or 3 contiguous holes failing.
 */

inline const int N = 6;

std::vector<std::pair<std::bitset<N>, real_t>> getProbabilitySpace(real_t p1, real_t p2, real_t p3, real_t p4)
{
    std::vector<std::pair<std::bitset<N>, real_t>> probability_space;

    if (p1 < 0 || p2 < 0 || p3 < 0 || p4 < 0 || (N * p1 + (N - 1) * p2 + (N - 2) * p3 + (N - 3) * p4 >= 1))
    {
        throw std::runtime_error("Ill defined probabilities.");
    }
    if (N < 3)
    {
        throw std::runtime_error("Ill defined number of holds");
    }

    real_t p0 = 1.0 - N * p1 - (N - 1) * p2 - (N - 2) * p3 - (N - 3) * p4;
    // real_t p0 = 1.0 - (N - 1) * p2 - (N - 3) * p4;

    std::bitset<N> noFailures;
    probability_space.emplace_back(noFailures, p0);

    // Example: pair of a vector of ints and a float
    // probability_space.emplace_back(std::bitset<N>(), p0);

    for (int i = 0; i < N; i++)
    {
        std::bitset<N> singleFailure;
        singleFailure.set(i);
        probability_space.emplace_back(singleFailure, p1);
    }

    for (int i = 0; i < N - 1; i++)
    {
        std::bitset<N> doubleFailure;
        doubleFailure.set(i);
        doubleFailure.set(i + 1);
        probability_space.emplace_back(doubleFailure, p2);
    }

    for (int i = 0; i < N - 2; i++)
    {
         std::bitset<N> tripleFailure;
         tripleFailure.set(i);
         tripleFailure.set(i + 1);
         tripleFailure.set(i + 2);
         probability_space.emplace_back(tripleFailure, p3);
    }

    for (int i = 0; i < N - 3; i++)
    {
        std::bitset<N> quadrupleFailure;
        quadrupleFailure.set(i);
        quadrupleFailure.set(i + 1);
        quadrupleFailure.set(i + 2);
        quadrupleFailure.set(i + 3);
        probability_space.emplace_back(quadrupleFailure, p4);
    }

    return probability_space;
}

// For testing. Just the basic vector.
std::vector<std::pair<std::bitset<N>, real_t>> nonProbabilitySpace()
{
    std::vector<std::pair<std::bitset<N>, real_t>> probability_space;

    std::bitset<N> noFailures;
    probability_space.emplace_back(noFailures, 1.0);

    return probability_space;
}


/**
 * @brief Initialize latent variable with a symmetric hole in the left-center of the domain.
 *        
 *        HOW IT WORKS - The Eval paradigm:
 *        - HoleCoeff is a "spatial function" that MFEM evaluates at quadrature points
 *        - When ProjectCoefficient() is called, MFEM loops through all mesh elements
 *        - For each element, it calls Eval(T, ip) at each quadrature/integration point
 *        - T.Transform(ip, transip) converts reference element coords → physical domain coords
 *        - We check if the physical point is inside the hole, return different densities accordingly
 * 
 * @param odens_latent The latent density field to initialize
 * @param target_volume Target material volume (vol_fraction * domain_volume)
 * @param domain_volume Total domain volume
 * @param myid MPI rank (for output)
 * @param hole_radius Radius of the hole relative to domain size (default 0.1)
 * @param hole_strength How much to reduce density in the hole (0-1, where 1 removes all material)
 * @param hole_size_x Position of hole center in x-direction as fraction from left edge (default 0.25)
 */
void initialize_with_hole(ParGridFunction &odens_latent,
                          real_t target_volume, real_t domain_volume, int myid,
                          real_t hole_radius = 0.1, real_t hole_strength = 0.8,
                          real_t hole_size_x = 0.25,
                          bool use_heaviside_projection = false,
                          real_t heaviside_eta = 0.5,
                          real_t heaviside_beta = 1.0)
{
    auto &fes = *odens_latent.FESpace();
    auto &mesh = *fes.GetMesh();
    
    // Get mesh bounding box — reduce across all MPI ranks to get global bounds
    const real_t *v0 = mesh.GetVertex(0);
    real_t xmin = v0[0], xmax = v0[0], ymin = v0[1], ymax = v0[1];
    for (int i = 1; i < mesh.GetNV(); i++)
    {
        const real_t *coords = mesh.GetVertex(i);
        xmin = std::min(xmin, coords[0]);
        xmax = std::max(xmax, coords[0]);
        ymin = std::min(ymin, coords[1]);
        ymax = std::max(ymax, coords[1]);
    }
    {
        const auto comm = odens_latent.ParFESpace()->GetComm();
        real_t buf;
        MPI_Allreduce(&xmin, &buf, 1, MPI_DOUBLE, MPI_MIN, comm); xmin = buf;
        MPI_Allreduce(&xmax, &buf, 1, MPI_DOUBLE, MPI_MAX, comm); xmax = buf;
        MPI_Allreduce(&ymin, &buf, 1, MPI_DOUBLE, MPI_MIN, comm); ymin = buf;
        MPI_Allreduce(&ymax, &buf, 1, MPI_DOUBLE, MPI_MAX, comm); ymax = buf;
    }
    if (myid == 0)
    {
        std::cout << "[initialize_with_hole] Global bounding box:"
                  << " x=[" << xmin << ", " << xmax << "]"
                  << " y=[" << ymin << ", " << ymax << "]" << std::endl;
    }
    
    real_t domain_width = xmax - xmin;
    real_t domain_height = ymax - ymin;
    real_t y_mid = 0.5 * (ymin + ymax);
    
    // Primary hole: left-center
    real_t hole_center_x = xmin + hole_size_x * domain_width;
    real_t hole_center_y = y_mid;
    // Mirror hole: reflected about y_mid (guarantees vertical symmetry)
    real_t hole_mirror_y = ymin + ymax - hole_center_y; // = y_mid when hole_center_y == y_mid
    
    // Baseline latent value corresponding to the target volume fraction
    real_t baseline_latent = inv_sigmoid(target_volume / domain_volume);

    // Initialize all DOFs to baseline
    odens_latent = baseline_latent;
    
    // Create a coefficient that reduces density in both hole and its mirror
    class HoleCoeff : public Coefficient {
    public:
        real_t center_x, center_y, mirror_y, radius, strength;
        real_t baseline_latent;
        
        HoleCoeff(real_t cx, real_t cy, real_t my, real_t r, real_t s, real_t bl) 
            : center_x(cx), center_y(cy), mirror_y(my), radius(r), strength(s), baseline_latent(bl) {}
        
        virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
        {
            // T.Transform converts reference element coordinates to physical coordinates
            Vector transip;
            T.Transform(ip, transip);
            
            real_t dx = transip(0) - center_x;

            // Check distance to primary hole
            real_t dy1 = transip(1) - center_y;
            real_t dist1 = std::sqrt(dx*dx + dy1*dy1);

            // Check distance to mirror hole (reflected about y_mid)
            real_t dy2 = transip(1) - mirror_y;
            real_t dist2 = std::sqrt(dx*dx + dy2*dy2);

            if (dist1 < radius || dist2 < radius) {
                // Inside either hole: interpolate toward near-zero latent (very little material)
                return baseline_latent - strength * (baseline_latent - inv_sigmoid(0.01));
            }
            // Outside both holes: return baseline latent value
            return baseline_latent;
        }
    } hole_coeff(hole_center_x, hole_center_y, hole_mirror_y, hole_radius, hole_strength, baseline_latent);
    
    // Project the hole coefficient onto the latent field
    odens_latent.ProjectCoefficient(hole_coeff);

    // Normalize
    real_t material_volume = proj(odens_latent, target_volume, domain_volume, 1e-12, 25,
                                  use_heaviside_projection, heaviside_eta, heaviside_beta);

    if (myid == 0) {
        std::cout << "Initialized with hole: target_volume=" << target_volume 
                  << ", actual_material_volume=" << material_volume 
                  << ", actual_volume_fraction=" << material_volume / domain_volume 
                  << std::endl;
    }
}
