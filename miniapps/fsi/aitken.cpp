#include "aitken.hpp"

using namespace mfem;
using namespace mfem::fsi;

AitkenRelaxation::AitkenRelaxation(real_t omega_init, real_t omega_min_, real_t omega_max_, int init_steps_)
    : last_omega(omega_init), omega_init(omega_init), omega_min(omega_min_), omega_max(omega_max_), init_steps(init_steps_)
{
   residuals = new ODEStateDataVector(max_history);
}

#ifdef MFEM_USE_MPI
AitkenRelaxation::AitkenRelaxation(MPI_Comm comm, real_t omega_init, real_t omega_min_, real_t omega_max_, int init_steps_)
    : AitkenRelaxation(omega_init, omega_min_, omega_max_, init_steps_)
{
   this->comm = comm;
}
#endif

void AitkenRelaxation::SetSize(const Vector &x0)
{
   auto mem_type = x0.GetMemory().GetMemoryType();
   vsize = x0.Size(); // Store the size of the solution vector
   x_prev.SetSize(vsize, mem_type);
   x_prev = x0;
   r.SetSize(vsize, mem_type);
   r = 0.0;
   residuals->SetSize(vsize, mem_type);
}

AitkenRelaxation::~AitkenRelaxation()
{
   delete residuals;
   residuals = nullptr;
}

real_t AitkenRelaxation::ComputeRelaxation(const Vector &x)
{
    ss++; // Increment stage counter

    // 1. Compute current residual: r_i = x - x_prev
    r = x;
    r -= x_prev;

    // Update stored residuals and previous solution
    residuals->Append(r); // Store current residual
    x_prev = x;           // Update previous solution vector

    // Declare new_omega outside the if-else for wider scope and clarity
    real_t new_omega;

    // 2. If we have previous residual, compute Aitken formula:
    //    ω_{i+1} = -ω_i * (r_{i-1}^T (r_i - r_{i-1})) / ||r_i - r_{i-1}||²
    if ((residuals->Size() > 1) && (ss > init_steps))
    {
        const Vector &r_prev = residuals->Get(1); // r_{i-1}
        r.Add(-1.0, r_prev); // r = r_i - r_{i-1}

#ifndef MFEM_USE_MPI
        real_t rtz = (r_prev * r); // r_prev * (r_i - r_{i-1}) 
        real_t norm_sq = r * r;    // ||z||^2 = ||r_i - r_{i-1}||^2 = rT r
#else
        real_t rtz = InnerProduct(comm, r_prev, r);
        real_t norm_sq = InnerProduct(comm, r, r);
#endif

        real_t effective_tol = TOL; // Default

        // If r_prev_norm_sq (which is last_r_i_norm_sq) is very small, we might be converging
        if (last_r_i_norm_sq > 0)
        {
            effective_tol = std::fmax(TOL, REL_TOL * last_r_i_norm_sq); 
        }

        if (norm_sq < effective_tol)
        {
           new_omega = safety_factor * last_omega; // Try to increase omega to avoid stagnation
        }
        else
        {
           new_omega = -last_omega * (rtz / norm_sq);
        }

        last_omega = new_omega > omega_max ? omega_max : new_omega < omega_min ? omega_min : new_omega; // Clamp to bounds
    }
    else // If this is the first iteration, use initial omega
    {
        last_omega = omega_init;
    }

    // --- Update for next iteration: store the norm of the *current* residual (r_i) ---
    // The `r` vector currently holds `z = r_i - r_{i-1}`.
    // The actual `r_i` is `residuals->Get(0)`.
    Vector &current_residual = residuals->Get(0);
    #ifndef MFEM_USE_MPI
        last_r_i_norm_sq = current_residual * current_residual;
    #else
        last_r_i_norm_sq = InnerProduct(comm, current_residual, current_residual);
    #endif

    return last_omega;
}

void AitkenRelaxation::Reset()
{
   residuals->Reset(); // Reset the residuals buffer
   last_omega = omega_init; // Reset to initial omega
   ss = 0; // Reset stage counter
   last_r_i_norm_sq = 0.0; // Do not carry the previous step's residual norm into the tolerance
}

void AitkenRelaxation::SetBounds(real_t omega_min_, real_t omega_max_)
{
   // Check bounds
   MFEM_ASSERT(omega_min_ < omega_max_, "Minimum bound must be less than maximum bound.");
   omega_min = omega_min_;
   omega_max = omega_max_;
}
