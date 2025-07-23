#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../fem/pfespace.hpp"
#include "../fem/pgridfunc.hpp"

namespace mfem
{

/// @brief Synchronize boundary values across processors, but only for DoFs marked in ldof_marker
/// @param[in] pfes Parallel finite element space
/// @param[in,out] values Array of values to synchronize
/// @param[in] ldof_marker Array marking which DoFs to synchronize (1) or leave untouched (0)
void SynchronizeMarkedDoFs(ParFiniteElementSpace *pfes, Array<double> &values, const Array<int> &ldof_marker);

/// @brief Synchronize boundary values across processors, but only for DoFs marked in ldof_marker
/// @param[in] pfes Parallel finite element space
/// @param[in,out] gf ParGridFunction with values to synchronize
/// @param[in] ldof_marker Array marking which DoFs to synchronize (1) or leave untouched (0)
void SynchronizeMarkedDoFs(ParFiniteElementSpace *pfes, ParGridFunction &gf, const Array<int> &ldof_marker);

} // namespace mfem

#endif // MFEM_USE_MPI
