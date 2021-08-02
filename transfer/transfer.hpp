#ifndef MFEM_TRANSFER
#define MFEM_TRANSFER

#include "../fem/fem.hpp"

#ifdef MFEM_USE_MOONOLITH
#include "mortarassembler.hpp"
#ifdef MFEM_USE_MPI
#include "parallel/pmortarassembler.hpp"
#endif // MFEM_USE_MPI
#endif // MFEM_USE_MOONOLITH

namespace mfem
{

void InitTransfer(int argc, char *argv[]);
int FinalizeTransfer();

#ifdef MFEM_USE_MPI
void InitTransfer(int argc, char *argv[], MPI_Comm comm);
#endif

} // namespace mfem

#endif // MFEM_TRANSFER
