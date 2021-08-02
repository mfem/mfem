#include "transfer.hpp"

#include "par_moonolith_instance.hpp"

namespace mfem
{

void InitTransfer(int argc, char *argv[])
{
   moonolith::Moonolith::Init(argc, argv);
}
int FinalizeTransfer() { return moonolith::Moonolith::Finalize(); }

#ifdef MFEM_USE_MPI
void InitTransfer(int argc, char *argv[], MPI_Comm comm)
{
   moonolith::Moonolith::Init(argc, argv, comm);
}
#endif

} // namespace mfem