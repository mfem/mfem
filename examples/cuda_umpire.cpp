//#if defined(MFEM_USE_UMPIRE) && (defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP))

#include "mfem.hpp"
#include <unistd.h>
#include <stdio.h>
#include "umpire/Umpire.hpp"

#define REQUIRE(a) if (!(a)) { printf("check failed: %s\n", #a); }


using namespace mfem;

constexpr unsigned num_elems = 1024;
constexpr unsigned num_bytes = num_elems * sizeof(double);
constexpr double host_val = 1.0;
constexpr double dev_val = 1.0;

static long alloc_size(int id)
{
   auto &rm = umpire::ResourceManager::getInstance();
   auto a                    = rm.getAllocator(id);
   return a.getCurrentSize();
}

int main(int argc, char ** argv)
{
   auto &rm = umpire::ResourceManager::getInstance();

   const int permanent = umpire::Allocator(
                            rm.makeAllocator<
                            umpire::strategy::DynamicPoolMap,
                            true>("MFEM-Permanent-Device-Pool",
                                  rm.getAllocator("DEVICE"), 0, 0))
                         .getId();

   const int temporary = umpire::Allocator(
                            rm.makeAllocator<
                            umpire::strategy::DynamicPoolList,
                            true>("MFEM-Temporary-Device-Pool",
                                  rm.getAllocator("DEVICE"), 0, 0))
                         .getId();

   Device::SetHostUmpire(false);
   Device::SetDeviceUmpire(true);
   MemoryManager::SetUmpireDeviceAllocatorId(permanent);
   MemoryManager::SetUmpireDeviceTempAllocatorId(temporary);
   Device device("cuda");

   printf("Both pools should be empty at startup: ");
   REQUIRE(alloc_size(permanent) == 0);
   REQUIRE(alloc_size(temporary) == 0);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // allocate on host, use permanent device memory when needed
   Vector host_perm(num_elems);
   // allocate on host, use temporary device memory when needed
   Vector host_temp(num_elems); host_temp = host_val; host_temp.UseTemporary(true);

   printf("Allocated %u bytes on the host, pools should still be empty: ",
          num_bytes*2);
   REQUIRE((alloc_size(permanent) == 0 && alloc_size(temporary) == 0));
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // uses permanent device memory
   host_perm.Write();

   printf("Write of size %u to perm, temp should still be empty: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes)
   REQUIRE(alloc_size(temporary) == 0);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // uses temporary device memory
   double * d_host_temp = host_temp.ReadWrite();
   //MFEM_FORALL(i, num_elems, { d_host_temp[i] = dev_val; });

   printf("Write of size %u to temp: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes)
   REQUIRE(alloc_size(temporary) == num_bytes);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // allocates in permanent device memory
   Vector dev_perm(num_elems, MemoryClass::DEVICE);

   printf("Allocate %u more bytes in permanent memory: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes*2);
   REQUIRE(alloc_size(temporary) == num_bytes);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // allocates in temporary device memory
   // shorten with using statement
   using mc = mfem::MemoryClass;
   Vector dev_temp(num_elems, mc::DEVICE_TEMP);
   //double * d_dev_temp = dev_temp.Write();
   //MFEM_FORALL(i, num_elems, { d_dev_temp[i] = dev_val; });

   printf("Allocate %u more bytes in temporary memory: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes*2)
   REQUIRE(alloc_size(temporary) == num_bytes*2);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // remove from temporary memory
   // don't copy to host, verify that the value is still the "host" value
   host_temp.DeleteDevice(false);
   REQUIRE(host_temp[0] == host_val);
   // copy to host, verify that the value is the "device" value
   dev_temp.DeleteDevice();
   // TODO TMS
   REQUIRE(dev_temp[0] == dev_val);

   printf("Delete all temporary memory: ");
   REQUIRE(alloc_size(permanent) == num_bytes*2);
   REQUIRE(alloc_size(temporary) == 0);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   return 0;
}

//#endif // MFEM_USE_UMPIRE && MFEM_USE_CUDA
