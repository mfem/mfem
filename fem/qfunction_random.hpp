#ifndef MFEM_QFUNCTION_RANDOM
#define MFEM_QFUNCTION_RANDOM

#include "../config/config.hpp"
#include <cstdint>

namespace mfem
{
namespace internal
{

struct PhiloxResult
{
   std::uint32_t words[4];
};

MFEM_HOST_DEVICE inline PhiloxResult Philox4x32(
   std::uint32_t counter_low, std::uint32_t stream,
   std::uint32_t sample_low, std::uint32_t sample_high,
   std::uint32_t key_low, std::uint32_t key_high)
{
   PhiloxResult result{{counter_low, stream, sample_low, sample_high}};
   for (int round = 0; round < 10; ++round)
   {
      const std::uint64_t product_low =
         std::uint64_t(0xd2511f53U) * result.words[0];
      const std::uint64_t product_high =
         std::uint64_t(0xcd9e8d57U) * result.words[2];
      const std::uint32_t next_low =
         std::uint32_t(product_high >> 32) ^ result.words[1] ^ key_low;
      const std::uint32_t next_high =
         std::uint32_t(product_low >> 32) ^ result.words[3] ^ key_high;
      result.words[0] = next_low;
      result.words[1] = std::uint32_t(product_high);
      result.words[2] = next_high;
      result.words[3] = std::uint32_t(product_low);
      key_low += 0x9e3779b9U;
      key_high += 0xbb67ae85U;
   }
   return result;
}

}
}

#endif
