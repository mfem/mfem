#include "reusable_storage.hpp"

#include <type_traits>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#include <immintrin.h>
#endif

namespace mfem
{
namespace internal
{
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) ||                     \
   defined(__INTEL_COMPILER)
namespace
{
template <class T> struct IdxTypeHelper;

template <class R, class T0, class T1> struct IdxTypeHelper<R (*)(T0, T1)>
{
   using type = std::remove_pointer_t<T0>;
};

/// Identifies the base index type for _BitScanForward64
template <class T> struct IdxType : IdxTypeHelper<std::add_pointer_t<T>>
{};
} // namespace
#endif

/// index of first unset bit + 1 starting from the LSB, or 0 if all bits are set
static int LowestUnset(uint64_t val)
{
#if defined(__GNUC__) || defined(__clang__)
   static_assert(sizeof(long long) == 8, "long long expected to be 64-bits");
   return __builtin_ffsll(~val);
#elif defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) ||                   \
   defined(__INTEL_COMPILER)
   if (val == ~0ull)
   {
      return 0;
   }
   // argument 0 has a (potentially) different type between MSVC and Intel
   // compilers
   typename IdxType<decltype(_BitScanForward64)>::type res;
   _BitScanForward64(&res, ~val);
   return res + 1;
#else
   for (int i = 0; i < sizeof(val) * 8; ++i)
   {
      if (val & (1ull << i))
      {
         return i + 1;
      }
   }
   return 0;
#endif
}

void ReusableStorageBase::EraseInternal(size_t idx)
{
   next = std::min<size_t>(idx, next);
   --idx;
   status.at(idx >> 6) &= ~(1ull << (idx & 0x3f));
}

void ReusableStorageBase::CreateNextInternal()
{
   size_t idx = (next - 1) >> 6;
   while (idx < status.size())
   {
      size_t tmp = LowestUnset(status.at(idx));
      if (tmp)
      {
         next = (idx << 6) + tmp;
         break;
      }
      ++idx;
   }
   if (idx >= status.size())
   {
      status.push_back(0);
      next = (idx << 6) + 1;
   }
   status.at(idx) |= 1ull << ((next - 1) & 0x3f);
}

} // namespace internal
} // namespace mfem
