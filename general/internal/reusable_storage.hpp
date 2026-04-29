#ifndef MFEM_REUSABLE_STORAGE_HPP
#define MFEM_REUSABLE_STORAGE_HPP

#include <cstdint>
#include <vector>

#include "../error.hpp"

namespace mfem
{
namespace internal
{

/// Flat storage of entries of type T such that their index is stable (but
/// address is not). Can quickly delete and create values at any index.
struct ReusableStorageBase
{
protected:
   /// each bit corresponds to if an entry in data is valid or not
   std::vector<uint64_t> status;
   /// where to start searching for the next entry in data is free (offset by 1)
   size_t next = 1;

   void EraseInternal(size_t idx);

   /// next entry has stable index next after calling this function
   void CreateNextInternal();
};

/// Flat storage of entries of type T such that their index is stable (but
/// address is not). Can quickly "delete" (mark unused) and create values at any
/// index.
/// T must be default constructable and either copy or move constructable.
template <class T> struct ReusableStorage : public ReusableStorageBase
{
private:
   std::vector<T> data;

public:
   T &Get(size_t idx) { return data.at(idx - 1); }
   const T &Get(size_t idx) const { return data.at(idx - 1); }
   void Erase(size_t idx)
   {
      MFEM_ASSERT(idx, "invalid idx");
      MFEM_ASSERT(idx <= data.size(), "idx oob");
      EraseInternal(idx);
      if (idx == data.size())
      {
         while (data.size())
         {
            --idx;
            if (status.at(idx >> 6) & (1ull << (idx & 0x3f)))
            {
               break;
            }
            data.pop_back();
         }
      }
   }
   /// @return stable index
   size_t CreateNext()
   {
      CreateNextInternal();
      size_t res = next;
      if (res > data.size())
      {
         MFEM_ASSERT(res == data.size() + 1,"");
         data.emplace_back();
      }
      else
      {
         data[res - 1] = T{};
      }
      return res;
   }
};
} // namespace internal
} // namespace mfem

#endif
