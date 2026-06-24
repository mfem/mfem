#include "../forall.hpp"
#include "../mem_manager.hpp"

namespace mfem
{
#ifdef MFEM_ENABLE_MEM_OP_DEBUG
namespace internal
{
std::pair<size_t, size_t> mem_op_tracker::add_allocation(const void *start,
                                                         const void *stop)
{
   size_t res = counter++;
   auto v = allocations.emplace(std::make_pair(start, stop),
                                std::pair<size_t, size_t>(res, 1));
   if (!v.second)
   {
      ++v.first->second.second;
      --counter;
   }
   return v.first->second;
}

typename mem_op_tracker::key_type
mem_op_tracker::find_containing(const void *start, const void *stop)
{
   auto iter = allocations.upper_bound(std::make_pair(start, stop));
   if (iter != allocations.begin())
   {
      --iter;
   }
   while (iter != allocations.end())
   {
      if (iter->first.first > start)
      {
         return allocations.end();
      }
      if (iter->first.first <= start && iter->first.second >= stop)
      {
         return iter;
      }
      ++iter;
   }
   return allocations.end();
}

typename mem_op_tracker::key_type
mem_op_tracker::find_containing(const void *start)
{
   auto iter = allocations.upper_bound(std::make_pair(start, start));
   if (iter != allocations.begin())
   {
      --iter;
   }
   while (iter != allocations.end())
   {
      if (iter->first.first > start)
      {
         return allocations.end();
      }
      if (iter->first.first <= start && iter->first.second >= start)
      {
         return iter;
      }
      ++iter;
   }
   return allocations.end();
}

mem_op_tracker::key_type mem_op_tracker::find_allocation(const void *start,
                                                         const void *stop)
{
   auto iter = allocations.upper_bound(std::make_pair(start, stop));
   if (iter != allocations.begin())
   {
      --iter;
   }
   while (iter != allocations.end())
   {
      if (iter->first.first > start)
      {
         mfem::err << "unable to find " << start << ", " << stop << std::endl;
         MFEM_ABORT("no allocation");
      }
      if (iter->first.first == start && iter->first.second == stop)
      {
         return iter;
      }
      ++iter;
   }
   mfem::err << "unable to find " << start << ", " << stop << std::endl;
   MFEM_ABORT("no allocation");
}

typename mem_op_tracker::key_type
mem_op_tracker::find_allocation(const void *start)
{
   auto iter = allocations.upper_bound(std::make_pair(start, start));
   if (iter != allocations.begin())
   {
      --iter;
   }
   while (iter != allocations.end())
   {
      if (iter->first.first > start)
      {
         MFEM_ABORT("unable to find " << start);
      }
      if (iter->first.first == start)
      {
         auto it2 = iter;
         ++it2;
         if (it2 != allocations.end())
         {
            if (it2->first.first <= start)
            {
               MFEM_ABORT("overlapping allocations");
            }
         }
         return iter;
      }
      ++iter;
   }
   MFEM_ABORT("unable to find " << start);
}

std::pair<size_t, size_t> mem_op_tracker::remove_allocation(const void *start,
                                                            const void *stop)
{
   auto iter = find_allocation(start, stop);
   auto res = iter->second;
   allocations.erase(iter);
   return res;
}

std::pair<size_t, size_t> mem_op_tracker::remove_allocation(const void *start)
{
   auto iter = find_allocation(start);
   auto res = iter->second;
   allocations.erase(iter);
   return res;
}

static OutStream &nullstream()
{
   static std::stringstream buf;
   static OutStream str(buf);
   str.Disable();
   return str;
}

size_t mem_op_debug(size_t idx)
{
   static std::vector<size_t> idcs;
   while (idcs.size() <= idx)
   {
      idcs.push_back(0);
   }
   return idcs.at(idx)++;
}

std::ostream &mem_op_debug(size_t idx, int)
{
   auto pidx = mem_op_debug(idx);
   // cannot use mfem::out because of potential static variiable initialization
   // order errors: https://isocpp.org/wiki/faq/ctors#static-init-order
   return std::cout << "[DEBUG] " << pidx << ": ";
}

std::ostream &mem_op_debug_add(size_t idx, const void *start,
                               const void *stop)
{
   auto &tracker = mem_op_tracker::instance();
   if (start != nullptr)
   {
      auto v = tracker.add_allocation(start, stop);
      auto pidx = mem_op_debug(idx);
      // cannot use mfem::out because of potential static variiable initialization
      // order errors: https://isocpp.org/wiki/faq/ctors#static-init-order
      return std::cout << "[DEBUG] " << pidx << ", add "
             << reinterpret_cast<const char *>(stop) -
             reinterpret_cast<const char *>(start)
             << " Bytes " << v.first << " [" << start << ":" << stop
             << "]: ";
   }
   else
   {
      return nullstream();
   }
}

std::ostream &mem_op_debug_remove(size_t idx, const void *start,
                                  const void *stop)
{
   auto &tracker = mem_op_tracker::instance();
   if (start != nullptr)
   {
      auto v = tracker.remove_allocation(start, stop);
      auto pidx = mem_op_debug(idx);
      // cannot use mfem::out because of potential static variiable initialization
      // order errors: https://isocpp.org/wiki/faq/ctors#static-init-order
      return std::cout << "[DEBUG] " << pidx << ", remove "
             << reinterpret_cast<const char *>(stop) -
             reinterpret_cast<const char *>(start)
             << " Bytes " << v.first << ": ";
   }
   else
   {
      return nullstream();
   }
}

std::ostream &mem_op_debug_remove(size_t idx, const void *start)
{
   auto &tracker = mem_op_tracker::instance();
   if (start != nullptr)
   {
      auto iter = tracker.find_allocation(start);
      auto nbytes = reinterpret_cast<const char *>(iter->first.second) -
                    reinterpret_cast<const char *>(iter->first.first);
      auto v = tracker.remove_allocation(start);
      auto pidx = mem_op_debug(idx);
      // cannot use mfem::out because of potential static variiable initialization
      // order errors: https://isocpp.org/wiki/faq/ctors#static-init-order
      return std::cout << "[DEBUG] " << pidx << ", remove " << nbytes << " Bytes "
             << v.first << ": ";
   }
   else
   {
      return nullstream();
   }
}

std::ostream &mem_op_debug_use(size_t idx, const void *start,
                               const void *stop)
{
   auto &tracker = mem_op_tracker::instance();
   if (start != nullptr)
   {
      auto iter = tracker.find_containing(start, stop);
      auto pidx = mem_op_debug(idx);
      if (iter != tracker.allocations.end())
      {
         // cannot use mfem::out because of potential static variiable
         // initialization order errors:
         // https://isocpp.org/wiki/faq/ctors#static-init-order
         return std::cout << "[DEBUG] " << pidx << ", " << iter->second.first
                << "["
                << reinterpret_cast<const char *>(start) -
                reinterpret_cast<const char *>(iter->first.first)
                << ":"
                << reinterpret_cast<const char *>(stop) -
                reinterpret_cast<const char *>(iter->first.first)
                << "]: ";
      }
      else
      {
         // cannot use mfem::out because of potential static variiable
         // initialization order errors:
         // https://isocpp.org/wiki/faq/ctors#static-init-order
         return std::cout << "[DEBUG] " << pidx << ", external[" << start << ":"
                << stop << "]: ";
      }
   }
   else
   {
      auto pidx = mem_op_debug(idx);
      // cannot use mfem::out because of potential static variiable initialization
      // order errors: https://isocpp.org/wiki/faq/ctors#static-init-order
      return std::cout << "[DEBUG] " << pidx << ", use nullptr "
             << ": ";
   }
}

std::ostream &mem_op_debug_sync_alias(size_t idx, const void *astart,
                                      const void *bstart, size_t nbytes)
{
   auto &tracker = mem_op_tracker::instance();
   if (astart != nullptr)
   {
      MFEM_VERIFY(bstart != nullptr, "bstart null");
      MFEM_VERIFY(reinterpret_cast<const char *>(astart) >=
                  reinterpret_cast<const char *>(bstart),
                  "alias start before base start");
      auto iter = tracker.find_containing(bstart);
      auto pidx = mem_op_debug(idx);
      if (iter != tracker.allocations.end())
      {
         MFEM_VERIFY(iter->first.first == bstart,
                     "registered ptr doesn't match base");
         auto seg_len = reinterpret_cast<const char *>(iter->first.second) -
                        reinterpret_cast<const char *>(iter->first.first);
         MFEM_VERIFY(size_t(seg_len) >= nbytes, "alias size too large");
         MFEM_VERIFY(reinterpret_cast<const char *>(astart) -
                     reinterpret_cast<const char *>(bstart) >=
                     0,
                     "astart not inside of base segment");
         MFEM_VERIFY(reinterpret_cast<const char *>(astart) + nbytes -
                     reinterpret_cast<const char *>(bstart) <=
                     seg_len,
                     "astart not inside of base segment");
         // cannot use mfem::out because of potential static variiable
         // initialization order errors:
         // https://isocpp.org/wiki/faq/ctors#static-init-order
         return std::cout << "[DEBUG] " << pidx << ", SYNC_ALIAS "
                << iter->second.first << "["
                << reinterpret_cast<const char *>(astart) -
                reinterpret_cast<const char *>(bstart)
                << ":"
                << reinterpret_cast<const char *>(astart) -
                reinterpret_cast<const char *>(bstart) + nbytes
                << "]: " << nbytes << " Bytes" << std::endl;
      }
      else
      {
         // cannot use mfem::out because of potential static variiable
         // initialization order errors:
         // https://isocpp.org/wiki/faq/ctors#static-init-order
         return std::cout << "[DEBUG] " << pidx << ", sync alias external "
                << bstart << "["
                << reinterpret_cast<const char *>(astart) -
                reinterpret_cast<const char *>(bstart)
                << ":"
                << reinterpret_cast<const char *>(astart) -
                reinterpret_cast<const char *>(bstart) + nbytes
                << "]: " << nbytes << " Bytes" << std::endl;
      }
   }
   else
   {
      auto pidx = mem_op_debug(idx);
      // cannot use mfem::out because of potential static variiable initialization
      // order errors: https://isocpp.org/wiki/faq/ctors#static-init-order
      return std::cout << "[DEBUG] " << pidx << ", sync alias nullptr"
             << std::endl;
   }
}

std::ostream &mem_op_debug_batch_mem_copy(size_t idx, const void *src_start,
                                          const void *dst_start, size_t nbytes,
                                          MemoryType src_loc,
                                          MemoryType dst_loc)
{
   auto &tracker = mem_op_tracker::instance();
   auto src_iter = tracker.find_containing(
                      src_start, reinterpret_cast<const char *>(src_start) + nbytes);
   auto dst_iter = tracker.find_containing(
                      dst_start, reinterpret_cast<const char *>(dst_start) + nbytes);
   auto pidx = mem_op_debug(idx);
   // cannot use mfem::out because of potential static variiable initialization
   // order errors: https://isocpp.org/wiki/faq/ctors#static-init-order
   std::cout << "[DEBUG] " << pidx << ": BATCH_MEM_COPY "
             << mem_op_debug_copy_type(src_loc, dst_loc) << " " << nbytes
             << " Bytes, ";
   if (src_iter != tracker.allocations.end())
   {
      // cannot use mfem::out because of potential static variiable initialization
      // order errors: https://isocpp.org/wiki/faq/ctors#static-init-order
      std::cout << src_iter->second.first << "["
                << reinterpret_cast<const char *>(src_start) -
                reinterpret_cast<const char *>(src_iter->first.first)
                << "] -> ";
   }
   else
   {
      // cannot use mfem::out because of potential static variiable initialization
      // order errors: https://isocpp.org/wiki/faq/ctors#static-init-order
      std::cout << "external src " << src_start << " -> ";
   }
   if (dst_iter != tracker.allocations.end())
   {
      // cannot use mfem::out because of potential static variiable initialization
      // order errors: https://isocpp.org/wiki/faq/ctors#static-init-order
      return std::cout << dst_iter->second.first << "["
             << reinterpret_cast<const char *>(dst_start) -
             reinterpret_cast<const char *>(
                dst_iter->first.first)
             << "]";
   }
   else
   {
      // cannot use mfem::out because of potential static variiable initialization
      // order errors: https://isocpp.org/wiki/faq/ctors#static-init-order
      return std::cout << "external dst " << dst_start;
   }
}

std::string mem_op_debug_copy_type(MemoryType src_loc, MemoryType dst_loc)
{
   std::stringstream res;
   switch (src_loc)
   {
      case MemoryType::HOST:
      case MemoryType::HOST_32:
      case MemoryType::HOST_64:
      case MemoryType::HOST_DEBUG:
      case MemoryType::HOST_UMPIRE:
      case MemoryType::HOST_PINNED:
         res << 'H';
         break;
      case MemoryType::MANAGED:
      case MemoryType::DEVICE:
      case MemoryType::DEVICE_DEBUG:
      case MemoryType::DEVICE_UMPIRE:
      case MemoryType::DEVICE_UMPIRE_2:
         res << 'D';
         break;
      default:
         break;
   }
   res << "to";
   switch (dst_loc)
   {
      case MemoryType::HOST:
      case MemoryType::HOST_32:
      case MemoryType::HOST_64:
      case MemoryType::HOST_DEBUG:
      case MemoryType::HOST_UMPIRE:
      case MemoryType::HOST_PINNED:
         res << 'H';
         break;
      case MemoryType::MANAGED:
      case MemoryType::DEVICE:
      case MemoryType::DEVICE_DEBUG:
      case MemoryType::DEVICE_UMPIRE:
      case MemoryType::DEVICE_UMPIRE_2:
         res << 'D';
         break;
      default:
         break;
   }
   return res.str();
}
} // namespace internal
#endif

#ifdef MFEM_ENABLE_MEM_BENCH
namespace internal
{
BenchTimer &BenchTimer::Instance()
{
   static BenchTimer v;
   return v;
}

BenchTimer::BenchTimer()
{
   glob_start = timer.now();
   start_points.resize(11);
   durations.resize(start_points.size());
   call_counts.resize(start_points.size());
}

BenchTimer::~BenchTimer()
{
   auto tot_time = timer.now() - glob_start;
   std::vector<double> sums;
   for (auto &v : durations)
   {
      sums.emplace_back(
         std::chrono::duration_cast<std::chrono::duration<double>>(v).count());
   }
   mfem::out << "Total time: "
             << std::chrono::duration_cast<std::chrono::duration<double>>(
                tot_time)
             .count()
             << std::endl;
   mfem::out << "ReadWrite [" << call_counts[0] << "]: " << sums[0] << " s"
             << std::endl;
   mfem::out << "Read [" << call_counts[1] << "]: " << sums[1] << " s"
             << std::endl;
   mfem::out << "Write [" << call_counts[2] << "]: " << sums[2] << " s"
             << std::endl;
   mfem::out << "SyncAlias [" << call_counts[3] << "]: " << sums[3] << " s"
             << std::endl;
   mfem::out << "Copy [" << call_counts[4] << "]: " << sums[4] << " s"
             << std::endl;
   mfem::out << "CopyFromHost [" << call_counts[5] << "]: " << sums[5] << " s"
             << std::endl;
   mfem::out << "CopyToHost [" << call_counts[6] << "]: " << sums[6] << " s"
             << std::endl;
   mfem::out << "MarkInvalid [" << call_counts[7] << "]: " << sums[7] << " s"
             << std::endl;
   mfem::out << "MarkValid [" << call_counts[8] << "]: " << sums[8] << " s"
             << std::endl;
   mfem::out << "CheckValid [" << call_counts[9] << "]: " << sums[9] << " s"
             << std::endl;
   mfem::out << "MemCopy [" << call_counts[10] << "]: " << sums[10] << " s"
             << std::endl;
}

ScopeBench::ScopeBench(size_t i, bool do_sync) : idx(i), sync(do_sync)
{
   ++BenchTimer::Instance().call_counts[i];
   if (do_sync)
   {
      MFEM_DEVICE_SYNC;
   }
   BenchTimer::Instance().start_points[i] = BenchTimer::Instance().timer.now();
}

ScopeBench::~ScopeBench()
{
   if (sync)
   {
      MFEM_DEVICE_SYNC;
   }
   BenchTimer::Instance().durations[idx] +=
      BenchTimer::Instance().timer.now() -
      BenchTimer::Instance().start_points[idx];
}
} // namespace internal
#endif
} // namespace mfem
