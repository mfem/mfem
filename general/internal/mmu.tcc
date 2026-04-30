#ifndef MFEM_MMU_TCC
#define MFEM_MMU_TCC

// this file should only be included in mem_manager.cpp or resource_manager.cpp!

#ifndef _WIN32
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace mfem
{

#ifndef _WIN32
static uintptr_t pagesize = 0;
static uintptr_t pagemask = 0;

static struct sigaction old_segv_action;
static struct sigaction old_bus_action;
  
/// Returns the restricted base address of the DEBUG segment
static const void *MmuAddrR(const void *ptr)
{
   const uintptr_t addr = (uintptr_t)ptr;
   return (addr & pagemask) ? (void *)((addr + pagesize) & ~pagemask) : ptr;
}

/// Returns the prolongated base address of the MMU segment
static const void *MmuAddrP(const void *ptr)
{
   const uintptr_t addr = (uintptr_t)ptr;
   return (void *)(addr & ~pagemask);
}

/// Compute the restricted length for the MMU segment
static uintptr_t MmuLengthR(const void *ptr, const size_t bytes)
{
   // a ---->A:|    |:B<---- b
   const uintptr_t a = (uintptr_t)ptr;
   const uintptr_t A = (uintptr_t)MmuAddrR(ptr);
   MFEM_ASSERT(a <= A, "");
   const uintptr_t b = a + bytes;
   const uintptr_t B = b & ~pagemask;
   MFEM_ASSERT(B <= b, "");
   const uintptr_t length = B > A ? B - A : 0;
   MFEM_ASSERT(length % pagesize == 0, "");
   return length;
}

/// Compute the prolongated length for the MMU segment
static uintptr_t MmuLengthP(const void *ptr, const size_t bytes)
{
   // |:A<----a |    |  b---->B:|
   const uintptr_t a = (uintptr_t)ptr;
   const uintptr_t A = (uintptr_t)MmuAddrP(ptr);
   MFEM_ASSERT(A <= a, "");
   const uintptr_t b = a + bytes;
   const uintptr_t B = b & pagemask ? (b + pagesize) & ~pagemask : b;
   MFEM_ASSERT(b <= B, "");
   MFEM_ASSERT(B >= A, "");
   const uintptr_t length = B - A;
   MFEM_ASSERT(length % pagesize == 0, "");
   return length;
}

/// The protected access error, used for the host
static void MmuError(int sig, siginfo_t *si, void *context)
{
   constexpr size_t buf_size = 64;
   fflush(0);
   char str[buf_size];
   const void *ptr = si->si_addr;
   snprintf(str, buf_size, "Error while accessing address %p!", ptr);
   mfem::out << std::endl << "An illegal memory access was made!";
   mfem::out << std::endl
             << "Caught signal " << sig << ", code " << si->si_code << " at "
             << ptr << std::endl;
   // chain to previous handler
   struct sigaction *old_action =
      (sig == SIGSEGV) ? &old_segv_action : &old_bus_action;
   if (old_action->sa_flags & SA_SIGINFO && old_action->sa_sigaction)
   {
      // old action uses three argument handler.
      old_action->sa_sigaction(sig, si, context);
   }
   else if (old_action->sa_handler == SIG_DFL)
   {
      // reinstall and raise the default handler.
      sigaction(sig, old_action, NULL);
      raise(sig);
   }
   MFEM_ABORT(str);
}

/// MMU initialization, setting SIGBUS & SIGSEGV signals to MmuError
static void MmuInit()
{
   if (pagesize > 0)
   {
      return;
   }
   struct sigaction sa;
   sa.sa_flags = SA_SIGINFO;
   sigemptyset(&sa.sa_mask);
   sa.sa_sigaction = MmuError;
   if (sigaction(SIGBUS, &sa, &old_bus_action) == -1)
   {
      mfem_error("SIGBUS");
   }
   if (sigaction(SIGSEGV, &sa, &old_segv_action) == -1)
   {
      mfem_error("SIGSEGV");
   }
   pagesize = (uintptr_t)sysconf(_SC_PAGE_SIZE);
   MFEM_ASSERT(pagesize > 0, "pagesize must not be less than 1");
   pagemask = pagesize - 1;
}

/// MMU allocation, through ::mmap
static void MmuAlloc(void **ptr, const size_t bytes)
{
   const size_t length = bytes == 0 ? 8 : bytes;
   const int prot = PROT_READ | PROT_WRITE;
   const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
   *ptr = ::mmap(NULL, length, prot, flags, -1, 0);
   if (*ptr == MAP_FAILED)
   {
      throw ::std::bad_alloc();
   }
}

/// MMU deallocation, through ::munmap
static void MmuDealloc(void *ptr, const size_t bytes)
{
   const size_t length = bytes == 0 ? 8 : bytes;
   if (::munmap(ptr, length) == -1)
   {
      mfem_error("Dealloc error!");
   }
}

/// MMU protection, through ::mprotect with no read/write accesses
static void MmuProtect(const void *ptr, const size_t bytes)
{
   static const bool mmu_protect_error = GetEnv("MFEM_MMU_PROTECT_ERROR");
   if (!::mprotect(const_cast<void *>(ptr), bytes, PROT_NONE))
   {
      return;
   }
   if (mmu_protect_error)
   {
      mfem_error("MMU protection (NONE) error");
   }
}

/// MMU un-protection, through ::mprotect with read/write accesses
static void MmuAllow(const void *ptr, const size_t bytes)
{
   const int RW = PROT_READ | PROT_WRITE;
   static const bool mmu_protect_error = GetEnv("MFEM_MMU_PROTECT_ERROR");
   if (!::mprotect(const_cast<void *>(ptr), bytes, RW))
   {
      return;
   }
   if (mmu_protect_error)
   {
      mfem_error("MMU protection (R/W) error");
   }
}
#else
static void MmuInit() {}
static void MmuAlloc(void **ptr, const size_t bytes)
{
   *ptr = std::malloc(bytes);
}
static void MmuDealloc(void *ptr, const size_t) { std::free(ptr); }
static void MmuProtect(const void *, const size_t) {}
static void MmuAllow(const void *, const size_t) {}
static const void *MmuAddrR(const void *a) { return a; }
static const void *MmuAddrP(const void *a) { return a; }
static uintptr_t MmuLengthR(const void *, const size_t) { return 0; }
static uintptr_t MmuLengthP(const void *, const size_t) { return 0; }
#endif
  
} // namespace mfem
#endif
