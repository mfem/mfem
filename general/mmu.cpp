// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../general/okina.hpp"

#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>

namespace mfem
{

// *****************************************************************************
static void MmuSigSeg(int sig, siginfo_t *si, void *unused)
{
   dbg("\033[31;1m%p\n", si->si_addr);
   mfem_error("MmuSigSeg");
}

// *****************************************************************************
static bool mmu = false;
static long pagesize = -1;

// *****************************************************************************
static inline bool MmuFilter(void)
{
   if (!config::UsingMM()) { return true; }
   if (config::DeviceDisabled()) { return true; }
   if (!config::DeviceHasBeenEnabled()) { return true; }
   if (config::UsingOcca()) { mfem_error("config::UsingOcca()"); }
   return false;
}
// *****************************************************************************
void MmuDisableAccess(void *ptr, const size_t bytes){
   if (MmuFilter()) { return; }
   dbg("\033[31;1m%p %ld", ptr, bytes);
   mprotect(ptr, bytes, PROT_NONE);
   //mprotect(ptr, bytes, PROT_READ | PROT_WRITE);
}

// *****************************************************************************
void MmuEnableAccess(void *ptr, const size_t bytes){
   if (MmuFilter()) { return; }
   dbg("\033[32;1m%p %ld", ptr, bytes);
   mprotect(ptr, bytes, PROT_READ | PROT_WRITE);
}

// *****************************************************************************
void MmuInit(){
   mmu = true;
   pagesize = sysconf(_SC_PAGE_SIZE);
   dbg("pagesize=%ldKB",pagesize/1024);
   if (pagesize == -1) mfem_error("sysconf");
   // Prepare handler
   struct sigaction sa;
   sa.sa_flags = SA_SIGINFO;
   sigemptyset(&sa.sa_mask);
   sa.sa_sigaction = MmuSigSeg;
   if (sigaction(SIGSEGV, &sa, NULL) == -1) mfem_error("sigaction");
   if (sigaction(SIGBUS, &sa, NULL) == -1) mfem_error("sigaction");
}

// *****************************************************************************
#undef MFEM_MMU_USE_MEMALIGN

// *****************************************************************************
void *MmuAllocate(const size_t bytes){
   assert(bytes>0);
   if (!mmu) MmuInit();
   MFEM_ASSERT(bytes>0,"!(bytes>0)");
#ifndef MFEM_MMU_USE_MEMALIGN
#warning bytes instead of pages
   //const long pages = 1 + bytes / pagesize;
   const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
   void *ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
   //dbg("bytes: %ld => pages=%ld %p", bytes, pages, ptr);
   if (ptr == MAP_FAILED) mfem_error("mmap");
#else
   void *ptr;
   if (posix_memalign(&ptr, pagesize, bytes) != 0) mfem_error("posix_memalign");
   //dbg("bytes: %ld => %p", bytes, ptr);
#endif
   return ptr;
}

// *****************************************************************************
void MmuDelete(void *ptr, const size_t bytes){
   assert(ptr);
   //dbg("%p", ptr);
   assert(bytes>0);
#ifndef MFEM_MMU_USE_MEMALIGN
   if (munmap(ptr, bytes) == -1) mfem_error("munmap");
#else
   free(ptr);
#endif
}

} // namespace mfem
