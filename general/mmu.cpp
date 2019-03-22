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
   mfem_error("");
}

// *****************************************************************************
static bool mmu = false;
static long pagesize = -1;

// *****************************************************************************
void MmuDisableAccess(void *ptr, const size_t bytes){
   mprotect(ptr, bytes, 0);
}

// *****************************************************************************
void MmuEnableAccess(void *ptr, const size_t bytes){
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
}

// *****************************************************************************
void *MmuAllocate(const size_t bytes){
   assert(bytes>0);
   if (!mmu) MmuInit();
   MFEM_ASSERT(bytes>0,"!(bytes>0)");
   const long pages = 1 + bytes / pagesize;
   const int flags = MAP_ANONYMOUS | MAP_PRIVATE;
   void *ptr = mmap(NULL, pages, PROT_READ | PROT_WRITE, flags, -1, 0);
   dbg("bytes: %ld => pages=%ld %p", bytes, pages, ptr);
   if (ptr == MAP_FAILED) mfem_error("mmap");
   return ptr;
}

// *****************************************************************************
void MmuDelete(void *ptr, const size_t bytes){
   assert(ptr);
   dbg("%p", ptr);
   assert(bytes>0);
   if (munmap(ptr, bytes) == -1) mfem_error("munmap");
}

} // namespace mfem
