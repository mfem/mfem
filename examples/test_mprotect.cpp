#include <sys/mman.h>
#include <iostream>
#include <unistd.h>

int main()
{
   // Allocate memory
   void *addr = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (addr == MAP_FAILED)
   {
      std::cerr << "mmap failed\n";
      return 1;
   }

   std::cout << "mprotect called, PID: " << getpid() << std::endl;
   sleep(10);

   // Call mprotect
   if (mprotect(addr, 4096, PROT_READ) == -1)
   {
      std::cerr << "mprotect failed\n";
      return 1;
   }

   // Keep the process running to allow DTrace to attach
   sleep(30);

   // Clean up
   munmap(addr, 4096);
   return 0;
}
