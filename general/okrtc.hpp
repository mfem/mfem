#ifndef OKRTC_HPP
#define OKRTC_HPP

#include <dlfcn.h>
#include <fcntl.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/mman.h>
#include <unordered_map>

using namespace std;

namespace ok{

   // ***************************************************************************
   // * Hash functions: combine the args, seed comes from kernel source
   // ***************************************************************************
   template <class T>
   inline size_t hash_combine(const size_t &seed, const T &v) noexcept {
      hash<T> hasher;
      return seed^(hasher(v)+0x9e3779b9ul+(seed<<6)+(seed>>2));
   }
   template<typename T>
   size_t hash_args(const size_t &seed, const T &that) noexcept {
      return hash_combine(seed,that);
   }
   template<typename T, typename... Args>
   size_t hash_args(const size_t &seed, const T &first, Args... args) noexcept {
      return hash_args(hash_combine(seed,first), args...);
   }
   // default OpenKernels::hash ⇒ std::hash **************************************
   template <typename T> struct hash {
      size_t operator()(const T& obj) const noexcept {
         auto hashfn = hash<T>{};
         return hashfn(obj);
      }
   };
   // ***************************************************************************
   static size_t hash_bytes(const char *data, size_t i) noexcept{
      size_t hash = 0xcbf29ce484222325ull;
      for(;i;i--) hash = (hash*0x100000001b3ull)^data[i];
      return hash;
   }
   // const char* specialization ************************************************
   template <> struct hash<const char*> {
      size_t operator()(const char *const s) const noexcept {
         return hash_bytes(s, strlen(s));
      }
   };
   
   // **************************************************************************
   // * uint64 ⇒ char*
   // **************************************************************************
   static void uint32str(uint64_t x, char *s, const size_t offset){      
      x=((x&0xFFFFull)<<32)|((x&0xFFFF0000ull)>>16);
      x=((x&0x0000FF000000FF00ull)>>8)|(x&0x000000FF000000FFull)<<16;
      x=((x&0x00F000F000F000F0ull)>>4)|(x&0x000F000F000F000Full)<<8;
      const uint64_t mask = ((x+0x0606060606060606ull)>>4)&0x0101010101010101ull;
      x|=0x3030303030303030ull;
      x+=0x27ull*mask;
      *(uint64_t*)(s+offset)=x;
   }
   static void uint64str(uint64_t num, char *s, const size_t offset){
      uint32str(num>>32,s,offset); uint32str(num&0xFFFFFFFFull,s+8,offset);
   }

   // ***************************************************************************
   // * compile
   // ***************************************************************************
   template<typename... Args>
   const char *compile(const bool dbg, const size_t hash, const char *xcc,
                       const char *src, const char *incs, Args... args){
      char soName[21] = "k0000000000000000.so";
      char ccName[21] = "k0000000000000000.cc";
      uint64str(hash,soName,1);
      uint64str(hash,ccName,1);
      const int fd = open(ccName,O_CREAT|O_RDWR,S_IRUSR|S_IWUSR);
      assert(fd>=0);
      dprintf(fd,src,hash,args...);
      close(fd);
      const size_t cmd_sz = 4096;
      char xccCommand[cmd_sz];
      const char *CCFLAGS = "-fPIC";
      const char *NVFLAGS = "--compiler-options '-fPIC' -lcuda";
      const bool nvcc = !strncmp("nvcc",xcc,4);
      const char *xflags = nvcc?NVFLAGS:CCFLAGS;
      if (snprintf(xccCommand,cmd_sz,
                   "%s -D__OKRTC__ -shared %s -o %s %s %s",
                   xcc,incs,soName,ccName,xflags)<0) return NULL;
      if (dbg) printf("\033[32;1m[compile] %s\033[m\n",xccCommand);
      if (system(xccCommand)<0) return NULL;
      if (!dbg) unlink(ccName);
      return src;
   }

   // ***************************************************************************
   // * lookup
   // ***************************************************************************
   template<typename... Args>
   void *lookup(const bool dbg, const size_t hash, const char *xcc,
                const char *src, const char* incs, Args... args){
      char soName[21] = "k0000000000000000.so";
      uint64str(hash,soName,1);
      void *handle = dlopen(soName,RTLD_LAZY);
      if (!handle && !compile(dbg,hash,xcc,src,incs,args...)) return NULL;
      if (!(handle=dlopen(soName,RTLD_LAZY))) return NULL;
      return handle;
   }

   // ***************************************************************************
   // * getSymbol
   // ***************************************************************************
   static void *getSymbol(const size_t hash,void *handle){
      char symbol[18] = "k0000000000000000";
      uint64str(hash,symbol,1);
      void *address = dlsym(handle, symbol);
      assert(address);
      return address;
   }

   // ***************************************************************************
   // * okrtc
   // ***************************************************************************
   template<typename kernel_t> class okrtc{
   private:
      bool dbg;
      size_t seed, hash;
      void *handle;
      kernel_t __kernel;
   public:
      template<typename... Args>
      okrtc(const char *xcc, const char *src,
            const char* incs, Args... args):
         dbg(!!getenv("DBG")||!!getenv("dbg")),
         seed(ok::hash<const char*>()(src)),
         hash(hash_args(seed,/*xcc,incs,*/args...)),
         handle(lookup(dbg,hash,xcc,src,incs,args...)),
         __kernel((kernel_t)getSymbol(hash,handle)) {
         if (dbg) printf("\033[32m[okrtc] seed:%016lx, hash:%016lx\033[m\n",seed,hash);
      }
      template<typename... Args>
      void operator_void(Args... args){ __kernel(args...); }
      template<typename return_t,typename... Args>
      return_t operator()(const return_t rtn, Args... args){
         return __kernel(rtn,args...); }
      ~okrtc(){ dlclose(handle); }
   };
  
} // namespace ok
  
#endif // OKRTC_HPP
