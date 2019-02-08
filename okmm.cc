#include "/home/camier1/home/okstk/stk.hpp"
#include <dlfcn.h>
#include <cassert>
#include <unordered_map>

// *****************************************************************************
typedef void *malloc_t(size_t);
typedef void *calloc_t(size_t, size_t);
typedef void free_t(void*);
typedef void *realloc_t(void*, size_t);
typedef void *memalign_t(size_t, size_t);
typedef std::unordered_map<void*,size_t> mm_t;

// *****************************************************************************
static bool dbg = false;
static bool hooked = false;
static bool dlsymd = false;

// *****************************************************************************
static mm_t *_mm = NULL;
static free_t *_free = NULL;
static malloc_t *_malloc = NULL;
static calloc_t *_calloc = NULL;
static realloc_t *_realloc = NULL;
static memalign_t *_memalign = NULL;

// *****************************************************************************
static void _init(void){
   if (getenv("OKMM")) dbg = true;
   _free = (free_t*) dlsym(RTLD_NEXT, "free");
   _calloc = (calloc_t*) dlsym(RTLD_NEXT, "calloc");
   _malloc = (malloc_t*) dlsym(RTLD_NEXT, "malloc");
   _realloc = (realloc_t*) dlsym(RTLD_NEXT, "realloc");
   _memalign = (memalign_t*) dlsym(RTLD_NEXT, "memalign");
   _mm = new mm_t();
   assert(_free and _malloc and _calloc and _realloc and _memalign and _mm);
   hooked = dlsymd = true;
}

// *****************************************************************************
void *malloc(size_t size){ // Red
   if (!_malloc) _init();
   if (!hooked) return _malloc(size);
   hooked = false;
   void *ptr = _malloc(size);
   assert(ptr);
   if (dbg){
      if ((*_mm)[ptr]) {
         printf("\n\033[31;1m[malloc] %p(%ld)\033[m", ptr, size);
      }else{
         printf("\n\033[31m[malloc] %p(%ld)\033[m", ptr, size);
         (*_mm)[ptr] = size;
      }
   }
   const bool show_all_stack = false;
   const bool new_or_del = true;
   stk(ptr, new_or_del, show_all_stack);
   hooked = true;
   return ptr;
}

// *****************************************************************************
void free(void *ptr){ // Green
   if (!_free) _init();
   if (!hooked) return _free(ptr);
   hooked = false;
   if (dbg and ptr){
      if ((*_mm)[ptr]){
         printf("\n\033[32;1m[free] %p\033[m", ptr);
      }else{
         printf("\n\033[32m[free]%p\033[m", ptr);
      }
   }
   const bool show_all_stack = false;
   const bool new_or_del = false;
   // tell the stack unwinder to make sure this pointer is not known
   if (ptr) stk(ptr, new_or_del, show_all_stack);
   _free(ptr);
   hooked = true;
}

// *****************************************************************************
void *calloc(size_t nmemb, size_t size){ // Yellow
   if (not dlsymd) { // if we are not yet dlsym'ed, just do it ourselves
      static const size_t MEM_MAX = 8192;
      static char mem[MEM_MAX];
      static size_t m = 0;
      const size_t bytes = nmemb*size;
      void *ptr = &mem[m];
      m += bytes;
      assert(m<MEM_MAX);
      for(size_t k=0;k<bytes;k+=1) *(((char*)ptr)+k) = 0;
      return ptr;
   }
   if (!hooked) return _calloc(nmemb, size);
   hooked = false;
   void *ptr = _calloc(nmemb, size);
   if (dbg and ptr){
      if ((*_mm)[ptr]){
         printf("\n\033[33;1m[calloc] %p(%ld)\033[m", ptr, size);
      }else{
         printf("\n\033[33m[calloc] %p(%ld)\033[m", ptr, size);
      }
   }
   stk();
   hooked = true;
   return ptr;
}

// *****************************************************************************
void *realloc(void *ptr, size_t size){ // Blue
   if (!_realloc) _init();
   if (!hooked) return _realloc(ptr, size);
   hooked = false;
   void *nptr = _realloc(ptr, size);
   assert(nptr);
   if (dbg and ptr){
      if ((*_mm)[ptr]){
         printf("\n\033[34;7m[realloc] %p(%ld)\033[m", nptr, size);
      }else{
         printf("\n\033[34m[realloc] %p(%ld)\033[m", nptr, size);
      }
   }
   stk();
   hooked = true;
   return nptr;
}

// *****************************************************************************
void *memalign(size_t alignment, size_t size){ // Magenta
   if (!_memalign) _init();
   if (!hooked) return _memalign(alignment, size);
   hooked = false;
   void *ptr = _memalign(alignment, size);
   assert(ptr);
   if (dbg and ptr){
      if ((*_mm)[ptr]){
         printf("\n\033[35;7m[memalign] %p(%ld)\033[m", ptr, size);
      }else{
         printf("\n\033[35m[memalign] %p(%ld)\033[m", ptr, size);
      }
   }
   stk();
   hooked = true;
   return ptr;
}
