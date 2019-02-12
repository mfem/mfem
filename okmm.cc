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
static free_t *_free = NULL;
static malloc_t *_malloc = NULL;
static calloc_t *_calloc = NULL;
static realloc_t *_realloc = NULL;
static memalign_t *_memalign = NULL;

// *****************************************************************************
static void _init(void){
   if (getenv("DBG")) dbg = true;
   _free = (free_t*) dlsym(RTLD_NEXT, "free");
   _calloc = (calloc_t*) dlsym(RTLD_NEXT, "calloc");
   _malloc = (malloc_t*) dlsym(RTLD_NEXT, "malloc");
   _realloc = (realloc_t*) dlsym(RTLD_NEXT, "realloc");
   _memalign = (memalign_t*) dlsym(RTLD_NEXT, "memalign");
   assert(_free and _malloc and _calloc and _realloc and _memalign);
   hooked = dlsymd = true;
}


// *****************************************************************************
void *malloc(size_t size){ // Red
   if (!_malloc) _init();
   if (!hooked) return _malloc(size);
   hooked = false;
   void *ptr = _malloc(size);
   assert(ptr);
   if (dbg) printf("\n\033[31m[malloc] %p (%ld)\033[m", ptr, size);
   backtrace(ptr, true, false); // new, dont show full stack
   hooked = true;
   return ptr;
}

// *****************************************************************************
void free(void *ptr){ // Green
   if (!_free) _init();
   if (!hooked) return _free(ptr);
   if (!ptr) return;
   hooked = false;
   if (dbg) printf("\n\033[32m[free] %p\033[m", ptr);
   backtrace(ptr, false, false); // delete, dont show full stack
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
   if (dbg) printf("\n\033[33m[calloc] %p (%ld)\033[m", ptr, size);
   backtrace(ptr, true, false); // new, show full stack
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
   if (dbg) printf("\n\033[34;7m[realloc] %p(%ld)\033[m", nptr, size);
   backtrace(nptr, true, true); // new, show full stack
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
   if (dbg) printf("\n\033[35;7m[memalign] %p(%ld)\033[m", ptr, size);
   backtrace(ptr, true, true); // new, show full stack
   hooked = true;
   return ptr;
}
