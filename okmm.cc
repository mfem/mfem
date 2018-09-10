//#define _GNU_SOURCE
#include <cassert>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>

// *****************************************************************************
static bool hooked = false;
static void  (*_free)(void*) = NULL;
static void* (*_malloc)(size_t) = NULL;
static void* (*_calloc)(size_t,size_t) = NULL;
static void* (*_realloc)(void*,size_t) = NULL;
static void* (*_memalign)(size_t,size_t) = NULL;
static std::unordered_map<void*,size_t> *_mm = NULL;

// *****************************************************************************
static void _init(void){
   _free = (void (*)(void*))dlsym(RTLD_NEXT, "free");
   _malloc = (void* (*)(size_t))dlsym(RTLD_NEXT, "malloc");
   _calloc = (void* (*)(size_t, size_t))dlsym(RTLD_NEXT, "calloc");
   _realloc = (void* (*)(void*, size_t))dlsym(RTLD_NEXT, "realloc");
   _memalign = (void* (*)(size_t, size_t))dlsym(RTLD_NEXT, "memalign");
   const bool dls = _malloc and _free and _calloc and _realloc and _memalign;
   _mm = new std::unordered_map<void*,size_t>();
   assert(_mm);
   assert(dls);
   hooked = true;
}

// *****************************************************************************
void *malloc(size_t size){
   if (!_malloc) _init();
   if (!hooked) return _malloc(size);
   hooked = false;
   void *ptr = _malloc(size);
   assert(ptr);
   if (!(*_mm)[ptr]) {
      printf("\033[32;7m%p(%ld)\033[m", ptr, size);
      (*_mm)[ptr] = size;
   }else{
      printf("\033[32m%p(%ld)\033[m", ptr, size);
   }
   hooked = true;
   return ptr;
}

// *****************************************************************************
void free(void *ptr){
   if (!_free) _init();
   if (!hooked) return _free(ptr);
   hooked = false;
   if (ptr){
      if ((*_mm)[ptr]){
         printf("\033[33;7m%p\033[m", ptr);
      }else{
         printf("\033[33m%p\033[m", ptr);
      }
   }
   _free(ptr);
   hooked = true;
}

// *****************************************************************************
void *realloc(void *ptr, size_t size){
   if (!_realloc) _init();
   void *nptr = _realloc(ptr, size);
   if (!hooked) return nptr;
   hooked = false;
   if (ptr){
      if ((*_mm)[ptr]){
         printf("\033[33;7m%p(%ld)\033[m", nptr, size);
      }else{
         printf("\033[33m%p(%ld)\033[m", nptr, size);
      }
   }
   hooked = true;
   return nptr;
}

// *****************************************************************************
void *calloc(size_t nmemb, size_t size){
   if (!_calloc) _init();
   void *ptr = _calloc(nmemb, size);
   if (!hooked) return ptr;
   hooked = false;
   if (ptr){
      if ((*_mm)[ptr]){
         printf("\033[35;7m%p(%ld)\033[m", ptr, size);
      }else{
         printf("\033[35m%p(%ld)\033[m", ptr, size);
      }
   }
   hooked = true;
   return ptr;
}

// *****************************************************************************
void *memalign(size_t alignment, size_t size){
   if (!_memalign) _init();
   void *ptr = _memalign(alignment, size);
   if (!hooked) return ptr;
   hooked = false;
   if (ptr){
      if ((*_mm)[ptr]){
         printf("\033[37;7m%p(%ld)\033[m", ptr, size);
      }else{
         printf("\033[37m%p(%ld)\033[m", ptr, size);
      }
   }
   hooked = true;
   return ptr;
}
