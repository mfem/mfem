//#define _GNU_SOURCE
#include <cassert>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// *****************************************************************************
static void * (*__calloc)(size_t, size_t) = NULL;
static void * (*__malloc)(size_t) = NULL;
static void   (*__free)(void*) = NULL;
static void * (*__realloc)(void*, size_t) = NULL;
static void * (*__memalign)(size_t, size_t) = NULL;

// *****************************************************************************
static void __init(void){
   __malloc     = (void* (*)(size_t))dlsym(RTLD_NEXT, "malloc");
   __free       = (void (*)(void*))dlsym(RTLD_NEXT, "free");
   __calloc     = (void* (*)(size_t, size_t))dlsym(RTLD_NEXT, "calloc");
   __realloc    = (void* (*)(void*, size_t))dlsym(RTLD_NEXT, "realloc");
   __memalign   = (void* (*)(size_t, size_t))dlsym(RTLD_NEXT, "memalign");
   const bool dlsyms = __malloc and __free and __calloc and __realloc and __memalign;
   if (!dlsyms) exit(fprintf(stderr, "Error in `dlsym`: %s\n", dlerror()));
}

// *****************************************************************************
void *malloc(size_t size){
   if (!__malloc) __init();
   void *ptr = NULL;
   ptr = __malloc(size);
   assert(ptr);
   printf("\033[32m%p(%ld)\033[m", ptr, size);
   return ptr;
}

// *****************************************************************************
void free(void *ptr){
   if (ptr) printf("\033[33m%p\033[m", ptr);
   __free(ptr);
}

// *****************************************************************************
void *realloc(void *ptr, size_t size){
   if (!__realloc) __init();
   void *nptr = __realloc(ptr, size);
   printf("\033[33m%p(%ld)\033[m", ptr, size);
   return nptr;
}

// *****************************************************************************
void *calloc(size_t nmemb, size_t size){
   if (!__calloc) __init();
   void *ptr = __calloc(nmemb, size);
   printf("\033[35m%p(%ld)\033[m", ptr, size);
   return ptr;
}

// *****************************************************************************
void *memalign(size_t alignment, size_t size){
   if (!__memalign) __init();
   void *ptr = __memalign(alignment, size);
   printf("\033[37m%p(%ld)\033[m", ptr, size);
   return ptr;
}
