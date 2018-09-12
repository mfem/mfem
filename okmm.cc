#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
//#include <unordered_map>

// *****************************************************************************
static char tmpbuf[1024];
static size_t tmppos = 0;
static size_t tmpallocs = 0;

static void* dummy_malloc(size_t size) {
   if (tmppos + size >= sizeof(tmpbuf))
      exit(1|fprintf(stderr,"\033[31;1merror\033[m\n"));
   void *retptr = tmpbuf + tmppos;
   tmppos += size;
   ++tmpallocs;
   return retptr;
}

static void* dummy_calloc(size_t nmemb, size_t size) {
   void *ptr = dummy_malloc(nmemb * size);
   bzero(ptr,size);
   return ptr;
}

static void dummy_free(void *ptr) {}

// *****************************************************************************
static bool hooked = false;
static void  (*__free)(void*) = NULL;
static void* (*__malloc)(size_t) = NULL;
static void* (*__calloc)(size_t,size_t) = NULL;
static void* (*__realloc)(void*,size_t) = NULL;
static void* (*__memalign)(size_t,size_t) = NULL;

static void  (*_free)(void*) = NULL;
static void* (*_malloc)(size_t) = NULL;
static void* (*_calloc)(size_t,size_t) = NULL;
static void* (*_realloc)(void*,size_t) = NULL;
static void* (*_memalign)(size_t,size_t) = NULL;
//static std::unordered_map<void*,size_t> *_mm = NULL;

// *****************************************************************************
void __attribute__((constructor)) _init(void){  
   //_malloc = dummy_malloc;
   //_calloc = dummy_calloc;
   //_free = dummy_free;
   fprintf(stdout,"\033[32;1m[here]\033[m\n");

   __malloc = (void* (*)(size_t)) dlsym(RTLD_NEXT, "malloc");
   //__calloc = (void* (*)(size_t, size_t)) dlsym(RTLD_NEXT, "calloc");
   __realloc =(void* (*)(void*, size_t)) dlsym(RTLD_NEXT, "realloc");
   __free =  (void  (*)(void*)) dlsym(RTLD_NEXT, "free");
   __memalign = (void* (*)(size_t, size_t)) dlsym(RTLD_NEXT, "memalign");
   
   const bool fmc = __free and __malloc and __calloc and __realloc and __memalign;
   if (!fmc)
      exit(1|fprintf(stderr,"\033[31;1merror\033[m\n"));
   
   _free = __free;
   _malloc = __malloc;
   //_calloc = __calloc;
   _realloc = __realloc;
   _memalign = __memalign;
   //_mm = new std::unordered_map<void*,size_t>();
   //const bool dls = _free and _malloc and _calloc and _realloc and _memalign;
   //assert(dls);// and _mm);
   hooked = false;
}

// *****************************************************************************
void *malloc(size_t size){
   //if (!hooked) return _malloc(size);
   hooked = false;
   void *ptr = _malloc(size);
   //assert(ptr);
   /*if (!(*_mm)[ptr]) {
      printf("\033[32m%p(%ld)\033[m", ptr, size);
      (*_mm)[ptr] = size;
   }else{
      printf("\033[32;7m%p(%ld)\033[m", ptr, size);
      }*/
   hooked = true;
   return ptr;
}

// *****************************************************************************
void free(void *ptr){
   //if (!hooked) _free(ptr);
   hooked = false;
   /* if (ptr){
      if ((*_mm)[ptr]){
      printf("\033[33;7m%p\033[m", ptr);
      }else{
      printf("\033[33m%p\033[m", ptr);
      }
      }*
      _free(ptr);
   */
   hooked = true;
}

// *****************************************************************************
void *realloc(void *ptr, size_t size){
   //if (!hooked) return _realloc(ptr, size);
   hooked = false;
   void *rptr = _realloc(ptr, size);
   /*if (rptr){
     if ((*_mm)[rptr]){
     printf("\033[33;7m%p(%ld)\033[m", rptr, size);
     }else{
     printf("\033[33m%p(%ld)\033[m", rptr, size);
     }
     }*/
   hooked = true;
   return rptr;
}

// *****************************************************************************
/*void *calloc(size_t nmemb, size_t size){
   if (!hooked) return _calloc(nmemb, size);
   hooked = false;
   void *ptr = _calloc(nmemb, size);
   if (ptr){
   if ((*_mm)[ptr]){
   printf("\033[35;7m%p(%ld)\033[m", ptr, size);
   }else{
   printf("\033[35m%p(%ld)\033[m", ptr, size);
   }
   }
   hooked = true;
   return ptr;
}*/

// *****************************************************************************
void *memalign(size_t alignment, size_t size){
   //if (!hooked) return _memalign(alignment, size);
   hooked = false;
   void *ptr = _memalign(alignment, size);
   /*if (ptr){
     if ((*_mm)[ptr]){
     printf("\033[37;7m%p(%ld)\033[m", ptr, size);
     }else{
     printf("\033[37m%p(%ld)\033[m", ptr, size);
     }
     }*/
   hooked = true;
   return ptr;
}
