//
//  userumat.h
//  
//
//  Created by Carson, Robert Allen on 10/31/18.
//

#ifndef userumat_h
#define userumat_h

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

#define real8 double

extern "C" {
   
#ifdef WIN32
#define UMAT_API __declspec(dllexport)
#else
#define UMAT_API
#define UMATTEST umattest_
#endif
   
   // A fortran function defined in umat.f
   void UMATTEST(real8 *stress, real8 *statev, real8 *ddsdde,
                 real8 *sse, real8 *spd, real8 *scd, real8 *rpl,
                 real8 *ddsdt, real8 *drplde, real8 *drpldt,
                 real8 *stran, real8 *dstran, real8 *time,
                 real8 *deltaTime, real8 *tempk, real8 *dtemp, real8 *predef,
                 real8 *dpred, real8 *cmname, int *ndi, int *nshr, int *ntens,
                 int *nstatv, real8 *props, int *nprops, real8 *coords,
                 real8 *drot, real8 *pnewdt, real8 *celent,
                 real8 *dfgrd0, real8 *dfgrd1, int *noel, int *npt,
                 int *layer, int *kspt, int *kstep, int *kinc);
   // The C entry point function for my umat
   UMAT_API void
   umat(real8 *stress, real8 *statev, real8 *ddsdde,
                    real8 *sse, real8 *spd, real8 *scd, real8 *rpl,
                    real8 *ddsdt, real8 *drplde, real8 *drpldt,
                    real8 *stran, real8 *dstran, real8 *time,
                    real8 *deltaTime, real8 *tempk, real8 *dtemp, real8 *predef,
                    real8 *dpred, real8 *cmname, int *ndi, int *nshr, int *ntens,
                    int *nstatv, real8 *props, int *nprops, real8 *coords,
                    real8 *drot, real8 *pnewdt, real8 *celent,
                    real8 *dfgrd0, real8 *dfgrd1, int *noel, int *npt,
                    int *layer, int *kspt, int *kstep, int *kinc);
   
   UMAT_API void
   abaqus_umat_remap_test(real8 *stress, real8 *statev, real8 *ddsdde,
                          real8 *sse, real8 *spd, real8 *scd, real8 *rpl,
                          real8 *ddsdt, real8 *drplde, real8 *drpldt,
                          real8 *stran, real8 *dstran, real8 *time,
                          real8 *deltaTime, real8 *tempk, real8 *dtemp, real8 *predef,
                          real8 *dpred, real8 *cmname, int *ndi, int *nshr, int *ntens,
                          int *nstatv, real8 *props, int *nprops, real8 *coords,
                          real8 *drot, real8 *pnewdt, real8 *celent,
                          real8 *dfgrd0, real8 *dfgrd1, int *noel, int *npt,
                          int *layer, int *kspt, int *kstep, int *kinc);
}


#endif /* userumat_h */
