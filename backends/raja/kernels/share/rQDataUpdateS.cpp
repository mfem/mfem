// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "../raja.hpp"

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DIM,
         const int NUM_QUAD,
         const int NUM_QUAD_1D,
         const int NUM_DOFS_1D> kernel
#endif
void rUpdateQuadratureData2S(
#ifndef __TEMPLATES__
                             const int NUM_DIM,
                             const int NUM_QUAD,
                             const int NUM_QUAD_1D,
                             const int NUM_DOFS_1D,
#endif
                             const double GAMMA,
                             const double H0,
                             const double CFL,
                             const bool USE_VISCOSITY,
                             const int numElements,
                             const double* restrict dofToQuad,
                             const double* restrict dofToQuadD,
                             const double* restrict quadWeights,
                             const double* restrict v,
                             const double* restrict e,
                             const double* restrict rho0DetJ0w,
                             const double* restrict invJ0,
                             const double* restrict J,
                             const double* restrict invJ,
                             const double* restrict detJ,
                             double* restrict stressJinvT,
                             double* restrict dtEst) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_DOFS_1D = (NUM_QUAD_1D * NUM_DOFS_1D);
  const int NUM_MAX_1D = (NUM_QUAD_1D<NUM_DOFS_1D)?NUM_DOFS_1D:NUM_QUAD_1D;
#ifdef __LAMBDA__
  forall(el,numElements,
#else
  const int idx = blockIdx.x;
  const int el = idx;
  if (el < numElements)
#endif
  {
    share double s_dofToQuad[NUM_QUAD_DOFS_1D];//@dim(NUM_QUAD_1D, NUM_DOFS_1D);
    share double s_dofToQuadD[NUM_QUAD_DOFS_1D];//@dim(NUM_QUAD_1D, NUM_DOFS_1D);

    share double s_xy[NUM_DIM * NUM_QUAD_DOFS_1D];//@dim(NUM_DIM, NUM_DOFS_1D, NUM_QUAD_1D);
    share double s_xDy[NUM_DIM * NUM_QUAD_DOFS_1D];//@dim(NUM_DIM, NUM_DOFS_1D, NUM_QUAD_1D);

    share double s_gradv[NUM_DIM * NUM_DIM * NUM_QUAD_2D];//@dim(NUM_DIM, NUM_DIM, NUM_QUAD_2D);

    double r_v[NUM_DIM * NUM_DOFS_1D];//@dim(NUM_DIM, NUM_DOFS_1D);

#ifdef __LAMBDA__
    for (int x = 0; x < NUM_MAX_1D; ++x) {
#else
    {  const int x = threadIdx.x;
#endif
      for (int id = x; id < NUM_QUAD_DOFS_1D; id += NUM_MAX_1D) {
        s_dofToQuad[id]  = dofToQuad[id];
        s_dofToQuadD[id] = dofToQuadD[id];
      }
    }

    sync;
#ifdef __LAMBDA__    
    for (int dx = 0; dx < NUM_MAX_1D; ++dx) {
#else
    {  const int dx = threadIdx.x;
#endif
      if (dx < NUM_DOFS_1D) {
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int vi = 0; vi < NUM_DIM; ++vi) {
            s_xy[ijkNM(vi, dx, qy,NUM_DIM,NUM_DOFS_1D)] = 0;
            s_xDy[ijkNM(vi, dx, qy,NUM_DIM,NUM_DOFS_1D)] = 0;
          }
        }
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          for (int vi = 0; vi < NUM_DIM; ++vi) {
            r_v[ijN(vi, dy,NUM_DIM)] = v[_ijklNM(vi,dx,dy,el,NUM_DOFS_1D,numElements)];
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          double xy[NUM_DIM];
          double xDy[NUM_DIM];
          for (int vi = 0; vi < NUM_DIM; ++vi) {
            xy[vi]  = 0;
            xDy[vi] = 0;
          }
          for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
            for (int vi = 0; vi < NUM_DIM; ++vi) {
              xy[vi]  += r_v[ijN(vi, dy,NUM_DIM)] * s_dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
              xDy[vi] += r_v[ijN(vi, dy,NUM_DIM)] * s_dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
            }
          }
          for (int vi = 0; vi < NUM_DIM; ++vi) {
            s_xy[ijkNM(vi, dx, qy,NUM_DIM,NUM_DOFS_1D)]  = xy[vi];
            s_xDy[ijkNM(vi, dx, qy,NUM_DIM,NUM_DOFS_1D)] = xDy[vi];
          }
        }
      }
    }

    sync;
#ifdef __LAMBDA__
    for (int qy = 0; qy < NUM_MAX_1D; ++qy) {
#else
    {  const int qy = threadIdx.x;
#endif
      if (qy < NUM_QUAD_1D) {
        for (int qx = 0; qx < NUM_MAX_1D; ++qx) {
          double gradX[NUM_DIM];
          double gradY[NUM_DIM];
          for (int vi = 0; vi < NUM_DIM; ++vi) {
            gradX[vi] = 0;
            gradY[vi] = 0;
          }
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            for (int vi = 0; vi < NUM_DIM; ++vi) {
              gradX[vi] += s_xy[ijkNM(vi, dx, qy,NUM_DIM,NUM_DOFS_1D)]  * s_dofToQuadD[ijN(qx, dx,NUM_QUAD_1D)];
              gradY[vi] += s_xDy[ijkNM(vi, dx, qy,NUM_DIM,NUM_DOFS_1D)] * s_dofToQuad[ijN(qx, dx,NUM_QUAD_1D)];
            }
          }
          for (int vi = 0; vi < NUM_DIM; ++vi) {
            s_gradv[ijkN(vi, 0, qx + qy*NUM_QUAD_1D,NUM_DIM)] = gradX[vi];
            s_gradv[ijkN(vi, 1, qx + qy*NUM_QUAD_1D,NUM_DIM)] = gradY[vi];
          }
        }
      }
    }

    sync;
#ifdef __LAMBDA__
    for (int qBlock = 0; qBlock < NUM_MAX_1D; ++qBlock) {
#else
    {  const int qBlock = threadIdx.x;
#endif
      for (int q = qBlock; q < NUM_QUAD; q += NUM_MAX_1D) {
        double q_gradv[NUM_DIM * NUM_DIM];//@dim(NUM_DIM, NUM_DIM);
        double q_stress[NUM_DIM * NUM_DIM];//@dim(NUM_DIM, NUM_DIM);

        const double invJ_00 = invJ[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
        const double invJ_10 = invJ[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
        const double invJ_01 = invJ[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
        const double invJ_11 = invJ[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];

        q_gradv[ijN(0,0,2)] = ((s_gradv[ijkN(0,0,q,2)]*invJ_00) + (s_gradv[ijkN(1,0,q,2)]*invJ_01));
        q_gradv[ijN(1,0,2)] = ((s_gradv[ijkN(0,0,q,2)]*invJ_10) + (s_gradv[ijkN(1,0,q,2)]*invJ_11));
        q_gradv[ijN(0,1,2)] = ((s_gradv[ijkN(0,1,q,2)]*invJ_00) + (s_gradv[ijkN(1,1,q,2)]*invJ_01));
        q_gradv[ijN(1,1,2)] = ((s_gradv[ijkN(0,1,q,2)]*invJ_10) + (s_gradv[ijkN(1,1,q,2)]*invJ_11));

        const double q_Jw = detJ[ijN(q,el,NUM_QUAD)]*quadWeights[q];

        const double q_rho = rho0DetJ0w[ijN(q,el,NUM_QUAD)]/q_Jw;
        const double q_e   = fmax(0.0,e[ijN(q,el,NUM_QUAD)]);

        // TODO: Input OccaVector eos(q,e) -> (stress, soundSpeed)
        const double s = -(GAMMA - 1.0) * q_rho * q_e;
        q_stress[ijN(0,0,2)] = s; q_stress[ijN(1,0,2)] = 0;
        q_stress[ijN(0,1,2)] = 0; q_stress[ijN(1,1,2)] = s;

        const double gradv00 = q_gradv[ijN(0,0,2)];
        const double gradv11 = q_gradv[ijN(1,1,2)];
        const double gradv10 = 0.5 * (q_gradv[ijN(1,0,2)] + q_gradv[ijN(0,1,2)]);
        q_gradv[ijN(1,0,2)] = gradv10;
        q_gradv[ijN(0,1,2)] = gradv10;

        double comprDirX = 1;
        double comprDirY = 0;
        double minEig = 0;
        // linalg/densemat.cpp: Eigensystem2S()
        if (gradv10 == 0) {
          minEig = (gradv00 < gradv11) ? gradv00 : gradv11;
        } else {
          const double zeta  = (gradv11 - gradv00) / (2.0 * gradv10);
          const double azeta = fabs(zeta);
          double t = 1.0 / (azeta + sqrt(1.0 + zeta*zeta));
          if ((t < 0) != (zeta < 0)) {
            t = -t;
          }

          const double c = sqrt(1.0 / (1.0 + t*t));
          const double s = c * t;
          t *= gradv10;

          if ((gradv00 - t) <= (gradv11 + t)) {
            minEig = gradv00 - t;
            comprDirX = c;
            comprDirY = -s;
          } else {
            minEig = gradv11 + t;
            comprDirX = s;
            comprDirY = c;
          }
        }

        // Computes the initial->physical transformation Jacobian.
        const double J_00 = J[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
        const double J_10 = J[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
        const double J_01 = J[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
        const double J_11 = J[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];

        const double invJ0_00 = invJ0[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)];
        const double invJ0_10 = invJ0[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)];
        const double invJ0_01 = invJ0[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)];
        const double invJ0_11 = invJ0[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)];

        const double Jpi_00 = ((J_00 * invJ0_00) + (J_10 * invJ0_01));
        const double Jpi_10 = ((J_00 * invJ0_10) + (J_10 * invJ0_11));
        const double Jpi_01 = ((J_01 * invJ0_00) + (J_11 * invJ0_01));
        const double Jpi_11 = ((J_01 * invJ0_10) + (J_11 * invJ0_11));

        const double physDirX = (Jpi_00 * comprDirX) + (Jpi_10 * comprDirY);
        const double physDirY = (Jpi_01 * comprDirX) + (Jpi_11 * comprDirY);

        const double q_h = H0 * sqrt((physDirX * physDirX) + (physDirY * physDirY));

        // TODO: soundSpeed will be an input as well (function call or values per q)
        const double soundSpeed = sqrt(GAMMA * (GAMMA - 1.0) * q_e);
        dtEst[ijN(q, el,NUM_QUAD)] = CFL * q_h / soundSpeed;

        if (USE_VISCOSITY) {
          // TODO: Check how we can extract outside of kernel
          const double mu = minEig;
          double coeff = 2.0 * q_rho * q_h * q_h * fabs(mu);
          if (mu < 0) {
            coeff += 0.5 * q_rho * q_h * soundSpeed;
          }
          for (int y = 0; y < NUM_DIM; ++y) {
            for (int x = 0; x < NUM_DIM; ++x) {
              q_stress[ijN(x,y,2)] += coeff * q_gradv[ijN(x,y,2)];
            }
          }
        }
        const double S00 = q_stress[ijN(0,0,2)]; const double S10 = q_stress[ijN(1,0,2)];
        const double S01 = q_stress[ijN(0,1,2)]; const double S11 = q_stress[ijN(1,1,2)];

        stressJinvT[ijklNM(0,0,q,el,NUM_DIM,NUM_QUAD)] = q_Jw * ((S00 * invJ_00) + (S10 * invJ_01));
        stressJinvT[ijklNM(1,0,q,el,NUM_DIM,NUM_QUAD)] = q_Jw * ((S00 * invJ_10) + (S10 * invJ_11));

        stressJinvT[ijklNM(0,1,q,el,NUM_DIM,NUM_QUAD)] = q_Jw * ((S01 * invJ_00) + (S11 * invJ_01));
        stressJinvT[ijklNM(1,1,q,el,NUM_DIM,NUM_QUAD)] = q_Jw * ((S01 * invJ_10) + (S11 * invJ_11));
      }
    }
  }
#ifdef __LAMBDA__
         );
#endif
}
// *****************************************************************************
typedef void (*fUpdateQuadratureDataS)(const double GAMMA,
                                       const double H0,
                                       const double CFL,
                                       const bool USE_VISCOSITY,
                                       const int numElements,
                                       const double* restrict dofToQuad,
                                       const double* restrict dofToQuadD,
                                       const double* restrict quadWeights,
                                       const double* restrict v,
                                       const double* restrict e,
                                       const double* restrict rho0DetJ0w,
                                       const double* restrict invJ0,
                                       const double* restrict J,
                                       const double* restrict invJ,
                                       const double* restrict detJ,
                                       double* restrict stressJinvT,
                                       double* restrict dtEst);

// *****************************************************************************
void rUpdateQuadratureDataS(const double GAMMA,
                            const double H0,
                            const double CFL,
                            const bool USE_VISCOSITY,
                            const int NUM_DIM,
                            const int NUM_QUAD,
                            const int NUM_QUAD_1D,
                            const int NUM_DOFS_1D,
                            const int nzones,
                            const double* restrict dofToQuad,
                            const double* restrict dofToQuadD,
                            const double* restrict quadWeights,
                            const double* restrict v,
                            const double* restrict e,
                            const double* restrict rho0DetJ0w,
                            const double* restrict invJ0,
                            const double* restrict J,
                            const double* restrict invJ,
                            const double* restrict detJ,
                            double* restrict stressJinvT,
                            double* restrict dtEst){
  push(Green);
#ifndef __LAMBDA__
  const int grid = nzones;
  const int blck = (NUM_QUAD_1D<NUM_DOFS_1D)?NUM_DOFS_1D:NUM_QUAD_1D;
#endif
#ifdef __TEMPLATES__
  assert(LOG2(NUM_DIM)<=4);
  assert(LOG2(NUM_DOFS_1D-2)<=4);
  assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
  assert(IROOT(NUM_DIM,NUM_QUAD)==NUM_QUAD_1D);
  const unsigned int id = (NUM_DIM<<4)|(NUM_DOFS_1D-2);
  static std::unordered_map<unsigned int,fUpdateQuadratureDataS> call = {
    // 2D
    {0x20,&rUpdateQuadratureData2S<2,2*2,2,2>},
    {0x21,&rUpdateQuadratureData2S<2,4*4,4,3>},
    {0x22,&rUpdateQuadratureData2S<2,6*6,6,4>},
    {0x23,&rUpdateQuadratureData2S<2,8*8,8,5>},
    {0x24,&rUpdateQuadratureData2S<2,10*10,10,6>},
    {0x25,&rUpdateQuadratureData2S<2,12*12,12,7>},     
    {0x26,&rUpdateQuadratureData2S<2,14*14,14,8>},
    {0x27,&rUpdateQuadratureData2S<2,16*16,16,9>},
    {0x28,&rUpdateQuadratureData2S<2,18*18,18,10>},
    {0x29,&rUpdateQuadratureData2S<2,20*20,20,11>},
    {0x2A,&rUpdateQuadratureData2S<2,22*22,22,12>},
    {0x2B,&rUpdateQuadratureData2S<2,24*24,24,13>},
    {0x2C,&rUpdateQuadratureData2S<2,26*26,26,14>},
    {0x2D,&rUpdateQuadratureData2S<2,28*28,28,15>},
    //{0x2E,&rUpdateQuadratureData2S<2,30*30,30,16>}, uses too much shared data
    //{0x2F,&rUpdateQuadratureData2S<2,32*32,32,17>}, uses too much shared data
    // 3D
/*    {0x30,&rUpdateQuadratureData3S<3,2*2*2,2,2>},
    {0x31,&rUpdateQuadratureData3S<3,4*4*4,4,3>},
    {0x32,&rUpdateQuadratureData3S<3,6*6*6,6,4>},
    {0x33,&rUpdateQuadratureData3S<3,8*8*8,8,5>},
    {0x34,&rUpdateQuadratureData3S<3,10*10*10,10,6>},
    {0x35,&rUpdateQuadratureData3S<3,12*12*12,12,7>},
    {0x36,&rUpdateQuadratureData3S<3,14*14*14,14,8>},
    {0x37,&rUpdateQuadratureData3S<3,16*16*16,16,9>},
    {0x38,&rUpdateQuadratureData3S<3,18*18*18,18,10>},
    {0x39,&rUpdateQuadratureData3S<3,20*20*20,20,11>},
    {0x3A,&rUpdateQuadratureData3S<3,22*22*22,22,12>},
    {0x3B,&rUpdateQuadratureData3S<3,24*24*24,24,13>},
    {0x3C,&rUpdateQuadratureData3S<3,26*26*26,26,14>},
    {0x3D,&rUpdateQuadratureData3S<3,28*28*28,28,15>},
    {0x3E,&rUpdateQuadratureData3S<3,30*30*30,30,16>},
    {0x3F,&rUpdateQuadratureData3S<3,32*32*32,32,17>},
*/
  };
  if (!call[id]){
    printf("\n[rUpdateQuadratureDataS] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rUpdateQuadratureData2S,id,grid,blck,
        GAMMA,H0,CFL,USE_VISCOSITY,
        nzones,dofToQuad,dofToQuadD,quadWeights,
        v,e,rho0DetJ0w,invJ0,J,invJ,detJ,
        stressJinvT,dtEst);
#else
  if (NUM_DIM==2)
    call0(rUpdateQuadratureData2S,id,grid,blck,
          NUM_DIM,NUM_QUAD,NUM_QUAD_1D,NUM_DOFS_1D,
          GAMMA,H0,CFL,USE_VISCOSITY,
          nzones,dofToQuad,dofToQuadD,quadWeights,
          v,e,rho0DetJ0w,invJ0,J,invJ,detJ,
          stressJinvT,dtEst);
  else assert(false);
#endif
  pop();
}
