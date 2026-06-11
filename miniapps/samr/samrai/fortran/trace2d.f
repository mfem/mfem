c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   F77 routines for trace in 2d.
c
c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   m4 include file for dimensioning 2d arrays in FORTRAN routines.
c
c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   m4 include file for 2d trace calculation.
c

      subroutine inittraceflux2d(
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  uval,
     &  tracelft0,tracelft1,
     &  tracergt0,tracergt1,
     &  fluxriem0,fluxriem1)
c***********************************************************************
      implicit none
c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   m4 include file defining FORTRAN constants.
c
      double precision zero,sixth,fourth,third,half,twothird,rt75,one,
     &  onept5,two,three,pi,four,seven,smallr
      parameter (zero=0.d0)
      parameter (sixth=0.16666666666667d0)
      parameter (fourth=.25d0)
      parameter (third=.333333333333333d0)
      parameter (half=.5d0)
      parameter (twothird=.66666666666667d0)
      parameter (rt75=.8660254037844d0)
      parameter (one=1.d0)
      parameter (onept5=1.5d0)
      parameter (two=2.d0)
      parameter (three=3.d0)
      parameter(pi=3.14159265358979323846d0)
      parameter (four=4.d0)
      parameter (seven=7.d0)
      parameter (smallr=1.0d-32)
c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   m4 include file defining common block.
c
      common/probparams/
     &  PIECEWISE_CONSTANT_X,PIECEWISE_CONSTANT_Y,PIECEWISE_CONSTANT_Z,
     &  SINE_CONSTANT_X,SINE_CONSTANT_Y,SINE_CONSTANT_Z,SPHERE,
     &  CELLG,FACEG,FLUXG
      integer 
     &  PIECEWISE_CONSTANT_X,PIECEWISE_CONSTANT_Y,PIECEWISE_CONSTANT_Z,
     &  SINE_CONSTANT_X,SINE_CONSTANT_Y,SINE_CONSTANT_Z,SPHERE,
     &  CELLG,FACEG,FLUXG
c***********************************************************************
c***********************************************************************
c input arrays:
      integer ifirst0,ilast0,ifirst1,ilast1
      double precision
     &     uval(ifirst0-CELLG:ilast0+CELLG,
     &          ifirst1-CELLG:ilast1+CELLG),
     &     fluxriem0(ifirst0-FLUXG:ilast0+1+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG),
     &     tracelft0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     tracergt0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     fluxriem1(ifirst1-FLUXG:ilast1+1+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG),
     &     tracelft1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     tracergt1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG)
c***********************************************************************
c initialize left and right states at cell edges
c (first-order upwind)
c***********************************************************************
c
      integer ic0,ic1,ie0,ie1
c*********************************************************************** 

         do  ic1=ifirst1-FACEG,ilast1+FACEG
           ie0=ifirst0-FACEG
           tracergt0(ie0,ic1)=uval(ie0,ic1)
           tracelft0(ie0,ic1)=uval(ie0,ic1)

           do  ie0=ifirst0+1-FACEG,ilast0+FACEG
             tracelft0(ie0,ic1)=uval(ie0-1,ic1)
             tracergt0(ie0,ic1)=uval(ie0,ic1)
           enddo

           ie0=ilast0+FACEG+1
           tracelft0(ie0,ic1)=uval(ie0-1,ic1)
           tracergt0(ie0,ic1)=uval(ie0-1,ic1)
         enddo
 
         do  ic0=ifirst0-FACEG,ilast0+FACEG
           ie1=ifirst1-FACEG
           tracergt1(ie1,ic0)=uval(ic0,ie1)
           tracelft1(ie1,ic0)=uval(ic0,ie1)

           do  ie1=ifirst1+1-FACEG,ilast1+FACEG
             tracelft1(ie1,ic0)=uval(ic0,ie1-1)
             tracergt1(ie1,ic0)=uval(ic0,ie1)
           enddo

           ie1=ilast1+FACEG+1
           tracelft1(ie1,ic0)=uval(ic0,ie1-1)
           tracergt1(ie1,ic0)=uval(ic0,ie1-1)
         enddo
 
c
c     we initialize the flux to be zero

      do ic1=ifirst1-FLUXG,ilast1+FLUXG
        do ie0=ifirst0-FLUXG,ilast0+FLUXG+1
            fluxriem0(ie0,ic1) = zero
        enddo
      enddo
c
      do ic0=ifirst0-FLUXG,ilast0+FLUXG
        do ie1=ifirst1-FLUXG,ilast1+FLUXG+1
            fluxriem1(ie1,ic0) = zero
        enddo
      enddo
c
c     call flush(6)
      return
      end 
c
c***********************************************************************
c***********************************************************************
c***********************************************************************
      subroutine chartracing2d0(dt,
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  mc,
     &  dx,
     &  advecspeed,igdnv,
     &  tracelft,tracergt,
     &  ttcelslp,ttedgslp, 
     &  ttraclft,ttracrgt)
c***********************************************************************
      implicit none
c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   m4 include file defining common block.
c
      common/probparams/
     &  PIECEWISE_CONSTANT_X,PIECEWISE_CONSTANT_Y,PIECEWISE_CONSTANT_Z,
     &  SINE_CONSTANT_X,SINE_CONSTANT_Y,SINE_CONSTANT_Z,SPHERE,
     &  CELLG,FACEG,FLUXG
      integer 
     &  PIECEWISE_CONSTANT_X,PIECEWISE_CONSTANT_Y,PIECEWISE_CONSTANT_Z,
     &  SINE_CONSTANT_X,SINE_CONSTANT_Y,SINE_CONSTANT_Z,SPHERE,
     &  CELLG,FACEG,FLUXG
c input arrays:
      integer ifirst0,ilast0,ifirst1,ilast1
      integer mc,igdnv
      double precision dt 
c variables in 1d axis indexed
      double precision 
     &     dx,advecspeed
c variables in 2d axis indexed
      double precision
     &     tracelft(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     tracergt(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG)
c  side variables ifirst0 to ifirst0+mc plus ghost cells
      double precision
     &  ttedgslp(ifirst0-FACEG:ifirst0+mc+FACEG),
     &  ttraclft(ifirst0-FACEG:ifirst0+mc+FACEG),
     &  ttracrgt(ifirst0-FACEG:ifirst0+mc+FACEG)
c  cell variables ifirst0 to ifirst0+mc-1 plus ghost cells
      double precision 
     &  ttcelslp(ifirst0-CELLG:ifirst0+mc-1+CELLG)
c***********************************************************************
c
      integer ic0,ic1,idir
c*********************************************************************** 
c***********************************************************************
c trace higher order states at cell edges
c***********************************************************************
      idir = 0
          do ic1=ifirst1-2,ilast1+2
            do ic0=ifirst0-FACEG,ilast0+FACEG
                ttraclft(ic0) = tracelft(ic0,ic1)
                ttracrgt(ic0) = tracergt(ic0,ic1)
            enddo

            call trace(dt,ifirst0,ilast0,mc,
     &        dx,idir,advecspeed,igdnv,
     &        ttraclft,ttracrgt,
     &        ttcelslp,ttedgslp)
            do ic0=ifirst0-FACEG,ilast0+FACEG
                tracelft(ic0,ic1) = ttraclft(ic0)
                tracergt(ic0,ic1) = ttracrgt(ic0)
            enddo
          enddo
c***********************************************************************
      return
      end
c
c***********************************************************************
c***********************************************************************
c***********************************************************************
      subroutine chartracing2d1(dt,
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  mc,
     &  dx,
     &  advecspeed,igdnv,
     &  tracelft,tracergt,
     &  ttcelslp,ttedgslp,  
     &  ttraclft,ttracrgt)
c***********************************************************************
      implicit none
c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   m4 include file defining common block.
c
      common/probparams/
     &  PIECEWISE_CONSTANT_X,PIECEWISE_CONSTANT_Y,PIECEWISE_CONSTANT_Z,
     &  SINE_CONSTANT_X,SINE_CONSTANT_Y,SINE_CONSTANT_Z,SPHERE,
     &  CELLG,FACEG,FLUXG
      integer 
     &  PIECEWISE_CONSTANT_X,PIECEWISE_CONSTANT_Y,PIECEWISE_CONSTANT_Z,
     &  SINE_CONSTANT_X,SINE_CONSTANT_Y,SINE_CONSTANT_Z,SPHERE,
     &  CELLG,FACEG,FLUXG
c input arrays:
      integer ifirst0,ilast0,ifirst1,ilast1
      integer mc,igdnv
      double precision dt
c variables in 1d axis indexed
      double precision 
     &     dx,advecspeed
c variables in 2d axis indexed
      double precision
     &     tracelft(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     tracergt(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG)
c  side variables ifirst1 to ifirst1+mc plus ghost cells
      double precision
     &  ttedgslp(ifirst1-FACEG:ifirst1+mc+FACEG),
     &  ttraclft(ifirst1-FACEG:ifirst1+mc+FACEG),
     &  ttracrgt(ifirst1-FACEG:ifirst1+mc+FACEG)
c  cell variables ifirst1 to ifirst1+mc-1 plus ghost cells
      double precision 
     &  ttcelslp(ifirst1-CELLG:ifirst1+mc-1+CELLG)
c***********************************************************************
c
      integer ic0,ic1,idir
c*********************************************************************** 
c***********************************************************************
c trace higher order states at cell edges
c***********************************************************************
      idir = 1
          do ic0=ifirst0-2,ilast0+2
            do ic1=ifirst1-FACEG,ilast1+FACEG
                ttraclft(ic1) = tracelft(ic1,ic0)
                ttracrgt(ic1) = tracergt(ic1,ic0)
            enddo

            call trace(dt,ifirst1,ilast1,mc,
     &        dx,idir,advecspeed,igdnv,
     &        ttraclft,ttracrgt,
     &        ttcelslp,ttedgslp)
            do ic1=ifirst1-FACEG,ilast1+FACEG
                tracelft(ic1,ic0) = ttraclft(ic1)
                tracergt(ic1,ic0) = ttracrgt(ic1)
            enddo
          enddo
c***********************************************************************
      return
      end
c
c***********************************************************************
c***********************************************************************
c***********************************************************************
