c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   F77 routines for trace in 3d.
c
c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   m4 include file for dimensioning 3d arrays in FORTRAN routines.
c
c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   m4 include file for 3d trace calculation.
c

      subroutine inittraceflux3d(
     &  ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
     &  uval,
     &  tracelft0,tracelft1,tracelft2,
     &  tracergt0,tracergt1,tracergt2,
     &  fluxriem0,fluxriem1,fluxriem2)
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
      integer ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2
      double precision
     &     uval(ifirst0-CELLG:ilast0+CELLG,
     &          ifirst1-CELLG:ilast1+CELLG,
     &          ifirst2-CELLG:ilast2+CELLG),
     &     fluxriem0(ifirst0-FLUXG:ilast0+1+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG),
     &     tracelft0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG),
     &     tracergt0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG),
     &     fluxriem1(ifirst1-FLUXG:ilast1+1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG),
     &     tracelft1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     tracergt1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     fluxriem2(ifirst2-FLUXG:ilast2+1+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG),
     &     tracelft2(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     tracergt2(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG)
c***********************************************************************
c initialize left and right states at cell edges
c (first-order upwind)
c***********************************************************************
c
      integer ic0,ic1,ie0,ie1,ic2,ie2
c***********************************************************************

      do  ic2=ifirst2-FACEG,ilast2+FACEG
         do  ic1=ifirst1-FACEG,ilast1+FACEG
           ie0=ifirst0-FACEG
           tracergt0(ie0,ic1,ic2)=uval(ie0,ic1,ic2)
           tracelft0(ie0,ic1,ic2)=uval(ie0,ic1,ic2)

           do  ie0=ifirst0+1-FACEG,ilast0+FACEG
             tracelft0(ie0,ic1,ic2)=uval(ie0-1,ic1,ic2)
             tracergt0(ie0,ic1,ic2)=uval(ie0,ic1,ic2)
           enddo

           ie0=ilast0+FACEG+1
           tracelft0(ie0,ic1,ic2)=uval(ie0-1,ic1,ic2)
           tracergt0(ie0,ic1,ic2)=uval(ie0-1,ic1,ic2)
         enddo
      enddo

 

      do  ic0=ifirst0-FACEG,ilast0+FACEG
         do  ic2=ifirst2-FACEG,ilast2+FACEG
           ie1=ifirst1-FACEG
           tracergt1(ie1,ic2,ic0)=uval(ic0,ie1,ic2)
           tracelft1(ie1,ic2,ic0)=uval(ic0,ie1,ic2)

           do  ie1=ifirst1+1-FACEG,ilast1+FACEG
             tracelft1(ie1,ic2,ic0)=uval(ic0,ie1-1,ic2)
             tracergt1(ie1,ic2,ic0)=uval(ic0,ie1,ic2)
           enddo

           ie1=ilast1+FACEG+1
           tracelft1(ie1,ic2,ic0)=uval(ic0,ie1-1,ic2)
           tracergt1(ie1,ic2,ic0)=uval(ic0,ie1-1,ic2)
         enddo
      enddo

 

      do  ic1=ifirst1-FACEG,ilast1+FACEG
         do  ic0=ifirst0-FACEG,ilast0+FACEG
           ie2=ifirst2-FACEG
           tracergt2(ie2,ic0,ic1)=uval(ic0,ic1,ie2)
           tracelft2(ie2,ic0,ic1)=uval(ic0,ic1,ie2)

           do  ie2=ifirst2+1-FACEG,ilast2+FACEG
             tracelft2(ie2,ic0,ic1)=uval(ic0,ic1,ie2-1)
             tracergt2(ie2,ic0,ic1)=uval(ic0,ic1,ie2)
           enddo

           ie2=ilast2+FACEG+1
           tracelft2(ie2,ic0,ic1)=uval(ic0,ic1,ie2-1)
           tracergt2(ie2,ic0,ic1)=uval(ic0,ic1,ie2-1)
         enddo
      enddo

 

c
c     we initialize the flux to be zero

      do ic2=ifirst2-FLUXG,ilast2+FLUXG
         do ic1=ifirst1-FLUXG,ilast1+FLUXG
           do ie0=ifirst0-FLUXG,ilast0+FLUXG+1
               fluxriem0(ie0,ic1,ic2) = zero
           enddo
         enddo
      enddo
c
      do ic2=ifirst2-FLUXG,ilast2+FLUXG
         do ic0=ifirst0-FLUXG,ilast0+FLUXG
            do ie1=ifirst1-FLUXG,ilast1+FLUXG+1
                 fluxriem1(ie1,ic2,ic0) = zero
            enddo
         enddo
      enddo
c
      do ic1=ifirst1-FLUXG,ilast1+FLUXG
         do ic0=ifirst0-FLUXG,ilast0+FLUXG
            do ie2=ifirst2-FLUXG,ilast2+FLUXG+1
                 fluxriem2(ie2,ic0,ic1) = zero
            enddo
         enddo
      enddo
c
c      call flush(6)
      return
      end 
c
c***********************************************************************
c***********************************************************************
c***********************************************************************
c
      subroutine chartracing3d0(dt,
     &  ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
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
      integer ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2
      integer mc,igdnv
      double precision dt 
c variables in 1d axis indexed
      double precision 
     &     dx,advecspeed
c variables in 2d axis indexed
      double precision
     &     tracelft(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG),
     &     tracergt(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG)
c  side variables ifirst0 to ifirst0+mc plus ghost cells
      double precision
     & ttedgslp(ifirst0-FACEG:ifirst0+mc+FACEG),
     & ttraclft(ifirst0-FACEG:ifirst0+mc+FACEG),
     & ttracrgt(ifirst0-FACEG:ifirst0+mc+FACEG)
c  cell variables ifirst0 to ifirst0+mc-1 plus ghost cells
      double precision 
     &  ttcelslp(ifirst0-CELLG:ifirst0+mc-1+CELLG)
c***********************************************************************
      integer ic0,ic1,ic2,idir
c***********************************************************************
c trace higher order states at cell edges
c***********************************************************************
      idir = 0
        do ic2=ifirst2-2,ilast2+2
          do ic1=ifirst1-2,ilast1+2
              do ic0=ifirst0-FACEG,ilast0+FACEG
                ttraclft(ic0) = tracelft(ic0,ic1,ic2)
                ttracrgt(ic0) = tracergt(ic0,ic1,ic2)
              enddo
   
            call trace(dt,ifirst0,ilast0,mc,
     &        dx,idir,advecspeed,igdnv,
     &        ttraclft,ttracrgt,
     &        ttcelslp,ttedgslp)
            do ic0=ifirst0-FACEG,ilast0+FACEG
                tracelft(ic0,ic1,ic2) = ttraclft(ic0)
                tracergt(ic0,ic1,ic2) = ttracrgt(ic0)
            enddo
          enddo
        enddo
c***********************************************************************
      return
      end
c***********************************************************************
c***********************************************************************
c***********************************************************************
      subroutine chartracing3d1(dt,
     &  ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
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
      integer ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2
      integer mc,igdnv
      double precision dt 
c variables in 1d axis indexed
      double precision 
     &     dx,advecspeed
c variables in 2d axis indexed
      double precision
     &     tracelft(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     tracergt(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG)
      double precision
     & ttedgslp(ifirst1-FACEG:ifirst1+mc+FACEG),
     & ttraclft(ifirst1-FACEG:ifirst1+mc+FACEG),
     & ttracrgt(ifirst1-FACEG:ifirst1+mc+FACEG)
c  cell variables ifirst1 to ifirst1+mc-1 plus ghost cells
      double precision 
     &  ttcelslp(ifirst1-CELLG:ifirst1+mc-1+CELLG)
c***********************************************************************
      integer ic0,ic1,ic2,idir
c***********************************************************************
c trace higher order states at cell edges
c***********************************************************************
      idir = 1
        do ic0=ifirst0-2,ilast0+2
          do ic2=ifirst2-2,ilast2+2
              do ic1=ifirst1-FACEG,ilast1+FACEG
                ttraclft(ic1) = tracelft(ic1,ic2,ic0)
                ttracrgt(ic1) = tracergt(ic1,ic2,ic0)
              enddo
   
            call trace(dt,ifirst1,ilast1,mc,
     &        dx,idir,advecspeed,igdnv,
     &        ttraclft,ttracrgt,
     &        ttcelslp,ttedgslp)
            do ic1=ifirst1-FACEG,ilast1+FACEG
                tracelft(ic1,ic2,ic0) = ttraclft(ic1)
                tracergt(ic1,ic2,ic0) = ttracrgt(ic1)
            enddo
          enddo
        enddo
c***********************************************************************
      return
      end
c
c***********************************************************************
c***********************************************************************
c***********************************************************************
      subroutine chartracing3d2(dt,
     &  ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
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
      integer ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2
      integer mc,igdnv
      double precision dt 
c variables in 1d axis indexed
      double precision 
     &     dx,advecspeed
c variables in 2d axis indexed
      double precision
     &     tracelft(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     tracergt(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG)
      double precision
     & ttedgslp(ifirst2-FACEG:ifirst2+mc+FACEG),
     & ttraclft(ifirst2-FACEG:ifirst2+mc+FACEG),
     & ttracrgt(ifirst2-FACEG:ifirst2+mc+FACEG)
c  cell variables ifirst2 to ifirst2+mc-1 plus ghost cells
      double precision 
     &  ttcelslp(ifirst2-CELLG:ifirst2+mc-1+CELLG)
c***********************************************************************
      integer ic0,ic1,ic2,idir
c***********************************************************************
c trace higher order states at cell edges
c***********************************************************************
      idir = 2
        do ic1=ifirst1-2,ilast1+2
          do ic0=ifirst0-2,ilast0+2
              do ic2=ifirst2-FACEG,ilast2+FACEG
                ttraclft(ic2) = tracelft(ic2,ic0,ic1)
                ttracrgt(ic2) = tracergt(ic2,ic0,ic1)
              enddo
   
            call trace(dt,ifirst2,ilast2,mc,
     &        dx,idir,advecspeed,igdnv,
     &        ttraclft,ttracrgt,
     &        ttcelslp,ttedgslp)
            do ic2=ifirst2-FACEG,ilast2+FACEG
                tracelft(ic2,ic0,ic1) = ttraclft(ic2)
                tracergt(ic2,ic0,ic1) = ttracrgt(ic2)
            enddo
          enddo
        enddo
c***********************************************************************
      return
      end
c
c***********************************************************************
c***********************************************************************
c***********************************************************************

