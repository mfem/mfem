c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   F77 routines for flux computation in 3d.
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
c Description:   m4 include file for 3d flux calculation.
c

      subroutine fluxcorrec2d(dt,
     &  ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
     &  dx,advecspeed,idir,
     &  flux0,flux1,flux2,
     &  tracelft0,tracelft1,tracelft2,
     &  tracergt0,tracergt1,tracergt2,
     &  ttracelft0,ttracelft1,ttracelft2,
     &  ttracergt0,ttracergt1,ttracergt2)
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
      double precision dt 
      integer idir
c variables in 1d axis indexed
c
      double precision 
     &     dx(0:3-1)
c variables in 2d cell indexed         
      double precision
     &     advecspeed(0:3-1),
c
     &     flux0(ifirst0-FLUXG:ilast0+1+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG),
     &     flux1(ifirst1-FLUXG:ilast1+1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG), 
     &     flux2(ifirst2-FLUXG:ilast2+1+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG), 
c
     &     tracelft0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG),
     &     tracelft1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     tracelft2(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     tracergt0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG),
     &     tracergt1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     tracergt2(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
c
     &     ttracelft0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG),
     &     ttracelft1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     ttracelft2(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     ttracergt0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG),
     &     ttracergt1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     ttracergt2(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG) 
c
c***********************************************************************     
c
      integer ic0,ic1,ic2
      double precision trnsvers
     
c     ******************************************************************
c     * complete tracing at cell edges
c     ******************************************************************
c         
c
c  "Forward" computation of transverse flux terms
c
      if (idir.eq.1) then
c
c   correct the 0-direction with 1-fluxes
      do ic2=ifirst2-1,ilast2+1
         do ic1=ifirst1,ilast1
           do ic0=ifirst0-1,ilast0+1
             trnsvers=
     &           (flux1(ic1+1,ic2,ic0)-flux1(ic1,ic2,ic0))*half/dx(1)
c
             ttracelft0(ic0+1,ic1,ic2)=tracelft0(ic0+1,ic1,ic2)
     &                                    - trnsvers
             ttracergt0(ic0  ,ic1,ic2)=tracergt0(ic0  ,ic1,ic2)
     &                                    - trnsvers
           enddo
         enddo
      enddo
c
c   correct the 1-direction with 0-fluxes
      do ic2=ifirst2-1,ilast2+1
         do ic0=ifirst0,ilast0
           do ic1=ifirst1-1,ilast1+1
             trnsvers=
     &           (flux0(ic0+1,ic1,ic2)-flux0(ic0,ic1,ic2))*half/dx(0)
c
             ttracelft1(ic1+1,ic2,ic0)=tracelft1(ic1+1,ic2,ic0)
     &                                    - trnsvers
             ttracergt1(ic1  ,ic2,ic0)=tracergt1(ic1  ,ic2,ic0)
     &                                    - trnsvers
           enddo
         enddo
      enddo
c
c   correct the 2-direction with 0-fluxes
      do ic1=ifirst1-1,ilast1+1
         do ic0=ifirst0,ilast0
           do ic2=ifirst2-1,ilast2+1
             trnsvers=
     &           (flux0(ic0+1,ic1,ic2)-flux0(ic0,ic1,ic2))*half/dx(0)
c
             ttracelft2(ic2+1,ic0,ic1)=tracelft2(ic2+1,ic0,ic1)
     &                                    - trnsvers
             ttracergt2(ic2  ,ic0,ic1)=tracergt2(ic2  ,ic0,ic1)
     &                                    - trnsvers
           enddo
         enddo
      enddo
c
c  "Backward" computation of transverse flux terms
c
      elseif (idir.eq.-1) then
c
c   correct the 0-direction with 2-fluxes
      do ic1=ifirst1-1,ilast1+1
         do ic2=ifirst2,ilast2
           do ic0=ifirst0-1,ilast0+1
             trnsvers=
     &           (flux2(ic2+1,ic0,ic1)-flux2(ic2,ic0,ic1))*half/dx(2)
c
             ttracelft0(ic0+1,ic1,ic2)=tracelft0(ic0+1,ic1,ic2)
     &                                    - trnsvers
             ttracergt0(ic0  ,ic1,ic2)=tracergt0(ic0  ,ic1,ic2)
     &                                    - trnsvers
           enddo
         enddo
      enddo
c
c   correct the 1-direction with 2-fluxes
      do ic0=ifirst0-1,ilast0+1
         do ic2=ifirst2,ilast2
           do ic1=ifirst1-1,ilast1+1
             trnsvers=
     &           (flux2(ic2+1,ic0,ic1)-flux2(ic2,ic0,ic1))*half/dx(2)
c
             ttracelft1(ic1+1,ic2,ic0)=tracelft1(ic1+1,ic2,ic0)
     &                                    - trnsvers
             ttracergt1(ic1  ,ic2,ic0)=tracergt1(ic1  ,ic2,ic0)
     &                                    - trnsvers
           enddo
         enddo
      enddo
c
c   correct the 2-direction with 1-fluxes
      do ic0=ifirst0-1,ilast0+1
         do ic1=ifirst1,ilast1
           do ic2=ifirst2-1,ilast2+1
             trnsvers=
     &           (flux1(ic1+1,ic2,ic0)-flux1(ic1,ic2,ic0))*half/dx(1)
c
             ttracelft2(ic2+1,ic0,ic1)=tracelft2(ic2+1,ic0,ic1)
     &                                    - trnsvers
             ttracergt2(ic2  ,ic0,ic1)=tracergt2(ic2  ,ic0,ic1)
     &                                    - trnsvers
           enddo
         enddo
      enddo
c
      endif
c
      return
      end   
c
c***********************************************************************
c***********************************************************************
c***********************************************************************
      subroutine fluxcorrec3d(dt,
     &  ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
     &  dx,
     &  advecspeed,
     &  fluxa0,fluxa1,fluxa2,
     &  fluxb0,fluxb1,fluxb2,
     &  tracelft0,tracelft1,tracelft2,
     &  tracergt0,tracergt1,tracergt2)
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
      double precision dt 
c variables in 1d axis indexed
c
      double precision 
     &     dx(0:3-1)
c variables in 2d cell indexed         
      double precision
     &     advecspeed(0:3-1),
     &     fluxa0(ifirst0-FLUXG:ilast0+1+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG),
     &     fluxa1(ifirst1-FLUXG:ilast1+1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG), 
     &     fluxa2(ifirst2-FLUXG:ilast2+1+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG), 
     &     fluxb0(ifirst0-FLUXG:ilast0+1+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG),
     &     fluxb1(ifirst1-FLUXG:ilast1+1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG), 
     &     fluxb2(ifirst2-FLUXG:ilast2+1+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG), 
     &     tracelft0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG),
     &     tracergt0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG),
     &     tracelft1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     tracergt1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     tracelft2(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     tracergt2(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG)
c
c***********************************************************************     
c
      integer ic0,ic1,ic2
      double precision trnsvers
     
c     ******************************************************************
c     * complete tracing at cell edges
c     ******************************************************************
c         
c   correct the 2-direction with 01-fluxes
      do ic2=ifirst2-1,ilast2+1
         do ic1=ifirst1,ilast1
           do ic0=ifirst0,ilast0
             trnsvers=half*(
     &          (fluxa0(ic0+1,ic1,ic2)-fluxa0(ic0,ic1,ic2))/dx(0)+
     &          (fluxa1(ic1+1,ic2,ic0)-fluxa1(ic1,ic2,ic0))/dx(1))
  
             tracelft2(ic2+1,ic0,ic1)=tracelft2(ic2+1,ic0,ic1)
     &                                           - trnsvers
             tracergt2(ic2  ,ic0,ic1)=tracergt2(ic2  ,ic0,ic1)
     &                                           - trnsvers
           enddo
         enddo
      enddo
c         
c   correct the 1-direction with 20-fluxes
      do ic1=ifirst1-1,ilast1+1
         do ic0=ifirst0,ilast0
           do ic2=ifirst2,ilast2
             trnsvers=half*(
     &          (fluxa2(ic2+1,ic0,ic1)-fluxa2(ic2,ic0,ic1))/dx(2)+
     &          (fluxb0(ic0+1,ic1,ic2)-fluxb0(ic0,ic1,ic2))/dx(0))
  
             tracelft1(ic1+1,ic2,ic0)=tracelft1(ic1+1,ic2,ic0)
     &                                           - trnsvers
             tracergt1(ic1  ,ic2,ic0)=tracergt1(ic1  ,ic2,ic0)
     &                                           - trnsvers
           enddo
         enddo
      enddo
c         
c   correct the 0-direction with 12-fluxes
      do ic0=ifirst0-1,ilast0+1
         do ic2=ifirst2,ilast2
           do ic1=ifirst1,ilast1
             trnsvers=half*(
     &          (fluxb1(ic1+1,ic2,ic0)-fluxb1(ic1,ic2,ic0))/dx(1)+
     &          (fluxb2(ic2+1,ic0,ic1)-fluxb2(ic2,ic0,ic1))/dx(2))
  
             tracelft0(ic0+1,ic1,ic2)=tracelft0(ic0+1,ic1,ic2)
     &                                           - trnsvers
             tracergt0(ic0  ,ic1,ic2)=tracergt0(ic0  ,ic1,ic2)
     &                                           - trnsvers
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
      subroutine fluxcalculation3d(dt,xcell0,xcell1,visco,dx,
     &  ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
     &  advecspeed,
     &  flux0,flux1,flux2,
     &  trlft0,trlft1,trlft2,
     &  trrgt0,trrgt1,trrgt2)
     
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
      integer xcell0,xcell1,visco
      double precision dt 
      double precision 
     &     dx(0:3-1)
c variables in 2d cell indexed         
      double precision
     &     advecspeed(0:3-1)
c variables in 2d side indexed         
      double precision
     &     flux0(ifirst0-FLUXG:ilast0+1+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG),
     &     flux1(ifirst1-FLUXG:ilast1+1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG), 
     &     flux2(ifirst2-FLUXG:ilast2+1+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG), 
     &     trlft0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG),
     &     trrgt0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG),
     &     trlft1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     trrgt1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst2-FACEG:ilast2+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     trlft2(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     trrgt2(ifirst2-FACEG:ilast2+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG)
c
c***********************************************************************     
c
      integer ic0,ic1,ic2,ie0,ie1,ie2
      double precision   riemst
c
c***********************************************************************
c solve riemann problems for conservative flux
c  arguments: ( axis for RP, other axis, extra cells-direction)
c***********************************************************************
c      

c     write(6,*) "checking onedr sol in riemann solve  "
c     write(6,*) "         dt= ",dt
      do ic2=ifirst2-xcell1,ilast2+xcell1
         do ic1=ifirst1-xcell0,ilast1+xcell0
           do ie0=ifirst0,ilast0+1
 
             if (advecspeed(0).ge.zero) then
               riemst= trlft0(ie0,ic1,ic2)
             else
               riemst= trrgt0(ie0,ic1,ic2)
             endif
             flux0(ie0,ic1,ic2)= dt*riemst*advecspeed(0)

           enddo
         enddo
      enddo

c
c     write(6,*) "checking onedr sol in riemann solve  "
c     write(6,*) "         dt= ",dt
      do ic2=ifirst2-xcell1,ilast2+xcell1
         do ic0=ifirst0-xcell0,ilast0+xcell0
           do ie1=ifirst1,ilast1+1
 
             if (advecspeed(1).ge.zero) then
               riemst= trlft1(ie1,ic2,ic0)
             else
               riemst= trrgt1(ie1,ic2,ic0)
             endif
             flux1(ie1,ic2,ic0)= dt*riemst*advecspeed(1)

           enddo
         enddo
      enddo

c
c     write(6,*) "checking onedr sol in riemann solve  "
c     write(6,*) "         dt= ",dt
      do ic1=ifirst1-xcell1,ilast1+xcell1
         do ic0=ifirst0-xcell0,ilast0+xcell0
           do ie2=ifirst2,ilast2+1
 
             if (advecspeed(2).ge.zero) then
               riemst= trlft2(ie2,ic0,ic1)
             else
               riemst= trrgt2(ie2,ic0,ic1)
             endif
             flux2(ie2,ic0,ic1)= dt*riemst*advecspeed(2)

           enddo
         enddo
      enddo

      if (visco.eq.1) then
      write(6,*) "doing artificial viscosity"
c
crtificial_viscosity1(0,1,2)c
crtificial_viscosity1(1,2,0)c
crtificial_viscosity1(2,0,1)c
      endif
c      call flush(6)     
      return
      end 
c***********************************************************************
c***********************************************************************
c***********************************************************************
  
      subroutine consdiff3d(
     &  ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
     &  dx,
     &  flux0,flux1,flux2,
     &  advecspeed,src,uval)
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
      integer ifirst0, ilast0,ifirst1, ilast1,ifirst2,ilast2
      double precision dx(0:3-1)
      double precision
     &     flux0(ifirst0-FLUXG:ilast0+1+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG),
     &     flux1(ifirst1-FLUXG:ilast1+1+FLUXG,
     &          ifirst2-FLUXG:ilast2+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG),
     &     flux2(ifirst2-FLUXG:ilast2+1+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG),
     &     advecspeed(0:3-1),src,
     &     uval(ifirst0-CELLG:ilast0+CELLG,
     &          ifirst1-CELLG:ilast1+CELLG,
     &          ifirst2-CELLG:ilast2+CELLG)
c
      integer ic0,ic1,ic2
      
c***********************************************************************
c update velocity to full time
c note the reversal of indices in 2nd coordinate direction
c***********************************************************************
c***********************************************************************
      do ic2=ifirst2,ilast2
         do ic1=ifirst1,ilast1
           do ic0=ifirst0,ilast0
             uval(ic0,ic1,ic2) = uval(ic0,ic1,ic2) +src
     &          -(flux0(ic0+1,ic1,ic2)-flux0(ic0,ic1,ic2))/dx(0)
     &          -(flux1(ic1+1,ic2,ic0)-flux1(ic1,ic2,ic0))/dx(1)
     &          -(flux2(ic2+1,ic0,ic1)-flux2(ic2,ic0,ic1))/dx(2)
           enddo
         enddo
      enddo
      return
      end
c***********************************************************************
