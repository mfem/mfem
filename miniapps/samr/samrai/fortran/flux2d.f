c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   F77 routines for flux computation in 2d.
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
c Description:   m4 include file for 2d flux calculation.
c

      subroutine fluxcorrec(dt,
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  dx,
     &  advecspeed,
     &  flux0,flux1,
     &  trlft0,trlft1,
     &  trrgt0,trrgt1)
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
      double precision dt 
c variables in 1d axis indexed
c
      double precision 
     &     dx(0:2-1)
c variables in 2d cell indexed         
      double precision
     &     advecspeed(0:2-1),
     &     flux0(ifirst0-FLUXG:ilast0+1+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG),
     &     flux1(ifirst1-FLUXG:ilast1+1+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG), 
     &     trlft0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     trrgt0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     trlft1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     trrgt1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG)
c
c***********************************************************************     
c
      integer ic0,ic1
      double precision trnsvers
     
c     write(6,*) "In fluxcorrec()"
c     ******************************************************************
c     * complete tracing at cell edges
c     ******************************************************************
c   correct the 1-direction with 0-fluxes
c     write(6,*) " correct the 1-direction with 0-fluxes"
      do ic1=ifirst1-1,ilast1+1
        do ic0=ifirst0-1,ilast0+1
          trnsvers= (flux0(ic0+1,ic1)-flux0(ic0,ic1))*0.5/dx(0)

          trrgt1(ic1  ,ic0)= trrgt1(ic1  ,ic0) - trnsvers
          trlft1(ic1+1,ic0)= trlft1(ic1+1,ic0) - trnsvers
        enddo
      enddo
c     call flush(6)     

c   correct the 0-direction with 1-fluxes
c     write(6,*) " correct the 0-direction with 1-fluxes"
      do ic0=ifirst0-1,ilast0+1
        do ic1=ifirst1-1,ilast1+1
          trnsvers= (flux1(ic1+1,ic0)-flux1(ic1,ic0))*0.5/dx(1)
          trrgt0(ic0  ,ic1)= trrgt0(ic0  ,ic1) - trnsvers
          trlft0(ic0+1,ic1)= trlft0(ic0+1,ic1) - trnsvers
        enddo
      enddo
c     call flush(6)     
      return
      end   
c
c***********************************************************************
c***********************************************************************
c***********************************************************************
      subroutine fluxcalculation2d(dt,extra_cell,visco,dx,
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  advecspeed,
     &  flux0,flux1,
     &  trlft0,trlft1,trrgt0,trrgt1)
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
c input arrays:
      integer ifirst0,ilast0,ifirst1,ilast1,extra_cell,visco
      double precision dt 
      double precision 
     &     dx(0:2-1)
c variables in 2d cell indexed         
      double precision
     &     advecspeed(0:2-1),
c variables in 2d side indexed         
     &     flux0(ifirst0-FLUXG:ilast0+1+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG),
     &     flux1(ifirst1-FLUXG:ilast1+1+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG), 
     &     trlft0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     trrgt0(ifirst0-FACEG:ilast0+1+FACEG,
     &          ifirst1-FACEG:ilast1+FACEG),
     &     trlft1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG),
     &     trrgt1(ifirst1-FACEG:ilast1+1+FACEG,
     &          ifirst0-FACEG:ilast0+FACEG)
c
c***********************************************************************     
c
      integer ic0,ic1,ie0,ie1
      double precision   riemst
c
c***********************************************************************
c solve riemann problems for conservative flux
c  arguments: ( axis for RP, other axis, extra cells-direction)
c***********************************************************************
c      
c     write(6,*) "In fluxcalculation2d(",extra_cell,")"
c     write(6,*) "ifirst0,ilast0,ifirst1,ilast1,extra_cell",
c    &       ifirst0,ilast0,ifirst1,ilast1,extra_cell

c     write(6,*) "checking onedr sol in riemann solve  "
c     write(6,*) "         dt= ",dt
c     write(6,*) "  calculating flux0, 1+extra_cell= ",extra_cell
c     write(6,*) "  ic1=",ifirst1-1-extra_cell,ilast1+1+extra_cell
c     write(6,*) "  ie0=",ifirst0-1-extra_cell,ilast0+1+1+extra_cell
        do ic1=ifirst1-extra_cell,ilast1+extra_cell
          do ie0=ifirst0-extra_cell,ilast0+1+extra_cell
 
            if (advecspeed(0).ge.zero) then
               riemst= trlft0(ie0,ic1)
             else
               riemst= trrgt0(ie0,ic1)
             endif
 
            flux0(ie0,ic1)= dt*riemst*advecspeed(0)
c           write(6,*) "   flux0(",ie0,ic1,")= ",flux0(ie0,ic1,1),
c    &                   flux0(ie0,ic1,2),
c    &          flux0(ie0,ic1,3),flux0(ie0,ic1,4)
          enddo
        enddo
c
c     write(6,*) "checking onedr sol in riemann solve  "
c     write(6,*) "         dt= ",dt
c     write(6,*) "  calculating flux1, 1+extra_cell= ",extra_cell
c     write(6,*) "  ic0=",ifirst0-1-extra_cell,ilast0+1+extra_cell
c     write(6,*) "  ie1=",ifirst1-1-extra_cell,ilast1+1+1+extra_cell
        do ic0=ifirst0-extra_cell,ilast0+extra_cell
          do ie1=ifirst1-extra_cell,ilast1+1+extra_cell
 
            if (advecspeed(1).ge.zero) then
               riemst= trlft1(ie1,ic0)
             else
               riemst= trrgt1(ie1,ic0)
             endif
 
            flux1(ie1,ic0)= dt*riemst*advecspeed(1)
c           write(6,*) "   flux1(",ie1,ic0,")= ",flux1(ie1,ic0,1),
c    &                   flux1(ie1,ic0,2),
c    &          flux1(ie1,ic0,3),flux1(ie1,ic0,4)
          enddo
        enddo

      if (visco.eq.1) then
      write(6,*) "doing artificial viscosity"
c
crtificial_viscosity1(0,1)c
crtificial_viscosity1(1,0)c
      endif
c     call flush(6)     
      return
      end 
c***********************************************************************
c***********************************************************************
c***********************************************************************
  
      subroutine consdiff2d(
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  dx,
     &  flux0,flux1,
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
      integer ifirst0, ilast0,ifirst1, ilast1
      double precision dx(0:2-1)
      double precision
     &     flux0(ifirst0-FLUXG:ilast0+1+FLUXG,
     &          ifirst1-FLUXG:ilast1+FLUXG),
     &     flux1(ifirst1-FLUXG:ilast1+1+FLUXG,
     &          ifirst0-FLUXG:ilast0+FLUXG),
     &     advecspeed(0:2-1),src,
     &     uval(ifirst0-CELLG:ilast0+CELLG,
     &          ifirst1-CELLG:ilast1+CELLG)
c
      integer ic0,ic1
      
c***********************************************************************
c update velocity to full time
c note the reversal of indices in 2nd coordinate direction
c***********************************************************************
c***********************************************************************
c     write(6,*) "at top of consdiff2d"
c         call flush(6)
c     write(6,*) "flux0"
c     do  ic1=ifirst1,ilast1
c       do  ic0=ifirst0,ilast0+1
c         write(6,*) "ic0,flux0= ",ic0,ic1,
c    &                  flux0(ic0,ic1,1),flux0(ic0,ic1,2),
c    &                  flux0(ic0,ic1,3),flux0(ic0,ic1,4)
c         call flush(6)
c       enddo
c     enddo
c     write(6,*) "flux1"
c     do  ic1=ifirst1,ilast1
c       do  ic0=ifirst0,ilast0+1
c         write(6,*) "ic0,flux1= ",ic0,ic1,
c    &                  flux1(ic0,ic1,1),flux1(ic0,ic1,2),
c    &                  flux1(ic0,ic1,3),flux1(ic0,ic1,4)
c         call flush(6)
c       enddo
c     enddo
c     do  ic1=ifirst1,ilast1
c       do  ic0=ifirst0,ilast0
c         write(6,*) "ic0,ic1,all = ",ic0,ic1,
c    &                        density(ic0,ic1),
c    &                        velocity(ic0,ic1,1),velocity(ic0,ic1,2),
c    &                        pressure(ic0,ic1)
c         call flush(6)
c       enddo
c     enddo
c***********************************************************************
      do ic1=ifirst1,ilast1
        do ic0=ifirst0,ilast0
          uval(ic0,ic1) = uval(ic0,ic1) +src
     &          -(flux0(ic0+1,ic1)-flux0(ic0,ic1))/dx(0)
     &          -(flux1(ic1+1,ic0)-flux1(ic1,ic0))/dx(1)
        enddo
      enddo
c***********************************************************************
c     write(6,*) "in consdiff2d"
c     do  ic1=ifirst1,ilast1+1
c       do  ic0=ifirst0,ilast0+1
c         write(6,*) "ic0,ic1,flux0= ",ic0,ic1,
c    &                        flux0(ic0,ic1,1),flux0(ic0,ic1,2),
c    &                        flux0(ic0,ic1,3),flux0(ic0,ic1,4) 
c         call flush(6)
c       enddo
c     enddo
c     do  ic1=ifirst1,ilast1+1
c       do  ic0=ifirst0,ilast0+1
c         write(6,*) "ic0,ic1,flux1= ",ic0,ic1,
c    &                        flux1(ic0,ic1,1),flux1(ic0,ic1,2),
c    &                        flux1(ic0,ic1,3),flux1(ic0,ic1,4) 
c         call flush(6)
c       enddo
c     enddo
c***********************************************************************
c***********************************************************************
c     do  ic1=ifirst1,ilast1
c       do  ic0=ifirst0,ilast0
c         write(6,*) "ic0,ic1,all = ",ic0,ic1,
c    &                        density(ic0,ic1),
c    &                        velocity(ic0,ic1,1),velocity(ic0,ic1,2),
c    &                        momentum(ic0,ic1,1),momentum(ic0,ic1,2),
c    &                        pressure(ic0,ic1),
c    &                        energy(ic0,ic1)
c         call flush(6)
c       enddo
c     enddo
      return
      end
