c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   F77 routines for initialization in 2d.
c
c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   m4 include file for dimensioning 2d arrays in FORTRAN routines.
c

      subroutine linadvinit2d(data_problem,dx,xlo,xhi,
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  gcw0,gcw1,
     &  uval,
     &  nintervals,front,
     &  i_uval)
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
      integer gcw0,gcw1
      integer data_problem
      integer nintervals
      double precision front(1:nintervals)
      double precision 
     &     dx(0:2-1),xlo(0:2-1),xhi(0:2-1)
      double precision
     &     i_uval(1:nintervals)
      double precision
     &     uval(ifirst0-gcw0:ilast0+gcw0,
     &          ifirst1-gcw1:ilast1+gcw1)
c
c***********************************************************************     
c
      integer ic0,ic1,dir,ifr
      double precision xc(0:2-1)
c
c   dir 0 two linear states (L,R) indp of y,z
c   dir 1 two linear states (L,R) indp of x,z

c     write(6,*) "Inside initplane"
c     write(6,*) "data_problem= ",data_problem
c     write(6,*) "dx= ",dx(0), dx(1)
c     write(6,*) "xlo= ",xlo(0), xlo(1),", xhi = ",xhi(0), xhi(1)
c     write(6,*) "ifirst, ilast= ",ifirst0,ilast0,"and ",ifirst1,ilast1
c     call flush(6)

      dir = 0
      if (data_problem.eq.PIECEWISE_CONSTANT_X) then
         dir = 0
      else if (data_problem.eq.PIECEWISE_CONSTANT_Y) then
         dir = 1
      endif

      if (dir.eq.0) then
         ifr = 1
         do ic0=ifirst0,ilast0
            xc(0) = xlo(0) + dx(0)*(dble(ic0-ifirst0)+half)
            if (xc(dir).gt.front(ifr)) then
              ifr = ifr+1
            endif
            do ic1=ifirst1,ilast1
               uval(ic0,ic1) = i_uval(ifr)
           enddo
         enddo
      else if (dir.eq.1) then
         ifr = 1
         do ic1=ifirst1,ilast1
            xc(1) =xlo(1)+ dx(1)*(dble(ic1-ifirst1)+half)
            if (xc(dir).gt.front(ifr)) then
               ifr = ifr+1
            endif
            do ic0=ifirst0,ilast0
               uval(ic0,ic1) = i_uval(ifr)
           enddo
         enddo
      endif
c
      return
      end

c***********************************************************************
c
c    Initialization routine where we use a spherical profile 
c
c***********************************************************************
      subroutine initsphere2d(data_problem,dx,xlo,xhi,
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  gcw0,gcw1,
     &  uval,
     &  i_uval,o_uval,
     &  center,radius)
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
c***********************************************************************
c***********************************************************************     
c input arrays:
      integer ifirst0,ilast0,ifirst1,ilast1
      integer gcw0,gcw1
      integer data_problem
      double precision i_uval,o_uval
      double precision radius,center(0:2-1)
      double precision 
     &     dx(0:2-1),xlo(0:2-1),xhi(0:2-1)
c variables in 2d cell indexed         
      double precision
     &     uval(ifirst0-gcw0:ilast0+gcw0,
     &          ifirst1-gcw1:ilast1+gcw1)
c
c***********************************************************************     
c
      integer ic0,ic1
      double precision xc(0:2-1),x0,x1
      double precision angle

c     write(6,*) "in initsphere"
c     write(6,*) "data_problem= ",data_problem
c     write(6,*) "dx = ",(dx(i),i=0,2-1)
c     write(6,*) "xlo = ",(xlo(i),i=0,2-1)
c     write(6,*) "xhi = ",(xhi(i),i=0,2-1)
c     write(6,*) "ce = ",(ce(i),i=0,2-1)
c     write(6,*) "radius = ",radius
c     write(6,*) "ifirst0,ilast0 = ",ifirst0,ilast0
c     write(6,*) "ifirst1,ilast1 = ",ifirst1,ilast1
c

      do ic1=ifirst1,ilast1
        xc(1) = xlo(1)+dx(1)*(dble(ic1-ifirst1)+half)
        x1 = xc(1)-center(1)
        do ic0=ifirst0,ilast0
           xc(0) = xlo(0)+dx(0)*(dble(ic0-ifirst0)+half)
           x0 = xc(0)-center(0)
           if (x1.eq.zero .and. x0.eq.zero) then
              angle = zero
           else
              angle = atan2(x1,x0)
           endif
           if ((x0**2+x1**2).lt.radius**2) then
              uval(ic0,ic1) = i_uval
           else
              uval(ic0,ic1) = o_uval
           endif
         enddo
      enddo
c
      return
      end

c***********************************************************************
c
c   Sine profile
c
c***********************************************************************
 
      subroutine linadvinitsine2d(data_problem,dx,xlo,
     &  domain_xlo, domain_length,
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  gcw0,gcw1,
     &  uval,
     &  nintervals,front,
     &  i_uval,
     &  amplitude,frequency)
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
      integer gcw0,gcw1
      integer data_problem
      integer nintervals
      double precision
     &     dx(0:2-1),xlo(0:2-1),
     &     domain_xlo(0:2-1),domain_length(0:2-1)
      double precision front(1:nintervals)
      double precision i_uval(1:nintervals)
      double precision amplitude,frequency(0:2-1)
c variables in 2d cell indexed
      double precision
     &     uval(ifirst0-gcw0:ilast0+gcw0,
     &          ifirst1-gcw1:ilast1+gcw1)
c
c***********************************************************************
c
      integer ic0,ic1,j,ifr
      double precision xc(0:2-1),xmid(1:10)
      double precision coef(0:2-1),coscoef(1:2)
c
c     write(6,*) "Inside eulerinitsine"
c     write(6,*) "data_problem= ",data_problem
c     write(6,*) "dx= ",dx(0), dx(1)
c     write(6,*) "mc= ",mc(0), mc(1)
c     write(6,*) "xlo= ",xlo(0), xlo(1),", xhi = ",xhi(0), xhi(1)
c     write(6,*) "ifirst, ilast= ",ifirst0,ilast0,ifirst1,ilast1
c     write(6,*) "gamma= ",gamma
c     call flush(6)
 
      if (data_problem.eq.SINE_CONSTANT_Y) then
         write(6,*) "Sorry, Y direction not implemented :-("
         return
      endif
 
      coef(0) = zero
      do j=1,2-1
        coef(j) = two*pi*frequency(j)/domain_length(j)
      enddo
 
      do ic1=ifirst1,ilast1
        xc(1) = xlo(1)+dx(1)*(dble(ic1-ifirst1)+half)
        coscoef(1) = amplitude*
     &         cos((xc(1)-domain_xlo(1))*coef(1))
        do j=1,(nintervals-1)
           xmid(j) = front(j) + coscoef(1)
        enddo
        do ic0=ifirst0,ilast0
           xc(0) = xlo(0)+dx(0)*(dble(ic0-ifirst0)+half)
           ifr = 1
           do j=1,(nintervals-1)
              if( xc(0) .gt. xmid(j) ) ifr = j+1
           enddo
           uval(ic0,ic1) = i_uval(ifr)
        enddo
      enddo
 
      return
      end
