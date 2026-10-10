c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   F77 routine for compuation of stable dt in 3d.
c
c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   m4 include file for dimensioning 3d arrays in FORTRAN routines.
c

      subroutine stabledt3d(dx,
     &  ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
     &  ngc0,ngc1,ngc2,
     &  advecspeed,stabdt)
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      double precision stabdt,dx(0:3-1)
      integer ifirst0,ilast0,ifirst1,ilast1,ifirst2,ilast2,
     & ngc0,ngc1,ngc2
c
      double precision
     &  advecspeed(0:3-1)
c
      double precision maxspeed(0:3-1)
c
      maxspeed(0)=zero
      maxspeed(1)=zero
      maxspeed(2)=zero

      maxspeed(0) = max(maxspeed(0), abs(advecspeed(0)))
      maxspeed(1) = max(maxspeed(1), abs(advecspeed(1)))
      maxspeed(2) = max(maxspeed(2), abs(advecspeed(2)))

c     Do the following with checks for zero
c      stabdt = min((dx(1)/maxspeed(1)),(dx(0)/maxspeed(0)))
c      stabdt = min((dx(2)/maxspeed(2)),stabdt)

      if ( maxspeed(0) .EQ. 0.0 ) then
         if( maxspeed(1) .EQ. 0.0 ) then
            stabdt = 1.0E9
         else 
            stabdt = dx(1)/maxspeed(1)
         endif
      elseif ( maxspeed(1) .EQ. 0.0 ) then
            stabdt = dx(0)/maxspeed(0) 
      else
         stabdt = min((dx(1)/maxspeed(1)),(dx(0)/maxspeed(0)))
      endif

      if (maxspeed(2) .NE. 0.0 ) then
         stabdt = min((dx(2)/maxspeed(2)),stabdt)
      endif

      return
      end
