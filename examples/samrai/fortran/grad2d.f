c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   F77 routines for gradient computation in 2d.
c
c
c This file is part of the SAMRAI distribution.  For full copyright
c information, see COPYRIGHT and LICENSE.
c
c Copyright:     (c) 1997-2025 Lawrence Livermore National Security, LLC
c Description:   m4 include file for dimensioning 2d arrays in FORTRAN routines.
c

      subroutine detectgrad2d(
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  vghost0,tagghost0,ttagghost0,
     &  vghost1,tagghost1,ttagghost1,
     &  dx,
     &  gradtol,
     &  dotag,donttag,
     &  var,
     &  tags,temptags)
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
c input arrays:
      integer
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  dotag,donttag,
     &  vghost0,vghost1,
     &  tagghost0,tagghost1,
     &  ttagghost0,ttagghost1
      double precision
     &  dx(0:2-1),
     &  gradtol
c variables indexed as 2dimensional
      double precision
     &  var(ifirst0-vghost0:ilast0+vghost0,
     &          ifirst1-vghost1:ilast1+vghost1)
      integer
     &  tags(ifirst0-tagghost0:ilast0+tagghost0,
     &          ifirst1-tagghost1:ilast1+tagghost1),
     &  temptags(ifirst0-ttagghost0:ilast0+ttagghost0,
     &          ifirst1-ttagghost1:ilast1+ttagghost1)
c
      double precision tol
      double precision facejump, loctol
      double precision presm1,presp1,diag01
      logical tagcell
      integer ic0,ic1
c
c***********************************************************************
c
      tol = gradtol
      diag01 = sqrt(dx(0)**2+dx(1)**2)

      do ic1=ifirst1,ilast1
        do ic0=ifirst0,ilast0

          if (tags(ic0,ic1) .ne. 0) then
            loctol = 0.125*tol
          else 
            loctol = tol
          endif
   
          tagcell = .false.

          presm1 = var(ic0-1,ic1)
          presp1 = var(ic0+1,ic1)
          facejump = abs(var(ic0,ic1)-presm1)
          facejump = max(facejump,abs(var(ic0,ic1)-presp1))
          tagcell = ((facejump).gt.(loctol*dx(0)))
          if (.not.tagcell) then
            presm1 = var(ic0,ic1-1)
            presp1 = var(ic0,ic1+1)
            facejump = abs(var(ic0,ic1)-presm1)
            facejump = max(facejump,abs(var(ic0,ic1)-presp1))
            tagcell = ((facejump).gt.(loctol*dx(1)))
          endif

          if (.not.tagcell) then
            presm1 = var(ic0-1,ic1-1)
            presp1 = var(ic0+1,ic1+1)
            facejump = abs(var(ic0,ic1)-presm1)
            facejump = max(facejump,abs(var(ic0,ic1)-presp1))
            tagcell = ((facejump).gt.(loctol*diag01))
          endif
          if (.not.tagcell) then
            presm1 = var(ic0-1,ic1+1)
            presp1 = var(ic0+1,ic1-1)
            facejump = abs(var(ic0,ic1)-presm1)
            facejump = max(facejump,abs(var(ic0,ic1)-presp1))
            tagcell = ((facejump).gt.(loctol*diag01))
          endif

          if ( tagcell ) then
            temptags(ic0,ic1) = dotag
          endif
        enddo
      enddo
      return
      end

      subroutine detectshock2d(
     &  ifirst0,ilast0,ifirst1,ilast1,
     &  vghost0,tagghost0,ttagghost0,
     &  vghost1,tagghost1,ttagghost1,
     &  dx,
     &  gradtol,gradonset,
     &  dotag,donttag,
     &  var,
     &  tags,temptags)
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
c input arrays:
      integer
     &  ifirst0,ifirst1,ilast0,ilast1,
     &  dotag,donttag,
     &  vghost0,vghost1,
     &  tagghost0,tagghost1,
     &  ttagghost0,ttagghost1
      double precision
     &  dx(0:2-1),
     &  gradtol,gradonset
c variables indexed as 2dimensional
      double precision
     &  var(ifirst0-vghost0:ilast0+vghost0,
     &          ifirst1-vghost1:ilast1+vghost1)
      integer
     &  tags(ifirst0-tagghost0:ilast0+tagghost0,
     &          ifirst1-tagghost1:ilast1+tagghost1),
     &  temptags(ifirst0-ttagghost0:ilast0+ttagghost0,
     &          ifirst1-ttagghost1:ilast1+ttagghost1)
c
      double precision tol,onset
      double precision jump1, jump2, facejump, loctol,locon
      double precision presm1,presm2,presp1,presp2
      double precision diag01 
      logical tagcell
      integer ic0,ic1
c
c***********************************************************************
c
      tol = gradtol
      onset = gradonset
      diag01 = sqrt(dx(0)**2+dx(1)**2)

      do ic1=ifirst1,ilast1
        do ic0=ifirst0,ilast0

          if (tags(ic0,ic1) .ne. 0) then
            loctol = 0.125*tol
            locon = 0.66*onset
          else 
            loctol = tol
            locon = onset
          endif
   
          tagcell = .false.

          presm1 = var(ic0-1,ic1)
          presm2 = var(ic0-2,ic1)
          presp1 = var(ic0+1,ic1)
          presp2 = var(ic0+2,ic1)
          jump2 = presp2-presm2
          jump1 = presp1-presm1
          facejump = abs(var(ic0,ic1)-presm1)
          facejump = max(facejump,abs(var(ic0,ic1)-presp1))
          tagcell = ((((abs(jump2)*locon).le.abs(jump1)).or.
     &                ((jump1*jump2).lt.zero)).and.
     &               ((facejump).gt.(loctol*dx(0))))
          if (.not.tagcell) then
            presm1 = var(ic0,ic1-1)
            presm2 = var(ic0,ic1-2)
            presp1 = var(ic0,ic1+1)
            presp2 = var(ic0,ic1+2)
            jump2 = presp2-presm2
            jump1 = presp1-presm1
            facejump = abs(var(ic0,ic1)-presm1)
            facejump = max(facejump,abs(var(ic0,ic1)-presp1))
            tagcell = ((((abs(jump2)*locon).le.abs(jump1)).or.
     &                ((jump1*jump2).lt.zero)).and.
     &               ((facejump).gt.(loctol*dx(1)))) 
          endif

          if (.not.tagcell) then
            presm1 = var(ic0-1,ic1-1)
            presp1 = var(ic0+1,ic1+1)
            presm2 = var(ic0-2,ic1-2)
            presp2 = var(ic0+2,ic1+2)
            jump1 = presp1-presm1
            jump2 = presp2-presm2
            facejump = abs(var(ic0,ic1)-presm1)
            facejump = max(facejump,abs(var(ic0,ic1)-presp1))
            tagcell = ((((abs(jump2)*locon).le.abs(jump1)).or.
     &                  ((jump1*jump2).lt.zero)).and.
     &                 ((facejump).gt.(loctol*diag01)))
          endif
          if (.not.tagcell) then
            presm1 = var(ic0-1,ic1+1)
            presp1 = var(ic0+1,ic1-1)
            presm2 = var(ic0-2,ic1+2)
            presp2 = var(ic0+2,ic1-2)
            jump1 = presp1-presm1
            jump2 = presp2-presm2
            facejump = abs(var(ic0,ic1)-presm1)
            facejump = max(facejump,abs(var(ic0,ic1)-presp1))
            tagcell = ((((abs(jump2)*locon).le.abs(jump1)).or.
     &                  ((jump1*jump2).lt.zero)).and.
     &                 ((facejump).gt.(loctol*diag01)))
          endif

          if ( tagcell ) then
            temptags(ic0,ic1) = dotag
          endif
        enddo
      enddo
      return
      end


