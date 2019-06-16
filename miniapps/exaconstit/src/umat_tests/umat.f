C
C     Format borrowed from another umat file.
C     The purpose of this file is just to print out the input and then
C     return updated variables for those expected to be updated.

      SUBROUTINE umat(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,
     1 STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     1 NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     1 CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
C
      IMPLICIT NONE

C     Parameter variables
      INTEGER NGRAIN
      PARAMETER (NGRAIN = 1)
      INTEGER NSYS
      PARAMETER (NSYS = 12)
      REAL*8 ZERO
      PARAMETER (ZERO = 0.D0)
C
C     Argument variables
      REAL*8 CMNAME
      INTEGER KINC, KSPT, KSTEP, LAYER, NDI, NOEL, NPROPS, NPT, NSHR
      INTEGER NSTATV, NTENS
      REAL*8 CELENT, COORDS(3), DDSDDE(NTENS,NTENS), DDSDDT(NTENS)
      REAL*8 DFGRD0(3,3), DFGRD1(3,3), DPRED(1), DROT(3,3)
      REAL*8 DRPLDE(NTENS), DRPLDT, DSTRAN(NTENS), DTEMP, DTIME, PNEWDT
      REAL*8 PREDEF(1), PROPS(NPROPS), RPL, SCD, SPD, SSE
      REAL*8 STATEV(NSTATV), STRAN(NTENS), STRESS(NTENS), TEMP, TIME(2)
C
C     File handler
      INTEGER un
C     Loop counters
      INTEGER i, j
      LOGICAL ex
      
      un = 14

C      inquire(file='umat_input.txt', EXIST=ex)
      
C      IF (.NOT. ex) THEN
C         OPEN(unit = un, FILE='umat_input.txt', STATUS='NEW')
C      ELSE
C         OPEN(unit = un, FILE='umat_input.txt', STATUS='OLD', ACCESS='APPEND')
C      ENDIF

C      write(un,*) 'NOEL = ', NOEL
C      write(un,*) 'NPT = ', NPT
C      write(un,*) 'KINC = ', KINC
C      write(un,*) 'KSTEP = ', KSTEP
C      write(un,*) 'KSPT = ', KSPT
C      write(un,*) 'LAYER = ', LAYER
C      write(un,*) 'DFGRAD0 = ', ((DFGRD0(I,J), J=1,3), I=1,3)
C      write(un,*) 'DFGRAD1 = ', ((DFGRD1(I,J), J=1,3), I=1,3)
C      write(un,*) 'CELENT = ', CELENT
C      write(un,*) 'DROT = ', ((DROT(I,J), J=1,3), I=1,3)
C      write(un,*) 'COORDS = ', (COORDS(I), I=1,3)
C      write(un,*) 'NPROPS = ', NPROPS
C      write(un,*) 'NSTATV = ', NSTATV
C      write(un,*) 'PROPS = ', (PROPS(I), I=1,NPROPS)
C      write(un,*) 'NTENS = ', NTENS
C      write(un,*) 'NSHR = ', NSHR
C      write(un,*) 'NDI = ', NDI
C      write(un,*) 'CMNAME = ', CMNAME
C      write(un,*) 'DPRED = ', DPRED
C      write(un,*) 'PREDEF = ', PREDEF
C      write(un,*) 'DTEMP = ', DTEMP
C      write(un,*) 'TEMP = ', TEMP
C      write(un,*) 'DTIME = ', DTIME
C      write(un,*) 'TIME = ', TIME
C      write(un,*) 'STRAN = ', (STRAN(I), I=1,NTENS)
C      write(un,*) 'DSTRAN = ', (DSTRAN(I), I=1,NTENS)
C      write(un,*) 'PNEWDT = ', PNEWDT
C      write(un,*) '%%%%%%%%%%%%%%%%'

C      CLOSE(un)

C      un = 14
C      inquire(file='umat_adapt.txt', EXIST=ex)
C      IF (.NOT. ex) THEN
C         OPEN(unit = un, FILE='umat_adapt.txt', STATUS='NEW')
C      ELSE
C         OPEN(unit = un, FILE='umat_adapt.txt', STATUS='OLD', ACCESS='APPEND')
C      ENDIF

C      write(un,*) 'DDSDDE = ', ((DDSDDE(I,J), J=1,NTENS), I=1,NTENS)
C      write(un,*) 'STRESS = ', (STRESS(I), I=1,NTENS)
C      write(un,*) 'STATEV = ', (STATEV(I), I=1,NSTATV)
C      write(un,*) 'SSE = ', SSE
C      write(un,*) 'SPD = ', SPD
C      write(un,*) 'SCD = ', SCD
C      write(un,*) 'RPL = ', RPL
C      write(un,*) 'DDSDDT = ', (DDSDDT(I), I=1,NTENS)
C      write(un,*) 'DRPLDE = ', DRPLDE
C      write(un,*) 'DRPLDT = ', DRPLDT
C      write(un,*) '%%%%%%%%%%%%%%%%'

C      CLOSE(un)

C  We are now changing the updated variables by incrementing their value by
C  1. Therefore, it should be fairly easy to check and see that our changes
C  were made.

      SSE = SSE + 1.0
      SPD = SPD + 1.0
      SCD = SCD + 1.0
      RPL = RPL + 1.0
      DRPLDE = DRPLDE + 1.0
      DRPLDT = DRPLDT + 1.0



      DO 10, i = 1, NTENS
         STRESS(i) = STRESS(i) + 1.0
         DDSDDT(i) = DDSDDT(i) + 1.0
         DO 20, j = 1, NTENS
            DDSDDE(i, j) = DDSDDE(i, j) + 1.0
   20    CONTINUE
   10 CONTINUE


      RETURN
      END
C

