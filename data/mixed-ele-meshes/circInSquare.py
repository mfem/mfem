#Title:circInSquare.py
#Author:T. M. McManus
#Date:10-7-18
#Purpose: Fill a circular sector with triangles and a bounding region,
#defined by 3 nodes, with quads.  Then reflect/preserve QuadI twice to
#create a complete disc bounded in a square.

import scipy as sp
import argparse
import sys
import subprocess
import time

parser=argparse.ArgumentParser(description='Fill a circular sector with triangles and a bounding region,\
defined by 3 nodes, with quads.  Then reflect/preserve QuadI twice to create a complete disc bounded in a square.'
,epilog='Sample run: python circInSquare.py -r 1 -e 2 -n 8 -g ../../../glvis/glvis')

parser.add_argument('-r','--circRad', nargs='?',const=1, default = 1.0, type=float, help='Radius of circle')
parser.add_argument('-e','--edgeLength', nargs='?',const=1,default=2.0,type=float,help='Edge-length of bounding square')
parser.add_argument('-n','--numEdges',nargs='?',const=1,default=6,type=int,help='n-gon approximation of internal circle')
parser.add_argument('-o','--outputFile',nargs='?',const=1,default='circInSquare', help='Output file name.')
parser.add_argument('-g','--glvis',nargs='?',const=1,default='',type=str,help='Abs. or rel. path of glvis binary.')
args=parser.parse_args()

r=args.circRad 
edgeLength=args.edgeLength
numEdges=args.numEdges
outputName=args.outputFile
glvis=args.glvis

visMesh=False;

if glvis!='':
    visMesh=True

if r >= edgeLength:
    print("Circle radius must be less than bounding square edge length")
    sys.exit(1)

if sp.mod(numEdges,2) != 0:
    print("Currently this mixed element generator only supports an even numbers of edges.")
    sys.exit(1)

    
#The basic idea:
#1. Construct topology for regions
#2. Combine topologies
#3. Construct boundary
#4. Construct geometry for regions
#5. Combine geometries
#6. Output

def eleMatCirc(numEdges):
    
    nNodesSeq=sp.zeros([numEdges])
    nNodesSeq[0]=3
    if numEdges != 1:
        for n in range(1,numEdges):
            nNodesSeq[n]=nNodesSeq[n-1]+(2+n)
            
    numCircNodesTot =int(((numEdges+1)*(numEdges+2))/2)
    
    b=range(numCircNodesTot)
    row_size=1
    A=sp.zeros([numEdges+1,numEdges+1])
    start=0;stop=1;
    for m in range(numEdges+1):
        if m==0:
            A[m,range(m+1)]=b[0:1]
            start=0
            stop=1
        else:
            start=stop
            stop=stop+m+1
            A[m,range(m+1)]=b[start:stop]

    M=sp.ones([numEdges**2,5])
    m_row=0
    for m in range(numEdges):
        if m==0:
            M[0,:]=[1,2,0,1,2]
            m_row+=1
        else:
           holder=sp.size(sp.nonzero(A[m,:]))
           for n in range(holder):
               if n!=holder-1:
                   M[m_row,:]=[1,2,A[m,n],A[m,n+1],A[m+1,n+1]]
                   m_row+=1
                   M[m_row,:]=[1,2,A[m,n],A[m+1,n],A[m+1,n+1]]
                   m_row+=1
               else:
                   M[m_row,:]=[1,2,A[m,n],A[m+1,n],A[m+1,n+1]]
                   m_row+=1
            
    return M.astype(int),numCircNodesTot

def eleMatQuad(numEdges):
    S0=numEdges*(numEdges+1)/(2.0)
    A=sp.linspace(S0,(S0+(numEdges+1)**2)-1,(numEdges+1)**2)
    A=A.reshape([numEdges+1,numEdges+1])
    quadNode=sp.delete(A,-1,1)
    quadNode=sp.delete(quadNode,-1,0)
    quadNode=quadNode.flatten()
    M=sp.zeros([numEdges**2,6])
    for n in range(numEdges**2):
        M[n,:]=[2,3,quadNode[n],quadNode[n]+1,quadNode[n]+numEdges+2,quadNode[n]+numEdges+1]
    return M.astype(int)

def boundMatTot(numEdges):
    triS1=sp.zeros(numEdges+1)
    triS3=sp.zeros(numEdges+1)
    quadS1=sp.zeros(numEdges)
    quadS2=sp.zeros(numEdges-1)
    quadS3=sp.zeros(numEdges)

    triS1[0]=0;
    triS3[0]=0;
    for n in range(1,numEdges+1):
        triS1[n]=triS1[n-1]+n
        triS3[n]=triS1[n]+n
    ref1=triS3
    
    triS3=sp.flipud(triS3)
    quadS1[0]=triS1[-1]+numEdges+1
    quadS3[0]=triS1[-1]+2*numEdges+1

    for n in range(1,numEdges):
        quadS1[n]=quadS1[n-1]+(numEdges+1)
        quadS3[n]=quadS3[n-1]+(numEdges+1)    
    ref2=quadS3
    xAxisRootRef=sp.concatenate([triS1.copy(),quadS1],axis=0)
    quadS3=sp.flipud(quadS3)
    quadS2=range(int(quadS1[-1]+1),int(quadS3[0]),1)
    STOT=sp.concatenate([triS1,quadS1,quadS2,quadS3,triS3],axis=0)

    filler=sp.zeros(1)
    filler[0]=quadS3[0]
    fillerFirst=sp.zeros(1)
    fillerFirst[0]=quadS1[-1]
    sTotRef=sp.concatenate([triS1,quadS1,quadS2,filler],axis=0)
    newsTotRef=sp.concatenate([fillerFirst,quadS2,filler],axis=0)
    boundMat=sp.zeros([STOT.size-1,4])
    boundMatRef=sp.zeros([sTotRef.size-1,4])
    new_boundMat_ref=sp.zeros([newsTotRef.size-1,4])
    
    for n in range(STOT.size-1):
        boundMat[n,:]=[1,1,STOT[n],STOT[n+1]]
    for n in range(sTotRef.size-1):
        boundMatRef[n,:]=[1,1,sTotRef[n],sTotRef[n+1]]
    for n in range(newsTotRef.size-1):
        new_boundMat_ref[n,:]=[1,1,newsTotRef[n],newsTotRef[n+1]]
        
    ref=sp.concatenate([ref1,ref2],axis=0).astype(int)
    return boundMat.astype(int),ref,boundMatRef.astype(int),xAxisRootRef.astype(int),new_boundMat_ref.astype(int)

def vertMatCirc(numEdges):
    r_o=sp.linspace(0,r,numEdges+1)
    counter=0
    vertMat=sp.zeros([numCircNodesTot,2])
    for m in range(numEdges+1):
        theta=sp.linspace(0,sp.pi/2.0,m+1)
        for n in range(sp.size(theta)):
            vertMat[counter,:]=[r_o[m]*sp.cos(theta[n]),r_o[m]*sp.sin(theta[n])]
            counter+=1
    return vertMat

def vertMatQuad(numEdges):

    theta=sp.linspace(0,sp.pi/2.0,numEdges+1)
    AX=sp.zeros([numEdges+1,numEdges+1])
    AY=sp.zeros([numEdges+1,numEdges+1])
    AX[0,:]=r*sp.cos(theta)
    AY[0,:]=r*sp.sin(theta)

    vertLinSpace=sp.linspace(0,edgeLength,(numEdges/2)+1)
    horzLineSpace=sp.linspace(edgeLength,0,(numEdges/2)+1)

    #Assigning node locations along the boundary
    vertCount=0
    horzCount=1
    for n in range(numEdges+1):
        if n < (numEdges/2):
            AX[-1,n]=edgeLength
            AY[-1,n]=vertLinSpace[vertCount]
            vertCount+=1
        elif n == int(numEdges/2):
            AX[-1,n]=edgeLength
            AY[-1,n]=edgeLength
        else:
            AX[-1,n]=horzLineSpace[horzCount]
            AY[-1,n]=edgeLength
            horzCount+=1

    #Linearly spacing nodes between the inner/outer boundaries
    #One could then smooth this via r-based adaptivity
    for col in range(numEdges+1):
        for row in range(1,numEdges):
            AX[row,col]=sp.linspace(AX[0,col],AX[-1,col],numEdges+1)[row]
            AY[row,col]=sp.linspace(AY[0,col],AY[-1,col],numEdges+1)[row]

    AX=sp.delete(AX,0,0)
    AY=sp.delete(AY,0,0)
    AX=AX.flatten()
    AY=AY.flatten()
    AX_reshape = AX.flatten()
    
    numQuadNodesTot=numEdges*(numEdges+1)
    vertMat=sp.zeros([numQuadNodesTot,2])
    for n in range(numQuadNodesTot):
        vertMat[n,:]=[AX[n],AY[n]]    
    return vertMat

def orient(A):
    aOrient=sp.zeros([A.shape[0],A.shape[1]])
    triCounter=0
    quadCounter=0
    #Determine the number of triangle and quad elments in the given element matrix
    for n in range(A.shape[0]):
        if A[n,1]==2:
            triCounter+=1
        else:
            quadCounter+=1
    edgeMatTotal=sp.zeros([3*triCounter+4*quadCounter,2])
    counter=0
    for n in range(A.shape[0]):
        detected=0
        if A[n,1]==2:
            for m in range(edgeMatTotal.shape[0]):
                if detected != 1:
                    if edgeMatTotal[m,0]==A[n,2] and edgeMatTotal[m,1]==A[n,3]:
                        aOrient[n,:]=[1,2,A[n,2],A[n,4],A[n,3],0]
                        detected=1
                        #print("reorder:[{} {} {}] to [{} {} {}]".format(A[n,2],A[n,3],A[n,4],int(aOrient[n,2]),int(aOrient[n,3]),int(aOrient[n,4])))
                    elif edgeMatTotal[m,0]==A[n,4] and edgeMatTotal[m,1]==A[n,2]:
                        aOrient[n,:]=[1,2,A[n,2],A[n,4],A[n,3],0]
                        detected=1
                    else:
                        aOrient[n,:]=A[n,:]

            edgeMatTotal[counter,:]=[aOrient[n,2],aOrient[n,3]]
            counter+=1
            edgeMatTotal[counter,:]=[aOrient[n,3],aOrient[n,4]]
            counter+=1
            edgeMatTotal[counter,:]=[aOrient[n,4],aOrient[n,2]]
            counter+=1
        else:
            for m in range(edgeMatTotal.shape[0]):
                if detected != 1: 
                    if edgeMatTotal[m,0]==A[n,2] and edgeMatTotal[m,1]==A[n,3]:
                        aOrient[n,:]=[2,3,A[n,2],A[n,5],A[n,4],A[n,3]]
                        detected=1
                        #print("reorder:[{} {} {} {}] to [{} {} {} {}]".format(A[n,2],A[n,3],A[n,4],A[n,5],int(aOrient[n,2]),int(aOrient[n,3]),int(aOrient[n,4]),int(aOrient[n,5])))
                    elif edgeMatTotal[m,0]==A[n,5] and edgeMatTotal[m,1]==A[n,2]:
                        aOrient[n,:]=[2,3,A[n,2],A[n,5],A[n,4],A[n,3]]
                        detected=1
                    else:
                        aOrient[n,:]=A[n,:]
            edgeMatTotal[counter,:]=[aOrient[n,2],aOrient[n,3]]
            counter+=1
            edgeMatTotal[counter,:]=[aOrient[n,3],aOrient[n,4]]
            counter+=1
            edgeMatTotal[counter,:]=[aOrient[n,4],aOrient[n,5]]
            counter+=1
            edgeMatTotal[counter,:]=[aOrient[n,5],aOrient[n,2]]
            counter+=1
    return aOrient.astype(int)

def gVis(_glvis,_meshFile):

    if(_glvis==''):
        print("Failure: Set glvis location via -g switch")
        sys.exit(1)

    colFuncFileName=_meshFile.replace('.mesh','.gf')
    glvsScriptFileName=_meshFile.replace('.mesh','.glvs')
    imageFileName=_meshFile.replace('.mesh','.png')
    
    #Create Coloring Function for mesh
    _colFuncCommand=_glvis+ ' -m '+ _meshFile +' -sc -k q'
    args=_colFuncCommand.split()
    p=subprocess.Popen(args)#Create 'GLVis_coloring.gf'

    _renameCommand='mv GLVis_coloring.gf {}'.format(colFuncFileName)
    args=_renameCommand.split()
    p=subprocess.Popen(args)

    #Glvis script template
    f=open(glvsScriptFileName,'w')
    f.write('window 0 0 800 800\n'+'\n')
    f.write('solution {} {}\n'.format(_meshFile,colFuncFileName)+'\n')
    f.write('{\n'+'perspective off\n'+'zoom 1.5\n'+'keys gAeeRM\n'+'solution {} {} screenshot {}\n'.format(_meshFile,colFuncFileName,imageFileName)+'keys q\n'+'}\n')
    f.close()

    _runGlvisCommand=_glvis+' -run {}'.format(glvsScriptFileName)
    args=_runGlvisCommand.split()
    p=subprocess.Popen(args)
    p.wait()
    
    return 0
def quadInterDof(_edge,_linEleMat,_linVertMatRound):
    _state=False
    for n in range(_linEleMat.shape[0]):
        if _linEleMat[n,1]==3:
            if sp.any(_edge[0]==_linEleMat[n,2:6]) and sp.any(_edge[1]==_linEleMat[n,2:6]):
                print("{} is possibly in {}".format(_edge,_linEleMat[n,2:6]))
                _n1Loc=sp.where(_edge[0]==_linEleMat[n,2:6])[0][0]
                _n2Loc=sp.where(_edge[1]==_linEleMat[n,2:6])[0][0]
                if _n1Loc==sp.mod(_n2Loc+1,4) or _n1Loc==sp.mod(_n2Loc-1,4):
                    _state=True
                    xcent=(_linVertMatRound[_linEleMat[n,2],0]+_linVertMatRound[_linEleMat[n,3],0]+_linVertMatRound[_linEleMat[n,4],0]+_linVertMatRound[_linEleMat[n,5],0])/4.0
                    ycent=(_linVertMatRound[_linEleMat[n,2],1]+_linVertMatRound[_linEleMat[n,3],1]+_linVertMatRound[_linEleMat[n,4],1]+_linVertMatRound[_linEleMat[n,5],1])/4.0
                    _interDof=sp.zeros(2)
                    _interDof[0]=sp.round_((_linVertMatRound[_edge[0],0]+_linVertMatRound[_edge[1],0]+xcent)/3.0,5)
                    _interDof[1]=sp.round_((_linVertMatRound[_edge[0],1]+_linVertMatRound[_edge[1],1]+ycent)/3.0,5)
                    print("dof loc is {},{}".format(_interDof[0],_interDof[1]))
                    return(_state,_interDof[0],_interDof[1])

    return(_state,0,0)

[eleMatTriHolder,numCircNodesTot]=eleMatCirc(numEdges) #Construct tri element matrix for the region inside circular sector
eleMatQuadHolder=eleMatQuad(numEdges) #Construct quad element matrix for region outside the circular sector

#Combining eleMatTriHolder and eleMatQuadHolder
linEleMat=sp.zeros([eleMatTriHolder.shape[0]+eleMatQuadHolder.shape[0],6])
counter=0
for n in range(eleMatTriHolder.shape[0]):
    linEleMat[n,[0,1,2,3,4]]=eleMatTriHolder[n,:]
    counter+=1
for n in range(eleMatQuadHolder.shape[0]):
    linEleMat[counter+n,:]=eleMatQuadHolder[n,:]

linEleMat=linEleMat.astype(int)
linBoundMat=boundMatTot(numEdges)[0] #Construct the boundary
vertMatCircHolder = vertMatCirc(numEdges) #Construct vertex matrix for triang region
vertMatQuadHolder = vertMatQuad(numEdges) #Construct vertex matrix for the quad region


#Combining the two vertex matrices in Quadrant I (q1)
linVertMat=sp.zeros([vertMatCircHolder.shape[0]+vertMatQuadHolder.shape[0],2])
counter=0
for n in range(vertMatCircHolder.shape[0]):
    linVertMat[n,:]=vertMatCircHolder[n,:]
    counter+=1
for n in range(vertMatQuadHolder.shape[0]):
    linVertMat[counter+n,:]=vertMatQuadHolder[n,:]

#Outputting P1/Q1 mesh to a .mesh file
g=open(outputName+'Lin.mesh','w')
g.write('MFEM mesh v1.0\n'+'\n')
g.write('dimension\n'+'2\n'+'\n')
g.write('elements\n'+'{}\n'.format(linEleMat.shape[0]))
for n in range(linEleMat.shape[0]):
    if linEleMat[n,1]==2:
        g.write('{} {} {} {} {}\n'.format(linEleMat[n,0],linEleMat[n,1],linEleMat[n,2],linEleMat[n,3],linEleMat[n,4]))
    else:
        g.write('{} {} {} {} {} {}\n'.format(linEleMat[n,0],linEleMat[n,1],linEleMat[n,2],linEleMat[n,3],linEleMat[n,4],linEleMat[n,5]))
g.write('\n'+'boundary\n'+'{}\n'.format(linBoundMat.shape[0]))
for n in range(linBoundMat.shape[0]):
    g.write('{} {} {} {}\n'.format(linBoundMat[n,0],linBoundMat[n,1],linBoundMat[n,2],linBoundMat[n,3]))
g.write('\n'+'vertices\n'+'{}\n'.format(linVertMat.shape[0])+'2\n')
for n in range(linVertMat.shape[0]):
    g.write('{} {}\n'.format(linVertMat[n,0],linVertMat[n,1]))
g.close()

if(visMesh==True):
    gVis(glvis,outputName+'Lin.mesh')

#Quadratic (P2/Q2) Element Generation

#1.)Create Edge list from previously generated linear elements
edgeMat=sp.zeros([3*eleMatTriHolder.shape[0]+4*eleMatQuadHolder.shape[0],2])
linEleMat=orient(linEleMat)#Make sure that element orientation is in agreement with MFEM requirements

counter=0
for n in range(linEleMat.shape[0]):
    if linEleMat[n,1]==2:
        edgeMat[counter,:]=[linEleMat[n,2],linEleMat[n,3]]
        counter+=1
        edgeMat[counter,:]=[linEleMat[n,3],linEleMat[n,4]]
        counter+=1
        edgeMat[counter,:]=[linEleMat[n,4],linEleMat[n,2]]
        counter+=1
    else:
        edgeMat[counter,:]=[linEleMat[n,2],linEleMat[n,3]]
        counter+=1
        edgeMat[counter,:]=[linEleMat[n,3],linEleMat[n,4]]
        counter+=1
        edgeMat[counter,:]=[linEleMat[n,4],linEleMat[n,5]]
        counter+=1
        edgeMat[counter,:]=[linEleMat[n,5],linEleMat[n,2]]
        counter+=1

#Remove duplicates
holder=[]
for n in range(edgeMat.shape[0]):
    counter=0
    for m in range(edgeMat.shape[0]):
        if edgeMat[n,0]==edgeMat[m,0] and edgeMat[n,1]==edgeMat[m,1] and m!=n:
            holder.append([n,m])
        elif edgeMat[n,1]==edgeMat[m,0] and edgeMat[n,0]==edgeMat[m,1] and m!=n:
            holder.append([n,m])

removeIndices=sp.zeros(len(holder))
for n in range(len(holder)):
    if holder[n][0]>holder[n][1]:
        removeIndices[n]=holder[n][0]
    else:
        removeIndices[n]=holder[n][1]
removeIndices=sp.unique(removeIndices).astype(int)
edgeMat=sp.delete(edgeMat,removeIndices,0)
edgeMat=edgeMat.astype(int)

edgeDofMat=sp.zeros([edgeMat.shape[0],2])#These will be the new DoFs that appear after the Element Vertices within the .mesh file
linVertMatRound=sp.round_(linVertMat,5) 
 
counter=0

for n in edgeMat:
    if linVertMatRound[n[0],1] == linVertMatRound[n[1],1]:
        xmid=(linVertMatRound[n[0],0]+linVertMatRound[n[1],0])/2.0
        ymid=linVertMatRound[n[0],1]
        edgeDofMat[counter,:]=[xmid,ymid]
    elif linVertMatRound[n[0],0] == linVertMatRound[n[1],0]:
        xmid=linVertMatRound[n[0],0]
        ymid=(linVertMatRound[n[0],1]+linVertMatRound[n[1],1])/2.0
        edgeDofMat[counter,:]=[xmid,ymid]
    else:
        r0=sp.sqrt(linVertMatRound[n[0],0]**2+linVertMatRound[n[0],1]**2)
        r1=sp.sqrt(linVertMatRound[n[1],0]**2+linVertMatRound[n[1],1]**2)
        rmid = (r0+r1)/2.0 #should not be needed
        xmidOld=(linVertMatRound[n[0],0]+linVertMatRound[n[1],0])/2.0
        ymidOld=(linVertMatRound[n[0],1]+linVertMatRound[n[1],1])/2.0
        midtheta=sp.arctan(ymidOld/xmidOld)
        xmid=rmid*sp.cos(midtheta)
        ymid=rmid*sp.sin(midtheta)
        edgeDofMat[counter,:]=[xmid,ymid]
    counter+=1
edgeDofMat = sp.round_(edgeDofMat,5)

#Determine midpoints of all Q1 elements:
quadCentroidLoc=sp.zeros([eleMatQuadHolder.shape[0],2])
for n in range(eleMatQuadHolder.shape[0]):
    quadCentroidLoc[n,0]=(linVertMatRound[eleMatQuadHolder[n,2],0]+linVertMatRound[eleMatQuadHolder[n,3],0]+linVertMatRound[eleMatQuadHolder[n,4],0]+linVertMatRound[eleMatQuadHolder[n,5],0])/4.0
    quadCentroidLoc[n,1]=(linVertMatRound[eleMatQuadHolder[n,2],1]+linVertMatRound[eleMatQuadHolder[n,3],1]+linVertMatRound[eleMatQuadHolder[n,4],1]+linVertMatRound[eleMatQuadHolder[n,5],1])/4.0

quadCentroidLoc = sp.round_(quadCentroidLoc,5)

#3.)Populate nodes section
g=open(outputName+'Quad.mesh','w')
g.write('MFEM mesh v1.0\n'+'\n')
g.write('dimension\n'+'2\n'+'\n')
g.write('elements\n'+'{}\n'.format(linEleMat.shape[0]))
for n in range(linEleMat.shape[0]):
    if linEleMat[n,1]==2:
        g.write('{} {} {} {} {}\n'.format(linEleMat[n,0],linEleMat[n,1],linEleMat[n,2],linEleMat[n,3],linEleMat[n,4]))
    else:
        g.write('{} {} {} {} {} {}\n'.format(linEleMat[n,0],linEleMat[n,1],linEleMat[n,2],linEleMat[n,3],linEleMat[n,4],linEleMat[n,5]))
g.write('\n'+'boundary\n'+'{}\n'.format(linBoundMat.shape[0]))
for n in range(linBoundMat.shape[0]):
    g.write('{} {} {} {}\n'.format(linBoundMat[n,0],linBoundMat[n,1],linBoundMat[n,2],linBoundMat[n,3]))
g.write('\n'+'vertices\n'+'{}\n'.format(linVertMat.shape[0]))
g.write('\n'+'nodes'+'\n'+'FiniteElementSpace'+'\n'+'FiniteElementCollection: H1_2D_P2'+'\n'+'VDim: 2'+'\n'+'Ordering: 1' +'\n\n')
for n in range(linVertMatRound.shape[0]):
    g.write('{} {}\n'.format(linVertMatRound[n,0],linVertMatRound[n,1]))
for n in range(edgeDofMat.shape[0]):
    g.write('{} {}\n'.format(edgeDofMat[n,0],edgeDofMat[n,1]))
for n in range(quadCentroidLoc.shape[0]):
    g.write('{} {}\n'.format(quadCentroidLoc[n,0],quadCentroidLoc[n,1]))
g.close()

if(visMesh==True):
    gVis(glvis,outputName+'Quad.mesh')

#Cubic (P3/Q3) Element Generation

cubeDofMat=sp.zeros([2*edgeMat.shape[0],2])#These will be the new DoFs that appear after the Element Vertices within the .mesh file

counter=0
for n in edgeMat: #Here DoF ordering matters. 
    if linVertMatRound[n[0],1] == linVertMatRound[n[1],1]:
        xmid=(linVertMatRound[n[0],0]+linVertMatRound[n[1],0])/2.0
        ymid=linVertMatRound[n[0],1]
        xmid1=(linVertMatRound[n[0],0]+xmid)/2.0
        ymid1=linVertMatRound[n[0],1]
        xmid2=(linVertMatRound[n[1],0]+xmid)/2.0
        ymid2=linVertMatRound[n[0],1]
        if n[0] > n[1]:
            cubeDofMat[counter,:]=[xmid2,ymid2]
            counter+=1
            cubeDofMat[counter,:]=[xmid1,ymid1]
            counter+=1
        else:
            cubeDofMat[counter,:]=[xmid1,ymid1]
            counter+=1
            cubeDofMat[counter,:]=[xmid2,ymid2]
            counter+=1
            
    elif linVertMatRound[n[0],0] == linVertMatRound[n[1],0]:
        xmid=linVertMatRound[n[0],0]
        ymid=(linVertMatRound[n[0],1]+linVertMatRound[n[1],1])/2.0
        xmid1=linVertMatRound[n[0],0]
        ymid1=(linVertMatRound[n[0],1]+ymid)/2.0
        xmid2=linVertMatRound[n[0],0]
        ymid2=(linVertMatRound[n[1],1]+ymid)/2.0
        if n[0] > n[1]:
            cubeDofMat[counter,:]=[xmid2,ymid2]
            counter+=1
            cubeDofMat[counter,:]=[xmid1,ymid1]
            counter+=1
        else:
            cubeDofMat[counter,:]=[xmid1,ymid1]
            counter+=1
            cubeDofMat[counter,:]=[xmid2,ymid2]
            counter+=1
    else:
        r0=sp.sqrt(linVertMatRound[n[0],0]**2+linVertMatRound[n[0],1]**2)
        r1=sp.sqrt(linVertMatRound[n[1],0]**2+linVertMatRound[n[1],1]**2)
        rmid = (r0+r1)/2.0 #should not be needed
        xmidOld=(linVertMatRound[n[0],0]+linVertMatRound[n[1],0])/2.0
        ymidOld=(linVertMatRound[n[0],1]+linVertMatRound[n[1],1])/2.0
        midtheta=sp.arctan(ymidOld/xmidOld)
        xmid=rmid*sp.cos(midtheta)
        ymid=rmid*sp.sin(midtheta)
        xmid1=(linVertMatRound[n[0],0]+xmid)/2.0
        ymid1=(linVertMatRound[n[0],1]+ymid)/2.0
        xmid2=(linVertMatRound[n[1],0]+xmid)/2.0
        ymid2=(linVertMatRound[n[1],1]+ymid)/2.0
        if n[0] > n[1]:
            cubeDofMat[counter,:]=[xmid2,ymid2]
            counter+=1
            cubeDofMat[counter,:]=[xmid1,ymid1]
            counter+=1
        else:
            cubeDofMat[counter,:]=[xmid1,ymid1]
            counter+=1
            cubeDofMat[counter,:]=[xmid2,ymid2]
            counter+=1
            
cubeDofMat = sp.round_(cubeDofMat,5)

triCentroidLoc=sp.zeros([eleMatTriHolder.shape[0],2])

for n in range(eleMatTriHolder.shape[0]):
    triCentroidLoc[n,0]=(linVertMatRound[eleMatTriHolder[n,2],0]+linVertMatRound[eleMatTriHolder[n,3],0]+linVertMatRound[eleMatTriHolder[n,4],0])/3.0
    triCentroidLoc[n,1]=(linVertMatRound[eleMatTriHolder[n,2],1]+linVertMatRound[eleMatTriHolder[n,3],1]+linVertMatRound[eleMatTriHolder[n,4],1])/3.0

quadCentroidLocCubic=sp.zeros([4*eleMatQuadHolder.shape[0],2])

counter=0
for n in range(eleMatQuadHolder.shape[0]):
    xcent=quadCentroidLoc[n,0];ycent=quadCentroidLoc[n,1]
    a=eleMatQuadHolder[n,2:6]
    aMinIndex=sp.where(a[:]==a.min())[0][0]
    dof0=0.5*sp.array([xcent+linVertMatRound[a[aMinIndex],0],ycent+linVertMatRound[a[aMinIndex],1]])
    quadCentroidLocCubic[counter,:]=dof0
    counter+=1
    if aMinIndex==0:
        aLeft=-1
        aRight=1
        aLast=2
    else:
        aLeft=aMinIndex-1
        aRight=aMinIndex+1
        aLast=sp.delete(a,[aMinIndex,aLeft,aRight])[0]
    edge1=[a[aMinIndex], a[aLeft]]
    edge2=[a[aMinIndex], a[aRight]]
    edge1Index=0
    edge2Index=0
    edgeCounter=0
    for edge in edgeMat:
        if(edge[0]==edge1[0] and edge[1]==edge1[1]) or (edge[1]==edge1[0] and edge[0]==edge1[1]):
            edge1Index=edgeCounter
        if(edge[0]==edge2[0] and edge[1]==edge2[1]) or (edge[1]==edge2[0] and edge[0]==edge2[1]):
            edge2Index=edgeCounter
        edgeCounter+=1

    if (edge1Index > edge2Index):
        dof1=0.5*sp.array([xcent+linVertMatRound[a[aLeft],0],ycent+linVertMatRound[a[aLeft],1]])
        quadCentroidLocCubic[counter,:]=dof1
        counter+=1
        dof2=0.5*sp.array([xcent+linVertMatRound[a[aRight],0],ycent+linVertMatRound[a[aRight],1]])
        quadCentroidLocCubic[counter,:]=dof2
        counter+=1
        dof3=0.5*sp.array([xcent+linVertMatRound[a[aLast],0],ycent+linVertMatRound[a[aLast],1]])
        quadCentroidLocCubic[counter,:]=dof3
        counter+=1
    else:
        dof1=0.5*sp.array([xcent+linVertMatRound[a[aRight],0],ycent+linVertMatRound[a[aRight],1]])
        quadCentroidLocCubic[counter,:]=dof1
        counter+=1
        dof2=0.5*sp.array([xcent+linVertMatRound[a[aLeft],0],ycent+linVertMatRound[a[aLeft],1]])
        quadCentroidLocCubic[counter,:]=dof2
        counter+=1
        dof3=0.5*sp.array([xcent+linVertMatRound[a[aLast],0],ycent+linVertMatRound[a[aLast],1]])
        quadCentroidLocCubic[counter,:]=dof3
        counter+=1
        
truCentroidLoc=sp.round_(triCentroidLoc,5)
    
#3.)Populate nodes section
g=open(outputName+'Cub.mesh','w')
g.write('MFEM mesh v1.0\n'+'\n')
g.write('dimension\n'+'2\n'+'\n')
g.write('elements\n'+'{}\n'.format(linEleMat.shape[0]))
for n in range(linEleMat.shape[0]):
    if linEleMat[n,1]==2:
        g.write('{} {} {} {} {}\n'.format(linEleMat[n,0],linEleMat[n,1],linEleMat[n,2],linEleMat[n,3],linEleMat[n,4]))
    else:
        g.write('{} {} {} {} {} {}\n'.format(linEleMat[n,0],linEleMat[n,1],linEleMat[n,2],linEleMat[n,3],linEleMat[n,4],linEleMat[n,5]))
g.write('\n'+'boundary\n'+'{}\n'.format(linBoundMat.shape[0]))
for n in range(linBoundMat.shape[0]):
    g.write('{} {} {} {}\n'.format(linBoundMat[n,0],linBoundMat[n,1],linBoundMat[n,2],linBoundMat[n,3]))
g.write('\n'+'vertices\n'+'{}\n'.format(linVertMat.shape[0]))
g.write('\n'+'nodes'+'\n'+'FiniteElementSpace'+'\n'+'FiniteElementCollection: H1_2D_P3'+'\n'+'VDim: 2'+'\n'+'Ordering: 1' +'\n\n')
for n in range(linVertMatRound.shape[0]):
    g.write('{} {}\n'.format(linVertMatRound[n,0],linVertMatRound[n,1]))
for n in range(cubeDofMat.shape[0]):
    g.write('{} {}\n'.format(cubeDofMat[n,0],cubeDofMat[n,1]))
for n in range(triCentroidLoc.shape[0]):
    g.write('{} {}\n'.format(triCentroidLoc[n,0],triCentroidLoc[n,1]))
for n in range(quadCentroidLocCubic.shape[0]):
    g.write('{} {}\n'.format(quadCentroidLocCubic[n,0],quadCentroidLocCubic[n,1]))
g.close()

if(visMesh==True):
    gVis(glvis,outputName+'Cub.mesh')
#raw_input()
#'Reflecting' topology about one of its edges and append it to itself
upperPlaneEleMat = sp.zeros([2*linEleMat.shape[0],6])
for n in range(linEleMat.shape[0]):
    upperPlaneEleMat[n,:]=linEleMat[n,:]
    
#Create ele_mat_holder.shape[0]x2 matrix for mapping
refEdge=boundMatTot(numEdges)[1]
q1NumNodes=linVertMat.shape[0]

mapping = sp.zeros([q1NumNodes])
counter=0
for n in range(q1NumNodes):
     if (sp.any(refEdge == n)):
         mapping[n]=n
     else:
         mapping[n]=counter+q1NumNodes
         counter+=1

mapping=mapping.astype(int)
#Implement mapping

counter=0
for n in range(linEleMat.shape[0],2*linEleMat.shape[0]):
    upperPlaneEleMat[n,0]=linEleMat[counter,0]
    upperPlaneEleMat[n,1]=linEleMat[counter,1]
    upperPlaneEleMat[n,2]=mapping[linEleMat[counter,2]]
    upperPlaneEleMat[n,3]=mapping[linEleMat[counter,3]]
    upperPlaneEleMat[n,4]=mapping[linEleMat[counter,4]]
    upperPlaneEleMat[n,5]=mapping[linEleMat[counter,5]]
    counter+=1
    
upperPlaneEleMat = upperPlaneEleMat.astype(int)

#Reflecting boundary matrix
origBound=boundMatTot(numEdges)[2]
upperPlaneBoundMat=sp.zeros([2*origBound.shape[0],4])
for n in range(origBound.shape[0]):
    upperPlaneBoundMat[n,:]=origBound[n,:]
counter=0
newOrigBound=origBound.copy()
newOrigBound[:,2]=sp.flipud(origBound[:,3])
newOrigBound[:,3]=sp.flipud(origBound[:,2])
for n in range(newOrigBound.shape[0],upperPlaneBoundMat.shape[0]):
    upperPlaneBoundMat[n,0]=newOrigBound[counter,0]
    upperPlaneBoundMat[n,1]=newOrigBound[counter,1]
    upperPlaneBoundMat[n,2]=mapping[newOrigBound[counter,2]]
    upperPlaneBoundMat[n,3]=mapping[newOrigBound[counter,3]]
    counter+=1
upperPlaneBoundMat=upperPlaneBoundMat.astype(int)

#Reflecting vertex matrix about the y-axis and appending it to itself
upperPlaneNumNodes=q1NumNodes+(q1NumNodes-refEdge.shape[0])
upperPlaneVertMat = sp.zeros([upperPlaneNumNodes,2])
for n in range(linVertMat.shape[0]):
    upperPlaneVertMat[n,:]=linVertMat[n,:]
counter=0
for n in range(linVertMat.shape[0],upperPlaneNumNodes):
        upperPlaneVertMat[n,0]=-1.0*linVertMat[sp.where(mapping==n)[0][0],0]
        upperPlaneVertMat[n,1]=linVertMat[sp.where(mapping==n)[0][0],1]
        counter+=1

upperPlaneEleMat=orient(upperPlaneEleMat)
g=open(outputName+'UpperPlaneLin.mesh','w')
g.write('MFEM mesh v1.0\n'+'\n')
g.write('dimension\n'+'2\n'+'\n')
g.write('elements\n'+'{}\n'.format(upperPlaneEleMat.shape[0]))
for n in range(upperPlaneEleMat.shape[0]):
    if upperPlaneEleMat[n,1]==2:
        g.write('{} {} {} {} {}\n'.format(upperPlaneEleMat[n,0],upperPlaneEleMat[n,1],upperPlaneEleMat[n,2],upperPlaneEleMat[n,3],upperPlaneEleMat[n,4]))
    else:
        g.write('{} {} {} {} {} {}\n'.format(upperPlaneEleMat[n,0],upperPlaneEleMat[n,1],upperPlaneEleMat[n,2],upperPlaneEleMat[n,3],upperPlaneEleMat[n,4],upperPlaneEleMat[n,5]))
g.write('\n'+'boundary\n'+'{}\n'.format(upperPlaneBoundMat.shape[0]))
for n in range(upperPlaneBoundMat.shape[0]):
    g.write('{} {} {} {}\n'.format(upperPlaneBoundMat[n,0],upperPlaneBoundMat[n,1],upperPlaneBoundMat[n,2],upperPlaneBoundMat[n,3]))
g.write('\n'+'vertices\n'+'{}\n'.format(upperPlaneVertMat.shape[0])+'2\n')
for n in range(upperPlaneVertMat.shape[0]):
    g.write('{} {}\n'.format(upperPlaneVertMat[n,0],upperPlaneVertMat[n,1]))
g.close()

if(visMesh==True):
    gVis(glvis,outputName+'UpperPlaneLin.mesh')

#'Reflecting' topology about one of its edges and append it to itself
wholePlaneEleMat = sp.zeros([2*upperPlaneEleMat.shape[0],6])
for n in range(upperPlaneEleMat.shape[0]):
    wholePlaneEleMat[n,:]=upperPlaneEleMat[n,:]

quad1Edge=boundMatTot(numEdges)[3]
newRefEdge=sp.zeros(2*quad1Edge.shape[0]-1)
for n in range(quad1Edge.shape[0]):
    newRefEdge[n]=quad1Edge[n]
counter=0
for n in range(quad1Edge.shape[0],newRefEdge.shape[0]):
    newRefEdge[n]=mapping[quad1Edge[counter]]
    counter+=1
newRefEdge=sp.unique(newRefEdge)
newRefEdge=newRefEdge.astype(int)
newTotNumNodes=upperPlaneVertMat.shape[0]

newMapping=sp.zeros([newTotNumNodes])
counter=0
for n in range(newTotNumNodes):
    if (sp.any(newRefEdge == n)):
        newMapping[n]=n
    else:
        newMapping[n]=counter+newTotNumNodes
        counter+=1
newMapping=newMapping.astype(int)

counter=0
for n in range(upperPlaneEleMat.shape[0],2*upperPlaneEleMat.shape[0]):
    wholePlaneEleMat[n,0]=upperPlaneEleMat[counter,0]
    wholePlaneEleMat[n,1]=upperPlaneEleMat[counter,1]
    wholePlaneEleMat[n,2]=newMapping[upperPlaneEleMat[counter,2]]
    wholePlaneEleMat[n,3]=newMapping[upperPlaneEleMat[counter,3]]
    wholePlaneEleMat[n,4]=newMapping[upperPlaneEleMat[counter,4]]
    wholePlaneEleMat[n,5]=newMapping[upperPlaneEleMat[counter,5]]
    counter+=1

wholePlaneEleMat=wholePlaneEleMat.astype(int)

#Reflecting boundary matrix
newOrigBoundQuad1=boundMatTot(numEdges)[4]
newFirstBoundMatHolder=sp.zeros([2*newOrigBoundQuad1.shape[0],4])
for n in range(newOrigBoundQuad1.shape[0]):
    newFirstBoundMatHolder[n,:]=newOrigBoundQuad1[n,:]

newNewOrigBoundQuad1=newOrigBoundQuad1.copy()
newNewOrigBoundQuad1[:,2]=sp.flipud(newOrigBoundQuad1[:,3])
newNewOrigBoundQuad1[:,3]=sp.flipud(newOrigBoundQuad1[:,2])
counter=0
for n in range(newOrigBoundQuad1.shape[0],newFirstBoundMatHolder.shape[0]):
    newFirstBoundMatHolder[n,0]=newNewOrigBoundQuad1[counter,0]
    newFirstBoundMatHolder[n,1]=newNewOrigBoundQuad1[counter,1]
    newFirstBoundMatHolder[n,2]=mapping[newNewOrigBoundQuad1[counter,2]]
    newFirstBoundMatHolder[n,3]=mapping[newNewOrigBoundQuad1[counter,3]]
    counter+=1

upperQuadMat=newFirstBoundMatHolder.copy()
wholePlaneBoundMat=sp.zeros([2*upperQuadMat.shape[0],4])
for n in range(upperQuadMat.shape[0]):
    wholePlaneBoundMat[n,:]=upperQuadMat[n,:]

counter=0
newNewOrigBound=upperQuadMat.copy()
newNewOrigBound[:,2]=sp.flipud(upperQuadMat[:,3])
newNewOrigBound[:,3]=sp.flipud(upperQuadMat[:,2])
newNewOrigBound=newNewOrigBound.astype(int)
for n in range(newNewOrigBound.shape[0],wholePlaneBoundMat.shape[0]):
    wholePlaneBoundMat[n,0]=newNewOrigBound[counter,0]
    wholePlaneBoundMat[n,1]=newNewOrigBound[counter,1]
    wholePlaneBoundMat[n,2]=newMapping[newNewOrigBound[counter,2]]
    wholePlaneBoundMat[n,3]=newMapping[newNewOrigBound[counter,3]]
    counter+=1
wholePlaneBoundMat=wholePlaneBoundMat.astype(int)

wholePlaneNumNodes=newTotNumNodes+(newTotNumNodes-newRefEdge.shape[0])
wholePlaneVertMat = sp.zeros([wholePlaneNumNodes,2])
for n in range(upperPlaneVertMat.shape[0]):
    wholePlaneVertMat[n,:]=upperPlaneVertMat[n,:]
counter=0
for n in range(upperPlaneVertMat.shape[0],wholePlaneNumNodes):
        wholePlaneVertMat[n,0]=upperPlaneVertMat[sp.where(newMapping==n)[0][0],0]
        wholePlaneVertMat[n,1]=-1.0*upperPlaneVertMat[sp.where(newMapping==n)[0][0],1]
        counter+=1

g=open(outputName+'WholePlaneLin.mesh','w')
g.write('MFEM mesh v1.0\n'+'\n')
g.write('dimension\n'+'2\n'+'\n')
g.write('elements\n'+'{}\n'.format(wholePlaneEleMat.shape[0]))
for n in range(wholePlaneEleMat.shape[0]):
    if wholePlaneEleMat[n,1]==2:
        g.write('{} {} {} {} {}\n'.format(wholePlaneEleMat[n,0],wholePlaneEleMat[n,1],wholePlaneEleMat[n,2],wholePlaneEleMat[n,3],wholePlaneEleMat[n,4]))
    else:
        g.write('{} {} {} {} {} {}\n'.format(wholePlaneEleMat[n,0],wholePlaneEleMat[n,1],wholePlaneEleMat[n,2],wholePlaneEleMat[n,3],wholePlaneEleMat[n,4],wholePlaneEleMat[n,5]))
g.write('\n'+'boundary\n'+'{}\n'.format(wholePlaneBoundMat.shape[0]))
for n in range(wholePlaneBoundMat.shape[0]):
    g.write('{} {} {} {}\n'.format(wholePlaneBoundMat[n,0],wholePlaneBoundMat[n,1],wholePlaneBoundMat[n,2],wholePlaneBoundMat[n,3]))
g.write('\n'+'vertices\n'+'{}\n'.format(wholePlaneVertMat.shape[0])+'2\n')
for n in range(wholePlaneVertMat.shape[0]):
    g.write('{} {}\n'.format(wholePlaneVertMat[n,0],wholePlaneVertMat[n,1]))
g.close()

if(visMesh==True):
    gVis(glvis,outputName+'WholePlaneLin.mesh')

#1.)Create Edge list from elements
wholePlaneEleMat=orient(wholePlaneEleMat)
triCounter=0;quadCounter=0;
for n in range(wholePlaneEleMat.shape[0]):
    if wholePlaneEleMat[n,1]==2:
        triCounter+=1
    else:
        quadCounter+=1
edgeMat=sp.zeros([3*triCounter+4*quadCounter,2])
counter=0
for n in range(wholePlaneEleMat.shape[0]):
    if wholePlaneEleMat[n,1]==2:
        edgeMat[counter,:]=[wholePlaneEleMat[n,2],wholePlaneEleMat[n,3]]
        counter+=1
        edgeMat[counter,:]=[wholePlaneEleMat[n,3],wholePlaneEleMat[n,4]]
        counter+=1
        edgeMat[counter,:]=[wholePlaneEleMat[n,4],wholePlaneEleMat[n,2]]
        counter+=1
    else:
        edgeMat[counter,:]=[wholePlaneEleMat[n,2],wholePlaneEleMat[n,3]]
        counter+=1
        edgeMat[counter,:]=[wholePlaneEleMat[n,3],wholePlaneEleMat[n,4]]
        counter+=1
        edgeMat[counter,:]=[wholePlaneEleMat[n,4],wholePlaneEleMat[n,5]]
        counter+=1
        edgeMat[counter,:]=[wholePlaneEleMat[n,5],wholePlaneEleMat[n,2]]
        counter+=1

#Remove duplicates
holder=[]
for n in range(edgeMat.shape[0]):
    counter=0
    for m in range(edgeMat.shape[0]):
        if edgeMat[n,0]==edgeMat[m,0] and edgeMat[n,1]==edgeMat[m,1] and m!=n:
            holder.append([n,m])
        elif edgeMat[n,1]==edgeMat[m,0] and edgeMat[n,0]==edgeMat[m,1] and m!=n:
            holder.append([n,m])
removeIndices=sp.zeros(len(holder))
for n in range(len(holder)):
    if holder[n][0]>holder[n][1]:
        removeIndices[n]=holder[n][0]
    else:
        removeIndices[n]=holder[n][1]
removeIndices=sp.unique(removeIndices).astype(int)
edgeMat=sp.delete(edgeMat,removeIndices,0)
edgeMat=edgeMat.astype(int)

edgeDofMat=sp.zeros([edgeMat.shape[0],2])

wholePlaneVertMatRound=sp.round_(wholePlaneVertMat,5)

counter=0
for n in edgeMat:
    if wholePlaneVertMatRound[n[0],1] == wholePlaneVertMatRound[n[1],1]:
        xmid=(wholePlaneVertMatRound[n[0],0]+wholePlaneVertMatRound[n[1],0])/2.0
        ymid=wholePlaneVertMatRound[n[0],1]
        edgeDofMat[counter,:]=[xmid,ymid]
    elif wholePlaneVertMatRound[n[0],0] == wholePlaneVertMatRound[n[1],0]:
        xmid=wholePlaneVertMatRound[n[0],0]
        ymid=(wholePlaneVertMatRound[n[0],1]+wholePlaneVertMatRound[n[1],1])/2.0
        edgeDofMat[counter,:]=[xmid,ymid]
    else:
        r0=sp.sqrt(wholePlaneVertMatRound[n[0],0]**2+wholePlaneVertMatRound[n[0],1]**2)
        r1=sp.sqrt(wholePlaneVertMatRound[n[1],0]**2+wholePlaneVertMatRound[n[1],1]**2)
        rmid = (r0+r1)/2.0 #should not be needed
        xmidOld=(wholePlaneVertMatRound[n[0],0]+wholePlaneVertMatRound[n[1],0])/2.0
        ymidOld=(wholePlaneVertMatRound[n[0],1]+wholePlaneVertMatRound[n[1],1])/2.0
        midtheta=sp.arctan2(ymidOld,xmidOld)
        xmid=rmid*sp.cos(midtheta)
        ymid=rmid*sp.sin(midtheta)
        edgeDofMat[counter,:]=[xmid,ymid]
    counter+=1
edgeDofMat = sp.round_(edgeDofMat,5)

#2.)Create correct dof locations
#Determine midpoints of all quads:
quadCentroidLoc=sp.zeros([quadCounter,2])
counter=0
for n in range(wholePlaneEleMat.shape[0]):
    if wholePlaneEleMat[n,1]==3:
        quadCentroidLoc[counter,0]=(wholePlaneVertMatRound[wholePlaneEleMat[n,2],0]+wholePlaneVertMatRound[wholePlaneEleMat[n,3],0]+wholePlaneVertMatRound[wholePlaneEleMat[n,4],0]+wholePlaneVertMatRound[wholePlaneEleMat[n,5],0])/4.0
        quadCentroidLoc[counter,1]=(wholePlaneVertMatRound[wholePlaneEleMat[n,2],1]+wholePlaneVertMatRound[wholePlaneEleMat[n,3],1]+wholePlaneVertMatRound[wholePlaneEleMat[n,4],1]+wholePlaneVertMatRound[wholePlaneEleMat[n,5],1])/4.0
        counter+=1

quadCentroidLoc = sp.round_(quadCentroidLoc,5)

#3.)Populate nodes section
g=open(outputName+'WholePlaneQuad.mesh','w')
g.write('MFEM mesh v1.0\n'+'\n')
g.write('dimension\n'+'2\n'+'\n')
g.write('elements\n'+'{}\n'.format(wholePlaneEleMat.shape[0]))
for n in range(wholePlaneEleMat.shape[0]):
    if wholePlaneEleMat[n,1]==2:
        g.write('{} {} {} {} {}\n'.format(wholePlaneEleMat[n,0],wholePlaneEleMat[n,1],wholePlaneEleMat[n,2],wholePlaneEleMat[n,3],wholePlaneEleMat[n,4]))
    else:
        g.write('{} {} {} {} {} {}\n'.format(wholePlaneEleMat[n,0],wholePlaneEleMat[n,1],wholePlaneEleMat[n,2],wholePlaneEleMat[n,3],wholePlaneEleMat[n,4],wholePlaneEleMat[n,5]))
g.write('\n'+'boundary\n'+'{}\n'.format(wholePlaneBoundMat.shape[0]))
for n in range(wholePlaneBoundMat.shape[0]):
    g.write('{} {} {} {}\n'.format(wholePlaneBoundMat[n,0],wholePlaneBoundMat[n,1],wholePlaneBoundMat[n,2],wholePlaneBoundMat[n,3]))
g.write('\n'+'vertices\n'+'{}\n'.format(wholePlaneVertMat.shape[0]))
g.write('\n'+'nodes'+'\n'+'FiniteElementSpace'+'\n'+'FiniteElementCollection: H1_2D_P2'+'\n'+'VDim: 2'+'\n'+'Ordering: 1' +'\n\n')
for n in range(wholePlaneVertMatRound.shape[0]):
    g.write('{} {}\n'.format(wholePlaneVertMatRound[n,0],wholePlaneVertMatRound[n,1]))
for n in range(edgeDofMat.shape[0]):
    g.write('{} {}\n'.format(edgeDofMat[n,0],edgeDofMat[n,1]))
for n in range(quadCentroidLoc.shape[0]):
    g.write('{} {}\n'.format(quadCentroidLoc[n,0],quadCentroidLoc[n,1]))
g.close()

if(visMesh==True):
    gVis(glvis,outputName+'WholePlaneQuad.mesh')
