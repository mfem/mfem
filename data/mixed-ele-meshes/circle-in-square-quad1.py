#Title:circle-in-square.py
#Author:Tim McManus
#Date:5-20-17
#Purpose: Fill a circular sector with triangles and a bounding region,
#defined by 3 nodes, with quads


import scipy as sp
import argparse
import sys

parser=argparse.ArgumentParser()
parser.add_argument('-r','--circ_rad',help='Radius of circle',required=True)
parser.add_argument('-e','--edge_length',help='Edge-length of bounding square',required=True)
parser.add_argument('-n','--num_edges',help='n-gon approximation of internal circle',required=True)
parser.add_argument('-o','--outputFile', help='Output file name.',required=True)
args=parser.parse_args()

r=float(args.circ_rad) #radius of disc
edge_length=float(args.edge_length)#edge length of bounding square
num_edges=int(args.num_edges) #number of edges along the sector
output_name=args.outputFile

if r >= edge_length:
    print("Circle radius must be less than bounding square edge length")
    sys.exit(1)

if sp.mod(num_edges,2) != 0:
    print("Currently this mixed element generator only supports an even numbers of edges.")
    sys.exit(1)

    
#The basic idea:
#1. Construct topology for regions
#2. Combine topologies
#3. Construct boundary
#4. Construct geometry for regions
#5. Combine geometries (relax/smooth?)
#6. Output


#Construct triangle element matrix for the region bounded
#by a circular sector
def ele_mat_circ(num_edges):
    ele_mat = sp.zeros([num_edges**2,num_edges+1])
    
    #This produces the correct result, but it seems to be reducible
    n_nodes_seq=sp.zeros([num_edges])
    n_nodes_seq[0]=3
    if num_edges != 1:
        for n in range(1,num_edges):
            n_nodes_seq[n]=n_nodes_seq[n-1]+(2+n)
    num_nodes_tot = n_nodes_seq[-1]

    #Stencil...probably could be reduced
    b=range(int(num_nodes_tot))
    row_size=1
    A=sp.zeros([num_edges+1,num_edges+1])
    start=0;stop=1;
    for m in range(num_edges+1):
        if m==0:
            A[m,range(m+1)]=b[0:1]
            start=0
            stop=1
        else:
            start=stop
            stop=stop+m+1
            A[m,range(m+1)]=b[start:stop]

    M=sp.ones([num_edges**2,5])
    m_row=0
    for m in range(num_edges):
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
            
    return M.astype(int),num_nodes_tot

[ele_mat_tri_holder,num_nodes_tot]=ele_mat_circ(num_edges)


#Construct square element matrix for the region outside of
#the circular sector

def ele_mat_quad(num_edges):
    S0=num_edges*(num_edges+1)/(2.0)
    A=sp.linspace(S0,(S0+(num_edges+1)**2)-1,(num_edges+1)**2)
    A=A.reshape([num_edges+1,num_edges+1])
    quadNode=sp.delete(A,-1,1)
    quadNode=sp.delete(quadNode,-1,0)
    quadNode=quadNode.flatten()
    M=sp.zeros([num_edges**2,6])
    for n in range(num_edges**2):
        M[n,:]=[2,3,quadNode[n],quadNode[n]+num_edges+1,quadNode[n]+num_edges+2,quadNode[n]+1]
        
    return M.astype(int)

ele_mat_quad_holder=ele_mat_quad(num_edges)


#Combining ele_mat_tri and ele_mat_quad
ele_holder_tot=sp.zeros([ele_mat_tri_holder.shape[0]+ele_mat_quad_holder.shape[0],6])
counter=0
for n in range(ele_mat_tri_holder.shape[0]):
    ele_holder_tot[n,[0,1,2,3,4]]=ele_mat_tri_holder[n,:]
    counter+=1
for n in range(ele_mat_quad_holder.shape[0]):
    ele_holder_tot[counter+n,:]=ele_mat_quad_holder[n,:]

ele_mat_holder=ele_holder_tot.astype(int)

#Constructing boundary
def bound_mat_tot(num_edges):
    triS1=sp.zeros(num_edges+1)
    triS3=sp.zeros(num_edges+1)
    quadS1=sp.zeros(num_edges)
    quadS2=sp.zeros(num_edges-1)
    quadS3=sp.zeros(num_edges)

    triS1[0]=0;
    triS3[0]=0;
    for n in range(1,num_edges+1):
        triS1[n]=triS1[n-1]+n
        triS3[n]=triS1[n]+n

    triS3=sp.flipud(triS3)
    quadS1[0]=triS1[-1]+num_edges+1
    quadS3[0]=triS1[-1]+2*num_edges+1

    for n in range(1,num_edges):
        quadS1[n]=quadS1[n-1]+(num_edges+1)
        quadS3[n]=quadS3[n-1]+(num_edges+1)    

    quadS3=sp.flipud(quadS3)
    quadS2=range(int(quadS1[-1]+1),int(quadS3[0]),1)
    STOT=sp.concatenate([triS1,quadS1,quadS2,quadS3,triS3],axis=0)

    boundMat=sp.zeros([STOT.size-1,4])

    for n in range(STOT.size-1):
        boundMat[n,:]=[1,1,STOT[n],STOT[n+1]]
    
    return boundMat.astype(int)
    
bound_mat_holder=bound_mat_tot(num_edges)
    
def vert_mat_circ(num_edges,num_nodes_tot):
    r_o=sp.linspace(0,r,num_edges+1)
    counter=0
    vert_mat=sp.zeros([int(num_nodes_tot),2])
    for m in range(num_edges+1):
        theta=sp.linspace(0,sp.pi/2.0,m+1)
        for n in range(sp.size(theta)):
            vert_mat[counter,:]=[r_o[m]*sp.cos(theta[n]),r_o[m]*sp.sin(theta[n])]
            counter+=1
    return vert_mat
vert_mat_circ_holder = vert_mat_circ(num_edges,num_nodes_tot)

#Constructing vertex matrix for quads regions
def vert_mat_quad(num_edges):
    
    theta=sp.linspace(0,sp.pi/2.0,num_edges+1)
    AX=sp.zeros([num_edges+1,num_edges+1])
    AY=sp.zeros([num_edges+1,num_edges+1])
    AX[0,:]=r*sp.cos(theta)
    AY[0,:]=r*sp.sin(theta)

    vert_lin_space=sp.linspace(0,edge_length,(num_edges/2)+1)
    horz_lin_space=sp.linspace(edge_length,0,(num_edges/2)+1)

    #Assigning node locations along the boundary
    vert_count=0
    horz_count=1
    for n in range(num_edges+1):
        if n < (num_edges/2):
            AX[-1,n]=edge_length
            AY[-1,n]=vert_lin_space[vert_count]
            vert_count+=1
        elif n == int(num_edges/2):
            AX[-1,n]=edge_length
            AY[-1,n]=edge_length
        else:
            AX[-1,n]=horz_lin_space[horz_count]
            AY[-1,n]=edge_length
            horz_count+=1

    #Linearly spacing nodes between the inner/outer boundaries
    #This could then be relaxed
    for col in range(num_edges+1):
        for row in range(1,num_edges):
            AX[row,col]=sp.linspace(AX[0,col],AX[-1,col],num_edges+1)[row]
            AY[row,col]=sp.linspace(AY[0,col],AY[-1,col],num_edges+1)[row]

    AX=sp.delete(AX,0,0)
    AY=sp.delete(AY,0,0)
    AX=AX.flatten()
    AY=AY.flatten()
    AX_reshape = AX.flatten()
    
    num_new_nodes=num_edges*(num_edges+1)
    vert_mat=sp.zeros([num_new_nodes,2])

    for n in range(num_new_nodes):
        vert_mat[n,:]=[AX[n],AY[n]]    
    return vert_mat

vert_mat_quad_holder = vert_mat_quad(num_edges)


#Combining the two vertex matrices
vert_holder_tot=sp.zeros([vert_mat_circ_holder.shape[0]+vert_mat_quad_holder.shape[0],2])
counter=0
for n in range(vert_mat_circ_holder.shape[0]):
    vert_holder_tot[n,:]=vert_mat_circ_holder[n,:]
    counter+=1
for n in range(vert_mat_quad_holder.shape[0]):
    vert_holder_tot[counter+n,:]=vert_mat_quad_holder[n,:]

vert_mat_holder=vert_holder_tot.copy()

#Outputting to a .mesh file
g=open(output_name,'w')
g.write('MFEM mesh v1.0\n'+'\n')
g.write('dimension\n'+'2\n'+'\n')
g.write('elements\n'+'{}\n'.format(ele_mat_holder.shape[0]))
for n in range(ele_mat_holder.shape[0]):
    if ele_mat_holder[n,1]==2:
        g.write('{} {} {} {} {}\n'.format(ele_mat_holder[n,0],ele_mat_holder[n,1],ele_mat_holder[n,2],ele_mat_holder[n,3],ele_mat_holder[n,4]))
    else:
        g.write('{} {} {} {} {} {}\n'.format(ele_mat_holder[n,0],ele_mat_holder[n,1],ele_mat_holder[n,2],ele_mat_holder[n,3],ele_mat_holder[n,4],ele_mat_holder[n,5]))
g.write('\n'+'boundary\n'+'{}\n'.format(bound_mat_holder.shape[0]))
for n in range(bound_mat_holder.shape[0]):
    g.write('{} {} {} {}\n'.format(bound_mat_holder[n,0],bound_mat_holder[n,1],bound_mat_holder[n,2],bound_mat_holder[n,3]))
g.write('\n'+'vertices\n'+'{}\n'.format(vert_mat_holder.shape[0])+'2\n')
for n in range(vert_mat_holder.shape[0]):
    g.write('{} {}\n'.format(vert_mat_holder[n,0],vert_mat_holder[n,1]))
g.close()
