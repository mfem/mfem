clear all;
%this is a simple matlab scrip to generate 3*2^n meshes
m=2^2;

elements=m^2;

onev=ones(1,m^2);

i1=0:1:(m^2-1);
i4=m:1:(m*(m+1)-1);
for i=1:m
    for j=1:m
        
        ind=j+(i-1)*m;
        if (j~=m)
            i2(ind)=j+(i-1)*m;
        else
            i2(ind)=(i-1)*m;
        end

        i3(ind)=i2(ind)+m;
    end
end

ielements=[i1+1; onev*3; i1; i2; i3; i4];

%boundary 2*m
onev2=ones(1,2*m);
l1=[0:1:m-1 fliplr([(m^2+1):1:(m^2+m-1) m^2]) ];
l2=[1:1:m-1 0  fliplr(m^2:1:m^2+m-1)];
iboundary=[onev2; onev2; l1; l2]; 

dx=2/m;

xx=-1:dx:1;
yy=xx;

fileID = fopen('test.mesh','w');

fprintf(fileID, 'MFEM mesh v1.0\n\ndimension\n2\n');
fprintf(fileID, '\nelements\n%i\n', m^2);
fprintf(fileID, '%i %i %i %i %i %i\n', ielements);
fprintf(fileID, '\n\nboundary\n%i\n', 2*m);
fprintf(fileID, '%i %i %i %i\n', iboundary);

fprintf(fileID, '\n\nvertices\n%i\n', m*(m+1));

fprintf(fileID, '\nnodes\n');
fprintf(fileID, 'FiniteElementSpace\n');
fprintf(fileID, 'FiniteElementCollection: L2_T1_2D_P1\n');
fprintf(fileID, 'VDim: 2\n');
fprintf(fileID, 'Ordering: 1\n\n');



for i=0:m^2-1;
    ii=mod(i,m);
    jj=(i-ii)/m;
    ii=ii+1; jj=jj+1;
    fprintf(fileID,'%12.8f %12.8f\n', xx(ii), yy(jj) );
    fprintf(fileID,'%12.8f %12.8f\n', xx(ii+1), yy(jj));
    fprintf(fileID,'%12.8f %12.8f\n', xx(ii), yy(jj+1));
    fprintf(fileID,'%12.8f %12.8f\n', xx(ii+1), yy(jj+1));
    fprintf(fileID,'\n');
end

fclose(fileID);
