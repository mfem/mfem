//
//  fe_skgrad.cpp
//  mfem
//
//  Created by psaber on 6/20/25.
//

#include "fe_curl.hpp"
#include "face_map_utils.hpp"
#include "../coefficient.hpp"
#include <fstream>


namespace mfem
{

using namespace std;


const double HCurl_PentatopeElement::tk1[10][4] =
{{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1},{-1,1,0,0},{-1,0,1,0},{0,-1,1,0},{-1,0,0,1},{-1,0,1,0},{0,0,-1,1}};

const double HCurl_PentatopeElement::tk2[14][4] =
{{-1,1,0,0},{-1,0,1,0},{-1,0,0,1},{0,-1,1,0},{0,-1,0,1},{0,0,-1,1},{0,-1,1,0},{0,-1,0,1},{0,0,-1,1},{0,0,-1,1},{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};


HCurl_PentatopeElement::HCurl_PentatopeElement(const int p)
: VectorFiniteElement(4, Geometry::PENTATOPE, p*(p*p*p + 8*p*p + 19*p +12)/4,
                      p, H_CURL, FunctionSpace::Pk),
dof2tk1(dof), dof2tk2(dof) , doftrans(p)
{
    
    const real_t *fop = poly1d.OpenPoints(p - 1);
    const real_t *ftop = (p > 1) ? poly1d.OpenPoints(p - 2) : NULL;
    const real_t *iop = (p > 2) ? poly1d.OpenPoints(p - 3) : NULL;
    
    const int pm1 = p - 1, pm2 = p - 2, pm3 = p - 3, pm4 = p -4;
    
    // Lamda function for computing the scaled skew-symmetric-outer-product of two four vectors
    auto skw_sym_outerprod_fnc = [&](const std::vector<double>& vec1, const std::vector<double>& vec2)
    {
        DenseMatrix result(4,4);
        for (int i = 0; i < vec1.size(); ++i)
        {
            for (int j = 0; j < vec2.size(); ++j)
            {
                // Forward Product
                double outer_prod_1 = vec1[i] * vec2[j];
                
                // "Backward" Product
                double outer_prod_2 = vec2[i] * vec1[j];
                
                // Compute Result
                result(i,j) = (outer_prod_1 - outer_prod_2);
            }
        }
        return result;
    }; // end of lamda function

    
#ifndef MFEM_THREAD_SAFE
    shape_x.SetSize(p);
    shape_y.SetSize(p);
    shape_z.SetSize(p);
    shape_t.SetSize(p);
    shape_l.SetSize(p);
    dshape_x.SetSize(p);
    dshape_y.SetSize(p);
    dshape_z.SetSize(p);
    dshape_t.SetSize(p);
    dshape_l.SetSize(p);
    u.SetSize(dof, dim, dim);
#else
    Vector shape_x(p), shape_y(p), shape_z(p), shape_t(p),
    shape_l(p);
#endif
    int o = 0;
    // faces (see Mesh::GeneratePlanars in mesh/mesh.cpp)
    for (int j = 0; j < p; j++)
       for (int i=0; i + j < p; i++) // (0,1,2)
       {
          double w = fop[i] + fop[j] + fop[pm1-i-j];
          Nodes.IntPoint(o).Set4(fop[i]/w, fop[j]/w, 0.0, 0.0);
          dof2tk1[o] = 0;
          dof2tk2[o++] = 0;
       }
    for (int j = 0; j < p; j++)
       for (int i=0; i + j < p; i++) // (0,1,3)
       {
          double w = fop[i] + fop[j] + fop[pm1-i-j];
          Nodes.IntPoint(o).Set4(fop[i]/w, 0.0, fop[j]/w, 0.0);
          dof2tk1[o] = 0;
          dof2tk2[o++] = 1;
       }
    for (int j = 0; j < p; j++)
       for (int i=0; i + j < p; i++) // (0,1,4)
       {
          double w = fop[i] + fop[j] + fop[pm1-i-j];
          Nodes.IntPoint(o).Set4(fop[i]/w, 0.0, 0.0, fop[j]/w);
          dof2tk1[o] = 0;
          dof2tk2[o++] = 2;

       }
    for (int j = 0; j < p; j++)
       for (int i=0; i + j < p; i++) // (0,2,3)
       {
          double w = fop[i] + fop[j] + fop[pm1-i-j];
          Nodes.IntPoint(o).Set4(0.0, fop[i]/w, fop[j]/w, 0.0);
          dof2tk1[o] = 1;
          dof2tk2[o++] = 3;

       }
    for (int j = 0; j < p; j++)
       for (int i=0; i + j < p; i++) // (0,2,4)
       {
          double w = fop[i] + fop[j] + fop[pm1-i-j];
          Nodes.IntPoint(o).Set4(0.0, fop[i]/w, 0.0, fop[j]/w);
          dof2tk1[o] = 1;
          dof2tk2[o++] = 4;
       }
    for (int j = 0; j < p; j++)
       for (int i=0; i + j < p; i++) // (0,3,4)
       {
          double w = fop[i] + fop[j] + fop[pm1-i-j];
          Nodes.IntPoint(o).Set4(0.0, 0.0, fop[i]/w, fop[j]/w);
          dof2tk1[o] = 2;
          dof2tk2[o++] = 5;
       }
    for (int j = 0; j < p; j++)
       for (int i=0; i + j < p; i++) // (1,2,3)
       {
          double w = fop[i] + fop[j] + fop[pm1-i-j];
          Nodes.IntPoint(o).Set4(fop[pm1-i-j]/w, fop[i]/w, fop[j]/w, 0.0);
          dof2tk1[o] = 4;
          dof2tk2[o++] = 6;
       }
    for (int j = 0; j < p; j++)
       for (int i=0; i + j < p; i++) // (1,2,4)
       {
          double w = fop[i] + fop[j] + fop[pm1-i-j];
          Nodes.IntPoint(o).Set4(fop[pm1-i-j]/w, fop[i]/w, 0.0, fop[j]/w);
          dof2tk1[o] = 4;
          dof2tk2[o++] = 7;

       }
    for (int j = 0; j < p; j++)
       for (int i=0; i + j < p; i++) // (1,3,4)
       {
          double w = fop[i] + fop[j] + fop[pm1-i-j];
          Nodes.IntPoint(o).Set4(fop[pm1-i-j]/w, 0.0, fop[i]/w, fop[j]/w);
          dof2tk1[o] = 5;
          dof2tk2[o++] = 8;

       }
    for (int j = 0; j < p; j++)
       for (int i=0; i + j < p; i++) // (2,3,4)
       {
          double w = fop[i] + fop[j] + fop[pm1-i-j];
          Nodes.IntPoint(o).Set4(0.0, fop[pm1-i-j]/w, fop[i]/w, fop[j]/w);
          dof2tk1[o] = 6;
          dof2tk2[o++] = 9;
       }
    
    // facets (see Mesh::GenerateFaces in mesh/mesh.cpp)
    for (int k = 0; k < pm1; k++)
       for (int j = 0; j + k < pm1; j++)
          for (int i = 0; i + j + k < pm1; i++)  // (0,1,2,3)
          {
              //std::cout << "inside facet dof" << std::endl;
             double w = ftop[i] + ftop[j] + ftop[k] + ftop[pm2-i-j-k];
             Nodes.IntPoint(o).Set4(ftop[i]/w, ftop[j]/w, ftop[k]/w, 0.0);
             const IntegrationPoint &ip = Nodes.IntPoint(o);
             dof2tk1[o] = 1;
             dof2tk2[o++] = 10;
             Nodes.IntPoint(o).Set4(ftop[i]/w, ftop[j]/w, ftop[k]/w, 0.0);
             dof2tk1[o] = 0;
             dof2tk2[o++] = 12;
             Nodes.IntPoint(o).Set4(ftop[i]/w, ftop[j]/w, ftop[k]/w, 0.0);
             dof2tk1[o] = 2;
             dof2tk2[o++] = 11;
          }
    for (int k = 0; k < pm1; k++)
       for (int j = 0; j + k < pm1; j++)
          for (int i = 0; i + j + k < pm1; i++)  // (0,2,1,4)
          {
             double w = ftop[i] + ftop[j] + ftop[k] + ftop[pm2-i-j-k];
             Nodes.IntPoint(o).Set4(ftop[j]/w, ftop[i]/w, 0.0, ftop[k]/w);
              const IntegrationPoint &ip = Nodes.IntPoint(o);
             dof2tk1[o] = 0;
             dof2tk2[o++] = 11;
             Nodes.IntPoint(o).Set4(ftop[j]/w, ftop[i]/w, 0.0, ftop[k]/w);
             dof2tk1[o] = 1;
             dof2tk2[o++] = 13;
             Nodes.IntPoint(o).Set4(ftop[j]/w, ftop[i]/w, 0.0, ftop[k]/w);
             dof2tk1[o] = 3;
             dof2tk2[o++] = 10;
          }
    for (int k = 0; k < pm1; k++)
       for (int j = 0; j + k < pm1; j++)
          for (int i = 0; i + j + k < pm1; i++)  // (0,1,3,4)
          {
             double w = ftop[i] + ftop[j] + ftop[k] + ftop[pm2-i-j-k];
             Nodes.IntPoint(o).Set4(ftop[i]/w, 0.0, ftop[j]/w, ftop[k]/w);
              const IntegrationPoint &ip = Nodes.IntPoint(o);
             dof2tk1[o] = 2;
             dof2tk2[o++] = 10;
             Nodes.IntPoint(o).Set4(ftop[i]/w, 0.0, ftop[j]/w, ftop[k]/w);
             dof2tk1[o] = 0;
             dof2tk2[o++] = 13;
             Nodes.IntPoint(o).Set4(ftop[i]/w, 0.0, ftop[j]/w, ftop[k]/w);
             dof2tk1[o] = 3;
             dof2tk2[o++] = 12;
          }
    for (int k = 0; k < pm1; k++)
       for (int j = 0; j + k < pm1; j++)
          for (int i = 0; i + j + k < pm1; i++)  // (0,3,2,4)
          {
             double w = ftop[i] + ftop[j] + ftop[k] + ftop[pm2-i-j-k];
             Nodes.IntPoint(o).Set4(0.0, ftop[j]/w, ftop[i]/w, ftop[k]/w);
              const IntegrationPoint &ip = Nodes.IntPoint(o);
             dof2tk1[o] = 1;
             dof2tk2[o++] = 12;
             Nodes.IntPoint(o).Set4(0.0, ftop[j]/w, ftop[i]/w, ftop[k]/w);
             dof2tk1[o] = 2;
             dof2tk2[o++] = 13;
             Nodes.IntPoint(o).Set4(0.0, ftop[j]/w, ftop[i]/w, ftop[k]/w);
             dof2tk1[o] = 3;
             dof2tk2[o++] = 11;


          }
    for (int k = 0; k < pm1; k++)
       for (int j = 0; j + k < pm1; j++)
          for (int i = 0; i + j + k < pm1; i++)  // (1,2,3,4)
          {
             double w = ftop[i] + ftop[j] + ftop[k] + ftop[pm2-i-j-k];
             Nodes.IntPoint(o).Set4(ftop[pm2-i-j-k]/w, ftop[i]/w, ftop[j]/w, ftop[k]/w);
              const IntegrationPoint &ip = Nodes.IntPoint(o);
             dof2tk1[o] = 5;
             dof2tk2[o++] = 0;
             Nodes.IntPoint(o).Set4(ftop[pm2-i-j-k]/w, ftop[i]/w, ftop[j]/w, ftop[k]/w);
             dof2tk1[o] = 4;
             dof2tk2[o++] = 2;
             Nodes.IntPoint(o).Set4(ftop[pm2-i-j-k]/w, ftop[i]/w, ftop[j]/w, ftop[k]/w);
             dof2tk1[o] = 7;
             dof2tk2[o++] = 1;

          }
    

     //interior bubbles
    for (int l = 0; l < pm2; l++)
       for (int k = 0; k + l < pm2; k++)
          for (int j = 0; j + k + l < pm2; j++)
             for (int i = 0; i + j + k + l < pm2; i++)
             {
                double w = iop[i] + iop[j] + iop[k] + iop[l] + iop[pm3-i-j-k-l];
                Nodes.IntPoint(o).Set4(iop[i]/w, iop[j]/w, iop[k]/w, iop[l]/w);
                dof2tk1[o] = 1;
                dof2tk2[o++] = 10;
                Nodes.IntPoint(o).Set4(iop[i]/w, iop[j]/w, iop[k]/w, iop[l]/w);
                dof2tk1[o] = 0;
                dof2tk2[o++] = 12;
                Nodes.IntPoint(o).Set4(iop[i]/w, iop[j]/w, iop[k]/w, iop[l]/w);
                dof2tk1[o] = 0;
                dof2tk2[o++] = 13;
                Nodes.IntPoint(o).Set4(iop[i]/w, iop[j]/w, iop[k]/w, iop[l]/w);
                dof2tk1[o] = 2;
                dof2tk2[o++] = 11;
                Nodes.IntPoint(o).Set4(iop[i]/w, iop[j]/w, iop[k]/w, iop[l]/w);
                dof2tk1[o] = 3;
                dof2tk2[o++] = 11;
                Nodes.IntPoint(o).Set4(iop[i]/w, iop[j]/w, iop[k]/w, iop[l]/w);
                dof2tk1[o] = 3;
                dof2tk2[o++] = 12;
                 
             }
    
    DenseMatrix T(dof);
    
    // Initialize result to zero
    for (int j = 0; j < dof; j++) {
        for (int n = 0; n < dof; n++) {
            T(n,j) = 0.0;
        }
    }

    DenseMatrix B(4, 4);
    int num_dof = dof;

    double Tensor_B[4][4][num_dof];
    double Tensor_HC[4][4][num_dof];

    DenseMatrix Mat_B(num_dof, dim);
    
    for (int q = 0; q < dof; q++)
    {
        
        const IntegrationPoint &ip = Nodes.IntPoint(q);
        
        const double tm1[4] = {tk1[dof2tk1[q]][0], tk1[dof2tk1[q]][1], tk1[dof2tk1[q]][2], tk1[dof2tk1[q]][3]};
        
        const double tm2[4] = {tk2[dof2tk2[q]][0], tk2[dof2tk2[q]][1], tk2[dof2tk2[q]][2], tk2[dof2tk2[q]][3]};
                
        int o = 0;
        
        //compute barycentric coordinates as function of ip
        std::vector<double> bary_vector{ip.x, ip.y, ip.z, ip.t, (1.0 - ip.x - ip.y - ip.z - ip.t)};
        
        // compute the gradient of the barycentric coords
        std::vector<double> gradL1{1,0,0,0};
        std::vector<double> gradL2{0,1,0,0};
        std::vector<double> gradL3{0,0,1,0};
        std::vector<double> gradL4{0,0,0,1};
        std::vector<double> gradL5{-1,-1,-1,-1};
        
        std::vector<std::vector<double>> gradbary_vector{gradL1, gradL2, gradL3, gradL4, gradL5};
        
        double La, Lb, Lc, Ld, Le;
        
        std::vector<double> grad_La, grad_Lb, grad_Lc, grad_Ld, grad_Le;
        
        // Faces
        for(int a=0; a<5;a++)
        {
            for(int b=0; b<5;b++)
            {
                for(int c=0; c<5;c++)
                {
                    for(int i=0; i<p;i++)
                    {
                        for(int j=0; j<p;j++)
                        {

                            if((a<b)&&(b<c)&&((i+j)<p))
                            {

                                La = bary_vector[a];
                                Lb = bary_vector[b];
                                Lc = bary_vector[c];
                                
                                grad_La = gradbary_vector[a];
                                grad_Lb = gradbary_vector[b];
                                grad_Lc = gradbary_vector[c];
                                
                                // compute polynomials
                                std::vector<double> Legendre_i;
                                double x = Lb;
                                double y = La + Lb;
                                poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                                
                                std::vector<double> Jacobi_j;
                                x = Lc;
                                y = La + Lb + Lc;
                                double alpha = 2*i + 1;
                                poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                                
                                
                                // Scaled Skew-sym outer product
                                DenseMatrix skw_sym_outerprod_1 = skw_sym_outerprod_fnc(grad_Lb, grad_Lc);
                                DenseMatrix skw_sym_outerprod_2 = skw_sym_outerprod_fnc(grad_Lc, grad_La);
                                DenseMatrix skw_sym_outerprod_3 = skw_sym_outerprod_fnc(grad_La, grad_Lb);
                                
                                // Add Basis Funcitons
                                Tensor_B[0][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(0,0) + Lb*skw_sym_outerprod_2(0,0) + Lc*skw_sym_outerprod_3(0,0));
                                Tensor_B[0][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(0,1) + Lb*skw_sym_outerprod_2(0,1) + Lc*skw_sym_outerprod_3(0,1));
                                Tensor_B[0][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(0,2) + Lb*skw_sym_outerprod_2(0,2) + Lc*skw_sym_outerprod_3(0,2));
                                Tensor_B[0][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(0,3) + Lb*skw_sym_outerprod_2(0,3) + Lc*skw_sym_outerprod_3(0,3));
                                
                                
                                Tensor_B[1][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(1,0) + Lb*skw_sym_outerprod_2(1,0) + Lc*skw_sym_outerprod_3(1,0));
                                Tensor_B[1][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(1,1) + Lb*skw_sym_outerprod_2(1,1) + Lc*skw_sym_outerprod_3(1,1));
                                Tensor_B[1][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(1,2) + Lb*skw_sym_outerprod_2(1,2) + Lc*skw_sym_outerprod_3(1,2));
                                Tensor_B[1][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(1,3) + Lb*skw_sym_outerprod_2(1,3) + Lc*skw_sym_outerprod_3(1,3));
                                
                                
                                Tensor_B[2][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(2,0) + Lb*skw_sym_outerprod_2(2,0) + Lc*skw_sym_outerprod_3(2,0));
                                Tensor_B[2][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(2,1) + Lb*skw_sym_outerprod_2(2,1) + Lc*skw_sym_outerprod_3(2,1));
                                Tensor_B[2][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(2,2) + Lb*skw_sym_outerprod_2(2,2) + Lc*skw_sym_outerprod_3(2,2));
                                Tensor_B[2][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(2,3) + Lb*skw_sym_outerprod_2(2,3) + Lc*skw_sym_outerprod_3(2,3));
                                
                                Tensor_B[3][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(3,0) + Lb*skw_sym_outerprod_2(3,0) + Lc*skw_sym_outerprod_3(3,0));
                                Tensor_B[3][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(3,1) + Lb*skw_sym_outerprod_2(3,1) + Lc*skw_sym_outerprod_3(3,1));
                                Tensor_B[3][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(3,2) + Lb*skw_sym_outerprod_2(3,2) + Lc*skw_sym_outerprod_3(3,2));
                                Tensor_B[3][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(3,3) + Lb*skw_sym_outerprod_2(3,3) + Lc*skw_sym_outerprod_3(3,3));
                                
                                
                                o++;
                                
                                
                            }
                        }
                    }
                }
            }
        }
        // end of faces
        
        //Facets
        int a;
        int b;
        int c;
        int d;
        
        for (int f=0; f<5; f++)
        {
            // Define each facet
            if (f==0)
            {
                a = 0;
                b = 1;
                c = 2;
                d = 3;
                
            }
            else if(f==1)
            {
                // Define Facet
                a = 0;
                b = 2;
                c = 1;
                d = 4;
                
            }
            // Define each facet
            else if (f==2)
            {
                // Define Facet
                a = 0;
                b = 1;
                c = 3;
                d = 4;
                
            }
            // Define each facet
            else if (f==3)
            {
                // Define Facet
                a = 0;
                b = 3;
                c = 2;
                d = 4;
                
            }
            // Define each facet
            else if (f==4)
            {
                // Define Facet
                a = 1;
                b = 2;
                c = 3;
                d = 4;
                
            }
            else
            {
                mfem_error("Invaild facet");
            }
            
            
            for(int i=0; i<p;i++)
            {
                for(int j=0; j<p;j++)
                {
                    for(int l=1; l<p;l++)
                    {
                        
                        if((i+j+l)<p)
                        {
                            
                            int Family = 1;
                            
                            if (Family == 1)
                            {
                                
                                // Family I:
                                
                                // Define Barycentric Coordinates
                                La = bary_vector[a];
                                Lb = bary_vector[b];
                                Lc = bary_vector[c];
                                Ld = bary_vector[d];
                                
                                grad_La = gradbary_vector[a];
                                grad_Lb = gradbary_vector[b];
                                grad_Lc = gradbary_vector[c];
                                grad_Ld = gradbary_vector[d];
                                
                                // compute polynomials
                                
                                // compute polynomials
                                std::vector<double> Legendre_i;
                                double x = Lb;
                                double y = La + Lb;
                                poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                                
                                std::vector<double> Jacobi_j;
                                x = Lc;
                                y = La + Lb + Lc;
                                double alpha = 2*i + 1;
                                poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                                
                                
                                std::vector<double> Int_Jacobi_l;
                                x = Ld;
                                y = La + Lb + Lc + Ld;
                                alpha = 2*(i + j + 1);
                                poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                                
                                
                                // Scaled Skew-sym outer product
                                DenseMatrix skw_sym_outerprod_1 = skw_sym_outerprod_fnc(grad_Lb, grad_Lc);
                                DenseMatrix skw_sym_outerprod_2 = skw_sym_outerprod_fnc(grad_Lc, grad_La);
                                DenseMatrix skw_sym_outerprod_3 = skw_sym_outerprod_fnc(grad_La, grad_Lb);
                                
                                
                                // Add Basis Funcitons
                                Tensor_B[0][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,0) + Lb*skw_sym_outerprod_2(0,0) + Lc*skw_sym_outerprod_3(0,0));
                                Tensor_B[0][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,1) + Lb*skw_sym_outerprod_2(0,1) + Lc*skw_sym_outerprod_3(0,1));
                                Tensor_B[0][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,2) + Lb*skw_sym_outerprod_2(0,2) + Lc*skw_sym_outerprod_3(0,2));
                                Tensor_B[0][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,3) + Lb*skw_sym_outerprod_2(0,3) + Lc*skw_sym_outerprod_3(0,3));
                                
                                
                                Tensor_B[1][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,0) + Lb*skw_sym_outerprod_2(1,0) + Lc*skw_sym_outerprod_3(1,0));
                                Tensor_B[1][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,1) + Lb*skw_sym_outerprod_2(1,1) + Lc*skw_sym_outerprod_3(1,1));
                                Tensor_B[1][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,2) + Lb*skw_sym_outerprod_2(1,2) + Lc*skw_sym_outerprod_3(1,2));
                                Tensor_B[1][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,3) + Lb*skw_sym_outerprod_2(1,3) + Lc*skw_sym_outerprod_3(1,3));
                                
                                
                                Tensor_B[2][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,0) + Lb*skw_sym_outerprod_2(2,0) + Lc*skw_sym_outerprod_3(2,0));
                                Tensor_B[2][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,1) + Lb*skw_sym_outerprod_2(2,1) + Lc*skw_sym_outerprod_3(2,1));
                                Tensor_B[2][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,2) + Lb*skw_sym_outerprod_2(2,2) + Lc*skw_sym_outerprod_3(2,2));
                                Tensor_B[2][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,3) + Lb*skw_sym_outerprod_2(2,3) + Lc*skw_sym_outerprod_3(2,3));
                                
                                
                                Tensor_B[3][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,0) + Lb*skw_sym_outerprod_2(3,0) + Lc*skw_sym_outerprod_3(3,0));
                                Tensor_B[3][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,1) + Lb*skw_sym_outerprod_2(3,1) + Lc*skw_sym_outerprod_3(3,1));
                                Tensor_B[3][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,2) + Lb*skw_sym_outerprod_2(3,2) + Lc*skw_sym_outerprod_3(3,2));
                                Tensor_B[3][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,3) + Lb*skw_sym_outerprod_2(3,3) + Lc*skw_sym_outerprod_3(3,3));
                            
                                o++;
                            }
                            
                            Family++;
                            
                            
                            if (Family == 2)
                            {
                                // Define Barycentric Coordinates
                                La = bary_vector[b];
                                Lb = bary_vector[c];
                                Lc = bary_vector[d];
                                Ld = bary_vector[a];
                                
                                grad_La = gradbary_vector[b];
                                grad_Lb = gradbary_vector[c];
                                grad_Lc = gradbary_vector[d];
                                grad_Ld = gradbary_vector[a];
                                
                                // Family II:
                                
                                // compute polynomials
                                std::vector<double> Legendre_i;
                                double x = Lb;
                                double y = La + Lb;
                                poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                                
                                
                                std::vector<double> Jacobi_j;
                                x = Lc;
                                y = La + Lb + Lc;
                                double alpha = 2*i + 1;
                                poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                                
                                
                                std::vector<double> Int_Jacobi_l;
                                x = Ld;
                                y = La + Lb + Lc + Ld;
                                alpha = 2*(i +j +1);
                                poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                                
                                
                                // Scaled Skew-sym outer product
                                DenseMatrix skw_sym_outerprod_1 = skw_sym_outerprod_fnc(grad_Lb, grad_Lc);
                                DenseMatrix skw_sym_outerprod_2 = skw_sym_outerprod_fnc(grad_Lc, grad_La);
                                DenseMatrix skw_sym_outerprod_3 = skw_sym_outerprod_fnc(grad_La, grad_Lb);
                                
                                
                                
                                // Add Basis Funcitons
                                Tensor_B[0][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,0) + Lb*skw_sym_outerprod_2(0,0) + Lc*skw_sym_outerprod_3(0,0));
                                Tensor_B[0][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,1) + Lb*skw_sym_outerprod_2(0,1) + Lc*skw_sym_outerprod_3(0,1));
                                Tensor_B[0][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,2) + Lb*skw_sym_outerprod_2(0,2) + Lc*skw_sym_outerprod_3(0,2));
                                Tensor_B[0][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,3) + Lb*skw_sym_outerprod_2(0,3) + Lc*skw_sym_outerprod_3(0,3));
                                
                                
                                Tensor_B[1][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,0) + Lb*skw_sym_outerprod_2(1,0) + Lc*skw_sym_outerprod_3(1,0));
                                Tensor_B[1][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,1) + Lb*skw_sym_outerprod_2(1,1) + Lc*skw_sym_outerprod_3(1,1));
                                Tensor_B[1][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,2) + Lb*skw_sym_outerprod_2(1,2) + Lc*skw_sym_outerprod_3(1,2));
                                Tensor_B[1][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,3) + Lb*skw_sym_outerprod_2(1,3) + Lc*skw_sym_outerprod_3(1,3));
                                
                                
                                Tensor_B[2][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,0) + Lb*skw_sym_outerprod_2(2,0) + Lc*skw_sym_outerprod_3(2,0));
                                Tensor_B[2][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,1) + Lb*skw_sym_outerprod_2(2,1) + Lc*skw_sym_outerprod_3(2,1));
                                Tensor_B[2][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,2) + Lb*skw_sym_outerprod_2(2,2) + Lc*skw_sym_outerprod_3(2,2));
                                Tensor_B[2][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,3) + Lb*skw_sym_outerprod_2(2,3) + Lc*skw_sym_outerprod_3(2,3));
                                
                                
                                Tensor_B[3][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,0) + Lb*skw_sym_outerprod_2(3,0) + Lc*skw_sym_outerprod_3(3,0));
                                Tensor_B[3][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,1) + Lb*skw_sym_outerprod_2(3,1) + Lc*skw_sym_outerprod_3(3,1));
                                Tensor_B[3][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,2) + Lb*skw_sym_outerprod_2(3,2) + Lc*skw_sym_outerprod_3(3,2));
                                Tensor_B[3][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,3) + Lb*skw_sym_outerprod_2(3,3) + Lc*skw_sym_outerprod_3(3,3));
                        
                                
                                o++;
                                
                            }
                            
                            Family++;
                            
                            if (Family == 3)
                            {
                                
                                // Define Barycentric Coordinates
                                La = bary_vector[c];
                                Lb = bary_vector[d];
                                Lc = bary_vector[a];
                                Ld = bary_vector[b];
                                
                                grad_La = gradbary_vector[c];
                                grad_Lb = gradbary_vector[d];
                                grad_Lc = gradbary_vector[a];
                                grad_Ld = gradbary_vector[b];
                                
                                // Family III:
                                
                                // compute polynomials
                                std::vector<double> Legendre_i;
                                double x = Lb;
                                double y = La + Lb;
                                poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                                
                                
                                std::vector<double> Jacobi_j;
                                x = Lc;
                                y = La + Lb + Lc;
                                double alpha = 2*i + 1;
                                poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                                
                                
                                std::vector<double> Int_Jacobi_l;
                                x = Ld;
                                y = La + Lb + Lc + Ld;
                                alpha = 2*(i +j +1);
                                poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
 
                                // Scaled Skew-sym outer product
                                DenseMatrix skw_sym_outerprod_1 = skw_sym_outerprod_fnc(grad_Lb, grad_Lc);
                                DenseMatrix skw_sym_outerprod_2 = skw_sym_outerprod_fnc(grad_Lc, grad_La);
                                DenseMatrix skw_sym_outerprod_3 = skw_sym_outerprod_fnc(grad_La, grad_Lb);

                                // Add Basis Funcitons
                                Tensor_B[0][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,0) + Lb*skw_sym_outerprod_2(0,0) + Lc*skw_sym_outerprod_3(0,0));
                                Tensor_B[0][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,1) + Lb*skw_sym_outerprod_2(0,1) + Lc*skw_sym_outerprod_3(0,1));
                                Tensor_B[0][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,2) + Lb*skw_sym_outerprod_2(0,2) + Lc*skw_sym_outerprod_3(0,2));
                                Tensor_B[0][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,3) + Lb*skw_sym_outerprod_2(0,3) + Lc*skw_sym_outerprod_3(0,3));
                                
                                
                                Tensor_B[1][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,0) + Lb*skw_sym_outerprod_2(1,0) + Lc*skw_sym_outerprod_3(1,0));
                                Tensor_B[1][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,1) + Lb*skw_sym_outerprod_2(1,1) + Lc*skw_sym_outerprod_3(1,1));
                                Tensor_B[1][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,2) + Lb*skw_sym_outerprod_2(1,2) + Lc*skw_sym_outerprod_3(1,2));
                                Tensor_B[1][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,3) + Lb*skw_sym_outerprod_2(1,3) + Lc*skw_sym_outerprod_3(1,3));
                                
                                
                                Tensor_B[2][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,0) + Lb*skw_sym_outerprod_2(2,0) + Lc*skw_sym_outerprod_3(2,0));
                                Tensor_B[2][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,1) + Lb*skw_sym_outerprod_2(2,1) + Lc*skw_sym_outerprod_3(2,1));
                                Tensor_B[2][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,2) + Lb*skw_sym_outerprod_2(2,2) + Lc*skw_sym_outerprod_3(2,2));
                                Tensor_B[2][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,3) + Lb*skw_sym_outerprod_2(2,3) + Lc*skw_sym_outerprod_3(2,3));
                                
                                
                                Tensor_B[3][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,0) + Lb*skw_sym_outerprod_2(3,0) + Lc*skw_sym_outerprod_3(3,0));
                                Tensor_B[3][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,1) + Lb*skw_sym_outerprod_2(3,1) + Lc*skw_sym_outerprod_3(3,1));
                                Tensor_B[3][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,2) + Lb*skw_sym_outerprod_2(3,2) + Lc*skw_sym_outerprod_3(3,2));
                                Tensor_B[3][3][o] = 0.5 *Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,3) + Lb*skw_sym_outerprod_2(3,3) + Lc*skw_sym_outerprod_3(3,3));
                                
                                
                                o++;
                                
                            }
                            
                        }
                    }
                }
            }
            
        }// end of Facets
        
        //Interiors
        for (int r = 0; r<6; r++)
        {
            if (r ==0)
            {
                //case where (a,b,c,d,e) = (0,1,2,3,4)
                // define lamda
                La = bary_vector[0];
                Lb = bary_vector[1];
                Lc = bary_vector[2];
                Ld = bary_vector[3];
                Le = bary_vector[4];
                
                // define a,b,c,d
                a = 0;
                b = 1;
                c = 2;
                d = 3;
                
            }
            else if (r==1)
            {
                //case where (a,b,c,d,e) = (1,2,3,4,0)
                // define lamda
                La = bary_vector[1];
                Lb = bary_vector[2];
                Lc = bary_vector[3];
                Ld = bary_vector[4];
                Le = bary_vector[0];
                
                // define a,b,c,d
                a = 1;
                b = 2;
                c = 3;
                d = 4;
                
                
            }
            else if (r==2)
            {
                //case where (a,b,c,d,e) = (2,3,4,0,1)
                // define lamda
                La = bary_vector[2];
                Lb = bary_vector[3];
                Lc = bary_vector[4];
                Ld = bary_vector[0];
                Le = bary_vector[1];
                
                // define a,b,c,d
                a = 2;
                b = 3;
                c = 4;
                d = 0;
                
            }
            else if (r==3)
            {
                //case where (a,b,c,d,e) = (3,4,0,1,2)
                // define lamda
                La = bary_vector[3];
                Lb = bary_vector[4];
                Lc = bary_vector[0];
                Ld = bary_vector[1];
                Le = bary_vector[2];
                
                // define a,b,c,d
                a = 3;
                b = 4;
                c = 0;
                d = 1;
                
            }
            else if (r==4)
            {
                //case where (a,b,c,d,e) = (4,0,1,2,3)
                // define lamda
                La = bary_vector[4];
                Lb = bary_vector[0];
                Lc = bary_vector[1];
                Ld = bary_vector[2];
                Le = bary_vector[3];
                
                // define a,b,c,d
                a = 4;
                b = 0;
                c = 1;
                d = 2;
                
                
            }
            else if (r==5)
            {
                //case where (a,b,c,d,e) = (0,1,3,2,4)
                // define lamda
                La = bary_vector[0];
                Lb = bary_vector[1];
                Lc = bary_vector[3];
                Ld = bary_vector[2];
                Le = bary_vector[4];
                
                // define a,b,c,d
                a = 0;
                b = 1;
                c = 3;
                d = 2;
                
            }
            else
            {
                mfem_error("Invaild Bubble");
            }
            
            for(int i=0; i<p;i++)
            {
                for(int j=0; j<p;j++)
                {
                    for(int l=1; l<p;l++)
                    {
                        for (int m=1; m<p; m++)
                        {
                            if((i+j+l+m)<p)
                            {
                                
                                // Define Barycentric Coordinates
                                grad_La = gradbary_vector[a];
                                grad_Lb = gradbary_vector[b];
                                grad_Lc = gradbary_vector[c];
                                grad_Ld = gradbary_vector[d];
                                
                                
                                // compute polynomials
                                std::vector<double> Legendre_i;
                                double x = Lb;
                                double y = La + Lb;
                                poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                                
                                std::vector<double> Jacobi_j;
                                x = Lc;
                                y = La + Lb + Lc;
                                double alpha = 2*i + 1;
                                poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                                
                                
                                std::vector<double> Int_Jacobi_l;
                                x = Ld;
                                y = La + Lb + Lc + Ld;
                                alpha = 2*(i + j + 1);
                                poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                                
                                std::vector<double> Int_Jacobi_m;
                                x = Le;
                                y = 1.0;
                                alpha = 2*(i + j + l);
                                poly1d.CalcIntJacobi(m, x, y, alpha, Int_Jacobi_m);
                                
                                
                                // Scaled Skew-sym outer product
                                DenseMatrix skw_sym_outerprod_1 = skw_sym_outerprod_fnc(grad_Lb, grad_Lc);
                                DenseMatrix skw_sym_outerprod_2 = skw_sym_outerprod_fnc(grad_Lc, grad_La);
                                DenseMatrix skw_sym_outerprod_3 = skw_sym_outerprod_fnc(grad_La, grad_Lb);
                                
                                
                                // Add Basis Funcitons
                                Tensor_B[0][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(0,0) + Lb*skw_sym_outerprod_2(0,0) + Lc*skw_sym_outerprod_3(0,0));
                                Tensor_B[0][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(0,1) + Lb*skw_sym_outerprod_2(0,1) + Lc*skw_sym_outerprod_3(0,1));
                                Tensor_B[0][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(0,2) + Lb*skw_sym_outerprod_2(0,2) + Lc*skw_sym_outerprod_3(0,2));
                                Tensor_B[0][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(0,3) + Lb*skw_sym_outerprod_2(0,3) + Lc*skw_sym_outerprod_3(0,3));
                                
                                
                                Tensor_B[1][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(1,0) + Lb*skw_sym_outerprod_2(1,0) + Lc*skw_sym_outerprod_3(1,0));
                                Tensor_B[1][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(1,1) + Lb*skw_sym_outerprod_2(1,1) + Lc*skw_sym_outerprod_3(1,1));
                                Tensor_B[1][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(1,2) + Lb*skw_sym_outerprod_2(1,2) + Lc*skw_sym_outerprod_3(1,2));
                                Tensor_B[1][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(1,3) + Lb*skw_sym_outerprod_2(1,3) + Lc*skw_sym_outerprod_3(1,3));
                                
                                
                                Tensor_B[2][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(2,0) + Lb*skw_sym_outerprod_2(2,0) + Lc*skw_sym_outerprod_3(2,0));
                                Tensor_B[2][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(2,1) + Lb*skw_sym_outerprod_2(2,1) + Lc*skw_sym_outerprod_3(2,1));
                                Tensor_B[2][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(2,2) + Lb*skw_sym_outerprod_2(2,2) + Lc*skw_sym_outerprod_3(2,2));
                                Tensor_B[2][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(2,3) + Lb*skw_sym_outerprod_2(2,3) + Lc*skw_sym_outerprod_3(2,3));
                                
                                
                                Tensor_B[3][0][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(3,0) + Lb*skw_sym_outerprod_2(3,0) + Lc*skw_sym_outerprod_3(3,0));
                                Tensor_B[3][1][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(3,1) + Lb*skw_sym_outerprod_2(3,1) + Lc*skw_sym_outerprod_3(3,1));
                                Tensor_B[3][2][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(3,2) + Lb*skw_sym_outerprod_2(3,2) + Lc*skw_sym_outerprod_3(3,2));
                                Tensor_B[3][3][o] = 0.5 * Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(3,3) + Lb*skw_sym_outerprod_2(3,3) + Lc*skw_sym_outerprod_3(3,3));
                                
                                o++;
                                
                            }
                        }
                    }
                }
            }
            
        }// End of Interiors
         
        

        // Initialize result to zero
        for (int j = 0; j < 4; j++) {
            for (int n = 0; n < num_dof; n++) {
                Mat_B(n,j) = 0.0;
            }
        }
        

        // Perform contraction over i
        for (int n = 0; n < num_dof; n++) 
        {
            for (int j = 0; j < 4; j++) 
            {
                // Reset Sum
                double sum = 0.0;
                
                for (int i = 0; i < 4; i++)
                {
                    sum += Tensor_B[i][j][n] * tm1[i];
                }
                // Store as transpose
                Mat_B(n, j) = sum;
            }
        }
        Mat_B.Mult(tm2, T.GetColumn(q));
    }
            
    DenseMatrixInverse T_temp(T);
    T_temp.GetInverseMatrix(T_inv); // M_inv now contains M^-1

    //mfem::out << "HCurl_PentatopeElement(" << p << ") : "; T_inv.TestInversion(); // DO NOT USE ****
    
}


void HCurl_PentatopeElement::CalcVShape(const IntegrationPoint &ip,
                                           DenseMatrix &shape) const
{
    
    
    const int p = order;
    int test_order = order;
    int reset_o;
    
#ifdef MFEM_THREAD_SAFE
    Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_t(p + 1),
    shape_l(p + 1);
    DenseMatrix u(Dof, Dim);
#endif
    int size_ip = Nodes.Size();
    
    int o = 0;
    int num_dof = dof;
    u.SetSize(num_dof, dim, dim);
    DenseTensor u_hc;
    u_hc.SetSize(num_dof, dim, dim);
    DenseMatrix shape_pre(num_dof,4*dim);
    
    //compute barycentric coordinates as function of ip
    std::vector<double> bary_vector{ip.x, ip.y, ip.z, ip.t, (1.0 - ip.x - ip.y - ip.z - ip.t)};
    
    
    // compute the gradient of the barycentric coords
    std::vector<double> gradL1{1,0,0,0};
    std::vector<double> gradL2{0,1,0,0};
    std::vector<double> gradL3{0,0,1,0};
    std::vector<double> gradL4{0,0,0,1};
    std::vector<double> gradL5{-1,-1,-1,-1};
    
    std::vector<std::vector<double>> gradbary_vector{gradL1, gradL2, gradL3, gradL4, gradL5};
    
    double La, Lb, Lc, Ld, Le;
    
    std::vector<double> grad_La, grad_Lb, grad_Lc, grad_Ld, grad_Le;
    
    
    // Lamda function for computing the scaled skew-symmetric-outer-product of two four vectors
    auto skw_sym_outerprod_fnc = [&](const std::vector<double>& vec1, const std::vector<double>& vec2)
    {

        DenseMatrix result(4,4);
        for (int i = 0; i < vec1.size(); ++i)
        {
            for (int j = 0; j < vec2.size(); ++j)
            {
                // Forward Product
                double outer_prod_1 = vec1[i] * vec2[j];
                
                // "Backward" Product
                double outer_prod_2 = vec2[i] * vec1[j];
                
                // Compute Result
                result(i,j) = (outer_prod_1 - outer_prod_2);
            }
        }
        
        return result;
    }; // end of lamda function

    

    // Faces
    for(int a=0; a<5;a++)
    {
        for(int b=0; b<5;b++)
        {
            for(int c=0; c<5;c++)
            {
                for(int i=0; i<p;i++)
                {
                    for(int j=0; j<p;j++)
                    {
                        if((a<b)&&(b<c)&&((i+j)<p))
                        {
                            
                            La = bary_vector[a];
                            Lb = bary_vector[b];
                            Lc = bary_vector[c];

                            
                            grad_La = gradbary_vector[a];
                            grad_Lb = gradbary_vector[b];
                            grad_Lc = gradbary_vector[c];
                            
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                            

                            
                            std::vector<double> Jacobi_j;
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i + 1;
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            

                            
                            // Scaled Skew-sym outer product
                            DenseMatrix skw_sym_outerprod_1 = skw_sym_outerprod_fnc(grad_Lb, grad_Lc);

                            DenseMatrix skw_sym_outerprod_2 = skw_sym_outerprod_fnc(grad_Lc, grad_La);

                            DenseMatrix skw_sym_outerprod_3 = skw_sym_outerprod_fnc(grad_La, grad_Lb);

                            
                            // Add Basis Funcitons
                            
                            u(o, 0, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(0,0) + Lb*skw_sym_outerprod_2(0,0) + Lc*skw_sym_outerprod_3(0,0));
                                                        
                            u(o, 0, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(0,1) + Lb*skw_sym_outerprod_2(0,1) + Lc*skw_sym_outerprod_3(0,1));
                                                        
                            u(o, 0, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(0,2) + Lb*skw_sym_outerprod_2(0,2) + Lc*skw_sym_outerprod_3(0,2));
                                                        
                            u(o, 0, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(0,3) + Lb*skw_sym_outerprod_2(0,3) + Lc*skw_sym_outerprod_3(0,3));
                            
                            
                            u(o, 1, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(1,0) + Lb*skw_sym_outerprod_2(1,0) + Lc*skw_sym_outerprod_3(1,0));
                            
                            u(o, 1, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(1,1) + Lb*skw_sym_outerprod_2(1,1) + Lc*skw_sym_outerprod_3(1,1));
                            
                            u(o, 1, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(1,2) + Lb*skw_sym_outerprod_2(1,2) + Lc*skw_sym_outerprod_3(1,2));
                            
                            u(o, 1, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(1,3) + Lb*skw_sym_outerprod_2(1,3) + Lc*skw_sym_outerprod_3(1,3));
                            

                            u(o, 2, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(2,0) + Lb*skw_sym_outerprod_2(2,0) + Lc*skw_sym_outerprod_3(2,0));
                            
                            u(o, 2, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(2,1) + Lb*skw_sym_outerprod_2(2,1) + Lc*skw_sym_outerprod_3(2,1));
                            
                            u(o, 2, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(2,2) + Lb*skw_sym_outerprod_2(2,2) + Lc*skw_sym_outerprod_3(2,2));
                            
                            u(o, 2, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(2,3) + Lb*skw_sym_outerprod_2(2,3) + Lc*skw_sym_outerprod_3(2,3));
                            
                            
                            u(o, 3, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(3,0) + Lb*skw_sym_outerprod_2(3,0) + Lc*skw_sym_outerprod_3(3,0));
                            
                            u(o, 3, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(3,1) + Lb*skw_sym_outerprod_2(3,1) + Lc*skw_sym_outerprod_3(3,1));
                            
                            u(o, 3, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(3,2) + Lb*skw_sym_outerprod_2(3,2) + Lc*skw_sym_outerprod_3(3,2));
                            
                            u(o, 3, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (La*skw_sym_outerprod_1(3,3) + Lb*skw_sym_outerprod_2(3,3) + Lc*skw_sym_outerprod_3(3,3));
                            
                            o++;
                            
                            
                        }
                    }
                }
            }
        }
    }
    // end of faces
    
    //Facets
    int a;
    int b;
    int c;
    int d;
    
    for (int f=0; f<5; f++)
    {
        // Define each facet
        if (f==0)
        {
            a = 0;
            b = 1;
            c = 2;
            d = 3;
            
        }
        else if(f==1)
        {
            // Define Facet
            a = 0;
            b = 2;
            c = 1;
            d = 4;
            
        }
        // Define each facet
        else if (f==2)
        {
            // Define Facet
            a = 0;
            b = 1;
            c = 3;
            d = 4;
            
        }
        // Define each facet
        else if (f==3)
        {
            // Define Facet
            a = 0;
            b = 3;
            c = 2;
            d = 4;
            
        }
        // Define each facet
        else if (f==4)
        {
            // Define Facet
            a = 1;
            b = 2;
            c = 3;
            d = 4;
            
        }
        else
        {
            mfem_error("Invaild facet");
        }
        
        
        //int num_facesH = 0;
        for(int i=0; i<p;i++)
        {
            for(int j=0; j<p;j++)
            {
                for(int l=1; l<p;l++)
                {
                    if((i+j+l)<p)
                    {
                        int Family = 1;
                        
                        if (Family == 1)
                        {
                            // Family I:
                            
                            // Define Barycentric Coordinates
                            La = bary_vector[a];
                            Lb = bary_vector[b];
                            Lc = bary_vector[c];
                            Ld = bary_vector[d];
                            
                            grad_La = gradbary_vector[a];
                            grad_Lb = gradbary_vector[b];
                            grad_Lc = gradbary_vector[c];
                            grad_Ld = gradbary_vector[d];
                            
                            
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcLegendreShifted(i, x, y, Legendre_i);

                            
                            std::vector<double> Jacobi_j;
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i + 1;
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            
 
                            
                            std::vector<double> Int_Jacobi_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i + j + 1);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                                

                            
                            // Scaled Skew-sym outer product
                            DenseMatrix skw_sym_outerprod_1 = skw_sym_outerprod_fnc(grad_Lb, grad_Lc);
                            DenseMatrix skw_sym_outerprod_2 = skw_sym_outerprod_fnc(grad_Lc, grad_La);
                            DenseMatrix skw_sym_outerprod_3 = skw_sym_outerprod_fnc(grad_La, grad_Lb);

                            
                            // Add Basis Funcitons
                            u(o, 0, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,0) + Lb*skw_sym_outerprod_2(0,0) + Lc*skw_sym_outerprod_3(0,0));
                            u(o, 0, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,1) + Lb*skw_sym_outerprod_2(0,1) + Lc*skw_sym_outerprod_3(0,1));
                            u(o, 0, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,2) + Lb*skw_sym_outerprod_2(0,2) + Lc*skw_sym_outerprod_3(0,2));
                            u(o, 0, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,3) + Lb*skw_sym_outerprod_2(0,3) + Lc*skw_sym_outerprod_3(0,3));
                            
                            
                            u(o, 1, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,0) + Lb*skw_sym_outerprod_2(1,0) + Lc*skw_sym_outerprod_3(1,0));
                            u(o, 1, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,1) + Lb*skw_sym_outerprod_2(1,1) + Lc*skw_sym_outerprod_3(1,1));
                            u(o, 1, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,2) + Lb*skw_sym_outerprod_2(1,2) + Lc*skw_sym_outerprod_3(1,2));
                            u(o, 1, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,3) + Lb*skw_sym_outerprod_2(1,3) + Lc*skw_sym_outerprod_3(1,3));
                            
                            
                            u(o, 2, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,0) + Lb*skw_sym_outerprod_2(2,0) + Lc*skw_sym_outerprod_3(2,0));
                            u(o, 2, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,1) + Lb*skw_sym_outerprod_2(2,1) + Lc*skw_sym_outerprod_3(2,1));
                            u(o, 2, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,2) + Lb*skw_sym_outerprod_2(2,2) + Lc*skw_sym_outerprod_3(2,2));
                            u(o, 2, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,3) + Lb*skw_sym_outerprod_2(2,3) + Lc*skw_sym_outerprod_3(2,3));
                            
                            
                            u(o, 3, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,0) + Lb*skw_sym_outerprod_2(3,0) + Lc*skw_sym_outerprod_3(3,0));
                            u(o, 3, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,1) + Lb*skw_sym_outerprod_2(3,1) + Lc*skw_sym_outerprod_3(3,1));
                            u(o, 3, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,2) + Lb*skw_sym_outerprod_2(3,2) + Lc*skw_sym_outerprod_3(3,2));
                            u(o, 3, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,3) + Lb*skw_sym_outerprod_2(3,3) + Lc*skw_sym_outerprod_3(3,3));
                            


                            o++;
                        }
                        
                        Family++;
                        
                        
                        if (Family == 2)
                        {
                            // Define Barycentric Coordinates
                            La = bary_vector[b];
                            Lb = bary_vector[c];
                            Lc = bary_vector[d];
                            Ld = bary_vector[a];
                            
                            grad_La = gradbary_vector[b];
                            grad_Lb = gradbary_vector[c];
                            grad_Lc = gradbary_vector[d];
                            grad_Ld = gradbary_vector[a];
                            
                            // Family II:
                            
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                            
                            
                            std::vector<double> Jacobi_j;
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i + 1;
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            
                            
                            std::vector<double> Int_Jacobi_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i +j +1);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                                
                            
                            // Scaled Skew-sym outer product
                            DenseMatrix skw_sym_outerprod_1 = skw_sym_outerprod_fnc(grad_Lb, grad_Lc);
                            DenseMatrix skw_sym_outerprod_2 = skw_sym_outerprod_fnc(grad_Lc, grad_La);
                            DenseMatrix skw_sym_outerprod_3 = skw_sym_outerprod_fnc(grad_La, grad_Lb);
                            

                            // Add Basis Funcitons
                            u(o, 0, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,0) + Lb*skw_sym_outerprod_2(0,0) + Lc*skw_sym_outerprod_3(0,0));
                            u(o, 0, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,1) + Lb*skw_sym_outerprod_2(0,1) + Lc*skw_sym_outerprod_3(0,1));
                            u(o, 0, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,2) + Lb*skw_sym_outerprod_2(0,2) + Lc*skw_sym_outerprod_3(0,2));
                            u(o, 0, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,3) + Lb*skw_sym_outerprod_2(0,3) + Lc*skw_sym_outerprod_3(0,3));
                            
                            
                            u(o, 1, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,0) + Lb*skw_sym_outerprod_2(1,0) + Lc*skw_sym_outerprod_3(1,0));
                            u(o, 1, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,1) + Lb*skw_sym_outerprod_2(1,1) + Lc*skw_sym_outerprod_3(1,1));
                            u(o, 1, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,2) + Lb*skw_sym_outerprod_2(1,2) + Lc*skw_sym_outerprod_3(1,2));
                            u(o, 1, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,3) + Lb*skw_sym_outerprod_2(1,3) + Lc*skw_sym_outerprod_3(1,3));
                            
                            
                            u(o, 2, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,0) + Lb*skw_sym_outerprod_2(2,0) + Lc*skw_sym_outerprod_3(2,0));
                            u(o, 2, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,1) + Lb*skw_sym_outerprod_2(2,1) + Lc*skw_sym_outerprod_3(2,1));
                            u(o, 2, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,2) + Lb*skw_sym_outerprod_2(2,2) + Lc*skw_sym_outerprod_3(2,2));
                            u(o, 2, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,3) + Lb*skw_sym_outerprod_2(2,3) + Lc*skw_sym_outerprod_3(2,3));
                            
                            
                            u(o, 3, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,0) + Lb*skw_sym_outerprod_2(3,0) + Lc*skw_sym_outerprod_3(3,0));
                            u(o, 3, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,1) + Lb*skw_sym_outerprod_2(3,1) + Lc*skw_sym_outerprod_3(3,1));
                            u(o, 3, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,2) + Lb*skw_sym_outerprod_2(3,2) + Lc*skw_sym_outerprod_3(3,2));
                            u(o, 3, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,3) + Lb*skw_sym_outerprod_2(3,3) + Lc*skw_sym_outerprod_3(3,3));
                            

                            

                            o++;
                            
                        }
                        
                        Family++;
                        
                        if (Family == 3)
                        {
                            
                            // Define Barycentric Coordinates
                            La = bary_vector[c];
                            Lb = bary_vector[d];
                            Lc = bary_vector[a];
                            Ld = bary_vector[b];
                            
                            grad_La = gradbary_vector[c];
                            grad_Lb = gradbary_vector[d];
                            grad_Lc = gradbary_vector[a];
                            grad_Ld = gradbary_vector[b];
                            
                            // Family III:
                            
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                            
                            
                            std::vector<double> Jacobi_j;
                            x = Lc;
                            y = La + Lb+ Lc;
                            double alpha = 2*i + 1;
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);

                            
                            std::vector<double> Int_Jacobi_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i +j +1);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            

                            
                            // Scaled Skew-sym outer product
                            DenseMatrix skw_sym_outerprod_1 = skw_sym_outerprod_fnc(grad_Lb, grad_Lc);
                            DenseMatrix skw_sym_outerprod_2 = skw_sym_outerprod_fnc(grad_Lc, grad_La);
                            DenseMatrix skw_sym_outerprod_3 = skw_sym_outerprod_fnc(grad_La, grad_Lb);

                            // Add Basis Funcitons
                            u(o, 0, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,0) + Lb*skw_sym_outerprod_2(0,0) + Lc*skw_sym_outerprod_3(0,0));
                            u(o, 0, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,1) + Lb*skw_sym_outerprod_2(0,1) + Lc*skw_sym_outerprod_3(0,1));
                            u(o, 0, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,2) + Lb*skw_sym_outerprod_2(0,2) + Lc*skw_sym_outerprod_3(0,2));
                            u(o, 0, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(0,3) + Lb*skw_sym_outerprod_2(0,3) + Lc*skw_sym_outerprod_3(0,3));
    
                            u(o, 1, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,0) + Lb*skw_sym_outerprod_2(1,0) + Lc*skw_sym_outerprod_3(1,0));
                            u(o, 1, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,1) + Lb*skw_sym_outerprod_2(1,1) + Lc*skw_sym_outerprod_3(1,1));
                            u(o, 1, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,2) + Lb*skw_sym_outerprod_2(1,2) + Lc*skw_sym_outerprod_3(1,2));
                            u(o, 1, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(1,3) + Lb*skw_sym_outerprod_2(1,3) + Lc*skw_sym_outerprod_3(1,3));
                            
                            u(o, 2, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,0) + Lb*skw_sym_outerprod_2(2,0) + Lc*skw_sym_outerprod_3(2,0));
                            u(o, 2, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(2,1) + Lb*skw_sym_outerprod_2(2,1) + Lc*skw_sym_outerprod_3(2,1));
                            u(o, 2, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,2) + Lb*skw_sym_outerprod_2(2,2) + Lc*skw_sym_outerprod_3(2,2));
                            u(o, 2, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]* (La*skw_sym_outerprod_1(2,3) + Lb*skw_sym_outerprod_2(2,3) + Lc*skw_sym_outerprod_3(2,3));
                            
                            u(o, 3, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,0) + Lb*skw_sym_outerprod_2(3,0) + Lc*skw_sym_outerprod_3(3,0));
                            u(o, 3, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,1) + Lb*skw_sym_outerprod_2(3,1) + Lc*skw_sym_outerprod_3(3,1));
                            u(o, 3, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,2) + Lb*skw_sym_outerprod_2(3,2) + Lc*skw_sym_outerprod_3(3,2));
                            u(o, 3, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (La*skw_sym_outerprod_1(3,3) + Lb*skw_sym_outerprod_2(3,3) + Lc*skw_sym_outerprod_3(3,3));
                            

                            
                            o++;
                            
                        }
                        
                    }
                }
            }
        }
    }// end of Facets
    
    //Interiors
    for (int r = 0; r<6; r++)
    {
        if (r ==0)
        {
            //case where (a,b,c,d,e) = (0,1,2,3,4)
            // define lamda
            La = bary_vector[0];
            Lb = bary_vector[1];
            Lc = bary_vector[2];
            Ld = bary_vector[3];
            Le = bary_vector[4];
            
            // define a,b,c,d
            a = 0;
            b = 1;
            c = 2;
            d = 3;
            
        }
        else if (r==1)
        {
            //case where (a,b,c,d,e) = (1,2,3,4,0)
            // define lamda
            La = bary_vector[1];
            Lb = bary_vector[2];
            Lc = bary_vector[3];
            Ld = bary_vector[4];
            Le = bary_vector[0];
            
            // define a,b,c,d
            a = 1;
            b = 2;
            c = 3;
            d = 4;
            
            
        }
        else if (r==2)
        {
            //case where (a,b,c,d,e) = (2,3,4,0,1)
            // define lamda
            La = bary_vector[2];
            Lb = bary_vector[3];
            Lc = bary_vector[4];
            Ld = bary_vector[0];
            Le = bary_vector[1];
            
            // define a,b,c,d
            a = 2;
            b = 3;
            c = 4;
            d = 0;
            
        }
        else if (r==3)
        {
            //case where (a,b,c,d,e) = (3,4,0,1,2)
            // define lamda
            La = bary_vector[3];
            Lb = bary_vector[4];
            Lc = bary_vector[0];
            Ld = bary_vector[1];
            Le = bary_vector[2];
            
            // define a,b,c,d
            a = 3;
            b = 4;
            c = 0;
            d = 1;
            
        }
        else if (r==4)
        {
            //case where (a,b,c,d,e) = (4,0,1,2,3)
            // define lamda
            La = bary_vector[4];
            Lb = bary_vector[0];
            Lc = bary_vector[1];
            Ld = bary_vector[2];
            Le = bary_vector[3];
            
            // define a,b,c,d
            a = 4;
            b = 0;
            c = 1;
            d = 2;
            
            
        }
        else if (r==5)
        {
            //case where (a,b,c,d,e) = (0,1,3,2,4)
            // define lamda
            La = bary_vector[0];
            Lb = bary_vector[1];
            Lc = bary_vector[3];
            Ld = bary_vector[2];
            Le = bary_vector[4];
            
            // define a,b,c,d
            a = 0;
            b = 1;
            c = 3;
            d = 2;
            
        }
        else
        {
            mfem_error("Invaild Bubble");
        }
        
        for(int i=0; i<p;i++)
        {
            for(int j=0; j<p;j++)
            {
                for(int l=1; l<p;l++)
                {
                    for (int m=1; m<p; m++)
                    {
                        if((i+j+l+m)<p)
                        {
                            
                            // Define Barycentric Coordinates
                            grad_La = gradbary_vector[a];
                            grad_Lb = gradbary_vector[b];
                            grad_Lc = gradbary_vector[c];
                            grad_Ld = gradbary_vector[d];
                            
                            
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                            
                            std::vector<double> Jacobi_j;
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i + 1;
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            
                            
                            std::vector<double> Int_Jacobi_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i + j + 1);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            
                            std::vector<double> Int_Jacobi_m;
                            x = Le;
                            y = 1.0;
                            alpha = 2*(i + j + l);
                            poly1d.CalcIntJacobi(m, x, y, alpha, Int_Jacobi_m);
                            
                            
                            // Scaled Skew-sym outer product
                            DenseMatrix skw_sym_outerprod_1 = skw_sym_outerprod_fnc(grad_Lb, grad_Lc);
                            DenseMatrix skw_sym_outerprod_2 = skw_sym_outerprod_fnc(grad_Lc, grad_La);
                            DenseMatrix skw_sym_outerprod_3 = skw_sym_outerprod_fnc(grad_La, grad_Lb);
                            
                            
                            // Add Basis Funcitons
                            u(o, 0, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(0,0) + Lb*skw_sym_outerprod_2(0,0) + Lc*skw_sym_outerprod_3(0,0));
                            u(o, 0, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(0,1) + Lb*skw_sym_outerprod_2(0,1) + Lc*skw_sym_outerprod_3(0,1));
                            u(o, 0, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(0,2) + Lb*skw_sym_outerprod_2(0,2) + Lc*skw_sym_outerprod_3(0,2));
                            u(o, 0, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(0,3) + Lb*skw_sym_outerprod_2(0,3) + Lc*skw_sym_outerprod_3(0,3));
    
                            u(o, 1, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(1,0) + Lb*skw_sym_outerprod_2(1,0) + Lc*skw_sym_outerprod_3(1,0));
                            u(o, 1, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(1,1) + Lb*skw_sym_outerprod_2(1,1) + Lc*skw_sym_outerprod_3(1,1));
                            u(o, 1, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(1,2) + Lb*skw_sym_outerprod_2(1,2) + Lc*skw_sym_outerprod_3(1,2));
                            u(o, 1, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(1,3) + Lb*skw_sym_outerprod_2(1,3) + Lc*skw_sym_outerprod_3(1,3));
                            
                            u(o, 2, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(2,0) + Lb*skw_sym_outerprod_2(2,0) + Lc*skw_sym_outerprod_3(2,0));
                            u(o, 2, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(2,1) + Lb*skw_sym_outerprod_2(2,1) + Lc*skw_sym_outerprod_3(2,1));
                            u(o, 2, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(2,2) + Lb*skw_sym_outerprod_2(2,2) + Lc*skw_sym_outerprod_3(2,2));
                            u(o, 2, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(2,3) + Lb*skw_sym_outerprod_2(2,3) + Lc*skw_sym_outerprod_3(2,3));
                            
                            u(o, 3, 0) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(3,0) + Lb*skw_sym_outerprod_2(3,0) + Lc*skw_sym_outerprod_3(3,0));
                            u(o, 3, 1) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(3,1) + Lb*skw_sym_outerprod_2(3,1) + Lc*skw_sym_outerprod_3(3,1));
                            u(o, 3, 2) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(3,2) + Lb*skw_sym_outerprod_2(3,2) + Lc*skw_sym_outerprod_3(3,2));
                            u(o, 3, 3) = Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * (La*skw_sym_outerprod_1(3,3) + Lb*skw_sym_outerprod_2(3,3) + Lc*skw_sym_outerprod_3(3,3));
                            

                            
                            o++;
                            
                        }
                    }
                }
            }
        }
        
    }// End of Interiors
    
    DenseTensor shape_Ten(dof,dim,dim);
    
    for (int l=0; l < num_dof; l++)
    {
        for (int j=0; j<dim; j++)
        {
            for (int k=0; k<dim; k++)
            {
                
                double sum = 0.0;
                for (int m=0; m < num_dof; m++)
                {
                    sum += T_inv(l,m) * u(m,j,k); // original
                }
                shape_Ten(l,j,k) = sum;
            }
        }
    }
    // Now unroll and store in shape
    for (int l=0; l < num_dof; l++)
    {
        int N = 0;
        for (int j=0; j<dim; j++)
        {
            for (int k=0; k<dim; k++)
            {
                shape(l,N) = 0.5 * shape_Ten(l,j,k); // 0.5 factor include becuase of 2-form proxy
                N++;
            }
            
        }
    }

}




void HCurl_PentatopeElement::CalcCurlShape(const IntegrationPoint &ip,
                                                 DenseMatrix &Curlshape) const
{
    const int p = order;
    
#ifdef MFEM_THREAD_SAFE
    Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_t(p + 1),
    shape_l(p + 1);
    DenseMatrix u(Dof, Dim);
#endif
    int size_ip = Nodes.Size();
    Curlu.SetSize(dof, int(4));
    int o = 0;
    int num_dof = dof;
    
    //compute barycentric coordinates as function of ip
    std::vector<double> bary_vector{ip.x, ip.y, ip.z, ip.t, (1.0 - ip.x - ip.y - ip.z - ip.t)};
    
    // compute the gradient of the barycentric coords
    std::vector<double> gradL1{1,0,0,0};
    std::vector<double> gradL2{0,1,0,0};
    std::vector<double> gradL3{0,0,1,0};
    std::vector<double> gradL4{0,0,0,1};
    std::vector<double> gradL5{-1,-1,-1,-1};
    
    std::vector<std::vector<double>> gradbary_vector{gradL1, gradL2, gradL3, gradL4, gradL5};
    
    double La, Lb, Lc, Ld, Le;
    
    std::vector<double> grad_La, grad_Lb, grad_Lc, grad_Ld, grad_Le;
    

    double s_factor = 0.5;

    // Lamda function for computing the skew-symmetric-outer-product of two four vectors
    auto skw_sym_outerprod_fnc = [&](const std::vector<double>& vec1, const std::vector<double>& vec2)
    {

        DenseMatrix result(4,4);
        for (int i = 0; i < vec1.size(); ++i)
        {
            for (int j = 0; j < vec2.size(); ++j)
            {
                // Forward Product
                double outer_prod_1 = vec1[i] * vec2[j];
                
                // "Backward" Product
                double outer_prod_2 = vec2[i] * vec1[j];
                
                // Compute Result
                result(i,j) = (outer_prod_1 - outer_prod_2);
            }
        }
        
        return result;
    }; // end of lamda function
    
    // Faces
    for(int a=0; a<5;a++)
    {
        for(int b=0; b<5;b++)
        {
            for(int c=0; c<5;c++)
            {
                for(int i=0; i<p;i++)
                {
                    for(int j=0; j<p;j++)
                    {
                        if((a<b)&&(b<c)&&((i+j)<p))
                        {
                            
                            La = bary_vector[a];
                            Lb = bary_vector[b];
                            Lc = bary_vector[c];

                            
                            grad_La = gradbary_vector[a];
                            grad_Lb = gradbary_vector[b];
                            grad_Lc = gradbary_vector[c];
                            
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            std::vector<double> Legendre_i_dx;
                            std::vector<double> Legendre_i_dt;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcScaledLegendreDerivative(i, x, y, Legendre_i, Legendre_i_dx, Legendre_i_dt);

                    
                            std::vector<double> Jacobi_j;
                            std::vector<double> Jacobi_j_ref;
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i + 1;
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            // Derivative Jacobi Poly
                            std::vector<double> Jacobi_j_dx;
                            std::vector<double> Jacobi_j_dt;
                            poly1d.CalcScaledJacobiDerivative(j, alpha, x, y, Jacobi_j_ref, Jacobi_j_dx, Jacobi_j_dt);
                                                        
                            
                            // grad of Legendre and Jacobi Polynomials (scalar part)
                            double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0])) * Jacobi_j[Jacobi_j.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[0] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[0] + grad_Lb[0] + grad_Lc[0]) );
                            
                            
                            double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1])) * Jacobi_j[Jacobi_j.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[1] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[1] + grad_Lb[1] + grad_Lc[1]) );
                            
                            
                            double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2])) * Jacobi_j[Jacobi_j.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[2] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[2] + grad_Lb[2] + grad_Lc[2]) );
                            
                            
                            double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3])) * Jacobi_j[Jacobi_j.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[3] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[3] + grad_Lb[3] + grad_Lc[3]) );
                            
                            
                            // curl of skw-sym part
                            std::vector<double> F_vec(4);
                            // x-component
                            F_vec[0] = 2.0 * ( grad_La[1]*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + grad_Lb[1]*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + grad_Lc[1]*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[2]*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + grad_Lb[2]*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + grad_Lc[2]*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[3]*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + grad_Lb[3]*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + grad_Lc[3]*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) );
                            
                            
                            // y-component
                            F_vec[1] = -2.0 * ( grad_La[0]*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + grad_Lb[0]*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + grad_Lc[0]*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[2]*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + grad_Lb[2]*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + grad_Lc[2]*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[3]*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + grad_Lb[3]*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + grad_Lc[3]*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) );
                            
                            
                            // z-component
                            F_vec[2] = 2.0 * ( grad_La[0]*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + grad_Lb[0]*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + grad_Lc[0]*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[1]*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + grad_Lb[1]*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + grad_Lc[1]*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[3]*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + grad_Lb[3]*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + grad_Lc[3]*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // t-component
                            F_vec[3] = -2.0 * ( grad_La[0]*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + grad_Lb[0]*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + grad_Lc[0]*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) )
                             
                            
                            + 2.0 * ( grad_La[1]*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + grad_Lb[1]*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + grad_Lc[1]*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) )
                            
                            
                            - 2.0 * ( grad_La[2]*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + grad_Lb[2]*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + grad_Lc[2]*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // grad(scalar-part) X skw-sym_mat
                            
                            std::vector<double> McrossN(4);
                            // x-component
                            McrossN[0] = 2.0 * dscalar_y * ( La*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + Lb*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + Lc*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_z * ( La*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + Lb*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + Lc*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_t * ( La*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + Lb*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + Lc*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) );
                            
                            
                            // y-component
                            McrossN[1] = -2.0 * dscalar_x * ( La*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + Lb*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + Lc*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_z * ( La*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + Lb*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + Lc*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_t * ( La*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + Lb*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + Lc*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) );
                            
                            
                            // z-component
                            McrossN[2] = 2.0 * dscalar_x * ( La*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + Lb*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + Lc*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_y * ( La*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + Lb*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + Lc*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_t * ( La*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + Lb*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + Lc*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // t-component
                            McrossN[3] = -2.0 * dscalar_x * ( La*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + Lb*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + Lc*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) )
                             
                            
                            + 2.0 * dscalar_y * ( La*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + Lb*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + Lc*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) )
                            
                            
                            - 2.0 * dscalar_z * ( La*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + Lb*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + Lc*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            
                            // Add Basis Funcitons
                            
                            Curlu(o, 0) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * F_vec[0]) + McrossN[0] );
                            
                            Curlu(o, 1) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * F_vec[1]) + McrossN[1] );

                            Curlu(o, 2) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * F_vec[2]) + McrossN[2] );

                            Curlu(o, 3) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * F_vec[3]) + McrossN[3] );
                            

                            o++;
                            
                            
                        }
                    }
                }
            }
        }
    }
    // end of faces
    
    //Facets
    int a;
    int b;
    int c;
    int d;
    
    for (int f=0; f<5; f++)
    {
        // Define each facet
        if (f==0)
        {
            a = 0;
            b = 1;
            c = 2;
            d = 3;
            
        }
        else if(f==1)
        {
            // Define Facet
            a = 0;
            b = 2;
            c = 1;
            d = 4;
            
        }
        // Define each facet
        else if (f==2)
        {
            // Define Facet
            a = 0;
            b = 1;
            c = 3;
            d = 4;
            
        }
        // Define each facet
        else if (f==3)
        {
            // Define Facet
            a = 0;
            b = 3;
            c = 2;
            d = 4;
            
        }
        // Define each facet
        else if (f==4)
        {
            // Define Facet
            a = 1;
            b = 2;
            c = 3;
            d = 4;
            
        }
        else
        {
            mfem_error("Invaild facet");
        }
        
        
        for(int i=0; i<p;i++)
        {
            for(int j=0; j<p;j++)
            {
                for(int l=1; l<p;l++)
                {
                    if((i+j+l)<p)
                    {
                        int Family = 1;
                        
                        if (Family == 1)
                        {
                            // Family I:
                            
                            La = bary_vector[a];
                            Lb = bary_vector[b];
                            Lc = bary_vector[c];
                            Ld = bary_vector[d];


                            
                            grad_La = gradbary_vector[a];
                            grad_Lb = gradbary_vector[b];
                            grad_Lc = gradbary_vector[c];
                            grad_Ld = gradbary_vector[d];

                            
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            std::vector<double> Legendre_i_dx;
                            std::vector<double> Legendre_i_dt;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcScaledLegendreDerivative(i, x, y, Legendre_i, Legendre_i_dx, Legendre_i_dt);

                            std::vector<double> Jacobi_j;
                            std::vector<double> Jacobi_j_ref;
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i + 1;
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            // Derivative Jacobi Poly
                            std::vector<double> Jacobi_j_dx;
                            std::vector<double> Jacobi_j_dt;
                            poly1d.CalcScaledJacobiDerivative(j, alpha, x, y, Jacobi_j_ref, Jacobi_j_dx, Jacobi_j_dt);
                            
                            
                            std::vector<double> Int_Jacobi_l;
                            std::vector<double> Jacobi_l;
                            std::vector<double> R_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i + j + 1);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            poly1d.CalcJacobi(l, x, y, alpha, Jacobi_l);
                            poly1d.CalcRJacobi(l, x, y, alpha, R_l);
                            
                            
                            // grad of Legendre and Jacobi Polynomials (scalar part)
                            double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[0] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[0] + grad_Lb[0] + grad_Lc[0]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[0] + R_l[R_l.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0] + grad_Ld[0]));
                            
                            
                            double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[1] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[1] + grad_Lb[1] + grad_Lc[1]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[1] + R_l[R_l.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1] + grad_Ld[1]));
                            
                            
                            double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[2] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[2] + grad_Lb[2] + grad_Lc[2]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[2] + R_l[R_l.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2] + grad_Ld[2]));
                            
                            
                            double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[3] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[3] + grad_Lb[3] + grad_Lc[3]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[3] + R_l[R_l.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3] + grad_Ld[3]));
                            
  
                            // curl of skw-sym part
                            
                            std::vector<double> F_vec(4);
                            // x-component
                            F_vec[0] = 2.0 * ( grad_La[1]*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + grad_Lb[1]*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + grad_Lc[1]*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[2]*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + grad_Lb[2]*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + grad_Lc[2]*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[3]*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + grad_Lb[3]*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + grad_Lc[3]*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) );
                            
                            
                            // y-component
                            F_vec[1] = -2.0 * ( grad_La[0]*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + grad_Lb[0]*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + grad_Lc[0]*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[2]*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + grad_Lb[2]*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + grad_Lc[2]*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[3]*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + grad_Lb[3]*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + grad_Lc[3]*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) );
                            
                            
                            // z-component
                            F_vec[2] = 2.0 * ( grad_La[0]*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + grad_Lb[0]*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + grad_Lc[0]*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[1]*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + grad_Lb[1]*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + grad_Lc[1]*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[3]*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + grad_Lb[3]*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + grad_Lc[3]*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // t-component
                            F_vec[3] = -2.0 * ( grad_La[0]*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + grad_Lb[0]*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + grad_Lc[0]*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) )
                             
                            
                            + 2.0 * ( grad_La[1]*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + grad_Lb[1]*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + grad_Lc[1]*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) )
                            
                            
                            - 2.0 * ( grad_La[2]*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + grad_Lb[2]*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + grad_Lc[2]*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            

                            // grad(scalar-part) X skw-sym_mat
                            
                            std::vector<double> McrossN(4);
                            // x-component
                            McrossN[0] = 2.0 * dscalar_y * ( La*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + Lb*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + Lc*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_z * ( La*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + Lb*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + Lc*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_t * ( La*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + Lb*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + Lc*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) );
                            
                            
                            // y-component
                            McrossN[1] = -2.0 * dscalar_x * ( La*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + Lb*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + Lc*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_z * ( La*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + Lb*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + Lc*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_t * ( La*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + Lb*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + Lc*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) );
                            
                            
                            // z-component
                            McrossN[2] = 2.0 * dscalar_x * ( La*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + Lb*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + Lc*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_y * ( La*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + Lb*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + Lc*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_t * ( La*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + Lb*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + Lc*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // t-component
                            McrossN[3] = -2.0 * dscalar_x * ( La*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + Lb*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + Lc*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) )
                             
                            
                            + 2.0 * dscalar_y * ( La*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + Lb*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + Lc*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) )
                            
                            
                            - 2.0 * dscalar_z * ( La*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + Lb*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + Lc*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // Add Basis Funcitons
                            
                            Curlu(o, 0) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[0]) + McrossN[0] );
                            
                            Curlu(o, 1) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[1]) + McrossN[1] );

                            Curlu(o, 2) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[2]) + McrossN[2] );

                            Curlu(o, 3) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[3]) + McrossN[3] );
                            

                            o++;
                        }
                        
                        Family++;
                        
                        
                        if (Family == 2)
                        {
                            
                            // Family II:

                            
                            // Define Barycentric Coordinates
                            La = bary_vector[b];
                            Lb = bary_vector[c];
                            Lc = bary_vector[d];
                            Ld = bary_vector[a];
                            
                            grad_La = gradbary_vector[b];
                            grad_Lb = gradbary_vector[c];
                            grad_Lc = gradbary_vector[d];
                            grad_Ld = gradbary_vector[a];
                            

                            // compute polynomials
                            std::vector<double> Legendre_i;
                            std::vector<double> Legendre_i_dx;
                            std::vector<double> Legendre_i_dt;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcScaledLegendreDerivative(i, x, y, Legendre_i, Legendre_i_dx, Legendre_i_dt);

                            std::vector<double> Jacobi_j;
                            std::vector<double> Jacobi_j_ref;
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i + 1;
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            // Derivative Jacobi Poly
                            std::vector<double> Jacobi_j_dx;
                            std::vector<double> Jacobi_j_dt;
                            poly1d.CalcScaledJacobiDerivative(j, alpha, x, y, Jacobi_j_ref, Jacobi_j_dx, Jacobi_j_dt);
                            
                            
                            std::vector<double> Int_Jacobi_l;
                            std::vector<double> Jacobi_l;
                            std::vector<double> R_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i + j + 1);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            poly1d.CalcJacobi(l, x, y, alpha, Jacobi_l);
                            poly1d.CalcRJacobi(l, x, y, alpha, R_l);
                            
                            
                            
                            // grad of Legendre and Jacobi Polynomials (scalar part)
                            double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[0] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[0] + grad_Lb[0] + grad_Lc[0]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[0] + R_l[R_l.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0] + grad_Ld[0]));
                            
                            
                            double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[1] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[1] + grad_Lb[1] + grad_Lc[1]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[1] + R_l[R_l.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1] + grad_Ld[1]));
                            
                            
                            double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[2] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[2] + grad_Lb[2] + grad_Lc[2]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[2] + R_l[R_l.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2] + grad_Ld[2]));
                            
                            
                            double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[3] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[3] + grad_Lb[3] + grad_Lc[3]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[3] + R_l[R_l.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3] + grad_Ld[3]));
                            
                            
                            
                            // curl of skw-sym part
                            
                            std::vector<double> F_vec(4);
                            // x-component
                            F_vec[0] = 2.0 * ( grad_La[1]*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + grad_Lb[1]*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + grad_Lc[1]*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[2]*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + grad_Lb[2]*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + grad_Lc[2]*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[3]*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + grad_Lb[3]*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + grad_Lc[3]*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) );
                            
                            
                            // y-component
                            F_vec[1] = -2.0 * ( grad_La[0]*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + grad_Lb[0]*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + grad_Lc[0]*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[2]*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + grad_Lb[2]*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + grad_Lc[2]*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[3]*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + grad_Lb[3]*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + grad_Lc[3]*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) );
                            
                            
                            // z-component
                            F_vec[2] = 2.0 * ( grad_La[0]*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + grad_Lb[0]*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + grad_Lc[0]*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[1]*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + grad_Lb[1]*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + grad_Lc[1]*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[3]*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + grad_Lb[3]*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + grad_Lc[3]*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // t-component
                            F_vec[3] = -2.0 * ( grad_La[0]*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + grad_Lb[0]*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + grad_Lc[0]*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) )
                             
                            
                            + 2.0 * ( grad_La[1]*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + grad_Lb[1]*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + grad_Lc[1]*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) )
                            
                            
                            - 2.0 * ( grad_La[2]*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + grad_Lb[2]*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + grad_Lc[2]*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // grad(scalar-part) X skw-sym_mat
                            
                            std::vector<double> McrossN(4);
                            // x-component
                            McrossN[0] = 2.0 * dscalar_y * ( La*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + Lb*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + Lc*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_z * ( La*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + Lb*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + Lc*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_t * ( La*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + Lb*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + Lc*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) );
                            
                            
                            // y-component
                            McrossN[1] = -2.0 * dscalar_x * ( La*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + Lb*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + Lc*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_z * ( La*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + Lb*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + Lc*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_t * ( La*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + Lb*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + Lc*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) );
                            
                            
                            // z-component
                            McrossN[2] = 2.0 * dscalar_x * ( La*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + Lb*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + Lc*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_y * ( La*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + Lb*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + Lc*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_t * ( La*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + Lb*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + Lc*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // t-component
                            McrossN[3] = -2.0 * dscalar_x * ( La*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + Lb*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + Lc*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) )
                             
                            
                            + 2.0 * dscalar_y * ( La*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + Lb*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + Lc*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) )
                            
                            
                            - 2.0 * dscalar_z * ( La*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + Lb*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + Lc*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            
                            // Add Basis Funcitons
                            
                            Curlu(o, 0) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[0]) + McrossN[0] );
                            
                            Curlu(o, 1) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[1]) + McrossN[1] );

                            Curlu(o, 2) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[2]) + McrossN[2] );

                            Curlu(o, 3) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[3]) + McrossN[3] );
                            

                            o++;
                        }
                        
                        Family++;
                        
                        if (Family == 3)
                        {
                            // Family III:

                            
                            // Define Barycentric Coordinates
                            La = bary_vector[c];
                            Lb = bary_vector[d];
                            Lc = bary_vector[a];
                            Ld = bary_vector[b];
                            
                            grad_La = gradbary_vector[c];
                            grad_Lb = gradbary_vector[d];
                            grad_Lc = gradbary_vector[a];
                            grad_Ld = gradbary_vector[b];
                            
                            
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            std::vector<double> Legendre_i_dx;
                            std::vector<double> Legendre_i_dt;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcScaledLegendreDerivative(i, x, y, Legendre_i, Legendre_i_dx, Legendre_i_dt);


                            std::vector<double> Jacobi_j;
                            std::vector<double> Jacobi_j_ref;
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i + 1;
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            // Derivative Jacobi Poly
                            std::vector<double> Jacobi_j_dx;
                            std::vector<double> Jacobi_j_dt;
                            poly1d.CalcScaledJacobiDerivative(j, alpha, x, y, Jacobi_j_ref, Jacobi_j_dx, Jacobi_j_dt);
                            
                            
                            std::vector<double> Int_Jacobi_l;
                            std::vector<double> Jacobi_l;
                            std::vector<double> R_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i + j + 1);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            poly1d.CalcJacobi(l, x, y, alpha, Jacobi_l);
                            poly1d.CalcRJacobi(l, x, y, alpha, R_l);
                            

                            
                            // grad of Legendre and Jacobi Polynomials (scalar part)
                            double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[0] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[0] + grad_Lb[0] + grad_Lc[0]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[0] + R_l[R_l.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0] + grad_Ld[0]));
                            
                            
                            double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[1] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[1] + grad_Lb[1] + grad_Lc[1]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[1] + R_l[R_l.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1] + grad_Ld[1]));
                            
                            
                            double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[2] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[2] + grad_Lb[2] + grad_Lc[2]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[2] + R_l[R_l.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2] + grad_Ld[2]));
                            
                            
                            double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[3] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[3] + grad_Lb[3] + grad_Lc[3]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[3] + R_l[R_l.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3] + grad_Ld[3]));
                            
                            
                            
                            // curl of skw-sym part
                            
                            std::vector<double> F_vec(4);
                            // x-component
                            F_vec[0] = 2.0 * ( grad_La[1]*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + grad_Lb[1]*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + grad_Lc[1]*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[2]*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + grad_Lb[2]*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + grad_Lc[2]*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[3]*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + grad_Lb[3]*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + grad_Lc[3]*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) );
                            
                            
                            // y-component
                            F_vec[1] = -2.0 * ( grad_La[0]*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + grad_Lb[0]*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + grad_Lc[0]*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[2]*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + grad_Lb[2]*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + grad_Lc[2]*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[3]*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + grad_Lb[3]*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + grad_Lc[3]*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) );
                            
                            
                            // z-component
                            F_vec[2] = 2.0 * ( grad_La[0]*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + grad_Lb[0]*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + grad_Lc[0]*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[1]*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + grad_Lb[1]*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + grad_Lc[1]*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[3]*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + grad_Lb[3]*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + grad_Lc[3]*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // t-component
                            F_vec[3] = -2.0 * ( grad_La[0]*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + grad_Lb[0]*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + grad_Lc[0]*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) )
                             
                            
                            + 2.0 * ( grad_La[1]*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + grad_Lb[1]*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + grad_Lc[1]*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) )
                            
                            
                            - 2.0 * ( grad_La[2]*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + grad_Lb[2]*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + grad_Lc[2]*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // grad(scalar-part) X skw-sym_mat
                            
                            std::vector<double> McrossN(4);
                            // x-component
                            McrossN[0] = 2.0 * dscalar_y * ( La*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + Lb*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + Lc*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_z * ( La*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + Lb*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + Lc*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_t * ( La*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + Lb*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + Lc*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) );
                            
                            
                            // y-component
                            McrossN[1] = -2.0 * dscalar_x * ( La*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + Lb*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + Lc*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_z * ( La*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + Lb*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + Lc*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_t * ( La*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + Lb*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + Lc*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) );
                            
                            
                            // z-component
                            McrossN[2] = 2.0 * dscalar_x * ( La*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + Lb*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + Lc*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_y * ( La*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + Lb*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + Lc*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_t * ( La*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + Lb*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + Lc*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // t-component
                            McrossN[3] = -2.0 * dscalar_x * ( La*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + Lb*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + Lc*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) )
                             
                            
                            + 2.0 * dscalar_y * ( La*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + Lb*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + Lc*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) )
                            
                            
                            - 2.0 * dscalar_z * ( La*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + Lb*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + Lc*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            
                            // Add Basis Funcitons
                            
                            Curlu(o, 0) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[0]) + McrossN[0] );
                            
                            Curlu(o, 1) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[1]) + McrossN[1] );

                            Curlu(o, 2) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[2]) + McrossN[2] );

                            Curlu(o, 3) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * F_vec[3]) + McrossN[3] );
                            

                            o++;
                            
                        }
                        
                    }
                }
            }
        }
    }// end of Facets
    
    //Interiors
    for (int r = 0; r<6; r++)
    {
        if (r ==0)
        {
            //case where (a,b,c,d,e) = (0,1,2,3,4)
            // define lamda
            La = bary_vector[0];
            Lb = bary_vector[1];
            Lc = bary_vector[2];
            Ld = bary_vector[3];
            Le = bary_vector[4];
            
            // define grad(lamda)
            grad_La = gradbary_vector[0];
            grad_Lb = gradbary_vector[1];
            grad_Lc = gradbary_vector[2];
            grad_Ld = gradbary_vector[3];
            grad_Le = gradbary_vector[4];
            
            // define a,b,c,d
            a = 0;
            b = 1;
            c = 2;
            d = 3;
            
        }
        else if (r==1)
        {
            //case where (a,b,c,d,e) = (1,2,3,4,0)
            // define lamda
            La = bary_vector[1];
            Lb = bary_vector[2];
            Lc = bary_vector[3];
            Ld = bary_vector[4];
            Le = bary_vector[0];
            
            // define grad(lamda)
            grad_La = gradbary_vector[1];
            grad_Lb = gradbary_vector[2];
            grad_Lc = gradbary_vector[3];
            grad_Ld = gradbary_vector[4];
            grad_Le = gradbary_vector[0];
            
            // define a,b,c,d
            a = 1;
            b = 2;
            c = 3;
            d = 4;
            
            
        }
        else if (r==2)
        {
            //case where (a,b,c,d,e) = (2,3,4,0,1)
            // define lamda
            La = bary_vector[2];
            Lb = bary_vector[3];
            Lc = bary_vector[4];
            Ld = bary_vector[0];
            Le = bary_vector[1];
            
            // define grad(lamda)
            grad_La = gradbary_vector[2];
            grad_Lb = gradbary_vector[3];
            grad_Lc = gradbary_vector[4];
            grad_Ld = gradbary_vector[0];
            grad_Le = gradbary_vector[1];
            
            // define a,b,c,d
            a = 2;
            b = 3;
            c = 4;
            d = 0;
            
        }
        else if (r==3)
        {
            //case where (a,b,c,d,e) = (3,4,0,1,2)
            // define lamda
            La = bary_vector[3];
            Lb = bary_vector[4];
            Lc = bary_vector[0];
            Ld = bary_vector[1];
            Le = bary_vector[2];
            
            // define grad(lamda)
            grad_La = gradbary_vector[3];
            grad_Lb = gradbary_vector[4];
            grad_Lc = gradbary_vector[0];
            grad_Ld = gradbary_vector[1];
            grad_Le = gradbary_vector[2];
            
            // define a,b,c,d
            a = 3;
            b = 4;
            c = 0;
            d = 1;
            
        }
        else if (r==4)
        {
            //case where (a,b,c,d,e) = (4,0,1,2,3)
            // define lamda
            La = bary_vector[4];
            Lb = bary_vector[0];
            Lc = bary_vector[1];
            Ld = bary_vector[2];
            Le = bary_vector[3];
            
            // define grad(lamda)
            grad_La = gradbary_vector[4];
            grad_Lb = gradbary_vector[0];
            grad_Lc = gradbary_vector[1];
            grad_Ld = gradbary_vector[2];
            grad_Le = gradbary_vector[3];
            
            // define a,b,c,d
            a = 4;
            b = 0;
            c = 1;
            d = 2;
            
            
        }
        else if (r==5)
        {
            //case where (a,b,c,d,e) = (0,1,3,2,4)
            // define lamda
            La = bary_vector[0];
            Lb = bary_vector[1];
            Lc = bary_vector[3];
            Ld = bary_vector[2];
            Le = bary_vector[4];
            
            // define grad(lamda)
            grad_La = gradbary_vector[0];
            grad_Lb = gradbary_vector[1];
            grad_Lc = gradbary_vector[3];
            grad_Ld = gradbary_vector[2];
            grad_Le = gradbary_vector[4];
            
            // define a,b,c,d
            a = 0;
            b = 1;
            c = 3;
            d = 2;
            
        }
        else
        {
            mfem_error("Invaild Bubble");
        }
        
        for(int i=0; i<p;i++)
        {
            for(int j=0; j<p;j++)
            {
                for(int l=1; l<p;l++)
                {
                    for (int m=1; m<p; m++)
                    {
                        if((i+j+l+m)<p)
                        {
                            
                            
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            std::vector<double> Legendre_i_dx;
                            std::vector<double> Legendre_i_dt;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcScaledLegendreDerivative(i, x, y, Legendre_i, Legendre_i_dx, Legendre_i_dt);

                        
                            std::vector<double> Jacobi_j;
                            std::vector<double> Jacobi_j_ref;
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i + 1;
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            // Derivative Jacobi Poly
                            std::vector<double> Jacobi_j_dx;
                            std::vector<double> Jacobi_j_dt;
                            poly1d.CalcScaledJacobiDerivative(j, alpha, x, y, Jacobi_j_ref, Jacobi_j_dx, Jacobi_j_dt);
                            
                            
                            std::vector<double> Int_Jacobi_l;
                            std::vector<double> Jacobi_l;
                            std::vector<double> R_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i + j + 1);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            poly1d.CalcJacobi(l, x, y, alpha, Jacobi_l);
                            poly1d.CalcRJacobi(l, x, y, alpha, R_l);
                            
                            std::vector<double> Int_Jacobi_m;
                            std::vector<double> Jacobi_m;
                            x = Le;
                            y = 1.0;
                            alpha = 2*(i + j + l);
                            poly1d.CalcIntJacobi(m, x, y, alpha, Int_Jacobi_m);
                            poly1d.CalcJacobi(m, x, y, alpha, Jacobi_m);

                            
                            
                            // grad of Legendre and Jacobi Polynomials (scalar part)
                            double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[0] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[0] + grad_Lb[0] + grad_Lc[0]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[0] + R_l[R_l.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0] + grad_Ld[0])) * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2]*grad_Le[0]);
                            
                            
                            double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] *Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[1] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[1] + grad_Lb[1] + grad_Lc[1]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[1] + R_l[R_l.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1] + grad_Ld[1])) * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2]*grad_Le[1]);
                            
                            
                            double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] *Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[2] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[2] + grad_Lb[2] + grad_Lc[2]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[2] + R_l[R_l.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2] + grad_Ld[2])) * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2]*grad_Le[2]);
                            
                            
                            double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3])) * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] *Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j_dx[Jacobi_j_dx.size()-1]*grad_Lc[3] + Jacobi_j_dt[Jacobi_j_dt.size()-1] * (grad_La[3] + grad_Lb[3] + grad_Lc[3]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[3] + R_l[R_l.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3] + grad_Ld[3])) * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2]*grad_Le[3]);
                            
                            

                            
                            // curl of skw-sym part
                            
                            std::vector<double> F_vec(4);
                            // x-component
                            F_vec[0] = 2.0 * ( grad_La[1]*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + grad_Lb[1]*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + grad_Lc[1]*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[2]*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + grad_Lb[2]*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + grad_Lc[2]*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[3]*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + grad_Lb[3]*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + grad_Lc[3]*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) );
                            
                            
                            // y-component
                            F_vec[1] = -2.0 * ( grad_La[0]*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + grad_Lb[0]*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + grad_Lc[0]*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[2]*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + grad_Lb[2]*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + grad_Lc[2]*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[3]*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + grad_Lb[3]*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + grad_Lc[3]*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) );
                            
                            
                            // z-component
                            F_vec[2] = 2.0 * ( grad_La[0]*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + grad_Lb[0]*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + grad_Lc[0]*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            - 2.0 * ( grad_La[1]*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + grad_Lb[1]*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + grad_Lc[1]*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            + 2.0 * ( grad_La[3]*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + grad_Lb[3]*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + grad_Lc[3]*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // t-component
                            F_vec[3] = -2.0 * ( grad_La[0]*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + grad_Lb[0]*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + grad_Lc[0]*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) )
                             
                            
                            + 2.0 * ( grad_La[1]*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + grad_Lb[1]*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + grad_Lc[1]*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) )
                            
                            
                            - 2.0 * ( grad_La[2]*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + grad_Lb[2]*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + grad_Lc[2]*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            

                            // grad(scalar-part) X skw-sym_mat
                            
                            std::vector<double> McrossN(4);
                            // x-component
                            McrossN[0] = 2.0 * dscalar_y * ( La*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + Lb*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + Lc*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_z * ( La*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + Lb*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + Lc*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_t * ( La*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + Lb*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + Lc*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) );
                            
                            
                            // y-component
                            McrossN[1] = -2.0 * dscalar_x * ( La*(grad_Lb[2]*grad_Lc[3] - grad_Lc[2]*grad_Lb[3]) + Lb*(grad_Lc[2]*grad_La[3] - grad_La[2]*grad_Lc[3]) + Lc*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_z * ( La*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + Lb*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + Lc*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_t * ( La*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + Lb*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + Lc*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) );
                            
                            
                            // z-component
                            McrossN[2] = 2.0 * dscalar_x * ( La*(grad_Lb[1]*grad_Lc[3] - grad_Lc[1]*grad_Lb[3]) + Lb*(grad_Lc[1]*grad_La[3] - grad_La[1]*grad_Lc[3]) + Lc*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3]) )
                            
                            
                            - 2.0 * dscalar_y * ( La*(grad_Lb[0]*grad_Lc[3] - grad_Lc[0]*grad_Lb[3]) + Lb*(grad_Lc[0]*grad_La[3] - grad_La[0]*grad_Lc[3]) + Lc*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3]) )
                            
                            
                            + 2.0 * dscalar_t * ( La*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + Lb*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + Lc*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // t-component
                            McrossN[3] = -2.0 * dscalar_x * ( La*(grad_Lb[1]*grad_Lc[2] - grad_Lc[1]*grad_Lb[2]) + Lb*(grad_Lc[1]*grad_La[2] - grad_La[1]*grad_Lc[2]) + Lc*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2]) )
                             
                            
                            + 2.0 * dscalar_y * ( La*(grad_Lb[0]*grad_Lc[2] - grad_Lc[0]*grad_Lb[2]) + Lb*(grad_Lc[0]*grad_La[2] - grad_La[0]*grad_Lc[2]) + Lc*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2]) )
                            
                            
                            - 2.0 * dscalar_z * ( La*(grad_Lb[0]*grad_Lc[1] - grad_Lc[0]*grad_Lb[1]) + Lb*(grad_Lc[0]*grad_La[1] - grad_La[0]*grad_Lc[1]) + Lc*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1]) );
                            
                            
                            // Add Basis Funcitons
                            
                            Curlu(o, 0) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * F_vec[0]) + McrossN[0] );
                            
                            Curlu(o, 1) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * F_vec[1]) + McrossN[1] );

                            Curlu(o, 2) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * F_vec[2]) + McrossN[2] );

                            Curlu(o, 3) = 0.5 * ( (Legendre_i[Legendre_i.size()-1] * Jacobi_j[Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * F_vec[3]) + McrossN[3] );
                        
                            o++;
                            
                        }
                    }
                }
            }
        }
        
    }// End of Interiors

    
    
    Curlshape = 0.0;
    for (int i=0; i < num_dof; i++)
    {
        for (int j=0; j<dim; j++)
        {
            for (int k=0; k<num_dof; k++)
            {
                Curlshape(i,j) +=  T_inv(i,k) * Curlu(k,j);
            }
        }
    }
                

}

void HCurl_PentatopeElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
    
    Vector v(6);
    double t1[4]; Vector t1i(t1, 4);
    double t2[4]; Vector t2i(t2, 4);
    Vector Mt(4);
    DenseMatrix mat(4,4); mat = 0.0;

    int dof_num = FiniteElement::dof;
    
    dofs.SetSize(dof_num); dofs = 0.0;
    for (int k = 0; k < dof_num; k++)
    {

       Trans.SetIntPoint (&Nodes.IntPoint (k));
       const DenseMatrix &J = Trans.Jacobian();

       vc.Eval(v, Trans, Nodes.IntPoint (k));
        
        // New Proxy
                             mat(0,1) =  v(0); mat(0,2) =  v(1); mat(0,3) =  v(2);
          mat(1,0) = v(6);                     mat(1,2) =  v(3); mat(1,3) =  v(4);
          mat(2,0) = v(7);  mat(2,1) = v(8);                     mat(2,3) =  v(5);
          mat(3,0) = v(9);  mat(3,1) = v(10); mat(3,2) = v(11);
        
        
       J.Mult(tk1[dof2tk1[k]],t1);
       J.Mult(tk2[dof2tk2[k]],t2);
                
       mat.Mult(t2i, Mt);
        
       dofs(k) = t1i * Mt;
    }

}

//void HCurl_PentatopeElement::Project(const FiniteElement &fe,
//                                        ElementTransformation &Trans,
//                                        DenseMatrix &Id) const
//{
//   int dim = fe.GetDim();
//   int dof = fe.GetDof();
//
//   Id.SetSize(10,6*dof); Id = 0.0;
//
//   double t1[4]; Vector t1i(t1, 4);
//   double t2[4]; Vector t2i(t2, 4);
//   Vector Mt(4);
//   DenseMatrix mat(4,4); mat = 0.0;
//
//   int indI[6] = {0,0,0,1,1,2};
//   int indJ[6] = {1,2,3,2,3,3};
//
//   Vector shape(dof);
//   for (int k = 0; k < 10; k++)
//   {
//      Trans.SetIntPoint(&Nodes.IntPoint(k));
//      const DenseMatrix &J = Trans.Jacobian();
//      fe.CalcShape(Nodes.IntPoint(k), shape);
//
//      J.Mult(tk1[k],t1);
//      J.Mult(tk2[k],t2);
//
//      for (int di=0; di<6; di++)
//      {
//         for (int j=0; j<dof; j++)
//         {
//            mat = 0.0;
//            mat(indI[di], indJ[di]) =  shape(j);
//            mat(indJ[di], indI[di]) = -shape(j);
//
//            mat.Mult(t2i, Mt);
//
//            Id(k, di*dof+j) = t1i * Mt;
//         }
//      }
//   }
//}

void HCurl_PentatopeElement::ProjectDivSkew(const FiniteElement& fe,
                                               ElementTransformation& Trans, DenseMatrix& DivSkew)
{
    mfem_error("SkwGrad_PentatopeElement_Fuentes::ProjectDivSkew not implemented");
    
}
} // end of namespace
