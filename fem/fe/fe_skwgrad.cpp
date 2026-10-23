//
//  fe_skgrad.cpp
//  mfem
//
//  Created by psaber on 6/20/25.
//

#include "fe_skwgrad.hpp"
#include "face_map_utils.hpp"
#include "../coefficient.hpp"
#include <fstream>


namespace mfem
{

using namespace std;

const double HSkwGrad_PentatopeElement::tk[40] =
{ 1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1,  -1,1,0,0,  -1,0,1,0,  0,-1,1,0,  -1,0,0,1,  0,-1,0,1,  0,0,-1,1 };

const double HSkwGrad_PentatopeElement::c = 1./5.;

HSkwGrad_PentatopeElement::HSkwGrad_PentatopeElement(const int p)
: VectorFiniteElement(4, Geometry::PENTATOPE, p*(p + 2)*(p + 3)*(p + 4)/6,
                      p, H_SkwGrad, FunctionSpace::Pk),
dof2tk(dof), doftrans(p)
{
    
    const real_t *eop = poly1d.OpenPoints(p - 1);
    const real_t *fop = (p > 1) ? poly1d.OpenPoints(p - 2) : NULL;
    const real_t *ftop = (p > 2) ? poly1d.OpenPoints(p - 3) : NULL;
    const real_t *iop = (p > 3) ? poly1d.OpenPoints(p - 4) : NULL;
    
    const int pm1 = p - 1, pm2 = p - 2, pm3 = p - 3, pm4 = p -4;

    
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
    u.SetSize(dof, dim);
#else
    Vector shape_x(p), shape_y(p), shape_z(p), shape_t(p),
    shape_l(p);
#endif
    // edges (see Tetrahedron::edges in mesh/tetrahedron.cpp)
    int o = 0;
    for (int i = 0; i < p; i++)  // (0,1)
    {
       Nodes.IntPoint(o).Set4(eop[i], 0.0, 0.0, 0.0);
        dof2tk[o++] = 0;
    }
    for (int i = 0; i < p; i++)  // (0,2)
    {
       Nodes.IntPoint(o).Set4(0.0, eop[i], 0.0, 0.0);
        dof2tk[o++] = 1;
    }
    for (int i = 0; i < p; i++)  // (0,3)
    {
       Nodes.IntPoint(o).Set4(0.0, 0.0, eop[i], 0.0);
       dof2tk[o++] = 2;
    }
    for (int i = 0; i < p; i++)  // (0,4)
    {
       Nodes.IntPoint(o).Set4(0.0, 0.0, 0.0, eop[i]);
       dof2tk[o++] = 3;
    }
    for (int i = 0; i < p; i++)  // (1,2)
    {
       Nodes.IntPoint(o).Set4(eop[pm1-i], eop[i], 0.0, 0.0);
       dof2tk[o++] = 4;
    }
    for (int i = 0; i < p; i++)  // (1,3)
    {
       Nodes.IntPoint(o).Set4(eop[pm1-i], 0.0, eop[i], 0.0);
       dof2tk[o++] = 5;
    }
    for (int i = 0; i < p; i++)  // (1,4)
    {
       Nodes.IntPoint(o).Set4(eop[pm1-i], 0.0, 0.0, eop[i]);
       dof2tk[o++] = 7;
    }
    for (int i = 0; i < p; i++)  // (2,3)
    {
       Nodes.IntPoint(o).Set4(0.0, eop[pm1-i], eop[i], 0.0);
       dof2tk[o++] = 6;
    }
    for (int i = 0; i < p; i++)  // (2,4)
    {
       Nodes.IntPoint(o).Set4(0.0, eop[pm1-i], 0.0, eop[i]);
       dof2tk[o++] = 8;
    }
    for (int i = 0; i < p; i++)  // (3,4)
    {
       Nodes.IntPoint(o).Set4(0.0, 0.0, eop[pm1-i], eop[i]);
       dof2tk[o++] = 9;
    }

    // faces (see Mesh::GeneratePlanars in mesh/mesh.cpp)
    for (int j = 0; j <= pm2; j++)
       for (int i=0; i + j <= pm2; i++) // (0,1,2)
       {
          double w = fop[i] + fop[j] + fop[pm2-i-j];
          Nodes.IntPoint(o).Set4(fop[i]/w, fop[j]/w, 0.0, 0.0);
          dof2tk[o++] = 0;
          Nodes.IntPoint(o).Set4(fop[i]/w, fop[j]/w, 0.0, 0.0);
          dof2tk[o++] = 1;
       }
    for (int j = 0; j <= pm2; j++)
       for (int i=0; i + j <= pm2; i++) // (0,1,3)
       {
          double w = fop[i] + fop[j] + fop[pm2-i-j];
          Nodes.IntPoint(o).Set4(fop[i]/w, 0.0, fop[j]/w, 0.0);
          dof2tk[o++] = 0;
          Nodes.IntPoint(o).Set4(fop[i]/w, 0.0, fop[j]/w, 0.0);
          dof2tk[o++] = 2;
       }
    for (int j = 0; j <= pm2; j++)
       for (int i=0; i + j <= pm2; i++) // (0,1,4)
       {
          double w = fop[i] + fop[j] + fop[pm2-i-j];
          Nodes.IntPoint(o).Set4(fop[i]/w, 0.0, 0.0, fop[j]/w);
          dof2tk[o++] = 0;
          Nodes.IntPoint(o).Set4(fop[i]/w, 0.0, 0.0, fop[j]/w);
          dof2tk[o++] = 3;
       }
    for (int j = 0; j <= pm2; j++)
       for (int i=0; i + j <= pm2; i++) // (0,2,3)
       {
          double w = fop[i] + fop[j] + fop[pm2-i-j];
          Nodes.IntPoint(o).Set4(0.0, fop[i]/w, fop[j]/w, 0.0);
          dof2tk[o++] = 1;
          Nodes.IntPoint(o).Set4(0.0, fop[i]/w, fop[j]/w, 0.0);
          dof2tk[o++] = 2;
       }
    for (int j = 0; j <= pm2; j++)
       for (int i=0; i + j <= pm2; i++) // (0,2,4)
       {
          double w = fop[i] + fop[j] + fop[pm2-i-j];
          Nodes.IntPoint(o).Set4(0.0, fop[i]/w, 0.0, fop[j]/w);
          dof2tk[o++] = 1;
          Nodes.IntPoint(o).Set4(0.0, fop[i]/w, 0.0, fop[j]/w);
          dof2tk[o++] = 3;
       }
    for (int j = 0; j <= pm2; j++)
       for (int i=0; i + j <= pm2; i++) // (0,3,4)
       {
          double w = fop[i] + fop[j] + fop[pm2-i-j];
          Nodes.IntPoint(o).Set4(0.0, 0.0, fop[i]/w, fop[j]/w);
          dof2tk[o++] = 2;
          Nodes.IntPoint(o).Set4(0.0, 0.0, fop[i]/w, fop[j]/w);
          dof2tk[o++] = 3;
       }
    for (int j = 0; j <= pm2; j++)
       for (int i=0; i + j <= pm2; i++) // (1,2,3)
       {
          double w = fop[i] + fop[j] + fop[pm2-i-j];
          Nodes.IntPoint(o).Set4(fop[pm2-i-j]/w, fop[i]/w, fop[j]/w, 0.0);
          dof2tk[o++] = 4;
          Nodes.IntPoint(o).Set4(fop[pm2-i-j]/w, fop[i]/w, fop[j]/w, 0.0);
          dof2tk[o++] = 5;
       }
    for (int j = 0; j <= pm2; j++)
       for (int i=0; i + j <= pm2; i++) // (1,2,4)
       {
          double w = fop[i] + fop[j] + fop[pm2-i-j];
          Nodes.IntPoint(o).Set4(fop[pm2-i-j]/w, fop[i]/w, 0.0, fop[j]/w);
          dof2tk[o++] = 4;
          Nodes.IntPoint(o).Set4(fop[pm2-i-j]/w, fop[i]/w, 0.0, fop[j]/w);
          dof2tk[o++] = 7;
       }
    for (int j = 0; j <= pm2; j++)
       for (int i=0; i + j <= pm2; i++) // (1,3,4)
       {
          double w = fop[i] + fop[j] + fop[pm2-i-j];
          Nodes.IntPoint(o).Set4(fop[pm2-i-j]/w, 0.0, fop[i]/w, fop[j]/w);
          dof2tk[o++] = 5;
          Nodes.IntPoint(o).Set4(fop[pm2-i-j]/w, 0.0, fop[i]/w, fop[j]/w);
          dof2tk[o++] = 7;
       }
    for (int j = 0; j <= pm2; j++)
       for (int i=0; i + j <= pm2; i++) // (2,3,4)
       {
          double w = fop[i] + fop[j] + fop[pm2-i-j];
          Nodes.IntPoint(o).Set4(0.0, fop[pm2-i-j]/w, fop[i]/w, fop[j]/w);
          dof2tk[o++] = 6;
          Nodes.IntPoint(o).Set4(0.0, fop[pm2-i-j]/w, fop[i]/w, fop[j]/w);
          dof2tk[o++] = 8;
       }

    // facets (see Mesh::GenerateFaces in mesh/mesh.cpp)
    for (int k = 0; k <= pm3; k++)
       for (int j = 0; j + k <= pm3; j++)
          for (int i = 0; i + j + k <= pm3; i++)  // (0,1,2,3)
          {
             double w = ftop[i] + ftop[j] + ftop[k] + ftop[pm3-i-j-k];
             Nodes.IntPoint(o).Set4(ftop[i]/w, ftop[j]/w, ftop[k]/w, 0.0);
             dof2tk[o++] = 0;
             Nodes.IntPoint(o).Set4(ftop[i]/w, ftop[j]/w, ftop[k]/w, 0.0);
             dof2tk[o++] = 1;
             Nodes.IntPoint(o).Set4(ftop[i]/w, ftop[j]/w, ftop[k]/w, 0.0);
             dof2tk[o++] = 2;
          }
    for (int k = 0; k <= pm3; k++)
       for (int j = 0; j + k <= pm3; j++)
          for (int i = 0; i + j + k <= pm3; i++)  // (0,2,1,4)
          {
             double w = ftop[i] + ftop[j] + ftop[k] + ftop[pm3-i-j-k];
             Nodes.IntPoint(o).Set4(ftop[j]/w, ftop[i]/w, 0.0, ftop[k]/w);
             dof2tk[o++] = 1;
             Nodes.IntPoint(o).Set4(ftop[j]/w, ftop[i]/w, 0.0, ftop[k]/w);
             dof2tk[o++] = 0;
             Nodes.IntPoint(o).Set4(ftop[j]/w, ftop[i]/w, 0.0, ftop[k]/w);
             dof2tk[o++] = 3;
          }
    for (int k = 0; k <= pm3; k++)
       for (int j = 0; j + k <= pm3; j++)
          for (int i = 0; i + j + k <= pm3; i++)  // (0,1,3,4)
          {
             double w = ftop[i] + ftop[j] + ftop[k] + ftop[pm3-i-j-k];
             Nodes.IntPoint(o).Set4(ftop[i]/w, 0.0, ftop[j]/w, ftop[k]/w);
             dof2tk[o++] = 0;
             Nodes.IntPoint(o).Set4(ftop[i]/w, 0.0, ftop[j]/w, ftop[k]/w);
             dof2tk[o++] = 2;
             Nodes.IntPoint(o).Set4(ftop[i]/w, 0.0, ftop[j]/w, ftop[k]/w);
             dof2tk[o++] = 3;
          }
    for (int k = 0; k <= pm3; k++)
       for (int j = 0; j + k <= pm3; j++)
          for (int i = 0; i + j + k <= pm3; i++)  // (0,3,2,4)
          {
             double w = ftop[i] + ftop[j] + ftop[k] + ftop[pm3-i-j-k];
             Nodes.IntPoint(o).Set4(0.0, ftop[j]/w, ftop[i]/w, ftop[k]/w);
             dof2tk[o++] = 2;
             Nodes.IntPoint(o).Set4(0.0, ftop[j]/w, ftop[i]/w, ftop[k]/w);
             dof2tk[o++] = 1;
             Nodes.IntPoint(o).Set4(0.0, ftop[j]/w, ftop[i]/w, ftop[k]/w);
             dof2tk[o++] = 3;
          }
    for (int k = 0; k <= pm3; k++)
       for (int j = 0; j + k <= pm3; j++)
          for (int i = 0; i + j + k <= pm3; i++)  // (1,2,3,4)
          {
             double w = ftop[i] + ftop[j] + ftop[k] + ftop[pm3-i-j-k];
             Nodes.IntPoint(o).Set4(ftop[pm3-i-j-k]/w, ftop[i]/w, ftop[j]/w, ftop[k]/w);
              const IntegrationPoint &ip = Nodes.IntPoint(o);
             dof2tk[o++] = 4;
             Nodes.IntPoint(o).Set4(ftop[pm3-i-j-k]/w, ftop[i]/w, ftop[j]/w, ftop[k]/w);
             dof2tk[o++] = 5;
             Nodes.IntPoint(o).Set4(ftop[pm3-i-j-k]/w, ftop[i]/w, ftop[j]/w, ftop[k]/w);
             dof2tk[o++] = 7;
          }

    // interior bubbles
    for (int l = 0; l <= pm4; l++)
       for (int k = 0; k + l <= pm4; k++)
          for (int j = 0; j + k + l <= pm4; j++)
             for (int i = 0; i + j + k + l <= pm4; i++)
             {
                double w = iop[i] + iop[j] + iop[k] + iop[l] + iop[pm4-i-j-k-l];
                Nodes.IntPoint(o).Set4(iop[i]/w, iop[j]/w, iop[k]/w, iop[l]/w);
                const IntegrationPoint &ip = Nodes.IntPoint(o);
                dof2tk[o++] = 0;
                Nodes.IntPoint(o).Set4(iop[i]/w, iop[j]/w, iop[k]/w, iop[l]/w);
                dof2tk[o++] = 1;
                Nodes.IntPoint(o).Set4(iop[i]/w, iop[j]/w, iop[k]/w, iop[l]/w);
                dof2tk[o++] = 2;
                Nodes.IntPoint(o).Set4(iop[i]/w, iop[j]/w, iop[k]/w, iop[l]/w);
                dof2tk[o++] = 3;

             }
    
    DenseMatrix T(dof);
    DenseMatrix B(dof, dim);
    
    for (int q = 0; q < dof; q++)
    {
        const IntegrationPoint &ip = Nodes.IntPoint(q);
                
        const Vector tm({tk[4*dof2tk[q]], tk[4*dof2tk[q]+1], tk[4*dof2tk[q]+2], tk[4*dof2tk[q]+3]});
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
        
        // Lamda Function for computing Whitney function part of basis function
        
        auto WH_func = [&](int A,int B) -> std::vector<double>{
            
            double La_WF = bary_vector[A];
            double Lb_WF = bary_vector[B];

            std::vector<double> grad_La_WF = gradbary_vector[A];
            std::vector<double> grad_Lb_WF = gradbary_vector[B];
            
            double i_hat, j_hat, k_hat, l_hat;
            
            i_hat = La_WF*grad_Lb_WF[0] - Lb_WF*grad_La_WF[0];
            j_hat = La_WF*grad_Lb_WF[1] - Lb_WF*grad_La_WF[1];
            k_hat = La_WF*grad_Lb_WF[2] - Lb_WF*grad_La_WF[2];
            l_hat = La_WF*grad_Lb_WF[3] - Lb_WF*grad_La_WF[3];
            
            std::vector<double> WH_func = {i_hat, j_hat, k_hat, l_hat};
            
            return WH_func;
            
        }; // end of lamda function
        
        //Edges
        for(int i=0; i<p; i++)
        {
            for(int a=0; a<5; a++)
            {
                for(int b=0; b<5; b++)
                {
                    if(a<b)
                    {
                        La = bary_vector[a];
                        Lb = bary_vector[b];
                                                
                        // compute polynomials
                        std::vector<double> Legendre_i;
                        double x = Lb;
                        double y = La + Lb;
                        poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                        
                        // Whitney Function
                        std::vector<double> Whit_Vec = WH_func(a,b);
                        
                        // Add Basis Funcitons
                        double x_comp = Legendre_i[Legendre_i.size()-1]*Whit_Vec[0];
                        double Whitx = Whit_Vec[0];
                        B(o, 0) = x_comp;
                        
                        double y_comp = Legendre_i[Legendre_i.size()-1]*Whit_Vec[1];
                        double Whity = Whit_Vec[1];

                        B(o, 1) = y_comp;
                        
                        double z_comp = Legendre_i[Legendre_i.size()-1]*Whit_Vec[2];
                        double Whitz = Whit_Vec[2];

                        B(o, 2) = z_comp;
                        
                        double t_comp = Legendre_i[Legendre_i.size()-1]*Whit_Vec[3];
                        double Whitt = Whit_Vec[3];

                        B(o, 3) = t_comp;
                        
                        o++;
                        
                    }
                }
            }
        }  // end of edges
        
        
        //Faces
        for(int i=0; i<p;i++)
        {
            for(int j=1; j<p;j++)
            {
                for(int a=0; a<5;a++)
                {
                    for(int b=0; b<5;b++)
                    {
                        for(int c=0; c<5;c++)
                        {
                            if((a<b)&&(b<c)&&((i+j)<p))
                            {
                                // Family I:
                                La = bary_vector[a];
                                Lb = bary_vector[b];
                                Lc = bary_vector[c];
                                
                                // compute polynomials
                                std::vector<double> Legendre_i;
                                double x = Lb;
                                double y = La + Lb;
                                poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                                
                                std::vector<double> Int_Jacobi_j;
                                x = Lc;
                                y = La + Lb + Lc;
                                double alpha = 2*i +1;
                                poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                                
                                // Whitney Function
                                std::vector<double> Whit_Vec = WH_func(a,b);
                                
                                // Add Basis Funcitons
                                B(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[0];
                                
                                B(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[1];
                                
                                B(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[2];
                                
                                B(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[3];
                                
                                o++;
                                
                                // Family II:
                                // Clean up storage for Polynomials
                                Legendre_i.clear();
                                Int_Jacobi_j.clear();
                                // compute polynomials
                                x = Lc;
                                y = Lc + Lb;
                                poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                                
                                x = La;
                                y = La + Lb + Lc;
                                alpha = 2*i +1;
                                poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                                
                                // Whitney Function
                                Whit_Vec = WH_func(b,c);
                                
                                // Add Basis Funcitons
                                B(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[0];
                                
                                B(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[1];
                                
                                B(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[2];
                                
                                B(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[3];
                                
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
        
            
            // Define Barycentric Coordinates
            La = bary_vector[a];
            Lb = bary_vector[b];
            Lc = bary_vector[c];
            Ld = bary_vector[d];
            
            
            for(int i=0; i<=p;i++)
            {
                for(int j=1; j<=p;j++)
                {
                    for(int l=1; l<=p;l++)
                    {
                        if((i+j+l)<p)
                        {
                            
                            // Family I:
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                            
                            std::vector<double> Int_Jacobi_j;
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i +1;
                            poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                            
                            std::vector<double> Int_Jacobi_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i +j);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            
                            // Whitney Function
                            std::vector<double> Whit_Vec = WH_func(a,b);
                            
                            // Add Basis Funcitons
                            B(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[0];
                            
                            B(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[1];
                            
                            B(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[2];
                            
                            B(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[3];
                            
                            o++;
                            
                            // Family II:
                            //Clean up storage for Polynomials
                            Legendre_i.clear();
                            Int_Jacobi_j.clear();
                            Int_Jacobi_l.clear();

                            // compute polynomials
                            x = Lc;
                            y = Lc + Lb;
                            poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                            
                            x = Ld;
                            y = Lb + Lc + Ld;
                            alpha = 2*i + 1;
                            poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                            
                            x = La;
                            y = Lb + Lc + Ld + La;
                            alpha = 2*(i +j);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            
                            // Whitney Function
                            Whit_Vec = WH_func(b,c);
                            
                            // Add Basis Funcitons
                            B(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[0];
                            
                            B(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[1];
                            
                            B(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[2];
                            
                            B(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[3];
                            
                            o++;
                            
                            // Family III:
                            //Clean up storage for Polynomials
                            Legendre_i.clear();
                            Int_Jacobi_j.clear();
                            Int_Jacobi_l.clear();
                            // compute polynomials
                            x = Ld;
                            y = Lc + Ld;
                            poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                            
                            x = La;
                            y = Lc + Ld + La;
                            alpha = 2*i + 1;
                            poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                            
                            x = Lb;
                            y = Lc + Ld + La + Lb;
                            alpha = 2*(i +j);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            
                            // Whitney Function
                            Whit_Vec = WH_func(c,d);
                            
                            // Add Basis Funcitons
                            B(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[0];
                            
                            B(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[1];
                            
                            B(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[2];
                            
                            B(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[3];
                            
                            o++;
                            
                        }
                    }
                }
            }
        }// end of Facets

        
        //Interiors
        for (int r = 0; r<4; r++)
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
            else
            {
                mfem_error("Invaild Bubble");
            }
            
            for (int i=0; i<=p; i++)
            {
                for (int j=1; j<=p; j++)
                {
                    for (int l=1; l<=p; l++)
                    {
                        for (int m=1; m<=p; m++)
                        {
                            if ((i+j+l+m)<(p))
                            {
                                
                                // compute polynomials
                                std::vector<double> Legendre_i;
                                double x = Lb;
                                double y = La + Lb;
                                poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                                
                                std::vector<double> Int_Jacobi_j;
                                x = Lc;
                                y = La + Lb + Lc;
                                double alpha = 2*i +1;
                                poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                                
                                std::vector<double> Int_Jacobi_l;
                                x = Ld;
                                y = La + Lb + Lc + Ld;
                                alpha = 2*(i +j);
                                poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                                
                                std::vector<double> Int_Jacobi_m;
                                x = Le;
                                y = 1.0;
                                alpha = 2*(i + j + l);
                                poly1d.CalcIntJacobi(m, x, y, alpha, Int_Jacobi_m);
                                
                                // Whitney Function
                                std::vector<double> Whit_Vec = WH_func(a,b);
                                
                                // Add Basis Funcitons
                                B(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * Whit_Vec[0];
                                
                                B(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * Whit_Vec[1];
                                
                                B(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * Whit_Vec[2];
                                
                                B(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * Whit_Vec[3];
                                
                                o++;
                                
                            }
                        }
                    }
                }
            }
            
        }// End of Interiors
         

        B.Mult(tm, T.GetColumn(q));
    }
    
    Ti.Factor(T);
    mfem::out << "Sk_grad_PentatopeElement(" << p << ") : "; Ti.TestInversion();
}

void HSkwGrad_PentatopeElement::CalcVShape(const IntegrationPoint &ip,
                                           DenseMatrix &shape) const
{
    const int p = order;
    int reset_o;
    
#ifdef MFEM_THREAD_SAFE
    Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_t(p + 1),
    shape_l(p + 1);
    DenseMatrix u(Dof, Dim);
#endif
    int size_ip = Nodes.Size();
    
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
    
    // Lamda Function for computing Whitney function part of basis function
    
    auto WH_func = [&](int A,int B) -> std::vector<double>{
        
        double La_WF = bary_vector[A];
        double Lb_WF = bary_vector[B];
        std::vector<double> grad_La_WF = gradbary_vector[A];
        std::vector<double> grad_Lb_WF = gradbary_vector[B];
        
        double i_hat, j_hat, k_hat, l_hat;
        
        i_hat = La_WF*grad_Lb_WF[0] - Lb_WF*grad_La_WF[0];
        j_hat = La_WF*grad_Lb_WF[1] - Lb_WF*grad_La_WF[1];
        k_hat = La_WF*grad_Lb_WF[2] - Lb_WF*grad_La_WF[2];
        l_hat = La_WF*grad_Lb_WF[3] - Lb_WF*grad_La_WF[3];
        
        std::vector<double> WH_func = {i_hat, j_hat, k_hat, l_hat};
        
        return WH_func;
        
    }; // end of lamda function
    
    
    //Edges
    for(int i=0; i<p; i++)
    {
        for(int a=0; a<5; a++)
        {
            for(int b=0; b<5; b++)
            {
                if(a<b)
                {
                    La = bary_vector[a];
                    Lb = bary_vector[b];
                    
                    // compute polynomials
                    std::vector<double> Legendre_i;
                    double x = Lb;
                    double y = La + Lb;
                    poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                    
                    // Whitney Function
                    std::vector<double> Whit_Vec = WH_func(a,b);
                    
                    // Add Basis Funcitons
                    u(o, 0) = Legendre_i[Legendre_i.size()-1]*Whit_Vec[0];
                    
                    u(o, 1)  = Legendre_i[Legendre_i.size()-1]*Whit_Vec[1];
                    
                    u(o, 2)  = Legendre_i[Legendre_i.size()-1]*Whit_Vec[2];
                    
                    u(o, 3) = Legendre_i[Legendre_i.size()-1]*Whit_Vec[3];
                    
                    o++;
                    
                }
            }
        }
    }  // end of edges
    

    
    //Faces
    for(int i=0; i<p;i++)
    {
        for(int j=1; j<p;j++)
        {
            for(int a=0; a<5;a++)
            {
                for(int b=0; b<5;b++)
                {
                    for(int c=0; c<5;c++)
                    {
                        if((a<b)&&(b<c)&&((i+j)<p))
                        {
                            // Define Family
                            int Family = 1;
                            
                            if (Family == 1)
                            {
                                
                                // Family I:
                                La = bary_vector[a];
                                Lb = bary_vector[b];
                                Lc = bary_vector[c];
                                
                                // compute polynomials
                                std::vector<double> Legendre_i;
                                double x = Lb;
                                double y = La + Lb;
                                poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                                
                                std::vector<double> Int_Jacobi_j;
                                x = Lc;
                                y = La + Lb + Lc;
                                double alpha = 2*i +1;
                                poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                                
                                // Whitney Function
                                std::vector<double> Whit_Vec = WH_func(a,b);
                                
                                // Add Basis Funcitons
                                u(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[0];
                                
                                u(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[1];
                                
                                u(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[2];
                                
                                u(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[3];
                                
                                o++;
                                
                            }
                            
                            Family++;

                            
                            if (Family == 2)
                            {
                                
                                // Family II:
                                La = bary_vector[b];
                                Lb = bary_vector[c];
                                Lc = bary_vector[a];
                                
                                // compute polynomials
                                std::vector<double> Legendre_i;
                                double x = Lb;
                                double y = La + Lb;
                                poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                                
                                std::vector<double> Int_Jacobi_j;
                                x = Lc;
                                y = La + Lb + Lc;
                                double alpha = 2*i +1;
                                poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                                
                                // Whitney Function
                                std::vector<double> Whit_Vec = WH_func(b,c);
                                
                                // Add Basis Funcitons
                                u(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[0];
                                
                                u(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[1];
                                
                                u(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[2];
                                
                                u(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Whit_Vec[3];
                                
                                o++;
                                
                            }
                            
                            
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
        
        // Define Barycentric Coordinates
        La = bary_vector[a];
        Lb = bary_vector[b];
        Lc = bary_vector[c];
        Ld = bary_vector[d];
        
        for(int i=0; i<=p;i++)
        {
            for(int j=1; j<=p;j++)
            {
                for(int l=1; l<=p;l++)
                {
                    if((i+j+l)<p)
                    {
                        
                        // Family I:
                        // compute polynomials
                        std::vector<double> Legendre_i;
                        double x = Lb;
                        double y = La + Lb;
                        poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                        
                        std::vector<double> Int_Jacobi_j;
                        x = Lc;
                        y = La + Lb + Lc;
                        double alpha = 2*i +1;
                        poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                        
                        std::vector<double> Int_Jacobi_l;
                        x = Ld;
                        y = La + Lb + Lc + Ld;
                        alpha = 2*(i +j);
                        poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                        
                        // Whitney Function
                        std::vector<double> Whit_Vec = WH_func(a,b);
                        
                        // Add Basis Funcitons
                        u(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[0];
                        
                        u(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[1];
                        
                        u(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[2];
                        
                        u(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[3];
                        
                        o++;
                        
                        // Family II:
                        //Clean up storage for Polynomials
                        Legendre_i.clear();
                        Int_Jacobi_j.clear();
                        Int_Jacobi_l.clear();
                        // compute polynomials
                        x = Lc;
                        y = Lb + Lc;
                        poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                        
                        x = Ld;
                        y = Lb + Lc + Ld;
                        alpha = 2*i + 1;
                        poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                        
                        x = La;
                        y = Lb + Lc + Ld + La;
                        alpha = 2*(i +j);
                        poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                        
                        // Whitney Function
                        Whit_Vec = WH_func(b,c);
                        
                        // Add Basis Funcitons
                        u(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[0];
                        
                        u(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[1];
                        
                        u(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[2];
                        
                        u(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[3];
                        
                        
                        o++;
                        
                        // Family III:
                        //Clean up storage for Polynomials
                        Legendre_i.clear();
                        Int_Jacobi_j.clear();
                        Int_Jacobi_l.clear();
                        // compute polynomials
                        x = Ld;
                        y = Lc + Ld;
                        poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                        
                        x = La;
                        y = Lc + Ld + La;
                        alpha = 2*i + 1;
                        poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                        
                        x = Lb;
                        y = Lc + Ld + La + Lb;
                        alpha = 2*(i +j);
                        poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                        
                        // Whitney Function
                        Whit_Vec = WH_func(c,d);
                        
                        // Add Basis Funcitons
                        u(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[0];
                        
                        u(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[1];
                        
                        u(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[2];
                        
                        u(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Whit_Vec[3];
                        
                        o++;
                        
                    }
                }
            }
        }
    }// end of Facets
    
    
    //Interiors
    for (int r = 0; r<4; r++)
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
        else
        {
            mfem_error("Invaild Bubble");
        }
        
        for (int i=0; i<=(p+1); i++)
        {
            for (int j=1; j<=(p+1); j++)
            {
                for (int l=1; l<=(p+1); l++)
                {
                    for (int m=1; m<=(p+1); m++)
                    {
                        if ((i+j+l+m)<=(p-1))
                        {
                            
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcLegendreShifted(i, x, y, Legendre_i);
                            
                            std::vector<double> Int_Jacobi_j;
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i +1;
                            poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                            
                            std::vector<double> Int_Jacobi_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i +j);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            
                            std::vector<double> Int_Jacobi_m;
                            x = Le;
                            y = 1.0;
                            alpha = 2*(i + j + l);
                            poly1d.CalcIntJacobi(m, x, y, alpha, Int_Jacobi_m);
                            
                            // Whitney Function
                            std::vector<double> Whit_Vec = WH_func(a,b);
                            
                            // Add Basis Funcitons
                            u(o, 0) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * Whit_Vec[0];
                            
                            u(o, 1) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * Whit_Vec[1];
                            
                            u(o, 2) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * Whit_Vec[2];
                            
                            u(o, 3) = Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1] * Whit_Vec[3];
                            
                            o++;
                            
                        }
                    }
                }
            }
        }
        
    }// End of Interiors
     
    Ti.Mult(u, shape);

}

void HSkwGrad_PentatopeElement::CalcSkwGradShape(const IntegrationPoint &ip,
                                                 DenseMatrix &SkwGradshape) const
{
    const int p = order;
    
#ifdef MFEM_THREAD_SAFE
    Vector shape_x(p + 1), shape_y(p + 1), shape_z(p + 1), shape_t(p + 1),
    shape_l(p + 1);
    DenseMatrix u(Dof, Dim);
#endif
    int size_ip = Nodes.Size();
    SkwGradu.SetSize(dof, int(6));
    int o = 0;
    int num_dof = dof;
    real_t s_factor = 0.5;
    
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
    
    // Lamda Function for computing Whitney function part of basis function
    
    auto WH_func = [&](int A,int B) -> std::vector<double>{
        
        double La_WF = bary_vector[A];
        double Lb_WF = bary_vector[B];
        std::vector<double> grad_La_WF = gradbary_vector[A];
        std::vector<double> grad_Lb_WF = gradbary_vector[B];
        
        double i_hat, j_hat, k_hat, l_hat;
        
        i_hat = La_WF*grad_Lb_WF[0] - Lb_WF*grad_La_WF[0];
        j_hat = La_WF*grad_Lb_WF[1] - Lb_WF*grad_La_WF[1];
        k_hat = La_WF*grad_Lb_WF[2] - Lb_WF*grad_La_WF[2];
        l_hat = La_WF*grad_Lb_WF[3] - Lb_WF*grad_La_WF[3];
        
        std::vector<double> WH_func = {i_hat, j_hat, k_hat, l_hat};
        
        return WH_func;
        
    }; // end of lamda function

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
                result(i,j) = 0.5*(outer_prod_1 - outer_prod_2);
                //std::cout << 0.5*(outer_prod_1 - outer_prod_2) << std::endl;
            }
            //std::cout << "New COl --------" << std::endl;
        }
        
        return result;
    }; // end of lamda function

    
    //Edges
    for(int i=0; i<p; i++)
    {
        for(int a=0; a<5; a++)
        {
            for(int b=0; b<5; b++)
            {
                if(a<b)
                {
                    // Set Barycentric lamdas
                    La = bary_vector[a];
                    Lb = bary_vector[b];
                    
                    // Set grad(Lamda)
                    grad_La = gradbary_vector[a];
                    grad_Lb = gradbary_vector[b];
                    
                    // Compute parts of SkewGrad(Whitney Function)
                    
                    // Q0 = 0
                    double Q1 = 1.0*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1] - grad_La[1]*grad_Lb[0] + grad_Lb[1]*grad_La[0]);
                    double Q2 = 1.0*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2] - grad_La[2]*grad_Lb[0] + grad_Lb[2]*grad_La[0]);
                    double Q3 = 1.0*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3] - grad_La[3]*grad_Lb[0] + grad_Lb[3]*grad_La[0]);
                    double Q4 = -1.0*Q1;
                    // Q5 = 0
                    double Q6 = 1.0*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2] - grad_La[2]*grad_Lb[1] + grad_Lb[2]*grad_La[1]);
                    double Q7 = 1.0*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3] - grad_La[3]*grad_Lb[1] + grad_Lb[3]*grad_La[1]);
                    double Q8 = -1.0*Q2;
                    double Q9 = -1.0*Q6;
                    // Q10 = 0
                    double Q11 = 1.0*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3] - grad_La[3]*grad_Lb[2] + grad_Lb[3]*grad_La[2]);
                    double Q12 = -1.0*Q3;
                    double Q13 = -1.0*Q7;
                    double Q14 = -1.0*Q11;
                    // Q15 = 0

                    Q1 = Q1 * s_factor;
                    Q2 = Q2 * s_factor;
                    Q3 = Q3 * s_factor;
                    Q4 = Q4 * s_factor;
                    Q6 = Q6 * s_factor;
                    Q7 = Q7 * s_factor;
                    Q8 = Q8 * s_factor;
                    Q9 = Q9 * s_factor;
                    Q11 = Q11 * s_factor;
                    Q12 = Q12 * s_factor;
                    Q13 = Q13 * s_factor;
                    Q14 = Q14 * s_factor;
                    


                    // compute polynomials
                    std::vector<double> Legendre_i;
                    std::vector<double> Legendre_i_dx;
                    std::vector<double> Legendre_i_dt;
                    
                    double x = Lb;
                    double y = La + Lb;
                    
                    poly1d.CalcScaledLegendreDerivative(i, x, y, Legendre_i, Legendre_i_dx, Legendre_i_dt);
                    
                    // Whitney Function
                    std::vector<double> Whit_Vec = WH_func(a,b);
                    
                    // grad of Legendre Polynomials (scalar part)
                    double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0]));
                    
                    double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1]));
                    
                    double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2]));
                    
                    double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3]));
                    
                    // Gradient of Scalar part
                    std::vector<double> grad_scalar = {dscalar_x, dscalar_y, dscalar_z, dscalar_t};
                    
                    // skew-sym outer product
                    DenseMatrix skw_sym_outerprod = skw_sym_outerprod_fnc(grad_scalar,Whit_Vec);
                    
                    
                    // Add Basis Funcitons
                    
                    SkwGradu(o, 0) = Q1 * Legendre_i[Legendre_i.size()-1] + skw_sym_outerprod(0, 1);
                    SkwGradu(o, 1) = Q2 * Legendre_i[Legendre_i.size()-1] + skw_sym_outerprod(0, 2);
                    SkwGradu(o, 2) = Q3 * Legendre_i[Legendre_i.size()-1] + skw_sym_outerprod(0, 3);
                    SkwGradu(o, 3) = Q6 * Legendre_i[Legendre_i.size()-1] + skw_sym_outerprod(1, 2);
                    SkwGradu(o, 4) = Q7 * Legendre_i[Legendre_i.size()-1] + skw_sym_outerprod(1, 3);
                    SkwGradu(o, 5) = Q11 * Legendre_i[Legendre_i.size()-1] + skw_sym_outerprod(2, 3);
                    
                    o++;
                    
                }
            }
        }
    }  // end of edges
    
    
    //Faces
    for(int i=0; i<p;i++)
    {
        for(int j=1; j<p;j++)
        {
            for(int a=0; a<5;a++)
            {
                for(int b=0; b<5;b++)
                {
                    for(int c=0; c<5;c++)
                    {
                        if((a<b)&&(b<c)&&((i+j)<p))
                        {
                            //num_faces++;
                            // Define Family
                            int Family = 1;
                            
                            if (Family == 1) 
                            {
                                
                                // Family I:
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
                                
                                std::vector<double> Int_Jacobi_j;
                                std::vector<double> Jacobi_j;
                                std::vector<double> R_j;
                                
                                x = Lc;
                                y = La + Lb + Lc;
                                double alpha = 2*i +1;
                                poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                                poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                                poly1d.CalcRJacobi(j, x, y, alpha, R_j);
                                
                                // Whitney Function
                                std::vector<double> Whit_Vec = WH_func(a,b);
                                
                                // Compute parts of SkewGrad(Whitney Function)
                                
                                // Q0 = 0
                                double Q1 = 0.5*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1] - grad_La[1]*grad_Lb[0] + grad_Lb[1]*grad_La[0]);
                                double Q2 = 0.5*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2] - grad_La[2]*grad_Lb[0] + grad_Lb[2]*grad_La[0]);
                                double Q3 = 0.5*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3] - grad_La[3]*grad_Lb[0] + grad_Lb[3]*grad_La[0]);
                                double Q4 = -1.0*Q1;
                                // Q5 = 0
                                double Q6 = 0.5*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2] - grad_La[2]*grad_Lb[1] + grad_Lb[2]*grad_La[1]);
                                double Q7 = 0.5*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3] - grad_La[3]*grad_Lb[1] + grad_Lb[3]*grad_La[1]);
                                double Q8 = -1.0*Q2;
                                double Q9 = -1.0*Q6;
                                // Q10 = 0
                                double Q11 = 0.5*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3] - grad_La[3]*grad_Lb[2] + grad_Lb[3]*grad_La[2]);
                                double Q12 = -1.0*Q3;
                                double Q13 = -1.0*Q7;
                                double Q14 = -1.0*Q11;
                                // Q15 = 0
                                
                                
                                // grad of Legendre and Jacobi Polynomials (scalar part)
                                double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0])) * Int_Jacobi_j[Int_Jacobi_j.size()-1]
                                
                                + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[0] + R_j[R_j.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0]) );
                                
                                
                                double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1])) * Int_Jacobi_j[Int_Jacobi_j.size()-1]
                                
                                + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[1] + R_j[R_j.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1]) );
                                
                                double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2])) * Int_Jacobi_j[Int_Jacobi_j.size()-1]
                                
                                + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[2] + R_j[R_j.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2]) );
                                
                                double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3])) * Int_Jacobi_j[Int_Jacobi_j.size()-1]
                                
                                + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[3] + R_j[R_j.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3]) );
                                
                                // Gradient of Scalar part
                                std::vector<double> grad_scalar = {dscalar_x, dscalar_y, dscalar_z, dscalar_t};
                                
                                // skew-sym outer product
                                DenseMatrix skw_sym_outerprod = skw_sym_outerprod_fnc(grad_scalar,Whit_Vec);
                                
                                
                                // Add Basis Funcitons
                                
                                SkwGradu(o, 0) = Q1 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1] + skw_sym_outerprod(0, 1);
                                SkwGradu(o, 1) = Q2 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]  + skw_sym_outerprod(0, 2);
                                SkwGradu(o, 2) = Q3 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]  + skw_sym_outerprod(0, 3);
                                SkwGradu(o, 3) = Q6 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]  + skw_sym_outerprod(1, 2);
                                SkwGradu(o, 4) = Q7 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]  + skw_sym_outerprod(1, 3);
                                SkwGradu(o, 5) = Q11 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]  + skw_sym_outerprod(2, 3);
                                
                                o++;
                                
                            }
                            
                            Family++;
                        
                            // Family II:  ----------------------------
                            if (Family == 2)
                            {
                                
                                // Family II:
                                La = bary_vector[b];
                                Lb = bary_vector[c];
                                Lc = bary_vector[a];
                                
                                grad_La = gradbary_vector[b];
                                grad_Lb = gradbary_vector[c];
                                grad_Lc = gradbary_vector[a];
                                
                                // compute polynomials
                                std::vector<double> Legendre_i;
                                std::vector<double> Legendre_i_dx;
                                std::vector<double> Legendre_i_dt;
                                
                                double x = Lb;
                                double y = La + Lb;
                                poly1d.CalcScaledLegendreDerivative(i, x, y, Legendre_i, Legendre_i_dx, Legendre_i_dt);
                                
                                std::vector<double> Int_Jacobi_j;
                                std::vector<double> Jacobi_j;
                                std::vector<double> R_j;
                                
                                x = Lc;
                                y = La + Lb + Lc;
                                double alpha = 2*i +1;
                                poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                                poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                                poly1d.CalcRJacobi(j, x, y, alpha, R_j);
                                
                                // Whitney Function
                                std::vector<double> Whit_Vec = WH_func(b,c);
                                
                                // Compute parts of SkewGrad(Whitney Function)
                                
                                // Q0 = 0
                                double Q1 = 0.5*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1] - grad_La[1]*grad_Lb[0] + grad_Lb[1]*grad_La[0]);
                                double Q2 = 0.5*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2] - grad_La[2]*grad_Lb[0] + grad_Lb[2]*grad_La[0]);
                                double Q3 = 0.5*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3] - grad_La[3]*grad_Lb[0] + grad_Lb[3]*grad_La[0]);
                                double Q4 = -1.0*Q1;
                                // Q5 = 0
                                double Q6 = 0.5*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2] - grad_La[2]*grad_Lb[1] + grad_Lb[2]*grad_La[1]);
                                double Q7 = 0.5*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3] - grad_La[3]*grad_Lb[1] + grad_Lb[3]*grad_La[1]);
                                double Q8 = -1.0*Q2;
                                double Q9 = -1.0*Q6;
                                // Q10 = 0
                                double Q11 = 0.5*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3] - grad_La[3]*grad_Lb[2] + grad_Lb[3]*grad_La[2]);
                                double Q12 = -1.0*Q3;
                                double Q13 = -1.0*Q7;
                                double Q14 = -1.0*Q11;
                                // Q15 = 0
                                
                                
                                // grad of Legendre and Jacobi Polynomials (scalar part)
                                double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0])) * Int_Jacobi_j[Int_Jacobi_j.size()-1]
                                
                                + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[0] + R_j[R_j.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0]) );
                                
                                
                                double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1])) * Int_Jacobi_j[Int_Jacobi_j.size()-1]
                                
                                + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[1] + R_j[R_j.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1]) );
                                
                                double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2])) * Int_Jacobi_j[Int_Jacobi_j.size()-1]
                                
                                + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[2] + R_j[R_j.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2]) );
                                
                                double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3])) * Int_Jacobi_j[Int_Jacobi_j.size()-1]
                                
                                + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[3] + R_j[R_j.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3]) );
                                
                                // Gradient of Scalar part
                                //std::cout << "Grad_vec = {" << dscalar_x << ", " << dscalar_y << ", " << dscalar_z << ", " << dscalar_t << std::endl;
                                std::vector<double> grad_scalar = {dscalar_x, dscalar_y, dscalar_z, dscalar_t};
                                
                                // skew-sym outer product
                                DenseMatrix skw_sym_outerprod = skw_sym_outerprod_fnc(grad_scalar,Whit_Vec);
                                
                                
                                // Add Basis Funcitons
                                
                                SkwGradu(o, 0) = Q1 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1] + skw_sym_outerprod(0, 1);
                                SkwGradu(o, 1) = Q2 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]  + skw_sym_outerprod(0, 2);
                                SkwGradu(o, 2) = Q3 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]  + skw_sym_outerprod(0, 3);
                                SkwGradu(o, 3) = Q6 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]  + skw_sym_outerprod(1, 2);
                                SkwGradu(o, 4) = Q7 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]  + skw_sym_outerprod(1, 3);
                                SkwGradu(o, 5) = Q11 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]  + skw_sym_outerprod(2, 3);
                                
                                o++;
                                
                            }
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
        
        for(int i=0; i<=p;i++)
        {
            for(int j=1; j<=p;j++)
            {
                for(int l=1; l<=p;l++)
                {
                    if((i+j+l)<p)
                    {
                        
                        
                        // Define Family
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
                            
                            std::vector<double> Int_Jacobi_j;
                            std::vector<double> Jacobi_j;
                            std::vector<double> R_j;
                            
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i +1;
                            poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            poly1d.CalcRJacobi(j, x, y, alpha, R_j);
                            
                            
                            std::vector<double> Int_Jacobi_l;
                            std::vector<double> Jacobi_l;
                            std::vector<double> R_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i +j);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            poly1d.CalcJacobi(l, x, y, alpha, Jacobi_l);
                            poly1d.CalcRJacobi(l, x, y, alpha, R_l);
                            
                            // Whitney Function
                            std::vector<double> Whit_Vec = WH_func(a,b);
                            
                            // Compute parts of SkewGrad(Whitney Function)
                            
                            // Q0 = 0
                            double Q1 = 0.5*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1] - grad_La[1]*grad_Lb[0] + grad_Lb[1]*grad_La[0]);
                            double Q2 = 0.5*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2] - grad_La[2]*grad_Lb[0] + grad_Lb[2]*grad_La[0]);
                            double Q3 = 0.5*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3] - grad_La[3]*grad_Lb[0] + grad_Lb[3]*grad_La[0]);
                            double Q4 = -1.0*Q1;
                            // Q5 = 0
                            double Q6 = 0.5*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2] - grad_La[2]*grad_Lb[1] + grad_Lb[2]*grad_La[1]);
                            double Q7 = 0.5*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3] - grad_La[3]*grad_Lb[1] + grad_Lb[3]*grad_La[1]);
                            double Q8 = -1.0*Q2;
                            double Q9 = -1.0*Q6;
                            // Q10 = 0
                            double Q11 = 0.5*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3] - grad_La[3]*grad_Lb[2] + grad_Lb[3]*grad_La[2]);
                            double Q12 = -1.0*Q3;
                            double Q13 = -1.0*Q7;
                            double Q14 = -1.0*Q11;
                            // Q15 = 0
                            
                            
                            // grad of Legendre and Jacobi Polynomials (scalar part)
                            double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[0] + R_j[R_j.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[0] + R_l[R_l.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0] + grad_Ld[0]) );
                            
                            double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[1] + R_j[R_j.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[1] + R_l[R_l.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1] + grad_Ld[1]) );
                            
                            double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[2] + R_j[R_j.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[2] + R_l[R_l.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2] + grad_Ld[2]) );
                            
                            double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[3] + R_j[R_j.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[3] + R_l[R_l.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3] + grad_Ld[3]) );
                            
                            
                            // Gradient of Scalar part
                            std::vector<double> grad_scalar = {dscalar_x, dscalar_y, dscalar_z, dscalar_t};
                            
                            // skew-sym outer product
                            DenseMatrix skw_sym_outerprod = skw_sym_outerprod_fnc(grad_scalar,Whit_Vec);
                            
                            
                            // Add Basis Funcitons
                            
                            SkwGradu(o, 0) = Q1 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1] + skw_sym_outerprod(0, 1);
                            SkwGradu(o, 1) = Q2 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(0, 2);
                            SkwGradu(o, 2) = Q3 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(0, 3);
                            SkwGradu(o, 3) = Q6 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(1, 2);
                            SkwGradu(o, 4) = Q7 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(1, 3);
                            SkwGradu(o, 5) = Q11 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(2, 3);
                            
                            o++;
                            
                        }
                        
                        Family++;
                        
                        if (Family == 2)
                        {
                            
                            // Family I:
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
                            
                            std::vector<double> Int_Jacobi_j;
                            std::vector<double> Jacobi_j;
                            std::vector<double> R_j;
                            
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i +1;
                            poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            poly1d.CalcRJacobi(j, x, y, alpha, R_j);
                            
                            
                            std::vector<double> Int_Jacobi_l;
                            std::vector<double> Jacobi_l;
                            std::vector<double> R_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i +j);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            poly1d.CalcJacobi(l, x, y, alpha, Jacobi_l);
                            poly1d.CalcRJacobi(l, x, y, alpha, R_l);
                            
                            // Whitney Function
                            std::vector<double> Whit_Vec = WH_func(b,c);
                            
                            // Compute parts of SkewGrad(Whitney Function)
                            
                            // Q0 = 0
                            double Q1 = 0.5*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1] - grad_La[1]*grad_Lb[0] + grad_Lb[1]*grad_La[0]);
                            double Q2 = 0.5*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2] - grad_La[2]*grad_Lb[0] + grad_Lb[2]*grad_La[0]);
                            double Q3 = 0.5*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3] - grad_La[3]*grad_Lb[0] + grad_Lb[3]*grad_La[0]);
                            double Q4 = -1.0*Q1;
                            // Q5 = 0
                            double Q6 = 0.5*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2] - grad_La[2]*grad_Lb[1] + grad_Lb[2]*grad_La[1]);
                            double Q7 = 0.5*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3] - grad_La[3]*grad_Lb[1] + grad_Lb[3]*grad_La[1]);
                            double Q8 = -1.0*Q2;
                            double Q9 = -1.0*Q6;
                            // Q10 = 0
                            double Q11 = 0.5*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3] - grad_La[3]*grad_Lb[2] + grad_Lb[3]*grad_La[2]);
                            double Q12 = -1.0*Q3;
                            double Q13 = -1.0*Q7;
                            double Q14 = -1.0*Q11;
                            // Q15 = 0
                            
                            
                            // grad of Legendre and Jacobi Polynomials (scalar part)
                            double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[0] + R_j[R_j.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[0] + R_l[R_l.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0] + grad_Ld[0]) );
                            
                            double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[1] + R_j[R_j.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[1] + R_l[R_l.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1] + grad_Ld[1]) );
                            
                            double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[2] + R_j[R_j.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[2] + R_l[R_l.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2] + grad_Ld[2]) );
                            
                            double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[3] + R_j[R_j.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[3] + R_l[R_l.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3] + grad_Ld[3]) );
                            
                            
                            // Gradient of Scalar part
                            std::vector<double> grad_scalar = {dscalar_x, dscalar_y, dscalar_z, dscalar_t};
                            
                            // skew-sym outer product
                            DenseMatrix skw_sym_outerprod = skw_sym_outerprod_fnc(grad_scalar,Whit_Vec);
                            
                            
                            // Add Basis Funcitons
                            
                            SkwGradu(o, 0) = Q1 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1] + skw_sym_outerprod(0, 1);
                            SkwGradu(o, 1) = Q2 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(0, 2);
                            SkwGradu(o, 2) = Q3 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(0, 3);
                            SkwGradu(o, 3) = Q6 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(1, 2);
                            SkwGradu(o, 4) = Q7 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(1, 3);
                            SkwGradu(o, 5) = Q11 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(2, 3);
                            
                            o++;
                            
                        }
                        
                        Family++;
                        
                        if (Family == 3)
                        {
                            
                            // Family I:
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
                            
                            std::vector<double> Int_Jacobi_j;
                            std::vector<double> Jacobi_j;
                            std::vector<double> R_j;
                            
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i +1;
                            poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            poly1d.CalcRJacobi(j, x, y, alpha, R_j);
                            
                            
                            std::vector<double> Int_Jacobi_l;
                            std::vector<double> Jacobi_l;
                            std::vector<double> R_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i +j);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            poly1d.CalcJacobi(l, x, y, alpha, Jacobi_l);
                            poly1d.CalcRJacobi(l, x, y, alpha, R_l);
                            
                            // Whitney Function
                            std::vector<double> Whit_Vec = WH_func(c,d);
                            
                            // Compute parts of SkewGrad(Whitney Function)
                            
                            // Q0 = 0
                            double Q1 = 0.5*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1] - grad_La[1]*grad_Lb[0] + grad_Lb[1]*grad_La[0]);
                            double Q2 = 0.5*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2] - grad_La[2]*grad_Lb[0] + grad_Lb[2]*grad_La[0]);
                            double Q3 = 0.5*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3] - grad_La[3]*grad_Lb[0] + grad_Lb[3]*grad_La[0]);
                            double Q4 = -1.0*Q1;
                            // Q5 = 0
                            double Q6 = 0.5*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2] - grad_La[2]*grad_Lb[1] + grad_Lb[2]*grad_La[1]);
                            double Q7 = 0.5*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3] - grad_La[3]*grad_Lb[1] + grad_Lb[3]*grad_La[1]);
                            double Q8 = -1.0*Q2;
                            double Q9 = -1.0*Q6;
                            // Q10 = 0
                            double Q11 = 0.5*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3] - grad_La[3]*grad_Lb[2] + grad_Lb[3]*grad_La[2]);
                            double Q12 = -1.0*Q3;
                            double Q13 = -1.0*Q7;
                            double Q14 = -1.0*Q11;
                            // Q15 = 0
                            
                            
                            // grad of Legendre and Jacobi Polynomials (scalar part)
                            double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[0] + R_j[R_j.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[0] + R_l[R_l.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0] + grad_Ld[0]) );
                            
                            double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[1] + R_j[R_j.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[1] + R_l[R_l.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1] + grad_Ld[1]) );
                            
                            double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[2] + R_j[R_j.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[2] + R_l[R_l.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2] + grad_Ld[2]) );
                            
                            double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[3] + R_j[R_j.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[3] + R_l[R_l.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3] + grad_Ld[3]) );
                            
                            
                            // Gradient of Scalar part
                            std::vector<double> grad_scalar = {dscalar_x, dscalar_y, dscalar_z, dscalar_t};
                            
                            // skew-sym outer product
                            DenseMatrix skw_sym_outerprod = skw_sym_outerprod_fnc(grad_scalar,Whit_Vec);
                            
                            
                            // Add Basis Funcitons
                            
                            SkwGradu(o, 0) = Q1 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1] + skw_sym_outerprod(0, 1);
                            SkwGradu(o, 1) = Q2 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(0, 2);
                            SkwGradu(o, 2) = Q3 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(0, 3);
                            SkwGradu(o, 3) = Q6 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(1, 2);
                            SkwGradu(o, 4) = Q7 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(1, 3);
                            SkwGradu(o, 5) = Q11 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]  + skw_sym_outerprod(2, 3);
                            
                            o++;
                            
                        }
                        
                        
                    }
                }
            }
        }
    }// end of Facets
    
    
    //Interiors
    for (int r = 0; r<4; r++)
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
        else
        {
            mfem_error("Invaild Bubble");
        }
        int num_bub = 0;
        for (int i=0; i<=(p+1); i++)
        {
            for (int j=1; j<=(p+1); j++)
            {
                for (int l=1; l<=(p+1); l++)
                {
                    for (int m=1; m<=(p+1); m++)
                    {
                        if ((i+j+l+m)<=(p-1))
                        {
                            num_bub++;
                            // compute polynomials
                            std::vector<double> Legendre_i;
                            std::vector<double> Legendre_i_dx;
                            std::vector<double> Legendre_i_dt;
                            
                            double x = Lb;
                            double y = La + Lb;
                            poly1d.CalcScaledLegendreDerivative(i, x, y, Legendre_i, Legendre_i_dx, Legendre_i_dt);
                            
                            std::vector<double> Int_Jacobi_j;
                            std::vector<double> Jacobi_j;
                            std::vector<double> R_j;
                            
                            x = Lc;
                            y = La + Lb + Lc;
                            double alpha = 2*i +1;
                            poly1d.CalcIntJacobi(j, x, y, alpha, Int_Jacobi_j);
                            poly1d.CalcJacobi(j, x, y, alpha, Jacobi_j);
                            poly1d.CalcRJacobi(j, x, y, alpha, R_j);
                            
                            
                            std::vector<double> Int_Jacobi_l;
                            std::vector<double> Jacobi_l;
                            std::vector<double> R_l;
                            x = Ld;
                            y = La + Lb + Lc + Ld;
                            alpha = 2*(i +j);
                            poly1d.CalcIntJacobi(l, x, y, alpha, Int_Jacobi_l);
                            poly1d.CalcJacobi(l, x, y, alpha, Jacobi_l);
                            poly1d.CalcRJacobi(l, x, y, alpha, R_l);
                            
                            std::vector<double> Int_Jacobi_m;
                            std::vector<double> Jacobi_m;
                            std::vector<double> R_m;
                            x = Le;
                            y = 1.0;
                            alpha = 2*(i + j + l);
                            poly1d.CalcIntJacobi(m, x, y, alpha, Int_Jacobi_m);
                            poly1d.CalcJacobi(m, x, y, alpha, Jacobi_m);
                            poly1d.CalcRJacobi(m, x, y, alpha, R_m);
                            
                            // Whitney Function
                            std::vector<double> Whit_Vec = WH_func(a,b);
                            
                            // Compute parts of SkewGrad(Whitney Function)
                            
                            // Q0 = 0
                            double Q1 = 0.5*(grad_La[0]*grad_Lb[1] - grad_Lb[0]*grad_La[1] - grad_La[1]*grad_Lb[0] + grad_Lb[1]*grad_La[0]);
                            double Q2 = 0.5*(grad_La[0]*grad_Lb[2] - grad_Lb[0]*grad_La[2] - grad_La[2]*grad_Lb[0] + grad_Lb[2]*grad_La[0]);
                            double Q3 = 0.5*(grad_La[0]*grad_Lb[3] - grad_Lb[0]*grad_La[3] - grad_La[3]*grad_Lb[0] + grad_Lb[3]*grad_La[0]);
                            double Q4 = -1.0*Q1;
                            // Q5 = 0
                            double Q6 = 0.5*(grad_La[1]*grad_Lb[2] - grad_Lb[1]*grad_La[2] - grad_La[2]*grad_Lb[1] + grad_Lb[2]*grad_La[1]);
                            double Q7 = 0.5*(grad_La[1]*grad_Lb[3] - grad_Lb[1]*grad_La[3] - grad_La[3]*grad_Lb[1] + grad_Lb[3]*grad_La[1]);
                            double Q8 = -1.0*Q2;
                            double Q9 = -1.0*Q6;
                            // Q10 = 0
                            double Q11 = 0.5*(grad_La[2]*grad_Lb[3] - grad_Lb[2]*grad_La[3] - grad_La[3]*grad_Lb[2] + grad_Lb[3]*grad_La[2]);
                            double Q12 = -1.0*Q3;
                            double Q13 = -1.0*Q7;
                            double Q14 = -1.0*Q11;
                            // Q15 = 0
                            
                            
                            // grad of Legendre and Jacobi Polynomials (scalar part)
                            double dscalar_x = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[0] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[0] + grad_Lb[0])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[0] + R_j[R_j.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[0] + R_l[R_l.size()-2] * (grad_La[0] + grad_Lb[0] + grad_Lc[0] + grad_Ld[0])) * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2]*grad_Le[0]);
                            
                            
                            double dscalar_y = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[1] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[1] + grad_Lb[1])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[1] + R_j[R_j.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[1] + R_l[R_l.size()-2] * (grad_La[1] + grad_Lb[1] + grad_Lc[1] + grad_Ld[1])) * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2]*grad_Le[1]);
                            
                            
                            double dscalar_z = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[2] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[2] + grad_Lb[2])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[2] + R_j[R_j.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[2] + R_l[R_l.size()-2] * (grad_La[2] + grad_Lb[2] + grad_Lc[2] + grad_Ld[2])) * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2]*grad_Le[2]);
                            
                            
                            double dscalar_t = (Legendre_i_dx[Legendre_i_dx.size()-1]*grad_Lb[3] + Legendre_i_dt[Legendre_i_dt.size()-1]*(grad_La[3] + grad_Lb[3])) * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * (Jacobi_j[Jacobi_j.size()-2]*grad_Lc[3] + R_j[R_j.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3]) ) * Int_Jacobi_l[Int_Jacobi_l.size()-1] * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * (Jacobi_l[Jacobi_l.size()-2]*grad_Ld[3] + R_l[R_l.size()-2] * (grad_La[3] + grad_Lb[3] + grad_Lc[3] + grad_Ld[3])) * Int_Jacobi_m[Int_Jacobi_m.size()-1]
                            
                            + Legendre_i[Legendre_i.size()-1] * Int_Jacobi_j[Int_Jacobi_j.size()-1] * Int_Jacobi_l[Int_Jacobi_l.size()-1] * (Jacobi_m[Jacobi_m.size()-2]*grad_Le[3]);
                            

                            
                            // Gradient of Scalar part
                            std::vector<double> grad_scalar = {dscalar_x, dscalar_y, dscalar_z, dscalar_t};
                            
                            // skew-sym outer product
                            DenseMatrix skw_sym_outerprod = skw_sym_outerprod_fnc(grad_scalar,Whit_Vec);
                            
                            
                            // Add Basis Funcitons
                            
                            SkwGradu(o, 0) = Q1 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]*Int_Jacobi_m[Int_Jacobi_m.size()-1]  + skw_sym_outerprod(0, 1);
                            SkwGradu(o, 1) = Q2 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]*Int_Jacobi_m[Int_Jacobi_m.size()-1]  + skw_sym_outerprod(0, 2);
                            SkwGradu(o, 2) = Q3 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]*Int_Jacobi_m[Int_Jacobi_m.size()-1]  + skw_sym_outerprod(0, 3);
                            SkwGradu(o, 3) = Q6 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]*Int_Jacobi_m[Int_Jacobi_m.size()-1]   + skw_sym_outerprod(1, 2);
                            SkwGradu(o, 4) = Q7 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]*Int_Jacobi_m[Int_Jacobi_m.size()-1]   + skw_sym_outerprod(1, 3);
                            SkwGradu(o, 5) = Q11 * Legendre_i[Legendre_i.size()-1]*Int_Jacobi_j[Int_Jacobi_j.size()-1]*Int_Jacobi_l[Int_Jacobi_l.size()-1]*Int_Jacobi_m[Int_Jacobi_m.size()-1]   + skw_sym_outerprod(2, 3);
                            
                            o++;
                            
                            
                        }
                    }
                }
            }
        }

    }// End of Interiors
    
    Ti.Mult(SkwGradu, SkwGradshape);


}

void HSkwGrad_PentatopeElement::ProjectDivSkew(const FiniteElement& fe,
                                               ElementTransformation& Trans, DenseMatrix& DivSkew)
{
    mfem_error("SkwGrad_PentatopeElement_Fuentes::ProjectDivSkew not implemented");
    
}

} // end of namespace
