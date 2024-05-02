#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class PyramidScalarFunc : public Coefficient
{
public:
   enum PyrSFunc {LAMBDA1 = 1, LAMBDA2, LAMBDA3, LAMBDA4, LAMBDA5,
                  PHI_Q, PHI_T
                 };

private:
   FuentesPyramid pyr;

   PyrSFunc func;
   int p, q, r;

   mutable DenseMatrix u_ij;

public:

   PyramidScalarFunc() = default;

   void SetFunc(PyrSFunc f) { func = f; }

   void SetP(int p_) { p = p_; }
   void SetPQ(int p_, int q_) { p = p_; q = q_; }
   void SetPQR(int p_, int q_, int r_) { p = p_; q = q_; r = r_; }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      Vector u(3);
      T.Transform(ip, u);

      real_t x = u(0);
      real_t y = u(1);
      real_t z = u(2);
      Vector xy({x,y});

      switch (func)
      {
         case LAMBDA1:
            return pyr.lam1(x, y, z);
         case LAMBDA2:
            return pyr.lam2(x, y, z);
         case LAMBDA3:
            return pyr.lam3(x, y, z);
         case LAMBDA4:
            return pyr.lam4(x, y, z);
         case LAMBDA5:
            return pyr.lam5(x, y, z);
         case PHI_Q:
         {
            int n = std::max(p, q);
            u_ij.SetSize(n+1, n+1);
            pyr.phi_Q(n, pyr.mu01(z, xy, 1), pyr.mu01(z, xy, 2), u_ij);
            return u_ij(p, q);
         }
         case PHI_T:
         {
            int n = p + q;
            u_ij.SetSize(n+1, n+1);
            pyr.phi_T(n, pyr.nu012(z, xy, 1), u_ij);
            return u_ij(p, q);
         }
         default:
            return 0.0;
      }
   }
};

class PyramidVectorFunc : public VectorCoefficient
{
public:
   enum PyrVFunc {DLAMBDA1 = 1, DLAMBDA2, DLAMBDA3, DLAMBDA4, DLAMBDA5,
                  E_E, E_Q, DE_Q, V_Q, V_T, V_L, V_R
                 };

private:
   FuentesPyramid pyr;

   PyrVFunc func;
   int p, q, r;

   mutable DenseMatrix u_ij;
   mutable DenseTensor u_ijk;
   mutable DenseTensor du_ijk;

public:

   PyramidVectorFunc() : VectorCoefficient(3) {}

   void SetFunc(PyrVFunc f) { func = f; }

   void SetP(int p_) { p = p_; }
   void SetPQ(int p_, int q_) { p = p_; q = q_; }
   void SetPQR(int p_, int q_, int r_) { p = p_; q = q_; r = r_; }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      V.SetSize(3);

      Vector u(3);
      T.Transform(ip, u);
      V = u;

      real_t x = u(0);
      real_t y = u(1);
      real_t z = u(2);
      Vector xy({x,y});

      switch (func)
      {
         case DLAMBDA1:
            V = pyr.grad_lam1(x, y, z);
            break;
         case DLAMBDA2:
            V = pyr.grad_lam2(x, y, z);
            break;
         case DLAMBDA3:
            V = pyr.grad_lam3(x, y, z);
            break;
         case DLAMBDA4:
            V = pyr.grad_lam4(x, y, z);
            break;
         case DLAMBDA5:
            V = pyr.grad_lam5(x, y, z);
            break;
         case E_Q:
         {
            int n = std::max(p, q) + 1;
            u_ijk.SetSize(n+1, n+2, 3);
            du_ijk.SetSize(n+1, n+2, 3);
            pyr.E_Q(p,
                    pyr.mu01(z, xy, 1), pyr.grad_mu01(z, xy, 1),
                    pyr.mu01(z, xy, 2), pyr.grad_mu01(z, xy, 2),
                    u_ijk, du_ijk);
            V(0) = u_ijk(p, q, 0);
            V(1) = u_ijk(p, q, 1);
            V(2) = u_ijk(p, q, 2);
         }
         break;
         case DE_Q:
         {
            int n = std::max(p, q) + 1;
            u_ijk.SetSize(n+1, n+2, 3);
            du_ijk.SetSize(n+1, n+2, 3);
            pyr.E_Q(p,
                    pyr.mu01(z, xy, 1), pyr.grad_mu01(z, xy, 1),
                    pyr.mu01(z, xy, 2), pyr.grad_mu01(z, xy, 2),
                    u_ijk, du_ijk);
            V(0) = du_ijk(p, q, 0);
            V(1) = du_ijk(p, q, 1);
            V(2) = du_ijk(p, q, 2);
         }
         break;
         case V_Q:
         {
            int n = std::max(p, q) + 1;
            u_ijk.SetSize(n+1, n+1, 3);
            pyr.V_Q(n, pyr.mu01(z, xy, 1), pyr.mu01_grad_mu01(z, xy, 1),
                    pyr.mu01(z, xy, 2), pyr.mu01_grad_mu01(z, xy, 2), u_ijk);
            V(0) = u_ijk(p, q, 0);
            V(1) = u_ijk(p, q, 1);
            V(2) = u_ijk(p, q, 2);
         }
         break;
         case V_T:
         {
            int n = std::max(p, q) + 1;
            u_ijk.SetSize(n+1, n+1, 3);
            pyr.V_T(n, pyr.nu012(z, xy, 1), pyr.nu012_grad_nu012(z, xy, 1), u_ijk);
            V(0) = u_ijk(p, q, 0);
            V(1) = u_ijk(p, q, 1);
            V(2) = u_ijk(p, q, 2);
         }
         break;
         case V_L:
         {
            int n = std::max(p, q);
            u_ijk.SetSize(n+1, n+1, 3);
            pyr.V_L(n, pyr.mu01(z, xy, 1), pyr.grad_mu01(z, xy, 1),
                    pyr.mu01(z, xy, 2), pyr.grad_mu01(z, xy, 2),
                    pyr.mu0(z), pyr.grad_mu0(z), u_ijk);
            V(0) = u_ijk(p, q, 0);
            V(1) = u_ijk(p, q, 1);
            V(2) = u_ijk(p, q, 2);
         }
         break;
         case V_R:
         {
            u_ij.SetSize(p + 1, 3);
            pyr.V_R(p, pyr.mu01(z, xy, 1), pyr.grad_mu01(z, xy, 1),
                    pyr.mu1(z, xy, 2), pyr.grad_mu1(z, xy, 2),
                    pyr.mu0(z), pyr.grad_mu0(z), u_ij);
            V(0) = u_ij(p, 0);
            V(1) = u_ij(p, 1);
            V(2) = u_ij(p, 2);
         }
         break;
         default:
            V = 0.0;
      }
   }
};

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/ref-pyramid-2tet.mesh";
   int ref_levels = 2;
   int order = 2;
   bool disc = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&disc, "-disc", "--discontinuous", "-cont",
                  "--continuous",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   H1_FECollection fec_h1(order, dim);
   L2_FECollection fec_l2(order, dim);
   FiniteElementCollection *fec = &fec_h1;
   if (disc)
   {
      fec = &fec_l2;
   }
   FiniteElementSpace sfes(&mesh, fec);
   FiniteElementSpace vfes(&mesh, fec, dim);

   GridFunction s(&sfes);
   s = 0.0;

   PyramidScalarFunc pyrCoef;

   for (int c=1; c<8; c++)
   {
      pyrCoef.SetFunc((PyramidScalarFunc::PyrSFunc)c);

      if (c == 6)
      {
         pyrCoef.SetPQ(2,2);
      }
      else if (c == 7)
      {
         pyrCoef.SetPQ(2,1);
      }

      s.ProjectCoefficient(pyrCoef);

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << s
                  << "window_title 'S" << c << "'\n" << flush;
      }
   }

   GridFunction v(&vfes);

   PyramidVectorFunc pyrVCoef;

   for (int c=1; c<10; c++)
   {
      pyrVCoef.SetFunc((PyramidVectorFunc::PyrVFunc)c);

      if (c == PyramidVectorFunc::E_Q)
      {
         for (int j=2; j<=2; j++)
            for (int i=1; i<2; i++)
            {
               pyrVCoef.SetPQ(i, j);
               v.ProjectCoefficient(pyrVCoef);
               if (visualization)
               {
                  char vishost[] = "localhost";
                  int  visport   = 19916;
                  socketstream sol_sock(vishost, visport);
                  sol_sock.precision(8);
                  sol_sock << "solution\n" << mesh << v
                           << "window_title 'E_Q " << i << j << "'\n" << flush;
               }
            }
      }
      if (c == PyramidVectorFunc::DE_Q)
      {
         for (int j=2; j<=2; j++)
            for (int i=1; i<2; i++)
            {
               pyrVCoef.SetPQ(i, j);
               v.ProjectCoefficient(pyrVCoef);
               if (visualization)
               {
                  char vishost[] = "localhost";
                  int  visport   = 19916;
                  socketstream sol_sock(vishost, visport);
                  sol_sock.precision(8);
                  sol_sock << "solution\n" << mesh << v
                           << "window_title 'dE_Q " << i << j << "'\n" << flush;
               }
            }
      }
      else if (c == PyramidVectorFunc::V_Q)
      {
         for (int j=0; j<2; j++)
            for (int i=0; i<2; i++)
            {
               pyrVCoef.SetPQ(i,j);
               v.ProjectCoefficient(pyrVCoef);
               if (visualization)
               {
                  char vishost[] = "localhost";
                  int  visport   = 19916;
                  socketstream sol_sock(vishost, visport);
                  sol_sock.precision(8);
                  sol_sock << "solution\n" << mesh << v
                           << "window_title 'V_Q " << i << j << "'\n" << flush;
               }
            }
      }
      else if (c == PyramidVectorFunc::V_T)
      {
         for (int j=0; j<2; j++)
            for (int i=0; i+j<2; i++)
            {
               pyrVCoef.SetPQ(i,j);
               v.ProjectCoefficient(pyrVCoef);
               if (visualization)
               {
                  char vishost[] = "localhost";
                  int  visport   = 19916;
                  socketstream sol_sock(vishost, visport);
                  sol_sock.precision(8);
                  sol_sock << "solution\n" << mesh << v
                           << "window_title 'V_T "<< i << j << "'\n" << flush;
               }
            }
      }
      else if (c == PyramidVectorFunc::V_L)
      {
         pyrVCoef.SetPQ(2,2);
         v.ProjectCoefficient(pyrVCoef);
         if (visualization)
         {
            char vishost[] = "localhost";
            int  visport   = 19916;
            socketstream sol_sock(vishost, visport);
            sol_sock.precision(8);
            sol_sock << "solution\n" << mesh << v
                     << "window_title 'V_L " << 2 << 2 << "'\n" << flush;
         }
      }
      else if (c == PyramidVectorFunc::V_R)
      {
         pyrVCoef.SetP(2);
         v.ProjectCoefficient(pyrVCoef);
         if (visualization)
         {
            char vishost[] = "localhost";
            int  visport   = 19916;
            socketstream sol_sock(vishost, visport);
            sol_sock.precision(8);
            sol_sock << "solution\n" << mesh << v
                     << "window_title 'V_R " << 2 << "'\n" << flush;
         }
      }
      else
      {
         v.ProjectCoefficient(pyrVCoef);

         if (visualization)
         {
            char vishost[] = "localhost";
            int  visport   = 19916;
            socketstream sol_sock(vishost, visport);
            sol_sock.precision(8);
            sol_sock << "solution\n" << mesh << v
                     << "window_title 'V" << c << "'\n" << flush;
         }
      }
   }
}
