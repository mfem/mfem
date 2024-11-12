//                                MFEM Example 0
//
// Compile with: make getdetjelem
//
// Sample runs:  make getdetjelem && ./getdetjelem -nmin -0.2 -nmax 0.2 -o 2 -nt 100000
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   int order = 1;
   real_t noise_min = -0.1;
   real_t noise_max = 0.1;
   int seed = 5;
   int ntries = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&noise_min, "-nmin", "--nmin",
                  "Finite element polynomial degree");
   args.AddOption(&noise_max, "-nmax", "--nmax",
                  "Finite element polynomial degree");
   args.AddOption(&seed, "-seed", "--seed", "Finite element polynomial degree");
   args.AddOption(&ntries, "-nt", "--seed", "Finite element polynomial degree");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.

   Mesh mesh(Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, 1, 1.0, 1.0));

   mesh.SetCurvature(order, false, -1, Ordering::byNODES);

   GridFunction *x = mesh.GetNodes();
   Vector xsav = *x;
   Vector noise(x->Size());
   bool quaddetJ = false;

   IntegrationRules IntRulesMine = IntegrationRules(0, Quadrature1D::GaussLobatto);
   int n1D = order+1;
   int nJac1D = (2*order)*(order)+1;

   int nb1 = 16*order;
   int nb2 = 32*order;

   for (int n = 0; n < ntries && !quaddetJ; n++)
   {
      *x = xsav;

      noise.Randomize(seed*(n+1));
      noise *= noise_max - noise_min;
      noise += noise_min;

      for (int i = 0; i < x->Size(); i++)
      {
         (*x)(i) += noise(i);
      }

      const FiniteElement *fe = x->FESpace()->GetFE(0);
      const IntegrationRule ir = fe->GetNodes();
      ElementTransformation *transf = mesh.GetElementTransformation(0);
      DenseMatrix Jac(fe->GetDim());
      bool nodaldetJ = true;
      if (n == 0)
      {
         std::cout << ir.GetNPoints() << " Nodes" << std::endl;
         std::cout << nb1 << " IntRule1" << std::endl;
         std::cout << nb2 << " IntRule2" << std::endl;
      }
      for (int q = 0; q < ir.GetNPoints() && nodaldetJ; q++)
      {
         IntegrationPoint ip = ir.IntPoint(q);
         transf->SetIntPoint(&ip);
         Jac = transf->Jacobian();
         double detj = Jac.Det();
         if (detj <= 0.0)
         {
            nodaldetJ = false;
         }
      }
      IntegrationPoint ip;
      for (int i = 0; i < nb1 && nodaldetJ; i++)
      {
         for (int j = 0; j < nb1 && nodaldetJ; j++)
         {
            ip.x = i/(nb1-1.0);
            ip.y = j/(nb1-1.0);
            transf->SetIntPoint(&ip);
            Jac = transf->Jacobian();
            double detj = Jac.Det();
            if (detj <= 0.0)
            {
               nodaldetJ = false;
            }
         }
      }

      if (!nodaldetJ)
      {
         continue;
      }
      else
      {
         IntegrationPoint ip;
         for (int i = 0; i < nb2 && !quaddetJ; i++)
         {
            for (int j =0; j < nb2 && !quaddetJ; j++)
            {
               ip.x = i/(nb2-1.0);
               ip.y = j/(nb2-1.0);
               transf->SetIntPoint(&ip);
               Jac = transf->Jacobian();
               double detj = Jac.Det();
               if (detj <= 0.0)
               {
                  quaddetJ = true;
                  std::cout << "found a case: " << detj << std::endl;
                  break;
               }
            }
         }
      }
   }



   mesh.Save("semi-invert.mesh");


   if (true)
   {
      osockstream sock(19916, "localhost");
      sock << "mesh\n";
      mesh.Print(sock);
      sock.send();
      sock << "window_title 'Displacements'\n"
           << "window_geometry "
           << 00 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   return 0;
}
