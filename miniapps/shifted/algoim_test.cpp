#include "mfem.hpp"
#include "integ_algoim.hpp"

using namespace mfem;
using namespace std;

//Level set function for sphere in 3D and circle in 2D
double sphere_ls(const Vector &x)
{
   double r2= x*x;
   return -sqrt(r2)+1.0;//the radius is 0.5
}

//Level set function for a sinusoidal wave.
//Resulting zero isocontour is at y=0.5-(0.1*sin(3*pi*x+pi/2))
double sinusoidal_ls(const Vector &x)
{
   double a1 = 20., a2 = 2., a3 = 3.;
   return tanh(a1*(x(1)-0.5) + a2*sin(a3*(x(0)-0.5)*M_PI));
}

int main(int argc, char *argv[])
{
   //   Parse command-line options
   const char *mesh_file = "../../data/star-q3.mesh";
   int ser_ref_levels = 1;
   int order = 2;
   int iorder = 2; //MFEM integration integration points
   int aorder = 2; //Algoim integration integration points
   bool visualization = true;
   int print_level = 0;
   int ls_type = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&iorder,
                  "-io",
                  "--iorder",
                  "MFEM Integration order.");
   args.AddOption(&aorder,
                  "-ao",
                  "--aorder",
                  "Algoim Integration order.");
   args.AddOption(&ls_type,
                  "-ls",
                  "--ls-type",
                  "Level set type: 1: circle, 2 sinusoidal wave");
   args.AddOption(&visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption((&print_level), "-prt", "--print-level", "Print level.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   //   Read the (serial) mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   //    Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // Define the finite element spaces for the solution
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(mesh, &fec, 1, Ordering::byVDIM);
   int glob_size = fespace.GetTrueVSize();
   std::cout << "Number of finite element unknowns: " << glob_size << std::endl;

   // Define the level set grid function
   GridFunction x(&fespace);

   // Define the level set coefficient
   Coefficient *ls_coeff = nullptr;
   if (ls_type == 1)
   {
      ls_coeff=new FunctionCoefficient(sphere_ls);
   }
   else if (ls_type == 2)
   {
      ls_coeff=new FunctionCoefficient(sinusoidal_ls);
   }
   else
   {
      MFEM_ABORT("Level set coefficient not defined");
   }

   // Project the coefficient onto the LS grid function
   x.ProjectCoefficient(*ls_coeff);

   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      x.Save(sock);
      sock.send();
      sock << "window_title 'Level set'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmmclA" << endl;
   }

   ElementTransformation *trans;

   double w;
   const IntegrationRule* ir=nullptr;

   // loop over the elements
   double vol=0.0;
   double area=0.0;

#ifdef MFEM_USE_ALGOIM
   DenseMatrix bmat; //gradients of the shape functions in isoparametric space
   DenseMatrix pmat; //gradients of the shape functions in physical space
   Vector inormal; //normal to the level set in isoparametric space
   Vector tnormal; //normal to the level set in physical space
   Vector lsfun; //level set function restricted to an element
   DofTransformation *doftrans;
   Array<int> vdofs;
   for (int i=0; i<fespace.GetNE(); i++)
   {
      const FiniteElement* el=fespace.GetFE(i);
      //get the element transformation
      trans = fespace.GetElementTransformation(i);

      //extract the element vector from the level-set
      doftrans = fespace.GetElementVDofs(i,vdofs);
      x.GetSubVector(vdofs, lsfun);

      //contruct Algoim integration object
      AlgoimIntegrationRule* air=new AlgoimIntegrationRule(aorder,*el,
                                                           *trans,lsfun);

      //compute the volume contribution from the element
      ir=air->GetVolumeIntegrationRule();
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         trans->SetIntPoint(&ip);
         w = trans->Weight();
         w = ip.weight * w;
         vol=vol+w;
      }

      //compute the perimeter/area contribution from the element
      bmat.SetSize(el->GetDof(),el->GetDim());
      pmat.SetSize(el->GetDof(),el->GetDim());
      inormal.SetSize(el->GetDim());
      tnormal.SetSize(el->GetDim());

      ir=air->GetSurfaceIntegrationRule();
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         trans->SetIntPoint(&ip);

         el->CalcDShape(ip,bmat);
         Mult(bmat, trans->AdjugateJacobian(), pmat);
         //compute the normal to the LS in isoparametric space
         bmat.MultTranspose(lsfun,inormal);
         //compute the normal to the LS in physicsl space
         pmat.MultTranspose(lsfun,tnormal);
         w = ip.weight*tnormal.Norml2()/inormal.Norml2();
         area=area+w;
      }

      //delete the Algoim integration rule object
      //the integration rules are constructed for the
      //particular element and level set
      delete air;
   }
#endif
   double exact_volume = -10, exact_area = -10;
   if (ls_type == 1)
   {
      if (strncmp(mesh_file,"../../data/star-q3.mesh",100) == 0)
      {
         exact_volume = M_PI;
         exact_area   = M_PI*2;
      }
      else if (strncmp(mesh_file, "../../data/inline-quad.mesh",100) == 0)
      {
         exact_volume = M_PI/4;
         exact_area   = M_PI/2;
      }
   }
   else if (ls_type == 2)
   {
      if (strncmp(mesh_file, "../../data/inline-quad.mesh",100) == 0)
      {
         exact_volume = 0.5;
         exact_area   = 1.194452300992437;
      }
   }
   if (exact_volume > 0)
   {
      std::cout<<"Algoim Volume="<<vol<<" Error="<<vol-exact_volume<<std::endl;
      std::cout<<"Algoim Area="<<area<<" Error="<<area-exact_area<<std::endl;
   }
   else
   {
      std::cout<<"Algoim Volume="<<vol<<std::endl;
      std::cout<<"Algoim Area="<<area<<std::endl;
   }

   //Perform standard MFEM integration
   vol=0.0;
   for (int i=0; i<fespace.GetNE(); i++)
   {
      const FiniteElement* el=fespace.GetFE(i);
      //get the element transformation
      trans = fespace.GetElementTransformation(i);

      //compute the volume contribution from the element
      ir=&IntRules.Get(el->GetGeomType(), iorder);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         trans->SetIntPoint(&ip);
         double vlsf=x.GetValue(*trans,ip);
         if (vlsf>=0.0)
         {
            w = trans->Weight();
            w = ip.weight * w;
            vol=vol+w;
         }
      }

   }
   if (exact_volume > 0.0)
   {
      std::cout<<"MFEM Volume="<<vol<<" Error="<<vol-exact_volume<<std::endl;
   }
   else
   {
      std::cout<<"MFEM Volume="<<vol<<std::endl;
   }

   ParaViewDataCollection dacol("ParaViewDistance", mesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("dist",&x);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();


   delete ls_coeff;
   delete mesh;

   return 0;
}
