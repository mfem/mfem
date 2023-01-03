#include "mfem.hpp"
#include "integ_algoim.hpp"
#include "shape_grad.hpp"
#include "marking.hpp"

using namespace mfem;
using namespace std;

//Level set function for sphere in 3D and circle in 2D
double sphere_ls(const Vector &x)
{
   double r2= x*x;
   return -sqrt(r2)+1.0;//the radius is 1.0
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
   int order = 1;
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
   args.AddOption(&ls_type,
                  "-ls",
                  "--ls-type",
                  "Level set type: 1: circle, 2 sinusoidal wave");
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

   mesh->EnsureNodes();
   GridFunction* nodes=mesh->GetNodes();

   // Define the finite element space for the level-set function.
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

   ConstantCoefficient one(1.0);
   Array<int> el_marks; el_marks.SetSize(mesh->GetNE());
   for(int i=0;i<mesh->GetNE();i++){ el_marks[i]= ElementMarker::SBElementType::CUT;}


   NonlinearForm nf(&fespace);
   nf.AddDomainIntegrator(new DVolShapeIntegrator(one,el_marks));

   double vol=nf.GetEnergy(x);
   Vector grad(x.Size());
   nf.Mult(x,grad);
   std::cout<<setprecision(8);
   grad.Print(std::cout,x.Size());

   std::cout<<"FD+"<<std::endl;
   double dd=0.00000001;
   for(int i=0;i<x.Size();i++){
       x[i]=x[i]-dd;
       double vol1=nf.GetEnergy(x);
       x[i]=x[i]+dd;
       std::cout<<(vol1-vol)/dd<<" ";
   }
   std::cout<<std::endl;

   std::cout<<"vol="<<vol<<std::endl;

   Vector di(nodes->Size()); di=0.0;
   Vector si(nodes->Size()); si=0.0;

   for(int i=0;i<nodes->Size();i++)
   {
       (*nodes)[i]=(*nodes)[i]+dd;
       mesh->DeleteGeometricFactors();
       double vol1=nf.GetEnergy(x);
       si[i]=(vol1-vol)/dd;
       (*nodes)[i]=(*nodes)[i]-dd;
   }

   std::cout<<"si"<<std::endl;
   si.Print(std::cout,2);


   // ParaView output.
   ParaViewDataCollection dacol("ParaViewDistance", mesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("x", &x);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();




   delete ls_coeff;
   delete mesh;

   return 0;
}
