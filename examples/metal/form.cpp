#include "mfem.hpp"
using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/star.mesh";
   int order = 2;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   Device device(device_config);
   device.Print();

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);

   FiniteElementSpace fespace(&mesh, &fec);

   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   GridFunction x(&fespace), y(&fespace);
   x = 0.0;

   BilinearForm a(&fespace);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   ConstantCoefficient one(1.0);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   //    a.AddDomainIntegrator(new BilinearFormIntegrator("u·v"));

   a.Assemble();
   a.Mult(x, y);

   // ir
   // fe, mesh = fes
   // a = inner(grad(u), grad(v))*dx = ∇u·∇v *dx = ∇u·∇v
   // a = inner(u, v)*dx = u·v *dx u·v

   return 0;
}