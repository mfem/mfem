#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "linalg/tensor/factories/factories.hpp"
#include "linalg/tensor/operators/operators.hpp"
#include "linalg/tensor/utilities/utilities.hpp"

using namespace std;
using namespace mfem;

template <int Dim,
          int VDim,
          bool IsTensor,
          int Dofs = Dynamic,
          int Quads = Dynamic,
          int BatchSize = 1>
static void ApplyDGMassInverse(const int ne,
                               const Array<double> &b,
                               const Array<double> &bt,
                               const Vector &d,
                               const Vector &x,
                               Vector &y,
                               const int dofs = Dofs,
                               const int quads = Quads)
{
   // config_static_device_tensor_is<
   // ThreadTensor<Dim>::template static_type
   // > static_tensor;
   auto config  = MakeConfig(quads,
                             config_dim_is<Dim>(),
                             config_is_tensor<IsTensor>(),
                             config_quads_is<Quads>());
   auto B       = MakeBasis<Dofs>(config, dofs, quads, b.Read(), bt.Read());
   const auto X = MakeDoFs<Dofs,VDim>(config, dofs, x.Read(), ne);
   const auto D = MakeQData<0>(config, d.Read(), ne);
   auto Y       = MakeDoFs<Dofs,VDim>(config, dofs, y.ReadWrite(), ne);
   MFEM_FORALL_CONFIG(config, e, ne,
   {
      auto op = transpose(B) * D(e) * B;
      Identity P;
      int iter = 400;
      double tol = 1e-12;
      Y(e) += conjugate_gradient(op, X(e), P, iter, tol);
   });
}

static void DGPAMassInverseApply(const int dim,
                                 const int D1D,
                                 const int Q1D,
                                 const int NE,
                                 const Array<double> &B,
                                 const Array<double> &Bt,
                                 const Vector &D,
                                 const Vector &X,
                                 Vector &Y)
{
   const int id = (D1D << 4) | Q1D;
   if (dim == 1)
   {
      switch (id)
      {
         // case 0x22: return Apply1D<2,2,16>(NE,B,Bt,D,X,Y);
         default:   mfem_error("default impl not yet implemented.");
      }
   }
   else if (dim == 2)
   {
      switch (id)
      {
         // default:   return PAMassApply2D(NE,B,Bt,D,X,Y,D1D,Q1D);
         case 0x22: return ApplyDGMassInverse<2,0,true,2,2,16>(NE,B,Bt,D,X,Y);
         case 0x24: return ApplyDGMassInverse<2,0,true,2,4,16>(NE,B,Bt,D,X,Y);
         case 0x33: return ApplyDGMassInverse<2,0,true,3,3,16>(NE,B,Bt,D,X,Y);
         case 0x34: return ApplyDGMassInverse<2,0,true,3,4,16>(NE,B,Bt,D,X,Y);
         case 0x35: return ApplyDGMassInverse<2,0,true,3,5,16>(NE,B,Bt,D,X,Y);
         case 0x36: return ApplyDGMassInverse<2,0,true,3,6,16>(NE,B,Bt,D,X,Y);
         case 0x44: return ApplyDGMassInverse<2,0,true,4,4,8>(NE,B,Bt,D,X,Y);
         case 0x46: return ApplyDGMassInverse<2,0,true,4,6,8>(NE,B,Bt,D,X,Y);
         case 0x48: return ApplyDGMassInverse<2,0,true,4,8,4>(NE,B,Bt,D,X,Y);
         case 0x55: return ApplyDGMassInverse<2,0,true,5,5,8>(NE,B,Bt,D,X,Y);
         case 0x57: return ApplyDGMassInverse<2,0,true,5,7,8>(NE,B,Bt,D,X,Y);
         case 0x58: return ApplyDGMassInverse<2,0,true,5,8,2>(NE,B,Bt,D,X,Y);
         case 0x66: return ApplyDGMassInverse<2,0,true,6,6,4>(NE,B,Bt,D,X,Y);
         case 0x77: return ApplyDGMassInverse<2,0,true,7,7,4>(NE,B,Bt,D,X,Y);
         case 0x88: return ApplyDGMassInverse<2,0,true,8,8,2>(NE,B,Bt,D,X,Y);
         case 0x99: return ApplyDGMassInverse<2,0,true,9,9,2>(NE,B,Bt,D,X,Y);
         // default:   return ApplyDGMassInverse<2,0,true>(NE,B,Bt,D,X,Y,D1D,Q1D);
         default:   MFEM_ABORT("default impl not yet implemented with D1D=" << D1D <<
                                  ", Q1D=" << Q1D << ".");
      }
   }
   else if (dim == 3)
   {
      switch (id)
      {
         case 0x23: return ApplyDGMassInverse<3,0,true,2,3>(NE,B,Bt,D,X,Y);
         case 0x24: return ApplyDGMassInverse<3,0,true,2,4>(NE,B,Bt,D,X,Y);
         case 0x34: return ApplyDGMassInverse<3,0,true,3,4>(NE,B,Bt,D,X,Y);
         case 0x35: return ApplyDGMassInverse<3,0,true,3,5>(NE,B,Bt,D,X,Y);
         case 0x36: return ApplyDGMassInverse<3,0,true,3,6>(NE,B,Bt,D,X,Y);
         case 0x37: return ApplyDGMassInverse<3,0,true,3,7>(NE,B,Bt,D,X,Y);
         case 0x45: return ApplyDGMassInverse<3,0,true,4,5>(NE,B,Bt,D,X,Y);
         case 0x46: return ApplyDGMassInverse<3,0,true,4,6>(NE,B,Bt,D,X,Y);
         case 0x48: return ApplyDGMassInverse<3,0,true,4,8>(NE,B,Bt,D,X,Y);
         case 0x56: return ApplyDGMassInverse<3,0,true,5,6>(NE,B,Bt,D,X,Y);
         case 0x58: return ApplyDGMassInverse<3,0,true,5,8>(NE,B,Bt,D,X,Y);
         case 0x67: return ApplyDGMassInverse<3,0,true,6,7>(NE,B,Bt,D,X,Y);
         case 0x78: return ApplyDGMassInverse<3,0,true,7,8>(NE,B,Bt,D,X,Y);
         case 0x89: return ApplyDGMassInverse<3,0,true,8,9>(NE,B,Bt,D,X,Y);
         case 0x9A: return ApplyDGMassInverse<3,0,true,9,10>(NE,B,Bt,D,X,Y);
         // default:   return ApplyDGMassInverse<3,0,true>(NE,B,Bt,D,X,Y,D1D,Q1D);
         default:   MFEM_ABORT("default impl not yet implemented with D1D=" << D1D <<
                                  ", Q1D=" << Q1D << ".");
      }
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Unknown kernel.");
}

class DGMassInverse: public Operator, private MassIntegrator
{
public:
   DGMassInverse(const FiniteElementSpace &fes, Coefficient &coeff)
      : Operator(fes.GetVSize()), MassIntegrator(coeff)
   {
      AssemblePA(fes);
   }

   void Mult(const Vector &x, Vector &y) const
   {
      DGPAMassInverseApply(dim, dofs1D, quad1D, ne, maps->B, maps->Bt, pa_data, x, y);
   }
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 2;
   int order = 1;
   const char *device_config = "cpu";
   // bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Device device(device_config);
   device.Print();

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   Array<int> ess_tdof_list;

   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   /// Legacy CG
   GridFunction x_ref(&fespace);
   x_ref = 0.0;
   cout << std::endl << "Building Old CG " << std::endl;
   tic_toc.Clear();
   tic_toc.Start();
   BilinearForm a(&fespace);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.AddDomainIntegrator(new MassIntegrator(one));
   a.Assemble();
   cout << "Form Linear System " << std::endl;
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x_ref, b, A, X, B);
   tic_toc.Stop();
   const double old_cg_setup = tic_toc.RealTime();
   cout << "Legacy CG setup time: " << old_cg_setup << std::endl;

   cout << "Size of linear system: " << A->Height() << endl;

   const int print_level = 1;
   const int max_iter = 400;
   const double rtol = 1e-12;
   const double atol = 0.0;
   tic_toc.Clear();
   tic_toc.Start();
   CG(*A, B, X, print_level, max_iter, rtol, atol);
   tic_toc.Stop();
   const double old_cg_compute = tic_toc.RealTime();
   cout << "Legacy CG computation time: " << old_cg_compute << std::endl;
   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x_ref);

   /// New CG
   GridFunction x_test(&fespace);
   x_test = 0.0;
   // Setup
   cout << std::endl << "Building New CG " << std::endl;
   tic_toc.Clear();
   tic_toc.Start();
   DGMassInverse new_op(fespace,one);
   tic_toc.Stop();
   const double new_cg_setup = tic_toc.RealTime();
   cout << "New CG setup time: " << new_cg_setup << std::endl;
   // Compute
   tic_toc.Clear();
   tic_toc.Start();
   new_op.Mult(b,x_test);
   tic_toc.Stop();
   const double new_cg_compute = tic_toc.RealTime();
   cout << "New CG computation time: " << new_cg_compute << std::endl;

   GridFunction diff(&fespace);
   diff = x_ref;
   diff -= x_test;

   cout << std::endl << "==Error==" << std::endl;
   cout << "Error norm: " << diff.Norml2() << "." << std::endl;
   cout << "==Setup==" << std::endl;
   cout << "Legacy CG setup time: " << old_cg_setup << "s." << std::endl;
   cout << "New CG setup time: " << new_cg_setup << "s." << std::endl;
   cout << "CG setup diff: " << new_cg_setup - old_cg_setup << "s." << std::endl;
   cout << "CG setup speedup: " << old_cg_setup / new_cg_setup << "x." <<
        std::endl;
   cout << "==Compute==" << std::endl;
   cout << "Legacy CG computation time: " << old_cg_compute << "s." << std::endl;
   cout << "New CG computation time: " << new_cg_compute << "s." << std::endl;
   cout << "CG compute diff: " << new_cg_compute - old_cg_compute << "s." <<
        std::endl;
   cout << "CG compute speedup: " << old_cg_compute / new_cg_compute << "x." <<
        std::endl;

   // // 13. Save the refined mesh and the solution. This output can be viewed later
   // //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   // ofstream mesh_ofs("refined.mesh");
   // mesh_ofs.precision(8);
   // mesh.Print(mesh_ofs);
   // ofstream sol_ofs("sol.gf");
   // sol_ofs.precision(8);
   // x_ref.Save(sol_ofs);

   // // 14. Send the solution by socket to a GLVis server.
   // if (visualization)
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock.precision(8);
   //    sol_sock << "solution\n" << mesh << x << flush;
   // }

   return 0;
}
