#include "mfem.hpp"
#include <iostream>
#include <memory>

#include "discrete_divergence.hpp"
#include "hdiv_linear_solver.hpp"

#include "../solvers/lor_mms.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref = 0, int par_ref = 0);

double one_over_r(const Vector &xvec)
{
   // xvec = [z, r]
   const double r = xvec[1];
   return r == 0.0 ? 0.0 : 1.0/r;
}

double f_rz(const Vector &xvec)
{
   const double z = xvec[0];
   const double r = xvec[1];

   // alpha is the coefficient in the equation -Delta(u) + alpha*u = f
   const double alpha = 1.0;
   const double f = -cos(z)*(4*sin(r) + 5*r*cos(r) - (2 + alpha)*r*r*sin(r));

   // scale integral by r because of coordinate transformation
   return r*f;
}

double u_rz(const Vector &xvec)
{
   const double z = xvec[0];
   const double r = xvec[1];

   return r*r*sin(r)*cos(z);
}

class RobinCoefficient : public Coefficient
{
   double Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      double xdata[3];
      Vector xvec(xdata, 3);
      T.Transform(ip, xvec);
      const int dim = xvec.Size();

      Vector n(dim);
      CalcOrtho(T.Jacobian(), n);
      n /= n.Norml2();

      const double p_val = u_rz(xvec);
      const double z = xvec[0];
      const double r = xvec[1];

      if (dim == 2)
      {
         const double dpdz = -r*r*sin(r)*sin(z);
         const double dpdr = r*cos(z)*(r*cos(r) + 2*sin(r));
         const double u_val = n[0]*dpdz + n[1]*dpdr;
         return p_val + u_val;
      }
      else
      {
         MFEM_ABORT("Not implemented");
      }
      return 0.0;
   }
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "rz.mesh";
   const char *device_config = "cpu";
   int ser_ref = 1;
   int par_ref = 1;
   int order = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");

   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLegendre;
   const int mt = FiniteElement::VALUE;
   RT_FECollection fec_rt(order-1, dim, b1, b2);
   L2_FECollection fec_l2(order-1, dim, b2, mt);
   ParFiniteElementSpace fes_rt(&mesh, &fec_rt);
   ParFiniteElementSpace fes_l2(&mesh, &fec_l2);

   HYPRE_BigInt ndofs_rt = fes_rt.GlobalTrueVSize();
   HYPRE_BigInt ndofs_l2 = fes_l2.GlobalTrueVSize();

   if (Mpi::Root())
   {
      cout << "\nRT DOFs: " << ndofs_rt << "\nL2 DOFs: " << ndofs_l2 << endl;
   }

   Array<int> ess_rt_dofs; // empty

   // f is the RHS, u is the exact solution
   FunctionCoefficient f_coeff(f_rz), u_coeff(u_rz);

   // Assemble the right-hand side for the scalar (L2) unknown.
   ParLinearForm b_l2(&fes_l2);
   // f_coeff has to include the r scaling for the coordinate transformation
   b_l2.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
   b_l2.UseFastAssembly(true);
   b_l2.Assemble();

   // Coefficient to enforce Robin boundary condition
   RobinCoefficient bc_coeff;

   // Enforce Robin boundary conditions by adding the boundary term to the flux
   // equation.
   ParLinearForm b_rt(&fes_rt);
   b_rt.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(bc_coeff));
   b_rt.UseFastAssembly(true);
   b_rt.Assemble();

   if (Mpi::Root()) { cout << "\nSaddle point solver... " << flush; }
   tic_toc.Clear(); tic_toc.Start();

   // Have to scale the RT mass matrix by (1/r)
   FunctionCoefficient r_recip_coeff(one_over_r);
   // Have to scale the L2 mass matrix by r
   FunctionCoefficient r_coeff([](const Vector &xvec) { return xvec[1]; });

   const auto solver_mode = HdivSaddlePointSolver::Mode::DARCY;
   HdivSaddlePointSolver saddle_point_solver(
      mesh, fes_rt, fes_l2, r_coeff, r_recip_coeff, r_recip_coeff, ess_rt_dofs, solver_mode);

   const Array<int> &offsets = saddle_point_solver.GetOffsets();
   BlockVector X_block(offsets), B_block(offsets);

   b_l2.ParallelAssemble(B_block.GetBlock(0));
   b_rt.ParallelAssemble(B_block.GetBlock(1));
   B_block.SyncFromBlocks();

   X_block = 0.0;
   saddle_point_solver.Mult(B_block, X_block);
   X_block.SyncToBlocks();

   if (Mpi::Root())
   {
      cout << "Done.\nIterations: "
           << saddle_point_solver.GetNumIterations()
           << "\nElapsed: " << tic_toc.RealTime() << endl;
   }

   ParGridFunction x(&fes_l2);
   x.SetFromTrueDofs(X_block.GetBlock(0));
   ParGridFunction flux(&fes_rt);
   flux.SetFromTrueDofs(X_block.GetBlock(1));

   const double error = x.ComputeL2Error(u_coeff);
   if (Mpi::Root()) { cout << "L2 error: " << error << endl; }

   ParGridFunction u_ex(&fes_l2), er(&fes_l2);
   u_ex.ProjectCoefficient(u_coeff);
   er = x;
   er -= u_ex;

   ParaViewDataCollection pv("RZ", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(order + 1);
   pv.RegisterField("u", &x);
   pv.RegisterField("flux", &flux);
   pv.RegisterField("exact", &u_ex);
   pv.RegisterField("error", &er);
   pv.SetCycle(0);
   pv.SetTime(0);
   pv.Save();

   return 0;
}

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}
