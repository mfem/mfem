#include "mfem.hpp"
#include <memory>

using namespace mfem;

struct _b_UserCtx
{
   Device            device;
   Mesh              mesh;
   const std::string smootherType,solverType,amgConfig;
   const int         order,refine;
   const double      jacobiVal;

   _b_UserCtx(const char *dev,const char *file, const char *smoother,
              const char *solver, const char *amg, int ord, int ref, double jacobi) :
     device(dev),mesh(file,1,1),smootherType(smoother),solverType(solver),amgConfig(amg),
     order(ord),refine(ref),jacobiVal(jacobi)
   {
      for (int i = 0; i < ref; ++i) { mesh.UniformRefinement(); }
      device.Print();
   }

};

using UserCtx = std::unique_ptr<_b_UserCtx>;

namespace exact
{
static constexpr double M_PI2 = M_PI*M_PI;

namespace helmholtz
{
double u(const mfem::Vector &xvec)
{
   const int    dim = xvec.Size();
   const double x = M_PI*xvec[0], y = M_PI*xvec[1];

   if (dim == 2)
   {
      return sin(x)*sin(y);
   }
   else
   {
      const double z = M_PI*xvec[2];
      return sin(x)*sin(y)*sin(z);
   }
}

double f(const mfem::Vector &xvec)
{
   const int    dim = xvec.Size();
   const double x = M_PI*xvec[0], y = M_PI*xvec[1];

   if (dim == 2)
   {
      return (sin(x)*sin(y))+(2*M_PI2*sin(x)*sin(y));
   }
   else
   {
      const double z = M_PI*xvec[2];
      return (sin(x)*sin(y)*sin(z))+(3*M_PI2*sin(x)*sin(y)*sin(z));
   }
}
} // namespace helmholtz

namespace maxwell
{
void u_vec(const mfem::Vector &xvec, mfem::Vector &u)
{
   const int    dim = xvec.Size();
   const double x = M_PI*xvec[0], y = M_PI*xvec[1];

   if (dim == 2)
   {
      u[0] = cos(x)*sin(y);
      u[1] = sin(x)*cos(y);
   }
   else
   {
      const double z = M_PI*xvec[2];
      u[0] = cos(x)*sin(y)*sin(z);
      u[1] = sin(x)*cos(y)*sin(z);
      u[2] = sin(x)*sin(y)*cos(z);
   }
}

void f_vec(const mfem::Vector &xvec, mfem::Vector &f)
{
   const int    dim = xvec.Size();
   const double x = M_PI*xvec[0], y = M_PI*xvec[1];

   if (dim == 2)
   {
      f[0] = cos(x)*sin(y);
      f[1] = sin(x)*cos(y);
   }
   else
   {
      const double z = M_PI*xvec[2];
      f[0] = cos(x)*sin(y)*sin(z);
      f[1] = sin(x)*cos(y)*sin(z);
      f[2] = sin(x)*sin(y)*cos(z);
   }
}
} // namespace maxwell
} // namespace exact

static UserCtx ParseCommandLineOptions(int argc, char *argv[])
{
   const char *meshFile = "../data/star.mesh";
   const char *smoother = "DR";
   const char *solver   = "amgx";
   const char *device   = "cuda";
   const char *amg      = "amgx.json";
   int        nRefine   = 1, order = 4;
   double     jacobiVal = 0.6666;
   OptionsParser args(argc,argv);

   args.AddOption(&device, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&meshFile,"-m","--mesh","Input mesh file");
   args.AddOption(&smoother,"-s","--smoother",
                  "Smoother to use (one of GS-Gauss Seidel, DR-distributive relaxation)");
   args.AddOption(&solver,"-S","--solver","Which solver to use (either direct, simpleamg, or amgx");
   args.AddOption(&amg,"-A","--amg-config","Path to amg config file, if using amgx");
   args.AddOption(&order,"-o","--order","Polynomial degree");
   args.AddOption(&jacobiVal,"-jv","--jacobi-value","Jacobi smoother value");
   args.AddOption(&nRefine,"-r","--refine",
                  "Number of times to refine the mesh uniformly");
   args.ParseCheck();

   return UserCtx{new _b_UserCtx{device,meshFile,smoother,solver,amg,order,nRefine,jacobiVal}};
}

int main(int argc, char *argv[])
{
   const UserCtx ctx = ParseCommandLineOptions(argc,argv);
   const int     dim = ctx->mesh.Dimension();

   FunctionCoefficient       f_coeff(exact::helmholtz::f);
   FunctionCoefficient       u_coeff(exact::helmholtz::u);
   VectorFunctionCoefficient f_vec_coeff(dim,exact::maxwell::f_vec);
   VectorFunctionCoefficient u_vec_coeff(dim,exact::maxwell::u_vec);

   std::unique_ptr<FiniteElementCollection> fec;
   fec.reset(new H1_FECollection(ctx->order,dim,BasisType::GaussLobatto));

   FiniteElementSpace fes(&ctx->mesh,fec.get());
   out<<"Number of finite element unknowns: "<<fes.GetTrueVSize()<<std::endl;

   Array<int> ess_dofs;
   fes.GetBoundaryTrueDofs(ess_dofs);

   ConstantCoefficient one(1.0);
   BilinearForm        a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble();
   a.SetDiagonalPolicy(Matrix::DIAG_ONE);

   LinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   GridFunction   x(&fes);
   Vector         X,B;
   OperatorHandle A;

   x = 0.0;
   a.FormLinearSystem(ess_dofs,x,b,A,X,B);

   StopWatch               chrono;
   LORDiscretization       lor(a,ess_dofs);
   const SparseMatrix      &ALor = lor.GetAssembledMatrix();
   std::unique_ptr<Solver> smoother;

   chrono.Start();
   if (ctx->smootherType == "DR") {
     LORInfo lorInfo(*lor.GetFESpace().GetMesh(),ctx->mesh,ctx->order);
     smoother.reset(new DRSmoother(lorInfo.Cluster(),&ALor,dim == 3));
   } else if (ctx->smootherType == "GS") {
     smoother.reset(new GSSmoother(ALor));
   } else if (ctx->smootherType == "J") {
     smoother.reset(new DSmoother(ALor,0,ctx->jacobiVal));
   } else {
     MFEM_ABORT("Unknown smoother type "<<ctx->smootherType);
   }

   LORSolver<SimpleAMG> lorSol(lor,ALor,*smoother,SimpleAMG::solverBackend::AMG_AMGX,MPI_COMM_WORLD,ctx->amgConfig);
   chrono.Stop();
   out<<"Setup time = "<<chrono.RealTime()<<std::endl;
   chrono.Clear();

   chrono.Start();
   PCG(*A,lorSol,B,X,1,500,1e-12,0.0);
   chrono.Stop();
   out<<"Solve time = "<<chrono.RealTime()<<std::endl;

   //a.RecoverFEMSolution(X,b,x);
   return 0;
}
