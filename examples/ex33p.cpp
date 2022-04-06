#include "mfem.hpp"

using namespace mfem;
using namespace std;

double SourceFunction(const Vector &x)
{
   return (2.*M_PI*M_PI + .1)*sin(M_PI*x(0))*sin(M_PI*x(1));
}

double ExactFunction(const Vector &x)
{
   return sin(M_PI*x(0))*sin(M_PI*x(1));
}

int main(int argc, char* argv[])
{
   MPI_Init(NULL, NULL);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int Ne = 10;
   int p = 1;
   double kappa = 0;
   bool use_beta = true;
   OptionsParser args(argc, argv);
   args.AddOption(&Ne, "-n", "--Ne", "number of elements");
   args.AddOption(&p, "-p", "--fe_order", "finite element order");
   args.AddOption(&kappa, "-k", "--kappa", "penalty parameter");
   args.AddOption(&use_beta, "-m", "--mdldg", "-l", "--ldg", "use MDLDG");
   args.Parse();
   if (!args.Good())
   {
      if (rank==0) { args.PrintUsage(cout); }
      return 1;
   }
   if (rank==0) { args.PrintOptions(cout); }
   Mesh smesh(Ne, Ne, Element::QUADRILATERAL);
   int dim = smesh.Dimension();
   ParMesh mesh(MPI_COMM_WORLD, smesh);
   L2_FECollection sfec(p, dim, BasisType::GaussLobatto);
   L2_FECollection vfec(p, dim, BasisType::GaussLobatto);

   ParFiniteElementSpace sfes(&mesh, &sfec);
   ParFiniteElementSpace vfes(&mesh, &vfec, dim);

   Vector b(sfes.GetVSize());
   ParLinearForm fform(&sfes);
   FunctionCoefficient f(SourceFunction);
   fform.AddDomainIntegrator(new DomainLFIntegrator(f));
   fform.Assemble();
   fform.ParallelAssemble(b);

   ParBilinearForm Miform(&vfes);
   Miform.AddDomainIntegrator(new InverseIntegrator(new VectorMassIntegrator));
   Miform.Assemble();
   Miform.Finalize();
   HypreParMatrix *Mi = Miform.ParallelAssemble();

   Vector beta(dim);
   beta(0) = 1.;
   beta(1) = 2.;
   ParMixedBilinearForm Dform(&vfes, &sfes);
   ConstantCoefficient none(-1.);
   Dform.AddDomainIntegrator(new TransposeIntegrator(new GradientIntegrator(
                                                        none)));
   Dform.AddInteriorFaceIntegrator(new LDGTraceIntegrator(use_beta ? &beta :
                                                          nullptr));
   Dform.AddBdrFaceIntegrator(new LDGTraceIntegrator);
   Dform.Assemble();
   Dform.Finalize();
   HypreParMatrix *D = Dform.ParallelAssemble();
   HypreParMatrix *DT = D->Transpose();

   ParBilinearForm Pform(&sfes);
   ConstantCoefficient abs(.1);
   Pform.AddDomainIntegrator(new MassIntegrator(abs));
   Pform.AddInteriorFaceIntegrator(new DGJumpJumpIntegrator(kappa));
   Pform.AddBdrFaceIntegrator(new DGJumpJumpIntegrator(pow(p+1,2), false));
   Pform.Assemble();
   Pform.Finalize();
   HypreParMatrix *P = Pform.ParallelAssemble();

   HypreParMatrix *DMi = ParMult(D, Mi, true);
   HypreParMatrix *DMiDt = ParMult(DMi, DT, true);
   HypreParMatrix *S = ParAdd(DMiDt, P);

   HypreBoomerAMG amg;
   amg.SetPrintLevel(0);

   BiCGSTABSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(1e-6);
   cg.SetMaxIter(100);
   cg.SetPreconditioner(amg);
   cg.SetOperator(*S);

   ParGridFunction T(&sfes);
   cg.Mult(b, T);

   FunctionCoefficient Tex(ExactFunction);
   double err = T.ComputeL2Error(Tex);
   if (rank==0) { printf("err = %.3e\n", err); }
   if (rank==0) { printf("cg iter = %d, final norm = %.3e\n", cg.GetNumIterations(), cg.GetFinalNorm()); }

   ParaViewDataCollection dc("solution", &mesh);
   dc.RegisterField("T", &T);
   dc.Save();

   return 0;
}
