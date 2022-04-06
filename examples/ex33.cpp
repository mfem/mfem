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
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   Mesh mesh(Ne, Ne, Element::QUADRILATERAL);
   int dim = mesh.Dimension();
   L2_FECollection sfec(p, dim, BasisType::GaussLobatto);
   L2_FECollection vfec(p, dim, BasisType::GaussLobatto);

   FiniteElementSpace sfes(&mesh, &sfec);
   FiniteElementSpace vfes(&mesh, &vfec, dim);

   LinearForm fform(&sfes);
   FunctionCoefficient f(SourceFunction);
   fform.AddDomainIntegrator(new DomainLFIntegrator(f));
   fform.Assemble();

   BilinearForm Miform(&vfes);
   Miform.AddDomainIntegrator(new InverseIntegrator(new VectorMassIntegrator));
   Miform.Assemble();
   Miform.Finalize();
   SparseMatrix Mi(Miform.SpMat());

   Vector beta(dim);
   beta(0) = 1.;
   beta(1) = 2.;
   MixedBilinearForm Dform(&vfes, &sfes);
   ConstantCoefficient none(-1.);
   Dform.AddDomainIntegrator(new TransposeIntegrator(new GradientIntegrator(
                                                        none)));
   Dform.AddInteriorFaceIntegrator(new LDGTraceIntegrator(use_beta ? &beta :
                                                          nullptr));
   Dform.AddBdrFaceIntegrator(new LDGTraceIntegrator);
   Dform.Assemble();
   Dform.Finalize();
   SparseMatrix D(Dform.SpMat());

   BilinearForm Pform(&sfes);
   ConstantCoefficient abs(.1);
   Pform.AddDomainIntegrator(new MassIntegrator(abs));
   Pform.AddInteriorFaceIntegrator(new DGJumpJumpIntegrator(kappa));
   Pform.AddBdrFaceIntegrator(new DGJumpJumpIntegrator(pow(p+1,2)));
   Pform.Assemble();
   Pform.Finalize();
   SparseMatrix P(Pform.SpMat());

   SparseMatrix *tmp = RAP(Mi, D);
   SparseMatrix *S = Add(*tmp, P);

   KLUSolver klu(*S);
   GridFunction T(&sfes);
   klu.Mult(fform, T);

   FunctionCoefficient Tex(ExactFunction);
   double err = T.ComputeL2Error(Tex);
   printf("err = %.3e\n", err);

   ParaViewDataCollection dc("solution", &mesh);
   dc.RegisterField("T", &T);
   dc.Save();
}
