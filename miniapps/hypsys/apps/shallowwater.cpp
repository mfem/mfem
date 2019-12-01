#include "shallowwater.hpp"

int ConfigNum;
Vector bbMin, bbMax;

double InitialCondition(const Vector &x);
double Inflow(const Vector &x);

ShallowWater::ShallowWater(const Vector &bbmin, const Vector &bbmax,
                           const int config, const double tEnd,
                           const int _dim, FiniteElementSpace *_fes)
   : tFinal(tEnd), dim(_dim), NumEq(_dim+1), fes(_fes)
{
   ConfigNum = config;
   bbMin = bbmin;
   bbMax = bbmax;

   vol = new VolumeTerms(fes, dim);
}

void VolumeTerms::EvalFluxFunction(const Vector &u, DenseMatrix &FluxEval)
{
   FluxEval.SetSize(NumEq, fes->GetMesh()->Dimension());
   double H0 = 0.05;

   if (u.Size() != NumEq) { MFEM_ABORT("Invalid solution vector."); }
   if (u(0) < H0) { MFEM_ABORT("Water height too small."); }

   switch (dim)
   {
      case 1:
      {
         FluxEval(0,0) = u(1);
         FluxEval(1,0) = u(1)*u(1)/u(0) + 9.81 / 2. * u(0)*u(0);
      }
      case 2:
      {
         FluxEval(0,0) = u(1);
         FluxEval(0,1) = u(2);
         FluxEval(1,0) = u(1)*u(1)/u(0) + 9.81 / 2. * u(0)*u(0);
         FluxEval(1,1) = u(1)*u(2)/u(0);
         FluxEval(2,0) = u(2)*u(1)/u(0);
         FluxEval(2,1) = u(2)*u(2)/u(0) + 9.81 / 2. * u(0)*u(0);
      }
      default: MFEM_ABORT("Invalid space dimensions.");
   }
}

void ShallowWater::PreprocessProblem(FiniteElementSpace *fes, GridFunction &u)
{
   Mesh *mesh = fes->GetMesh();

   // Model parameters.
   FunctionCoefficient u0(InitialCondition);
   FunctionCoefficient inflow(Inflow);



   // Initialize solution vector.
   u.ProjectCoefficient(u0);
}

void ShallowWater::PostprocessProblem(const GridFunction &u,
                                      Array<double> &errors)
{
   if (SolutionKnown)
   {
      switch (ConfigNum)
      {
         case 0:
         {
            FunctionCoefficient uAnalytic(Inflow);
            errors[0] = u.ComputeLpError(1., uAnalytic);
            errors[1] = u.ComputeLpError(2., uAnalytic);
            errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
            break;
         }
         case 1:
         {
            FunctionCoefficient uAnalytic(InitialCondition);
            errors[0] = u.ComputeLpError(1., uAnalytic);
            errors[1] = u.ComputeLpError(2., uAnalytic);
            errors[2] = u.ComputeLpError(numeric_limits<double>::infinity(), uAnalytic);
            break;
         }
         default: MFEM_ABORT("No such test case implemented.");
      }
   }
}

void VolumeTerms::AssembleElementVolumeTerms(const int e,
                                             const DenseMatrix &uEl, DenseMatrix &VolTerms)
{
   cout << "H" << endl;
   //    const FiniteElement *el = fes->GetFE(e);
   //    const int nd = el->GetDof();
   //    const int nq = ir->GetNPoints();
   //    int i, k;

   //    VolTerms.SetSize(nd, NumEq); VolTerms = 0.;
   //    DenseMatrix uQ(nq, NumEq), FluxEval, tmp4(nd, NumEq);
   //    Vector uQuad(nq), tmp(NumEq), tmp2(dim), tmp3(nd);
   //
   //    ElementTransformation *trans = fes->GetElementTransformation(e);
   //    DenseMatrix adjJ = trans->AdjugateJacobian();
   //
   //    for (i = 0; i < NumEq; i++)
   //    {
   //       shape.Mult(uEl.GetColumn(i), uQuad);
   //       uQ.SetCol(i, uQuad);
   //    }
   //
   //    for (k = 0; k < nq; k++)
   //    {
   //       const IntegrationPoint &ip = ir->IntPoint(k);
   //       uQ.GetRow(k, tmp);
   //       EvalFluxFunction(tmp, FluxEval);
   //       for (i = 0; i < NumEq; i++)
   //       {
   //          FluxEval.GetRow(i, tmp);
   //          adjJ.Mult(tmp, tmp2);
   //          dShape(k).Mult(tmp2, tmp3);
   //          tmp4.SetCol(i, tmp3);
   //       }
   //       VolTerms += tmp4;
   //    }
}

double InitialCondition(const Vector &x)
{
   // Map to the reference [-1,1] domain.
   int dim = x.Size();
   Vector X(dim); X = x;
   for (int i = 0; i < dim; i++)
   {
      double center = (bbMin(i) + bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (bbMax(i) - bbMin(i));
   }

   switch (ConfigNum)
   {
      case 0: // Smooth solution used for grid convergence studies.
      {
         Vector Y(dim); Y = 1.;
         X.Add(1., Y);
         X *= 0.5;
         double r = X.Norml2();
         double a = 0.5, b = 0.03, c = 0.1;
         return 0.25 * (1. + tanh((r+c-a)/b)) * (1. - tanh((r-c-a)/b));
      }
      case 1: // Solid body rotation.
      {
         double scale = 0.0225;
         double coef = (0.5/sqrt(scale));
         double slit = (X(0) <= -0.05) || (X(0) >= 0.05) || (X(1) >= 0.7);
         double cone = coef * sqrt(pow(X(0), 2.) + pow(X(1) + 0.5, 2.));
         double hump = coef * sqrt(pow(X(0) + 0.5, 2.) + pow(X(1), 2.));

         return (slit && ((pow(X(0),2.) + pow(X(1)-.5,2.))<=4.*scale)) ? 1. : 0.
                + (1. - cone) * (pow(X(0), 2.) + pow(X(1)+.5, 2.) <= 4.*scale)
                + .25 * (1. + cos(M_PI*hump))
                * ((pow(X(0)+.5, 2.) + pow(X(1), 2.)) <= 4.*scale);
      }
      default: { MFEM_ABORT("No such test case implemented."); }
   }
}

double Inflow(const Vector &x)
{
   // Map to the reference [-1,1] domain.
   int dim = x.Size();
   Vector X(dim); X = x;
   for (int i = 0; i < dim; i++)
   {
      double center = (bbMin(i) + bbMax(i)) * 0.5;
      X(i) = 2. * (x(i) - center) / (bbMax(i) - bbMin(i));
   }

   switch (ConfigNum)
   {
      case 0:
      {
         double r = x.Norml2();
         double a = 0.5, b = 0.03, c = 0.1;
         return 0.25 * (1. + tanh((r+c-a)/b)) * (1. - tanh((r-c-a)/b));
      }
      default: { return 0.; }
   }
}
