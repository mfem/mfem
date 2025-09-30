/// Example 0: AD Function Example
#include "mfem.hpp"
#include "ad_native.hpp"

using namespace std;
using namespace mfem;

void Print(DenseMatrix &H, bool last_linebreak=true);
void Print(DenseTensor &H);

struct MyADFunction : public ADFunction
{
public:
   MyADFunction(int n_input): ADFunction(n_input) { }

   AD_IMPL(T, V, M, x, return sin(x(0))*exp(x(1)) + pow(x(2), 3.0);)
};

struct MyADVecFunction : public ADVectorFunction
{
public:
   MyADVecFunction()
      : ADVectorFunction(3, 2) { }

   AD_VEC_IMPL(T, V, M, x, result,
   {
      result[0]=sin(x[0]*x[1]);
      result[1]=cos(x[0]*x[1]*x[2]);
   });
};

void jacobian(const Vector &x, Vector &J)
{
   J.SetSize(x.Size());
   J[0] = std::cos(x(0)) * std::exp(x(1));
   J[1] = std::sin(x(0)) * std::exp(x(1));
   J[2] = 3.0 * std::pow(x(2), 2.0);
}

void hessian(const Vector &x, DenseMatrix &H)
{
   // J[0] = cos(x(0)) * exp(x(1));
   H.SetSize(x.Size(), x.Size());
   H(0, 0) = -std::sin(x(0)) * std::exp(x(1));
   H(0, 1) = std::cos(x(0)) *std::exp(x(1));
   H(0, 2) = 0.0;

   // J[1] = sin(x(0)) * exp(x(1));
   H(1, 0) = std::cos(x(0)) * std::exp(x(1));
   H(1, 1) = std::sin(x(0)) * std::exp(x(1));
   H(1, 2) = 0.0;

   // J[2] = 3.0 * pow(x(2), 2.0);
   H(2, 0) = 0.0;
   H(2, 1) = 0.0;
   H(2, 2) = 6.0 * std::pow(x(2), 1.0);
}

void jacobian(const Vector &x, DenseMatrix &J)
{
   // result[0]=sin(x[0]*x[1]);
   // result[1]=cos(x[0]*x[1]*x[2]);)
   J.SetSize(2,3);
   J(0,0) = x(1) * std::cos(x(0) * x(1));
   J(0,1) = x(0) * std::cos(x(0) * x(1));
   J(0,2) = 0.0;
   J(1,0) = -x(1) * x(2) * std::sin(x(0) * x(1) * x(2));
   J(1,1) = -x(0) * x(2) * std::sin(x(0) * x(1) * x(2));
   J(1,2) = -x(0) * x(1) * std::sin(x(0) * x(1) * x(2));
}
void hessian(const Vector &X, DenseTensor &H)
{
   real_t x(X(0)), y(X(1)), z(X(2));
   H.SetSize(3, 3, 2);
   H = 0.0;
   using std::sin;
   using std::cos;

   // result[0]=sin(x[0]*x[1]);
   H(0,0,0) = -y*y*sin(x*y);
   H(0,1,0) = cos(x*y) - x*y*sin(x*y);
   H(1,0,0) = cos(x*y) - x*y*sin(x*y);
   H(1,1,0) = -x*x*sin(x*y);
   // result[1]=cos(x[0]*x[1]*x[2]);)
   H(0,0,1) = -y*y*z*z*cos(x*y*z);
   H(1,0,1) = -x*y*z*z*cos(x*y*z) - z*sin(x*y*z);
   H(2,0,1) = -x*y*y*z*cos(x*y*z) - y*sin(x*y*z);
   H(0,1,1) = -x*y*z*z*cos(x*y*z) - z*sin(x*y*z);
   H(1,1,1) = -x*x*z*z*cos(x*y*z);
   H(2,1,1) = -x*x*y*z*cos(x*y*z) - x*sin(x*y*z);
   H(0,2,1) = -x*y*y*z*cos(x*y*z) - y*sin(x*y*z);
   H(1,2,1) = -x*x*y*z*cos(x*y*z) - x*sin(x*y*z);
   H(2,2,1) = -x*x*y*y*cos(x*y*z);
}

int main(int argc, char *argv[])
{
   Vector x({0.5, 1.0, -1.0});

   MyADFunction f(3);

   Vector jac, jac_ref;
   f.Gradient(x, jac);
   jacobian(x, jac_ref);

   DenseMatrix hess, hess_ref;
   f.Hessian(x, hess);
   hessian(x, hess_ref);

   MyADVecFunction f2;

   DenseMatrix jac2, jac2_ref;
   f2.Gradient(x, jac2);
   jacobian(x, jac2_ref);

   DenseTensor hess2, hess2_ref;
   f2.Hessian(x, hess2);
   hessian(x, hess2_ref);


   out << "Value : " << f(x) << std::endl;

   out << "Jacobian  : ";
   jac.Print();
   out << "Reference : ";
   jac_ref.Print();
   jac -= jac_ref;

   out << "Hessian : " << std::endl;
   Print(hess);
   out << "Reference: " << std::endl;
   Print(hess_ref);
   hess -= hess_ref;

   out << std::endl;
   out << "Jacobian error: " << jac.DistanceTo(jac_ref) << std::endl;
   out << "Hessian error: " << hess.MaxMaxNorm() << std::endl;
   out << "-------------------------" << std::endl;

   out << "Jacobian2 : " << std::endl;
   Print(jac2);
   out << "Reference : " << std::endl;
   Print(jac2_ref);
   jac2 -= jac2_ref;

   out << "Hess2 : " << std::endl;
   Print(hess2);
   out << "Reference : " << std::endl;
   Print(hess2_ref);
   out << std::endl;
   out << "Jacobian2 error: " << jac2.MaxMaxNorm() << std::endl;
   for (int k=0; k<hess2.SizeK(); k++)
   {
      hess2(k) -= hess2_ref(k);
      out << "Hessian[" << k << "] error: " << hess2(k).MaxMaxNorm() << std::endl;
   }

   return 0;
}

void Print(DenseMatrix &H, bool last_linebreak)
{
   for (int i = 0; i < H.Height(); i++)
   {
      for (int j = 0; j < H.Width(); j++)
      {
         cout << H(i, j) << " ";
      }
      cout << ";";
      if (i < H.Height() - 1 || last_linebreak)
      {
         cout << "\n";
      }
   }
   cout << std::flush;
}

void Print(DenseTensor &H)
{
   for (int k=0; k<H.SizeK(); k++)
   {
      out << "{ ";
      Print(H(k), false);
      out << " }\n";
   }
   cout << std::flush;
}
