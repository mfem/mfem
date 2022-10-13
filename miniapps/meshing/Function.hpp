#ifndef FUNCTION
#define FUNCTION

#include <iostream>
#include <vector>
#include <cmath>
#include <assert.h>

namespace ImplicitGeometry
{

class Function   
{
public:
   Function(int n);
   virtual ~Function();

   virtual double          Eval()=0;
   std::vector<double>     Grad();
   int                     GetNumParams()           {return num_exp_;}
   Function*               GetExplicitParam(int i)  {return exp_params_[i];} 
   Function*               GetImplicitParam(int i)  {return imp_params_[i];} 
protected:
   void                    SetExplicitParam(int i, Function *p); 
   void                    Finalize();
   std::vector<double>     exp_grad_;
   double PI{3.14159265359};
private:
   virtual void            ComputeExplicitGrad()=0;
   int                     GradientSize()           {return gradSize_;}
   std::vector<Function*>  exp_params_;
   std::vector<Function*>  imp_params_;
   std::vector<double>     tot_grad_;
   int                     num_exp_;
   int                     gradSize_;
};

class Scalar : public Function
{
public:
   Scalar(double a);
   ~Scalar() {};
   inline void SetValue(double a) {a_=a;}
   inline double Eval() {return a_;}
private:
   inline void ComputeExplicitGrad() {exp_grad_[0]=1.0;}
   double a_;
};

class Addition : public Function
{
public:
   Addition(Function *a, Function *b);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *a_, *b_;
};

class Subtraction : public Function
{
public:
   Subtraction(Function *a, Function *b);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *a_, *b_;
};

class Multiplication : public Function
{
public:
   Multiplication(Function *a, Function *b);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *a_, *b_;
};

class Division : public Function
{
public:
   Division(Function *a, Function *b);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *a_, *b_;
};

class Power : public Function
{
public:
   Power(Function *a, double p);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *a_;
   double p_;
};

class Polynomial : public Function
{
public:
   Polynomial(Function *x, std::vector<double> p);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *x_;
   std::vector<double> p_;
};

class SineWave : public Function
{
public:
   SineWave(Function *x, Function *L, Function *phi, Function *l);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *x_, *L_, *phi_, *l_;
};

class StairCase : public Function
{
public:
   StairCase(Function *x, Function *a, Function *b, int N=20);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *x_, *a_, *b_;
   int N_;
};

class TransformedCoordinate : public Function
{
public:
   TransformedCoordinate(Function *x, Function *y, Function *z);
   virtual double Eval()=0;
protected:
   Function *x_, *y_, *z_;
private:
   virtual void ComputeExplicitGrad()=0;
};

class ProjectedCoordinate : public TransformedCoordinate
{
public:
   ProjectedCoordinate(Function *x, Function *y, Function *z, double i, double j, double k);
   double Eval();
private:
   void ComputeExplicitGrad();
   double a0_, a1_, a2_;
};

class CylindricalRadius : public TransformedCoordinate
{
public:
   CylindricalRadius(Function *x, Function *y, Function *z);
   double Eval();
private:
   void ComputeExplicitGrad();
};

class CylindricalTheta : public TransformedCoordinate
{
public:
   CylindricalTheta(Function *x, Function *y, Function *z);
   double Eval();
private:
   void ComputeExplicitGrad();
};

class SphericalRadius : public TransformedCoordinate
{
public:
   SphericalRadius(Function *x, Function *y, Function *z);
   double Eval();
private:
   void ComputeExplicitGrad();
};

class SphericalTheta : public TransformedCoordinate
{
public:
   SphericalTheta(Function *x, Function *y, Function *z);
   double Eval();
private:
   void ComputeExplicitGrad();
};

class SphericalPhi : public TransformedCoordinate
{
public:
   SphericalPhi(Function *x, Function *y, Function *z);
   double Eval();
private:
   void ComputeExplicitGrad();
};

// generating function IS a signed distance function!
class Sphere : public Function
{
public:
   Sphere(Function *x, Function *y, Function *z, Function *x0, Function *y0, Function *z0, Function *r);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *x_, *y_, *z_, *x0_, *y0_, *z0_, *r_;
};

// generating function IS a signed distance function!
class CappedCylindricalRod : public Function
{
public:
   CappedCylindricalRod(Function *x,  Function *y,  Function *z, 
                        Function *x0, Function *y0, Function *z0,
                        Function *x1, Function *y1, Function *z1,
                        Function *r);
   double Eval();
private:
   double InnerProduct(const double (&a)[3], const double (&b)[3]);
   double Norml2(const double (&a)[3]);
   void MatVec(const double (&A)[3][3], const double (&b)[3], double (&c)[3]);
   void MatMat(const double (&A)[3][3], const double (&B)[3][3], double (&C)[3][3]);
   void MatMatTrans(const double (&A)[3][3], const double (&B)[3][3], double (&C)[3][3]);
   void MatAdd(double (&A)[3][3], const double (&B)[3][3]);
   void MatScale(double (&A)[3][3], double c);

   void EvaluateVectors();
   void BuildOrthogonalProjector();
   void ComputeDg(const double (&Da)[3], 
                  const double (&Db)[3],
                        double (&Dg)[3]);
   
   void ComputeExplicitGrad();
   Function *x_, *y_, *z_, *x0_, *y0_, *z0_, *x1_, *y1_, *z1_, *r_;
   double X_[3], X0_[3], X1_[3], a_[3], b_[3], e_[3], g_[3];
   double P_[3][3];
};

class SphericallySampledRodVolumeFraction : public Function
{
public:
   SphericallySampledRodVolumeFraction(Function *phi, Function *r);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *phi_, *r_;
};

class SmoothMax : public Function
{
public:
   SmoothMax(std::vector<Function*> x, double alpha);
   double Eval();
private:
   void ComputeExplicitGrad();
   std::vector<Function*> x_;
   int n_;
   double alpha_;
   double numer_, denom_;
};

class PNorm : public Function
{
public:
   PNorm(std::vector<Function*> x, double p);
   double Eval();
private:
   void ComputeExplicitGrad();
   std::vector<Function*> x_;
   int n_;
   double p_;
};

class RpFunction : public Function
{
public:
   RpFunction(Function *f1, Function *f2, Function *a0, Function *a1, Function *a2, double a, double b, double c, double p);
   RpFunction(Function *f1, Function *f2, double a, double b, double c, double p);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *f1_, *f2_, *a0_, *a1_, *a2_;
   double    a_, b_, c_, p_;
   double    eps_;
   bool      blend_;
};

class ROr : public RpFunction
{
public:
   ROr(Function *f1, Function *f2, Function *a0, Function *a1, Function *a2, double p=2.0)
     : RpFunction(f1, f2, a0, a1, a2, 1., 1., 1., p) {}
   ROr(Function *f1, Function *f2, double p=2.0)
     : RpFunction(f1, f2, 1., 1., 1., p) {}
};

class RAnd : public RpFunction
{
public:
   RAnd(Function *f1, Function *f2, Function *a0, Function *a1, Function *a2, double p=2.0)
     : RpFunction(f1, f2, a0, a1, a2, 1., 1., -1., p) {}
   RAnd(Function *f1, Function *f2, double p=2.0)
     : RpFunction(f1, f2, 1., 1., -1., p) {}
};

class RMinus : public RpFunction
{
public:
   RMinus(Function *f1, Function *f2, Function *a0, Function *a1, Function *a2, double p=2.0)
     : RpFunction(f1, f2, a0, a1, a2, 1., -1., -1., p) {}
   RMinus(Function *f1, Function *f2, double p=2.0)
     : RpFunction(f1, f2, 1., -1., -1., p) {}
};

class Heaviside : public Function
{ 
public:
   Heaviside(Function *x, double alpha);
   double Eval();
private:
   void ComputeExplicitGrad();
   Function *x_;
   double alpha_; 
};

Function::Function(int n)
{
   num_exp_ = n;
   exp_params_.resize(num_exp_);
   for (int i=0; i<num_exp_; i++) 
      {exp_params_[i] = NULL;}

   exp_grad_.resize(n);
}
 
Function::~Function()
{
   //for (int i=0; i<num_exp_; i++)
   //   {delete exp_params_[i]; } //exp_params_[i] = NULL; }
}

void Function::SetExplicitParam(int i, Function *p)
{
   assert(p != NULL);
   assert(0 <= i && i < num_exp_);
   assert(exp_params_[i] == NULL);
   exp_params_[i] = p;
}

void Function::Finalize()
{
   imp_params_.clear();
   for (int i=0; i<num_exp_; i++)
   {
      Function *param = exp_params_[i];
      if (param == NULL)
         {imp_params_.push_back(this);}
      else
      {
         for (int j=0; j<param->GradientSize(); j++)
            {imp_params_.push_back(param->GetImplicitParam(j));}
      }
   }
   gradSize_ = imp_params_.size();
   tot_grad_.resize(gradSize_);
}

// -> function computes gradient via composition and performs chain rule automatically
std::vector<double> Function::Grad()
{
   this->ComputeExplicitGrad();
   int k=0;
   for (int i=0; i<num_exp_; i++)
   {
      if (exp_params_[i] == NULL) // Grab the explicit derivative of scalars
         {tot_grad_[k++] = exp_grad_[i];}
      else                        // Perform chain rule
      {
         std::vector<double> paramGrad = exp_params_[i]->Grad();
         for (int j=0; j<paramGrad.size(); j++)
            {tot_grad_[k++] = exp_grad_[i]*paramGrad[j];}
      }
   }
   return tot_grad_;
}

Scalar::Scalar(double a)
  : Function(1), a_(a)
{
   this->Finalize();
}

Addition::Addition(Function *a, Function *b)
  : Function(2), a_(a), b_(b)
{
   this->SetExplicitParam(0, a);
   this->SetExplicitParam(1, b);
   this->Finalize();
}

double Addition::Eval()
{
   return a_->Eval() + b_->Eval();   
}

void Addition::ComputeExplicitGrad()
{
   exp_grad_[0] = 1.0;
   exp_grad_[1] = 1.0;
}

Subtraction::Subtraction(Function *a, Function *b)
  : Function(2), a_(a), b_(b)
{
   this->SetExplicitParam(0, a);
   this->SetExplicitParam(1, b);
   this->Finalize();
}

double Subtraction::Eval()
{
   return a_->Eval() - b_->Eval();   
}

void Subtraction::ComputeExplicitGrad()
{
   exp_grad_[0] =  1.0;
   exp_grad_[1] = -1.0;
}

Multiplication::Multiplication(Function *a, Function *b)
  : Function(2), a_(a), b_(b)
{
   this->SetExplicitParam(0, a);
   this->SetExplicitParam(1, b);
   this->Finalize();
}

double Multiplication::Eval()
{
   return a_->Eval() * b_->Eval();   
}

void Multiplication::ComputeExplicitGrad()
{
   exp_grad_[0] = b_->Eval();
   exp_grad_[1] = a_->Eval();
}

Division::Division(Function *a, Function *b)
  : Function(2), a_(a), b_(b)
{
   this->SetExplicitParam(0, a);
   this->SetExplicitParam(1, b);
   this->Finalize();
}

double Division::Eval()
{
   return a_->Eval() / b_->Eval();   
}

void Division::ComputeExplicitGrad()
{
   double b = b_->Eval();
   exp_grad_[0] = 1. / b;
   exp_grad_[1] = -a_->Eval() / (b*b);
}

Power::Power(Function *a, double p)
  : Function(1), a_(a), p_(p)
{
   this->SetExplicitParam(0, a);
   this->Finalize();
}

double Power::Eval()
{
   return pow(a_->Eval(), p_);   
}

void Power::ComputeExplicitGrad()
{
   exp_grad_[0] = p_ * pow(a_->Eval(), p_-1.);
}

Polynomial::Polynomial(Function *x, std::vector<double> p)
  : Function(1), x_(x), p_(p)
{
   this->SetExplicitParam(0, x);
   this->Finalize();
}

double Polynomial::Eval()
{
   double     x = x_->Eval();
   double value = 0.0;
   double   x_p = 1.0;
   for (int i=0; i<p_.size(); i++)
   {
      value += p_[i]*x_p;
      x_p   *= x;
   }
   return value;
}

void Polynomial::ComputeExplicitGrad()
{   
   double    x = x_->Eval();
   double grad = 0.0;
   double  x_p = 1.0;
   for (int i=1; i<p_.size(); i++)
   {
      grad += i*p_[i]*x_p;
      x_p  *= x;
   }
   exp_grad_[0] = grad;
}


SineWave::SineWave(Function *x, Function *L, Function *phi, Function *l)
  : Function(4), x_(x), L_(L), phi_(phi), l_(l)
{
   this->SetExplicitParam(0, x);
   this->SetExplicitParam(1, L);
   this->SetExplicitParam(2, phi);
   this->SetExplicitParam(3, l);
   this->Finalize();
}

double SineWave::Eval()
{
   return sin(2.*PI*x_->Eval()/L_->Eval() + phi_->Eval()) + l_->Eval();
}

void SineWave::ComputeExplicitGrad()
{
   double   x = x_->Eval();
   double   L = L_->Eval();
   double phi = phi_->Eval();
   double   l = l_->Eval();
   exp_grad_[0] = cos(2*PI*x/L + phi)*( 2*PI  / L );   // d_dx
   exp_grad_[1] = cos(2*PI*x/L + phi)*(-2*PI*x/(L*L)); // d_dL
   exp_grad_[2] = cos(2*PI*x/L + phi);                 // d_dphi
   exp_grad_[3] = 1;                                   // d_dl
}

StairCase::StairCase(Function *x, Function *a, Function *b, int N)
  : Function(3), x_(x), a_(a), b_(b), N_(N)
{
   this->SetExplicitParam(0, x);
   this->SetExplicitParam(1, a);
   this->SetExplicitParam(2, b);
   this->Finalize();
}

double StairCase::Eval()
{
   double     x = x_->Eval();
   double     a = a_->Eval();
   double     b = b_->Eval();
   double value = x*b/a;
   for (int n=1; n<=N_; n++)
   {
      value += pow(-1, n) * b * sin(2*n*PI*x/a) / (n*PI);
   }
   return value;
}

void StairCase::ComputeExplicitGrad()
{
   double     x = x_->Eval();
   double     a = a_->Eval();
   double     b = b_->Eval();
   double gradx = b/a;
   double grada = -1*x*b/(a*a);
   double gradb = x/a;
   for (int n=1; n<=N_; n++)
   {
      gradx += pow(-1, n) *   2*b/a        * cos(2*n*PI*x/a);
      grada += pow(-1, n) * (-2*x*b/(a*a)) * cos(2*n*PI*x/a);
      gradb += pow(-1, n) * sin(2*n*PI*x/a) / (n*PI);
   }
   exp_grad_[0] = gradx;
   exp_grad_[1] = grada;
   exp_grad_[2] = gradb;
}

TransformedCoordinate::TransformedCoordinate(Function *x, Function *y, Function *z)
  : Function(3), x_(x), y_(y), z_(z)
{
   this->SetExplicitParam(0, x); 
   this->SetExplicitParam(1, y); 
   this->SetExplicitParam(2, z); 
   this->Finalize();
}


ProjectedCoordinate::ProjectedCoordinate(Function *x, Function *y, Function *z, double i, double j, double k)
  : TransformedCoordinate(x, y, z)
{
   double mag = sqrt(i*i + j*j + k*k);
   a0_ = i/mag;
   a1_ = j/mag;
   a2_ = k/mag;
}

double ProjectedCoordinate::Eval()
{
   return x_->Eval()*a0_ + y_->Eval()*a1_ + z_->Eval()*a2_;
}

void ProjectedCoordinate::ComputeExplicitGrad()
{
   exp_grad_[0] = a0_;
   exp_grad_[1] = a1_;
   exp_grad_[2] = a2_;
}

CylindricalRadius::CylindricalRadius(Function *x, Function *y, Function *z)
  : TransformedCoordinate(x, y, z)
{}

double CylindricalRadius::Eval()
{
   double x = x_->Eval();
   double y = y_->Eval();
   return sqrt(x*x + y*y);  
}

void CylindricalRadius::ComputeExplicitGrad()
{
   double x = x_->Eval();
   double y = y_->Eval();
   double r = sqrt(x*x + y*y);
   exp_grad_[0] = x / r;  // d_dx
   exp_grad_[1] = y / r;  // d_dy
   exp_grad_[2] = 0.0;    // d_dz
}

CylindricalTheta::CylindricalTheta(Function *x, Function *y, Function *z)
  : TransformedCoordinate(x, y, z)
{}

double CylindricalTheta::Eval()
{
   double x = x_->Eval();
   double y = y_->Eval();
   return atan2(y, x);
}

void CylindricalTheta::ComputeExplicitGrad()
{
   double  x = x_->Eval();
   double  y = y_->Eval();
   double r2 = x*x + y*y;   
   exp_grad_[0] = -y/r2;  // d_dx
   exp_grad_[1] =  x/r2;  // d_dy
   exp_grad_[2] =  0.0;   // d_dz
}

SphericalRadius::SphericalRadius(Function *x, Function *y, Function *z)
  : TransformedCoordinate(x, y, z)
{}

double SphericalRadius::Eval()
{
   double x = x_->Eval();
   double y = y_->Eval();
   double z = z_->Eval();
   return sqrt(x*x + y*y + z*z);
}

void SphericalRadius::ComputeExplicitGrad()
{
   double x = x_->Eval();
   double y = y_->Eval();
   double z = z_->Eval();
   double r = sqrt(x*x + y*y + z*z);   
   exp_grad_[0] = x/r;  // d_dx
   exp_grad_[1] = y/r;  // d_dy
   exp_grad_[2] = z/r;  // d_dz
}

SphericalTheta::SphericalTheta(Function *x, Function *y, Function *z)
  : TransformedCoordinate(x, y, z)
{}

double SphericalTheta::Eval()
{
   double x = x_->Eval();
   double y = y_->Eval();
   double z = z_->Eval();
   return acos(z / sqrt( x*x + y*y + z*z));
}

void SphericalTheta::ComputeExplicitGrad()
{
   double     x = x_->Eval();
   double     y = y_->Eval();
   double     z = z_->Eval();
   double    r2 = x*x + y*y + z*z;
   double     r = sqrt(r2);   
   double dacos = -1 / sqrt(1-(z*z)/(r*r)); 
   exp_grad_[0] = dacos *         z*(-x)*pow(r2, -1.5);   // d_dx
   exp_grad_[1] = dacos *         z*(-y)*pow(r2, -1.5);   // d_dy
   exp_grad_[2] = dacos * ( 1/r + z*(-z)*pow(r2, -1.5));  // d_dz
}

SphericalPhi::SphericalPhi(Function *x, Function *y, Function *z)
  : TransformedCoordinate(x, y, z)
{}

double SphericalPhi::Eval()
{
   double x = x_->Eval();
   double y = y_->Eval();
   return atan2(y, x);  
}

void SphericalPhi::ComputeExplicitGrad()
{
   double     x = x_->Eval();
   double     y = y_->Eval();
   exp_grad_[0] = -y/(x*x + y*y); // d_dx
   exp_grad_[1] =  x/(x*x + y*y); // d_dy
   exp_grad_[2] = 0.0;            // d_dz
}

Sphere::Sphere(Function *x, Function *y, Function *z, Function *x0, Function *y0, Function *z0, Function *r)
  : Function(7), x_(x), y_(y), z_(z), x0_(x0), y0_(y0), z0_(z0), r_(r)
{
   this->SetExplicitParam(0, x);
   this->SetExplicitParam(1, y);
   this->SetExplicitParam(2, z);
   this->SetExplicitParam(3, x0);
   this->SetExplicitParam(4, y0);
   this->SetExplicitParam(5, z0);
   this->SetExplicitParam(6, r);
   this->Finalize();
}

double Sphere::Eval()
{
   double  x =  x_->Eval();
   double  y =  y_->Eval();
   double  z =  z_->Eval();
   double x0 = x0_->Eval();
   double y0 = y0_->Eval();
   double z0 = z0_->Eval();
   double  r =  r_->Eval();
   return r - sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0));
}

void Sphere::ComputeExplicitGrad()
{
   double  x =  x_->Eval();
   double  y =  y_->Eval();
   double  z =  z_->Eval();
   double x0 = x0_->Eval();
   double y0 = y0_->Eval();
   double z0 = z0_->Eval();
   double  r =  r_->Eval();
   double scale = -1/sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0));
   exp_grad_[0] =  scale*(x-x0); 
   exp_grad_[1] =  scale*(y-y0);
   exp_grad_[2] =  scale*(z-z0);
   exp_grad_[3] = -scale*(x-x0);
   exp_grad_[4] = -scale*(y-y0);      
   exp_grad_[5] = -scale*(z-z0);
   exp_grad_[6] =  1;
}

CappedCylindricalRod::CappedCylindricalRod(Function *x,  Function *y,  Function *z, 
                                           Function *x0, Function *y0, Function *z0,
                                           Function *x1, Function *y1, Function *z1,
                                           Function *r)
  : Function(10), x_(x),   y_(y),   z_(z), 
                  x0_(x0), y0_(y0), z0_(z0), 
                  x1_(x1), y1_(y1), z1_(z1), r_(r)
{
   this->SetExplicitParam(0, x);
   this->SetExplicitParam(1, y);
   this->SetExplicitParam(2, z);
   this->SetExplicitParam(3, x0);
   this->SetExplicitParam(4, y0);
   this->SetExplicitParam(5, z0);
   this->SetExplicitParam(6, x1);
   this->SetExplicitParam(7, y1);
   this->SetExplicitParam(8, z1);
   this->SetExplicitParam(9, r);
   this->Finalize();
}

double CappedCylindricalRod::InnerProduct(const double (&a)[3], const double (&b)[3])
{
   double prod = 0.0;
   for (int i=0; i<3; i++) 
      {prod += a[i]*b[i];}
   return prod;
}

double CappedCylindricalRod::Norml2(const double (&a)[3])
{
   double norm2 = this->InnerProduct(a, a);
   return sqrt(norm2);
}

// A*b -> c
void CappedCylindricalRod::MatVec(const double (&A)[3][3], const double (&b)[3], double (&c)[3]) 
{
   for (int i=0; i<3; i++) 
   { 
      c[i] = 0.0;
      for (int j=0; j<3; j++) 
      { 
         c[i] += A[i][j]*b[j]; 
      } 
   }    
}

// A*B -> C
void CappedCylindricalRod::MatMat(const double (&A)[3][3], const double (&B)[3][3], double (&C)[3][3]) 
{
   for (int i=0; i<3; i++) 
   {
      for (int j=0; j<3; j++) 
      {
         C[i][j] = 0.0;
         for (int k=0; k<3; k++) 
         {
            C[i][j] += A[i][k]*B[k][j];
         }   
      }
   }    
}

// A*B^T -> C
void CappedCylindricalRod::MatMatTrans(const double (&A)[3][3], const double (&B)[3][3], double (&C)[3][3]) 
{
   for (int i=0; i<3; i++) 
   {
      for (int j=0; j<3; j++) 
      {
         C[i][j] = 0.0;
         for (int k=0; k<3; k++) 
         {
            C[i][j] += A[i][k]*B[j][k];
         }   
      }
   }    
}

// A+B -> A
void CappedCylindricalRod::MatAdd(double (&A)[3][3], const double (&B)[3][3]) 
{
   for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
         A[i][j] += B[i][j];
}

// A -> c*A
void CappedCylindricalRod::MatScale(double (&A)[3][3], double c) 
{
   for (int i=0; i<3; i++) 
      for (int j=0; j<3; j++) 
         A[i][j] *= c; 
}

void CappedCylindricalRod::EvaluateVectors()
{
   X_[0]  = x_->Eval();
   X_[1]  = y_->Eval();
   X_[2]  = z_->Eval();

   X0_[0] = x0_->Eval();
   X0_[1] = y0_->Eval();
   X0_[2] = z0_->Eval();

   X1_[0] = x1_->Eval();
   X1_[1] = y1_->Eval();
   X1_[2] = z1_->Eval();

   for (int i=0; i<3; i++) { a_[i] = X1_[i] - X0_[i]; }
   for (int i=0; i<3; i++) { b_[i] =  X_[i] - X0_[i]; }   
}

double CappedCylindricalRod::Eval()
{
   double r = r_->Eval();

   this->EvaluateVectors(); // sets X_, X0_, X1_, a_, and b_

   double d;
   if (this->InnerProduct(a_, b_) <= 0.0)
   {
      d = Norml2(b_);
   }
   else if (this->InnerProduct(a_, b_) >= this->InnerProduct(a_, a_))
   {
      for (int i=0; i<3; i++) { e_[i] =  X_[i] - X1_[i]; }
      d  = this->Norml2(e_);
   }
   else
   {
      this->BuildOrthogonalProjector(); // sets P_
      this->MatVec(P_, b_, g_);
      d = this->Norml2(g_);
   }
   return r-d;
}

void CappedCylindricalRod::BuildOrthogonalProjector()
{
   // compute a dyad a
   for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
         P_[i][j] = a_[i]*a_[j];

   // scale by a dot a
   MatScale(P_, -1.0/InnerProduct(a_, a_));
     
   // add Identity
   for (int i=0; i<3; i++) { P_[i][i] += 1.0; }      
}

void CappedCylindricalRod::ComputeDg(const double (&Da)[3],
                                     const double (&Db)[3],
                                           double (&Dg)[3])
{
   // assumes BuildOrthoganalProjector() has been called already in ComputeExplicitGrad()
   double DP[3][3];

   double aDa[3][3];
   for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
         aDa[i][j] = a_[i]*Da[j];

   double tempMat[3][3];
   MatMat(aDa, P_, tempMat);
   MatMatTrans(P_, aDa, DP);
   MatAdd(DP, tempMat);
   MatScale(DP, -1.0/InnerProduct(a_, a_));

   double tempVec[3];
   MatVec(DP, b_, tempVec);
   MatVec(P_, Db, Dg);
   
   for (int i=0; i<3; i++) 
      {Dg[i] += tempVec[i];}
}

void CappedCylindricalRod::ComputeExplicitGrad()
{
   this->EvaluateVectors();

   if (InnerProduct(a_, b_) <= 0.0)
   {
      double scale = -1/(2*Norml2(b_));
      exp_grad_[0] =  scale*2*b_[0];
      exp_grad_[1] =  scale*2*b_[1];
      exp_grad_[2] =  scale*2*b_[2];
      exp_grad_[3] = -scale*2*b_[0];
      exp_grad_[4] = -scale*2*b_[1];
      exp_grad_[5] = -scale*2*b_[2];
      exp_grad_[6] =  0.0;
      exp_grad_[7] =  0.0;
      exp_grad_[8] =  0.0;
   }
   else if (InnerProduct(a_, b_) >= InnerProduct(a_, a_))
   {
      for (int i=0; i<3; ++i) { e_[i] =  X_[i] - X1_[i]; }
      double scale = -1/(2*Norml2(e_));
      exp_grad_[0] =  scale*2*e_[0];
      exp_grad_[1] =  scale*2*e_[1];
      exp_grad_[2] =  scale*2*e_[2];
      exp_grad_[3] =  0.0;
      exp_grad_[4] =  0.0;
      exp_grad_[5] =  0.0;
      exp_grad_[6] = -scale*2*e_[0];
      exp_grad_[7] = -scale*2*e_[1];
      exp_grad_[8] = -scale*2*e_[2];
   }
   else
   {
      this->BuildOrthogonalProjector();
      MatVec(P_, b_, g_);

      double scale = -1/(2*Norml2(g_));
      
      double Da[3], Db[3], Dg[3];

      // d_dx
      Da[0] = 0.0; Da[1] = 0.0; Da[2] = 0.0;
      Db[0] = 1.0; Db[1] = 0.0; Db[2] = 0.0;
      this->ComputeDg(Da, Db, Dg);
      exp_grad_[0] = scale*2*InnerProduct(g_, Dg);

      // d_dy
      Db[0] = 0.0; Db[1] = 1.0; Db[2] = 0.0;
      this->ComputeDg(Da, Db, Dg);
      exp_grad_[1] = scale*2*InnerProduct(g_, Dg);

      // d_dz
      Db[0] = 0.0; Db[1] = 0.0; Db[2] = 1.0;
      this->ComputeDg(Da, Db, Dg);
      exp_grad_[2] = scale*2*InnerProduct(g_, Dg);

      // d_dx0
      Da[0] = -1.0; Da[1] = 0.0; Da[2] = 0.0;
      Db[0] = -1.0; Db[1] = 0.0; Db[2] = 0.0;
      this->ComputeDg(Da, Db, Dg);
      exp_grad_[3] = scale*2*InnerProduct(g_, Dg);

      // d_dy0
      Da[0] = 0.0; Da[1] = -1.0; Da[2] = 0.0;
      Db[0] = 0.0; Db[1] = -1.0; Db[2] = 0.0;
      this->ComputeDg(Da, Db, Dg);
      exp_grad_[4] = scale*2*InnerProduct(g_, Dg);

      // d_dz0
      Da[0] = 0.0; Da[1] = 0.0; Da[2] = -1.0;
      Db[0] = 0.0; Db[1] = 0.0; Db[2] = -1.0;
      this->ComputeDg(Da, Db, Dg);
      exp_grad_[5] = scale*2*InnerProduct(g_, Dg);

      // d_dx1
      Db[0] = 0.0; Db[1] = 0.0; Db[2] = 0.0;
      Da[0] = 1.0; Da[1] = 0.0; Da[2] = 0.0;
      this->ComputeDg(Da, Db, Dg);
      exp_grad_[6] = scale*2*InnerProduct(g_, Dg);

      // d_dy1
      Da[0] = 0.0; Da[1] = 1.0; Da[2] = 0.0;
      this->ComputeDg(Da, Db, Dg);
      exp_grad_[7] = scale*2*InnerProduct(g_, Dg);

      // d_dz1
      Da[0] = 0.0; Da[1] = 0.0; Da[2] = 1.0;
      this->ComputeDg(Da, Db, Dg);
      exp_grad_[8] = scale*2*InnerProduct(g_, Dg);
   }
   exp_grad_[9] = 1.0;  // d_dr
}

SphericallySampledRodVolumeFraction::SphericallySampledRodVolumeFraction(
                                        Function *phi, Function *r)
  : Function(2), phi_(phi), r_(r)
{
   this->SetExplicitParam(0, phi); 
   this->SetExplicitParam(1, r); 
   this->Finalize();
}

double SphericallySampledRodVolumeFraction::Eval()
{
   double phi = phi_->Eval();   
   double   r = r_->Eval();
   double  vf;
   if (phi > r)
   {
      vf = 1.0;
   }
   else if (phi < -r)
   {
      vf = 0.0;
   }
   else
   {
      vf = (r+phi)*(r+phi)*(2.*r-phi) / (4.*r*r*r);
   }
   return vf;
}

void SphericallySampledRodVolumeFraction::ComputeExplicitGrad()
{
   double phi = phi_->Eval();
   double   r = r_->Eval();
   if ( (phi > r) || (phi < -r) )
   {
      exp_grad_[0] = 0.0;
      exp_grad_[1] = 0.0;
   }
   else
   {
      exp_grad_[0] =  (3.*(r*r-phi*phi)) / (4.*r*r*r);
      exp_grad_[1] =  (3.*phi*(phi*phi-r*r)) / (4.*r*r*r*r); 
   }
}

SmoothMax::SmoothMax(std::vector<Function*> x, double alpha)
  : Function(x.size()), n_(x.size()), x_(x), alpha_(alpha)
{
   for (int i=0; i<n_; i++)
      {this->SetExplicitParam(i, x[i]);}
   this->Finalize();
}

double SmoothMax::Eval()
{
   numer_ = 0.0;
   denom_ = 0.0;
   for (int i=0; i<n_; i++)
   {
      double xi = x_[i]->Eval();
      numer_   += xi*exp(alpha_*xi);
      denom_   +=    exp(alpha_*xi);
   }
   return numer_/denom_;
}

void SmoothMax::ComputeExplicitGrad()
{
   double Sa = this->Eval(); // sets denom_
   for (int i=0; i<n_; i++)
   {
      double xi = x_[i]->Eval();
      exp_grad_[i] = (exp(alpha_*xi) / denom_) * (1.0 + alpha_*(xi-Sa));
   }
}

PNorm::PNorm(std::vector<Function*> x, double p)
  : Function(x.size()), n_(x.size()), x_(x), p_(p)
{
   for (int i=0; i<n_; i++)
      {this->SetExplicitParam(i, x[i]);}
   this->Finalize();
}

double PNorm::Eval()
{
   double sum_xp = 0.0;
   for (int i=0; i<n_; i++)
      {sum_xp += pow(x_[i]->Eval(), p_);}
   return pow(sum_xp, 1./p_);
}

void PNorm::ComputeExplicitGrad()
{
   double sum_xp = 0.0;
   for (int i=0; i<n_; i++)
      {sum_xp += pow(x_[i]->Eval(), p_);}

   double scale = pow(sum_xp, (1./p_)-1.);
   for (int i=0; i<n_; i++)
      {exp_grad_[i] = scale * pow(x_[i]->Eval(), p_-1.);}
}

RpFunction::RpFunction(Function *f1, Function *f2, Function *a0, Function *a1, Function *a2, double a, double b, double c, double p)
  : Function(5), f1_(f1), f2_(f2), a0_(a0), a1_(a1), a2_(a2), a_(a), b_(b), c_(c), p_(p), blend_(true), eps_(1e-15)
{
   this->SetExplicitParam(0, f1);
   this->SetExplicitParam(1, f2);
   this->SetExplicitParam(2, a0);
   this->SetExplicitParam(3, a1);
   this->SetExplicitParam(4, a2);
   this->Finalize();
}

RpFunction::RpFunction(Function *f1, Function *f2, double a, double b, double c, double p)
  : Function(2), f1_(f1), f2_(f2), a_(a), b_(b), c_(c), p_(p), blend_(false), eps_(1e-15)
{
   this->SetExplicitParam(0, f1);
   this->SetExplicitParam(1, f2);
   this->Finalize();
}

double RpFunction::Eval()
{
   double f1 = f1_->Eval();
   double f2 = f2_->Eval();
   double value = a_*f1 + b_*f2 + c_*pow( pow(f1, p_) + pow(f2, p_) + eps_, 1./p_);
   if (blend_) 
   {
      double a0 = a0_->Eval();
      double a1 = a1_->Eval();
      double a2 = a2_->Eval();
      value += a0 / ( 1.0 + (f1*f1)/(a1*a1) + (f2*f2)/(a2*a2) );
   }
   return value;
}

void RpFunction::ComputeExplicitGrad()
{
   double f1 = f1_->Eval();
   double f2 = f2_->Eval();
   exp_grad_[0] = a_ + (c_*pow(pow(f1, p_) + pow(f2, p_) + eps_, 1./p_ - 1.)*pow(f1, p_-1)); // d_df1
   exp_grad_[1] = b_ + (c_*pow(pow(f1, p_) + pow(f2, p_) + eps_, 1./p_ - 1.)*pow(f2, p_-1)); // d_df2
   if (blend_)
   {
      double    a0 = a0_->Eval();
      double    a1 = a1_->Eval();
      double    a2 = a2_->Eval();
      double denom = 1.0 + (f1*f1)/(a1*a1) + (f2*f2)/(a2*a2);
      exp_grad_[0] += (-2*a0*f1/(a1*a1)) / (denom*denom);       // d_df1
      exp_grad_[1] += (-2*a0*f2/(a2*a2)) / (denom*denom);       // d_df2
      exp_grad_[2]  =  1 / denom;                               // d_da0
      exp_grad_[3]  = (2*a0*f1*f1/(a1*a1*a1)) / (denom*denom);  // d_da1
      exp_grad_[4]  = (2*a0*f2*f2/(a2*a2*a2)) / (denom*denom);  // d_da2
   }
}

Heaviside::Heaviside(Function *x, double alpha)
  : Function(1), x_(x), alpha_(alpha)
{
   this->SetExplicitParam(0, x);
   this->Finalize();
}

double Heaviside::Eval()
{
   return 0.5 * (1.0 + tanh(alpha_*x_->Eval()));
}

void Heaviside::ComputeExplicitGrad()
{
   exp_grad_[0]  = 0.5 * pow( cosh(alpha_*x_->Eval()), -2.0 ) * alpha_;
}

}  //  Implicit Geometry namespace
#endif
