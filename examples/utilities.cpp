///////////////////////////
// SOME USEFUL FUNCTIONS //
///////////////////////////

////////////////////////////////////////////
// INITIAL CONDITION FOR X AND Os PROBLEM //
////////////////////////////////////////////
double box(std::pair<double,double> p1, std::pair<double,double> p2, double theta, 
	   std::pair<double,double> origin, double x, double y)
{
  double xmin=p1.first;
  double xmax=p2.first;
  double ymin=p1.second;
  double ymax=p2.second;
  double ox=origin.first;
  double oy=origin.second;
  
  double pi = M_PI;
  double s=std::sin(theta*pi/180);
  double c=std::cos(theta*pi/180);

  double xn=c*(x-ox)-s*(y-oy)+ox;
  double yn=s*(x-ox)+c*(y-oy)+oy;

  if (xn>xmin && xn<xmax && yn>ymin && yn<ymax)
    return 1.0;
  else 
    return 0.0;
}

double get_cross(double rect1, double rect2)
{
  double intersection=rect1*rect2;
  return rect1+rect2-intersection; //union
}

double ring(double cx, double cy, double rin, double rout, double x, double y)
{
  double r=std::sqrt(pow(x-cx,2)+pow(y-cy,2));
  if (r>rin && r<rout)
    return 1.0;
  else 
    return 0.0;
}

///////////////////////
// VELOCITY FUNCTION //
///////////////////////
void velocity_function(const Vector &x, Vector &v)
{
#if 1
  v(0)=1;
  v(1)=0;
#elif 0
  v(0)=10.;
  v(1)=10.;
#elif 1
  double Pi = M_PI;
  double t = current_time;
  v(0) = std::sin(Pi*x(0))*std::cos(Pi*x(1))*std::sin(2*Pi*t);
  v(1) = -std::cos(Pi*x(0))*std::sin(Pi*x(1))*std::sin(2*Pi*t);
#else
   const double angular_v = 2 * M_PI;
   v = 0.;
   v(0) = -x(1) * angular_v;
   v(1) = +x(0) * angular_v;
#endif
}

/////////////////////
// INFLOW FUNCTION //
/////////////////////
double inflow_function(Vector &x)
{
   return 0.;
   //return 1.0;
}

///////////////////////////////////
// INIT CONDITION FOR MATERIAL 1 //
///////////////////////////////////
double u0Mat1_function(Vector &x)
{
   int dim = x.Size();
#if 0
   // 1D periodic smooth solution on [0,1]
   return cos(2*M_PI*(x(0)-0.5));

#elif 1
   // 1D square in (0,1)
   return ( x(0) > 0.4 && x(0) < 0.6) ? 1.0 : 0.0;
   //return ( x(0) > 0.15 && x(0) < 0.35) ? 1.0 : 0.0;

#elif 0
   // 2D sin function
   return std::sin(M_PI*x(1));
   //return x(0)*cos(x(1))+x(1)*cos(x(0));

#elif 0
   // 2D rotation of hyp tangent on a unit square
   return std::tanh((x(1)-0.5)/0.25);
#elif 0
   // 2D square
   return (fabs(x(0)+0.5) < 0.2 && fabs(x(1)+0.5) < 0.2) ? 1.0 : 0.0;
   //return (fabs(x(0)-0.0) < 0.2 && fabs(x(1)-0.0) < 0.2) ? 1.0 : 0.0;
   //return (fabs(x(0)-20.) < 10. && fabs(x(1)-20.) < 10.) ? 1.0 : 0.0;
#else 
   std::pair<double, double> p1;
   std::pair<double, double> p2;
   std::pair<double, double> origin;

   // cross
   p1.first=14; p1.second=3;
   p2.first=17; p2.second=26;
   origin.first = 0.5*(32 - 7);
   origin.second = 0.5*(26 - 3);
   double rect1=box(p1,p2,-45,origin,x(0),x(1));
   p1.first=7; p1.second=10;
   p2.first=32; p2.second=13;
   double rect2=box(p1,p2,-45,origin,x(0),x(1));
   double cross=get_cross(rect1,rect2);
   // rings
   double ring1 = ring(40,40,7,10,x(0),x(1));
   double ring2 = ring(40,20,3,7,x(0),x(1));
   double rings = ring1+ring2;
   // cross and rings
   return (cross+rings);
#endif
}

///////////////////////////////////
// INIT CONDITION FOR MATERIAL 2 //
///////////////////////////////////
double u0Mat2_function(Vector &x)
{
  std::pair<double, double> p1;
  std::pair<double, double> p2;
  std::pair<double, double> origin;
  // MATERIAL 2
  //cross
  p1.first=9; p1.second=23;
  p2.first=12; p2.second=46;
  origin.first=0; origin.second=0;
  double rect1=box(p1,p2,0,origin,x(0),x(1));
  p1.first=2; p1.second=30;
  p2.first=27; p2.second=33;
  double rect2=box(p1,p2,0,origin,x(0),x(1));
  double cross=get_cross(rect1,rect2);
  // rings
  double ring1 = ring(40,40,0,7,x(0),x(1));
  double ring2 = ring(40,20,0,3,x(0),x(1));
  double ring3 = ring(40,20,7,10,x(0),x(1));
  double rings = ring1+ring2+ring3;
  // cross and rings
  return (cross+rings);
}

///////////////////////////////////
// INIT CONDITION FOR MATERIAL 3 //
///////////////////////////////////
double u0Mat3_function(Vector &x)
{
  return 1.-(u0Mat1_function(x)+u0Mat2_function(x));
}


///////////////////////////////////
// COMPUTE (CONVENTIONAL) BOUNDS //
///////////////////////////////////
// original code
void ComputeBounds(const SparseMatrix &K, const Vector &x,
                   Vector &x_min, Vector &x_max)
{
   const int *I = K.GetI(), *J = K.GetJ(), size = K.Size();

   for (int i = 0, k = 0; i < size; i++)
   {
      double x_i_min = numeric_limits<double>::infinity();
      double x_i_max = -x_i_min;
      for (int end = I[i+1]; k < end; k++)
      {
         double x_j = x(J[k]);

         if (x_j > x_i_max)
            x_i_max = x_j;
         if (x_j < x_i_min)
            x_i_min = x_j;
      }

      //x_min(i) = max(x_i_min,u_MIN);
      //x_max(i) = min(x_i_max,u_MAX);
      x_min(i) = x_i_min;
      x_max(i) = x_i_max;
   }
}

//////////////////////////
// SPARSITY OF MATRICES //
//////////////////////////
int *SparseMatrix_Build_smap(SparseMatrix &A)
{
   // assuming that A is finalized
   const int *I = A.GetI(), *J = A.GetJ(), n = A.Size();
   int *smap = new int[I[n]];

   for (int row = 0, j = 0; row < n; row++)
   {
      for (int end = I[row+1]; j < end; j++)
      {
         int col = J[j];

         // find the offset, _j, of the (col,row) entry and store it in smap[j]:
         for (int _j = I[col], _end = I[col+1]; true; _j++)
         {
            if (_j == _end)
               mfem_error("SparseMatrix_Build_smap");

            if (J[_j] == row)
            {
               smap[j] = _j;
               break;
            }
         }
      }
   }

   return smap;
}

/////////////////////////////////////
// TO SOLVE NON-LINEARITY PER CELL //
/////////////////////////////////////
double get_lambda_times_sum_z(double lambda, Vector &w, Vector &flux)
{
  //double tol=1e-30;
  Vector lambda_times_z(w.Size());
  double lambda_times_sum_z=0.;
  for (int j=0; j<w.Size(); j++)
    {
      //if (abs(flux(j)) > tol) //flux(j)!=0
      if (flux(j)!=0)
	{
	  //lambda_times_z(j) = lambda*w(j) + flux(j)*min(0.,1-lambda*w(j)/flux(j));	  
	  lambda_times_z(j) = ((abs(flux(j)) >= lambda*abs(w(j))) ? lambda*w(j) : flux(j));
	}
      else 
	lambda_times_z(j) = 0;
      lambda_times_sum_z += lambda_times_z(j);
    }  
  return lambda_times_sum_z;
}

void get_z(double lambda, Vector &w, Vector &flux, Vector &zz)
{
  if (lambda==0)
    zz = 0.;
  else
    for (int j=0; j<w.Size(); j++)
      {
	if (flux(j)!=0)
	  //zz(j) = w(j) + flux(j)/lambda*min(0.,1-lambda*w(j)/flux(j));
	  zz(j) = ((abs(flux(j)) >= lambda*abs(w(j))) ? w(j) : flux(j)/lambda);
	else
	  zz(j) = 0;
      }
}

////////////
// OTHERS //
////////////
void print_map_of_maps(map<int, map<int,int> > &M) 
{
  // iterate on first 
  for (map<int, map<int,int> >::iterator iter=M.begin(); iter!=M.end(); iter++)
    {
      cout << "x: " << iter->first << ", ys: ";
      for (map<int,int>::iterator it=iter->second.begin(); it!=iter->second.end(); it++)
	{
	  cout << it->second << ", ";
	}
      cout << endl;
    }
}
