// 1D transformation at the right boundary.
double right(const double eps, const double x)
{
   return (x <= 0.5) ? (2-eps) * x : 1 + eps*(x-1);
}

// 1D transformation at the left boundary
double left(const double eps, const double x)
{
   return 1-right(eps,1-x);
}

// Transition from a value of "a" for x=0, to a value of "b" for x=1.  Optionally
// smooth -- see the commented versions at the end.
double step(const double a, const double b, double x)
{
   if (x <= 0) return a;
   if (x >= 1) return b;
   return a + (b-a) * (x);
   // return a + (b-a) * (x*x*(3-2*x));
   // return a + (b-a) * (x*x*x*(x*(6*x-15)+10));
}

// 3D version of a generalized Kershaw mesh transformation, see D. Kershaw,
// "Differencing of the diffusion equation in Lagrangian hydrodynamic codes",
// JCP, 39:375–395, 1981.
//
// The input mesh should be Cartesian nx x ny x nz with nx divisible by 6 and
// ny, nz divisible by 2.
//
// The eps parameters are in (0, 1]. Uniform mesh is recovered for epsy=epsz=1.
void kershaw(const double epsy, const double epsz,
             const double x, const double y, const double z,
             double &X, double &Y, double &Z)
{
   X = x;

   int layer = x*6.0;
   double lambda = (x-layer/6.0)*6;

   // The x-range is split in 6 layers going from left-to-left, left-to-right,
   // right-to-left (2 layers), left-to-right and right-to-right yz-faces.
   switch (layer)
   {
   case 0:
      Y = left(epsy, y);
      Z = left(epsz, z);
      break;
   case 1:
   case 4:
      Y = step(left(epsy, y), right(epsy, y), lambda);
      Z = step(left(epsz, z), right(epsz, z), lambda);
      break;
   case 2:
      Y = step(right(epsy, y), left(epsy, y), lambda/2);
      Z = step(right(epsz, z), left(epsz, z), lambda/2);
      break;
   case 3:
      Y = step(right(epsy, y), left(epsy, y), (1+lambda)/2);
      Z = step(right(epsz, z), left(epsz, z), (1+lambda)/2);
      break;
   default:
      Y = right(epsy, y);
      Z = right(epsz, z);
      break;
   }
}
