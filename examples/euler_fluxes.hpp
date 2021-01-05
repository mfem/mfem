/// Functions related to Euler equations

#include <algorithm> // std::max

/// For constants related to the Euler equations
namespace euler
{
/// gas constant
const double R = 287;
/// heat capcity ratio for air
const double gamma = 1.4;
/// ratio minus one
const double gami = gamma - 1.0;
} // namespace euler

/// for performing loop unrolling of dot-products using meta-programming
/// \tparam double - `double` 
/// \tparam dim - number of dimensions for array
/// This was adapted from http://www.informit.com/articles/article.aspx?p=30667&seqNum=7
template <int dim>
class DotProduct {
  public:
    static double result(const double *a, const double *b)
    {
        return (*a) * (*b)  +  DotProduct<dim-1>::result(a+1,b+1);
    }
};
// partial specialization as end criteria
template <>
class DotProduct<1> {
  public:
    static double result(const double *a, const double *b)
    {
        return *a * *b;
    }
};
/// dot product of two arrays that uses an unrolled loop
/// \param[in] a - first vector involved in product
/// \param[in] b - second vector involved in product
/// \tparam double - typically `double` 
/// \tparam dim - number of array dimensions
template <int dim>
inline double dot(const double *a, const double *b)
{
    return DotProduct<dim>::result(a,b);
}

/// Pressure based on the ideal gas law equation of state
/// \param[in] q - the conservative variables
/// \tparam double - either double or adouble
/// \tparam dim - number of physical dimensions
template <int dim>
inline double pressure(const double *q)
{
   double vsq = 0.0;
   for (int k = 0; k<dim; ++k)
   {
       vsq += q[k+1]*q[k+1];
   }
   return euler::gami * (q[dim + 1] - 0.5 * vsq / q[0]);
}

/// Convert entropy variables `w` to conservative variables `q`
/// \param[in] w - entropy variables we want to convert from
/// \param[out] q - conservative variables that we want to convert to
/// \tparam double - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
void calcConservativeVars(const double *w, double *q)
{
   double u[dim];
   double Vel2 = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      u[i] = -w[i + 1] / w[dim + 1];
      Vel2 += u[i]*u[i];
   }
   double s = euler::gamma + euler::gami*(0.5*Vel2*w[dim+1] - w[0]);
   q[0] = pow(-exp(-s)/w[dim+1], 1.0/euler::gami);
   for (int i = 0; i < dim; ++i)
   {
      q[i + 1] = q[0]*u[i];
   }
   double p = -q[0]/w[dim+1];
   q[dim+1] = p/euler::gami + 0.5*q[0]*Vel2;  
}

/// Mathematical entropy function rho*s/(gamma-1), where s = ln(p/rho^gamma)
/// \param[in] q - state variables (either conservative or entropy variables)
/// \tparam double - either double or adouble
/// \tparam dim - number of physical dimensions
/// \tparam entvar - if true q = conservative vars, if false q = entropy vars
template <int dim, bool entvar = false>
inline double entropy(const double *q)
{
   if (entvar)
   {
    //   double Vel2 = dot<double, dim>(q + 1, q + 1); // Vel2*rho^2/p^2
    //   double s = -euler::gamma + euler::gami*(q[0] - 0.5*Vel2/q[dim+1]); // -s
    //   double rho = pow(-exp(s)/q[dim+1], 1.0/euler::gami);
    //   return rho*s/euler::gami;
   }
   else
   {
      return -q[0]*log(pressure<dim>(q)/pow(q[0],euler::gamma))/euler::gami;
   }
}

/// Euler flux function in a given (scaled) direction
/// \param[in] dir - direction in which the flux is desired
/// \param[in] q - conservative variables
/// \param[out] flux - fluxes in the direction `dir`
/// \tparam double - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
void calcEulerFlux(const double *dir, const double *q, double *flux)
{
   using namespace std;
   double press = pressure<dim>(q);
   double U = dot<dim>(q + 1, dir);
   flux[0] = U;
   U /= q[0];
   for (int i = 0; i < dim; ++i)
   {
      flux[i + 1] = q[i + 1] * U + dir[i] * press;
   }
   flux[dim + 1] = (q[dim + 1] + press) * U;
}

/// Log-average function used in the Ismail-Roe flux function
/// \param[in] aL - nominal left state variable to average
/// \param[in] aR - nominal right state variable to average
/// \returns the logarithmic average of `aL` and `aR`.
/// \tparam double - typically `double` or `adept::adouble`
double logavg(const double &aL, const double &aR)
{
   double xi = aL / aR;
   double f = (xi - 1) / (xi + 1);
   double u = f * f;
   double eps = 1.0e-3;
   double F;
   if (u < eps)
   {
      F = 1.0 + u * (1. / 3. + u * (1. / 5. + u * (1. / 7. + u / 9.)));
   }
   else
   {
      F = (log(xi) / 2.0) / f;
   }
   return (aL + aR) / (2.0 * F);
}


/// The spectral radius of flux Jacobian in direction `dir` w.r.t. conservative
/// \param[in] dir - desired direction of flux Jacobian
/// \param[in] u - state variables used to evaluate Jacobian
/// \returns absolute value of the largest eigenvalue of the Jacobian
/// \tparam double - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true u = conservative vars, if false u = entropy vars
template <int dim, bool entvar = false>
double calcSpectralRadius(const double *dir, const double *u)
{
   double q[dim+2];
   if (entvar)
   {
      calcConservativeVars<dim>(u, q);
   }
   else
   {
      for (int i = 0; i < dim+2; ++i)
      {
         q[i] = u[i];
      }
   }
   double press = pressure<dim>(q);
   double sndsp = sqrt(euler::gamma * press / q[0]);
   // U = u*dir[0] + v*dir[1] + ...
   double U = dot<dim>(q + 1, dir) / q[0];
   double dir_norm = sqrt(dot<dim>(dir, dir));
   return fabs(U) + sndsp * dir_norm;
}

// TODO: How should we return matrices, particularly when they will be differentiated?
template <int dim>
void calcdQdWProduct(const double *q, const double *vec, double *dqdw_vec)
{
   double p = pressure<dim>(q);
   double rho_inv = 1.0 / q[0];
   double h = (q[dim + 1] + p) * rho_inv;  // scaled version of h
   double a2 = euler::gamma * p * rho_inv; // square of speed of sound

   // first row of dq/dw times vec
   dqdw_vec[0] = 0.0;
   for (int i = 0; i < dim + 2; ++i)
   {
      dqdw_vec[0] += q[i] * vec[i];
   }

   // second through dim-1 rows of dq/dw times vec
   for (int j = 0; j < dim; ++j)
   {
      dqdw_vec[j + 1] = 0.0;
      double u = q[j + 1] * rho_inv;
      for (int i = 0; i < dim + 2; ++i)
      {
         dqdw_vec[j + 1] += u * q[i] * vec[i];
      }
      dqdw_vec[j + 1] += p * vec[j + 1];
      dqdw_vec[j + 1] += p * u * vec[dim + 1];
   }

   // dim-th row of dq/dw times vec
   dqdw_vec[dim + 1] = q[dim + 1] * vec[0];
   for (int i = 0; i < dim; ++i)
   {
      dqdw_vec[dim + 1] += q[i + 1] * h * vec[i + 1];
   }
   dqdw_vec[dim + 1] += (q[0] * h * h - a2 * p / euler::gami) * vec[dim + 1];
}


/// Boundary flux that uses characteristics to determine which state to use
/// \param[in] dir - direction in which the flux is desired
/// \param[in] qbnd - boundary values of the conservative variables
/// \param[in] q - interior domain values of the conservative variables
/// \param[in] work - a work vector of size `dim+2`
/// \param[out] flux - fluxes in the direction `dir`
/// \tparam double - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note the "flux Jacobian" is computed using `qbnd`, so the boundary values
/// define what is inflow and what is outflow.
template <int dim>
void calcBoundaryFlux(const double *dir, const double *qbnd, const double *q,
                      double *work, double *flux)
{
   using std::max;

   // Define some constants
   const double sat_Vn = 0.0; // 0.025
   const double sat_Vl = 0.0; // 0.025
   // Define some constants used to construct the "Jacobian"
   const double dA = sqrt(dot<dim>(dir, dir));
   const double fac = 1.0 / qbnd[0];
   const double phi = 0.5 * dot<dim>(qbnd + 1, qbnd + 1) * fac * fac;
   const double H = euler::gamma * qbnd[dim + 1] * fac - euler::gami * phi;
   const double a = sqrt(euler::gami * (H - phi));
   const double Un = dot< dim>(qbnd + 1, dir) * fac;
   double lambda1 = Un + dA * a;
   double lambda2 = Un - dA * a;
   double lambda3 = Un;
   const double rhoA = fabs(Un) + dA * a;
   lambda1 = 0.5 * (max(fabs(lambda1), sat_Vn * rhoA) - lambda1);
   lambda2 = 0.5 * (max(fabs(lambda2), sat_Vn * rhoA) - lambda2);
   lambda3 = 0.5 * (max(fabs(lambda3), sat_Vl * rhoA) - lambda3);

   double *dq = work;
   for (int i = 0; i < dim + 2; ++i)
   {
      dq[i] = q[i] - qbnd[i];
   }
   calcEulerFlux<dim>(dir, q, flux);
   // cout << "euler flux " <<  endl;
   // for (int k = 0; k < dim + 2; ++k)
   // {
   //    cout << flux[k] << endl;
   // }
   // diagonal matrix multiply; note that flux was initialized by calcEulerFlux
   for (int i = 0; i < dim + 2; ++i)
   {
      flux[i] += lambda3 * dq[i];
   }

   // some scalars needed for E1*dq, E2*dq, E3*dq, and E4*dq
   double tmp1 = 0.5 * (lambda1 + lambda2) - lambda3;
   double E1dq_fac = tmp1 * euler::gami / (a * a);
   double E2dq_fac = tmp1 / (dA * dA);
   double E34dq_fac = 0.5 * (lambda1 - lambda2) / (dA * a);

   // get E1*dq + E4*dq and add to flux
   double Edq = phi * dq[0] + dq[dim + 1] - dot<dim>(qbnd + 1, dq + 1) * fac;
   flux[0] += E1dq_fac * Edq;
   for (int i = 0; i < dim; ++i)
   {
      flux[i + 1] += Edq * (E1dq_fac * qbnd[i + 1] * fac + euler::gami * E34dq_fac * dir[i]);
   }
   flux[dim + 1] += Edq * (E1dq_fac * H + euler::gami * E34dq_fac * Un);

   // get E2*dq + E3*dq and add to flux
   Edq = -Un * dq[0] + dot< dim>(dir, dq + 1);
   flux[0] += E34dq_fac * Edq;
   for (int i = 0; i < dim; ++i)
   {
      flux[i + 1] += Edq * (E2dq_fac * dir[i] + E34dq_fac * qbnd[i + 1] * fac);
   }
   flux[dim + 1] += Edq * (E2dq_fac * Un + E34dq_fac * H);
}

/// Boundary flux that uses characteristics to determine which state to use
/// \param[in] dir - direction in which the flux is desired
/// \param[in] qbnd - boundary values of the **conservative** variables
/// \param[in] q - interior domain values of the state variables
/// \param[in] work - a work vector of size `dim+2`
/// \param[out] flux - fluxes in the direction `dir`
/// \tparam double - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, `q` is entropy var; otherwise, `q` is conservative
/// \note the "flux Jacobian" is computed using `qbnd`, so the boundary values
/// define what is inflow and what is outflow.
template <int dim, bool entvar = false>
void calcFarFieldFlux(const double *dir, const double *qbnd, const double *q,
                      double *work, double *flux)
{
   if (entvar)
   {
      double qcons[dim+2];
      calcConservativeVars<dim>(q, qcons);
      calcBoundaryFlux< dim>(dir, qbnd, qcons, work, flux);
   }
   else
   {
      calcBoundaryFlux<dim>(dir, qbnd, q, work, flux);
   }
}

/// Isentropic vortex exact state as a function of position
/// \param[in] x - location at which the exact state is desired
/// \param[out] qbnd - vortex conservative variable at `x`
/// \tparam double - typically `double` or `adept::adouble`
/// \note  I reversed the flow direction to be clockwise, so the problem and
/// mesh are consistent with the LPS paper (that is, because the triangles are
/// subdivided from the quads using the opposite diagonal)
void calcIsentropicVortexState(const double *x, double *qbnd)
{
   double ri = 1.0;
   double Mai = 0.5; //0.95
   double rhoi = 2.0;
   double prsi = 1.0 / euler::gamma;
   double rinv = ri / sqrt(x[0] * x[0] + x[1] * x[1]);
   double rho = rhoi * pow(1.0 + 0.5 * euler::gami * Mai * Mai * (1.0 - rinv * rinv),
                            1.0 / euler::gami);
   double Ma = sqrt((2.0 / euler::gami) * ((pow(rhoi / rho, euler::gami)) *
                                                (1.0 + 0.5 * euler::gami * Mai * Mai) -
                                            1.0));
   double theta;
   if (x[0] > 1e-15)
   {
      theta = atan(x[1] / x[0]);
   }
   else
   {
      theta = M_PI / 2.0;
   }
   double press = prsi * pow((1.0 + 0.5 * euler::gami * Mai * Mai) /
                                  (1.0 + 0.5 * euler::gami * Ma * Ma),
                              euler::gamma / euler::gami);
   double a = sqrt(euler::gamma * press / rho);

   qbnd[0] = rho;
   qbnd[1] = -rho * a * Ma * sin(theta);
   qbnd[2] = rho * a * Ma * cos(theta);
   qbnd[3] = press / euler::gami + 0.5 * rho * a * a * Ma * Ma;
}

/// A wrapper for `calcBoundaryFlux` in the case of the isentropic vortex
/// \param[in] x - location at which the boundary flux is desired
/// \param[in] dir - desired (scaled) direction of the flux
/// \param[in] q - state variable on the interior of the boundary
/// \param[out] flux - the boundary flux in the direction `dir`
/// \tparam double - typically `double` or `adept::adouble`
/// \tparam entvar - if true, `q` is entropy var; otherwise, `q` is conservative
template <bool entvar = false>
void calcIsentropicVortexFlux(const double *x, const double *dir,
                              const double *q, double *flux)
{
   double qbnd[4];
   double work[4];
   calcIsentropicVortexState(x, qbnd);
   if (entvar)
   {
      double qcons[4];
      calcConservativeVars< 2>(q, qcons);
      calcBoundaryFlux< 2>(dir, qbnd, qcons, work, flux);
   }
   else {
      calcBoundaryFlux<2>(dir, qbnd, q, work, flux);
   }
}

/// removes the component of momentum normal to the wall from `q`
/// \param[in] dir - vector perpendicular to the wall (does not need to be unit)
/// \param[in] q - the state whose momentum is being projected
/// \param[in] qbnd - the state with the normal component removed
/// \tparam double - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
void projectStateOntoWall(const double *dir, const double *q, double *qbnd)
{
   double nrm[dim];
   double fac = 1.0 / sqrt(dot<dim>(dir, dir));
   double Unrm = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      nrm[i] = dir[i] * fac;
      Unrm += nrm[i] * q[i + 1];
   }
   qbnd[0] = q[0];
   qbnd[dim + 1] = q[dim + 1];
   for (int i = 0; i < dim; ++i)
   {
      qbnd[i + 1] = q[i + 1] - nrm[i] * Unrm;
   }
}
/// Lax-Friedrichs flux function
/// \param[in] dir - vector direction in which flux is wanted
/// \param[in] qL - conservative variables at "left" state
/// \param[in] qR - conservative variables at "right" state
/// \param[out] flux - fluxes in the direction `di`
/// \tparam double - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
void calcLaxFriedrichsFlux(const double *dir, const double *qL, const double *qR,
                       double *flux)
{
   double fluxL[dim + 2];
   double fluxR[dim + 2];
   double q_ave[dim + 2];
   double q_diff[dim + 2];
   calcEulerFlux<dim>(dir, qL, fluxL);
   calcEulerFlux<dim>(dir, qR, fluxR);
   for (int i = 0; i < dim + 2; i++)
   {
      q_ave[i] = 0.5 * (qL[i] + qR[i]);
      q_diff[i] = -qR[i] + qL[i];
   }
   double lambda  = 1.0 * calcSpectralRadius<dim, false>(dir, q_ave);
   for (int k = 0; k < dim + 2; ++k)
   {
      flux[k] = fluxL[k] + fluxR[k] + (lambda * q_diff[k]);
      flux[k] *= 0.5;
   }
}

/// Roe flux function in direction `dir`
/// \param[in] dir - vector direction in which flux is wanted
/// \param[in] qL - conservative variables at "left" state
/// \param[in] qR - conservative variables at "right" state
/// \param[out] flux - fluxes in the direction `dir`
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
void calcRoeFaceFlux(const double *dir, const double *qL,
                     const double *qR, double *flux)
{
   using std::max;
   // Define some constants
   const double sat_Vn = 0.025;
   const double sat_Vl = 0.025;
   // Define the Roe-average state 
   const double sqL = sqrt(qL[0]);
   const double sqR = sqrt(qR[0]);
   const double facL = 1.0/qL[0];
   const double facR = 1.0/qR[0];
   const double fac = 1.0/(sqL + sqR);
   double u[dim]; // should pass in another work array?
   for (int i = 0; i < dim; ++i)
   {
      u[i] = (sqL*qL[i+1]*facL + sqR*qR[i+1]*facR)*fac;
   }
   const double phi = 0.5*dot<dim>(u, u);
   const double Un = dot<dim>(dir, u);
   const double HL = (euler::gamma * qL[dim + 1] - 0.5 * euler::gami *         
                       dot<dim>(qL + 1, qL + 1) * facL) * facL;
   const double HR = (euler::gamma * qR[dim + 1] - 0.5 * euler::gami *         
                       dot<dim>(qR + 1, qR + 1) * facR) * facR;
   const double H = (sqL*HL + sqR*HR)*fac;
   const double a = sqrt(euler::gami * (H - phi));
   const double dA = sqrt(dot<dim>(dir, dir));
   // Define the wave speeds
   const double rhoA = fabs(Un) + dA * a;
   const double lambda1 = 0.5 * max(fabs(Un + dA * a), sat_Vn * rhoA);
   const double lambda2 = 0.5 * max(fabs(Un - dA * a), sat_Vn * rhoA);
   const double lambda3 = 0.5 * max(fabs(Un), sat_Vl * rhoA);
   // start flux computation by averaging the Euler flux
   double dq[dim+2];
   calcEulerFlux<dim>(dir, qL, flux);
   calcEulerFlux<dim>(dir, qR, dq);
   for (int i = 0; i < dim + 2; ++i)
   {
      flux[i] = 0.5*(flux[i] + dq[i]);
      dq[i] = qL[i] - qR[i];
      flux[i] += lambda3 * dq[i]; // diagonal matrix multiply 
   }
   // some scalars needed for E1*dq, E2*dq, E3*dq, and E4*dq
   double tmp1 = 0.5 * (lambda1 + lambda2) - lambda3;
   double E1dq_fac = tmp1 * euler::gami / (a * a);
   double E2dq_fac = tmp1 / (dA * dA);
   double E34dq_fac = 0.5 * (lambda1 - lambda2) / (dA * a);
   // get E1*dq + E4*dq and add to flux
   double Edq = phi * dq[0] + dq[dim + 1] - dot<dim>(u, dq + 1);
   flux[0] += E1dq_fac * Edq;
   for (int i = 0; i < dim; ++i)
   {
      flux[i + 1] += Edq * (E1dq_fac * u[i] + euler::gami * E34dq_fac * dir[i]);
   }
   flux[dim + 1] += Edq * (E1dq_fac * H + euler::gami * E34dq_fac * Un);
   // get E2*dq + E3*dq and add to flux
   Edq = -Un * dq[0] + dot<dim>(dir, dq + 1);
   flux[0] += E34dq_fac * Edq;
   for (int i = 0; i < dim; ++i)
   {
      flux[i + 1] += Edq * (E2dq_fac * dir[i] + E34dq_fac * u[i]);
   }
   flux[dim + 1] += Edq * (E2dq_fac * Un + E34dq_fac * H);
}

/// computes an adjoint consistent slip wall boundary condition
/// \param[in] x - not used
/// \param[in] dir - desired (scaled) normal vector to the wall
/// \param[in] q - conservative state variable on the boundary
/// \param[out] flux - the boundary flux in the direction `dir`
/// \tparam double - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, `q` = ent. vars; otherwise, `q` = conserv. vars
template <int dim, bool entvar = false>
void calcSlipWallFlux(const double *x, const double *dir, const double *q,
                      double *flux)
{
#if 0
   double qbnd[dim+2];
   projectStateOntoWall<double,dim>(dir, q, qbnd);
   calcEulerFlux<double,dim>(dir, qbnd, flux);
   //calcIsentropicVortexFlux<double>(x, dir, q, flux);
#endif
   double press;
   if (entvar)
   {
      double Vel2 = dot<dim>(q+1, q+1);
      double s = euler::gamma + euler::gami*(0.5*Vel2/q[dim+1] - q[0]);
      press = -pow(-exp(-s)/q[dim+1], 1.0/euler::gami)/q[dim+1];
   }
   else
   {
      press = pressure<dim>(q);
   }
   flux[0] = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      flux[i + 1] = dir[i] * press;
   }
   flux[dim + 1] = 0.0;
}

/// Convert conservative variables `q` to entropy variables `w`
/// \param[in] q - conservative variables that we want to convert from
/// \param[out] w - entropy variables we want to convert to
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
void calcEntropyVars(const double *q, double *w)
{
   double u[dim];
   for (int i = 0; i < dim; ++i)
   {
      u[i] = q[i + 1] / q[0];
   }
   double p = pressure<dim>(q);
   double s = log(p / pow(q[0], euler::gamma));
   double fac = 1.0 / p;
   w[0] = (euler::gamma - s) / euler::gami - 0.5 * dot<dim>(u, u) * fac * q[0];
   for (int i = 0; i < dim; ++i)
   {
      w[i + 1] = q[i + 1] * fac;
   }
   w[dim + 1] = -q[0] * fac;
}

