# include <cfloat>
# include <cmath>
# include <cstdlib>
# include <ctime>
# include <iomanip>
# include <iostream>

using namespace std;

# include "elliptic_integral.hpp"

//****************************************************************************80

double elliptic_ea ( double a )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_EA evaluates the complete elliptic integral E(A).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      E(a) = RF ( 0, 1-sin^2(a), 1 ) - 1/3 sin^2(a) RD ( 0, 1-sin^2(a), 1 ).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double A, the argument.
//
//    Output, double ELLIPTIC_EA, the function value.
//
{
  double errtol;
  int ierr;
  double k;
  const double r8_pi = 3.141592653589793;
  double value;
  double x;
  double y;
  double z;

  k = sin ( a * r8_pi / 180.0 );

  x = 0.0;
  y = ( 1.0 - k ) * ( 1.0 + k );
  z = 1.0;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr ) 
    - k * k * rd ( x, y, z, errtol, ierr ) / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_ea_values ( int &n_data, double &x, double &fx )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_EA_VALUES returns values of the complete elliptic integral E(A).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic
//    integral of the second kind.
//
//    The function is defined by the formula:
//
//      E(A) = integral ( 0 <= T <= PI/2 )
//        sqrt ( 1 - sin ( A )^2 * sin ( T )^2 ) dT
//
//    In Mathematica, the function can be evaluated by:
//
//      EllipticE[(Sin[Pi*a/180])^2]
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 August 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 before the
//    first call.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &X, the argument of the function, measured
//    in degrees.
//
//    Output, double &FX, the value of the function.
//
{
# define N_MAX 18

  static double fx_vec[N_MAX] = {
     1.570796326794897E+00,
     1.567809073977622E+00,
     1.558887196601596E+00,
     1.544150496914673E+00,
     1.523799205259774E+00,
     1.498114928422116E+00,
     1.467462209339427E+00,
     1.432290969306756E+00,
     1.393140248523812E+00,
     1.350643881047676E+00,
     1.305539094297794E+00,
     1.258679624779997E+00,
     1.211056027568459E+00,
     1.163827964493139E+00,
     1.118377737969864E+00,
     1.076405113076403E+00,
     1.040114395706010E+00,
     1.012663506234396E+00 };

  static double x_vec[N_MAX] = {
      0.0E+00,
      5.0E+00,
     10.0E+00,
     15.0E+00,
     20.0E+00,
     25.0E+00,
     30.0E+00,
     35.0E+00,
     40.0E+00,
     45.0E+00,
     50.0E+00,
     55.0E+00,
     60.0E+00,
     65.0E+00,
     70.0E+00,
     75.0E+00,
     80.0E+00,
     85.0E+00 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    x = 0.0;
    fx = 0.0;
  }
  else
  {
    x = x_vec[n_data-1];
    fx = fx_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_ek ( double k )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_EK evaluates the complete elliptic integral E(K).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      E(k) = RF ( 0, 1-k^2, 1 ) - 1/3 k^2 RD ( 0, 1-k^2, 1 ).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double K, the argument.
//
//    Output, double ELLIPTIC_EK, the function value.
//
{
  double errtol;
  int ierr;
  double value;
  double x;
  double y;
  double z;

  x = 0.0;
  y = ( 1.0 - k ) * ( 1.0 + k );
  z = 1.0;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr ) 
    - k * k * rd ( x, y, z, errtol, ierr ) / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_ek_values ( int &n_data, double &x, double &fx )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_EK_VALUES returns values of the complete elliptic integral E(K).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic
//    integral of the second kind.
//
//    The function is defined by the formula:
//
//      E(K) = integral ( 0 <= T <= PI/2 )
//        sqrt ( 1 - K^2 * sin ( T )^2 ) dT
//
//    In Mathematica, the function can be evaluated by:
//
//      EllipticE[k^2]
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    29 May 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 before the
//    first call.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &X, the argument of the function.
//
//    Output, double &FX, the value of the function.
//
{
# define N_MAX 20

  static double fx_vec[N_MAX] = {
     1.570796326794897E+00,
     1.550973351780472E+00,
     1.530757636897763E+00,
     1.510121832092819E+00,
     1.489035058095853E+00,
     1.467462209339427E+00,
     1.445363064412665E+00,
     1.422691133490879E+00,
     1.399392138897432E+00,
     1.375401971871116E+00,
     1.350643881047676E+00,
     1.325024497958230E+00,
     1.298428035046913E+00,
     1.270707479650149E+00,
     1.241670567945823E+00,
     1.211056027568459E+00,
     1.178489924327839E+00,
     1.143395791883166E+00,
     1.104774732704073E+00,
     1.060473727766278E+00 };

  static double x_vec[N_MAX] = {
     0.0000000000000000, 
     0.2236067977499790, 
     0.3162277660168379, 
     0.3872983346207417, 
     0.4472135954999579, 
     0.5000000000000000, 
     0.5477225575051661, 
     0.5916079783099616, 
     0.6324555320336759, 
     0.6708203932499369, 
     0.7071067811865476, 
     0.7416198487095663, 
     0.7745966692414834, 
     0.8062257748298550, 
     0.8366600265340756, 
     0.8660254037844386, 
     0.8944271909999159, 
     0.9219544457292888, 
     0.9486832980505138, 
     0.9746794344808963 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    x = 0.0;
    fx = 0.0;
  }
  else
  {
    x = x_vec[n_data-1];
    fx = fx_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_em ( double m )

//****************************************************************************80

//
//  Purpose:
//
//    ELLIPTIC_EM evaluates the complete elliptic integral E(M).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      E(m) = RF ( 0, 1-m, 1 ) - 1/3 m RD ( 0, 1-m, 1 ).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double M, the argument.
//
//    Output, double ELLIPTIC_EM, the function value.
//
{
  double errtol;
  int ierr;
  double value;
  double x;
  double y;
  double z;

  x = 0.0;
  y = 1.0 - m;
  z = 1.0;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr ) 
    - m * rd ( x, y, z, errtol, ierr ) / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_em_values ( int &n_data, double &x, double &fx )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_EM_VALUES returns values of the complete elliptic integral E(M).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic
//    integral of the second kind.
//
//    The function is defined by the formula:
//
//      E(M) = integral ( 0 <= T <= PI/2 )
//        sqrt ( 1 - M * sin ( T )^2 ) dT
//
//    In Mathematica, the function can be evaluated by:
//
//      EllipticE[m]
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 August 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 before the
//    first call.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &X, the argument of the function.
//
//    Output, double &FX, the value of the function.
//
{
# define N_MAX 20

  static double fx_vec[N_MAX] = {
     1.570796326794897E+00,
     1.550973351780472E+00,
     1.530757636897763E+00,
     1.510121832092819E+00,
     1.489035058095853E+00,
     1.467462209339427E+00,
     1.445363064412665E+00,
     1.422691133490879E+00,
     1.399392138897432E+00,
     1.375401971871116E+00,
     1.350643881047676E+00,
     1.325024497958230E+00,
     1.298428035046913E+00,
     1.270707479650149E+00,
     1.241670567945823E+00,
     1.211056027568459E+00,
     1.178489924327839E+00,
     1.143395791883166E+00,
     1.104774732704073E+00,
     1.060473727766278E+00 };

  static double x_vec[N_MAX] = {
     0.00E+00,
     0.05E+00,
     0.10E+00,
     0.15E+00,
     0.20E+00,
     0.25E+00,
     0.30E+00,
     0.35E+00,
     0.40E+00,
     0.45E+00,
     0.50E+00,
     0.55E+00,
     0.60E+00,
     0.65E+00,
     0.70E+00,
     0.75E+00,
     0.80E+00,
     0.85E+00,
     0.90E+00,
     0.95E+00 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    x = 0.0;
    fx = 0.0;
  }
  else
  {
    x = x_vec[n_data-1];
    fx = fx_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_fa ( double a )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_FA evaluates the complete elliptic integral F(A).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      F(a) = RF ( 0, 1-sin^2(a), 1 ).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double A, the argument.
//
//    Output, double ELLIPTIC_FA, the function value.
//
{
  double errtol;
  int ierr;
  double k;
  const double r8_pi = 3.141592653589793;
  double value;
  double x;
  double y;
  double z;

  k = sin ( a * r8_pi / 180.0 );
  x = 0.0;
  y = ( 1.0 - k ) * ( 1.0 + k );
  z = 1.0;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr );

  return value;
}
//****************************************************************************80

void elliptic_fa_values ( int &n_data, double &x, double &fx )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_FA_VALUES returns values of the complete elliptic integral F(A).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic integral
//    of the first kind.
//
//    The function is defined by the formula:
//
//      F(A) = integral ( 0 <= T <= PI/2 )
//        dT / sqrt ( 1 - sin ( A )^2 * sin ( T )^2 )
//
//    In Mathematica, the function can be evaluated by:
//
//      EllipticK[(Sin[a*Pi/180])^2]
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 August 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 before the
//    first call.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &X, the argument of the function, measured
//    in degrees.
//
//    Output, double &FX, the value of the function.
//
{
# define N_MAX 18

  static double fx_vec[N_MAX] = {
     0.1570796326794897E+01,
     0.1573792130924768E+01,
     0.1582842804338351E+01,
     0.1598142002112540E+01,
     0.1620025899124204E+01,
     0.1648995218478530E+01,
     0.1685750354812596E+01,
     0.1731245175657058E+01,
     0.1786769134885021E+01,
     0.1854074677301372E+01,
     0.1935581096004722E+01,
     0.2034715312185791E+01,
     0.2156515647499643E+01,
     0.2308786798167196E+01,
     0.2504550079001634E+01,
     0.2768063145368768E+01,
     0.3153385251887839E+01,
     0.3831741999784146E+01 };

  static double x_vec[N_MAX] = {
      0.0E+00,
      5.0E+00,
     10.0E+00,
     15.0E+00,
     20.0E+00,
     25.0E+00,
     30.0E+00,
     35.0E+00,
     40.0E+00,
     45.0E+00,
     50.0E+00,
     55.0E+00,
     60.0E+00,
     65.0E+00,
     70.0E+00,
     75.0E+00,
     80.0E+00,
     85.0E+00 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    x = 0.0;
    fx = 0.0;
  }
  else
  {
    x = x_vec[n_data-1];
    fx = fx_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_fk ( double k )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_FK evaluates the complete elliptic integral F(K).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      F(k) = RF ( 0, 1-k^2, 1 ).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double K, the argument.
//
//    Output, double ELLIPTIC_FK, the function value.
//
{
  double errtol;
  int ierr;
  double value;
  double x;
  double y;
  double z;

  x = 0.0;
  y = ( 1.0 - k ) * ( 1.0 + k );
  z = 1.0;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr );

  return value;
}
//****************************************************************************80

void elliptic_fk_values ( int &n_data, double &x, double &fx )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_FK_VALUES returns values of the complete elliptic integral F(K).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic
//    integral of the first kind.
//
//    The function is defined by the formula:
//
//      F(K) = integral ( 0 <= T <= PI/2 )
//        dT / sqrt ( 1 - K^2 * sin ( T )^2 )
//
//    In Mathematica, the function can be evaluated by:
//
//      EllipticK[k^2]
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    29 May 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 before the
//    first call.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &X, the argument of the function.
//
//    Output, double &FX, the value of the function.
//
{
# define N_MAX 20

  static double fx_vec[N_MAX] = {
     1.570796326794897E+00,
     1.591003453790792E+00,
     1.612441348720219E+00,
     1.635256732264580E+00,
     1.659623598610528E+00,
     1.685750354812596E+00,
     1.713889448178791E+00,
     1.744350597225613E+00,
     1.777519371491253E+00,
     1.813883936816983E+00,
     1.854074677301372E+00,
     1.898924910271554E+00,
     1.949567749806026E+00,
     2.007598398424376E+00,
     2.075363135292469E+00,
     2.156515647499643E+00,
     2.257205326820854E+00,
     2.389016486325580E+00,
     2.578092113348173E+00,
     2.908337248444552E+00 };

  static double x_vec[N_MAX] = {
     0.0000000000000000E+00,
     0.2236067977499790E+00,
     0.3162277660168379E+00,
     0.3872983346207417E+00,
     0.4472135954999579E+00,
     0.5000000000000000E+00,
     0.5477225575051661E+00,
     0.5916079783099616E+00,
     0.6324555320336759E+00,
     0.6708203932499369E+00,
     0.7071067811865476E+00,
     0.7416198487095663E+00,
     0.7745966692414834E+00,
     0.8062257748298550E+00,
     0.8366600265340756E+00,
     0.8660254037844386E+00,
     0.8944271909999159E+00,
     0.9219544457292888E+00,
     0.9486832980505138E+00,
     0.9746794344808963E+00 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    x = 0.0;
    fx = 0.0;
  }
  else
  {
    x = x_vec[n_data-1];
    fx = fx_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_fm ( double m )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_FM evaluates the complete elliptic integral F(M).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      F(m) = RF ( 0, 1-m, 1 ).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double M, the argument.
//
//    Output, double ELLIPTIC_FM, the function value.
//
{
  double errtol;
  int ierr;
  double value;
  double x;
  double y;
  double z;

  x = 0.0;
  y = 1.0 - m;
  z = 1.0;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr );

  return value;
}
//****************************************************************************80

void elliptic_fm_values ( int &n_data, double &x, double &fx )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_FM_VALUES returns values of the complete elliptic integral F(M).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic
//    integral of the first kind.
//
//    The function is defined by the formula:
//
//      F(M) = integral ( 0 <= T <= PI/2 )
//        dT / sqrt ( 1 - M * sin ( T )^2 )
//
//    In Mathematica, the function can be evaluated by:
//
//      EllipticK[m]
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    10 August 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 before the
//    first call.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &X, the argument of the function.
//
//    Output, double &FX, the value of the function.
//
{
# define N_MAX 20

  static double fx_vec[N_MAX] = {
     1.570796326794897E+00,
     1.591003453790792E+00,
     1.612441348720219E+00,
     1.635256732264580E+00,
     1.659623598610528E+00,
     1.685750354812596E+00,
     1.713889448178791E+00,
     1.744350597225613E+00,
     1.777519371491253E+00,
     1.813883936816983E+00,
     1.854074677301372E+00,
     1.898924910271554E+00,
     1.949567749806026E+00,
     2.007598398424376E+00,
     2.075363135292469E+00,
     2.156515647499643E+00,
     2.257205326820854E+00,
     2.389016486325580E+00,
     2.578092113348173E+00,
     2.908337248444552E+00 };

  static double x_vec[N_MAX] = {
    0.00E+00,
    0.05E+00,
    0.10E+00,
    0.15E+00,
    0.20E+00,
    0.25E+00,
    0.30E+00,
    0.35E+00,
    0.40E+00,
    0.45E+00,
    0.50E+00,
    0.55E+00,
    0.60E+00,
    0.65E+00,
    0.70E+00,
    0.75E+00,
    0.80E+00,
    0.85E+00,
    0.90E+00,
    0.95E+00  };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    x = 0.0;
    fx = 0.0;
  }
  else
  {
    x = x_vec[n_data-1];
    fx = fx_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_inc_ea ( double phi, double a )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_EA evaluates the incomplete elliptic integral E(PHI,A).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      k = sin ( a * pi / 180 )
//      E(phi,a) = 
//                  sin ( phi )   RF ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 ) 
//        - 1/3 k^2 sin^3 ( phi ) RD ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 ).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double PHI, A, the arguments.
//    0 <= PHI <= PI/2.
//    0 <= sin^2 ( A * pi / 180 ) * sin^2(PHI) <= 1.
//
//    Output, double ELLIPTIC_INC_EA, the function value.
//
{
  double cp;
  double errtol;
  int ierr;
  double k;
  const double r8_pi = 3.141592653589793;
  double sp;
  double value;
  double value1;
  double value2;
  double x;
  double y;
  double z;

  k = sin ( a * r8_pi / 180.0 );

  cp = cos ( phi );
  sp = sin ( phi );
  x = cp * cp;
  y = ( 1.0 - k * sp ) * ( 1.0 + k * sp );
  z = 1.0;
  errtol = 1.0E-03;

  value1 = rf ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_EA - Fatal error!\n"; 
    cout << "  RF returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value2 = rd ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_EA - Fatal error!\n";
    cout << "  RD returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value = sp * value1 - k * k * sp * sp * sp * value2 / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_inc_ea_values ( int &n_data, double &phi, double &a, double &ea )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_EA_VALUES: values of the incomplete elliptic integral E(PHI,A).
//
//  Discussion:
//
//    This is one form of the incomplete elliptic integral of the second kind.
//
//      E(PHI,A) = integral ( 0 <= T <= PHI ) 
//        sqrt ( 1 - sin^2 ( A ) * sin^2 ( T ) ) dT
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    US Department of Commerce, 1964.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Wolfram Media / Cambridge University Press, 1999.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 
//    before the first call.  On each call, the routine increments N_DATA by 1, 
//    and returns the corresponding data when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &PHI, &A, the arguments of the function.
//
//    Output, double &EA, the value of the function.
//
{
# define N_MAX 20

  static double a_vec[N_MAX] = {
         123.0821233267548,
         11.26931745051486,
        -94.88806452075445,
        -99.71407853545323,
         57.05881039324191,
        -19.71363287074183,
         56.31230299738043,
        -91.55605346417718,
        -27.00654574696468,
        -169.2293728595904,
         61.96859564803047,
        -158.7324398933148,
         105.0883958999383,
        -48.95883872360177,
        -42.58568835110901,
         11.65603284687828,
        -8.398113719173338,
         17.69362213019626,
          73.8803420626852,
        -69.82492339645128 };

  static double ea_vec[N_MAX] = {
        0.3384181367348019,
         1.292924624509506,
        0.6074183768796306,
        0.3939726730783567,
       0.06880814097089803,
        0.0969436473376824,
        0.6025937791452033,
        0.9500549494837583,
         1.342783372140486,
        0.1484915631401388,
         1.085432887050926,
        0.1932136916085597,
        0.3983689593057807,
        0.1780054133336934,
         1.164525270273536,
         1.080167047541845,
         1.346684963830312,
         1.402100272685504,
        0.2928091845544553,
        0.5889342583405707 };

   static double phi_vec[N_MAX] = {
        0.3430906586047127,
         1.302990057703935,
        0.6523628380743488,
        0.4046022501376546,
       0.06884642871852312,
        0.0969609046794745,
         0.630370432896175,
         1.252375418911598,
         1.409796082144801,
        0.1485105463502483,
         1.349466184634646,
        0.1933711786970301,
        0.4088829927466769,
        0.1785430666405224,
         1.292588374416351,
         1.087095515757691,
         1.352794600489329,
         1.432530166308616,
        0.2968093345769761,
        0.6235880396594726 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  if ( N_MAX <= n_data )
  {
    n_data = 0;
    a = 0.0;
    ea = 0.0;
    phi = 0.0;
  }
  else
  {
    a = a_vec[n_data];
    ea = ea_vec[n_data];
    phi = phi_vec[n_data];
    n_data = n_data + 1;
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_inc_ek ( double phi, double k )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_EK evaluates the incomplete elliptic integral E(PHI,K).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      E(phi,k) = 
//                  sin ( phi )   RF ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 ) 
//        - 1/3 k^2 sin^3 ( phi ) RD ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 ).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double PHI, K, the arguments.
//    0 <= PHI <= PI/2.
//    0 <= K^2 * sin^2(PHI) <= 1.
//
//    Output, double ELLIPTIC_INC_EK, the function value.
//
{
  double cp;
  double errtol;
  int ierr;
  const double r8_pi = 3.141592653589793;
  double sp;
  double value;
  double value1;
  double value2;
  double x;
  double y;
  double z;

  cp = cos ( phi );
  sp = sin ( phi );
  x = cp * cp;
  y = ( 1.0 - k * sp ) * ( 1.0 + k * sp );
  z = 1.0;
  errtol = 1.0E-03;

  value1 = rf ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_EK - Fatal error!\n"; 
    cout << "  RF returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value2 = rd ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_EK - Fatal error!\n";
    cout << "  RD returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value = sp * value1 - k * k * sp * sp * sp * value2 / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_inc_ek_values ( int &n_data, double &phi, double &k, double &ek )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_EK_VALUES: values of the incomplete elliptic integral E(PHI,K).
//
//  Discussion:
//
//    This is the incomplete elliptic integral of the second kind.
//
//      E(PHI,K) = integral ( 0 <= T <= PHI ) 
//        sqrt ( 1 - K^2 * sin ( T )^2 ) dT
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    22 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    US Department of Commerce, 1964.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Wolfram Media / Cambridge University Press, 1999.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 
//    before the first call.  On each call, the routine increments N_DATA by 1, 
//    and returns the corresponding data when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &PHI, &K, the arguments.
//
//    Output, double &EK, the function value.
//
{
# define N_MAX 20

  static double ek_vec[N_MAX] = {
        0.2852345328295404,
         1.298690225567921,
        0.5508100202571943,
        0.3575401358115371,
       0.06801307805507453,
       0.09679584980231837,
        0.6003112504412838,
        0.8996717721794724,
         1.380715261453875,
        0.1191644625202453,
         1.196994838171557,
        0.1536260979667945,
        0.3546768920544152,
        0.1758756066650882,
         1.229819109410569,
          1.08381066114337,
          1.35023378157378,
         1.419775884709218,
        0.2824895528020034,
        0.5770427720982867 };

  static double k_vec[N_MAX] = {
         2.712952582080266,
        0.1279518954120547,
        -1.429437513650137,
        -1.981659235625333,
         3.894801879555818,
        -1.042486024983672,
        0.8641142168759754,
        -1.049058412826877,
       -0.3024062128402472,
        -6.574288841527263,
        0.6987397421988888,
         -5.12558591600033,
         2.074947853793764,
        -1.670886158426681,
       -0.4843595000931672,
        0.1393061679635559,
       -0.0946527302537008,
        0.1977207111754007,
         1.788159919089993,
        -1.077780624681256 };

  static double phi_vec[N_MAX] = {
        0.3430906586047127,
         1.302990057703935,
        0.6523628380743488,
        0.4046022501376546,
       0.06884642871852312,
        0.0969609046794745,
         0.630370432896175,
         1.252375418911598,
         1.409796082144801,
        0.1485105463502483,
         1.349466184634646,
        0.1933711786970301,
        0.4088829927466769,
        0.1785430666405224,
         1.292588374416351,
         1.087095515757691,
         1.352794600489329,
         1.432530166308616,
        0.2968093345769761,
        0.6235880396594726 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  if ( N_MAX <= n_data )
  {
    n_data = 0;
    ek = 0.0;
    k = 0.0;
    phi = 0.0;
  }
  else
  {
    ek = ek_vec[n_data];
    k = k_vec[n_data];
    phi = phi_vec[n_data];
    n_data = n_data + 1;
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_inc_em ( double phi, double m )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_EM evaluates the incomplete elliptic integral E(PHI,M).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      E(phi,m) = 
//                sin ( phi )   RF ( cos^2 ( phi ), 1-m sin^2 ( phi ), 1 ) 
//        - 1/3 m sin^3 ( phi ) RD ( cos^2 ( phi ), 1-m sin^2 ( phi ), 1 ).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double PHI, M, the arguments.
//    0 <= PHI <= PI/2.
//    0 <= M * sin^2(PHI) <= 1.
//
//    Output, double ELLIPTIC_INC_EM, the function value.
//
{
  double cp;
  double errtol;
  int ierr;
  const double r8_pi = 3.141592653589793;
  double sp;
  double value;
  double value1;
  double value2;
  double x;
  double y;
  double z;

  cp = cos ( phi );
  sp = sin ( phi );
  x = cp * cp;
  y = 1.0 - m * sp * sp;
  z = 1.0;
  errtol = 1.0E-03;

  value1 = rf ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_EM - Fatal error!\n"; 
    cout << "  RF returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value2 = rd ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_EM - Fatal error!\n";
    cout << "  RD returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value = sp * value1 - m * sp * sp * sp * value2 / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_inc_em_values ( int &n_data, double &phi, double &m, double &em )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_EM_VALUES: values of the incomplete elliptic integral E(PHI,M).
//
//  Discussion:
//
//    This is the incomplete elliptic integral of the second kind.
//
//      E(PHI,M) = integral ( 0 <= T <= PHI ) 
//        sqrt ( 1 - M * sin ( T )^2 ) dT
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    US Department of Commerce, 1964.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Wolfram Media / Cambridge University Press, 1999.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 
//    before the first call.  On each call, the routine increments N_DATA by 1, 
//    and returns the corresponding data when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &PHI, &M, the arguments.
//
//    Output, double &EM, the function value.
//
{
# define N_MAX 20

  static double em_vec[N_MAX] = {
        0.2732317284159052,
         1.124749725099781,
        0.6446601913679151,
        0.3968902354370061,
       0.06063960799944668,
       0.08909411577948728,
         0.532402014802015,
         1.251888640660265,
          1.28897116191626,
        0.1481718153599732,
         1.038090185639913,
        0.1931275771541276,
        0.3304419611986801,
         0.167394796063963,
         1.214501175324736,
        0.9516560179840655,
         1.203682959526176,
         1.206426326185419,
        0.2522791382096692,
        0.6026499038720986 };

  static double m_vec[N_MAX] = {
         8.450689756874594,
        0.6039878267930615,
        0.1794126658351454,
        0.7095689301026752,
         133.9643389059188,
         47.96621393936416,
         2.172070586163255,
      0.002038130569431913,
        0.3600036705339421,
        0.6219544540067304,
        0.8834215943508453,
        0.2034290670379481,
         5.772526076430922,
         11.14853902343298,
        0.2889238477277305,
        0.7166617182589116,
        0.4760623731559658,
        0.6094948502068943,
         8.902276887883076,
        0.5434439226321253 };

  static double phi_vec[N_MAX] = {
      0.3430906586047127, 
       1.302990057703935, 
      0.6523628380743488, 
      0.4046022501376546, 
     0.06884642871852312, 
      0.0969609046794745, 
       0.630370432896175, 
       1.252375418911598, 
       1.409796082144801, 
      0.1485105463502483, 
       1.349466184634646, 
      0.1933711786970301, 
      0.4088829927466769, 
      0.1785430666405224, 
       1.292588374416351, 
       1.087095515757691, 
       1.352794600489329, 
       1.432530166308616, 
      0.2968093345769761, 
      0.6235880396594726 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  if ( N_MAX <= n_data )
  {
    n_data = 0;
    em = 0.0;
    m = 0.0;
    phi = 0.0;
  }
  else
  {
    em = em_vec[n_data];
    m = m_vec[n_data];
    phi = phi_vec[n_data];
    n_data = n_data + 1;
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_inc_fa ( double phi, double a )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_FA evaluates the incomplete elliptic integral F(PHI,A).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      k = sin ( a * pi / 180 )
//      F(phi,k) = sin(phi) * RF ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double PHI, A, the arguments.
//    0 <= PHI <= PI/2.
//    0 <= sin^2 ( A * pi / 180 ) * sin^2(PHI) <= 1.
//
//    Output, double ELLIPTIC_INC_FA, the function value.
//
{
  double cp;
  double errtol;
  int ierr;
  double k;
  const double r8_pi = 3.141592653589793;
  double sp;
  double value;
  double x;
  double y;
  double z;

  k = sin ( a * r8_pi / 180.0 );

  cp = cos ( phi );
  sp = sin ( phi );
  x = cp * cp;
  y = ( 1.0 - k * sp ) * ( 1.0 + k * sp );
  z = 1.0;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_FA - Fatal error!\n"; 
    cout << "  RF returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value = sp * value;

  return value;
}
//****************************************************************************80

void elliptic_inc_fa_values ( int &n_data, double &phi, double &a, double &fa )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_FA_VALUES: values of the incomplete elliptic integral F(PHI,A).
//
//  Discussion:
//
//    This is the incomplete elliptic integral of the first kind.
//
//      F(PHI,A) = integral ( 0 <= T <= PHI ) 
//        dT / sqrt ( 1 - sin^2 ( A ) * sin^2 ( T ) )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    US Department of Commerce, 1964.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Wolfram Media / Cambridge University Press, 1999.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 
//    before the first call.  On each call, the routine increments N_DATA by 1, 
//    and returns the corresponding data when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &PHI, &A, the arguments.
//
//    Output, double &FA, the function value.
//
{
# define N_MAX 20

  static double a_vec[N_MAX] = {
         123.0821233267548,
         11.26931745051486,
        -94.88806452075445,
        -99.71407853545323,
         57.05881039324191,
        -19.71363287074183,
         56.31230299738043,
        -91.55605346417718,
        -27.00654574696468,
        -169.2293728595904,
         61.96859564803047,
        -158.7324398933148,
         105.0883958999383,
        -48.95883872360177,
        -42.58568835110901,
         11.65603284687828,
        -8.398113719173338,
         17.69362213019626,
          73.8803420626852,
        -69.82492339645128 };

  static double fa_vec[N_MAX] = {
        0.3478806460316299,
         1.313180577009584,
        0.7037956689264326,
        0.4157626844675118,
       0.06888475483285136,
       0.09697816754845832,
        0.6605394722518515,
          1.82758346036751,
         1.482258783392487,
        0.1485295339221232,
         1.753800062701494,
         0.193528896465351,
        0.4199100508706138,
        0.1790836490491233,
         1.446048832279763,
         1.094097652100984,
         1.358947908427035,
          1.46400078231538,
        0.3009092014525799,
        0.6621341112075102 };

  static double phi_vec[N_MAX] = {
        0.3430906586047127,
         1.302990057703935,
        0.6523628380743488,
        0.4046022501376546,
       0.06884642871852312,
        0.0969609046794745,
         0.630370432896175,
         1.252375418911598,
         1.409796082144801,
        0.1485105463502483,
         1.349466184634646,
        0.1933711786970301,
        0.4088829927466769,
        0.1785430666405224,
         1.292588374416351,
         1.087095515757691,
         1.352794600489329,
         1.432530166308616,
        0.2968093345769761,
        0.6235880396594726 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  if ( N_MAX <= n_data )
  {
    n_data = 0;
    a = 0.0;
    fa = 0.0;
    phi = 0.0;
  }
  else
  {
    a = a_vec[n_data];
    fa = fa_vec[n_data];
    phi = phi_vec[n_data];
    n_data = n_data + 1;
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_inc_fk ( double phi, double k )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_FK evaluates the incomplete elliptic integral F(PHI,K).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      F(phi,k) = sin(phi) * RF ( cos^2 ( phi ), 1-k^2 sin^2 ( phi ), 1 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double PHI, K, the arguments.
//    0 <= PHI <= PI/2.
//    0 <= K^2 * sin^2(PHI) <= 1.
//
//    Output, double ELLIPTIC_INC_FK, the function value.
//
{
  double cp;
  double errtol;
  int ierr;
  const double r8_pi = 3.141592653589793;
  double sp;
  double value;
  double x;
  double y;
  double z;

  cp = cos ( phi );
  sp = sin ( phi );
  x = cp * cp;
  y = ( 1.0 - k * sp ) * ( 1.0 + k * sp );
  z = 1.0;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_FK - Fatal error!\n"; 
    cout << "  RF returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value = sp * value;

  return value;
}
//****************************************************************************80

void elliptic_inc_fk_values ( int &n_data, double &phi, double &k, double &fk )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_FK_VALUES: values of the incomplete elliptic integral F(PHI,K).
//
//  Discussion:
//
//    This is the incomplete elliptic integral of the first kind.
//
//      F(PHI,K) = integral ( 0 <= T <= PHI ) 
//        dT / sqrt ( 1 - K^2 * sin ( T )^2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    US Department of Commerce, 1964.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Wolfram Media / Cambridge University Press, 1999.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 
//    before the first call.  On each call, the routine increments N_DATA by 1, 
//    and returns the corresponding data when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &PHI, &K, the arguments.
//
//    Output, double &FK, the value of the function.
//
{
# define N_MAX 20

  static double fk_vec[N_MAX] = {
       0.4340870330108736,
        1.307312511398114,
       0.8005154258533936,
       0.4656721451084328,
      0.06969849613441773,
      0.09712646708750489,
       0.6632598061016007,
          2.2308677858579,
        1.439846282888019,
       0.2043389243773096,
        1.537183574881771,
       0.2749229901565622,
       0.4828388342828284,
       0.1812848567886627,
        1.360729522341841,
         1.09039680912027,
        1.355363051581808,
        1.445462819732441,
       0.3125355489354676,
       0.6775731623807174 };

  static double k_vec[N_MAX] = {
        2.712952582080266,
       0.1279518954120547,
       -1.429437513650137,
       -1.981659235625333,
        3.894801879555818,
       -1.042486024983672,
       0.8641142168759754,
       -1.049058412826877,
      -0.3024062128402472,
       -6.574288841527263,
       0.6987397421988888,
        -5.12558591600033,
        2.074947853793764,
       -1.670886158426681,
      -0.4843595000931672,
       0.1393061679635559,
      -0.0946527302537008,
       0.1977207111754007,
        1.788159919089993,
       -1.077780624681256 };

  static double phi_vec[N_MAX] = {
       0.3430906586047127,
        1.302990057703935,
       0.6523628380743488,
       0.4046022501376546,
      0.06884642871852312,
       0.0969609046794745,
        0.630370432896175,
        1.252375418911598,
        1.409796082144801,
       0.1485105463502483,
        1.349466184634646,
       0.1933711786970301,
       0.4088829927466769,
       0.1785430666405224,
        1.292588374416351,
        1.087095515757691,
        1.352794600489329,
        1.432530166308616,
       0.2968093345769761,
       0.6235880396594726 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  if ( N_MAX <= n_data )
  {
    n_data = 0;
    fk = 0.0;
    k = 0.0;
    phi = 0.0;
  }
  else
  {
    fk = fk_vec[n_data];
    k = k_vec[n_data];
    phi = phi_vec[n_data];
    n_data = n_data + 1;
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_inc_fm ( double phi, double m )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_FM evaluates the incomplete elliptic integral F(PHI,M).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      F(phi,m) = sin(phi) * RF ( cos^2 ( phi ), 1-m sin^2 ( phi ), 1 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double PHI, M, the arguments.
//    0 <= PHI <= PI/2.
//    0 <= M * sin^2(PHI) <= 1.
//
//    Output, double ELLIPTIC_INC_FM, the function value.
//
{
  double cp;
  double errtol;
  int ierr;
  const double r8_pi = 3.141592653589793;
  double sp;
  double value;
  double x;
  double y;
  double z;

  cp = cos ( phi );
  sp = sin ( phi );
  x = cp * cp;
  y = 1.0 - m * sp * sp;
  z = 1.0;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_FM - Fatal error!\n"; 
    cout << "  RF returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value = sp * value;

  return value;
}
//****************************************************************************80

void elliptic_inc_fm_values ( int &n_data, double &phi, double &m, double &fm )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_FM_VALUES: values of the incomplete elliptic integral F(PHI,M).
//
//  Discussion:
//
//    This is the incomplete elliptic integral of the first kind.
//
//      F(PHI,M) = integral ( 0 <= T <= PHI ) 
//        dT / sqrt ( 1 - M * sin ( T )^2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    US Department of Commerce, 1964.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Wolfram Media / Cambridge University Press, 1999.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 
//    before the first call.  On each call, the routine increments N_DATA by 1, 
//    and returns the corresponding data when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &PHI, &M, the arguments.
//
//    Output, double &FM, the value of the function.
//
{
# define N_MAX 20

  static double fm_vec[N_MAX] = {
        0.4804314075855023,
         1.535634981092025,
        0.6602285297476601,
        0.4125884303785135,
       0.07964566007155376,
        0.1062834070535258,
        0.7733990864393913,
         1.252862499892228,
         1.549988686611532,
        0.1488506735822822,
         1.892229900799662,
        0.1936153327753556,
        0.5481932935424454,
        0.1911795073571756,
         1.379225069349756,
         1.261282453331402,
         1.535239838525378,
         1.739782418156071,
        0.3616930047198503,
        0.6458627645916422 };

  static double m_vec[N_MAX] = {
         8.450689756874594,
        0.6039878267930615,
        0.1794126658351454,
        0.7095689301026752,
         133.9643389059188,
         47.96621393936416,
         2.172070586163255,
      0.002038130569431913,
        0.3600036705339421,
        0.6219544540067304,
        0.8834215943508453,
        0.2034290670379481,
         5.772526076430922,
         11.14853902343298,
        0.2889238477277305,
        0.7166617182589116,
        0.4760623731559658,
        0.6094948502068943,
         8.902276887883076,
        0.5434439226321253 };

  static double phi_vec[N_MAX] = {
        0.3430906586047127,
         1.302990057703935,
        0.6523628380743488,
        0.4046022501376546,
       0.06884642871852312,
        0.0969609046794745,
         0.630370432896175,
         1.252375418911598,
         1.409796082144801,
        0.1485105463502483,
         1.349466184634646,
        0.1933711786970301,
        0.4088829927466769,
        0.1785430666405224,
         1.292588374416351,
         1.087095515757691,
         1.352794600489329,
         1.432530166308616,
        0.2968093345769761,
        0.6235880396594726 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  if ( N_MAX <= n_data )
  {
    n_data = 0;
    fm = 0.0;
    m = 0.0;
    phi = 0.0;
  }
  else
  {
    fm = fm_vec[n_data];
    m = m_vec[n_data];
    phi = phi_vec[n_data];
    n_data = n_data + 1;
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_inc_pia ( double phi, double n, double a )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_PIA evaluates the incomplete elliptic integral Pi(PHI,N,A).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      Pi(PHI,N,A) = integral ( 0 <= T <= PHI )
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - sin^2(A*pi/180) * sin ( T )^2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double PHI, N, A, the arguments.
//
//    Output, double ELLIPTIC_INC_PIA, the function value.
//
{
  double cp;
  double errtol;
  int ierr;
  double k;
  double p;
  const double r8_pi = 3.141592653589793;
  double sp;
  double value;
  double value1;
  double value2;
  double x;
  double y;
  double z;

  k = sin ( a * r8_pi / 180.0 );

  cp = cos ( phi );
  sp = sin ( phi );
  x = cp * cp;
  y = ( 1.0 - k * sp ) * ( 1.0 + k * sp );
  z = 1.0;
  p = 1.0 - n * sp * sp;
  errtol = 1.0E-03;

  value1 = rf ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_PIA - Fatal error!\n"; 
    cout << "  RF returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value2 = rj ( x, y, z, p, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_PIA - Fatal error!\n"; 
    cout << "  RJ returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value = sp * value1 + n * sp * sp * sp * value2 / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_inc_pia_values ( int &n_data, double &phi, double &n, double &a, 
  double &pia )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_PIA_VALUES: values of incomplete elliptic integral Pi(PHI,N,A).
//
//  Discussion:
//
//    This is the incomplete elliptic integral of the third kind.
//
//      Pi(PHI,N,A) = integral ( 0 <= T <= PHI ) 
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - sin^2(A) * sin ( T )^2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    US Department of Commerce, 1964.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Wolfram Media / Cambridge University Press, 1999.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 
//    before the first call.  On each call, the routine increments N_DATA by 1, 
//    and returns the corresponding data when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &PHI, &N, &A, the arguments of the function.
//
//    Output, double &PIA, the value of the function.
//
{
# define N_MAX 20

  static double a_vec[N_MAX] = {
         88.87822485052908,
        -86.55208740039521,
        -116.6195703112117,
        -9.742878017582015,
         65.73480919446207,
        -115.0387719677141,
         124.9421177735846,
        -89.78704401263703,
        -98.42673771271734,
        -53.74936192418378,
         68.28047574440727,
         20.82174673810708,
         -29.1042364797769,
        -37.80176710944693,
        -55.81173355852393,
        -37.66594589748672,
        -80.09408170610219,
         52.23806528467412,
         74.30945212430545,
        -17.22920703094039 };

  static double n_vec[N_MAX] = {
         8.064681366127422,
       -0.2840588974558835,
        -5.034023488967104,
        -1.244606253942751,
         1.465981775919188,
         95338.12857321106,
        -44.43130633436311,
       -0.8029374966926196,
         5.218883222649502,
         2.345821782626782,
         0.157358332363011,
         1.926593468907062,
         6.113982855261652,
         1.805710621498681,
       -0.4072847419780592,
       -0.9416404038595624,
        0.7009655305226739,
        -1.019830985340273,
       -0.4510798219577842,
        0.6028821390092596 };

  static double phi_vec[N_MAX] = {
        0.3430906586047127,
        0.8823091382756705,
        0.4046022501376546,
        0.9958310121985398,
         0.630370432896175,
      0.002887706662908567,
        0.1485105463502483,
         1.320800086884777,
        0.4088829927466769,
         0.552337007372852,
         1.087095515757691,
        0.7128175949111615,
        0.2968093345769761,
        0.2910907344062498,
        0.9695030752034163,
         1.122288759723523,
         1.295911610809573,
         1.116491437736542,
         1.170719322533712,
         1.199360682338851 };

  static double pia_vec[N_MAX] = {
        0.7099335174334724,
        0.9601963779142505,
        0.3362852532098376,
        0.7785343427543768,
         0.857889755214478,
      0.004630772344931844,
        0.1173842687902911,
         1.505788070660267,
        0.7213264194624553,
        0.8073261799642218,
         1.402853811110838,
         1.259245331474513,
        0.3779079263971614,
        0.3088493910496766,
        0.9782829177005183,
        0.9430491574504173,
         3.320796277384155,
        0.9730988737054799,
         1.301988094953789,
          1.64558360445259 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  if ( N_MAX <= n_data )
  {
    n_data = 0;
    a = 0.0;
    n = 0.0;
    phi = 0.0;
    pia = 0.0;
  }
  else
  {
    a = a_vec[n_data];
    n = n_vec[n_data];
    phi = phi_vec[n_data];
    pia = pia_vec[n_data];
    n_data = n_data + 1;
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_inc_pik ( double phi, double n, double k )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_PIK evaluates the incomplete elliptic integral Pi(PHI,N,K).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      Pi(PHI,N,K) = integral ( 0 <= T <= PHI )
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - k^2 * sin ( T )^2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double PHI, N, K, the arguments.
//
//    Output, double ELLIPTIC_INC_PIK, the function value.
//
{
  double cp;
  double errtol;
  int ierr;
  double p;
  const double r8_pi = 3.141592653589793;
  double sp;
  double value;
  double value1;
  double value2;
  double x;
  double y;
  double z;

  cp = cos ( phi );
  sp = sin ( phi );
  x = cp * cp;
  y = ( 1.0 - k * sp ) * ( 1.0 + k * sp );
  z = 1.0;
  p = 1.0 - n * sp * sp;
  errtol = 1.0E-03;

  value1 = rf ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_PIK - Fatal error!\n"; 
    cout << "  RF returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value2 = rj ( x, y, z, p, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_PIK - Fatal error!\n"; 
    cout << "  RJ returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value = sp * value1 + n * sp * sp * sp * value2 / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_inc_pik_values ( int &n_data, double &phi, double &n, double &k, 
  double &pik )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_PIK_VALUES: values of incomplete elliptic integral Pi(PHI,N,K).
//
//  Discussion:
//
//    This is the incomplete elliptic integral of the third kind.
//
//      Pi(PHI,N,K) = integral ( 0 <= T <= PHI ) 
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - K^2 * sin ( T )^2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    US Department of Commerce, 1964.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Wolfram Media / Cambridge University Press, 1999.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 
//    before the first call.  On each call, the routine increments N_DATA by 1, 
//    and returns the corresponding data when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &PHI, &N, &K, the arguments of the function.
//
//    Output, double &PIK, the value of the function.
//
{
# define N_MAX 20

  static double k_vec[N_MAX] = {
       1.959036804709882,
      -1.123741823223131,
      -2.317629084640271,
     -0.1202582658444815,
       1.008702896970963,
      -103.3677494756118,
       4.853800240677973,
      -1.016577251056124,
       -1.94341484065839,
     -0.8876593284500023,
      0.8160487832898813,
      0.2994546721661018,
     -0.7044232294525243,
     -0.9266523277404759,
     -0.6962608926846425,
     -0.4453932031991797,
     -0.9104582513322106,
      0.6187501419936026,
      0.8672305032589989,
     -0.1996772638241632 };

  static double n_vec[N_MAX] = {
       8.064681366127422,
     -0.2840588974558835,
      -5.034023488967104,
      -1.244606253942751,
       1.465981775919188,
       95338.12857321106,
      -44.43130633436311,
     -0.8029374966926196,
       5.218883222649502,
       2.345821782626782,
       0.157358332363011,
       1.926593468907062,
       6.113982855261652,
       1.805710621498681,
     -0.4072847419780592,
     -0.9416404038595624,
      0.7009655305226739,
      -1.019830985340273,
     -0.4510798219577842,
      0.6028821390092596 };

  static double phi_vec[N_MAX] = {
      0.3430906586047127,
      0.8823091382756705,
      0.4046022501376546,
      0.9958310121985398,
       0.630370432896175,
    0.002887706662908567,
      0.1485105463502483,
       1.320800086884777,
      0.4088829927466769,
       0.552337007372852,
       1.087095515757691,
      0.7128175949111615,
      0.2968093345769761,
      0.2910907344062498,
      0.9695030752034163,
       1.122288759723523,
       1.295911610809573,
       1.116491437736542,
       1.170719322533712,
       1.199360682338851 };

  static double pik_vec[N_MAX] = {
      0.7982975462595892,
       1.024022134726036,
        0.40158120852642,
      0.7772649487439858,
      0.8737159913132074,
    0.004733334297691273,
      0.1280656893638068,
       1.594376037512564,
      0.8521145133671923,
      0.8154325229803082,
        1.31594514075427,
        1.25394623148424,
      0.3796503567258643,
      0.3111034454739552,
      0.9442477901112342,
      0.9153111661980959,
       2.842080644328393,
      0.9263253777034376,
       1.212396018757624,
       1.628083572710471 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  if ( N_MAX <= n_data )
  {
    n_data = 0;
    k = 0.0;
    n = 0.0;
    phi = 0.0;
    pik = 0.0;
  }
  else
  {
    k = k_vec[n_data];
    n = n_vec[n_data];
    phi = phi_vec[n_data];
    pik = pik_vec[n_data];
    n_data = n_data + 1;
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_inc_pim ( double phi, double n, double m )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_PIM evaluates the incomplete elliptic integral Pi(PHI,N,M).
//
//  Discussion:
//
//    The value is computed using Carlson elliptic integrals:
//
//      Pi(PHI,N,M) = integral ( 0 <= T <= PHI )
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - m * sin ( T )^2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double PHI, N, M, the arguments.
//
//    Output, double ELLIPTIC_INC_PIM, the function value.
//
{
  double cp;
  double errtol;
  int ierr;
  double p;
  const double r8_pi = 3.141592653589793;
  double sp;
  double value;
  double value1;
  double value2;
  double x;
  double y;
  double z;

  cp = cos ( phi );
  sp = sin ( phi );
  x = cp * cp;
  y = 1.0 - m * sp * sp;
  z = 1.0;
  p = 1.0 - n * sp * sp;
  errtol = 1.0E-03;

  value1 = rf ( x, y, z, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_PIM - Fatal error!\n"; 
    cout << "  RF returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value2 = rj ( x, y, z, p, errtol, ierr );

  if ( ierr != 0 )
  {
    cout << "\n";
    cout << "ELLIPTIC_INC_PIM - Fatal error!\n"; 
    cout << "  RJ returned IERR = " << ierr << "\n";
    exit ( 1 );
  }

  value = sp * value1 + n * sp * sp * sp * value2 / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_inc_pim_values ( int &n_data, double &phi, double &n, double &m, 
  double &pim )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_INC_PIM_VALUES: values of incomplete elliptic integral Pi(PHI,N,M).
//
//  Discussion:
//
//    This is the incomplete elliptic integral of the third kind.
//
//      Pi(PHI,N,M) = integral ( 0 <= T <= PHI ) 
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - M * sin ( T )^2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    US Department of Commerce, 1964.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Wolfram Media / Cambridge University Press, 1999.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 
//    before the first call.  On each call, the routine increments N_DATA by 1, 
//    and returns the corresponding data when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &PHI, &N, &M, the arguments of the function.
//
//    Output, double &PIM, the value of the function.
//
{
# define N_MAX 20

  static double m_vec[N_MAX] = {
       7.330122710928245,
      0.1108806690614566,
      0.2828355944410993,
      0.6382999794812498,
       2.294718938593894,
       42062.55329826538,
        39.2394337789563,
    0.008002151065098688,
      0.7190579590867517,
      0.9703767630929055,
       1.098881295982823,
       1.398066725917478,
       4.641021931654496,
       4.455969064311461,
      0.3131448239736511,
      0.3686443684703166,
     0.06678210908100803,
      0.9635538974026796,
       1.060208762696207,
      0.4687160847955397 };

  static double n_vec[N_MAX] = {
       8.064681366127422,
     -0.2840588974558835,
      -5.034023488967104,
      -1.244606253942751,
       1.465981775919188,
       95338.12857321106,
      -44.43130633436311,
     -0.8029374966926196,
       5.218883222649502,
       2.345821782626782,
       0.157358332363011,
       1.926593468907062,
       6.113982855261652,
       1.805710621498681,
     -0.4072847419780592,
     -0.9416404038595624,
      0.7009655305226739,
      -1.019830985340273,
     -0.4510798219577842,
      0.6028821390092596 };

  static double phi_vec[N_MAX] = {
      0.3430906586047127,
      0.8823091382756705,
      0.4046022501376546,
      0.9958310121985398,
       0.630370432896175,
    0.002887706662908567,
      0.1485105463502483,
       1.320800086884777,
      0.4088829927466769,
       0.552337007372852,
       1.087095515757691,
      0.7128175949111615,
      0.2968093345769761,
      0.2910907344062498,
      0.9695030752034163,
       1.122288759723523,
       1.295911610809573,
       1.116491437736542,
       1.170719322533712,
       1.199360682338851 };

  static double pim_vec[N_MAX] = {
         1.0469349800785,
       0.842114448140669,
      0.3321642201520043,
      0.8483033529960849,
       1.055753817656772,
    0.005108896144265593,
      0.1426848042785896,
       1.031350958206424,
      0.7131013701418496,
      0.8268044665355507,
        1.57632867896015,
       1.542817120857211,
      0.4144629799126912,
      0.3313231611366746,
      0.9195822851915201,
      0.9422320754002217,
       2.036599002815859,
       1.076799231499882,
       1.416084462957852,
       1.824124922310891 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  if ( N_MAX <= n_data )
  {
    n_data = 0;
    m = 0.0;
    n = 0.0;
    phi = 0.0;
    pim = 0.0;
  }
  else
  {
    m = m_vec[n_data];
    n = n_vec[n_data];
    phi = phi_vec[n_data];
    pim = pim_vec[n_data];
    n_data = n_data + 1;
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_pia ( double n, double a )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_PIA evaluates the complete elliptic integral Pi(N,A).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic
//    integral of the third kind.
//
//    The double is defined by the formula:
//
//      Pi(N,A) = integral ( 0 <= T <= PI/2 )
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - sin^2(A) * sin ( T )^2 )
//
//    In MATLAB, the double can be evaluated by:
//
//      ellipticPi(n,(sin(a*pi/180)^2)
//
//    The value is computed using Carlson elliptic integrals:
//
//      k = sin ( a * pi / 180 )
//      Pi(n,k) = RF ( 0, 1 - k^2, 1 ) + 1/3 n RJ ( 0, 1 - k^2, 1, 1 - n )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double N, A, the arguments.
//
//    Output, double ELLIPTIC_PIA, the function value.
//
{
  double errtol;
  int ierr;
  double k;
  double p;
  const double r8_pi = 3.141592653589793;
  double value;
  double x;
  double y;
  double z;

  k = sin ( a * r8_pi / 180.0 );
  x = 0.0;
  y = ( 1.0 - k ) * ( 1.0 + k );
  z = 1.0;
  p = 1.0 - n;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr ) 
    + n * rj ( x, y, z, p, errtol, ierr ) / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_pia_values ( int &n_data, double &n, double &a, double &pia )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_PIA_VALUES returns values of the complete elliptic integral Pi(A).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic
//    integral of the third kind.
//
//    The function is defined by the formula:
//
//      Pi(N,A) = integral ( 0 <= T <= PI/2 )
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - sin^2(A) * sin ( T )^2 )
//
//    In MATLAB, the function can be evaluated by:
//
//      ellipticPi(n,(sin(A*pi/180))^2)
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    31 May 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 before the
//    first call.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &N, &A, the arguments.
//
//    Output, double &PIA, the value of the function.
//
{
# define N_MAX 20

  static double a_vec[N_MAX] = {
    30.00000000000000,
    45.00000000000000,
    60.00000000000000,
    77.07903361841643,
    30.00000000000000,
    45.00000000000000,
    60.00000000000000,
    77.07903361841643,
    30.00000000000000,
    45.00000000000000,
    60.00000000000000,
    77.07903361841643,
    30.00000000000000,
    45.00000000000000,
    60.00000000000000,
    77.07903361841643,
    30.00000000000000,
    45.00000000000000,
    60.00000000000000,
    77.07903361841643 };

  static double n_vec[N_MAX] = {
    -10.0,
    -10.0,
    -10.0,
    -10.0,
     -3.0,
     -3.0,
     -3.0,
     -3.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.5,
      0.5,
      0.5,
      0.5 };

  static double pia_vec[N_MAX] = {
    0.4892245275965397,
    0.5106765677902629,
    0.5460409271920561,
    0.6237325893535237,
    0.823045542660675,
    0.8760028274011437,
    0.9660073560143946,
    1.171952391481798,
    1.177446843000566,
    1.273127366749682,
    1.440034318657551,
    1.836472172302591,
    1.685750354812596,
    1.854074677301372,
    2.156515647499643,
    2.908337248444552,
    2.413671504201195,
    2.701287762095351,
    3.234773471249465,
    4.633308147279891 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    a = 0.0;
    n = 0.0;
    pia = 0.0;
  }
  else
  {
    a = a_vec[n_data-1];
    n = n_vec[n_data-1];
    pia = pia_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_pik ( double n, double k )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_PIK evaluates the complete elliptic integral Pi(N,K).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic
//    integral of the third kind.
//
//    The double is defined by the formula:
//
//      Pi(N,K) = integral ( 0 <= T <= PI/2 )
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - K^2 * sin ( T )^2 )
//
//    In MATLAB, the double can be evaluated by:
//
//      ellipticPi(n,k^2)
//
//    The value is computed using Carlson elliptic integrals:
//
//      Pi(n,k) = RF ( 0, 1 - k^2, 1 ) + 1/3 n RJ ( 0, 1 - k^2, 1, 1 - n )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double N, K, the arguments.
//
//    Output, double ELLIPTIC_PIK, the function value.
//
{
  double errtol;
  int ierr;
  double p;
  double value;
  double x;
  double y;
  double z;

  x = 0.0;
  y = ( 1.0 - k ) * ( 1.0 + k );
  z = 1.0;
  p = 1.0 - n;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr ) 
    + n * rj ( x, y, z, p, errtol, ierr ) / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_pik_values ( int &n_data, double &n, double &k, double &pik )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_PIK_VALUES returns values of the complete elliptic integral Pi(K).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic
//    integral of the third kind.
//
//    The function is defined by the formula:
//
//      Pi(N,K) = integral ( 0 <= T <= PI/2 )
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - K^2 * sin ( T )^2 )
//
//    In MATLAB, the function can be evaluated by:
//
//      ellipticPi(n,k^2)
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    31 May 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 before the
//    first call.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &N, &K, the arguments.
//
//    Output, double &PIK, the value of the function.
//
{
# define N_MAX 20

  static double k_vec[N_MAX] = {
    0.5000000000000000,
    0.7071067811865476,
    0.8660254037844386,
    0.9746794344808963,
    0.5000000000000000,
    0.7071067811865476,
    0.8660254037844386,
    0.9746794344808963,
    0.5000000000000000,
    0.7071067811865476,
    0.8660254037844386,
    0.9746794344808963,
    0.5000000000000000,
    0.7071067811865476,
    0.8660254037844386,
    0.9746794344808963,
    0.5000000000000000,
    0.7071067811865476,
    0.8660254037844386,
    0.9746794344808963 };

  static double n_vec[N_MAX] = {
    -10.0,
    -10.0,
    -10.0,
    -10.0,
     -3.0,
     -3.0,
     -3.0,
     -3.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.5,
      0.5,
      0.5,
      0.5 };

  static double pik_vec[N_MAX] = {
    0.4892245275965397,
    0.5106765677902629,
    0.5460409271920561,
    0.6237325893535237,
    0.823045542660675,
    0.8760028274011437,
    0.9660073560143946,
    1.171952391481798,
    1.177446843000566,
    1.273127366749682,
    1.440034318657551,
    1.836472172302591,
    1.685750354812596,
    1.854074677301372,
    2.156515647499643,
    2.908337248444552,
    2.413671504201195,
    2.701287762095351,
    3.234773471249465,
    4.633308147279891 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    k = 0.0;
    n = 0.0;
    pik = 0.0;
  }
  else
  {
    k = k_vec[n_data-1];
    n = n_vec[n_data-1];
    pik = pik_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double elliptic_pim ( double n, double m )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_PIM evaluates the complete elliptic integral Pi(N,M).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic
//    integral of the third kind.
//
//    The double is defined by the formula:
//
//      Pi(N,M) = integral ( 0 <= T <= PI/2 )
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - M * sin ( T )^2 )
//
//    In MATLAB, the double can be evaluated by:
//
//      ellipticPi(n,m)
//
//    The value is computed using Carlson elliptic integrals:
//
//      Pi(n,m) = RF ( 0, 1 - m, 1 ) + 1/3 n RJ ( 0, 1 - m, 1, 1 - n )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 June 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double N, M, the arguments.
//
//    Output, double ELLIPTIC_PIM, the function value.
//
{
  double errtol;
  int ierr;
  double p;
  double value;
  double x;
  double y;
  double z;

  x = 0.0;
  y = 1.0 - m;
  z = 1.0;
  p = 1.0 - n;
  errtol = 1.0E-03;

  value = rf ( x, y, z, errtol, ierr ) 
    + n * rj ( x, y, z, p, errtol, ierr ) / 3.0;

  return value;
}
//****************************************************************************80

void elliptic_pim_values ( int &n_data, double &n, double &m, double &pim )

//****************************************************************************80
//
//  Purpose:
//
//    ELLIPTIC_PIM_VALUES returns values of the complete elliptic integral Pi(M).
//
//  Discussion:
//
//    This is one form of what is sometimes called the complete elliptic
//    integral of the third kind.
//
//    The function is defined by the formula:
//
//      Pi(N,M) = integral ( 0 <= T <= PI/2 )
//        dT / (1 - N sin^2(T) ) sqrt ( 1 - M * sin ( T )^2 )
//
//    In MATLAB, the function can be evaluated by:
//
//      ellipticPi(n,m)
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    31 May 2018
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Parameters:
//
//    Input/output, int &N_DATA.  The user sets N_DATA to 0 before the
//    first call.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    Output, double &N, &M, the arguments.
//
//    Output, double &PIM, the value of the function.
//
{
# define N_MAX 20

  static double m_vec[N_MAX] = {
    0.25,
    0.50,
    0.75,
    0.95,
    0.25,
    0.50,
    0.75,
    0.95,
    0.25,
    0.50,
    0.75,
    0.95,
    0.25,
    0.50,
    0.75,
    0.95,
    0.25,
    0.50,
    0.75,
    0.95 };

  static double n_vec[N_MAX] = {
    -10.0,
    -10.0,
    -10.0,
    -10.0,
     -3.0,
     -3.0,
     -3.0,
     -3.0,
     -1.0,
     -1.0,
     -1.0,
     -1.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.5,
      0.5,
      0.5,
      0.5 };

  static double pim_vec[N_MAX] = {
    0.4892245275965397,
    0.5106765677902629,
    0.5460409271920561,
    0.6237325893535237,
    0.823045542660675,
    0.8760028274011437,
    0.9660073560143946,
    1.171952391481798,
    1.177446843000566,
    1.273127366749682,
    1.440034318657551,
    1.836472172302591,
    1.685750354812596,
    1.854074677301372,
    2.156515647499643,
    2.908337248444552,
    2.413671504201195,
    2.701287762095351,
    3.234773471249465,
    4.633308147279891 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    m = 0.0;
    n = 0.0;
    pim = 0.0;
  }
  else
  {
    m = m_vec[n_data-1];
    n = n_vec[n_data-1];
    pim = pim_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double jacobi_cn ( double u, double m )

//****************************************************************************80
//
//  Purpose:
//
//    JACOBI_CN evaluates the Jacobi elliptic function CN(U,M).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    26 June 2018
//
//  Author:
//
//    Original ALGOL version by Roland Bulirsch.
//    C++ version by John Burkardt
//
//  Reference:
//
//    Roland Bulirsch,
//    Numerical calculation of elliptic integrals and elliptic functions,
//    Numerische Mathematik,
//    Volume 7, Number 1, 1965, pages 78-90.
//
//  Parameters:
//
//    Input, double U, M, the arguments.
//
//    Output, double JACOBI_CN, the function value.
//
{
  double cn;
  double dn;
  double sn;

  sncndn ( u, m, sn, cn, dn );

  return cn;
}
//****************************************************************************80

void jacobi_cn_values ( int &n_data, double &u, double &a, double &k, 
  double &m, double &fx )

//****************************************************************************80
//
//  Purpose:
//
//    jacobi_cn_values returns values of the Jacobi elliptic function CN(U,M).
//
//  Discussion:
//
//    In Mathematica, the function can be evaluated by:
//
//      JacobiCN[ u, m ]
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 November 2020
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Input:
//
//    int &N_DATA.  The user sets N_DATA to 0 before the first call.  
//
//  Output:
//
//    int &N_DATA.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    double &U the argument of the function.
//
//    double &A, &K, &M, the parameters of the function.
//
//    double &FX, the value of the function.
//
{
# define N_MAX 20

  static double m_vec[N_MAX] = {
     0.0E+00,
     0.0E+00,
     0.0E+00,
     0.0E+00,
     0.0E+00,
     0.5E+00,
     0.5E+00,
     0.5E+00,
     0.5E+00,
     0.5E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00 };

  static double fx_vec[N_MAX] = {
      0.9950041652780258E+00,
      0.9800665778412416E+00,
      0.8775825618903727E+00,
      0.5403023058681397E+00,
     -0.4161468365471424E+00,
      0.9950124626090582E+00,
      0.9801976276784098E+00,
      0.8822663948904403E+00,
      0.5959765676721407E+00,
     -0.1031836155277618E+00,
      0.9950207489532265E+00,
      0.9803279976447253E+00,
      0.8868188839700739E+00,
      0.6480542736638854E+00,
      0.2658022288340797E+00,
      0.3661899347368653E-01,
      0.9803279976447253E+00,
      0.8868188839700739E+00,
      0.6480542736638854E+00,
      0.2658022288340797E+00 };

  static double u_vec[N_MAX] = {
      0.1E+00,
      0.2E+00,
      0.5E+00,
      1.0E+00,
      2.0E+00,
      0.1E+00,
      0.2E+00,
      0.5E+00,
      1.0E+00,
      2.0E+00,
      0.1E+00,
      0.2E+00,
      0.5E+00,
      1.0E+00,
      2.0E+00,
      4.0E+00,
     -0.2E+00,
     -0.5E+00,
     -1.0E+00,
     -2.0E+00 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    a = 0.0;
    k = 0.0;
    m = 0.0;
    u = 0.0;
    fx = 0.0;
  }
  else
  {
    m = m_vec[n_data-1];
    k = sqrt ( m );
    a = asin ( k );
    u = u_vec[n_data-1];
    fx = fx_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double jacobi_dn ( double u, double m )

//****************************************************************************80
//
//  Purpose:
//
//    JACOBI_DN evaluates the Jacobi elliptic function DN(U,M).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    26 June 2018
//
//  Author:
//
//    Original ALGOL version by Roland Bulirsch.
//    C++ version by John Burkardt
//
//  Reference:
//
//    Roland Bulirsch,
//    Numerical calculation of elliptic integrals and elliptic functions,
//    Numerische Mathematik,
//    Volume 7, Number 1, 1965, pages 78-90.
//
//  Parameters:
//
//    Input, double U, M, the arguments.
//
//    Output, double JACOBI_DN, the function value.
//
{
  double cn;
  double dn;
  double sn;

  sncndn ( u, m, sn, cn, dn );

  return dn;
}
//****************************************************************************80

void jacobi_dn_values ( int &n_data, double &u, double &a, double &k, 
  double &m, double &fx )

//****************************************************************************80
//
//  Purpose:
//
//    jacobi_dn_values returns values of the Jacobi elliptic function DN(U,M).
//
//  Discussion:
//
//    In Mathematica, the function can be evaluated by:
//
//      JacobiDN[ u, m ]
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 November 2020
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Input:
//
//    int &N_DATA.  The user sets N_DATA to 0 before the first call.  
//
//  Output:
//
//    int &N_DATA.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    double &U the argument of the function.
//
//    double &A, &K, &M, the parameters of the function.
//
//    double &FX, the value of the function.
//
{
# define N_MAX 20

  static double m_vec[N_MAX] = {
     0.0E+00,
     0.0E+00,
     0.0E+00,
     0.0E+00,
     0.0E+00,
     0.5E+00,
     0.5E+00,
     0.5E+00,
     0.5E+00,
     0.5E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00 };

  static double fx_vec[N_MAX] = {
     0.1000000000000000E+01,
     0.1000000000000000E+01,
     0.1000000000000000E+01,
     0.1000000000000000E+01,
     0.1000000000000000E+01,
     0.9975093485144243E+00,
     0.9901483195224800E+00,
     0.9429724257773857E+00,
     0.8231610016315963E+00,
     0.7108610477840873E+00,
     0.9950207489532265E+00,
     0.9803279976447253E+00,
     0.8868188839700739E+00,
     0.6480542736638854E+00,
     0.2658022288340797E+00,
     0.3661899347368653E-01,
     0.9803279976447253E+00,
     0.8868188839700739E+00,
     0.6480542736638854E+00,
     0.2658022288340797E+00  };

  static double u_vec[N_MAX] = {
      0.1E+00,
      0.2E+00,
      0.5E+00,
      1.0E+00,
      2.0E+00,
      0.1E+00,
      0.2E+00,
      0.5E+00,
      1.0E+00,
      2.0E+00,
      0.1E+00,
      0.2E+00,
      0.5E+00,
      1.0E+00,
      2.0E+00,
      4.0E+00,
     -0.2E+00,
     -0.5E+00,
     -1.0E+00,
     -2.0E+00 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    a = 0.0;
    k = 0.0;
    m = 0.0;
    u = 0.0;
    fx = 0.0;
  }
  else
  {
    m = m_vec[n_data-1];
    k = sqrt ( m );
    a = asin ( k );
    u = u_vec[n_data-1];
    fx = fx_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double jacobi_sn ( double u, double m )

//****************************************************************************80
//
//  Purpose:
//
//    JACOBI_SN evaluates the Jacobi elliptic function SN(U,M).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    26 June 2018
//
//  Author:
//
//    Original ALGOL version by Roland Bulirsch.
//    C++ version by John Burkardt
//
//  Reference:
//
//    Roland Bulirsch,
//    Numerical calculation of elliptic integrals and elliptic functions,
//    Numerische Mathematik,
//    Volume 7, Number 1, 1965, pages 78-90.
//
//  Parameters:
//
//    Input, double U, M, the arguments.
//
//    Output, double JACOBI_SN, the function value.
//
{
  double cn;
  double dn;
  double sn;

  sncndn ( u, m, sn, cn, dn );

  return sn;
}
//****************************************************************************80

void jacobi_sn_values ( int &n_data, double &u, double &a, double &k, 
  double &m, double &fx )

//****************************************************************************80
//
//  Purpose:
//
//    jacobi_sn_values returns values of the Jacobi elliptic function SN(U,M).
//
//  Discussion:
//
//    In Mathematica, the function can be evaluated by:
//
//      JacobiSN[ u, m ]
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 November 2020
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Milton Abramowitz, Irene Stegun,
//    Handbook of Mathematical Functions,
//    National Bureau of Standards, 1964,
//    ISBN: 0-486-61272-4,
//    LC: QA47.A34.
//
//    Stephen Wolfram,
//    The Mathematica Book,
//    Fourth Edition,
//    Cambridge University Press, 1999,
//    ISBN: 0-521-64314-7,
//    LC: QA76.95.W65.
//
//  Input:
//
//    int &N_DATA.  The user sets N_DATA to 0 before the first call.  
//
//  Output:
//
//    int &N_DATA.  On each call, the routine increments N_DATA by 1, and
//    returns the corresponding data; when there is no more data, the
//    output value of N_DATA will be 0 again.
//
//    double &U the argument of the function.
//
//    double &A, &K, &M, the parameters of the function.
//
//    double &FX, the value of the function.
//
{
# define N_MAX 20

  static double m_vec[N_MAX] = {
     0.0E+00,
     0.0E+00,
     0.0E+00,
     0.0E+00,
     0.0E+00,
     0.5E+00,
     0.5E+00,
     0.5E+00,
     0.5E+00,
     0.5E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00,
     1.0E+00 };

  static double fx_vec[N_MAX] = {
      0.9983341664682815E-01,
      0.1986693307950612E+00,
      0.4794255386042030E+00,
      0.8414709848078965E+00,
      0.9092974268256817E+00,
      0.9975068547462484E-01,
      0.1980217429819704E+00,
      0.4707504736556573E+00,
      0.8030018248956439E+00,
      0.9946623253580177E+00,
      0.9966799462495582E-01,
      0.1973753202249040E+00,
      0.4621171572600098E+00,
      0.7615941559557649E+00,
      0.9640275800758169E+00,
      0.9993292997390670E+00,
     -0.1973753202249040E+00,
     -0.4621171572600098E+00,
     -0.7615941559557649E+00,
     -0.9640275800758169E+00  };

  static double u_vec[N_MAX] = {
      0.1E+00,
      0.2E+00,
      0.5E+00,
      1.0E+00,
      2.0E+00,
      0.1E+00,
      0.2E+00,
      0.5E+00,
      1.0E+00,
      2.0E+00,
      0.1E+00,
      0.2E+00,
      0.5E+00,
      1.0E+00,
      2.0E+00,
      4.0E+00,
     -0.2E+00,
     -0.5E+00,
     -1.0E+00,
     -2.0E+00 };

  if ( n_data < 0 )
  {
    n_data = 0;
  }

  n_data = n_data + 1;

  if ( N_MAX < n_data )
  {
    n_data = 0;
    a = 0.0;
    k = 0.0;
    m = 0.0;
    u = 0.0;
    fx = 0.0;
  }
  else
  {
    m = m_vec[n_data-1];
    k = sqrt ( m );
    a = asin ( k );
    u = u_vec[n_data-1];
    fx = fx_vec[n_data-1];
  }

  return;
# undef N_MAX
}
//****************************************************************************80

double rc ( double x, double y, double errtol, int &ierr )

//****************************************************************************80
//
//  Purpose:
//
//    RC computes the elementary integral RC(X,Y).
//
//   Discussion:
//
//     This function computes the elementary integral
//
//       RC(X,Y) = Integral ( 0 <= T < oo )
//
//                   -1/2     -1
//         (1/2)(T+X)    (T+Y)  DT,
//
//     where X is nonnegative and Y is positive.  The duplication
//     theorem is iterated until the variables are nearly equal,
//     and the function is then expanded in Taylor series to fifth
//     order.
//
//     Logarithmic, inverse circular, and inverse hyperbolic
//     functions can be expressed in terms of RC.
//
//     Check by addition theorem:
//
//       RC(X,X+Z) + RC(Y,Y+Z) = RC(0,Z),
//       where X, Y, and Z are positive and X * Y = Z * Z.
//
//   Licensing:
//
//     This code is distributed under the GNU LGPL license.
//
//   Modified:
//
//     02 June 2018
//
//   Author:
//
//     Original FORTRAN77 version by Bille Carlson, Elaine Notis.
//     This C++ version by John Burkardt.
//
//   Reference:
//
//     Bille Carlson,
//     Computing Elliptic Integrals by Duplication,
//     Numerische Mathematik,
//     Volume 33, 1979, pages 1-16.
//
//     Bille Carlson, Elaine Notis,
//     Algorithm 577, Algorithms for Incomplete Elliptic Integrals,
//     ACM Transactions on Mathematical Software,
//     Volume 7, Number 3, pages 398-403, September 1981.
//
//   Parameters:
//
//     Input, double X, Y, the arguments in the integral.
//
//     Input, double ERRTOL, the error tolerance.
//     Relative error due to truncation is less than
//       16 * ERRTOL ^ 6 / (1 - 2 * ERRTOL).
//     Sample choices:
//       ERRTOL   Relative truncation error less than
//       1.D-3    2.D-17
//       3.D-3    2.D-14
//       1.D-2    2.D-11
//       3.D-2    2.D-8
//       1.D-1    2.D-5
//
//     Output, int &IERR, the error flag.
//     0, no error occurred.
//     1, abnormal termination.
//
{
  double c1;
  double c2;
  double lamda;
  const double lolim = 3.0E-78;
  double mu;
  double s;
  double sn;
  const double uplim = 1.0E+75;
  double value;
  double xn;
  double yn;
//
//   LOLIM and UPLIM determine the range of valid arguments.
//   LOLIM IS NOT LESS THAN THE MACHINE MINIMUM MULTIPLIED BY 5.
//   UPLIM IS NOT GREATER THAN THE MACHINE MAXIMUM DIVIDED BY 5.
//
  if ( 
    x < 0.0 || 
    y <= 0.0 || 
    ( x + y ) < lolim || 
    uplim < x || 
    uplim < y )
  {
    cout << "\n";
    cout << "RC - Error!\n";
    cout << "  Invalid input arguments.\n";
    cout << "  X = " << x << "\n";
    cout << "  Y = " << y << "\n";
    cout << "\n";
    ierr = 1;
    value = 0.0;
    return value;
  }

  ierr = 0;
  xn = x;
  yn = y;

  while ( true )
  {
    mu = ( xn + yn + yn ) / 3.0;
    sn = ( yn + mu ) / mu - 2.0;

    if ( fabs ( sn ) < errtol )
    {
      c1 = 1.0 / 7.0;
      c2 = 9.0 / 22.0;
      s = sn * sn * ( 0.3 + sn * ( c1 + sn * ( 0.375 + sn * c2 ) ) );
      value = ( 1.0 + s ) / sqrt ( mu );
      return value;
    }

    lamda = 2.0 * sqrt ( xn ) * sqrt ( yn ) + yn;
    xn = ( xn + lamda ) * 0.25;
    yn = ( yn + lamda ) * 0.25;
  }

}
//****************************************************************************80

double rd ( double x, double y, double z, double errtol, int &ierr )

//****************************************************************************80
//
//  Purpose:
//
//    RD computes an incomplete elliptic integral of the second kind, RD(X,Y,Z).
//
//   Discussion:
//
//     This function computes an incomplete elliptic integral of the second kind.
//
//     RD(X,Y,Z) = Integral ( 0 <= T < oo )
//
//                     -1/2     -1/2     -3/2
//           (3/2)(T+X)    (T+Y)    (T+Z)    DT,
//
//     where X and Y are nonnegative, X + Y is positive, and Z is positive.
//
//     If X or Y is zero, the integral is complete.
//
//     The duplication theorem is iterated until the variables are
//     nearly equal, and the function is then expanded in Taylor
//     series to fifth order.
//
//     Check:
//
//       RD(X,Y,Z) + RD(Y,Z,X) + RD(Z,X,Y) = 3 / sqrt ( X * Y * Z ),
//       where X, Y, and Z are positive.
//
//   Licensing:
//
//     This code is distributed under the GNU LGPL license.
//
//   Modified:
//
//     02 June 2018
//
//   Author:
//
//     Original FORTRAN77 version by Bille Carlson, Elaine Notis.
//     This C++ version by John Burkardt.
//
//   Reference:
//
//     Bille Carlson,
//     Computing Elliptic Integrals by Duplication,
//     Numerische Mathematik,
//     Volume 33, 1979, pages 1-16.
//
//     Bille Carlson, Elaine Notis,
//     Algorithm 577, Algorithms for Incomplete Elliptic Integrals,
//     ACM Transactions on Mathematical Software,
//     Volume 7, Number 3, pages 398-403, September 1981.
//
//   Parameters:
//
//     Input, double X, Y, Z, the arguments in the integral.
//
//     Input, double ERRTOL, the error tolerance.
//     The relative error due to truncation is less than
//       3 * ERRTOL ^ 6 / (1-ERRTOL) ^ 3/2.
//     Sample choices:
//       ERRTOL   Relative truncation error less than
//       1.D-3    4.D-18
//       3.D-3    3.D-15
//       1.D-2    4.D-12
//       3.D-2    3.D-9
//       1.D-1    4.D-6
//
//     Output, int &IERR, the error flag.
//     0, no error occurred.
//     1, abnormal termination.
//
{
  double c1;
  double c2;
  double c3;
  double c4;
  double ea;
  double eb;
  double ec;
  double ed;
  double ef;
  double epslon;
  double lamda;
  const double lolim = 6.0E-51;
  double mu;
  double power4;
  double sigma;
  double s1;
  double s2;
  const double uplim = 1.0E+48;
  double value;
  double xn;
  double xndev;
  double xnroot;
  double yn;
  double yndev;
  double ynroot;
  double zn;
  double zndev;
  double znroot;
//
//   LOLIM and UPLIM determine the range of valid arguments.
//   LOLIM IS NOT LESS THAN 2 / (MACHINE MAXIMUM) ^ (2/3).
//   UPLIM IS NOT GREATER THAN (0.1 * ERRTOL / MACHINE
//   MINIMUM) ^ (2/3), WHERE ERRTOL IS DESCRIBED BELOW.
//   IN THE FOLLOWING TABLE IT IS ASSUMED THAT ERRTOL WILL
//   NEVER BE CHOSEN SMALLER THAN 1.D-5.
//
  if ( 
    x < 0.0 || 
    y < 0.0 || 
    x + y < lolim || 
    z < lolim || 
    uplim < x || 
    uplim < y || 
    uplim < z )
  {
    cout << "\n";
    cout << "RD - Error!\n";
    cout << "  Invalid input arguments.\n";
    cout << "  X = " << x << "\n";
    cout << "  Y = " << y << "\n";
    cout << "  Z = " << z << "\n";
    cout << "\n";
    ierr = 1;
    value = 0.0;
    return value;
  }

  ierr = 0;
  xn = x;
  yn = y;
  zn = z;
  sigma = 0.0;
  power4 = 1.0;

  while ( true )
  {
    mu = ( xn + yn + 3.0 * zn ) * 0.2;
    xndev = ( mu - xn ) / mu;
    yndev = ( mu - yn ) / mu;
    zndev = ( mu - zn ) / mu;
    epslon = fmax ( fabs ( xndev ), 
      fmax ( fabs ( yndev ), fabs ( zndev ) ) );

    if ( epslon < errtol )
    {
      c1 = 3.0 / 14.0;
      c2 = 1.0 / 6.0;
      c3 = 9.0 / 22.0;
      c4 = 3.0 / 26.0;
      ea = xndev * yndev;
      eb = zndev * zndev;
      ec = ea - eb;
      ed = ea - 6.0 * eb;
      ef = ed + ec + ec;
      s1 = ed * ( - c1 + 0.25 * c3 * ed - 1.5 * c4 * zndev * ef );
      s2 = zndev  * ( c2 * ef + zndev * ( - c3 * ec + zndev * c4 * ea ) );
      value = 3.0 * sigma  + power4 * ( 1.0 + s1 + s2 ) / ( mu * sqrt ( mu ) );
      return value;
    }

    xnroot = sqrt ( xn );
    ynroot = sqrt ( yn );
    znroot = sqrt ( zn );
    lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot;
    sigma = sigma + power4 / ( znroot * ( zn + lamda ) );
    power4 = power4 * 0.25;
    xn = ( xn + lamda ) * 0.25;
    yn = ( yn + lamda ) * 0.25;
    zn = ( zn + lamda ) * 0.25;
  }

}
//****************************************************************************80

double rf ( double x, double y, double z, double errtol, int &ierr )

//****************************************************************************80
//
//  Purpose:
//
//    RF computes an incomplete elliptic integral of the first kind, RF(X,Y,Z).
//
//   Discussion:
//
//     This function computes the incomplete elliptic integral of the first kind.
//
//     RF(X,Y,Z) = Integral ( 0 <= T < oo )
//
//                     -1/2     -1/2     -1/2
//           (1/2)(T+X)    (T+Y)    (T+Z)    DT,
//
//     where X, Y, and Z are nonnegative and at most one of them is zero.
//
//     If X or Y or Z is zero, the integral is complete.
//
//     The duplication theorem is iterated until the variables are
//     nearly equal, and the function is then expanded in Taylor
//     series to fifth order.
//
//     Check by addition theorem:
//
//       RF(X,X+Z,X+W) + RF(Y,Y+Z,Y+W) = RF(0,Z,W),
//       where X, Y, Z, W are positive and X * Y = Z * W.
//
//   Licensing:
//
//     This code is distributed under the GNU LGPL license.
//
//   Modified:
//
//     02 June 2018
//
//   Author:
//
//     Original FORTRAN77 version by Bille Carlson, Elaine Notis.
//     This C++ version by John Burkardt.
//
//   Reference:
//
//     Bille Carlson,
//     Computing Elliptic Integrals by Duplication,
//     Numerische Mathematik,
//     Volume 33, 1979, pages 1-16.
//
//     Bille Carlson, Elaine Notis,
//     Algorithm 577, Algorithms for Incomplete Elliptic Integrals,
//     ACM Transactions on Mathematical Software,
//     Volume 7, Number 3, pages 398-403, September 1981.
//
//   Parameters:
//
//     Input, double X, Y, Z, the arguments in the integral.
//
//     Input, double ERRTOL, the error tolerance.
//     Relative error due to truncation is less than
//       ERRTOL ^ 6 / (4 * (1 - ERRTOL)).
//     Sample choices:
//       ERRTOL   Relative truncation error less than
//       1.D-3    3.D-19
//       3.D-3    2.D-16
//       1.D-2    3.D-13
//       3.D-2    2.D-10
//       1.D-1    3.D-7
//
//     Output, int &IERR, the error flag.
//     0, no error occurred.
//     1, abnormal termination.
//
{
  double c1;
  double c2;
  double c3;
  double e2;
  double e3;
  double epslon;
  double lamda;
  const double lolim = 3.0E-78;
  double mu;
  double s;
  const double uplim = 1.0E+75;
  double value;
  double xn;
  double xndev;
  double xnroot;
  double yn;
  double yndev;
  double ynroot;
  double zn;
  double zndev;
  double znroot;
//
//   LOLIM and UPLIM determine the range of valid arguments.
//   LOLIM IS NOT LESS THAN THE MACHINE MINIMUM MULTIPLIED BY 5.
//   UPLIM IS NOT GREATER THAN THE MACHINE MAXIMUM DIVIDED BY 5.
//
  if ( 
    x < 0.0 || 
    y < 0.0 || 
    z < 0.0 || 
    x + y < lolim || 
    x + z < lolim || 
    y + z < lolim || 
    uplim <= x || 
    uplim <= y || 
    uplim <= z )
  {
    cout << "\n";
    cout << "RF - Error!\n";
    cout << "  Invalid input arguments.\n";
    cout << "  X = " << x << "\n";
    cout << "  Y = " << y << "\n";
    cout << "  Z = " << z << "\n";
    cout << "\n";
    ierr = 1;
    value = 0.0;
    return value;
  }

  ierr = 0;
  xn = x;
  yn = y;
  zn = z;

  while ( true )
  {
    mu = ( xn + yn + zn ) / 3.0;
    xndev = 2.0 - ( mu + xn ) / mu;
    yndev = 2.0 - ( mu + yn ) / mu;
    zndev = 2.0 - ( mu + zn ) / mu;
    epslon = fmax ( fabs ( xndev ), 
      fmax ( fabs ( yndev ), fabs ( zndev ) ) );

    if ( epslon < errtol )
    {
      c1 = 1.0 / 24.0;
      c2 = 3.0 / 44.0;
      c3 = 1.0 / 14.0;
      e2 = xndev * yndev - zndev * zndev;
      e3 = xndev * yndev * zndev;
      s = 1.0 + ( c1 * e2 - 0.1 - c2 * e3 ) * e2 + c3 * e3;
      value = s / sqrt ( mu );
      return value;
    }

    xnroot = sqrt ( xn );
    ynroot = sqrt ( yn );
    znroot = sqrt ( zn );
    lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot;
    xn = ( xn + lamda ) * 0.25;
    yn = ( yn + lamda ) * 0.25;
    zn = ( zn + lamda ) * 0.25;
  }

}
//****************************************************************************80

double rj ( double x, double y, double z, double p, double errtol, int &ierr )

//****************************************************************************80
//
//  Purpose:
//
//    RJ computes an incomplete elliptic integral of the third kind, RJ(X,Y,Z,P).
//
//   Discussion:
//
//     This function computes an incomplete elliptic integral of the third kind.
//
//     RJ(X,Y,Z,P) = Integral ( 0 <= T < oo )
//
//                   -1/2     -1/2     -1/2     -1
//         (3/2)(T+X)    (T+Y)    (T+Z)    (T+P)  DT,
//
//     where X, Y, and Z are nonnegative, at most one of them is
//     zero, and P is positive.
//
//     If X or Y or Z is zero, then the integral is complete.
//
//     The duplication theorem is iterated until the variables are nearly equal,
//     and the function is then expanded in Taylor series to fifth order.
//
//     Check by addition theorem:
//
//       RJ(X,X+Z,X+W,X+P)
//       + RJ(Y,Y+Z,Y+W,Y+P) + (A-B) * RJ(A,B,B,A) + 3 / sqrt ( A)
//       = RJ(0,Z,W,P), where X,Y,Z,W,P are positive and X * Y
//       = Z * W,  A = P * P * (X+Y+Z+W),  B = P * (P+X) * (P+Y),
//       and B - A = P * (P-Z) * (P-W).
//
//     The sum of the third and fourth terms on the left side is 3 * RC(A,B).
//
//   Licensing:
//
//     This code is distributed under the GNU LGPL license.
//
//   Modified:
//
//     02 June 2018
//
//   Author:
//
//     Original FORTRAN77 version by Bille Carlson, Elaine Notis.
//     This C++ version by John Burkardt.
//
//   Reference:
//
//     Bille Carlson,
//     Computing Elliptic Integrals by Duplication,
//     Numerische Mathematik,
//     Volume 33, 1979, pages 1-16.
//
//     Bille Carlson, Elaine Notis,
//     Algorithm 577, Algorithms for Incomplete Elliptic Integrals,
//     ACM Transactions on Mathematical Software,
//     Volume 7, Number 3, pages 398-403, September 1981.
//
//   Parameters:
//
//     Input, double X, Y, Z, P, the arguments in the integral.
//
//     Input, double ERRTOL, the error tolerance.
//     Relative error due to truncation of the series for rj
//     is less than 3 * ERRTOL ^ 6 / (1 - ERRTOL) ^ 3/2.
//     An error tolerance (ETOLRC) will be passed to the subroutine
//     for RC to make the truncation error for RC less than for RJ.
//     Sample choices:
//       ERRTOL   Relative truncation error less than
//       1.D-3    4.D-18
//       3.D-3    3.D-15
//       1.D-2    4.D-12
//       3.D-2    3.D-9
//       1.D-1    4.D-6
//
//     Output, int &IERR, the error flag.
//     0, no error occurred.
//     1, abnormal termination.
//
{
  double alfa;
  double beta;
  double c1;
  double c2;
  double c3;
  double c4;
  double ea;
  double eb;
  double ec;
  double e2;
  double e3;
  double epslon;
  double etolrc;
  double lamda;
  const double lolim = 2.0E-26;
  double mu;
  double pn;
  double pndev;
  double power4;
  double sigma;
  double s1;
  double s2;
  double s3;
  const double uplim = 3.0E+24;
  double value;
  double xn;
  double xndev;
  double xnroot;
  double yn;
  double yndev;
  double ynroot;
  double zn;
  double zndev;
  double znroot;
//
//   LOLIM and UPLIM determine the range of valid arguments.
//   LOLIM IS NOT LESS THAN THE CUBE ROOT OF THE VALUE
//   OF LOLIM USED IN THE SUBROUTINE FOR RC.
//   UPLIM IS NOT GREATER THAN 0.3 TIMES THE CUBE ROOT OF
//   THE VALUE OF UPLIM USED IN THE SUBROUTINE FOR RC.
//
  if ( 
    x < 0.0 || 
    y < 0.0 || 
    z < 0.0 || 
    x + y < lolim || 
    x + z < lolim || 
    y + z < lolim || 
    p < lolim || 
    uplim < x || 
    uplim < y || 
    uplim < z || 
    uplim < p )
  {
    cout << "\n";
    cout << "RJ - Error!\n";
    cout << "  Invalid input arguments.\n";
    cout << "  X = " << x << "\n";
    cout << "  Y = " << y << "\n";
    cout << "  Z = " << z << "\n";
    cout << "  P = " << p << "\n";
    cout << "\n";
    ierr = 1;
    value = 0.0;
    return value;
  }

  ierr = 0;
  xn = x;
  yn = y;
  zn = z;
  pn = p;
  sigma = 0.0;
  power4 = 1.0;
  etolrc = 0.5 * errtol;

  while ( true )
  {
    mu = ( xn + yn + zn + pn + pn ) * 0.2;
    xndev = ( mu - xn ) / mu;
    yndev = ( mu - yn ) / mu;
    zndev = ( mu - zn ) / mu;
    pndev = ( mu - pn ) / mu;
    epslon = fmax ( fabs ( xndev ), 
      fmax ( fabs ( yndev ), 
      fmax ( fabs ( zndev ), fabs ( pndev ) ) ) );

    if ( epslon < errtol )
    {
      c1 = 3.0 / 14.0;
      c2 = 1.0 / 3.0;
      c3 = 3.0 / 22.0;
      c4 = 3.0 / 26.0;
      ea = xndev * ( yndev + zndev ) + yndev * zndev;
      eb = xndev * yndev * zndev;
      ec = pndev * pndev;
      e2 = ea - 3.0 * ec;
      e3 = eb + 2.0 * pndev * ( ea - ec );
      s1 = 1.0 + e2 * ( - c1 + 0.75 * c3 * e2 - 1.5 * c4 * e3 );
      s2 = eb * ( 0.5 * c2 + pndev * ( - c3 - c3 + pndev * c4 ) );
      s3 = pndev * ea * ( c2 - pndev * c3 ) - c2 * pndev * ec;
      value = 3.0 * sigma + power4 * ( s1 + s2 + s3 ) / ( mu * sqrt ( mu ) );
      return value;
    }

    xnroot = sqrt ( xn );
    ynroot = sqrt ( yn );
    znroot = sqrt ( zn );
    lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot;
    alfa = pn * ( xnroot + ynroot + znroot ) + xnroot * ynroot * znroot;
    alfa = alfa * alfa;
    beta = pn * ( pn + lamda ) * ( pn + lamda );
    sigma = sigma + power4 * rc ( alfa, beta, etolrc, ierr );

    if ( ierr != 0 )
    {
      value = 0.0;
      return value;
    }

    power4 = power4 * 0.25;
    xn = ( xn + lamda ) * 0.25;
    yn = ( yn + lamda ) * 0.25;
    zn = ( zn + lamda ) * 0.25;
    pn = ( pn + lamda ) * 0.25;
  }

}
//****************************************************************************80

void sncndn ( double u, double m, double &sn, double &cn, double &dn )

//****************************************************************************80
//
//  Purpose:
//
//    SNCNDN evaluates Jacobi elliptic functions.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    26 June 2018
//
//  Author:
//
//    Original ALGOL version by Roland Bulirsch.
//    C++ version by John Burkardt
//
//  Reference:
//
//    Roland Bulirsch,
//    Numerical calculation of elliptic integrals and elliptic functions,
//    Numerische Mathematik,
//    Volume 7, Number 1, 1965, pages 78-90.
//
//  Parameters:
//
//    Input, double U, M, the arguments.
//
//    Output, double &SN, &CN, &DN, the value of the Jacobi
//    elliptic functions sn(u,m), cn(u,m), and dn(u,m).
//
{
  double a;
  double b;
  double c;
  double ca;
  double d;
  int i;
  int l;
  double *m_array;
  double m_comp;
  double *n_array;
  double u_copy;

  m_comp = 1.0 - m;
  u_copy = u;

  if ( m_comp == 0.0 )
  {
    cn = 1.0 / cosh ( u_copy );
    dn = cn;
    sn = tanh ( u_copy );
    return;
  }

  if ( 1.0 < m )
  {
    d = 1.0 - m_comp;
    m_comp = - m_comp / d;
    d = sqrt ( d );
    u_copy = d * u_copy;
  }

  ca = sqrt ( DBL_EPSILON );

  a = 1.0;
  dn = 1.0;
  l = 24;

  m_array = new double[25];
  n_array = new double[25];

  for ( i = 0; i < 25; i++ )
  {
    m_array[i] = a;
    m_comp = sqrt ( m_comp );
    n_array[i] = m_comp;
    c = 0.5 * ( a + m_comp );
    if ( fabs ( a - m_comp ) <= ca * a )
    {
      l = i;
      break;
    }
    m_comp = a * m_comp;
    a = c;
  }

  u_copy = c * u_copy;
  sn = sin ( u_copy );
  cn = cos ( u_copy );

  if ( sn != 0.0 )
  {
    a = cn / sn;
    c = a * c;

    for ( i = l; 0 <= i; i-- )
    {
      b = m_array[i];
      a = c * a;
      c = dn * c;
      dn = ( n_array[i] + a ) / ( b + a );
      a = c / b;
    }

    a = 1.0 / sqrt ( c * c + 1.0 );

    if ( sn < 0.0 )
    {
      sn = - a;
    }
    else
    {
      sn = a;
    }
    cn = c * sn;
  }

  if ( 1.0 < m )
  {
    a = dn;
    dn = cn;
    cn = a;
    sn = sn / d;
  }

  delete [] m_array;
  delete [] n_array;

  return;
}
//****************************************************************************80

void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    19 March 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct std::tm *tm_ptr;
  std::time_t now;

  now = std::time ( NULL );
  tm_ptr = std::localtime ( &now );

  std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );

  std::cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}
