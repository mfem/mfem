namespace manufactured
{

//----------------------------------------------------------
void adv(const Vector & x, Vector & a)
{
   a[1] = 1.0/(1.0 + att_param*att_param);
   a[0] = sqrt(1.0 - a[1]*a[1]);
}

//----------------------------------------------------------
real_t kappa(const Vector & x)
{
   return kappa_param;
}

//----------------------------------------------------------
real_t force(const Vector & x)
{
   int d = x.Size();

   Vector a(d);
   adv(x, a);
   real_t ax = a[0];
   real_t ay = 0.0;
   real_t az = 0.0;
   real_t k = kappa(x);

   real_t sx = sin(pi*x[0]);
   real_t cx = cos(pi*x[0]);
   real_t sy = 1.0;
   real_t cy = 1.0;
   real_t sz = 1.0;
   real_t cz = 1.0;

   if (d >= 2)
   {
      sy = sin(pi*x[1]);
      cy = cos(pi*x[1]);
      ay = a[1];
   }
   if (d >= 3)
   {
      sz = sin(pi*x[2]);
      cz = cos(pi*x[2]);
      az = a[2];
   }

   return ax*pi*cx*sy*sz
        + ay*pi*sx*cy*sz
        + az*pi*sx*sy*cz + d*k*pi*pi*sx*sy*sz;
}

//----------------------------------------------------------
real_t sol(const Vector & x)
{
   real_t sx = sin(pi*x[0]);
   real_t sy = 1.0;
   real_t sz = 1.0;

   int d = x.Size();
   if (d >= 2)
   {
      sy = sin(pi*x[1]);
   }
   if (d >= 3)
   {
      sz = sin(pi*x[2]);
   }

   return sx*sy*sz;
}


//----------------------------------------------------------
void grad(const Vector & x, Vector &grad)
{
   real_t sx = sin(pi*x[0]);
   real_t sy = 1.0;
   real_t sz = 1.0;

   real_t gx = pi*cos(pi*x[0]);
   real_t gy = 0.0;
   real_t gz = 0.0;

   grad[0] = gx;

   int d = x.Size();
   if (d >= 2)
   {
      sy = sin(pi*x[1]);
      gy = pi*cos(pi*x[1]);

      grad[0] = gx*sy;
      grad[1] = sx*gy;
   }
   if (d >= 3)
   {
      sz = sin(pi*x[2]);
      gz = pi*cos(pi*x[2]);

      grad[0] = gx*sy*sz;
      grad[1] = sx*gy*sz;
      grad[2] = sx*sy*gz;
   }
}

//----------------------------------------------------------
real_t laplace(const Vector & x)
{
   return -x.Size()*pi*pi*sol(x);
}

}
