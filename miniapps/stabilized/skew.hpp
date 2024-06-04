namespace skew
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
   return 0.0;
}

//----------------------------------------------------------
real_t sol(const Vector & x)
{
   if ((x[1] - x[0] - 0.2 < 0.0)
      &(x[0] + x[1] -0.99 < 0.0))
   {
      return 1.0;
   }
   return 0.0;
}

//----------------------------------------------------------
void grad(const Vector & x, Vector &grad)
{
   grad = 0.0;
}

//----------------------------------------------------------
real_t laplace(const Vector & x)
{
   return 0.0;
}

}
