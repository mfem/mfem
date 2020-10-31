
void SlipWallBC::calcFlux(const mfem::Vector &x,
                                       const mfem::Vector &dir,
                                       const mfem::Vector &q,
                                       mfem::Vector &flux_vec)
{
   double press;
   press = computePressure(q, dim);
   flux[0] = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      flux[i + 1] = dir[i] * press;
   }
   flux[dim + 1] = 0.0;
}
