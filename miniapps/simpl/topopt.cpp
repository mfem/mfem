#include "topopt.hpp"

namespace mfem
{

DesignDensity::DesignDensity(
   FiniteElementSpace &fes_control, const real_t tot_vol,
   const real_t min_vol, const real_t max_vol,
   LegendreEntropy *entropy)
   :fes_control(fes_control), tot_vol(tot_vol),
    min_vol(min_vol), max_vol(max_vol), entropy(entropy)
{
   hasPassiveElements = fes_control.GetMesh()->attributes.Max() > 1;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace * pfes_control = dynamic_cast<ParFiniteElementSpace*>
                                          (&fes_control);
   if (pfes_control)
   {
      zero.reset(new ParGridFunction(pfes_control));
   }
   else
   {
      zero.reset(new GridFunction(&fes_control));
   }
#else
   zero.reset(new GridFunction(&fes_control));
#endif
   *zero = 0.0;
}

real_t DesignDensity::ApplyVolumeProjection(GridFunction &x)
{
   real_t mu = 0.0; // constant perturbation

   // define density with perturbation
   std::function<real_t(const real_t)> density_fun;
   if (entropy)
   {
      // if entropy exists, then use the Bregman projection
      // assuming x is the dual variable
      density_fun = [this, &mu](const real_t psi) { return this->entropy->backward(psi + mu); };
   }
   else
   {
      // if entropy does not exist, then use the L2 projection
      // assuming x is the primal variable
      density_fun = [&mu](const real_t rho) {return std::max(0.0, std::min(rho + mu, 1.0));};
   }
   MappedGFCoefficient density(x, density_fun);

   // Check the volume constraints and determine the target volume
   real_t curr_vol = zero->ComputeL1Error(density);
   real_t target_vol=-1;
   if (curr_vol > max_vol)
   {
      target_vol = max_vol;
   }
   else if (curr_vol < min_vol)
   {
      target_vol = min_vol;
   }
   if (Mpi::Root()) out << "\t\tVolume projection: Target Volume = " << target_vol << ", Current = " <<curr_vol << std::endl;

   // if target volume is -1, then it is already satisfied
   if (target_vol == -1) { return curr_vol; }

   // Get lower and upper bound
   // Baseline is computed by considering a constant density that has target volume
   // Then the lower/upper bounds of mu can be found by
   // subtracting max/min of the current variable from the baseline
   // This is possible because our density mapping is an increasing function
   real_t baseline = entropy
                     ? entropy->forward(target_vol / tot_vol)
                     : target_vol / tot_vol;
   real_t upper = baseline - x.Min();
   real_t lower = baseline - x.Max();
#ifdef MFEM_USE_MPI
   ParGridFunction *px = dynamic_cast<ParGridFunction*>(&x);
   MPI_Comm comm;
   if (px)
   {
      comm = px->ParFESpace()->GetComm();
      MPI_Allreduce(MPI_IN_PLACE, &upper, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);
      MPI_Allreduce(MPI_IN_PLACE, &lower, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);
   }
#endif

   // bisection
   real_t target_accuracy = 1e-12;
   real_t dc = (upper - lower)*0.5; // increament
   mu = (upper + lower) * 0.5; // initial choice
   while (dc > target_accuracy)
   {
      curr_vol = zero->ComputeL1Error(density);
      dc *= 0.5;
      mu += curr_vol < target_vol ? dc : -dc;
   }
   x += mu;
   return curr_vol;
}

} // end of namespace mfem
