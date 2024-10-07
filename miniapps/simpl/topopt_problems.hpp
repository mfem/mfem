#ifndef TOPOPT_PROBLEMS_HPP
#define TOPOPT_PROBLEMS_HPP
#include "mfem.hpp"
#include "topopt.hpp"


namespace mfem
{

enum TopoptProblem
{
   // 2D
   Cantilever2=21,
   MBB2=22,
   Arch2=23,
   Bridge2=24,

   // 3D
   Cantilever3=31,
   MBB3=32,
   Arch3=33,
   Bridge3=34,
   Torsion3=35,

   // 2D Compliant mechanism
   ForceInverter2=-21,
};

void MarkBoundaries(Mesh &mesh, int attr,
                    std::function<bool(const Vector &x)> marker);
void MarkElements(Mesh &mesh, int attr,
                  std::function<bool(const Vector &x)> marker);

Mesh * GetTopoptMesh(TopoptProblem prob, std::stringstream &filename,
                     real_t &r_min, real_t &tot_vol, real_t &min_vol, real_t &max_vol,
                     real_t &E, real_t &nu,
                     Array2D<int> &ess_bdr_displacement, Array<int> &ess_bdr_filter,
                     int &solid_attr, int &void_attr,
                     int ser_ref_levels, int par_ref_levels=-1);

void SetupTopoptProblem(TopoptProblem prob,
                        HelmholtzFilter &filter, ElasticityProblem &elasticity,
                        GridFunction &gf_filter, GridFunction &gf_state);

} // end of namespace mfem
#endif
