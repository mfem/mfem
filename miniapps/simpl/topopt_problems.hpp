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

   // 2D Compliant mechanism
   ForceInverter2=-21,
};

void MarkBoundaries(Mesh &mesh, int attr, std::function<bool(const Vector &x)> marker);
void MarkElements(Mesh &mesh, int attr, std::function<bool(const Vector &x)> marker);

Mesh * GetTopoptMesh(TopoptProblem prob,
                     real_t &r_min, real_t &min_vol, real_t &max_vol,
                     real_t &lambda, real_t &mu,
                     int ser_ref_levels, int par_ref_levels=-1);

void SetupTopoptProblem(TopoptProblem prob, ElasticityProblem &elasticity,
                        GridFunction &gf_filter, GridFunction &gf_state);

} // end of namespace mfem
