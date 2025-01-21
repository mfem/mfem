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

   // 3D Compliant mechanism
   ForceInverter3=-31,

   // Multi-objective. 3 Digits
   // 1st digit: the number of objectives
   // 2nd digit: spatial dimension
   // 3rd digit: problem type
   // sign: need to solve adjoint (negative) or not (positive)

   // 2D
   MultiFrame2=221,
   MultiBridge2=222,
};

enum ThermalTopoptProblem
{
   // 2D
   HeatSink2=21,

};

void MarkBoundaries(Mesh &mesh, int attr,
                    std::function<bool(const Vector &x)> marker);
void MarkElements(Mesh &mesh, int attr,
                  std::function<bool(const Vector &x)> marker);

// Mesh, BC, filter radius, material properties
Mesh * GetTopoptMesh(TopoptProblem prob, std::stringstream &filename,
                     real_t &r_min, real_t &tot_vol, real_t &min_vol, real_t &max_vol,
                     real_t &E, real_t &nu,
                     Array2D<int> &ess_bdr_displacement, Array<int> &ess_bdr_filter,
                     int &solid_attr, int &void_attr,
                     int ser_ref_levels, int par_ref_levels=-1);
Mesh * GetThermalTopoptMesh(ThermalTopoptProblem prob, std::stringstream &filename,
                            real_t &r_min, real_t &tot_vol, real_t &min_vol, real_t &max_vol,
                            real_t &kappa,
                            Array<int> &ess_bdr_heat, Array<int> &ess_bdr_filter,
                            int &solid_attr, int &void_attr,
                            int ser_ref_levels, int par_ref_levels=-1);

// Right hand side. Force or adjoint problems
void SetupTopoptProblem(TopoptProblem prob,
                        HelmholtzFilter &filter, ElasticityProblem &elasticity,
                        GridFunction &gf_filter, std::vector<std::unique_ptr<GridFunction>> &gf_state);
void SetupThermalTopoptProblem(ThermalTopoptProblem prob,
                               HelmholtzFilter &filter, DiffusionProblem &diffusion,
                               GridFunction &filter_gf, GridFunction &state_gf);

real_t DistanceToSegment(const Vector &p, const Vector &v, const Vector &w);
void ForceInverterInitialDesign(GridFunction &x,
                                LegendreEntropy *entropy=nullptr);
void ForceInverter3InitialDesign(GridFunction &x, LegendreEntropy *entropy=nullptr);

} // end of namespace mfem
#endif
