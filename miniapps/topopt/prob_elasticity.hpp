#ifndef TOPOPT_ELASTICITY
#define TOPOPT_ELASTICITY

#include "mfem.hpp"
#include <string>
namespace mfem
{
double DistanceToSegment(const Vector &p, const Vector &v, const Vector &w);

void initialDesign(GridFunction& psi, Vector domain_center,
                   Array<Vector*> ports,
                   double target_volume, double domain_size, double lower, double upper,
                   bool toDensity=false);

void uniformRefine(std::unique_ptr<Mesh>& mesh, int ser_ref_levels,
                   int par_ref_levels);

enum ElasticityProblem
{
   Cantilever,
   MBB,
   Bridge,
   LBracket,
   Cantilever3,
   Torsion3,
   MBB_selfloading, // below this, everything should be self-loading case
   Arch2,
   SelfLoading3
};

void GetElasticityProblem(const ElasticityProblem problem,
                          double &filter_radius, double &vol_fraction,
                          std::unique_ptr<Mesh> &mesh, std::unique_ptr<VectorCoefficient> &vforce_cf,
                          Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                          std::string &prob_name, int ref_levels, int par_ref_levels=-1);

enum CompliantMechanismProblem
{
   ForceInverter,
   ForceInverter3
};

void GetCompliantMechanismProblem(const CompliantMechanismProblem problem,
                                  double &filter_radius, double &vol_fraction,
                                  std::unique_ptr<Mesh> &mesh,
                                  double &k_in, double &k_out,
                                  Vector &d_in, Vector &d_out,
                                  std::unique_ptr<VectorCoefficient> &t_in,
                                  Array<int> &bdr_in, Array<int> &bdr_out,
                                  Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                                  std::string &prob_name, int ref_levels, int par_ref_levels=-1);

void CantileverPreRefine(double &filter_radius, double &vol_fraction,
                         std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                         std::unique_ptr<VectorCoefficient> &vforce_cf);
void CantileverPostRefine(int ser_ref_levels, int par_ref_levels,
                          std::unique_ptr<Mesh> &mesh);

void MBBPreRefine(double &filter_radius, double &vol_fraction,
                  std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                  std::unique_ptr<VectorCoefficient> &vforce_cf);
void MBBPostRefine(int ser_ref_levels, int par_ref_levels,
                   std::unique_ptr<Mesh> &mesh);

void BridgePreRefine(double &filter_radius, double &vol_fraction,
                     std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                     std::unique_ptr<VectorCoefficient> &vforce_cf);
void BridgePostRefine(int ser_ref_levels, int par_ref_levels,
                      std::unique_ptr<Mesh> &mesh);

void LBracketPreRefine(double &filter_radius, double &vol_fraction,
                       std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                       std::unique_ptr<VectorCoefficient> &vforce_cf);
void LBracketPostRefine(int ser_ref_levels, int par_ref_levels,
                        std::unique_ptr<Mesh> &mesh);

void Cantilever3PreRefine(double &filter_radius, double &vol_fraction,
                          std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                          std::unique_ptr<VectorCoefficient> &vforce_cf);
void Cantilever3PostRefine(int ser_ref_levels, int par_ref_levels,
                           std::unique_ptr<Mesh> &mesh);

void Torsion3PreRefine(double &filter_radius, double &vol_fraction,
                       std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                       std::unique_ptr<VectorCoefficient> &vforce_cf);
void Torsion3PostRefine(int ser_ref_levels, int par_ref_levels,
                        std::unique_ptr<Mesh> &mesh);

void MBB_selfloadingPreRefine(double &filter_radius, double &vol_fraction,
                              std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                              std::unique_ptr<VectorCoefficient> &vforce_cf);
void MBB_selfloadingPostRefine(int ser_ref_levels, int par_ref_levels,
                               std::unique_ptr<Mesh> &mesh);

void Arch2PreRefine(double &filter_radius, double &vol_fraction,
                    std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                    std::unique_ptr<VectorCoefficient> &vforce_cf);
void Arch2PostRefine(int ser_ref_levels, int par_ref_levels,
                     std::unique_ptr<Mesh> &mesh);

void SelfLoading3PreRefine(double &filter_radius, double &vol_fraction,
                           std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                           std::unique_ptr<VectorCoefficient> &vforce_cf);
void SelfLoading3PostRefine(int ser_ref_levels, int par_ref_levels,
                            std::unique_ptr<Mesh> &mesh);

void ForceInverterPreRefine(double &filter_radius, double &vol_fraction,
                            std::unique_ptr<Mesh> &mesh, Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter,
                            double &k_in, Vector &d_in, Array<int> &bdr_in,
                            std::unique_ptr<VectorCoefficient> &t_in,
                            double &k_out, Vector &d_out, Array<int> &bdr_out);
void ForceInverterPostRefine(int ser_ref_levels, int par_ref_levels,
                             std::unique_ptr<Mesh> &mesh);
}
#endif