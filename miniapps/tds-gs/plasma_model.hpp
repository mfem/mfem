#ifndef PLASMA_MODEL
#define PLASMA_MODEL

#include "mfem.hpp"
#include <set>
using namespace mfem;
using namespace std;


/*
  Contains functions associated with the plasma model. 
  They appear on the RHS of the equation.
*/
class PlasmaModel
{
private:
  double alpha;
  double beta;
  double lambda;
  double gamma;
  double mu0;
  double r0;
public:
  PlasmaModel(double & alpha_, double & beta_, double & lambda_, double & gamma_, double & mu0_, double & r0_) :
    alpha(alpha_), beta(beta_), lambda(lambda_), gamma(gamma_), mu0(mu0_), r0(r0_) { }
  double S_p_prime(double & psi_N) const;
  double S_ff_prime(double & psi_N) const;
  double S_prime_p_prime(double & psi_N) const;
  double S_prime_ff_prime(double & psi_N) const;
  double get_mu() const {return mu0;}
  ~PlasmaModel() { }
};

/*
  Coefficient for the nonlinear term
  option:
  1: r S_p' + S_ff' / mu r
  2: (r S'_p' + S'_ff' / mu r) / (psi_bdp - psi_max)
  3: - (r S'_p' + S'_ff' / mu r) (1 - psi_N) / (psi_bdp - psi_max)
  4: - (r S'_p' + S'_ff' / mu r) psi_N / (psi_bdp - psi_max)
*/
class NonlinearGridCoefficient : public Coefficient
{
private:
  const GridFunction *psi;
  PlasmaModel *model;
  int option;
  double psi_max;
  double psi_bdp;
  set<int> plasma_inds;
  int attr_lim;
public:
  NonlinearGridCoefficient(PlasmaModel *pm, int option_, const GridFunction *psi_,
                           double & psi_max_, double & psi_bdp_, set<int> plasma_inds_,
                           int attr_lim_) :
    model(pm), option(option_), psi(psi_),
    psi_max(psi_max_), psi_bdp(psi_bdp_), plasma_inds(plasma_inds_), attr_lim(attr_lim_) { }
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~NonlinearGridCoefficient() { }
};

void compute_plasma_points(const GridFunction & z, const Mesh & mesh,
                           const map<int, vector<int>> & vertex_map,
                           set<int> & plasma_inds,
                           int &ind_min, int &ind_max, double &min_val, double & max_val,
                           int iprint);
map<int, vector<int>> compute_vertex_map(Mesh & mesh, int with_attrib = -1);

#endif
