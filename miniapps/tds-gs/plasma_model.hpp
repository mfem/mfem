#ifndef PLASMA_MODEL
#define PLASMA_MODEL

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <set>
using namespace mfem;
using namespace std;

class PlasmaModelBase
{
private:
  double coeff_u2 = 0.0; // coefficient of u^2, for debugging
  double mu0;
  double alpha;
  double beta;
  double gamma;
public:
  PlasmaModelBase() {}
  virtual double S_p_prime(double & psi_N) const {return 1.0;};
  virtual double S_ff_prime(double & psi_N) const {return 1.0;};
  virtual double S_prime_p_prime(double & psi_N) const {return 1.0;};
  virtual double S_prime_ff_prime(double & psi_N) const {return 1.0;};
  virtual double get_mu() const {return mu0;}
  virtual double get_coeff_u2() const {return coeff_u2;}
  virtual double get_alpha() const {return alpha;}
  virtual double get_beta() const {return beta;}
  virtual double get_gamma() const {return gamma;}
  // ~PlasmaModel() {}
};

/*
  Contains functions associated with the plasma model. 
  They appear on the RHS of the equation.
*/
class PlasmaModel : public PlasmaModelBase
{
private:
  double alpha;
  double beta;
  double lambda;
  double gamma;
  double r0;
  double mu0;
  double coeff_u2 = 0.0; // coefficient of u^2, for debugging
  const char *data_file;
  bool use_model = true;

  
public:
  PlasmaModel(double & alpha_, double & beta_, double & lambda_, double & gamma_, double & mu0_, double & r0_) :
    alpha(alpha_), beta(beta_), lambda(lambda_), gamma(gamma_), mu0(mu0_), r0(r0_)
  {
  }
  double S_p_prime(double & psi_N) const;
  double S_ff_prime(double & psi_N) const;
  double S_prime_p_prime(double & psi_N) const;
  double S_prime_ff_prime(double & psi_N) const;
  double get_mu() const {return mu0;}
  double get_coeff_u2() const {return coeff_u2;}
  ~PlasmaModel() { }
};


class PlasmaModelFile : public PlasmaModelBase
{
private:
  const char *data_file;
  vector<double> ffprime_vector;
  vector<double> pprime_vector;
  double alpha;
  double beta;
  double gamma;
  double mu0;
  int N;
  double dx;
public:
  PlasmaModelFile(double & mu0_, const char *data_file_, double alpha=1.0, double beta=1.0, double gamma=1.0) :
    mu0(mu0_), data_file(data_file_), alpha(alpha), beta(beta), gamma(gamma)
  {

    // cout << data_file_ << endl;
    
    ifstream inFile;
    inFile.open(data_file_);

    if (!inFile) {
      cerr << "Unable to open file" << data_file << endl;
      exit(1);   // call system to stop
    }

    string line;
    istringstream *iss;

    double fpol, pres, ffprime, pprime;
    while (getline(inFile, line)) {
      iss = new istringstream(line);
      *iss >> fpol >> pres >> ffprime >> pprime;
      // cout << fpol << " " << pres << " " << ffprime << " " << pprime << endl;
      ffprime_vector.push_back(ffprime);
      pprime_vector.push_back(pprime);
    }

    N = ffprime_vector.size();
    dx = 1.0 / (N - 1.0);

  }
  double S_p_prime(double & psi_N) const;
  double S_ff_prime(double & psi_N) const;
  double S_prime_p_prime(double & psi_N) const;
  double S_prime_ff_prime(double & psi_N) const;
  double get_mu() const {return mu0;}
  double get_alpha() const {return alpha;}
  double get_beta() const {return beta;}
  double get_gamma() const {return gamma;}
  double set_alpha(double alpha_) {alpha = alpha_;}
  ~PlasmaModelFile() { }
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
  PlasmaModelBase *model;
  int option;
  double psi_max;
  double psi_bdp;
  set<int> plasma_inds;
  int attr_lim;
public:
  NonlinearGridCoefficient(PlasmaModelBase *pm, int option_, const GridFunction *psi_,
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
