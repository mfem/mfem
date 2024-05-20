#ifndef AMR
#define AMR

#include "mfem.hpp"

using namespace mfem;
using namespace std;

class RegionalThresholdRefiner : public MeshOperator
{
protected:
  ErrorEstimator &estimator;
  AnisotropicErrorEstimator *aniso_estimator;

  double total_norm_p;
  double total_err_goal;
  double total_fraction;
  double local_err_goal;
  long long max_elements;
  long   amr_levels, xRange_levels, yRange_levels;
  bool   yRange, xRange;
  double xmin, xmax, ymin, ymax;

  double threshold;
  double threshold_outside;
  
  long long num_marked_elements;

  Array<Refinement> marked_elements;
  long current_sequence;

  int non_conforming;
  int nc_limit;

  double GetNorm(const Vector &local_err, Mesh &mesh) const;  
  int ApplyImpl(Mesh &mesh) {}
 public:
  RegionalThresholdRefiner(ErrorEstimator &est);
  void SetTotalErrorFraction(double fraction) { total_fraction = fraction; }
  void SetMaxElements(long long max_elem) { max_elements = max_elem; }
  int ApplyRef(Mesh &mesh, int attrib, double levels_inside, double levels_outside);
  virtual void Reset();
};

#endif
