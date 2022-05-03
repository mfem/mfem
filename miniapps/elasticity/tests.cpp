#include <mfem.hpp>

#include "materials/neohookean.hpp"

using namespace std;
using namespace mfem;


int main()
{
  NeoHookeanMaterial<3, GradientType::Symbolic, GradientType::Symbolic> mat;
  tensor<double, 3, 3> H{{{0.337494265892494, 0.194238454581911, 0.307832573181341},
                          {0.090147365480304, 0.610402517912401, 0.458978918716148},
                          {0.689309323130592, 0.198321409053159, 0.901973313462065}}};

  double W = mat.strain_energy_density(H);
  std::cout << "SED = " << W << endl;

  auto P_symbolic = mat.stress_symbolic(H);
  auto P_fd = mat.stress_fd(H);
  auto P_enzyme_rev = mat.stress_enzyme_rev(H);
  auto P_enzyme_fwd = mat.stress_enzyme_fwd(H);
  cout << "symbolic\t" << P_symbolic << endl;
  cout << "finitediff\t" << P_fd << endl;
  cout << "enzymerev\t" << P_enzyme_rev << endl;
  cout << "enzymefwd\t" << P_enzyme_fwd << endl;
  return 0;
}
