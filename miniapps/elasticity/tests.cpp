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
  std::cout << "Strain energy density = " << W << "\n" << endl;

  auto P_symbolic = mat.stress_symbolic(H);
  auto P_fd = mat.stress_fd(H);
  auto P_enzyme_rev = mat.stress_enzyme_rev(H);
  auto P_enzyme_fwd = mat.stress_enzyme_fwd(H);
  auto P_dual = mat.stress_dual(H);
  cout << "Stress\n"
       << "------" << endl;
  cout << "symbolic\t" << P_symbolic << endl;
  cout << "finite diff\t" << P_fd << endl;
  cout << "enzyme rev\t" << P_enzyme_rev << endl;
  cout << "enzyme fwd\t" << P_enzyme_fwd << endl;
  cout << "dual numbers\t" << P_dual << endl;

  cout << "\n\n" << endl;

  tensor<double, 3, 3> Hdot{{{0.191653881479253, 0.445956210862074, 0.038732049150475},
                             {0.589233685844341, 0.092360587237104, 0.259746940075709},
                             {0.830970655782669, 0.485472875958392, 0.03308538443643 }}};

  auto dP_symbolic = mat.action_of_gradient_symbolic(H, Hdot);
  auto dP_fd = mat.action_of_gradient_fd(H, Hdot);
  auto dP_enzyme_rev = mat.action_of_gradient_enzyme_rev(H, Hdot);
  auto dP_enzyme_fwd = mat.action_of_gradient_enzyme_fwd(H, Hdot);
  auto dP_dual = mat.action_of_gradient_dual(H, Hdot);
  auto dP_full = ddot(mat.gradient(H), Hdot);
  cout << "Elasticities\n"
       << "------------" << endl;
  cout << "symbolic\t" << dP_symbolic << endl;
  cout << "finite diff\t" << dP_fd << endl;
  cout << "enzyme rev\t" << dP_enzyme_rev << endl;
  cout << "enzyme fwd\t" << dP_enzyme_fwd << endl;
  cout << "dual numbers\t" << dP_dual << endl;
  cout << "contraction of symbolic elastic tensor with \ntangent dispgrad\t" << dP_full << endl;

  return 0;
}
