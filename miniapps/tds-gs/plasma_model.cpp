#include "mfem.hpp"
#include "plasma_model.hpp"

#include <iostream>
#include <set>
#include <list>
using namespace mfem;
using namespace std;

double PlasmaModel::S_p_prime(double & psi_N) const
{
  // return zero when derivative is singular
  if ((gamma < 1.0) && (psi_N == 1.0)) {
    return 0.0;
  }
  if ((alpha < 1.0) && (psi_N == 0.0)) {
    return 0.0;
  }
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }
  return lambda * beta * pow(1.0 - pow(psi_N, alpha), gamma) / r0;
}
double PlasmaModel::S_prime_p_prime(double & psi_N) const
{
  // return zero when derivative is singular
  if ((gamma < 1.0) && (psi_N == 1.0)) {
    return 0.0;
  }
  if ((alpha < 1.0) && (psi_N == 0.0)) {
    return 0.0;
  }
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }
  return - alpha * gamma * lambda * beta
    * pow(1.0 - pow(psi_N, alpha), gamma - 1.0)
    * pow(psi_N, alpha - 1.0) / r0;
}
double PlasmaModel::S_ff_prime(double & psi_N) const
{
  // return zero when derivative is singular
  if ((gamma < 1.0) && (psi_N == 1.0)) {
    return 0.0;
  }
  if ((alpha < 1.0) && (psi_N == 0.0)) {
    return 0.0;
  }
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }
  return lambda * (1.0 - beta) * mu0 * r0 * pow(1.0 - pow(psi_N, alpha), gamma);
}
double PlasmaModel::S_prime_ff_prime(double & psi_N) const
{
  // return zero when derivative is singular
  if ((gamma < 1.0) && (psi_N == 1.0)) {
    return 0.0;
  }
  if ((alpha < 1.0) && (psi_N == 0.0)) {
    return 0.0;
  }
  // outside plasma, return 0
  if ((psi_N > 1.0) || (psi_N < 0.0)) {
    return 0.0;
  }

  return - alpha * gamma * lambda * (1.0 - beta) * mu0 * r0
    * pow(1.0 - pow(psi_N, alpha), gamma - 1.0)
    * pow(psi_N, alpha - 1.0);
}
double normalized_psi(double & psi, double & psi_max, double & psi_bdp)
{
  if (false) {
    return psi;
  }
  return (psi - psi_max) / (psi_bdp - psi_max);
}

double NonlinearGridCoefficient::Eval(ElementTransformation & T,
                                      const IntegrationPoint & ip)
{

  if (true) {
    // check that we are in the limiter region
    if (T.Attribute != attr_lim) {
      return 0.0;
    }

    // check to see if integration point is inside an element that is
    // part of the plasma region
    const int *v = T.mesh->GetElement(T.ElementNo)->GetVertices();
    set<int>::iterator plasma_inds_it;
    for (int i = 0; i < 3; ++i) {
      plasma_inds_it = plasma_inds.find(v[i]);
      if (plasma_inds_it == plasma_inds.end()) {
        return 0.0;
      }
    }
  }
  
  double x_[3];
  Vector x(x_, 3);
  T.Transform(ip, x);
  double ri(x(0));
  if (false) {
    // check to see if we're inside the exact plasma region
    double r0_ = 1.0;
    double z0_ = 0.0;
    double L_ = 0.35;
    double zi(x(1));
    if (abs(ri - r0_) + abs(zi - z0_) > L_) {
      return 0.0;
    }
  }
 
  double psi_val;
  Mesh *gf_mesh = psi->FESpace()->GetMesh();
  int Component = 0;

  psi_val = psi->GetValue(T, ip, Component);
  double psi_N = normalized_psi(psi_val, psi_max, psi_bdp);
  double mu = model->get_mu();
  double coeff_u2 = model->get_coeff_u2();

  if (option == 1) {
    // integrand
    return
      + ri * (model->S_p_prime(psi_N)) + (model->S_ff_prime(psi_N)) / (mu * ri)
      + coeff_u2 * pow(psi_val, 2.0);
  } else {
    // derivative of integrand
    double coeff = 1.0;
    if (option == 2) {
      // coefficient for phi
      coeff = 1.0 / (psi_bdp - psi_max);
    } else if (option == 3) {
      // coefficient for phi_ma
      coeff = - 1.0 * (1 - psi_N) / (psi_bdp - psi_max);
    } else if (option == 4) {
      // coefficient for phi_x
      coeff = - 1.0 * psi_N / (psi_bdp - psi_max);
    }

    return
      coeff * (ri * (model->S_prime_p_prime(psi_N))
               + (model->S_prime_ff_prime(psi_N)) / (mu * ri))
      + coeff_u2 * 2.0 * psi_val;
  }
}

map<int, vector<int>> compute_vertex_map(Mesh & mesh, int with_attrib) {
  // get map between vertices and neighboring vertices
  map<int, vector<int>> vertex_map;
  for (int i = 0; i < mesh.GetNE(); i++) {
    const int *v = mesh.GetElement(i)->GetVertices();
    const int ne = mesh.GetElement(i)->GetNEdges();
    const int attrib = mesh.GetElement(i)->GetAttribute();

    if ((with_attrib == -1) || (attrib == with_attrib)) {
      for (int j = 0; j < ne; j++) {
        const int *e = mesh.GetElement(i)->GetEdgeVertices(j);
        vertex_map[v[e[0]]].push_back(v[e[1]]);
      }
    }
  }
  return vertex_map;
}

void compute_plasma_points(const GridFunction & z, const Mesh & mesh,
                           const map<int, vector<int>> & vertex_map,
                           set<int> & plasma_inds,
                           int &ind_min, int &ind_max, double &min_val, double & max_val,
                           int iprint) {

  // mag ax point: global minimum in z
  // saddle point: closest saddle point to mag ax point, otherwise maximum on limiter boundary

  // keep track of elements inside of plasma region
  
   Vector nval;
   z.GetNodalValues(nval);

   min_val = + numeric_limits<double>::infinity();
   max_val = - numeric_limits<double>::infinity();
   ind_min = 0;
   ind_max = 0;

   vector<int> candidate_x_points;
     
   int count = 0;
   // **********************************************************************************
   // loop through vertices and check neighboring vertices to see if we found a saddle point
   for(int iv = 0; iv < mesh.GetNV(); ++iv) {

     // ensure point is in vertex map
     vector<int> adjacent;
     try {
       adjacent = vertex_map.at(iv);
     } catch (...) {
       continue;
     }

     // min/max checker
     if (nval[iv] < min_val) {
       min_val = nval[iv];
       ind_min = iv;
     }
     if (nval[iv] > max_val) {
       max_val = nval[iv];
       ind_max = iv;
     }
     
     // saddle point checker
     int j = 0;
     const double* x0 = mesh.GetVertex(iv);
     const double* a = mesh.GetVertex(adjacent[j]);

     map<double, double> clock;
     set<double> ordered_angs;
     for (j = 0; j < adjacent.size(); ++j) {
       const int jv = adjacent[j];
       const double* b = mesh.GetVertex(jv);
       double diff = nval[jv] - nval[iv];
       // cout << b[0] << ", " << b[1] << endl;

       double ax = a[0]-x0[0];
       double ay = a[1]-x0[1];
       double bx = b[0]-x0[0];
       double by = b[1]-x0[1];

       double ang = atan2(by, bx);
       clock[ang] = diff;
       ordered_angs.insert(ang);
     }

     int sign_changes = 0;
     set<double>::iterator it = ordered_angs.begin();
     double init = clock[*it];
     double prev = clock[*it];
     ++it;
     for (; it != ordered_angs.end(); ++it) {
       if (clock[*it] * prev < 0.0) {
         ++sign_changes;
       }
       prev = clock[*it];
     }
     if (prev * init < 0.0) {
       ++sign_changes;
     }

     if (sign_changes >= 4) {
       if (iprint) {
         printf("Found saddle at (%9.6f, %9.6f), val=%9.6f\n", x0[0], x0[1], nval[iv]);
       }
       candidate_x_points.push_back(iv);
       ++count;
     } 
   }

   int ind_x = ind_max;
   double x_val = max_val;
   for (int i = 0; i < candidate_x_points.size(); ++i) {
     int iv = candidate_x_points[i];
     if (nval[iv] < x_val) {
       x_val = nval[iv];
       ind_x = iv;
     }
   }

   const double* x_min = mesh.GetVertex(ind_min);
   const double* x_max = mesh.GetVertex(ind_max);
   const double* x_x = mesh.GetVertex(ind_x);
   if (iprint) {
     cout << "total saddles found: " << count << endl;
     printf("  min of %9.6f at (%9.6f, %9.6f), ind %d\n", min_val, x_min[0], x_min[1], ind_min);
     printf("  max of %9.6f at (%9.6f, %9.6f), ind %d\n", max_val, x_max[0], x_max[1], ind_max);
     printf("x_val of %9.6f at (%9.6f, %9.6f), ind %d\n", x_val, x_x[0], x_x[1], ind_x);
   }

   // DAS: we need to return the x_val, not the max_val.
   // TODO, refactor to make less confusing...
   max_val = x_val;
   ind_max = ind_x;

   // **********************************************************************************
   // start at x_min and mark vertices that are in the plasma
   list<int> queue;
   set<int>::iterator plasma_inds_it;
   queue.push_back(ind_min);
   plasma_inds.insert(ind_min);
   plasma_inds.insert(ind_x);
   while (!queue.empty()) {
     // get a point that is already in the plasma region
     int iv = queue.front();
     queue.pop_front();

     // check for neighboring points
     vector<int> adjacent;
     try {
       adjacent = vertex_map.at(iv);
     } catch (...) {
       continue;
     }

     // check if neighboring points are in the plasma
     for (int i = 0; i < adjacent.size(); ++i) {
       double val = nval[adjacent[i]];
       plasma_inds_it = plasma_inds.find(adjacent[i]);
       // check that found vertex is not already accounted for
       if (plasma_inds_it == plasma_inds.end()) {
         // point is not yet in plasma inds
         if ((val >= min_val) && (val <= x_val)) {
           // value at this point is between min and max, it is part of plasma
           // add the point to the queue
           queue.push_back(adjacent[i]);
           plasma_inds.insert(adjacent[i]);
         } else {
           // value at point is not between min and max, it is adjacent to the plasma
           plasma_inds.insert(adjacent[i]);
         }
       }
     }
   }
   
   
}
