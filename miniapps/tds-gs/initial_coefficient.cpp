#include "mfem.hpp"
#include "initial_coefficient.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


double InitialCoefficient::Eval(ElementTransformation & T,
                                const IntegrationPoint & ip)
{

  if (mask_plasma) {
    const int *v = T.mesh->GetElement(T.ElementNo)->GetVertices();
    set<int>::iterator plasma_inds_it;
    for (int i = 0; i < 3; ++i) {
      plasma_inds_it = plasma_inds.find(v[i]);
      if (plasma_inds_it == plasma_inds.end()) {
        return 0.0;
      }
    }
  }
  if (use_manufactured) {
    // double L = 0.35;
    // if (abs(r - 0.625-0.75/2)+abs(z) <= L)
    return exact_coeff.Eval(T, ip);
    // else {
    //   return 0.0;
    // }
  }

  
  double x_[3];
  Vector x(x_, 3);
  T.Transform(ip, x);
  double r(x(0));
  double z(x(1));

  
  double mf = (r - r0) / dr;
  double nf = (z - z0) / dz;
  int m = int(mf);
  int n = int(nf);
  double rlc = r0 + m * dr;
  double zlc = z0 + n * dz;

  if ((mf < 0) || (mf > nr-2) || (nf < 0) || (nf > nz-2)) {
    return 0.0;
  }
  double ra, rb, rc;
  double za, zb, zc;
  double va, vb, vc;
  if (fmod(mf, 1.0) > 0.5) {
    // choose two points to the right
    ra = rlc+dr; za = zlc; va = psizr[n][m+1];
    rb = rlc+dr; zb = zlc+dz; vb = psizr[n+1][m+1];
    if (fmod(nf, 1.0) > 0.5) {
      // top left
      rc = rlc; zc = zlc+dz; vc = psizr[n+1][m];
    } else {
      // bot left
      rc = rlc; zc = zlc; vc = psizr[n][m];
    }
  } else {
    // choose two points to the left
    ra = rlc; za = zlc; va = psizr[n][m];
    rb = rlc; zb = zlc+dz; vb = psizr[n+1][m];
    if (fmod(nf, 1.0) > 0.5) {
      // top right
      rc = rlc+dr; zc = zlc+dz; vc = psizr[n+1][m+1];
    } else {
      // bot right
      rc = rlc+dr; zc = zlc; vc = psizr[n][m+1];
    }
  }

  double wa = ((zb-zc)*(r-rc)+(rc-rb)*(z-zc))
    /((zb-zc)*(ra-rc)+(rc-rb)*(za-zc));
  double wb = ((zc-za)*(r-rc)+(ra-rc)*(z-zc))
    /((zb-zc)*(ra-rc)+(rc-rb)*(za-zc));
  double wc = 1 - wb - wa;

  // if ((wa < 0) || (wb < 0) || (wc < 0)) {
  //   cout << "weights are out of bounds, check me!" << endl;
  //   printf("(r, z) = (%f, %f)\n", r, z);
  //   printf("(ra, za) = (%f, %f)\n", ra, za);
  //   printf("(rb, zb) = (%f, %f)\n", rb, zb);
  //   printf("(rc, zc) = (%f, %f)\n", rc, zc);
  //   printf("(mf, nf) = (%f, %f)\n", mf, nf);
  //   printf("(m, n) = (%d, %d)\n", m, n);
  //   printf("(wa, wb, wc) = (%f, %f, %f)\n", wa, wb, wc);
  //   return 0.0;
  // }
  return wa*va+wb*vb+wc*vc;

}

InitialCoefficient from_manufactured_solution() {

  // center of limiter
  double r0 = 0.625+0.75/2;
  double z0 = 0.0;
  double L = 0.35;
  double k = M_PI/(2.0*L);

  ExactCoefficient exact_coeff(r0, z0, k);
  InitialCoefficient initial_coeff(exact_coeff);
  return initial_coeff;
}


InitialCoefficient read_data_file(const char *data_file) {
  ifstream inFile;
  inFile.open(data_file);

  if (!inFile) {
    cerr << "Unable to open file datafile.txt";
    exit(1);   // call system to stop
  }

  string line;
  istringstream *iss;
  
  while (getline(inFile, line)) {
    if (line.find("nw") != std::string::npos) {
      getline(inFile, line);
      break;
    }
  }
  iss = new istringstream(line);
  int idum, nw, nh;
  *iss >> idum >> nw >> nh;
  // cout << nw << " " << nh << endl;

  while (getline(inFile, line)) {
    if (line.find("rdim") != std::string::npos) {
      getline(inFile, line);
      break;
    }
  }
  iss = new istringstream(line);
  double rdim, zdim, rcentr, rleft, zmid;
  *iss >> rdim >> zdim >> rcentr >> rleft >> zmid;
  // cout << rdim << endl;

  double r0, r1, z0, z1;
  r0 = rleft;
  r1 = rleft+rdim;
  z0 = zmid-zdim/2.0;
  z1 = zmid+zdim/2.0;
  // DAS - this is overrided since mesh isn't based on ITER geometry yet
  r0 = .67; r1 = 1.321;
  z0 = -0.556; z1 = 0.5556;
  
  while (getline(inFile, line)) {
    if (line.find("psizr") != std::string::npos) {
      getline(inFile, line);
      break;
    }
  }
  iss = new istringstream(line);
  double **psizr;
  // [nh][nw];
  psizr = new double *[nh];
  int i, j;
  for (i = 0; i < nh; ++i) {
    psizr[i] = new double[nw];
    for (j = 0; j < nw; ++j) {
      *iss >> psizr[i][j];
    }
  }

  // nz=nh, nr=nw
  InitialCoefficient init_coeff(psizr, r0, r1, z0, z1, nh, nw);

  return init_coeff;
}
