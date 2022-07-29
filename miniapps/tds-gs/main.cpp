/* 
   Compile with: make dan

   Sample runs:  
   ./dan
   ./dan -m meshes/gs_mesh.msh
   ./dan -m meshes/gs_mesh.msh -o 2

   After, run:
   glvis -m mesh.mesh -g sol.gf

   Description: 
   Solve the Grad-Shafranov equation using a newton iteration:
   d_psi a(psi^k, v, phi^k) = l(I, v) - a(psi^k, v), for all v in V
   
   a = + int 1/(mu r) grad psi dot grad v dr dz  (term1)
       - int (r Sp + 1/(mu r) Sff) v dr dz       (term2)
       + int_Gamma 1/mu psi(x) N(x) v(x) dS(x)   (term3)
       + int_Gamma int_Gamma 1/(2 mu) (psi(x) - psi(y)) M(x, y) (v(x) - v(y)) dS(x) dS(y)  (term4)
   
   d_psi a = + int 1/(mu r) grad phi dot grad v dr dz           (term1')
             - int (r Sp' + 1/(mu r) Sff') d_psi psi_N v dr dz  (term2')
             + int_Gamma 1/mu phi(x) N(x) v(x) dS(x)            (term3')
             + int_Gamma int_Gamma 1/(2 mu) (phi(x) - phi(y)) M(x, y) (v(x) - v(y)) dS(x) dS(y)  (term4')
             
   l(I, v): coil_term:     coil contribution
   term1:   diff_operator: diffusion integrator
   term2:   plasma_term:   nonlinear contribution from plasma
   term3:   
   term4:   
   term1':  diff_operator:      diffusion integrator (shared with term1)
   term2':  diff_plasma_term_i: derivative of nonlinear contribution from plasma (i=1,2,3)
   term3':  
   term4':

   Mesh attributes:
   831:  r=0 boundary
   900:  far-field boundary
   1000: limiter
   2000: exterior
   everything else: coils

   TODO: double boundary integral
*/

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <set>

using namespace std;
using namespace mfem;

int test();
double one_over_r_mu(const Vector & x, double & mu);

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
  Coefficient for diffusion integrator.
*/
class DiffusionIntegratorCoefficient : public Coefficient
{
private:
  // double mu;
  PlasmaModel *model;
public:
  // DiffusionIntegratorCoefficient(double & mu_) : mu(mu_) { }
  DiffusionIntegratorCoefficient(PlasmaModel *model_) : model(model_) { }
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~DiffusionIntegratorCoefficient() { }
};

/*
  Used to test saddle point calculator
 */
class TestCoefficient : public Coefficient
{
public:
  TestCoefficient() { }
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~TestCoefficient() { }
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
  const PlasmaModel *model;
  int option;
public:
  NonlinearGridCoefficient(const PlasmaModel *pm, int option_) : model(pm), option(option_) { }
  void set_grid_function(const GridFunction *psi_) {psi = psi_;}
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
  virtual ~NonlinearGridCoefficient() { }
};


int main(int argc, char *argv[])
{
   // Parse command line options.
   const char *mesh_file = "meshes/gs_mesh.msh";
   int order = 1;

   // constants associated with plasma model
   double alpha = 2.0;
   double beta = 1.0;
   double lambda = 1.0;
   double gamma = 2.0;
   double mu = 1.0;
   double r0 = 1.0;

   int attr_r_eq_0_bdr = 831;
   int attr_ff_bdr = 900;
   int attr_lim = 1000;
   int attr_ext = 2000;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&mu, "-mu", "--magnetic_permeability", "Magnetic permeability of a vaccuum");
   args.ParseCheck();

   // save options in model
   PlasmaModel model(alpha, beta, lambda, gamma, mu, r0);
   
   // unit tests
   test();

   // Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   mesh.UniformRefinement();

   // Define a finite element space on the mesh. Here we use H1 continuous
   // high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // Extract the list of all the boundary DOFs. These will be marked as
   // Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   Array<int> bdr_attribs(mesh.bdr_attributes);
   Array<int> ess_bdr(bdr_attribs.Max());
   ess_bdr = 1;
   ess_bdr[attr_ff_bdr-1] = 0;
   fespace.GetEssentialTrueDofs(ess_bdr, boundary_dofs, 1);
   
   // Define the solution x as a finite element grid function in fespace. Set
   // the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // Set up the contribution from the coils
   LinearForm coil_term(&fespace);
   // these are the unique element attributes used by the mesh
   Array<int> attribs(mesh.attributes);
   Vector coil_current(attribs.Max());
   coil_current = 0.0;
   for (int i = 0; i < attribs.Size(); ++i) {
     int attrib = attribs[i];
     if (attrib == 2000) {
       // exterior domain
     } else if (attrib == 1000) {
       // limiter domain
     } else {
       // coil domain
       coil_current(attrib-1) = 1.0;
     }
   }
   PWConstCoefficient coil_current_pw(coil_current);
   coil_term.AddDomainIntegrator(new DomainLFIntegrator(coil_current_pw));
   coil_term.Assemble();

   // Set up the bilinear form diff_operator corresponding to the diffusion integrator
   DiffusionIntegratorCoefficient diff_op_coeff(&model);
   BilinearForm diff_operator(&fespace);
   diff_operator.AddDomainIntegrator(new DiffusionIntegrator(diff_op_coeff));

   // GreenCoefficient green(greenfunc);
   // LinearForm fi(&fespace);
   // fi.AddBoundaryIntegrator(new BoundaryLFIntegrator(green));

   // ConstantCoefficient coeff(1.0);
   // YFixSurfaceCoefficient coeff(&fespace);
   // diff_operator.AddBoundaryIntegrator(new MassIntegrator(coeff));
   diff_operator.Assemble();

   // boundary integral
   // https://en.cppreference.com/w/cpp/experimental/special_functions
   

   // Form the linear system A X = B. This includes eliminating boundary
   // conditions, applying AMR constraints, and other transformations.
   SparseMatrix A;
   Vector B, X;
   diff_operator.FormLinearSystem(boundary_dofs, x, coil_term, A, X, B);

   // Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
   diff_operator.RecoverFEMSolution(X, coil_term, x);

   // plasma term
   NonlinearGridCoefficient nlgcoeff1(&model, 1);
   nlgcoeff1.set_grid_function(&x);
   LinearForm plasma_term(&fespace);
   plasma_term.AddDomainIntegrator(new DomainLFIntegrator(nlgcoeff1));
   plasma_term.Assemble();

   // derivative of plasma term
   NonlinearGridCoefficient nlgcoeff2(&model, 2);
   nlgcoeff2.set_grid_function(&x);
   BilinearForm diff_plasma_term(&fespace);
   diff_plasma_term.AddDomainIntegrator(new MassIntegrator(nlgcoeff2));
   diff_plasma_term.Assemble();
   
   x.Save("sol.gf");
   mesh.Save("mesh.mesh");

   //
   GridFunction y(&fespace);
   y= 0.0;
   diff_operator.Mult(x, y);
   y.Save("y.gf");

   //
   TestCoefficient tcoeff;
   GridFunction z(&fespace);
   z.ProjectCoefficient(tcoeff);
   z.Save("z.gf");

   /*
     Test vertex to vertex mapping...
    */

   Vector nval;
   z.GetNodalValues(nval);
   map<int, vector<int>> vertex_map;

   DSTable v_to_v(mesh.GetNV());
   int stop = 0;
   // get vertex to vertex mapping
   for (int i = 0; i < mesh.GetNE(); i++) {
     const int *v = mesh.GetElement(i)->GetVertices();
     const int ne = mesh.GetElement(i)->GetNEdges();
     for (int j = 0; j < ne; j++) {
       const int *e = mesh.GetElement(i)->GetEdgeVertices(j);
       v_to_v.Push(v[e[0]], v[e[1]]);

       vertex_map[v[e[0]]].push_back(v[e[1]]);
       
       // cout << v[e[0]] << " : " << v[e[1]] << endl;
       // cout << nval[v[e[0]]] << " : " << nval[v[e[1]]] << endl;
       // stop++;
     }
     // if (stop > 10) {
     //   break;
     // }
   }

   int count = 0;
   // loop through vertices and check neighboring vertices to see if we found a saddle point
   for(int iv = 0; iv < mesh.GetNV(); ++iv) {
     vector<int> adjacent = vertex_map[iv];
     int j = 0;
     double* x0 = mesh.GetVertex(iv);
     double* a = mesh.GetVertex(adjacent[j]);

     // cout << pow(x0[0] - 1.0, 2.0) + pow(x0[1], 2.0) << endl;
     // cout << nval[iv] << endl;
     // cout << x0[0] << ", " << x0[1] << endl;
     map<double, double> clock;
     set<double> ordered_angs;
     for (j = 0; j < adjacent.size(); ++j) {
       int jv = adjacent[j];
       double* b = mesh.GetVertex(jv);
       double diff = nval[jv] - nval[iv];
       // cout << b[0] << ", " << b[1] << endl;

       double ax = a[0]-x0[0];
       double ay = a[1]-x0[1];
       double bx = b[0]-x0[0];
       double by = b[1]-x0[1];

       double ang = atan2(by, bx);
       // double ang = asin( (ax*by-bx*ay)
       //                    / (sqrt(pow(ax,2.0)+pow(ay,2.0))
       //                       * sqrt(pow(bx,2.0)+pow(by,2.0))) );
       clock[ang] = diff;
       ordered_angs.insert(ang);
     }

     int sign_changes = 0;
     set<double>::iterator it = ordered_angs.begin();
     double init = clock[*it];
     double prev = clock[*it];
     // cout << *it << " : " << clock[*it] << endl;
     ++it;
     for (; it != ordered_angs.end(); ++it) {
       if (clock[*it] * prev < 0.0) {
         ++sign_changes;
       }
       prev = clock[*it];
       // cout << *it << " : " << clock[*it] << endl;
     }
     if (prev * init < 0.0) {
       ++sign_changes;
     }
     // cout << "sign changes: " << sign_changes << endl;
     // cout << endl;

     if (sign_changes >= 4) {
       printf("Found saddle at (%9.6f, %9.6f)\n", x0[0], x0[1]);
       ++count;
     }
 
     // stop++;
     // if (stop > 5) {
     //   break;
     // }
   }

   cout << "total saddles: " << count << endl;
   int key = 62198;
   vector<int> adjacent = vertex_map[key];

   // cout << key;
   // for (int j = 0; j < adjacent.size(); ++j)
   //   {
   //     cout << " " << adjacent[j];
   //   }
   // cout << endl;

   

   int i;
   for (DSTable::RowIterator it(v_to_v, i); !it; ++it) {
     it.Column();
     // J_v2v.Append(Pair<int,int>(it.Column(), it.Index()));
   }
   
   // cout << "rows: " << v_to_v.NumberOfRows() << endl;
   // cout << "entries: " << v_to_v.NumberOfEntries() << endl;

   

   // Table* vertex_map = mesh.GetVertexToElementTable();
   // for (int i = 0; i < vertex_map->Size(); ++i) {
   //   int num_elems = vertex_map->RowSize(i);
   //   const int *elems = vertex_map->GetRow(i);
   //   cout << i << " ";
   //   for (int j = 0; j < num_elems; ++j) {
   //     cout << elems[j] << " ";

   //     Array<int> vertices;
   //     fespace.GetElementVertices(j, vertices);
       
   //   }
   //   cout << endl;
   // }
   
   // map<int, vector<int>> vertex_map;
   // // vector<int> empty;
   // // vertex_map[1] = new vector<int>;
   // // vertex_map[1] = 3;
   // // vertex_map[1].push_back(3);
   // // vertex_map[1].push_back(4);
   // // cout << vertex_map[1] << endl;
   // // cout << vertex_map[1][0] << endl;
   
   
   // // ElementTransformation *Trans;
   // const FiniteElement *fe;
   // ElementTransformation *transf;
   // Vector shape;
   // Array<int> vdofs;
   // int fdof, d, i, intorder, j, k;
   // for (i = 0; i < fespace.GetNE(); i++)
   //   {
   //    fe = fespace.GetFE(i);
   //    fdof = fe->GetDof();
   //    transf = fespace.GetElementTransformation(i);
   //    shape.SetSize(fdof);
   //    intorder = 2*fe->GetOrder() + 3; // <----------
   //    const IntegrationRule *ir;
   //    fespace.GetElementVDofs(i, vdofs);

   //    Array<double> nval;
   //    z.GetNodalValues(i, nval);

   //    Array<int> vert;
   //    mesh.GetFaceVertices(i, vert);
   //    if (vert.Size() == 3) {
   //      vertex_map[vert[0]].push_back(vert[1]);
   //      vertex_map[vert[0]].push_back(vert[2]);
   //      vertex_map[vert[1]].push_back(vert[0]);
   //      vertex_map[vert[1]].push_back(vert[2]);
   //      vertex_map[vert[2]].push_back(vert[0]);
   //      vertex_map[vert[2]].push_back(vert[1]);
        
   //    }
   //    // ir = &(IntRules.Get(fe->GetGeomType(), intorder));
   //    // for (j = 0; j < ir->GetNPoints(); j++)
   //    // {
   //    //    const IntegrationPoint &ip = ir->IntPoint(j);
         
   //    // }
   //   }

   // for(map<int, vector<int>>::iterator iter = vertex_map.begin(); iter != vertex_map.end(); ++iter)
   //   {
   //     int key = iter->first;
   //     vector<int> adjacent = iter->second;

   //     cout << key;
   //     for (int j = 0; j < adjacent.size(); ++j)
   //       {
   //         cout << adjacent[j];
   //       }
   //     cout << endl;
   //   }
   

   //     Trans = fespace.GetElementTransformation(i);
   //     const IntegrationRule *ir;
   //     int order = Trans->OrderGrad(&fe) + Trans->Order() + fe.GetOrder();
   //     ir = &IntRules.Get(fe.GetGeomType(), order);
   //     double x_[3];
   //     Vector x(x_, 3);       
   //     Trans->Transform(ir, x);

   //     double x1(x(0));
   //     double x2(x(1));

   //     const int *pd = elem_pdof.GetRow(i);

   //     pow(x1, 2.0) * pow(x2, 3.0);
   //   }
   

   

   return 0;
}

double PlasmaModel::S_p_prime(double & psi_N) const
{
  return lambda * beta * pow(1.0 - pow(psi_N, alpha), gamma) / r0;
}
double PlasmaModel::S_prime_p_prime(double & psi_N) const
{
  return - alpha * gamma * lambda * beta
    * pow(1.0 - pow(psi_N, alpha), gamma - 1.0)
    * pow(psi_N, alpha - 1.0) / r0;
}
double PlasmaModel::S_ff_prime(double & psi_N) const
{
  return lambda * (1.0 - beta) * mu0 * r0 * pow(1.0 - pow(psi_N, alpha), gamma);
}
double PlasmaModel::S_prime_ff_prime(double & psi_N) const
{
  return - alpha * gamma * lambda * (1.0 - beta) * mu0 * r0
    * pow(1.0 - pow(psi_N, alpha), gamma - 1.0)
    * pow(psi_N, alpha - 1.0);
}
double normalized_psi(double & psi, double & psi_max, double & psi_bdp)
{
  return max(0.0,
             min(1.0,
                 (psi - psi_max) / (psi_bdp - psi_max)));
}


int test()
{
  double a = 1.0;
  // double b = S_p_prime(a);
  // cout << "a" << a << "b" << b << endl;
  return 1;
}

double NonlinearGridCoefficient::Eval(ElementTransformation & T,
                                      const IntegrationPoint & ip)
{
  double psi_val;
  Mesh *gf_mesh = psi->FESpace()->GetMesh();
  int Component = 1;
  if (T.mesh == gf_mesh)
    {
      psi_val = psi->GetValue(T, ip, Component);
    }
  else
    {
      cout << "problem!!!" << endl;
      // IntegrationPoint coarse_ip;
      // ElementTransformation *coarse_T = RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);
      // psi_val = psi->GetValue(*coarse_T, coarse_ip, Component);
      psi_val = 1.0;
    }

   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);

   double ri(x(0));
   // todo, need to compute these
   double psi_max = 1.0;
   double psi_bdp = 0.0;
   double psi_N = normalized_psi(psi_val, psi_max, psi_bdp);

   if (option == 1) {
     return ri * (model->S_p_prime(psi_N)) + (model->S_ff_prime(psi_N)) / (model->get_mu() * ri);
   } else {
     double coeff = 1.0;
     if (option == 2) {
       coeff = 1.0 / (psi_bdp - psi_max);
     } else if (option == 3) {
       coeff = - (1 - psi_N) / (psi_bdp - psi_max);
     } else if (option == 4) {
       coeff = - psi_N / (psi_bdp - psi_max);
     }
     
     return coeff * (ri * (model->S_prime_p_prime(psi_N))
                     + (model->S_prime_ff_prime(psi_N)) / (model->get_mu() * ri));
   }
}


double DiffusionIntegratorCoefficient::Eval(ElementTransformation & T,
                                            const IntegrationPoint & ip)
{
   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);
   double ri(x(0));

   if (T.Attribute == 900) {

     cout << "asdf" << endl;
   }
   return 1.0 / (ri * model->get_mu());
}

double TestCoefficient::Eval(ElementTransformation & T,
                             const IntegrationPoint & ip)
{
   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);
   double x1(x(0));
   double x2(x(1));
   double k = 4.0;
   
   // return pow(x1, 2.0) * pow(x2, 3.0);
   return cos(k * x1) * cos(k * x2) * exp(- pow(x1, 2.0) - pow(x2, 2.0));
   // return pow(x1 - 1.0, 2.0) + pow(x2, 2.0);
   // return pow(x1 - 1.0, 2.0) - pow(x2, 2.0);
}
