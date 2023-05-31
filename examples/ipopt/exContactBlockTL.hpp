//                                Contact example
//
// Compile with: make exContactBlockTL
//
// Sample runs:  ./exContactBlockTL -m1 block1.mesh -m2 block2.mesh -at "5 6 7 8"
// Sample runs:  ./exContactBlockTL -m1 block1_d.mesh -m2 block2_d.mesh -at "5 6 7 8"

#ifndef EXCONTACTBLOCKTL_HPP
#define EXCONTACTBLOCKTL_HPP

#include "mfem.hpp"
#include "IpTNLP.hpp"

using namespace std;
using namespace mfem;
using namespace Ipopt;


class ExContactBlockTL: public TNLP
{
public:
   /** default constructor */
   ExContactBlockTL(int argc, char *argv[]);

   /** default destructor */
   virtual ~ExContactBlockTL();

   /**@name Overloaded from TNLP */
   /** Method to return some info about the nlp */
   virtual bool get_nlp_info(
      Index&          n,
      Index&          m,
      Index&          nnz_jac_g,
      Index&          nnz_h_lag,
      IndexStyleEnum& index_style
   );

   /** Method to return the bounds for my problem */
   virtual bool get_bounds_info(
      Index   n,
      Number* x_l,
      Number* x_u,
      Index   m,
      Number* g_l,
      Number* g_u
   );

   /** Method to return the starting point for the algorithm */
   virtual bool get_starting_point(
      Index   n,
      bool    init_x,
      Number* x,
      bool    init_z,
      Number* z_L,
      Number* z_U,
      Index   m,
      bool    init_lambda,
      Number* lambda
   );

   /** Method to return the objective value */
   virtual bool eval_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number&       obj_value
   );

   /** Method to return the gradient of the objective */
   virtual bool eval_grad_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number*       grad_f
   );

   /** Method to return the constraint residuals */
   virtual bool eval_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Number*       cons
   );

   /** Method to return:
    *   1) The structure of the Jacobian (if "values" is NULL)
    *   2) The values of the Jacobian (if "values" is not NULL)
    */
   virtual bool eval_jac_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Index         nele_jac,
      Index*        iRow,
      Index*        jCol,
      Number*       values
   );

   /** Method to return:
    *   1) The structure of the Hessian of the Lagrangian (if "values" is NULL)
    *   2) The values of the Hessian of the Lagrangian (if "values" is not NULL)
    */
   virtual bool eval_h(
      Index         n,
      const Number* x,
      bool          new_x,
      Number        obj_factor,
      Index         m,
      const Number* lambda,
      bool          new_lambda,
      Index         nele_hess,
      Index*        iRow,
      Index*        jCol,
      Number*       values
   );

   /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
   virtual void finalize_solution(
      SolverReturn               status,
      Index                      n,
      const Number*              x,
      const Number*              z_L,
      const Number*              z_U,
      Index                      m,
      const Number*              g,
      const Number*              lambda,
      Number                     obj_value,
      const IpoptData*           ip_data,
      IpoptCalculatedQuantities* ip_cq
   );

private:
   void update_g();
   void update_jac();
   void update_hess();

private:
   /**@name Methods to block default compiler methods.
    *
    * The compiler automatically generates the following three methods.
    *  Since the default compiler implementation is generally not what
    *  you want (for all but the most simple classes), we usually
    *  put the declarations of these methods in the private section
    *  and never implement them. This prevents the compiler from
    *  implementing an incorrect "default" behavior without us
    *  knowing. (See Scott Meyers book, "Effective C++")
    */
   ExContactBlockTL(
      const ExContactBlockTL&
   );

   ExContactBlockTL& operator=(
      const ExContactBlockTL&
   );

   Array<int> attr;
   Array<int> m_attr;
   Array<int> s_conn; // connectivity of the second/slave mesh
   std::string mesh_file1;
   std::string mesh_file2;
   Mesh* mesh1;
   Mesh* mesh2;
   FiniteElementCollection* fec1;
   FiniteElementCollection* fec2;
   FiniteElementSpace* fespace1;
   FiniteElementSpace* fespace2;
   Array<int> ess_tdof_list1;
   Array<int> ess_tdof_list2;
   GridFunction nodes0;
   GridFunction* nodes1;
   GridFunction* nodes2;
   GridFunction* x1;
   GridFunction* x2;
   LinearForm* b1;
   LinearForm* b2;
   PWConstCoefficient* lambda1_func;
   PWConstCoefficient* lambda2_func;
   PWConstCoefficient* mu1_func;
   PWConstCoefficient* mu2_func;
   BilinearForm* a1;
   BilinearForm* a2;

   mfem::Vector lambda1;
   mfem::Vector lambda2;
   mfem::Vector mu1;
   mfem::Vector mu2;
   mfem::Vector xyz;

   std::set<int> bdryVerts2;

   int dim;
   // degrees of freedom of both meshes
   int ndof_1;
   int ndof_2;
   int ndofs;
   // number of nodes for each mesh
   int nnd_1;
   int nnd_2;
   int nnd;

   int npoints;

   SparseMatrix A1;
   mfem::Vector B1, X1;
   SparseMatrix A2;
   mfem::Vector B2, X2;

   SparseMatrix* K;
   mfem::Vector g;
   mfem::Vector m_xi;
   mfem::Vector xs;

   Array<int> m_conn; // only works for linear elements that have 4 vertices!
   DenseMatrix* coordsm;
   SparseMatrix* M;

   std::vector<SparseMatrix>* dM;

   Array<int> Dirichlet_dof;
   Array<double> Dirichlet_val;

public: 
   Mesh * GetMesh1() {return mesh1;}
   Mesh * GetMesh2() {return mesh2;}

};

#endif
