
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

double sx, sy;

Vector vel;

void Prefine(FiniteElementSpace & fes_old,
             GridFunction &u, Coefficient &gf_ex, GridFunction &orders_gf,
             double min_thresh, double max_thresh);

void Hrefine(GridFunction &u, Coefficient &gf_ex, double min_thresh,
             double max_thresh);

// void Hrefine2(GridFunction &u, Coefficient &gf_ex, double min_thresh,
//   double max_thresh);
Table * Hrefine2(GridFunction &u, GridFunction &u_ref, Table * refT,
                 Coefficient &gf_ex, double min_thresh,
                 double max_thresh);

Table * Refine(Array<int> ref_actions, GridFunction &u, int depth_limit = 100);

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x, double);

// Inflow boundary condition
double inflow_function(const Vector &x);

class FE_Evolution : public TimeDependentOperator
{
private:
   BilinearForm &M, &K;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;

   mutable Vector z;

public:
   FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_);
   void Update();
   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution();
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 1;
   sx = 1.0;
   sy = 1.0;
   double t_final = 1.0;
   double dt = 0.002;
   bool visualization = true;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&sx, "-sx", "--sx",
                  "mesh length in x direction");
   args.AddOption(&sy, "-sy", "--sy",
                  "mesh length in y direction");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh0 = Mesh::MakeCartesian2D(16, 16, mfem::Element::QUADRILATERAL,false,
                                      sx,
                                      sy);


   std::vector<Vector> translations = {Vector({sx,0.0}), Vector({0.0,sy})};


   Mesh mesh = Mesh::MakePeriodic(mesh0,
                                  mesh0.CreatePeriodicVertexMapping(translations));


   mesh.EnsureNCMesh();

   // compute reference solution
   Mesh ref_mesh(mesh);



   int dim = mesh.Dimension();

   ODESolver *ode_solver = new RK4Solver;
   ODESolver *ref_ode_solver = new RK4Solver;


   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);
   FiniteElementSpace ref_fes(&ref_mesh, &fec);
   FiniteElementSpace fes_old(&mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm m(&fes);
   BilinearForm k(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   constexpr double alpha = -1.0;
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
   k.AddInteriorFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));
   k.AddBdrFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, alpha,-0.5));

   m.Assemble();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   b.Assemble();
   m.Finalize();
   k.Finalize(skip_zeros);

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u0.SetTime(0.);
   u.ProjectCoefficient(u0);

   // reference solution
   BilinearForm m_ref(&ref_fes);
   BilinearForm k_ref(&ref_fes);
   m_ref.AddDomainIntegrator(new MassIntegrator);
   k_ref.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
   k_ref.AddInteriorFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));
   k_ref.AddBdrFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));

   LinearForm b_ref(&ref_fes);
   b_ref.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, alpha,-0.5));

   m_ref.Assemble();
   k_ref.Assemble(skip_zeros);
   b_ref.Assemble();
   m_ref.Finalize();
   k_ref.Finalize(skip_zeros);

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u_ref(&ref_fes);
   u_ref.ProjectCoefficient(u0);


   Array<int> refinements(ref_mesh.GetNE());
   refinements = 1;

   Table * T1 = Refine(refinements, u_ref, 2);
   refinements.SetSize(ref_mesh.GetNE());
   refinements = 1;
   Table * T2 = Refine(refinements, u_ref, 2);

   Table * refT = Mult(*T1,*T2);

   m_ref.Update();
   m_ref.Assemble();
   m_ref.Finalize();
   k_ref.Update();
   k_ref.Assemble(skip_zeros);
   k_ref.Finalize(skip_zeros);
   b_ref.Update();
   b_ref.Assemble();


   L2_FECollection orders_fec(0,dim);
   FiniteElementSpace orders_fes(&mesh,&orders_fec);
   GridFunction orders_gf(&orders_fes);
   for (int i = 0; i<mesh.GetNE(); i++) { orders_gf(i) = order; }




   socketstream sout;
   // socketstream s_refout;
   // socketstream meshout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      // s_refout.open(vishost, visport);
      // meshout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << mesh << u;
         sout << flush;
         // s_refout.precision(precision);
         // s_refout << "solution\n" << ref_mesh << u_ref;
         // s_refout << flush;


         // meshout.precision(precision);
         // meshout << "solution\n" << mesh << orders_gf;
         // meshout << "mesh\n" << mesh ;
         // meshout << flush;
         // cin.get();
      }
   }


   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(m, k, b);
   FE_Evolution ref_adv(m_ref, k_ref, b_ref);

   double t = 0.0;
   adv.SetTime(t);
   double ref_t = 0.0;
   ref_adv.SetTime(ref_t);

   FunctionCoefficient u_ex(u0_function);

   ode_solver->Init(adv);
   ref_ode_solver->Init(ref_adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(u, t, dt_real);
      ref_ode_solver->Step(u_ref, ref_t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         u_ex.SetTime(t);
         // Prefine(fes_old,u,u_ex, orders_gf, 5e-5, 5e-4);

         mfem::out << "Global L2 Error = " << u.ComputeL2Error(u_ex) << std::endl;
         // Hrefine2(u,u_ex, 5e-5, 5e-4);
         // mfem::out << "refT size = " << refT->Size() << " x " << refT->Width() << endl;
         refT = Hrefine2(u,u_ref, refT, u_ex, 5e-5, 5e-4);
         // mfem::out << "refT size = " << refT->Size() << " x " << refT->Width() << endl;
         // mfem::out << "number of elements = " << mesh.GetNE() << endl;


         m.Update();
         m.Assemble();
         m.Finalize();
         k.Update();
         k.Assemble(skip_zeros);
         k.Finalize(skip_zeros);
         b.Update();
         b.Assemble();
         adv.Update();
         ode_solver->Init(adv);


         // ref_adv.Update();
         // ref_ode_solver->Init(ref_adv);
         if (visualization)
         {
            // GridFunction gf_ex(&fes);
            // gf_ex.ProjectCoefficient(u_ex);
            GridFunction * pr_u = ProlongToMaxOrder(&u);
            sout << "solution\n" << mesh << *pr_u << flush;
            // s_refout << "solution\n" << ref_mesh << u_ref << flush;

            // meshout << "solution\n" << mesh << orders_gf << flush;
            // meshout << "mesh\n" << mesh << flush;
         }
      }
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete ref_ode_solver;

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_)
   : TimeDependentOperator(M_.Height()), M(M_), K(K_), b(b_), z(M_.Height())
{
   Array<int> ess_tdof_list;
   M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetOperator(M);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Update()
{
   height = M.Height();
   width = M.Width();
   z.SetSize(M.Height());

   Array<int> ess_tdof_list;
   delete M_prec;
   M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetOperator(M);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

FE_Evolution::~FE_Evolution()
{
   delete M_prec;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   v.SetSize(2);
   v(0) = 1.;
   v(1) = 1.;
}

// Initial condition
double u0_function(const Vector &x, double t)
{
   // give x0, y0;

   // Rotation matrix
   double theta = M_PI/4;
   //

   double x0 = 0.5;
   double y0 = 0.5;
   double w = 100.;
   double c = 1.;
   double ds = c*t;
   Vector a(2);
   a(0) = cos(theta);
   a(1) = sin(theta);
   // double xx = x(0) - a(0)*ds;
   // double yy = x(1) - a(1)*ds;
   double xx = x(0) - ds;
   double yy = x(1) - ds;

   double tol = 1e-6;
   if (xx>= sx+tol || xx<= 0.0-tol)
   {
      xx -= floor(xx/sx) * sx;
   }
   if (yy>= sy+tol || yy<= 0.0-tol)
   {
      yy -= floor(yy/sy) * sy;
   }



   // double d = (xx-x0)*a(0) + (yy-y0)*a(1);
   // double d1 = (xx-x0-0.5)*a(0) + (yy-y0-0.5)*a(1);
   // double d2 = (xx-x0+0.5)*a(0) + (yy-y0+0.5)*a(1);
   // return 1. + exp(-w*(d*d)) + exp(-w*(d1*d1)) + exp(-w*(d2*d2));
   double dr_x = (xx-x0)*(xx-x0);
   double dr_y = (yy-y0)*(yy-y0);
   return 1. + exp(-w*(dr_x+dr_y));
   // return 1. + exp(-w*(dr_x));
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   return 1.0;
}


void Prefine(FiniteElementSpace & fes_old,
             GridFunction &u, Coefficient &ex, GridFunction &orders_gf,
             double min_thresh, double max_thresh)
{
   // get element errors
   FiniteElementSpace * fes = u.FESpace();
   int ne = fes->GetMesh()->GetNE();
   Vector errors(ne);
   u.ComputeElementL2Errors(ex,errors);
   for (int i = 0; i<ne; i++)
   {
      double error = errors(i);
      int order = fes->GetElementOrder(i);
      if (error < min_thresh && order > 1)
      {
         fes->SetElementOrder(i,order-1);
      }
      else if (error > max_thresh && order < 2)
      {
         fes->SetElementOrder(i, order+1);
      }
      else
      {
         // do nothing
      }
   }

   fes->Update(false);

   PRefinementTransferOperator * T = new PRefinementTransferOperator(fes_old,*fes);

   GridFunction u_fine(fes);
   T->Mult(u,u_fine);

   // copy the orders to the old space
   for (int i = 0; i<ne; i++)
   {
      int order = fes->GetElementOrder(i);
      fes_old.SetElementOrder(i,order);
      orders_gf(i) = order;
   }
   fes_old.Update(false);

   delete T;

   // update old gridfuntion;
   u = u_fine;

}

// void Hrefine2(GridFunction &u, Coefficient & ex_coeff, double min_thresh,
//               double max_thresh)
Table * Hrefine2(GridFunction &u, GridFunction &u_ref, Table * refT,
                 Coefficient & ex_coeff, double min_thresh,
                 double max_thresh)
{
   FiniteElementSpace * fes = u.FESpace();
   Mesh * mesh = fes->GetMesh();
   int ne = mesh->GetNE();
   Vector errors(ne);
   // u.ComputeElementL2Errors(ex_coeff,errors);

   // copy the fespace, refine it up to element depth 2 and calculate the errors
   // copy mesh
   Mesh fine_mesh(*mesh);

   FiniteElementSpace fes_copy(&fine_mesh,fes->FEColl());
   GridFunction u_fine(&fes_copy);
   // copy data;
   u_fine = u;
   Array<int>refinements(fine_mesh.GetNE());
   refinements = 1;
   Table * T1 = Refine(refinements,u_fine,2);
   refinements.SetSize(fine_mesh.GetNE());
   refinements = 1;
   Table * T2 = Refine(refinements,u_fine,2);
   Table * T = Mult(*T1, *T2);

   delete T1;
   delete T2;

   // constract map
   int n = T->Size();
   int m = T->Width();
   Array<int> elem_map(m);
   for (int i = 0; i< n; i++)
   {
      int nr = T->RowSize(i);
      int * row = T->GetRow(i);
      int * ref_row = refT->GetRow(i);
      for (int j = 0; j<nr ; j++ )
      {
         elem_map[row[j]] = ref_row[j];
      }
   }


   // Table *fine2refT = Transpose(*Mult(*Transpose(*T), *refT));


   // char vishost[] = "localhost";
   // int  visport   = 19916;
   // socketstream pr_out(vishost, visport);
   // pr_out << "solution\n" << fine_mesh << u_fine << flush;

   // calculate error

   GridFunction diff(u_fine);
   // this needs to change for reordering
   diff-= u_ref;

   ConstantCoefficient zero(0.0);
   Vector fine_errors(fine_mesh.GetNE());
   diff.ComputeElementL2Errors(zero,fine_errors);

   // combine fine errors to current mesh;
   // Table *Tt = Transpose(*T);
   // mfem::out << "Tt->Size = " << Tt->Size() << endl;
   // mfem::out << "errors = " << errors.Size() << endl;
   for (int i = 0; i<T->Size(); i++)
   {
      int m = T->RowSize(i);
      int *row = T->GetRow(i);
      double err = 0.;
      for (int j = 0; j<m; j++)
      {
         err += fine_errors[row[j]]*fine_errors[row[j]];
      }
      errors[i] = sqrt(err);
   }



   // cin.get();

   // compute element errors by
   // 1. refine the mesh up to mesh limit 2
   // 2. Prolongate the current solution to the refined mesh
   // 3. Calculate errors and combine them (for the coarse elements)
   // 4. Derifine mesh

   //copy the mesh
   // Mesh * ref_mesh = new Mesh(*mesh);
   // Array<int> ref_actions(ne);
   // ref_actions = 1;
   // NCMesh * ref_ncmesh = ref_mesh->ncmesh;




   Array<int> actions(ne);
   for (int i = 0; i<ne; i++)
   {
      double error = errors(i);
      if (error > max_thresh)
      {
         actions[i] = 1;
      }
      else if (error < min_thresh)
      {
         actions[i] = -1;
      }
      else
      {
         actions[i] = 0;
      }
   }

   Table * T3 = Refine(actions,u,1);


   if (T3)
   {
      Table *Ttt = Mult(*Transpose(*T3), *refT);
      delete T3;
      delete refT;
      refT = Ttt;
   }

   return refT;

   // construct a list of possible ref actions
   // Array<int> actions(ne);
   // for (int i = 0; i<ne; i++)
   // {
   //    double error = errors(i);
   //    if (error > max_thresh && mesh->ncmesh->GetElementDepth(i) < 1)
   //    {
   //       actions[i] = 1;
   //    }
   //    else
   //    {
   //       actions[i] = 0;
   //    }
   // }

   // // list of possible dref actions
   // Array<int> derefactions(ne); derefactions = 0;
   // const Table & dref_table = mesh->ncmesh->GetDerefinementTable();
   // for (int i = 0; i<dref_table.Size(); i++)
   // {
   //    int size = dref_table.RowSize(i);
   //    const int * row = dref_table.GetRow(i);
   //    double error = 0.;
   //    for (int j = 0; j<size; j++)
   //    {
   //       error += errors[row[j]];
   //    }
   //    if (error < min_thresh)
   //    {
   //       for (int j = 0; j<size; j++)
   //       {
   //          actions[row[j]] += -1;
   //       }
   //    }
   // }

   // // now refine the elements that have score >0 and deref the elements that have score < 0
   // Array<Refinement> elements_to_refine;
   // for (int i = 0; i<ne; i++)
   // {
   //    if (actions[i] > 0)
   //    {
   //       elements_to_refine.Append(Refinement(i,0b01));
   //    }
   // }

   // mesh->GeneralRefinement(elements_to_refine);
   // fes->Update();
   // u.Update();

   // // map old actions to new mesh
   // Array<int> new_actions(mesh->GetNE());
   // if (mesh->GetLastOperation() == mesh->REFINE)
   // {
   //    const CoarseFineTransformations &tr = mesh->GetRefinementTransforms();
   //    Table coarse2fine;
   //    tr.MakeCoarseToFineTable(coarse2fine);
   //    new_actions = 1;
   //    for (int i = 0; i<coarse2fine.Size(); i++)
   //    {
   //       if (coarse2fine.RowSize(i) == 1)
   //       {
   //          int * el = coarse2fine.GetRow(i);
   //          new_actions[el[0]] = actions[i];
   //       }
   //    }
   // }
   // else
   // {
   //    new_actions = actions;
   // }

   // // create a dummy error vector
   // Vector new_errors(mesh->GetNE());
   // new_errors = infinity();
   // for (int i = 0; i< new_errors.Size(); i++)
   // {
   //    if (new_actions[i] < 0)
   //    {
   //       new_errors[i] = 0.;
   //    }
   // }

   // // any threshold would do here
   // mesh->DerefineByError(new_errors,min_thresh);

   // fes->Update();
   // u.Update();
}


Table * Refine(Array<int> ref_actions, GridFunction &u, int depth_limit)
{
   FiniteElementSpace * fes = u.FESpace();
   Mesh * mesh = fes->GetMesh();
   int ne = mesh->GetNE();


   //  ovewrite to no action if an element is marked for refinement but it exceeds the depth limit
   for (int i = 0; i<ne; i++)
   {
      int depth = mesh->ncmesh->GetElementDepth(i);
      if (depth >= depth_limit && ref_actions[i] == 1)
      {
         ref_actions[i] = 0;
      }
   }

   // current policy to map agent_actions to actions
   // 1. All elements that are marked for refinement are to perform the refinement
   // 2. All of the "siblings" (i) of a marked element for refinement are assigned action=max(0,agent_actions[i])
   //   i.e., a) if the action is to be refined then they are refined
   //         b) if the action is to be derefined or no action then they get no action
   // 3. If among the "siblings" there is no refinement action then the group is marked
   //    for derefinement if the majority (including a tie) of the siblings are marked for derefinement
   //    otherwise they are marked for no action
   //  h-refine:   action =  1
   //  h-derefine: action = -1
   //  do nothing: action =  0

   Array<int> actions(ne);
   Array<int> actions_marker(ne);
   actions_marker = 0;

   const Table & deref_table = mesh->ncmesh->GetDerefinementTable();

   for (int i = 0; i<deref_table.Size(); i++)
   {
      int n = deref_table.RowSize(i);
      const int * row = deref_table.GetRow(i);
      int sum_of_actions = 0;
      bool ref_flag = false;
      for (int j = 0; j<n; j++)
      {
         int action = ref_actions[row[j]];
         sum_of_actions+=action;
         if (action == 1)
         {
            ref_flag = true;
            break;
         }
      }
      if (ref_flag)
      {
         for (int j = 0; j<n; j++)
         {
            actions[row[j]] = max(0,ref_actions[row[j]]);
            actions_marker[row[j]] = 1;
         }
      }
      else
      {
         bool dref_flag = (2*abs(sum_of_actions) >= n) ? true : false;
         for (int j = 0; j<n; j++)
         {
            actions[row[j]] = (dref_flag) ? -1 : 0;
            actions_marker[row[j]] = 1;
         }
      }
   }

   for (int i = 0; i<ne; i++)
   {
      if (actions_marker[i] != 1)
      {
         if (ref_actions[i] == -1)
         {
            actions[i] = 0;
         }
         else
         {
            actions[i] = ref_actions[i];
         }
      }
   }

   // now the actions array holds feasible actions of -1,0,1
   Array<Refinement> refinements;
   for (int i = 0; i<ne; i++)
   {
      if (actions[i] == 1) {refinements.Append(Refinement(i,0b11));}
   }
   if (refinements.Size())
   {
      mesh->GeneralRefinement(refinements);
      fes->Update();
      u.Update();
      ne = mesh->GetNE();
   }

   Table * ref_table = nullptr;
   Table * dref_table = nullptr;
   // now the derefinements
   Array<int> new_actions(ne);
   if (refinements.Size())
   {
      new_actions = 1;
      const CoarseFineTransformations & tr = mesh->GetRefinementTransforms();
      ref_table = new Table();
      tr.MakeCoarseToFineTable(*ref_table);
      for (int i = 0; i<ref_table->Size(); i++)
      {
         int n = ref_table->RowSize(i);
         if (n == 1)
         {
            int * row = ref_table->GetRow(i);
            new_actions[row[0]] = actions[i];
         }
      }
   }
   else
   {
      new_actions = actions;
   }

   Vector dummy_errors(ne);
   dummy_errors = 1.0;
   for (int i = 0; i<ne; i++)
   {
      if (new_actions[i] < 0)
      {
         dummy_errors[i] = 0.;
      }
   }
   mesh->DerefineByError(dummy_errors,0.5);

   fes->Update();
   u.Update();

   if (mesh->GetNE() < ne)
   {
      const CoarseFineTransformations & tr =
         mesh->ncmesh->GetDerefinementTransforms();
      Table coarse_to_fine_table;
      tr.MakeCoarseToFineTable(coarse_to_fine_table);
      dref_table = Transpose(coarse_to_fine_table);
   }

   // Build combined table of mesh modifications
   Table * T = nullptr;
   if (ref_table && dref_table)
   {
      T = Mult(*ref_table, * dref_table);
      delete dref_table;
      delete ref_table;
   }
   else if (ref_table)
   {
      T = ref_table;
      delete dref_table;
   }
   else if (dref_table)
   {
      T= dref_table;
      delete ref_table;
   }
   else
   {
      // do nothing: no mesh modifications happened
   }

   return T;
}
