//                       MFEM Example 2 - Parallel Version
//
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder with f being a constant pull down
//               vector on boundary elements with attribute 2, and zero
//               otherwise. 
//Sample PUMI RUN
//mpirun -np 2 ./pumi_upright_ex2p -m ../../data/pumi/parallel/upright/2p5kg1.smb -p ../../data/pumi/geom/upright_defeatured_geomsim.smd -bf ../../data/pumi/serial/boundary_upright.mesh -ar 0.04
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

#include "../../general/text.hpp"

#ifdef MFEM_USE_SIMMETRIX
#include <SimUtil.h>
#include <gmi_sim.h>
#endif
#include <apfMDS.h>
#include <gmi_null.h>
#include <PCU.h>
#include <spr.h>
#include <apfConvert.h>
#include <gmi_mesh.h>
#include <crv.h>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   //initilize mpi
   int num_procs, myId;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myId);

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/pumi/parallel/upright/2p5kg1.smb";
   const char *boundary_file = "../../data/pumi/serial/boundary_upright.mesh";
#ifdef MFEM_USE_SIMMETRIX
   const char *model_file = "../../data/pumi/geom/upright_defeatured_geomsim.smd";
#else
   const char *model_file = "../../data/pumi/geom/pillbox.dmg";
#endif

   bool static_cond = false;
   bool visualization = 1;
   int geom_order = 1;
   int order = 1;
   bool amg_elast = 0;
   double adapt_ratio = 0.2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&amg_elast, "-elast", "--amg-for-elasticity", "-sys",
                  "--amg-for-systems",
                  "Use the special AMG elasticity solver (GM/LN approaches), "
                  "or standard AMG for systems (unknown approach).");  
   args.AddOption(&model_file, "-p", "--parasolid",
                  "Parasolid model to use.");
   args.AddOption(&geom_order, "-go", "--geometry_order",
                  "Geometric order of the model");
   args.AddOption(&boundary_file, "-bf", "--txt",
                  "txt file containing boundary tags");
   args.AddOption(&adapt_ratio, "-ar", "--adapt_ratio",
                  "adaptation factor used in MeshAdapt");   

   
   args.Parse();
   if (!args.Good())
   {
      if (myId == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myId == 0)
   {
      args.PrintOptions(cout);
   }   

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   // 3. Read the SCOREC Mesh
   PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
   Sim_readLicenseFile(0);
   gmi_sim_start();
   gmi_register_sim();
#endif
   gmi_register_mesh();

   apf::Mesh2* pumi_mesh;
   pumi_mesh = apf::loadMdsMesh(model_file, mesh_file);

   // 4. Increase the geometry order if necessary.
   if (geom_order > 1)
   {
      crv::BezierCurver bc(pumi_mesh, geom_order, 0);
      bc.run();
   }
   pumi_mesh->verify();
   

   //Read boundary
   string bdr_tags;
   named_ifgzstream input_bdr(boundary_file);
   input_bdr >> ws;
   getline(input_bdr, bdr_tags);
   filter_dos(bdr_tags);
   if (myId == 0) cout << " the boundary tag is : " << bdr_tags << endl;
   Array<int> Dirichlet;
   int numOfent;
   if (bdr_tags == "Dirichlet")
   {
      input_bdr >> numOfent;
      if (myId == 0) cout << " num of Dirirchlet bdr conditions : " << numOfent << endl;
      Dirichlet.SetSize(numOfent);
      for (int kk = 0; kk < numOfent; kk++)
      {
         input_bdr >> Dirichlet[kk];
      }
   }
   Dirichlet.Print();

   Array<int> load_bdr;
   skip_comment_lines(input_bdr, '#');
   input_bdr >> bdr_tags;
   filter_dos(bdr_tags);
   if (myId == 0) cout << " the boundary tag is : " << bdr_tags << endl;
   if (bdr_tags == "Load")
   {
      input_bdr >> numOfent;
      load_bdr.SetSize(numOfent);
      if (myId == 0) cout << " num of load bdr conditions : " << numOfent << endl;
      for (int kk = 0; kk < numOfent; kk++)
      {
         input_bdr >> load_bdr[kk];
      }
   }
   load_bdr.Print();

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   ParMesh *pmesh = new ParPumiMesh(MPI_COMM_WORLD, pumi_mesh);
   int dim = pumi_mesh->getDimension();

   //Hack for the boundary condition
   apf::MeshIterator* itr = pumi_mesh->begin(dim-1);
   apf::MeshEntity* ent ;
   int bdr_cnt = 0;
   while ((ent = pumi_mesh->iterate(itr)))
   {
      apf::ModelEntity *me = pumi_mesh->toModel(ent);
      if (pumi_mesh->getModelType(me) == (dim-1))
      {
         //Evrywhere 3 as initial
         (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(3);
         int tag = pumi_mesh->getModelTag(me);
         if (Dirichlet.Find(tag) != -1)
         {
            //Dirichlet attr -> 1
            (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(1);
         }
         else if (load_bdr.Find(tag) != -1)
         {
            //load attr -> 2
            (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(2);
         }
         bdr_cnt++;
      }
   }
   pumi_mesh->end(itr);

   //assign attr for elements
   double ppt[3];
   Vector cent(ppt, dim);
   for (int el = 0; el < pmesh->GetNE(); el++)
   {
      (pmesh->GetElementTransformation(el))->Transform(Geometries.GetCenter(
                                                         pmesh->GetElementBaseGeometry(el)),cent);
      if (cent(1) <= 0.01)
      {
         pmesh->SetAttribute(el , 1);
      }
      else if (cent(1) >= 0.1)
      {
         pmesh->SetAttribute(el , 2);
      }
      else
      {
         pmesh->SetAttribute(el , 3);
      }

   }
   pmesh->SetAttributes();

   cout << " elem attr max " << pmesh->attributes.Max() << " bdr attr max " <<
        pmesh->bdr_attributes.Max() <<endl;
   if (pmesh->attributes.Max() < 2 || pmesh->bdr_attributes.Max() < 2)
   {
      cerr << "\nInput mesh should have at least two materials and "
           << "two boundary attributes! (See schematic in ex2.cpp)\n"
           << endl;
      return 3;
   }
   

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use vector finite elements, i.e. dim copies of a scalar finite element
   //    space. We use the ordering by vector dimension (the last argument of
   //    the FiniteElementSpace constructor) which is expected in the systems
   //    version of BoomerAMG preconditioner. For NURBS meshes, we use the
   //    (degree elevated) NURBS space associated with the mesh nodes.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   const bool use_nodal_fespace = pmesh->NURBSext && !amg_elast;
   if (use_nodal_fespace)
   {
      fec = NULL;
      fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      fespace = new ParFiniteElementSpace(pmesh, fec, dim);//, Ordering::byVDIM
   }
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myId == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl
           << "Assembling: " << flush;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined by
   //    marking only boundary attribute 1 from the mesh as essential and
   //    converting it to a list of true dofs.
   //Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
   //ess_bdr = 0;
   //ess_bdr[0] = 1;
   //fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system. In this case, b_i equals the
   //    boundary integral of f*phi_i where f represents a "pull down" force on
   //    the Neumann part of the boundary and phi_i are the basis functions in
   //    the finite element fespace. The force is defined by the object f, which
   //    is a vector of Coefficient objects. The fact that f is non-zero on
   //    boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.
   VectorArrayCoefficient f(dim);
   f.Set(0, new ConstantCoefficient(0.0));
   f.Set(1, new ConstantCoefficient(0.0));
   f.Set(2, new ConstantCoefficient(0.0));   


   //ParLinearForm *b = new ParLinearForm(fespace);
   //b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   if (myId == 0)
   {
      cout << "r.h.s. ... " << flush;
   }
   //b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with piece-wise
   //     constants coefficient lambda and mu.
   Vector lambda(pmesh->attributes.Max());
   lambda = 1.0;
   lambda(0) = lambda(1)*50;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh->attributes.Max());
   mu = 1.0;
   mu(0) = mu(1)*50;
   PWConstCoefficient mu_func(mu);

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
   

    // 12. Assemble the parallel bilinear form and the corresponding linear
    //     system, applying any necessary transformations such as: parallel
    //     assembly, eliminating boundary conditions, applying conforming
    //     constraints for non-conforming AMR, static condensation, etc.
    if (myId == 0) { cout << "matrix ... " << flush; }
    if (static_cond) { a->EnableStaticCondensation(); }   
   
   apf::Field* Tmag_field = 0;
   apf::Field* temp_field = 0;
   apf::Field* ipfield = 0;
   apf::Field* sizefield = 0;     
   
   int max_iter = 4;

   for (int Itr = 0; Itr < max_iter; Itr++)
   {   

     a->Assemble();
        
     //Hack for the boundary condition
     itr = pumi_mesh->begin(dim-1);
     bdr_cnt = 0;
     while ((ent = pumi_mesh->iterate(itr)))
       {
           apf::ModelEntity *me = pumi_mesh->toModel(ent);
           if (pumi_mesh->getModelType(me) == (dim-1))
           {
              //Evrywhere 3 as initial
              (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(3);
              int tag = pumi_mesh->getModelTag(me);
              if (Dirichlet.Find(tag) != -1)
              {
                 //Dirichlet attr -> 1
                 (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(1);
              }
              else if (load_bdr.Find(tag) != -1)
              {
                 //load attr -> 2
                 (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(2);
              }
              bdr_cnt++;
           }
       }
       pumi_mesh->end(itr);

      //assign attr for elements
      Vector cent(ppt, dim);
      for (int el = 0; el < pmesh->GetNE(); el++)
       {
           (pmesh->GetElementTransformation(el))->Transform(Geometries.GetCenter(
                                                              pmesh->GetElementBaseGeometry(el)),cent);
           if (cent(1) <= 0.01)
           {
              pmesh->SetAttribute(el , 1);
           }
           else if (cent(1) >= 0.1)
           {
              pmesh->SetAttribute(el , 2);
           }
           else
           {
              pmesh->SetAttribute(el , 3);
           }

       }
       pmesh->SetAttributes();       
        
      f.Set(0, new ConstantCoefficient(0.0));
      f.Set(1, new ConstantCoefficient(0.0));
      f.Set(2, new ConstantCoefficient(0.0));
      {
           Vector pull_force(pmesh->bdr_attributes.Max());
           pull_force = 0.0;
           pull_force(1) =  1.0e-2;
           f.Set(1, new PWConstCoefficient(pull_force));
      }        
      ParLinearForm *b = new ParLinearForm(fespace);
      b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
      b->Assemble();
        
      Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[0] = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);        
        
      HypreParMatrix A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
      if (myId == 0)
        {
           cout << "done." << endl;
           cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
        }

      // 13. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
      //     preconditioner from hypre.
      HypreBoomerAMG *amg = new HypreBoomerAMG(A);
      if (amg_elast && !a->StaticCondensationIsEnabled())
      {
           amg->SetElasticityOptions(fespace);
      }
      else
      {
           amg->SetSystemsOptions(dim);
      }
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-7);
      pcg->SetMaxIter(250);
      pcg->SetPrintLevel(2);
      pcg->SetPreconditioner(*amg);
      pcg->Mult(B, X);

      // 14. Recover the parallel grid function corresponding to X. This is the
      //     local finite element solution on each processor.
      a->RecoverFEMSolution(X, *b, x);

      // 17. Send the above data by socket to a GLVis server.  Use the "n" and "b"
      //     keys in GLVis to visualize the displacements.
      if (visualization)
       {
           char vishost[] = "localhost";
           int  visport   = 19916;
           socketstream sol_sock(vishost, visport);
           sol_sock << "parallel " << num_procs << " " << myId << "\n";
           sol_sock.precision(8);
           sol_sock << "solution\n" << *pmesh << x << flush;
       }

       // 12. The main AMR loop. In each iteration we solve the problem on the
       //     current mesh, visualize the solution, and adapt the mesh.
       //
       //write vtk file
       apf::writeVtkFiles("upright_Before_ma", pumi_mesh);    


       // 18. Field transfer. Scalar solution field and magnitude field for
       //     error estimation are created the pumi mesh.
       if (order > geom_order)
        {
              Tmag_field = apf::createField(pumi_mesh, "field_mag",
                                            apf::SCALAR, apf::getLagrange(order));
              temp_field = apf::createField(pumi_mesh, "T_field",
                                            apf::VECTOR, apf::getLagrange(order));
        }
        else
        {
             Tmag_field = apf::createFieldOn(pumi_mesh, "field_mag",apf::VECTOR);
             temp_field = apf::createFieldOn(pumi_mesh, "T_field", apf::VECTOR);
        }

        ParPumiMesh* pPPmesh = dynamic_cast<ParPumiMesh*>(pmesh);
        pPPmesh->VectorFieldMFEMtoPUMI(pumi_mesh, &x, temp_field, Tmag_field);

        ipfield= spr::getGradIPField(temp_field, "MFEM_gradip", 2);
        sizefield = spr::getSPRSizeField(ipfield, adapt_ratio);

        apf::destroyField(Tmag_field);
        apf::destroyField(ipfield);
        apf::destroyNumbering(pumi_mesh->findNumbering("LocalVertexNumbering"));

        // 19. Perform MesAdapt
        ma::Input* erinput = ma::configure(pumi_mesh, sizefield);
        erinput->shouldFixShape = true;
        erinput->shouldSnap = true;
        erinput->maximumIterations = 2;
        erinput->shouldRunMidParma = true;
        if ( geom_order > 1)
        {
            crv::adapt(erinput);
        }
         else
        {
            ma::adapt(erinput);
        }
        pumi_mesh->verify();

        //write vtk file
        apf::writeVtkFiles("upright_After_ma", pumi_mesh);      

        ParMesh* Adapmesh = new ParPumiMesh(MPI_COMM_WORLD, pumi_mesh);
        pPPmesh->UpdateMesh(Adapmesh);
        delete Adapmesh;   

        fespace->Update();
        x.Update();
        x = 0.0;      

        pPPmesh->VectorFieldPUMItoMFEM(pumi_mesh, temp_field, &x);
        a->Update();
        b->Update();           

        //Destroy fields
        apf::destroyField(temp_field);
        apf::destroyField(sizefield);   
           
        delete pcg;
        delete amg;
        delete b;
     
   }

   // 18. Free the used memory.
   delete a;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete pmesh;

   pumi_mesh->destroyNative();
   apf::destroyMesh(pumi_mesh);
   PCU_Comm_Free();

#ifdef MFEM_USE_SIMMETRIX
   gmi_sim_stop();
   Sim_unregisterAllKeys();
#endif
   
   MPI_Finalize();

   return 0;
}
