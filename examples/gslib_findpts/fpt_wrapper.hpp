IntegrationRules IntRulesGLL(0, Quadrature1D::GaussLobatto);
class findpts_gslib
{
private:
        IntegrationRule ir;
        double *fmesh;
        struct findpts_data_2 *fda;
        struct findpts_data_3 *fdb;
        struct comm cc;
        int dim, nel, qo, msz;

public:
#ifdef MFEM_USE_MPI
      findpts_gslib (ParFiniteElementSpace *pfes, ParMesh *pmesh, int QORDER);
#else
      findpts_gslib (FiniteElementSpace *pfes, Mesh *pmesh, int QORDER);
#endif

//    sets up findpts
      void gslib_findpts_setup(double bb_t, double newt_tol, int npt_max);

//    finds r,s,t,e,p for given x,y,z
      void gslib_findpts(Array<uint> *pcode,Array<uint> *pproc,Array<uint> *pel,
      Vector *pr,Vector *pd,Vector *xp, Vector *yp, Vector *zp, int nxyz);

      void gslib_findpts(Array<uint> *pcode,Array<uint> *pproc,Array<uint> *pel,
      Vector *pr,Vector *pd,Vector *xyzp, int nxyz);

//    Interpolates fieldin for given r,s,t,e,p and puts it in fieldout
      void gslib_findpts_eval (Vector *fieldout,Array<uint> *pcode,Array<uint> *pproc,Array<uint> *pel,
            Vector *pr,double *fieldin, int nxyz);

//    Interpolates fieldin for given r,s,t,e,p and puts it in fieldout
#ifdef MFEM_USE_MPI
      void gslib_findpts_eval (Vector *fieldout,Array<uint> *pcode,Array<uint> *pproc,Array<uint> *pel,
            Vector *pr,ParGridFunction *fieldin, int nxyz);
#else
      void gslib_findpts_eval (Vector *fieldout,Array<uint> *pcode,Array<uint> *pproc,Array<uint> *pel,
            Vector *pr,GridFunction *fieldin, int nxyz);
#endif

//    clears up memory
      void gslib_findpts_free ();

      ~findpts_gslib();
};

#ifdef MFEM_USE_MPI
findpts_gslib::findpts_gslib (ParFiniteElementSpace *pfes, ParMesh *pmesh, int QORDER)
#else
findpts_gslib::findpts_gslib (FiniteElementSpace *pfes, Mesh *pmesh, int QORDER)
#endif
{
   const int geom_type = pfes->GetFE(0)->GetGeomType();
   this->ir = IntRulesGLL.Get(geom_type, QORDER); 
   dim = pmesh->Dimension();
   nel = pmesh->GetNE();
   qo = sqrt(ir.GetNPoints());
   if (dim==3) qo = cbrt(ir.GetNPoints());
   int nsp = pow(qo,dim);
   msz = nel*nsp;
   this->fmesh = new double[dim*msz];

   int npt = nel*nsp;
#ifdef MFEM_USE_MPI
   ParGridFunction nodes(pfes);
#else
   GridFunction nodes(pfes);
#endif
   pmesh->GetNodes(nodes);

   int np = 0; 
   for (int i = 0; i < nel; i++)
   {  
      for (int j = 0; j < nsp; j++)
      { 
        const IntegrationPoint &ip = this->ir.IntPoint(j);
        for (int k = 0; k < dim; k++)
        {
         this->fmesh[k*npt+np] = nodes.GetValue(i, ip, k+1);
        }
        np = np+1;
      }
   }
}

void findpts_gslib::gslib_findpts_setup(double bb_t, double newt_tol, int npt_max)
{
   const int NE = nel, nsp = this->ir.GetNPoints(), NR = qo;
#ifdef MFEM_USE_MPI
   comm_init(&this->cc,MPI_COMM_WORLD);
#else
   comm_init(&this->cc,0);
#endif
   int ntot = pow(NR,dim)*NE;
   if (dim==2)
   {
    unsigned nr[2] = {NR,NR};
    unsigned mr[2] = {2*NR,2*NR};
    double *const elx[2] = {&this->fmesh[0],&this->fmesh[ntot]};
    this->fda=findpts_setup_2(&this->cc,elx,nr,NE,mr,bb_t,ntot,ntot,npt_max,newt_tol);
   }
   else
   {
    unsigned nr[3] = {NR,NR,NR};
    unsigned mr[3] = {2*NR,2*NR,2*NR};
    double *const elx[3] = {&this->fmesh[0],&this->fmesh[ntot],&this->fmesh[2*ntot]};
    this->fdb=findpts_setup_3(&this->cc,elx,nr,NE,mr,bb_t,ntot,ntot,npt_max,newt_tol);
   }
}

void findpts_gslib::gslib_findpts(Array<uint> *pcode,Array<uint> *pproc,Array<uint> *pel,Vector *pr,Vector *pd,Vector *xp,Vector *yp, Vector *zp, int nxyz)
{
    uint *const code_base = pcode->GetData();
    uint *const proc_base = pproc->GetData();
    uint *const el_base = pel->GetData();
    double *const dist_base = pd->GetData();
    if (dim==2)
    {
    const double *xv_base[2];
    xv_base[0]=xp->GetData(), xv_base[1]=yp->GetData();
    unsigned xv_stride[2];
    xv_stride[0] = sizeof(double),xv_stride[1] = sizeof(double);
    findpts_2(
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr->GetData(),sizeof(double)*dim,
      dist_base,sizeof(double),
      xv_base,     xv_stride,
      nxyz,this->fda);
    }
   else
   {
    const double *xv_base[3];
    xv_base[0]=xp->GetData(), xv_base[1]=yp->GetData();xv_base[2]=zp->GetData();
    unsigned xv_stride[3];
    xv_stride[0] = sizeof(double),xv_stride[1] = sizeof(double),xv_stride[2] = sizeof(double);
    findpts_3(
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr->GetData(),sizeof(double)*dim,
      dist_base,sizeof(double),
      xv_base,     xv_stride,
      nxyz,this->fdb);
   }
// Recast from [-1,1] to [0,1] for MFEM
   for (int i = 0; i < pr->Size(); i++)
   {
    pr->GetData()[i] = pr->GetData()[i]*0.5 + 0.5;
   }
}

void findpts_gslib::gslib_findpts(Array<uint> *pcode,Array<uint> *pproc,Array<uint> *pel,Vector *pr,Vector *pd,Vector *xyzp, int nxyz)
{
    uint *const code_base = pcode->GetData();
    uint *const proc_base = pproc->GetData();
    uint *const el_base = pel->GetData();
    double *const dist_base = pd->GetData();
    if (dim==2)
    {
    const double *xv_base[2];
    xv_base[0]=xyzp->GetData(), xv_base[1]=xyzp->GetData()+nxyz;
    unsigned xv_stride[2];
    xv_stride[0] = sizeof(double),xv_stride[1] = sizeof(double);
    findpts_2(
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr->GetData(),sizeof(double)*dim,
      dist_base,sizeof(double),
      xv_base,     xv_stride,
      nxyz,this->fda);
    }
   else
   {
    const double *xv_base[3];
    xv_base[0]=xyzp->GetData(), xv_base[1]=xyzp->GetData()+nxyz;xv_base[2]=xyzp->GetData()+nxyz;
    unsigned xv_stride[3];
    xv_stride[0] = sizeof(double),xv_stride[1] = sizeof(double),xv_stride[2] = sizeof(double);
    findpts_3(
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr->GetData(),sizeof(double)*dim,
      dist_base,sizeof(double),
      xv_base,     xv_stride,
      nxyz,this->fdb);
   }
// Recast from [-1,1] to [0,1] for MFEM
   for (int i = 0; i < pr->Size(); i++)
   {
    pr->GetData()[i] = pr->GetData()[i]*0.5 + 0.5;
   }
}

void findpts_gslib::gslib_findpts_eval(
            Vector *fieldout, Array<uint> *pcode, Array<uint>  *pproc, Array<uint>  *pel, Vector *pr,
            double *fieldin, int nxyz)
{
// Recast from [0,1] to [-1,1] for MFEM
    for (int i = 0; i < pr->Size(); i++)
    {
     pr->GetData()[i] = pr->GetData()[i]*2.0 - 1.0;
    }
    uint *const code_base = pcode->GetData();
    uint *const proc_base = pproc->GetData();
    uint *const el_base = pel->GetData();
    double *const out_base = fieldout->GetData();
    double *const in_base = fieldin;
    int npt = nel*pow(qo,dim);
    if (dim==2)
    {
    findpts_eval_2(out_base,sizeof(double),
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr->GetData(),sizeof(double)*dim,
      nxyz,fieldin,this->fda);
    }
   else
   {
    findpts_eval_3(out_base,sizeof(double),
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr->GetData(),sizeof(double)*dim,
      nxyz,fieldin,this->fdb);
   }
// Recast from [-1,1] to [0,1] for MFEM
   for (int i = 0; i < pr->Size(); i++)
   {
    pr->GetData()[i] = pr->GetData()[i]*0.5 + 0.5;
   }
}

#ifdef MFEM_USE_MPI
void findpts_gslib::gslib_findpts_eval(
            Vector *fieldout, Array<uint> *pcode, Array<uint>  *pproc, Array<uint>  *pel, Vector *pr,
            ParGridFunction *fieldin, int nxyz)
#else
void findpts_gslib::gslib_findpts_eval(
            Vector *fieldout, Array<uint> *pcode, Array<uint>  *pproc, Array<uint>  *pel, Vector *pr,
            GridFunction *fieldin, int nxyz)
#endif
{

// convert gridfunction to double field
   double* fin =  new double[dim*msz]; 
   int nsp = pow(qo,dim);
   int np = 0;  
   for (int i = 0; i < nel; i++)
   {  
      for (int j = 0; j < nsp; j++)
      { 
        const IntegrationPoint &ip = this->ir.IntPoint(j);
        fin[np] = fieldin->GetValue(i, ip);
        np = np+1;
      }
   }

// Recast from [0,1] to [-1,1] for MFEM
    for (int i = 0; i < pr->Size(); i++)
    {
     pr->GetData()[i] = pr->GetData()[i]*2.0 - 1.0;
    }

//  
    uint *const code_base = pcode->GetData();
    uint *const proc_base = pproc->GetData();
    uint *const el_base = pel->GetData();
    double *const out_base = fieldout->GetData();
    double *const in_base = fin;
    int npt = nel*pow(qo,dim);
    if (dim==2)
    {
    findpts_eval_2(out_base,sizeof(double),
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr->GetData(),sizeof(double)*dim,
      nxyz,fin,this->fda);
    }
   else
   {
    findpts_eval_3(out_base,sizeof(double),
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr->GetData(),sizeof(double)*dim,
      nxyz,fin,this->fdb);
   }
// Recast from [-1,1] to [0,1] for MFEM
   for (int i = 0; i < pr->Size(); i++)
   {
    pr->GetData()[i] = pr->GetData()[i]*0.5 + 0.5;
   }

   delete[] fin;
}

void findpts_gslib::gslib_findpts_free ()
{
 if (dim==2)
 {
  findpts_free_2(this->fda);
 }
 else
 {
  findpts_free_3(this->fdb);
 }
}
