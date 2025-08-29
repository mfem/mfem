import csv
from pylab import *

fields=[
    ['code ID', 'str'],
    ['preconditioner ID', 'str'],
    ['machine ID', 'str'],
    ['number of nodes', 'int'],
    ['number of MPI ranks', 'int'],
    ['n_x', 'int'], ['n_y', 'int'], ['n_z', 'int'],
    ['solution polynomial degree', 'int'],
    ['number of 1D quadrature points', 'float'],
    ['eps_y', 'float'], ['eps_z', 'float'],
    ['ndofs (including Dirichlet boundary)', 'int'],
    ['niter', 'int'],
    ['initial residual', 'float'], ['final residual', 'float'],
    ['error', 'float'],
    ['t_setup (preconditioner setup)', 'float'],
    ['t_solve (total iter time)', 'float']]
fields_dict=dict(fields)

def convert(obj, type_str):
    ctor=getattr(__builtins__, type_str)
    return ctor(obj)

input_csv='run-001.csv'
print('reading %s ...' % input_csv)
runs = []
with open(input_csv) as csvfile:
    csvreader = csv.DictReader(csvfile, fieldnames=[f[0] for f in fields],
                               restkey='additional notes')
    for row in csvreader:
        for i in fields_dict:
            row[i]=convert(row[i], fields_dict[i])
        runs.append(row)

orders=[r['solution polynomial degree'] for r in runs]
orders=unique(orders) # numpy function
# orders=[1]

nps=[r['number of MPI ranks'] for r in runs]
nps=unique(nps)
if len(nps) > 1:
    print('multiple num-ranks present: %s' % nps)
    quit()
np=nps[0]

# plot fx (or fx/fn) vs fy, (or fx/fn/fy, etc) for all orders
fn='number of MPI ranks'
fx='ndofs (including Dirichlet boundary)'
fy='t_solve (total iter time)'
# fy='niter'
# fy='error'
fz='niter'

figure()
for p in orders:
    rr=[r for r in runs if (r['solution polynomial degree']==p and
                            r['niter']>0)]
    if len(rr)==0:
        continue

    # pl_data=asarray([[r[fx],r[fx]/r[fy]] for r in rr])
    # pl_data=asarray([[r[fx],r[fy]] for r in rr])
    # pl_data=asarray([[r[fx],r[fx]/(r[fy]/r[fz])] for r in rr])

    pl_data=asarray([[r[fx]/r[fn],r[fx]/r[fn]/r[fy]] for r in rr])
    # pl_data=asarray([[r[fx]/r[fn],r[fy]] for r in rr])

    plot(pl_data[:,0],pl_data[:,1], 'o-', label='p=%i'%p)
    rnx=asarray([r['n_x'] for r in rr])
    rerr=asarray([r['error'] for r in rr])
    rate=arange(1.0,len(rnx))
    for l in range(1,len(rnx)):
        rate[l-1]=log(rerr[l-1]/rerr[l])/log(rnx[l]/rnx[l-1])
    set_printoptions(formatter={'float':"{:6.2f}".format},linewidth=120)
    print(f"p={p} rate:{rate}")

# xscale('log', basex=10) # older matplotlib
xscale('log', base=10)
# xlim(4e4,3.1e7)
xlim(4e4,5e6)
# yscale('log', basey=10) # older matplotlib
# yscale('log', base=10)
# ylim(1e5,2e7)
# ylim(0,2.55e7)
# ylim(0,3.25e7)
# ylim(0,5e6)
ymin,ymax=ylim()
ylim(0,ymax)
# ylim(1e-2,2e1)
# ylim(3e-3,6e-2)
# xlabel(fx)
# xlabel('# DOFs')
xlabel('# DOFs / # Ranks')
# ylabel(fx + ' / ' + fy)
# ylabel(fy)
# ylabel('# DOFs / t_solve')
ylabel('# DOFs / # Ranks / t_solve')
# ylabel('t_solve')
# ylabel('# DOFs / (t_solve / # Iter)')
# ylabel('# Iter')
# ylabel('L2 error')
# ylabel('Grad L2 error')
grid('on', color='gray', ls='dotted')
grid('on', axis='both', which='minor', color='gray', ls='dotted')
legend(ncol=2, loc='best')
ranks='1 MPI rank'
if np > 1:
   ranks='%s MPI ranks' % (np,np)
hypre='hypre CPU'
# hypre='hypre HIP'
# prec=hypre+', p-MG(1,1)'
prec=hypre+', LOR'
# prec='Jacobi'
# eps='1'
eps='0.3'
mfem='MFEM CPU'
# mfem='MFEM HIP'
title(mfem + ', ' + prec + ', $\\varepsilon = ' + eps + '$, ' + ranks)

if 1: # write .pdf file?
    pdf_file='plot.pdf'
    print('saving figure --> %s'%pdf_file)
    savefig(pdf_file, format='pdf', bbox_inches='tight')

if 0: # show the figures?
    print('\nshowing figures ...')
    show()
