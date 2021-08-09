import numpy as np
import pandas as pd
from pylab import *
import seaborn as sns
sns.set()
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")

order = 2

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

df = pd.read_csv('conv_order%i.csv' % order, index_col=0)
dofs = df.index.to_numpy()
H10_errors = df.iloc[:, 0].to_numpy()
ZZ_errors = df.iloc[:, 1].to_numpy()

H10_error_rate = np.log(H10_errors[-1]/H10_errors[-10])/np.log(dofs[-10]/dofs[-1])
ZZ_error_rate = np.log(ZZ_errors[-1]/ZZ_errors[-10])/np.log(dofs[-10]/dofs[-1])
print('H10 error rate: ', H10_error_rate)
print('ZZ error rate:  ', ZZ_error_rate)

fig1,(ax1) = plt.subplots(1, 1)
ax1.loglog(dofs,H10_errors,'-og',lw=1.5,markersize = 9.0, alpha=.7, label=r'$H^1_0$ (rate: %.2f)' % H10_error_rate)
ax1.loglog(dofs,ZZ_errors,'-or',lw=1.5,markersize = 9.0, alpha=.7, label=r'ZZ (rate: %.2f)' % ZZ_error_rate)
ax1.set_ylabel(r"Error")
ax1.set_xlabel(r"DOFs")
ax1.legend(fontsize=15)
ax1.grid(True)

ax1.set_title(r'Order %i. Expected rate: %.1f' % (order, order/2)  )
plt.savefig('ConvergencePlotOrder%i.pdf' % order)
plt.show()