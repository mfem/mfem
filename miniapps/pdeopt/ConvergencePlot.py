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

df_GD = pd.read_csv('conv_order%i_GD.csv' % order, index_col=0)
steps_GD = df_GD.index[1:].to_numpy()
Compliance_GD = df_GD.iloc[1:, 0].to_numpy()
MassFraction_GD = df_GD.iloc[1:, 1].to_numpy()

df_Subgrad = pd.read_csv('conv_order%i_Subgrad.csv' % order, index_col=0)
steps_Subgrad = df_Subgrad.index[1:].to_numpy()
Compliance_Subgrad = df_Subgrad.iloc[1:, 0].to_numpy()
MassFraction_Subgrad = df_Subgrad.iloc[1:, 1].to_numpy()

df_KL = pd.read_csv('conv_order%i_KL.csv' % order, index_col=0)
steps_KL = df_KL.index[1:].to_numpy()
Compliance_KL = df_KL.iloc[1:, 0].to_numpy()
MassFraction_KL = df_KL.iloc[1:, 1].to_numpy()


fig1,(ax1) = plt.subplots(1, 1)
ax1.plot(steps_GD,Compliance_GD,'-k',lw=1.5,markersize = 9.0, alpha=.7, label=r'GD: Fixed Stepsize')
# ax1.plot(steps_Subgrad,Compliance_Subgrad,'-r',lw=1.5,markersize = 9.0, alpha=.7, label=r'GD: Decaying Stepsize')
ax1.plot(steps_KL,Compliance_KL,'-g',lw=1.5,markersize = 9.0, alpha=.7, label=r'Mirror Descent (KL-Divergence)')
ax1.set_ylabel(r"Compliance")
ax1.set_xlabel(r"Design Update")
ax1.legend(fontsize=15)
ax1.grid(True)

ax1.set_title(r'Order %i FEM Discretization' % order  )
plt.savefig('ConvergencePlotOrder%i.pdf' % order)
plt.show()