"""
src/scripts/plot_fd_experiment.py

Creating finite-differences plot with/out corrected symplectic Euler. Used standalone, together with mean error values
in the resulting experiment after running `src/scripts/run_finite_differences_experiment.sh`
"""

from matplotlib import pyplot as plt
import numpy as np

hs = np.array([0.05, 0.1, 0.2, 0.4, 0.8])

# true flow simulated with RK45
y_symp_euler = np.array([             0.007458685509513693,   0.01492905495050296,     0.029952207567551365,  0.06063653697572398,  0.12633854796468025])

# true flow simulated with symp euler
y_symp_euler = np.array([             0.007655487417115211,   0.014893745411252607,    0.029887015855865662,  0.06056706558727502,  0.12626741161758956])

# true flow simulated with RK45
y_corrected_symp_euler = np.array([   0.0001049473483956588,  0.00042048988193375044,  0.0016931638666154113, 0.006948469496834716, 0.030395443784396382])

# true flow simulated with symp euler
y_corrected_symp_euler = np.array([   0.002044280831390535,   0.0012417564032197134,   0.0019579162937092324, 0.007044428242534738, 0.030413997915585196 ])

y_vec_field = np.ones_like(hs) * 6.348834733533535e-07

fig, ax = plt.subplots()
# ax.loglog(hs, y_forward_euler, 'o-')
ax.loglog(hs, y_symp_euler, 'o-')
ax.loglog(hs, y_corrected_symp_euler, 'o-')

ax.loglog(hs, hs, '--', color='grey', linewidth=1)
ax.loglog(hs, hs**2, '--', color='grey', linewidth=1)
# ax.loglog(hs, y_vec_field, 'o-')

# ax.legend(['forward Euler', 'symp. Euler', 'corrected symp. Euler'], fontsize='medium', shadow=True, loc='best')
ax.legend(['symp. Euler', 'corrected symp. Euler'], fontsize='medium', shadow=True, loc='best')

ax.set_xlabel(r'Step size $h$')
ax.set_ylabel(r'Rel. $L^2$ error, mean of 10 runs, on $\log_{10}$ scale')

ax.set_xticks(hs)
ax.set_xticklabels(map(str, hs))
ax.text(0.5, 0.4+0.02, r'$\varepsilon \in \mathcal{O}(h)$',
          color='grey', rotation=15, verticalalignment='bottom', horizontalalignment='center')
ax.text(0.5, 0.4**2+0.02, r'$\varepsilon \in \mathcal{O}(h^2)$',
          color='grey', rotation=25, verticalalignment='bottom', horizontalalignment='center')
ax.grid()
fig.tight_layout()
fig.savefig('./fd_figures.pdf')
plt.show()
