import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

exp1 = dict()
exp1['A1'] = []

domains = ['B','D','E','K']
tasks = []
for soucre in domains:
    for target in domains:
        tasks.append(soucre+'2'+target)
exp1['A1'] = [np.random.random() for task in tasks]

pd_acc =pd.DataFrame(exp1)

N = len(tasks)

theta = np.linspace(0,2*np.pi,N,endpoint=False)
theta = theta.tolist()
theta.append(theta[0])
# theta = [aaxis+0.2 for aaxis in theta]
r = pd_acc['A1'].tolist()
r.append(r[0])

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(theta,r,'-',linewidth = 1)
ax.plot(theta,r,'o-',linewidth = 1)

aaxis = np.array(theta[:-1])
ax.set_thetagrids(aaxis*180/np.pi,tasks)

plt.rgrids([0.2, 0.4, 0.6, 0.8])

labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
legend = plt.legend(labels, loc=(0.9, .95), labelspacing=0.1)

plt.figtext(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
                ha='center', color='black', weight='bold', size='large')

plt.ylim(0,1)
plt.show()

print()