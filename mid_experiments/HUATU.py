import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print(matplotlib.matplotlib_fname())
exp_pd = pd.read_csv('all_exp_data.csv')

# 所有任务
domains = ['B', 'D', 'E', 'K']
tasks = []
for soucre in domains:
    for target in domains:
        if soucre != target:
            tasks.append(soucre + '->' + target)
N = len(tasks)

# ['SVM', 'mmd_svm', 'mk_svm', 'lh_SVM',
# 'text_CNN', 'mmd_text_CNN','mk_text_CNN', 'lh_text_CNN',
# 'mmd_TCA', 'mk_TCA', 'lh_TCA',
# 'HIDC', 'mk_HIDC', 'lh_HIDC',
# 'MMD_DDC', 'MK_DDC', 'lh_DDC',
# 'roberta']

# 纵向对照
svm = ['SVM', 'mmd_svm', 'mk_svm']
text_CNN = ['text_CNN', 'mmd_text_CNN', 'mk_text_CNN', 'lh_text_CNN']
tca = ['mmd_TCA', 'mk_TCA', 'lh_TCA']
hidc = ['HIDC', 'mk_HIDC', 'lh_HIDC']
ddc = ['MMD_DDC', 'MK_DDC', 'lh_DDC']

# 横向对照
mmd = ['mmd_svm', 'mmd_text_CNN', 'mmd_TCA', 'HIDC', 'MMD_DDC']
mk = ['mk_svm', 'mk_text_CNN', 'mk_TCA', 'mk_HIDC', 'MK_DDC']
lh = ['lh_SVM','lh_text_CNN','lh_TCA','lh_HIDC','lh_DDC']
last = ['lh_SVM','lh_text_CNN','lh_DDC','MDDA','PMBDAN']

# fig
fig1 = ['SVM', 'text_CNN']
fig2 = ['mmd_svm','mk_svm']
fig3 = ['mmd_text_CNN','mk_text_CNN']
fig4 = ['mmd_TCA','mk_TCA']
fig5 = ['HIDC','mk_HIDC']
fig6 = ['MMD_DDC','MK_DDC']
all1 = ['mmd_svm','mk_svm', 'mmd_text_CNN','mk_text_CNN', 'mmd_TCA','mk_TCA', 'HIDC','mk_HIDC', 'MMD_DDC','MK_DDC']
fig7 = ['mk_svm','lh_SVM']
fig8 = ['mk_text_CNN','lh_text_CNN']
fig9 = ['mk_TCA','lh_TCA']
fig10 = ['mk_HIDC','lh_HIDC']
fig11 = ['MK_DDC','lh_DDC']
all2 = ['mk_svm','lh_SVM','mk_text_CNN','lh_text_CNN','mk_TCA','lh_TCA','mk_HIDC','lh_HIDC','MK_DDC','lh_DDC']
fig12 = ['SVM','text_CNN','roberta']

data = last # chose
tittle = "基于亚马逊评论数据集的性能比较"

# 雷达线
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
theta = theta.tolist()
theta.append(theta[0])
for algor in data:
    exp_data = exp_pd[algor].tolist()
    exp_data.append(exp_data[0])
    ax.plot(theta, exp_data, '-o', linewidth=1)
legend = plt.legend(data, loc=(0.9, .95), labelspacing=0.1)
plt.figtext(0.5, 0.985, tittle, ha='center', color='black', weight='bold', size='large')
aaxis = np.array(theta[:-1])
ax.set_thetagrids(aaxis * 180 / np.pi, tasks)
plt.rgrids([0.2, 0.4, 0.6, 0.8])
plt.ylim(0, 1)
fig.tight_layout()
plt.savefig('figa.jpg')
plt.show()

# 直方图
x = np.arange(len(tasks))
width = 0.8
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
lbound = x-width/2 + width/len(data)/2
for algor in data:
    exp_data = exp_pd[algor].tolist()
    ax.bar(lbound, exp_data, width/len(data), label=algor)
    lbound += width/len(data)

ax.set_ylabel('分类精度')
ax.set_title(tittle)
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.legend(loc=4)
plt.grid()
fig.tight_layout()
plt.savefig('figb.jpg')
plt.show()
print()
