import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

data = pd.read_csv('LinearRegression/random.csv')

fig = plt.figure(figsize=(10, 6))
t = fig.suptitle('Polynomial Regression', fontsize=14)

# 散點圖
ax1 = fig.add_subplot(121, projection='3d')
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['green','yellow','orange', 'red'])
scatter = ax1.scatter(data['random2'], data['random3'], data['random4'], c=data['random5'], cmap=cmap, s=data['random1'], marker='o', edgecolors='none', alpha=0.8)
ax1.set_xlabel('random2')
ax1.set_ylabel('random3')
ax1.set_zlabel('random4')


# 添加颜色條
cb1 = plt.colorbar(scatter, ax=ax1, pad=0.2)
cb1.set_label('Salary')


ax2 = fig.add_subplot(122)
ax2.scatter(data['random2'], data['random3'], c=data['random5'], s=data['random5'], marker='o', edgecolors='none', alpha=0.6, cmap=cmap)
ax2.set_xlabel('random2')
ax2.set_ylabel('random3')


cb2 = plt.colorbar(scatter, ax=ax2)
cb2.set_label('random5')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('LinearRegression/test7_2.png')
plt.show()
