import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
data = pd.read_csv('LinearRegression/random.csv')

fig = plt.figure(figsize=(10, 6))
t = fig.suptitle('Polynomial Regression', fontsize=14)
ax = fig.add_subplot(111, projection='3d')

cmap = LinearSegmentedColormap.from_list('custom_cmap', ['green','yellow','orange', 'red'])
#使用 Salary 列作為颜色映射的依据
scatter = ax.scatter(data['random2'], data['random3'], data['random4'], c=data['random5'], cmap=cmap, s=data['random1']*5, marker='o', edgecolors='none', alpha=0.8)

# 設置坐標軸標籤
ax.set_xlabel('random2')
ax.set_ylabel('random3')
ax.set_zlabel('random4')

# 創建颜色條
cb = plt.colorbar(scatter,pad=0.15)
cb.set_label('random5')

def update(frame):
    # 模擬動態效果，可根据需求更改

    ax.view_init(elev=10, azim=frame)


# 創建動畫
animation = FuncAnimation(fig, update, frames=range(0, 360 , 1), interval=50, repeat=True)
# 保存為png
plt.savefig('LinearRegression/test7_1.png')
# 保存為GIF
animation.save('LinearRegression/test7_1.gif', writer='imagemagick')
print('Done')