import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
dataset=pd.read_csv('data/'+os.listdir('data')[0])
df = pd.DataFrame(dataset)

# 計算特徵和特徵之間的相關係數
correlation_matrix = df.corr()

# 可視化相關係數矩陣
plt.figure(figsize=(8, 6))

heatmap = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# 添加相關係數的標籤
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        plt.text(j, i, '{:.2f}'.format(correlation_matrix.iloc[i, j]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='black', fontsize=10)

# 設置顏色軸標籤
cbar = plt.colorbar(heatmap)
cbar.set_label('Correlation', rotation=270, labelpad=20)

# 設置坐標軸標籤
plt.xticks(np.arange(len(correlation_matrix)), correlation_matrix.columns)
plt.yticks(np.arange(len(correlation_matrix)), correlation_matrix.index)

# 添加標題
plt.title('Correlation Matrix')

# 顯示圖形
plt.show()