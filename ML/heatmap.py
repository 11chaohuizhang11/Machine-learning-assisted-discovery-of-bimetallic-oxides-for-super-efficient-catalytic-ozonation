import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
excel_file_path = '/mnt/sg001/home/fz_nankai_cyj/ZCH/heatmap/inputdatav4.xlsx'
df = pd.read_excel(excel_file_path)

# 计算相关性矩阵
correlation_matrix = df.corr()

# 创建一个与相关性矩阵相同形状的布尔掩码，其中的 True 表示空白部分
mask = np.zeros_like(correlation_matrix, dtype=bool)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.heatmap(correlation_matrix, cmap="BrBG", annot=True)
plt.show()
plt.savefig('/mnt/sg001/home/fz_nankai_cyj/ZCH/heatmap/heatmapv1-1.tif',dpi=300)



