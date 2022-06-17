import umap.umap_ as umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

X_test = pd.read_excel('/content/drive/MyDrive/Luận_văn_PTDL/Train/test_data_final_2.xlsx', sheet_name ='data')
Y_test_vs  = pd.read_excel('/content/drive/MyDrive/Luận_văn_PTDL/Train/test_data_visualize.xlsx')
encoder = LabelEncoder()
encoder.fit(Y_test_vs)

#umap 2D visualize data
encoded_Y_test_vs = encoder.transform(Y_test_vs)
reducer = umap.UMAP()

getures_data = X_test[["ax",
                  "ay",
                  "az",
                  "pinky",
                  "ring",
                  "middle",
                  "index",
                  "thumb"
]].values
scaled_getures_data = StandardScaler().fit_transform(getures_data)
embedding = reducer.fit_transform(scaled_getures_data)
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in Y_test_vs.Label.map({"Yes":0, "Stop":1, "Give":2,"Drink":3,"Eat":4,"None":5, "No":6})])
plt.gca().set_aspect('equal', 'datalim')
plt.title('Dữ liệu nhận dạng cử chỉ', fontsize=18)
plt.show()

#PCA 3D-visualize data
ig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X_test)
X = pca.transform(X_test)
# Reorder the labels to have colors matching the cluster results
#y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=encoded_Y_test_vs)
#cmap=plt.cm.nipy_spectral,
# edgecolor="k"
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()


