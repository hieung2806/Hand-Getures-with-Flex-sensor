import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

#preprocessing
X = pd.read_excel('/content/drive/MyDrive/Luận_văn_PTDL/Train/Output_Final.xlsx', sheet_name = 'data')
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
X_data = scaled_X.reshape(2100,322*8)
my_pca = PCA(n_components = 2)
X_data_pca = my_pca.fit_transform(X_data)
Y = pd.read_excel('/content/drive/MyDrive/Luận_văn_PTDL/Train/Output_Final.xlsx',sheet_name = 'Label')
Le = LabelEncoder()
#trainning
y_data = Le.fit_transform(Y)
svc = SVC(kernel = 'rbf')
svc.fit(X_data_pca,y_data)

#plotting decision boundary
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
 
fig, ax = plt.subplots()
title = ('Trực quan phân lớp dữ liệu với PCA và SVM ')
# Set-up grid for plotting.
X0, X1 = X_data_pca[:, 0], X_data_pca[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax,svc, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_data, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('Principal Component 2')
ax.set_xlabel('Principal Component 1')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()
