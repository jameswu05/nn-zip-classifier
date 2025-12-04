import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 4:
    print("Usage: python3 plot_decision_boundary.py <grid_csv> <train_csv>")
    sys.exit(1)

grid_file = sys.argv[1]    
train_file = sys.argv[2]   
output_file = sys.argv[3]
plot_title = sys.argv[4]

train_data = np.loadtxt(train_file)
X_train = train_data[:, 1:3]
y_train = train_data[:, 0]

grid_data = np.loadtxt(grid_file, delimiter=',')
X_grid = grid_data[:, :2]
Z_grid = grid_data[:, 2]

x1_vals = np.unique(X_grid[:, 0])
x2_vals = np.unique(X_grid[:, 1])

Z = Z_grid.reshape(len(x1_vals), len(x2_vals)).T 

X1, X2 = np.meshgrid(x1_vals, x2_vals)

plt.figure(figsize=(8,6))
plt.contour(X1, X2, Z, levels=[0], colors='red', linewidths=2)
plt.contourf(X1, X2, Z, levels=[Z.min(), 0, Z.max()], colors=['#FFAAAA','#AAAAFF'], alpha=0.3)
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='blue', marker='o', label='Digit 1')
plt.scatter(X_train[y_train==-1,0], X_train[y_train==-1,1], c='green', marker='x', label='Not 1')

plt.xlabel("Symmetry")
plt.ylabel("Intensity")
plt.title(f"Decision Boundary of Neural Network ({plot_title})")
plt.legend()
plt.tight_layout()
plt.savefig(output_file)
plt.show()