import matplotlib.pyplot as plt
import csv
import sys

if len(sys.argv) < 4:
    print("Usage: python plot_errors.py <csv_file> <output_file> <plot_title> [val_csv]")
    sys.exit(1)

csv_file = sys.argv[1]
output_file = sys.argv[2]
plot_title = sys.argv[3]

# Optional validation CSV is argv[4]
if len(sys.argv) > 4:
    val_csv_file = sys.argv[4]
else:
    val_csv_file = None

iters = []
e_in = []

with open(csv_file) as f:
    reader = csv.reader(f)
    for row in reader:
        iters.append(int(row[0]))
        e_in.append(float(row[1]))

plt.plot(iters, e_in, label="Training Error")

if val_csv_file:
    e_val = []
    with open(val_csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            e_val.append(float(row[1]))
    if len(e_val) == len(iters):
        plt.plot(iters, e_val, label="Validation Error")

plt.xlabel("Iteration")
plt.ylabel("E_in")
plt.title(f"E_in vs Iterations ({plot_title})")
plt.grid(True)
plt.legend()
plt.savefig(output_file)
plt.show()