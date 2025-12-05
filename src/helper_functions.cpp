#include <random>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include "nn.h"
#include "layers.h"

void load_data(
    const std::string& filename,
    std::vector<Eigen::VectorXd>& X,
    std::vector<Eigen::VectorXd>& y
) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    X.clear();
    y.clear();

    std::string line;
    while (std::getline(fin, line)) {
        if (line.size() == 0) {
            continue;
        }

        std::stringstream ss(line);
        double label, f1, f2;
        ss >> label >> f1 >> f2;
        
        Eigen::VectorXd x(2);
        x << f1, f2;

        Eigen::VectorXd t(1);
        if (label == 1) {
            t(0) = 1.0;
        } else {
            t(0) = -1.0;
        }

        X.push_back(x);
        y.push_back(t);
    }

    fin.close();
}

void initialize_random_weights(Layer& layer, double min_val = -0.01, double max_val = 0.01) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min_val, max_val);

    Eigen::MatrixXd layer_w(layer.get_weights().rows(), layer.get_weights().cols());
    Eigen::VectorXd layer_b(layer.get_biases().size());

    for (int i = 0; i < layer_w.rows(); i++) {
        layer_b(i) = dis(gen);
        for (int j = 0; j < layer_w.cols(); j++) {
            layer_w(i, j) = dis(gen);
        }
    }

    layer.set_weights(layer_w);
    layer.set_biases(layer_b);
}

void split_training_validation(
    const std::vector<Eigen::VectorXd>& X,
    const std::vector<Eigen::VectorXd>& y,
    std::vector<Eigen::VectorXd>& X_train,
    std::vector<Eigen::VectorXd>& y_train,
    std::vector<Eigen::VectorXd>& X_val,
    std::vector<Eigen::VectorXd>& y_val,
    int train_size,
    int val_size
) {
    std::cout << X.size() << " " << train_size << " " << val_size << std::endl;
    assert (X.size() >= train_size + val_size);
    assert(X.size() == y.size());

    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    X_train.clear();
    y_train.clear();
    X_val.clear();
    y_val.clear();

    for (int i = 0; i < train_size; i++) {
        X_train.push_back(X[indices[i]]);
        y_train.push_back(y[indices[i]]);
    }

    for (int i = train_size; i < train_size + val_size; i++) {
        X_val.push_back(X[indices[i]]);
        y_val.push_back(y[indices[i]]);
    }
}

Eigen::VectorXd tanh_func(const Eigen::VectorXd& z) {
    return z.array().tanh();
}

Eigen::VectorXd tanh_derivative(const Eigen::VectorXd& z) { 
    return (1.0 - z.array().square()).matrix();
}

Eigen::VectorXd identity_func(const Eigen::VectorXd& z) { 
    return z;
}

Eigen::VectorXd identity_derivative(const Eigen::VectorXd& z) { 
    return Eigen::VectorXd::Ones(z.size());
}

Eigen::VectorXd sign_func(const Eigen::VectorXd& z) {
    return z.unaryExpr([](double v){ return (v >= 0 ? 1.0 : -1.0); });
}

Eigen::VectorXd sign_derivative(const Eigen::VectorXd& z) {
    return Eigen::VectorXd::Zero(z.size()); 
}

void plot_errors_from_csv(
    const std::string& csv_file,
    const std::string& output_file,
    const std::string& python_script,
    const std::string& plot_title,
    const std::string& val_csv_file = ""
) {
    std::string quoted_title = "\"" + plot_title + "\"";
    std::string command = 
        "python3.11 ../src/" + python_script + " " +
        "../data/" + csv_file + " " +
        "../plots/" + output_file + " " +
        quoted_title;

    if (!val_csv_file.empty()) {
        command += " ../data/" + val_csv_file;
    }
    
    std::cout << "Running command: " << command << std::endl;

    int ret = std::system(command.c_str());
    if (ret != 0) {
        std::cerr << "Error: Python script exited with code " << ret << std::endl;
    }
}

void plot_decision_boundary_from_NN(
    NN& network,
    const std::vector<Eigen::VectorXd>& X_train,
    const std::string& train_csv,
    const std::string& grid_csv,
    const std::string& python_script,
    const std::string& output_file,
    const std::string& plot_title,
    int steps=200
) {
    double x1_min = std::numeric_limits<double>::max();
    double x1_max = std::numeric_limits<double>::lowest();
    double x2_min = std::numeric_limits<double>::max();
    double x2_max = std::numeric_limits<double>::lowest();

    for (const auto& x : X_train) {
        if (x(0) < x1_min) {
            x1_min = x(0);
        }

        if (x(0) > x1_max) {
            x1_max = x(0);
        }

        if (x(1) < x2_min) {
            x2_min = x(1);
        }

        if (x(1) > x2_max) {
            x2_max = x(1);
        }
    }

    double margin_x1 = (x1_max - x1_min) * 0.05;
    double margin_x2 = (x2_max - x2_min) * 0.05;
    x1_min -= margin_x1;
    x1_max += margin_x1;
    x2_min -= margin_x2;
    x2_max += margin_x2;

    std::ofstream fout(grid_csv);
    for (int i = 0; i < steps; i++) {
        double x1 = x1_min + i * (x1_max - x1_min) / (steps - 1);
        for (int j = 0; j < steps; j++) {
            double x2 = x2_min + j * (x2_max - x2_min) / (steps - 1);

            Eigen::VectorXd input(2);
            input << x1, x2;

            Eigen::VectorXd output = network.forward(input, identity_func);
            fout << x1 << "," << x2 << "," << output(0) << "\n";
        }
    }
    fout.close();

    std::string quoted_title = "\"" + plot_title + "\"";
    std::string command = "python3.11 " + python_script + " " + grid_csv + " " + train_csv + " " + output_file + " " + quoted_title;
    std::cout << "Running command: " << command << std::endl;
    int ret = std::system(command.c_str());
    if (ret != 0) {
        std::cerr << "Error: Python script exited with code " << ret << std::endl;
    }
}