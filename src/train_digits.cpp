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
#include "helper_functions.cpp"

int main() {
    std::vector<Eigen::VectorXd> X_train, y_train;
    std::vector<Eigen::VectorXd> X_test, y_test;

    load_data("zipDigitsRandom.train", X_train, y_train);
    load_data("zipDigitsRandom.test", X_test, y_test);

    std::cout << "Training samples: " << X_train.size() << std::endl;
    std::cout << "Testing samples: " << X_test.size() << std::endl;

    Layer hidden(2, 10);
    initialize_random_weights(hidden);
    
    Layer output(10, 1);
    initialize_random_weights(output);

    std::vector<Layer> layers = {hidden, output};
    NN network(layers);

    double eta0 = 0.01;
    double alpha = 1.05;
    double beta = 0.7;
    int max_iters = 2000000;
    double stopping_criteria = 1e-6;
    double reg_lambda = 0.0;
    double tol = 0;
    int patience = 0;
    std::vector<double> errors;
    std::vector<double> val_errors;
    std::vector<Eigen::VectorXd> X_val = {};
    std::vector<Eigen::VectorXd> y_val = {};
    
    network.variable_lr_gradient_descent(
        X_train,
        y_train,
        errors,
        val_errors,
        reg_lambda,
        eta0,
        alpha,
        beta,
        max_iters,
        tol,
        patience,
        stopping_criteria,
        identity_func,
        identity_derivative,
        X_val,
        y_val
    );

    std::cout << "Finished training.\n" << std::endl;

    std::ofstream fout("errors_var.csv");
    for (size_t i = 0; i < errors.size(); i++) {
        fout << i << "," << errors[i] << "\n";
    }
    fout.close();

    std::cout << "Errors saved to errors_var.csv." << std::endl;
    plot_errors_from_csv(
        "errors_var.csv", 
        "errors_var_lr_gd.png",
        "plot_errors.py",
        "Variable-LR Gradient Descent"
    );
    std::cout << "Errors saved to errors_var_lr_gd.png." << std::endl;

    plot_decision_boundary_from_NN(
        network,
        X_train,
        "zipDigitsRandom.train",
        "grid_predictions_VLRGD.csv",
        "plot_decision_boundary.py",
        "decision_boundary_var.png",
        "Variable-LR Gradient Descent"
    );
    std::cout << "Decision boundary saved to decision_boundary_var.png." << std::endl;

    double test_error = network.test_error(
        X_test,
        y_test,
        identity_func
    );
    std::cout << "Final Test Error: " << test_error << std::endl;

    return 0;
}