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

    double eta = 0.01;
    double tol = 0;
    int patience = 0;
    int max_iters = 20000000;
    int N = X_train.size();
    double reg_lambda = 0.01 / N;
    std::vector<double> errors;
    std::vector<double> val_errors;
    std::vector<Eigen::VectorXd> X_val = {};
    std::vector<Eigen::VectorXd> y_val = {};
    
    network.stochastic_gradient_descent(
        X_train,
        y_train,
        errors,
        val_errors,
        reg_lambda,
        eta,
        tol,
        patience,
        max_iters,
        identity_func,
        identity_derivative,
        X_val,
        y_val
    );

    std::cout << "Finished training.\n" << std::endl;

    std::ofstream fout("errors_sgd_wd.csv");
    for (size_t i = 0; i < errors.size(); i++) {
        fout << i << "," << errors[i] << "\n";
    }
    fout.close();

    std::cout << "Errors saved to errors_sgd_wd.csv." << std::endl;
    plot_errors_from_csv(
        "errors_sgd_wd.csv", 
        "errors_sgd_wd.png",
        "plot_errors.py",
        "SGD with Weight Decay"
    );
    std::cout << "Errors saved to errors_sgd_wd.png." << std::endl;

    plot_decision_boundary_from_NN(
        network,
        X_train,
        "zipDigitsRandom.train",
        "grid_predictions_SGDWD.csv",
        "plot_decision_boundary.py",
        "decision_boundary_sgd_wd.png",
        "SGD with Weight Decay"
    );
    std::cout << "Decision boundary saved to decision_boundary_sgd_wd.png." << std::endl;

    double test_error = network.test_error(
        X_test,
        y_test,
        identity_func
    );
    std::cout << "Final Test Error: " << test_error << std::endl;

    return 0;
}