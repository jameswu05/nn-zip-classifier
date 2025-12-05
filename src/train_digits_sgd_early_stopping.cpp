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
    std::vector<Eigen::VectorXd> X, y;
    std::vector<Eigen::VectorXd> X_train, y_train;
    std::vector<Eigen::VectorXd> X_val, y_val;
    std::vector<Eigen::VectorXd> X_test, y_test;

    load_data("../src/ZipDigitsRandom.train", X, y);
    load_data("../src/ZipDigitsRandom.test", X_test, y_test);

    split_training_validation(
        X, y,
        X_train, y_train,
        X_val, y_val,
        250,
        50
    );

    std::cout << "Training samples: " << X_train.size() << std::endl;
    std::cout << "Validation samples: " << X_val.size() << std::endl;

    Layer hidden(2, 10);
    initialize_random_weights(hidden);
    
    Layer output(10, 1);
    initialize_random_weights(output);

    std::vector<Layer> layers = {hidden, output};
    NN network(layers);

    double eta = 0.01;
    double tol = 1e-6;
    int patience = 100;
    int max_iters = 20000000;
    int N = X_train.size();
    double reg_lambda = 0.0;
    std::vector<double> errors;
    std::vector<double> val_errors;

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

    std::ofstream fout("../data/errors_sgd_es.csv");
    for (size_t i = 0; i < errors.size(); i++) {
        fout << i << "," << errors[i] << "\n";
    }
    fout.close();

    std::cout << "Errors saved to data/errors_sgd_es.csv." << std::endl;

    std::ofstream f_out("../data/val_errors_sgd_es.csv");
    for (size_t i = 0; i < val_errors.size(); i++) {
        f_out << i << "," << val_errors[i] << "\n";
    }
    f_out.close();

    std::cout << "Errors saved to data/val_errors_sgd_es.csv" << std::endl;

    plot_errors_from_csv(
        "errors_sgd_es.csv", 
        "errors_sgd_es.png",
        "plot_errors.py",
        "SGD with Early Stopping",
        "val_errors_sgd_es.csv"
    );
    std::cout << "Errors saved to plots/errors_sgd_es.png." << std::endl;

    plot_decision_boundary_from_NN(
        network,
        X_train,
        "../src/zipDigitsRandom.train",
        "../data/grid_predictions_SGDES.csv",
        "../src/plot_decision_boundary.py",
        "../plots/decision_boundary_sgd_es.png",
        "SGD with Early Stopping"
    );
    std::cout << "Decision boundary saved to plots/decision_boundary_sgd_es.png." << std::endl;

    double test_error = network.test_error(
        X_test, 
        y_test, 
        identity_func
    );
    std::cout << "Final Test Error: " << test_error << std::endl;

    return 0;
}