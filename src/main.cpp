#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>
#include "nn.h"
#include "layers.h"
#include "helper_functions.cpp"

Eigen::MatrixXd flatten_weights(
    const Eigen::MatrixXd& weights,
    const Eigen::VectorXd& biases
) {
    int n_neurons = weights.rows();
    int n_inputs = weights.cols();

    Eigen::MatrixXd flat(n_neurons, n_inputs + 1);

    for (int i = 0; i < n_neurons; i++) {
        flat(i, 0) = biases(i);
        flat.row(i).segment(1, n_inputs) = weights.row(i);
    }

    return flat.transpose();
}

void run_one_layer(
    NN& network, 
    const std::vector<Eigen::VectorXd>& X,
    const std::vector<Eigen::VectorXd>& y,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_func,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_derivative,
    double e_in,
    std::vector<Eigen::MatrixXd> grad_w,
    std::vector<Eigen::VectorXd> grad_b
) {
    std::cout << "\n" << std::endl;
    double reg_lambda = 0.0;

    network.gradient_descent(
        X, y, 
        e_in, 
        grad_w, 
        grad_b, 
        reg_lambda,
        activation_func, 
        activation_derivative
    );

    std::cout << "E_in = " << e_in << std::endl;

    for (size_t l = 0; l < grad_w.size(); l++) {
        Eigen::MatrixXd gradient_layer = flatten_weights(grad_w[l], grad_b[l]);
        std::cout << "Layer " << l+1 << ":\n" << gradient_layer << std::endl;
    }

    std::cout << "\n" << std::endl;
}

int main() {
    Layer hidden(2, 2);
    Layer output(2, 1);

    std::vector<Layer> layers = {hidden, output};
    NN network(layers);

    std::vector<Eigen::VectorXd> X(1), y(1);
    X[0] = Eigen::VectorXd(2);
    X[0] << 2, 1;
    y[0] = Eigen::VectorXd(1);
    y[0] << -1;

    double e_in;
    std::vector<Eigen::MatrixXd> grad_w;
    std::vector<Eigen::VectorXd> grad_b;

    std::cout << "tanh results:\n" << std::endl;
    run_one_layer(
        network, X, y, 
        tanh_func, 
        tanh_derivative, 
        e_in, 
        grad_w, 
        grad_b
    );

    std::cout << "identity results:\n" << std::endl;
    run_one_layer(
        network, X, y, 
        identity_func, 
        identity_derivative, 
        e_in, 
        grad_w, 
        grad_b
    );

    std::cout << "\n" << std::endl;

    for (auto& layer : network.get_layers()) {
        Eigen::MatrixXd w = Eigen::MatrixXd::Constant(
            layer.get_weights().rows(), 
            layer.get_weights().cols(), 
            0.15 + 1e-4
        );

        Eigen::VectorXd b = Eigen::VectorXd::Constant(
            layer.get_biases().size(), 
            0.15 + 1e-4
        );

        layer.set_weights(w);
        layer.set_biases(b);
    }

    double perturbed_e_in;
    std::vector<Eigen::MatrixXd> perturbed_grad_w;
    std::vector<Eigen::VectorXd> perturbed_grad_b;

    std::cout << "tanh perturbed results:\n" << std::endl;
    run_one_layer(
        network, X, y, 
        tanh_func, 
        tanh_derivative, 
        perturbed_e_in, 
        perturbed_grad_w, 
        perturbed_grad_b
    );

    std::cout << "identity perturbed results:\n" << std::endl;
    run_one_layer(
        network, X, y, 
        identity_func, 
        identity_derivative, 
        perturbed_e_in, 
        perturbed_grad_w, 
        perturbed_grad_b
    );

    return 0;
}