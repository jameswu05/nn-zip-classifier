#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <cassert>
#include <vector>
#include <cstdlib>

class Layer {
private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd input;
    Eigen::VectorXd output;
    Eigen::VectorXd delta;

public:
    Layer(size_t input_size, size_t output_size) {
        weights = Eigen::MatrixXd::Constant(output_size, input_size, 0.15);
        biases = Eigen::VectorXd::Constant(output_size, 0.15);
    }

    Eigen::MatrixXd get_weights() const {
        return weights;
    }

    Eigen::VectorXd get_biases() const {
        return biases;
    }

    Eigen::VectorXd get_input() const {
        return input;
    }

    Eigen::VectorXd get_output() const {
        return output;
    }

    Eigen::VectorXd get_delta() const {
        return delta;
    }

    void set_weights(Eigen::MatrixXd new_weights) {
        weights = new_weights;
    }

    void set_biases(Eigen::VectorXd new_biases) {
        biases = new_biases;
    }

    void set_input(Eigen::VectorXd new_input) {
        input = new_input;
    }

    void set_output(Eigen::VectorXd new_output) {
        output = new_output;
    }

    void set_delta(Eigen::VectorXd new_delta) {
        delta = new_delta;
    }

    Eigen::VectorXd tanh(const Eigen::VectorXd& z) const {
        return z.array().tanh();
    }

    Eigen::VectorXd tanh_derivative(const Eigen::VectorXd& z) const { 
        return (1.0 - output.array().square()).matrix();
    }

    Eigen::VectorXd identity(const Eigen::VectorXd& z) const { 
        return z;
    }

    Eigen::VectorXd identity_derivative(const Eigen::VectorXd& z) const { 
        return Eigen::VectorXd::Ones(z.size());
    }

    Eigen::VectorXd sign_func(const Eigen::VectorXd& z) const {
        return z.unaryExpr([](double v){ return (v >= 0 ? 1.0 : -1.0); });
    }

    Eigen::VectorXd sign_derivative(const Eigen::VectorXd& z) const {
        return Eigen::VectorXd::Zero(z.size()); 
    }

    Eigen::VectorXd propagate(
        const Eigen::MatrixXd& weights, 
        const Eigen::VectorXd& x, 
        const Eigen::VectorXd& biases
    ) {
        input = x;
        output = tanh(weights * x + biases);
        return output;
    }

    Eigen::VectorXd last_layer_propagate(
        const Eigen::MatrixXd& weights, 
        const Eigen::VectorXd& x, 
        const Eigen::VectorXd& biases,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_func
    ) {
        input = x;
        output = activation_func(weights * x + biases);
        return output;
    }

    Eigen::VectorXd backpropagate(
        const Eigen::MatrixXd& next_weights,
        const Eigen::VectorXd& next_delta
    ) {
        Eigen::VectorXd deriv = tanh_derivative(output);
        delta = deriv.cwiseProduct(next_weights.transpose() * next_delta);
        return delta;
    } 
    
    Eigen::VectorXd last_layer_backpropagate(
        const Eigen::MatrixXd& last_output,
        const Eigen::VectorXd& target,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> deriv_func
    ) {
        Eigen::VectorXd deriv = deriv_func(output);
        delta = ((1.0 / 2.0) * (last_output - target)).cwiseProduct(deriv);
        return delta;
    }

    Eigen::MatrixXd calculate_gradient(const Eigen::VectorXd& x_prev) {
        return delta * x_prev.transpose();
    }

    void update_weights(
        const Eigen::MatrixXd& grad_w,
        const Eigen::VectorXd& grad_b,
        double eta
    ) {
        weights -= eta * grad_w;
        biases -= eta * grad_b;
    }
};

#endif