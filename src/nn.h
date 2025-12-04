#ifndef NN_H
#define NN_H

#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>
#include "layers.h"

class NN {
private:
    std::vector<Layer> layers;

public:
    NN(std::vector<Layer> layers) : layers(layers) {}

    std::vector<Layer> get_layers() {
        return layers;
    }

    void initialize_layers(const std::vector<Layer>& new_layers) {
        layers = new_layers;
    }

    void add_layer(const Layer& new_layer) {
        layers.push_back(new_layer);
    }

    Eigen::VectorXd forward(
        const Eigen::VectorXd& input,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> last_layer_activation = nullptr
    ) {
        Eigen::VectorXd current_input = input;
        int num_layers = layers.size();

        for (int l = 0; l < num_layers - 1; l++) {
            Layer& current_layer = layers[l];
            Eigen::MatrixXd current_weights = current_layer.get_weights();
            Eigen::VectorXd current_biases = current_layer.get_biases();
            current_layer.set_input(current_input);

            Eigen::VectorXd current_output = current_layer.propagate(
                current_weights,
                current_input,
                current_biases
            );

            current_layer.set_output(current_output);
            current_input = current_output;
        }

        Layer& last_layer = layers[num_layers - 1];
        last_layer.set_input(current_input);
        Eigen::VectorXd output;
        Eigen::MatrixXd last_weights = last_layer.get_weights();
        Eigen::VectorXd last_biases = last_layer.get_biases();
        
        if (last_layer_activation) {
            output = last_layer.last_layer_propagate(
                last_weights, 
                current_input, 
                last_biases,
                last_layer_activation
            );
        } else {
            output = last_layer.propagate(
                last_weights,
                current_input,
                last_biases
            );
        }

        return output;
    }

    double total_error(
        const Eigen::VectorXd& output, 
        const Eigen::VectorXd& target,
        const double& reg_lambda
    ) {
        // std::cout << "output: " << output.transpose() << " target: " << target.transpose() << std::endl;
        int N = output.size();
        Eigen::VectorXd diff = output - target;
        double loss_acc = diff.squaredNorm();
        double e_in = (1.0 / (4.0 * N)) * loss_acc;

        double reg_term = 0.0;
        if (reg_lambda != 0) {
            for (const auto& layer : layers) {
                reg_term += layer.get_weights().squaredNorm();
            }
            reg_term = (reg_lambda / N) * reg_term;
        }

        return e_in + reg_term;
    }

    void backwards(
        const Eigen::VectorXd& output,
        const Eigen::VectorXd& target,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> last_layer_deactivate
    ) {
        int num_layers = layers.size();
        Layer& last_layer = layers[num_layers - 1];

        Eigen::VectorXd next_delta = last_layer.last_layer_backpropagate(
            output,
            target,
            last_layer_deactivate
        );
        
        for (int l = num_layers - 2; l >= 0; l--) {
            Layer& current_layer = layers[l];
            Layer& next_layer = layers[l+1];
            Eigen::MatrixXd next_weights = next_layer.get_weights();

            Eigen::VectorXd current_delta = current_layer.backpropagate(
                next_weights, 
                next_delta
            );

            current_layer.set_delta(current_delta);
            next_delta = current_delta;
        }
    }

    void gradient_descent(
        const std::vector<Eigen::VectorXd>& X,
        const std::vector<Eigen::VectorXd>& y,
        double& e_in,
        std::vector<Eigen::MatrixXd>& gradients,
        std::vector<Eigen::VectorXd>& grad_b,
        double& reg_lambda,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> last_layer_activate,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> last_layer_deactivate
    ) {
        int num_layers = layers.size();
        int N = X.size();

        e_in = 0.0;
        gradients.resize(num_layers);
        grad_b.resize(num_layers);

        for (int i = 0; i <= num_layers - 1; i++) {
            gradients[i] = (
                Eigen::MatrixXd::Zero(
                    layers[i].get_weights().rows(), 
                    layers[i].get_weights().cols()
                )
            );

            grad_b[i] = (
                Eigen::VectorXd::Zero(
                    layers[i].get_biases().size()
                )
            );
        }

        for (int n = 0; n < N; n++) {
            Eigen::VectorXd x_output = forward(
                X[n], 
                last_layer_activate
            );

            backwards(
                x_output, 
                y[n], 
                last_layer_deactivate
            );

            e_in += total_error(
                x_output, 
                y[n],  
                reg_lambda
            );

            for (int l = 0; l <= num_layers - 1; l++) {
                Eigen::VectorXd x_prev;

                if (l == 0) {
                    x_prev = X[n];
                } else {
                    x_prev = layers[l-1].get_output();
                }

                gradients[l] += layers[l].calculate_gradient(x_prev) / N;
                grad_b[l] += layers[l].get_delta() / N;
            }

            if (reg_lambda != 0.0) {
                for (int l = 0; l <= num_layers - 1; l++) {
                    gradients[l] += ((2 * reg_lambda) / N) * layers[l].get_weights();
                }
            }
        }

        e_in /= N;
    }

    void variable_lr_gradient_descent(
        const std::vector<Eigen::VectorXd>& X,
        const std::vector<Eigen::VectorXd>& y,
        std::vector<double>& errors,
        std::vector<double>& val_errors,
        double reg_lambda=0.0,
        double eta0=0.01,
        double alpha=1.05,
        double beta=0.7,
        int max_iters=1000,
        double tol=1e-6,
        int patience=10,
        double stopping_criteria=1e-6,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> last_layer_activate=nullptr,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> last_layer_deactivate=nullptr,
        const std::vector<Eigen::VectorXd>& X_val = {},
        const std::vector<Eigen::VectorXd>& y_val = {}
    ) {
        bool use_val = (!X_val.empty() && !y_val.empty());
        double eta = eta0;
        int num_layers = layers.size();
        int N = X.size();
        int N_val = X_val.size();

        std::vector<Eigen::MatrixXd> gradients(num_layers);
        std::vector<Eigen::VectorXd> grad_b(num_layers);

        std::vector<Eigen::MatrixXd> best_weights(num_layers);
        std::vector<Eigen::VectorXd> best_biases(num_layers);
        double best_val_error = std::numeric_limits<double>::max();
        int patience_counter = 0;

        for (int l = 0; l < num_layers; l++) {
            best_weights[l] = layers[l].get_weights();
            best_biases[l] = layers[l].get_biases();
        }

        double prev_e_in = std::numeric_limits<double>::infinity();
        int num_iters = 0;
        while (num_iters < max_iters) {
            if (num_iters % 10000 == 0) {
                std::cout << "Finished iteration " << num_iters << std::endl;
            }

            std::vector<Eigen::MatrixXd> prev_weights(num_layers);
            std::vector<Eigen::VectorXd> prev_biases(num_layers);
            for (int l = 0; l < num_layers; l++) {
                prev_weights[l] = layers[l].get_weights();
                prev_biases[l] = layers[l].get_biases();
            }

            double e_in = 0.0;
            gradient_descent(
                X, y, e_in,
                gradients,
                grad_b,
                reg_lambda,
                last_layer_activate,
                last_layer_deactivate
            );

            errors.push_back(e_in);

            for (int l = 0; l < num_layers; l++) {
                layers[l].update_weights(
                    gradients[l],
                    grad_b[l],
                    eta
                );
            }

            double new_e_in = 0.0;
            gradient_descent(
                X, y, new_e_in,
                gradients,
                grad_b,
                reg_lambda,
                last_layer_activate,
                last_layer_deactivate
            );

            if (new_e_in < e_in) {
                prev_e_in = new_e_in;
                eta *= alpha;
            } else {
                for (int l = 0; l < num_layers; l++) {
                    layers[l].set_weights(prev_weights[l]);
                    layers[l].set_biases(prev_biases[l]);
                }

                eta *= beta;
            }

            if (use_val) {
                double val_error = 0.0;
                for (int n = 0; n < N_val; n++) {
                    Eigen::VectorXd val_output = forward(
                        X_val[n], 
                        last_layer_activate
                    );

                    val_error += total_error(
                        val_output,
                        y_val[n],
                        reg_lambda
                    );
                }
                val_error /= N_val;

                val_errors.push_back(val_error);

                if (val_error < best_val_error - tol) {
                    best_val_error = val_error;
                    patience_counter = 0;

                    for (int l = 0; l < num_layers; l++) {
                        best_weights[l] = layers[l].get_weights();
                        best_biases[l] = layers[l].get_biases();
                    }

                } else {
                    patience_counter++;
                    if (patience_counter >= patience) {
                        std::cout << "Early stopping triggered at iteration " 
                        << num_iters << ". Minimum Validation Error = "
                        << best_val_error << std::endl;
                        break;
                    }
                }
            }

            num_iters++;
        }

        if (use_val) {
            for (int l = 0; l < num_layers; l++) {
                layers[l].set_weights(best_weights[l]);
                layers[l].set_biases(best_biases[l]);
            }
        }
    }

    void stochastic_gradient_descent(
        const std::vector<Eigen::VectorXd>& X,
        const std::vector<Eigen::VectorXd>& y,
        std::vector<double>& errors,
        std::vector<double>& val_errors,
        double reg_lambda=0.0,
        double eta=0.01,
        double tol=1e-6,
        int patience=10,
        int max_updates=1000,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> last_layer_activate = nullptr,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> last_layer_deactivate = nullptr,
        const std::vector<Eigen::VectorXd>& X_val = {},
        const std::vector<Eigen::VectorXd>& y_val = {}
    ) {
        bool use_val = (!X_val.empty() && !y_val.empty());
        int num_layers = layers.size();
        int N = X.size();
        int N_val = X_val.size();

        errors.clear();
        val_errors.clear();

        double best_val_error = std::numeric_limits<double>::infinity();
        int patience_counter = 0;

        std::vector<Eigen::MatrixXd> best_weights(num_layers);
        std::vector<Eigen::VectorXd> best_biases(num_layers);

        for (int l = 0; l < num_layers; l++) {
            best_weights[l] = layers[l].get_weights();
            best_biases[l] = layers[l].get_biases();
        }

        int num_updates = 0;
        while (num_updates < max_updates) {
            if (num_updates % 1000000 == 0) {
                std::cout << "Finished iteration " << num_updates << std::endl;
            }

            int idx = rand() % N;

            Eigen::VectorXd x_output = forward(
                X[idx], 
                last_layer_activate
            );

            backwards(
                x_output, 
                y[idx], 
                last_layer_deactivate
            );

            for (int l = 0; l < num_layers; l++) {
                Eigen::VectorXd x_prev;

                if (l == 0) {
                    x_prev = X[idx];
                } else {
                    x_prev = layers[l-1].get_output();
                }

                Eigen::MatrixXd gradients_layer = layers[l].calculate_gradient(x_prev);
                gradients_layer += ((2.0 * reg_lambda) / N) * layers[l].get_weights();

                Eigen::VectorXd grad_b_layer = layers[l].get_delta();

                layers[l].update_weights(
                    gradients_layer, 
                    grad_b_layer, 
                    eta
                );
            }

            if (num_updates % N == 0) {
                double e_in = 0.0;

                for (int i = 0; i < N; i++) {
                    Eigen::VectorXd output = forward(
                        X[i], 
                        last_layer_activate
                    );

                    e_in += total_error(
                        output, 
                        y[i],
                        reg_lambda
                    );
                }

                double error = e_in / N;
                errors.push_back(error);

                if (use_val) {
                    double val_error = 0.0;
                    for (int i = 0; i < N_val; i++) {
                        Eigen::VectorXd val_output = forward(
                            X_val[i], 
                            last_layer_activate
                        );

                        val_error += total_error(
                            val_output,
                            y_val[i],
                            reg_lambda
                        );
                    }
                    val_error /= N_val;

                    val_errors.push_back(val_error);

                    if (val_error < best_val_error - tol) {
                        best_val_error = val_error;
                        patience_counter = 0;

                        for (int l = 0; l < num_layers; l++) {
                            best_weights[l] = layers[l].get_weights();
                            best_biases[l] = layers[l].get_biases();
                        }

                    } else {
                        patience_counter++;
                        if (patience_counter >= patience) {
                            std::cout << "Early stopping triggered at iteration "
                            << num_updates << ". Minimum Validation Error = "
                            << best_val_error << std::endl;

                            for (int l = 0; l < num_layers; l++) {
                                layers[l].set_weights(best_weights[l]);
                                layers[l].set_biases(best_biases[l]);
                            }

                            return;
                        }
                    }
                }
            }

            num_updates++;
        }
    }

    double test_error(
        const std::vector<Eigen::VectorXd>& X_test,
        const std::vector<Eigen::VectorXd>& y_test,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> last_layer_activate=nullptr
    ) {
        int N = X_test.size();
        double total = 0.0;

        for (int i = 0; i < N; i++) {
            Eigen::VectorXd output = forward(X_test[i], last_layer_activate);
            Eigen::VectorXd diff = output - y_test[i];
            double loss = diff.squaredNorm(); 
            total += loss;
        }

        return (1.0 / (4.0 * N)) * total;
    }

};

#endif