//
// Created by zwpdbh on 17/03/2018.
//

#include "neuralnetwork.hpp"
#include <math.h>
#include <stdlib.h>

using namespace std;

namespace nn {
NeuralNetwork::NeuralNetwork(int    num_of_input_neurons,
                             int    num_of_hidden_neurons,
                             int    num_of_output_neurons,
                             double learning_rate,
                             double momentum,
                             double learning_criterion,
                             int    epoches) {

    this->_num_of_input_neurons = num_of_input_neurons;
    this->_num_of_hidden_neurons = num_of_hidden_neurons;
    this->_num_of_output_neurons = num_of_output_neurons;
    this->_learning_rate         = learning_rate;
    this->_momentum              = momentum;
    this->_learning_criterion    = learning_criterion;
    this->_epoches               = epoches;

    // initialize weights
    this->_first_layer_weights = new double*[this->_num_of_hidden_neurons];
    for (int j = 0; j < _num_of_hidden_neurons; j++) {
        _first_layer_weights[j] = new double[this->_num_of_input_neurons];
    }

    this->_second_layer_weights = new double*[this->_num_of_output_neurons];
    for (int k = 0; k < _num_of_output_neurons; k++) {
        _second_layer_weights[k] = new double[this->_num_of_hidden_neurons];
    }

    this->_input_layer  = new double[this->_num_of_input_neurons];
    this->_hidden_layer = new double[this->_num_of_hidden_neurons];
    this->_output_layer = new double[this->_num_of_output_neurons];

    this->_biases_of_hidden_layer = new double[this->_num_of_output_neurons];
    this->_biases_of_output_layer = new double[this->_num_of_hidden_neurons];
}

double double_rand(double min, double max) {
    double scale = rand() / (double)RAND_MAX;
    return min + scale * (max - min);
}

void NeuralNetwork::initializeWeightsRandomly(double min, double max) {

    for (int k = 0; k < this->_num_of_output_neurons; k++) {
        for (int j = 0; j < this->_num_of_hidden_neurons; j++) {
            this->_second_layer_weights[k][j] = double_rand(min, max);
            cout << "second_layer_W_" << k << "_" << j << " = " << this->_second_layer_weights[k][j] << endl;
        }
        this->_biases_of_output_layer[k] = double_rand(min, max);
        cout << "bias_of_output_layer_" << k << "_" << this->_biases_of_output_layer[k] << endl;
    }

    for (int j = 0; j < this->_num_of_hidden_neurons; j++) {
        for (int i = 0; i < this->_num_of_input_neurons; i++) {
            this->_first_layer_weights[j][i] = double_rand(min, max);
            cout << "first_layer_W_" << j << "_" << i << " = " << this->_first_layer_weights[j][i] << endl;
        }
        this->_biases_of_hidden_layer[j] = double_rand(min, max);
        cout << "bias_of_hidden_layer_" << j << "_" << this->_biases_of_output_layer[j] << endl;
    }


    // for testing a specific setting of weights
//    _biases_of_output_layer[0] = -0.3;
//    _biases_of_output_layer[1] = 0.4;
//
//    _second_layer_weights[0][0] = -0.8;
//    _second_layer_weights[0][1] = 0.7;
//    _second_layer_weights[1][0] = 0.6;
//    _second_layer_weights[1][1] = -0.5;
//
//    _biases_of_hidden_layer[0] = 0.2;
//    _biases_of_hidden_layer[1] = -0.2;
//
//    _first_layer_weights[0][0] = 1.1;
//    _first_layer_weights[0][1] = 1.2;
//    _first_layer_weights[1][0] = -1.3;
//    _first_layer_weights[1][1] = -1.4;

}

void NeuralNetwork::computeForward(const Dataset& dataset, int index) {
    // the first layer's state = input
    for (int i = 0; i < this->_num_of_input_neurons; i++) {
        this->_input_layer[i] = dataset.getInputPattern()[index][i];
    }

    // the second/third layer's input = sum(w * (output of previous layer + bias
    // = 1)) go through activation function
    for (int j = 0; j < this->_num_of_hidden_neurons; j++) {
        double sum = 0.0;
        for (int i = 0; i < this->_num_of_input_neurons; i++) {
            sum += this->_input_layer[i] * this->_first_layer_weights[j][i];
        }
        sum += this->_biases_of_hidden_layer[j];

        this->_hidden_layer[j] = (1 / (1 + exp(-1 * (sum))));
    }

    for (int k = 0; k < this->_num_of_output_neurons; k++) {
        double sum = 0.0;
        for (int j = 0; j < this->_num_of_hidden_neurons; j++) {
            sum += this->_hidden_layer[j] * this->_second_layer_weights[k][j];
        }
        sum += this->_biases_of_output_layer[k];

        this->_output_layer[k] = (1 / (1 + exp(-1 * (sum))));
    }
}

void NeuralNetwork::computeBackpropagation(double* backPropagationErrors) {
    double** saveAdjustedSecondLayerWeights =
        new double*[this->_num_of_output_neurons];
    for (int k = 0; k < this->_num_of_output_neurons; k++) {
        saveAdjustedSecondLayerWeights[k] =
            new double[this->_num_of_hidden_neurons];
    }

    // for second layer weights
    double* deltaPKs = new double[this->_num_of_output_neurons];
    for (int k = 0; k < this->_num_of_output_neurons; k++) {
        deltaPKs[k] = backPropagationErrors[k] * _output_layer[k] *
                      (1 - _output_layer[k]);

        for (int j = 0; j < this->_num_of_hidden_neurons; j++) {
            saveAdjustedSecondLayerWeights[k][j] =
                _second_layer_weights[k][j] +
                (_learning_rate * deltaPKs[k] * _hidden_layer[j]);
        }
        _biases_of_output_layer[k] += (_learning_rate * deltaPKs[k] * 1.0);
    }

    // for first layer weights
    for (int j = 0; j < _num_of_hidden_neurons; j++) {
        double sumFromOutputLayer = 0.0;
        for (int k = 0; k < _num_of_output_neurons; k++) {
            sumFromOutputLayer += (deltaPKs[k] * _second_layer_weights[k][j]);
        }
        double deltaPJ =
            _hidden_layer[j] * (1 - _hidden_layer[j]) * sumFromOutputLayer;
        for (int i = 0; i < _num_of_input_neurons; i++) {
            _first_layer_weights[j][i] +=
                (_learning_rate * deltaPJ * _input_layer[i]);
        }
        _biases_of_hidden_layer[j] += (_learning_rate * deltaPJ * 1.0);
    }

    // set the weights for second layer
    for (int k = 0; k < _num_of_output_neurons; k++) {
        for (int j = 0; j < _num_of_hidden_neurons; j++) {
            this->_second_layer_weights[k][j] =
                saveAdjustedSecondLayerWeights[k][j];
        }
    }
}

void NeuralNetwork::train(Dataset& dataset) {
    this->initializeWeightsRandomly(0.1, 0.3);

    int epoch = 0;
    while (this->_epoches > 0) {
        double popErr = 0.0;
        double sum    = 0.0;
        double backPropagationErrors[this->_num_of_output_neurons];

        for (int k = 0; k < this->_num_of_output_neurons; k++) {
            backPropagationErrors[k] = 0.0;
        }

        // for each epoch
        for (int index = 0; index < dataset.get_num_of_patterns(); index++) {
            // compute forward
            this->computeForward(dataset, index);

            for (int k = 0; k < this->_num_of_output_neurons; k++) {
                double err = this->_output_layer[k] -
                             dataset.getOutputPattern()[index][k];
                sum += (err * err);
                backPropagationErrors[k] += err;
            }
        }
        // compute back propagation
        this->computeBackpropagation(backPropagationErrors);

        // compute population error;
        popErr = sum /
                 (this->_num_of_output_neurons * dataset.get_num_of_patterns());
        epoch++;
        if (epoch % 100 == 0) {
            this->_learning_rate = (_learning_rate - (_learning_rate / 10));
            cout << "epoch " << epoch << ", popErr = " << popErr << endl;
            cout << "change learning rate to : " << this->_learning_rate << endl << endl;
        }
        this->_epoches--;

        if (popErr < this->_learning_criterion) {
            break;
        }
    }
}

const int NeuralNetwork::get_num_of_input_neurons() const {
    return _num_of_input_neurons;
}


    const int NeuralNetwork::get_num_of_hidden_neurons() const {
        return _num_of_hidden_neurons;
    }

    const int NeuralNetwork::get_num_of_output_neurons() const {
        return _num_of_output_neurons;
    }

    const double NeuralNetwork::get_learning_rate() const {
        return _learning_rate;
    }

    const double NeuralNetwork::get_momentum() const {
        return _momentum;
    }

    const double NeuralNetwork::get_learning_criterion() const {
        return _learning_criterion;
    }

} // namespace nn