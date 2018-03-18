//
// Created by zwpdbh on 17/03/2018.
//

#ifndef NERUAL_NETWORK_NEURALNETWORK_HPP
#define NERUAL_NETWORK_NEURALNETWORK_HPP

#include <dataset.hpp>
#include <vector>

namespace nn {

class NeuralNetwork {
  private:
    int _num_of_input_neurons;
    int _num_of_hidden_neurons;
    int _num_of_output_neurons;

    double _learning_rate;
    double _momentum;
    double _learning_criterion;

    int      _epoches;
    double** _first_layer_weights;
    double** _second_layer_weights;
    double*  _biases_of_hidden_layer;
    double*  _biases_of_output_layer;

    double* _input_layer;
    double* _hidden_layer;
    double* _output_layer;

  public:
    NeuralNetwork(int    num_of_input_neurons,
                  int    num_of_hidden_neurons,
                  int    num_of_output_neurons,
                  double learning_rate,
                  double momentum,
                  double learning_criterion,
                  int    epoches);

    void   train(nn::Dataset& dataset);
    void   computeForward(const nn::Dataset& dataset, int index);
    void   computeBackpropagation(double* backpropagationError);
    void   initializeWeightsRandomly(double min, double max);
    const int get_num_of_input_neurons() const;
    const int get_num_of_hidden_neurons() const;
    const int get_num_of_output_neurons() const;
    const double get_learning_rate() const ;
    const double get_momentum() const;
    const double get_learning_criterion() const;
};

} // namespace nn

#endif // NERUAL_NETWORK_NEURALNETWORK_HPP
