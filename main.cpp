#include "dataset.hpp"
#include "neuralnetwork.hpp"
#include <fstream>

using namespace std;
using namespace nn;


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "need to specify 3 files: param.txt, in.txt and teach.txt"
                  << std::endl;

        std::cout << "You have entered " << argc << " arguments: "
                  << "\n";

        for (int i = 0; i < argc; i++) {
            std::cout << argv[i] << "\n";
        }
        std::exit(1);
    }

    std::ifstream paramfs(argv[1]);
    int           num_of_inputLayerNeurons;
    int           num_of_hiddenLayerNeurons;
    int           num_of_outputLayerNeurons;
    double        learning_rate;
    double        momentum;
    double        trainning_criterion;

    if (paramfs.good()) {
        // process parameters
        int i = 0;
        for (std::string line; getline(paramfs, line);) {
            switch (i) {
            case 0:
                num_of_inputLayerNeurons = stoi(line);
                break;
            case 1:
                num_of_hiddenLayerNeurons = stoi(line);
                break;
            case 2:
                num_of_outputLayerNeurons = stoi(line);
                break;
            case 3:
                learning_rate = stod(line);
                break;
            case 4:
                momentum = stod(line);
                break;
            case 5:
                trainning_criterion = stod(line);
                break;
            default:
                break;
            }
            i++;
        }

        // process data: load them into dataSet
        Dataset dataset(argv[2], argv[3]);
        cout << "the settings for Dataset is: " << endl;
        cout << "the number of patterns is: " << dataset.get_num_of_patterns()
             << endl;
        cout << "the number of input attributes are: "
             << dataset.get_num_of_input_attribute() << endl;
        cout << "the number of output attributes are: "
             << dataset.get_num_of_output_attribute() << endl;

        // train
        nn::NeuralNetwork nn(num_of_inputLayerNeurons,
                             num_of_hiddenLayerNeurons,
                             num_of_outputLayerNeurons,
                             learning_rate,
                             momentum,
                             trainning_criterion,
                             7000);

        cout << "\nthe settings for NN: " << endl;
        cout << "num_of_inputLayerNeurons = " << nn.get_num_of_input_neurons() << endl;
        cout << "num_of_hiddenLayerNeurons = " << nn.get_num_of_hidden_neurons() << endl;
        cout << "num_of_outputLayerNeurons = " << nn.get_num_of_output_neurons() << endl;
        cout << "learning_rate = " << nn.get_learning_rate() << endl;
        cout << "momentum = " << nn.get_momentum() << endl;
        cout << "trainning_criterion = " << nn.get_learning_criterion() << endl << endl;

        nn.train(dataset);

    } else {
        std::cout << "can't open param.txt file to initialize neural network "
                     "structure, exit..."
                  << std::endl;
        std::exit(1);
    }
    return 0;
}