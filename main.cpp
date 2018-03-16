#include "dataset.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace Neuralnetwork;

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
        cout << "the number of patterns is: " << dataset.get_num_of_patterns()
             << endl;
        cout << "the number of input attributes are: "
             << dataset.get_num_of_input_attribute() << endl;
        cout << "the number of output attributes are: "
             << dataset.get_num_of_output_attribute() << endl;
    } else {
        std::cout << "can't open param.txt file to initialize neural network "
                     "structure, exit..."
                  << std::endl;
        std::exit(1);
    }
    return 0;
}