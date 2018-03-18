#include "dataset.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace nn {
Dataset::Dataset(const ::string& input_filename,
                 const ::string& output_filename) {

    ifstream inputfs(input_filename);
    ifstream teachingInputfs(output_filename);

    if (!inputfs.good()) {
        cout << "can not load input file: " << input_filename << endl;
        exit(1);
    }
    if (!teachingInputfs.good()) {
        cout << "can not load teaching input file: " << output_filename << endl;
        exit(1);
    }

    for (string line; getline(inputfs, line);) {
        stringstream ss;
        ss << line;
        string         token;
        vector<double> inputPatternRecord;
        while (ss >> token) {
            inputPatternRecord.push_back(stod(token));
        }
        this->inputPattern.push_back(inputPatternRecord);
    }
    this->num_of_input_attribute = this->inputPattern[0].size();

    for (string line; getline(teachingInputfs, line);) {
        stringstream ss;
        ss << line;
        string         token;
        vector<double> outputPatternRecord;
        while (ss >> token) {
            outputPatternRecord.push_back(stod(token));
        }
        this->outputPattern.push_back(outputPatternRecord);
    }
    this->num_of_output_attribute = this->outputPattern[0].size();

    if (this->inputPattern.size() == this->outputPattern.size()) {
        this->num_of_patterns = this->inputPattern.size();
    } else {
        cout << "The inputPattern and outputPatten doesn't match: \n"
             << "the number of inputPatter is: " << this->inputPattern.size()
             << "\n the number of outputPattern is: "
             << this->outputPattern.size() << endl;
        exit(1);
    }
}

const vector<vector<double>>& Dataset::getInputPattern() const {
    return this->inputPattern;
}
const vector<vector<double>>& Dataset::getOutputPattern() const {
    return this->outputPattern;
}

const int Dataset::get_num_of_input_attribute() const {
    return this->num_of_input_attribute;
}

const int Dataset::get_num_of_output_attribute() const {
    return this->num_of_output_attribute;
}

const int Dataset::get_num_of_patterns() const {
    return this->num_of_patterns;
}

} // namespace nn