#ifndef neural_netowrk_dataset_hpp
#define neural_netowrk_dataset_hpp

#include <iostream>
#include <vector>

namespace nn {

class Dataset {
  private:
    std::vector<std::vector<double>> inputPattern;
    std::vector<std::vector<double>> outputPattern;
    int                              num_of_patterns;
    int                              num_of_input_attribute;
    int                              num_of_output_attribute;

  public:
    Dataset(const std::string& input_filename,
            const std::string& output_filename);
    const std::vector<std::vector<double>>& getInputPattern() const;
    const std::vector<std::vector<double>>& getOutputPattern() const;
    const int                               get_num_of_input_attribute() const;
    const int                               get_num_of_output_attribute() const;
    const int                               get_num_of_patterns() const;
};
} // namespace nn

#endif // !neural_netowrk_dataset_hpp
