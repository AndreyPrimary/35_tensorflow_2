#include <stdio.h>                                                                        
#include <stdlib.h>                                                                       
#include <tensorflow/c/c_api.h>                                                           

#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>
#include <regex>

void Deallocator([[maybe_unused]]void* data, [[maybe_unused]]size_t length, [[maybe_unused]]void* arg) {
        // std::cout << "Deallocator called\n";
        // free(data);
        // *reinterpret_cast<bool*>(arg) = true;
}

class TF_container {

public:  
  TF_Buffer* RunOpts = nullptr;
  TF_Status* status = nullptr;
  TF_Graph* graph = nullptr;
  TF_SessionOptions* session_opts = nullptr;
  TF_Session* session = nullptr;

  TF_container()/* : status(TF_NewStatus(), TF_DeleteStatus)*/ {
    allocate();
  }

  ~TF_container() {
    deallocate();
  }

  void allocate () {
    deallocate();

    status = TF_NewStatus();
    graph = TF_NewGraph();
    session_opts = TF_NewSessionOptions();
  }

  void deallocate() {
    if (session != nullptr) {

      if (status == nullptr) {
        status = TF_NewStatus();
      }

      TF_CloseSession(session, status);

      if (TF_GetCode(status) != TF_OK) {
        std::cout << "ERROR: Unable to close session " << TF_Message(status) << std::endl;
      }

      TF_DeleteSession(session, status);

      if (TF_GetCode(status) != TF_OK) {
        std::cout << "ERROR: Unable to delete session " << TF_Message(status) << std::endl;
      }

      session = nullptr;
    }

    if (session_opts != nullptr) {
      TF_DeleteSessionOptions(session_opts);
      session_opts = nullptr;
    }

    if (graph != nullptr) {
      TF_DeleteGraph(graph);
      graph = nullptr;
    }

    if (status != nullptr) {
      TF_DeleteStatus(status);
      status = nullptr;
    }

    if (RunOpts != nullptr) {
      TF_DeleteBuffer(RunOpts);
      RunOpts = nullptr;
    }
  }
};

using features_t = std::vector<float>;
using probas_t = std::vector<float>;
using features_data_t = struct {
  int product_type;
  features_t features; 
};
//
using feat_arr_t = std::vector<features_data_t>;

feat_arr_t read_file (const std::string &testDataFile);

// ######################
// Set up graph inputs
// ######################
int SetGraphInput(
  std::unique_ptr<TF_container> &tf_cont,
  features_data_t               &features_data,

  std::vector<TF_Output>        &inputs, 
  std::vector<TF_Tensor*>       &input_values
  ) 
{  
  // Create vectors to store graph input operations and input tensors
  // std::vector<TF_Output> inputs;
  // std::vector<TF_Tensor*> input_values;

  // Create variables to store the size of the input variables
  int64_t in_dims[] = {1, 28, 28, 1};

  // Set input dimensions - this should match the dimensionality of the input in
  // the loaded graph, in this case it's three dimensional.
  const int num_bytes_in = 28 * 28 * sizeof(float);

  // Pass the graph and a string name of your input operation
  // (make sure the operation name is correct)
  TF_Operation* input_op = TF_GraphOperationByName(tf_cont->graph, "serving_default_input");
  if (input_op == nullptr) {
    std::cout << "Operation 'serving_default_input' not found in graph" << std::endl;
    return 1;
  }
  // std::cout << "Input op info: " << TF_OperationNumOutputs(input_op) << "\n";

  TF_Output input_opout = {input_op, 0};
  inputs.push_back(input_opout);

  // Create the input tensor using the dimension (in_dims) and size (num_bytes_in)
  // variables created earlier
  TF_Tensor* input = TF_NewTensor(TF_FLOAT, in_dims, 4, features_data.features.data(), num_bytes_in, &Deallocator, 0);
  if (input == nullptr) {
      std::cerr << "Error: TF_NewTensor" << std::endl;
      return 1;
  }
  // Optionally, you can check that your input_op and input tensors are correct
  // by using some of the functions provided by the C API.
  // std::cout << "Input data info: " << TF_Dim(input, 0) << "\n";

  input_values.push_back(input);

  return 0;
}
  
// ######################
// Set up graph outputs (similar to setting up graph inputs)
// ######################
int SetGraphOutput(
  std::unique_ptr<TF_container> &tf_cont,

  std::vector<TF_Output>        &outputs,
  std::vector<TF_Tensor*>       &output_values
  ) 
{  
  // Create variables to store the size of the output variables
  int64_t out_dims[] = {1, 10};

  // Set input dimensions - this should match the dimensionality of the input in
  // the loaded graph, in this case it's three dimensional.
  const int num_bytes_out = 10 * sizeof(float);

  // Create vector to store graph output operations
  // std::vector<TF_Output> outputs;

  TF_Operation* output_op = TF_GraphOperationByName(tf_cont->graph, "StatefulPartitionedCall");
  if (output_op == nullptr) {
    std::cout << "Operation 'StatefulPartitionedCall' not found in graph" << std::endl;
    return 1;
  }
  TF_Output output_opout = {output_op, 0};
  outputs.push_back(output_opout);

  // Create TF_Tensor* vector
  // std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);
  output_values.resize(outputs.size(), nullptr);

  // Similar to creating the input tensor, however here we don't yet have the
  // output values, so we use TF_AllocateTensor()
  TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 2, num_bytes_out);
  output_values.push_back(output_value);

  // As with inputs, check the values for the output operation and output tensor
  // std::cout << "Output: " << TF_OperationName(output_op) << "\n";
  // std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";
  // std::cout << "Output info: " << TF_Dim(output_value, 1) << "\n";

  return 0;
}

int CalculatePrediction(
  std::unique_ptr<TF_container> &tf_cont,

  std::vector<TF_Output>        &inputs,
  std::vector<TF_Tensor*>       &input_values, 
  std::vector<TF_Output>        &outputs,
  std::vector<TF_Tensor*>       &output_values,
  int                           &predicted
  )
{
  // Call TF_SessionRun
  TF_SessionRun(tf_cont->session, nullptr,
                &inputs[0], &input_values[0], inputs.size(),
                &outputs[0], &output_values[0], outputs.size(),
                nullptr, 0, nullptr, tf_cont->status);

  if (TF_GetCode(tf_cont->status) != TF_OK) {
          std::cout << "ERROR: Unable to run session " << TF_Message(tf_cont->status) << std::endl;
          return 1;
  }

  probas_t probas;
  float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
  for (int i = 0; i < 10; ++i)
  {
      probas.push_back(*out_vals);
      // std::cout << "Output values[" << i << "] info: " << *out_vals << "\n";
      out_vals++;
  }

  auto argmax = std::max_element(probas.begin(), probas.end());
  predicted = std::distance(probas.begin(), argmax);

  return 0;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
      std::cout << "Usage: " << argv[0] <<  " modelpath test_data_file" << std::endl;
      return 1;
  }    

  std::string   export_dir      = argv[1];
  std::string   testDataFile    = argv[2];
  const char*   tags = "serve"; // default model serving tag;
  int ntags = 1;

  auto feat_arr = read_file (testDataFile);

  std::unique_ptr<TF_container> tf_cont(new TF_container);

  // Load saved TensorFlow session
  std::cout << "Start load TensorFlow saved model from " << export_dir << std::endl;

  // - `export_dir` must be set to the path of the exported SavedModel.
  // - `tags` must include the set of tags used to identify one MetaGraphDef in
  //    the SavedModel.
  // - `graph` must be a graph newly allocated with TF_NewGraph().
  tf_cont->session = TF_LoadSessionFromSavedModel(
    tf_cont->session_opts,      tf_cont->RunOpts, 
    export_dir.c_str(),         &tags, 
    ntags,                      tf_cont->graph, 
    NULL,                       tf_cont->status
    );

  if (TF_GetCode(tf_cont->status) != TF_OK) {
          std::cout << "ERROR: Unable to load session from saved model '" << export_dir 
            << "' Error: " << TF_Message(tf_cont->status) << std::endl;

          return 1;
  } 

  std::cout << "Successfully load session" << std::endl;

  // size_t pos = 0;
  // TF_Operation* oper;
  // while ((oper = TF_GraphNextOperation(tf_cont->graph, &pos)) != nullptr) {
  //     std::cout << "Input: " << TF_OperationName(oper) << "\n";
  // }

  int count_positive = 0;
  int count_negative = 0;

  // Check on test data
  for (auto features_data : feat_arr) {
   
    std::vector<TF_Output>        inputs;
    std::vector<TF_Tensor*>       input_values;

    int result = SetGraphInput(tf_cont, features_data, inputs, input_values);

    if (result != 0) {
      std::cout << "ERROR: SetGraphInput " << std::endl;
      return result;
    }

    std::vector<TF_Output>        outputs;
    std::vector<TF_Tensor*>       output_values;

    result = SetGraphOutput(tf_cont, outputs, output_values);
  
    if (result != 0) {
      std::cout << "ERROR: SetGraphOutput " << std::endl;
      return result;
    }

    int    predicted;

    result = CalculatePrediction(tf_cont, inputs, input_values, outputs, output_values, predicted);

    if (result != 0) {
      std::cout << "ERROR: CalculatePrediction " << std::endl;
      return result;
    }

    // std::cout << "Predicted values : " << predicted << "\n";
    // std::cout << "Estimated values : " << features_data.product_type << "\n";

    if (predicted == features_data.product_type) {
      count_positive++;
    } else {
      count_negative++;
    }

  }

  float prediction = 1;

  if (count_positive + count_negative > 0) {
    prediction = (float)count_positive / (count_positive + count_negative);
  } else {
    prediction = 0;
  }

  std::cout << "Successfully run session" << std::endl;

  std::cout << "*** Prediction " << prediction << std::endl;

  return 0;
}

feat_arr_t read_file (const std::string &testDataFile)
{
  feat_arr_t    feat_arr;
  std::string   line{};

  const std::regex comma(",");
  
  std::ifstream test_data{testDataFile.c_str()};
  if (!test_data.is_open() ) {
    std::string err = "Input data file " + testDataFile + " not found";
        throw std::runtime_error(err.c_str());
  }

  std::cout << "Start load test.csv" << std::endl;

  while ( getline (test_data,line) ) {

        std::stringstream s(line);

        std::vector<std::string>  row;
        std::string               val;

        while (getline(s, val, ',')) {
          row.push_back(val);
        }        

        features_data_t    features_data;
        features_data.features.reserve(row.size());

        // first byte - product type
        auto it = row.begin();
        features_data.product_type = stoi(*it++);

        // Divide each bytes by 255
        std::transform(it, row.end(), std::back_inserter(features_data.features),
                [](std::string val) { return stof(val) / 255;});

        feat_arr.push_back(features_data);
  }

  std::cout << "End load" << std::endl;

  return feat_arr;
}
