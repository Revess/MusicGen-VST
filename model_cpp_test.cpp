#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <iostream>

// Softmax function in C++
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> exp_values(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Calculate exp(logits - max) for numerical stability
    float sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exp_values[i] = std::exp(logits[i] - max_logit);
        sum_exp += exp_values[i];
    }

    // Normalize
    for (size_t i = 0; i < logits.size(); ++i) {
        exp_values[i] /= sum_exp;
    }

    return exp_values;
}

// Multinomial sampling in C++
int multinomial(const std::vector<float>& probabilities) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    return dist(gen);
}


int main() {
    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelTest");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    Ort::Session session(env, "./musicgen-small/text_encoder.onnx", session_options);

    


    // // Prepare input data (replace with your actual input)
    // std::vector<int64_t> input_shape = {4, 16};  // For example, 4x16 input
    // std::vector<int64_t> input_data(input_shape[0] * input_shape[1], -1);

    // // Create OrtMemoryInfo
    // Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // // Create input tensor with the correct API
    // Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    // // Input names (replace with actual names from the ONNX model)
    // const char* input_names[] = {"input_ids"};

    // // Output names (replace with actual output names from the ONNX model)
    // const char* output_names[] = {"logits"};

    // // Run inference
    // auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // // Extract output (logits)
    // float* logits = output_tensors[0].GetTensorMutableData<float>();

    return 0;
}