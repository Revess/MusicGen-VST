#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <random>
#include <filesystem>
#include <sstream>
#include <regex>

#include "./include/tokenizer.hpp"
#include <core/onnxruntime_cxx_api.h>

// Define a helper function to convert std::string to ORTCHAR_T
std::basic_string<ORTCHAR_T> ConvertToOrtChar(const std::string& str) {
#ifdef _WIN32
    // On Windows, ORTCHAR_T is wchar_t, so we need to convert std::string to std::wstring
    std::wstring wstr(str.begin(), str.end());
    return wstr;
#else
    // On non-Windows platforms, ORTCHAR_T is char, so we can directly return the std::string
    return str;
#endif
}

// Softmax function
std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x, int axis = -1) {
    std::vector<std::vector<float>> exp_x(x.size(), std::vector<float>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        float max_val = *std::max_element(x[i].begin(), x[i].end());
        float sum_exp = 0.0f;
        for (size_t j = 0; j < x[i].size(); ++j) {
            exp_x[i][j] = std::exp(x[i][j] - max_val);
            sum_exp += exp_x[i][j];
        }
        for (size_t j = 0; j < x[i].size(); ++j) {
            exp_x[i][j] /= sum_exp;
        }
    }
    return exp_x;
}

// Multinomial sampling
std::vector<int> multinomial(const std::vector<float>& probs) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return {dist(gen)};
}

// Apply decoder mask
std::vector<std::vector<int>> apply_mask(const std::vector<std::vector<int>>& mask, const std::vector<std::vector<int>>& ids) {
    std::vector<std::vector<int>> result = ids;
    size_t seq_len = ids[0].size();
    for (size_t i = 0; i < mask.size(); ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            if (mask[i][j] != -1) {
                result[i][j] = mask[i][j];
            }
        }
    }
    return result;
}

std::vector<std::vector<int64_t>> run_delay_pattern_mask(
    Ort::Session& session,
    std::vector<int64_t>& input_ids,
    int64_t pad_token_id,
    int64_t max_length) {

    // Step 3: Prepare input tensors
    Ort::AllocatorWithDefaultOptions allocator;

    // Define input names (replace with actual input names from your model)
    const char* input_names[] = {"input_ids", "pad_token_id", "max_length"};

    // Define input shapes
    std::vector<int64_t> input_ids_shape = {4, 16}; // Adjust size accordingly
    std::vector<int64_t> pad_token_id_shape = {1};
    std::vector<int64_t> max_length_shape = {1};

    // Step 4: Create input tensors
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());
    Ort::Value pad_token_id_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, &pad_token_id, 1, pad_token_id_shape.data(), pad_token_id_shape.size());
    Ort::Value max_length_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, &max_length, 1, max_length_shape.data(), max_length_shape.size());

    auto tensor_info = input_ids_tensor.GetTensorTypeAndShapeInfo();  // Get the type and shape info
    std::vector<int64_t> shape = tensor_info.GetShape();    // Get the shape as a vector of int64_t

    // Step 5: Define output names (replace with actual output names from your model)
    std::vector<const char*> output_names = {"input_ids_edited", "delay_pattern_mask"};  // Replace with the actual output name

    // Step 6: Prepare input tensors array (use std::move to transfer ownership)
    std::array<Ort::Value, 3> input_tensors = {std::move(input_ids_tensor), std::move(pad_token_id_tensor), std::move(max_length_tensor)};

    // Step 7: Run the model (pass the input_tensors by reference to avoid copying)
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

    // Step 8: Get the output tensor (decoder_pattern_mask)
    int64_t* decoder_pattern_mask = output_tensors[1].GetTensorMutableData<int64_t>();

    // Get the tensor shape (expected [4, max_len] i.e. [4, 256])
    std::vector<int64_t> output_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

    // Convert the output tensor to a 2D vector (vector of vectors)
    std::vector<std::vector<int64_t>> output_vector(output_shape[0], std::vector<int64_t>(output_shape[1], 0));

    // Fill the 2D vector from the flat array data
    for (size_t i = 0; i < output_shape[0]; ++i) {
        for (size_t j = 0; j < output_shape[1]; ++j) {
            output_vector[i][j] = decoder_pattern_mask[i * output_shape[1] + j];
        }
    }

    return output_vector;  // Return the vector containing the output
}

std::vector<std::vector<std::vector<float>>> run_text_encoder(
    Ort::Session& session,
    std::vector<std::vector<int64_t>>& input_ids,
    std::vector<std::vector<int64_t>> attention_mask) {

    // Step 1: Flatten the input vectors (tokens and attention_mask)
    std::vector<int64_t> flat_tokens;
    std::vector<int64_t> flat_attention_mask;
    size_t batch_size = input_ids.size();
    size_t seq_length = input_ids[0].size();  // Assume all sequences are the same length

    for (size_t i = 0; i < batch_size; ++i) {
        flat_tokens.insert(flat_tokens.end(), input_ids[i].begin(), input_ids[i].end());
        flat_attention_mask.insert(flat_attention_mask.end(), attention_mask[i].begin(), attention_mask[i].end());
    }

    const char* input_names[] = {"input_ids", "attention_mask"};

    // Step 2: Prepare input shapes for ONNX
    std::vector<int64_t> tokens_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_length)};
    std::vector<int64_t> attention_mask_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_length)};

    // Step 3: Create input tensors
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, flat_tokens.data(), flat_tokens.size(), tokens_shape.data(), tokens_shape.size());
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, flat_attention_mask.data(), flat_attention_mask.size(), attention_mask_shape.data(), attention_mask_shape.size());

    // Step 4: Define output names (replace with actual output names from your model)
    std::vector<const char*> output_names = {"last_hidden_state"};  // Replace with the actual output name

    // Step 5: Prepare input tensors array
    std::array<Ort::Value, 2> input_tensors = {std::move(input_ids_tensor), std::move(attention_mask_tensor)};

    // Step 6: Run the model
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

    // Step 7: Get the output tensor (expected shape [batch_size, seq_length, hidden_size])
    float* encoded_output = output_tensors[0].GetTensorMutableData<float>();

    // Get the tensor shape
    std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t actual_batch_size = output_shape[0];
    size_t actual_seq_length = output_shape[1];
    size_t actual_hidden_size = output_shape[2];

    // Prepare the output vector of size [batch_size, seq_length, hidden_size]
    std::vector<std::vector<std::vector<float>>> output_vector(actual_batch_size, std::vector<std::vector<float>>(actual_seq_length, std::vector<float>(actual_hidden_size, 0.0f)));

    // Fill the 3D vector from the flat array data
    for (size_t i = 0; i < actual_batch_size; ++i) {
        for (size_t j = 0; j < actual_seq_length; ++j) {
            for (size_t k = 0; k < actual_hidden_size; ++k) {
                output_vector[i][j][k] = encoded_output[i * actual_seq_length * actual_hidden_size + j * actual_hidden_size + k];
            }
        }
    }

    return output_vector;   // Return the vector containing the output
}

int main() {
    std::string folder = "./musicgen-small";

    int cfg = 3;
    float temperature = 0.7;
    int top_k = 250;
    float top_p = 0.5;
    int max_len = 256;

    // Load the tokenizer with the SentencePiece model and special tokens file
    Tokenizer tokenizer(folder + "/spiece.model", folder + "/special_tokens_map.json");
    std::cout << "Loaded the tokenizer" << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Load the different ort sessions. We require the combined string to be sortof combined
    Ort::Session ort_session_mask(env, ConvertToOrtChar(folder + "/build_delay_pattern_mask.onnx").c_str(), session_options);
    Ort::Session ort_text_encoder(env, ConvertToOrtChar(folder + "/text_encoder.onnx").c_str(), session_options);
    Ort::Session ort_text_decoder(env, ConvertToOrtChar(folder + "/decoder_model.onnx").c_str(), session_options);
    Ort::Session ort_audio_decoder(env, ConvertToOrtChar(folder + "/encodec_decode.onnx").c_str(), session_options);

    std::cout << "Model loaded successfully!" << std::endl;

    // Tokenize the input text
    std::vector<std::string> input_texts = {"80s pop track with bassy drums and synth"}; // This can be of size n_samples
    
    std::vector<std::vector<int64_t>> tokens;
    std::vector<std::vector<int64_t>> attention_mask;
    for (size_t i = 0; i < input_texts.size(); ++i) {
        std::vector<int> outcome = tokenizer.encode(input_texts[i]);
        std::vector<int64_t> tokenized(outcome.size());
        std::transform(outcome.begin(), outcome.end(), tokenized.begin(), [](int x) {
            return static_cast<int64_t>(x);
        });
        tokens.push_back(tokenized);
        std::vector<int64_t> attentions(tokenized.size(), 1);
        attention_mask.push_back(attentions);
    }

    std::cout << "Input Text is Tokenized" << std::endl;



    std::cout << "Build delay pattern mask" << std::endl;

    std::vector<int64_t> dpm_input_ids(4 * 16, -1);  // (4, 16) filled with -1
    int64_t pad_token_id = 2048;
    int64_t max_length = 256;

    // Call the function to run the model and get the output
    std::vector<std::vector<int64_t>> decoder_pattern_mask = run_delay_pattern_mask(ort_session_mask, dpm_input_ids, pad_token_id, max_length);

    std::cout << "Delay pattern mask build" << std::endl;

    // // Print the output
    // std::cout << "Decoder Pattern Mask Output:" << std::endl;
    // std::cout << "Printing" << std::endl;
    // for (size_t i = 0; i < decoder_pattern_mask.size(); ++i) {
    //     std::cout << "[" << " ";
    //     for (size_t j = 0; j < decoder_pattern_mask[i].size(); ++j) {
    //         std::cout << decoder_pattern_mask[i][j] << " ";
    //     }
    //     std::cout << "]," << std::endl;
    // }

    std::cout << "Running text token decoder" << std::endl;

    // std::vector<int64_t> dpm_input_ids(4 * 16, -1);  // (4, 16) filled with -1
    // int64_t pad_token_id = 2048;
    // int64_t max_length = 256;

    // // Call the function to run the model and get the output
    std::vector<std::vector<std::vector<float>>> encoded = run_text_encoder(ort_text_encoder, tokens, attention_mask);
    
    std::cout << encoded[0][0][0] << std::endl;
    std::cout << "Decoded text tokens" << std::endl;



    std::cout << "Running audio decoder" << std::endl;
    
    std::cout << "Done decoding audio" << std::endl;
}
