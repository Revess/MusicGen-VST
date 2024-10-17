#include "./include/tokenizer.hpp"
#include <iostream>

int main() {
    // Load the tokenizer with the SentencePiece model and special tokens file
    Tokenizer tokenizer("./musicgen-small/spiece.model", "./musicgen-small/special_tokens_map.json");

    std::string input_text = "Hello, this is a test.";

    // Encode the text into token IDs
    std::vector<int> tokens = tokenizer.encode(input_text);
    std::cout << "Encoded tokens: ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    // Decode the token IDs back into text
    std::string decoded_text = tokenizer.decode(tokens);
    std::cout << "Decoded text: " << decoded_text << std::endl;

    return 0;
}