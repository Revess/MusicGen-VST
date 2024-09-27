#include <iostream>
#include <string>
#include <cstring>      // For memset
#include <sys/socket.h> // For socket functions
#include <arpa/inet.h>  // For inet_addr
#include <unistd.h>     // For close()

int main() {
    // Create a socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation error!" << std::endl;
        return -1;
    }

    // Server address setup
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(65432);  // Same port as Python server

    // Convert IPv4 address from text to binary
    if (inet_pton(AF_INET, "127.0.0.1", &server_address.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported!" << std::endl;
        return -1;
    }

    // Connect to the server
    if (connect(sock, (struct sockaddr*)&server_address, sizeof(server_address)) < 0) {
        std::cerr << "Connection failed!" << std::endl;
        return -1;
    }

    // Send input data to the server in the format: 'prompt|temperature|topk|topp|cfg|samples|duration'
    std::string prompt = "test_prompt";
    float temperature = 0.7;
    int topk = 250;
    float topp = 0.95;
    int cfg = 2;
    int samples = 3;
    int duration = 15;

    // Create the formatted input string for the Python server
    std::string input_data = prompt + "|" + 
                             std::to_string(temperature) + "|" + 
                             std::to_string(topk) + "|" + 
                             std::to_string(topp) + "|" + 
                             std::to_string(cfg) + "|" + 
                             std::to_string(samples) + "|" + 
                             std::to_string(duration);

    // Send the data to the server
    send(sock, input_data.c_str(), input_data.size(), 0);
    std::cout << "Request sent to server." << std::endl;

    // Buffer to receive the response from the server
    char buffer[4096] = {0};

    // Read the response from the server
    int bytes_received = read(sock, buffer, sizeof(buffer) - 1);
    if (bytes_received > 0) {
        std::cout << "Response from server: " << buffer << std::endl;
    } else {
        std::cerr << "Failed to receive response from server." << std::endl;
    }

    // Close the socket connection
    close(sock);

    return 0;
}
