#include "MusicGen.hpp"

START_NAMESPACE_DISTRHO

MusicGen::MusicGen() : Plugin(kParameterCount, 0, 0)
{

}

MusicGen::~MusicGen() { 

}

void MusicGen::initParameter(uint32_t index, Parameter &parameter)
{
    switch (index)
    {
        case kCharFloat:
            parameter.hints = kParameterIsAutomatable|kParameterIsBoolean;
            parameter.name = "char_float_val";
            parameter.ranges.min = -1.0f;
            parameter.ranges.max = 1000000.0f;
            parameter.ranges.def = 0.0f;
            parameter.groupId = kCharFloat;
            parameter.symbol = "char_float_val";
            break;
        default:
            break;
    }
}

void MusicGen::initAudioPort(bool input, uint32_t index, AudioPort& port)
{
    // treat meter audio ports as stereo
    port.groupId = kPortGroupStereo;

    // everything else is as default
    Plugin::initAudioPort(input, index, port);
}

float MusicGen::getParameterValue(uint32_t index) const 
{
    return 0.5;
}

void MusicGen::setParameterValue(uint32_t index, float value)
{
    if(index == 0){
        if(value > 0){
            readFilePath.push_back(static_cast<char>(value));
        } else if(value == -1) { // Start of list
            startOfFile = true;
            readFilePath = "";
            bufferPosition = 0;
            buffer.clear();
        } else if(value == -2) { // End of list
            startOfFile = false;
            SF_INFO sfInfo;
            SNDFILE* sndFile = sf_open(readFilePath.c_str(), SFM_READ, &sfInfo);
            if (!sndFile) {
                std::cerr << "Error: Could not open file: " << sf_strerror(sndFile) << std::endl;
                buffer.clear();
            } else {
                // Display audio file information
                std::cout << "Sample Rate: " << sfInfo.samplerate << std::endl;
                std::cout << "Channels: " << sfInfo.channels << std::endl;
                std::cout << "Frames: " << sfInfo.frames << std::endl;

                // Read the audio data into a buffer
                buffer.resize(sfInfo.frames * sfInfo.channels);
                sf_count_t numFrames = sf_readf_float(sndFile, buffer.data(), sfInfo.frames);

                // Close the audio file
                sf_close(sndFile);
            }

            
        } 
    } else if(index == 1){

    }
}

void MusicGen::setState(const char* key, const char* value)
{

}

String MusicGen::getState(const char* key) const
{
    String retString = String("undefined state");
    return retString;
}

void MusicGen::initState(unsigned int index, String &stateKey, String &defaultStateValue)
{

}

void MusicGen::run(
    const float **,              // incoming audio
    float **outputs,             // outgoing audio
    uint32_t numFrames,          // size of block to process
    const MidiEvent *midiEvents, // MIDI pointer
    uint32_t midiEventCount      // Number of MIDI events in block
)
{
    float* outLeft = outputs[0];
    float* outRight = outputs[1];

    // Process each frame
    for (uint32_t i = 0; i < numFrames; ++i) {
        if (bufferPosition < buffer.size() / 2) {
            // Copy samples from the buffer to the outputs
            outLeft[i] = buffer[2 * bufferPosition];     // Left channel
            outRight[i] = buffer[2 * bufferPosition + 1]; // Right channel
            bufferPosition++;
        } else {
            // If we've reached the end of the buffer, output silence
            outLeft[i] = 0.0f;
            outRight[i] = 0.0f;
        }
    }

}

void MusicGen::readFile(std::string selectedFile)
{
    
}



Plugin *createPlugin()
{
    return new MusicGen();
}

END_NAMESPACE_DISTRHO