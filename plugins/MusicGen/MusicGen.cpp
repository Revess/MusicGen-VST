#include "MusicGen.hpp"

START_NAMESPACE_DISTRHO

MusicGen::MusicGen() : Plugin(0,0,0) 
{
}

MusicGen::~MusicGen() { 

}

void MusicGen::initParameter(uint32_t index, Parameter &parameter)
{

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

}



Plugin *createPlugin()
{
    return new MusicGen();
}

END_NAMESPACE_DISTRHO