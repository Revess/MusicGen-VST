#ifndef DISTRHO_PLUGIN_INFO_H_INCLUDED
#define DISTRHO_PLUGIN_INFO_H_INCLUDED

#define DISTRHO_PLUGIN_BRAND "Thunderboom Records"
#define DISTRHO_PLUGIN_CLAP_ID "waive.midiinstrument"
#define DISTRHO_PLUGIN_NAME  "MusicGen"
#define DISTRHO_PLUGIN_URI   ""

#define DISTRHO_PLUGIN_NUM_INPUTS               0
#define DISTRHO_PLUGIN_NUM_OUTPUTS              2
#define DISTRHO_PLUGIN_IS_RT_SAFE               1
#define DISTRHO_PLUGIN_IS_SYNTH                 1
#define DISTRHO_PLUGIN_MINIMUM_BUFFER_SIZE      2048
#define DISTRHO_PLUGIN_WANT_LATENCY             0
#define DISTRHO_PLUGIN_WANT_MIDI_OUTPUT         0
#define DISTRHO_PLUGIN_WANT_PARAMETER_VALUE_CHANGE_REQUEST   0
#define DISTRHO_PLUGIN_WANT_DIRECT_ACCESS       1
#define DISTRHO_PLUGIN_WANT_PROGRAMS            0
#define DISTRHO_PLUGIN_WANT_STATE               0
#define DISTRHO_PLUGIN_WANT_FULL_STATE          0
#define DISTRHO_PLUGIN_WANT_TIMEPOS             0
#define DISTRHO_UI_FILE_BROWSER                 0
#define DISTRHO_PLUGIN_HAS_EXTERNAL_UI          1
#define DISTRHO_PLUGIN_HAS_EMBED_UI             1
#define DISTRHO_PLUGIN_HAS_UI                   1
#define DISTRHO_UI_USE_CUSTOM                   1
#define DISTRHO_UI_DEFAULT_WIDTH                300
#define DISTRHO_UI_DEFAULT_HEIGHT               300
#define DISTRHO_UI_USE_NANOVG                   1
#define DISTRHO_UI_USER_RESIZABLE               1
#define DISTRHO_PLUGIN_LV2_CATEGORY   "lv2:InstrumentPlugin"
#define DISTRHO_PLUGIN_VST3_CATEGORIES   "Instrument|Synth|Stereo"
#define DISTRHO_PLUGIN_CLAP_FEATURES   "instrument", "stereo"

enum Parameters {
    kGain,
    kParameterWidth = 0,
    kParameterHeight,
    kParameterCount
};

#endif