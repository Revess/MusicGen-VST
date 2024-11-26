#ifndef MUSICGEN_UI_HPP
#define MUSICGEN_UI_HPP
#include <string>
#include <iostream>
#include <ctime>
#include <curl/curl.h>
#include <json/json.h>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <atomic>
#include <functional>
#include <condition_variable>

#include "DistrhoUI.hpp"
#include "NanoVG.hpp"
#include "Window.hpp"
#include "Color.hpp"

#include "MusicGen.hpp"

#include "./src/SimpleButton.hpp"
#include "./src/Knob.hpp"
#include "./src/Panel.hpp"
#include "./src/TextInput.hpp"
#include "./src/WAIVEColors.hpp"
#include "./src/Label.hpp"
#include "./src/ValueIndicator.hpp"
#include "./src/Checkbox.hpp"
#include "tinyfiledialogs.h"
#include "PluginParams.h"

START_NAMESPACE_DISTRHO

const unsigned int UI_W = 1500;
const unsigned int UI_H = 800;

class MusicGenUI : public UI,
                   public DGL::Button::Callback,
                   public Knob::Callback,
                   public Checkbox::Callback,
                   public TextInput::Callback
{
public:
    MusicGenUI();
    ~MusicGenUI();

protected:
    // Plugin callbacks
    void parameterChanged(uint32_t index, float value) override;
    void stateChanged(const char *key, const char *value) override;
    void onNanoDisplay() override;
    void uiScaleFactorChanged(const double scaleFactor) override;

    // Callback handler
    void buttonClicked(Button *button) override;
    void knobDragStarted(Knob *knob) override;
    void knobDragFinished(Knob *knob, float value) override;
    void knobValueChanged(Knob *knob, float value) override;

    void textEntered(TextInput *textInput, std::string text) override;
    void textInputChanged(TextInput *textInput, std::string text) override;

    void checkboxUpdated(Checkbox *checkbox, bool value) override;

    void addSampleToPanel(float padding, std::string name);

    void generateFn(std::atomic<bool>& done);
    void loaderAnim(std::atomic<bool>& done);

    void startPollingForCompletion(std::atomic<bool>* done);
    void addTimer(std::function<bool()> callback, int interval);

private:
    MusicGen *plugin;

    Checkbox *localOnlineSwitch;
    Label *localOnlineSwitchLabel;

    Panel *generatePanel;
    Panel *promptPanel;
    Panel *controlsPanel;

    TextInput *textPrompt;
    TextInput *promptTempo;
    TextInput *promptInstrumentation;

    Label *textPromptLabel;
    Label *promptTempoLabel;
    Label *promptInstrumentationLabel;

    Panel *temperaturePanel;
    Knob *temperatureKnob;
    ValueIndicator *temperatureLabel;
    Panel *nSamplesPanel;
    Knob *nSamplesKnob;
    ValueIndicator *nSamplesLabel;
    Panel *genLengthPanel;
    Knob *genLengthKnob;
    ValueIndicator *genLengthLabel;

    Checkbox *advancedSettings;
    Panel *advancedSettingsPanel;
    Label *advancedSettingsLabel;

    Panel *topKPanel;
    Knob *topKKnob;
    ValueIndicator *topKLabel;
    Panel *topPPanel;
    Knob *topPKnob;
    ValueIndicator *topPLabel;
    Panel *CFGPanel;
    Knob *CFGKnob;
    ValueIndicator *CFGLabel;

    Button *generateButton;
    Button *openFolderButton;
    Button *importButton;
    Button *clearImportedSample;
    Label *loadedFile;
    Label *loaderSpinner;

    Panel *samplesListPanel;
    Panel *samplesListInner;
    Panel *loaderPanel;

    Panel *popupPanel;
    Button *popupButton;
    Label *popupLabel;

    std::vector<Panel*> samplePanels;
    std::vector<Button*> sampleButtons;
    std::vector<Button*> samplesRemove;

    std::string userid = "undefined";

    std::string playBtn = "▶";

    double fScaleFactor;
    float fScale;
    float fscaleMult;
    int yOffset = 0;
    std::condition_variable cv;
    std::string selectedFile = "";
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MusicGenUI);
};

UI *createUI()
{
    return new MusicGenUI();
}

END_NAMESPACE_DISTRHO

#endif