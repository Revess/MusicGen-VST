#ifndef MUSICGEN_UI_HPP
#define MUSICGEN_UI_HPP
#include <string>

#include "DistrhoUI.hpp"
#include "NanoVG.hpp"
#include "Window.hpp"
#include "Color.hpp"
#include "SourceList.hpp"

#include "MusicGen.hpp"

#include "./src/SimpleButton.hpp"
#include "./src/Knob.hpp"
#include "./src/Panel.hpp"
#include "./src/TextInput.hpp"
#include "./src/WAIVEColors.hpp"
#include "./src/Label.hpp"
#include "./src/ValueIndicator.hpp"
#include "./src/Checkbox.hpp"

START_NAMESPACE_DISTRHO

const unsigned int UI_W = 1500;
const unsigned int UI_H = 760;

class MusicGenUI : public UI,
                   DGL::Button::Callback,
                   Knob::Callback,
                   Checkbox::Callback,
                   TextInput::Callback
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

private:
    MusicGen *plugin;

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

    DGL::Button *generateButton;

    Panel *samplesList;

    double fScaleFactor;

    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MusicGenUI);
};

UI *createUI()
{
    return new MusicGenUI();
}

END_NAMESPACE_DISTRHO

#endif