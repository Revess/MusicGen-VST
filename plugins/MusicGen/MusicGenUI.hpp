#ifndef MUSICGEN_UI_HPP
#define MUSICGEN_UI_HPP

#include "DistrhoUI.hpp"
#include "NanoVG.hpp"
#include "Window.hpp"

#include "MusicGen.hpp"

#include <random>

START_NAMESPACE_DISTRHO

const unsigned int UI_W = 1000;
const unsigned int UI_H = 582;

class MusicGenUI;

class MusicGenUI : public UI {
public:
    MusicGenUI();
    ~MusicGenUI();

protected:
    // Plugin callbacks
    void parameterChanged(uint32_t index, float value) override;
    void stateChanged(const char *key, const char *value) override;
    void onNanoDisplay() override;
    void uiScaleFactorChanged(const double scaleFactor) override;

private:
    MusicGen *plugin;

    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MusicGenUI);
};

UI *createUI()
{
    return new MusicGenUI();
}

END_NAMESPACE_DISTRHO

#endif