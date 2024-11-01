#include "MusicGenUI.hpp"

START_NAMESPACE_DISTRHO

MusicGenUI::MusicGenUI() : UI(UI_W, UI_H)
{
    plugin = static_cast<MusicGen *>(getPluginInstancePointer());
    random.seed();

}

MusicGenUI::~MusicGenUI()
{

}

void MusicGenUI::parameterChanged(uint32_t index, float value)
{

}

void MusicGenUI::stateChanged(const char *key, const char *value)
{

}

void MusicGenUI::onNanoDisplay()
{

}

void MusicGenUI::uiScaleFactorChanged(const double scaleFactor)
{

}


END_NAMESPACE_DISTRHO