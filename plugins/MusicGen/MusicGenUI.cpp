#include "MusicGenUI.hpp"

START_NAMESPACE_DISTRHO

MusicGenUI::MusicGenUI() : UI(UI_W, UI_H)
{
    plugin = static_cast<MusicGen *>(getPluginInstancePointer());

    float width = getWidth();
    float height = getHeight();

    generateButton = new Button(this);
    generateButton->setLabel("Generate");
    generateButton->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
    generateButton->resizeToFit();
    generateButton->setCallback(this);

    // generateButton.setAbsolutePos((width / 2) - (150 / 2), (height / 2) - (75 / 2));
    // generateButton.setLabel("Generate");
    // generateButton.setSize(150, 75);
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
    float width = getWidth();
    float height = getHeight();

    beginPath();
    fillColor(Color(255, 255, 255));
    rect(0.0f, 0.0f, width, height);
    fill();
    closePath();
}

void MusicGenUI::uiScaleFactorChanged(const double scaleFactor)
{

}

void MusicGenUI::buttonClicked(Button *button)
{

}

END_NAMESPACE_DISTRHO