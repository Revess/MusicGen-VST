#include "DistrhoUI.hpp"

START_NAMESPACE_DISTRHO

class MusicGenUI : public UI {
public:
    MusicGenUI()
    {
        // setGeometryConstraints(DISTRHO_UI_DEFAULT_WIDTH, DISTRHO_UI_DEFAULT_HEIGHT, true, true);
    }

protected:
    void onMusicGenDisplay()
    {
    }

    // void parameterChanged(const uint32_t index, const float value) override
    // {
    //     switch (index)
    //     {
    //     case kParameterKnob:
    //         fKnob->setValue(value);
    //         break;
    //     case kParameterTriState:
    //         fWidgetClickable->setColorId(static_cast<int>(value + 0.5f));
    //         break;
    //     case kParameterButton:
    //         fButton->setDown(value > 0.5f);
    //         break;
    //     }
    // }

};

UI* createUI()
{
    return new MusicGenUI();
}

END_NAMESPACE_DISTRHO
