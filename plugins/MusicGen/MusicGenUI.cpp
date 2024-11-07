#include "MusicGenUI.hpp"

START_NAMESPACE_DISTRHO

MusicGenUI::MusicGenUI() : UI(UI_W, UI_H),
                           fScaleFactor(getScaleFactor())
{
    plugin = static_cast<MusicGen *>(getPluginInstancePointer());

    float width = UI_W * fScaleFactor;
    float height = UI_H * fScaleFactor;

    float padding = 4.f * fScaleFactor;

    float fontsize = 8.f * fScaleFactor;

    // Main input panel
    int promptPanelHeight = 0;
    int controltPanelHeight = 0;
    {
        {
            generatePanel = new Panel(this);
            generatePanel->setSize((width * 0.5f * 0.5f) - (padding * 2.0), (height * 0.5f) - (padding * 2.0), true);
            generatePanel->setAbsolutePos(padding, padding);
            generatePanel->background_color = WaiveColors::grey2;
        }

        {
            promptPanel = new Panel(this);
            promptPanel->setSize(generatePanel->getWidth(), (generatePanel->getHeight() * 0.9) - (padding * 2.0), true);
            promptPanel->onTop(generatePanel, CENTER, START, padding);
            promptPanel->background_color = WaiveColors::grey2;
        }

        // Now add input box for prompt
        float inpBoxW = promptPanel->getWidth() - (padding * 2);
        {
            textPromptLabel = new Label(this, "Prompt");
            textPromptLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            textPromptLabel->setFontSize(fontsize);
            textPromptLabel->text_color = WaiveColors::text;
            textPromptLabel->resizeToFit();
            textPromptLabel->onTop(promptPanel, START, START, padding);

            textPrompt = new TextInput(this);
            textPrompt->setSize(inpBoxW, 40, true);
            textPrompt->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            textPrompt->setFontSize(fontsize);
            textPrompt->align = Align::ALIGN_CENTER;
            textPrompt->foreground_color = WaiveColors::light1;
            textPrompt->background_color = WaiveColors::light1;
            textPrompt->below(textPromptLabel, START, padding);
            textPrompt->setCallback(this);
        }

        // Now add input box for tempo information
        {
            promptTempoLabel = new Label(this, "Tempo");
            promptTempoLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            promptTempoLabel->setFontSize(fontsize);
            promptTempoLabel->text_color = WaiveColors::text;
            promptTempoLabel->resizeToFit();
            promptTempoLabel->below(textPrompt, START, padding);

            promptTempo = new TextInput(this);
            promptTempo->setSize((inpBoxW * 0.5f) - (padding * 0.5f), 40, true);
            promptTempo->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            promptTempo->setFontSize(fontsize);
            promptTempo->align = Align::ALIGN_CENTER;
            promptTempo->foreground_color = WaiveColors::light1;
            promptTempo->background_color = WaiveColors::light1;
            promptTempo->below(promptTempoLabel, START, padding);
            promptTempo->setCallback(this);
        }

        // Now add input box for instrumentation
        {
            promptInstrumentation = new TextInput(this);
            promptInstrumentation->setSize((inpBoxW * 0.5f) - (padding * 0.5f), 40, true);
            promptInstrumentation->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            promptInstrumentation->setFontSize(fontsize);
            promptInstrumentation->align = Align::ALIGN_CENTER;
            promptInstrumentation->foreground_color = WaiveColors::light1;
            promptInstrumentation->background_color = WaiveColors::light1;
            promptInstrumentation->rightOf(promptTempo, START, padding);
            promptInstrumentation->setCallback(this);

            promptInstrumentationLabel = new Label(this, "Instrumentation");
            promptInstrumentationLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            promptInstrumentationLabel->setFontSize(fontsize);
            promptInstrumentationLabel->text_color = WaiveColors::text;
            promptInstrumentationLabel->resizeToFit();
            promptInstrumentationLabel->above(promptInstrumentation, START, padding);
            // promptInstrumentationLabel->rightOf(promptTempoLabel, START, padding);
        }

        promptPanelHeight += promptInstrumentation->getHeight();
        promptPanelHeight += promptInstrumentationLabel->getHeight();
        promptPanelHeight += promptTempo->getHeight();
        promptPanelHeight += promptTempoLabel->getHeight();
        promptPanelHeight += textPrompt->getHeight();
        promptPanelHeight += textPromptLabel->getHeight();

        promptPanel->setSize(generatePanel->getWidth() - (padding * 2.0), promptPanelHeight  - (padding * 2.0), true);

        // Control panel
        {
            controlsPanel = new Panel(this);
            controlsPanel->setSize(promptPanel->getWidth(), (generatePanel->getHeight() * 0.6f) - padding, true);
            controlsPanel->below(promptPanel, CENTER, padding);
            controlsPanel->background_color = WaiveColors::grey2;
        }

        // Now add knobs
        int knobW = 200;
        int knobH = 200;
        // Length
        {
            genLengthPanel = new Panel(this);
            genLengthPanel->setSize(knobW, knobH, true);
            genLengthPanel->onTop(controlsPanel, CENTER, START, padding);
            genLengthPanel->background_color = WaiveColors::grey2;

            genLengthKnob = new Knob(this);
            genLengthKnob->setName("Length");
            genLengthKnob->max = 30.0;
            genLengthKnob->min = 0.4;
            genLengthKnob->label = "Length";
            genLengthKnob->setFontSize(fontsize);
            genLengthKnob->setRadius(20.f);
            genLengthKnob->gauge_width = 3.0f * fScaleFactor;
            genLengthKnob->setValue(8);
            genLengthKnob->resizeToFit();
            genLengthKnob->onTop(genLengthPanel, CENTER, CENTER, padding);
            genLengthKnob->setCallback(this);

            genLengthLabel = new ValueIndicator(this);
            genLengthLabel->setSize(70, 20);
            genLengthLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            genLengthLabel->background_color = WaiveColors::grey2;
            genLengthLabel->above(genLengthKnob, CENTER, padding);
            genLengthLabel->setValue(genLengthKnob->getValue());
        }
        // Temperature
        {
            temperaturePanel = new Panel(this);
            temperaturePanel->setSize(knobW, knobH, true);
            temperaturePanel->leftOf(genLengthPanel, START, padding);
            temperaturePanel->background_color = WaiveColors::grey2;
        
            temperatureKnob = new Knob(this);
            temperatureKnob->setName("Temperature");
            temperatureKnob->max = 2.0;
            temperatureKnob->min = 0.0;
            temperatureKnob->label = "Temperature";
            temperatureKnob->setFontSize(fontsize);
            temperatureKnob->setRadius(20.f);
            temperatureKnob->gauge_width = 3.0f * fScaleFactor;
            temperatureKnob->setValue(0.7);
            temperatureKnob->resizeToFit();
            temperatureKnob->onTop(temperaturePanel, CENTER, CENTER, padding);
            temperatureKnob->setCallback(this);

            temperatureLabel = new ValueIndicator(this);
            temperatureLabel->setSize(70, 20);
            temperatureLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            temperatureLabel->background_color = WaiveColors::grey2;
            temperatureLabel->above(temperatureKnob, CENTER, padding);
            temperatureLabel->setValue(temperatureKnob->getValue());
        }
        // nSamples
        {
            nSamplesPanel = new Panel(this);
            nSamplesPanel->setSize(knobW, knobH, true);
            nSamplesPanel->rightOf(genLengthPanel, START, padding);
            nSamplesPanel->background_color = WaiveColors::grey2;

            nSamplesKnob = new Knob(this);
            nSamplesKnob->setName("Samples");
            nSamplesKnob->max = 12;
            nSamplesKnob->min = 1;
            nSamplesKnob->label = "Samples";
            nSamplesKnob->setFontSize(fontsize);
            nSamplesKnob->setRadius(20.f);
            nSamplesKnob->integer = true;
            nSamplesKnob->gauge_width = 3.0f * fScaleFactor;
            nSamplesKnob->setValue(1);
            nSamplesKnob->resizeToFit();
            nSamplesKnob->onTop(nSamplesPanel, CENTER, CENTER, padding);
            nSamplesKnob->setCallback(this);

            nSamplesLabel = new ValueIndicator(this);
            nSamplesLabel->setSize(70, 20);
            nSamplesLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            nSamplesLabel->background_color = WaiveColors::grey2;
            nSamplesLabel->above(nSamplesKnob, CENTER, padding);
            nSamplesLabel->setValue(nSamplesKnob->getValue());
        }

        // Advanced Options
        {
            advancedSettingsPanel = new Panel(this);
            advancedSettingsPanel->setSize(promptPanel->getWidth() - (padding * 2), 100, true);
            advancedSettingsPanel->below(genLengthPanel, CENTER, padding);
            advancedSettingsPanel->background_color = WaiveColors::grey2;
            
            advancedSettings = new Checkbox(this);
            advancedSettings->setSize(25, 25, true);
            advancedSettings->background_color = WaiveColors::text;
            advancedSettings->foreground_color = WaiveColors::text;
            advancedSettings->setChecked(false, false);
            advancedSettings->onTop(advancedSettingsPanel, START, CENTER, padding);
            advancedSettings->setCallback(this);

            advancedSettingsLabel = new Label(this, "Advanced Settings");
            advancedSettingsLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            advancedSettingsLabel->setFontSize(fontsize);
            advancedSettingsLabel->text_color = WaiveColors::text;
            advancedSettingsLabel->resizeToFit();
            advancedSettingsLabel->rightOf(advancedSettings, CENTER, padding);
            
            
            // generate Button
            {
                generateButton = new Button(this);
                generateButton->setLabel("Generate");
                generateButton->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
                generateButton->setFontSize(fontsize);
                generateButton->resizeToFit();
                generateButton->background_color = WaiveColors::light1;
                generateButton->onTop(advancedSettingsPanel, END, CENTER, padding);
                generateButton->setCallback(this);
            }
        }

        controltPanelHeight += advancedSettingsPanel->getHeight();
        controltPanelHeight += genLengthPanel->getHeight() * 6;

        // Now add knobs of advanced
        // Top P
        {
            topPPanel = new Panel(this);
            topPPanel->setSize(knobW, knobH, true);
            topPPanel->below(advancedSettingsPanel, CENTER, padding);
            topPPanel->background_color = WaiveColors::grey2;

            topPKnob = new Knob(this);
            topPKnob->setName("Top P");
            topPKnob->max = 1.0;
            topPKnob->min = 0.0;
            topPKnob->label = "Top P";
            topPKnob->setFontSize(fontsize);
            topPKnob->setRadius(20.f);
            topPKnob->gauge_width = 3.0f * fScaleFactor;
            topPKnob->setValue(0.0);
            topPKnob->resizeToFit();
            topPKnob->onTop(topPPanel, CENTER, CENTER, padding);
            topPKnob->setCallback(this);

            topPLabel = new ValueIndicator(this);
            topPLabel->setSize(70, 20);
            topPLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            topPLabel->background_color = WaiveColors::grey2;
            topPLabel->above(topPKnob, CENTER, padding);
            topPLabel->setValue(topPKnob->getValue());

            topPPanel->setVisible(false);
            topPKnob->setVisible(false);
            topPLabel->setVisible(false);
            topPPanel->hide();
            topPKnob->hide();
            topPLabel->hide();
        }
        // Top K
        {
            topKPanel = new Panel(this);
            topKPanel->setSize(knobW, knobH, true);
            topKPanel->leftOf(topPPanel, START, padding);
            topKPanel->background_color = WaiveColors::grey2;
        
            topKKnob = new Knob(this);
            topKKnob->setName("Top K");
            topKKnob->max = 1000;
            topKKnob->min = 1;
            topKKnob->integer = true;
            topKKnob->label = "Top K";
            topKKnob->setFontSize(fontsize);
            topKKnob->setRadius(20.f);
            topKKnob->gauge_width = 3.0f * fScaleFactor;
            topKKnob->setValue(500);
            topKKnob->resizeToFit();
            topKKnob->onTop(topKPanel, CENTER, CENTER, padding);
            topKKnob->setCallback(this);

            topKLabel = new ValueIndicator(this);
            topKLabel->setSize(70, 20);
            topKLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            topKLabel->background_color = WaiveColors::grey2;
            topKLabel->above(topKKnob, CENTER, padding);
            topKLabel->setValue(topKKnob->getValue());

            topKPanel->setVisible(false);
            topKKnob->setVisible(false);
            topKLabel->setVisible(false);
            topKPanel->hide();
            topKKnob->hide();
            topKLabel->hide();
        }
        // CFG
        {
            CFGPanel = new Panel(this);
            CFGPanel->setSize(knobW, knobH, true);
            CFGPanel->rightOf(topPPanel, START, padding);
            CFGPanel->background_color = WaiveColors::grey2;

            CFGKnob = new Knob(this);
            CFGKnob->setName("CFG");
            CFGKnob->max = 10;
            CFGKnob->min = 1;
            CFGKnob->label = "CFG";
            CFGKnob->setFontSize(fontsize);
            CFGKnob->setRadius(20.f);
            CFGKnob->integer = true;
            CFGKnob->gauge_width = 3.0f * fScaleFactor;
            CFGKnob->setValue(5);
            CFGKnob->resizeToFit();
            CFGKnob->onTop(CFGPanel, CENTER, CENTER, padding);
            CFGKnob->setCallback(this);

            CFGLabel = new ValueIndicator(this);
            CFGLabel->setSize(70, 20);
            CFGLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            CFGLabel->background_color = WaiveColors::grey2;
            CFGLabel->above(CFGKnob, CENTER, padding);
            CFGLabel->setValue(CFGKnob->getValue());

            CFGPanel->setVisible(false);
            CFGKnob->setVisible(false);
            CFGLabel->setVisible(false);
            CFGPanel->hide();
            CFGKnob->hide();
            CFGLabel->hide();
        }
    
        controlsPanel->setSize(controlsPanel->getWidth(), controltPanelHeight  - (padding * 2.0), true);
        advancedSettings->setChecked(true, true);
        advancedSettings->setChecked(false, true);
    }
    
    {
        {
            samplesList = new Panel(this);
            samplesList->setSize((width * 0.5f * 0.5f) - padding, (height * 0.5f) - (padding * 2.0), true);
            samplesList->setAbsolutePos(generatePanel->getWidth() + (padding * 2.0), padding);
            samplesList->background_color = WaiveColors::grey2;
        }
    }
    
    repaint();
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
    fillColor(WaiveColors::grey1);
    rect(0.0f, 0.0f, width, height);
    fill();
    closePath();
}

void MusicGenUI::uiScaleFactorChanged(const double scaleFactor)
{

}

void MusicGenUI::buttonClicked(Button *button)
{
    // Start making the request
    if(button == generateButton){
        float length = genLengthKnob->getValue();
        float temperature = temperatureKnob->getValue();
        int samples = nSamplesKnob->getValue();
        // std::string textPrompt = textPrompt.getText;
        // textPrompt.append(" ");
        // textPrompt.append(promptTempo.text);
        // textPrompt.append(" ");
        // textPrompt.append(promptInstrumentation.text);

        // Make the request etc.
    }
}

void MusicGenUI::knobDragStarted(Knob *knob)
{
    if (knob == temperatureKnob){
        temperatureLabel->setValue(knob->getValue());
    } else if (knob == genLengthKnob){
        genLengthLabel->setValue(knob->getValue());
    } else if (knob == nSamplesKnob){
        nSamplesLabel->setValue(knob->getValue());
    } else if (knob == topKKnob){
        topKLabel->setValue(knob->getValue());
    } else if (knob == topPKnob){
        topPLabel->setValue(knob->getValue());
    } else if (knob == CFGKnob){
        CFGLabel->setValue(knob->getValue());
    }
}

void MusicGenUI::knobDragFinished(Knob *knob, float value)
{
    // plugin->triggerPreview();
}

void MusicGenUI::knobValueChanged(Knob *knob, float value)
{
    if (knob == temperatureKnob){
        temperatureLabel->setValue(knob->getValue());
    } else if (knob == genLengthKnob){
        genLengthLabel->setValue(knob->getValue());
    } else if (knob == nSamplesKnob){
        nSamplesLabel->setValue(knob->getValue());
    } else if (knob == topKKnob){
        topKLabel->setValue(knob->getValue());
    } else if (knob == topPKnob){
        topPLabel->setValue(knob->getValue());
    } else if (knob == CFGKnob){
        CFGLabel->setValue(knob->getValue());
    }
}

void MusicGenUI::textEntered(TextInput *textInput, std::string text){

}

void MusicGenUI::textInputChanged(TextInput *textInput, std::string text)
{
    
}

void MusicGenUI::checkboxUpdated(Checkbox *checkbox, bool value)
{
    float padding = 4.f * fScaleFactor;
    float width = UI_W * fScaleFactor;
    float height = UI_H * fScaleFactor;

    if(checkbox == advancedSettings){
        if(value == true){
            promptPanel->setSize(generatePanel->getWidth() - (padding * 2.0), (generatePanel->getHeight() * 0.4) - (padding * 2.0), true);
            controlsPanel->setSize(promptPanel->getWidth(), (generatePanel->getHeight() * 0.6f) - padding, true);

            topKPanel->setVisible(true);
            topKKnob->setVisible(true);
            topKLabel->setVisible(true);
            topKPanel->show();
            topKKnob->show();
            topKLabel->show();

            topPPanel->show();
            topPKnob->show();
            topPLabel->show();

            CFGPanel->setVisible(true);
            CFGKnob->setVisible(true);
            CFGLabel->setVisible(true);
            CFGPanel->show();
            CFGKnob->show();
            CFGLabel->show();
        } else {
            promptPanel->setSize(generatePanel->getWidth() - (padding * 2.0), (generatePanel->getHeight() * 0.4) - (padding * 2.0), true);
            controlsPanel->setSize(promptPanel->getWidth(), (generatePanel->getHeight() * 0.6f) - padding, true);

            topKPanel->setVisible(false);
            topKKnob->setVisible(false);
            topKLabel->setVisible(false);
            topKPanel->hide();
            topKKnob->hide();
            topKLabel->hide();

            topPPanel->setVisible(false);
            topPKnob->setVisible(false);
            topPLabel->setVisible(false);
            topPPanel->hide();
            topPKnob->hide();
            topPLabel->hide();

            CFGPanel->setVisible(false);
            CFGKnob->setVisible(false);
            CFGLabel->setVisible(false);
            CFGPanel->hide();
            CFGKnob->hide();
            CFGLabel->hide();
        }
    }
    repaint();
}

END_NAMESPACE_DISTRHO