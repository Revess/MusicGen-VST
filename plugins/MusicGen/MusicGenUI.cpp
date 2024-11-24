#include "MusicGenUI.hpp"


// TODO :
// Test localhost toggle with python code.
// Add a loading screen.

// Some small pop ups for the user like:
    // Error no local server found
    // Error no remote server found

std::string getCurrentDateTime() {
    // Get the current time
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);

    // Format the date and time
    std::ostringstream oss;
    oss << (now->tm_year % 100)  // Year (last two digits)
        << (now->tm_mon + 1 < 10 ? "0" : "") << (now->tm_mon + 1)  // Month
        << (now->tm_mday < 10 ? "0" : "") << now->tm_mday          // Day
        << (now->tm_hour < 10 ? "0" : "") << now->tm_hour          // Hour
        << (now->tm_min < 10 ? "0" : "") << now->tm_min            // Minute
        << (now->tm_sec < 10 ? "0" : "") << now->tm_sec;           // Second

    return oss.str();
}

START_NAMESPACE_DISTRHO

MusicGenUI::MusicGenUI() : UI(UI_W, UI_H),
                           fScaleFactor(2.0f),
                           fScale(1.0f)
{
    // getScaleFactor();
    const char* homeDir = std::getenv("HOME"); // Works on Unix-like systems
    std::filesystem::path documentsPath = std::filesystem::path(homeDir) / "Documents" / "MusicGenVST";
    std::filesystem::create_directory(documentsPath);
    documentsPath = std::filesystem::path(homeDir) / "Documents" / "MusicGenVST" / "generated";
    std::filesystem::create_directory(documentsPath);

    plugin = static_cast<MusicGen *>(getPluginInstancePointer());

    float width = UI_W * fScaleFactor;
    float height = UI_H * fScaleFactor;

    float padding = 4.f * fScaleFactor;

    if(getScaleFactor() == 2){
        fscaleMult = 1.0;
    } else{
        fscaleMult = 2.0;
    }

    float fontsize = 8.f * fScaleFactor * fscaleMult;

    // Main input panel
    int promptPanelHeight = 0;
    int controltPanelHeight = 0;
    {
        {
            generatePanel = new Panel(this);
            generatePanel->setSize((width * 0.5f * 0.5f) - (padding * 2.0), (height * 0.5f) - (padding * 2.0), true);
            generatePanel->setAbsolutePos(padding, padding);
        }

        {
            promptPanel = new Panel(this);
            promptPanel->setSize(generatePanel->getWidth(), (generatePanel->getHeight() * 0.9) - (padding * 2.0), true);
            promptPanel->onTop(generatePanel, CENTER, START, padding);
        }

        {
            localOnlineSwitch = new Checkbox(this);
            localOnlineSwitch->setSize(25, 25, true);
            localOnlineSwitch->background_color = WaiveColors::text;
            localOnlineSwitch->foreground_color = WaiveColors::text;
            localOnlineSwitch->setChecked(false, false);
            localOnlineSwitch->onTop(promptPanel, END, START, padding);
            localOnlineSwitch->setCallback(this);

            localOnlineSwitchLabel = new Label(this, "Use local server");
            localOnlineSwitchLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            localOnlineSwitchLabel->setFontSize(fontsize);
            localOnlineSwitchLabel->text_color = WaiveColors::text;
            localOnlineSwitchLabel->resizeToFit();
            localOnlineSwitchLabel->leftOf(localOnlineSwitch, CENTER, padding);
            localOnlineSwitch->hide();
            localOnlineSwitchLabel->hide();
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

        {
            importButton = new Button(this);
            importButton->setLabel("Import Sample");
            importButton->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            importButton->setFontSize(fontsize);
            importButton->resizeToFit();
            importButton->below(promptTempo, CENTER, padding);
            importButton->setCallback(this);
        }

        {
            clearImportedSample = new Button(this);
            clearImportedSample->setLabel("Clear Sample");
            clearImportedSample->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            clearImportedSample->setFontSize(fontsize);
            clearImportedSample->resizeToFit();
            clearImportedSample->below(promptInstrumentation, CENTER, padding);
            clearImportedSample->setCallback(this);

            {
                loadedFile = new Label(this, "This is the test loc of the load filepath");
                loadedFile->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
                loadedFile->setFontSize(fontsize * 0.9);
                loadedFile->text_color = WaiveColors::text;
                loadedFile->resizeToFit();
                loadedFile->below(importButton, START, padding);
            }
        }

        promptPanelHeight += promptInstrumentation->getHeight();
        promptPanelHeight += promptInstrumentationLabel->getHeight();
        promptPanelHeight += promptTempo->getHeight();
        promptPanelHeight += promptTempoLabel->getHeight();
        promptPanelHeight += textPrompt->getHeight();
        promptPanelHeight += textPromptLabel->getHeight();
        promptPanelHeight += clearImportedSample->getHeight();
        promptPanelHeight += loadedFile->getHeight();
        loadedFile->hide();

        promptPanel->setSize(generatePanel->getWidth() - (padding * 2.0), promptPanelHeight  + (padding * 1.0), true);

        // Control panel
        {
            controlsPanel = new Panel(this);
            controlsPanel->setSize(promptPanel->getWidth(), (generatePanel->getHeight() * 0.6f) - padding, true);
            controlsPanel->below(promptPanel, CENTER, padding);
        }

        // Now add knobs
        int knobW = 200;
        int knobH = 125;
        // Length
        {
            genLengthPanel = new Panel(this);
            genLengthPanel->setSize(knobW, knobH, true);
            genLengthPanel->onTop(controlsPanel, CENTER, START, padding);

            genLengthKnob = new Knob(this);
            genLengthKnob->setName("Length");
            genLengthKnob->max = 30.0;
            genLengthKnob->min = 0.4;
            genLengthKnob->label = "Length";
            genLengthKnob->setFontSize(fontsize);
            genLengthKnob->setRadius(10.f * fScaleFactor);
            genLengthKnob->gauge_width = 2.0f * fScaleFactor;
            genLengthKnob->setValue(8);
            genLengthKnob->resizeToFit();
            genLengthKnob->onTop(genLengthPanel, CENTER, CENTER, padding);
            genLengthKnob->setCallback(this);

            genLengthLabel = new ValueIndicator(this);
            genLengthLabel->setSize(70, 20);
            genLengthLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            genLengthLabel->setFontSize(fontsize);
            genLengthLabel->above(genLengthKnob, CENTER, padding);
            genLengthLabel->setValue(genLengthKnob->getValue());
        }
        // Temperature
        {
            temperaturePanel = new Panel(this);
            temperaturePanel->setSize(knobW, knobH, true);
            temperaturePanel->leftOf(genLengthPanel, START, padding);
        
            temperatureKnob = new Knob(this);
            temperatureKnob->setName("Temperature");
            temperatureKnob->max = 2.0;
            temperatureKnob->min = 0.0;
            temperatureKnob->label = "Temperature";
            temperatureKnob->setFontSize(fontsize);
            temperatureKnob->setRadius(10.f * fScaleFactor);
            temperatureKnob->gauge_width = 3.0f * fScaleFactor;
            temperatureKnob->setValue(0.7);
            temperatureKnob->resizeToFit();
            temperatureKnob->onTop(temperaturePanel, CENTER, CENTER, padding);
            temperatureKnob->setCallback(this);

            temperatureLabel = new ValueIndicator(this);
            temperatureLabel->setSize(70, 20);
            temperatureLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            temperatureLabel->setFontSize(fontsize);
            temperatureLabel->above(temperatureKnob, CENTER, padding);
            temperatureLabel->setValue(temperatureKnob->getValue());
        }
        // nSamples
        {
            nSamplesPanel = new Panel(this);
            nSamplesPanel->setSize(knobW, knobH, true);
            nSamplesPanel->rightOf(genLengthPanel, START, padding);

            nSamplesKnob = new Knob(this);
            nSamplesKnob->setName("Samples");
            nSamplesKnob->max = 10;
            nSamplesKnob->min = 1;
            nSamplesKnob->label = "Samples";
            nSamplesKnob->setFontSize(fontsize);
            nSamplesKnob->setRadius(10.f * fScaleFactor);
            nSamplesKnob->integer = true;
            nSamplesKnob->gauge_width = 3.0f * fScaleFactor;
            nSamplesKnob->setValue(1);
            nSamplesKnob->resizeToFit();
            nSamplesKnob->onTop(nSamplesPanel, CENTER, CENTER, padding);
            nSamplesKnob->setCallback(this);

            nSamplesLabel = new ValueIndicator(this);
            nSamplesLabel->setSize(70, 20);
            nSamplesLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            nSamplesLabel->setFontSize(fontsize);
            nSamplesLabel->above(nSamplesKnob, CENTER, padding);
            nSamplesLabel->setValue(nSamplesKnob->getValue());
        }

        // Advanced Options
        {
            advancedSettingsPanel = new Panel(this);
            advancedSettingsPanel->setSize(promptPanel->getWidth() - (padding), 100, true);
            advancedSettingsPanel->below(genLengthPanel, CENTER, padding);
            
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
                generateButton->onTop(advancedSettingsPanel, END, CENTER, padding);
                generateButton->setCallback(this);
            }

            // Sample prompt Button
        }

        controltPanelHeight += advancedSettingsPanel->getHeight();
        controltPanelHeight += genLengthPanel->getHeight() * 6;

        // Now add knobs of advanced
        // Top P
        {
            topPPanel = new Panel(this);
            topPPanel->setSize(knobW, knobH, true);
            topPPanel->below(advancedSettingsPanel, CENTER, padding*4);

            topPKnob = new Knob(this);
            topPKnob->setName("Top P");
            topPKnob->max = 1.0;
            topPKnob->min = 0.0;
            topPKnob->label = "Top P";
            topPKnob->setFontSize(fontsize);
            topPKnob->setRadius(10.f * fScaleFactor);
            topPKnob->gauge_width = 3.0f * fScaleFactor;
            topPKnob->setValue(0.0);
            topPKnob->resizeToFit();
            topPKnob->onTop(topPPanel, CENTER, CENTER, padding);
            topPKnob->setCallback(this);

            topPLabel = new ValueIndicator(this);
            topPLabel->setSize(70, 20);
            topPLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            topPLabel->setFontSize(fontsize);
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
        
            topKKnob = new Knob(this);
            topKKnob->setName("Top K");
            topKKnob->max = 1000;
            topKKnob->min = 1;
            topKKnob->integer = true;
            topKKnob->label = "Top K";
            topKKnob->setFontSize(fontsize);
            topKKnob->setRadius(10.f * fScaleFactor);
            topKKnob->gauge_width = 3.0f * fScaleFactor;
            topKKnob->setValue(500);
            topKKnob->resizeToFit();
            topKKnob->onTop(topKPanel, CENTER, CENTER, padding);
            topKKnob->setCallback(this);

            topKLabel = new ValueIndicator(this);
            topKLabel->setSize(70, 20);
            topKLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            topKLabel->setFontSize(fontsize);
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

            CFGKnob = new Knob(this);
            CFGKnob->setName("CFG");
            CFGKnob->max = 10;
            CFGKnob->min = 1;
            CFGKnob->label = "CFG";
            CFGKnob->setFontSize(fontsize);
            CFGKnob->setRadius(10.f * fScaleFactor);
            CFGKnob->integer = true;
            CFGKnob->gauge_width = 3.0f * fScaleFactor;
            CFGKnob->setValue(5);
            CFGKnob->resizeToFit();
            CFGKnob->onTop(CFGPanel, CENTER, CENTER, padding);
            CFGKnob->setCallback(this);

            CFGLabel = new ValueIndicator(this);
            CFGLabel->setSize(70, 20);
            CFGLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            CFGLabel->setFontSize(fontsize);
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
            samplesListPanel = new Panel(this);
            samplesListPanel->setSize((width * 0.5f * 0.5f) - padding, (height * 0.5f) - (padding * 2.0), true);
            samplesListPanel->setAbsolutePos(generatePanel->getWidth() + (padding * 2.0), padding);

            openFolderButton = new Button(this);
            openFolderButton->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
            openFolderButton->setLabel("Open Generate Folder");
            openFolderButton->setFontSize(fontsize);
            openFolderButton->resizeToFit();
            openFolderButton->onTop(samplesListPanel, CENTER, END, padding);
            openFolderButton->setCallback(this);

            samplesListInner = new Panel(this);
            samplesListInner->setSize((width * 0.5f * 0.5f) - (padding * 2), ((27 * fScaleFactor)*10 + padding*11), true);
            samplesListInner->onTop(samplesListPanel, CENTER, START, padding);
            samplesListInner->background_color = WaiveColors::dark;

            // scrollBar = new Panel(this);
            // scrollBar->setSize((padding * 2), samplesListInner->getHeight() - (padding * 2), true);
            // scrollBar->onTop(samplesListInner, END, CENTER, 0);
            // scrollBar->background_color = WaiveColors::light1;
            // scrollBarHeight = scrollBar->getHeight();

            // Loop over it change the positioning to take into account the scroll top etc.
            for(int i = 0; i < 10; i++) {
                addSampleToPanel(padding, std::string("test"));
                // if(samplePanels.size() > 12){
                //     scrollBar->setSize((padding * 2), round(static_cast<float>(scrollBarHeight) * ((12.0) / static_cast<float>(samplePanels.size()))), true);
                // }
            }

            // sampleButtons
            // samplesRemove
        }
    }

    {
        loaderPanel = new Panel(this);
        loaderPanel->setSize((width * 0.5f) - (padding * 2.0), (height * 0.5f) - (padding * 2.0), true);
        loaderPanel->setAbsolutePos(padding, padding);
        loaderPanel->toFront();

        static const Color bg_col(35, 35, 37, 0.5);
        loaderPanel->background_color = bg_col;
        loaderPanel->hide();

        loaderSpinner = new Label(this, " ");
        loaderSpinner->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
        loaderSpinner->setFontSize(fontsize * 4.0f);
        loaderSpinner->text_color = WaiveColors::text;
        loaderSpinner->resizeToFit();
        loaderSpinner->onTop(loaderPanel, CENTER, CENTER, padding);
        loaderSpinner->hide();
    }

    {
        popupPanel = new Panel(this);
        popupPanel->setSize((width * 0.25f) - (padding * 2.0), (height * 0.25f) - (padding * 2.0), true);
        popupPanel->onTop(loaderPanel, CENTER, CENTER, padding);
        popupPanel->toFront();

        static const Color bg_col(35, 35, 37, 0.5);
        popupPanel->background_color = bg_col;
        popupPanel->hide();

        popupLabel = new Label(this, "Oke");
        popupLabel->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
        popupLabel->setFontSize(fontsize);
        popupLabel->text_color = WaiveColors::text;
        popupLabel->resizeToFit();
        popupLabel->onTop(popupPanel, CENTER, CENTER, padding);
        popupLabel->hide();

        popupButton = new Button(this);
        popupButton->setFont("Poppins-Light", Poppins_Light, Poppins_Light_len);
        popupButton->setLabel("Ok");
        popupButton->setFontSize(fontsize);
        popupButton->resizeToFit();
        popupButton->below(popupLabel, CENTER, padding);
        popupButton->setCallback(this);
        popupButton->hide();
    }
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
    fillColor(WaiveColors::dark);
    rect(0.0f, 0.0f, width, height);
    fill();
    closePath();
}

void MusicGenUI::uiScaleFactorChanged(const double scaleFactor)
{

}

// Callback function to capture the response from the server
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

static size_t WriteCallbackStream (void* contents, size_t size, size_t nmemb, void* userp)
{
    std::ofstream* out = static_cast<std::ofstream*>(userp);
    size_t totalSize = size * nmemb;
    out->write(static_cast<char*>(contents), totalSize);
    return totalSize;
}

// Function to extract the basename from a URL
std::string getBasename(const std::string& url)
{
    size_t lastSlash = url.find_last_of("/");
    if (lastSlash == std::string::npos)
    {
        return url; // Return the full URL if no slash is found
    }
    return url.substr(lastSlash + 1);
}

void MusicGenUI::generateFn(std::atomic<bool>& done)
{
    // TODO: add way for audio prompt
    // Make if else statment here to update the IP to localhost if in offline mode.
    std::string ip = "";
    if(!localOnlineSwitch->getChecked()){
        ip = "http://82.217.111.120/";
    } else {
        ip = "http://127.0.0.1:55000/";
    }

    float duration = genLengthKnob->getValue();
    float temperature = temperatureKnob->getValue();
    float topp = topPKnob->getValue();
    int samples = nSamplesKnob->getValue();
    int topk = topKKnob->getValue();
    int CFG = CFGKnob->getValue();
    std::string prompt = textPrompt->getText();
    prompt.append(" ");
    prompt.append(promptTempo->getText());
    prompt.append(" ");
    prompt.append(promptInstrumentation->getText());

    // Make a request
    CURL *curl;
    CURLcode res;
    curl = curl_easy_init();

    // Parse and print the response using a JSON library
    Json::Value jsonData;
    Json::CharReaderBuilder readerBuilder;

    // Do a generate request
    if(curl) {
        std::string readBuffer;

        // Prepare curl form data
        struct curl_httppost* form = NULL;
        struct curl_httppost* last = NULL;

        // Add the audio file
        if(selectedFile.size() > 0){
            curl_formadd(&form, &last,
                    CURLFORM_COPYNAME, "audioInput",
                    CURLFORM_FILE, selectedFile.c_str(),
                    CURLFORM_CONTENTTYPE, "audio/wav",
                    CURLFORM_END);
        }
        
        // Add other form data
        curl_formadd(&form, &last,
                    CURLFORM_COPYNAME, "prompt",
                    CURLFORM_COPYCONTENTS, prompt.c_str(),
                    CURLFORM_END);

        curl_formadd(&form, &last,
                    CURLFORM_COPYNAME, "userid",
                    CURLFORM_COPYCONTENTS, userid.c_str(),
                    CURLFORM_END);

        curl_formadd(&form, &last,
                    CURLFORM_COPYNAME, "Temperature",
                    CURLFORM_COPYCONTENTS, std::to_string(temperature).c_str(),
                    CURLFORM_END);

        curl_formadd(&form, &last,
                    CURLFORM_COPYNAME, "Top K",
                    CURLFORM_COPYCONTENTS, std::to_string(topk).c_str(),
                    CURLFORM_END);

        curl_formadd(&form, &last,
                    CURLFORM_COPYNAME, "Top P",
                    CURLFORM_COPYCONTENTS, std::to_string(topp).c_str(),
                    CURLFORM_END);

        curl_formadd(&form, &last,
                    CURLFORM_COPYNAME, "Samples",
                    CURLFORM_COPYCONTENTS, std::to_string(samples).c_str(),
                    CURLFORM_END);

        curl_formadd(&form, &last,
                    CURLFORM_COPYNAME, "Classifier Free Guidance",
                    CURLFORM_COPYCONTENTS, std::to_string(CFG).c_str(),
                    CURLFORM_END);

        curl_formadd(&form, &last,
                    CURLFORM_COPYNAME, "Duration",
                    CURLFORM_COPYCONTENTS, std::to_string(duration).c_str(),
                    CURLFORM_END);

        // Set URL and form data
        curl_easy_setopt(curl, CURLOPT_URL, ip.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPPOST, form);

        // Set the write callback function
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        // Perform the request and store the result
        res = curl_easy_perform(curl);
        // Check for errors
        if(res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            if(ip == "http://127.0.0.1:55000/"){
                popupLabel->setLabel("Local server, try starting local server.");
            } else {
                popupLabel->setLabel("Could not reach server, try with local server.");
            }
            popupLabel->resizeToFit();
            float padding = 4.f * fScaleFactor;

            popupLabel->onTop(popupPanel, CENTER, CENTER, padding);

            popupPanel->show();
            popupButton->show();
            popupLabel->show();
        } else {
            std::string errs;
            std::istringstream ss(readBuffer);
            std::cout << "Raw Response: " << readBuffer << std::endl;

            if (Json::parseFromStream(readerBuilder, ss, &jsonData, &errs)) {
                // Access the values from the JSON response
                bool success = jsonData["success"].asBool();
                std::string userId = jsonData["userid"].asString();
                const Json::Value downloadLinks = jsonData["download_links"];

                userid = userId;
                std::cout << "Status: " << (success ? "ok" : "error") << std::endl;
                std::cout << "User ID: " << userId << std::endl;
                std::cout << "Download Links:" << std::endl;

                for (const auto &link : downloadLinks) {
                    std::cout << " - " << link.asString() << std::endl;
                }
            } else {
                std::cerr << "Failed to parse JSON: " << errs << std::endl;
            }
        }

        // Cleanup
        curl_easy_cleanup(curl);
        curl_formfree(form);
    }

    const char* homeDir = std::getenv("HOME"); // Works on Unix-like systems

    // Download the files into the generated folder   
    if(curl) {
        for(std::size_t i = 0; i < samplePanels.size(); i++){
            samplePanels[i]->hide();
            sampleButtons[i]->hide();
        }
        std::string readBuffer;
        const Json::Value downloadLinks = jsonData["download_links"];
        int i = 0;
        for (const auto &link : downloadLinks) {
            const std::string url = std::string(ip) + link.asString();
            std::string datetime = getCurrentDateTime() + std::string("_") + std::to_string(i) + std::string(".wav");
            std::filesystem::path outputFilename = std::filesystem::path(homeDir) / "Documents" / "MusicGenVST" / "generated" / datetime;

            curl = curl_easy_init();

            std::ofstream outFile(outputFilename, std::ios::binary);
            if (!outFile)
            {
                std::cerr << "Failed to open file for writing" << std::endl;
            }

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallbackStream);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outFile);


            res = curl_easy_perform(curl);
            if (res != CURLE_OK)
            {
                std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            }

            curl_easy_cleanup(curl);

            outFile.close();

            samplePanels[i]->show();
            sampleButtons[i]->setLabel(datetime);
            sampleButtons[i]->show();

            i++;
        }
    }
    done = true;
}

void MusicGenUI::startPollingForCompletion(std::atomic<bool>* done) {
    // Use a timer or periodic task in your framework (DPF doesn’t have a built-in timer)
    int i = 0;
    const std::string spinner = "|/-\\";
    auto pollCompletion = [this, done, i, spinner]() mutable {
        if (done->load()) {
            loaderPanel->hide();
            loaderSpinner->hide();
            repaint(); // Update the UI to hide the loader

            delete done; // Free the memory allocated for `done`
            return true; // Stop polling
        }

        loaderSpinner->setLabel(std::string(1, spinner[i % spinner.size()]));
        loaderSpinner->resizeToFit();
        std::cout << "\rLoading " << spinner[i % 4] << std::flush;
        i++;

        repaint(); // Keep the UI responsive while polling
        return false; // Continue polling
    };

    // Replace this with your framework's timer or loop mechanism
    addTimer(pollCompletion, 50); // Call `pollCompletion` every 50ms
}

void MusicGenUI::addTimer(std::function<bool()> callback, int interval) {
    std::thread([callback, interval]() {
        while (true) {
            if (callback()) break; // Stop the timer if the callback returns true
            std::this_thread::sleep_for(std::chrono::milliseconds(interval));
        }
    }).detach();
}

void MusicGenUI::buttonClicked(Button *button)
{
    
    if(!loaderPanel->isVisible()){
        // Start making the request
        if(button == generateButton){
            loaderPanel->show();
            loaderSpinner->show();
            repaint();
            std::atomic<bool>* done = new std::atomic<bool>(false);

            // Start worker threads
            std::thread(&MusicGenUI::generateFn, this, std::ref(*done)).detach();

            // Set up a timer to poll for the `done` flag and update the UI
            startPollingForCompletion(done);
        } else if(button == importButton) {
            const char* filters[] = { "*.wav" };
            const char* filePath = tinyfd_openFileDialog(
                "Select a File",    // Title of the dialog
                "",                 // Default path
                1,                  // Number of filters
                filters,            // File filters
                "WAV Files",        // Filter description
                0                   // Do not allow multiple selections
            );

            if (filePath) {
                std::cout << "Selected file: " << filePath << std::endl;
                selectedFile = static_cast<std::string>(filePath);
                loadedFile->setLabel(selectedFile);
                loadedFile->resizeToFit();
                loadedFile->show();
            } else {
                std::cout << "No file selected." << std::endl;
                selectedFile = "";
            }

        } else if(button == clearImportedSample) {
            selectedFile = "";
            loadedFile->hide();
        } else if(button == openFolderButton) {
            #ifdef _WIN32
                // Windows
                std::system("explorer .");
            #elif __APPLE__
                // macOS
                std::system("open ~/Documents/MusicGenVST/generated");
                std::cerr << "open ~/Documents/MusicGenVST/generated" << std::endl;
            #elif __linux__
                // Linux
                std::system("xdg-open ~/Documents/MusicGenVST/generated");
            #else
                std::cerr << "Unsupported operating system." << std::endl;
            #endif
        } else if(button == popupButton){
            popupPanel->hide();
            popupButton->hide();
            popupLabel->hide();
        } else{
            for(std::size_t i = 0; i < sampleButtons.size(); i++){
                if (button == sampleButtons[i]){
                    const char* homeDir = std::getenv("HOME"); // Works on Unix-like systems
                    std::filesystem::path outputFilename = std::filesystem::path(homeDir) / "Documents" / "MusicGenVST" / "generated" / sampleButtons[i]->getLabel();
                    std::string selectedFile = static_cast<std::string>(outputFilename);
                    plugin->setParameterValue(0, -1.0f);
                    for(std::size_t i = 0; i < selectedFile.size(); i++){
                        plugin->setParameterValue(0, static_cast<float>(selectedFile[i]));
                        // std::cout << selectedFile[i] << std::endl;
                    }
                    plugin->setParameterValue(0, -2.0f);
                    break;
                }
            }   
        }
    }
}

void MusicGenUI::knobDragStarted(Knob *knob)
{
    if(!loaderPanel->isVisible()){
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
}

void MusicGenUI::knobDragFinished(Knob *knob, float value)
{
    // plugin->triggerPreview();
}

void MusicGenUI::knobValueChanged(Knob *knob, float value)
{
    if(!loaderPanel->isVisible()){
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

            localOnlineSwitch->show();
            localOnlineSwitchLabel->show();
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

            localOnlineSwitch->hide();
            localOnlineSwitchLabel->hide();
        }
    } else if(checkbox == localOnlineSwitch){
        if(value == true){

        } else {

        }
    }
    repaint();
}

bool MusicGenUI::onScroll(const ScrollEvent &ev)
{
    float mouseX = ev.absolutePos.getX();
    float mouseY = ev.absolutePos.getY();
    float padding = 4.f * fScaleFactor;
    float yOffsetPrev = yOffset;
    if(
        mouseX >= samplesListInner->getAbsoluteX() && 
        mouseX <= (samplesListInner->getAbsoluteX() + samplesListInner->getWidth()) &&
        mouseY >= samplesListInner->getAbsoluteY() && 
        mouseY <= (samplesListInner->getAbsoluteY() + samplesListInner->getHeight()) &&
        1 == 0
    )
    {
        if(samplePanels[0]->getAbsoluteY() == samplesListInner->getAbsoluteY() - padding){

        }
        double yScroll = -ev.delta.getY();
        // std::cout << ev.delta.getY() << std::endl;
        int scrollbarOffset = -round(static_cast<float>(samplesListInner->getHeight() - (padding * 2) - scrollBar->getHeight()) / static_cast<float>((samplePanels.size() - 12)));
        if(yScroll < 0 && samplePanels[0]->getAbsoluteY() != samplesListInner->getAbsoluteY() + padding){ // Down when can actually go up
            yOffset += samplePanels[0]->getHeight() + padding;
        } else if(samplesListInner->getAbsoluteY() + samplesListInner->getHeight() - padding - samplePanels[0]->getHeight() <= samplePanels.back()->getAbsoluteY()) { // Up when actually can go
            yOffset -= samplePanels[0]->getHeight() + padding;
            scrollbarOffset = -scrollbarOffset;
        }

        if(yOffsetPrev != yOffset){ // Actually do a scroll
            this->start = std::chrono::high_resolution_clock::now(); // Reset the time
            scrollBar->setAbsolutePos(scrollBar->getAbsoluteX(), scrollBar->getAbsoluteY() + scrollbarOffset);
            for(std::size_t i = 0; i < samplePanels.size(); i++){
                if(i == 0){
                    samplePanels[i]->setAbsolutePos(samplesListInner->getAbsoluteX() + padding, samplesListInner->getAbsoluteY() + padding + yOffset);
                } else {
                    samplePanels[i]->below(samplePanels[i - 1], CENTER, padding);
                }
                sampleButtons[i]->onTop(samplePanels[i], START, START, 0);
                samplesRemove[i]->onTop(samplePanels[i], END, END, 0);
                if(samplePanels[i]->getAbsoluteY() <= samplesListInner->getAbsoluteY() - padding || 
                samplePanels[i]->getAbsoluteY() + samplePanels[i]->getHeight() >= samplesListInner->getAbsoluteY() + samplesListInner->getHeight()){
                    samplePanels[i]->hide();
                    sampleButtons[i]->hide();
                    samplesRemove[i]->hide();
                } else {
                    samplePanels[i]->show();
                    sampleButtons[i]->show();
                    samplesRemove[i]->show();
                }
            }
            repaint();
        }
    }
}

void MusicGenUI::addSampleToPanel(float padding, std::string name)
{
    int h = 27 * fscaleMult;
    samplePanels.push_back(new Panel(this));
    samplePanels.back()->setSize((samplesListInner->getWidth() * 0.5f * fscaleMult) - (padding), h);
    samplePanels.back()->background_color = WaiveColors::grey2;
    if(samplePanels.size() == 1){
        samplePanels.back()->onTop(samplesListInner, START, START, padding);
        samplePanels.back()->setAbsolutePos(samplesListInner->getAbsoluteX() + padding, samplesListInner->getAbsoluteY() + padding + yOffset);
    } else {
        samplePanels.back()->below(samplePanels[samplePanels.size()-2], CENTER, padding);
    }
    sampleButtons.push_back(new Button(this));
    sampleButtons.back()->setSize((samplePanels.back()->getWidth() * 0.5 * fscaleMult), h);
    sampleButtons.back()->setLabel(name);
    sampleButtons.back()->textAlign(Align::ALIGN_LEFT); // Fix in the header of the button to change the text alignment
    sampleButtons.back()->background_color = WaiveColors::grey2;
    sampleButtons.back()->onTop(samplePanels.back(), START, START, 0);
    sampleButtons.back()->setCallback(this);

    samplePanels.back()->hide();
    sampleButtons.back()->hide();

    // samplesRemove.push_back(new Button(this));
    // samplesRemove.back()->setSize((samplePanels.back()->getWidth() * 0.5) * 0.25, 25);
    // samplesRemove.back()->setLabel("⌫");
    // samplesRemove.back()->background_color = WaiveColors::grey2;
    // samplesRemove.back()->onTop(samplePanels.back(), END, END, 0);
}

END_NAMESPACE_DISTRHO