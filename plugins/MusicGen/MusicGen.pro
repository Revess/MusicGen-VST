# Name of the target
TEMPLATE = app
TARGET = DragDropApp

# List of source files
SOURCES += MusicGenUI.cpp \
            ./src/SimpleButton.cpp \
            ./src/Knob.cpp \
            ./src/TextInput.cpp \
            ./src/Panel.cpp \
            ./src/Label.cpp \
            ./src/Checkbox.cpp \
            ./src/ValueIndicator.cpp \
            ./src/WidgetGroup.cpp \
            ./src/Layout.cpp \
            ./src/WAIVEWidget.cpp \
            draghandler.cpp \
            tinyfiledialogs.c

# List of header files
HEADERS += MusicGenUI.hpp \
            ./src/SimpleButton.hpp \
            ./src/Knob.hpp \
            ./src/TextInput.hpp \
            ./src/Panel.hpp \
            ./src/Label.hpp \
            ./src/Checkbox.hpp \
            ./src/ValueIndicator.hpp \
            ./src/WidgetGroup.hpp \
            ./src/Layout.hpp \
            ./src/WAIVEWidget.hpp \
            draghandler.h \
            tinyfiledialogs.h

# Qt modules
QT += core gui widgets
