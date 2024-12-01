# Name of the target
TEMPLATE = app
TARGET = DragDropApp

# List of source files
SOURCES += main.cpp \
           mainwindow.cpp \
           draghandler.cpp

# List of header files
HEADERS += mainwindow.h draghandler.h

# Qt modules
QT += core gui widgets
