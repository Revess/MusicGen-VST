# Variables
QT_VERSION = 6 # Change this to match your installed Qt version
QT_INSTALL_PATH = /opt/homebrew/opt/qt# Update this to your Qt installation path
QT_BIN = $(QT_INSTALL_PATH)/bin
CXX = $(QT_BIN)/qmake
MAKE = make
BUILD_DIR = build
TARGET = DragDropApp
SOURCES = main.cpp mainwindow.cpp draghandler.cpp
HEADERS = mainwindow.h draghandler.h
FORMS = mainwindow.ui

# Default target
all: build

# Build the project
build:
	@echo "Building project..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(CXX) .. && $(MAKE)

# Clean build files
clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)

# Run the app
run: build
	@echo "Running the application..."
	./$(BUILD_DIR)/$(TARGET)

.PHONY: all build clean run
