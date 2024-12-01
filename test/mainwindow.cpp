#include "mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), selectedFilePath(""), dragHandler(new DragHandler(this))
{
    setupUI();
}

MainWindow::~MainWindow()
{
}

void MainWindow::setupUI()
{
    // Configure the main window
    setFixedSize(400, 200);
    setWindowTitle("Drag and Drop File");

    // Load button
    loadButton = new QPushButton("Load File", this);
    loadButton->setGeometry(50, 80, 100, 50);
    connect(loadButton, &QPushButton::clicked, this, [this]() {
        QString filePath = QFileDialog::getOpenFileName(this, "Select a file to drag");
        if (!filePath.isEmpty()) {
            selectedFilePath = filePath;
            dragButton->setEnabled(true); // Enable drag button if a file is selected
            QMessageBox::information(this, "File Loaded", "File selected:\n" + filePath);
        } else {
            QMessageBox::warning(this, "Warning", "No file selected!");
        }
    });

    // Drag button
    dragButton = new QPushButton("Drag File", this);
    dragButton->setGeometry(250, 80, 100, 50);
    dragButton->setEnabled(false); // Disabled until a file is loaded

    connect(dragButton, &QPushButton::pressed, this, [this]() {
        dragHandler->startDrag(selectedFilePath, dragButton);
    });
}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
    QMainWindow::mousePressEvent(event);
}
