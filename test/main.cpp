#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Create and show the main window
    MainWindow mainWindow;
    mainWindow.show();

    // Start the event loop
    return app.exec();
}
