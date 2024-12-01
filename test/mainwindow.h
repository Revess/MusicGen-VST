#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QString>
#include "draghandler.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    QPushButton *loadButton;
    QPushButton *dragButton;
    QString selectedFilePath;
    DragHandler *dragHandler;

    void setupUI();

protected:
    void mousePressEvent(QMouseEvent *event) override;
};

#endif // MAINWINDOW_H
