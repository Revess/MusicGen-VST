#ifndef DRAGHANDLER_H
#define DRAGHANDLER_H

#include <QObject>
#include <QString>
#include <QDrag>
#include <QMimeData>
#include <QWidget>

class DragHandler : public QObject
{
    Q_OBJECT

public:
    explicit DragHandler(QObject *parent = nullptr);

    // Function to trigger a drag operation
    void startDrag(const QString &filePath, QWidget *dragSource);

private:
    bool isValidFile(const QString &filePath);
};

#endif // DRAGHANDLER_H
