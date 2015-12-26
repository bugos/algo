#include <QApplication>
#include <QLabel>
#include <QCursor>
#include <QDebug>
#include <QtWidgets>
#include "main.h"


testWidget::testWidget(QWidget *parent) : QWidget(parent)
{
    setMinimumSize(800,600);
    show();
}

void testWidget::mouseMoveEvent(QMouseEvent *event)
{
    QPoint before(mapFromGlobal(QCursor::pos()));
    QPoint center = mapToGlobal(QPoint(width()/2,height()/2));
    QCursor::setPos(center);
    qDebug()<<"Before:"<<before<<"After:"<<mapFromGlobal(QCursor::pos());
}

int main(int argc, char *argv[])
{
	//Q_INIT_RESOURCE(testWidget);
    QApplication app(argc, argv);

    testWidget *tw = new testWidget;
    tw->show();
    //QLabel *label = new QLabel("Hello Qt!");
    //label->show();
    return app.exec();
}
