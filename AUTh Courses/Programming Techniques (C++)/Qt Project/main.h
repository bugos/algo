#include <QWidget>

class testWidget : public QWidget
{
 //   Q_OBJECT
public:
    testWidget(QWidget *parent = 0);
protected:
    virtual void mouseMoveEvent(QMouseEvent* event);
};