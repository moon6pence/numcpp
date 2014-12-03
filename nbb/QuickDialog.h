#ifndef QUICK_DIALOG_H_
#define QUICK_DIALOG_H_

#include "Object.h"
#include "Context.h"

// create Qt widget
class QWidget;

QWidget *createWidget(Object &object, QWidget *parent = 0);

#endif // QUICK_DIALOG_H_