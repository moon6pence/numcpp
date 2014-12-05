#ifndef QUICK_DIALOG_H_
#define QUICK_DIALOG_H_

#include "object.h"
#include "context.h"

// create Qt widget
class QWidget;

namespace nbb {

QWidget *createWidget(Object &object, QWidget *parent = 0);

} // namespace nbb

#endif // QUICK_DIALOG_H_