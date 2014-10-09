#ifndef QD_CONTEXT_WINDOW_H_
#define QD_CONTEXT_WINDOW_H_

#include "Object.h"
#include "Context.h"

#include <QtWidgets>

namespace Ui { class ContextWindowClass; }

class ContextWindow : public QMainWindow
{
	Q_OBJECT

public:
	ContextWindow();
	~ContextWindow();

	void resetContext();
	void loadContextFile(const std::string &filepath);
	void updateContextUI();

	Context &getContext() { return _context; }
	bool FoldAllPanelsByDefault;

private slots:
	void actionNew();
	void actionOpen();
	void actionSave();

protected:
	void dragEnterEvent(QDragEnterEvent* event) override;
	void dragMoveEvent(QDragMoveEvent* event) override;
	void dragLeaveEvent(QDragLeaveEvent* event) override;
	void dropEvent(QDropEvent* event) override;

private:
	Context _context;
	std::string _filepath;

	// User interface
	Ui::ContextWindowClass *ui;
	void addObjectUI(Object &object);
};

#endif // QD_CONTEXT_WINDOW_H_