#ifndef MAINWINDOW_H_
#define MAINWINDOW_H_

#include "Object.h"
#include "Context.h"

#include <QtWidgets>

namespace Ui { class MainWindowClass; }

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow();
	~MainWindow();

	void resetContext();
	void loadContextFile(const std::string &filename);
	Context &getContext() { return _context; }

private slots:
	void actionNew();
	void actionOpen();
	void actionSave();

private:
	Context _context;
	std::string _filename;

	// User interface
	Ui::MainWindowClass *ui;
	void addObjectUI(Object &object);
};

#endif // MAINWINDOW_H_