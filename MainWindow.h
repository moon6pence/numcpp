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

	// operations
	void setContext(std::unique_ptr<Context> context, const std::string &filename);

	// User interface
	Ui::MainWindowClass *ui;

private slots:
	void actionNew();
	void actionOpen();
	void actionSave();

private:
	void setupEvents();

	void addObjectUI(Object &object);

	std::string _filename;
	std::unique_ptr<Context> _context;
};

#endif // MAINWINDOW_H_