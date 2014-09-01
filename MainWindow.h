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

	void setFileName(const std::string &filepath);

	void resetContext();
	void loadContextFile(const std::string &filepath);
	Context &getContext() { return _context; }

private slots:
	void actionNew();
	void actionOpen();
	void actionSave();

private:
	Context _context;
	std::string _filepath;

	// User interface
	Ui::MainWindowClass *ui;
	void addObjectUI(Object &object);
};

#endif // MAINWINDOW_H_