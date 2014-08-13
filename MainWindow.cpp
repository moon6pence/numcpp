#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "moc_MainWindow.cpp"

#include <QuickDialog/QuickDialog.h>
#include <QuickDialog/QuickJSON.h>

using namespace std;

MainWindow::MainWindow() : ui(nullptr)
{
	ui = new Ui::MainWindowClass();

	// Build user interface
	ui->setupUi(this);

	// Connect signal/slots
	setupEvents();
}

MainWindow::~MainWindow()
{
	if (ui) { delete ui; ui = nullptr; }
}

void MainWindow::setupEvents()
{
	QObject::connect(ui->actionNew, &QAction::triggered, this, &MainWindow::actionNew);
	QObject::connect(ui->actionOpen, &QAction::triggered, this, &MainWindow::actionOpen);
	QObject::connect(ui->actionSave, &QAction::triggered, this, &MainWindow::actionSave);
	QObject::connect(ui->actionSaveAs, &QAction::triggered, this, &MainWindow::actionSaveAs);
}

void MainWindow::setContext(std::unique_ptr<Context> context, const std::string &filename)
{
	if (_context)
	{
		// Clear UI
		auto remove_all_children = [](QLayout *layout)
		{
			while (QLayoutItem *item = layout->takeAt(0))
				layout->removeItem(item);
		};

		remove_all_children(ui->panel_objectList->layout());
	}

	_context = std::move(context);

	// Add UI for objects
	for (unique_ptr<Object> &object : _context->objects())
		addObjectUI(*object);

	// Set filename and window title
	_filename = filename;

	if (_filename == "")
		setWindowTitle("Untitled - Erasmus3");
	else
		setWindowTitle(QString((_filename + " - Erasmus3").c_str()));
}

void MainWindow::addObjectUI(Object &object)
{
	const std::string label = object.getName() + " : " + object.getTypeName();

	// Add folding panel for task UI
	QWidget *panel = new QWidget();
	panel->setWindowOpacity(1.0);

	// Layout
	QVBoxLayout *layout = new QVBoxLayout(panel);
	layout->setSpacing(0);
	layout->setContentsMargins(0, 0, 0, 0);

	// Title button with monospace font
	QFont font("Monospace");
	font.setStyleHint(QFont::Monospace);

	QPushButton *button_fold = new QPushButton(QString(label.c_str()), panel);
	button_fold->setCheckable(true);
	button_fold->setMinimumHeight(25);
	button_fold->setFont(font);
	button_fold->setStyleSheet("text-align:left;");
	layout->addWidget(button_fold);

	QWidget *widget = createWidget(object, ui->panel_objectList);
	layout->addWidget(widget);

	auto clicked = [widget, button_fold, label](bool checked)
	{
		widget->setHidden(!checked);

		QString icon = checked ? "[-] " : "[+] ";
		button_fold->setText(icon + QString(label.c_str()));
	};

	QObject::connect(
		button_fold, 
		&QPushButton::clicked, 
		clicked);

	// Fold all objects except main process
	if (object.getName() == "main")
	{
		button_fold->setChecked(true);
		clicked(true);
	}
	else
	{
		button_fold->setChecked(false);
		clicked(false);
	}

	// Add panel
	ui->panel_objectList->layout()->addWidget(panel);
}

void MainWindow::actionNew()
{
	puts("Action: New");

	// Set new context
	setContext(unique_ptr<Context>(new Context), "");
}

void MainWindow::actionOpen()
{
	puts("Action: Open");

	QString filename = QFileDialog::getOpenFileName(this, "Open", "", "Context File (*.json)");
	if (filename == "") return;

	unique_ptr<Context> context(new Context);

	// Load context from json file
	load(*context, filename.toStdString()); // TODO: exception
	
	// Set context
	setContext(std::move(context), filename.toStdString());
}

void MainWindow::actionSave()
{
	puts("Action: Save");

	save(*_context, _filename);
}

void MainWindow::actionSaveAs()
{
	puts("Action: SaveAs");
}