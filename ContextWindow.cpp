#include "ContextWindow.h"
#include "ui_ContextWindow.h"
#include "moc_ContextWindow.cpp"

#include <QuickDialog/QuickDialog.h>
#include <QuickDialog/QuickJSON.h>

#include <QFileInfo>
#include <filesystem>

using namespace std;

ContextWindow::ContextWindow() : ui(nullptr)
{
	ui = new Ui::ContextWindowClass();

	// Build user interface
	ui->setupUi(this);

	// Enable drag & drop file
	this->setAcceptDrops(true);

	// Connect signal/slots
	QObject::connect(ui->actionNew, &QAction::triggered, this, &ContextWindow::actionNew);
	QObject::connect(ui->actionOpen, &QAction::triggered, this, &ContextWindow::actionOpen);
	QObject::connect(ui->actionSave, &QAction::triggered, this, &ContextWindow::actionSave);
}

ContextWindow::~ContextWindow()
{
	if (ui) { delete ui; ui = nullptr; }
}

void ContextWindow::loadContextFile(const std::string &filepath)
{
	// Clear current context UI
	QLayoutItem *item;
	while ((item = ui->panel_objectList->layout()->takeAt(0)) != nullptr)
	{
		delete item->widget();
		delete item;
	}

	// Clear context
	_context.clear();

	// Set filepath and window title
	_filepath = filepath;
	if (_filepath == "")
	{
		// Just clean context if filepath == ""
		setWindowTitle("Untitled - Erasmus3");
	}
	else
	{
		setWindowTitle(QFileInfo(_filepath.c_str()).fileName() + " - Erasmus3");

		// Load context from json file
		if (!readJson(_context, filepath))
			return;

		// Add UI for objects
		for (unique_ptr<Object> &object : _context.objects())
			addObjectUI(*object);
	}

	// Change current directory to json file dir
	using namespace std::tr2::sys;
	current_path(path(filepath).parent_path());
}

void ContextWindow::addObjectUI(Object &object)
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

void ContextWindow::actionNew()
{
	puts("Action: New");

	loadContextFile("");
}

void ContextWindow::actionOpen()
{
	puts("Action: Open");

	QString filename = QFileDialog::getOpenFileName(this, "Open", "", "Context File (*.json)");
	if (filename == "") return;

	loadContextFile(filename.toStdString());
}

void ContextWindow::actionSave()
{
	puts("Action: Save");

	writeJson(_context, _filepath);
}

void ContextWindow::dragEnterEvent(QDragEnterEvent* event)
{
	event->acceptProposedAction();
}

void ContextWindow::dragMoveEvent(QDragMoveEvent* event)
{
	event->acceptProposedAction();
}

void ContextWindow::dragLeaveEvent(QDragLeaveEvent* event)
{
	event->accept();
}

void ContextWindow::dropEvent(QDropEvent* event)
{
	const QMimeData* mimeData = event->mimeData();

	if (mimeData->hasUrls())
	{
		// Open first file
		loadContextFile(mimeData->urls().at(0).toLocalFile().toStdString());
	}
}
