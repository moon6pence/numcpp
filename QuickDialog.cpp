#include "QuickDialog.h"
#include <QtWidgets>

#include <iostream>

struct QuickDialog : public property_visitor
{
	QWidget *widget;
    QFormLayout *layout;

	QuickDialog(QWidget *parent = 0)
	{
		widget = new QWidget(parent);
        layout = new QFormLayout(widget);
	}

	void visit(property<bool> &property) const
	{
		QCheckBox *checkBox = new QCheckBox(" ", widget);
		checkBox->setChecked(property);

		// Update property when checkbox is clicked
		QObject::connect(
			checkBox, 
			&QCheckBox::clicked,  
			[&property](bool checked)
			{
				property.set(checked);
			});

		// Update checkbox if property is changed
		property.valueChanged += [checkBox](bool value)
		{
			checkBox->setChecked(value);
		};

		addFormWidget(property.name(), checkBox);
	}

	class QuickSpinBox : public QSpinBox
	{
	public:
		QuickSpinBox(QWidget *parent) : QSpinBox(parent) { }

		void wheelEvent(QWheelEvent *event) 
		{
			if (!hasFocus()) 
			{
				event->ignore();
			} 
			else 
			{
				QSpinBox::wheelEvent(event);
			}
		}
	};

	void visit(property<int> &property) const
	{
		QuickSpinBox *spinBox = new QuickSpinBox(widget);
		spinBox->setRange(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
		spinBox->setValue(property.get());
		spinBox->setFocusPolicy(Qt::StrongFocus);

		// Update property when spinbox is changed
		QObject::connect(
			spinBox, 
			static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), 
			[&property](int value)
			{
				property.set(value);
			});

		// Update spinbox if property is changed
		property.valueChanged += [spinBox](int value)
		{
			spinBox->metaObject()->invokeMethod(spinBox, "setValue", Qt::QueuedConnection, Q_ARG(int, value));
		};

		addFormWidget(property.name(), spinBox);
	}

	class QDoubleSpinBoxIgnoreWheel : public QDoubleSpinBox
	{
	public:
		QDoubleSpinBoxIgnoreWheel(QWidget *parent = 0) : QDoubleSpinBox(parent) { }

		void wheelEvent(QWheelEvent *event) override
		{
			if (!hasFocus()) 
			{
				event->ignore();
			} 
			else 
			{
				QDoubleSpinBox::wheelEvent(event);
			}
		}

		// To prevent too long spin box
		QSize sizeHint () const override
		{
			QSize size = QDoubleSpinBox::sizeHint();
			return QSize(0, size.height());
		}

		QSize minimumSizeHint () const override
		{
			QSize size = QDoubleSpinBox::minimumSizeHint();
			return QSize(0, size.height());
		}
	};

	void visit(property<float> &property) const
	{
		QDoubleSpinBox *spinBox = new QDoubleSpinBoxIgnoreWheel(widget);
		spinBox->setRange(property.min(), property.max());
		spinBox->setSingleStep(property.step());
		spinBox->setValue(property.get());
		spinBox->setFocusPolicy(Qt::StrongFocus);

		// Update property when spinbox is changed
		QObject::connect(
			spinBox, 
			static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), 
			[&property](double value)
			{
				property.set(value);
			});

		// Update spinbox if property is changed
		property.valueChanged += [spinBox](float value)
		{
			spinBox->setValue(value);
		};

		addFormWidget(property.name(), spinBox);
	}

	void visit(property<std::string> &property) const
	{
		QLineEdit *edit = new QLineEdit(widget);
		edit->setText(QString(property.get().c_str()));

		// Update property when textbox is changed
		QObject::connect(
			edit, 
			&QLineEdit::editingFinished, 
			[&property, edit]()
			{
				property.set(edit->text().toStdString());
			});

		// Update textbox when property is changed
		property.valueChanged += [edit](std::string value)
		{
			edit->setText(QString(value.c_str()));
		};

		addFormWidget(property.name(), edit);
	}

	void visit(property<Object> &property) const
	{
		QComboBox *comboBox = new QComboBox(widget);

		Context &context = property.getContext();
		for (auto &object : context.objects())
		{
			comboBox->addItem(QString(object->getName().c_str()));

			// Select current value
			if (property.get().compare(object->getName()) == 0)
				comboBox->setCurrentIndex(comboBox->count() - 1);
		}

		// Update property when textbox is changed
		QObject::connect(
			comboBox, 
	        static_cast<void (QComboBox::*)(const QString &)>(&QComboBox::currentIndexChanged),
			[&property](const QString &text)
			{
				property.set(text.toStdString());
			});

		// Update textbox when property is changed
		property.valueChanged += [comboBox](std::string value)
		{
			for (int i = 0; i < comboBox->count(); i++)
				if (comboBox->itemText(i).toStdString().compare(value) == 0)
					comboBox->setCurrentIndex(i);
		};

		addFormWidget(property.name(), comboBox);
	}
	
	void visit(operation &operation) const
	{
		int rowCount = layout->rowCount();

		QPushButton *button = new QPushButton(widget);
		button->setText(QString(operation.name().c_str()));
		button->setMinimumHeight(30);
		layout->setWidget(rowCount, QFormLayout::SpanningRole, button);

		// signal: clicked()
		QObject::connect(
			button, 
			&QPushButton::clicked, 
			[&operation]()
			{
				operation.run();
			});
	}

	void addLine() const
	{
		int rowCount = layout->rowCount();

        QFrame *line = new QFrame(widget);
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);
		layout->setWidget(rowCount, QFormLayout::SpanningRole, line);
	}

private:
	void addFormWidget(const std::string &label_text, QWidget *widget) const
	{
		int rowCount = layout->rowCount();

		QLabel *label = new QLabel(widget);
		label->setText(QString(label_text.c_str()));
		layout->setWidget(rowCount, QFormLayout::LabelRole, label);

		layout->setWidget(rowCount, QFormLayout::FieldRole, widget);
	}
};


QWidget *createWidget(Object &object, QWidget *parent)
{
	QuickDialog dialog(parent);
	object.accept(dialog);

	return dialog.widget;
}