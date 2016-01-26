#ifndef QIMAGE_VIEW_H_
#define QIMAGE_VIEW_H_

#include <QtWidgets>

#include <numcpp/array.h>
#include <functional>

class QImageView : public QWidget
{
public:
	QImageView(QWidget *parent = 0) : QWidget(parent), 
		_standardColorTable(256),
		_invertedColorTable(256)
    {
        // prepare color table
		for (int i = 0; i < 256; i++)
			_standardColorTable[i] = 0xFF000000 + i * 0x010101;

		for (int i = 0; i < 256; i++)
			_invertedColorTable[i] = 0xFF000000 + (255 - i) * 0x010101; // Inverted color

        _qimage = QImage(256, 256, QImage::Format_Indexed8);
        _qimage.setColorTable(_standardColorTable);
    }

    void setImage(const np::Array<uint8_t, 2> &image)
    {
        // lazy initialize QImage
        if (_qimage.width() != image.size(0) || _qimage.height() != image.size(1))
        {
            _qimage = QImage(image.size(0), image.size(1), QImage::Format_Indexed8);
			_qimage.setColorTable(_inverted ? _invertedColorTable : _standardColorTable);
		}

        // copy image data
        uchar *targetBuffer = _qimage.bits();
        memcpy(targetBuffer, image, np::byteSize(image));

        // refresh image view
        update();
    }

    void setCrop(int left, int top, int right, int bottom)
    {
        _left = left;
        _top = top;
        _right = right;
        _bottom = bottom;

        update();
    }

	void setInverted(bool inverted)
	{
		_inverted = inverted;

		if (!_qimage.isNull())
		{
			_qimage.setColorTable(_inverted ? _invertedColorTable : _standardColorTable);
			update();
		}
	}

	// signal
	std::function<void(Qt::MouseButton button)> onClicked;

protected:
    void paintEvent(QPaintEvent *event) override
    {
        QPainter painter(this);
        painter.drawImage(
            QRect(0, 0, width(), height()), 
            _qimage, 
            QRect(_left, _top, _qimage.width() - _left - _right, _qimage.height() - _top - _bottom));
        painter.end();
    }

    void mousePressEvent(QMouseEvent *event) override
    {
        if (onClicked)
            onClicked(event->button());
    }

private:
	QImage _qimage;
	QVector<QRgb> _standardColorTable, _invertedColorTable;
	bool _inverted = false;
	int _left = 0, _top = 0, _right = 0, _bottom = 0; // crop
};

#endif // QIMAGE_VIEW_H_