#ifndef QIMAGE_VIEW_H_
#define QIMAGE_VIEW_H_

#include <QtWidgets>

#include <numcpp/array.h>
#include <functional>

class QImageView : public QWidget
{
public:
    QImageView(QWidget *parent = 0) : QWidget(parent), _colorTable(256)
    {
        // prepare color table
        for (int i = 0; i < 256; i++)
            _colorTable[i] = 0xFF000000 + i * 0x010101; 

        _qimage = QImage(256, 256, QImage::Format_Indexed8);
        _qimage.setColorTable(_colorTable);
    }

    void setImage(const np::Array<uint8_t, 2> &image)
    {
        // lazy initialize QImage
        if (_qimage.width() != image.size(0) || _qimage.height() != image.size(1))
        {
            _qimage = QImage(image.size(0), image.size(1), QImage::Format_Indexed8);
            _qimage.setColorTable(_colorTable);
        }

        // copy image data
        uchar *targetBuffer = _qimage.bits();
        memcpy(targetBuffer, image, np::byteSize(image));

        // refresh image view
        update();
    }

	// signal
	std::function<void(Qt::MouseButton button)> onClicked;

protected:
    void paintEvent(QPaintEvent *event) override
    {
        QPainter painter(this);
        painter.drawImage(
            QRect(0, 0, width(), height()),
            _qimage);
        painter.end();
    }

    void mousePressEvent(QMouseEvent *event) override
    {
        if (onClicked)
            onClicked(event->button());
    }

private:
	QImage _qimage;
    QVector<QRgb> _colorTable;
};

#endif // QIMAGE_VIEW_H_