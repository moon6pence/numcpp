#ifndef QO_OBJECT_H_
#define QO_OBJECT_H_

#include "property.h"

struct property_visitor
{
	virtual void visit(property<bool> &property) const = 0;
	virtual void visit(property<int> &property) const = 0;
	virtual void visit(property<float> &property) const = 0;
	virtual void visit(property<std::string> &property) const = 0;
	virtual void visit(operation &operation) const = 0;
};

template <class visitor>
struct templated_property_visitor : public property_visitor
{
	void visit(property<bool> &property) const 
	{ 
		static_cast<const visitor *>(this)->visit(property); 
	}

	void visit(property<int> &property) const
	{ 
		static_cast<const visitor *>(this)->visit(property); 
	}

	void visit(property<float> &property) const
	{ 
		static_cast<const visitor *>(this)->visit(property); 
	}

	void visit(property<std::string> &property) const
	{ 
		static_cast<const visitor *>(this)->visit(property); 
	}

	virtual void visit(operation &operation) const = 0;
};

class Object
{
public:
	virtual ~Object() { }

	virtual const std::string getTypeName() = 0;
	virtual void accept(property_visitor &visitor) = 0;
	virtual Object *clone() = 0;

	const std::string &getName() const { return _name; }
	void setName(const std::string &name) { _name = name; }

private:
	std::string _name;
};

// print object to console
void print(Object &object);

// Json serialize
namespace Json
{
	class Value;
};

void readJson(Object &object, const Json::Value &json);
void writeJson(Object &object, Json::Value &json);

// create Qt widget
class QWidget;

QWidget *createWidget(Object &object, QWidget *parent = 0);

#endif // QO_OBJECT_H_