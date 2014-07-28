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
	Object(const std::string &typeName) : _typeName(typeName) { }
	virtual ~Object() { }

	virtual void accept(property_visitor &visitor) = 0;
	virtual Object *clone() = 0;

	const std::string getTypeName() { return _typeName; }

	const std::string &getName() const { return _name; }
	void setName(const std::string &name) { _name = name; }

private:
	const std::string _typeName;
	std::string _name;
};

// print object to console
void print(Object &object);

#endif // QO_OBJECT_H_