#include "QuickDialog.h"
#include <iostream>

using namespace std;

struct print_visitor : public templated_property_visitor<print_visitor>
{
	template <typename T>
	void visit(property<T> &property) const 
	{ 
		cout << property.name() << " = " << property.get() << endl;
	}

	void visit(operation &operation) const
	{
		cout << operation.name() << "()" << endl;
	}
};

void print(Object &object)
{
	cout << object.getName() << " : " << object.getTypeName() << endl;

	print_visitor visitor;
	object.accept(visitor);
}