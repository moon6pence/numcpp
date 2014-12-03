#include "Object.h"

using namespace std;

namespace nbb {

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

template <typename PropertyType>
struct get_property : templated_property_visitor<get_property<PropertyType>>
{
	std::string name;
	mutable typename property<PropertyType>::value_type value;

	get_property(const std::string &name) : name(name)
	{
	}

	template <typename T>
	void visit(property<T> &property) const 
	{
		// do nothing, type not match (T != PropertyType)
	}

	template <>
	void visit(property<PropertyType> &property) const 
	{
		if (property.name() == name)
			value = property.get();
	}

	void visit(operation &operation) const
	{
	}
};

template <typename PropertyType>
struct set_property : templated_property_visitor<set_property<PropertyType>>
{
	std::string name;
	typename property<PropertyType>::value_type value;

	set_property(const std::string &name, const typename property<PropertyType>::value_type &value) : name(name), value(value)
	{
	}

	template <typename T>
	void visit(property<T> &property) const 
	{
		// do nothing, type not match (T != PropertyType)
	}

	template <>
	void visit(property<PropertyType> &property) const 
	{
		if (property.name() == name)
			property.set(value);
	}

	void visit(operation &operation) const
	{
	}
};

struct assign_properties : templated_property_visitor<assign_properties>
{
	Object &dst;
	Object &src;

	assign_properties(Object &dst, Object &src) : dst(dst), src(src)
	{
	}

	template <typename T>
	void visit(property<T> &property) const 
	{
		// FIXME: O(n^2)?
		get_property<T> getter(property.name());
		src.accept(getter);

		set_property<T> setter(property.name(), getter.value);
		dst.accept(setter);
	}

	void visit(operation &operation) const
	{
	}
};

Object *clone(Object &object)
{
	Object *new_object = object.clone();

	assign_properties visitor(*new_object, object);
	new_object->accept(visitor);

	return new_object;
}

void copy_properties(Object &dst, Object &src)
{
	assign_properties visitor(dst, src);
	dst.accept(visitor);
}

} // namespace nbb