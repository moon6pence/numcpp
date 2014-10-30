#include "Context.h"

void Context::addPrototype(Object *prototype)
{
	_prototypes.push_back(std::unique_ptr<Object>(prototype));
}

Object *Context::create(const std::string &typeName)
{
	for (std::unique_ptr<Object> &prototype: _prototypes)
		if (prototype->getTypeName() == typeName)
		{
			Object *new_object = prototype->clone();
			addObject(new_object);

			return new_object;
		}

	return nullptr;
}

void Context::addObject(Object *object)
{
	_objects.push_back(std::unique_ptr<Object>(object));
}

Object *Context::object(const std::string &name)
{
	for (std::unique_ptr<Object> &object: _objects)
		if (object->getName() == name)
			return object.get();

	return nullptr;
}

Object &Context::getObject(const std::string &name)
{
	for (std::unique_ptr<Object> &object: _objects)
		if (object->getName() == name)
			return *object.get();

	throw std::exception("Unknown object name");
}
