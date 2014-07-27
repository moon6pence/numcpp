#include "Context.h"

void Context::addPrototype(const std::string &typeName, Object *prototype)
{
	_prototypes[typeName] = std::unique_ptr<Object>(prototype);
}

Object *Context::create(const std::string &typeName)
{
	// find prototype
	if (_prototypes.find(typeName) == _prototypes.end())
	{
		// TODO: exception
		return nullptr;
	}

	Object *new_object = _prototypes[typeName]->clone();
	addObject(new_object);

	return new_object;
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
