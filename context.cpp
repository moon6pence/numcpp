#include "Context.h"

#include <fstream>
#include <json/json.h>

using namespace std;

void Context::addPrototype(const std::string &typeName, Object *prototype)
{
	_prototypes[typeName] = std::unique_ptr<Object>(prototype);
}

Object *Context::create(const string &typeName)
{
	// find prototype
	if (_prototypes.find(typeName) == _prototypes.end())
	{
		// TODO: exception
		return nullptr;
	}

	Object *new_object = _prototypes[typeName]->clone();

	// add object
	_objects.push_back(std::unique_ptr<Object>(new_object));

	return new_object;
}

Object *Context::object(const std::string &name)
{
	for (unique_ptr<Object> &object: _objects)
		if (object->getName() == name)
			return object.get();

	return nullptr;
}

void Context::load(const std::string &filename)
{
	cout << "Load json file: " << filename << endl;
	fstream json_file(filename, ios::in);
	if (!json_file.is_open())
	{
		cout << "Failed to open json file." << endl;
		return;
	}

	Json::Value root;
	Json::Reader reader;

	if (!reader.parse(json_file, root))
	{
		cout  << "Failed to parse configuration\n" << reader.getFormatedErrorMessages();
		return;
	}

	for (auto i = begin(root); i != end(root); ++i)
	{
		Json::Value &instance = *i;
		const string name = instance["name"].asString();
		const string &typeName = instance["typeName"].asString();

		// Create object
		Object *new_object = create(typeName);
		new_object->setName(name);

		// Read object properties
		readJson(*new_object, instance["properties"]);
	}
}

void Context::save(const std::string &filename)
{
	cout << "Save json file: " << filename << endl;
	Json::Value root;

	for (unique_ptr<Object> &object : _objects)
	{
		Json::Value instance;
		{
			instance["name"] = object->getName();
			instance["typeName"] = object->getTypeName();

			Json::Value json_properties;
			writeJson(*object, json_properties);

			instance["properties"] = json_properties;
		}

		root.append(instance);
	}

	Json::StyledWriter writer;
	fstream json_file(filename, ios::out);
	json_file << writer.write(root);
}
