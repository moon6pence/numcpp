#include "QuickJSON.h"

#include <json/json.h>
#include <iostream>
#include <fstream>

using namespace std;

namespace nbb {

struct json_reader : public property_visitor
{
	const Json::Value &dictionary;

	json_reader(const Json::Value &dictionary) : dictionary(dictionary) { }

	void visit(property<bool> &property) const 
	{
		if (!property.readonly)
		{
			if (dictionary.isMember(property.name()))
				property.set(dictionary[property.name()].asBool());
			else
				std::cout << "Warning: cannot find property \"" << property.name() << "\"" << std::endl;
		}
	}

	void visit(property<int> &property) const
	{
		if (!property.readonly)
		{
			if (dictionary.isMember(property.name()))
				property.set(dictionary[property.name()].asInt());
			else
				std::cout << "Warning: cannot find property \"" << property.name() << "\"" << std::endl;
		}
	}

	void visit(property<float> &property) const
	{ 
		if (!property.readonly)
		{
			if (dictionary.isMember(property.name()))
				property.set(static_cast<float>(dictionary[property.name()].asDouble()));
			else
				std::cout << "Warning: cannot find property \"" << property.name() << "\"" << std::endl;
		}
	}

	void visit(property<std::string> &property) const
	{ 
		if (!property.readonly)
		{
			if (dictionary.isMember(property.name()))
				property.set(dictionary[property.name()].asString());
			else
				std::cout << "Warning: cannot find property \"" << property.name() << "\"" << std::endl;
		}
	}

	void visit(property<Object> &property) const
	{
		if (dictionary.isMember(property.name()))
			property.set(dictionary[property.name()].asString());
		else
			std::cout << "Warning: cannot find property \"" << property.name() << "\"" << std::endl;
	}

	void visit(operation &operation) const
    {
    }
};

struct json_writer : public templated_property_visitor<json_writer>
{
	Json::Value &json;

	json_writer(Json::Value &json) : json(json) { }

	template <typename T>
	void visit(property<T> &property) const
    {
		if (!property.readonly)
			json[property.name()] = property.get();
    }

	void visit(operation &operation) const
    {
    }
};

void readJson(Object &object, const Json::Value &json)
{
	json_reader reader(json);
	object.accept(reader);
}

void writeJson(Object &object, Json::Value &json)
{
	json_writer writer(json);
	object.accept(writer);
}

bool readJson(Context &context, const std::string &filename)
{
	cout << "Load json file: " << filename << endl;
	fstream json_file(filename, ios::in);
	if (!json_file.is_open())
	{
		cout << "Failed to open json file!" << endl;
		return false;
	}

	Json::Value root;
	Json::Reader reader;

	if (!reader.parse(json_file, root))
	{
		cout  << "Failed to parse context json file!\n" << reader.getFormattedErrorMessages() << endl;
		return false;
	}

	for (auto i = begin(root); i != end(root); ++i)
	{
		Json::Value &instance = *i;
		const string name = instance["name"].asString();
		const string &typeName = instance["typeName"].asString();

		// Create object
		Object *new_object = context.create(typeName);
		if (new_object == nullptr)
		{
			cout << "Cannot find object type: " << typeName << endl;
			continue;
		}

		// Read object properties
		new_object->setName(name);
		readJson(*new_object, instance["properties"]);
	}

	return true;
}

void writeJson(Context &context, const std::string &filename)
{
	cout << "Save json file: " << filename << endl;
	Json::Value root;

	for (unique_ptr<Object> &object : context.getObjectList())
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

} // namespace nbb