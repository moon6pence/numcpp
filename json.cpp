#include "QuickDialog.h"
#include <json/json.h>

struct json_reader : public property_visitor
{
	const Json::Value &dictionary;

	json_reader(const Json::Value &dictionary) : dictionary(dictionary) { }

	void visit(property<bool> &property) const 
	{
		if (dictionary.isMember(property.name()))
			property.set(dictionary[property.name()].asBool());
		else
			std::cout << "Warning: cannot find property \"" << property.name() << "\"" << std::endl;
	}

	void visit(property<int> &property) const
	{
		if (dictionary.isMember(property.name()))
			property.set(dictionary[property.name()].asInt());
		else
			std::cout << "Warning: cannot find property \"" << property.name() << "\"" << std::endl;
	}

	void visit(property<float> &property) const
	{ 
		if (dictionary.isMember(property.name()))
			property.set(static_cast<float>(dictionary[property.name()].asDouble()));
		else
			std::cout << "Warning: cannot find property \"" << property.name() << "\"" << std::endl;
	}

	void visit(property<std::string> &property) const
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