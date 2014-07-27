#ifndef QUICK_JSON_H_
#define QUICK_JSON_H_

#include "Object.h"
#include "Context.h"

// Json serialize
namespace Json
{
	class Value;
};

void readJson(Object &object, const Json::Value &json);
void writeJson(Object &object, Json::Value &json);

void load(Context &context, const std::string &filename);
void save(Context &context, const std::string &filename);

#endif // QUICK_JSON_H_