#ifndef QUICK_JSON_H_
#define QUICK_JSON_H_

#include "object.h"
#include "context.h"

// Json serialize
namespace Json
{
	class Value;
};

namespace nbb {

void readJson(Object &object, const Json::Value &json);
void writeJson(Object &object, Json::Value &json);

bool readJson(Context &context, const std::string &filename);
void writeJson(Context &context, const std::string &filename);

} // namespace nbb

#endif // QUICK_JSON_H_