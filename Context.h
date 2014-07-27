#ifndef CONTEXT_H_
#define CONTEXT_H_

#include <memory>
#include <vector>
#include <map>

#include "object.h"

class Context
{
public:
	void addPrototype(const std::string &typeName, Object *prototype);
	Object *create(const std::string &typeName);
	void addObject(Object *object);

	Object *object(const std::string &name);

	template <typename T>
	T *object(const std::string &name) 
	{ 
		return static_cast<T *>(object(name));
	}

	std::vector<std::unique_ptr<Object>> &objects() { return _objects; }

private:
	std::map<std::string, std::unique_ptr<Object>> _prototypes;
	std::vector<std::unique_ptr<Object>> _objects;
};

#endif // CONTEXT_H_