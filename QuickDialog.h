#ifndef QUICK_OBJECT_H_
#define QUICK_OBJECT_H_

#include <string>
#include <vector>
#include <functional>

template <typename Arg>
struct signal
{
public:
	void connect(const std::function<void(Arg)> &slot)
	{
		_slots.push_back(slot);
	}

	void operator+=(const std::function<void(Arg)> &slot)
	{
		connect(slot);
	}

	void invoke(const Arg &arg)
	{
		for (auto i = begin(_slots); i != end(_slots); ++i)
			(*i)(arg);
	}

	void operator()(const Arg &arg)
	{
		invoke(arg);
	}

private:
	std::vector<std::function<void(Arg)>> _slots;
};

template <typename T>
struct property
{
	signal<T> valueChanged;

	property(const std::string name, T default_value = T()) :
		_name(name)
	{
		set(default_value);
	}

	const T& get() const
	{
		return _value; 
	}

	void set(const T& new_value)
	{
		if (_value != new_value)
		{
			_value = new_value;

			// Fire valueChanged event
			valueChanged(new_value);
		}
	}

	const std::string& name() const
	{
		return _name;
	}

	// Syntatic sugars
	operator const T& () const
	{
		return _value;
	}

	const T& operator=(const T& new_value)
	{
		set(new_value);
		return _value;
	}

	const T& operator=(const property<T> &property)
	{
		set(property.get());
		return _value;
	}

private:
	std::string _name;
	T _value;
};

struct operation
{
	template <class Class>
	operation(const std::string name, Class *instance, void (Class::*function)(void)) :
		_name(name), 
		_function(std::bind(function, instance))
	{
	}

	template <class Func>
	operation(const std::string name, Func function) :
		_name(name), 
		_function(function)
	{
	}

	void run() const
	{
		_function();
	}

	const std::string& name() const
	{
		return _name;
	}

	// Syntatic sugars
	void operator() () const
	{
		run();
	}

private:
	std::string _name;
	std::function<void()> _function;
};

struct property_visitor
{
	virtual void visit(property<bool> &property) const = 0;
	virtual void visit(property<int> &property) const = 0;
	virtual void visit(property<float> &property) const = 0;
	virtual void visit(property<std::string> &property) const = 0;
	virtual void visit(operation &operation) const = 0;
};

template <class visitor>
struct templated_property_visitor : public property_visitor
{
	void visit(property<bool> &property) const 
	{ 
		static_cast<const visitor *>(this)->visit(property); 
	}

	void visit(property<int> &property) const
	{ 
		static_cast<const visitor *>(this)->visit(property); 
	}

	void visit(property<float> &property) const
	{ 
		static_cast<const visitor *>(this)->visit(property); 
	}

	void visit(property<std::string> &property) const
	{ 
		static_cast<const visitor *>(this)->visit(property); 
	}

	virtual void visit(operation &operation) const = 0;
};

class Object
{
public:
	virtual ~Object() { }

	virtual void accept(property_visitor &visitor) = 0;
	virtual const std::string getTypeName() = 0;

	const std::string &getName() const { return _name; }
	void setName(const std::string &name) { _name = name; }

private:
	std::string _name;
};

// print object to console
void print(Object &object);

// Json serialize
namespace Json
{
	class Value;
};

void readJson(Object &object, const Json::Value &json);
void writeJson(Object &object, Json::Value &json);

// create Qt widget
class QWidget;

QWidget *createWidget(Object &object, QWidget *parent = 0);

#endif // QUICK_OBJECT_H_