#ifndef QO_PROPERTY_H_
#define QO_PROPERTY_H_

#include <string>
#include <vector>
#include <functional>
#include <limits>

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

	property(const std::string &name, T default_value = T()) :
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

// template specialization for float
template <>
struct property<float>
{
	typedef float T;

	signal<T> valueChanged;

	property(const std::string &name) : 
		_name(name), _step(1.0f), _min(std::numeric_limits<T>::min()), _max(std::numeric_limits<T>::max())
	{
	}

	property(const std::string &name, 
		T default_value, 
		T step = 1.0f, 
		T min = std::numeric_limits<T>::min(), 
		T max = std::numeric_limits<T>::max()) : _name(name), _step(step), _min(min), _max(max)
	{
		set(default_value);
	}

	const T& get() const
	{
		return _value; 
	}

	void set(const T& new_value)
	{
		if (_value == new_value)
			return; // do not fire update event
		else if (new_value > max())
			_value = max();
		else if (new_value < min())
			_value = min();
		else
			_value = new_value;

		// Fire valueChanged event
		valueChanged(new_value);
	}

	// getters
	const std::string& name() const	{ return _name; }
	T step() const { return _step; }
	T min() const { return _min; }
	T max() const { return _max; }

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
	const T _step, _min, _max;
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

#endif // QO_PROPERTY_H_