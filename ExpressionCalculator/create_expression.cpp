#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm> // for_each
#include <numeric> // accumulate
#include <functional> // function

namespace detail
{
    template <class T>
    T AddOp (std::vector<T> values) 
    {
        
        return std::accumulate(values.begin(), values.end(), 0);
    }

    template <class T>
    T MulOp (std::vector<T> values)
    {
        T i = 1;
        std::for_each(values.begin(), values.end(), [&i](int n) { i *= n; });
        return i;
    }
}

template <class T>
class Expression{
    std::unordered_map<std::string, std::function<T(std::vector<T>)>> FunctionList =
    {
            {"add", detail::AddOp<T>},
            {"mul", detail::MulOp<T>}
    };

public:
    std::string function;
    Expression(std::string name): function(name){};
    T apply_function(std::vector<T> op){ return FunctionList[function](op);}
};

class StringExpression
{
    std::vector<std::tuple<StringExpression, std::string>> function_operators;
    std::string function_name;
public:
    StringExpression(): function_name("__NULL"){};
    StringExpression(std::string name) : function_name(name){};
};




int main()
{
    Expression<float> exp("add");
    std::cout << "add " << exp.apply_function({1,2,3,4}) << std::endl;
    return 0;
}
