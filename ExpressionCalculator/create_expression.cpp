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

    template <class T>
    T PowOp (std::vector<T> values, int base) 
    {
        return std::accumulate(values.begin(), values.end(), 0);
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
public:
    std::vector<std::tuple<StringExpression, std::string>> function_operators;
    std::string function_type;

    StringExpression(): function_type("__NULL"){};
    StringExpression(std::string name) : function_type(name){};
    void add_child(StringExpression exp)
    {
        function_operators.push_back({exp, ""});
    }
    void add_child(std::string variable)
    {
        function_operators.push_back({StringExpression(), variable});
    }
};

template <class T>
T get_string_exp(StringExpression exp, std::unordered_map<std::string, T> varToOp)
{   
    std::vector<T> temp_op_list;
    for(auto op : exp.function_operators)
    {
        std::cout << "Operation " << exp.function_type << std::endl;
        if (std::get<1>(op) == "")
        {
            std::cout << "Expanding " << std::endl;
            auto temp_res = get_string_exp(std::get<0>(op), varToOp);
            temp_op_list.push_back(temp_res);
            std::cout << "push back " << std::get<1>(op) << " : " << varToOp[std::get<1>(op)] << std::endl;
        } 
        else
        {
            std::cout << "push back " << std::get<1>(op) << " : " << varToOp[std::get<1>(op)] << std::endl;
            temp_op_list.push_back(varToOp[std::get<1>(op)]);
        }
    }
    Expression<T> temp_exp(exp.function_type);
    std::cout << "Result " << temp_exp.apply_function(temp_op_list) << std::endl;
    return temp_exp.apply_function(temp_op_list);
}

void create_test_string_exp()
{
    StringExpression exp_child("add");
    exp_child.add_child("var1");
    exp_child.add_child("var2");
    StringExpression exp("mul");
    exp.add_child("var3");
    exp.add_child(exp_child);

    auto res = get_string_exp<float>(
        exp, {{"var1", 1}, {"var2", 2}, {"var3", 3}});
    std::cout << "exp result: " << res << std::endl;
}

int main()
{
    create_test_string_exp();
    return 0;
}
