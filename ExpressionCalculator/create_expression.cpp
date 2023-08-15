#include <iostream>
#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <memory>
#include <functional>

template <class T>
class Expression;

template <class T>
class Variable
{
    friend class IO;
    private:
        std::string name;
        T value;
    public:
        Variable(): name("n/a"){}
        Variable(std::string var_name, T var_value): name(var_name){ value=var_value;}
        std::string get_name(){ return name; }
        T get_value(){ return value; }
};


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
class Expression
{
    std::vector<std::tuple<Expression<T>, Variable<T> *>> operand_list;
    std::string name;
    std::function<T(std::vector<T>)> op;

public:
    Expression(){name="__NULL";}
    Expression(std::function<T(std::vector<T>)> o): op(o){name="n/a";}
    Expression(std::function<T(std::vector<T>)> o, std::string s):name(s), op(o){}

    void addOperand(const Expression<T> &ex)
    {
        operand_list.push_back({ex, NULL});
    }

    void addOperand(Variable<T> *var)
    {
        operand_list.push_back({Expression<T>(), var});
    }

    T compute()
    {
        std::vector<T> values(operand_list.size());
        for (size_t idx=0; idx < operand_list.size(); idx++)
        {
            if (std::get<1>(operand_list[idx]) == NULL)
            {
                values[idx] = std::get<0>(operand_list[idx]).compute();
            }else
            {
                values[idx] = std::get<1>(operand_list[idx])->get_value();
            }
        }
        return op(values);
    }
};

class IO{
    private:
        std::string io_name;
    public:
        IO(std::string name){ io_name=name; }
        template <class T>
        Variable<T> DefineVariable(const std::string name, const T value)
        {
            Variable<T> v(name, value);
            return v;
        }
};

int main ()
{
    IO io("Test");
    auto var1 = io.DefineVariable<float>("/coord/x", 10);
    std::cout << "Variable name: " << var1.get_name() << ", value: " << var1.get_value() << std::endl;

    auto var2 = io.DefineVariable<float>("/coord/y", 5);
    std::cout << "Variable name: " << var2.get_name() << ", value: " << var2.get_value() << std::endl;
/*
    Expression<float> expr = io.DefineDerivedExpression<float>(name,
        "sqrt(px^2+py^2) \n"
        "px:/sqrt/x \n"
        "py:/coord/y")

    Expression<float> expr = io.DefineDerivedExpression<float>(name,
        "sqrt('/coord/x'^2+'/coord/y'^2)")
 */
    Expression<float> ex(detail::AddOp<float>, "add");
    ex.addOperand(&var1);
    ex.addOperand(&var2);
    Expression<float> ex2(detail::MulOp<float>, "complex");
    ex2.addOperand(ex);
    ex2.addOperand(&var1);
    std::cout << "'(var1 + var2) * var1': " << ex2.compute() << std::endl;
    return 0;
}
