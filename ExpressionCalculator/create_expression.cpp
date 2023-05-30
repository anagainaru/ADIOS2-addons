#include <iostream>
#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <memory>

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

template <class T>
class Operation
{
public:
    virtual T apply (std::vector<T> values) = 0;
};

template <class T>
class AddOp: public Operation<T>
{
public:
    T apply (std::vector<T> values) 
    { 
        return std::accumulate(values.begin(), values.end(), 0);
    }
};

template <class T>
class MulOp: public Operation<T>
{
public:
    T apply (std::vector<T> values)
    {
        T i = 1;
        std::for_each(values.begin(), values.end(), [&i](int n) { i *= n; });
        return i;
    }
};

template <class T>
class Expression
{
    std::vector<std::tuple<Expression<T>, Variable<T> *>> operand_list;
    std::string name;
    std::shared_ptr<Operation<T>> op;

public:
    Expression(){name="n/a";}
    Expression(std::shared_ptr<Operation<T>> o): op(o){name="n/a";}
    Expression(std::shared_ptr<Operation<T>> o, std::string s):name(s), op(o){}

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
        return op->apply(values);
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

template <class T>
class MathParser
{
    std::string formula;
    std::vector<std::tuple<std::string, Variable<T> *>> var_list;
public:
    MathParser(std::string f): formula(f){}
    void setVariable(std::string op, Variable<T> var)
    {
        // check that we have the op 
        var_list.push_back({op, &var});
    }
    Expression<T> apply()
    { 
        // check that we have enough Variables 
        AddOp<float> addOp;
        Expression<float> ex1(std::make_shared<AddOp<float>>(addOp), "sum");
        for (size_t i=0; i<var_list.size(); i++)
        {
            ex1.addOperand(std::get<1>(var_list[i]));
        }
        return ex1;
    }
};

int main ()
{
    IO io("Test");
    auto var1 = io.DefineVariable<float>("data", 10);
    std::cout << "Variable name: " << var1.get_name() << ", value: " << var1.get_value() << std::endl;

    auto var2 = io.DefineVariable<float>("data", 5);
    std::cout << "Variable name: " << var2.get_name() << ", value: " << var2.get_value() << std::endl;

    MathParser<float> mp("{1}+{2}");
    mp.setVariable("{1}", var1);
    mp.setVariable("{2}", var2);
    Expression<float> ex = mp.apply();
     std::cout << "'var1 + var2': " << ex.compute() << std::endl;
 
    MulOp<float> mulOp;
    Expression<float> ex2(std::make_shared<MulOp<float>>(mulOp), "complex");
    ex2.addOperand(ex);
    ex2.addOperand(&var2);
    std::cout << "'(var1 + var2) * var2': " << ex2.compute() << std::endl;
    return 0;
}
