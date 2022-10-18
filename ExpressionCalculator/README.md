

VisIT expression calculator: [https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.2.0/gui_manual/Quantitative/Expressions.html](https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.2.0/gui_manual/Quantitative/Expressions.html)

```c++
adios2::Engine Writer = io.Open("test", adios2::Mode::Write);

std::vector<adios2::Variable<float>> varA(variablesSize);
auto varB = io.DefineVariable<float>(name, {size * Nx}, {rank * Nx}, {Nx});

Writer.Expression(adios2::Expr::Sum, varA, varB);
for (i=0; i<steps; i++){
    Writer.Put<float>(varA, value);
    Writer.Put<float>(varB, value);
}
```
