# Coeus
**COEUS: Advanced metadata to support accelerating complex queries on scientific data**

1. Expression Calculator
2. Hints for multi-tier movement

### Expression Calculator

Based on the VisIT expression calculator: [https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.2.0/gui_manual/Quantitative/Expressions.html](https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.2.0/gui_manual/Quantitative/Expressions.html)

ADIOS2 provides Expressions (default and user defined).

```c++
adios2::Expression expr(adios2::ExpressionType::Sum);
auto expr = io.DefineExpression(--user defined expression--);
```

The Expression can be applied on ADIOS2 variables within an engine and once variables are populated the derived quantity is also created.

```c++
adios2::Engine Writer = io.Open("test", adios2::Mode::Write);

std::vector<adios2::Variable<float>> varA(variablesSize);
auto varB = io.DefineVariable<float>(name, {size * Nx}, {rank * Nx}, {Nx});

Writer.Expression(expr, varA, varB);
for (i=0; i<steps; i++){
    Writer.Put<float>(varA, value);
    Writer.Put<float>(varB, value);
}
```

The expression can be created in metadata or in both metadata and stored in data buffers.
