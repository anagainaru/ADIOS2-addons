# Math Parser

Given a string describing a math expression, we need to translate it into ADIOS expressions.

```c++
auto var = io.DefineVariable<float>("test", shape, start, count);

adios2::Expression ex("x+sin(x)");           // expression check (add, sin)
ex.set("x", var);                            // variable check

auto derVar = io.DefineDerivedVariable(ex);  // sanity check
```

- During initialization we check that we support all the expressions used in the string.
- During `set` we check that the variable has the type and shape that fits the expression.
- During `DefineDerivedVariable` we check that all variables have been set.

### Existing software

List of math parsers from https://github.com/ArashPartow/math-parser-benchmark-project

|#    |  Library                                                  |  Author                       |  License                                                  |      Numeric Type     |
| --- | :-------------------------------------------------------- | :-----------------------------| :---------------------------------------------------------| :--------------------:|
| 00  | [ATMSP](http://sourceforge.net/projects/atmsp/)           | Heinz van Saanen              | [GPL v3](http://www.opensource.org/licenses/gpl-3.0.html) | double, MPFR          |
| 01  | [ExprTk](http://www.partow.net/programming/exprtk/)       | Arash Partow                  | [MIT](https://opensource.org/licenses/MIT)                | double, float, MPFR   |
| 02  | [FParser](http://warp.povusers.org/FunctionParser/)       | Juha Nieminen & Joel Yliluoma | [LGPL](http://www.gnu.org/copyleft/lesser.html)           | double                |
| 03  | [Lepton](https://simtk.org/home/lepton)                   | Peter Eastman                 | [MIT](https://opensource.org/licenses/MIT)                | double                |
| 04  | [MathExpr](http://www.yann-ollivier.org/mathlib/mathexpr) | Yann Ollivier                 | [Copyright Notice 1997-2000](http://www.yann-ollivier.org/mathlib/mathexpr#C)     | double |
| 05  | [METL](https://github.com/TillHeinzel/METL)               | Till Heinzel                  | [Apache](https://opensource.org/licenses/Apache-2.0)      | double                |
| 06  | [MTParser](http://www.codeproject.com/Articles/7335/An-extensible-math-expression-parser-with-plug-ins)| Mathieu Jacques | [CPOL](http://www.codeproject.com/info/cpol10.aspx)| double |
| 07  | [muParser](http://muparser.beltoforion.de)                | Ingo Berg                     | [MIT](http://www.opensource.org/licenses/mit-license.php) | double, float         |
| 08  | [muParserX](http://muparserx.beltoforion.de)              | Ingo Berg                     | [MIT](http://www.opensource.org/licenses/mit-license.php) | double, float         |
| 09  | [TinyExpr](https://github.com/codeplea/tinyexpr)          | Lewis Van Winkle              | [Zlib](https://opensource.org/licenses/Zlib)              | double                |
