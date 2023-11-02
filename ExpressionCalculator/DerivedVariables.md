## Build system

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index ebb992c1f..aebbd5b91 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -170,7 +170,8 @@ adios_option(Profiling  "Enable support for profiling" AUTO)
 adios_option(Endian_Reverse "Enable support for Little/Big Endian Interoperability" AUTO)
 adios_option(Sodium     "Enable support for Sodium for encryption" AUTO)
 adios_option(Catalyst   "Enable support for in situ visualization plugin using ParaView Catalyst" AUTO)
+adios_option(Derived    "Enable support for derived variables" OFF)
 include(${PROJECT_SOURCE_DIR}/cmake/DetectOptions.cmake)

 if(ADIOS2_HAVE_CUDA OR ADIOS2_HAVE_Kokkos_CUDA)
@@ -243,8 +244,8 @@ endif()
 set(ADIOS2_CONFIG_OPTS
     DataMan DataSpaces HDF5 HDF5_VOL MHS SST Fortran MPI Python Blosc2 BZip2
     LIBPRESSIO MGARD PNG SZ ZFP DAOS IME O_DIRECT Sodium Catalyst SysVShMem UCX
-    ZeroMQ Profiling Endian_Reverse AWSSDK GPU_Support CUDA Kokkos Kokkos_CUDA
-    Kokkos_HIP Kokkos_SYCL
+    ZeroMQ Profiling Endian_Reverse Derived AWSSDK GPU_Support CUDA Kokkos
+    Kokkos_CUDA Kokkos_HIP Kokkos_SYCL
 )

 GenerateADIOSHeaderConfig(${ADIOS2_CONFIG_OPTS})
diff --git a/bindings/CXX11/CMakeLists.txt b/bindings/CXX11/CMakeLists.txt
index 861764313..3e62fdcb8 100644
--- a/bindings/CXX11/CMakeLists.txt
+++ b/bindings/CXX11/CMakeLists.txt
@@ -37,6 +37,12 @@ target_include_directories(adios2_cxx11

 add_library(adios2::cxx11 ALIAS adios2_cxx11)

+if (ADIOS2_HAVE_Derived)
+    target_sources(adios2_cxx11 PRIVATE
+      adios2/cxx11/VariableDerived.cpp
+  )
+endif()
+
 if(ADIOS2_HAVE_MPI)
   add_library(adios2_cxx11_mpi
     adios2/cxx11/ADIOSMPI.cpp
@@ -79,6 +85,14 @@ install(
   COMPONENT adios2_cxx11-development
 )

+if (ADIOS2_HAVE_Derived)
+    install(
+        FILES adios2/cxx11/VariableDerived.h
+        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/adios2/cxx11
+        COMPONENT adios2_cxx11-development
+    )
+endif()
+
 install(
   FILES adios2/cxx11/ADIOS.h
         adios2/cxx11/ADIOS.inl
diff --git a/cmake/DetectOptions.cmake b/cmake/DetectOptions.cmake
index 3f511e02a..dde40a883 100644
--- a/cmake/DetectOptions.cmake
+++ b/cmake/DetectOptions.cmake
@@ -193,6 +193,10 @@ endif()

 set(mpi_find_components C)

+if(ADIOS2_USE_Derived)
+    set(ADIOS2_HAVE_Derived TRUE)
+endif()
+
 if(ADIOS2_USE_Kokkos AND ADIOS2_USE_CUDA)
   message(FATAL_ERROR "ADIOS2_USE_Kokkos is incompatible with ADIOS2_USE_CUDA")
 endif()
diff --git a/examples/derived/CMakeLists.txt b/examples/derived/CMakeLists.txt
new file mode 100644
index 000000000..aa1246de9
--- /dev/null
+++ b/examples/derived/CMakeLists.txt
@@ -0,0 +1,5 @@
+add_executable(bpExpression_write BP5Write.cpp)
+target_link_libraries(bpExpression_write MPI::MPI_C adios2::cxx11 adios2::cxx11_mpi)
+
+add_executable(bpExpression_read BP5Read.cpp)
+target_link_libraries(bpExpression_read MPI::MPI_C adios2::cxx11 adios2::cxx11_mpi)
diff --git a/source/adios2/CMakeLists.txt b/source/adios2/CMakeLists.txt
index 343c70d42..267588ae3 100644
--- a/source/adios2/CMakeLists.txt
+++ b/source/adios2/CMakeLists.txt
@@ -127,6 +127,21 @@ add_library(adios2_core
 set_property(TARGET adios2_core PROPERTY EXPORT_NAME core)
 set_property(TARGET adios2_core PROPERTY OUTPUT_NAME adios2${ADIOS2_LIBRARY_SUFFIX}_core)

+set(maybe_adios2_core_derived)
+if (ADIOS2_HAVE_Derived)
+  target_sources(adios2_core PRIVATE
+      core/VariableDerived.cpp
+      toolkit/derived/Expression.cpp
+      toolkit/derived/Function.cpp toolkit/derived/Function.tcc
+      toolkit/derived/parser/ExprHelper.h)
+  add_library(adios2_core_derived
+      toolkit/derived/parser/lexer.cpp
+      toolkit/derived/parser/parser.cpp
+      toolkit/derived/parser/ASTNode.cpp)
+  target_link_libraries(adios2_core PRIVATE adios2_core_derived)
+  set(maybe_adios2_core_derived adios2_core_derived)
+endif()
+
 set(maybe_adios2_core_cuda)
 if(ADIOS2_HAVE_CUDA)
   add_library(adios2_core_cuda helper/adiosCUDA.cu)
@@ -447,10 +462,11 @@ install(DIRECTORY toolkit/
   PATTERN "*/*.inl"
   REGEX "sst/util" EXCLUDE
   REGEX "sst/dp" EXCLUDE
+  REGEX "derived/parser" EXCLUDE
 )

 # Library installation
-install(TARGETS adios2_core ${maybe_adios2_core_mpi} ${maybe_adios2_core_cuda} ${maybe_adios2_core_kokkos} ${maybe_adios2_blosc2} EXPORT adios2Exports
+install(TARGETS adios2_core ${maybe_adios2_core_mpi} ${maybe_adios2_core_cuda} ${maybe_adios2_core_kokkos} ${maybe_adios2_blosc2} ${maybe_adios2_core_derived} EXPORT adios2Expor
ts
   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT adios2_core-runtime
   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT adios2_core-libraries NAMELINK_COMPONENT adios2_core-development
   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT adios2_core-development
```

## Bindings

```diff
diff --git a/bindings/CXX11/adios2/cxx11/IO.cpp b/bindings/CXX11/adios2/cxx11/IO.cpp
index 8018c06d7..fb5cb55ca 100644
--- a/bindings/CXX11/adios2/cxx11/IO.cpp
+++ b/bindings/CXX11/adios2/cxx11/IO.cpp
@@ -179,6 +179,16 @@ VariableNT IO::DefineVariable(const DataType type, const std::string &name, cons
     }
 }

+#ifdef ADIOS2_HAVE_DERIVED
+VariableDerived IO::DefineDerivedVariable(const std::string &name, const std::string &expression,
+                                          const DerivedVarType varType)
+{
+    helper::CheckForNullptr(m_IO,
+                            "for variable name " + name + ", in call to IO::DefineDerivedVariable");
+
+    return VariableDerived(&m_IO->DefineDerivedVariable(name, expression, varType));
+}
+#endif
 StructDefinition IO::DefineStruct(const std::string &name, const size_t size)
 {
     helper::CheckForNullptr(m_IO, "for struct name " + name + ", in call to IO::DefineStruct");
diff --git a/bindings/CXX11/adios2/cxx11/IO.h b/bindings/CXX11/adios2/cxx11/IO.h
index 1702c769f..b3ea8150f 100644
--- a/bindings/CXX11/adios2/cxx11/IO.h
+++ b/bindings/CXX11/adios2/cxx11/IO.h
@@ -20,6 +20,9 @@
 #include "Group.h"
 #include "Operator.h"
 #include "Variable.h"
+#ifdef ADIOS2_HAVE_DERIVED
+#include "VariableDerived.h"
+#endif
 #include "VariableNT.h"
 #include "adios2/common/ADIOSMacros.h"
 #include "adios2/common/ADIOSTypes.h"
@@ -151,7 +154,11 @@ public:
     Variable<T> DefineVariable(const std::string &name, const Dims &shape = Dims(),
                                const Dims &start = Dims(), const Dims &count = Dims(),
                                const bool constantDims = false);
-
+#ifdef ADIOS2_HAVE_DERIVED
+    VariableDerived
+    DefineDerivedVariable(const std::string &name, const std::string &expression,
+                          const DerivedVarType varType = DerivedVarType::MetadataOnly);
+#endif
     VariableNT DefineVariable(const DataType type, const std::string &name,
                               const Dims &shape = Dims(), const Dims &start = Dims(),
                               const Dims &count = Dims(), const bool constantDims = false);
diff --git a/bindings/CXX11/adios2/cxx11/VariableDerived.cpp b/bindings/CXX11/adios2/cxx11/VariableDerived.cpp
new file mode 100644
index 000000000..ce6a4dc0e
--- /dev/null
+++ b/bindings/CXX11/adios2/cxx11/VariableDerived.cpp
@@ -0,0 +1,8 @@
+#include "VariableDerived.h"
+
+#include "adios2/core/VariableDerived.h"
+
+namespace adios2
+{
+VariableDerived::VariableDerived(core::VariableDerived *variable) : m_VariableDerived(variable) {}
+} // end namespace adios2
diff --git a/bindings/CXX11/adios2/cxx11/VariableDerived.h b/bindings/CXX11/adios2/cxx11/VariableDerived.h
new file mode 100644
index 000000000..cc69273c2
--- /dev/null
+++ b/bindings/CXX11/adios2/cxx11/VariableDerived.h
@@ -0,0 +1,43 @@
+#ifndef ADIOS2_BINDINGS_CXX11_VARIABLE_DERIVED_H_
+#define ADIOS2_BINDINGS_CXX11_VARIABLE_DERIVED_H_
+
+#include "Operator.h"
+#include "adios2/common/ADIOSTypes.h"
+
+namespace adios2
+{
+
+/// \cond EXCLUDE_FROM_DOXYGEN
+// forward declare
+class IO; // friend
+namespace core
+{
+
+class VariableDerived; // private implementation
+}
+/// \endcond
+
+class VariableDerived
+{
+    friend class IO;
+
+public:
+    /**
+     * Empty (default) constructor, use it as a placeholder for future
+     * variables from IO:DefineVariableDerived<T> or IO:InquireVariableDerived<T>.
+     * Can be used with STL containers.
+     */
+    VariableDerived() = default;
+
+    /** Default, using RAII STL containers */
+    ~VariableDerived() = default;
+
+private:
+    core::VariableDerived *m_VariableDerived = nullptr;
+
+    VariableDerived(core::VariableDerived *variable);
+};
+
+} // end namespace adios2
+
+#endif // ADIOS2_BINDINGS_CXX11_VARIABLE_DERIVED_H_
```

## Grammar

All the files in toolkit/derived/parser

## Storing derived variables

All the files in toolkit/derived/*.h/cpp/tcc

```diff
diff --git a/source/adios2/common/ADIOSTypes.h b/source/adios2/common/ADIOSTypes.h
index 7697c3462..fc14274ab 100644
--- a/source/adios2/common/ADIOSTypes.h
+++ b/source/adios2/common/ADIOSTypes.h
@@ -32,6 +32,16 @@
 namespace adios2
 {

+#ifdef ADIOS2_HAVE_DERIVED
+/** Type of derived variables */
+enum class DerivedVarType
+{
+    MetadataOnly,     ///< Store only the metadata (default)
+    ExpressionString, ///< Store only the expression string
+    StoreData         ///< Store data and metadata
+};
+#endif
+
 /** Memory space for the user provided buffers */
 enum class MemorySpace
 {
diff --git a/source/adios2/core/IO.cpp b/source/adios2/core/IO.cpp
index 1b99463b5..ecfe2083f 100644
--- a/source/adios2/core/IO.cpp
+++ b/source/adios2/core/IO.cpp
@@ -288,6 +288,9 @@ void IO::SetTransportParameter(const size_t transportIndex, const std::string ke
 }

 const VarMap &IO::GetVariables() const noexcept { return m_Variables; }
+#ifdef ADIOS2_HAVE_DERIVED
+const VarMap &IO::GetDerivedVariables() const noexcept { return m_VariablesDerived; }
+#endif

 const AttrMap &IO::GetAttributes() const noexcept { return m_Attributes; }

@@ -808,6 +811,92 @@ void IO::CheckTransportType(const std::string type) const
     }
 }

+#ifdef ADIOS2_HAVE_DERIVED
+VariableDerived &IO::DefineDerivedVariable(const std::string &name, const std::string &exp_string,
+                                           const DerivedVarType varType)
+{
+    PERFSTUBS_SCOPED_TIMER("IO::DefineDerivedVariable");
+
+    {
+        auto itVariable = m_VariablesDerived.find(name);
+        if (itVariable != m_VariablesDerived.end())
+        {
+            helper::Throw<std::invalid_argument>("Core", "IO", "DefineDerivedVariable",
+                                                 "derived variable " + name +
+                                                     " already defined in IO " + m_Name);
+        }
+        else
+        {
+            auto itVariable = m_Variables.find(name);
+            if (itVariable != m_Variables.end())
+            {
+                helper::Throw<std::invalid_argument>(
+                    "Core", "IO", "DefineDerivedVariable",
+                    "derived variable " + name +
+                        " trying to use an already defined variable name in IO " + m_Name);
+            }
+        }
+    }
+
+    derived::Expression derived_exp(exp_string);
+    std::vector<std::string> var_list = derived_exp.VariableNameList();
+    DataType expressionType = DataType::None;
+    bool isConstant = true;
+    std::map<std::string, std::tuple<Dims, Dims, Dims>> name_to_dims;
+    // check correctness for the variable names and types within the expression
+    for (auto var_name : var_list)
+    {
+        auto itVariable = m_Variables.find(var_name);
+        if (itVariable == m_Variables.end())
+            helper::Throw<std::invalid_argument>("Core", "IO", "DefineDerivedVariable",
+                                                 "using undefine variable " + var_name +
+                                                     " in defining the derived variable " + name);
+        DataType var_type = InquireVariableType(var_name);
+        if (expressionType == DataType::None)
+            expressionType = var_type;
+        if (expressionType != var_type)
+            helper::Throw<std::invalid_argument>("Core", "IO", "DefineDerivedVariable",
+                                                 "all variables within a derived variable "
+                                                 " must have the same type ");
+        if ((itVariable->second)->IsConstantDims() == false)
+            isConstant = false;
+        name_to_dims.insert({var_name,
+                             {(itVariable->second)->m_Start, (itVariable->second)->m_Count,
+                              (itVariable->second)->m_Shape}});
+    }
+    std::cout << "Derived variable " << name << ": PASS : variables exist and have the same type"
+              << std::endl;
+    // set the initial shape of the expression and check correcness
+    derived_exp.SetDims(name_to_dims);
+    std::cout << "Derived variable " << name << ": PASS : initial variable dimensions are valid"
+              << std::endl;
+
+    // create derived variable with the expression
+    auto itVariablePair = m_VariablesDerived.emplace(
+        name, std::unique_ptr<VariableBase>(
+                  new VariableDerived(name, derived_exp, expressionType, isConstant, varType)));
+    VariableDerived &variable = static_cast<VariableDerived &>(*itVariablePair.first->second);
+
+    // check IO placeholder for variable operations
+    auto itOperations = m_VarOpsPlaceholder.find(name);
+    if (itOperations != m_VarOpsPlaceholder.end())
+    {
+        // allow to apply an operation only for derived variables that save the data
+        if (varType != DerivedVarType::StoreData)
+            helper::Throw<std::invalid_argument>(
+                "Core", "IO", "DefineDerivedVariable",
+                "Operators for derived variables can only be applied "
+                " for DerivedVarType::StoreData types.");
+        variable.m_Operations.reserve(itOperations->second.size());
+        for (auto &operation : itOperations->second)
+        {
+            variable.AddOperation(operation.first, operation.second);
+        }
+    }
+    return variable;
+}
+#endif
+
 StructDefinition &IO::DefineStruct(const std::string &name, const size_t size)
 {
     return m_ADIOS.m_StructDefinitions.emplace(name, StructDefinition(name, size))->second;
diff --git a/source/adios2/core/IO.h b/source/adios2/core/IO.h
index 4d64982f3..cec260064 100644
--- a/source/adios2/core/IO.h
+++ b/source/adios2/core/IO.h
@@ -28,6 +28,9 @@
 #include "adios2/core/CoreTypes.h"
 #include "adios2/core/Group.h"
 #include "adios2/core/Variable.h"
+#ifdef ADIOS2_HAVE_DERIVED
+#include "adios2/core/VariableDerived.h"
+#endif
 #include "adios2/core/VariableStruct.h"

 namespace adios2
@@ -179,7 +182,11 @@ public:
     Variable<T> &DefineVariable(const std::string &name, const Dims &shape = Dims(),
                                 const Dims &start = Dims(), const Dims &count = Dims(),
                                 const bool constantDims = false);
-
+#ifdef ADIOS2_HAVE_DERIVED
+    VariableDerived &
+    DefineDerivedVariable(const std::string &name, const std::string &expression,
+                          const DerivedVarType varType = DerivedVarType::MetadataOnly);
+#endif
     VariableStruct &DefineStructVariable(const std::string &name, StructDefinition &def,
                                          const Dims &shape = Dims(), const Dims &start = Dims(),
                                          const Dims &count = Dims(),
@@ -304,6 +311,9 @@ public:
      * </pre>
      */
     const VarMap &GetVariables() const noexcept;
+#ifdef ADIOS2_HAVE_DERIVED
+    const VarMap &GetDerivedVariables() const noexcept;
+#endif

     /**
      * Retrieves hash holding internal Attributes identifiers
@@ -500,6 +510,9 @@ private:
     adios2::IOMode m_IOMode = adios2::IOMode::Independent;

     VarMap m_Variables;
+#ifdef ADIOS2_HAVE_DERIVED
+    VarMap m_VariablesDerived;
+#endif

     AttrMap m_Attributes;
diff --git a/source/adios2/core/VariableDerived.cpp b/source/adios2/core/VariableDerived.cpp
new file mode 100644
index 000000000..bf7d9ad0f
--- /dev/null
+++ b/source/adios2/core/VariableDerived.cpp
@@ -0,0 +1,140 @@
+#include "VariableDerived.h"
+#include "adios2/helper/adiosType.h"
+
+namespace adios2
+{
+namespace core
+{
+
+VariableDerived::VariableDerived(const std::string &name, adios2::derived::Expression expr,
+                                 const DataType exprType, const bool isConstant,
+                                 const DerivedVarType varType)
+: VariableBase(name, exprType, helper::GetDataTypeSize(exprType), expr.GetShape(), expr.GetStart(),
+               expr.GetCount(), isConstant),
+  m_Expr(expr), m_DerivedType(varType)
+{
+    std::cout << "Creating derived variable " << name << std::endl;
+    std::cout << "Set initial dimensions for derived variable " << name << " start: " << m_Start
+              << " count: " << m_Count << std::endl;
+}
+
+DerivedVarType VariableDerived::GetDerivedType() { return m_DerivedType; }
+
+std::vector<std::string> VariableDerived::VariableNameList() { return m_Expr.VariableNameList(); }
+void VariableDerived::UpdateExprDim(std::map<std::string, std::tuple<Dims, Dims, Dims>> NameToDims)
+{
+    m_Expr.SetDims(NameToDims);
+    m_Shape = m_Expr.GetShape();
+    m_Start = m_Expr.GetStart();
+    m_Count = m_Expr.GetCount();
+}
+
+std::vector<std::tuple<void *, Dims, Dims>>
+VariableDerived::ApplyExpression(std::map<std::string, MinVarInfo *> NameToMVI)
+{
+    size_t numBlocks = 0;
+    // check that all variables have the same number of blocks
+    for (auto variable : NameToMVI)
+    {
+        if (numBlocks == 0)
+            numBlocks = variable.second->BlocksInfo.size();
+        if (numBlocks != variable.second->BlocksInfo.size())
+            helper::Throw<std::invalid_argument>("Core", "VariableDerived", "ApplyExpression",
+                                                 " variables do not have the same number of blocks "
+                                                 " in computing the derived variable " +
+                                                     m_Name);
+    }
+    std::cout << "Derived variable " << m_Name
+              << ": PASS : variables have written the same num of blocks" << std::endl;
+
+    std::map<std::string, std::vector<adios2::derived::DerivedData>> inputData;
+    // create the map between variable name and DerivedData object
+    for (auto variable : NameToMVI)
+    {
+        // add the dimensions of all blocks into a vector
+        std::vector<adios2::derived::DerivedData> varData;
+        for (size_t i = 0; i < numBlocks; i++)
+        {
+            Dims start;
+            Dims count;
+            for (size_t d = 0; d < variable.second->Dims; d++)
+            {
+                start.push_back(variable.second->BlocksInfo[i].Start[d]);
+                count.push_back(variable.second->BlocksInfo[i].Count[d]);
+            }
+            varData.push_back(adios2::derived::DerivedData(
+                {variable.second->BlocksInfo[i].BufferP, start, count}));
+        }
+        inputData.insert({variable.first, varData});
+    }
+    // TODO check that the dimensions are still corrects
+    std::vector<adios2::derived::DerivedData> outputData =
+        m_Expr.ApplyExpression(m_Type, numBlocks, inputData);
+
+    std::vector<std::tuple<void *, Dims, Dims>> blockData;
+    for (size_t i = 0; i < numBlocks; i++)
+    {
+        blockData.push_back({outputData[i].Data, outputData[i].Start, outputData[i].Count});
+    }
+
+    // DEBUG - TODO remove the switch
+    switch (m_DerivedType)
+    {
+    case DerivedVarType::MetadataOnly:
+        std::cout << "Store only metadata for derived variable " << m_Name << std::endl;
+        break;
+    case DerivedVarType::StoreData:
+        std::cout << "Store data and metadata for derived variable " << m_Name << std::endl;
+        break;
+    default:
+        // we currently only support Metadata/StoreData modes, TODO ExpressionString
+        break;
+    }
+
+    return blockData;
+}
+
+std::vector<void *>
+VariableDerived::ApplyExpression(std::map<std::string, std::vector<void *>> NameToData,
+                                 std::map<std::string, std::tuple<Dims, Dims, Dims>> NameToDims)
+{
+    size_t numBlocks = 0;
+    std::map<std::string, std::vector<adios2::derived::DerivedData>> inputData;
+    // check that all variables have the same number of blocks
+    for (auto variable : NameToData)
+    {
+        if (numBlocks == 0)
+            numBlocks = variable.second.size();
+        if (numBlocks != variable.second.size())
+            helper::Throw<std::invalid_argument>("Core", "VariableDerived", "ApplyExpression",
+                                                 " variables do not have the same number of blocks "
+                                                 " in computing the derived variable " +
+                                                     m_Name);
+    }
+    std::cout << "Derived variable " << m_Name
+              << ": PASS : variables have written the same num of blocks" << std::endl;
+    // create the map between variable name and DerivedData object
+    for (auto variable : NameToData)
+    {
+        // add the dimensions of all blocks into a vector
+        std::vector<adios2::derived::DerivedData> varData;
+        for (size_t i = 0; i < numBlocks; i++)
+        {
+            varData.push_back(adios2::derived::DerivedData(
+                {variable.second[i], std::get<0>(NameToDims[variable.first]),
+                 std::get<1>(NameToDims[variable.first])}));
+        }
+        inputData.insert({variable.first, varData});
+    }
+    std::vector<adios2::derived::DerivedData> outputData =
+        m_Expr.ApplyExpression(m_Type, numBlocks, inputData);
+    std::vector<void *> blockData;
+    for (size_t i = 0; i < numBlocks; i++)
+    {
+        blockData.push_back(outputData[i].Data);
+    }
+    return blockData;
+}
+
+} // end namespace core
+} // end namespace adios2
diff --git a/source/adios2/core/VariableDerived.h b/source/adios2/core/VariableDerived.h
new file mode 100644
index 000000000..26b079b64
--- /dev/null
+++ b/source/adios2/core/VariableDerived.h
@@ -0,0 +1,41 @@
+#ifndef ADIOS2_CORE_VARIABLE_DERIVED_H_
+#define ADIOS2_CORE_VARIABLE_DERIVED_H_
+
+#include "adios2/common/ADIOSTypes.h"
+#include "adios2/core/VariableBase.h"
+#include "adios2/helper/adiosType.h"
+#include "adios2/toolkit/derived/Expression.h"
+
+namespace adios2
+{
+namespace core
+{
+
+/**
+ * @param Base (parent) class for template derived (child) class Variable.
+ */
+class VariableDerived : public VariableBase
+{
+    DerivedVarType m_DerivedType;
+
+public:
+    adios2::derived::Expression m_Expr;
+    VariableDerived(const std::string &name, adios2::derived::Expression expr,
+                    const DataType exprType, const bool isConstant, const DerivedVarType varType);
+    ~VariableDerived() = default;
+
+    DerivedVarType GetDerivedType();
+    std::vector<std::string> VariableNameList();
+    void UpdateExprDim(std::map<std::string, std::tuple<Dims, Dims, Dims>> NameToDims);
+
+    std::vector<void *>
+    ApplyExpression(std::map<std::string, std::vector<void *>> NameToData,
+                    std::map<std::string, std::tuple<Dims, Dims, Dims>> NameToDims);
+    std::vector<std::tuple<void *, Dims, Dims>>
+    ApplyExpression(std::map<std::string, MinVarInfo *> mvi);
+};
+
+} // end namespace core
+} // end namespace adios2
+
+#endif /* ADIOS2_CORE_VARIABLE_DERIVED_H_ */
```

## Computing derived variables

```diff
diff --git a/source/adios2/engine/bp5/BP5Writer.cpp b/source/adios2/engine/bp5/BP5Writer.cpp
index 465a44d95..403ec4e93 100644
--- a/source/adios2/engine/bp5/BP5Writer.cpp
+++ b/source/adios2/engine/bp5/BP5Writer.cpp
@@ -495,8 +495,73 @@ void BP5Writer::MarshalAttributes()
     }
 }

+#ifdef ADIOS2_HAVE_DERIVED
+void BP5Writer::ComputeDerivedVariables()
+{
+    auto const &m_VariablesDerived = m_IO.GetDerivedVariables();
+    auto const &m_Variables = m_IO.GetVariables();
+    // parse all derived variables
+    std::cout << " Parsing " << m_VariablesDerived.size() << " derived variables" << std::endl;
+    for (auto it = m_VariablesDerived.begin(); it != m_VariablesDerived.end(); it++)
+    {
+        // identify the variables used in the derived variable
+        auto derivedVar = dynamic_cast<core::VariableDerived *>((*it).second.get());
+        std::vector<std::string> varList = derivedVar->VariableNameList();
+        // to create a mapping between variable name and the varInfo (dim and data pointer)
+        std::map<std::string, MinVarInfo *> nameToVarInfo;
+        bool computeDerived = true;
+        for (auto varName : varList)
+        {
+            auto itVariable = m_Variables.find(varName);
+            if (itVariable == m_Variables.end())
+                helper::Throw<std::invalid_argument>("Core", "IO", "DefineDerivedVariable",
+                                                     "using undefine variable " + varName +
+                                                         " in defining the derived variable " +
+                                                         (*it).second->m_Name);
+            // extract the dimensions and data for each variable
+            VariableBase *varBase = itVariable->second.get();
+            auto mvi = WriterMinBlocksInfo(*varBase);
+            if (mvi->BlocksInfo.size() == 0)
+            {
+                computeDerived = false;
+                std::cout << "Variable " << itVariable->first << " not written in this step";
+                std::cout << " .. skip derived variable " << (*it).second->m_Name << std::endl;
+                break;
+            }
+            nameToVarInfo.insert({varName, mvi});
+        }
+        // skip computing derived variables if it contains variables that are not written this step
+        if (!computeDerived)
+            continue;
+
+        // compute the values for the derived variables that are not type ExpressionString
+        std::vector<std::tuple<void *, Dims, Dims>> DerivedBlockData;
+        if (derivedVar->GetDerivedType() != DerivedVarType::ExpressionString)
+        {
+            DerivedBlockData = derivedVar->ApplyExpression(nameToVarInfo);
+        }
+
+        // Send the derived variable to ADIOS2 internal logic
+        for (auto derivedBlock : DerivedBlockData)
+        {
+            // set the shape of the variable for each block
+            if (!(*it).second->IsConstantDims())
+            {
+                (*it).second->m_Start = std::get<1>(derivedBlock);
+                (*it).second->m_Count = std::get<2>(derivedBlock);
+            }
+            PutCommon(*(*it).second.get(), std::get<0>(derivedBlock), true /* sync */);
+            free(std::get<0>(derivedBlock));
+        }
+    }
+}
+#endif
+
 void BP5Writer::EndStep()
 {
+#ifdef ADIOS2_HAVE_DERIVED
+    ComputeDerivedVariables();
+#endif
     m_BetweenStepPairs = false;
     PERFSTUBS_SCOPED_TIMER("BP5Writer::EndStep");
     m_Profiler.Start("ES");
@@ -504,26 +569,6 @@ void BP5Writer::EndStep()
     m_Profiler.Start("ES_close");
     MarshalAttributes();

-#ifdef NOT_DEF
-    const auto &vars = m_IO.GetVariables();
-    for (const auto &varPair : vars)
-    {
-        auto baseVar = varPair.second.get();
-        auto mvi = WriterMinBlocksInfo(*baseVar);
-        if (mvi)
-        {
-            std::cout << "Info for Variable " << varPair.first << std::endl;
-            PrintMVI(std::cout, *mvi);
-            if (baseVar->m_Type == DataType::Double)
-                std::cout << "Double value is " << *((double *)mvi->BlocksInfo[0].BufferP)
-                          << std::endl;
-            delete mvi;
-        }
-        else
-            std::cout << "Variable " << varPair.first << " not written on this step" << std::endl;
-    }
-#endif
-
     // true: advances step
     auto TSInfo = m_BP5Serializer.CloseTimestep((int)m_WriterStep,
                                                 m_Parameters.AsyncWrite || m_Parameters.DirectIO);
diff --git a/source/adios2/engine/bp5/BP5Writer.h b/source/adios2/engine/bp5/BP5Writer.h
index eaddf93b9..cc0d307ee 100644
--- a/source/adios2/engine/bp5/BP5Writer.h
+++ b/source/adios2/engine/bp5/BP5Writer.h
@@ -119,6 +119,10 @@ private:
     /** Inform about computation block through User->ADIOS->IO */
     void ExitComputationBlock() noexcept;

+#ifdef ADIOS2_HAVE_DERIVED
+    void ComputeDerivedVariables();
+#endif
+
 #define declare_type(T)                                                                            \
     void DoPut(Variable<T> &variable, typename Variable<T>::Span &span, const bool initialize,     \
                const T &value) final;
diff --git a/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp b/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
index cf9fbefc1..6e6ea7e80 100644
--- a/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
+++ b/source/adios2/toolkit/format/bp5/BP5Deserializer.cpp
@@ -1,3 +1,4 @@
+
 /*
  * Distributed under the OSI-approved Apache License, Version 2.0.  See
  * accompanying file Copyright.txt for details.
@@ -238,8 +239,8 @@ void BP5Deserializer::BreakdownArrayName(const char *Name, char **base_name_p, D
 {
     /* string formatted as bp5_%d_%d_actualname */
     char *p;
-    // + 3 to skip BP5_ or bp5_ prefix
-    long n = strtol(Name + 4, &p, 10);
+    // Prefix has already been skipped
+    long n = strtol(Name, &p, 10);
     *element_size_p = static_cast<int>(n);
     ++p; // skip '_'
     long Type = strtol(p, &p, 10);
@@ -295,6 +296,59 @@ BP5Deserializer::BP5VarRec *BP5Deserializer::CreateVarRec(const char *ArrayName)
     return Ret;
 }

+/*
+ * Decode base64 data to 'output'.  Decode in-place if 'output' is NULL.
+ * Return the length of the decoded data, or -1 if there was an error.
+ */
+static const char signed char_to_num[256] = {
+    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
+    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,
+    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,
+    7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,
+    -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
+    49, 50, 51, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
+    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
+    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
+    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
+    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
+    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
+};
+static int base64_decode(unsigned char *input, unsigned char *output)
+{
+    int len = 0;
+    int c1, c2, c3, c4;
+
+    if (output == NULL)
+        output = input;
+    while (*input)
+    {
+        c1 = *input++;
+        if (char_to_num[c1] == -1)
+            return -1;
+        c2 = *input++;
+        if (char_to_num[c2] == -1)
+            return -1;
+        c3 = *input++;
+        if (c3 != '=' && char_to_num[c3] == -1)
+            return -1;
+        c4 = *input++;
+        if (c4 != '=' && char_to_num[c4] == -1)
+            return -1;
+        *output++ = (char_to_num[c1] << 2) | (char_to_num[c2] >> 4);
+        ++len;
+        if (c3 == '=')
+            break;
+        *output++ = ((char_to_num[c2] << 4) & 0xf0) | (char_to_num[c3] >> 2);
+        ++len;
+        if (c4 == '=')
+            break;
+        *output++ = ((char_to_num[c3] << 6) & 0xc0) | char_to_num[c4];
+        ++len;
+    }
+
+    return len;
+}
+
 BP5Deserializer::ControlInfo *BP5Deserializer::BuildControl(FMFormat Format)
 {
     FMStructDescList FormatList = format_list_of_FMFormat(Format);
@@ -312,6 +366,9 @@ BP5Deserializer::ControlInfo *BP5Deserializer::BuildControl(FMFormat Format)
     size_t VarIndex = 0;
     while (FieldList[i].field_name)
     {
+        size_t HeaderSkip;
+        char *ExprStr = NULL;
+        int Derived = 0;
         ret = (ControlInfo *)realloc(ret, sizeof(*ret) + ControlCount * sizeof(struct ControlInfo));
         struct ControlStruct *C = &(ret->Controls[ControlCount]);
         ControlCount++;
@@ -336,6 +393,29 @@ BP5Deserializer::ControlInfo *BP5Deserializer::BuildControl(FMFormat Format)
             C->OrigShapeID = ShapeID::LocalArray;
             break;
         }
+        if (FieldList[i].field_name[3] == '_')
+        {
+            HeaderSkip = 4;
+        }
+        else if (FieldList[i].field_name[3] == '-')
+        {
+            // Expression follows
+            Derived = 1;
+            int EncLen;
+            int NumberLen;
+            if (sscanf(&FieldList[i].field_name[4], "%d%n", &EncLen, &NumberLen) == 1)
+            { // Expression
+                ExprStr = (char *)malloc(EncLen + 1);
+                const char *Dash = strchr(&FieldList[i].field_name[4], '-');
+                base64_decode((unsigned char *)Dash + 1, (unsigned char *)ExprStr);
+                HeaderSkip = 6 + NumberLen + EncLen;
+            }
+            else
+            {
+                fprintf(stderr, "Bad Expression spec in field %s\n", FieldList[i].field_name);
+            }
+        }
+        //
         BP5VarRec *VarRec = nullptr;
         if (NameIndicatesArray(FieldList[i].field_name))
         {
@@ -356,8 +436,8 @@ BP5Deserializer::ControlInfo *BP5Deserializer::BuildControl(FMFormat Format)
             else
             {
                 BreakdownFieldType(FieldList[i].field_type, Operator, MinMax);
-                BreakdownArrayName(FieldList[i].field_name, &ArrayName, &Type, &ElementSize,
-                                   &StructFormat);
+                BreakdownArrayName(FieldList[i].field_name + HeaderSkip, &ArrayName, &Type,
+                                   &ElementSize, &StructFormat);
             }
             VarRec = LookupVarByName(ArrayName);
             if (!VarRec)
@@ -366,6 +446,8 @@ BP5Deserializer::ControlInfo *BP5Deserializer::BuildControl(FMFormat Format)
                 VarRec->Type = Type;
                 VarRec->ElementSize = ElementSize;
                 VarRec->OrigShapeID = C->OrigShapeID;
+                VarRec->Derived = Derived;
+                VarRec->ExprStr = ExprStr;
                 if (StructFormat)
                 {
                     core::StructDefinition *Def =
@@ -1501,6 +1583,8 @@ BP5Deserializer::GenerateReadRequests(const bool doAllocTempBuffers, size_t *max
                     RR.Timestep = Req->Step;
                     RR.WriterRank = WriterRank;
                     RR.StartOffset = writer_meta_base->DataBlockLocation[NeededBlock];
+                    if (RR.StartOffset == (size_t)-1)
+                        throw std::runtime_error("No data exists for this variable");
                     if (Req->MemSpace != MemorySpace::Host)
                         RR.DirectToAppMemory = false;
                     else
@@ -1574,6 +1658,8 @@ BP5Deserializer::GenerateReadRequests(const bool doAllocTempBuffers, size_t *max
                             RR.StartOffset = writer_meta_base->DataBlockLocation[Block];
                             RR.ReadLength = writer_meta_base->DataBlockSize[Block];
                             RR.DestinationAddr = nullptr;
+                            if (RR.StartOffset == (size_t)-1)
+                                throw std::runtime_error("No data exists for this variable");
                             if (doAllocTempBuffers)
                             {
                                 RR.DestinationAddr = (char *)malloc(RR.ReadLength);
@@ -1611,6 +1697,8 @@ BP5Deserializer::GenerateReadRequests(const bool doAllocTempBuffers, size_t *max
                             RR.WriterRank = WriterRank;
                             RR.StartOffset =
                                 writer_meta_base->DataBlockLocation[Block] + StartOffsetInBlock;
+                            if (writer_meta_base->DataBlockLocation[Block] == (size_t)-1)
+                                throw std::runtime_error("No data exists for this variable");
                             RR.ReadLength = EndOffsetInBlock - StartOffsetInBlock;
                             if (Req->MemSpace != MemorySpace::Host)
                                 RR.DirectToAppMemory = false;
diff --git a/source/adios2/toolkit/format/bp5/BP5Deserializer.h b/source/adios2/toolkit/format/bp5/BP5Deserializer.h
index 3c6aefa35..c2d6f3956 100644
--- a/source/adios2/toolkit/format/bp5/BP5Deserializer.h
+++ b/source/adios2/toolkit/format/bp5/BP5Deserializer.h
@@ -120,6 +120,8 @@ private:
         size_t JoinedDimen = SIZE_MAX;
         size_t *LastJoinedOffset = NULL;
         size_t *LastJoinedShape = NULL;
+        bool Derived = false;
+        char *ExprStr = NULL;
         ShapeID OrigShapeID;
         core::StructDefinition *Def = nullptr;
         core::StructDefinition *ReaderDef = nullptr;
diff --git a/source/adios2/toolkit/format/bp5/BP5Serializer.cpp b/source/adios2/toolkit/format/bp5/BP5Serializer.cpp
index 4bd014017..60d96f642 100644
--- a/source/adios2/toolkit/format/bp5/BP5Serializer.cpp
+++ b/source/adios2/toolkit/format/bp5/BP5Serializer.cpp
@@ -10,6 +10,9 @@
 #include "adios2/core/Engine.h"
 #include "adios2/core/IO.h"
 #include "adios2/core/VariableBase.h"
+#ifdef ADIOS2_HAVE_DERIVED
+#include "adios2/core/VariableDerived.h"
+#endif
 #include "adios2/helper/adiosFunctions.h"
 #include "adios2/toolkit/format/buffer/ffs/BufferFFS.h"

@@ -234,25 +237,104 @@ char *BP5Serializer::BuildVarName(const char *base_name, const ShapeID Shape, co
     return Ret;
 }

+/*
+ * Do base64 encoding of binary buffer, returning a malloc'd string
+ */
+static const char num_to_char[] =
+    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
+
+static char *base64_encode(const char *buffer, unsigned int len)
+{
+    char *buf;
+    int buflen = 0;
+    int c1, c2, c3;
+    int maxlen = len * 4 / 3 + 4;
+#ifdef OVERKILL
+    maxlen = len * 2 + 2;
+#endif
+
+    buf = (char *)malloc(maxlen * sizeof(char));
+    if (buf == NULL)
+    {
+        return NULL;
+    }
+    else
+    {
+        memset(buf, 0, maxlen * sizeof(char));
+    }
+
+    while (len)
+    {
+
+        c1 = (unsigned char)*buffer++;
+        buf[buflen++] = num_to_char[c1 >> 2];
+
+        if (--len == 0)
+            c2 = 0;
+        else
+            c2 = (unsigned char)*buffer++;
+        buf[buflen++] = num_to_char[((c1 & 0x3) << 4) | ((c2 & 0xf0) >> 4)];
+
+        if (len == 0)
+        {
+            buf[buflen++] = '=';
+            buf[buflen++] = '=';
+            break;
+        }
+
+        if (--len == 0)
+            c3 = 0;
+        else
+            c3 = (unsigned char)*buffer++;
+
+        buf[buflen++] = num_to_char[((c2 & 0xf) << 2) | ((c3 & 0xc0) >> 6)];
+        if (len == 0)
+        {
+            buf[buflen++] = '=';
+
+            break;
+        }
+
+        --len;
+        buf[buflen++] = num_to_char[c3 & 0x3f];
+    }
+
+    buf[buflen] = 0;
+
+    return buf;
+}
 static char *BuildLongName(const char *base_name, const ShapeID Shape, const int type,
-                           const size_t element_size, const char *StructID)
+                           const size_t element_size, const char *StructID, const char *ExprStr)
 {
     const char *Prefix = NamePrefix(Shape);
     size_t StructIDLen = 0;
+    size_t ExprLen = 0;
+    char *ExpressionInsert = (char *)"_";
     if (StructID)
         StructIDLen = strlen(StructID);
-    size_t Len = strlen(base_name) + 3 + strlen(Prefix) + StructIDLen + 16;
+    if (ExprStr)
+    {
+        char *ExprEnc = base64_encode(ExprStr, (int)(strlen(ExprStr) + 1));
+        ExprLen = strlen(ExprEnc);
+        ExpressionInsert = (char *)malloc(ExprLen + 16); // str + enough for len and separators
+        snprintf(ExpressionInsert, ExprLen + 16, "-%zu-%s-", ExprLen, ExprEnc);
+        free(ExprEnc);
+    }
+    size_t Len = strlen(base_name) + 3 + ExprLen + strlen(Prefix) + StructIDLen + 16;
     char *Ret = (char *)malloc(Len);
     if (StructID)
     {
-        snprintf(Ret, Len, "%s_%zd_%d_%s", Prefix, element_size, type, StructID);
+        snprintf(Ret, Len, "%s%s%zd_%d_%s", Prefix, ExpressionInsert, element_size, type, StructID);
     }
     else
     {
-        snprintf(Ret, Len, "%s_%zd_%d", Prefix, element_size, type);
+        snprintf(Ret, Len, "%s%s%zd_%d", Prefix, ExpressionInsert, element_size, type);
     }
     strcat(Ret, "_");
     strcat(Ret, base_name);
+    if (ExprStr)
+        free(ExpressionInsert);
     return Ret;
 }
@@ -420,6 +502,9 @@ BP5Serializer::BP5WriterRec BP5Serializer::CreateWriterRec(void *Variable, const
                                                            size_t DimCount)
 {
     core::VariableBase *VB = static_cast<core::VariableBase *>(Variable);
+#ifdef ADIOS2_HAVE_DERIVED
+    core::VariableDerived *VD = dynamic_cast<core::VariableDerived *>(VB);
+#endif
     auto obj = Info.RecMap.insert(std::make_pair(Variable, _BP5WriterRec()));
     BP5WriterRec Rec = &obj.first->second;
     if (Type == DataType::String)
@@ -497,7 +582,12 @@ BP5Serializer::BP5WriterRec BP5Serializer::CreateWriterRec(void *Variable, const
         }
         // Array field.  To Metadata, add FMFields for DimCount, Shape, Count
         // and Offsets matching _MetaArrayRec
-        char *LongName = BuildLongName(Name, VB->m_ShapeID, (int)Type, ElemSize, TextStructID);
+        const char *ExprString = NULL;
+#ifdef ADIOS2_HAVE_DERIVED
+        ExprString = VD ? VD->m_Expr.ExprString.c_str() : NULL;
+#endif
+        char *LongName =
+            BuildLongName(Name, VB->m_ShapeID, (int)Type, ElemSize, TextStructID, ExprString);

         const char *ArrayTypeName = "MetaArray";
         int FieldSize = sizeof(MetaArrayRec);
@@ -643,7 +733,18 @@ void BP5Serializer::Marshal(void *Variable, const char *Name, const DataType Typ
     };

     core::VariableBase *VB = static_cast<core::VariableBase *>(Variable);
+#ifdef ADIOS2_HAVE_DERIVED
+    core::VariableDerived *VD = dynamic_cast<core::VariableDerived *>(VB);
+#endif

+    bool WriteData = true;
+#ifdef ADIOS2_HAVE_DERIVED
+    if (VD)
+    {
+        // All other types of Derived types we don't write data
+        WriteData = (VD->GetDerivedType() == DerivedVarType::StoreData);
+    }
+#endif
     BP5MetadataInfoStruct *MBase;

     BP5WriterRec Rec = LookupWriterRec(Variable);
@@ -714,7 +815,12 @@ void BP5Serializer::Marshal(void *Variable, const char *Name, const DataType Typ

         MinMaxStruct MinMax;
         MinMax.Init(Type);
-        if ((m_StatsLevel > 0) && !Span)
+        bool DerivedWithoutStats = false;
+#ifdef ADIOS2_HAVE_DERIVED
+        DerivedWithoutStats = VD && (VD->GetDerivedType() == DerivedVarType::ExpressionString);
+#endif
+        bool DoMinMax = ((m_StatsLevel > 0) && !DerivedWithoutStats);
+        if (DoMinMax && !Span)
         {
             GetMinMax(Data, ElemCount, (DataType)Rec->Type, MinMax, MemSpace);
         }
@@ -744,6 +850,11 @@ void BP5Serializer::Marshal(void *Variable, const char *Name, const DataType Typ
                     VB->m_Operations[0]->GetHeaderSize(), MemSpace);
             CurDataBuffer->DownsizeLastAlloc(AllocSize, CompressedSize);
         }
+        else if (!WriteData)
+        {
+            DataOffset = (size_t)-1;
+            DeferAddToVec = false;
+        }
         else if (Span == nullptr)
         {
             if (!DeferAddToVec)
@@ -781,7 +892,7 @@ void BP5Serializer::Marshal(void *Variable, const char *Name, const DataType Typ
                 MetaEntry->Offsets = CopyDims(DimCount, Offsets);
             else
                 MetaEntry->Offsets = NULL;
-            if (m_StatsLevel > 0)
+            if (DoMinMax)
             {
                 void **MMPtrLoc = (void **)(((char *)MetaEntry) + Rec->MinMaxOffset);
                 *MMPtrLoc = (void *)malloc(ElemSize * 2);
@@ -823,7 +934,7 @@ void BP5Serializer::Marshal(void *Variable, const char *Name, const DataType Typ
                     (size_t *)realloc(OpEntry->DataBlockSize, OpEntry->BlockCount * sizeof(size_t));
                 OpEntry->DataBlockSize[OpEntry->BlockCount - 1] = CompressedSize;
             }
-            if (m_StatsLevel > 0)
+            if (DoMinMax)
             {
                 void **MMPtrLoc = (void **)(((char *)MetaEntry) + Rec->MinMaxOffset);
                 *MMPtrLoc = (void *)realloc(*MMPtrLoc, MetaEntry->BlockCount * ElemSize * 2);
@@ -854,6 +965,18 @@ void BP5Serializer::Marshal(void *Variable, const char *Name, const DataType Typ
     }
 }

+const void *BP5Serializer::SearchDeferredBlocks(size_t MetaOffset, size_t BlockID)
+{
+    for (auto &Def : DeferredExterns)
+    {
+        if ((Def.MetaOffset == MetaOffset) && (Def.BlockID == BlockID))
+        {
+            return Def.Data;
+        }
+    }
+    return NULL;
+}
+
 MinVarInfo *BP5Serializer::MinBlocksInfo(const core::VariableBase &Var)
 {
     BP5WriterRec VarRec = LookupWriterRec((void *)&Var);
@@ -912,8 +1035,10 @@ MinVarInfo *BP5Serializer::MinBlocksInfo(const core::VariableBase &Var)
             }
             else
             {
-                Blk.BufferP = CurDataBuffer->GetPtr(MetaEntry->DataBlockLocation[b] -
-                                                    m_PriorDataBufferSizeTotal);
+                Blk.BufferP = (void *)SearchDeferredBlocks(VarRec->MetaOffset, b);
+                if (!Blk.BufferP)
+                    Blk.BufferP = CurDataBuffer->GetPtr(MetaEntry->DataBlockLocation[b] -
+                                                        m_PriorDataBufferSizeTotal);
             }
             MV->BlocksInfo.push_back(Blk);
         }
diff --git a/source/adios2/toolkit/format/bp5/BP5Serializer.h b/source/adios2/toolkit/format/bp5/BP5Serializer.h
index 94025f1b2..d07aa727f 100644
--- a/source/adios2/toolkit/format/bp5/BP5Serializer.h
+++ b/source/adios2/toolkit/format/bp5/BP5Serializer.h
@@ -244,6 +244,9 @@ private:
         size_t ElemCount;
         void *Array;
     } ArrayRec;
+
+private:
+    const void *SearchDeferredBlocks(size_t MetaOffset, size_t blocknum);
 };

 } // end namespace format
```

## Unit test

All the files in testing/derived
