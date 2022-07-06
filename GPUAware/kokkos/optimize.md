```c++
namespace detail
{
template <typename... Ts>
struct void_t
{
};

template <typename C, typename = void>
struct is_container : std::false_type
{
};

template <typename C>
struct is_container<
    C, typename std::conditional<
           false,
           void_t<typename C::value_type, typename C::size_type,
                  decltype(const_cast<const typename C::value_type *>(
                      std::declval<C>().data())),
                  decltype(std::declval<C>().size())>,
           void>::type> : public std::true_type
{
};

}
```


```c++
 /**
     * Put data associated with a container-like thing, e.g., a Kokkos::View
     */
    template <
        class T, class C,
        typename = typename std::enable_if<
            detail::is_container<C>::value &&
            std::is_same<typename std::remove_cv<typename C::value_type>::type,
                         T>::value>::type>
    void Put(Variable<T> variable, const C &container,
             const Mode launch = Mode::Deferred)
    {
        Put(variable, const_cast<const T *>(container.data()), launch);
    }
```
