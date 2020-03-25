/*
 * =====================================================================================
 *
 *       Filename:  singleton.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2020年01月04日 19时53分59秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  yipeng
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef MOON_UTILITY_SINGLETON_HPP_
#define MOON_UTILITY_SINGLETON_HPP_

//单例模式 懒汉式
namespace moon {
namespace utility {

template <typename T>
class Singleton {
public:
    //T为类
    typedef T ObjectType;

    static ObjectType& Instance() {
        //这里就调用了真正做事的类的构造
        static ObjectType obj;
        create_object.DoNothing();

        return obj;
    }
    
private:
    Singleton() = default;
    ~Singleton() = default;
    Singleton(const Singleton&) = delete;
    Singleton& operator =(const Singleton&) = delete;
    Singleton(Singleton&&) = delete;
    Singleton& operator =(Singleton&&) = delete;

    struct CreateObject {
        CreateObject() {
            Singleton<T>::Instance();
        }
        inline void DoNothing() const {}
    };
    static CreateObject create_object;
};

//初始化静态成员 是内部类成员 typename声明这个是个类型而不是对象
template <typename T>
typename Singleton<T>::CreateObject Singleton<T>::create_object;

}        //namespace utility
}        //namespace moon

#endif   //MOON_UTILITY_SINGLETON_HPP_
