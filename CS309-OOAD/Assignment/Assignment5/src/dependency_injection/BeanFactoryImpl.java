package dependency_injection;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.*;
import java.util.Properties;

/**
 * TODO you should complete the class
 */
public class BeanFactoryImpl implements BeanFactory {

    public static Properties injectProperties = new Properties();
    public static Properties valueProperties = new Properties();

    @Override
    public void loadInjectProperties(File file) {
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
            injectProperties.load(bufferedReader);
        } catch (IOException ignored) {
        }
    }

    @Override
    public void loadValueProperties(File file) {
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
            valueProperties.load(bufferedReader);
        } catch (IOException ignored) {
        }
    }

    @Override
    public <T> T createInstance(Class<T> clazz) {
        // 1. 找实现类 implClazz
        Class<T> implClazz = clazz;
        if (BeanFactoryImpl.injectProperties.containsKey(clazz.getName())) {
            String implClazzName = BeanFactoryImpl.injectProperties.getProperty(clazz.getName());
            try {
                implClazz = (Class<T>) Class.forName(implClazzName);
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }
        }

        // 2. 通过实现类找 Constructor
        //    - 找带有 Inject 注解的
        //    - getConstructor();
        Constructor<T> constructor = null;
        for (Constructor c : implClazz.getDeclaredConstructors()) {
            if (c.getAnnotation(Inject.class) != null) {
                constructor = c;
                break;
            }
        }
        if (constructor == null) {
            try {
                constructor = implClazz.getConstructor();
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            }
        }

        // 3. 通过 constructor，找constructor中的所有parameters
        assert constructor != null;
        Parameter[] parameters = constructor.getParameters();

        // 4. 创建一个 Object[] 为了存放每个 Parameter 的实例
        Object[] objects = new Object[parameters.length];

        // 5. 遍历每个 Parameters，分别创建实例放入 Object[] 中
        //    - 如果是用户自定义类，递归调用，类似于createInstance方法
        //    - 如果不是，那么一定带有value注解，通过value注解注入
        //    - if(p.getType() == int.class)
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].getAnnotation(Value.class) == null) {
                objects[i] = createInstance(parameters[i].getType());
            } else {
                Class parameterClass = parameters[i].getType();
                Value keyAnnotation = parameters[i].getAnnotation(Value.class);
                objects[i] = injectPrimitiveParameters(parameterClass, keyAnnotation);
            }
        }

        T instance = null;
        // 6. 根据 Object[] 以及 constructor，创建当前 implClazz 的实例
        try {
            instance = constructor.newInstance(objects);
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException e) {
            e.printStackTrace();
        }

        // 7. 找当前implClazz的所有 field，遍历每一个 field
        //    - 如果是用户自定义类
        //         - 如果带有Inject注解，递归调用，类似于createInstance方法
        //         - 如果没有注解忽略
        //    - 如果是基本数据类型+String且带有@Value注解，按照value注解注入
        assert instance != null;
        Field[] fields = instance.getClass().getDeclaredFields();
        for (Field field : fields) {
            if (field.getAnnotation(Value.class) != null) {
                Class parameterClass = field.getType();
                Value keyAnnotation = field.getAnnotation(Value.class);
                field.setAccessible(true);
                try {
                    field.set(instance, injectPrimitiveParameters(parameterClass, keyAnnotation));
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
                field.setAccessible(false);
            } else if (field.getAnnotation(Inject.class) != null) {
                field.setAccessible(true);
                try {
                    field.set(instance, createInstance(field.getType()));
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
                field.setAccessible(false);
            }

        }
        // 8. 找所有带有Inject注解的普通方法 set 方法
        // Method method
        // method.invoke(object, xxx)
        // 按照方法5调用
        Method[] methods = instance.getClass().getDeclaredMethods();
        for (Method method : methods) {
            if (method.getAnnotation(Inject.class) != null) {
                Parameter[] methodParameters = method.getParameters();
                Object[] methodObjects = new Object[methodParameters.length];
                for (int i = 0; i < methodObjects.length; i++) {
                    methodObjects[i] = createInstance(methodParameters[i].getType());
                }
                try {
                    method.invoke(instance, methodObjects);
                } catch (IllegalAccessException | InvocationTargetException e) {
                    e.printStackTrace();
                }
            }
        }
        return instance;
    }


    public Object injectPrimitiveParameters(Class parameterClass, Value keyAnnotation) {
        String[] vals = BeanFactoryImpl.valueProperties
                .getProperty(keyAnnotation.value())
                .split(keyAnnotation.delimiter());
        if (parameterClass == byte.class) {
            for (String val : vals) {
                if (Byte.parseByte(val) >= keyAnnotation.min() && Byte.parseByte(val) <= keyAnnotation.max()) {
                    return Byte.parseByte(val);
                }
            }
            return (byte) 0;
        } else if (parameterClass == short.class) {
            for (String val : vals) {
                if (Short.parseShort(val) >= keyAnnotation.min() && Short.parseShort(val) <= keyAnnotation.max()) {
                    return Short.parseShort(val);
                }
            }
            return (short) 0;
        } else if (parameterClass == int.class) {
            for (String val : vals) {
                if (Integer.parseInt(val) >= keyAnnotation.min() && Integer.parseInt(val) <= keyAnnotation.max()) {
                    return Integer.parseInt(val);
                }
            }
            return 0;
        } else if (parameterClass == long.class) {
            for (String val : vals) {
                if (Long.parseLong(val) >= keyAnnotation.min() && Long.parseLong(val) <= keyAnnotation.max()) {
                    return Long.parseLong(val);
                }
            }
            return (long) 0;
        } else if (parameterClass == float.class) {
            for (String val : vals) {
                if (Float.parseFloat(val) >= keyAnnotation.min() && Float.parseFloat(val) <= keyAnnotation.max()) {
                    return Float.parseFloat(val);
                }
            }
            return 0.0f;
        } else if (parameterClass == double.class) {
            for (String val : vals) {
                if (Double.parseDouble(val) >= keyAnnotation.min() && Double.parseDouble(val) <= keyAnnotation.max()) {
                    return Double.parseDouble(val);
                }
            }
            return 0.0;
        } else if (parameterClass == boolean.class) {
            return Boolean.parseBoolean(vals[0]);
        } else if (parameterClass == char.class) {
            return vals[0].charAt(0);
        } else if (parameterClass == String.class) {
            for (String val : vals) {
                if (val.length() >= keyAnnotation.min() && val.length() <= keyAnnotation.max()) {
                    return val;
                }
            }
            return "default value";
        }
        return null;
    }
}
