package dependency_injection;

import java.io.File;

public interface BeanFactory {
    void loadInjectProperties(File file);

    void loadValueProperties(File file);

    <T> T createInstance(Class<T> clazz);
}
