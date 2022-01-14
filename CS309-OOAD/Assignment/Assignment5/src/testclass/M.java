package testclass;

import dependency_injection.Value;

public class M {
    @Value(value = "m.strings", delimiter = "-", min = 1,max = 2)
    private String str;

    public String getStr() {
        return str;
    }
}
