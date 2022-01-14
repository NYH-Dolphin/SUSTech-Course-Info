package testclass;

import dependency_injection.Value;

public class N {

    @Value(value = "n.bytes", delimiter = "-", min = 0, max = 100000)
    private byte b;


}
