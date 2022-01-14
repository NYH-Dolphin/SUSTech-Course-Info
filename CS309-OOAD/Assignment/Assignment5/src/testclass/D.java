package testclass;


import dependency_injection.Value;

public class D {
    @Value("d.val")
    private int val;

    public int getVal() {
        return val;
    }
}