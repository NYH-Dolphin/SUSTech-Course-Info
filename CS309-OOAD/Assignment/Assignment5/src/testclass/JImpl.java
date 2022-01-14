package testclass;

import dependency_injection.Inject;
import dependency_injection.Value;

public class JImpl implements J {
    private E eDep;

    @Value(value = "j.integers", delimiter = "-", min = 5)
    private int integer;

    private String string;

    @Inject
    public JImpl(E eDep) {
        this.eDep = eDep;
    }

    @Override
    public E getEDep() {
        return eDep;
    }

    @Override
    public int getInt() {
        return integer;
    }

    @Override
    public String getString() {
        return string;
    }
}
