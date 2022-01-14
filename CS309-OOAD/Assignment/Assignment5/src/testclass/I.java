package testclass;


import dependency_injection.Inject;

public class I {
    @Inject
    public I(D dDep) {
        this.dDep = dDep;
    }

    @Inject
    public void setADep(A aDep) {
        this.aDep = aDep;
    }

    public A getADep() {
        return aDep;
    }

    public C getCDep() {
        return cDep;
    }

    public D getDDep() {
        return dDep;
    }

    @Inject
    private A aDep;

    @Inject
    private C cDep;

    private D dDep;


}
