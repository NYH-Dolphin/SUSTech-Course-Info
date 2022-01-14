package testclass;


import dependency_injection.Inject;

public class B {

    private C cDep;

    public C getCDep() {
        return cDep;
    }

    public D getDDep() {
        return dDep;
    }

    private D dDep;

    @Inject
    public B(C cDep, D dDep) {
        this.cDep = cDep;
        this.dDep = dDep;
    }
}
