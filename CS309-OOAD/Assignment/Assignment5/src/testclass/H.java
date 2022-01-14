package testclass;


import dependency_injection.Inject;

public class H {

    @Inject
    public void setCDep(C cDep) {
        this.cDep = cDep;
    }

    @Inject
    public void setDDep(D dDep) {
        this.dDep = dDep;
    }

    public C getCDep() {
        return cDep;
    }

    public D getDDep() {
        return dDep;
    }

    private C cDep;

    private D dDep;


}
