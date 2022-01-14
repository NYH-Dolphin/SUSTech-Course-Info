package testclass;


import dependency_injection.Inject;

public class K {

    private E eDep;
    private F fDep;

    @Inject
    public K(E eDep, F fDep) {
        this.eDep = eDep;
        this.fDep = fDep;
    }

    public E getEDep() {
        return eDep;
    }

    public F getFDep() {
        return fDep;
    }
}
