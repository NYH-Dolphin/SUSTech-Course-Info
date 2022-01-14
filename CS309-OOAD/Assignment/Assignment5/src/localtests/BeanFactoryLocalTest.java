package localtests;

import dependency_injection.BeanFactory;
import dependency_injection.BeanFactoryImpl;
import testclass.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.*;

import static org.junit.jupiter.api.Assertions.*;

public class BeanFactoryLocalTest {

    private BeanFactory beanFactory;

    @BeforeEach
    public void setup() {
        this.beanFactory = new BeanFactoryImpl();
        beanFactory.loadInjectProperties(new File("local-inject.properties"));
        beanFactory.loadValueProperties(new File("local-value.properties"));
//        System.setOut(new PrintStream(OutputStream.nullOutputStream()));
//        System.setErr(new PrintStream(OutputStream.nullOutputStream()));
    }

    @Test
    public void testResolveWithNoDependency() {
        A instance = beanFactory.createInstance(A.class);
        assertNotNull(instance);
    }

    @Test
    public void testWithConstructorDependency() {
        B instance = beanFactory.createInstance(B.class);
        assertNotNull(instance);
        assertNotNull(instance.getCDep());
        assertNotNull(instance.getDDep());
    }

    @Test
    public void testWithAnnotationDependency() {
        G instance = beanFactory.createInstance(G.class);
        assertNotNull(instance);
        assertNotNull(instance.getCDep());
        assertNotNull(instance.getDDep());
    }

    @Test
    public void testWithMethodDependency() {
        H instance = beanFactory.createInstance(H.class);
        assertNotNull(instance);
        assertNotNull(instance.getCDep());
        assertNotNull(instance.getDDep());
    }

    @Test
    public void testMixedDependencies() {
        I instance = beanFactory.createInstance(I.class);
        assertNotNull(instance);
        assertNotNull(instance.getADep());
        assertNotNull(instance.getCDep());
        assertNotNull(instance.getDDep());
    }

    @Test
    public void testImplTypeForInterface() {
        E instance = beanFactory.createInstance(E.class);
        assertNotNull(instance);
        assertTrue(instance instanceof EImpl);
    }

    @Test
    public void testImplTypeForAbstractClass() {
        F instance = beanFactory.createInstance(F.class);
        assertNotNull(instance);
        assertTrue(instance instanceof FEnhanced);
    }

    @Test
    public void testDependencyImpl() {
        K instance = beanFactory.createInstance(K.class);
        assertNotNull(instance);
        assertNotNull(instance.getEDep());
        assertNotNull(instance.getFDep());
        assertTrue(instance.getEDep() instanceof EImpl);
        assertTrue(instance.getFDep() instanceof FEnhanced);
    }

    @Test
    public void testConstructorInject() {
        L instance = beanFactory.createInstance(L.class);
        assertNotNull(instance);
        assertNotNull(instance.getBDep());
        assertTrue(instance.isBool());
    }

    @Test
    public void testFieldWithValue() {
        D instance = beanFactory.createInstance(D.class);
        assertEquals(10, instance.getVal());
    }

    @Test
    public void testPrimitiveArrayValues() {
        J instance = beanFactory.createInstance(J.class);
        assertNotNull(instance);
        assertTrue(instance instanceof JImpl);
    }

    @Test
    public void testWrappedPrimitiveArrayValues() {
        J instance = beanFactory.createInstance(J.class);
        assertNotNull(instance);
        assertTrue(instance instanceof JImpl);
        assertEquals(8, instance.getInt());
    }

    @Test
    public void testStringArrayValues() {
        J instance = beanFactory.createInstance(J.class);
        assertNotNull(instance);
        assertTrue(instance instanceof JImpl);
    }


    @Test
    public void testDefaultString(){
        M instance = beanFactory.createInstance(M.class);
        assertEquals(instance.getStr(), "default value");
    }


    @Test
    public void testByteValue(){
        N instance = beanFactory.createInstance(N.class);
        System.out.println(1);
    }
}
