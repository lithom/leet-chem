package tech.molecules.analytics;

public interface NumericalMMPInstance extends MMPInstance {

    public String getNumericalAttributeName();
    public void setNumericalAttributeName(String attr);

    public double getMeanA();
    public double getMeanB();

    default public double getShiftAtoB() {
        return getMeanB() - getMeanA();
    }

}
