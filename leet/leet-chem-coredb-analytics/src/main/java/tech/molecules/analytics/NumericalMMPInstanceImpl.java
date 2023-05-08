package tech.molecules.analytics;

public class NumericalMMPInstanceImpl extends MMPInstanceImpl implements NumericalMMPInstance {

    private String attribute;

    private double meanA;

    private double meanB;

    public NumericalMMPInstanceImpl(MMPFragmentDecomposition a, MMPFragmentDecomposition b, String attribute, double mean_a, double mean_b) {
        super(a, b);
        this.attribute = attribute;
        this.meanA = mean_a;
        this.meanB = mean_b;
    }

    @Override
    public String getNumericalAttributeName() {
        return this.attribute;
    }

    @Override
    public void setNumericalAttributeName(String attr) {
        this.attribute = attr;
    }

    @Override
    public double getMeanA() {
        return meanA;
    }

    @Override
    public double getMeanB() {
        return meanB;
    }
}
