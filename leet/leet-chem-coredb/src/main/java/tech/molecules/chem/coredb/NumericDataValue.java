package tech.molecules.chem.coredb;

public class NumericDataValue implements DataValue {

    private double val;

    public NumericDataValue(double val) {
        this.val = val;
    }

    @Override
    public double getAsDouble() {
        return 0;
    }

    @Override
    public String getAsText() {
        return ""+this.val;
    }
}
