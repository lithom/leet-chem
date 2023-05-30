package tech.molecules.chem.coredb;

public class NumericDataValue implements DataValue {

    private double val;

    public NumericDataValue(double val) {
        this.val = val;
    }

    @Override
    public double getAsDouble() {
        return this.val;
    }

    @Override
    public String getAsText() {
        return ""+this.val;
    }
}
