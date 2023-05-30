package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.DataValue;

public class DataValueImpl implements DataValue {

    private double doubleValue;

    private String textValue;

    public DataValueImpl(double doubleValue, String textValue) {
        this.doubleValue = doubleValue;
        this.textValue = textValue;
    }

    @Override
    public double getAsDouble() {
        return this.doubleValue;
    }

    @Override
    public String getAsText() {
        return this.textValue;
    }
}
