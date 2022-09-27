package tech.molecules.leet.table;

public interface NumericalDatasource<U> {
    public String getName();
    public NColumn<U,?> getColumn();
    public boolean hasValue(String row);
    public double getValue(String row);
}
