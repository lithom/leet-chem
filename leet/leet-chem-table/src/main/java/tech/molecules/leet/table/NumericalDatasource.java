package tech.molecules.leet.table;

public interface NumericalDatasource<U> {
    public String getName();
    public NColumn<U,?> getColumn();
    public boolean hasValue(U dp, String row);
    public double getValue(U dp, String row);
}
