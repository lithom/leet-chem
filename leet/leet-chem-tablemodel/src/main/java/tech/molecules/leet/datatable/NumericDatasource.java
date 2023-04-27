package tech.molecules.leet.datatable;

public interface NumericDatasource<U> {
    public DataTableColumn<?,U> getColumn();
    public boolean hasValue(String row);
    public double getValue(String row);
}