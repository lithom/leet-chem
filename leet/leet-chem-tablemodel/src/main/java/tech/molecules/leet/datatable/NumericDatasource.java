package tech.molecules.leet.datatable;

public interface NumericDatasource<U> extends DataRepresentation<U,Double> {
    public DataTableColumn<?,U> getColumn();
    public boolean hasValue(String row);
    //public double getValue(String row);
}