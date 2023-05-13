package tech.molecules.leet.datatable;

import java.util.DoubleSummaryStatistics;

public interface NumericDatasource<U> extends DataRepresentation<U,Double> {
    public DataTableColumn<?,U> getColumn();
    public boolean hasValue(String row);
    //public double getValue(String row);

}