package tech.molecules.leet.datatable;

import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;
import java.util.List;

public interface NumericDatasource<U> extends DataRepresentation<U,Double> {
    public DataTableColumn<?,U> getColumn();
    public boolean hasValue(String row);
    //public double getValue(String row);

    default List<Double> getDataVisibleColumns(DataTable table) {
        List<String> rows = table.getVisibleKeysSorted();
        List<Double> data = new ArrayList<>();
        DataTableColumn<?,U> ci = getColumn();
        for(String si : rows) {
            data.add( evaluate( ci.getRawValue(si) ) );
        }
        return data;
    }

}