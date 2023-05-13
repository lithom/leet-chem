package tech.molecules.leet.datatable.column;

import tech.molecules.leet.datatable.AbstractNumericDatasource;
import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.NumericDatasource;

import java.util.ArrayList;
import java.util.List;

public class NumericColumn extends AbstractDataTableColumn<Double,Double> {

    public NumericColumn() {
        super(Double.class);
    }

    public NumericColumn(DataProvider<Double> dp) {
        super(Double.class, dp);
    }

    @Override
    public List<NumericDatasource> getNumericDatasources() {
        NumericDatasource<Double> nd = new AbstractNumericDatasource<Double>("Column",this) {
            @Override
            public Double evaluate(Double original) {
                return original;
            }
        };
        List<NumericDatasource> nds = new ArrayList<>();
        nds.add(nd);
        return nds;
    }

    @Override
    public Double processData(Double data) {
        return data;
    }

}
