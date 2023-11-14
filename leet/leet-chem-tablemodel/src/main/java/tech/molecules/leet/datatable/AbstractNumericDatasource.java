package tech.molecules.leet.datatable;

import java.io.Serializable;
import java.util.DoubleSummaryStatistics;

public abstract class AbstractNumericDatasource<U> implements NumericDatasource<U> , Serializable {

    private String name;
    private DataTableColumn<?,U> col;

    public AbstractNumericDatasource(String name, DataTableColumn<?,U> col) {
        this.name = name;
        this.col = col;
    }

    @Override
    public String getName() {
        return this.name;
    }

    @Override
    public Class<Double> getRepresentationClass() {
        return Double.class;
    }

    @Override
    public DataTableColumn<?, U> getColumn() {
        return this.col;
    }

    @Override
    public boolean hasValue(String row) {
        return col.getValue(row)!=null;
    }

}
