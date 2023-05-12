package tech.molecules.leet.datatable.column;

public class NumericColumn extends AbstractDataTableColumn<Double,Double> {

    @Override
    public Double processData(Double data) {
        return data;
    }

}
