package tech.molecules.leet.datatable.column;

import tech.molecules.leet.data.NumericArray;
import tech.molecules.leet.datatable.AbstractNumericDatasource;
import tech.molecules.leet.datatable.NumericDatasource;
import tech.molecules.leet.datatable.numeric.AggregatedNumericArray;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class NumericArrayColumn extends AbstractDataTableColumn<NumericArray, AggregatedNumericArray> implements Serializable {

    // Explicit serialVersionUID for interoperability
    private static final long serialVersionUID = 1L;

    public NumericArrayColumn() {
        super(AggregatedNumericArray.class);
    }

    @Override
    public AggregatedNumericArray processData(NumericArray data) {
        return new AggregatedNumericArray(data);
    }

    @Override
    public List<NumericDatasource> getNumericDatasources() {
        List<NumericDatasource> datasources = new ArrayList<>();
        NumericDatasource mean = new AbstractNumericDatasource<AggregatedNumericArray>("Mean",this) {
            @Override
            public Double evaluate(AggregatedNumericArray original) {
                if(original == null) {return null;}
                return original.getMean();
            }
        };
        datasources.add(mean);
        return datasources;
    }
}
