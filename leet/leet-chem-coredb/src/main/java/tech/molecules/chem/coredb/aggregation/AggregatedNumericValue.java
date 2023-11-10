package tech.molecules.chem.coredb.aggregation;

import tech.molecules.chem.coredb.AssayResult;
import tech.molecules.chem.coredb.CoreDB;

import java.util.List;

public class AggregatedNumericValue {

    public final double value;
    public final NumericAggregationInfo aggregationInfo;
    public final List<AssayResult> dataFiltered;
    public final double[] dataNumeric;

    public AggregatedNumericValue(double value, NumericAggregationInfo aggregationInfo, List<AssayResult> dataFiltered, double[] dataNumeric) {
        this.value = value;
        this.aggregationInfo = aggregationInfo;
        this.dataFiltered = dataFiltered;
        this.dataNumeric = dataNumeric;
    }

}
