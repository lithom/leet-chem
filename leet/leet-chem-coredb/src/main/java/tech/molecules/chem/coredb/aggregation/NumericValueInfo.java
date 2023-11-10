package tech.molecules.chem.coredb.aggregation;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

public class NumericValueInfo {

    @JsonPropertyDescription("Assay")
    @JsonProperty("Assay")
    private FilteredAssayInfo assay;

    //@JsonPropertyDescription("OutValue, i.e. the value that we are interested in")
    //@JsonProperty("outValue")
    //private String outValue;

    @JsonPropertyDescription("Aggregation info")
    @JsonProperty("aggregation")
    private NumericAggregationInfo aggregation; // = new NumericAggregationInfo("",NumericAggregationInfo.AGGREGATION_MEAN);

    public NumericValueInfo(FilteredAssayInfo assay, NumericAggregationInfo aggregation) {
        this.assay = assay;
        this.aggregation = aggregation;
    }

    public FilteredAssayInfo getAssay() {
        return assay;
    }

    public void setAssay(FilteredAssayInfo assay) {
        this.assay = assay;
    }

    public NumericAggregationInfo getAggregation() {
        return aggregation;
    }

    public void setAggregation(NumericAggregationInfo aggregation) {
        this.aggregation = aggregation;
    }
}
