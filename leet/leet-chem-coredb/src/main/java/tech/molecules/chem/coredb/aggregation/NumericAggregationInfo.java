package tech.molecules.chem.coredb.aggregation;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

public class NumericAggregationInfo {

    public static final String AGGREGATION_MEAN = "mean";

    @JsonPropertyDescription("method used for aggregation to double value (this is applied after result filtering)")
    @JsonProperty("method")
    private String method = AGGREGATION_MEAN;

    public NumericAggregationInfo() {}

    public NumericAggregationInfo(String method) {
        this.method = method;
    }

    public String getMethod() {
        return this.method;
    }

    public void setMethod(String m) {this.method = m;}

}
