package tech.molecules.chem.coredb.aggregation;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

public class NumericValueInfo {

    @JsonPropertyDescription("Assay")
    @JsonProperty("Assay")
    private FilteredAssayInfo assay;

    @JsonPropertyDescription("OutValue, i.e. the value that we are interested in")
    @JsonProperty("outValue")
    private String outValue;


    @JsonPropertyDescription("Aggregation info")
    @JsonProperty("aggregation")
    private NumericAggregationInfo aggregation = new NumericAggregationInfo(NumericAggregationInfo.AGGREGATION_MEAN);
    @JsonPropertyDescription("Handle assay scores logarithmically?")
    @JsonProperty("handleLogarithmic")
    private boolean handleLogarithmic = false;


    //@JsonPropertyDescription("Scoring description")
    //@JsonProperty("Score")
    //private NumericScore score = null;

    @JsonPropertyDescription("Manually set lower bound (NaN means not set)")
    @JsonProperty("fixedLB")
    private double fixedLB = Double.NaN;

    @JsonPropertyDescription("Manually set upper bound (NaN means not set)")
    @JsonProperty("fixedUB")
    private double fixedUB = Double.NaN;



}
