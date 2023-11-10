package tech.molecules.chem.coredb.aggregation;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import javafx.css.StyleConverter;

public class NumericAggregationInfo {

    public static final String AGGREGATION_MEAN = "mean";

    @JsonPropertyDescription("parameter name of the assay result values that is considered")
    @JsonProperty("parameterName")
    private String parameterName;

    @JsonPropertyDescription("method used for aggregation to double value (this is applied after result filtering)")
    @JsonProperty("method")
    private String method = AGGREGATION_MEAN;

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


    public NumericAggregationInfo() {}

    public NumericAggregationInfo(String parameterName, String method) {
        this.parameterName = parameterName;
        this.method = method;
    }

    public NumericAggregationInfo(String parameterName, String method, boolean handleLogarithmic, double fixedLB, double fixedUB) {
        this.parameterName = parameterName;
        this.method = method;
        this.handleLogarithmic = handleLogarithmic;
        this.fixedLB = fixedLB;
        this.fixedUB = fixedUB;
    }

    public String getMethod() {
        return this.method;
    }

    public void setMethod(String m) {this.method = m;}

    public String getParameterName() {
        return parameterName;
    }

    public void setParameterName(String parameterName) {
        this.parameterName = parameterName;
    }

    public boolean isHandleLogarithmic() {
        return handleLogarithmic;
    }

    public void setHandleLogarithmic(boolean handleLogarithmic) {
        this.handleLogarithmic = handleLogarithmic;
    }

    public double getFixedLB() {
        return fixedLB;
    }

    public void setFixedLB(double fixedLB) {
        this.fixedLB = fixedLB;
    }

    public double getFixedUB() {
        return fixedUB;
    }

    public void setFixedUB(double fixedUB) {
        this.fixedUB = fixedUB;
    }
}
