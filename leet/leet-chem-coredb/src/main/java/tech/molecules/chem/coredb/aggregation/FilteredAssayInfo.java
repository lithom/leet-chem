package tech.molecules.chem.coredb.aggregation;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

public class FilteredAssayInfo {

    @JsonPropertyDescription("assay id")
    @JsonProperty("assayID")
    public Integer assayID;

    @JsonPropertyDescription("filter based on input parameters")
    @JsonProperty("osirisFilter")
    public CoreDBFilterInfo coreDBFilter;

    @JsonPropertyDescription("filter based on assay result properties (e.g. quality))")
    @JsonProperty("assayResultFilter")
    public AssayResultFilterInfo assayResultFilter;

}
