package tech.molecules.chem.coredb.aggregation;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

public class AssayResultFilterInfo {

    @JsonPropertyDescription("min coredb assayresult quality")
    @JsonProperty("minQuality")
    private int minQuality = -1;

    public AssayResultFilterInfo() {}

    public AssayResultFilterInfo(int minQuality) {
        this.minQuality = minQuality;
    }

    public int getMinQuality() {
        return this.minQuality;
    }

    public void setMinQuality(int minQuality) { this.minQuality = minQuality; }

}
