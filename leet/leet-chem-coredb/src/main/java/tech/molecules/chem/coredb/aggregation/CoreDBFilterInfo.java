package tech.molecules.chem.coredb.aggregation;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class CoreDBFilterInfo {

    public static class AttrFilterInfo {
        @JsonPropertyDescription("attribute name to be filtered")
        @JsonProperty("attr")
        public final String attr;
        @JsonPropertyDescription("accepted values for attribute")
        @JsonProperty("values")
        public final List<String> values;
        public AttrFilterInfo(String attr, List<String> values) {
            this.attr = attr;
            this.values = values;
        }
    }
    @JsonPropertyDescription("attribute filters, all attributes not in this list are not considered for filtering")
    @JsonProperty("attrFilters")
    private List<AttrFilterInfo> attrFilters;

    public List<AttrFilterInfo> getAttrFilters() {
        return this.attrFilters;
    }

    public CoreDBFilterInfo() {
        this(new ArrayList<>());
    }

    public CoreDBFilterInfo(List<AttrFilterInfo> attrFilters) {
        this.attrFilters = attrFilters;
    }

    public Map<String, AttrFilterInfo> getAttrFiltersSorted() {
        Map<String, AttrFilterInfo> fs = new HashMap<>();
        for(AttrFilterInfo ati : this.attrFilters) {
            fs.put(ati.attr,ati);
        }
        return fs;
    }

    public void CoreDBFilterInfo(String attrName, AttrFilterInfo fi_new) {
        if(this.getAttrFiltersSorted().containsKey(attrName)) {
            this.attrFilters.removeAll( this.attrFilters.stream().filter( ai -> ai.attr.equals(attrName) ).collect(Collectors.toList()));
        }
        this.attrFilters.add(fi_new);
    }

}
