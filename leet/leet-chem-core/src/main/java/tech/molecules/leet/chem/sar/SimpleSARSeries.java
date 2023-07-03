package tech.molecules.leet.chem.sar;

import com.actelion.research.chem.StereoMolecule;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class SimpleSARSeries {

    @JsonPropertyDescription("Series name")
    @JsonProperty("name")
    private String name;

    @JsonPropertyDescription("Multi synthon representation of series")
    @JsonProperty("seriesDecomposition")
    private SimpleMultiSynthonStructure seriesDecomposition;

    public SimpleSARSeries() {
    }

    public SimpleSARSeries(String name, SimpleMultiSynthonStructure seriesDecomposition) {
        this.name = name;
        this.seriesDecomposition = seriesDecomposition;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public SimpleMultiSynthonStructure getSeriesDecomposition() {
        return seriesDecomposition;
    }

    public void setSeriesDecomposition(SimpleMultiSynthonStructure seriesDecomposition) {
        this.seriesDecomposition = seriesDecomposition;
    }

    @JsonIgnore
    public List<String> getLabels() {
        Set<String> labels = new HashSet<>();

        for(StereoMolecule mi : getSeriesDecomposition().getAllStructures()) {
            List<String> labels_i = SimpleSARDecomposition.extractAllLabels(mi);
            labels.addAll(labels_i);
        }
        return new ArrayList<>( labels.stream().sorted().collect(Collectors.toList()) );
    }

    @Override
    public String toString() {
        return "Series: "+this.getName();
    }


}
