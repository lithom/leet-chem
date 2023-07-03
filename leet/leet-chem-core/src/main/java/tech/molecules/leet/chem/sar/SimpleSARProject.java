package tech.molecules.leet.chem.sar;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

import java.util.ArrayList;
import java.util.List;

public class SimpleSARProject {

    @JsonPropertyDescription("List of series that define the different SARs of the project")
    @JsonProperty("series")
    private List<SimpleSARSeries> series = new ArrayList<>();

    public SimpleSARProject() {}

    public SimpleSARProject(List<SimpleSARSeries> series) {
        this.series = series;
    }

    public List<SimpleSARSeries> getSeries() {
        return series;
    }

    public void setSeries(List<SimpleSARSeries> series) {
        this.series = series;
    }

    @Override
    public String toString() {
        return "Project";
    }
}
