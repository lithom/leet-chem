package tech.molecules.leet.chem.sar;

import com.actelion.research.chem.StereoMolecule;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

import java.util.List;

public class SimpleSynthonSet {

    @JsonPropertyDescription("synthons of this simple synthon set")
    @JsonProperty("synthons")
    private List<StereoMolecule> synthons;

    public SimpleSynthonSet() {
    }

    public SimpleSynthonSet(List<StereoMolecule> synthons) {
        this.synthons = synthons;
    }

    public List<StereoMolecule> getSynthons() {
        return this.synthons;
    }

    public void setSynthons(List<StereoMolecule> synthons) {
        this.synthons = synthons;
    }

    @Override
    public String toString() {
        return String.format( "SynthonSet (Size=%d)",synthons.size());
    }
}
