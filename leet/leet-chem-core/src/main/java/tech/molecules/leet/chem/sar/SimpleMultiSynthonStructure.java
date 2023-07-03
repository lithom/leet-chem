package tech.molecules.leet.chem.sar;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.chemicalspaces.synthon.SynthonReactor;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class SimpleMultiSynthonStructure {

    @JsonPropertyDescription("The different sets of synthons that build the decompositions of this series")
    @JsonProperty("synthonSets")
    private List<SimpleSynthonSet> synthonSets = new ArrayList<>();

    public SimpleMultiSynthonStructure() {
    }

    public SimpleMultiSynthonStructure(StereoMolecule mi) {
        this(Collections.singletonList( new SimpleSynthonSet(Collections.singletonList(mi))));
    }

    public SimpleMultiSynthonStructure(List<SimpleSynthonSet> synthonSets) {
        this.synthonSets = synthonSets;
    }

    public List<StereoMolecule> getAllStructures() {
        if(this.synthonSets.isEmpty()) {
            return new ArrayList<>();
        }
        List<StereoMolecule> assembled = new ArrayList<>();
        List<List<StereoMolecule>> allSets = generateAllSets();
        for(List<StereoMolecule> ssi : allSets) {
            if(ssi.size()==1) {assembled.add(ssi.get(0));}
            else {
                StereoMolecule mi = SynthonReactor.react(ssi);
                mi.ensureHelperArrays(Molecule.cHelperCIP);
                assembled.add(mi);
            }
        }
        return assembled;
    }

    @JsonIgnore
    public List<List<StereoMolecule>> generateAllSets() {
        List<List<StereoMolecule>> allSets = new ArrayList<>();
        generateSetsRecursively(allSets, new ArrayList<>(), 0);
        return allSets;
    }


    private void generateSetsRecursively(List<List<StereoMolecule>> allSets, List<StereoMolecule> currentSet, int synthonSetIndex) {
        if (synthonSetIndex == synthonSets.size()) {
            allSets.add(new ArrayList<>(currentSet));
            return;
        }

        SimpleSynthonSet synthonSet = synthonSets.get(synthonSetIndex);
        List<StereoMolecule> synthonSetMolecules = synthonSet.getSynthons();
        for (StereoMolecule molecule : synthonSetMolecules) {
            currentSet.add(molecule);
            generateSetsRecursively(allSets, currentSet, synthonSetIndex + 1);
            currentSet.remove(currentSet.size() - 1);
        }
    }

    public List<SimpleSynthonSet> getSynthonSets() {
        return synthonSets;
    }

    public void setSynthonSets(List<SimpleSynthonSet> synthonSets) {
        this.synthonSets = synthonSets;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("MultiSynthonStructure: ");
        List<String> synthonSetSizes = this.synthonSets.stream().map(xi -> ""+xi.getSynthons().size()).collect(Collectors.toList());
        sb.append( String.join("x" ,synthonSetSizes) );
        return sb.toString();
    }
}
