package tech.molecules.analytics;

import tech.molecules.leet.chem.mutator.FragmentDecompositionSynthon;

import java.util.List;

public interface MMPFragmentDecomposition {


    public int getAssayID();

    public List<Long> getAssayResultIDs();

    public FragmentDecompositionSynthon getDecompositionSynthon();

    public String getFragmentIDCode();

    public String getRemainderIDCode();

    default public String getDecompositionID() {
        return ( getFragmentIDCode()+"::++::"+getRemainderIDCode() );
    }

}
