package tech.molecules.analytics;

import tech.molecules.leet.chem.mutator.FragmentDecompositionSynthon;

import java.util.List;

public class MMPFragmentDecompositionImpl implements MMPFragmentDecomposition {


    private FragmentDecompositionSynthon decompositionSynthon;
    private int assayID;
    private List<Long> assayResultIDs;


    public MMPFragmentDecompositionImpl(FragmentDecompositionSynthon frag,
                                        int assayID,
                                        List<Long> assayResultIds) {
            this.setDecompositionSynthon(frag);
            this.setAssayID(assayID);
            this.setAssayResultIDs(assayResultIds);
    }

    @Override
    public String getFragmentIDCode() {
        return getDecompositionSynthon().getSynthonIDCode();
    }

    @Override
    public String getRemainderIDCode() {
        return getDecompositionSynthon().getRemainderIDCode();
    }


    public FragmentDecompositionSynthon getDecompositionSynthon() {
        return decompositionSynthon;
    }

    public void setDecompositionSynthon(FragmentDecompositionSynthon decompositionSynthon) {
        this.decompositionSynthon = decompositionSynthon;
    }

    public int getAssayID() {
        return assayID;
    }

    public void setAssayID(int assayID) {
        this.assayID = assayID;
    }

    public List<Long> getAssayResultIDs() {
        return assayResultIDs;
    }

    public void setAssayResultIDs(List<Long> assayResultIDs) {
        this.assayResultIDs = assayResultIDs;
    }

}
