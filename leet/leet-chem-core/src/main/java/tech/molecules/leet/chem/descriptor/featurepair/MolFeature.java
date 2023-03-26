package tech.molecules.leet.chem.descriptor.featurepair;

public interface MolFeature {

    //public double similarity(FeaturePairDescriptor.Feature f);
    public int getCentralAtom();
    public int[] getCoveredAtoms();


}
