package tech.molecules.leet.chem.descriptor.featurepair;

import com.actelion.research.chem.StereoMolecule;

import java.util.List;

public interface FeatureHandler<F extends MolFeature> {
    public List<F> detectFeatures(StereoMolecule m);

    /**
     *
     * @param a
     * @param b
     * @return
     */
    public double computeFeatureSimilarity(F a, F b);

    public double computeFeatureImportance(F a);

}
