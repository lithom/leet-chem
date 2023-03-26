package tech.molecules.leet.chem.descriptor.featurepair;

import com.actelion.research.chem.StereoMolecule;
import org.apache.commons.lang3.tuple.Pair;

import java.util.List;

public interface FeaturePairHandler<D extends MolFeatureDistance> {

    public <F extends MolFeature> List<FeaturePair<D,F>> detectPairs(StereoMolecule m, List<F> features);

    public double computeDistanceImportance(D da);

    /**
     *
     * @param da
     * @param db
     * @return similarity (zero to one)
     */
    public double computeDistanceSimilarity(D da, D db);

}
