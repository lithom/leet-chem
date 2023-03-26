package tech.molecules.leet.chem.descriptor.featurepair;

public class FeaturePair<D extends MolFeatureDistance,F extends MolFeature> {

    public final D d;
    public final F a;
    public final F b;

    public FeaturePair(D d, F a, F b) {
        this.d = d;
        this.a = a;
        this.b = b;
    }

    /**
     * Combines the importances of the two features and the distance by taking the product
     * of all three.
     *
     * @param fh
     * @param fph
     * @return
     */
    public double evaluateImportance(FeatureHandler fh, FeaturePairHandler fph) {
        double wca = fh.computeFeatureImportance(a) * fh.computeFeatureImportance(b);
        double wd  = fph.computeDistanceImportance(d);
        return wca*wd;
    }
}
