package tech.molecules.leet.chem.descriptor.featurepair;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.descriptor.DescriptorHandler;
import com.actelion.research.chem.descriptor.DescriptorInfo;
import com.actelion.research.chem.descriptor.pharmacophoretree.HungarianAlgorithm;
import org.apache.commons.lang3.tuple.Pair;

import java.util.Arrays;
import java.util.List;

public class FeaturePairDescriptor<D extends MolFeatureDistance,F extends MolFeature> implements DescriptorHandler<List<FeaturePair<D,F>>, StereoMolecule> {


    public static interface FeatureAndPairSimilarityCombinator {
        public double combinedSimilarity(double simDistance, double simFeatureA, double simFeatureB);
    }


    public static enum MATCHING_MODE {JACCARD};

    private MATCHING_MODE matchingMode = MATCHING_MODE.JACCARD;

    private FeatureHandler<F> featureHandler;
    private FeaturePairHandler<D> featurePairHandler;
    private FeatureAndPairSimilarityCombinator combinator;

    /**
     * Options: JACCARD: means, we match all A <-> B to compute avg. similarity of matchings, then we estimate
     *                   the average distance from all pairs A x B, then we return 1 - ( avg.matching.sim / avg.sim )
     * @param mode
     */
    public void setMatchingMode(MATCHING_MODE mode) {
        this.matchingMode = mode;
    }


    public FeaturePairDescriptor(FeatureHandler<F> featureHandler, FeaturePairHandler<D> featurePairHandler, FeatureAndPairSimilarityCombinator combinator) {
        this.featureHandler = featureHandler;
        this.featurePairHandler = featurePairHandler;
        this.combinator = combinator;
    }

    @Override
    public DescriptorInfo getInfo() {
        return null;
    }

    @Override
    public String getVersion() {
        return null;
    }

    @Override
    public String encode(List<FeaturePair<D, F>> featurePairs) {
        return null;
    }

    @Override
    public List<FeaturePair<D, F>> decode(String s) {
        return null;
    }

    @Override
    public List<FeaturePair<D, F>> decode(byte[] bytes) {
        return null;
    }

    @Override
    public List<FeaturePair<D, F>> createDescriptor(StereoMolecule m) {
        List<F> features = featureHandler.detectFeatures(m);
        List<FeaturePair<D, F>>  pairs =  featurePairHandler.detectPairs(m,features);
        return pairs;
    }

    @Override
    public boolean calculationFailed(List<FeaturePair<D, F>> featurePairs) {
        return false;
    }

    @Override
    public DescriptorHandler<List<FeaturePair<D, F>>, StereoMolecule> getThreadSafeCopy() {
        return null;
    }





    public final static class MatchResult {
        public final double[] weightsA;
        public final double[] weightsB;
        public final double[][] similarities;
        public final double[][] weightedSimilarities;
        public final int[][] match;
        public final double similarity;

        public MatchResult(double weightsA[], double weightsB[], double[][] similarities, double[][] weightedSimilarities, int[][] match, double similarity) {
            this.weightsA = weightsA;
            this.weightsB = weightsB;
            this.similarities = similarities;
            this.weightedSimilarities = weightedSimilarities;
            this.match = match;
            this.similarity = similarity;
        }
    }

    public static MatchResult computeMatch(double similarityMatrix_unweighted[][], double weightsA[], double weightsB[], MATCHING_MODE mode) {
        // compute weights into similarity matrix:
        double[][] similarityMatrix = new double[similarityMatrix_unweighted.length][similarityMatrix_unweighted[0].length];
        for(int zi=0;zi<weightsA.length;zi++) {
            for(int zj=0;zj<weightsB.length;zj++) {
                similarityMatrix[zi][zj] = similarityMatrix_unweighted[zi][zj] * (weightsA[zi] + weightsB[zj]);
            }
        }

        if(mode==MATCHING_MODE.JACCARD) {
            int[][] match = HungarianAlgorithm.hgAlgorithm(similarityMatrix,"max");

            double matchedSimilarity = 0;

            int na = weightsA.length;
            int nb = weightsB.length;

            for (int zi = 0; zi < na; zi++) {
                int xa = match[zi][0];
                int xb = match[zi][1];
                if (xa < na && xb < nb) {
                    matchedSimilarity += similarityMatrix[xa][xb];
                    //matchedSimilarity += similarityMatrix[xa][xb] * (weightsA[xa] + weightsB[xb]);
                }
                // add zero for all unmatched..
            }

            double totalWeight = Arrays.stream(weightsA).sum() + Arrays.stream(weightsB).sum();
            double score = matchedSimilarity / totalWeight;

            //System.out.println("matched weighted similarity = "+matchedSimilarity);
            //System.out.println("total weight = " + totalWeight);
            //System.out.println("score= " +score);
            return new MatchResult(weightsA,weightsB,similarityMatrix_unweighted,similarityMatrix,match,score);
        }
        return null;
    }


    public MatchResult computeFullSimilarity(List<FeaturePair<D, F>> featurePairs, List<FeaturePair<D, F>> t1) {
        if(matchingMode == MATCHING_MODE.JACCARD) {
            List<FeaturePair<D, F>> pa = null;
            List<FeaturePair<D, F>> pb = null;

            if (featurePairs.size() >= t1.size()) {
                pa = featurePairs;
                pb = t1;
            } else {
                pa = t1;
                pb = featurePairs;
            }

            //PPPSimilarity ppp = new PPPSimilarity();
            /**
             * We compute a distance matrix as follows:
             * We consider the better matching version of possible a->b feature mappings.
             */

            int maxsize = Math.max(pa.size(), pb.size());
            double similarities[][] = new double[maxsize][maxsize];
            for (int zi = 0; zi < pa.size(); zi++) {
                for (int zj = 0; zj <  pb.size(); zj++) {
                    FeaturePair<D, F> pi = pa.get(zi);
                    FeaturePair<D, F> pj = pb.get(zj);
                    // we check if a->a and a->b is better or a->b b->a for the two pairs..
                    double fs_1a = featureHandler.computeFeatureSimilarity(pi.a, pj.a);
                    double fs_1b = featureHandler.computeFeatureSimilarity(pi.b, pj.b);
                    double fs_2a = featureHandler.computeFeatureSimilarity(pi.a, pj.b);
                    double fs_2b = featureHandler.computeFeatureSimilarity(pi.b, pj.a);
                    double ds_a = featurePairHandler.computeDistanceSimilarity(pi.d, pj.d);

                    double sim_1 = combinator.combinedSimilarity(ds_a, fs_1a, fs_1b);
                    double sim_2 = combinator.combinedSimilarity(ds_a, fs_2a, fs_2b);
                    double sim = Math.max(sim_1, sim_2);
                    similarities[zi][zj] = sim;
//                    double dx = nomatch_distance;
//                    if (zi < pa.size() && zj < pb.size()) {
//                        dx = ppp.distance(pa.get(zi), pb.get(zj));
//                    }
//                    dx = Math.min(dx, nomatch_distance);
//                    dm[zi][zj] = dx;
                }
            }

            double[] weightsA = pa.stream().mapToDouble(xi -> xi.evaluateImportance(featureHandler, featurePairHandler)).toArray();
            double[] weightsB = pb.stream().mapToDouble(xi -> xi.evaluateImportance(featureHandler, featurePairHandler)).toArray();
            MatchResult match = computeMatch(similarities, weightsA, weightsB, matchingMode);
            return match;
        }
        return null;
    }

    @Override
    public float getSimilarity(List<FeaturePair<D, F>> featurePairs, List<FeaturePair<D, F>> t1) {
        MatchResult mi = computeFullSimilarity(featurePairs,t1);
        return (float) mi.similarity;
    }


}
