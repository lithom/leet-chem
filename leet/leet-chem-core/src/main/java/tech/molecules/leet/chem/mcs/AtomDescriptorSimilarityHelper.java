package tech.molecules.leet.chem.mcs;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.BitSetUtils;

import java.util.*;
import java.util.stream.Collectors;

public class AtomDescriptorSimilarityHelper {

    protected int selectionStrategy1(BitSet c) {
        //return c.stream().findFirst().getAsInt();
        //List<Integer> list = c.stream().boxed().collect(Collectors.toList());
        //list.sort( (xi,yi) -> -Double.compare( bestMatchValueInA[xi] , bestMatchValueInA[yi] ) );
        //int[] result =  list.stream().mapToInt( li -> li ).toArray();
        double vmax = -1;
        int    imax = -1;
        for(int vi : c.stream().toArray()) {
            if(bestMatchValueInA[vi]>vmax) {
                vmax = bestMatchValueInA[vi];
                imax = vi;
            }
        }
        return imax;
    }

    protected int[] selectionStrategy2(int v, BitSet ci) {
        List<Integer> list = ci.stream().boxed().collect(Collectors.toList());
        list.sort( (xi,yi) -> -Double.compare( similarities[v][xi] , similarities[v][yi] ) );
        int[] result =  list.stream().mapToInt( li -> li ).toArray();
        return result;
    }

    private BitSet[] descriptors_A = null;
    private BitSet[] descriptors_B = null;


    /**
     * atom similarities
     */
    private double[][] similarities = null;

    /**
     * Position i contains highest match value for atom i in molecule a.
     */
    private double[] bestMatchValueInA = null;



    public void setAtomDescriptors(BitSet[] a, BitSet[] b) {
        this.descriptors_A = a;
        this.descriptors_B = b;

        this.similarities = new double[a.length][b.length];
        this.bestMatchValueInA = new double[a.length];
        // compute similarity stuff:
        for(int zi=0;zi<a.length;zi++) {
            double bestmatch_a = -1;
            for(int zj=0;zj<b.length;zj++) {
                double simij = BitSetUtils.tanimoto_similarity(a[zi],b[zj]);
                similarities[zi][zj] = simij;
                bestmatch_a = Math.max(bestmatch_a,simij);
            }
            bestMatchValueInA[zi] = bestmatch_a;
        }
    }

    public int findBestCandidateInA(BitSet candidates_a) {
        int best = -1;
        if(true) {
            // strategy: take the one with the best (similarity^2 * num_bits)
            double best_score = -1;
            for (int ca : candidates_a.stream().toArray()) {
                double score_a = this.bestMatchValueInA[ca];
                if (score_a >= best_score) {
                    best = ca;
                    best_score = score_a;
                }
            }
        }

        // strategy: take the one with the best similarity score, use number of bits as tie breaker
        if(false) {
            double best_similarity = -1;
            int num_bits = -1;
            for (int ca : candidates_a.stream().toArray()) {
                if (this.bestMatchValueInA[ca] >= best_similarity) {
                    if (this.bestMatchValueInA[ca] > best_similarity) {
                        best = ca;
                        best_similarity = this.bestMatchValueInA[ca];
                        num_bits = this.descriptors_A[ca].cardinality();
                    } else {
                        if (this.descriptors_A[ca].cardinality() > num_bits) {
                            best = ca;
                            num_bits = this.descriptors_A[ca].cardinality();
                        }
                    }
                }
            }
        }
        return best;
    }

    public static final class AtomWithSimilarity implements Comparable<AtomWithSimilarity> {
        public final int atom;
        public final double similarity;
        public AtomWithSimilarity(int atom, double similarity) {
            this.atom = atom;
            this.similarity = similarity;
        }

        @Override
        public int compareTo(AtomWithSimilarity o) {
            int simcomparison = Double.compare(this.similarity,o.similarity);
            if(simcomparison!=0) {return simcomparison;}
            return Integer.compare(this.atom,o.atom);
        }
    }

    public int[] findBestCandidateInB(int v, BitSet ci) {
        TreeSet<AtomWithSimilarity> options = new TreeSet<>();

        for(int ni : ci.stream().toArray()) {
            options.add(new AtomWithSimilarity(ni,similarities[v][ni]));
        }

        int[] sorted = new int[options.size()];
        int pi = 0;
        Iterator<AtomWithSimilarity> it = options.descendingIterator();
        while( it.hasNext() ) {
            sorted[pi] = it.next().atom;
            pi++;
        }
        return sorted;
    }
}
