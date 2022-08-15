package tech.molecules.leet.chem;

import java.util.BitSet;

public class BitSetUtils {

    /**
     * tests if a is subset of b
     *
     * @param a
     * @param b
     * @return
     */
    public static boolean test_subset(BitSet a, BitSet b) {
        BitSet ti = (BitSet) b.clone();
        ti.or(a);
        return ti.equals(b);
    }

    /**
     * tests if a is subset of b
     *
     * @param a
     * @param b
     * @return
     */
    public static boolean test_subset(long a[], long b[]) {
        boolean is_subset = true;
        for(int zi=0;zi<a.length;zi++) {
            long bi = (b.length>zi)?b[zi]:0x0;
            long a2 = a[zi];
            is_subset = (a2 | bi) == bi;
            if(!is_subset) {break;}
        }
        return is_subset;
    }

    public static int hamming(BitSet a, BitSet b) {
        BitSet xored = (BitSet) a.clone();
        xored.xor(b);
        return xored.cardinality();
    }

    public static double tanimoto_similarity(BitSet a, BitSet b) {
        BitSet all_bits    = (BitSet) a.clone();
        BitSet shared_bits = (BitSet) a.clone();
        all_bits.or(b);
        shared_bits.and(b);

        double a_or_b = all_bits.cardinality();
        double a_and_b = shared_bits.cardinality();
        return a_and_b / a_or_b;
    }

}
