package tech.molecules.leet.table;

public interface NSimilarityColumn<U,T> extends NColumn<U,T> {

    /**
     * true for similarity, false for distance
     * @return
     */
    public boolean isSimilarity();

    /**
     * If value is always in between 0 and 1.
     * @return
     */
    public boolean isNormalized();

    public boolean isSymmetric();

    public double evaluateValue(U data, String rowid_a, String rowid_b);

}
