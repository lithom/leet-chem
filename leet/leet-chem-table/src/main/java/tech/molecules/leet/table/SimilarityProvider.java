package tech.molecules.leet.table;

public interface SimilarityProvider<U> {

    public String getName();
    public NColumn<U,?> getColumn();
    //public boolean hasValue(U dp, String row);
    public double evaluate(U dp, String row_a, String row_b);

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

}
