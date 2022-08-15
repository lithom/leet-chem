package tech.molecules.leet.table;

public class CachedDescriptorSimilarityProvider implements SimilarityProvider<NDataProvider.StructureDataProvider> {



    public CachedDescriptorSimilarityProvider(String dsc_shortname) {

    }

    @Override
    public String getName() {
        return null;
    }

    @Override
    public NColumn<NDataProvider.StructureDataProvider, ?> getColumn() {
        return null;
    }

    @Override
    public double evaluate(NDataProvider.StructureDataProvider dp, String row_a, String row_b) {
        return 0;
    }

    @Override
    public boolean isSimilarity() {
        return false;
    }

    @Override
    public boolean isNormalized() {
        return false;
    }

    @Override
    public boolean isSymmetric() {
        return false;
    }
}
