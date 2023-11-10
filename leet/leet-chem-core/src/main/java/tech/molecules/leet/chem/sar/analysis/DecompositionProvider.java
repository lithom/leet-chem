package tech.molecules.leet.chem.sar.analysis;

import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 *
 * Serves as a "database" that can be queried for all structures with
 * specific characteristics (in terms of the decomposition)
 *
 */
public interface DecompositionProvider {

    public Part getDecompositionForStructure(String s);

    public List<Pair<String,Part>> findAllStructuresWithPart(Part p);

    public List<Part> getAllVariantsForPart(String label);

    public List<Part> getAllVariantsForPart(List<String> labels);

    /**
     *
     * @param labels
     * @return
     */
    public Map<Part,List<Pair<String,Part>>> getAllVariantsForPart2(List<String> labels);

    public List<Pair<String,Part>> getAllDecompositions();

    public List<String> getAllLabels();

    /**
     * Factory method for Part objects
     *
     * @param parts
     * @return
     */
    public Part createPart(List<Pair<String,String>> parts);

    /**
     * Returns all structures that share the same remainder part with a different base part
     *
     * @param
     * @return
     */
    public MatchedSeries findComplementSeries(Part basePart, Part specificRemainderPart);

    /**
     *
     * @param
     * @return list of matched series for all remainders that we find for structures with the given base part
     */
    public default List<MatchedSeries> findAllComplementSeries(Part basePart) {
        List<MatchedSeries> series = new ArrayList<>();
        List<Pair<String,Part>> structures = findAllStructuresWithPart(basePart);
        for(Pair<String,Part> si : structures) {
            // extract the remainder parts:
            List<Pair<String,String>> remainder = PartHelper.getLabelPart( si.getRight() , basePart.getComplementLabels() );
            // for this remainder we create a series:
            MatchedSeries msi = findComplementSeries( basePart , createPart( remainder ) );
            series.add(msi);
        }
        return series;
    }

}
