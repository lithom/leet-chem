package tech.molecules.leet.chem.sar.analysis;

import org.apache.commons.lang3.tuple.Pair;

import java.util.List;
import java.util.Map;

public interface Part {

    public void initialize(List<Pair<String,String>> parts, List<String> complement);

    public List<String> getPartLabels();
    public List<String> getComplementLabels();


    /**
     *
     * @return hashmap with all elements, also with complement labels (they have empty string)
     */
    public Map<String,String> getFullMap();

    /**
     *
     * @return (part label, part variant)
     */
    public List<Pair<String,String>> getVariants();

    public default boolean isSingle() {
        return getPartLabels().size()==1;
    }

}
