package tech.molecules.leet.chem.sar.analysis;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.sar.SimpleSARDecomposition;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class DecompositionProviderFactory {

    public DecompositionProviderFactory() {}

    public DecompositionProvider createDecompositionProviderFromSimpleSARResult(List<SimpleSARDecomposition.SimpleSARResult> data) {

        List<String> labels = new ArrayList<>( data.get(0).keySet() );
        List<Pair<String,Part>> decomp = new ArrayList<>();

        DefaultDecompositionProvider ddp = new DefaultDecompositionProvider(labels,new ArrayList<>());

        List<SimpleSARDecomposition.SimpleSARResult> data_filtered = data.stream().filter(xi -> xi!=null).collect(Collectors.toList());
        for(SimpleSARDecomposition.SimpleSARResult ri : data_filtered) {
            try {
                List<Pair<String, String>> dci = new ArrayList<>();
                for (String li : labels) {
                    dci.add(Pair.of(li, ri.get(li).matchedFrag.getIDCode()));
                }
                Part pi = ddp.createPart(dci);
                decomp.add(Pair.of(ri.getStructure()[0], pi));
            }
            catch(Exception ex) {
                ex.printStackTrace();
            }
        }
        return new DefaultDecompositionProvider(labels,decomp);
    }

}
