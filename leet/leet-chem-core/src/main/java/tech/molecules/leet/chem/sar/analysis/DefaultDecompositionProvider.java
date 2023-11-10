package tech.molecules.leet.chem.sar.analysis;

import org.apache.commons.lang3.tuple.Pair;

import java.util.*;
import java.util.stream.Collectors;

public class DefaultDecompositionProvider implements DecompositionProvider {

    private List<String> labels;
    private List<Pair<String,Part>> decompositions;

    private List<Part> decompositionsList;
    private Map<Part,String> structuresByDecomposition;


    /**
     * Decompositions must be the fully decomposed molecules.
     *
     * @param decompositions
     */
    public DefaultDecompositionProvider(List<String> labels, List<Pair<String,Part>> decompositions) {
        this.decompositions = decompositions;
        this.structuresByDecomposition = new HashMap<>();
        this.decompositionsList = new ArrayList<>();
        for(Pair<String,Part> pi : decompositions) {
            this.decompositionsList.add(pi.getRight());
            this.structuresByDecomposition.put(pi.getRight(),pi.getLeft());
        }
        this.labels = labels;//new ArrayList<>( decompositions.get(0).getRight().getPartLabels() );
    }

    @Override
    public List<Pair<String,Part>> findAllStructuresWithPart(Part p) {
        List<Pair<String,Part>> filtered = decompositions.parallelStream().filter( xi -> filterDecomposition( xi.getRight() , p ) ).collect(Collectors.toList());
        return filtered;
    }

    public static boolean filterDecomposition(Part mol, Part filter) {
        boolean ok = true;
        for( Pair<String,String> pi : filter.getVariants() ) {
            // se if we find this:
            boolean found = mol.getVariants().stream().anyMatch( xi -> xi.getLeft().equals(pi.getLeft()) && xi.getRight().equals(pi.getRight()) );
            if(!found) {
                ok = false;
                break;
            }
        }
        return ok;
    }

    @Override
    public Part getDecompositionForStructure(String s) {
        return null;
    }


    @Override
    public List<String> getAllLabels() {
        return new ArrayList<>(labels);
    }

    @Override
    public Part createPart(List<Pair<String, String>> parts) {
        List<String> complement = getAllLabels();
        List<String> labels = parts.stream().map(xi -> xi.getLeft()).collect(Collectors.toList());
        complement.removeAll( labels );
        return new DefaultPart(parts,complement);
    }

    @Override
    public MatchedSeries findComplementSeries(Part basePart, Part specificRemainderPart) {
        return null;
    }

    @Override
    public List<Part> getAllVariantsForPart(String label) {
        Set<Part> variants = new HashSet<>();
        this.decompositionsList.forEach( xi -> variants.add( createPart(Collections.singletonList( Pair.of(label,PartHelper.getLabelPart(xi,label)) ) ) ) );
        return new ArrayList<>( variants );
    }

    @Override
    public List<Pair<String, Part>> getAllDecompositions() {
        return new ArrayList<>(this.decompositions);
    }

    @Override
    public List<Part> getAllVariantsForPart(List<String> labels) {
        Set<Part> variants = new HashSet<>();
        this.decompositionsList.forEach( xi -> variants.add( createPart( PartHelper.getLabelPart(xi,labels) ) ) );
        return new ArrayList<>( variants );
    }

    @Override
    public Map<Part, List<Pair<String, Part>>> getAllVariantsForPart2(List<String> labels) {
        Map<Part, List<Pair<String,Part>>> variantsSorted = new HashMap<>();
        for(Pair<String,Part> di : decompositions) {
            Part pi = createPart( PartHelper.getLabelPart(di.getRight(),labels) );
            variantsSorted.putIfAbsent(pi,new ArrayList<>());
            variantsSorted.get(pi).add( Pair.of(di.getLeft(),pi));
        }
        return variantsSorted;
    }
}
