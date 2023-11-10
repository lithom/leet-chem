package tech.molecules.leet.chem.sar.analysis;

import org.apache.commons.lang3.tuple.Pair;

import java.util.*;
import java.util.stream.Collectors;

public class PartHelper {

    public static List<Pair<String,String>> getLabelPart(Part pi, Collection<String> labels) {
        //Map<String,String> extracted = new HashMap<>();
        List<Pair<String,String>> extracted = new ArrayList<>();
        for(Pair<String,String> vi : pi.getVariants()) {
            if(labels.contains(vi.getLeft())) {
                extracted.add(Pair.of(vi.getLeft(),vi.getRight()));
            }
        }
        if(extracted.size()!=labels.size()) {
            System.out.println("[WARN] not all labels found..");
        }

        return extracted;
    }

    public static String getLabelPart(Part pi, String label) {
        for(Pair<String,String> vi : pi.getVariants()) {
            if(vi.getLeft().equals(label)) {
                return vi.getRight();
            }
        }
        System.out.println("[WARN] label not found..");
        return null;
    }

    public static List<List<MatchedSeries.MatchedSeriesElement>> computeAllQuadruples(String labelA, String labelB, DecompositionProvider provider) {
        List<List<MatchedSeries.MatchedSeriesElement>> allQuadruples = new ArrayList<>();

        // Get all possible Part variants for labelA and labelB
        List<Part> variantsA = provider.getAllVariantsForPart(labelA);
        List<Part> variantsB = provider.getAllVariantsForPart(labelB);

        // Generate all possible parts from the variants
        for (Part variantA : variantsA) {
            for (Part variantB : variantsB) {
                Part partA = variantA;
                Part partB = variantB;

                List<List<MatchedSeries.MatchedSeriesElement>> quadruples = getQuadruplesForParts(partA, partB, provider);
                allQuadruples.addAll(quadruples);
            }
        }

        // TODO: probably filter for unique ones? I guess we compute the same quadruples over and over again?
        Set<String> uniqueQuadrupleStrings = allQuadruples.stream()
                .map(PartHelper::quadrupleToString)
                .collect(Collectors.toSet());

        List<List<MatchedSeries.MatchedSeriesElement>> distinctQuadruples = uniqueQuadrupleStrings.stream()
                .map(s -> stringToQuadruple(s, provider))
                .collect(Collectors.toList());

        return distinctQuadruples;
    }

    private static String quadrupleToString(List<MatchedSeries.MatchedSeriesElement> quadruple) {
        return quadruple.stream()
                .map(MatchedSeries.MatchedSeriesElement::getStructure)
                .collect(Collectors.joining("\t"));
    }

    private static List<MatchedSeries.MatchedSeriesElement> stringToQuadruple(String s, DecompositionProvider provider) {
        String[] structures = s.split("\\t");
        List<MatchedSeries.MatchedSeriesElement> quadruple = new ArrayList<>();
        for (String structure : structures) {
            Part part = provider.getDecompositionForStructure(structure);
            MatchedSeries series = provider.findComplementSeries(part, part); // Using the same part as it's just a placeholder
            for (MatchedSeries.MatchedSeriesElement element : series.getSeries()) {
                if (element.getStructure().equals(structure)) {
                    quadruple.add(element);
                    break;
                }
            }
        }
        return quadruple;
    }

    /**
     * Returns the quadruples for partA
     *
     * @param partA
     * @param partB
     * @param provider
     * @return
     */
    public static List<List<MatchedSeries.MatchedSeriesElement>> getQuadruplesForParts(Part partA, Part partB, DecompositionProvider provider) {
        List<List<MatchedSeries.MatchedSeriesElement>> quadruples = new ArrayList<>();

        // Obtain series for both parts
        MatchedSeries seriesA = provider.findComplementSeries(partA, partB);
        MatchedSeries seriesB = provider.findComplementSeries(partB, partA);

        for (MatchedSeries.MatchedSeriesElement elementA : seriesA.getSeries()) {
            for (MatchedSeries.MatchedSeriesElement elementB : seriesB.getSeries()) {
                if (elementA.getCommonRemainder().equals(elementB.getCommonRemainder())) {
                    // Create a quadruple and add to the list
                    List<MatchedSeries.MatchedSeriesElement> quadruple = Arrays.asList(elementA, elementB);
                    quadruples.add(quadruple);
                }
            }
        }
        return quadruples;
    }

}
