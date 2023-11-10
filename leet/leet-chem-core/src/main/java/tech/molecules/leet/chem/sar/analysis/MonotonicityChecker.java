package tech.molecules.leet.chem.sar.analysis;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class MonotonicityChecker {

    private DecompositionProvider provider;

    public MonotonicityChecker(DecompositionProvider provider) {
        this.provider = provider;
    }

    private boolean checkMonotonicityForQuadruple(List<MatchedSeries.MatchedSeriesElement> quadruple, Map<String, Double> numericalData) {
        double delta1 = numericalData.get(quadruple.get(0).getStructure()) - numericalData.get(quadruple.get(1).getStructure());
        double delta2 = numericalData.get(quadruple.get(2).getStructure()) - numericalData.get(quadruple.get(3).getStructure());
        double delta3 = numericalData.get(quadruple.get(0).getStructure()) - numericalData.get(quadruple.get(3).getStructure());

        return (Math.signum(delta1) == Math.signum(delta2)) && (Math.signum(delta1) == Math.signum(delta3));
    }

    public double computeIMS(Part partA, Part partB, Map<String, Double> numericalData) {
        List<List<MatchedSeries.MatchedSeriesElement>> quadruples = PartHelper.getQuadruplesForParts(partA, partB, provider);
        int monotonicCount = 0;

        for (List<MatchedSeries.MatchedSeriesElement> quadruple : quadruples) {
            if (checkMonotonicityForQuadruple(quadruple, numericalData)) {
                monotonicCount++;
            }
        }

        return (double) monotonicCount / quadruples.size();
    }





}

