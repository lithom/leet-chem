package tech.molecules.leet.chem.sar.analysis;

import org.apache.commons.lang3.tuple.Pair;

import java.util.Collections;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class DescriptiveStats {

    //private Map<Pair<String,Part>, Double> numericalData;

    //public DescriptiveStats(Map<Pair<String,Part>, Double> numericalData) {
    //    this.numericalData = numericalData;
    //}

    public static class Stats {
        public final int n;
        public final double average;
        public final double median;
        public final double min;
        public final double max;
        public final double standardDeviation;

        public Stats(int n, double average, double median, double min, double max, double standardDeviation) {
            this.n       = n;
            this.average = average;
            this.median = median;
            this.min = min;
            this.max = max;
            this.standardDeviation = standardDeviation;
        }
    }

//    public Stats computeStatsForPart(Part part) {
//        List<Double> values = numericalData.entrySet().stream()
//                .filter(entry -> entry.getKey().equals(part))
//                .map(Map.Entry::getValue)
//                .collect(Collectors.toList());
//        return computeStats(part.toString(),values);
//    }

    public static Stats computeStats(List<Double> v) {
        int n = v.size();
        Collections.sort(v);
        DoubleSummaryStatistics dss = v.stream().mapToDouble(xi->xi).summaryStatistics();

        double median = (v.size()%2==0) ? 0.5*( v.get((v.size()/2)-1) +v.get(v.size()/2)) : v.get( (v.size()-1) /2);
        double mean   = dss.getAverage();

        double variance = dss.getSum() / v.size() - mean*mean;
        return new Stats(n,dss.getAverage(),median,dss.getMin(),dss.getMax(),Math.sqrt(variance));
    }

}
