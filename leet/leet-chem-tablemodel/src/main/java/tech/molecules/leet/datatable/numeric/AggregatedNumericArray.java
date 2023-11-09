package tech.molecules.leet.datatable.numeric;

import tech.molecules.leet.data.NumericArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class AggregatedNumericArray {

    public static enum AGGREGATION {NONE}



    private NumericArray data;
    private AGGREGATION aggregation;


    public AggregatedNumericArray(NumericArray data) {
        this(data,AGGREGATION.NONE);
    }

    public AggregatedNumericArray(NumericArray data, AGGREGATION aggregation) {
        this.data = data;
        this.aggregation = aggregation;
    }


    public List<Double> getValues() {
        switch(this.aggregation) {
            case NONE: return Arrays.stream(this.data.getData()).boxed().collect(Collectors.toList());
        }

        return new ArrayList<>();
    }

    public NumericArray getAllValuesNumericArray() {
        return this.data;
    }

    public String toString() {
        return Arrays.stream(this.data.getData()).mapToObj( xi -> ""+xi ).collect(Collectors.joining(";"));
    }

    public double getMean() {
        return Arrays.stream( this.data.getData() ).summaryStatistics().getAverage();
    }

}
