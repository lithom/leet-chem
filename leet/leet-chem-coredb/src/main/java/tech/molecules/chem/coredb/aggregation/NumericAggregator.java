package tech.molecules.chem.coredb.aggregation;

import tech.molecules.chem.coredb.AssayResult;
import tech.molecules.chem.coredb.DataValue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class NumericAggregator {


    public static AggregatedNumericValue aggregate(NumericAggregationInfo info , List<AssayResult> filteredResults) {

        List<Double> values = new ArrayList<>();

        // Filter
        for(AssayResult ri : filteredResults) {
            DataValue xvi = ri.getData(info.getParameterName());
            if(xvi != null) {
                double vi = xvi.getAsDouble();
                if(!Double.isNaN(vi)) {
                    if( Double.isFinite( info.getFixedLB() )) {
                        vi = Math.max(vi, info.getFixedLB());
                    }
                    if( Double.isFinite( info.getFixedUB() )) {
                        vi = Math.min(vi, info.getFixedUB());
                    }
                    if( info.isHandleLogarithmic() ) {
                        if(vi <= 0) {
                            continue;
                        }
                    }
                    values.add(vi);
                }
            }
        }

        // Aggregate:
        double aggregated = Double.NaN;
        if(info.isHandleLogarithmic()) {
            values = values.stream().map( xi -> Math.log(xi) ).collect(Collectors.toList());
        }

        // aggregate:
        aggregated = aggregate(info.getMethod(),values);

        if(info.isHandleLogarithmic()) {
            aggregated = Math.exp(aggregated);
        }

        return new AggregatedNumericValue(aggregated,info,filteredResults,values.stream().mapToDouble(xi->xi).toArray());
    }


    public static double aggregate(String method, List<Double> values ) {
        double aggregated = Double.NaN;
        if(values.size()==0) {
            return aggregated;
        }
        boolean handled = false;
        if( method.equals( NumericAggregationInfo.AGGREGATION_MEAN) ) {
            aggregated = values.stream().mapToDouble(xi -> xi ).summaryStatistics().getAverage();
            handled = true;
        }

        // fallback to default mean
        if(!handled) {
            System.out.println("[WARN] unknown aggregation method "+method+" -> fallback to mean");
            aggregated = values.stream().mapToDouble(xi -> xi ).summaryStatistics().getAverage();
            handled = true;
        }

        return aggregated;
    }

    public static List<AssayResult> filter(FilteredAssayInfo filterInfo , List<AssayResult> in) {
        List<AssayResult> results = new ArrayList<>(
                in.stream().filter(xi -> filter(filterInfo,xi)).collect(Collectors.toList())
        );
        return results;
    }

    public static boolean filter(FilteredAssayInfo filterInfo , AssayResult in) {
        boolean ok = true;

        // assay id
        if( ok ) {
            if( in.getAssay().getId() != filterInfo.assayID ) {
                ok = false;
            }
        }

        // apply assay result filter info
        if( ok ) {
            if( filterInfo.assayResultFilter != null ) {
                ok = filterAssayResultFilter(filterInfo.assayResultFilter,in);
            }
        }

        if( ok ) {
            if( filterInfo.coreDBFilter != null ) {
                ok = filterCoreDBFilter(filterInfo.coreDBFilter,in);
            }
        }

        return ok;
    }


    public static boolean filterAssayResultFilter(AssayResultFilterInfo info, AssayResult xi) {
        // TODO: implement quality information
        return true;
    }

    public static boolean filterCoreDBFilter(CoreDBFilterInfo info, AssayResult xi) {
        boolean ok = true;
        Map<String, CoreDBFilterInfo.AttrFilterInfo> paramFilters = info.getAttrFiltersSorted();
        for( String ki : paramFilters.keySet()) {
            DataValue vi = xi.getData(ki);
            if(! paramFilters.get(ki).values.contains(vi.getAsText())) {
                ok = false;
                break;
            }
        }
        return ok;
    }


}
