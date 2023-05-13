package tech.molecules.leet.datatable.filter;

import tech.molecules.leet.datatable.AbstractCachedDataFilter;
import tech.molecules.leet.datatable.DataFilter;
import tech.molecules.leet.datatable.DataFilterType;
import tech.molecules.leet.datatable.DataTableColumn;

import java.util.regex.Pattern;

public class NumericRangeFilter extends AbstractCachedDataFilter<Double> {

    public static NumericRangeFilter.NumericRangeFilterType TYPE = new NumericRangeFilter.NumericRangeFilterType();

    public static class NumericRangeFilterType implements DataFilterType<Double> {
        @Override
        public String getFilterName() {
            return "NumericRangeFilter";
        }

        @Override
        public boolean requiresInitialization() {
            return true;
        }

        @Override
        public DataFilter<Double> createInstance(DataTableColumn<?,Double> column) {
            return new NumericRangeFilter();
        }

    }

    public NumericRangeFilter() {
        this.setRange(new double[]{Double.NaN,Double.NaN});
    }

    /**
     * Configuration
     */
    private double range[];


    public void setRange(double[] range) {
        this.range = range;
    }

    @Override
    public boolean filterRow(Double vi) {
        //try {Thread.sleep(50);
        //} catch (InterruptedException e) { throw new RuntimeException(e);}

        boolean lb_ok = Double.isNaN(range[0]) ? true : vi >= range[0];
        boolean ub_ok = Double.isNaN(range[1]) ? true : vi <= range[1];

        return ! ( lb_ok && ub_ok );
    }

    @Override
    public DataFilterType<Double> getDataFilterType() {
        return TYPE;
    }

    @Override
    public double getApproximateFilterSpeed() {
        return 0.99;
    }

    public double[] getRange() {
        return this.range;
    }

}
