package tech.molecules.leet.datatable.filter;

import javafx.scene.control.TableColumn;
import tech.molecules.leet.datatable.*;

import java.util.regex.Pattern;

public class NumericRangeFilter<U> extends AbstractCachedDataFilter<U> {

    public static NumericRangeFilter.NumericRangeFilterType TYPE = new NumericRangeFilter.NumericRangeFilterType();

    public static class NumericRangeFilterType implements DataFilterType<Object> {
        @Override
        public String getFilterName() {
            return "NumericRangeFilter";
        }

        @Override
        public boolean requiresInitialization() {
            return true;
        }

        @Override
        public DataFilter<Object> createInstance(DataTableColumn column) {
            return new NumericRangeFilter(null);
        }

    }

    private NumericDatasource<U> numericDatasource;

    public NumericRangeFilter(NumericDatasource<U> nds) {
        this.numericDatasource = nds;
        this.setRange(new double[]{Double.NaN,Double.NaN});
    }

    public void setNumericDatasource(NumericDatasource nds) {
        this.numericDatasource = nds;
    }

    /**
     * Configuration
     */
    private double range[];


    public void setRange(double[] range) {
        this.range = range;
        fireFilterChanged();
    }

    @Override
    public boolean filterRow(U vi_a) {
        //try {Thread.sleep(50);
        //} catch (InterruptedException e) { throw new RuntimeException(e);}
        double vi = numericDatasource.evaluate(vi_a);

        boolean lb_ok = Double.isNaN(range[0]) ? true : vi >= range[0];
        boolean ub_ok = Double.isNaN(range[1]) ? true : vi <= range[1];

        return ! ( lb_ok && ub_ok );
    }

    @Override
    public DataFilterType getDataFilterType() {
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
