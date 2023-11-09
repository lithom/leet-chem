package tech.molecules.leet.datatable.chart.jfc;

import org.jfree.data.DomainOrder;
import org.jfree.data.general.DatasetChangeEvent;
import org.jfree.data.general.DatasetChangeListener;
import org.jfree.data.general.DatasetGroup;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYZDataset;
import tech.molecules.leet.datatable.CategoricDatasource;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.NumericDatasource;

import java.sql.Array;
import java.util.*;
import java.util.stream.Collectors;

public class LeetXYZDataSet implements XYZDataset, XYDataset {

    /**
     * Configuration data
     */

    private DataTable table;
    private NumericDatasource dataX = null;
    private NumericDatasource dataY = null;
    private NumericDatasource dataZ = null;
    private CategoricDatasource dataCategoric = null;


    public LeetXYZDataSet(DataTable table) {
        this.table = table;
        this.table.addDataTableListener(new DataTable.DataTableListener() {
            @Override
            public void tableDataChanged() {
                recompute();
            }

            @Override
            public void tableStructureChanged() {
                recompute();
            }

            @Override
            public void tableCellsChanged(List<int[]> cells) {
                recompute();
            }
        });
    }

    /**
     * Computed data
     */
    public static final class XYZSeries {
        public final String seriesKey;
        public final List<Double> x;
        public final List<Double> y;
        public final List<Double> z;

        public XYZSeries(String seriesKey, List<Double> x, List<Double> y, List<Double> z) {
            this.seriesKey = seriesKey;
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }

    private List<XYZSeries> data = new ArrayList<>();

    //private Map<XYZSeries,String> seriesToKey = new HashMap<>();


    /**
     *
     * @param dataX
     * @param dataY
     * @param dataZ
     * @param categoric
     */
    public void setDataSources(NumericDatasource dataX, NumericDatasource dataY, NumericDatasource dataZ, CategoricDatasource categoric) {
        this.dataX = dataX;
        this.dataY = dataY;
        this.dataZ = dataZ;
        this.dataCategoric = categoric;
        this.recompute();
    }

    /**
     * This is the function that reinitializes all of the data in this dataset.
     *
     *
     */
    private void recompute() {
        List<String> rows = table.getVisibleKeysSorted();
        double[] x_visible = new double[rows.size()];
        double[] y_visible = new double[rows.size()];
        double[] z_visible = new double[rows.size()];
        Arrays.fill(x_visible,Double.NaN);
        Arrays.fill(y_visible,Double.NaN);
        Arrays.fill(z_visible,Double.NaN);

        if(this.dataX!=null) {
             List<Double> di = this.dataX.getDataVisibleColumns(table);
             for(int zi=0;zi<di.size();zi++) {x_visible[zi]=di.get(zi);}
        }
        if(this.dataY!=null) {
            List<Double> di = this.dataY.getDataVisibleColumns(table);
            for(int zi=0;zi<di.size();zi++) {y_visible[zi]=di.get(zi);}
        }
        if(this.dataZ!=null) {
            List<Double> di = this.dataZ.getDataVisibleColumns(table);
            for(int zi=0;zi<di.size();zi++) {x_visible[zi]=di.get(zi);}
        }

        List<XYZSeries> series = new ArrayList<>();
        if(this.dataCategoric==null) {
            XYZSeries sall = new XYZSeries("all",new ArrayList(Arrays.stream(x_visible).boxed().collect(Collectors.toList())),
                    new ArrayList(Arrays.stream(y_visible).boxed().collect(Collectors.toList())),
                    new ArrayList(Arrays.stream(z_visible).boxed().collect(Collectors.toList())));
            series.add(sall);
        }
        else {

            // split by category..
            Map<String,List<Integer>> splits = new HashMap<>();
            List<String> categories = dataCategoric.getCategoryVisibleColumns(table);
            for(int zi=0;zi<rows.size();zi++) {
                splits.putIfAbsent(categories.get(zi),new ArrayList<>());
                splits.get(categories.get(zi)).add(zi);
            }
            // TODO: if category objects are comparable, then sort according to that..
            categories = categories.stream().sorted().collect(Collectors.toList());

            for(String ki : splits.keySet()) {
                List<Double> xxi = new ArrayList<>();
                List<Double> yyi = new ArrayList<>();
                List<Double> zzi = new ArrayList<>();
                for(int xi : splits.get(ki)) {
                    xxi.add(x_visible[xi]);
                    yyi.add(y_visible[xi]);
                    zzi.add(z_visible[xi]);
                }

                String seriesKey = "unclassified";
                if(ki!=null) {seriesKey = ki;}
                XYZSeries sall = new XYZSeries(seriesKey, xxi,yyi,zzi);
                series.add(sall);
            }
        }

        this.data = series;
        this.fireDataSetChanged();
    }

    @Override
    public Number getZ(int i, int i1) {
        return this.data.get(i).z.get(i1);
    }

    @Override
    public double getZValue(int i, int i1) {
        return this.data.get(i).z.get(i1);
    }

    @Override
    public DomainOrder getDomainOrder() {
        return null;
    }

    @Override
    public int getItemCount(int i) {
        return this.data.get(i).x.size();
    }

    @Override
    public Number getX(int i, int i1) {
        return this.data.get(i).x.get(i1);
    }

    @Override
    public double getXValue(int i, int i1) {
        return this.data.get(i).x.get(i1);
    }

    @Override
    public Number getY(int i, int i1) {
        return this.data.get(i).y.get(i1);
    }

    @Override
    public double getYValue(int i, int i1) {
        return this.data.get(i).y.get(i1);
    }

    @Override
    public int getSeriesCount() {
        return this.data.size();
    }

    @Override
    public Comparable getSeriesKey(int i) {
        return this.data.get(i).seriesKey;
    }

    @Override
    public int indexOf(Comparable comparable) {
        //int idx = -1;
        for(int zi=0;zi<this.data.size();zi++) {
            if(this.data.get(zi).seriesKey.equals(comparable)) {return zi;}
        }
        return -1;
    }

    private List<DatasetChangeListener> listeners = new ArrayList<>();

    @Override
    public void addChangeListener(DatasetChangeListener datasetChangeListener) {
        this.listeners.add(datasetChangeListener);
    }

    @Override
    public void removeChangeListener(DatasetChangeListener datasetChangeListener) {
        this.listeners.remove(datasetChangeListener);
    }

    private void fireDataSetChanged() {
        for(DatasetChangeListener li : listeners) {
            li.datasetChanged(new DatasetChangeEvent(this,this));
        }
    }

    private DatasetGroup group;

    @Override
    public DatasetGroup getGroup() {
        return group;
    }

    @Override
    public void setGroup(DatasetGroup datasetGroup) {
        this.group = datasetGroup;
    }

    public DataTable getTable() {
        return table;
    }

    public NumericDatasource getDataX() {
        return dataX;
    }

    public NumericDatasource getDataY() {
        return dataY;
    }

    public NumericDatasource getDataZ() {
        return dataZ;
    }
}
