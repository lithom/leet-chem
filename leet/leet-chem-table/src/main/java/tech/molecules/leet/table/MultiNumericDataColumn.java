package tech.molecules.leet.table;

import javax.swing.table.TableCellEditor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MultiNumericDataColumn implements NColumn<NDataProvider.NMultiNumericDataProvider,double[]> {

    private NDataProvider.NMultiNumericDataProvider dp;
    private String name;

    public MultiNumericDataColumn(String name) {
        this.name = name;
    }

    @Override
    public void setDataProvider(NDataProvider.NMultiNumericDataProvider dataprovider) {
        this.dp = dataprovider;
    }

    @Override
    public NDataProvider.NMultiNumericDataProvider getDataProvider() {
        return this.dp;
    }

    @Override
    public String getName() {
        return this.name;
    }

    @Override
    public double[] getData(String rowid) {
        return this.dp.getMultiNumericData(rowid);
    }

    @Override
    public TableCellEditor getCellEditor() {
        return new DefaultNumericCellEditor(this);
    }

    @Override
    public Map<String, NumericalDatasource<NDataProvider.NMultiNumericDataProvider>> getNumericalDataSources() {
        HashMap<String,NumericalDatasource<NDataProvider.NMultiNumericDataProvider>> nds = new HashMap<>();
        DefaultNumericRangeFilter<NDataProvider.NMultiNumericDataProvider,double[]> rf = new DefaultNumericRangeFilter<>(this,this.getName());
        //nds.put(rf.getFilterName(),rf);
        return nds;
    }

    @Override
    public void startAsyncReinitialization(NexusTableModel model) {

    }

    @Override
    public List<String> getRowFilterTypes() {
        return null;
    }

    @Override
    public NexusRowFilter<NDataProvider.NMultiNumericDataProvider> createRowFilter(NexusTableModel tableModel, String name) {
        return null;
    }

    @Override
    public void addColumnDataListener(ColumnDataListener cdl) {

    }

    @Override
    public boolean removeColumnDataListener(ColumnDataListener cdl) {
        return false;
    }
}
