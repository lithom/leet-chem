package tech.molecules.leet.table;

import javax.swing.table.TableCellEditor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MultiNumericDataColumn implements NColumn<NDataProvider.NMultiNumericDataProvider,double[]> {

    private String name;

    public MultiNumericDataColumn(String name) {
        this.name = name;
    }

    @Override
    public String getName() {
        return this.name;
    }

    @Override
    public double[] getData(NDataProvider.NMultiNumericDataProvider data, String rowid) {
        return data.getMultiNumericData(rowid);
    }

    @Override
    public TableCellEditor getCellEditor() {
        return new DefaultNumericCellEditor(this);
    }

    @Override
    public Map<String, NumericalDatasource<NDataProvider.NMultiNumericDataProvider>> getNumericalDataSources() {
        return new HashMap<>();
    }

    @Override
    public double evaluateNumericalDataSource(NDataProvider.NMultiNumericDataProvider dp, String datasource, String rowid) {
        return 0;
    }

    @Override
    public void startAsyncInitialization(NexusTableModel model, NDataProvider.NMultiNumericDataProvider dataprovider) {

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
