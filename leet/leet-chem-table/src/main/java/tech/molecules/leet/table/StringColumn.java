package tech.molecules.leet.table;

import javax.swing.table.TableCellEditor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StringColumn implements NColumn<NDataProvider.NStringDataProvider,String> {

    private String name;

    public StringColumn(String name){
        this.name = name;
    }

    @Override
    public String getName() {
        return this.name;
    }

    @Override
    public String getData(NDataProvider.NStringDataProvider data, String rowid) {
        return data.getStringData(rowid);
    }

    @Override
    public TableCellEditor getCellEditor() {
        return new DefaultStringCellEditor(this);
    }

    @Override
    public Map<String, NumericalDatasource<NDataProvider.NStringDataProvider>> getNumericalDataSources() {
        return new HashMap<>();
    }

    @Override
    public double evaluateNumericalDataSource(NDataProvider.NStringDataProvider dp, String datasource, String rowid) {
        return 0;
    }

    @Override
    public void startAsyncInitialization(NexusTableModel model, NDataProvider.NStringDataProvider dataprovider) {

    }

    @Override
    public List<String> getRowFilterTypes() {
        return null;
    }

    @Override
    public NexusRowFilter<NDataProvider.NStringDataProvider> createRowFilter(NexusTableModel tableModel, String name) {
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
