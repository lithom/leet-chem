package tech.molecules.leet.table.chem;

import tech.molecules.leet.table.NColumn;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.NumericalDatasource;

import javax.swing.table.TableCellEditor;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NSynthonWithContextColumn implements NColumn<NSynthonWithContextDataProvider,String> {

    public static enum SYNTHON_COLUMN_MODE {SHOW, SHOW_ASSEMBLED};

    private SYNTHON_COLUMN_MODE mode = SYNTHON_COLUMN_MODE.SHOW;

    private NSynthonWithContextDataProvider dp;

    @Override
    public String getName() {
        return "Synthon";
    }

    //public void setMode(SYNTHON_COLUMN_MODE mode, StereoMolecule )


    @Override
    public void setDataProvider(NSynthonWithContextDataProvider dataprovider) {
        this.dp = dp;
    }

    @Override
    public NSynthonWithContextDataProvider getDataProvider() {
        return this.dp;
    }

    @Override
    public String getData(String rowid) {
        return dp.getStructureData(rowid).toString();
    }



    @Override
    public TableCellEditor getCellEditor() {
        return null;
    }

    @Override
    public Map<String, NumericalDatasource<NSynthonWithContextDataProvider>> getNumericalDataSources() {
        return new HashMap<>();
    }


    @Override
    public void startAsyncReinitialization(NexusTableModel model) {

    }

    @Override
    public List<String> getRowFilterTypes() {
        return null;
    }

    @Override
    public NexusRowFilter<NSynthonWithContextDataProvider> createRowFilter(NexusTableModel tableModel, String name) {
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
