package tech.molecules.leet.table;

import javax.swing.*;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import java.io.Serializable;
import java.util.BitSet;
import java.util.List;
import java.util.Map;

public interface NColumn<U,T> {


    public String getName();
    public T getData(U data, String rowid);
    //public TableCellRenderer getCellRenderer();
    public void startAsyncInitialization(NexusTableModel model, U dataprovider);
    public TableCellEditor getCellEditor();

    //public List<Pair<String,Class>> getAvailableDataSources();
    //public Object getDataSource(String name);
    public Map<String,NumericalDatasource<U>> getNumericalDataSources();
    public double evaluateNumericalDataSource(U dp, String datasource, String rowid);

    public static abstract class CellSpecificAction extends AbstractAction {
        private NColumn column;
        private String rowid;
        public NColumn getColumn(){return this.column;}
        public String  getRowId(){return this.rowid;}
        public void setRowId(String rowid){this.rowid=rowid;}
        public CellSpecificAction(String name, NColumn column, String rowid) {
            super(name);
            this.column = column;
            this.rowid = rowid;
        }
    }

    //default List<Action> getHeaderActions

    default void addCellPopupAction(CellSpecificAction ca) {
    }

    /**
     * NOTE: There are columns that have different row filters available depending on the
     * colunn configuration. For these, the call to getRowFilterTypes() initializes the
     * currently available filters. I.e. to create a filter, it is necessary to call
     * getRowFilterTypes() before.
     *
     * @return
     */
    public List<String> getRowFilterTypes();

    /**
     * NOTE: There are columns that have different row filters available depending on the
     * colunn configuration. For these, the call to getRowFilterTypes() initializes the
     * currently available filters. I.e. to create a filter, it is necessary to call
     * getRowFilterTypes() before.
     *
     * @param tableModel
     * @param name
     * @return
     */
    public NexusRowFilter<U> createRowFilter(NexusTableModel tableModel, String name);
    //public List<NexusRowFilter<U>> getRowFilters();
    //public void removeRowFilter(NexusRowFilter<U> filter);

    public interface NexusRowFilter<U> extends Serializable {
        public String getFilterName();

        /**
         *
         * @param data
         * @param ids
         * @param filtered bits that are one indicate rows that are not yet filtered. For these
         *                 the function has to check if it should be filtered, and in that case
         *                 for the given position in the result bitset a zero must be returned.
         * @return
         */
        public BitSet filterNexusRows(U data, List<String> ids, BitSet filtered);
        public double getApproximateFilterSpeed();
        public void setupFilter(NexusTableModel model, U dp);
        public JPanel getFilterGUI();

        /**
         * For certain filters it may be possible that the column first has
         * to initialize specific datastructures asynchronously. In this case,
         * the filter will return false in this function until the data is ready.
         * Example would be the substructure filter and a structure column (loading
         * fingerprints asynchronously).
         *
         * @return
         */
        public boolean isReady();
    }

    public void addColumnDataListener(ColumnDataListener cdl);
    public boolean removeColumnDataListener(ColumnDataListener cdl);

    public interface ColumnDataListener {
        public void needFiltering();
    }

}
