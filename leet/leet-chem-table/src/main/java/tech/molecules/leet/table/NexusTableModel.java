package tech.molecules.leet.table;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;

import javax.swing.*;
import javax.swing.table.AbstractTableModel;
import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

public class NexusTableModel extends AbstractTableModel {

    //private NexusOsirisDataset data;

    /**
     * Here we register IDs for all objects that we need for
     * serialization / deserialization etc.
     */
    private Map<Integer,Pair<Class,Object>> registry;

    private List<String> allRows                 = new ArrayList<>();
    private List<String> visibleRows             = new ArrayList<>();
    private Map<String,Integer> visibleRowPositions = new HashMap<>();
    //private List<Pair<NColumn, NStructureDataProvider>> columns = new ArrayList<>();

    /**
     * The data providers for the columns are specified in the columnDataProvider map.
     */
    private List<NColumn> columns = new ArrayList<>();

    private Map<NColumn,NDataProvider> columnDataProviders = new HashMap<>();

    Map<NColumn,List<NColumn.NexusRowFilter>> rowFilters = new HashMap<>();

//    public void setNexusData(NexusOsirisDataset data) {
//        this.data = data;
//        this.visibleRows = new ArrayList<>( data.getData().keySet() );
//    }


    public NexusTableModel() {
        initNexusSelectionTypes();
    }

    public void setAllRows(List<String> rowids) {
        this.allRows = rowids;
        this.updateFiltering();
    }


    public List<String> getAllRows() {
        return this.allRows;
    }

    //public void setVisibleRows(List<String> rowids) {
    //    this.visibleRows = new ArrayList<>(rowids);
    //}

    public List<String> getVisibleRows() { return this.visibleRows; }

    private Map<NColumn, NColumn.ColumnDataListener> columnListeners = new HashMap<>();

    public <U,T> T getDatasetForColumn(NColumn<U,T> col) {
        //T dataset = null;
        //for(Pair<NColumn, NDataProvider> ci : this.columns) {
        //    if(ci.getLeft()==col) {return (U) ci.getRight();}
        //}
        return (T) this.columnDataProviders.get(col);
    }



    /**
     *
     * @param cols
     *
     */
    //public <U,T> void setNexusColumns(List<Pair<NColumn<U,T>, U>> cols) {
    public void setNexusColumnsWithDataProviders(List<Pair<NColumn, NDataProvider>> cols) {
        this.columns = new ArrayList<>( cols.stream().map( ci -> ci.getLeft() ).collect(Collectors.toList()) ) ;
        //List<NDataProvider> dps = new ArrayList<>( cols.stream().map( ci -> ci.getRight() ).collect(Collectors.toList()) ) ;;
        for(int zi=0;zi<this.columns.size();zi++) {
            NColumn cp = columns.get(zi);
            //NDataProvider dp = //this.columnDataProviders.get(cp);
            this.setDataProviderForColumn(cp, cols.get(zi).getRight());
        }
        reinitNexusColumns();
    }

    public void addNexusColumn(NDataProvider ndp, NColumn nc) {
        this.setNexusColumnWithDataProvider(this.columns.size(),Pair.of(nc,ndp));
    }

    public void removeNexusColumn(NColumn nc) {
        this.columns.remove(nc);
        this.rowFilters.remove(nc);
        this.updateFiltering();
    }

    /**
     * Provides mapping from table model row index to row id.
     *
     * @param visible_row_index
     * @return
     */
    public String getRowIdForVisibleRow(int visible_row_index) {
        return this.visibleRows.get(visible_row_index);
    }

    private void reinitNexusColumns() {
        for(int zi=0;zi<this.columns.size();zi++) {
            NColumn cp = columns.get(zi);
            cp.setDataProvider(columnDataProviders.get(cp));
            if(!columnListeners.containsKey(cp)) {
                NColumn.ColumnDataListener cdpl = new NColumn.ColumnDataListener() {
                    @Override
                    public void needFiltering() {
                        updateFiltering();
                    }
                };
                cp.addColumnDataListener(cdpl);
                columnListeners.put(cp,cdpl);
            }
        }
        this.updateFiltering();
        //fireTableStructureChanged();
        //fireNexusTableStructureChangedEvent();
        fireNexusTableStructureChangedEvent();
        fireTableStructureChanged();
    }

    public void setNexusColumnWithDataProvider(int idx, Pair<NColumn, NDataProvider> column) {
        this.columns.add(idx, column.getLeft());
        this.setDataProviderForColumn(column.getLeft(),column.getRight());
        reinitNexusColumns();
    }

    private void registerNexusColumn(NColumn nci) {
        Random ri = new Random();
        boolean found = false;
        int ii = -1;
        while(!found) {
            ii = Math.abs(ri.nextInt());
            found = !this.registry.containsKey(ii);
        }
        this.registry.put(ii, Pair.of(nci.getClass(),nci));
    }

    private void registerNexusDataProvider(NDataProvider ndp) {
        Random ri = new Random();
        boolean found = false;
        int ii = -1;
        while(!found) {
            ii = Math.abs(ri.nextInt());
            found = !this.registry.containsKey(ii);
        }
        this.registry.put(ii, Pair.of(ndp.getClass(),ndp));
    }

    public <U extends NDataProvider,T> void setDataProviderForColumn(NColumn<U,T> c, U np) {
        this.columnDataProviders.put(c,np);
        c.setDataProvider(np);
    }

    public List<NColumn> getNexusColumns() {
        return this.columns;
    }

    @Override
    public int getRowCount() {
        return this.visibleRows.size();
    }

    @Override
    public int getColumnCount() {
        return this.columns.size();
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        //return columns.get(columnIndex).getData( columns.get(columnIndex).getRight(),visibleRows.get(rowIndex));
        //return columns.get(columnIndex).getData( columnDataProviders.get(columns.get(columnIndex)),visibleRows.get(rowIndex));
        return columns.get(columnIndex).getData( visibleRows.get(rowIndex));
    }

    public <U,T> void  addRowFilter(NColumn<U,T> col, NColumn.NexusRowFilter<T> filter ) {
        filter.setupFilter(this, this.getDatasetForColumn(col));
        if( !this.rowFilters.containsKey(col) ) {this.rowFilters.put(col,new ArrayList<>());}
        this.rowFilters.get(col).add(filter);
        this.updateFiltering();
    }

    private void updateRowIndices() {

    }

    @Override
    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return true;
    }

    public void resetFilters() {
        this.rowFilters.clear();
    }

    public final Object lockTable = new Object();

    public void updateFiltering() {
        synchronized(lockTable) {
            List<Triple<NColumn, NStructureDataProvider,NColumn.NexusRowFilter>> filters = new ArrayList();
            for(Map.Entry<NColumn,List<NColumn.NexusRowFilter>> fiss : this.rowFilters.entrySet()) {
                for(NColumn.NexusRowFilter fi : fiss.getValue()) {
                    filters.add(Triple.of(fiss.getKey(), (NStructureDataProvider) getDatasetForColumn( fiss.getKey()),fi));
                }
            }
            // sort fastest filters first:
            filters.sort( (x,y) -> -Double.compare( x.getRight().getApproximateFilterSpeed() , y.getRight().getApproximateFilterSpeed() ) );

            BitSet in = new BitSet(this.allRows.size());
            in.set(0,this.allRows.size());
            for(int zi=0;zi<filters.size();zi++) {
                in.and( filters.get(zi).getRight().filterNexusRows( filters.get(zi).getMiddle() , this.allRows, in) );
            }
            this.visibleRows = new ArrayList<>();
            this.visibleRowPositions.clear();
            for(int zi=0;zi<this.allRows.size();zi++) {
                if(in.get(zi)) {
                    visibleRows.add(this.allRows.get(zi));
                    this.visibleRowPositions.put(this.allRows.get(zi),visibleRows.size()-1);
                }
            }
            fireTableDataChanged();
        }
    }

    List<NexusTableModelListener> nexusListeners = new ArrayList<>();

    public void addNexusListener(NexusTableModelListener li) {
        nexusListeners.add(li);}
    public void removeNexusListener(NexusTableModelListener li) {
        nexusListeners.remove(li);}

    public void fireNexusTableStructureChangedEvent() {
        for(NexusTableModelListener li : this.nexusListeners) {
            SwingUtilities.invokeLater(new Runnable(){
                @Override
                public void run() {
                    li.nexusTableStructureChanged();
                }
            });
        }
    }

    public List<Pair<NColumn, NDataProvider>> getNexusColumnsWithDataProviders() {
        //List<Pair<NColumn, NStructureDataProvider>> colwdp = new ArrayList<>();
        List<Pair<NColumn, NDataProvider>> colwdp = new ArrayList<>();
        for(NColumn ci : this.columns) {
            //colwdp.add(Pair.of(ci, (NStructureDataProvider) this.getDatasetForColumn(ci)));
            colwdp.add(Pair.of(ci, (NDataProvider) this.getDatasetForColumn(ci)));
        }
        return colwdp;
    }

    public Map<NColumn,Map<String,NumericalDatasource>> collectNumericDataSources() {
        Map<NColumn,Map<String,NumericalDatasource>> nds = new HashMap<>();
        for(NColumn nci : this.getNexusColumns()) {
            if(nci.getNumericalDataSources()!=null && nci.getNumericalDataSources().size()>0 ) {
                nds.put(nci,nci.getNumericalDataSources());
            }
        }
        return nds;
    }

    public String getKeyAtVisiblePosition(int ri) {
        return this.visibleRows.get(ri);
    }

    public int getVisiblePositionOfKey(String ki) {
        return this.visibleRowPositions.get(ki);
    }

    public static interface NexusTableModelListener {
        public void nexusTableStructureChanged();
    }


    /**
     * Describes the selection and highlighting status of a given row.
     *
     * highlightingColor is a single color that usually is used for
     * the background of the row
     *
     * selectionColor: there may be multiple selection at place at the
     * same time. The cell renderers should reflect all of them somehow
     *
     */
    public static class NexusHighlightingAndSelectionStatus {
        public final Color highlightingColor;
        public final List<SelectionType> selectionColor;
        public NexusHighlightingAndSelectionStatus(Color highlightingColor, List<SelectionType> selectionColor) {
            this.highlightingColor = highlightingColor;
            this.selectionColor = selectionColor;
        }
    }

    public static class SelectionType {
        private String name;
        private Color color;
        public SelectionType(String name, Color color) {
            this.name = name;
            this.color = color;
        }
        public String getName() {
            return name;
        }
        public Color getColor() {
            return color;
        }
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            SelectionType that = (SelectionType) o;
            return new EqualsBuilder().append(name, that.name).isEquals();
        }
        @Override
        public int hashCode() {
            return new HashCodeBuilder(17, 37).append(name).toHashCode();
        }
    }

    private Map<String,SelectionType> selectionTypeRegistry = new HashMap<>();
    public SelectionType getSelectionType(String name) {
        return this.selectionTypeRegistry.get(name);
    }

    public Set<String> getSelectionTypeRows(String selectionTypeName) {
        return this.selections.get(this.selectionTypeRegistry.get(selectionTypeName));
    }

    private void initNexusSelectionTypes() {
        this.registerSelectionType(new SelectionType(SELECTION_TYPE_SELECTED,Color.red.brighter().brighter()));
        this.registerSelectionType(new SelectionType(SELECTION_TYPE_MOUSE_OVER,Color.cyan.darker()));
    }

    public static final String SELECTION_TYPE_SELECTED   = "selected";
    public static final String SELECTION_TYPE_MOUSE_OVER = "mouseOver";

    /**
     * returns false if type was already registered
     * @param type
     * @return
     */
    public boolean registerSelectionType(SelectionType type) {
        if(this.selectionTypeRegistry.containsKey(type.getName())) {
            return false;
        }
        this.selectionTypeRegistry.put(type.getName(),type);
        return true;
    }

    private Map<String,List<SelectionType>> selectionTypes  = new HashMap<>();
    private Map<String,Color>       highlightColors = new HashMap<>();

    private Map<SelectionType,Set<String>> selections = new HashMap<>();

    public void addSelectionTypeToRows(SelectionType c, Collection<String> rows) {
        for(String rowid : rows) {
            if (!selectionTypes.containsKey(rowid)) {
                this.selectionTypes.put(rowid, new ArrayList<>());
            }
            this.selectionTypes.get(rowid).add(c);
        }
        if(!selections.containsKey(c)) {selections.put(c,new HashSet<>());}
        selections.get(c).addAll(rows);
    }


    public void removeSelectionTypeFromRows(SelectionType c, Collection<String> rows) {
        for(String rowid : rows) {
            List<SelectionType> ci = selectionTypes.get(rowid);
            if(ci!=null) {
                ci.remove(c);
            }
        }
        if(selections.containsKey(c)) {
            selections.get(c).removeAll(rows);
        }
    }

    public void setHighlightColors(Map<String,Color> colors) {
        this.highlightColors = colors;
    }

    public NexusHighlightingAndSelectionStatus getHighlightingAndSelectionStatus(int row) {
        List<SelectionType> selCols = selectionTypes.get( this.visibleRows.get(row) );
        return new NexusHighlightingAndSelectionStatus(this.highlightColors.get(this.visibleRows.get(row)),selCols);

//        if(false) {
//            if (row % 6 == 2) {
//                //List<Color> cta = new ArrayList<>();cta.add(Color.GREEN.darker());cta.add(Color.orange.darker());
//                return new NexusHighlightingAndSelectionStatus(Color.cyan.darker(), null);
//            }
//            if (row % 8 == 3) {
//                List<Color> cta = new ArrayList<>();
//                cta.add(Color.red.brighter());
//                cta.add(Color.blue.darker());
//                return new NexusHighlightingAndSelectionStatus(Color.orange.brighter(), cta);
//            }
//            if (row % 4 == 0) {
//                return new NexusHighlightingAndSelectionStatus(Color.blue.brighter(), null);
//            }
//        }
    }

    public static class NexusEvent {
        private Object source;

        public NexusEvent(Object source) {
            this.source = source;
        }

        public Object getSource() {
            return this.source;
        }

    }

    public static class NexusSelectionChangedEvent extends NexusEvent {
        Set<String> rows;

        public NexusSelectionChangedEvent(Object source, Set<String> rows) {
            super(source);
            this.rows = rows;
        }

        public Set<String> getRows() {return this.rows;}
    }

    public static class NexusHighlightingChangedEvent extends NexusEvent {
        Set<String> rows;

        public NexusHighlightingChangedEvent(Object source, Set<String> rows) {
            super(source);
            this.rows = rows;
        }

        public Set<String> getRows() {return this.rows;}
    }



    /**
     * This registers all necessary serializers in the Jackson object
     * mapper. After calling this, the mapper can correctly serialize
     * Nexus-related fields in Config objects (NColumn, NDataProvider etc.).
     *
     */
    public static void equipJacksonSerializer(ObjectMapper mapper) {

    }

    /**
     * Registers all necessary deserializers required for the Jackson
     * object mapper. For this, we require the information stored in the
     * registry, i.e. we must be able to fill the id-referenced values.
     *
     * @param mapper
     */
    public static void equipJacksonDeserializer(ObjectMapper mapper, Map<Integer,Pair<Class,Object>> registry) {

    }

}
